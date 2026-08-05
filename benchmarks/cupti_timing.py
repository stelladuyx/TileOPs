# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Direct CUPTI activity timing adapted from NVIDIA SOL-ExecBench.

Unlike the legacy TileOps timing path, this module does not use
``torch.profiler`` or Kineto annotation projection.  It discovers the GPU
activity sequence launched by one callable invocation, then uses timestamps in
the CUPTI clock domain to attribute the same sequence to every timed iteration.

The implementation is intentionally isolated from :mod:`benchmark_base` so the
activity matching can be unit-tested without a CUDA device or a Kineto trace.
"""

from __future__ import annotations

import bisect
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Literal

import torch


class DirectCuptiError(RuntimeError):
    """Base class for expected direct-CUPTI measurement failures."""


class DirectCuptiUnavailableError(DirectCuptiError):
    """The optional ``cupti-python`` dependency is unavailable."""


class DirectCuptiTraceError(DirectCuptiError):
    """A CUPTI trace cannot be attributed to complete benchmark calls."""


@dataclass(frozen=True)
class CuptiActivityInfo:
    """Normalized CUPTI activity for a kernel, memcpy, or memset operation."""

    name: str
    start: int
    end: int
    correlation_id: int
    copy_kind: int
    bytes: int
    value: int
    kind: Any

    @classmethod
    def from_activity(cls, activity: Any, cupti_api: Any) -> "CuptiActivityInfo":
        if activity.kind == cupti_api.ActivityKind.CONCURRENT_KERNEL:
            name = activity.name
        elif activity.kind == cupti_api.ActivityKind.MEMCPY:
            name = "MEMCPY"
        elif activity.kind == cupti_api.ActivityKind.MEMSET:
            name = "MEMSET"
        else:  # pragma: no cover - collector filters these kinds
            name = str(activity.kind)

        is_copy = activity.kind in (
            cupti_api.ActivityKind.MEMCPY,
            cupti_api.ActivityKind.MEMSET,
        )
        return cls(
            name=name,
            start=int(activity.start),
            end=int(activity.end),
            correlation_id=int(activity.correlation_id),
            copy_kind=(
                int(activity.copy_kind)
                if activity.kind == cupti_api.ActivityKind.MEMCPY
                else 0
            ),
            bytes=int(activity.bytes) if is_copy else 0,
            value=(
                int(activity.value)
                if activity.kind == cupti_api.ActivityKind.MEMSET
                else 0
            ),
            kind=activity.kind,
        )

    def identity(self) -> tuple[str, int, int, int, str]:
        """Return the stable activity identity used across iterations."""
        return (self.name, self.copy_kind, self.bytes, self.value, str(self.kind))


@dataclass
class CuptiActivityBuffers:
    """Mutable activity buffer populated by ``cupti-python`` callbacks."""

    activities: list[CuptiActivityInfo] = field(default_factory=list)


@dataclass(frozen=True)
class DirectCuptiMeasurement:
    """Per-iteration timings and the discovery sequence that produced them."""

    samples_ms: list[float]
    activity_sum_ms: list[float]
    activity_span_ms: list[float]
    activity_union_busy_ms: list[float]
    inter_activity_idle_ms: list[float]
    activity_overlap_ms: list[float]
    # Legacy algebraic value: span - sum == idle - overlap.  It is an exact
    # idle gap only when the selected activities do not overlap.
    inter_activity_gap_ms: list[float]
    expected_sequence: list[tuple[str, int, int, int, str]]
    metric: str
    # Conservative guard bands around the complete selected GPU activity span.
    # A pair is ``(first_activity_start - window_start,
    # window_end - last_activity_end)`` in nanoseconds.  Negative values mean
    # the normalized GPU span crossed a CPU timestamp boundary.
    boundary_margins_ns: list[tuple[int, int]]


def load_cupti_api() -> Any:
    """Load the optional SOL direct-CUPTI Python binding lazily."""
    try:
        from cupti import cupti
    except (ImportError, OSError) as exc:
        raise DirectCuptiUnavailableError(
            "direct CUPTI timing requires the optional cupti-python package "
            "and a compatible libcupti"
        ) from exc
    return cupti


def activity_sequence(
    activities: list[CuptiActivityInfo],
) -> list[tuple[str, int, int, int, str]]:
    return [activity.identity() for activity in activities]


def activity_counts(activities: list[CuptiActivityInfo]) -> Counter:
    return Counter(activity_sequence(activities))


def activity_timeline_components_ns(
    activities: list[CuptiActivityInfo],
) -> tuple[int, int, int, int, int]:
    """Return ``(sum, span, union_busy, idle, overlap)`` in nanoseconds."""
    if not activities:
        raise ValueError("activity timeline requires at least one activity")
    intervals = sorted((activity.start, activity.end) for activity in activities)
    duration_sum = sum(end - start for start, end in intervals)
    span = max(end for _, end in intervals) - intervals[0][0]
    union_busy = 0
    merged_start, merged_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= merged_end:
            merged_end = max(merged_end, end)
        else:
            union_busy += merged_end - merged_start
            merged_start, merged_end = start, end
    union_busy += merged_end - merged_start
    idle = span - union_busy
    overlap = duration_sum - union_busy
    return duration_sum, span, union_busy, idle, overlap


def _relative_order_score(
    candidate: list[tuple[str, int, int, int, str]],
    expected: list[tuple[str, int, int, int, str]],
) -> int:
    """Return the longest-common-subsequence score for two activity lists."""
    scores = [0] * (len(expected) + 1)
    for candidate_name in candidate:
        previous_diagonal = 0
        for idx, expected_name in enumerate(expected, start=1):
            previous_score = scores[idx]
            if candidate_name == expected_name:
                scores[idx] = max(scores[idx], previous_diagonal + 1)
            else:
                scores[idx] = max(scores[idx], scores[idx - 1])
            previous_diagonal = previous_score
    return scores[-1]


def select_activity_sequence(
    activities: list[CuptiActivityInfo],
    expected_sequence: list[tuple[str, int, int, int, str]],
    *,
    iteration: int,
) -> list[CuptiActivityInfo]:
    """Select one complete discovered activity sequence from a noisy window."""
    expected_count = len(expected_sequence)
    if expected_count == 0:
        raise DirectCuptiTraceError("no expected GPU activities were discovered")

    expected_names = set(expected_sequence)
    expected_counts = Counter(expected_sequence)
    candidates = [a for a in activities if a.identity() in expected_names]
    candidate_sequence = activity_sequence(candidates)

    for start_idx in range(len(candidates) - expected_count + 1):
        end_idx = start_idx + expected_count
        if candidate_sequence[start_idx:end_idx] == expected_sequence:
            return candidates[start_idx:end_idx]

    # Concurrent streams can reorder otherwise identical activity sets.  Keep
    # the best relative order, then the tightest GPU span, as SOL-ExecBench does.
    best: list[CuptiActivityInfo] | None = None
    best_score: tuple[int, int] | None = None
    for start_idx in range(len(candidates) - expected_count + 1):
        end_idx = start_idx + expected_count
        window_sequence = candidate_sequence[start_idx:end_idx]
        if Counter(window_sequence) != expected_counts:
            continue
        window = candidates[start_idx:end_idx]
        span = max(a.end for a in window) - min(a.start for a in window)
        score = (_relative_order_score(window_sequence, expected_sequence), -span)
        if best_score is None or score > best_score:
            best_score = score
            best = window

    if best is not None:
        return best

    raise DirectCuptiTraceError(
        "expected GPU activity sequence not found at iteration "
        f"{iteration}: {expected_sequence!r} not in {candidate_sequence!r}"
    )


def _request_activity_buffer() -> tuple[int, int]:
    # Eight MiB matches SOL-ExecBench and comfortably holds the default 50
    # iterations even for multi-kernel operators.
    return 8 * 1024 * 1024, 0


def _collect_activity_records(
    buffers: CuptiActivityBuffers,
    activities: list[Any],
    cupti_api: Any,
) -> None:
    accepted_kinds = (
        cupti_api.ActivityKind.CONCURRENT_KERNEL,
        cupti_api.ActivityKind.MEMCPY,
        cupti_api.ActivityKind.MEMSET,
    )
    for activity in activities:
        if activity.kind in accepted_kinds:
            buffers.activities.append(CuptiActivityInfo.from_activity(activity, cupti_api))


@contextmanager
def collect_cupti_activities(cupti_api: Any | None = None) -> Iterator[CuptiActivityBuffers]:
    """Collect direct CUPTI activities and always tear tracing state down."""
    api = cupti_api or load_cupti_api()
    buffers = CuptiActivityBuffers()
    enabled_kinds: list[Any] = []
    activity_kinds = (
        api.ActivityKind.CONCURRENT_KERNEL,
        api.ActivityKind.MEMCPY,
        api.ActivityKind.MEMSET,
    )
    try:
        for activity_kind in activity_kinds:
            api.activity_enable(activity_kind)
            enabled_kinds.append(activity_kind)
        api.activity_register_callbacks(
            _request_activity_buffer,
            partial(_collect_activity_records, buffers, cupti_api=api),
        )
        yield buffers
    finally:
        if enabled_kinds:
            try:
                api.activity_flush_all(0)
            finally:
                for activity_kind in enabled_kinds:
                    api.activity_disable(activity_kind)
                api.finalize()


def measure_direct_cupti(
    run_iteration: Callable[[int], Any],
    prepare_iteration: Callable[[int], None],
    repeats: int,
    *,
    metric: Literal["activity-sum", "activity-span"] = "activity-sum",
    cupti_api: Any | None = None,
) -> DirectCuptiMeasurement:
    """Measure complete per-call GPU activity without Kineto projection.

    ``prepare_iteration`` runs before the timestamp window and may enqueue L2
    clearing work, but it must synchronize before returning.  The timestamp at
    the end of each window is taken only after a CUDA synchronize, so delayed or
    non-default-stream work launched by the callable remains attributable.
    ``activity-sum`` preserves TileOps' historical pure-activity semantics;
    ``activity-span`` reproduces upstream SOL-ExecBench's first-start to
    last-end semantics, including inter-activity gaps.
    """
    if repeats <= 0:
        raise ValueError(f"repeats must be positive, got {repeats}")
    if metric not in ("activity-sum", "activity-span"):
        raise ValueError(f"unknown direct CUPTI metric: {metric!r}")

    api = cupti_api or load_cupti_api()

    # Discover the callable's activity sequence after cache/setup work drains.
    prepare_iteration(-1)
    with collect_cupti_activities(api) as discovery_buffers:
        run_iteration(-1)
        torch.cuda.synchronize()

    discovered = sorted(
        discovery_buffers.activities,
        key=lambda activity: (activity.start, activity.end, activity.correlation_id),
    )
    expected = activity_sequence(discovered)
    if not expected:
        raise DirectCuptiTraceError("no GPU activities recorded during discovery iteration")

    timestamp_windows: list[tuple[int, int]] = []
    with collect_cupti_activities(api) as timing_buffers:
        for iteration in range(repeats):
            prepare_iteration(iteration)
            start = int(api.get_timestamp())
            run_iteration(iteration)
            torch.cuda.synchronize()
            end = int(api.get_timestamp())
            if end <= start:
                raise DirectCuptiTraceError(
                    f"invalid CUPTI timestamp window at iteration {iteration}: {start}..{end}"
                )
            timestamp_windows.append((start, end))

    timed_activities = sorted(
        timing_buffers.activities,
        key=lambda activity: (activity.start, activity.end, activity.correlation_id),
    )
    starts = [activity.start for activity in timed_activities]
    samples_ms: list[float] = []
    activity_sum_ms: list[float] = []
    activity_span_ms: list[float] = []
    activity_union_busy_ms: list[float] = []
    inter_activity_idle_ms: list[float] = []
    activity_overlap_ms: list[float] = []
    inter_activity_gap_ms: list[float] = []
    boundary_margins_ns: list[tuple[int, int]] = []
    expected_counts = Counter(expected)
    for iteration, (start, end) in enumerate(timestamp_windows):
        left = bisect.bisect_left(starts, start)
        right = bisect.bisect_right(starts, end)
        selected = select_activity_sequence(
            timed_activities[left:right],
            expected,
            iteration=iteration,
        )
        if activity_counts(selected) != expected_counts:  # defensive invariant
            raise DirectCuptiTraceError(
                f"activity count mismatch at iteration {iteration}"
            )
        if any(activity.end <= activity.start for activity in selected):
            raise DirectCuptiTraceError(
                f"invalid GPU activity duration at iteration {iteration}"
            )
        first_activity_start = min(activity.start for activity in selected)
        last_activity_end = max(activity.end for activity in selected)
        boundary_margins_ns.append(
            (first_activity_start - start, end - last_activity_end)
        )
        sum_ns, span_ns, union_busy_ns, idle_ns, overlap_ns = (
            activity_timeline_components_ns(selected)
        )
        activity_sum_ms.append(sum_ns / 1e6)
        activity_span_ms.append(span_ns / 1e6)
        activity_union_busy_ms.append(union_busy_ns / 1e6)
        inter_activity_idle_ms.append(idle_ns / 1e6)
        activity_overlap_ms.append(overlap_ns / 1e6)
        inter_activity_gap_ms.append((span_ns - sum_ns) / 1e6)
        duration_ns = sum_ns if metric == "activity-sum" else span_ns
        samples_ms.append(duration_ns / 1e6)

    return DirectCuptiMeasurement(
        samples_ms=samples_ms,
        activity_sum_ms=activity_sum_ms,
        activity_span_ms=activity_span_ms,
        activity_union_busy_ms=activity_union_busy_ms,
        inter_activity_idle_ms=inter_activity_idle_ms,
        activity_overlap_ms=activity_overlap_ms,
        inter_activity_gap_ms=inter_activity_gap_ms,
        expected_sequence=expected,
        metric=metric,
        boundary_margins_ns=boundary_margins_ns,
    )
