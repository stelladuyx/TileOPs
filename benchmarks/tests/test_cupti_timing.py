"""Unit tests for direct CUPTI activity attribution (no GPU required)."""

import importlib.util
import statistics
from contextlib import contextmanager

import pytest
import torch

import benchmarks.cupti_timing as direct


def _activity(name, start, end, correlation_id, kind="kernel"):
    return direct.CuptiActivityInfo(
        name=name,
        start=start,
        end=end,
        correlation_id=correlation_id,
        copy_kind=0,
        bytes=0,
        value=0,
        kind=kind,
    )


def test_select_activity_sequence_accepts_reordered_complete_set():
    activities = [
        _activity("user_b", 100, 110, 1),
        _activity("user_a", 120, 130, 2),
        _activity("user_c", 140, 150, 3),
    ]
    expected = [activities[1].identity(), activities[0].identity(), activities[2].identity()]

    selected = direct.select_activity_sequence(activities, expected, iteration=0)

    assert {activity.correlation_id for activity in selected} == {1, 2, 3}


def test_select_activity_sequence_rejects_partial_call():
    expected_activities = [
        _activity("user_a", 10, 20, 1),
        _activity("user_b", 30, 40, 2),
    ]

    with pytest.raises(direct.DirectCuptiTraceError, match="expected GPU activity sequence"):
        direct.select_activity_sequence(
            [expected_activities[0]],
            direct.activity_sequence(expected_activities),
            iteration=4,
        )


@pytest.mark.parametrize(
    "metric, expected_samples",
    [
        ("activity-sum", [0.00005, 0.00007]),
        ("activity-span", [0.00007, 0.00008]),
    ],
)
def test_measure_direct_cupti_filters_setup_and_measures_multikernel(
    monkeypatch, metric, expected_samples
):
    discovery = direct.CuptiActivityBuffers(
        activities=[
            _activity("user_a", 10, 20, 1),
            _activity("user_b", 30, 50, 2),
        ]
    )
    timing = direct.CuptiActivityBuffers(
        activities=[
            _activity("cache_clear", 80, 95, 3),
            _activity("user_a", 120, 140, 4),
            _activity("user_b", 160, 190, 5),
            _activity("cache_clear", 280, 295, 6),
            _activity("user_a", 320, 350, 7),
            _activity("user_b", 360, 400, 8),
        ]
    )
    buffers = iter([discovery, timing])

    @contextmanager
    def collect(_api):
        yield next(buffers)

    class FakeCupti:
        def __init__(self):
            self._timestamps = iter([100, 200, 300, 450])

        def get_timestamp(self):
            return next(self._timestamps)

    calls = []
    monkeypatch.setattr(direct, "collect_cupti_activities", collect)
    monkeypatch.setattr(direct.torch.cuda, "synchronize", lambda: calls.append("sync"))

    result = direct.measure_direct_cupti(
        lambda iteration: calls.append(f"run:{iteration}"),
        lambda iteration: calls.append(f"prepare:{iteration}"),
        repeats=2,
        metric=metric,
        cupti_api=FakeCupti(),
    )

    assert result.samples_ms == pytest.approx(expected_samples)
    assert result.metric == metric
    assert result.expected_sequence == direct.activity_sequence(discovery.activities)
    assert result.boundary_margins_ns == [(20, 10), (20, 50)]
    assert calls.count("sync") == 3  # discovery plus two timed iterations


def test_measure_direct_cupti_fails_when_an_iteration_is_incomplete(monkeypatch):
    discovery = direct.CuptiActivityBuffers(
        activities=[
            _activity("user_a", 10, 20, 1),
            _activity("user_b", 30, 40, 2),
        ]
    )
    timing = direct.CuptiActivityBuffers(
        activities=[_activity("user_a", 120, 140, 3)]
    )
    buffers = iter([discovery, timing])

    @contextmanager
    def collect(_api):
        yield next(buffers)

    class FakeCupti:
        timestamps = iter([100, 200])

        @classmethod
        def get_timestamp(cls):
            return next(cls.timestamps)

    monkeypatch.setattr(direct, "collect_cupti_activities", collect)
    monkeypatch.setattr(direct.torch.cuda, "synchronize", lambda: None)

    with pytest.raises(direct.DirectCuptiTraceError, match="iteration 0"):
        direct.measure_direct_cupti(
            lambda _iteration: None,
            lambda _iteration: None,
            repeats=1,
            cupti_api=FakeCupti,
        )


def test_measure_direct_cupti_rejects_empty_discovery(monkeypatch):
    @contextmanager
    def collect(_api):
        yield direct.CuptiActivityBuffers()

    monkeypatch.setattr(direct, "collect_cupti_activities", collect)
    monkeypatch.setattr(direct.torch.cuda, "synchronize", lambda: None)

    with pytest.raises(direct.DirectCuptiTraceError, match="discovery"):
        direct.measure_direct_cupti(
            lambda _iteration: None,
            lambda _iteration: None,
            repeats=1,
            cupti_api=object(),
        )


@pytest.mark.nightly
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(importlib.util.find_spec("cupti") is None, reason="cupti-python required")
def test_direct_cupti_gpu_returns_stable_complete_samples():
    """Minimal H200 integration check; the full variance matrix lives in the tool."""
    x = torch.randn(65536, device="cuda", dtype=torch.bfloat16)
    y = torch.randn_like(x)
    l2_bytes = torch.cuda.get_device_properties(0).L2_cache_size
    cache = torch.empty(max(1, l2_bytes), device="cuda", dtype=torch.int8)

    def prepare(_iteration):
        cache.zero_()
        torch.cuda.synchronize()

    def run(_iteration):
        torch.add(x, y, out=x)

    first = direct.measure_direct_cupti(run, prepare, repeats=20)
    second = direct.measure_direct_cupti(run, prepare, repeats=20)

    assert len(first.samples_ms) == len(second.samples_ms) == 20
    assert all(sample > 0 for sample in first.samples_ms + second.samples_ms)
    ratio = statistics.median(second.samples_ms) / statistics.median(first.samples_ms)
    assert 0.8 < ratio < 1.2
    assert first.expected_sequence == second.expected_sequence
    assert len(first.boundary_margins_ns) == len(second.boundary_margins_ns) == 20
    assert all(
        left >= 0 and right >= 0
        for left, right in first.boundary_margins_ns + second.boundary_margins_ns
    )
