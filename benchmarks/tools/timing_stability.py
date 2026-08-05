#!/usr/bin/env python3
"""Compare TileOps timing backends and retain raw stability evidence.

Run this on the same H200 runner used by Nightly.  Fallback is forcibly
disabled so a direct-CUPTI attribution failure is reported as a failure rather
than being hidden behind CUDA-events data.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from benchmarks.benchmark_base import _bench_meta, bench_kernel
from benchmarks.cupti_timing import get_cupti_runtime_info, load_cupti_api

BACKENDS = ("cupti-direct", "kineto", "cuda-events")


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "cv": None}
    mean = statistics.mean(values)
    return {
        "count": len(values),
        "mean": mean,
        "median": statistics.median(values),
        "stdev": statistics.pstdev(values),
        "cv": statistics.pstdev(values) / mean if mean > 0 else None,
        "min": min(values),
        "p10": _percentile(values, 0.10),
        "p90": _percentile(values, 0.90),
        "max": max(values),
    }


def _cases(
    device: str, *, include_tileops_l2: bool, include_cuda_graph: bool
) -> dict[str, tuple[Callable, tuple[Any, ...]]]:
    dtype = torch.bfloat16

    fast_a = torch.randn(4096, device=device, dtype=dtype)
    fast_b = torch.randn_like(fast_a)

    def fast_elementwise(a, b):
        torch.add(a, b, out=a)

    medium_a = torch.randn(1024, 1024, device=device, dtype=dtype)
    medium_b = torch.randn_like(medium_a)
    medium_out = torch.empty_like(medium_a)

    def medium_matmul(a, b, out):
        torch.mm(a, b, out=out)

    multi_a = torch.randn(65536, device=device, dtype=dtype)
    multi_b = torch.randn_like(multi_a)

    def multi_kernel(a, b):
        torch.add(a, b, out=a)
        torch.mul(a, b, out=a)

    cases = {
        "fast-single-kernel": (fast_elementwise, (fast_a, fast_b)),
        "medium-single-kernel": (medium_matmul, (medium_a, medium_b, medium_out)),
        "fast-multi-kernel": (multi_kernel, (multi_a, multi_b)),
    }
    if include_cuda_graph:
        graph_a = multi_a.clone()
        graph_b = multi_b.clone()
        capture_stream = torch.cuda.Stream(device=device)
        capture_stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(capture_stream):
            # Populate allocator and library state before capture.
            multi_kernel(graph_a, graph_b)
        capture_stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture_stream):
            multi_kernel(graph_a, graph_b)

        def multi_kernel_graph():
            graph.replay()

        cases["fast-multi-kernel-graph"] = (multi_kernel_graph, ())
    if include_tileops_l2:
        from tileops.ops.reduction.vector_norm import L2NormFwdOp

        l2_input = torch.randn(2048, 4096, device=device, dtype=dtype)
        tileops_l2 = L2NormFwdOp(dtype=dtype, dim=-1)

        def torch_l2(x):
            return torch.linalg.vector_norm(x.float(), ord=2, dim=-1).to(x.dtype)

        cases["tileops-l2norm"] = (tileops_l2, (l2_input,))
        cases["torch-l2norm"] = (torch_l2, (l2_input,))
    return cases


def _git_revision() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _measurement(
    case_name: str,
    backend: str,
    round_index: int,
    fn: Callable,
    fn_args: tuple[Any, ...],
    args: argparse.Namespace,
) -> dict[str, Any]:
    os.environ["TILEOPS_TIMING_BACKEND"] = backend
    os.environ["TILEOPS_ALLOW_CUDA_EVENTS_FALLBACK"] = "0"
    os.environ["TILEOPS_DIRECT_CUPTI_METRIC"] = args.direct_metric
    os.environ["TILEOPS_VALIDATE_CUPTI_WITH_CUDA_EVENTS"] = (
        "1" if backend == "cupti-direct" and args.validate_with_cuda_events else "0"
    )
    try:
        latency = bench_kernel(
            fn,
            args=fn_args,
            n_warmup=args.warmup,
            n_repeat=args.repeats,
            n_trials=args.trials,
        )
    except Exception as exc:
        return {
            "case": case_name,
            "backend": backend,
            "round": round_index,
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }

    return {
        "case": case_name,
        "backend": backend,
        "round": round_index,
        "status": "ok",
        "latency_ms": latency,
        "raw_samples_ms": list(getattr(_bench_meta, "raw_samples_ms", [])),
        "trial_reductions_ms": list(
            getattr(_bench_meta, "trial_reductions_ms", [])
        ),
        "timing_reduction": getattr(_bench_meta, "timing_reduction", None),
        "within_measurement_cv": getattr(_bench_meta, "timing_cv", None),
        "activity_sequence": list(
            getattr(_bench_meta, "expected_activity_sequence", [])
        ),
        "direct_boundary_margins_ns": list(
            getattr(_bench_meta, "direct_boundary_margins_ns", [])
        ),
        "direct_activity_sum_ms": list(
            getattr(_bench_meta, "direct_activity_sum_ms", [])
        ),
        "direct_activity_span_ms": list(
            getattr(_bench_meta, "direct_activity_span_ms", [])
        ),
        "direct_activity_union_busy_ms": list(
            getattr(_bench_meta, "direct_activity_union_busy_ms", [])
        ),
        "direct_inter_activity_idle_ms": list(
            getattr(_bench_meta, "direct_inter_activity_idle_ms", [])
        ),
        "direct_activity_overlap_ms": list(
            getattr(_bench_meta, "direct_activity_overlap_ms", [])
        ),
        "direct_inter_activity_gap_ms": list(
            getattr(_bench_meta, "direct_inter_activity_gap_ms", [])
        ),
        "direct_cuda_event_span_ms": list(
            getattr(_bench_meta, "direct_cuda_event_span_ms", [])
        ),
        "kineto_activity_sum_ms": list(
            getattr(_bench_meta, "kineto_activity_sum_ms", [])
        ),
        "kineto_activity_span_ms": list(
            getattr(_bench_meta, "kineto_activity_span_ms", [])
        ),
        "cupti_library_path": getattr(_bench_meta, "cupti_library_path", None),
        "cupti_api_version": getattr(_bench_meta, "cupti_api_version", None),
    }


def _aggregate(
    measurements: list[dict[str, Any]],
    cases: list[str],
    rounds: int,
    clock_shift_risk_us: float = 2.0,
) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    for case_name in cases:
        aggregate[case_name] = {}
        for backend in BACKENDS:
            selected = [
                measurement
                for measurement in measurements
                if measurement["case"] == case_name
                and measurement["backend"] == backend
            ]
            successful = [item for item in selected if item["status"] == "ok"]
            latencies = [item["latency_ms"] for item in successful]
            raw_samples = [
                sample
                for item in successful
                for sample in item["raw_samples_ms"]
            ]
            boundary_margins = [
                margin
                for item in successful
                for margin in item.get("direct_boundary_margins_ns", [])
            ]
            left_margins = [margin[0] for margin in boundary_margins]
            right_margins = [margin[1] for margin in boundary_margins]
            activity_sums = [
                sample
                for item in successful
                for sample in item.get("direct_activity_sum_ms", [])
            ]
            activity_spans = [
                sample
                for item in successful
                for sample in item.get("direct_activity_span_ms", [])
            ]
            inter_activity_gaps = [
                sample
                for item in successful
                for sample in item.get("direct_inter_activity_gap_ms", [])
            ]
            activity_union_busy = [
                sample
                for item in successful
                for sample in item.get("direct_activity_union_busy_ms", [])
            ]
            inter_activity_idle = [
                sample
                for item in successful
                for sample in item.get("direct_inter_activity_idle_ms", [])
            ]
            activity_overlap = [
                sample
                for item in successful
                for sample in item.get("direct_activity_overlap_ms", [])
            ]
            cuda_event_spans = [
                sample
                for item in successful
                for sample in item.get("direct_cuda_event_span_ms", [])
            ]
            event_enclosure_margins = [
                event_span - activity_span
                for item in successful
                if item.get("direct_cuda_event_span_ms")
                for event_span, activity_span in zip(
                    item.get("direct_cuda_event_span_ms", []),
                    item.get("direct_activity_span_ms", []),
                    strict=True,
                )
            ]
            round_activity_sums = [
                statistics.mean(item["direct_activity_sum_ms"])
                for item in successful
                if item.get("direct_activity_sum_ms")
            ]
            round_activity_spans = [
                statistics.mean(item["direct_activity_span_ms"])
                for item in successful
                if item.get("direct_activity_span_ms")
            ]
            round_inter_activity_gaps = [
                statistics.mean(item["direct_inter_activity_gap_ms"])
                for item in successful
                if item.get("direct_inter_activity_gap_ms")
            ]
            kineto_activity_sums = [
                sample
                for item in successful
                for sample in item.get("kineto_activity_sum_ms", [])
            ]
            kineto_activity_spans = [
                sample
                for item in successful
                for sample in item.get("kineto_activity_span_ms", [])
            ]
            round_kineto_activity_sums = [
                statistics.mean(item["kineto_activity_sum_ms"])
                for item in successful
                if item.get("kineto_activity_sum_ms")
            ]
            round_kineto_activity_spans = [
                statistics.mean(item["kineto_activity_span_ms"])
                for item in successful
                if item.get("kineto_activity_span_ms")
            ]
            first_half = latencies[: max(1, len(latencies) // 2)]
            second_half = latencies[len(latencies) // 2 :]
            drift_ratio = None
            if first_half and second_half:
                drift_ratio = statistics.median(second_half) / statistics.median(first_half)
            aggregate[case_name][backend] = {
                "successes": len(successful),
                "failures": rounds - len(successful),
                "success_rate": len(successful) / rounds,
                "round_latency_ms": _summary(latencies),
                "raw_samples_ms": _summary(raw_samples),
                "left_boundary_margin_ns": _summary(left_margins),
                "right_boundary_margin_ns": _summary(right_margins),
                "direct_activity_sum_ms": _summary(activity_sums),
                "direct_activity_span_ms": _summary(activity_spans),
                "direct_activity_union_busy_ms": _summary(activity_union_busy),
                "direct_inter_activity_idle_ms": _summary(inter_activity_idle),
                "direct_activity_overlap_ms": _summary(activity_overlap),
                "direct_cuda_event_span_ms": _summary(cuda_event_spans),
                "direct_cuda_event_enclosure_margin_ms": _summary(
                    event_enclosure_margins
                ),
                "direct_inter_activity_gap_ms": _summary(inter_activity_gaps),
                "round_direct_activity_sum_ms": _summary(round_activity_sums),
                "round_direct_activity_span_ms": _summary(round_activity_spans),
                "round_direct_inter_activity_gap_ms": _summary(
                    round_inter_activity_gaps
                ),
                "kineto_activity_sum_ms": _summary(kineto_activity_sums),
                "kineto_activity_span_ms": _summary(kineto_activity_spans),
                "round_kineto_activity_sum_ms": _summary(
                    round_kineto_activity_sums
                ),
                "round_kineto_activity_span_ms": _summary(
                    round_kineto_activity_spans
                ),
                "clock_shift_unsafe_samples": sum(
                    left < clock_shift_risk_us * 1_000
                    or right < clock_shift_risk_us * 1_000
                    for left, right in boundary_margins
                ),
                "first_to_second_half_median_ratio": drift_ratio,
                "errors": [item for item in selected if item["status"] == "error"],
            }
    return aggregate


def _acceptance_failures(aggregate: dict[str, Any], args: argparse.Namespace) -> list[str]:
    failures = []
    for case_name, backends in aggregate.items():
        direct = backends["cupti-direct"]
        unsafe_samples = direct["clock_shift_unsafe_samples"]
        if unsafe_samples:
            failures.append(
                f"{case_name}: {unsafe_samples} direct CUPTI samples have less "
                f"than {args.clock_shift_risk_us:g} us timestamp guard"
            )
        if direct["success_rate"] < 1 - args.max_failure_rate:
            failures.append(
                f"{case_name}: direct CUPTI success rate "
                f"{direct['success_rate']:.1%} is below {1 - args.max_failure_rate:.1%}"
            )
        cv = direct["round_latency_ms"]["cv"]
        if cv is not None and cv > args.max_cv:
            failures.append(
                f"{case_name}: direct CUPTI round CV {cv:.2%} exceeds {args.max_cv:.2%}"
            )
        drift = direct["first_to_second_half_median_ratio"]
        if drift is not None and abs(drift - 1.0) > args.max_drift:
            failures.append(
                f"{case_name}: direct CUPTI half-run drift {drift:.3f} exceeds "
                f"±{args.max_drift:.1%}"
            )
        # Validate direct attribution/durations against Kineto independently;
        # validate span enclosure from the same direct capture below.
        direct_sum_median = direct["round_direct_activity_sum_ms"]["median"]
        direct_sum_median = (
            direct_sum_median
            if direct_sum_median is not None
            else direct["round_latency_ms"]["median"]
        )
        kineto = backends["kineto"]
        kineto_sum_median = kineto["round_kineto_activity_sum_ms"]["median"]
        kineto_sum_median = (
            kineto_sum_median
            if kineto_sum_median is not None
            else kineto["round_latency_ms"]["median"]
        )
        if direct_sum_median is not None and kineto_sum_median is not None:
            ratio = direct_sum_median / kineto_sum_median
            if abs(ratio - 1.0) > args.max_direct_kineto_delta:
                failures.append(
                    f"{case_name}: direct-sum/Kineto median ratio {ratio:.3f} exceeds "
                    f"±{args.max_direct_kineto_delta:.1%}"
                )
        # Span is sensitive to instrumentation-induced host launch gaps, so a
        # separate Kineto capture is not a valid oracle. Compare against CUDA
        # events inserted around the exact same execution instead.
        if args.validate_with_cuda_events:
            event_margin = direct["direct_cuda_event_enclosure_margin_ms"]
            if event_margin["count"] == 0:
                failures.append(f"{case_name}: no same-capture CUDA-event span control")
            elif event_margin["min"] < -args.max_event_containment_error_us * 1e-3:
                failures.append(
                    f"{case_name}: CUPTI activity span exceeds its same-capture "
                    f"CUDA-event enclosure by {-event_margin['min'] * 1e3:.3f} us"
                )
    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--trials", type=int, default=9)
    parser.add_argument("--max-cv", type=float, default=0.05)
    parser.add_argument("--max-drift", type=float, default=0.05)
    parser.add_argument("--max-failure-rate", type=float, default=0.0)
    parser.add_argument("--max-direct-kineto-delta", type=float, default=0.10)
    parser.add_argument(
        "--max-event-containment-error-us", type=float, default=2.0
    )
    parser.add_argument(
        "--validate-with-cuda-events",
        action="store_true",
        help="insert enclosing events for a correctness run; perturbs timing",
    )
    parser.add_argument(
        "--clock-shift-risk-us",
        type=float,
        default=2.0,
        help="fail if a direct activity span is closer than this to either window boundary",
    )
    parser.add_argument(
        "--direct-metric",
        choices=("activity-sum", "activity-span"),
        default="activity-span",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("timing_stability.json")
    )
    parser.add_argument("--include-tileops-l2", action="store_true")
    parser.add_argument("--include-cuda-graph", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        print("CUDA device is unavailable", file=sys.stderr)
        return 2

    # Fail before running a long experiment if the isolated environment is not
    # actually capable of loading the direct CUPTI API.
    load_cupti_api().get_timestamp()
    cupti_runtime = get_cupti_runtime_info()
    cases = _cases(
        args.device,
        include_tileops_l2=args.include_tileops_l2,
        include_cuda_graph=args.include_cuda_graph,
    )
    measurements: list[dict[str, Any]] = []
    old_backend = os.environ.get("TILEOPS_TIMING_BACKEND")
    old_fallback = os.environ.get("TILEOPS_ALLOW_CUDA_EVENTS_FALLBACK")
    old_direct_metric = os.environ.get("TILEOPS_DIRECT_CUPTI_METRIC")
    old_event_validation = os.environ.get("TILEOPS_VALIDATE_CUPTI_WITH_CUDA_EVENTS")
    try:
        for round_index in range(args.rounds):
            # Rotate backend order to reduce temperature/order bias.
            offset = round_index % len(BACKENDS)
            backend_order = BACKENDS[offset:] + BACKENDS[:offset]
            for case_name, (fn, fn_args) in cases.items():
                for backend in backend_order:
                    measurement = _measurement(
                        case_name, backend, round_index, fn, fn_args, args
                    )
                    measurements.append(measurement)
                    status = measurement["status"]
                    latency = measurement.get("latency_ms")
                    detail = f"{latency:.6f} ms" if latency is not None else measurement["error"]
                    print(
                        f"round={round_index:02d} case={case_name:<22} "
                        f"backend={backend:<13} status={status:<5} {detail}",
                        flush=True,
                    )
    finally:
        if old_backend is None:
            os.environ.pop("TILEOPS_TIMING_BACKEND", None)
        else:
            os.environ["TILEOPS_TIMING_BACKEND"] = old_backend
        if old_fallback is None:
            os.environ.pop("TILEOPS_ALLOW_CUDA_EVENTS_FALLBACK", None)
        else:
            os.environ["TILEOPS_ALLOW_CUDA_EVENTS_FALLBACK"] = old_fallback
        if old_direct_metric is None:
            os.environ.pop("TILEOPS_DIRECT_CUPTI_METRIC", None)
        else:
            os.environ["TILEOPS_DIRECT_CUPTI_METRIC"] = old_direct_metric
        if old_event_validation is None:
            os.environ.pop("TILEOPS_VALIDATE_CUPTI_WITH_CUDA_EVENTS", None)
        else:
            os.environ["TILEOPS_VALIDATE_CUPTI_WITH_CUDA_EVENTS"] = old_event_validation

    aggregate = _aggregate(
        measurements, list(cases), args.rounds, args.clock_shift_risk_us
    )
    failures = _acceptance_failures(aggregate, args)
    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_revision": _git_revision(),
        "environment": {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "device": args.device,
            "gpu": torch.cuda.get_device_name(args.device),
            "cupti_library_path": cupti_runtime.library_path,
            "cupti_api_version": cupti_runtime.api_version,
        },
        "config": vars(args) | {"output": str(args.output)},
        "measurements": measurements,
        "aggregate": aggregate,
        "acceptance_failures": failures,
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")
    for failure in failures:
        print(f"FAIL: {failure}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
