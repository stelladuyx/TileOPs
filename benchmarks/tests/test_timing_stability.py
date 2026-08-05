"""Tests for stability-report aggregation and acceptance gates."""

from types import SimpleNamespace

from benchmarks.tools.timing_stability import _acceptance_failures, _aggregate


def _measurement(case, backend, round_index, latency):
    return {
        "case": case,
        "backend": backend,
        "round": round_index,
        "status": "ok",
        "latency_ms": latency,
        "raw_samples_ms": [latency],
    }


def test_acceptance_detects_direct_instability_and_backend_bias():
    measurements = []
    for round_index, direct_latency in enumerate((1.0, 1.1, 1.1, 1.4)):
        measurements.extend(
            [
                _measurement("case", "cupti-direct", round_index, direct_latency),
                _measurement("case", "kineto", round_index, 1.0),
                _measurement("case", "cuda-events", round_index, 2.0),
            ]
        )
    aggregate = _aggregate(measurements, ["case"], rounds=4)
    args = SimpleNamespace(
        max_failure_rate=0.0,
        max_cv=0.05,
        max_drift=0.05,
        max_direct_kineto_delta=0.05,
        clock_shift_risk_us=2.0,
    )

    failures = _acceptance_failures(aggregate, args)

    assert any("round CV" in failure for failure in failures)
    assert any("half-run drift" in failure for failure in failures)
    assert any("direct-sum/Kineto" in failure for failure in failures)


def test_acceptance_passes_stable_matching_measurements():
    measurements = []
    for round_index, direct_latency in enumerate((1.00, 1.01, 0.99, 1.00)):
        measurements.extend(
            [
                _measurement("case", "cupti-direct", round_index, direct_latency),
                _measurement("case", "kineto", round_index, 1.0),
                _measurement("case", "cuda-events", round_index, 2.0),
            ]
        )
    aggregate = _aggregate(measurements, ["case"], rounds=4)
    args = SimpleNamespace(
        max_failure_rate=0.0,
        max_cv=0.05,
        max_drift=0.05,
        max_direct_kineto_delta=0.05,
        clock_shift_risk_us=2.0,
    )

    assert _acceptance_failures(aggregate, args) == []


def test_acceptance_detects_clock_shift_boundary_risk():
    measurements = []
    for backend in ("cupti-direct", "kineto", "cuda-events"):
        item = _measurement("case", backend, 0, 1.0)
        if backend == "cupti-direct":
            item["direct_boundary_margins_ns"] = [(1_999, 4_000)]
        measurements.append(item)
    aggregate = _aggregate(measurements, ["case"], rounds=1)
    args = SimpleNamespace(
        max_failure_rate=0.0,
        max_cv=0.05,
        max_drift=0.05,
        max_direct_kineto_delta=0.05,
        clock_shift_risk_us=2.0,
    )

    failures = _acceptance_failures(aggregate, args)

    assert any("timestamp guard" in failure for failure in failures)
