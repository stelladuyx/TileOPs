import contextlib
import logging
import os
import statistics
import subprocess
import sys
import threading
from abc import ABC, abstractmethod
from collections import Counter
from datetime import datetime
from typing import (
    Any,
    Callable,
    Generic,
    Optional,
    Protocol,
    TypeVar,
    runtime_checkable,
)

import pytest
import torch
from torch.autograd.profiler import DeviceType

from tileops.manifest import (
    WORKLOAD_RESERVED_KEYS,
    load_manifest,
    load_workloads,
    single_input_workload_contract,
)


def _workload_contract(op_name: str) -> tuple[str, frozenset[str]]:
    """Resolve the shared workload contract for an op known to exist."""
    sig = load_manifest()[op_name].get("signature") or {}
    contract = single_input_workload_contract(sig)
    if contract is None:
        raise KeyError(
            f"workloads_to_params({op_name!r}) needs exactly one manifest "
            "tensor input; multi-input ops use their own bench files."
        )
    return contract


# Benchmark capability protocols


@runtime_checkable
class ShapeDtypeWorkload(Protocol):
    """Structural type for workloads that carry shape and dtype metadata.

    Any object with ``shape`` and ``dtype`` satisfies this protocol.
    Used by helpers that only need tensor metadata, not input generation
    capability.
    """

    shape: tuple[int, ...]
    dtype: torch.dtype


@runtime_checkable
class InputGeneratingWorkload(Protocol):
    """Structural type for workloads that can generate benchmark inputs."""

    def gen_inputs(self) -> tuple[Any, ...]: ...


@runtime_checkable
class BenchmarkWorkload(ShapeDtypeWorkload, InputGeneratingWorkload, Protocol):
    """Full benchmark workload: shape/dtype metadata + input generation.

    This is the standard contract for benchmark workloads that need both
    roofline metadata extraction and input tensor generation.
    Workloads satisfy this protocol when they define ``shape`` and ``dtype``
    metadata in addition to implementing ``gen_inputs()``.
    """

    ...


W = TypeVar("W")


_logger = logging.getLogger("tileops.bench")

# Thread-local storage for conftest hook to pick up per-test bench results.
# A single test function may call record() multiple times (tileops + baseline).
_bench_results = threading.local()

# Latest bench_kernel measurement metadata; deviations from the default
# protocol are surfaced in results by BenchmarkBase._build_result.
_bench_meta = threading.local()


class _CuptiProjectionError(Exception):
    """CUPTI trace lacked a projected annotation window for every repeat."""


# Name of the ``record_function`` annotation wrapping the timed call. Kineto
# projects this scope onto the device timeline. The L2-flush ``cache.zero_()``
# is synchronized to completion before the window opens (see ``bench_kernel``),
# so its device event cannot fall inside a window regardless of how the
# projection behaves; kernels the timed call launches do.
_KERNEL_REGION = "tileops_bench_kernel"


def _sum_kernel_time_us(kineto_results):
    """Sum device time of the kernels the timed call launched.

    Sums only kernels inside a :data:`_KERNEL_REGION` annotation window, so the
    L2-flush fill is excluded and the kernel under test is counted regardless of
    its name. A call launching several kernels contributes all of them.

    Iterates the C++ Kineto events directly to bypass ``key_averages()``, which
    is ~16x slower (~130ms of Python parsing/tree-building) for large traces.

    Returns:
        ``(total_us, n_regions)``: summed kernel time in microseconds and the
        number of annotation windows. The caller checks ``n_regions ==
        n_repeat`` to confirm the scope projected on every iteration.
    """
    import bisect

    windows: list[tuple[int, int]] = []
    kernels: list[tuple[int, int]] = []  # (start_ns, duration_ns)
    for evt in kineto_results.events():
        if evt.device_type() != DeviceType.CUDA:
            continue
        if evt.is_user_annotation():
            if evt.name() == _KERNEL_REGION:
                windows.append((evt.start_ns(), evt.end_ns()))
            continue
        kernels.append((evt.start_ns(), evt.duration_ns()))

    windows.sort()
    starts = [w[0] for w in windows]
    ends = [w[1] for w in windows]
    total_us = 0.0
    for start_ns, dur_ns in kernels:
        # Count only kernels that fall inside a timed-call window; everything
        # outside (notably the L2-flush fill) is excluded.
        idx = bisect.bisect_right(starts, start_ns) - 1
        if idx >= 0 and start_ns < ends[idx]:
            total_us += dur_ns / 1000.0
    return total_us, len(windows)


# L2 cache flush buffer (sized to actual L2, allocated lazily)

_l2_flush_cache: Optional[torch.Tensor] = None


def _get_l2_flush_cache() -> torch.Tensor:
    global _l2_flush_cache
    if _l2_flush_cache is None:
        l2_bytes = torch.cuda.get_device_properties(0).L2_cache_size
        if l2_bytes <= 0:
            _logger.warning(
                "L2 cache size query returned %d; flushing a 256 MB buffer "
                "instead", l2_bytes,
            )
            l2_bytes = int(256e6)
        _l2_flush_cache = torch.empty(l2_bytes // 4, dtype=torch.int, device="cuda")
    return _l2_flush_cache


def _native_output_suppressor():
    """Return an fd-level output suppressor that is safe under pytest capture.

    tilelang's ``suppress_stdout_stderr`` dup2's ``/dev/null`` over
    ``sys.stdout.fileno()``; under pytest fd capture that fileno is the
    capture tmpfile and the redirect corrupts it (``EBADF`` on later reads).
    Suppress only when stdout/stderr are the process fds 1/2.
    """
    try:
        native = sys.stdout.fileno() == 1 and sys.stderr.fileno() == 2
    except (AttributeError, OSError, ValueError):
        # Streams without a real descriptor (io.StringIO, capsys) or with
        # fileno() unsupported: fd-level suppression is impossible.
        native = False
    if not native:
        return contextlib.nullcontext()
    from tilelang.profiler.bench import suppress_stdout_stderr
    return suppress_stdout_stderr()


_TIMING_BACKENDS = frozenset({"cupti-direct", "kineto", "cuda-events"})


def _bench_with_direct_cupti(
    run_iteration: Callable[[int], Any],
    cache: torch.Tensor,
    n_repeat: int,
    n_trials: int,
) -> tuple[list[float], list[float], list[Any]]:
    """Run SOL-style direct CUPTI activity timing without Kineto."""
    from benchmarks.cupti_timing import measure_direct_cupti

    metric = os.getenv("TILEOPS_DIRECT_CUPTI_METRIC", "activity-sum")
    trial_means: list[float] = []
    raw_samples: list[float] = []
    expected_sequence: list[Any] = []
    boundary_margins_ns: list[tuple[int, int]] = []
    activity_sum_ms: list[float] = []
    activity_span_ms: list[float] = []
    activity_union_busy_ms: list[float] = []
    inter_activity_idle_ms: list[float] = []
    activity_overlap_ms: list[float] = []
    inter_activity_gap_ms: list[float] = []

    def prepare_iteration(_iteration: int) -> None:
        cache.zero_()
        # Keep cache/setup activity outside the CUPTI timestamp window.
        torch.cuda.synchronize()

    for _ in range(n_trials):
        measurement = measure_direct_cupti(
            run_iteration,
            prepare_iteration,
            n_repeat,
            metric=metric,
        )
        if not expected_sequence:
            expected_sequence = measurement.expected_sequence
        elif Counter(measurement.expected_sequence) != Counter(expected_sequence):
            from benchmarks.cupti_timing import DirectCuptiTraceError

            raise DirectCuptiTraceError(
                "discovery activity sequence changed between timing trials"
            )
        raw_samples.extend(measurement.samples_ms)
        boundary_margins_ns.extend(measurement.boundary_margins_ns)
        activity_sum_ms.extend(measurement.activity_sum_ms)
        activity_span_ms.extend(measurement.activity_span_ms)
        activity_union_busy_ms.extend(measurement.activity_union_busy_ms)
        inter_activity_idle_ms.extend(measurement.inter_activity_idle_ms)
        activity_overlap_ms.extend(measurement.activity_overlap_ms)
        inter_activity_gap_ms.extend(measurement.inter_activity_gap_ms)
        trial_means.append(statistics.mean(measurement.samples_ms))
    _bench_meta.direct_boundary_margins_ns = boundary_margins_ns
    _bench_meta.direct_activity_sum_ms = activity_sum_ms
    _bench_meta.direct_activity_span_ms = activity_span_ms
    _bench_meta.direct_activity_union_busy_ms = activity_union_busy_ms
    _bench_meta.direct_inter_activity_idle_ms = inter_activity_idle_ms
    _bench_meta.direct_activity_overlap_ms = activity_overlap_ms
    _bench_meta.direct_inter_activity_gap_ms = inter_activity_gap_ms
    return trial_means, raw_samples, expected_sequence


def _bench_with_kineto(
    run_iteration: Callable[[int], Any],
    cache: torch.Tensor,
    n_repeat: int,
    n_trials: int,
) -> tuple[list[float], list[float], list[Any]]:
    """Run the legacy torch.profiler/Kineto projection timing path."""
    trial_means: list[float] = []
    for _ in range(n_trials):
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
        ) as profiler:
            for iteration in range(n_repeat):
                cache.zero_()
                torch.cuda.synchronize()
                with torch.profiler.record_function(_KERNEL_REGION):
                    run_iteration(iteration)
                torch.cuda.synchronize()
        total_us, n_regions = _sum_kernel_time_us(profiler.profiler.kineto_results)
        if n_regions != n_repeat:
            n_cuda_kernels = sum(
                1
                for event in profiler.profiler.kineto_results.events()
                if event.device_type() == DeviceType.CUDA and not event.is_user_annotation()
            )
            raise _CuptiProjectionError(
                f"{n_regions}/{n_repeat} annotation windows projected, "
                f"{n_cuda_kernels} CUDA kernels captured"
            )
        trial_means.append((total_us / n_repeat) * 1e-3)
    # Kineto exposes only the per-trial aggregate through this optimized parser.
    return trial_means, list(trial_means), []


def _bench_with_cuda_events(
    run_iteration: Callable[[int], Any],
    cache: torch.Tensor,
    n_repeat: int,
    n_trials: int,
) -> tuple[list[float], list[float], list[Any]]:
    """Run the diagnostic CUDA-events fallback and retain raw samples."""
    trial_means: list[float] = []
    raw_samples: list[float] = []
    for _ in range(n_trials):
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]

        for iteration in range(n_repeat):
            cache.zero_()
            torch.cuda.synchronize()
            start_events[iteration].record()
            run_iteration(iteration)
            end_events[iteration].record()
        torch.cuda.synchronize()

        samples = [
            start.elapsed_time(end)
            for start, end in zip(start_events, end_events, strict=True)
        ]
        raw_samples.extend(samples)
        trial_means.append(statistics.mean(samples))
    return trial_means, raw_samples, []


# NVIDIA SOL-ExecBench–style benchmark

def bench_kernel(
    fn: Callable,
    args: tuple[Any, ...] = (),
    n_warmup: int = 10,
    n_repeat: int = 50,
    n_trials: int = 3,
) -> float:
    """Benchmark a GPU callable with selectable GPU timing backends.

    Protocol (adapted from NVIDIA SOL-ExecBench, arxiv.org/abs/2603.19173):
      1. Lock GPU clocks externally (nvidia-smi).
      2. Run *n_warmup* un-timed iterations with L2 flush.
      3. For each of *n_trials* trials, profile *n_repeat* iterations
         with direct CUPTI activity timing (no Kineto projection).
         L2 is flushed before every iteration.  Input tensors are cloned
         each iteration so the kernel always sees fresh addresses.
      4. Report the median trial mean (robust to outlier trials).

    The default ``cupti-direct`` backend is adapted from NVIDIA SOL-ExecBench:
    it discovers the callable's GPU activity sequence, records per-iteration
    windows using CUPTI timestamps, and validates that every window contains a
    complete sequence. Set ``TILEOPS_TIMING_BACKEND=kineto`` for an A/B against
    the legacy projection path, or ``cuda-events`` for diagnostics.

    Args:
        fn: Callable to benchmark.  If *args* is provided, called as
            ``fn(*cloned_args)``; otherwise called as ``fn()``.
        args: Tensor arguments to clone each iteration.  Non-tensor
            values are passed through unchanged.
        n_warmup: Warmup iterations (default 10).
        n_repeat: Timed iterations per trial (default 50).
        n_trials: Independent trials (default 3).

    Returns:
        Kernel latency in **milliseconds**.
    """
    if not isinstance(args, tuple):
        raise TypeError(
            f"bench_kernel expects a tuple of args, got {type(args).__name__}. "
            "Check that gen_inputs() returns a tuple."
        )
    if n_warmup < 0:
        raise ValueError(f"n_warmup must be non-negative, got {n_warmup}")
    if n_repeat <= 0:
        raise ValueError(f"n_repeat must be positive, got {n_repeat}")
    if n_trials <= 0:
        raise ValueError(f"n_trials must be positive, got {n_trials}")

    backend = os.getenv("TILEOPS_TIMING_BACKEND", "cupti-direct")
    if backend not in _TIMING_BACKENDS:
        raise ValueError(
            f"unknown TILEOPS_TIMING_BACKEND={backend!r}; "
            f"expected one of {sorted(_TIMING_BACKENDS)}"
        )

    cache = _get_l2_flush_cache()
    has_args = len(args) > 0

    # Pre-clone a small pool of input tensors so the kernel sees different
    # addresses across iterations.  Skip cloning if total tensor memory
    # exceeds 1 GB to avoid OOM on large workloads.
    _N_CLONES = 3
    _MAX_CLONE_BYTES = 1 << 30  # 1 GB
    if has_args:
        tensor_mask = tuple(isinstance(a, torch.Tensor) for a in args)
        total_bytes = sum(a.nelement() * a.element_size()
                          for a, m in zip(args, tensor_mask, strict=True) if m)
        if total_bytes * _N_CLONES <= _MAX_CLONE_BYTES:
            arg_pool = [
                tuple(a.clone() if m else a for a, m in zip(args, tensor_mask, strict=True))
                for _ in range(_N_CLONES)
            ]
            def _run(i):
                return fn(*arg_pool[i % _N_CLONES])
        else:
            _logger.warning(
                "bench_kernel: inputs total %.2f GiB; skipping per-iteration "
                "cloning (kernel sees identical addresses)",
                total_bytes / (1 << 30),
            )
            arg_pool = None
            def _run(i):
                return fn(*args)
    else:
        arg_pool = None
        def _run(i):
            return fn()
    _bench_meta.inputs_cloned = arg_pool is not None or not has_args

    try:
        # Prevent diagnostics from a prior direct-CUPTI call leaking into a
        # Kineto/CUDA-events measurement or a failed direct attempt.
        _bench_meta.direct_boundary_margins_ns = []
        _bench_meta.direct_activity_sum_ms = []
        _bench_meta.direct_activity_span_ms = []
        _bench_meta.direct_activity_union_busy_ms = []
        _bench_meta.direct_inter_activity_idle_ms = []
        _bench_meta.direct_activity_overlap_ms = []
        _bench_meta.direct_inter_activity_gap_ms = []
        # Warmup is deliberately outside all profiler/activity contexts.
        for iteration in range(n_warmup):
            cache.zero_()
            _run(iteration)
        torch.cuda.synchronize()

        try:
            with _native_output_suppressor():
                if backend == "cupti-direct":
                    trial_means, raw_samples, expected_sequence = _bench_with_direct_cupti(
                        _run, cache, n_repeat, n_trials
                    )
                elif backend == "kineto":
                    trial_means, raw_samples, expected_sequence = _bench_with_kineto(
                        _run, cache, n_repeat, n_trials
                    )
                else:
                    trial_means, raw_samples, expected_sequence = _bench_with_cuda_events(
                        _run, cache, n_repeat, n_trials
                    )
            actual_backend = backend
            fallback_reason = None
        except _CuptiProjectionError as exc:
            measurement_error: Exception = exc
        except Exception as exc:
            # Only expected direct-CUPTI attribution/dependency failures may
            # enter the configured fallback. Kernel/runtime errors propagate.
            if backend != "cupti-direct":
                raise
            from benchmarks.cupti_timing import DirectCuptiError

            if not isinstance(exc, DirectCuptiError):
                raise
            measurement_error = exc
        else:
            measurement_error = None

        if measurement_error is not None:
            allow_fallback = os.getenv("TILEOPS_ALLOW_CUDA_EVENTS_FALLBACK", "1") == "1"
            if not allow_fallback:
                raise RuntimeError(
                    f"{backend} timing failed: {measurement_error}. "
                    "CUDA-events fallback is disabled "
                    "(TILEOPS_ALLOW_CUDA_EVENTS_FALLBACK=0)."
                ) from measurement_error

            _logger.warning(
                "%s timing failed (%s); falling back to CUDA events. "
                "Set TILEOPS_ALLOW_CUDA_EVENTS_FALLBACK=0 to fail closed.",
                backend,
                measurement_error,
            )
            trial_means, raw_samples, expected_sequence = _bench_with_cuda_events(
                _run, cache, n_repeat, n_trials
            )
            actual_backend = "cuda-events"
            fallback_reason = f"{type(measurement_error).__name__}: {measurement_error}"

        _bench_meta.timing = actual_backend
        _bench_meta.requested_timing = backend
        _bench_meta.raw_samples_ms = list(raw_samples)
        _bench_meta.trial_means_ms = list(trial_means)
        _bench_meta.expected_activity_sequence = list(expected_sequence)
        _bench_meta.timing_metric = (
            os.getenv("TILEOPS_DIRECT_CUPTI_METRIC", "activity-sum")
            if actual_backend == "cupti-direct"
            else "trial-mean" if actual_backend == "kineto" else "event-elapsed"
        )
        _bench_meta.fallback_reason = fallback_reason
        if raw_samples and statistics.mean(raw_samples) > 0:
            _bench_meta.timing_cv = statistics.pstdev(raw_samples) / statistics.mean(raw_samples)
        else:
            _bench_meta.timing_cv = 0.0

        trial_means.sort()
        return trial_means[len(trial_means) // 2]
    finally:
        # Release clone storage even when profiling or the callable fails.
        if arg_pool is not None:
            del arg_pool
        torch.cuda.empty_cache()


def _get_env_metadata() -> list[str]:
    """Collect GPU model, driver version, CUDA version, and torch version."""
    lines = []
    lines.append(f"- **Torch version**: {torch.__version__}")
    lines.append(f"- **CUDA version (torch)**: {torch.version.cuda or 'N/A'}")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        lines.append(f"- **GPU model**: {gpu_name}")
    else:
        lines.append("- **GPU model**: N/A (no CUDA device)")

    # Try to get NVIDIA driver version and clocks from nvidia-smi.
    gpu_query_fields = [
        "driver_version",
        "clocks.current.sm",
        "clocks.current.memory",
        "clocks.applications.graphics",
        "clocks.applications.memory",
    ]
    gpu_query_values = []
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={','.join(gpu_query_fields)}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            gpu_query_values = [
                part.strip() for part in result.stdout.splitlines()[0].split(",")
            ]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    driver = gpu_query_values[0] if len(gpu_query_values) == len(gpu_query_fields) else "N/A"
    lines.append(f"- **Driver version**: {driver}")

    if len(gpu_query_values) == len(gpu_query_fields):
        sm_clock, mem_clock, app_sm_clock, app_mem_clock = gpu_query_values[1:]
        lines.append(
            "- **GPU clocks**: "
            f"SM current {sm_clock} MHz, memory current {mem_clock} MHz, "
            f"application SM {app_sm_clock} MHz, "
            f"application memory {app_mem_clock} MHz"
        )

    return lines


class BenchmarkBase(Generic[W], ABC):
    """Abstract base class for op benchmarking.

    Generic over workload type so subclasses can declare the exact
    capability they need.  ``WorkloadBase`` remains the typical in-repo
    implementation, but the public contract is the type parameter.

    Subclass must implement calculate_flops() and calculate_memory().
    """

    def __init__(self, workload: W):
        self.workload = workload

    @abstractmethod
    def calculate_flops(self) -> Optional[float]:
        raise NotImplementedError

    @abstractmethod
    def calculate_memory(self) -> Optional[float]:
        raise NotImplementedError

    def profile(self,
                functor: Any,
                *inputs: Any) -> dict:
        """Profile a callable and return structured results.

        Uses the NVIDIA SOL-ExecBench protocol: CUPTI kernel timing,
        10 warmup, 50 repeats × 3 trials, L2 flush sized to actual
        cache, input tensors cloned each iteration.
        """
        with torch.no_grad():
            latency = bench_kernel(functor, args=inputs)
        return self._build_result(latency)

    def profile_autograd(self, functor: Any) -> dict:
        """Profile a callable that requires autograd (e.g. fwd+bwd).

        Same as profile() but without torch.no_grad(), so the callable
        can build autograd graphs and call .backward() internally.
        The functor must be a zero-arg closure that captures its inputs.
        """
        latency = bench_kernel(functor)
        return self._build_result(latency)

    def _build_result(self, latency: float) -> dict:
        result = {"latency_ms": latency}
        # Timing provenance is part of benchmark correctness: never mix direct
        # CUPTI, Kineto, and CUDA-events records without an explicit backend.
        timing = getattr(_bench_meta, "timing", None)
        if timing is not None:
            result["timing"] = timing
        requested_timing = getattr(_bench_meta, "requested_timing", None)
        if requested_timing is not None and requested_timing != timing:
            result["timing_requested"] = requested_timing
        fallback_reason = getattr(_bench_meta, "fallback_reason", None)
        if fallback_reason:
            result["timing_fallback_reason"] = fallback_reason
        if os.getenv("TILEOPS_RECORD_TIMING_DIAGNOSTICS", "0") == "1":
            raw_samples = getattr(_bench_meta, "raw_samples_ms", [])
            result["timing_sample_count"] = len(raw_samples)
            result["timing_cv"] = getattr(_bench_meta, "timing_cv", None)
            result["timing_metric"] = getattr(_bench_meta, "timing_metric", None)
            result["timing_trial_means_ms"] = getattr(
                _bench_meta, "trial_means_ms", []
            )
            boundary_margins = getattr(
                _bench_meta, "direct_boundary_margins_ns", []
            )
            if boundary_margins:
                result["timing_left_boundary_min_us"] = min(
                    margin[0] for margin in boundary_margins
                ) / 1_000
                result["timing_right_boundary_min_us"] = min(
                    margin[1] for margin in boundary_margins
                ) / 1_000
                result["timing_clock_shift_unsafe_2us"] = sum(
                    left < 2_000 or right < 2_000
                    for left, right in boundary_margins
                )
        if getattr(_bench_meta, "inputs_cloned", True) is False:
            result["inputs_cloned"] = False
        flops = self.calculate_flops()
        if flops is not None:
            result["tflops"] = flops / latency * 1e-9
        memory = self.calculate_memory()
        if memory is not None:
            result["bandwidth_tbs"] = memory / latency * 1e-9
        return result


# Manifest-driven benchmark helpers


def _workload_extra_params(w: dict, shape_key: str) -> dict[str, Any]:
    """Return op-call params on a workload entry, stripping reserved keys."""
    reserved = WORKLOAD_RESERVED_KEYS | {shape_key}
    return {
        k: v
        for k, v in w.items()
        if isinstance(k, str) and k not in reserved and not k.startswith("__")
    }


def workloads_to_params(op_name: str, include_extra: bool = False) -> list:
    """Convert manifest workload dicts for *op_name* to pytest params.

    Each entry becomes ``pytest.param(shape, dtype, id=...)``; with
    ``include_extra=True`` a third element carries the op-call params
    declared on the workload entry (e.g. ``{"dim": 0}``).
    """
    workloads = load_workloads(op_name)  # canonical not-found error
    shape_key, allowed = _workload_contract(op_name)
    params = []
    for w in workloads:
        if shape_key not in w:
            raise KeyError(
                f"workload {w.get('label', w)!r} of {op_name!r} is missing "
                f"{shape_key!r} (derived from the signature's input name)."
            )
        unknown = sorted(
            repr(k) for k in w
            if not isinstance(k, str) or (k not in allowed and not k.startswith("__"))
        )
        if unknown:
            raise KeyError(
                f"workload {w.get('label', w)!r} of {op_name!r} has unknown "
                f"keys {unknown}; allowed: {sorted(allowed)}."
            )
        shape = tuple(w[shape_key])
        label = w.get("label", "x".join(str(s) for s in shape))
        extra = _workload_extra_params(w, shape_key) if include_extra else {}
        for dtype_str in w["dtypes"]:
            dtype = getattr(torch, dtype_str)
            # Copy ``extra`` per parametrization so mutation in one test case
            # cannot leak into later cases sharing the workload entry.
            param_args = (
                (shape, dtype, dict(extra))
                if include_extra
                else (shape, dtype)
            )
            params.append(pytest.param(*param_args, id=f"{label}-{dtype_str}"))
    return params


def workload_field_params(workloads: list, keys: tuple) -> list:
    """Turn manifest workload dicts into pytest params.

    First workload is marked ``smoke``, the rest ``full``. Keys ending in
    ``dtype`` are resolved to ``torch.dtype`` values.
    """
    params = []
    for i, w in enumerate(workloads):
        args = [getattr(torch, w[k]) if k.endswith("dtype") else w[k] for k in keys]
        params.append(
            pytest.param(
                *args,
                marks=pytest.mark.smoke if i == 0 else pytest.mark.full,
                id=w["label"],
            )
        )
    return params

class ManifestBenchmark(BenchmarkBase[ShapeDtypeWorkload]):
    """Generic benchmark that reads FLOP/memory counts from an Op instance.

    Accepts an op name, an instantiated Op, and any workload satisfying
    :class:`ShapeDtypeWorkload`.  The op must implement ``eval_roofline()``.
    Dynamic-shape ops may bind roofline variables during ``forward()``, so
    this helper calls ``op.eval_roofline()`` only while building a result
    after profiling has executed the op.

    Usage::

        op = SumFwdOp(dtype=dtype, dim=0)
        bm = ManifestBenchmark("SumFwdOp", op, workload)
        result = bm.profile(op, *inputs)
    """

    def __init__(
        self,
        op_name: str,
        op: Any,
        workload: ShapeDtypeWorkload,
    ):
        super().__init__(workload)
        self._op_name = op_name
        self._op = op
        self._roofline_cache: Optional[tuple[float, float]] = None

    def _get_roofline(self) -> tuple[float, float]:
        if self._roofline_cache is None:
            flops, mem_bytes = self._op.eval_roofline()
            self._roofline_cache = (float(flops), float(mem_bytes))
        return self._roofline_cache

    def calculate_flops(self) -> Optional[float]:
        return self._get_roofline()[0]

    def calculate_memory(self) -> Optional[float]:
        return self._get_roofline()[1]


def _extract_op_config(op: object) -> Optional[dict]:
    """Return the kernel config for an Op instance, or None if unavailable.

    Handles the three Op patterns currently used in tileops:

      1. **Eager-init** (e.g. ``GemmOp``): ``op.kernel`` is a Kernel
         instance set in ``__init__``.
      2. **Lazy with dummy kernel** (e.g. ``FFTC2COp``): ``op.kernel`` is a
         default Kernel and ``op._kernel_cache`` may hold others.
      3. **Pure lazy cache** (e.g. ``_SoftmaxBaseOp`` and the spec-conformant
         reduction ops): ``op._kernel_cache`` is the only source; ``op.kernel``
         is unset.

    A direct ``op.config`` attribute (legacy / explicit override) takes
    precedence over kernel introspection.
    """
    op_config = getattr(op, "config", None)
    if op_config:
        return op_config

    kernel = getattr(op, "kernel", None)
    op_config = getattr(kernel, "config", None) if kernel is not None else None
    if op_config:
        return op_config

    # Pure lazy-cache pattern: pick any cached kernel's config. All cached
    # kernels for a given op share dtype/op_kind, so taking the first is
    # sufficient for the benchmark report (which records one entry per call).
    cache = getattr(op, "_kernel_cache", None)
    if cache:
        try:
            first_kernel = next(iter(cache.values()))
        except StopIteration:
            first_kernel = None
        if first_kernel is not None:
            op_config = getattr(first_kernel, "config", None)
            if op_config:
                return op_config

    return None


class BenchmarkReport:
    """Collects benchmark results and dumps a markdown report.

    All methods are static — use as BenchmarkReport.record(...).
    Call clear() at session start, dump() at session end.
    """
    _records: dict = {}

    @staticmethod
    def record(op_or_name, params: dict, result: dict, tag: str = "tileops") -> None:
        """Record a benchmark result.

        Args:
            op_or_name: Op instance or benchmark group name string.
                If an Op instance, class name and module are extracted automatically.
            params: Parameter dict (typically from locals())
            result: Dict with latency_ms, tflops, bandwidth_tbs
            tag: Label to distinguish implementations (e.g. "tileops", "FA3", "fla")
        """
        if isinstance(op_or_name, str):
            name = op_or_name
            op_module = None
            op_config = None
        else:
            name = op_or_name.__class__.__name__
            op_module = op_or_name.__class__.__module__
            op_config = _extract_op_config(op_or_name)

        # Filter params to only include serializable benchmark parameters.
        # Tuples of primitives (e.g. ``shape=(4096, 4096)``) are preserved
        # verbatim so the profile log carries the original input geometry
        # rather than a flattened element count.
        def _is_serializable(v: Any) -> bool:
            if isinstance(v, (int, float, bool, str, torch.dtype)):
                return True
            if isinstance(v, tuple):
                return all(_is_serializable(x) for x in v)
            return False

        filtered_params = {
            k: v for k, v in params.items()
            if k not in ("test", "bm", "op", "inputs", "result", "result_bl",
                         "baseline_fn", "tune")
            and not k.startswith("_")
            and _is_serializable(v)
        }
        record_entry = {
            "params": filtered_params,
            "result": result,
            "tag": tag,
        }
        if op_config:
            record_entry["config"] = op_config
        BenchmarkReport._records.setdefault(name, []).append(record_entry)

        # Accumulate in thread-local for conftest hook.
        if not hasattr(_bench_results, "entries"):
            _bench_results.entries = []
        entry = {"tag": tag, "op": name, **result}
        if op_module:
            entry["op_module"] = op_module
        _bench_results.entries.append(entry)

        _logger.info("op=%s module=%s tag=%s latency_ms=%.4f tflops=%.2f",
                      name, op_module or "N/A", tag,
                      result.get("latency_ms", 0),
                      result.get("tflops", 0))

    @staticmethod
    def dump(path: str) -> None:
        """Write all collected results to a markdown-formatted log file."""
        if not BenchmarkReport._records:
            return

        lines = [
            "# TileOPs Benchmark Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Environment",
            "",
        ]
        lines.extend(_get_env_metadata())
        lines.append("")

        default_result_keys = ["latency_ms", "tflops", "bandwidth_tbs"]

        for name, entries in BenchmarkReport._records.items():
            if not entries:
                continue

            lines.append(f"## {name}")
            lines.append("")

            # Group by tag
            tag_entries = {}
            for entry in entries:
                tag_entries.setdefault(entry["tag"], []).append(entry)
            result_keys = list(default_result_keys)
            for entry in entries:
                for key in entry["result"]:
                    if key not in result_keys:
                        result_keys.append(key)

            for tag, tag_group in tag_entries.items():
                lines.append(f"### {tag}")
                lines.append("")

                param_keys = list(tag_group[0]["params"].keys())
                has_config = any("config" in e for e in tag_group)
                header_parts = param_keys + result_keys
                if has_config:
                    header_parts.append("config")
                lines.append("| " + " | ".join(header_parts) + " |")
                lines.append("| " + " | ".join(["---"] * len(header_parts)) + " |")

                for entry in tag_group:
                    row = [str(entry["params"].get(k, "")) for k in param_keys]
                    for rk in result_keys:
                        val = entry["result"].get(rk)
                        if val is None:
                            row.append("N/A")
                        elif isinstance(val, (int, float)) and not isinstance(val, bool):
                            row.append(f"{val:.4f}")
                        else:
                            row.append(str(val))
                    if has_config:
                        cfg = entry.get("config")
                        row.append(str(cfg) if cfg else "")
                    lines.append("| " + " | ".join(row) + " |")

                lines.append("")

        with open(path, "w") as f:
            f.write("\n".join(lines))

        print(f"Benchmark report saved to {path}")

    @staticmethod
    def clear() -> None:
        """Clear all collected records."""
        BenchmarkReport._records.clear()
