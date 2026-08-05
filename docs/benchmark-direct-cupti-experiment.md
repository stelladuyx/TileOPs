# Direct CUPTI benchmark experiment

## Goal

Replace the default `torch.profiler`/Kineto projected-annotation timing path
with the direct CUPTI activity interface used by NVIDIA SOL-ExecBench, then
measure whether it remains stable over both isolated microbenchmarks and a
Nightly-like long-running process.

This branch deliberately keeps all three backends available:

| `TILEOPS_TIMING_BACKEND` | Purpose                                          |
| ------------------------ | ------------------------------------------------ |
| `cupti-direct`           | Default experiment backend; no Kineto projection |
| `kineto`                 | Legacy A/B control                               |
| `cuda-events`            | Diagnostic control, not valid history data       |

Direct CUPTI first discovers the callable's complete kernel/memcpy/memset
sequence. Every timed call gets a window from `cupti.get_timestamp()` before
the callable to another timestamp after `torch.cuda.synchronize()`. A sample is
accepted only when the complete discovered sequence can be selected inside the
window.

`TILEOPS_DIRECT_CUPTI_METRIC` controls how selected activities become latency:

| Value           | Semantics                                                                  | Status       |
| --------------- | -------------------------------------------------------------------------- | ------------ |
| `activity-sum`  | Sum complete activity durations, matching legacy TileOps Kineto semantics  | Default      |
| `activity-span` | First activity start to last activity end, matching upstream SOL-ExecBench | Experimental |

## Isolated environment

Do not install experiment packages into `flashmlaenv` or `tileopsenv`. The
following creates a lightweight venv that uses the known-working TileLang/Torch
stack from `flashmlaenv` read-only but receives its own writes:

```bash
/home/yuxian.du/miniconda3/envs/flashmlaenv/bin/python \
  -m venv --system-site-packages \
  /home/yuxian.du/.venvs/flashmla-sol-cupti

/home/yuxian.du/.venvs/flashmla-sol-cupti/bin/python \
  -m pip install --no-deps cupti-python==12.8.0
```

Why `--no-deps`: the CUDA 12.9 runner's PyTorch requires
`cuda-bindings==12.9.4`, while `cupti-python==12.8.0` declares
`cuda-bindings==12.8.0`. The CUPTI extension and `get_timestamp()` work with
the runner's 12.9 binding/library, but installing dependency closure would
downgrade the binding required by PyTorch. This is an experiment constraint,
not yet a production dependency decision.

## Unit and GPU integration tests

```bash
cd /home/yuxian.du/TileOPs-sol-cupti

/home/yuxian.du/.venvs/flashmla-sol-cupti/bin/python -m pytest -q \
  benchmarks/tests/test_cupti_timing.py \
  benchmarks/tests/test_benchmark_base.py -m smoke

TILEOPS_ALLOW_CUDA_EVENTS_FALLBACK=0 \
/home/yuxian.du/.venvs/flashmla-sol-cupti/bin/python -m pytest -q \
  benchmarks/tests/test_cupti_timing.py -m nightly
```

The first command needs no GPU. The Nightly-marked test requires CUDA and
`cupti-python`; it verifies complete samples and cross-run median stability.

## Controlled stability experiment

Run the three-backend matrix on the H200 Nightly runner:

```bash
cd /home/yuxian.du/TileOPs-sol-cupti

/home/yuxian.du/.venvs/flashmla-sol-cupti/bin/python \
  benchmarks/tools/timing_stability.py \
  --rounds 10 \
  --warmup 10 \
  --repeats 50 \
  --trials 3 \
  --direct-metric activity-sum \
  --max-cv 0.05 \
  --max-drift 0.05 \
  --max-direct-kineto-delta 0.10 \
  --max-failure-rate 0 \
  --include-tileops-l2 \
  --output timing_stability_h200.json
```

The tool rotates backend order between rounds, disables fallback, and retains:

- every reported latency and direct/CUDA-event raw sample;
- per-trial means and within-measurement CV;
- direct-CUPTI discovery activity sequence;
- attribution failures without converting them to another backend;
- round-level CV and first-half/second-half drift.

Initial acceptance criteria:

1. Direct CUPTI attribution success is `100%` for all three synthetic cases.
1. Round-level CV is at most `5%` with locked clocks.
1. First-half versus second-half median drift is at most `5%`.
1. For single-kernel cases, direct CUPTI and successful Kineto medians differ by
   at most `10%`.
1. CUDA-events results are reported only as an error-size comparison and never
   mixed into performance history.

## H200 controlled results

Measured on 2026-08-05 with an NVIDIA H200, driver 575.57.08, locked 1500 MHz
SM clock, Torch `2.11.0.dev20260107+cu129`, 10 rounds, 10 warmups, 50 repeats,
and 3 trials. All direct measurements completed without fallback.

### Accepted: direct CUPTI `activity-sum`

| Case                 | Direct median | Direct round CV | Half-run ratio | Kineto median | Direct / Kineto | CUDA Events median |
| -------------------- | ------------: | --------------: | -------------: | ------------: | --------------: | -----------------: |
| Fast single kernel   |      1.904 us |           0.57% |          0.998 |      1.901 us |          1.001x |          21.048 us |
| Medium single kernel |      7.499 us |           0.22% |          0.999 |      7.496 us |          1.000x |          23.958 us |
| Fast multi-kernel    |      3.653 us |           0.33% |          1.002 |      3.636 us |          1.005x |          32.149 us |

The activity-sum experiment passed all initial acceptance thresholds: 100%
attribution success, less than 1% round CV, less than 1% half-run drift, and
less than 1% difference from Kineto for all three cases.

### Real TileOps L2Norm matrix

A second run used the production-compatible Torch `2.10.0+cu129` / TileLang
`0.1.11+cuda.gitcd37ed5f` stack and added the real `(2048, 4096)` BF16 TileOps
L2Norm plus its PyTorch baseline. It ran 5 rounds with the same 50 repeats and
3 trials. Every direct attribution succeeded and the report had no acceptance
failures.

| Case           | Direct median | Direct round CV | Half-run ratio | Kineto median | Direct / Kineto | CUDA Events median |
| -------------- | ------------: | --------------: | -------------: | ------------: | --------------: | -----------------: |
| TileOps L2Norm |      9.192 us |           0.13% |          1.001 |      9.195 us |          1.000x |          89.008 us |
| PyTorch L2Norm |     51.035 us |           0.16% |          0.999 |     51.042 us |          1.000x |          65.142 us |

This also verifies repeated direct CUPTI enable/flush/disable/finalize cycles in
one process across synthetic, TileOps, and PyTorch callables.

## Two-microsecond CPU/GPU timestamp-offset risk

CUPTI normalizes GPU activity timestamps onto the CPU timeline using linear
interpolation. A constant offset does not change an activity's own duration,
but it can move an activity across a CPU timestamp-window boundary and make
attribution fail. The direct backend now records conservative per-sample guard
bands from the CPU window start to the first activity start and from the last
activity end to the CPU window end. The stability tool rejects any sample with
less than the configurable `--clock-shift-risk-us` guard (default 2 us).

On H200, 6,900 newly instrumented direct-CUPTI samples had zero 2-us-unsafe
windows. Representative minimum guards were:

| Case                                |  GPU span | Samples | Min left guard | Min right guard | Unsafe at 2 us |
| ----------------------------------- | --------: | ------: | -------------: | --------------: | -------------: |
| PyTorch add, 4K BF16                |  1.855 us |     900 |       9.343 us |        4.848 us |              0 |
| PyTorch two-kernel add/mul          | 11.966 us |     900 |       9.241 us |        4.676 us |              0 |
| TileOps ReLU decode, 1x4096 BF16    |  1.784 us |     300 |      27.157 us |       22.229 us |              0 |
| TileOps RMSNorm decode, 1x4096 BF16 |  2.676 us |     300 |      46.550 us |       26.715 us |              0 |
| TileOps L2Norm, 2048x4096 BF16      |  9.178 us |     900 |      48.769 us |        7.808 us |              0 |
| TileOps Sum, 2048x4096 BF16         |  9.114 us |     300 |      48.008 us |       20.785 us |              0 |
| TileOps GEMV, M1 N7168 K2048 BF16   | 18.923 us |     300 |      24.193 us |        6.743 us |              0 |
| TileOps GEMM, 1024x1024x1024 BF16   | 14.512 us |     300 |      33.077 us |        6.652 us |              0 |
| TileOps elementwise Neg, 16M BF16   | 18.769 us |     300 |      28.662 us |        7.858 us |              0 |
| PyTorch L2Norm, 2048x4096 BF16      | 53.222 us |     900 |      10.998 us |        4.830 us |              0 |

The shortest measured TileOps kernels are therefore not automatically the
highest attribution risk: their Python/TileLang dispatch before launch and the
post-call CUDA synchronization provide substantially more than 2 us of guard.
The remaining high-risk configurations are lower-overhead native launchers,
timestamp windows without a post-call synchronization, work launched from a
different host thread after the wrapper returns, and back-to-back windows on
non-default streams. Those configurations are not used by the current TileOps
benchmark path and require a separate native-launch stress test before support.

## Locked GPU1 activity-span gap decomposition

The activity-span experiment was repeated on GPU1 with the benchmark process
pinned to local-NUMA CPU2. During the run, one-second `nvidia-smi dmon` samples
reported an invariant 1500 MHz SM clock and 3201 MHz memory clock under both
idle and benchmark load. The run used 10 rounds, 100 repeats, 3 trials, disabled
fallback, and retained sum, span, and gap from the exact same CUPTI capture.

For the two-kernel BF16 add/mul case:

| Component                            | Result           |
| ------------------------------------ | ---------------- |
| Activity-sum round mean              | 3.650 us         |
| Activity-sum round CV                | 0.21%            |
| Kineto round median                  | 3.647 us         |
| Activity-span reported round median  | 11.478 us        |
| Activity-span reported round CV      | 7.61%            |
| Inter-activity gap median            | 7.584 us         |
| Inter-activity gap P90               | 9.152 us         |
| Maximum observed inter-activity gap  | 3920.672 us      |
| Inter-activity gap round-mean CV     | 46.06%           |
| Incomplete sequence / 2-us-risk miss | 0 / 3000 samples |

The kernel activity durations are stable while the gap is not. Frequency
variation, activity attribution, and sequence changes are ruled out for this
run; the activity-span instability is driven by time between the two GPU
activities, including rare millisecond-scale host submission stalls. Exact SOL
span semantics therefore measure launch-pipeline behavior for short
multi-kernel callables rather than only GPU execution time.

### Gap anatomy and tuning experiment

For sequential activities on one stream, the measured idle gap between
activity N and N+1 can contain host framework/dispatcher work that was not
finished before N completed, CUDA runtime and driver launch work, command-buffer
queue/submission delay, GPU scheduling or resource delay, stream/event
dependencies, context interference, and host descheduling. Work that overlaps
activity N does not contribute to the observed GPU idle interval.

For concurrent streams, `span - sum` is not a pure gap because activity
durations overlap. The collector therefore also computes the union of all
activity intervals and reports:

```text
span = union GPU busy + inter-activity idle
sum  = union GPU busy + multiply-counted overlap
```

CUPTI can further expose kernel `queued` and `submitted` timestamps when
latency timestamp collection is enabled before CUDA initialization. Together
with runtime/driver API correlation, those timestamps can separate host launch,
command-buffer submission, and GPU scheduling portions in a dedicated process.

To test whether gap reduction is actionable, the same locked-GPU1 add/mul pair
was compared as ordinary eager launches and as one CUDA Graph replay. Both
paths retained the same two kernel identities. The process remained pinned to
CPU2 and all sampled clocks remained 1500 MHz SM / 3201 MHz memory.

| Metric                     |      Eager | CUDA Graph | Change |
| -------------------------- | ---------: | ---------: | -----: |
| Activity-sum round median  |   3.626 us |   2.852 us | -21.3% |
| Idle gap median            |   7.616 us |   0.320 us | -95.8% |
| Activity-span round median |  11.565 us |   3.179 us | -72.5% |
| Activity-span round CV     |      5.25% |      0.27% |        |
| Maximum idle gap           | 113.408 us |   0.353 us |        |

Reducing gap is therefore a valid end-to-end tuning direction for repeated,
short multi-kernel workflows. Kernel fusion, CUDA Graph replay, native/batched
launch, persistent kernels, removing host round trips, and eliminating
unnecessary stream/event waits are candidate techniques. It should remain a
separate objective from kernel-body tuning: per-kernel implementation history
uses activity-sum, while application/workflow latency can use activity-span and
idle-gap diagnostics.

### Rejected as default: upstream `activity-span`

The same 10-round experiment with upstream SOL-ExecBench span semantics was
stable for both single-kernel cases, but the fast multi-kernel case measured
12.701 us versus Kineto's 3.685 us, had 8.49% round CV, and drifted 9.4% between
the first and second halves. The span includes GPU idle gaps while the host
launches the next short kernel. It remains useful as an end-to-end GPU-span
diagnostic but is not a stable replacement for TileOps' historical pure
activity-duration metric.

## Nightly-like workload matrix

After the synthetic matrix passes, run selected real cases with
`TILEOPS_ALLOW_CUDA_EVENTS_FALLBACK=0` under both `cupti-direct` and `kineto`:

- one fast norm/elementwise single-kernel case;
- one GQA case that previously produced projection mismatch;
- one Grouped GEMM or FP8 GEMM case;
- one multi-kernel DeltaNet/GatedDeltaNet backward case;
- one Conv3d case;
- `n_repeat = 1`, `10`, and `50`;
- isolated pytest process and normal Nightly file order.

For formal history comparison, persist `timing`, `timing_requested`, fallback
reason, sample count, CV, and trial means. The branch exposes these diagnostics
when `TILEOPS_RECORD_TIMING_DIAGNOSTICS=1`.

## Known risks to validate

- `cupti-python` currently has no 12.9 release; 12.8/12.9 ABI compatibility is
  demonstrated only for import/API access until the H200 run completes.
- Direct CUPTI tracing repeatedly enables, flushes, disables, and finalizes
  activity collection. Long-process stability must be tested, not inferred from
  one successful call.
- Data-dependent kernel sequences intentionally fail validation if discovery
  and timed iterations differ. Such a failure is safer than silently averaging
  incomplete calls but needs an explicit benchmark policy.
- Multi-stream callables are included through the post-call synchronize.
  `activity-sum` can double-count overlapping activities, while
  `activity-span` includes inter-activity gaps; real multi-stream workloads must
  therefore be checked separately before formal Nightly adoption.
