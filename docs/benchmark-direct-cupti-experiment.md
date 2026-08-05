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
