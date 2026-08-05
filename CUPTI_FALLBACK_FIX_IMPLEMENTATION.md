# CUPTI Fallback 修复说明

## 问题背景

Nightly benchmark 使用 `cuda-events` 而不是 `cupti`，导致：

- L2 Norm latency 从 0.0098 ms 误报为 0.0675 ms（6.9x）
- Roofline 从 37% 误报为 5%（7.4x）
- TileOPs 甚至显示比 PyTorch 还慢

**根本原因**: `torch.profiler` 在 nightly 环境中出现 CUPTI projection 失败，fallback 到了 cuda-events。

## 已实施的修改

### 1. 增强的错误诊断 (`benchmarks/benchmark_base.py:290-306`)

```python
if n_regions != n_repeat:
    # 添加了详细的诊断信息
    n_cuda_kernels = sum(...)
    _logger.debug(
        "CUPTI projection mismatch: %d annotation windows vs %d repeats "
        "(%d CUDA kernels captured). ...",
        n_regions,
        n_repeat,
        n_cuda_kernels,
    )
    raise _CuptiProjectionError(...)
```

**目的**: 当 CUPTI 失败时，记录详细信息帮助诊断根因。

### 2. 可控的 Fallback 行为 (`benchmarks/benchmark_base.py:299-313`)

```python
allow_fallback = os.getenv("TILEOPS_ALLOW_CUDA_EVENTS_FALLBACK", "1") == "1"

if not allow_fallback:
    raise RuntimeError("CUPTI profiling failed. CUDA-events fallback is disabled. ...")
```

**目的**: 通过环境变量控制是否允许 fallback。

### 3. 改进的警告消息

```python
_logger.warning(
    "CUPTI projection failed (%s); falling back to CUDA-events "
    "timing, which includes ~50-60us launch overhead per call. "
    "Latency will be inflated by ~6-7x for fast kernels (<10us). ...",
)
```

**目的**: 明确告知 fallback 的影响。

## 使用方法

### 场景 A: 允许 Fallback（默认，向后兼容）

```bash
# 默认行为，保持向后兼容
pytest benchmarks/ops/bench_vector_norm.py
```

如果 CUPTI 失败：

- 会有警告日志
- Fallback 到 cuda-events
- 生成数据但 latency 不准确

### 场景 B: 禁止 Fallback（推荐用于 CI）

```bash
# 在 nightly workflow 中设置
export TILEOPS_ALLOW_CUDA_EVENTS_FALLBACK=0
pytest benchmarks/ops/bench_vector_norm.py
```

如果 CUPTI 失败：

- 测试直接失败，不会 fallback
- 不会生成错误的 benchmark 数据
- 强制修复 CUPTI 问题

### 场景 C: 调试模式

```bash
# 启用详细日志
export TILEOPS_ALLOW_CUDA_EVENTS_FALLBACK=1
python -m pytest benchmarks/ops/bench_vector_norm.py -v --log-cli-level=DEBUG
```

查看日志中的：

```
DEBUG ... CUPTI projection mismatch: X annotation windows vs Y repeats (Z CUDA kernels captured)
```

## Nightly CI 修改建议

### 方案 1: 禁用 Fallback（推荐）

修改 `.github/workflows/nightly.yml`:

```yaml
- name: Run benchmark ops
  env:
    TILEOPS_ALLOW_CUDA_EVENTS_FALLBACK: "0"  # 添加这行
  run: |
    ...
    python3 -m pytest -q benchmarks/ops ...
```

**优点**:

- 强制修复 CUPTI 问题
- 不会生成错误数据

**缺点**:

- 如果 CUPTI 失败，整个 nightly 会失败

### 方案 2: 监控但允许 Fallback

```yaml
- name: Run benchmark ops
  run: |
    ...
    python3 -m pytest -q benchmarks/ops ... 2>&1 | tee bench.log

    # 检查是否有 fallback 警告
    if grep -q "falling back to CUDA-events" bench.log; then
      echo "::warning::Benchmarks used CUDA-events fallback. Results may be inaccurate."
    fi
```

**优点**:

- Nightly 不会失败
- 但会有明确的警告

**缺点**:

- 仍然会生成不准确的数据

## 下一步调查

要彻底修复 CUPTI projection 问题，需要：

1. **在 nightly 环境中运行诊断**:

   ```bash
   export TILEOPS_ALLOW_CUDA_EVENTS_FALLBACK=1
   pytest benchmarks/ops/bench_vector_norm.py --log-cli-level=DEBUG -v
   ```

1. **查看日志中的诊断信息**:

   - 有多少个 annotation windows?
   - 有多少个 CUDA kernels?
   - 为什么数量不匹配?

1. **可能的根因**:

   - `pytest-timeout --timeout-method=thread` 的线程干扰
   - PyTorch 2.10 的 bug
   - 环境配置问题

1. **潜在修复**:

   - 使用 `--timeout-method=signal` 而不是 `thread`
   - 升级/降级 PyTorch 版本
   - 使用分批 profiling
   - 测量并减去 launch overhead

## 验证修复

修复成功后，`profile_run.log` 应该显示：

```
| timing | config |
| cupti | ... |    ← 而不是 cuda-events
```

Latency 应该恢复正常：

```
| latency_ms |
| 0.0098     |  ← 而不是 0.0675
```

## 文件清单

修改的文件：

- `benchmarks/benchmark_base.py` - 主要修改

生成的文档：

- `FIX_CUPTI_FALLBACK.md` - 修复方案概述
- `CUPTI_FALLBACK_FIX_IMPLEMENTATION.md` - 本文档
- `L2_NORM_NIGHTLY_FINAL_DIAGNOSIS.md` - 完整调查报告
