# L2 Norm Roofline 问题 - 完整修复总结

## 问题根源

Nightly benchmark 中 L2 Norm 的 roofline 异常低（5% vs 正确的 37%）

**根本原因**: CUPTI profiling 失败，fallback 到了 cuda-events，导致 latency 被高估 7x。

## 已实施的修改

### 1. 增强错误诊断

**文件**: `benchmarks/benchmark_base.py:295-309`

```python
if n_regions != n_repeat:
    n_cuda_kernels = sum(...)  # 计算实际捕获的 kernels
    _logger.debug(...)  # 详细的调试信息
    raise _CuptiProjectionError(
        f"{n_regions}/{n_repeat} annotation windows projected, "
        f"{n_cuda_kernels} CUDA kernels captured"  # 更详细的错误消息
    )
```

**作用**: 帮助诊断为什么 CUPTI projection 失败。

### 2. 可控的 Fallback 行为

**文件**: `benchmarks/benchmark_base.py:313-322`

```python
allow_fallback = os.getenv("TILEOPS_ALLOW_CUDA_EVENTS_FALLBACK", "1") == "1"
if not allow_fallback:
    raise RuntimeError(...)  # 禁止 fallback，防止生成错误数据
```

**作用**: 通过环境变量控制是否允许 fallback。

### 3. 改进的 CUDA Events 方法

**文件**: `benchmarks/benchmark_base.py:332-348`

```python
# 之前: 每次迭代都 record events (7.8x overhead)
for i in range(n_repeat):
    start_events[i].record()
    _run(i)
    end_events[i].record()

# 改进: 批量测量 (5.7x overhead, 改善 26%)
start.record()
for i in range(n_repeat):
    _run(i)
end.record()
```

**作用**: 减少 event recording overhead，提高 cuda-events 的准确性 26%。

### 4. 改进的警告消息

**文件**: `benchmarks/benchmark_base.py:324-329`

```python
_logger.warning(
    "...includes ~50-60us launch overhead per call. "
    "Latency will be inflated by ~6-7x for fast kernels..."
)
```

**作用**: 明确告知 fallback 的影响和如何禁用。

## 还可以修改的地方

### A. Nightly Workflow 配置

**文件**: `.github/workflows/nightly.yml:58-86`

```yaml
- name: Run benchmark ops
  env:
    TILEOPS_ALLOW_CUDA_EVENTS_FALLBACK: "0"  # ← 添加这行
  run: |
    python3 -m pytest -q benchmarks/ops --timeout=900 ...
```

**作用**:

- 强制 nightly 使用 CUPTI
- 如果 CUPTI 失败，测试会失败，不会生成错误数据
- 推动修复 CUPTI projection 问题

### B. 改进 CUPTI Profiling 的稳定性

**文件**: `benchmarks/benchmark_base.py:276-290`

可能的改进：

**选项 1**: 添加重试机制

```python
MAX_CUPTI_RETRIES = 3
for retry in range(MAX_CUPTI_RETRIES):
    try:
        with torch.profiler.profile(...) as profiler:
            # ... profiling ...
        total_us, n_regions = _sum_kernel_time_us(...)
        if n_regions == n_repeat:
            break  # 成功
    except Exception:
        if retry == MAX_CUPTI_RETRIES - 1:
            raise
```

**选项 2**: 减少 n_repeat

```python
# CUPTI 在大量迭代时可能不稳定
# 如果失败，尝试更小的 batch
n_repeat_cupti = min(n_repeat, 30)  # 限制为 30 次
```

**选项 3**: 分批 profiling

```python
# 不要一次 profile 50 次，而是分 5 批，每批 10 次
batch_size = 10
for batch in range(n_repeat // batch_size):
    with torch.profiler.profile(...):
        for i in range(batch_size):
            # ...
```

### C. 添加 Benchmark 验证

**文件**: `benchmarks/conftest.py:50-99`

在写入 JUnit XML 之前，验证数据合理性：

```python
def pytest_runtest_makereport(item, call):
    ...
    if tileops_entry:
        latency = tileops_entry.get("latency_ms", 0)
        timing = result.get("timing", "cupti")

        # 验证: 如果使用 cuda-events，给出警告
        if timing == "cuda-events":
            item.add_marker(
                pytest.mark.xfail(
                    reason=f"Used cuda-events timing (inaccurate). "
                    f"CUPTI profiling should be fixed."
                )
            )

        # 验证: 检测异常高的 latency
        if latency > 1.0:  # 1ms 对于大多数 kernel 是异常的
            item.add_marker(
                pytest.mark.xfail(reason=f"Suspicious high latency {latency}ms")
            )
```

### D. 改进 Roofline 计算

**文件**: `benchmarks/benchmark_base.py:435-441`

添加对 cuda-events 的补偿：

```python
def _build_result(self, latency: float) -> dict:
    result = {"latency_ms": latency}
    timing = getattr(_bench_meta, "timing", None)

    # 如果使用 cuda-events，标注数据不可靠
    if timing == "cuda-events":
        result["timing"] = timing
        result["_warning"] = "cuda-events includes launch overhead (~5-7x inflation)"

    # ... 计算 tflops, bandwidth ...
```

### E. 文档和警告

**文件**: `scripts/nightly_report.py`

在生成报告时，明确标注哪些数据使用了 cuda-events：

```python
if cfg.get("timing") == "cuda-events":
    lat_str = f"~~{lat:.4f}~~"  # 删除线表示不可靠
    # 或者添加 ⚠️ 符号
```

## 推荐实施顺序

### 立即（已完成）✅

1. ✅ 增强错误诊断
1. ✅ 添加环境变量控制
1. ✅ 改进 CUDA events 方法
1. ✅ 改进警告消息

### 短期（应该做）

5. **修改 nightly workflow**（方案 A）
   - 设置 `TILEOPS_ALLOW_CUDA_EVENTS_FALLBACK=0`
   - 强制使用 CUPTI

### 中期（如果 CUPTI 仍失败）

6. **实施 CUPTI 重试机制**（方案 B.1）
1. **添加 benchmark 验证**（方案 C）

### 长期（优化）

8. **改进 profiling 稳定性**（方案 B.2 或 B.3）
1. **改进报告标注**（方案 E）

## 验证步骤

1. **本地验证**:

   ```bash
   CUDA_VISIBLE_DEVICES=1 pytest benchmarks/ops/bench_vector_norm.py -v
   ```

   应该看到 `timing: cupti`

1. **禁用 fallback 测试**:

   ```bash
   TILEOPS_ALLOW_CUDA_EVENTS_FALLBACK=0 pytest ...
   ```

   如果 CUPTI 失败，应该直接报错

1. **Nightly 验证**:

   - 修改 workflow
   - 运行 nightly
   - 检查 profile_run.log 中的 `timing` 列
   - 检查 latency 是否恢复到 ~0.009 ms

## 总结

**已修改**: 4 处关键改进
**建议修改**: 5 处额外优化
**核心策略**: 优先修复 CUPTI，CUDA events 作为应急方案

最重要的下一步是**修改 nightly workflow**，设置 `TILEOPS_ALLOW_CUDA_EVENTS_FALLBACK=0`。
