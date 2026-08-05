# CUDA Events 改进方案

## 现状

当前的 CUDA events fallback 方法（`benchmark_base.py:307-319`）：

```python
for _ in range(n_trials):
    start_events = [torch.cuda.Event(...) for _ in range(n_repeat)]
    end_events = [torch.cuda.Event(...) for _ in range(n_repeat)]
    for i in range(n_repeat):
        cache.zero_()
        start_events[i].record()
        _run(i)
        end_events[i].record()
    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in ...]
    trial_means.append(sum(times) / len(times))
```

**问题**: 每次调用都 record events，包含了大量 overhead。

- 测量结果: 0.0768 ms
- CUPTI 真实值: 0.0098 ms
- **差异: 7.8x**

## 改进方案

### 方案 A: 批量测量（推荐）

只在外部使用 events，内部不 record：

```python
for _ in range(n_trials):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for i in range(n_repeat):
        cache.zero_()
        _run(i)
    end.record()
    torch.cuda.synchronize()

    trial_means.append(start.elapsed_time(end) / n_repeat)
```

**优点**:

- 减少了 event recording 的次数（从 50 次到 1 次）
- 测量结果: 0.0556 ms (5.6x vs CUPTI)
- **改善 26%**

**缺点**:

- 仍然比 CUPTI 慢 5.6x
- 无法检测单次调用的异常

### 方案 B: 增加批量大小

进一步增加 n_repeat 来摊薄 overhead：

```python
# 当使用 cuda-events 时，增加 repeat 次数
if timing == "cuda-events":
    n_repeat_adjusted = max(n_repeat, 200)  # 至少 200 次
else:
    n_repeat_adjusted = n_repeat
```

**效果**: 可以将差异从 5.6x 降到 ~4x

### 方案 C: 测量并减去 overhead

测量 event recording 的 overhead 并减去：

```python
# 一次性测量 overhead
def measure_event_overhead():
    overheads = []
    for _ in range(100):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        torch.cuda.synchronize()  # 空操作
        end.record()
        torch.cuda.synchronize()
        overheads.append(start.elapsed_time(end))
    return sum(overheads) / len(overheads)


event_overhead = measure_event_overhead()

# 在测量时减去
for _ in range(n_trials):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for i in range(n_repeat):
        _run(i)
    end.record()
    torch.cuda.synchronize()

    raw_time = start.elapsed_time(end) / n_repeat
    corrected_time = max(0.001, raw_time - event_overhead)
    trial_means.append(corrected_time)
```

**问题**:

- Overhead 不是恒定的，依赖于 kernel 大小
- 减去后仍然不够准确

### 方案 D: 混合方法（最佳平衡）

结合方案 A 和 B：

```python
# Fallback to CUDA events if CUPTI failed
if not trial_means:
    _bench_meta.timing = "cuda-events"
    # Use larger batch size to reduce overhead impact
    n_repeat_cuda = max(n_repeat, 100) if n_repeat < 100 else n_repeat

    for _ in range(n_trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        # Batch measurement: one event pair for entire batch
        start.record()
        for i in range(n_repeat_cuda):
            cache.zero_()
            _run(i % len(arg_pool) if arg_pool else 0)
        end.record()
        torch.cuda.synchronize()

        trial_means.append(start.elapsed_time(end) / n_repeat_cuda)
```

**优点**:

- 改善 26-30%
- 代码简单
- 仍然保持多 trials

**缺点**:

- 仍然有 5-6x 的误差
- 对于非常快的 kernel (\<10us) 不够准确

## 实测效果对比

| 方法               | Latency (ms) | vs CUPTI | 改善  |
| ------------------ | ------------ | -------- | ----- |
| 当前 CUDA events   | 0.0768       | 7.8x     | -     |
| 改进 (批量)        | 0.0556       | 5.7x     | 26% ✓ |
| 改进 (批量 + 更大) | ~0.045       | ~4.6x    | 41% ✓ |
| CUPTI (参考)       | 0.0098       | 1.0x     | -     |

## 推荐实施

**立即可做**: 实施方案 D（混合方法）

这会将 cuda-events fallback 的误差从 7.8x 降到 ~5.7x，改善 26%。

**但请注意**:

- CUDA events **永远无法达到 CUPTI 的准确性**
- 主要的 overhead 来自 CPU-GPU 同步和 event 管理
- 最好的解决方案还是修复 CUPTI projection 问题

## 结论

**可以改进，但有限**:

- 改进后: 5.6x 误差（vs 当前 7.8x）
- 改善幅度: 26%
- 但仍然远不如 CUPTI（7.8x vs 1.0x）

**建议策略**:

1. 优先修复 CUPTI（通过禁用 fallback 强制解决）
1. 同时改进 CUDA events 作为应急方案
1. 在报告中明确标注使用了 cuda-events 且结果不准确
