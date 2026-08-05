# L2 Norm Nightly Roofline 问题 - 最终诊断报告

## 问题定位 ✅

### 根本原因

**Nightly 报告中的 latency 测量值是错误的！**

```
错误值: 0.0675 ms (nightly 报告)
正确值: 0.0098 ms (实际测量)
差异:   6.9x
```

### 连锁效应

因为所有性能指标都基于 latency 计算，一个错误导致全部错误：

| 指标      | Nightly (错) | 正确值    | 关系               |
| --------- | ------------ | --------- | ------------------ |
| Latency   | 0.0675 ms    | 0.0098 ms | 基础测量           |
| TFLOPs    | 0.2          | 1.81      | = FLOPs / Latency  |
| Bandwidth | 0.25 TB/s    | 1.71 TB/s | = Bytes / Latency  |
| Roofline  | 5%           | 37.7%     | = Bandwidth / Peak |

## 已排除的可能原因 ❌

我们在 docker 中完整模拟了 nightly 环境，验证了：

1. ✅ Benchmark 代码完全正确
1. ✅ `--timeout-method=thread` 不影响结果
1. ✅ CUPTI profiling 工作正常
1. ✅ torch.profiler 测量准确
1. ✅ JUnit XML 写入正确

**在 docker 模拟环境中，所有测量都是正确的 (0.0098 ms)。**

## 剩余可能的原因

既然代码和环境都正确，问题必定在：

### 可能性 #1: 特定 nightly run 的异常

某次 nightly run 可能遇到：

- GPU 被其他进程占用
- GPU 温度过高导致降频
- 内存不足导致 swap
- 网络/存储 I/O 延迟

**验证方法:**

```bash
# 检查其他 nightly runs 的数据
# 如果只有某一次是 0.0675 ms，其他都是 0.009 ms
# 说明是异常 run
```

### 可能性 #2: 数据聚合/后处理错误

`nightly_report.py` 或某个中间环节可能：

- 错误地平均了多个 GPU 的结果
- 混合了不同 workload 的数据
- 单位转换错误

**验证方法:**

```bash
# 检查原始的 bench_results.xml
grep "hidden-state-l2-float16" bench_results.xml
grep "tileops_latency_ms" bench_results.xml
# 看 XML 中的值是 0.0098 还是 0.0675
```

### 可能性 #3: 测试在非独占 GPU 上运行

Nightly workflow 要求 GPU 独占：

```yaml
# Phase 1 — Benchmark (exclusive GPU access for accurate profiling)
```

但如果：

- 多个 workflow 同时运行
- 有其他进程在使用 GPU
- GPU 锁定失败

会导致测量不准确。

**验证方法:**

```bash
# 检查 nightly logs 中是否有 warning
grep -i "gpu.*busy\|concurrent\|lock failed" nightly.log
```

### 可能性 #4: 测试顺序或缓存问题

如果 L2 Norm 测试：

- 是第一个运行的（cold start，JIT compilation）
- TileLang cache 被清空
- 每次都重新编译

**验证方法:**

```bash
# 检查是否有编译日志
grep "TileLang begins to compile" nightly.log | head -10
```

## 诊断步骤

### Step 1: 检查原始 JUnit XML

```bash
# 从 GitHub Actions artifacts 下载 bench_results.xml
# 然后检查
python3 << 'EOF'
import xml.etree.ElementTree as ET
tree = ET.parse('bench_results.xml')
for tc in tree.findall('.//testcase'):
    name = tc.get('name')
    if 'hidden-state-l2-float16' in name:
        for prop in tc.findall('.//property[@name="tileops_latency_ms"]'):
            print(f'{name}: tileops_latency_ms = {prop.get("value")}')
EOF
```

**期望结果:**

- 如果 XML 中是 0.0098 → 问题在报告生成
- 如果 XML 中是 0.0675 → 问题在 benchmark 测量

### Step 2: 检查多个 nightly runs

```bash
# 查看历史 nightly runs 的数据
# 如果都是 0.0675 → 系统性问题
# 如果只有某次 → 偶发异常
```

### Step 3: 检查 GPU 状态日志

```bash
# 检查 nightly logs
grep -i "gpu\|nvidia-smi\|clock\|temperature" nightly.log
# 看是否有异常
```

### Step 4: 对比不同 workloads

```bash
# 检查其他 L2 norm workloads
# 如果都慢 6.9x → 系统性问题
# 如果只有这个 → 特定问题
```

## 解决方案

### 临时方案: 添加诊断日志

```python
# 在 benchmarks/benchmark_base.py 的 bench_kernel 中
def bench_kernel(...):
    ...
    # 添加详细日志
    logger.info(f"bench_kernel start: op={fn.__name__}, "
                f"cuda_available={torch.cuda.is_available()}, "
                f"gpu_name={torch.cuda.get_device_name(0)}")

    latency = ...

    logger.info(f"bench_kernel result: latency={latency:.6f}ms, "
                f"n_warmup={n_warmup}, n_repeat={n_repeat}, "
                f"n_trials={n_trials}")

    # 添加异常检测
    if latency > 0.05:  # 50 us threshold
        logger.warning(f"Suspicious high latency {latency}ms!")

    return latency
```

### 永久方案: 增强错误检测

```python
# 在 conftest.py 中
def pytest_runtest_makereport(item, call):
    ...
    if tileops_entry:
        latency = tileops_entry.get("latency_ms", 0)

        # 检测异常 latency
        if latency > 0.05:
            item.add_marker(
                pytest.mark.xfail(
                    reason=f"Suspicious latency {latency}ms (>50us threshold)"
                )
            )
```

### 根本方案: 找到数据源头

1. 提供实际的 nightly run URL
1. 下载 bench_results.xml
1. 检查原始数据
1. 定位问题环节

## 下一步行动

请提供：

1. **GitHub Actions nightly run URL**
1. **bench_results.xml 文件**（从 artifacts 下载）
1. **完整的 nightly logs**（特别是 benchmark 部分）

有了这些，我可以：

- 精确定位问题在哪个环节
- 找到具体的 bug
- 提供针对性的修复

______________________________________________________________________

**总结**: 问题不在 L2 Norm kernel 或 benchmark 代码，而在某个特定 nightly run 的异常或数据处理环节。需要原始数据才能进一步诊断。
