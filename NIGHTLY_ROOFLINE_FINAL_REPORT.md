# L2 Norm Nightly Roofline 问题 - 最终调查报告

## 问题总结

Nightly 报告显示 L2 Norm 的性能异常：

```
Workload                    Latency     Roofline
hidden-state-l2-float16     0.0675 ms   5%        ← 错误
期望值                       0.0093 ms   37%       ← 正确
差异                         7.3x        7.4x
```

## 已验证的事实

### ✅ Benchmark 代码是正确的

1. **本地测试 (GPU1)**

   - Latency: 0.0093 ms
   - Roofline: 37.7%
   - 使用 CUPTI profiling

1. **容器中测试 (ci-runner-gpu2)**

   - Latency: 0.0098 ms
   - Roofline: 35.6%
   - `bench_kernel` 使用 CUPTI
   - torch.profiler 工作正常

1. **代码路径验证**

   ```
   bench_vector_norm.py::test_l2_norm_bench
   → bm.profile(op, *inputs)
   → bench_kernel(functor, args)
   → torch.profiler + CUPTI
   → 返回 {"latency_ms": 0.0098}
   → BenchmarkReport.record()
   → conftest.py 写入 JUnit XML
   → nightly_report.py 读取 XML 生成报告
   ```

### ❌ Nightly 数据异常

Nightly 报告的 0.0675 ms 必定来自以下某处的错误数据。

## 数据流分析

```
测试运行
  ↓
bench_kernel() 返回 latency_ms
  ↓
BenchmarkReport.record(result)
  ↓
conftest.py 读取 _bench_results
  ↓
写入 JUnit XML (user_properties)
  ↓
nightly_report.py 解析 XML
  ↓
生成 markdown 报告
```

**关键点:** 如果 nightly 报告显示 0.0675 ms，那么：

- 要么 `bench_kernel()` 返回了错误的值
- 要么数据在某个环节被错误地修改/替换

## 可能的根本原因

### 假设 #1: 多 GPU 测试导致的数据混淆

Nightly 可能在多个 GPU 上并行运行测试：

```bash
# 例如
pytest benchmarks/ops/bench_vector_norm.py -n 4  # 4个worker
```

**问题:**

- GPU0-3 同时运行，但有些 GPU 性能不同
- 或者某个 GPU 上有其他负载
- 结果被错误地聚合/平均

**验证方法:**

```bash
# 检查 CI 脚本中是否有 -n 参数
grep -r "pytest.*-n\|xdist" .github/workflows/
```

### 假设 #2: 测试在 CPU 上运行或 CUDA 不可用

如果某个 worker 的 CUDA 环境有问题：

```python
# bench_kernel fallback 到 wall time
if not torch.cuda.is_available():
    # 使用简单的 time.perf_counter()
    # 会包含所有开销 → 7x slower
```

**验证方法:**

```bash
# 检查 nightly 日志中是否有 CUDA warnings
grep -i "cuda.*not available\|no cuda" nightly.log
```

### 假设 #3: 不同的 benchmark 配置

Nightly 可能使用了不同的参数：

```python
# 可能在某处设置了
_bench_meta.timing = "event"  # 而不是默认的 CUPTI
# 或
n_warmup = 0  # 没有 warmup
n_repeat = 10  # 重复次数少
```

**验证方法:**
检查 CI 脚本中是否有环境变量设置：

```bash
export TILEOPS_TIMING=wall
export TILEOPS_BENCHMARK_WARMUP=0
```

### 假设 #4: JIT 编译时间被计入

如果每次测试都重新编译 kernel（cache 失效）：

```python
# 第一次调用包含编译时间
latency_with_jit = 0.809 ms  # 我们测到的

# 但如果 n_repeat 很少，平均下来仍然很高
average_latency = (0.809 + 0.01 * 9) / 10 = 0.089 ms
```

接近但不完全匹配 0.0675 ms。

**验证方法:**
检查 TileLang cache 目录是否被清空：

```bash
# 如果每次测试前运行
rm -rf $TILELANG_CACHE_DIR/*
```

## 下一步行动

### 立即可做

1. **检查实际的 JUnit XML 文件**

   ```bash
   # 找到 nightly 生成的 XML
   cat bench_results.xml | grep -A 5 "L2NormFwd.*hidden-state"
   # 看 tileops_latency_ms 的值是多少
   ```

1. **检查 CI workflow 配置**

   ```bash
   cat .github/workflows/nightly.yml | grep -A 20 "pytest.*bench"
   # 看有没有 -n 参数或其他特殊配置
   ```

1. **检查环境变量**

   ```bash
   # 在 workflow 中搜索
   grep -r "TILEOPS\|BENCHMARK\|TIMING" .github/workflows/
   ```

### 需要你提供的信息

1. **Nightly 的 JUnit XML 文件**

   - `bench_results.xml` 的内容（至少 L2NormFwd 部分）

1. **CI workflow 文件**

   - `.github/workflows/nightly.yml` 或类似文件

1. **Nightly 的完整日志**

   - 看是否有 warnings 或 errors
   - 特别是 CUDA/GPU 相关的消息

1. **确认测试环境**

   - Nightly 在哪个/哪些 GPU 上运行？
   - 是否有多个 worker 并行？
   - 使用的是哪个 docker image？

## 临时解决方案

如果无法立即修复根本原因，可以：

1. **添加诊断日志**

   ```python
   # 在 bench_kernel 中
   logger.info(
       f"bench_kernel: latency={latency:.6f}ms, "
       f"n_warmup={n_warmup}, n_repeat={n_repeat}, "
       f"cuda_available={torch.cuda.is_available()}"
   )
   ```

1. **添加断言**

   ```python
   # 在 conftest.py 中
   latency = tileops_entry.get("latency_ms", 0)
   if latency > 0.05:  # 50us 是合理上限
       logger.warning(f"Suspicious latency {latency}ms for {op}")
   ```

1. **强制使用 CUPTI**

   ```python
   # 如果 CUPTI 失败就报错，不要 fallback
   if not torch.profiler.kineto_available():
       raise RuntimeError("CUPTI not available, cannot benchmark")
   ```

______________________________________________________________________

**结论:** 问题不在 L2 Norm kernel 本身，也不在 benchmark 代码。问题在于 nightly 测试运行时的某种配置或环境问题，导致测量结果包含了额外开销或使用了错误的测量方法。需要查看实际的 CI 配置和日志来定位。
