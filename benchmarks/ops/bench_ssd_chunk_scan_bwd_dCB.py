"""
Benchmark: ssd_chunk_scan_bwd_dCB

Measures the TileOPs kernel against the Mamba-2 Triton reference
(_chunk_scan_bwd_dcb from mamba_ssm), falling back to the PyTorch
reference when mamba_ssm is not installed.

Kernel contract
---------------
  dCB[b, c, g, l, s] = sum_{h in group g} sum_p
      d_out[b,c*L+l,h,p] * exp(clamp(a_l - a_s, max=0)) * dt[b,h,c,s] * x[b,c*L+s,h,p]

Layout (official Mamba Triton conventions, same as mamba_ssm)
-------------------------------------------------------------
  x               [B, S, H, P]    dtype    S = C*L seqlen-fused
  dt              [B, H, C, L]    dtype
  dA_cumsum       [B, H, C, L]    float32
  d_out            [B, S, H, P]    dtype    S = C*L seqlen-fused
  valid_chunk_len [B, C]          int32
  output          [B, C, G, L, L] float32
"""
from typing import Optional

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tests.ops.test_ssd_chunk_scan_bwd_dCB import (
    SsdChunkScanBwdDCBTest,
    ssd_chunk_scan_bwd_dCB_ref,
)
from tileops.kernels.mamba.ssd_chunk_scan_bwd_dBC import SsdChunkScanBwdDCBKernel

# ---------------------------------------------------------------------------
# Optional mamba_ssm Triton baseline
# ---------------------------------------------------------------------------
try:
    from mamba_ssm.ops.triton.ssd_chunk_scan import _chunk_scan_bwd_dcb as _mamba_chunk_scan_bwd_dcb
except ImportError:
    _mamba_chunk_scan_bwd_dcb = None


def _to_mamba_inputs(
    x: torch.Tensor,            # [B, S, H, P]
    dt: torch.Tensor,           # [B, H, C, L]
    dA_cumsum: torch.Tensor,    # [B, H, C, L]
    d_out: torch.Tensor,         # [B, S, H, P]
):
    """Pass through — TileOPs now uses the same layout as mamba_ssm."""
    return x.contiguous(), dt.contiguous(), dA_cumsum.contiguous(), d_out.contiguous()


# ---------------------------------------------------------------------------
# Benchmark class
# ---------------------------------------------------------------------------

class SsdChunkScanBwdDCBBenchmark(BenchmarkBase):
    """FLOPs and memory bandwidth for ssd_chunk_scan_bwd_dCB."""

    def calculate_flops(self) -> Optional[float]:
        t = self.workload
        b, c, L, h, p, g = (
            t.batch, t.num_chunks, t.chunk_len,
            t.n_heads, t.d_head, t.n_groups,
        )
        # pair_base[b,c,h,l,s] = sum_p d_out[l,p] * x[s,p]
        #   GEMM [L, P] x [P, L] -> L*L*P*2 MACs per (b, c, h)
        return float(b * c * h * L * L * p * 2)

    def calculate_memory(self) -> Optional[float]:
        t = self.workload
        b, c, L, h, p, g = (
            t.batch, t.num_chunks, t.chunk_len,
            t.n_heads, t.d_head, t.n_groups,
        )
        S = c * L
        elem = torch.tensor([], dtype=t.dtype).element_size()
        # Reads (input dtype): x [B, S, H, P] + d_out [B, S, H, P]
        reads = (b * S * h * p + b * S * h * p) * elem
        # Reads (input dtype): dt [B, H, C, L]
        reads += b * h * c * L * elem
        # Reads (float32): dA_cumsum [B, H, C, L]
        reads += b * h * c * L * 4
        # Reads (int32): valid_chunk_len [B, C]
        reads += b * c * 4
        # Writes (float32): dCB_out [B, C, G, L, L]
        writes = b * c * g * L * L * 4
        return float(reads + writes)


# ---------------------------------------------------------------------------
# Benchmark parameters — aligned with bench_ssd_chunk_scan_bwd_dC.py
#
# Model-to-shape mapping (Mamba-2 defaults):
#   n_heads = d_model / 32,  head_dim = 64,  chunk_len = 256
#   num_chunks = seq_len // chunk_len
#   n_groups = 1 (Mamba-2 standard)
#
# Schema: (batch, num_chunks, chunk_len, n_heads, d_head, n_groups, dtype, tune)
# ---------------------------------------------------------------------------
_SSD_CHUNK_SCAN_BWD_DCB_BENCH_PARAMS = [
    # ── unit-scale ──
    pytest.param(1, 2,  64, 4,  64, 1, torch.float16,  False, id="b1-c2-L64-h4-p64-fp16"),
    pytest.param(2, 4,  64, 8,  64, 2, torch.float16,  False, id="b2-c4-L64-h8-p64-fp16"),
    pytest.param(1, 2,  64, 4,  64, 1, torch.bfloat16, False, id="b1-c2-L64-h4-p64-bf16"),
    pytest.param(2, 2,  64, 4,  64, 2, torch.bfloat16, False, id="b2-c2-L64-h4-p64-bf16"),
    # ── 130M (n_heads=24) ──
    pytest.param(1,  16, 256, 24, 64, 1, torch.float16, True, id="latency-130m-4k"),
    pytest.param(8,  16, 256, 24, 64, 1, torch.float16, True, id="serving-130m-4k"),
    pytest.param(4, 128, 256, 24, 64, 1, torch.float16, True, id="longctx-130m-32k"),
    # ── 370M (n_heads=32) ──
    pytest.param(1,  16, 256, 32, 64, 1, torch.float16, True, id="latency-370m-4k"),
    pytest.param(8,  16, 256, 32, 64, 1, torch.float16, True, id="serving-370m-4k"),
    pytest.param(4, 128, 256, 32, 64, 1, torch.float16, True, id="longctx-370m-32k"),
    pytest.param(32,  8, 256, 32, 64, 1, torch.float16, True, id="throughput-370m-2k"),
    # ── 780M (n_heads=48) ──
    pytest.param(1,  16, 256, 48, 64, 1, torch.float16, True, id="latency-780m-4k"),
    pytest.param(8,  16, 256, 48, 64, 1, torch.float16, True, id="serving-780m-4k"),
    pytest.param(4, 128, 256, 48, 64, 1, torch.float16, True, id="longctx-780m-32k"),
    pytest.param(16,  8, 256, 48, 64, 1, torch.float16, True, id="throughput-780m-2k"),
    # ── 1.3B (n_heads=64) ──
    pytest.param(1,  16, 256, 64, 64, 1, torch.float16, True, id="latency-1p3b-4k"),
    pytest.param(8,  16, 256, 64, 64, 1, torch.float16, True, id="serving-1p3b-4k"),
    pytest.param(2, 128, 256, 64, 64, 1, torch.float16, True, id="longctx-1p3b-32k"),
    pytest.param(8,   8, 256, 64, 64, 1, torch.float16, True, id="throughput-1p3b-2k"),
    # ── 2.7B (n_heads=80) ──
    pytest.param(1,  16, 256, 80, 64, 1, torch.float16, True, id="latency-2p7b-4k"),
    pytest.param(4,  16, 256, 80, 64, 1, torch.float16, True, id="serving-2p7b-4k"),
    pytest.param(2, 128, 256, 80, 64, 1, torch.float16, True, id="longctx-2p7b-32k"),
    pytest.param(4,   8, 256, 80, 64, 1, torch.float16, True, id="throughput-2p7b-2k"),
]


# ---------------------------------------------------------------------------
# Benchmark entry point
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "batch, num_chunks, chunk_len, n_heads, d_head, n_groups, dtype, tune",
    _SSD_CHUNK_SCAN_BWD_DCB_BENCH_PARAMS,
)
def test_ssd_chunk_scan_bwd_dCB_bench(
    batch: int,
    num_chunks: int,
    chunk_len: int,
    n_heads: int,
    d_head: int,
    n_groups: int,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    test = SsdChunkScanBwdDCBTest(
        batch, num_chunks, chunk_len, n_heads, d_head, n_groups, dtype,
        chunk_len_valid=None,  # complete chunks for perf benchmarking
    )
    bm = SsdChunkScanBwdDCBBenchmark(test)
    inputs = test.gen_inputs()  # x, dt, dA_cumsum, d_out, valid_chunk_len

    # ── TileOPs kernel ──
    kernel = SsdChunkScanBwdDCBKernel(
        batch, num_chunks, chunk_len, n_heads, d_head, n_groups, dtype, tune=tune,
    )
    result = bm.profile(kernel, *inputs)
    BenchmarkReport.record(kernel, locals(), result, tag="tileops")

    # ── Mamba-2 Triton baseline ──
    if _mamba_chunk_scan_bwd_dcb is not None:
        x, dt, dA_cumsum, d_out, valid_chunk_len = inputs
        x_m, dt_m, dA_m, d_out_m = _to_mamba_inputs(x, dt, dA_cumsum, d_out)

        def mamba_bwd_dcb():
            dcb = _mamba_chunk_scan_bwd_dcb(
                x_m, dt_m, dA_m, d_out_m, seq_idx=None, CB=None, ngroups=n_groups,
            )
            # dcb: [B, C, nsplits, G, L, L] -> sum splits -> [B, C, G, L, L]
            return dcb.sum(dim=2)

        result_mamba = bm.profile(mamba_bwd_dcb)
        BenchmarkReport.record(kernel, locals(), result_mamba, tag="mamba")
    else:
        # Fall back to PyTorch reference when mamba_ssm is not installed.
        def torch_ref(x, dt, dA_cumsum, d_out, valid_chunk_len):
            return ssd_chunk_scan_bwd_dCB_ref(
                x, dt, dA_cumsum, d_out, valid_chunk_len, n_groups,
            )

        result_bl = bm.profile(torch_ref, *inputs)
        BenchmarkReport.record(kernel, locals(), result_bl, tag="torch-ref")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
