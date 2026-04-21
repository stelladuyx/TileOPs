"""
Benchmark: ssd_chunk_scan_bwd_dC

Measures the TileOPs kernel against the Mamba-2 Triton reference
(_chunk_scan_bwd_dC from mamba_ssm), falling back to the PyTorch
reference when mamba_ssm is not installed.

Kernel contract
---------------
  Forward branch differentiated:

    y_prev[b, c*L+l, h, p] = exp(dA_cumsum[b,h,c,l]) * sum_n C[b,c*L+l,g,n] * states[b,c,h,p,n]

  Outputs:
    dC_out              [B, S, G, N]  float32   S = C*L seqlen-fused
    ddA_cumsum_prev_out [B, H, C, L]  float32

Layout (official Mamba Triton conventions, same as mamba_ssm)
-------------------------------------------------------------
  states    [B, C, H, P, N]
  dA_cumsum [B, H, C, L]
  d_out      [B, S, H, P]     S = C*L seqlen-fused
  C_in      [B, S, G, N]     S = C*L seqlen-fused
"""
from typing import Optional

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tests.ops.test_ssd_chunk_scan_bwd_dC import (
    SsdChunkScanBwdDCTest,
    ssd_chunk_scan_bwd_dC_ref,
)
from tileops.kernels.mamba.ssd_chunk_scan_bwd_dC import SsdChunkScanBwdDCKernel

# ---------------------------------------------------------------------------
# Optional mamba_ssm Triton baseline
# ---------------------------------------------------------------------------
try:
    from mamba_ssm.ops.triton.ssd_chunk_scan import _chunk_scan_bwd_dC as _mamba_chunk_scan_bwd_dC
except ImportError:
    _mamba_chunk_scan_bwd_dC = None


def _to_mamba_inputs(
    states: torch.Tensor,       # [B, C, H, P, N]
    dA_cumsum: torch.Tensor,    # [B, H, C, L]     already canonical
    d_out: torch.Tensor,         # [B, S, H, P]     seqlen-fused
    C_in: torch.Tensor,         # [B, S, G, N]     seqlen-fused
):
    """Pass through — TileOPs now uses the same layout as mamba_ssm."""
    return states.contiguous(), dA_cumsum.contiguous(), d_out.contiguous(), C_in.contiguous()


# ---------------------------------------------------------------------------
# Benchmark class
# ---------------------------------------------------------------------------

class SsdChunkScanBwdDCBenchmark(BenchmarkBase):
    """FLOPs and memory bandwidth for ssd_chunk_scan_bwd_dC."""

    def calculate_flops(self) -> Optional[float]:
        t = self.workload
        b, c, L, h, p, n, g = (
            t.batch, t.num_chunks, t.chunk_len,
            t.n_heads, t.d_head, t.d_state, t.n_groups,
        )
        L_valid = t.chunk_len_valid
        # dc_base[b,c,h,l,n] = sum_p d_out[l,p] * states[p,n]
        #   GEMM [L_valid, P] x [P, N]  ->  L_valid * P * N * 2 MACs per (b, c, h)
        gemm_flops = b * c * h * L_valid * p * n * 2
        # ddA_prev[b,c,h,l] = sum_n dc_acc[l,n] * C[l,n]
        #   dot product of length N per (b, c, h, l)
        dda_flops = b * c * h * L_valid * n * 2
        return float(gemm_flops + dda_flops)

    def calculate_memory(self) -> Optional[float]:
        t = self.workload
        b, c, L, h, p, n, g = (
            t.batch, t.num_chunks, t.chunk_len,
            t.n_heads, t.d_head, t.d_state, t.n_groups,
        )
        S = c * L
        elem = torch.tensor([], dtype=t.dtype).element_size()
        # Reads (input dtype): states + d_out + C_in
        reads = (
            b * c * h * p * n  # states   [B, C, H, P, N]
            + b * S * h * p    # d_out     [B, S, H, P]
            + b * S * g * n    # C_in     [B, S, G, N]
        ) * elem
        # Reads (float32): dA_cumsum [B, H, C, L]
        reads += b * h * c * L * 4
        # Reads (int32): valid_chunk_len [B, C]
        reads += b * c * 4
        # Writes (float32): dC_out [B, S, G, N] + ddA_cumsum_prev_out [B, H, C, L]
        writes = (b * S * g * n + b * h * c * L) * 4
        return float(reads + writes)


# ---------------------------------------------------------------------------
# Benchmark parameters — aligned with bench_ssd_chunk_scan_fwd.py
#
# The forward kernel is the Mamba-2 prefill primitive; the backward dC kernel
# runs in the same model contexts. Cases cover unit-scale configs and
# realistic Mamba-2 model workloads.
#
# Model-to-shape mapping (Mamba-2 defaults):
#   n_heads = d_model / 32,  head_dim = 64,  d_state = 128,  chunk_len = 256
#   num_chunks = seq_len // chunk_len  (chunk_len=256: 2k->8, 4k->16, 32k->128)
#
#   130M -> n_heads=24   370M -> n_heads=32   780M -> n_heads=48
#   1.3B -> n_heads=64   2.7B -> n_heads=80
#
# n_groups=1 throughout: Mamba-2 standard (no C/B grouping).
# All chunks are complete (chunk_len_valid=None -> chunk_len).
#
# Schema: (batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype, tune)
# ---------------------------------------------------------------------------
_SSD_CHUNK_SCAN_BWD_DC_BENCH_PARAMS = [
    # ── unit-scale ──
    pytest.param(1, 2,  64, 4,  64,  32, 1, torch.float16,  False, id="b1-c2-L64-h4-p64-n32-fp16"),
    pytest.param(2, 4,  64, 8,  64,  64, 1, torch.float16,  False, id="b2-c4-L64-h8-p64-n64-fp16"),
    pytest.param(1, 2, 128, 4, 128,  32, 1, torch.bfloat16, False, id="b1-c2-L128-h4-p128-n32-bf16"),
    pytest.param(2, 2,  64, 4,  64,  32, 1, torch.bfloat16, False, id="b2-c2-L64-h4-p64-n32-bf16"),
    # ── 130M (n_heads=24) ──
    pytest.param(1,  16, 256, 24, 64, 128, 1, torch.float16, True, id="latency-130m-4k"),
    pytest.param(8,  16, 256, 24, 64, 128, 1, torch.float16, True, id="serving-130m-4k"),
    pytest.param(4, 128, 256, 24, 64, 128, 1, torch.float16, True, id="longctx-130m-32k"),
    # ── 370M (n_heads=32) ──
    pytest.param(1,  16, 256, 32, 64, 128, 1, torch.float16, True, id="latency-370m-4k"),
    pytest.param(8,  16, 256, 32, 64, 128, 1, torch.float16, True, id="serving-370m-4k"),
    pytest.param(4, 128, 256, 32, 64, 128, 1, torch.float16, True, id="longctx-370m-32k"),
    pytest.param(32,  8, 256, 32, 64, 128, 1, torch.float16, True, id="throughput-370m-2k"),
    # ── 780M (n_heads=48) ──
    pytest.param(1,  16, 256, 48, 64, 128, 1, torch.float16, True, id="latency-780m-4k"),
    pytest.param(8,  16, 256, 48, 64, 128, 1, torch.float16, True, id="serving-780m-4k"),
    pytest.param(4, 128, 256, 48, 64, 128, 1, torch.float16, True, id="longctx-780m-32k"),
    pytest.param(16,  8, 256, 48, 64, 128, 1, torch.float16, True, id="throughput-780m-2k"),
    # ── 1.3B (n_heads=64) ──
    pytest.param(1,  16, 256, 64, 64, 128, 1, torch.float16, True, id="latency-1p3b-4k"),
    pytest.param(8,  16, 256, 64, 64, 128, 1, torch.float16, True, id="serving-1p3b-4k"),
    pytest.param(2, 128, 256, 64, 64, 128, 1, torch.float16, True, id="longctx-1p3b-32k"),
    pytest.param(8,   8, 256, 64, 64, 128, 1, torch.float16, True, id="throughput-1p3b-2k"),
    # ── 2.7B (n_heads=80) ──
    pytest.param(1,  16, 256, 80, 64, 128, 1, torch.float16, True, id="latency-2p7b-4k"),
    pytest.param(4,  16, 256, 80, 64, 128, 1, torch.float16, True, id="serving-2p7b-4k"),
    pytest.param(2, 128, 256, 80, 64, 128, 1, torch.float16, True, id="longctx-2p7b-32k"),
    pytest.param(4,   8, 256, 80, 64, 128, 1, torch.float16, True, id="throughput-2p7b-2k"),
]


# ---------------------------------------------------------------------------
# Benchmark entry point
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype, tune",
    _SSD_CHUNK_SCAN_BWD_DC_BENCH_PARAMS,
)
def test_ssd_chunk_scan_bwd_dC_bench(
    batch: int,
    num_chunks: int,
    chunk_len: int,
    n_heads: int,
    d_head: int,
    d_state: int,
    n_groups: int,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    test = SsdChunkScanBwdDCTest(
        batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype,
        chunk_len_valid=None,  # complete chunks for perf benchmarking
    )
    bm = SsdChunkScanBwdDCBenchmark(test)
    inputs = test.gen_inputs()  # states, dA_cumsum, d_out, C_in, valid_chunk_len

    # ── TileOPs kernel ──
    kernel = SsdChunkScanBwdDCKernel(
        batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype, tune=tune,
    )
    result = bm.profile(kernel, *inputs)
    BenchmarkReport.record(kernel, locals(), result, tag="tileops")

    # ── Mamba-2 Triton baseline ──
    if _mamba_chunk_scan_bwd_dC is not None:
        states, dA_cumsum, d_out, C_in, valid_chunk_len = inputs
        states_m, dA_m, d_out_m, C_m = _to_mamba_inputs(states, dA_cumsum, d_out, C_in)

        def mamba_bwd_dC():
            # Pass C to also compute ddA_cumsum_prev, matching the full TileOPs output.
            return _mamba_chunk_scan_bwd_dC(
                states_m, dA_m, d_out_m, C=C_m, ngroups=n_groups,
            )

        result_mamba = bm.profile(mamba_bwd_dC)
        BenchmarkReport.record(kernel, locals(), result_mamba, tag="mamba")
    else:
        # Fall back to PyTorch reference when mamba_ssm is not installed.
        def torch_ref(states, dA_cumsum, d_out, C_in, valid_chunk_len):
            return ssd_chunk_scan_bwd_dC_ref(
                states, dA_cumsum, d_out, C_in, valid_chunk_len, n_groups,
            )

        result_bl = bm.profile(torch_ref, *inputs)
        BenchmarkReport.record(kernel, locals(), result_bl, tag="torch-ref")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
