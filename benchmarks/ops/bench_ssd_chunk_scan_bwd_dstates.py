from typing import Optional

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tests.ops.test_ssd_chunk_scan_bwd_dstates import (
    SsdChunkScanBwdDstatesFixture,
    SsdChunkScanBwdDstatesTest,
    ssd_chunk_scan_bwd_dstates_ref,
)
from tileops.kernels.mamba.ssd_chunk_scan_bwd_dstates import SsdChunkScanBwdDstatesKernel

# ---------------------------------------------------------------------------
# Optional mamba_ssm Triton baseline
# ---------------------------------------------------------------------------
try:
    from mamba_ssm.ops.triton.ssd_chunk_scan import (
        _chunk_scan_bwd_dstates as _mamba_chunk_scan_bwd_dstates,
    )
except ImportError:
    _mamba_chunk_scan_bwd_dstates = None


class SsdChunkScanBwdDstatesBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.workload
        b, c, Q, h, p, n = (
            t.batch, t.num_chunks, t.chunk_len,
            t.n_heads, t.d_head, t.d_state,
        )
        # dstates[b,c,h,p,n] = sum_l d_out[l,h,p] * C[l,h,n] * exp(dA_cumsum[l])
        # einsum("bclhp,bclhn->bchpn") over L_valid positions
        L_valid = t.chunk_len_valid
        return float(b * c * h * n * p * 2 * L_valid)

    def calculate_memory(self) -> Optional[float]:
        t = self.workload
        b, c, Q, h, p, n = (
            t.batch, t.num_chunks, t.chunk_len,
            t.n_heads, t.d_head, t.d_state,
        )
        S = c * Q
        elem = torch.tensor([], dtype=t.dtype).element_size()
        # reads (dtype): d_out [B,S,H,P] + C [B,S,H,N]
        reads = (b * S * h * p + b * S * h * n) * elem
        # reads (float32): dA_cumsum [B,H,C,L]
        reads += b * h * c * Q * 4
        # reads (int32): valid_chunk_len [B, C]
        reads += b * c * 4
        # writes (float32): dstates [B,C,H,P,N]
        writes = b * c * h * p * n * 4
        return float(reads + writes)


@SsdChunkScanBwdDstatesFixture
def test_ssd_chunk_scan_bwd_dstates_bench(
    batch, num_chunks, chunk_len, n_heads, d_head, d_state, dtype, chunk_len_valid, tune,
):
    test = SsdChunkScanBwdDstatesTest(
        batch, num_chunks, chunk_len, n_heads, d_head, d_state, dtype, chunk_len_valid,
    )
    bm = SsdChunkScanBwdDstatesBenchmark(test)
    inputs = test.gen_inputs()  # d_out, C, dA_cumsum, valid_chunk_len

    kernel = SsdChunkScanBwdDstatesKernel(
        batch, num_chunks, chunk_len, n_heads, d_head, d_state, dtype, tune=tune,
    )
    result = bm.profile(kernel, *inputs)
    BenchmarkReport.record(kernel, locals(), result, tag="tileops")

    # ── Mamba-2 Triton baseline ──
    if _mamba_chunk_scan_bwd_dstates is not None:
        d_out, C, dA_cumsum, valid_chunk_len_t = inputs

        def mamba_bwd_dstates():
            return _mamba_chunk_scan_bwd_dstates(
                d_out.contiguous(), dA_cumsum.contiguous(), C.contiguous(),
                ngroups=1,
            )

        result_mamba = bm.profile(mamba_bwd_dstates)
        BenchmarkReport.record(kernel, locals(), result_mamba, tag="mamba")
    else:
        result_bl = bm.profile(ssd_chunk_scan_bwd_dstates_ref, *inputs)
        BenchmarkReport.record(kernel, locals(), result_bl, tag="torch-ref")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
