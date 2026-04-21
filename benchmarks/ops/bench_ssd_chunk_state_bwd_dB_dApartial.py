from typing import Optional

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tests.ops.test_ssd_chunk_state_bwd_dB_dApartial import (
    SsdChunkStateBwdDBDApartialFixture,
    SsdChunkStateBwdDBDApartialTest,
    ssd_chunk_state_bwd_dB_dApartial_ref,
)
from tileops.kernels.mamba.ssd_chunk_state_bwd_dB_dApartial import SsdChunkStateBwdDBDApartialKernel


class SsdChunkStateBwdDBDApartialBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.test
        b, c, Q, h, p, n = (
            t.batch, t.num_chunks, t.chunk_len,
            t.n_heads, t.d_head, t.d_state,
        )
        # dB[l,n] = sum_p weighted_X[l,p] * g_u[p,n]:  Q*p*n*2 MACs per (b,c,h)
        # g_Z[l]  = sum_{p,n} g_u[p,n] * X[l,p] * B[l,n]: Q*p*n*2 MACs per (b,c,h)
        flops = b * c * h * Q * p * n * 2 * 2
        return float(flops)

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        b, c, Q, h, p, n, g = (
            t.batch, t.num_chunks, t.chunk_len,
            t.n_heads, t.d_head, t.d_state, t.n_groups,
        )
        S = c * Q  # seqlen-fused
        elem = torch.tensor([], dtype=t.dtype).element_size()
        # Reads (input dtype): X [B,S,H,P] + Bmat [B,S,G,N]
        reads = (
            b * S * h * p    # X
            + b * S * g * n  # Bmat
        ) * elem
        # Reads (float32): g_u [B,C,H,P,N] + chunk_state_decay [B,H,C,Q]
        reads += (b * c * h * p * n + b * h * c * Q) * 4
        # Writes (float32): dB [B,S,G,N] + g_Z [B,H,C,Q]
        writes = (b * S * g * n + b * h * c * Q) * 4
        return float(reads + writes)


@SsdChunkStateBwdDBDApartialFixture
def test_ssd_chunk_state_bwd_dB_dApartial_bench(
    batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype, tune,
):
    test = SsdChunkStateBwdDBDApartialTest(
        batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype,
    )
    bm = SsdChunkStateBwdDBDApartialBenchmark(test)
    inputs = test.gen_inputs()

    op = SsdChunkStateBwdDBDApartialKernel(
        batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype, tune=tune,
    )
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    def baseline(g_u, X, Bmat, chunk_state_decay):
        return ssd_chunk_state_bwd_dB_dApartial_ref(g_u, X, Bmat, chunk_state_decay, n_groups)
    result_bl = bm.profile(baseline, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch-ref")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
