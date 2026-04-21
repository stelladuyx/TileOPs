"""Benchmark: TileOPs SsdStatePassingBwd vs mamba_ssm Triton _state_passing_bwd.

When mamba_ssm is not installed, falls back to a pure-torch reference baseline
so the benchmark can still run in CI without the optional dependency.

Layout notes:
  TileOPs:
    states / g_u_out:       [B, C, H, P, N]
    dA_chunk_cumsum:        [B, H, C]        -- log-decay per chunk (official layout)
    g_dA_chunk_cumsum_out:  [B, H, C]
    g_final_state_in:       [B, H, P, N]

  mamba_ssm Triton _state_passing_bwd expects:
    states:            [B, C, H, P*N]  (flattened state dim)
    dA_chunk_cumsum:   [B, H, C]       (same layout as TileOPs after fix)
    dout:              [B, C, H, P*N]  (flattened, upstream grad)
    dfinal_states:     [B, H, P*N]     optional
"""

from typing import Optional

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tests.ops.test_ssd_state_passing_bwd import (
    SsdStatePassingBwdTest,
    ssd_state_passing_bwd_ref,
)
from tileops.kernels.mamba import SsdStatePassingBwdKernel

try:
    from mamba_ssm.ops.triton.ssd_state_passing import (
        _state_passing_bwd as _triton_state_passing_bwd,
    )
except ImportError:
    _triton_state_passing_bwd = None


class SsdStatePassingBwdBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.workload
        B, C, H, P, N = t.batch, t.num_chunks, t.n_heads, t.d_head, t.d_state
        # Per chunk per (b, h): g_dA needs dot(g_sc, s_prev) -> 2*P*N FLOPs
        # g_u assignment + g_sc propagation are memory-bound, not compute-bound
        # Dominant: g_dA reduction: B * C * H * P * N * 2
        return float(B * C * H * P * N * 2)

    def calculate_memory(self) -> Optional[float]:
        t = self.workload
        B, C, H, P, N = t.batch, t.num_chunks, t.n_heads, t.d_head, t.d_state
        elem_in = torch.tensor([], dtype=t.dtype).element_size()
        f32 = 4
        # Reads:
        #   g_states_readout_in [B,C,H,P,N] float32
        #   dA_chunk_cumsum     [B,H,C]      float32
        #   states_before_chunk [B,C,H,P,N]  dtype
        reads = (
            B * C * H * P * N * f32       # g_states_readout_in
            + B * H * C * f32             # dA_chunk_cumsum
            + B * C * H * P * N * elem_in # states_before_chunk
        )
        if t.has_final_state_grad:
            reads += B * H * P * N * f32  # g_final_state_in
        # Writes:
        #   g_u_out  [B,C,H,P,N] float32
        #   g_dA_out [B,H,C]     float32
        writes = B * C * H * P * N * f32 + B * H * C * f32
        if t.has_initial_states:
            writes += B * H * P * N * f32  # g_initial_states_out
        if t.return_states:
            writes += B * C * H * P * N * f32  # states_out
        return float(reads + writes)


_BENCH_PARAMS = [
    # (batch, num_chunks, n_heads, d_head, d_state, dtype, has_final_state_grad, has_initial_states, return_states, tune)
    pytest.param(1, 8,  4,  64,  64,  torch.float16,  False, False, False, False, id="b1-c8-h4-p64-n64-fp16"),
    pytest.param(2, 16, 8,  64,  64,  torch.float16,  False, False, False, False, id="b2-c16-h8-p64-n64-fp16"),
    pytest.param(2, 16, 8,  64,  64,  torch.bfloat16, False, False, False, False, id="b2-c16-h8-p64-n64-bf16"),
    pytest.param(1, 8,  4,  64,  64,  torch.float16,  True,  True,  False, False, id="b1-c8-h4-p64-n64-fp16-full"),
    pytest.param(2, 16, 24, 128, 128, torch.float16,  False, False, False, False, id="b2-c16-h24-p128-n128-fp16"),
]


@pytest.mark.parametrize(
    "batch, num_chunks, n_heads, d_head, d_state, dtype, "
    "has_final_state_grad, has_initial_states, return_states, tune",
    _BENCH_PARAMS,
)
def test_ssd_state_passing_bwd_bench(
    batch, num_chunks, n_heads, d_head, d_state, dtype,
    has_final_state_grad, has_initial_states, return_states, tune,
):
    test = SsdStatePassingBwdTest(
        batch, num_chunks, n_heads, d_head, d_state, dtype,
        has_final_state_grad, has_initial_states, return_states,
    )
    bm = SsdStatePassingBwdBenchmark(test)
    inputs = test.gen_inputs()  # g_readout, dA_chunk_cumsum, states, g_final

    # ── TileOPs kernel ────────────────────────────────────────────────────────
    kernel = SsdStatePassingBwdKernel(
        batch, num_chunks, n_heads, d_head, d_state,
        has_final_state_grad=has_final_state_grad,
        has_initial_states=has_initial_states,
        return_states=return_states,
        dtype=dtype,
        tune=tune,
    )
    result = bm.profile(kernel, *inputs)
    BenchmarkReport.record(kernel, locals(), result, tag="tileops")

    # ── mamba_ssm Triton baseline ─────────────────────────────────────────────
    if _triton_state_passing_bwd is not None:
        g_readout, dA_chunk_cumsum, states, g_final = inputs
        B, C, H, P, N = batch, num_chunks, n_heads, d_head, d_state

        # Triton expects:
        #   states:           [B, C, H, P*N]  (our [B,C,H,P,N] flattened)
        #   dA_chunk_cumsum:  [B, H, C]       (same layout as TileOPs)
        #   dout:             [B, C, H, P*N]  (g_readout flattened)
        #   dfinal_states:    [B, H, P*N]     optional
        states_flat    = states.reshape(B, C, H, P * N).contiguous()
        dout_flat      = g_readout.reshape(B, C, H, P * N).contiguous()
        dfinal_flat    = (g_final.reshape(B, H, P * N).contiguous()
                          if g_final is not None else None)

        def triton_bwd():
            return _triton_state_passing_bwd(
                states_flat,
                dA_chunk_cumsum,
                dout_flat,
                dfinal_states=dfinal_flat,
                has_initial_states=has_initial_states,
                dstates_dtype=torch.float32,
                states_dtype=dtype,
            )

        result_tri = bm.profile(triton_bwd)
        BenchmarkReport.record(kernel, locals(), result_tri, tag="triton")

    else:
        # fallback: pure-torch reference
        def torch_ref(*args):
            return ssd_state_passing_bwd_ref(
                *args,
                has_initial_states=has_initial_states,
                return_states=return_states,
            )

        result_bl = bm.profile(torch_ref, *inputs)
        BenchmarkReport.record(kernel, locals(), result_bl, tag="torch-ref")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
