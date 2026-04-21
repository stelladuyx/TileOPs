"""
Mamba-2 SSD state passing backward kernel.

Forward recurrence:
  s_c = alpha_c * s_{c-1} + u_c

where alpha_c = exp(dA_chunk_cumsum[b, h, c]).

Saved tensor:
  states_before_chunk[b, c, h, :, :] = s_{c-1}
  (so states_before_chunk[b, 0, h] = s_{-1}, the initial state)

Backward formulas for chunk c (scanning c = C-1 ... 0):

  g_u[c]               = g_sc
  g_dA_chunk_cumsum[c] = alpha_c * sum_{p,n}(g_sc * s_{c-1})
  g_{s_{c-1}}          = g_states_readout_in[c] + alpha_c * g_sc

Inputs:
  g_states_readout_in  : (B, C, H, P, N)  -- direct grad onto s_{c-1} from chunk_scan
  dA_chunk_cumsum      : (B, H, C)         -- log-decay per chunk; alpha_c = exp(dA_chunk_cumsum[b,h,c])
  states_before_chunk  : (B, C, H, P, N)  -- saved s_{c-1} per chunk
  g_final_state_in     : (B, H, P, N)     -- upstream grad for s_{C-1} (zeros if unused)

Outputs (always 4, caller ignores unused ones):
  g_u_out              : (B, C, H, P, N)  -- gradient w.r.t. u_c
  g_dA_chunk_cumsum_out: (B, H, C)        -- gradient w.r.t. dA_chunk_cumsum (atomic-added)
  g_initial_states_out : (B, H, P, N)     -- gradient w.r.t. s_{-1} (zeros if unused)
  states_out           : (B, C, H, P, N)  -- pass-through cast of states_before_chunk
                                              (zeros if unused)

Parallelization:
  axis-0: fused batch*head (B*H)
  axis-1: tile over P (d_head)
  axis-2: tile over N (d_state)
  chunk dimension C is scanned serially (backward) inside each program.

Notation:
  B = batch, H = n_heads, C = num_chunks, P = d_head, N = d_state
"""

import itertools
from typing import Callable, Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel import Kernel

__all__ = ["SsdStatePassingBwdKernel"]


def _ssd_state_passing_bwd_kernel(
    batch: int,
    num_chunks: int,
    n_heads: int,
    d_head: int,
    d_state: int,
    has_final_state_grad: bool = False,
    has_initial_states: bool = False,
    return_states: bool = False,
    dtype: str = "float16",
) -> Callable:
    accum_dtype = "float"

    B = batch
    C = num_chunks
    H = n_heads
    P = d_head
    N = d_state

    # No out_idx: all outputs are pre-allocated as zeros by the caller
    # so that T.atomic_add on g_dA_chunk_cumsum_out is correct.
    @tilelang.jit()
    def kernel_func(
        block_p: int,
        block_n: int,
        threads: int,
    ):
        @T.prim_func
        def main(
            g_states_readout_in:   T.Tensor((B, C, H, P, N), accum_dtype),   # type: ignore
            dA_chunk_cumsum:       T.Tensor((B, H, C), accum_dtype),          # type: ignore
            states_before_chunk:   T.Tensor((B, C, H, P, N), dtype),          # type: ignore
            g_final_state_in:      T.Tensor((B, H, P, N), accum_dtype),       # type: ignore
            g_u_out:               T.Tensor((B, C, H, P, N), accum_dtype),    # type: ignore
            g_dA_chunk_cumsum_out: T.Tensor((B, H, C), accum_dtype),          # type: ignore
            g_initial_states_out:  T.Tensor((B, H, P, N), accum_dtype),       # type: ignore
            states_out:            T.Tensor((B, C, H, P, N), accum_dtype),    # type: ignore
        ):
            with T.Kernel(
                B * H,
                T.ceildiv(P, block_p),
                T.ceildiv(N, block_n),
                threads=threads,
            ) as (bh_fused, bp, bn):

                bb = bh_fused // H
                bh = bh_fused % H
                p0 = bp * block_p
                n0 = bn * block_n

                g_sc      = T.alloc_fragment((block_p, block_n), accum_dtype)
                s_prev    = T.alloc_fragment((block_p, block_n), accum_dtype)
                g_readout = T.alloc_fragment((block_p, block_n), accum_dtype)
                dot_col_shared = T.alloc_shared((block_n,), accum_dtype)
                dot_scalar_shared = T.alloc_shared((1,), accum_dtype)

                # --------------------------------------------------------
                # Initialize g_sc = g_{s_{C-1}}
                # --------------------------------------------------------
                if has_final_state_grad:
                    for i, j in T.Parallel(block_p, block_n):
                        pi = p0 + i
                        nj = n0 + j
                        g_sc[i, j] = T.if_then_else(
                            (pi < P) and (nj < N),
                            g_final_state_in[bb, bh, pi, nj],
                            T.float32(0.0),
                        )
                else:
                    T.clear(g_sc)

                # --------------------------------------------------------
                # Backward scan: c = C-1, C-2, ..., 1
                # --------------------------------------------------------
                for c in T.serial(C - 1, 0, -1):
                    for i, j in T.Parallel(block_p, block_n):
                        pi = p0 + i
                        nj = n0 + j
                        s_prev[i, j] = T.if_then_else(
                            (pi < P) and (nj < N),
                            T.cast(states_before_chunk[bb, c, bh, pi, nj], accum_dtype),
                            T.float32(0.0),
                        )

                    if return_states:
                        for i, j in T.Parallel(block_p, block_n):
                            pi = p0 + i
                            nj = n0 + j
                            if pi < P and nj < N:
                                states_out[bb, c, bh, pi, nj] = s_prev[i, j]

                    alpha_c = T.exp(dA_chunk_cumsum[bb, bh, c])

                    # 1) g_u[c] = g_sc
                    for i, j in T.Parallel(block_p, block_n):
                        pi = p0 + i
                        nj = n0 + j
                        if pi < P and nj < N:
                            g_u_out[bb, c, bh, pi, nj] = g_sc[i, j]

                    # 2) g_dA[c] = alpha_c * sum(g_sc * s_{c-1})
                    # Step a: elementwise product into fragment
                    dot_tile = T.alloc_fragment((block_p, block_n), accum_dtype)
                    for i, j in T.Parallel(block_p, block_n):
                        pi = p0 + i
                        nj = n0 + j
                        dot_tile[i, j] = T.if_then_else(
                            (pi < P) and (nj < N),
                            g_sc[i, j] * s_prev[i, j],
                            T.float32(0.0),
                        )
                    # Step b: reduce (block_p, block_n) -> col fragment -> shared -> scalar
                    dot_col = T.alloc_fragment((block_n,), accum_dtype)
                    T.reduce_sum(dot_tile, dot_col, dim=0)
                    for j in T.Parallel(block_n):
                        dot_col_shared[j] = dot_col[j]
                    T.sync_threads()
                    tx = T.get_thread_binding()
                    if tx == 0:
                        dot_scalar_shared[0] = T.float32(0.0)
                        for j in T.serial(block_n):
                            dot_scalar_shared[0] += dot_col_shared[j]
                        T.atomic_add(g_dA_chunk_cumsum_out[bb, bh, c],
                                     dot_scalar_shared[0] * alpha_c)
                    T.sync_threads()

                    # 3) g_{s_{c-1}} = g_readout[c] + alpha_c * g_sc
                    for i, j in T.Parallel(block_p, block_n):
                        pi = p0 + i
                        nj = n0 + j
                        g_readout[i, j] = T.if_then_else(
                            (pi < P) and (nj < N),
                            g_states_readout_in[bb, c, bh, pi, nj],
                            T.float32(0.0),
                        )
                    for i, j in T.Parallel(block_p, block_n):
                        g_sc[i, j] = g_readout[i, j] + alpha_c * g_sc[i, j]

                # --------------------------------------------------------
                # c = 0: s_0 = alpha_0 * s_{-1} + u_0
                # states_before_chunk[b, h, 0] = s_{-1}
                # --------------------------------------------------------
                for i, j in T.Parallel(block_p, block_n):
                    pi = p0 + i
                    nj = n0 + j
                    s_prev[i, j] = T.if_then_else(
                        (pi < P) and (nj < N),
                        T.cast(states_before_chunk[bb, 0, bh, pi, nj], accum_dtype),
                        T.float32(0.0),
                    )

                if return_states:
                    for i, j in T.Parallel(block_p, block_n):
                        pi = p0 + i
                        nj = n0 + j
                        if pi < P and nj < N:
                            states_out[bb, 0, bh, pi, nj] = s_prev[i, j]

                alpha_0 = T.exp(dA_chunk_cumsum[bb, bh, 0])

                # g_u[0] = g_sc
                for i, j in T.Parallel(block_p, block_n):
                    pi = p0 + i
                    nj = n0 + j
                    if pi < P and nj < N:
                        g_u_out[bb, 0, bh, pi, nj] = g_sc[i, j]

                if has_initial_states:
                    # g_dA[0] = alpha_0 * sum(g_sc * s_{-1})
                    dot_tile0 = T.alloc_fragment((block_p, block_n), accum_dtype)
                    for i, j in T.Parallel(block_p, block_n):
                        pi = p0 + i
                        nj = n0 + j
                        dot_tile0[i, j] = T.if_then_else(
                            (pi < P) and (nj < N),
                            g_sc[i, j] * s_prev[i, j],
                            T.float32(0.0),
                        )
                    dot_col0 = T.alloc_fragment((block_n,), accum_dtype)
                    T.reduce_sum(dot_tile0, dot_col0, dim=0)
                    for j in T.Parallel(block_n):
                        dot_col_shared[j] = dot_col0[j]
                    T.sync_threads()
                    tx = T.get_thread_binding()
                    if tx == 0:
                        dot_scalar_shared[0] = T.float32(0.0)
                        for j in T.serial(block_n):
                            dot_scalar_shared[0] += dot_col_shared[j]
                        T.atomic_add(g_dA_chunk_cumsum_out[bb, bh, 0],
                                     dot_scalar_shared[0] * alpha_0)
                    T.sync_threads()

                    # g_{s_{-1}} = g_readout[0] + alpha_0 * g_sc
                    for i, j in T.Parallel(block_p, block_n):
                        pi = p0 + i
                        nj = n0 + j
                        g_readout[i, j] = T.if_then_else(
                            (pi < P) and (nj < N),
                            g_states_readout_in[bb, 0, bh, pi, nj],
                            T.float32(0.0),
                        )
                    for i, j in T.Parallel(block_p, block_n):
                        pi = p0 + i
                        nj = n0 + j
                        if pi < P and nj < N:
                            g_initial_states_out[bb, bh, pi, nj] = (
                                g_readout[i, j] + alpha_0 * g_sc[i, j]
                            )
                # else: s_{-1} is constant zero; g_dA[0] stays zero, g_initial_states_out unused

        return main

    return kernel_func


@torch.library.custom_op("top::ssd_state_passing_bwd", mutates_args=())
def _ssd_state_passing_bwd_wrapped(
    batch: int,
    num_chunks: int,
    n_heads: int,
    d_head: int,
    d_state: int,
    has_final_state_grad: bool,
    has_initial_states: bool,
    return_states: bool,
    dtype: str,
    block_p: int,
    block_n: int,
    threads: int,
    g_states_readout_in: torch.Tensor,
    dA_chunk_cumsum: torch.Tensor,
    states_before_chunk: torch.Tensor,
    g_final_state_in: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    f32 = torch.float32
    dev = g_states_readout_in.device
    g_u    = torch.zeros(batch, num_chunks, n_heads, d_head, d_state, dtype=f32, device=dev)
    g_dA   = torch.zeros(batch, n_heads, num_chunks, dtype=f32, device=dev)
    g_init = torch.zeros(batch, n_heads, d_head, d_state, dtype=f32, device=dev)
    s_out  = torch.zeros(batch, num_chunks, n_heads, d_head, d_state, dtype=f32, device=dev)
    _ssd_state_passing_bwd_kernel(
        batch, num_chunks, n_heads, d_head, d_state,
        has_final_state_grad, has_initial_states, return_states, dtype,
    )(block_p, block_n, threads)(
        g_states_readout_in, dA_chunk_cumsum, states_before_chunk, g_final_state_in,
        g_u, g_dA, g_init, s_out,
    )
    return g_u, g_dA, g_init, s_out


@_ssd_state_passing_bwd_wrapped.register_fake
def _(
    batch: int,
    num_chunks: int,
    n_heads: int,
    d_head: int,
    d_state: int,
    has_final_state_grad: bool,
    has_initial_states: bool,
    return_states: bool,
    dtype: str,
    block_p: int,
    block_n: int,
    threads: int,
    g_states_readout_in: torch.Tensor,
    dA_chunk_cumsum: torch.Tensor,
    states_before_chunk: torch.Tensor,
    g_final_state_in: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    f32 = torch.float32
    g_u    = g_states_readout_in.new_empty((batch, num_chunks, n_heads, d_head, d_state), dtype=f32)
    g_dA   = dA_chunk_cumsum.new_empty((batch, n_heads, num_chunks), dtype=f32)
    g_init = g_states_readout_in.new_empty((batch, n_heads, d_head, d_state), dtype=f32)
    s_out  = states_before_chunk.new_empty((batch, num_chunks, n_heads, d_head, d_state), dtype=f32)
    return g_u, g_dA, g_init, s_out


class SsdStatePassingBwdKernel(Kernel):
    """Mamba-2 SSD state passing backward kernel.

    Computes gradients for the inter-chunk recurrent scan:

      s_c = exp(dA_chunk_cumsum[b, h, c]) * s_{c-1} + u_c

    Scanning c = C-1 ... 0:

      g_u[c]               = g_sc
      g_dA_chunk_cumsum[c] = alpha_c * sum(g_sc * s_{c-1})
      g_{s_{c-1}}          = g_states_readout_in[c] + alpha_c * g_sc

    Always returns 4 tensors; caller ignores unused ones based on flags:
      g_u_out               (B, C, H, P, N), float32  -- always valid
      g_dA_chunk_cumsum_out (B, H, C),       float32  -- always valid
      g_initial_states_out  (B, H, P, N),    float32  -- valid if has_initial_states
      states_out            (B, C, H, P, N), float32  -- valid if return_states
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    def __init__(
        self,
        batch: int,
        num_chunks: int,
        n_heads: int,
        d_head: int,
        d_state: int,
        has_final_state_grad: bool = False,
        has_initial_states: bool = False,
        return_states: bool = False,
        dtype: torch.dtype = torch.float16,
        config: Optional[dict] = None,
        tune: bool = False,
    ) -> None:
        super().__init__()
        self.batch = batch
        self.num_chunks = num_chunks
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_state = d_state
        self.has_final_state_grad = has_final_state_grad
        self.has_initial_states = has_initial_states
        self.return_states = return_states
        self.dtype = dtype
        self.kernel = _ssd_state_passing_bwd_kernel(
            batch, num_chunks, n_heads, d_head, d_state,
            has_final_state_grad, has_initial_states, return_states, self.dtype_str,
        )
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {
            "block_p": 32,
            "block_n": 32,
            "threads": 128,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        block_p = [16, 32, 64]
        block_n = [16, 32, 64]
        threads = [128, 256]
        _configs = list(itertools.product(block_p, block_n, threads))
        return [{"block_p": c[0], "block_n": c[1], "threads": c[2]} for c in _configs]

    def forward(
        self,
        g_states_readout_in: torch.Tensor,
        dA_chunk_cumsum: torch.Tensor,
        states_before_chunk: torch.Tensor,
        g_final_state_in: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            g_states_readout_in:  (B, C, H, P, N) float32
            dA_chunk_cumsum:      (B, H, C)        float32  -- log-decay per chunk
            states_before_chunk:  (B, C, H, P, N) dtype    = saved s_{c-1}
            g_final_state_in:     (B, H, P, N)    float32  optional; pass None if unused

        Returns:
            g_u_out, g_dA_chunk_cumsum_out, g_initial_states_out, states_out
        """
        if g_final_state_in is None:
            g_final_state_in = g_states_readout_in.new_zeros(
                self.batch, self.n_heads, self.d_head, self.d_state)
        return _ssd_state_passing_bwd_wrapped(
            self.batch, self.num_chunks, self.n_heads, self.d_head, self.d_state,
            self.has_final_state_grad, self.has_initial_states, self.return_states,
            self.dtype_str,
            self.config["block_p"], self.config["block_n"], self.config["threads"],
            g_states_readout_in, dA_chunk_cumsum, states_before_chunk, g_final_state_in,
        )
