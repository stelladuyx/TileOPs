"""
Mamba-2 SSD chunk scan backward kernel: gradient w.r.t. C and dA_cumsum (prev-state branch).

Corresponds to Triton _chunk_scan_bwd_dC (K4).

Mathematical contract
---------------------
Forward branch being differentiated:

  y_prev[b, s, h, p]
    = exp(dA_cumsum[b, h, c(s), l(s)]) * sum_n C[b, s, g(h), n] * states[b, c(s), h, p, n]

where s = c*L + l, c(s) = s // L, l(s) = s % L, g(h) = h // HEADS_PER_GROUP.

Intermediate:

  dc_base[b,c,h,l,n] = sum_p d_out[b, c*L+l, h, p] * states[b,c,h,p,n]
  dc_acc[b,c,h,l,n]  = exp(dA_cumsum[b,h,c,l]) * dc_base[b,c,h,l,n]

Primary output:

  dC[b, c*L+l, g, n] += sum_{h in group g} dc_acc[b,c,h,l,n]

Secondary output:

  ddA_cumsum_prev[b,h,c,l] += sum_n dc_acc[b,c,h,l,n] * C[b, c*L+l, g(h), n]

Canonical tensor layouts (official Mamba Triton conventions)
------------------------------------------------------------
  states              : [B, C, H, P, N]   dtype
  dA_cumsum           : [B, H, C, L]      float32  (H before C; exp taken inside kernel)
  d_out                : [B, S, H, P]      dtype    seqlen-fused; S = C*L
  C_in                : [B, S, G, N]      dtype    seqlen-fused; S = C*L
  valid_chunk_len     : [B, C]            int32
  dC_out              : [B, S, G, N]      float32  (atomic-accumulated)  S = C*L
  ddA_cumsum_prev_out : [B, H, C, L]      float32  (atomic-accumulated)

Implementation notes
--------------------
- Parallel axes: (B * H * C, ceil(L/block_l), ceil(N/block_n))
- Reduction axis: P (tiled in block_p steps)
- Each program owns one (l_tile, n_tile) for one (b, c, h).
- dC_out uses atomic_add because multiple heads map to the same group g.
- ddA_cumsum_prev_out uses atomic_add because different n-tiles contribute
  to the same l position.
- valid_chunk_len[b,c] masks positions l >= valid_chunk_len out of computation.
- chunk_start = bc * L is added to seqlen-fused indexing.
"""

import itertools
from typing import Callable, Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel import Kernel

__all__ = ["SsdChunkScanBwdDCKernel"]


def _ssd_chunk_scan_bwd_dC_kernel(
    batch: int,
    num_chunks: int,
    chunk_len: int,
    n_heads: int,
    d_head: int,
    d_state: int,
    n_groups: int,
    dtype: str = "float16",
) -> Callable:
    accum_dtype = "float"

    B = batch
    C = num_chunks
    L = chunk_len
    S = C * L           # seqlen-fused sequence length
    H = n_heads
    P = d_head
    N = d_state
    G = n_groups
    HEADS_PER_GROUP = H // G

    @tilelang.jit()
    def kernel_func(
        block_l: int,
        block_n: int,
        block_p: int,
        threads: int,
    ):
        # canonical layouts
        states_shape   = (B, C, H, P, N)   # [B, C, H, P, N]
        dA_cum_shape   = (B, H, C, L)      # [B, H, C, L]  float32
        d_out_shape     = (B, S, H, P)      # [B, S, H, P]  seqlen-fused
        c_in_shape     = (B, S, G, N)      # [B, S, G, N]  seqlen-fused
        vcl_shape      = (B, C)            # [B, C]  int32
        dc_out_shape   = (B, S, G, N)      # [B, S, G, N]  float32 output seqlen-fused
        dda_out_shape  = (B, H, C, L)      # [B, H, C, L]  float32 output

        @T.prim_func
        def main(
            states:              T.Tensor(states_shape, dtype),       # type: ignore
            dA_cumsum:           T.Tensor(dA_cum_shape, accum_dtype), # type: ignore
            d_out:                T.Tensor(d_out_shape, dtype),         # type: ignore
            C_in:                T.Tensor(c_in_shape, dtype),         # type: ignore
            valid_chunk_len:     T.Tensor(vcl_shape, "int32"),        # type: ignore
            dC_out:              T.Tensor(dc_out_shape, accum_dtype),  # type: ignore
            ddA_cumsum_prev_out: T.Tensor(dda_out_shape, accum_dtype), # type: ignore
        ):
            with T.Kernel(
                B * H * C,
                T.ceildiv(L, block_l),
                T.ceildiv(N, block_n),
                threads=threads,
            ) as (bhc, bl, bn):

                # decode fused axis
                bz = bhc // (H * C)
                bh = (bhc % (H * C)) // C
                bc = bhc % C
                l0 = bl * block_l
                n0 = bn * block_n

                bg = bh // HEADS_PER_GROUP
                chunk_start = bc * L

                # valid chunk length for this (b, c) pair
                chunk_valid = valid_chunk_len[bz, bc]

                # --------------------------------------------------------
                # Load dA_cumsum for this l-tile and pre-compute exp(a_l)
                # Mask positions >= chunk_valid to 0 so exp() -> 1 won't
                # contribute (we later mask before accumulation anyway).
                # --------------------------------------------------------
                a_l = T.alloc_fragment((block_l,), accum_dtype)
                for ll in T.Parallel(block_l):
                    l_abs = l0 + ll
                    a_l[ll] = T.if_then_else(
                        (l_abs < L) and (l_abs < chunk_valid),
                        dA_cumsum[bz, bh, bc, l_abs],
                        T.float32(0.0),
                    )

                # --------------------------------------------------------
                # dc_acc[l, n] = sum_p d_out[l, p] * states[p, n]
                # (GEMM over p; accumulate in float32)
                # --------------------------------------------------------
                dc_acc = T.alloc_fragment((block_l, block_n), accum_dtype)
                T.clear(dc_acc)

                d_out_tile  = T.alloc_shared((block_l, block_p), dtype)
                state_tile = T.alloc_shared((block_p, block_n), dtype)

                for p_blk in T.Serial(T.ceildiv(P, block_p)):
                    p0 = p_blk * block_p

                    # Load d_out tile: [B, S, H, P]
                    for ll, pp in T.Parallel(block_l, block_p):
                        l_abs = l0 + ll
                        p_abs = p0 + pp
                        d_out_tile[ll, pp] = T.if_then_else(
                            (l_abs < L) and (p_abs < P) and (l_abs < chunk_valid),
                            d_out[bz, chunk_start + l_abs, bh, p_abs],
                            T.cast(T.float32(0.0), dtype),
                        )

                    # Load states tile: [B, C, H, P, N]
                    for pp, nn in T.Parallel(block_p, block_n):
                        p_abs = p0 + pp
                        n_abs = n0 + nn
                        state_tile[pp, nn] = T.if_then_else(
                            (p_abs < P) and (n_abs < N),
                            states[bz, bc, bh, p_abs, n_abs],
                            T.cast(T.float32(0.0), dtype),
                        )

                    # GEMM:  dc_acc[l, n] += d_out_tile[l, p] * state_tile[p, n]
                    # [block_l, block_p] @ [block_p, block_n] -> [block_l, block_n]
                    T.gemm(d_out_tile, state_tile, dc_acc)

                # --------------------------------------------------------
                # Row-wise scale by exp(dA_cumsum[l]):
                #   dc_acc[l, n] *= exp(a_l[l])
                # --------------------------------------------------------
                for ll, nn in T.Parallel(block_l, block_n):
                    l_abs = l0 + ll
                    n_abs = n0 + nn
                    if (l_abs < chunk_valid) and (n_abs < N):
                        dc_acc[ll, nn] = dc_acc[ll, nn] * T.exp(a_l[ll])

                # --------------------------------------------------------
                # Primary output: dC[b,c,g,l,n] += dc_acc[l,n]
                # Atomic because multiple heads share the same group g.
                # --------------------------------------------------------
                for ll, nn in T.Parallel(block_l, block_n):
                    l_abs = l0 + ll
                    n_abs = n0 + nn
                    if (l_abs < chunk_valid) and (n_abs < N):
                        T.atomic_add(
                            dC_out[bz, chunk_start + l_abs, bg, n_abs],
                            dc_acc[ll, nn],
                        )

                # --------------------------------------------------------
                # Secondary output: ddA_cumsum_prev[b,c,h,l]
                #   += sum_n dc_acc[l,n] * C[b,c,g,l,n]
                # --------------------------------------------------------
                C_tile = T.alloc_shared((block_l, block_n), dtype)

                for ll, nn in T.Parallel(block_l, block_n):
                    l_abs = l0 + ll
                    n_abs = n0 + nn
                    C_tile[ll, nn] = T.if_then_else(
                        (l_abs < chunk_valid) and (n_abs < N),
                        C_in[bz, chunk_start + l_abs, bg, n_abs],
                        T.cast(T.float32(0.0), dtype),
                    )

                # Elementwise product dc_acc * C_tile, masked by validity.
                dc_x_C = T.alloc_fragment((block_l, block_n), accum_dtype)
                for ll, nn in T.Parallel(block_l, block_n):
                    l_abs = l0 + ll
                    n_abs = n0 + nn
                    dc_x_C[ll, nn] = T.if_then_else(
                        (l_abs < chunk_valid) and (n_abs < N),
                        dc_acc[ll, nn] * T.cast(C_tile[ll, nn], accum_dtype),
                        T.float32(0.0),
                    )

                # Reduce over n: ddA_acc[l] = sum_n dc_x_C[l, n]
                ddA_acc = T.alloc_fragment((block_l,), accum_dtype)
                T.reduce_sum(dc_x_C, ddA_acc, dim=1)

                # Atomic because different n-tiles contribute to the same l position.
                for ll in T.Parallel(block_l):
                    l_abs = l0 + ll
                    if l_abs < chunk_valid:
                        T.atomic_add(
                            ddA_cumsum_prev_out[bz, bh, bc, l_abs],
                            ddA_acc[ll],
                        )

        return main

    return kernel_func


@torch.library.custom_op("top::ssd_chunk_scan_bwd_dC", mutates_args=())
def _ssd_chunk_scan_bwd_dC_wrapped(
    batch: int,
    num_chunks: int,
    chunk_len: int,
    n_heads: int,
    d_head: int,
    d_state: int,
    n_groups: int,
    dtype: str,
    block_l: int,
    block_n: int,
    block_p: int,
    threads: int,
    states: torch.Tensor,
    dA_cumsum: torch.Tensor,
    d_out: torch.Tensor,
    C_in: torch.Tensor,
    valid_chunk_len: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    dC_out = torch.zeros(
        batch, num_chunks * chunk_len, n_groups, d_state,
        dtype=torch.float32, device=states.device,
    )
    ddA_cumsum_prev_out = torch.zeros(
        batch, n_heads, num_chunks, chunk_len,
        dtype=torch.float32, device=states.device,
    )
    _ssd_chunk_scan_bwd_dC_kernel(
        batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype,
    )(block_l, block_n, block_p, threads)(
        states, dA_cumsum, d_out, C_in, valid_chunk_len,
        dC_out, ddA_cumsum_prev_out,
    )
    return dC_out, ddA_cumsum_prev_out


@_ssd_chunk_scan_bwd_dC_wrapped.register_fake
def _(
    batch: int,
    num_chunks: int,
    chunk_len: int,
    n_heads: int,
    d_head: int,
    d_state: int,
    n_groups: int,
    dtype: str,
    block_l: int,
    block_n: int,
    block_p: int,
    threads: int,
    states: torch.Tensor,
    dA_cumsum: torch.Tensor,
    d_out: torch.Tensor,
    C_in: torch.Tensor,
    valid_chunk_len: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    dC_out = states.new_empty(
        (batch, num_chunks * chunk_len, n_groups, d_state), dtype=torch.float32,
    )
    ddA_cumsum_prev_out = states.new_empty(
        (batch, n_heads, num_chunks, chunk_len), dtype=torch.float32,
    )
    return dC_out, ddA_cumsum_prev_out


class SsdChunkScanBwdDCKernel(Kernel):
    """Mamba-2 SSD backward kernel: gradient w.r.t. C and dA_cumsum (prev-state branch).

    Differentiates the history branch of chunk scan:

      y_prev[l, p] = exp(dA_cumsum[l]) * sum_n C[l,n] * states[p,n]

    Computes:

      dC[b,c*L+l,g,n]              += exp(dA_cumsum[b,h,c,l]) * sum_p d_out[b,c*L+l,h,p] * states[b,c,h,p,n]
      ddA_cumsum_prev[b,h,c,l]     += sum_n dC[b,c*L+l,g(h),n] * C[b,c*L+l,g(h),n]

    Both outputs are float32 and use atomic accumulation to handle the group
    broadcast (multiple heads -> same group g) and n-tile reduction.

    Inputs:
      states          [B, C, H, P, N]  dtype
      dA_cumsum       [B, H, C, L]     float32
      d_out            [B, S, H, P]     dtype     S = C*L seqlen-fused
      C_in            [B, S, G, N]     dtype     S = C*L seqlen-fused
      valid_chunk_len [B, C]           int32

    Outputs:
      dC_out              [B, S, G, N]  float32   S = C*L seqlen-fused
      ddA_cumsum_prev_out [B, H, C, L]  float32
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    def __init__(
        self,
        batch: int,
        num_chunks: int,
        chunk_len: int,
        n_heads: int,
        d_head: int,
        d_state: int,
        n_groups: int,
        dtype: torch.dtype,
        config: Optional[dict] = None,
        tune: bool = False,
    ) -> None:
        super().__init__()
        self.batch = batch
        self.num_chunks = num_chunks
        self.chunk_len = chunk_len
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_state = d_state
        self.n_groups = n_groups
        self.dtype = dtype
        self.kernel = _ssd_chunk_scan_bwd_dC_kernel(
            batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, self.dtype_str,
        )
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {
            "block_l": 64,
            "block_n": 32,
            "block_p": 64,
            "threads": 128,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        block_l  = [32, 64]
        block_n  = [16, 32]
        block_p  = [32, 64]
        threads  = [128, 256]
        return [
            {"block_l": c[0], "block_n": c[1], "block_p": c[2], "threads": c[3]}
            for c in itertools.product(block_l, block_n, block_p, threads)
        ]

    def forward(
        self,
        states: torch.Tensor,
        dA_cumsum: torch.Tensor,
        d_out: torch.Tensor,
        C_in: torch.Tensor,
        valid_chunk_len: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            states:          [B, C, H, P, N]  dtype
            dA_cumsum:       [B, H, C, L]     float32
            d_out:            [B, S, H, P]     dtype     S = C*L seqlen-fused
            C_in:            [B, S, G, N]     dtype     S = C*L seqlen-fused
            valid_chunk_len: [B, C]           int32

        Returns:
            dC_out:              [B, S, G, N]  float32   S = C*L seqlen-fused
            ddA_cumsum_prev_out: [B, H, C, L]  float32
        """
        return _ssd_chunk_scan_bwd_dC_wrapped(
            self.batch, self.num_chunks, self.chunk_len, self.n_heads, self.d_head,
            self.d_state, self.n_groups, self.dtype_str,
            self.config["block_l"], self.config["block_n"], self.config["block_p"],
            self.config["threads"],
            states.contiguous(),
            dA_cumsum.contiguous(),
            d_out.contiguous(),
            C_in.contiguous(),
            valid_chunk_len.contiguous(),
        )
