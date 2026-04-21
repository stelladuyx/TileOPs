"""
Mamba-2 SSD backward kernel: unified BMM chunk backward (K8 / K9).

Goal
----
Project a chunk-space gradient back to token-space features by a
chunk-local matrix multiplication along the token axis.

Unified mathematical template
------------------------------
For one chunk (b, c), define:

    out[m, k] = sum_r  dout[r, m] * a[r, k]  +  residual[m, k]

where:
    r : reduction token index inside the chunk
    m : output token index inside the chunk
    k : feature / state axis

equivalently in matrix form:

    out_chunk = dout_chunk^T @ a_chunk  +  residual_chunk
                [M, R]          [R, K]     [M, K]

This kernel is generic and does NOT need to know whether it is computing
dB or dC.  That is determined entirely by which tensor is passed as `a`
and whether `dout` is transposed beforehand by the caller.

Official-style instantiations
------------------------------
K8: dCB -> dB

    dB[s, n] = sum_l  dCB[l, s] * C[l, n]

    map:
        r    = l          (dim=-2 of dCB)
        m    = s          (dim=-1 of dCB)
        k    = n
        a    = C          [B, S, G, N]
        dout = dCB        [B, C, G, L, S]   r=L, m=S  (no transpose)
        out  = dB         [B, S, G, N]

K9: dCB^T -> dC

    dC[l, n] = sum_s  dCB[l, s] * B[s, n]

    map:
        r    = s          (dim=-2 after transpose)
        m    = l          (dim=-1 after transpose)
        k    = n
        a    = B          [B, S, G, N]
        dout = dCB^T      [B, C, G, S, L]   r=S, m=L  (caller transposes)
        out  = dC         [B, S, G, N]

Canonical tensor layouts
------------------------
    a               : [B, S, G, K]       seqlen-fused; S = C * chunk_size
    dout            : [B, C, G, R, M]   chunk-local pairwise; R = M = chunk_size
    valid_chunk_len : [B, C]            int32
    residual_in     : [B, S, G, K]      optional, seqlen-fused
    out             : [B, S, G, K]      seqlen-fused

Notes
-----
- S = C * chunk_size  (padded / global sequence length).
- valid_chunk_len[b, c] gives the valid number of token positions in chunk c.
- local m maps to global token index:  t_m = c * chunk_size + m
- local r maps to global token index:  t_r = c * chunk_size + r
- Any local index >= valid_chunk_len[b, c] must be masked on BOTH r and m.

Kernel ownership
----------------
One program owns one unique output tile:

    (b, c, g, m_tile, k_tile)

and completes the entire reduction over r internally by looping over r-blocks.
Therefore:
    - no cross-program reduction is needed
    - no atomic_add is needed for the final output write

Main contraction
----------------
    dout_tile_t : [block_m, block_r]   (loaded in GEMM-ready transposed layout)
    a_tile      : [block_r, block_k]
    acc         : [block_m, block_k]

    acc += dout_tile_t @ a_tile

We load dout in GEMM-ready transposed layout:

    dout_tile_t[m, r] = dout[r, m]

so no extra transpose staging buffer is needed.

Grid: (B * C * G, ceildiv(chunk_size, block_m), ceildiv(K, block_k))
"""

import itertools
from typing import Callable, Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel import Kernel

__all__ = ["SsdBmmChunkBwdKernel"]


def _ssd_bmm_chunk_bwd_kernel(
    batch: int,
    num_chunks: int,
    chunk_size: int,
    n_groups: int,
    d_state: int,
    dtype: str = "float16",
) -> Callable:
    accum_dtype = "float"

    B = batch
    C = num_chunks
    L = chunk_size   # R = M = chunk_size
    S = C * L
    G = n_groups
    K = d_state

    @tilelang.jit(out_idx=[-1])
    def kernel_func(
        block_m: int,
        block_k: int,
        block_r: int,
        threads: int,
    ):
        a_shape       = (B, S, G, K)      # [B, S, G, K]   seqlen-fused
        dout_shape    = (B, C, G, L, L)   # [B, C, G, R, M]  R = M = chunk_size
        vcl_shape     = (B, C)            # [B, C]  int32
        residual_shape = (B, S, G, K)     # [B, S, G, K]   seqlen-fused
        out_shape     = (B, S, G, K)      # [B, S, G, K]   seqlen-fused

        @T.prim_func
        def main(
            a:               T.Tensor(a_shape, dtype),          # type: ignore
            dout:            T.Tensor(dout_shape, dtype),        # type: ignore
            valid_chunk_len: T.Tensor(vcl_shape, "int32"),       # type: ignore
            residual_in:     T.Tensor(residual_shape, dtype),    # type: ignore
            out:             T.Tensor(out_shape, accum_dtype),   # type: ignore
        ):
            # Grid: fuse (B, C, G) into one axis; decompose inside the kernel.
            with T.Kernel(
                B * C * G,
                T.ceildiv(L, block_m),
                T.ceildiv(K, block_k),
                threads=threads,
            ) as (bcg, bm, bk):

                # Decode fused axis
                bz = bcg // (C * G)
                bc = (bcg % (C * G)) // G
                bg = bcg % G

                m0 = bm * block_m
                k0 = bk * block_k

                chunk_start = bc * L
                chunk_valid = valid_chunk_len[bz, bc]

                # Global token indices for the output (m) axis
                # t_m[mm] = chunk_start + m0 + mm

                # ----------------------------------------------------------------
                # Accumulator: acc[m, k]  float32
                # ----------------------------------------------------------------
                acc = T.alloc_fragment((block_m, block_k), accum_dtype)
                T.clear(acc)

                # Shared tiles reused each r-block
                # dout_tile_t[m, r] = dout[r, m]   loaded in transposed GEMM layout
                dout_tile_t = T.alloc_shared((block_m, block_r), dtype)
                a_tile      = T.alloc_shared((block_r, block_k), dtype)

                # ----------------------------------------------------------------
                # Reduction loop over the r axis (inner chunk token dimension)
                # ----------------------------------------------------------------
                for r_blk in T.Serial(T.ceildiv(L, block_r)):
                    r0 = r_blk * block_r

                    # Load dout in GEMM-ready transposed layout:
                    #   dout_tile_t[mm, rr] = dout[bz, bc, bg, r_abs, m_abs]
                    # dim=-2 of dout is R (reduction), dim=-1 is M (output).
                    # Mask both m_abs and r_abs against chunk_valid.
                    for mm, rr in T.Parallel(block_m, block_r):
                        m_abs = m0 + mm
                        r_abs = r0 + rr
                        dout_tile_t[mm, rr] = T.if_then_else(
                            (m_abs < L) and (m_abs < chunk_valid)
                            and (r_abs < L) and (r_abs < chunk_valid),
                            dout[bz, bc, bg, r_abs, m_abs],
                            T.cast(0.0, dtype),
                        )

                    # Load a_tile: a[b, t_r, g, k]  from seqlen-fused layout.
                    # t_r = chunk_start + r_abs
                    for rr, kk in T.Parallel(block_r, block_k):
                        r_abs = r0 + rr
                        k_abs = k0 + kk
                        a_tile[rr, kk] = T.if_then_else(
                            (r_abs < L) and (r_abs < chunk_valid) and (k_abs < K),
                            a[bz, chunk_start + r_abs, bg, k_abs],
                            T.cast(0.0, dtype),
                        )

                    # GEMM: [block_m, block_r] @ [block_r, block_k] -> [block_m, block_k]
                    T.gemm(dout_tile_t, a_tile, acc)

                # ----------------------------------------------------------------
                # Optional residual merge:
                #   acc[m, k] += residual[b, t_m, g, k]
                # ----------------------------------------------------------------
                residual_tile = T.alloc_shared((block_m, block_k), dtype)

                for mm, kk in T.Parallel(block_m, block_k):
                    m_abs = m0 + mm
                    k_abs = k0 + kk
                    residual_tile[mm, kk] = T.if_then_else(
                        (m_abs < L) and (m_abs < chunk_valid) and (k_abs < K),
                        residual_in[bz, chunk_start + m_abs, bg, k_abs],
                        T.cast(0.0, dtype),
                    )

                for mm, kk in T.Parallel(block_m, block_k):
                    m_abs = m0 + mm
                    k_abs = k0 + kk
                    if (m_abs < chunk_valid) and (k_abs < K):
                        acc[mm, kk] += T.cast(residual_tile[mm, kk], accum_dtype)

                # ----------------------------------------------------------------
                # Write output: out[bz, chunk_start + m_abs, bg, k_abs]
                # Output tile is uniquely owned by this program -> no atomic needed.
                # Positions outside [0, chunk_valid) or [0, K) are written as 0
                # (tilelang allocates output with torch.empty, so we must zero them).
                # ----------------------------------------------------------------
                for mm, kk in T.Parallel(block_m, block_k):
                    m_abs = m0 + mm
                    k_abs = k0 + kk
                    if (m_abs < L) and (k_abs < K):
                        out[bz, chunk_start + m_abs, bg, k_abs] = T.if_then_else(
                            (m_abs < chunk_valid),
                            acc[mm, kk],
                            T.cast(0.0, accum_dtype),
                        )

        return main

    return kernel_func


@torch.library.custom_op("top::ssd_bmm_chunk_bwd", mutates_args=())
def _ssd_bmm_chunk_bwd_wrapped(
    batch: int,
    num_chunks: int,
    chunk_size: int,
    n_groups: int,
    d_state: int,
    dtype: str,
    block_m: int,
    block_k: int,
    block_r: int,
    threads: int,
    a: torch.Tensor,
    dout: torch.Tensor,
    valid_chunk_len: torch.Tensor,
    residual_in: torch.Tensor,
) -> torch.Tensor:
    return _ssd_bmm_chunk_bwd_kernel(
        batch, num_chunks, chunk_size, n_groups, d_state, dtype,
    )(block_m, block_k, block_r, threads)(
        a, dout, valid_chunk_len, residual_in,
    )


@_ssd_bmm_chunk_bwd_wrapped.register_fake
def _(
    batch: int,
    num_chunks: int,
    chunk_size: int,
    n_groups: int,
    d_state: int,
    dtype: str,
    block_m: int,
    block_k: int,
    block_r: int,
    threads: int,
    a: torch.Tensor,
    dout: torch.Tensor,
    valid_chunk_len: torch.Tensor,
    residual_in: torch.Tensor,
) -> torch.Tensor:
    return a.new_empty(
        (batch, num_chunks * chunk_size, n_groups, d_state), dtype=torch.float32,
    )


class SsdBmmChunkBwdKernel(Kernel):
    """Mamba-2 SSD backward kernel: unified BMM chunk backward (K8 / K9).

    Computes for each chunk (b, c):

        out[m, k] = sum_r  dout[r, m] * a[r, k]  +  residual[m, k]

    equivalently:

        out_chunk = dout_chunk^T @ a_chunk  +  residual_chunk

    Used as:
        K8 (dCB -> dB):  a=C,     dout=dCB,       out=dB
        K9 (dCB -> dC):  a=B,     dout=dCB^T,     out=dC

    Inputs:
        a               [B, S, G, K]    dtype    S = C*chunk_size seqlen-fused
        dout            [B, C, G, R, M] dtype    R = M = chunk_size
        valid_chunk_len [B, C]          int32
        residual_in     [B, S, G, K]    dtype    optional, seqlen-fused

    Output:
        out             [B, S, G, K]    float32  seqlen-fused
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    def __init__(
        self,
        batch: int,
        num_chunks: int,
        chunk_size: int,
        n_groups: int,
        d_state: int,
        dtype: torch.dtype,
        config: Optional[dict] = None,
        tune: bool = False,
    ) -> None:
        super().__init__()
        self.batch = batch
        self.num_chunks = num_chunks
        self.chunk_size = chunk_size
        self.n_groups = n_groups
        self.d_state = d_state
        self.dtype = dtype
        self.kernel = _ssd_bmm_chunk_bwd_kernel(
            batch, num_chunks, chunk_size, n_groups, d_state, self.dtype_str,
        )
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {
            "block_m": 64,
            "block_k": 32,
            "block_r": 64,
            "threads": 128,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        block_m  = [32, 64]
        block_k  = [16, 32, 64]
        block_r  = [32, 64]
        threads  = [128, 256]
        return [
            {"block_m": c[0], "block_k": c[1], "block_r": c[2], "threads": c[3]}
            for c in itertools.product(block_m, block_k, block_r, threads)
        ]

    def forward(
        self,
        a: torch.Tensor,
        dout: torch.Tensor,
        valid_chunk_len: torch.Tensor,
        residual_in: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            a:               [B, S, G, K]    dtype    S = C*chunk_size seqlen-fused
            dout:            [B, C, G, R, M] dtype    R = M = chunk_size
            valid_chunk_len: [B, C]          int32
            residual_in:     [B, S, G, K]    dtype    optional; zeros if None

        Returns:
            out: [B, S, G, K]  float32  seqlen-fused
        """
        if residual_in is None:
            residual_in = torch.zeros_like(a)
        return _ssd_bmm_chunk_bwd_wrapped(
            self.batch, self.num_chunks, self.chunk_size, self.n_groups,
            self.d_state, self.dtype_str,
            self.config["block_m"], self.config["block_k"], self.config["block_r"],
            self.config["threads"],
            a.contiguous(),
            dout.contiguous(),
            valid_chunk_len.contiguous(),
            residual_in.contiguous(),
        )
