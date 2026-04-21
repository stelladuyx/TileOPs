"""
Mamba-2 SSD chunk scan backward kernel: gradient w.r.t. CB (local/diag path).

Mathematical contract
---------------------
Forward local / diag path:

  out[b, c*L+l, h, p]
    = sum_{s <= l} CB[b, c, g(h), l, s] * exp(a_l - a_s) * dt[b, h, c, s] * x[b, c*L+s, h, p]

where:
  a_t    = dA_cumsum[b, h, c, t]
  g(h)   = h // HEADS_PER_GROUP

Backward w.r.t. CB:

  dCB[b, c, g, l, s]
    = sum_{h in group g} sum_p d_out[b, c*L+l, h, p] * exp(a_l - a_s) * dt[b, h, c, s] * x[b, c*L+s, h, p]

Rearranged as two steps:

  pair_base[l, s]
    = sum_p d_out[l, p] * x[s, p]       (GEMM over p)

  dCB[l, s]
    = pair_base[l, s] * exp(a_l - a_s) * dt[s]   (element-wise post-scale)

Canonical tensor layouts (official Mamba Triton conventions)
------------------------------------------------------------
  x               : [B, S, H, P]      dtype    seqlen-fused; S = C*L
  dt              : [B, H, C, L]      dtype    (H before C)
  dA_cumsum       : [B, H, C, L]      float32  (H before C)
  d_out            : [B, S, H, P]      dtype    seqlen-fused; S = C*L
  valid_chunk_len : [B, C]            int32
  dCB_out         : [B, C, G, L, L]   float32  (chunk-local, group-owned, atomic-accumulated)

Implementation notes
--------------------
- Parallel axes: (B*C*H, ceil(L/block_l), ceil(L/block_s))
- Reduction axis: P (tiled in block_p steps)
- Each program instance owns one (b, c, h, l_tile, s_tile).
- chunk_start = bc * L is added to seqlen-fused indexing for x and d_out.

Output ownership and atomic writeback
--------------------------------------
- dCB_out is indexed by group g = h // HEADS_PER_GROUP, not by head h.
- Multiple heads within the same group write to the same (b, c, g, l, s) tile.
- The final writeback therefore uses T.atomic_add to accumulate contributions
  from all heads in the group without a separate reduction pass.
- This is a first-version design choice. A future BCG-owned kernel could
  eliminate the atomic by parallelizing over groups instead of heads, but
  that requires a different launch grid and is deferred.

GEMM layout: direct load into transposed x buffer
--------------------------------------------------
- The GEMM computes: d_out_tile [block_l, block_p] @ x_tile_t [block_p, block_s]
- x in global memory is [B, S, H, P], so a natural tile is x[chunk_start+s, p].
- The GEMM requires x in [block_p, block_s] layout (p is the contraction axis,
  s is the output axis).
- Rather than loading x_tile[ss, pp] and then transposing into x_tile_t[pp, ss],
  we load directly into the transposed layout:
      x_tile_t[pp, ss] = x[bz, chunk_start + s_abs, bh, p_abs]
  This eliminates one shared-memory buffer and one extra parallel loop.
  T.copy is not appropriate here because it replicates the source layout;
  it cannot reinterpret the index axes.

Valid-chunk masking
-------------------
- valid_chunk_len[b, c] gives the number of valid positions in chunk (b, c).
- Positions l >= valid_chunk_len or s >= valid_chunk_len are masked to zero
  in all loads (d_out, x, dA_cumsum, dt) and skipped in post-scale / writeback.
- Tile-tail masking (l_abs >= L, s_abs >= L, p_abs >= P) is handled separately
  via T.if_then_else in the load loops.
- No seq_idx support in this version.
"""

import itertools
from typing import Callable, Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel import Kernel

__all__ = ["SsdChunkScanBwdDCBKernel"]


def _ssd_chunk_scan_bwd_dCB_kernel(
    batch: int,
    num_chunks: int,
    chunk_len: int,
    n_heads: int,
    d_head: int,
    n_groups: int,
    dtype: str = "float16",
) -> Callable:
    accum_dtype = "float"

    B = batch
    C = num_chunks
    L = chunk_len
    H = n_heads
    P = d_head
    G = n_groups
    HEADS_PER_GROUP = H // G

    @tilelang.jit
    def kernel_func(
        block_l: int,
        block_s: int,
        block_p: int,
        threads: int,
    ):
        S = C * L
        x_shape       = (B, S, H, P)      # [B, S, H, P]  seqlen-fused
        dt_shape      = (B, H, C, L)      # [B, H, C, L]
        dA_cum_shape  = (B, H, C, L)      # [B, H, C, L]  float32
        d_out_shape    = (B, S, H, P)      # [B, S, H, P]  seqlen-fused
        vcl_shape     = (B, C)            # [B, C]  int32
        dcb_out_shape = (B, C, G, L, L)   # [B, C, G, L, L]  float32

        @T.prim_func
        def main(
            x:               T.Tensor(x_shape, dtype),          # type: ignore
            dt:              T.Tensor(dt_shape, dtype),          # type: ignore
            dA_cumsum:       T.Tensor(dA_cum_shape, accum_dtype),# type: ignore
            d_out:            T.Tensor(d_out_shape, dtype),        # type: ignore
            valid_chunk_len: T.Tensor(vcl_shape, "int32"),       # type: ignore
            dCB_out:         T.Tensor(dcb_out_shape, accum_dtype), # type: ignore
        ):
            # CUDA supports at most 3D grids; fuse (B, C, H) into one axis
            # and recover bz, bc, bh inside the kernel.
            with T.Kernel(
                B * C * H,
                T.ceildiv(L, block_l),
                T.ceildiv(L, block_s),
                threads=threads,
            ) as (bch, bl, bs):

                bz = bch // (C * H)
                bc = (bch % (C * H)) // H
                bh = bch % H

                l0 = bl * block_l
                s0 = bs * block_s

                chunk_valid = valid_chunk_len[bz, bc]
                bg = bh // HEADS_PER_GROUP
                chunk_start = bc * L

                # --------------------------------------------------------
                # Load dA_cumsum for l-tile and s-tile, and dt for s-tile.
                # Mask positions >= chunk_valid to 0.
                # --------------------------------------------------------
                a_l  = T.alloc_fragment((block_l,), accum_dtype)
                a_s  = T.alloc_fragment((block_s,), accum_dtype)
                dt_s = T.alloc_fragment((block_s,), accum_dtype)

                for ll in T.Parallel(block_l):
                    l_abs = l0 + ll
                    a_l[ll] = T.if_then_else(
                        (l_abs < L) and (l_abs < chunk_valid),
                        dA_cumsum[bz, bh, bc, l_abs],
                        T.float32(0.0),
                    )

                for ss in T.Parallel(block_s):
                    s_abs = s0 + ss
                    a_s[ss] = T.if_then_else(
                        (s_abs < L) and (s_abs < chunk_valid),
                        dA_cumsum[bz, bh, bc, s_abs],
                        T.float32(0.0),
                    )
                    dt_s[ss] = T.if_then_else(
                        (s_abs < L) and (s_abs < chunk_valid),
                        dt[bz, bh, bc, s_abs],
                        T.float32(0.0),
                    )

                # --------------------------------------------------------
                # pair_base[l, s] = sum_p d_out[chunk_start+l, p] * x[chunk_start+s, p]
                #
                # GEMM:
                #   d_out_tile  [block_l, block_p]
                #   x_tile_t   [block_p, block_s]   loaded directly in GEMM layout
                #   pair_acc   [block_l, block_s]
                #
                # x in global memory is [B, S, H, P] (natural tile: [chunk_start+s, p]).
                # We need [block_p, block_s] for the GEMM contraction axis.
                # Load directly into the transposed layout to avoid a staging
                # buffer and an extra shared-memory round-trip.
                # --------------------------------------------------------
                pair_acc = T.alloc_fragment((block_l, block_s), accum_dtype)
                T.clear(pair_acc)

                d_out_tile = T.alloc_shared((block_l, block_p), dtype)
                x_tile_t  = T.alloc_shared((block_p, block_s), dtype)

                for p_blk in T.Serial(T.ceildiv(P, block_p)):
                    p0 = p_blk * block_p

                    # Load d_out tile: d_out[b, chunk_start+l, h, p] -> [block_l, block_p]
                    for ll, pp in T.Parallel(block_l, block_p):
                        l_abs = l0 + ll
                        p_abs = p0 + pp
                        d_out_tile[ll, pp] = T.if_then_else(
                            (l_abs < L) and (p_abs < P) and (l_abs < chunk_valid),
                            d_out[bz, chunk_start + l_abs, bh, p_abs],
                            T.cast(0.0, dtype),
                        )

                    # Load x directly into transposed layout: x[b, chunk_start+s, h, p] -> x_tile_t[pp, ss]
                    for pp, ss in T.Parallel(block_p, block_s):
                        s_abs = s0 + ss
                        p_abs = p0 + pp
                        x_tile_t[pp, ss] = T.if_then_else(
                            (s_abs < L) and (p_abs < P) and (s_abs < chunk_valid),
                            x[bz, chunk_start + s_abs, bh, p_abs],
                            T.cast(0.0, dtype),
                        )

                    # [block_l, block_p] @ [block_p, block_s] -> [block_l, block_s]
                    T.gemm(d_out_tile, x_tile_t, pair_acc)

                # --------------------------------------------------------
                # Post-scale:
                #
                #   dCB[l, s] = pair_base[l, s] * exp(a_l - a_s) * dt[s]
                # --------------------------------------------------------
                for ll, ss in T.Parallel(block_l, block_s):
                    l_abs = l0 + ll
                    s_abs = s0 + ss
                    if (l_abs < chunk_valid) and (s_abs < chunk_valid):
                        decay = T.exp(T.min(a_l[ll] - a_s[ss], T.float32(0.0)))
                        pair_acc[ll, ss] *= decay * dt_s[ss]

                # --------------------------------------------------------
                # Write group-owned output with atomic_add.
                # Multiple heads in the same group contribute to the same
                # dCB_out[bz, bc, bg, l, s] tile.
                # --------------------------------------------------------
                for ll, ss in T.Parallel(block_l, block_s):
                    l_abs = l0 + ll
                    s_abs = s0 + ss
                    if (l_abs < chunk_valid) and (s_abs < chunk_valid):
                        T.atomic_add(
                            dCB_out[bz, bc, bg, l_abs, s_abs],
                            pair_acc[ll, ss],
                        )

        return main

    return kernel_func


@torch.library.custom_op("top::ssd_chunk_scan_bwd_dCB", mutates_args=())
def _ssd_chunk_scan_bwd_dCB_wrapped(
    batch: int,
    num_chunks: int,
    chunk_len: int,
    n_heads: int,
    d_head: int,
    n_groups: int,
    dtype: str,
    block_l: int,
    block_s: int,
    block_p: int,
    threads: int,
    x: torch.Tensor,
    dt: torch.Tensor,
    dA_cumsum: torch.Tensor,
    d_out: torch.Tensor,
    valid_chunk_len: torch.Tensor,
) -> torch.Tensor:
    dCB_out = torch.zeros(
        batch, num_chunks, n_groups, chunk_len, chunk_len,
        dtype=torch.float32, device=x.device,
    )
    _ssd_chunk_scan_bwd_dCB_kernel(
        batch, num_chunks, chunk_len, n_heads, d_head, n_groups, dtype,
    )(block_l, block_s, block_p, threads)(
        x, dt, dA_cumsum, d_out, valid_chunk_len, dCB_out,
    )
    return dCB_out


@_ssd_chunk_scan_bwd_dCB_wrapped.register_fake
def _(
    batch: int,
    num_chunks: int,
    chunk_len: int,
    n_heads: int,
    d_head: int,
    n_groups: int,
    dtype: str,
    block_l: int,
    block_s: int,
    block_p: int,
    threads: int,
    x: torch.Tensor,
    dt: torch.Tensor,
    dA_cumsum: torch.Tensor,
    d_out: torch.Tensor,
    valid_chunk_len: torch.Tensor,
) -> torch.Tensor:
    return x.new_empty(
        (batch, num_chunks, n_groups, chunk_len, chunk_len), dtype=torch.float32,
    )


class SsdChunkScanBwdDCBKernel(Kernel):
    """Mamba-2 SSD backward kernel: gradient w.r.t. CB (local/diag path).

    Differentiates the local chunk-scan branch:

      out[l, p] = sum_{s <= l} CB[l, s] * exp(a_l - a_s) * dt[s] * x[s, p]

    Computes:

      dCB[b, c, g, l, s]
        += sum_{h in group g} sum_p d_out[b, c*L+l, h, p] * exp(a_l - a_s) * dt[b, h, c, s] * x[b, c*L+s, h, p]

    Output is float32 and uses atomic accumulation to handle the group broadcast
    (multiple heads -> same group g).

    Inputs:
      x               [B, S, H, P]    dtype    S = C*L seqlen-fused
      dt              [B, H, C, L]    dtype
      dA_cumsum       [B, H, C, L]    float32
      d_out            [B, S, H, P]    dtype    S = C*L seqlen-fused
      valid_chunk_len [B, C]          int32

    Output:
      dCB_out         [B, C, G, L, L]  float32
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    def __init__(
        self,
        batch: int,
        num_chunks: int,
        chunk_len: int,
        n_heads: int,
        d_head: int,
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
        self.n_groups = n_groups
        self.dtype = dtype
        self.kernel = _ssd_chunk_scan_bwd_dCB_kernel(
            batch, num_chunks, chunk_len, n_heads, d_head, n_groups, self.dtype_str,
        )
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {
            "block_l": 64,
            "block_s": 64,
            "block_p": 64,
            "threads": 128,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        block_l = [32, 64]
        block_s = [32, 64]
        block_p = [32, 64]
        threads = [128, 256]
        return [
            {"block_l": c[0], "block_s": c[1], "block_p": c[2], "threads": c[3]}
            for c in itertools.product(block_l, block_s, block_p, threads)
        ]

    def forward(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        dA_cumsum: torch.Tensor,
        d_out: torch.Tensor,
        valid_chunk_len: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:               [B, S, H, P]    dtype    S = C*L seqlen-fused
            dt:              [B, H, C, L]    dtype
            dA_cumsum:       [B, H, C, L]    float32
            d_out:            [B, S, H, P]    dtype    S = C*L seqlen-fused
            valid_chunk_len: [B, C]          int32

        Returns:
            dCB_out: [B, C, G, L, L]  float32
        """
        return _ssd_chunk_scan_bwd_dCB_wrapped(
            self.batch, self.num_chunks, self.chunk_len, self.n_heads,
            self.d_head, self.n_groups, self.dtype_str,
            self.config["block_l"], self.config["block_s"], self.config["block_p"],
            self.config["threads"],
            x.contiguous(),
            dt.contiguous(),
            dA_cumsum.contiguous(),
            d_out.contiguous(),
            valid_chunk_len.contiguous(),
        )
