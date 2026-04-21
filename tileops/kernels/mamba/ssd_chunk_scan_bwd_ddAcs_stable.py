"""
Mamba-2 SSD chunk scan backward kernel: local/diag-path contribution to ddA_cumsum.

K10: ssd_chunk_scan_bwd_ddAcs_stable

Mathematical contract
---------------------
Forward local / diag path:

  out_local[b, c*L+l, h, p]
    = sum_{s <= l} CB[b, c, g(h), l, s] * exp(a_l - a_s) * dt[b, h, c, s] * x[b, c*L+s, h, p]

where:
  a_t    = dA_cumsum[b, h, c, t]
  g(h)   = h // HEADS_PER_GROUP

Backward w.r.t. dA_cumsum (local/diag path only)
-------------------------------------------------
Define the pairwise weight:

  G[l, s]
    = (sum_p d_out[b, c*L+l, h, p] * x[b, c*L+s, h, p])
      * CB[b, c, g(h), l, s]
      * dt[b, h, c, s]
      * exp(a_l - a_s)

Valid only for strict lower-triangular pairs: l >= s + 1
(the diagonal l == s contributes 0 to ddA because exp(a_l - a_l) = 1 has
zero derivative w.r.t. both a_l and a_s at the diagonal).

The full contribution of pair (l, s) to ddA_cumsum is:
  - adds    G[l, s]  to position l  (differentiating exp(+a_l))
  - subtracts G[l, s]  to position s  (differentiating exp(-a_s))

Equivalently, G[l, s] contributes +G[l,s] to every t in (s, l]:
  ddA_cumsum[b, h, c, t] += G[l, s]   for all t with s < t <= l

Output:
  ddA_cumsum_local_out[b, h, c, t]
    = sum_{l >= t, s < t} G[l, s]    (strict-lower-tri pairs spanning t)

Canonical tensor layouts (official Mamba Triton conventions)
------------------------------------------------------------
  x               : [B, S, H, P]    dtype    seqlen-fused; S = C*L
  dt              : [B, H, C, L]    dtype    (H before C)
  dA_cumsum       : [B, H, C, L]    float32  (H before C)
  CB              : [B, C, G, L, L] dtype    group-owned pairwise coeff
  d_out            : [B, S, H, P]    dtype    seqlen-fused; S = C*L
  valid_chunk_len : [B, C]          int32
  ddA_local_out   : [B, H, C, L]    float32  (local/diag-path partial, atomic-accumulated)

Implementation notes
--------------------
- Parallel axes: (B*C*H, ceil(L/block_l))
- Inner serial loop: s-tiles (only up to hi = min(chunk_valid, l0+block_l)),
  p-tiles (ceil(P/block_p))
- Each program instance owns one (b, c, h, l_tile).
- For each l-tile, sweep s-tiles up to hi (s >= hi cannot contribute any
  valid pair (l, s) with l > s, so the sweep is bounded):
    1. GEMM: pair_base[l, s] = sum_p d_out[l, p] * x[s, p]
       using d_out_tile [block_l, block_p] @ x_tile_t [block_p, block_s]
    2. Post-scale + pre-cumsum mask:
         pair_weight[l, s] = pair_base * CB[l,s] * dt[s] * exp(min(a_l-a_s, 0))
       zeroed for l < s+1 (strict lower-triangular), out-of-bounds, or s >= hi.
    3. Interval-to-position scatter via prefix-sum carry:
       - rowsum_carry[l] holds sum_{ s' < s0 } G[l, s'] from earlier s-tiles.
       - cumsum pair_weight[l, *] along s (in-place), adding carry:
           pair_weight[l, ss]  <-- rowsum_carry[l] + sum_{s'=0}^{ss} G[l, s']
         Meaning after cumsum: pair_weight[l, ss] is the total G[l,*] weight
         that maps to output position t = s0+ss+1 (the contribution to all
         t in (s, l] where s = s0+ss).
       - Post-cumsum mask: zero pair_weight[l, ss] if t_abs = s0+ss+1 is
         outside (s_abs, l_abs] or outside chunk_valid. This is necessary
         because the prefix sum propagates earlier column values into later
         columns, so invalid positions beyond row l's endpoint must be
         re-zeroed after the cumsum.
       - col_acc[ss] = sum_l pair_weight[l, ss]
       - atomic_add col_acc[ss] to ddA_local_out at position t = s0+ss+1
         (not s0+ss: column ss represents the cumulative weight for output
         position t = s0+ss+1, since pair (l,s) contributes to (s, l], i.e.
         the first output position a given pair column contributes to is s+1).

- ddA_local_out (the local/diag-path partial ddA) is zero-initialized by the
  caller; this kernel atomic-adds into it. Multiple heads in the same (b, c)
  accumulate into overlapping positions via atomics.

- Valid-chunk masking:
    - valid_chunk_len[b, c] is the number of valid positions in chunk (b, c).
    - hi = min(chunk_valid, l0+block_l) bounds the s-tile sweep per l-tile.
    - All s-dependent loads use s_abs < hi as the validity guard.

Grid: (B*C*H, ceildiv(L, block_l))
"""

import itertools
from typing import Callable, Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel import Kernel

__all__ = ["SsdChunkScanBwdDdAcsStableKernel"]


def _ssd_chunk_scan_bwd_ddAcs_stable_kernel(
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
    S = C * L           # seqlen-fused sequence length
    H = n_heads
    P = d_head
    G = n_groups
    HEADS_PER_GROUP = H // G

    @tilelang.jit(out_idx=[-1])
    def kernel_func(
        block_l: int,
        block_s: int,
        block_p: int,
        threads: int,
    ):
        x_shape      = (B, S, H, P)      # [B, S, H, P]  seqlen-fused
        dt_shape     = (B, H, C, L)      # [B, H, C, L]
        dA_cum_shape = (B, H, C, L)      # [B, H, C, L]  float32
        cb_shape     = (B, C, G, L, L)   # [B, C, G, L, L]
        d_out_shape   = (B, S, H, P)      # [B, S, H, P]  seqlen-fused
        vcl_shape    = (B, C)            # [B, C]  int32
        dda_shape    = (B, H, C, L)      # [B, H, C, L]  float32 local/diag-path output

        @T.prim_func
        def main(
            x:               T.Tensor(x_shape, dtype),            # type: ignore
            dt:              T.Tensor(dt_shape, dtype),            # type: ignore
            dA_cumsum:       T.Tensor(dA_cum_shape, accum_dtype),  # type: ignore
            CB:              T.Tensor(cb_shape, dtype),            # type: ignore
            d_out:            T.Tensor(d_out_shape, dtype),          # type: ignore
            valid_chunk_len: T.Tensor(vcl_shape, "int32"),         # type: ignore
            ddA_local_out:   T.Tensor(dda_shape, accum_dtype),     # type: ignore
        ):
            # Grid: (B*C*H, ceildiv(L, block_l))
            with T.Kernel(
                B * C * H,
                T.ceildiv(L, block_l),
                threads=threads,
            ) as (bch, bl):

                bz = bch // (C * H)
                bc = (bch % (C * H)) // H
                bh = bch % H
                bg = bh // HEADS_PER_GROUP

                l0 = bl * block_l
                chunk_valid = valid_chunk_len[bz, bc]
                chunk_start = bc * L

                # --------------------------------------------------------
                # Bound the s-tile sweep for this l-tile.
                #
                # hi = min(chunk_valid, l0 + block_l)
                #
                # For the current l-tile, any s >= hi satisfies s >= l for
                # every l in the tile (since l <= l0+block_l-1 < hi <= s),
                # so no pair (l, s) with l > s can exist. Sweeping beyond
                # hi does unnecessary work and complicates the carry logic.
                # --------------------------------------------------------
                hi = T.min(chunk_valid, l0 + block_l)

                # --------------------------------------------------------
                # Load dA_cumsum for this l-tile  (a_l values).
                # Positions >= chunk_valid are masked to 0.
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
                # rowsum_carry[ll]:
                #   Stores  sum_{ s' < s0 } G[l0+ll, s']  for the current
                #   row across all s-tiles processed so far.  This is the
                #   carry term added at the start of each new s-tile's
                #   prefix sum so that the cumsum is global over all s < s0,
                #   not just local within the current tile.
                # --------------------------------------------------------
                rowsum_carry = T.alloc_fragment((block_l,), accum_dtype)
                T.clear(rowsum_carry)

                # --------------------------------------------------------
                # Sweep s-tiles from 0 up to ceildiv(hi, block_s).
                # s-tiles with s0 >= hi are skipped entirely.
                # --------------------------------------------------------
                for s_blk in T.Serial(T.ceildiv(hi, block_s)):
                    s0 = s_blk * block_s

                    # Load a_s and dt_s.
                    # Guard: s_abs < hi  (combines the L and chunk_valid bounds).
                    a_s  = T.alloc_fragment((block_s,), accum_dtype)
                    dt_s = T.alloc_fragment((block_s,), accum_dtype)

                    for ss in T.Parallel(block_s):
                        s_abs = s0 + ss
                        a_s[ss] = T.if_then_else(
                            s_abs < hi,
                            dA_cumsum[bz, bh, bc, s_abs],
                            T.float32(0.0),
                        )
                        dt_s[ss] = T.if_then_else(
                            s_abs < hi,
                            T.cast(dt[bz, bh, bc, s_abs], accum_dtype),
                            T.float32(0.0),
                        )

                    # --------------------------------------------------
                    # Step 1: pair_base[l, s] = sum_p d_out[l, p] * x[s, p]
                    #
                    # GEMM:
                    #   d_out_tile  [block_l, block_p]  (rows = l, cols = p)
                    #   x_tile_t   [block_p, block_s]  (loaded transposed;
                    #                                   guard: s_abs < hi)
                    #   pair_base  [block_l, block_s]  float32 accumulator
                    # --------------------------------------------------
                    pair_base = T.alloc_fragment((block_l, block_s), accum_dtype)
                    T.clear(pair_base)

                    d_out_tile = T.alloc_shared((block_l, block_p), dtype)
                    x_tile_t  = T.alloc_shared((block_p, block_s), dtype)

                    for p_blk in T.Serial(T.ceildiv(P, block_p)):
                        p0 = p_blk * block_p

                        # d_out[b, chunk_start+l, h, p] -> d_out_tile[ll, pp]
                        for ll, pp in T.Parallel(block_l, block_p):
                            l_abs = l0 + ll
                            p_abs = p0 + pp
                            d_out_tile[ll, pp] = T.if_then_else(
                                (l_abs < L) and (p_abs < P) and (l_abs < chunk_valid),
                                d_out[bz, chunk_start + l_abs, bh, p_abs],
                                T.cast(0.0, dtype),
                            )

                        # x[b, chunk_start+s, h, p] -> x_tile_t[pp, ss]  (transposed)
                        # Guard: s_abs < hi ensures we don't load padding.
                        for pp, ss in T.Parallel(block_p, block_s):
                            s_abs = s0 + ss
                            p_abs = p0 + pp
                            x_tile_t[pp, ss] = T.if_then_else(
                                (s_abs < hi) and (p_abs < P),
                                x[bz, chunk_start + s_abs, bh, p_abs],
                                T.cast(0.0, dtype),
                            )

                        # [block_l, block_p] @ [block_p, block_s] -> [block_l, block_s]
                        T.gemm(d_out_tile, x_tile_t, pair_base)

                    # --------------------------------------------------
                    # Step 2 + 3: post-scale and strict lower-tri mask.
                    #
                    # Before cumsum, pair_weight[l, s] = G[l, s]:
                    #
                    #   G[l, s] = pair_base[l, s]
                    #             * CB[b, c, g, l, s]
                    #             * dt[s]
                    #             * exp(min(a_l - a_s, 0))   (stable clamp)
                    #
                    # Zeroed for:
                    #   - l < s+1  (diagonal and upper triangle: no interval (s,l])
                    #   - s_abs >= hi  (out-of-bounds or past valid sweep range)
                    #   - l_abs >= chunk_valid  (out-of-bounds row)
                    # --------------------------------------------------
                    pair_weight = T.alloc_fragment((block_l, block_s), accum_dtype)

                    for ll, ss in T.Parallel(block_l, block_s):
                        l_abs = l0 + ll
                        s_abs = s0 + ss
                        keep = (
                            (l_abs < chunk_valid)
                            and (s_abs < hi)
                            and (l_abs >= s_abs + 1)
                        )
                        cb_ls = T.if_then_else(
                            keep,
                            T.cast(CB[bz, bc, bg, l_abs, s_abs], accum_dtype),
                            T.float32(0.0),
                        )
                        decay_ls = T.exp(T.min(a_l[ll] - a_s[ss], T.float32(0.0)))
                        pair_weight[ll, ss] = T.if_then_else(
                            keep,
                            pair_base[ll, ss] * cb_ls * dt_s[ss] * decay_ls,
                            T.float32(0.0),
                        )

                    # --------------------------------------------------
                    # Step 4a: row-wise prefix sum (cumsum along s) + carry.
                    #
                    # rowsum_carry[ll] holds  sum_{ s' < s0 } G[l0+ll, s'].
                    #
                    # After this loop, pair_weight[ll, ss] is REDEFINED:
                    #
                    #   pair_weight[ll, ss]  (after cumsum)
                    #     = rowsum_carry[ll] + sum_{s'=0}^{ss} G[l0+ll, s0+s']
                    #
                    # This value represents the total G weight from row l that
                    # maps to output position  t = s0 + ss + 1  (i.e. all pairs
                    # (l, s') with s' <= ss contribute to t > s', and the
                    # smallest such t for column ss is s0+ss+1).
                    #
                    # tile_rowsum[ll] = final running value at end of this
                    # s-tile; stored back into rowsum_carry for the next tile.
                    # --------------------------------------------------
                    tile_rowsum = T.alloc_fragment((block_l,), accum_dtype)
                    T.clear(tile_rowsum)

                    for ll in T.Parallel(block_l):
                        running = rowsum_carry[ll]
                        for ss in T.Serial(block_s):
                            running = running + pair_weight[ll, ss]
                            pair_weight[ll, ss] = running
                        tile_rowsum[ll] = running

                    # Update carry: rowsum_carry[ll] = sum_{ s' < s0+block_s } G[l, s']
                    for ll in T.Parallel(block_l):
                        rowsum_carry[ll] = tile_rowsum[ll]

                    # --------------------------------------------------
                    # Step 4b: post-cumsum mask.
                    #
                    # The prefix sum propagates earlier column values into
                    # later columns for each row l.  This can deposit weight
                    # at column ss even if t_abs = s0+ss+1 > l_abs (i.e.
                    # beyond the row's own endpoint) or outside chunk_valid.
                    #
                    # Re-zero any pair_weight[ll, ss] where the implied
                    # output position  t_abs = s0 + ss + 1  does not satisfy
                    #   s_abs < t_abs <= l_abs   and   t_abs < chunk_valid
                    # --------------------------------------------------
                    for ll, ss in T.Parallel(block_l, block_s):
                        l_abs  = l0 + ll
                        t_abs  = s0 + ss + 1   # output position this column maps to
                        valid_t = (t_abs <= l_abs) and (t_abs < chunk_valid)
                        pair_weight[ll, ss] = T.if_then_else(
                            valid_t,
                            pair_weight[ll, ss],
                            T.float32(0.0),
                        )

                    # --------------------------------------------------
                    # Step 5: column-reduce over l.
                    #
                    #   col_acc[ss] = sum_l pair_weight[l, ss]
                    #               = total ddA_local contribution to
                    #                 output position t = s0+ss+1.
                    # --------------------------------------------------
                    col_acc = T.alloc_fragment((block_s,), accum_dtype)
                    T.clear(col_acc)

                    for ss in T.Parallel(block_s):
                        tmp = T.float32(0.0)
                        for ll in T.Serial(block_l):
                            tmp = tmp + pair_weight[ll, ss]
                        col_acc[ss] = tmp

                    # --------------------------------------------------
                    # Step 6: atomic scatter to ddA_local_out.
                    #
                    # Column ss maps to output position t = s0 + ss + 1
                    # (NOT s0+ss), because pair (l, s) contributes to the
                    # open-on-left interval (s, l], so the first affected
                    # output position for s-column ss is s+1 = s0+ss+1.
                    #
                    # col_acc[ss] is already the correctly accumulated value;
                    # we do NOT shift the column index itself (col_acc[ss+1]
                    # would be wrong).  Only the output address shifts by +1.
                    # --------------------------------------------------
                    for ss in T.Parallel(block_s):
                        t_abs = s0 + ss + 1   # output position: +1 from column index
                        if (t_abs < L) and (t_abs < chunk_valid):
                            T.atomic_add(
                                ddA_local_out[bz, bh, bc, t_abs],
                                col_acc[ss],
                            )

        return main

    return kernel_func


@torch.library.custom_op("top::ssd_chunk_scan_bwd_ddAcs_stable", mutates_args=())
def _ssd_chunk_scan_bwd_ddAcs_stable_wrapped(
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
    CB: torch.Tensor,
    d_out: torch.Tensor,
    valid_chunk_len: torch.Tensor,
) -> torch.Tensor:
    # ddA_local_out: [B, H, C, L]  float32  local/diag-path partial ddA
    ddA_local_out = torch.zeros(
        batch, n_heads, num_chunks, chunk_len,
        dtype=torch.float32, device=x.device,
    )
    _ssd_chunk_scan_bwd_ddAcs_stable_kernel(
        batch, num_chunks, chunk_len, n_heads, d_head, n_groups, dtype,
    )(block_l, block_s, block_p, threads)(
        x, dt, dA_cumsum, CB, d_out, valid_chunk_len, ddA_local_out,
    )
    return ddA_local_out


@_ssd_chunk_scan_bwd_ddAcs_stable_wrapped.register_fake
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
    CB: torch.Tensor,
    d_out: torch.Tensor,
    valid_chunk_len: torch.Tensor,
) -> torch.Tensor:
    # output shape: [B, H, C, L]  float32  local/diag-path partial ddA
    return x.new_empty(
        (batch, n_heads, num_chunks, chunk_len), dtype=torch.float32,
    )


class SsdChunkScanBwdDdAcsStableKernel(Kernel):
    """Mamba-2 SSD backward kernel: local/diag-path contribution to ddA_cumsum (K10).

    Differentiates the local chunk-scan branch w.r.t. dA_cumsum:

      out_local[l, p] = sum_{s<=l} CB[l,s] * exp(a_l - a_s) * dt[s] * x[s, p]

    The pairwise gradient weight is:

      G[l, s] = (sum_p d_out[l,p] * x[s,p]) * CB[l,s] * dt[s] * exp(a_l - a_s)

    and pair (l, s) with l >= s+1 contributes G[l,s] to every t in (s, l]:

      ddA_cumsum_local[b, h, c, t] += sum_{l>=t, s<t} G[l, s]

    Algorithm: stable blockwise interval-folding
    --------------------------------------------
    For each l-tile, s is swept up to hi = min(chunk_valid, l0+block_l).
    Inside each s-tile:
      1. GEMM:  pair_base[l,s] = d_out[l,:] @ x[s,:]^T
      2. Scale + pre-cumsum mask:
           pair_weight[l,s] = G[l,s],  zeroed for l <= s or out-of-bounds
      3. Row-wise prefix sum (cumsum along s) + carry from prior s-tiles:
           pair_weight[l, ss]  (after) = sum_{s'<=ss} G[l,s'] + carry
           -> column ss now encodes the contribution to output position
              t = s0+ss+1  (not s0+ss, because pair (l,s) maps to (s, l])
      4. Post-cumsum mask: zero pair_weight[l, ss] where t_abs > l_abs
         or t_abs >= chunk_valid (prefix sum propagates into invalid cols)
      5. Column-reduce: col_acc[ss] = sum_l pair_weight[l, ss]
      6. Atomic scatter: ddA_local_out[t_abs] += col_acc[ss]
         where t_abs = s0 + ss + 1  (+1 offset, not +0)

    Inputs:
      x               [B, S, H, P]    dtype    S = C*L seqlen-fused
      dt              [B, H, C, L]    dtype
      dA_cumsum       [B, H, C, L]    float32
      CB              [B, C, G, L, L] dtype    group-owned pairwise coeff
      d_out            [B, S, H, P]    dtype    S = C*L seqlen-fused
      valid_chunk_len [B, C]          int32

    Output:
      ddA_local_out   [B, H, C, L]    float32  (local/diag-path partial;
                                                zero-initialized, atomically accumulated)
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
        self.kernel = _ssd_chunk_scan_bwd_ddAcs_stable_kernel(
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
        CB: torch.Tensor,
        d_out: torch.Tensor,
        valid_chunk_len: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:               [B, S, H, P]    dtype    S = C*L seqlen-fused
            dt:              [B, H, C, L]    dtype
            dA_cumsum:       [B, H, C, L]    float32
            CB:              [B, C, G, L, L] dtype    group-owned pairwise coeff
            d_out:            [B, S, H, P]    dtype    S = C*L seqlen-fused
            valid_chunk_len: [B, C]          int32,
                             defaults to all chunk_len (complete chunks)

        Returns:
            ddA_local_out: [B, H, C, L]  float32  local/diag-path partial ddA
        """
        if valid_chunk_len is None:
            valid_chunk_len = torch.full(
                (self.batch, self.num_chunks), self.chunk_len,
                dtype=torch.int32, device=x.device,
            )
        return _ssd_chunk_scan_bwd_ddAcs_stable_wrapped(
            self.batch, self.num_chunks, self.chunk_len, self.n_heads,
            self.d_head, self.n_groups, self.dtype_str,
            self.config["block_l"], self.config["block_s"], self.config["block_p"],
            self.config["threads"],
            x.contiguous(),
            dt.contiguous(),
            dA_cumsum.contiguous(),
            CB.contiguous(),
            d_out.contiguous(),
            valid_chunk_len.contiguous(),
        )
