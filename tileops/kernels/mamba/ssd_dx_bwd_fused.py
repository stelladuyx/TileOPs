"""
Mamba-2 SSD chunk-scan backward kernel: gradient w.r.t. x and dt (fused dx pass).

Corresponds to Triton _chunk_scan_chunk_state_bwd_dx (K7) — fuses the two
gradient contributions to ``dx`` and the partial ``ddt`` in a single kernel.

Mathematical contract
---------------------
Forward pass creates two contributions to the output y at position (b, c*L+m, h, p):

  (A) State path:
      y_state[m, p] = exp(min(a_end - a_m, 0)) * sum_n B[m, n] * states[p, n]

  (B) Local / diag chunk-scan path:
      y_local[m, p] = sum_{l >= m}^{L_valid-1}
                          CB[l, m] * exp(min(a_l - a_m, 0)) * dt[m] * x[m, p]

Accumulator (both paths share a single [block_m, block_p] fragment):

  acc[m, p]  = state_contrib[m, p] + local_contrib[m, p]

where:

  state_contrib[m, p]
    = exp(min(a_end - a_m, 0)) * sum_n B[m, n] * dstates[p, n]

  local_contrib[m, p]
    = sum_{l >= m} CB[l, m] * exp(min(a_l - a_m, 0)) * d_out[l, p]

Final outputs:

  dx[b, c*L+m, h, p]        = acc[m, p] * dt[m]
  ddt_partial[b, h, c, m]  += sum_p acc[m, p] * x[m, p]   (atomic over p-tiles)

Canonical tensor layouts (official Mamba Triton conventions):
  x          : [B, S, H, P]      dtype    seqlen-fused;  S = C*L
  dt         : [B, H, C, L]      dtype    (H before C)
  dA_cumsum  : [B, H, C, L]      float32  (H before C)
  B_in       : [B, S, G, N]      dtype    seqlen-fused, group-owned
  CB         : [B, C, G, L, L]   dtype    chunk-local, lower-triangular
  d_out       : [B, S, H, P]      dtype    seqlen-fused;  S = C*L
  dstates    : [B, C, H, P, N]   float32  upstream grad from state-passing bwd

Outputs:
  dx_out         : [B, S, H, P]  dtype  (written directly — no atomic)
  ddt_partial_out: [B, H, C, L]  float32 (atomic over p-tiles)

Implementation notes
--------------------
- Parallel axes: (B*C*H, ceil(L/block_m), ceil(P/block_p))
- PART A reduces over N (d_state) tiles → GEMM: B_tile [block_m, block_n] @ dstates_t [block_n, block_p]
- PART B reduces over l tiles (l >= m only) → GEMM: cb_t_tile [block_m, block_lr] @ d_out_tile [block_lr, block_p]
  where cb_t_tile[mm, ll] = CB[l, m] * exp(min(a_l - a_m, 0)) (pre-fused decay)
- dx_out is written directly (one p-tile per program → no atomic).
- ddt_partial_out uses atomic_add because the same (b, h, c, m) is written
  by every p-tile, so contributions must be summed across p-tile programs.
- n_groups / HEADS_PER_GROUP: B_in and CB use group axis g = h // HEADS_PER_GROUP.
- valid_chunk_len is not taken as an explicit argument here; the chunk length is
  implicitly bounded by min(L, S - chunk_start) so partial last chunks are handled.
"""

import itertools
from typing import Callable, Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel import Kernel

__all__ = ["SsdDxBwdFusedKernel"]


def _ssd_dx_bwd_fused_kernel(
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
        block_m: int,
        block_p: int,
        block_n: int,
        block_lr: int,
        threads: int,
    ):
        x_shape         = (B, S, H, P)       # [B, S, H, P]  seqlen-fused
        dt_shape        = (B, H, C, L)       # [B, H, C, L]
        dA_cum_shape    = (B, H, C, L)       # [B, H, C, L]  float32
        b_in_shape      = (B, S, G, N)       # [B, S, G, N]  seqlen-fused
        cb_shape        = (B, C, G, L, L)    # [B, C, G, L, L]
        d_out_shape      = (B, S, H, P)       # [B, S, H, P]  seqlen-fused
        dstates_shape   = (B, C, H, P, N)    # [B, C, H, P, N]  float32
        dx_shape        = (B, S, H, P)       # [B, S, H, P]  output
        ddt_shape       = (B, H, C, L)       # [B, H, C, L]  float32

        @T.prim_func
        def main(
            x:               T.Tensor(x_shape, dtype),          # type: ignore
            dt:              T.Tensor(dt_shape, dtype),          # type: ignore
            dA_cumsum:       T.Tensor(dA_cum_shape, accum_dtype),# type: ignore
            B_in:            T.Tensor(b_in_shape, dtype),        # type: ignore
            CB:              T.Tensor(cb_shape, dtype),          # type: ignore
            d_out:            T.Tensor(d_out_shape, dtype),        # type: ignore
            dstates:         T.Tensor(dstates_shape, accum_dtype),# type: ignore
            dx_out:          T.Tensor(dx_shape, dtype),          # type: ignore
            ddt_partial_out: T.Tensor(ddt_shape, accum_dtype),   # type: ignore
        ):
            with T.Kernel(
                B * C * H,
                T.ceildiv(L, block_m),
                T.ceildiv(P, block_p),
                threads=threads,
            ) as (bch, bm, bp):

                # ── decode fused axis ────────────────────────────────────────
                bz = bch // (C * H)
                bc = (bch % (C * H)) // H
                bh = bch % H
                bg = bh // HEADS_PER_GROUP

                m0 = bm * block_m
                p0 = bp * block_p
                chunk_start = bc * L

                # valid positions in this chunk (handles incomplete last chunk)
                valid_len = T.max(
                    T.int32(0),
                    T.min(T.int32(L), T.int32(S) - T.int32(chunk_start)),
                )

                # ── load per-row scalars: dA_cumsum[m] and dt[m] ─────────────
                a_m  = T.alloc_fragment((block_m,), accum_dtype)
                dt_m = T.alloc_fragment((block_m,), accum_dtype)

                for mm in T.Parallel(block_m):
                    m_abs = m0 + mm
                    a_m[mm] = T.if_then_else(
                        m_abs < valid_len,
                        dA_cumsum[bz, bh, bc, m_abs],
                        T.float32(0.0),
                    )
                    dt_m[mm] = T.if_then_else(
                        m_abs < valid_len,
                        T.cast(dt[bz, bh, bc, m_abs], accum_dtype),
                        T.float32(0.0),
                    )

                # ── load x tile: [block_m, block_p] ─────────────────────────
                x_tile = T.alloc_shared((block_m, block_p), dtype)
                for mm, pp in T.Parallel(block_m, block_p):
                    m_abs = m0 + mm
                    p_abs = p0 + pp
                    x_tile[mm, pp] = T.if_then_else(
                        (m_abs < valid_len) and (p_abs < P),
                        x[bz, chunk_start + m_abs, bh, p_abs],
                        T.cast(0.0, dtype),
                    )

                # ── shared accumulator ───────────────────────────────────────
                acc = T.alloc_fragment((block_m, block_p), accum_dtype)
                T.clear(acc)

                # ============================================================
                # PART A: state path
                #
                # state_contrib[m, p] =
                #   exp(min(a_end - a_m, 0)) * sum_n B[m, n] * dstates[p, n]
                #
                # GEMM: B_tile [block_m, block_n] @ dstates_t [block_n, block_p]
                # ============================================================
                a_end = T.if_then_else(
                    valid_len > T.int32(0),
                    dA_cumsum[bz, bh, bc, valid_len - 1],
                    T.float32(0.0),
                )

                acc_state   = T.alloc_fragment((block_m, block_p), accum_dtype)
                T.clear(acc_state)

                B_tile      = T.alloc_shared((block_m, block_n), dtype)
                dstates_t   = T.alloc_shared((block_n, block_p), dtype)

                for n_blk in T.Serial(T.ceildiv(N, block_n)):
                    n0 = n_blk * block_n

                    # load B_in[b, chunk_start+m, g, n]  → [block_m, block_n]
                    for mm, nn in T.Parallel(block_m, block_n):
                        m_abs = m0 + mm
                        n_abs = n0 + nn
                        B_tile[mm, nn] = T.if_then_else(
                            (m_abs < valid_len) and (n_abs < N),
                            B_in[bz, chunk_start + m_abs, bg, n_abs],
                            T.cast(0.0, dtype),
                        )

                    # load dstates[b, c, h, p, n] transposed → [block_n, block_p]
                    for nn, pp in T.Parallel(block_n, block_p):
                        n_abs = n0 + nn
                        p_abs = p0 + pp
                        dstates_t[nn, pp] = T.if_then_else(
                            (n_abs < N) and (p_abs < P),
                            T.cast(dstates[bz, bc, bh, p_abs, n_abs], dtype),
                            T.cast(0.0, dtype),
                        )

                    # acc_state[m, p] += B_tile[m, n] @ dstates_t[n, p]
                    T.gemm(B_tile, dstates_t, acc_state)

                # apply state-path decay: exp(min(a_end - a_m, 0))
                for mm, pp in T.Parallel(block_m, block_p):
                    m_abs = m0 + mm
                    p_abs = p0 + pp
                    if (m_abs < valid_len) and (p_abs < P):
                        decay_state = T.exp(T.min(a_end - a_m[mm], T.float32(0.0)))
                        acc[mm, pp] = acc[mm, pp] + acc_state[mm, pp] * decay_state

                # ============================================================
                # PART B: local / diag chunk-scan path
                #
                # local_contrib[m, p] =
                #   sum_{l >= m} CB[l, m] * exp(min(a_l - a_m, 0)) * d_out[l, p]
                #
                # Strategy: for each l-block (l >= m0) build
                #   cb_t_tile[mm, ll] = CB[l, m] * exp(min(a_l - a_m, 0))
                # then GEMM: cb_t_tile [block_m, block_lr] @ d_out_tile [block_lr, block_p]
                # ============================================================
                cb_t_tile  = T.alloc_shared((block_m, block_lr), dtype)
                d_out_tile  = T.alloc_shared((block_lr, block_p), dtype)
                a_l        = T.alloc_fragment((block_lr,), accum_dtype)

                # only iterate l-tiles from m0 onward (earlier l < m are invalid)
                for l_blk in T.Serial(T.ceildiv(valid_len - m0, block_lr)):
                    l0 = m0 + l_blk * block_lr

                    # load dA_cumsum for l-tile
                    for ll in T.Parallel(block_lr):
                        l_abs = l0 + ll
                        a_l[ll] = T.if_then_else(
                            l_abs < valid_len,
                            dA_cumsum[bz, bh, bc, l_abs],
                            T.float32(0.0),
                        )

                    # build pre-scaled cb_t_tile[mm, ll] = CB[l, m] * decay
                    # (causal: only l >= m contributes, upper triangle is zeroed)
                    for mm, ll in T.Parallel(block_m, block_lr):
                        m_abs = m0 + mm
                        l_abs = l0 + ll

                        keep = (m_abs < valid_len) and (l_abs < valid_len) and (l_abs >= m_abs)

                        coeff = T.if_then_else(
                            keep,
                            T.cast(CB[bz, bc, bg, l_abs, m_abs], accum_dtype),
                            T.float32(0.0),
                        )
                        decay_local = T.if_then_else(
                            keep,
                            T.exp(T.min(a_l[ll] - a_m[mm], T.float32(0.0))),
                            T.float32(0.0),
                        )
                        cb_t_tile[mm, ll] = T.cast(coeff * decay_local, dtype)

                    # load d_out[b, chunk_start+l, h, p]  → [block_lr, block_p]
                    for ll, pp in T.Parallel(block_lr, block_p):
                        l_abs = l0 + ll
                        p_abs = p0 + pp
                        d_out_tile[ll, pp] = T.if_then_else(
                            (l_abs < valid_len) and (p_abs < P),
                            d_out[bz, chunk_start + l_abs, bh, p_abs],
                            T.cast(0.0, dtype),
                        )

                    # acc[m, p] += cb_t_tile[m, l] @ d_out_tile[l, p]
                    T.gemm(cb_t_tile, d_out_tile, acc)

                # ============================================================
                # FINAL WRITEOUT
                #
                # dx[m, p]       = acc[m, p] * dt[m]
                # ddt_partial[m] += sum_p acc[m, p] * x[m, p]   (atomic, partial p-tile)
                # ============================================================
                ddt_tile = T.alloc_fragment((block_m,), accum_dtype)
                T.clear(ddt_tile)

                for mm, pp in T.Parallel(block_m, block_p):
                    m_abs = m0 + mm
                    p_abs = p0 + pp
                    if (m_abs < valid_len) and (p_abs < P):
                        dx_out[bz, chunk_start + m_abs, bh, p_abs] = T.cast(
                            acc[mm, pp] * dt_m[mm], dtype,
                        )
                        ddt_tile[mm] = ddt_tile[mm] + acc[mm, pp] * T.cast(
                            x_tile[mm, pp], accum_dtype,
                        )

                # ddt_partial is partial over p-tiles → atomic accumulation
                for mm in T.Parallel(block_m):
                    m_abs = m0 + mm
                    if m_abs < valid_len:
                        T.atomic_add(ddt_partial_out[bz, bh, bc, m_abs], ddt_tile[mm])

        return main

    return kernel_func


@torch.library.custom_op("top::ssd_dx_bwd_fused", mutates_args=())
def _ssd_dx_bwd_fused_wrapped(
    batch: int,
    num_chunks: int,
    chunk_len: int,
    n_heads: int,
    d_head: int,
    d_state: int,
    n_groups: int,
    dtype: str,
    block_m: int,
    block_p: int,
    block_n: int,
    block_lr: int,
    threads: int,
    x: torch.Tensor,
    dt: torch.Tensor,
    dA_cumsum: torch.Tensor,
    B_in: torch.Tensor,
    CB: torch.Tensor,
    d_out: torch.Tensor,
    dstates: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    dx_out = torch.zeros(
        batch, num_chunks * chunk_len, n_heads, d_head,
        dtype=x.dtype, device=x.device,
    )
    ddt_partial_out = torch.zeros(
        batch, n_heads, num_chunks, chunk_len,
        dtype=torch.float32, device=x.device,
    )
    _ssd_dx_bwd_fused_kernel(
        batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype,
    )(block_m, block_p, block_n, block_lr, threads)(
        x, dt, dA_cumsum, B_in, CB, d_out, dstates, dx_out, ddt_partial_out,
    )
    return dx_out, ddt_partial_out


@_ssd_dx_bwd_fused_wrapped.register_fake
def _(
    batch: int,
    num_chunks: int,
    chunk_len: int,
    n_heads: int,
    d_head: int,
    d_state: int,
    n_groups: int,
    dtype: str,
    block_m: int,
    block_p: int,
    block_n: int,
    block_lr: int,
    threads: int,
    x: torch.Tensor,
    dt: torch.Tensor,
    dA_cumsum: torch.Tensor,
    B_in: torch.Tensor,
    CB: torch.Tensor,
    d_out: torch.Tensor,
    dstates: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    dx_out = x.new_empty(
        (batch, num_chunks * chunk_len, n_heads, d_head),
    )
    ddt_partial_out = x.new_empty(
        (batch, n_heads, num_chunks, chunk_len), dtype=torch.float32,
    )
    return dx_out, ddt_partial_out


class SsdDxBwdFusedKernel(Kernel):
    """Mamba-2 SSD backward kernel: fused dx and partial ddt (K7).

    Fuses the state-path and local-scan-path gradient contributions to ``x``
    and computes the partial gradient w.r.t. ``dt`` in a single kernel pass.

    Two paths share a single [block_m, block_p] accumulator ``acc``:

    State path (PART A):
      acc[m, p] += exp(min(a_end - a_m, 0)) * sum_n B[m, n] * dstates[p, n]

    Local chunk-scan path (PART B):
      acc[m, p] += sum_{l >= m} CB[l, m] * exp(min(a_l - a_m, 0)) * d_out[l, p]

    Final:
      dx[b, c*L+m, h, p]        = acc[m, p] * dt[m]
      ddt_partial[b, h, c, m]  += sum_p acc[m, p] * x[m, p]

    Inputs:
      x          [B, S, H, P]      dtype    S = C*L seqlen-fused
      dt         [B, H, C, L]      dtype
      dA_cumsum  [B, H, C, L]      float32
      B_in       [B, S, G, N]      dtype    seqlen-fused, group-owned
      CB         [B, C, G, L, L]   dtype    chunk-local lower-triangular
      d_out       [B, S, H, P]      dtype    S = C*L seqlen-fused
      dstates    [B, C, H, P, N]   float32  upstream grad from state-passing bwd

    Outputs:
      dx_out         [B, S, H, P]  dtype   (same dtype as x)
      ddt_partial_out[B, H, C, L]  float32 (partial; caller sums contributions)
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
        self.kernel = _ssd_dx_bwd_fused_kernel(
            batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, self.dtype_str,
        )
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {
            "block_m": 64,
            "block_p": 64,
            "block_n": 32,
            "block_lr": 64,
            "threads": 128,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        block_m  = [32, 64]
        block_p  = [32, 64]
        block_n  = [16, 32]
        block_lr = [32, 64]
        threads  = [128, 256]
        return [
            {
                "block_m": c[0],
                "block_p": c[1],
                "block_n": c[2],
                "block_lr": c[3],
                "threads": c[4],
            }
            for c in itertools.product(block_m, block_p, block_n, block_lr, threads)
        ]

    def forward(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        dA_cumsum: torch.Tensor,
        B_in: torch.Tensor,
        CB: torch.Tensor,
        d_out: torch.Tensor,
        dstates: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:         [B, S, H, P]      dtype    S = C*L seqlen-fused
            dt:        [B, H, C, L]      dtype
            dA_cumsum: [B, H, C, L]      float32
            B_in:      [B, S, G, N]      dtype    seqlen-fused, group-owned
            CB:        [B, C, G, L, L]   dtype    chunk-local lower-triangular
            d_out:      [B, S, H, P]      dtype    S = C*L seqlen-fused
            dstates:   [B, C, H, P, N]   float32  upstream grad from state-passing bwd

        Returns:
            dx_out:          [B, S, H, P]  dtype
            ddt_partial_out: [B, H, C, L]  float32
        """
        return _ssd_dx_bwd_fused_wrapped(
            self.batch, self.num_chunks, self.chunk_len, self.n_heads,
            self.d_head, self.d_state, self.n_groups, self.dtype_str,
            self.config["block_m"], self.config["block_p"],
            self.config["block_n"], self.config["block_lr"],
            self.config["threads"],
            x.contiguous(),
            dt.contiguous(),
            dA_cumsum.contiguous(),
            B_in.contiguous(),
            CB.contiguous(),
            d_out.contiguous(),
            dstates.contiguous(),
        )
