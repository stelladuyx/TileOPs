"""
Mamba-2 SSD chunk-state backward kernel: dB and ddA_next.

Corresponds to Triton _chunk_state_bwd_dx (K5) — computes gradients w.r.t.
the B matrix and the partial dA decay used in chunk-state accumulation.

For each chunk c and each position l within the chunk:

  chunk_state[b, h, c, :, :] = sum_{l=0}^{Q-1}
      decay[b, h, c, l] * X[b, chunk_start+l, h, :] ⊗ B[b, chunk_start+l, g(h), :]

where  decay[b, h, c, l] = exp(dA_chunk_cumsum[b, h, c, l] - dA_chunk_cumsum[b, h, c, Q-1])
  i.e. chunk_state_decay[b, h, c, l] = exp(A_end_c - A_l).

NOTE: chunk_state_decay already incorporates dt (it is exp(dt * A_end - dt * A_l)).
      The kernel treats scale[l] = chunk_state_decay[b, h, c, l] as a single
      pre-multiplied scalar and does NOT split dt back out.

Given upstream gradient g_u[b, c, h, p, n] (= dL/d chunk_state), this kernel
computes per position l:

  dB[b, chunk_start+l, g(h), n]  += sum_p g_u[b,c,h,p,n] * X[b,chunk_start+l,h,p] * scale[b,h,c,l]
  ddA_next[b, h, c, l]            = (sum_{p,n} g_u[b,c,h,p,n] * X[b,chunk_start+l,h,p]
                                       * B[b,chunk_start+l,g(h),n]) * scale[b,h,c,l]

  ddA_next feeds into the dA_partial accumulation upstream.

Layouts (official Mamba Triton conventions):
  g_u:                  (B, C, H, P, N)   float32   -- upstream grad from state_passing_bwd
  X:                    (B, S, H, P)      dtype     -- seqlen-fused;  S = C * Q
  Bmat:                 (B, S, G, N)      dtype     -- seqlen-fused, group-owned
  chunk_state_decay:    (B, H, C, Q)      float32   -- scale[l] = exp(A_end - A_l), includes dt
  dB_from_chunk_state:  (B, S, G, N)      float32   -- output grad for B (atomic-added)
  ddA_next_chunk_state: (B, H, C, Q)      float32   -- output grad for dA partial

Parallelization:
  axis-0: fused batch * chunk * head  (B*C*H)
  axis-1: tile over Q (chunk positions)
  axis-2: tile over N (d_state)
  P (d_head) is reduced serially across tiles inside each program.

Notation:
  B = batch, C = num_chunks, Q = chunk_len, S = C*Q (seqlen)
  H = n_heads, G = n_groups, P = d_head, N = d_state
"""


import itertools
from typing import Callable

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel import Kernel

__all__ = ["SsdChunkStateBwdDBDApartialKernel"]


def _ssd_chunk_state_bwd_dB_dApartial_kernel(
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
    Q = chunk_len
    S = C * Q           # seqlen-fused sequence length
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
        @T.prim_func
        def main(
            g_u: T.Tensor((B, C, H, P, N), accum_dtype),                     # type: ignore
            X: T.Tensor((B, S, H, P), dtype),                                # type: ignore
            Bmat: T.Tensor((B, S, G, N), dtype),                             # type: ignore
            chunk_state_decay: T.Tensor((B, H, C, Q), accum_dtype),          # type: ignore
            dB_from_chunk_state: T.Tensor((B, S, G, N), accum_dtype),        # type: ignore
            ddA_next_chunk_state: T.Tensor((B, H, C, Q), accum_dtype),       # type: ignore
        ):
            with T.Kernel(
                B * C * H,
                T.ceildiv(Q, block_l),
                T.ceildiv(N, block_n),
                threads=threads,
            ) as (bch, bl, bn):

                bz = bch // (C * H)
                bc = (bch % (C * H)) // H
                bh = bch % H
                bg = bh // HEADS_PER_GROUP
                chunk_start = bc * Q

                l0 = bl * block_l
                n0 = bn * block_n

                # ── Step 1 accumulator ────────────────────────────────────────
                # dB_base[l, n] = sum_p X[l, p] * g_u[p, n]
                # Accumulated in fp32 to preserve precision across the p-reduction.
                dB_base = T.alloc_fragment((block_l, block_n), accum_dtype)
                T.clear(dB_base)

                # scale[l] = chunk_state_decay[b, h, c, l]
                # This scalar already includes dt (= exp(dt*(A_end - A_l))).
                # We do NOT split dt back out — it is treated as one pre-multiplied weight.
                # Out-of-bounds positions are masked to 0 here so they contribute nothing.
                scale = T.alloc_fragment((block_l,), accum_dtype)
                for ll in T.Parallel(block_l):
                    l_idx = l0 + ll
                    scale[ll] = T.if_then_else(
                        l_idx < Q,
                        chunk_state_decay[bz, bh, bc, l_idx],
                        T.cast(0.0, accum_dtype),
                    )

                # Precision note on shared-memory dtypes:
                #   X_tile      → dtype (fp16/bf16): low-precision input, no change needed.
                #   g_u_tile_lp → dtype (fp16/bf16): g_u is fp32 upstream, but we cast it
                #                 to dtype here so that T.gemm operand dtypes match.
                #                 This is safe because dB_base (the accumulator) stays fp32,
                #                 recovering precision after each tile accumulation.
                #   B_tile      → dtype (fp16/bf16): only used elementwise in step 4.
                #   scale       → fp32: applied after the GEMM, never quantised.
                X_tile      = T.alloc_shared((block_l, block_p), dtype)
                g_u_tile_lp = T.alloc_shared((block_p, block_n), dtype)
                B_tile      = T.alloc_shared((block_l, block_n), dtype)

                # Load B_tile once — independent of the p-reduction loop.
                # Valid-chunk masking: positions l_idx >= Q (incomplete last chunk)
                # are explicitly zeroed so they do not pollute the ddA_next reduction.
                for ll, nn in T.Parallel(block_l, block_n):
                    l_idx = l0 + ll
                    n_idx = n0 + nn
                    B_tile[ll, nn] = T.if_then_else(
                        (l_idx < Q) and (n_idx < N),
                        Bmat[bz, chunk_start + l_idx, bg, n_idx],
                        T.cast(T.float32(0.0), dtype),
                    )

                # ── Step 1: primary contraction — GEMM over p tiles ───────────
                #   dB_base[l, n] += X[l, p] * g_u[p, n]   (fp32 accumulator)
                for p_blk in T.Serial(T.ceildiv(P, block_p)):
                    p0 = p_blk * block_p

                    # Load X tile; mask out-of-bounds (tail tile and incomplete chunk).
                    for ll, pp in T.Parallel(block_l, block_p):
                        l_idx = l0 + ll
                        p_idx = p0 + pp
                        X_tile[ll, pp] = T.if_then_else(
                            (l_idx < Q) and (p_idx < P),
                            X[bz, chunk_start + l_idx, bh, p_idx],
                            T.cast(T.float32(0.0), dtype),
                        )

                    # Load g_u tile, cast fp32 → dtype for GEMM operand alignment.
                    for pp, nn in T.Parallel(block_p, block_n):
                        p_idx = p0 + pp
                        n_idx = n0 + nn
                        g_u_tile_lp[pp, nn] = T.if_then_else(
                            (p_idx < P) and (n_idx < N),
                            T.cast(g_u[bz, bc, bh, p_idx, n_idx], dtype),
                            T.cast(T.float32(0.0), dtype),
                        )

                    # GEMM: dB_base += X_tile @ g_u_tile_lp
                    # Both operands are dtype (fp16/bf16); accumulator dB_base is fp32.
                    T.gemm(X_tile, g_u_tile_lp, dB_base)

                # ── Step 2: row-wise scale ────────────────────────────────────
                # Apply scale[l] (which already includes dt) once, after all p-tiles.
                #   dB_acc[l, n] = dB_base[l, n] * scale[l]
                dB_acc = T.alloc_fragment((block_l, block_n), accum_dtype)
                for ll, nn in T.Parallel(block_l, block_n):
                    dB_acc[ll, nn] = dB_base[ll, nn] * scale[ll]

                # ── Step 3: write primary output dB ──────────────────────────
                # Atomic-add for multi-head → group reduction.
                for ll, nn in T.Parallel(block_l, block_n):
                    l_idx = l0 + ll
                    n_idx = n0 + nn
                    if (l_idx < Q) and (n_idx < N):
                        T.atomic_add(
                            dB_from_chunk_state[bz, chunk_start + l_idx, bg, n_idx],
                            dB_acc[ll, nn],
                        )

                # ── Step 4: secondary output ddA_next ────────────────────────
                # ddA_next[l] = sum_n dB_acc[l, n] * B[l, n]
                #
                # Derivation:
                #   dB_acc[l, n] = scale[l] * sum_p X[l,p] * g_u[p,n]
                #   sum_n dB_acc[l,n] * B[l,n]
                #     = scale[l] * sum_{p,n} X[l,p] * g_u[p,n] * B[l,n]
                #     = scale[l] * g_decay_state[l]
                #     = ddA_next[l]
                #
                # Valid-chunk masking: n_idx >= N positions were zeroed in B_tile,
                # so they contribute 0 to the reduction automatically.
                dB_x_B = T.alloc_fragment((block_l, block_n), accum_dtype)
                for ll, nn in T.Parallel(block_l, block_n):
                    n_idx = n0 + nn
                    dB_x_B[ll, nn] = T.if_then_else(
                        n_idx < N,
                        dB_acc[ll, nn] * T.cast(B_tile[ll, nn], accum_dtype),
                        T.float32(0.0),
                    )

                ddA_next_acc = T.alloc_fragment((block_l,), accum_dtype)
                T.reduce_sum(dB_x_B, ddA_next_acc, dim=1)

                for ll in T.Parallel(block_l):
                    l_idx = l0 + ll
                    if l_idx < Q:
                        T.atomic_add(
                            ddA_next_chunk_state[bz, bh, bc, l_idx],
                            ddA_next_acc[ll],
                        )

        return main

    return kernel_func


@torch.library.custom_op("top::ssd_chunk_state_bwd_dB_dApartial", mutates_args=())
def _ssd_chunk_state_bwd_dB_dApartial_wrapped(
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
    g_u: torch.Tensor,
    X: torch.Tensor,
    Bmat: torch.Tensor,
    chunk_state_decay: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    dB = torch.zeros(
        batch, num_chunks * chunk_len, n_groups, d_state,
        dtype=torch.float32, device=g_u.device,
    )
    ddA_next_chunk_state = torch.zeros(
        batch, n_heads, num_chunks, chunk_len,
        dtype=torch.float32, device=g_u.device,
    )
    _ssd_chunk_state_bwd_dB_dApartial_kernel(
        batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype
    )(block_l, block_n, block_p, threads)(
        g_u, X, Bmat, chunk_state_decay, dB, ddA_next_chunk_state
    )
    return dB, ddA_next_chunk_state


@_ssd_chunk_state_bwd_dB_dApartial_wrapped.register_fake
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
    g_u: torch.Tensor,
    X: torch.Tensor,
    Bmat: torch.Tensor,
    chunk_state_decay: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    dB = g_u.new_zeros((batch, num_chunks * chunk_len, n_groups, d_state), dtype=torch.float32)
    ddA_next = g_u.new_zeros((batch, n_heads, num_chunks, chunk_len), dtype=torch.float32)
    return dB, ddA_next


class SsdChunkStateBwdDBDApartialKernel(Kernel):
    """Mamba-2 SSD chunk-state backward kernel: dB and ddA_next.

    Computes per position l within each chunk c:

      dB[b, chunk_start+l, g(h), n] +=
          sum_p g_u[b,c,h,p,n] * X[b,chunk_start+l,h,p] * scale[b,h,c,l]

      ddA_next[b, h, c, l] =
          (sum_{p,n} g_u[b,c,h,p,n] * X[b,chunk_start+l,h,p]
           * B[b,chunk_start+l,g(h),n]) * scale[b,h,c,l]

    where scale[b,h,c,l] = chunk_state_decay[b,h,c,l] = exp(A_end_c - A_l),
    which already includes dt. chunk_start = c * Q.

    ddA_next feeds into the dA_partial accumulation upstream.

    Inputs:
      g_u:               (B, C, H, P, N)  float32
      X:                 (B, S, H, P)     dtype     S = C*Q seqlen-fused
      Bmat:              (B, S, G, N)     dtype     S = C*Q seqlen-fused
      chunk_state_decay: (B, H, C, Q)     float32   scale already includes dt

    Outputs:
      dB_from_chunk_state:  (B, S, G, N)  float32   (atomic-added into output)
      ddA_next_chunk_state: (B, H, C, Q)  float32
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
        config: dict | None = None,
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
        self.kernel = _ssd_chunk_state_bwd_dB_dApartial_kernel(
            batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, self.dtype_str
        )
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {
            "block_l": 32,
            "block_n": 32,
            "block_p": 64,
            "threads": 128,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        block_l = [16, 32, 64]
        block_n = [16, 32]
        block_p = [32, 64]
        threads = [128, 256]
        _configs = list(itertools.product(block_l, block_n, block_p, threads))
        return [{"block_l": c[0], "block_n": c[1], "block_p": c[2], "threads": c[3]} for c in _configs]

    def forward(
        self,
        g_u: torch.Tensor,
        X: torch.Tensor,
        Bmat: torch.Tensor,
        chunk_state_decay: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            g_u:               (B, C, H, P, N)  float32  -- upstream grad from state_passing_bwd
            X:                 (B, S, H, P)     dtype    -- seqlen-fused input; S = C*Q
            Bmat:              (B, S, G, N)     dtype    -- seqlen-fused B matrix, group-owned
            chunk_state_decay: (B, H, C, Q)     float32  -- scale[l] = exp(A_end-A_l), includes dt

        Returns:
            dB_from_chunk_state:  (B, S, G, N)  float32
            ddA_next_chunk_state: (B, H, C, Q)  float32
        """
        return _ssd_chunk_state_bwd_dB_dApartial_wrapped(
            self.batch, self.num_chunks, self.chunk_len, self.n_heads, self.d_head,
            self.d_state, self.n_groups, self.dtype_str,
            self.config["block_l"], self.config["block_n"], self.config["block_p"],
            self.config["threads"],
            g_u, X, Bmat, chunk_state_decay,
        )
