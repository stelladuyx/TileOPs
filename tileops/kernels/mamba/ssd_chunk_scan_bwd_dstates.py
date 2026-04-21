"""
Mamba-2 SSD chunk scan backward kernel: gradient w.r.t. chunk-entry states.

Stage: K1 / _chunk_scan_bwd_dstates

Mathematical contract (aligned with official Triton K1):

  dstates[b, c, h, p, n]
    = sum_{l=0}^{L_valid-1}
          dout[b, c*L+l, h, p]
        * C[b, c*L+l, h, n]
        * exp(dA_cumsum[b, h, c, l])

where L_valid <= L is the number of valid token positions in chunk (b, c)
(supports incomplete final chunks; positions l >= L_valid are masked out).

Canonical tensor layouts (official Mamba Triton conventions):
  dout:            [B, S, H, P]   -- upstream gradient, dtype     S = C*L seqlen-fused
  C_mat:           [B, S, H, N]   -- readout matrix, dtype        S = C*L seqlen-fused
  dA_cumsum:       [B, H, C, L]   -- cumulative A*dt, float32     H before C
  valid_chunk_len: [B, C]         -- valid token count per chunk, int32
  dstates:         [B, C, H, P, N]-- output gradient, float32

Implementation notes:
  - Parallel axes: (B * H * C, ceil(N/block_n), ceil(P/block_p))
  - Reduction axis: L (tiled in block_l steps)
  - Each program owns one (p_tile, n_tile) output tile for one (b, c, h)
    -> no atomic needed on output
  - chunk_start = bc * L is added to seqlen-fused indexing for dout and C_mat
  - exp(dA_cumsum) is computed inside the kernel (no pre-exponentiated helper)
  - Scaling is applied to C_tile (not dout_tile):
      scaled_C[l, n] = C[l, n] * exp(dA_cumsum[l])   (computed in float32)
      scaled_C is then cast to dtype for GEMM
  - Core GEMM: acc[n, p] += scaled_C_cast^T @ dout_tile
      scaled_C_cast: (block_l, block_n), transpose_A=True -> treats as (block_n, block_l)
      dout_tile:     (block_l, block_p)
      acc:           (block_n, block_p), float32
  - Precision: decay float32, scaled_C float32 then cast to dtype before GEMM,
    dout stays dtype, accumulator float32

Incomplete-chunk masking:
  - valid_chunk_len[b, c]: number of valid positions in chunk (b, c)
  - positions l >= valid_chunk_len[b, c] are zeroed in both dout_tile and scaled_C_tile
  - for complete chunks: valid_chunk_len[b, c] == L (no extra masking cost)

Grid: (B*H*C, ceildiv(N, block_n), ceildiv(P, block_p))
"""

import itertools
from typing import Callable, Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel import Kernel

__all__ = ["SsdChunkScanBwdDstatesKernel"]


def _ssd_chunk_scan_bwd_dstates_kernel(
    batch: int,
    num_chunks: int,
    chunk_len: int,
    n_heads: int,
    d_head: int,
    d_state: int,
    dtype: str = "float16",
) -> Callable:
    accum_dtype = "float"

    B = batch
    C = num_chunks
    Q = chunk_len
    S = C * Q          # seqlen-fused sequence length
    H = n_heads
    P = d_head
    N = d_state

    @tilelang.jit(out_idx=[-1])
    def kernel_func(
        block_n: int,
        block_p: int,
        block_l: int,
        threads: int,
    ):
        # canonical layouts
        dout_shape         = (B, S, H, P)    # [B, S, H, P]  seqlen-fused
        c_shape            = (B, S, H, N)    # [B, S, H, N]  seqlen-fused
        dA_cum_shape       = (B, H, C, Q)    # [B, H, C, L]  float32
        vcl_shape          = (B, C)          # [B, C]  int32
        dstates_shape      = (B, C, H, P, N) # [B, C, H, P, N]

        @T.prim_func
        def main(
            dout:            T.Tensor(dout_shape, dtype),          # type: ignore
            C_mat:           T.Tensor(c_shape, dtype),             # type: ignore
            dA_cumsum:       T.Tensor(dA_cum_shape, accum_dtype),  # type: ignore
            valid_chunk_len: T.Tensor(vcl_shape, "int32"),         # type: ignore
            dstates:         T.Tensor(dstates_shape, accum_dtype), # type: ignore
        ):
            with T.Kernel(
                B * H * C,
                T.ceildiv(N, block_n),
                T.ceildiv(P, block_p),
                threads=threads,
            ) as (bhc, bn, bp):
                # decode fused axis
                bz = bhc // (H * C)
                bh = (bhc % (H * C)) // C
                bc = bhc % C

                n0 = bn * block_n
                p0 = bp * block_p

                chunk_start  = bc * Q
                chunk_valid  = valid_chunk_len[bz, bc]

                # accumulator for one dstates tile [block_n, block_p], float32
                acc = T.alloc_fragment((block_n, block_p), accum_dtype)
                T.clear(acc)

                # shared memory tiles (reused each l-block)
                # scaled_C_shared: C scaled by decay, cast to dtype for GEMM
                scaled_C_shared  = T.alloc_shared((block_l, block_n), dtype)
                dout_shared      = T.alloc_shared((block_l, block_p), dtype)

                # float32 fragments for intermediate computation
                decay_frag       = T.alloc_fragment((block_l,), accum_dtype)
                scaled_C_f32     = T.alloc_fragment((block_l, block_n), accum_dtype)

                # reduce over L in block_l tiles
                for l_blk in T.Serial(T.ceildiv(Q, block_l)):
                    l0 = l_blk * block_l

                    # --------------------------------------------------------
                    # Load dA_cumsum[b, h, c, l] and compute exp(dA_cumsum).
                    # Zero out positions beyond chunk_valid (incomplete chunk).
                    # --------------------------------------------------------
                    for ll in T.Parallel(block_l):
                        l_abs = l0 + ll
                        decay_frag[ll] = T.if_then_else(
                            (l_abs < Q) and (l_abs < chunk_valid),
                            T.exp(dA_cumsum[bz, bh, bc, l_abs]),
                            T.float32(0.0),
                        )

                    # --------------------------------------------------------
                    # Load C_mat tile and scale by decay in float32:
                    #   scaled_C_f32[l, n] = C[chunk_start+l, h, n] * exp(dA_cumsum[l])
                    # Then cast to dtype for GEMM.
                    # --------------------------------------------------------
                    for ll, nn in T.Parallel(block_l, block_n):
                        l_abs = l0 + ll
                        n_abs = n0 + nn
                        c_val = T.if_then_else(
                            (l_abs < Q) and (n_abs < N)
                            and (l_abs < chunk_valid),
                            T.cast(C_mat[bz, chunk_start + l_abs, bh, n_abs], accum_dtype),
                            T.float32(0.0),
                        )
                        scaled_C_f32[ll, nn] = c_val * decay_frag[ll]

                    # cast float32 scaled_C to dtype for GEMM
                    # (TileLang GEMM requires both operands to share dtype)
                    T.copy(scaled_C_f32, scaled_C_shared)

                    # --------------------------------------------------------
                    # Load dout tile: dout[b, chunk_start+l, h, p]
                    # layout: [B, S, H, P]  seqlen-fused
                    # Zero out positions beyond chunk_valid
                    # --------------------------------------------------------
                    for ll, pp in T.Parallel(block_l, block_p):
                        l_abs = l0 + ll
                        p_abs = p0 + pp
                        dout_shared[ll, pp] = T.if_then_else(
                            (l_abs < Q) and (p_abs < P)
                            and (l_abs < chunk_valid),
                            dout[bz, chunk_start + l_abs, bh, p_abs],
                            T.cast(T.float32(0.0), dtype),
                        )

                    # --------------------------------------------------------
                    # Core GEMM:
                    #   acc[n, p] += scaled_C_cast^T @ dout_tile
                    #
                    #   scaled_C_shared: (block_l, block_n), transpose_A=True
                    #     -> treated as (block_n, block_l)
                    #   dout_shared:     (block_l, block_p)
                    #   acc:             (block_n, block_p), float32
                    # --------------------------------------------------------
                    T.gemm(scaled_C_shared, dout_shared, acc, transpose_A=True)

                # write output: dstates[bz, bc, bh, p, n]
                # layout: [B, C, H, P, N]
                for nn, pp in T.Parallel(block_n, block_p):
                    n_abs = n0 + nn
                    p_abs = p0 + pp
                    if n_abs < N and p_abs < P:
                        dstates[bz, bc, bh, p_abs, n_abs] = acc[nn, pp]

        return main

    return kernel_func


@torch.library.custom_op("top::ssd_chunk_scan_bwd_dstates", mutates_args=())
def _ssd_chunk_scan_bwd_dstates_wrapped(
    batch: int,
    num_chunks: int,
    chunk_len: int,
    n_heads: int,
    d_head: int,
    d_state: int,
    dtype: str,
    block_n: int,
    block_p: int,
    block_l: int,
    threads: int,
    dout: torch.Tensor,
    C: torch.Tensor,
    dA_cumsum: torch.Tensor,
    valid_chunk_len: torch.Tensor,
) -> torch.Tensor:
    return _ssd_chunk_scan_bwd_dstates_kernel(
        batch, num_chunks, chunk_len, n_heads, d_head, d_state, dtype)(
        block_n, block_p, block_l, threads,
    )(dout, C, dA_cumsum, valid_chunk_len)


@_ssd_chunk_scan_bwd_dstates_wrapped.register_fake
def _(
    batch: int,
    num_chunks: int,
    chunk_len: int,
    n_heads: int,
    d_head: int,
    d_state: int,
    dtype: str,
    block_n: int,
    block_p: int,
    block_l: int,
    threads: int,
    dout: torch.Tensor,
    C: torch.Tensor,
    dA_cumsum: torch.Tensor,
    valid_chunk_len: torch.Tensor,
) -> torch.Tensor:
    # output shape: [B, C, H, P, N]
    return dout.new_empty(
        (batch, num_chunks, n_heads, d_head, d_state), dtype=torch.float32)


class SsdChunkScanBwdDstatesKernel(Kernel):
    """Mamba-2 SSD backward kernel: gradient w.r.t. chunk-entry states (K1).

    Computes:
      dstates[b, c, h, p, n]
        = sum_{l=0}^{L_valid-1}
              dout[b, c*L+l, h, p]
            * C[b, c*L+l, h, n]
            * exp(dA_cumsum[b, h, c, l])

    Inputs:
      dout            [B, S, H, P]  dtype    S = C*L seqlen-fused
      C               [B, S, H, N]  dtype    S = C*L seqlen-fused
      dA_cumsum       [B, H, C, L]  float32
      valid_chunk_len [B, C]        int32

    Output:
      dstates         [B, C, H, P, N]  float32
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
        self.dtype = dtype
        self.kernel = _ssd_chunk_scan_bwd_dstates_kernel(
            batch, num_chunks, chunk_len, n_heads, d_head, d_state, self.dtype_str,
        )
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {
            "block_n": 32,
            "block_p": 64,
            "block_l": 64,
            "threads": 128,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        block_n = [16, 32]
        block_p = [32, 64]
        block_l = [32, 64]
        threads = [128, 256]
        return [{
            "block_n": c[0],
            "block_p": c[1],
            "block_l": c[2],
            "threads": c[3],
        } for c in itertools.product(block_n, block_p, block_l, threads)]

    def forward(
        self,
        dout: torch.Tensor,
        C: torch.Tensor,
        dA_cumsum: torch.Tensor,
        valid_chunk_len: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            dout:            [B, S, H, P]  dtype    S = C*L seqlen-fused
            C:               [B, S, H, N]  dtype    S = C*L seqlen-fused
            dA_cumsum:       [B, H, C, L]  float32
            valid_chunk_len: [B, C] int32, defaults to all chunk_len (complete chunks)

        Returns:
            dstates: [B, C, H, P, N]  float32
        """
        if valid_chunk_len is None:
            valid_chunk_len = torch.full(
                (self.batch, self.num_chunks), self.chunk_len,
                dtype=torch.int32, device=dout.device,
            )
        return _ssd_chunk_scan_bwd_dstates_wrapped(
            self.batch, self.num_chunks, self.chunk_len, self.n_heads, self.d_head,
            self.d_state, self.dtype_str,
            self.config["block_n"], self.config["block_p"], self.config["block_l"],
            self.config["threads"],
            dout.contiguous(),
            C.contiguous(),
            dA_cumsum.contiguous(),
            valid_chunk_len.contiguous(),
        )
