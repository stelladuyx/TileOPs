"""
Mamba-2 State-Space Dual (SSD) chunk cumsum backward kernel (K11).

Mathematical contract
---------------------
Forward chunk cumsum computes dA_raw = A * dt_work, where:

  dt_pre[h, t]  = dt_in[b, c*T+t, h] + dt_bias[h]   (if dt_bias is not None)
  dt_sp[h, t]   = softplus(dt_pre[h, t])              (if dt_softplus is True)
  dt_work[h, t] = clamp(dt_sp[h, t], dt_min, dt_max)
  dA_raw[h, t]  = A[h] * dt_work[h, t]               (what goes into dA_cumsum)

Backward (given ddA[B, H, C, T] and ddt_out[B, H, C, T]):

  ddt_work[h, t] = ddA[b, h, c, t] * A[h] + ddt_out[b, h, c, t]
  dA[h]         += sum_t( ddA[b, h, c, t] * dt_work[h, t] )   [per (b,c), summed to H]

  backward through clamp:
    ddt_sp[h, t] = 0                    if dt_sp < dt_min or dt_sp > dt_max
                 = ddt_work[h, t]       otherwise

  backward through softplus (if dt_softplus):
    ddt_pre[h, t] = ddt_sp[h, t] * sigmoid(dt_pre[h, t])
  else:
    ddt_pre[h, t] = ddt_sp[h, t]

  ddt[b, c*T+t, h] = ddt_pre[h, t]    (scatter back to sequence layout)
  ddt_bias[h]     += sum_t( ddt_pre[h, t] )   (if dt_bias is not None)

Canonical tensor layouts
------------------------
  ddA      : [B, H, C, T]   float32  (merged total ddA, already finalized)
  ddt_out  : [B, H, C, T]   float32  (partial ddt from earlier kernels)
  dt_in    : [B, S, H]      dtype    S = C*T, seqlen-fused
  A        : [H]             float32
  dt_bias  : [H]             float32  or None

  ddt      : [B, S, H]      float32  (output gradient w.r.t. dt_in)
  dA       : [H]             float32  (output gradient w.r.t. A, atomically accumulated)
  ddt_bias : [H]             float32  or None

Implementation notes
--------------------
- Grid: (B*C, H).  Each program instance owns one (b, c, h) triple.
- The T-dimension is tiled with block_t; positions past valid_len are masked.
- valid_len = min(T, S - c*T) to handle the tail chunk when S is not C*T.
- dA and ddt_bias are accumulated with T.atomic_add because all chunks for the
  same head h write to the same output location.
- dt_softplus and dt_limit are compile-time constants baked into the kernel via
  the outer Python closure; this avoids runtime branching on critical paths.
- Softplus is computed as log(1 + exp(x)); sigmoid as 1 / (1 + exp(-x)).

Grid: (B * C, H, ceildiv(T, block_t))
"""

import itertools
from typing import Callable, Optional

import tilelang
import torch

from tileops.kernels.kernel import Kernel

__all__ = ["SsdChunkCumsumBwdKernel"]


def _ssd_chunk_cumsum_bwd_kernel(
    batch: int,
    num_chunks: int,
    chunk_len: int,
    n_heads: int,
    seq_len: int,
    dt_softplus: bool = False,
    dt_min: float = 0.0,
    dt_max: float = float("inf"),
    has_dt_bias: bool = False,
    dtype: str = "float16",
) -> Callable:
    accum_dtype = "float"

    B = batch
    C = num_chunks
    T = chunk_len
    H = n_heads
    S = seq_len

    @tilelang.jit(out_idx=[-1])
    def kernel_func(block_t: int, threads: int):
        ddA_shape     = (B, H, C, T)   # [B, H, C, T]  float32
        ddt_out_shape = (B, H, C, T)   # [B, H, C, T]  float32
        dt_in_shape   = (B, S, H)      # [B, S, H]      dtype  seqlen-fused
        A_shape       = (H,)            # [H]            float32
        dt_bias_shape = (H,)            # [H]            float32  (may be unused)
        ddt_shape     = (B, S, H)      # [B, S, H]      float32  output

        @T.prim_func
        def main(
            ddA:       T.Tensor(ddA_shape, accum_dtype),      # type: ignore
            ddt_out:   T.Tensor(ddt_out_shape, accum_dtype),  # type: ignore
            dt_in:     T.Tensor(dt_in_shape, dtype),          # type: ignore
            A:         T.Tensor(A_shape, accum_dtype),        # type: ignore
            dt_bias:   T.Tensor(dt_bias_shape, accum_dtype),  # type: ignore
            dA_out:    T.Tensor(A_shape, accum_dtype),        # type: ignore
            ddt_bias_out: T.Tensor(dt_bias_shape, accum_dtype),  # type: ignore
            ddt:       T.Tensor(ddt_shape, accum_dtype),      # type: ignore
        ):
            # Grid: (B*C, H, ceildiv(T, block_t))
            with T.Kernel(
                B * C,
                H,
                T.ceildiv(T, block_t),
                threads=threads,
            ) as (bc_fused, bh, bt):

                bb = bc_fused // C
                bc = bc_fused % C

                t0 = bt * block_t
                chunk_start = bc * T

                # How many valid positions in this chunk?
                valid_len = T.max(T.int32(0), T.min(T, S - chunk_start))

                # Load A[h] once — scalar per block.
                A_h = A[bh]

                # --------------------------------------------------------
                # Step 1: Load dt_in for this (b, h, c, t-tile) block.
                #         Rebuild forward dt_work to enable backward through
                #         clamp and softplus.
                # --------------------------------------------------------
                dt_chunk  = T.alloc_fragment((block_t,), accum_dtype)
                dt_pre    = T.alloc_fragment((block_t,), accum_dtype)
                dt_sp     = T.alloc_fragment((block_t,), accum_dtype)
                dt_work   = T.alloc_fragment((block_t,), accum_dtype)

                for tt in T.Parallel(block_t):
                    t_abs = t0 + tt
                    seq_abs = chunk_start + t_abs
                    dt_chunk[tt] = T.if_then_else(
                        (t_abs < T) and (t_abs < valid_len),
                        T.cast(dt_in[bb, seq_abs, bh], accum_dtype),
                        T.float32(0.0),
                    )

                # Add bias if present (compile-time branch).
                if has_dt_bias:
                    bias_h = dt_bias[bh]
                    for tt in T.Parallel(block_t):
                        dt_pre[tt] = dt_chunk[tt] + bias_h
                else:
                    for tt in T.Parallel(block_t):
                        dt_pre[tt] = dt_chunk[tt]

                # Apply softplus if requested (compile-time branch).
                if dt_softplus:
                    for tt in T.Parallel(block_t):
                        x = dt_pre[tt]
                        dt_sp[tt] = T.if_then_else(
                            x <= T.float32(20.0),
                            T.log(T.float32(1.0) + T.exp(x)),
                            x,
                        )
                else:
                    for tt in T.Parallel(block_t):
                        dt_sp[tt] = dt_pre[tt]

                # Clamp.
                for tt in T.Parallel(block_t):
                    dt_work[tt] = T.min(T.max(dt_sp[tt], T.float32(dt_min)), T.float32(dt_max))

                # --------------------------------------------------------
                # Step 2: Load ddA and ddt_out for this tile.
                # --------------------------------------------------------
                ddA_tile    = T.alloc_fragment((block_t,), accum_dtype)
                ddt_out_tile = T.alloc_fragment((block_t,), accum_dtype)

                for tt in T.Parallel(block_t):
                    t_abs = t0 + tt
                    ddA_tile[tt] = T.if_then_else(
                        (t_abs < T) and (t_abs < valid_len),
                        ddA[bb, bh, bc, t_abs],
                        T.float32(0.0),
                    )
                    ddt_out_tile[tt] = T.if_then_else(
                        (t_abs < T) and (t_abs < valid_len),
                        ddt_out[bb, bh, bc, t_abs],
                        T.float32(0.0),
                    )

                # --------------------------------------------------------
                # Step 3: Gradient w.r.t. dt_work and accumulate dA.
                #
                #   ddt_work[t] = ddA[t] * A[h] + ddt_out[t]
                #   dA[h]      += sum_t( ddA[t] * dt_work[t] )
                # --------------------------------------------------------
                ddt_work = T.alloc_fragment((block_t,), accum_dtype)
                dA_tile  = T.alloc_fragment((block_t,), accum_dtype)

                for tt in T.Parallel(block_t):
                    ddt_work[tt] = ddA_tile[tt] * A_h + ddt_out_tile[tt]
                    dA_tile[tt]  = ddA_tile[tt] * dt_work[tt]

                # Reduce dA_tile -> scalar and atomic-add to dA_out[bh].
                dA_reduce = T.alloc_fragment((1,), accum_dtype)
                T.reduce_sum(dA_tile, dA_reduce, dim=0, clear=True)
                T.atomic_add(dA_out[bh], dA_reduce[0])

                # --------------------------------------------------------
                # Step 4: Backward through clamp.
                #
                #   clamp_mask = (dt_sp < dt_min) or (dt_sp > dt_max)
                #   ddt_sp[t] = 0               if clamp_mask
                #             = ddt_work[t]     otherwise
                # --------------------------------------------------------
                ddt_sp_grad = T.alloc_fragment((block_t,), accum_dtype)

                for tt in T.Parallel(block_t):
                    clamped = (dt_sp[tt] < T.float32(dt_min)) or (dt_sp[tt] > T.float32(dt_max))
                    ddt_sp_grad[tt] = T.if_then_else(clamped, T.float32(0.0), ddt_work[tt])

                # --------------------------------------------------------
                # Step 5: Backward through softplus.
                #
                #   if dt_softplus:
                #     ddt_pre[t] = ddt_sp[t] * sigmoid(dt_pre[t])
                #   else:
                #     ddt_pre[t] = ddt_sp[t]
                # --------------------------------------------------------
                ddt_pre_grad = T.alloc_fragment((block_t,), accum_dtype)

                if dt_softplus:
                    for tt in T.Parallel(block_t):
                        x = dt_pre[tt]
                        sig = T.if_then_else(
                            x <= T.float32(20.0),
                            T.float32(1.0) / (T.float32(1.0) + T.exp(-x)),
                            T.float32(1.0),
                        )
                        ddt_pre_grad[tt] = ddt_sp_grad[tt] * sig
                else:
                    for tt in T.Parallel(block_t):
                        ddt_pre_grad[tt] = ddt_sp_grad[tt]

                # --------------------------------------------------------
                # Step 6: Scatter ddt_pre back to ddt[b, seq_abs, h].
                # --------------------------------------------------------
                for tt in T.Parallel(block_t):
                    t_abs = t0 + tt
                    seq_abs = chunk_start + t_abs
                    if (t_abs < T) and (t_abs < valid_len):
                        ddt[bb, seq_abs, bh] = ddt_pre_grad[tt]

                # --------------------------------------------------------
                # Step 7: Accumulate ddt_bias (if requested).
                #
                #   ddt_bias[h] += sum_t( ddt_pre[t] )
                # --------------------------------------------------------
                if has_dt_bias:
                    ddt_bias_tile   = T.alloc_fragment((1,), accum_dtype)
                    T.reduce_sum(ddt_pre_grad, ddt_bias_tile, dim=0, clear=True)
                    T.atomic_add(ddt_bias_out[bh], ddt_bias_tile[0])

        return main

    return kernel_func


@torch.library.custom_op("top::ssd_chunk_cumsum_bwd", mutates_args=())
def _ssd_chunk_cumsum_bwd_wrapped(
    batch: int,
    num_chunks: int,
    chunk_len: int,
    n_heads: int,
    seq_len: int,
    dt_softplus: bool,
    dt_min: float,
    dt_max: float,
    has_dt_bias: bool,
    dtype: str,
    block_t: int,
    threads: int,
    ddA: torch.Tensor,
    ddt_out: torch.Tensor,
    dt_in: torch.Tensor,
    A: torch.Tensor,
    dt_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = ddA.device
    f32 = torch.float32
    ddt      = torch.zeros(batch, seq_len, n_heads, dtype=f32, device=device)
    dA_out   = torch.zeros(n_heads, dtype=f32, device=device)
    ddt_bias = torch.zeros(n_heads, dtype=f32, device=device)
    _ssd_chunk_cumsum_bwd_kernel(
        batch, num_chunks, chunk_len, n_heads, seq_len,
        dt_softplus, dt_min, dt_max, has_dt_bias, dtype,
    )(block_t, threads)(
        ddA, ddt_out, dt_in, A, dt_bias,
        dA_out, ddt_bias, ddt,
    )
    return ddt, dA_out, ddt_bias


@_ssd_chunk_cumsum_bwd_wrapped.register_fake
def _(
    batch: int,
    num_chunks: int,
    chunk_len: int,
    n_heads: int,
    seq_len: int,
    dt_softplus: bool,
    dt_min: float,
    dt_max: float,
    has_dt_bias: bool,
    dtype: str,
    block_t: int,
    threads: int,
    ddA: torch.Tensor,
    ddt_out: torch.Tensor,
    dt_in: torch.Tensor,
    A: torch.Tensor,
    dt_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    f32 = torch.float32
    ddt      = ddA.new_empty((batch, seq_len, n_heads), dtype=f32)
    dA_out   = A.new_empty((n_heads,), dtype=f32)
    ddt_bias = A.new_empty((n_heads,), dtype=f32)
    return ddt, dA_out, ddt_bias


class SsdChunkCumsumBwdKernel(Kernel):
    """Mamba-2 SSD chunk cumsum backward kernel (K11).

    Differentiates the chunk-local forward pass:

      dt_pre[t]  = dt_in[c*T+t] + dt_bias          (if dt_bias)
      dt_sp[t]   = softplus(dt_pre[t])              (if dt_softplus)
      dt_work[t] = clamp(dt_sp[t], dt_min, dt_max)
      dA_raw[t]  = A * dt_work[t]

    Given the merged total ddA[B, H, C, T] (already summed from all upstream
    paths) and ddt_out[B, H, C, T] (partial ddt from earlier kernels), this
    kernel computes:

      ddt[b, c*T+t, h]  -- gradient w.r.t. dt_in, shape [B, S, H]
      dA[h]             -- gradient w.r.t. A,      shape [H]
      ddt_bias[h]       -- gradient w.r.t. dt_bias, shape [H], or zeros if unused

    Always returns 3 tensors; caller ignores ddt_bias when has_dt_bias=False.

    Inputs:
      ddA      [B, H, C, T]   float32  (merged total ddA)
      ddt_out  [B, H, C, T]   float32  (partial ddt from earlier kernels)
      dt_in    [B, S, H]      dtype    S = C*T seqlen-fused
      A        [H]            float32
      dt_bias  [H]            float32  (pass zeros when has_dt_bias=False)

    Outputs:
      ddt      [B, S, H]      float32
      dA       [H]            float32
      ddt_bias [H]            float32  (valid only when has_dt_bias=True)
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    def __init__(
        self,
        batch: int,
        num_chunks: int,
        chunk_len: int,
        n_heads: int,
        seq_len: int,
        dt_softplus: bool = False,
        dt_limit: tuple[float, float] = (0.0, float("inf")),
        has_dt_bias: bool = False,
        dtype: torch.dtype = torch.float16,
        config: Optional[dict] = None,
        tune: bool = False,
    ) -> None:
        super().__init__()
        self.batch = batch
        self.num_chunks = num_chunks
        self.chunk_len = chunk_len
        self.n_heads = n_heads
        self.seq_len = seq_len
        self.dt_softplus = dt_softplus
        self.dt_min, self.dt_max = dt_limit
        self.has_dt_bias = has_dt_bias
        self.dtype = dtype
        self.kernel = _ssd_chunk_cumsum_bwd_kernel(
            batch, num_chunks, chunk_len, n_heads, seq_len,
            dt_softplus, self.dt_min, self.dt_max, has_dt_bias, self.dtype_str,
        )
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {
            "block_t": 64,
            "threads": 128,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        block_t = [32, 64, 128]
        threads = [64, 128, 256]
        return [
            {"block_t": c[0], "threads": c[1]}
            for c in itertools.product(block_t, threads)
        ]

    def forward(
        self,
        ddA: torch.Tensor,
        ddt_out: torch.Tensor,
        dt_in: torch.Tensor,
        A: torch.Tensor,
        dt_bias: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            ddA:     [B, H, C, T]   float32  merged total ddA
            ddt_out: [B, H, C, T]   float32  partial ddt from earlier kernels
            dt_in:   [B, S, H]      dtype    S = C*T seqlen-fused
            A:       [H]            float32
            dt_bias: [H]            float32, optional

        Returns:
            ddt:      [B, S, H]     float32
            dA:       [H]           float32
            ddt_bias: [H]           float32, or None if dt_bias was None
        """
        if dt_bias is None:
            dt_bias_in = A.new_zeros(self.n_heads)
        else:
            dt_bias_in = dt_bias.contiguous()

        ddt, dA, ddt_bias_out = _ssd_chunk_cumsum_bwd_wrapped(
            self.batch, self.num_chunks, self.chunk_len, self.n_heads, self.seq_len,
            self.dt_softplus, self.dt_min, self.dt_max, self.has_dt_bias,
            self.dtype_str,
            self.config["block_t"], self.config["threads"],
            ddA.contiguous(),
            ddt_out.contiguous(),
            dt_in.contiguous(),
            A.contiguous(),
            dt_bias_in,
        )
        return ddt, dA, ddt_bias_out if self.has_dt_bias else None
