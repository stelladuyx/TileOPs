from typing import Dict, Optional

import torch

from tileops.kernels.kernel import Kernel
from tileops.kernels.mamba.ssd_dx_bwd_fused import SsdDxBwdFusedKernel

from .op import Op

__all__ = ["SsdDxBwdFusedOp"]


class SsdDxBwdFusedOp(Op):
    """Mamba-2 SSD backward operator: fused dx and partial ddt (K7).

    Fuses the state-path and local-scan-path gradient contributions to ``x``
    into a single kernel pass and simultaneously computes the partial gradient
    w.r.t. ``dt``.

    Two paths share a single accumulator ``acc[m, p]``:

    State path (PART A):
      acc[m,p] += exp(min(a_end - a_m, 0)) * sum_n B[m,n] * dstates[p,n]

    Local chunk-scan path (PART B):
      acc[m,p] += sum_{l >= m} CB[l,m] * exp(min(a_l - a_m, 0)) * d_out[l,p]

    Final outputs:
      dx[b, c*L+m, h, p]        = acc[m,p] * dt[m]
      ddt_partial[b, h, c, m]  += sum_p acc[m,p] * x[m,p]

    Args:
        batch:      Batch size.
        num_chunks: Number of chunks (seq_len / chunk_len).
        chunk_len:  Tokens per chunk (L).
        n_heads:    Number of attention heads (H).
        d_head:     Head dimension (P).
        d_state:    State Space Model (SSM) state dimension (N).
        n_groups:   Number of B/CB matrix groups (G); n_heads must be divisible
                    by n_groups.
        dtype:      Data type for inputs (float16 or bfloat16).
        tune:       Whether to autotune tile config on construction.
        kernel_map: Optional override for kernel dispatch.
    """

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
        tune: bool = False,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ) -> None:
        self.batch = batch
        self.num_chunks = num_chunks
        self.chunk_len = chunk_len
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_state = d_state
        self.n_groups = n_groups
        self.dtype = dtype
        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["ssd_dx_bwd_fused"](
            batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups,
            dtype, tune=tune,
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"ssd_dx_bwd_fused": SsdDxBwdFusedKernel}

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
        """Run the fused dx backward pass.

        Args:
            x:         (batch, seq_len, n_heads, d_head)           dtype
            dt:        (batch, n_heads, num_chunks, chunk_len)      dtype
            dA_cumsum: (batch, n_heads, num_chunks, chunk_len)      float32
            B_in:      (batch, seq_len, n_groups, d_state)          dtype
            CB:        (batch, num_chunks, n_groups, chunk_len, chunk_len) dtype
            d_out:      (batch, seq_len, n_heads, d_head)            dtype
            dstates:   (batch, num_chunks, n_heads, d_head, d_state) float32

        Returns:
            dx_out:          (batch, seq_len, n_heads, d_head)           dtype
            ddt_partial_out: (batch, n_heads, num_chunks, chunk_len)     float32
        """
        if not x.is_cuda:
            raise ValueError("x must be a CUDA tensor")
        if x.dtype != self.dtype:
            raise ValueError(f"Expected dtype {self.dtype}, got {x.dtype}")

        return self.kernel.forward(
            x.contiguous(),
            dt.contiguous(),
            dA_cumsum.contiguous(),
            B_in.contiguous(),
            CB.contiguous(),
            d_out.contiguous(),
            dstates.contiguous(),
        )
