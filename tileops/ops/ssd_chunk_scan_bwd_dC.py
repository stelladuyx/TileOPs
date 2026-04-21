from typing import Dict, Optional

import torch

from tileops.kernels.kernel import Kernel
from tileops.kernels.mamba.ssd_chunk_scan_bwd_dC import SsdChunkScanBwdDCKernel

from .op import Op

__all__ = ["SsdChunkScanBwdDCOp"]


class SsdChunkScanBwdDCOp(Op):
    """Mamba-2 SSD backward operator: gradient w.r.t. C and dA_cumsum (prev-state branch).

    Differentiates the history branch of chunk scan:

      y_prev[l, p] = exp(dA_cumsum[l]) * sum_n C[l,n] * states[p,n]

    Computes:

      dC[b,c,g,l,n]            += exp(dA_cumsum[b,c,h,l]) * sum_p d_out[b,c,l,h,p] * states[b,c,h,p,n]
      ddA_cumsum_prev[b,c,h,l] += sum_n dC_contrib[b,c,h,l,n] * C[b,c,g(h),l,n]

    Both outputs are float32 and atomically accumulated.

    Args:
        batch:      Batch size.
        num_chunks: Number of chunks (T / chunk_len).
        chunk_len:  Tokens per chunk (L).
        n_heads:    Number of heads (H).
        d_head:     Head dimension (P).
        d_state:    SSM state dimension (N).
        n_groups:   Number of C/B groups (G); n_heads must be divisible by n_groups.
        dtype:      Data type for inputs (float16 or bfloat16).
        tune:       Whether to autotune tile config on construction.
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
    ):
        self.batch = batch
        self.num_chunks = num_chunks
        self.chunk_len = chunk_len
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_state = d_state
        self.n_groups = n_groups
        self.dtype = dtype
        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["ssd_chunk_scan_bwd_dC"](
            batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype, tune=tune,
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"ssd_chunk_scan_bwd_dC": SsdChunkScanBwdDCKernel}

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
            dA_cumsum:       [B, C, H, L]     float32
            d_out:            [B, C, L, H, P]  dtype
            C_in:            [B, C, G, L, N]  dtype
            valid_chunk_len: [B, C]           int32

        Returns:
            dC_out:              [B, C, G, L, N]  float32
            ddA_cumsum_prev_out: [B, C, H, L]     float32
        """
        if not states.is_cuda:
            raise ValueError("states must be a CUDA tensor")
        if states.dtype != self.dtype:
            raise ValueError(f"Expected dtype {self.dtype}, got {states.dtype}")
        return self.kernel.forward(
            states.contiguous(),
            dA_cumsum.contiguous(),
            d_out.contiguous(),
            C_in.contiguous(),
            valid_chunk_len.contiguous(),
        )
