from typing import Dict, Optional

import torch

from tileops.kernels.kernel import Kernel
from tileops.kernels.mamba import SsdChunkScanBwdDstatesKernel

from .op import Op

__all__ = ["SsdChunkScanBwdDstatesOp"]


class SsdChunkScanBwdDstatesOp(Op):
    """Mamba-2 SSD backward operator: gradient w.r.t. chunk-entry states (K1).

    Computes:
      dstates[b, c, h, p, n]
        = sum_{l=0}^{L_valid-1}
              d_out[b, c, l, h, p]
            * C[b, c, l, h, n]
            * state_decay_out[b, c, h, l]

    where state_decay_out = exp(dA_cumsum) is the forward history scale.
    """

    def __init__(
        self,
        batch: int,
        num_chunks: int,
        chunk_len: int,
        n_heads: int,
        d_head: int,
        d_state: int,
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
        self.dtype = dtype
        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["ssd_chunk_scan_bwd_dstates"](
            batch, num_chunks, chunk_len, n_heads, d_head, d_state, dtype, tune=tune,
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"ssd_chunk_scan_bwd_dstates": SsdChunkScanBwdDstatesKernel}

    def forward(
        self,
        d_out: torch.Tensor,
        C: torch.Tensor,
        state_decay_out: torch.Tensor,
        chunk_len_valid: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            d_out:            [B, C, L, H, P]  dtype
            C:               [B, C, L, H, N]  dtype
            state_decay_out: [B, C, H, L]     float32  helper = exp(dA_cumsum)
            chunk_len_valid: int, defaults to chunk_len (complete chunks)

        Returns:
            dstates: [B, C, H, P, N]  float32
        """
        if not d_out.is_cuda:
            raise ValueError("d_out must be a CUDA tensor")
        if d_out.dtype != self.dtype:
            raise ValueError(f"Expected dtype {self.dtype}, got {d_out.dtype}")
        return self.kernel.forward(
            d_out.contiguous(),
            C.contiguous(),
            state_decay_out.contiguous(),
            chunk_len_valid,
        )
