from typing import Any, Optional
import torch
from torch.autograd.function import FunctionCtx

from top.ops import (Fp8QuantOp, Fp8LightingIndexerOp, TopkSelectorOp,
                     DeepSeekSparseAttentionDecodeWithKVCacheOp)

from .function import Function

__all__ = ['FusedDeepSeekSparseAttentionFunc']


class FusedDSACtx(torch.autograd.Function):

    @staticmethod
    def forward(
            ctx: FunctionCtx,
            index_q: torch.Tensor,  # [batch, seq_len, heads, index_dim]
            index_k: torch.Tensor,  # [batch, seq_len_kv, index_dim] (原始float16)
            weights: torch.Tensor,  # [batch, seq_len, heads]
            cu_seqlen_ks: torch.Tensor,  # [batch, seq_len]
            cu_seqlen_ke: torch.Tensor,  # [batch, seq_len]
            starts: torch.Tensor,  # [batch] or [batch, seq_len]
            ends: torch.Tensor,  # [batch] or [batch, seq_len]
            query: torch.Tensor,  # [batch, seq_len, heads, dim]
            kv_cache: torch.Tensor,  # [batch, seq_len_kv, heads, dim+dim_tail]
            quant_op: Fp8QuantOp,
            indexer_op: Fp8LightingIndexerOp,
            topk_op: TopkSelectorOp,
            dsa_op: DeepSeekSparseAttentionDecodeWithKVCacheOp) -> torch.Tensor:

        # Step 1: Quantize index_k to FP8
        index_k_scale, index_k_fp8 = quant_op(index_k)

        # Step 2: Compute index scores using indexer
        # index_q: [batch, seq_len, heads, index_dim]
        # index_k_fp8: [batch, seq_len_kv, index_dim]
        # weights: [batch, seq_len, heads]
        # cu_seqlen_ks, cu_seqlen_ke: [batch, seq_len]
        seq_len = index_q.shape[1]
        index_dtype = index_k_fp8.dtype
        seq_len_kv = index_k_fp8.shape[1]

        cu_seqlen_ks = torch.zeros(seq_len, device='cuda', dtype=index_dtype)
        cu_seqlen_ke = torch.full((seq_len,),
                                  fill_value=seq_len_kv - 1,
                                  device='cuda',
                                  dtype=index_dtype)
        index_scores = indexer_op(index_q, index_k_fp8, weights, cu_seqlen_ks, cu_seqlen_ke)

        # Step 3: Select top-k indices
        # index_scores: [batch, seq_len, seq_len_kv]
        # starts, ends: [batch] or [batch, seq_len]
        indices = topk_op(index_scores, starts, ends)

        # Step 4: Apply sparse attention with selected indices
        # query: [batch, seq_len, heads, dim]
        # kv_cache: [batch, seq_len_kv, heads, dim+dim_tail]
        # indices: [batch, seq_len, topk]
        output = dsa_op(query, kv_cache, indices)

        return output

    @staticmethod
    def backward(ctx: FunctionCtx, do: torch.Tensor) -> Any:
        raise RuntimeError("Inference-only op")


class FusedDeepSeekSparseAttentionFunc(Function):

    def __init__(
            self,
            # Quant op parameters
            seq_len_kv: int,
            index_dim: int,

            # Indexer op parameters
            seq_len: int,
            heads: int,

            # Topk op parameters
            batch: int,
            topk: int,

            # DSA op parameters
            dim: int,
            dim_tail: int,
            stride_kv: int,
            group_kv: int,
            q_start_index_s: int,

            #default arguments
            quant_in_dtype: torch.dtype = torch.float16,
            clean_logits: bool = True,
            indexer_config: Optional[dict] = None,
            in_dtype: str = "float16",
            out_dtype: str = "int32",
            sm_scale: Any = None,
            is_causal: bool = True,
            dsa_dtype: torch.dtype = torch.float16,

            # Common parameters
            tune: bool = False):
        self.batch = batch
        self.seq_len = seq_len
        self.seq_len_kv = seq_len_kv
        self.heads = heads
        self.dim = dim
        self.dim_tail = dim_tail
        self.topk = topk
        self.index_dim = index_dim

        self.quant_op = Fp8QuantOp(
            seq_len_kv=seq_len_kv, index_dim=index_dim, in_dtype=quant_in_dtype, tune=tune)

        self.indexer_op = Fp8LightingIndexerOp(
            seq_len=seq_len,
            heads=heads,
            index_dim=index_dim,
            seq_len_kv=seq_len_kv,
            clean_logits=clean_logits,
            config=indexer_config,
            tune=tune)

        self.topk_op = TopkSelectorOp(
            batch=batch,
            seq_len=seq_len,
            topk=topk,
            in_dtype=in_dtype,
            out_dtype=out_dtype,
            tune=tune)

        self.dsa_op = DeepSeekSparseAttentionDecodeWithKVCacheOp(
            batch=batch,
            heads=heads,
            seq_len=seq_len,
            seq_len_kv=seq_len_kv,
            dim=dim,
            dim_tail=dim_tail,
            topk=topk,
            stride_kv=stride_kv,
            group_kv=group_kv,
            q_start_index_s=q_start_index_s,
            sm_scale=sm_scale,
            is_causal=is_causal,
            dtype=dsa_dtype,
            tune=tune)

    def forward(
        self,
        index_q: torch.Tensor,  # [batch, seq_len, heads, index_dim]
        index_k: torch.Tensor,  # [batch, seq_len_kv, index_dim] (原始float16)
        weights: torch.Tensor,  # [batch, seq_len, heads]
        cu_seqlen_ks: torch.Tensor,  # [batch, seq_len]
        cu_seqlen_ke: torch.Tensor,  # [batch, seq_len]
        starts: torch.Tensor,  # [batch] or [batch, seq_len]
        ends: torch.Tensor,  # [batch] or [batch, seq_len]
        query: torch.Tensor,  # [batch, seq_len, heads, dim]
        kv_cache: torch.Tensor  # [batch, seq_len_kv, heads, dim+dim_tail]
    ) -> torch.Tensor:
        """
        Sparse attention fusion forward propagation

        Parameters:
        index_q: Query index vector [batch, seq_len, heads, index_dim]
        index_k: Key index vector [batch, seq_len_kv, index_dim] (will be quantized to FP8)
        weights: Attention weights [batch, seq_len, heads]
        cu_seqlen_ks: Starting positions of KV for each query token [batch, seq_len]
        cu_seqlen_ke: Ending positions of KV for each query token [batch, seq_len]
        starts: Starting positions of topk selection [batch] or [batch, seq_len]
        ends: Ending positions of topk selection [batch] or [batch, seq_len]
        query: Query vector [batch, seq_len, heads, dim]
        kv_cache: KV cache [batch, seq_len_kv, heads, dim+dim_tail]

        Returns:
        output: Attention output [batch, seq_len, heads, dim]
        """

        assert index_q.shape == (self.batch, self.seq_len, self.heads, self.index_dim), \
            f"index_q shape mismatch: {index_q.shape} != ({self.batch}, {self.seq_len}, {self.heads}, {self.index_dim})"

        assert index_k.shape == (self.batch, self.seq_len_kv, self.index_dim), \
            f"index_k shape mismatch: {index_k.shape} != ({self.batch}, {self.seq_len_kv}, {self.index_dim})"

        assert weights.shape == (self.batch, self.seq_len, self.heads), \
            f"weights shape mismatch: {weights.shape} != ({self.batch}, {self.seq_len}, {self.heads})"

        assert query.shape == (self.batch, self.seq_len, self.heads, self.dim), \
            f"query shape mismatch: {query.shape} != ({self.batch}, {self.seq_len}, {self.heads}, {self.dim})"

        return FusedDSACtx.apply(index_q, index_k, weights, cu_seqlen_ks, cu_seqlen_ke, starts,
                                 ends, query, kv_cache, self.quant_op, self.indexer_op,
                                 self.topk_op, self.dsa_op)
