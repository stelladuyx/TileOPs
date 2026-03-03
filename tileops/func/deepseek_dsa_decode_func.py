from typing import Any

import time
import torch
from torch.autograd.function import FunctionCtx

from tileops.ops import (Fp8LightingIndexerOp, Fp8QuantOp, TopkSelectorOp,
                         DeepSeekSparseAttentionDecodeWithKVCacheOp)
from tileops.utils import str2dtype

from .function import Function

__all__ = ["DeepSeekDSAFusedFunc"]


class FusedDSACtx(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx: FunctionCtx,
        index_q: torch.Tensor,  # [batch, seq_len, heads, index_dim]
        index_k: torch.Tensor,  # [batch, seq_len_kv, kv_group, index_dim]
        weights: torch.Tensor,  # [seq_len, heads]
        cu_seqlen_ks: torch.Tensor,  # [seq_len]
        cu_seqlen_ke: torch.Tensor,  # [seq_len]
        starts: torch.Tensor,  # [batch, seq_len]
        ends: torch.Tensor,  # [batch, seq_len]
        query: torch.Tensor,  # [batch, seq_len, heads, dim + dim_tail]
        kv_cache: torch.Tensor,  # [batch, seq_len_kv, kv_group, dim + dim_tail]
        quant_op: Fp8QuantOp,
        indexer_op: Fp8LightingIndexerOp,
        topk_op: TopkSelectorOp,
        dsa_op: DeepSeekSparseAttentionDecodeWithKVCacheOp,
    ) -> torch.Tensor:
        t0 = time.time()
        print("[FusedDSA] forward start", flush=True)

        # 1) Quantize index_k to FP8
        print("[FusedDSA] step1 quant_op...", flush=True)
        index_k_scale, index_k_fp8 = quant_op(index_k)
        t1 = time.time()
        print(f"[FusedDSA] step1 done in {t1 - t0:.3f}s", flush=True)

        # 2) Compute index scores using indexer
        print("[FusedDSA] step2 indexer_op...", flush=True)
        index_scores = indexer_op(index_q, index_k_fp8, index_k_scale, weights, cu_seqlen_ks,
                                  cu_seqlen_ke)
        t2 = time.time()
        print(f"[FusedDSA] step2 done in {t2 - t1:.3f}s", flush=True)

        # 3) Select top‑k indices
        print("[FusedDSA] step3 topk_op...", flush=True)
        indices = topk_op(index_scores, starts, ends)
        t3 = time.time()
        print(f"[FusedDSA] step3 done in {t3 - t2:.3f}s", flush=True)

        # 4) Sparse attention decode with selected indices
        print("[FusedDSA] step4 dsa_op...", flush=True)
        output = dsa_op(query, kv_cache, indices)
        t4 = time.time()
        print(f"[FusedDSA] step4 done in {t4 - t3:.3f}s, total {t4 - t0:.3f}s", flush=True)
        return output

        # return index_k_scale

    @staticmethod
    def get_output_shape(input_shape: tuple[int, ...]) -> tuple[int, ...]:
        return input_shape

    @staticmethod
    def backward(ctx: FunctionCtx, do: torch.Tensor) -> Any:
        raise RuntimeError("Inference-only op")


class DeepSeekDSAFusedFunc(Function):

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
        # Default arguments
        quant_in_dtype: torch.dtype = torch.float16,
        clean_logits: bool = True,
        in_dtype: str = "float16",
        out_dtype: str = "int32",
        sm_scale: Any = None,
        is_causal: bool = True,
        dsa_dtype: torch.dtype = torch.float16,
        # Common parameters
        tune: bool = False,
    ):
        self.batch = batch
        self.seq_len = seq_len
        self.seq_len_kv = seq_len_kv
        self.heads = heads
        self.dim = dim
        self.dim_tail = dim_tail
        self.topk = topk
        self.index_dim = index_dim
        self.group_kv = group_kv

        # kv_group is the same as group_kv for these ops
        kv_group = group_kv

        self.quant_op = Fp8QuantOp(
            batch=batch,
            seq_len_kv=seq_len_kv,
            kv_group=kv_group,
            index_dim=index_dim,
            in_dtype=quant_in_dtype,
            tune=tune,
        )

        self.indexer_op = Fp8LightingIndexerOp(
            batch=batch,
            seq_len=seq_len,
            heads=heads,
            index_dim=index_dim,
            seq_len_kv=seq_len_kv,
            kv_group=kv_group,
            clean_logits=clean_logits,
            tune=tune,
        )

        self.topk_op = TopkSelectorOp(
            batch=batch,
            seq_len=seq_len,
            seq_len_kv=seq_len_kv,
            kv_group=kv_group,
            topk=topk,
            in_dtype=torch.float32,
            out_dtype=str2dtype[out_dtype],
            tune=tune,
        )

        self.dsa_op = DeepSeekSparseAttentionDecodeWithKVCacheOp(
            batch=batch,
            heads=heads,
            seq_len=seq_len,
            seq_len_kv=seq_len_kv,
            dim=dim,
            dim_tail=dim_tail,
            topk=topk,
            stride_kv=stride_kv,
            heads_kv=group_kv,
            q_start_index_s=q_start_index_s,
            sm_scale=sm_scale,
            is_causal=is_causal,
            dtype=dsa_dtype,
            tune=tune,
        )

    def forward(
        self,
        index_q: torch.Tensor,  # [batch, seq_len, heads, index_dim]
        index_k: torch.Tensor,  # [batch, seq_len_kv, kv_group, index_dim]
        weights: torch.Tensor,  # [seq_len, heads]
        cu_seqlen_ks: torch.Tensor,  # [seq_len]
        cu_seqlen_ke: torch.Tensor,  # [seq_len]
        starts: torch.Tensor,  # [batch, seq_len]
        ends: torch.Tensor,  # [batch, seq_len]
        query: torch.Tensor,  # [batch, seq_len, heads, dim + dim_tail]
        kv_cache: torch.Tensor,  # [batch, seq_len_kv, kv_group, dim + dim_tail]
    ) -> torch.Tensor:
        """
        Sparse attention fusion forward propagation.

        All tensors are validated to match the constructor‑provided shapes, then
        dispatched through the fused CUDA pipeline.
        """

        assert index_q.shape == (self.batch, self.seq_len, self.heads, self.index_dim), (
            f"index_q shape mismatch: {index_q.shape} "
            f"!= ({self.batch}, {self.seq_len}, {self.heads}, {self.index_dim})"
        )

        assert index_k.shape == (self.batch, self.seq_len_kv, self.group_kv, self.index_dim), (
            f"index_k shape mismatch: {index_k.shape} "
            f"!= ({self.batch}, {self.seq_len_kv}, {self.group_kv}, {self.index_dim})"
        )

        assert weights.shape == (self.seq_len, self.heads), (
            f"weights shape mismatch: {weights.shape} != ({self.seq_len}, {self.heads})"
        )

        expected_query_shape = (self.batch, self.seq_len, self.heads,
                                self.dim + self.dim_tail)
        assert query.shape == expected_query_shape, (
            f"query shape mismatch: {query.shape} "
            f"!= {expected_query_shape}"
        )

        expected_kv_shape = (self.batch, self.seq_len_kv, self.group_kv,
                             self.dim + self.dim_tail)
        assert kv_cache.shape == expected_kv_shape, (
            f"kv_cache shape mismatch: {kv_cache.shape} != {expected_kv_shape}"
        )

        return FusedDSACtx.apply(
            index_q,
            index_k,
            weights,
            cu_seqlen_ks,
            cu_seqlen_ke,
            starts,
            ends,
            query,
            kv_cache,
            self.quant_op,
            self.indexer_op,
            self.topk_op,
            self.dsa_op,
        )
