import pytest
import torch

from tests.test_base import TestBase, FixtureBase
from tileops.func import DeepSeekDSAFusedFunc
from tileops.ops import (DeepSeekSparseAttentionDecodeWithKVCacheOp,
                         Fp8LightingIndexerOp, Fp8QuantOp, TopkSelectorOp)
from tileops.utils import str2dtype


class DeepSeekDSAFusedFixture(FixtureBase):
    PARAMS = [
        ("batch, seq_len, seq_len_kv, heads, dim, index_dim, topk, dim_tail, stride_kv, "
         "group_kv, q_start_index_s, quant_in_dtype, clean_logits, in_dtype, out_dtype, "
         "sm_scale, is_causal, dsa_dtype, tune", [
             (1, 512, 1024, 64, 256, 32, 128, 32, 1, 1, 512, torch.float16, True, "float16",
              "int32", None, True, torch.float16, False),
         ]),
    ]


class DeepSeekDSAFusedTest(TestBase):

    def __init__(self,
                 batch: int,
                 seq_len: int,
                 seq_len_kv: int,
                 heads: int,
                 dim: int,
                 index_dim: int,
                 topk: int,
                 dim_tail: int,
                 stride_kv: int,
                 group_kv: int,
                 q_start_index_s: int,
                 quant_in_dtype: torch.dtype,
                 clean_logits: bool,
                 in_dtype: str,
                 out_dtype: str,
                 sm_scale,
                 is_causal: bool,
                 dsa_dtype: torch.dtype,
                 tune: bool) -> None:
        self.batch = batch
        self.seq_len = seq_len
        self.seq_len_kv = seq_len_kv
        self.heads = heads
        self.dim = dim
        self.index_dim = index_dim
        self.topk = topk
        self.dim_tail = dim_tail
        self.stride_kv = stride_kv
        self.group_kv = group_kv
        self.q_start_index_s = q_start_index_s
        self.quant_in_dtype = quant_in_dtype
        self.clean_logits = clean_logits
        self.in_dtype = in_dtype
        self.out_dtype = out_dtype
        self.sm_scale = sm_scale
        self.is_causal = is_causal
        self.dsa_dtype = dsa_dtype
        self.tune = tune

    def gen_inputs(self):
        device = "cuda"
        # Generate in a supported dtype, then cast to float8_e4m3fn for index_q
        index_q = torch.randn(
            self.batch,
            self.seq_len,
            self.heads,
            self.index_dim,
            dtype=self.quant_in_dtype,
            device=device,
        ).to(torch.float8_e4m3fn)
        index_k = torch.randn(
            self.batch,
            self.seq_len_kv,
            self.group_kv,
            self.index_dim,
            dtype=self.quant_in_dtype,
            device=device,
        )
        weights = torch.randn(self.seq_len, self.heads, dtype=torch.float32, device=device)
        cu_seqlen_ks = torch.zeros(self.seq_len, dtype=torch.int32, device=device)
        cu_seqlen_ke = torch.full((self.seq_len,),
                                  self.seq_len_kv,
                                  dtype=torch.int32,
                                  device=device)
        starts = torch.zeros(self.batch, self.seq_len, dtype=torch.int32, device=device)
        ends = torch.full(
            (self.batch, self.seq_len),
            self.seq_len_kv,
            dtype=torch.int32,
            device=device,
        )
        query = torch.randn(
            self.batch,
            self.seq_len,
            self.heads,
            self.dim,
            dtype=self.dsa_dtype,
            device=device,
        )
        kv_cache = torch.randn(
            self.batch,
            self.seq_len_kv,
            self.group_kv,
            self.dim + self.dim_tail,
            dtype=self.dsa_dtype,
            device=device,
        )
        return index_q, index_k, weights, cu_seqlen_ks, cu_seqlen_ke, starts, ends, query, kv_cache

    def ref_program(self,
                    index_q: torch.Tensor,
                    index_k: torch.Tensor,
                    weights: torch.Tensor,
                    cu_seqlen_ks: torch.Tensor,
                    cu_seqlen_ke: torch.Tensor,
                    starts: torch.Tensor,
                    ends: torch.Tensor,
                    query: torch.Tensor,
                    kv_cache: torch.Tensor) -> torch.Tensor:
        # Baseline: compose the individual ops used by the fused function.
        quant_op = Fp8QuantOp(
            batch=self.batch,
            seq_len_kv=self.seq_len_kv,
            kv_group=self.group_kv,
            index_dim=self.index_dim,
            in_dtype=self.quant_in_dtype,
            tune=self.tune,
        )
        index_k_scale, index_k_fp8 = quant_op(index_k)

        indexer_op = Fp8LightingIndexerOp(
            batch=self.batch,
            seq_len=self.seq_len,
            heads=self.heads,
            index_dim=self.index_dim,
            seq_len_kv=self.seq_len_kv,
            kv_group=self.group_kv,
            clean_logits=self.clean_logits,
            tune=self.tune,
        )
        index_scores = indexer_op(index_q, index_k_fp8, index_k_scale, weights, cu_seqlen_ks,
                                  cu_seqlen_ke)

        topk_op = TopkSelectorOp(
            batch=self.batch,
            seq_len=self.seq_len,
            seq_len_kv=self.seq_len_kv,
            kv_group=self.group_kv,
            topk=self.topk,
            in_dtype=torch.float32,
            out_dtype=str2dtype[self.out_dtype],
            tune=self.tune,
        )
        indices = topk_op(index_scores, starts, ends)

        dsa_op = DeepSeekSparseAttentionDecodeWithKVCacheOp(
            batch=self.batch,
            heads=self.heads,
            seq_len=self.seq_len,
            seq_len_kv=self.seq_len_kv,
            dim=self.dim,
            dim_tail=self.dim_tail,
            topk=self.topk,
            stride_kv=self.stride_kv,
            heads_kv=self.group_kv,
            q_start_index_s=self.q_start_index_s,
            sm_scale=self.sm_scale,
            is_causal=self.is_causal,
            dtype=self.dsa_dtype,
            tune=self.tune,
        )
        return dsa_op(query, kv_cache, indices)


@DeepSeekDSAFusedFixture
def test_deepseek_dsa_fused(batch: int,
                            seq_len: int,
                            seq_len_kv: int,
                            heads: int,
                            dim: int,
                            index_dim: int,
                            topk: int,
                            dim_tail: int,
                            stride_kv: int,
                            group_kv: int,
                            q_start_index_s: int,
                            quant_in_dtype: torch.dtype,
                            clean_logits: bool,
                            in_dtype: str,
                            out_dtype: str,
                            sm_scale,
                            is_causal: bool,
                            dsa_dtype: torch.dtype,
                            tune: bool) -> None:
    """Compare fused function against explicit composition of ops."""
    test = DeepSeekDSAFusedTest(
        batch,
        seq_len,
        seq_len_kv,
        heads,
        dim,
        index_dim,
        topk,
        dim_tail,
        stride_kv,
        group_kv,
        q_start_index_s,
        quant_in_dtype,
        clean_logits,
        in_dtype,
        out_dtype,
        sm_scale,
        is_causal,
        dsa_dtype,
        tune,
    )

    fused = DeepSeekDSAFusedFunc(
        seq_len_kv=seq_len_kv,
        index_dim=index_dim,
        seq_len=seq_len,
        heads=heads,
        batch=batch,
        topk=topk,
        dim=dim,
        dim_tail=dim_tail,
        stride_kv=stride_kv,
        group_kv=group_kv,
        q_start_index_s=q_start_index_s,
        quant_in_dtype=quant_in_dtype,
        clean_logits=clean_logits,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        sm_scale=sm_scale,
        is_causal=is_causal,
        dsa_dtype=dsa_dtype,
        tune=tune,
    )

    test.check(fused, *test.gen_inputs(), atol=3e-4, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
