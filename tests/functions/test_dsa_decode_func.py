import pytest
import torch

from tests.test_base import TestBase, FixtureBase
from tileops.func import DeepSeekDSAFusedFunc


class DeepSeekDSAFusedFixture(FixtureBase):
    PARAMS = [
        ("batch, seq_len, seq_len_kv, heads, dim, index_dim, topk, dim_tail, stride_kv, "
         "group_kv, q_start_index_s, quant_in_dtype, clean_logits, in_dtype, out_dtype, "
         "sm_scale, is_causal, dsa_dtype, tune", [
            #  (1, 512, 1024, 64, 256, 32, 128, 32, 1, 1, 512, torch.float16, True, "float16",
            #   "int32", None, True, torch.float16, False),
              (1, 1024, 2048, 128, 512, 64, 2048, 64, 1, 1, 1024, torch.float16,
          True, "float16", "int32", None, True, torch.float16, False),
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
            self.dim + self.dim_tail,
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
        # Baseline implementation for batch=1, group_kv=1.
        # Align shapes with einsum "mhd,nd->hmn".
        b_q, m, h, d = index_q.shape
        b_k, n, gk, d_k = index_k.shape
        # assert b_q == 1 and b_k == 1 and gk == 1, "ref_program currently assumes batch=1, group_kv=1"
        assert d_k == d, "index_q and index_k last dims must match"

        index_q = index_q[0].float()          # (m, h, d)
        k = index_k[0, :, 0, :].float()       # (n, d)

        seq_len_kv = n
        mask_lo = torch.arange(0, seq_len_kv, device="cuda")[None, :] >= cu_seqlen_ks[:, None]
        mask_hi = torch.arange(0, seq_len_kv, device="cuda")[None, :] < cu_seqlen_ke[:, None]
        mask = mask_lo & mask_hi

        score = torch.einsum("mhd,nd->hmn", index_q, k)
        indexer_weights = weights
        logits = (score.relu() * indexer_weights.unsqueeze(-1).transpose(0, 1)).sum(dim=0)
        logits = logits.masked_fill(~mask, float("-inf"))
        indices = torch.topk(logits, self.topk, dim=-1)[1]  # (seq_len, topk)

        query = query.float()
        kv_cache = kv_cache.float()
        # Reshape to match (b, g_index, seq_len, topk) for scatter.
        indices = indices.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, topk)
        b, sq, h, dim_q = query.shape
        b, sk, g, _ = kv_cache.shape
        q_start_index_s = self.q_start_index_s
        if self.q_start_index_s is None:
            q_start_index_s = sk * self.stride_kv - sq

        assert kv_cache.shape[-1] == self.dim + self.dim_tail, 'you should assign dim otherwise'
        dim = self.dim
        k = kv_cache
        v = kv_cache[..., :dim]

        b, _, _, dim_v = v.shape
        g_index = g
        h_index = h // g
        compressed_causal_mask = torch.arange(
            q_start_index_s, sq + q_start_index_s, dtype=torch.int32,
            device="cuda").view(-1, 1) >= torch.arange(
                self.stride_kv - 1,
                sk * self.stride_kv,
                self.stride_kv,
                dtype=torch.int32,
                device="cuda").view(1, -1)

        mask = query.new_zeros(
            b, g_index, sq, sk + 1, dtype=torch.bool).scatter(3, indices.long(), 1)
        mask = mask[..., :-1]
        mask = mask & compressed_causal_mask.view(1, 1, sq, sk)
        mask[:, :, :self.stride_kv - 1, 0] = True
        mask = mask.view(b, g_index, 1, sq, sk)

        query = query.view(b, sq, g, -1, dim_q)
        score = torch.einsum("bmghd,bngd->bghmn", query, k)
        sm_scale = dim_q**-0.5 if self.sm_scale is None else self.sm_scale
        score = score.masked_fill(~mask, float("-inf")).mul(sm_scale)
        p = score.softmax(dim=-1)
        p = p.view(b, g_index, h_index, -1, sq, sk)
        p = p.view(b, g, -1, sq, sk)
        o = torch.einsum("bghmn,bngd->bmghd", p.type(v.dtype), v)
        o = o.reshape(b, sq, h, dim_v)
        return o.to(torch.float16)


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
