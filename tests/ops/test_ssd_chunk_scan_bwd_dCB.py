import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.kernels.mamba.ssd_chunk_scan_bwd_dBC import SsdChunkScanBwdDCBKernel


def ssd_chunk_scan_bwd_dCB_ref(
    x: torch.Tensor,               # [B, S, H, P]  input dtype    S = C*L seqlen-fused
    dt: torch.Tensor,              # [B, H, C, L]  input dtype
    dA_cumsum: torch.Tensor,       # [B, H, C, L]  float32
    dout: torch.Tensor,            # [B, S, H, P]  input dtype    S = C*L seqlen-fused
    valid_chunk_len: torch.Tensor, # [B, C]        int32
    n_groups: int,
) -> torch.Tensor:
    """
    Math-spec reference for _chunk_scan_bwd_dcb.

    Corresponds to: ssd_chunk_scan._chunk_scan_bwd_dcb kernel
    Official file:  mamba_ssm/ops/triton/ssd_chunk_scan.py

    The forward pass local/diagonal path contributes to output via:
        y_local[b, c*L+l, h, p] = sum_{s<=l}
            CB[b,c,g(h),l,s] * exp(min(dA[l] - dA[s], 0)) * dt[b,h,c,s] * x[b,c*L+s,h,p]

    Differentiating wrt CB:
        dCB[b, c, g, l, s] = sum_{h in group g} sum_p
            dout[b,c*L+l,h,p] * x[b,c*L+s,h,p]
            * exp(min(dA_cumsum[b,h,c,l] - dA_cumsum[b,h,c,s], 0))
            * dt[b,h,c,s]

    Decomposed as:
        pair_base[b,c,h,l,s] = sum_p dout[b,c,l,h,p] * x[b,c,s,h,p]
        dCB_h[b,c,h,l,s]     = pair_base * exp(min(dA[l] - dA[s], 0)) * dt[b,h,c,s]
        dCB[b,c,g,l,s]       = sum_{h in group g} dCB_h[b,c,h,l,s]

    Official contract notes:
    - The kernel computes the lower-triangle mask (l >= s) and zeros upper triangle.
    - Decay is exp(min(dA_cs_m - dA_cs_n, 0)) — the stable-decay convention matching
      all other SSD backward kernels.
    - valid_chunk_len gates which positions (l, s) contribute.
    """
    B, S, H, P = x.shape
    _, H_, C, L = dA_cumsum.shape
    assert H == H_ and S == C * L
    heads_per_group = H // n_groups

    # --------------------------------------------------------
    # All arithmetic in float32; reshape seqlen-fused -> chunked
    # --------------------------------------------------------
    x_f    = x.float().reshape(B, C, L, H, P).permute(0, 1, 3, 2, 4)     # [B, C, H, L, P]
    dout_f = dout.float().reshape(B, C, L, H, P).permute(0, 1, 3, 2, 4)  # [B, C, H, L, P]
    # [B, H, C, L] -> [B, C, H, L]
    a  = dA_cumsum.float().permute(0, 2, 1, 3)   # [B, C, H, L]
    dt_f = dt.float().permute(0, 2, 1, 3)        # [B, C, H, L]

    # --------------------------------------------------------
    # Pairwise inner product over head dimension P:
    #   pair_base[b,c,h,l,s] = sum_p dout[l,p] * x[s,p]
    # Official kernel: dcb = tl.dot(dout, x)   [l, p] x [p, s] -> [l, s]
    # --------------------------------------------------------
    pair_base = torch.einsum("bchlp,bchsp->bchls", dout_f, x_f)  # [B, C, H, L, L]

    # --------------------------------------------------------
    # Stable decay: exp(min(dA_cumsum[l] - dA_cumsum[s], 0))
    # Official kernel: dcb *= tl.exp(tl.minimum(dA_cs_m - dA_cs_n, 0.0))
    # --------------------------------------------------------
    a_l = a.unsqueeze(-1)   # [B, C, H, L, 1]
    a_s = a.unsqueeze(-2)   # [B, C, H, 1, L]
    decay = torch.exp(torch.clamp(a_l - a_s, max=0.0))  # [B, C, H, L, L]

    # dt lives on source position s (official kernel: dt_n * dcb)
    dt_s = dt_f.unsqueeze(-2)  # [B, C, H, 1, L]  broadcast over l

    dCB_h = pair_base * decay * dt_s   # [B, C, H, L, L]

    # --------------------------------------------------------
    # Causal lower-triangle mask: l >= s  (official kernel: mask = offs_m >= offs_n)
    # --------------------------------------------------------
    l_idx = torch.arange(L, device=x.device)
    s_idx = torch.arange(L, device=x.device)
    causal_mask = (l_idx[:, None] >= s_idx[None, :])  # [L, L]
    dCB_h = dCB_h * causal_mask[None, None, None, :, :]  # broadcast over [B, C, H]

    # --------------------------------------------------------
    # Valid-length mask: position t in chunk c valid iff t < valid_chunk_len[b, c]
    # Both l and s must be valid.
    # Official kernel: chunk_size_limit_n = min(chunk_size_limit, (pid_m+1)*BLOCK_SIZE_M)
    # --------------------------------------------------------
    vcl = valid_chunk_len.int()                              # [B, C]
    l_mask = (l_idx[None, None, :, None] < vcl[:, :, None, None])   # [B, C, L, 1]
    s_mask = (s_idx[None, None, None, :] < vcl[:, :, None, None])   # [B, C, 1, L]
    valid_mask = (l_mask & s_mask).unsqueeze(2)              # [B, C, 1, L, L] for head broadcast
    dCB_h = dCB_h * valid_mask.float()                       # [B, C, H, L, L]

    # --------------------------------------------------------
    # Accumulate over heads into groups:
    #   dCB[b,c,g,l,s] = sum_{h: h//hpg == g} dCB_h[b,c,h,l,s]
    # --------------------------------------------------------
    h_idx   = torch.arange(H, device=x.device)
    g_idx   = h_idx // heads_per_group                       # [H]
    dCB = torch.zeros(B, C, n_groups, L, L, dtype=torch.float32, device=x.device)
    dCB.scatter_add_(
        dim=2,
        index=g_idx[None, None, :, None, None].expand(B, C, H, L, L),
        src=dCB_h,
    )

    return dCB


class SsdChunkScanBwdDCBFixture(FixtureBase):
    PARAMS = [
        ("batch, num_chunks, chunk_len, n_heads, d_head, n_groups, dtype, chunk_len_valid, tune", [
            # smoke: minimal shape, float16, no tune
            pytest.param(1, 2, 64, 4, 64, 1, torch.float16, None, False, marks=pytest.mark.smoke),
            # full: multi-group, larger batch
            pytest.param(2, 4, 64, 8, 64, 2, torch.float16, None, False, marks=pytest.mark.full),
            # full: bfloat16
            pytest.param(1, 2, 64, 4, 64, 1, torch.bfloat16, None, False, marks=pytest.mark.full),
            # full: multi-group bfloat16
            pytest.param(2, 2, 64, 4, 64, 2, torch.bfloat16, None, False, marks=pytest.mark.full),
            # full: partial chunk (valid_chunk_len < chunk_len)
            pytest.param(1, 2, 64, 4, 64, 1, torch.float16, 33, False, marks=pytest.mark.full),
            pytest.param(1, 2, 64, 4, 64, 1, torch.float16, 1, False, marks=pytest.mark.full),
            pytest.param(1, 2, 64, 4, 64, 1, torch.float16, 63, False, marks=pytest.mark.full),
            # full: chunk_len not a multiple of block_l
            pytest.param(1, 2, 48, 4, 64, 1, torch.float16, None, False, marks=pytest.mark.full),
            pytest.param(1, 2, 48, 4, 64, 1, torch.float16, 31, False, marks=pytest.mark.full),
            # full: tune
            pytest.param(1, 2, 64, 4, 64, 1, torch.float16, None, True, marks=pytest.mark.full),
        ]),
    ]


class SsdChunkScanBwdDCBTest(TestBase):
    def __init__(
        self,
        batch: int,
        num_chunks: int,
        chunk_len: int,
        n_heads: int,
        d_head: int,
        n_groups: int,
        dtype: torch.dtype,
        chunk_len_valid: int | None,
    ):
        self.batch = batch
        self.num_chunks = num_chunks
        self.chunk_len = chunk_len
        self.n_heads = n_heads
        self.d_head = d_head
        self.n_groups = n_groups
        self.dtype = dtype
        # None means all positions valid (complete chunk)
        self.chunk_len_valid = chunk_len if chunk_len_valid is None else chunk_len_valid

    def gen_inputs(self):
        B, C, L, H, P, G = (
            self.batch, self.num_chunks, self.chunk_len,
            self.n_heads, self.d_head, self.n_groups,
        )
        S = C * L
        # x: [B, S, H, P]  seqlen-fused
        x = torch.randn(B, S, H, P, dtype=self.dtype, device="cuda") * 0.1
        # dout: [B, S, H, P]  seqlen-fused
        dout = torch.randn(B, S, H, P, dtype=self.dtype, device="cuda") * 0.1
        # dA_cumsum: [B, H, C, L]  non-positive, monotone decreasing cumsum
        dA_cumsum = -torch.rand(B, H, C, L, dtype=torch.float32, device="cuda").cumsum(-1)
        # dt: [B, H, C, L]
        dt = torch.rand(B, H, C, L, dtype=self.dtype, device="cuda") * 0.1 + 0.01
        # valid_chunk_len: [B, C]
        valid_chunk_len = torch.full(
            (B, C), self.chunk_len_valid, dtype=torch.int32, device="cuda",
        )
        return x, dt, dA_cumsum, dout, valid_chunk_len

    def ref_program(self, x, dt, dA_cumsum, dout, valid_chunk_len):
        return ssd_chunk_scan_bwd_dCB_ref(
            x, dt, dA_cumsum, dout, valid_chunk_len, self.n_groups,
        )


@SsdChunkScanBwdDCBFixture
def test_ssd_chunk_scan_bwd_dCB(
    batch, num_chunks, chunk_len, n_heads, d_head, n_groups, dtype, chunk_len_valid, tune,
):
    test = SsdChunkScanBwdDCBTest(
        batch, num_chunks, chunk_len, n_heads, d_head, n_groups, dtype,
        chunk_len_valid,
    )
    op = SsdChunkScanBwdDCBKernel(
        batch, num_chunks, chunk_len, n_heads, d_head, n_groups, dtype, tune=tune,
    )
    inputs = test.gen_inputs()
    atol = 1e-2 if dtype == torch.float16 else 2e-2
    rtol = 1e-5
    test.check(op, *inputs, atol=atol, rtol=rtol)
