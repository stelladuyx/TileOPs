import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.kernels.mamba.ssd_bmm_chunk_bwd import SsdBmmChunkBwdKernel


def ssd_bmm_chunk_bwd_ref(
    a:               torch.Tensor,        # [B, S, G, K]   seqlen-fused; S = C * chunk_size
    d_out:            torch.Tensor,        # [B, C, G, R, M]  R = M = chunk_size
    valid_chunk_len: torch.Tensor,        # [B, C]          int32
    residual:        torch.Tensor | None, # [B, S, G, K]   optional, seqlen-fused
) -> torch.Tensor:
    """
    Math-spec torch reference for ssd_bmm_chunk_bwd.

    Unified contract:
        out[b,c,m,g,k] = sum_r d_out[b,c,g,r,m] * a[b,c,r,g,k] + residual[b,c,m,g,k]

    where:
        a is first reshaped from [B,S,G,K] -> [B,C,R,G,K]
        out is finally reshaped back to [B,S,G,K]

    valid_chunk_len masks both:
        - r axis (reduction tokens)
        - m axis (output tokens)
    """
    B, S, G, K = a.shape
    _, C, G2, R, M = d_out.shape
    assert G2 == G
    assert R == M
    chunk_size = R
    assert C * chunk_size == S, f"S={S} must equal C*chunk_size={C * chunk_size}"

    a_f = a.float()
    d_out_f = d_out.float()
    residual_f = residual.float() if residual is not None else None

    # --------------------------------------------------------
    # reshape a / residual into chunk-local view
    #   a_chunked: [B, C, R, G, K]
    # --------------------------------------------------------
    a_chunked = a_f.view(B, C, chunk_size, G, K)

    if residual_f is not None:
        residual_chunked = residual_f.view(B, C, chunk_size, G, K)
    else:
        residual_chunked = None

    # --------------------------------------------------------
    # build validity masks
    # valid[b,c,t] = (t < valid_chunk_len[b,c])
    # --------------------------------------------------------
    t = torch.arange(chunk_size, device=a.device)
    valid = t[None, None, :] < valid_chunk_len[..., None]   # [B, C, R]

    # r-mask for reduction axis
    valid_r = valid[:, :, None, :, None]    # [B, C, 1, R, 1]
    # m-mask for output axis
    valid_m = valid[:, :, None, None, :]    # [B, C, 1, 1, M]

    # mask d_out on both r and m axes
    d_out_masked = torch.where(
        valid_r & valid_m,
        d_out_f,
        torch.zeros_like(d_out_f),
    )  # [B, C, G, R, M]

    # --------------------------------------------------------
    # math-spec contraction:
    #   out[b,c,m,g,k] = sum_r d_out[b,c,g,r,m] * a[b,c,r,g,k]
    # --------------------------------------------------------
    out_chunked = torch.einsum(
        "bcgrm,bcrgk->bcmgk",
        d_out_masked,
        a_chunked,
    )  # [B, C, M, G, K]

    # add residual on valid output positions only
    if residual_chunked is not None:
        out_chunked = out_chunked + torch.where(
            valid[:, :, :, None, None],
            residual_chunked,
            torch.zeros_like(residual_chunked),
        )

    # zero invalid output positions explicitly
    out_chunked = torch.where(
        valid[:, :, :, None, None],
        out_chunked,
        torch.zeros_like(out_chunked),
    )

    # reshape back to [B, S, G, K]
    out = out_chunked.reshape(B, S, G, K)

    return out


# ──────────────────────────────────────────────────────────────────────────────
# K8 specialisation: dCB -> dB
#
#   dB[s, n] = sum_l  dCB[l, s] * C[l, n]  +  residual_dB[s, n]
#
#   Mapping to unified template:
#     r    = l   (dim=-2 of dCB, the row / l axis)
#     m    = s   (dim=-1 of dCB, the col / s axis)
#     k    = n
#     a    = C             [B, S, G, N]
#     d_out = dCB           [B, C, G, L, S]   already (r=L, m=S) — no transpose
#     out  = dB            [B, S, G, N]
# ──────────────────────────────────────────────────────────────────────────────
def ssd_bmm_chunk_bwd_K8_ref(
    C:               torch.Tensor,              # [B, S, G, N]  seqlen-fused
    dCB:             torch.Tensor,              # [B, C, G, L, S]
    valid_chunk_len: torch.Tensor,              # [B, C]  int32
    residual_dB:     torch.Tensor | None = None,  # [B, S, G, N]
) -> torch.Tensor:
    """dB[s,n] = sum_l dCB[l,s] * C[l,n]  +  residual_dB[s,n]"""
    return ssd_bmm_chunk_bwd_ref(
        a               = C,
        d_out            = dCB,       # dim=-2 = l = r,  dim=-1 = s = m
        valid_chunk_len = valid_chunk_len,
        residual        = residual_dB,
    )


# ──────────────────────────────────────────────────────────────────────────────
# K9 specialisation: dCB^T -> dC
#
#   dC[l, n] = sum_s  dCB[l, s] * B[s, n]  +  residual_dC[l, n]
#
#   Mapping to unified template:
#     r    = s   (the col / s axis of dCB, which becomes dim=-2 after transpose)
#     m    = l   (the row / l axis of dCB, which becomes dim=-1 after transpose)
#     k    = n
#     a    = B             [B, S, G, N]
#     d_out = dCB^T         [B, C, G, S, L]   (r=S, m=L)
#     out  = dC            [B, S, G, N]
# ──────────────────────────────────────────────────────────────────────────────
def ssd_bmm_chunk_bwd_K9_ref(
    B_mat:           torch.Tensor,              # [B, S, G, N]  seqlen-fused
    dCB:             torch.Tensor,              # [B, C, G, L, S]
    valid_chunk_len: torch.Tensor,              # [B, C]  int32
    residual_dC:     torch.Tensor | None = None,  # [B, S, G, N]
) -> torch.Tensor:
    """dC[l,n] = sum_s dCB[l,s] * B[s,n]  +  residual_dC[l,n]"""
    dCB_t = dCB.transpose(-1, -2)  # [B, C, G, S, L]  =>  dim=-2=S=r, dim=-1=L=m
    return ssd_bmm_chunk_bwd_ref(
        a               = B_mat,
        d_out            = dCB_t,
        valid_chunk_len = valid_chunk_len,
        residual        = residual_dC,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Test fixtures
# ──────────────────────────────────────────────────────────────────────────────

class SsdBmmChunkBwdFixture(FixtureBase):
    PARAMS = [
        ("batch, num_chunks, chunk_len, n_groups, d_state, dtype, chunk_len_valid", [
            # smoke: minimal shape, float16, complete chunks
            pytest.param(1, 2, 64, 1, 32, torch.float16, None,
                         marks=pytest.mark.smoke),
            # full: multi-group
            pytest.param(2, 4, 64, 2, 64, torch.float16, None,
                         marks=pytest.mark.full),
            # full: bfloat16
            pytest.param(1, 2, 64, 1, 32, torch.bfloat16, None,
                         marks=pytest.mark.full),
            # full: multi-group bfloat16
            pytest.param(2, 2, 64, 2, 32, torch.bfloat16, None,
                         marks=pytest.mark.full),
            # full: partial chunk — valid_chunk_len < chunk_len
            pytest.param(1, 2, 64, 1, 32, torch.float16, 33,
                         marks=pytest.mark.full),
            pytest.param(1, 2, 64, 1, 32, torch.float16, 1,
                         marks=pytest.mark.full),
            pytest.param(1, 2, 64, 1, 32, torch.float16, 63,
                         marks=pytest.mark.full),
            # full: chunk_len not a power-of-two multiple (tile-tail stress)
            pytest.param(1, 2, 48, 1, 32, torch.float16, None,
                         marks=pytest.mark.full),
            pytest.param(1, 2, 48, 1, 32, torch.float16, 31,
                         marks=pytest.mark.full),
        ]),
    ]


class SsdBmmChunkBwdTest(TestBase):
    def __init__(
        self,
        batch:           int,
        num_chunks:      int,
        chunk_len:       int,
        n_groups:        int,
        d_state:         int,
        dtype:           torch.dtype,
        chunk_len_valid: int | None,
    ):
        self.batch           = batch
        self.num_chunks      = num_chunks
        self.chunk_len       = chunk_len
        self.n_groups        = n_groups
        self.d_state         = d_state
        self.dtype           = dtype
        # None → all positions valid (complete chunk)
        self.chunk_len_valid = chunk_len if chunk_len_valid is None else chunk_len_valid

    def gen_inputs(self):
        B, C, L, G, K = (
            self.batch, self.num_chunks, self.chunk_len,
            self.n_groups, self.d_state,
        )
        S = C * L

        # a: [B, S, G, K]  seqlen-fused
        a = torch.randn(B, S, G, K, dtype=self.dtype, device="cuda") * 0.1
        # d_out: [B, C, G, R, M]  chunk-local pairwise; R = M = chunk_size
        d_out = torch.randn(B, C, G, L, L, dtype=self.dtype, device="cuda") * 0.1
        # valid_chunk_len: [B, C]  — same value for every (b, c) pair for simplicity
        valid_chunk_len = torch.full(
            (B, C), self.chunk_len_valid, dtype=torch.int32, device="cuda",
        )
        # residual: [B, S, G, K]  — always provided to exercise that path
        residual = torch.randn(B, S, G, K, dtype=self.dtype, device="cuda") * 0.1

        return a, d_out, valid_chunk_len, residual

    def ref_program(self, a, d_out, valid_chunk_len, residual):
        return ssd_bmm_chunk_bwd_ref(a, d_out, valid_chunk_len, residual)


@SsdBmmChunkBwdFixture
def test_ssd_bmm_chunk_bwd(
    batch, num_chunks, chunk_len, n_groups, d_state, dtype, chunk_len_valid,
):
    test = SsdBmmChunkBwdTest(
        batch, num_chunks, chunk_len, n_groups, d_state, dtype, chunk_len_valid,
    )
    inputs = test.gen_inputs()
    atol = 1e-2 if dtype == torch.float16 else 2e-2
    rtol = 1e-5

    # Cross-check: K8 and K9 against the unified reference on identical data.
    a, d_out, valid_chunk_len_t, residual = inputs
    B, C, G, L, _ = d_out.shape

    # --- K8 cross-check ---
    # C_mat plays the role of `a`,  d_out plays the role of dCB [B,C,G,L,S].
    # The unified ref and K8 wrapper must agree exactly (same code path).
    out_unified = ssd_bmm_chunk_bwd_ref(a, d_out, valid_chunk_len_t, residual)
    out_K8 = ssd_bmm_chunk_bwd_K8_ref(
        C               = a,
        dCB             = d_out,
        valid_chunk_len = valid_chunk_len_t,
        residual_dB     = residual,
    )
    torch.testing.assert_close(out_K8, out_unified, atol=0.0, rtol=0.0,
                                msg="K8 wrapper must be identical to unified ref")

    # --- K9 cross-check ---
    # dCB_for_K9[b,c,g,l,s] represents the same matrix as d_out but
    # the caller transposes it before passing; verify K9(dCB^T) == unified(dCB_t).
    dCB_for_K9 = d_out.transpose(-1, -2).contiguous()   # [B,C,G,S,L] -> pass as dCB
    out_K9 = ssd_bmm_chunk_bwd_K9_ref(
        B_mat           = a,
        dCB             = dCB_for_K9,
        valid_chunk_len = valid_chunk_len_t,
        residual_dC     = residual,
    )
    # K9 internally does dCB_for_K9.transpose(-1,-2) which recovers the original
    # d_out, so the result must match the unified ref to floating-point precision.
    torch.testing.assert_close(out_K9, out_unified, atol=1e-6, rtol=0.0,
                                msg="K9 wrapper must match unified ref")

    # --- Shape and finiteness ---
    S = C * L
    assert out_unified.shape == (batch, S, n_groups, d_state)
    assert out_unified.isfinite().all(), "reference produced non-finite values"

    # --- valid_chunk_len masking: positions beyond vcl must remain zero ---
    vcl = test.chunk_len_valid
    if vcl < chunk_len:
        # Build a version with no residual so zeros are meaningful.
        out_no_res = ssd_bmm_chunk_bwd_ref(a, d_out, valid_chunk_len_t, residual=None)
        for b in range(batch):
            for c in range(num_chunks):
                t_start = c * chunk_len
                t_vcl   = t_start + vcl
                t_end   = t_start + chunk_len
                # Positions [t_vcl, t_end) must be all-zero.
                tail = out_no_res[b, t_vcl:t_end, :, :]
                assert tail.abs().max().item() == 0.0, (
                    f"positions beyond valid_chunk_len={vcl} must be zero; "
                    f"got max={tail.abs().max().item()}"
                )

    # --- Kernel correctness: compare TileLang kernel output against reference ---
    op = SsdBmmChunkBwdKernel(
        batch, num_chunks, chunk_len, n_groups, d_state, dtype,
    )
    test.check(op, *inputs, atol=atol, rtol=rtol)
