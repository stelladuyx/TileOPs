import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.kernels.mamba.ssd_chunk_state_bwd_dB_dApartial import SsdChunkStateBwdDBDApartialKernel


def ssd_chunk_state_bwd_dB_dApartial_ref(
    g_u: torch.Tensor,               # (b, c, h, p, n)  float32  — gradient w.r.t. u_c
    X: torch.Tensor,                 # (b, s, h, p)     input dtype   s = c*q seqlen-fused
    Bmat: torch.Tensor,              # (b, s, g, n)     input dtype   s = c*q seqlen-fused
    chunk_state_decay: torch.Tensor, # (b, h, c, q)     float32
                                     #   = exp(min(A_end - A_l, 0)) * dt[l]
                                     #   where A_end = dA_cumsum[c, last_valid]
    n_groups: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Math-spec reference for _chunk_state_bwd_db (_chunk_state_bwd_db_kernel).

    Corresponds to: ssd_chunk_state._chunk_state_bwd_db_kernel
    Official file:  mamba_ssm/ops/triton/ssd_chunk_state.py

    The chunk-state forward accumulation is:
        state_c[h, p, n] = sum_{l=0}^{L-1}
            exp(min(A_end - A_l, 0)) * dt[l] * X[c*L+l, h, p] * B[c*L+l, g(h), n]

        (where A_end = dA_cumsum[c, last_valid], A_l = dA_cumsum[c, l])

    The ``chunk_state_decay`` input already encodes the full scale factor
        chunk_state_decay[b, h, c, l] = exp(min(A_end - A_l, 0)) * dt[l]
    so no dt or dA_cumsum splitting is needed here.

    Differentiating state_c wrt B[l]:
        dB[b, c*L+l, g, n] = sum_{h in group g} g_u[b,c,h,p,n] * X[b,c*L+l,h,p] * scale[b,h,c,l]

    Differentiating state_c wrt chunk_state_decay (combined ddA_next partial):
        ddA_next[b, h, c, l]
            = (sum_{p,n} g_u[b,c,h,p,n] * X[b,c*L+l,h,p] * B[b,c*L+l,g(h),n]) * scale[b,h,c,l]

    Note: the official kernel writes the ddA gradient shifted by +1 position
    (exclusive-reverse-cumsum step), but this reference produces the per-position
    partial that is then combined upstream.

    Returns:
        dB       : (b, s, g, n)  float32   s = c*q seqlen-fused
        ddA_next : (b, h, c, q)  float32
    """
    b, c, h, p, n = g_u.shape
    q = chunk_state_decay.shape[-1]
    hpg = h // n_groups   # heads_per_group

    # --------------------------------------------------------
    # All arithmetic in float32; reshape seqlen-fused -> chunked
    # --------------------------------------------------------
    g_u_f = g_u.float()                                                     # (b, c, h, p, n)
    # [b, s, h, p] -> [b, c, q, h, p] -> [b, c, h, q, p]
    X_f   = X.float().reshape(b, c, q, h, p).permute(0, 1, 3, 2, 4)        # (b, c, h, q, p)
    # [b, s, g, n] -> [b, c, q, g, n] -> [b, c, g, q, n]
    B_f   = Bmat.float().reshape(b, c, q, n_groups, n).permute(0, 1, 3, 2, 4)  # (b, c, g, q, n)
    # [b, h, c, q] -> [b, c, h, q]
    scale = chunk_state_decay.float().permute(0, 2, 1, 3)                   # (b, c, h, q)

    # --------------------------------------------------------
    # dB per head (before group reduction):
    #
    #   dB_h[b,c,h,l,n] = sum_p g_u[b,c,h,p,n] * X[b,c,h,l,p] * scale[b,c,h,l]
    #
    # Official kernel: db = tl.dot(x, dstates); db *= (scale * dt_m)[:, None]
    # (here scale already includes dt)
    # --------------------------------------------------------
    # weighted_X: weight each X position by its scale factor
    weighted_X = X_f * scale.unsqueeze(-1)  # (b, c, h, q, p)

    # Einsum over p: [b,c,h,p,n] x [b,c,h,q,p] -> [b,c,h,q,n]
    dB_per_head = torch.einsum("bchpn,bchlp->bchln", g_u_f, weighted_X)  # (b, c, h, q, n)

    # --------------------------------------------------------
    # Accumulate dB over heads into groups using scatter_add:
    #   dB[b,c,g,l,n] = sum_{h: group(h)==g} dB_per_head[b,c,h,l,n]
    # --------------------------------------------------------
    h_pos = torch.arange(h, device=X.device)
    g_pos = h_pos // hpg                                           # [H]
    dB_chunked = torch.zeros(b, c, n_groups, q, n, dtype=torch.float32, device=X.device)
    dB_chunked.scatter_add_(
        dim=2,
        index=g_pos[None, None, :, None, None].expand(b, c, h, q, n),
        src=dB_per_head,
    )
    # Reshape to seqlen-fused: (b, s, g, n) where s = c*q
    dB = dB_chunked.permute(0, 1, 3, 2, 4).reshape(b, c * q, n_groups, n)

    # --------------------------------------------------------
    # ddA_next (per-position decay partial):
    #
    #   ddA_next[b,h,c,l] = (sum_{p,n} g_u * X[l] * B[l]) * scale[l]
    #
    # Official kernel (HAS_DDA_CS path in _chunk_state_bwd_db_kernel):
    #   ddA_cs = tl.sum(db * b, axis=1)   -> summed over n
    #   tl.store(ddA_cumsum_ptrs + stride, ddA_cs, ...)
    # --------------------------------------------------------
    # Map B from group-indexed to head-indexed: (b, c, h, q, n)
    B_heads = B_f[:, :, g_pos, :, :]                                  # (b, c, h, q, n)

    # dB_no_scale[b,c,h,l,n] = sum_p g_u[b,c,h,p,n] * X[b,c,h,l,p]
    dB_no_scale = torch.einsum("bchpn,bchlp->bchln", g_u_f, X_f)     # (b, c, h, q, n)

    # scalar inner product with B, then scale
    ddA_partial = (dB_no_scale * B_heads).sum(dim=-1)                  # (b, c, h, q)
    ddA_next    = (ddA_partial * scale).permute(0, 2, 1, 3)            # (b, h, c, q)

    return dB, ddA_next


class SsdChunkStateBwdDBDApartialFixture(FixtureBase):
    PARAMS = [
        ("batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype, tune", [
            pytest.param(1, 2, 64, 4, 64, 32, 1, torch.float16, False, marks=pytest.mark.smoke),
            pytest.param(2, 4, 64, 8, 64, 64, 2, torch.float16, False, marks=pytest.mark.full),
            pytest.param(1, 2, 128, 4, 128, 32, 1, torch.bfloat16, False, marks=pytest.mark.full),
            pytest.param(2, 2, 64, 4, 64, 32, 2, torch.bfloat16, False, marks=pytest.mark.full),
        ]),
    ]


class SsdChunkStateBwdDBDApartialTest(TestBase):
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
    ):
        self.batch = batch
        self.num_chunks = num_chunks
        self.chunk_len = chunk_len
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_state = d_state
        self.n_groups = n_groups
        self.dtype = dtype

    def gen_inputs(self):
        b, c, Q, h, p, n, g = (
            self.batch, self.num_chunks, self.chunk_len,
            self.n_heads, self.d_head, self.d_state, self.n_groups,
        )
        S = c * Q  # seqlen-fused sequence length
        g_u = torch.randn(b, c, h, p, n, dtype=torch.float32, device="cuda") * 0.1
        X = torch.randn(b, S, h, p, dtype=self.dtype, device="cuda") * 0.1
        Bmat = torch.randn(b, S, g, n, dtype=self.dtype, device="cuda") * 0.1
        # decay = exp(A_end - A_l), values in (0, 1]; layout [B, H, C, Q]
        dA_cumsum = -torch.rand(b, h, c, Q, dtype=torch.float32, device="cuda").cumsum(-1)
        dA_end = dA_cumsum[..., -1:]
        chunk_state_decay = torch.exp(torch.clamp(dA_end - dA_cumsum, max=0.0))
        return g_u, X, Bmat, chunk_state_decay

    def ref_program(self, g_u, X, Bmat, chunk_state_decay):
        return ssd_chunk_state_bwd_dB_dApartial_ref(
            g_u, X, Bmat, chunk_state_decay, self.n_groups,
        )


@SsdChunkStateBwdDBDApartialFixture
def test_ssd_chunk_state_bwd_dB_dApartial(
    batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype, tune,
):
    test = SsdChunkStateBwdDBDApartialTest(
        batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype,
    )
    op = SsdChunkStateBwdDBDApartialKernel(
        batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype, tune=tune,
    )
    inputs = test.gen_inputs()
    atol = 1e-3 if dtype == torch.float16 else 2e-3
    rtol = 1e-5
    test.check(op, *inputs, atol=atol, rtol=rtol)
