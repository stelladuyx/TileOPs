import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.kernels.mamba.ssd_chunk_scan_bwd_dstates import SsdChunkScanBwdDstatesKernel


def ssd_chunk_scan_bwd_dstates_ref(
    d_out: torch.Tensor,             # [B, S, H, P]  input dtype    S = C*L seqlen-fused
    C: torch.Tensor,                # [B, S, H, N]  input dtype    S = C*L seqlen-fused
    dA_cumsum: torch.Tensor,        # [B, H, C, L]  float32
    valid_chunk_len: torch.Tensor,  # [B, C]        int32
) -> torch.Tensor:
    """
    Math-spec reference for _chunk_scan_bwd_dstates (K1).

    Corresponds to: ssd_chunk_scan._chunk_scan_bwd_dstates kernel
    Official file:  mamba_ssm/ops/triton/ssd_chunk_scan.py

    The history branch of the forward scan:
        y_prev[b, c*L+l, h, p] = exp(dA_cumsum[b,h,c,l]) * sum_n C[b,c*L+l,h,n] * states[b,c,h,p,n]

    Differentiating wrt states:
        dstates[b, c, h, p, n]
            = sum_{l: valid} exp(dA_cumsum[b,h,c,l]) * d_out[b,c*L+l,h,p] * C[b,c*L+l,h,n]

    Direct einsum after scaling C by exp(dA_cumsum[l]):

        scaled_C[b,c,l,h,n] = C[b,c,l,h,n] * exp(dA_cumsum[b,h,c,l]) * valid[b,c,l]
        dstates[b,c,h,p,n]  = sum_l d_out[b,c,l,h,p] * scaled_C[b,c,l,h,n]

    Official kernel notes:
    - ``C`` here is head-indexed (not group-indexed); this is the raw C tensor passed
      into the combined backward, not the group-contracted CB product.
    - The decay is exp(dA_cumsum[l]) — the direct cumulative-sum value, not the
      interval-relative decay used in K10 / dCB.
    - valid_chunk_len masks positions l >= valid_chunk_len[b,c].

    Returns:
        dstates: [B, C, H, P, N]  float32
    """
    B, S, H, P = d_out.shape
    _, H_, C_sz, L = dA_cumsum.shape
    assert H == H_ and C_sz * L == S
    N = C.shape[-1]

    # --------------------------------------------------------
    # All arithmetic in float32; reshape seqlen-fused -> chunked
    # --------------------------------------------------------
    d_out_f = d_out.float().reshape(B, C_sz, L, H, P)  # [B, C, L, H, P]
    C_f    = C.float().reshape(B, C_sz, L, H, N)     # [B, C, L, H, N]
    # [B, H, C, L] -> [B, C, L, H]
    decay  = torch.exp(dA_cumsum.float().permute(0, 2, 3, 1))  # [B, C, L, H]

    # --------------------------------------------------------
    # Valid-position mask: position l in chunk c valid iff l < valid_chunk_len[b, c]
    # Official kernel: chunk_size_limit = min(chunk_size, seqlen - pid_c*chunk_size)
    # --------------------------------------------------------
    l_idx = torch.arange(L, device=d_out.device)
    valid = (l_idx[None, None, :] < valid_chunk_len[:, :, None])  # [B, C, L]

    # --------------------------------------------------------
    # Scale C by decay and mask invalid positions:
    #   scaled_C[b,c,l,h,n] = C[b,c,l,h,n] * exp(dA_cumsum[b,h,c,l]) * valid[b,c,l]
    # Official kernel: dc = tl.dot(d_out, prev_states); dc *= scale[:, None]
    # --------------------------------------------------------
    scaled_C = C_f * decay.unsqueeze(-1)                           # [B, C, L, H, N]
    scaled_C = scaled_C * valid[:, :, :, None, None]               # zero invalid l

    # --------------------------------------------------------
    # dstates[b,c,h,p,n] = sum_l d_out[b,c,l,h,p] * scaled_C[b,c,l,h,n]
    # Direct einsum translation of: tl.dot(d_out, prev_states) with [L, P] x [P, N] -> [L, N]
    # (here summed over l to produce the full dstates)
    # --------------------------------------------------------
    return torch.einsum("bclhp,bclhn->bchpn", d_out_f, scaled_C)   # [B, C, H, P, N]


class SsdChunkScanBwdDstatesFixture(FixtureBase):
    PARAMS = [
        ("batch, num_chunks, chunk_len, n_heads, d_head, d_state, dtype, chunk_len_valid, tune", [
            # complete chunks (chunk_len_valid == chunk_len)
            pytest.param(1, 2, 64, 4, 64, 32, torch.float16, None, False,
                         marks=pytest.mark.smoke),
            pytest.param(2, 4, 64, 8, 64, 64, torch.float16, None, False,
                         marks=pytest.mark.full),
            pytest.param(1, 2, 128, 4, 128, 32, torch.bfloat16, None, False,
                         marks=pytest.mark.full),
            pytest.param(2, 2, 64, 4, 64, 32, torch.bfloat16, None, False,
                         marks=pytest.mark.full),
            # incomplete final chunk: chunk_len_valid < chunk_len
            pytest.param(1, 2, 64, 4, 64, 32, torch.float16, 33, False,
                         marks=pytest.mark.full),
            pytest.param(1, 2, 64, 4, 64, 32, torch.float16, 1, False,
                         marks=pytest.mark.full),
            pytest.param(1, 2, 64, 4, 64, 32, torch.float16, 63, False,
                         marks=pytest.mark.full),
            # small P / N (tail tile stress)
            pytest.param(1, 2, 64, 4, 16, 16, torch.float16, None, False,
                         marks=pytest.mark.full),
            # chunk_len not a multiple of block_l
            pytest.param(1, 2, 48, 4, 64, 32, torch.float16, None, False,
                         marks=pytest.mark.full),
            pytest.param(1, 2, 48, 4, 64, 32, torch.float16, 31, False,
                         marks=pytest.mark.full),
        ]),
    ]


class SsdChunkScanBwdDstatesTest(TestBase):
    def __init__(
        self,
        batch: int,
        num_chunks: int,
        chunk_len: int,
        n_heads: int,
        d_head: int,
        d_state: int,
        dtype: torch.dtype,
        chunk_len_valid: int | None,
    ):
        self.batch = batch
        self.num_chunks = num_chunks
        self.chunk_len = chunk_len
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_state = d_state
        self.dtype = dtype
        self.chunk_len_valid = chunk_len if chunk_len_valid is None else chunk_len_valid

    def gen_inputs(self):
        b, c, Q, h, p, n = (
            self.batch, self.num_chunks, self.chunk_len,
            self.n_heads, self.d_head, self.d_state,
        )
        S = c * Q
        # d_out: [B, S, H, P]  seqlen-fused
        d_out = torch.randn(b, S, h, p, dtype=self.dtype, device="cuda") * 0.1
        # C_mat: [B, S, H, N]  seqlen-fused
        C = torch.randn(b, S, h, n, dtype=self.dtype, device="cuda") * 0.1
        # dA_cumsum: [B, H, C, L]  non-positive, monotone decreasing cumsum
        dA_cumsum = -torch.rand(b, h, c, Q, dtype=torch.float32, device="cuda").cumsum(-1)
        # valid_chunk_len: [B, C]
        valid_chunk_len = torch.full(
            (b, c), self.chunk_len_valid, dtype=torch.int32, device="cuda",
        )
        return d_out, C, dA_cumsum, valid_chunk_len

    def ref_program(self, d_out, C, dA_cumsum, valid_chunk_len):
        return ssd_chunk_scan_bwd_dstates_ref(
            d_out, C, dA_cumsum, valid_chunk_len,
        )


@SsdChunkScanBwdDstatesFixture
def test_ssd_chunk_scan_bwd_dstates(
    batch, num_chunks, chunk_len, n_heads, d_head, d_state, dtype,
    chunk_len_valid, tune,
):
    test = SsdChunkScanBwdDstatesTest(
        batch, num_chunks, chunk_len, n_heads, d_head, d_state, dtype, chunk_len_valid,
    )
    kernel = SsdChunkScanBwdDstatesKernel(
        batch, num_chunks, chunk_len, n_heads, d_head, d_state, dtype, tune=tune,
    )
    inputs = test.gen_inputs()
    atol = 1e-3 if dtype == torch.float16 else 2e-3
    rtol = 1e-5
    test.check(kernel, *inputs, atol=atol, rtol=rtol)
