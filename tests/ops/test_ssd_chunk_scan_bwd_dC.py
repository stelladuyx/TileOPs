import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.kernels.mamba.ssd_chunk_scan_bwd_dC import SsdChunkScanBwdDCKernel


def ssd_chunk_scan_bwd_dC_ref(
    states: torch.Tensor,           # [B, C, H, P, N]  input dtype
    dA_cumsum: torch.Tensor,        # [B, H, C, L]     float32
    d_out: torch.Tensor,             # [B, S, H, P]     input dtype   S = C*L seqlen-fused
    C_in: torch.Tensor,             # [B, S, G, N]     input dtype   S = C*L seqlen-fused
    valid_chunk_len: torch.Tensor,  # [B, C]           int32
    n_groups: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Math-spec reference for _chunk_scan_bwd_dC.

    Corresponds to: ssd_chunk_scan._chunk_scan_bwd_dc_kernel
    Official file:  mamba_ssm/ops/triton/ssd_chunk_scan.py

    The history (previous-state) branch of the forward scan contributes:
        y_prev[b, c*L+l, h, p] = exp(dA_cumsum[b,h,c,l]) * sum_n C[b,c*L+l,g(h),n] * states[b,c,h,p,n]

    Differentiating wrt C (primary output) and dA_cumsum (secondary):

        dc_base[b,c,h,l,n] = sum_p d_out[b,c,l,h,p] * states[b,c,h,p,n]
        dc_acc[b,c,h,l,n]  = exp(dA_cumsum[b,h,c,l]) * dc_base[b,c,h,l,n]

        dC[b,c,g,l,n]      = sum_{h in group g} dc_acc[b,c,h,l,n]
        (reshaped to seqlen-fused: dC_out[b, c*L+l, g, n])

        ddA_cumsum_prev[b,h,c,l] = sum_n dc_acc[b,c,h,l,n] * C[b,c,g(h),l,n]

    Official kernel notes:
    - ``states`` are the states_before_chunk (s_{c-1} in the state-passing sense),
      stored in the kernel as ``prev_states``.
    - ``dA_cumsum`` here is the within-chunk cumulative sum (not chunk-level).
    - The kernel accumulates dc across heads for each group using scatter logic.
    - valid_chunk_len masks positions l beyond the valid chunk length.
    """
    B, C_sz, H, P, N = states.shape
    L = dA_cumsum.shape[-1]
    S = C_sz * L
    G = n_groups
    hpg = H // G   # heads_per_group

    # --------------------------------------------------------
    # All arithmetic in float32; reshape seqlen-fused -> chunked
    # --------------------------------------------------------
    states_f = states.float()                                              # [B, C, H, P, N]
    d_out_f   = d_out.float().reshape(B, C_sz, L, H, P).permute(0,1,3,2,4) # [B, C, H, L, P]
    C_f      = C_in.float().reshape(B, C_sz, L, G, N)                    # [B, C, L, G, N]
    # [B, H, C, L] -> [B, C, H, L]
    dA_f     = dA_cumsum.float().permute(0, 2, 1, 3)                      # [B, C, H, L]

    # --------------------------------------------------------
    # Valid-position mask: position l in chunk c valid iff l < valid_chunk_len[b, c]
    # --------------------------------------------------------
    l_idx  = torch.arange(L, device=states.device)
    valid  = (l_idx[None, None, :] < valid_chunk_len[:, :, None])  # [B, C, L]

    # --------------------------------------------------------
    # dc_base[b,c,h,l,n] = sum_p d_out[b,c,h,l,p] * states[b,c,h,p,n]
    # Direct einsum translation of: dc = tl.dot(d_out, prev_states)
    # --------------------------------------------------------
    dc_base = torch.einsum("bchlp,bchpn->bchln", d_out_f, states_f)   # [B, C, H, L, N]

    # Apply valid mask (zero out l positions beyond valid_chunk_len)
    dc_base = dc_base * valid[:, :, None, :, None]                    # [B, C, H, L, N]

    # --------------------------------------------------------
    # Scale by exp(dA_cumsum[l]):
    #   Official kernel: dc *= scale[:, None]  where scale = exp(dA_cs_m)
    # --------------------------------------------------------
    exp_a  = torch.exp(dA_f).unsqueeze(-1)                            # [B, C, H, L, 1]
    dc_acc = dc_base * exp_a                                           # [B, C, H, L, N]

    # --------------------------------------------------------
    # Primary output: dC[b,c,g,l,n] = sum_{h in group g} dc_acc[b,c,h,l,n]
    # Use scatter_add for a direct group-accumulation without a Python loop.
    # --------------------------------------------------------
    h_pos   = torch.arange(H, device=states.device)
    g_pos   = h_pos // hpg                                             # [H]
    dC_chunked = torch.zeros(B, C_sz, G, L, N, dtype=torch.float32, device=states.device)
    dC_chunked.scatter_add_(
        dim=2,
        index=g_pos[None, None, :, None, None].expand(B, C_sz, H, L, N),
        src=dc_acc,
    )
    # Reshape to seqlen-fused [B, S, G, N]
    dC_out = dC_chunked.permute(0, 1, 3, 2, 4).reshape(B, S, G, N)

    # --------------------------------------------------------
    # Secondary output:
    #   ddA_cumsum_prev[b,h,c,l] = sum_n dc_acc[b,c,h,l,n] * C[b,c,g(h),l,n]
    #
    # Official kernel:  ddA_cs = tl.sum(dc * c, axis=1)
    # Here dc corresponds to dc_acc[b,c,h,l,:] and c is C[b,c,g(h),l,:].
    # --------------------------------------------------------
    # Map C from group-indexed to head-indexed: [B, C, H, L, N]
    C_heads = C_f[:, :, :, g_pos, :].permute(0, 1, 3, 2, 4)           # [B, C, H, L, N]
    ddA_prev = (dc_acc * C_heads).sum(dim=-1)                          # [B, C, H, L]
    # Return in official layout [B, H, C, L]
    ddA_cumsum_prev_out = ddA_prev.permute(0, 2, 1, 3)                 # [B, H, C, L]

    return dC_out, ddA_cumsum_prev_out


class SsdChunkScanBwdDCFixture(FixtureBase):
    PARAMS = [
        ("batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype, chunk_len_valid, tune", [
            # complete chunks (all positions valid)
            pytest.param(1, 2, 64, 4, 64, 32, 1, torch.float16, None, False,
                         marks=pytest.mark.smoke),
            pytest.param(2, 4, 64, 8, 64, 64, 2, torch.float16, None, False,
                         marks=pytest.mark.full),
            pytest.param(1, 2, 128, 4, 128, 32, 1, torch.bfloat16, None, False,
                         marks=pytest.mark.full),
            pytest.param(2, 2, 64, 4, 64, 32, 2, torch.bfloat16, None, False,
                         marks=pytest.mark.full),
            # grouped C (n_heads > n_groups)
            pytest.param(1, 2, 64, 4, 64, 32, 2, torch.float16, None, False,
                         marks=pytest.mark.full),
            pytest.param(2, 2, 64, 8, 64, 32, 4, torch.float16, None, False,
                         marks=pytest.mark.full),
            # incomplete final chunk: chunk_len_valid < chunk_len
            pytest.param(1, 2, 64, 4, 64, 32, 1, torch.float16, 33, False,
                         marks=pytest.mark.full),
            pytest.param(1, 2, 64, 4, 64, 32, 1, torch.float16, 1, False,
                         marks=pytest.mark.full),
            pytest.param(1, 2, 64, 4, 64, 32, 1, torch.float16, 63, False,
                         marks=pytest.mark.full),
            # small N / P (tail tile stress)
            pytest.param(1, 2, 64, 4, 16, 16, 1, torch.float16, None, False,
                         marks=pytest.mark.full),
            # chunk_len not a multiple of block_l
            pytest.param(1, 2, 48, 4, 64, 32, 1, torch.float16, None, False,
                         marks=pytest.mark.full),
            pytest.param(1, 2, 48, 4, 64, 32, 1, torch.float16, 31, False,
                         marks=pytest.mark.full),
        ]),
    ]


class SsdChunkScanBwdDCTest(TestBase):
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
        chunk_len_valid: int | None,
    ):
        self.batch = batch
        self.num_chunks = num_chunks
        self.chunk_len = chunk_len
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_state = d_state
        self.n_groups = n_groups
        self.dtype = dtype
        # None means all positions valid (complete chunk)
        self.chunk_len_valid = chunk_len if chunk_len_valid is None else chunk_len_valid

    def gen_inputs(self):
        b, c, L, h, p, n, g = (
            self.batch, self.num_chunks, self.chunk_len,
            self.n_heads, self.d_head, self.d_state, self.n_groups,
        )
        S = c * L
        # states: [B, C, H, P, N]
        states = torch.randn(b, c, h, p, n, dtype=self.dtype, device="cuda") * 0.1
        # dA_cumsum: [B, H, C, L]  (non-positive, monotone decreasing cumsum)
        dA_cumsum = -torch.rand(b, h, c, L, dtype=torch.float32, device="cuda").cumsum(-1)
        # d_out: [B, S, H, P]  seqlen-fused
        d_out = torch.randn(b, S, h, p, dtype=self.dtype, device="cuda") * 0.1
        # C_in: [B, S, G, N]  seqlen-fused
        C_in = torch.randn(b, S, g, n, dtype=self.dtype, device="cuda") * 0.1
        # valid_chunk_len: [B, C] - use same value for all (b, c) pairs for simplicity
        valid_chunk_len = torch.full(
            (b, c), self.chunk_len_valid, dtype=torch.int32, device="cuda",
        )
        return states, dA_cumsum, d_out, C_in, valid_chunk_len

    def ref_program(self, states, dA_cumsum, d_out, C_in, valid_chunk_len):
        return ssd_chunk_scan_bwd_dC_ref(
            states, dA_cumsum, d_out, C_in, valid_chunk_len, self.n_groups,
        )


@SsdChunkScanBwdDCFixture
def test_ssd_chunk_scan_bwd_dC(
    batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype,
    chunk_len_valid, tune,
):
    test = SsdChunkScanBwdDCTest(
        batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype,
        chunk_len_valid,
    )
    kernel = SsdChunkScanBwdDCKernel(
        batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype, tune=tune,
    )
    inputs = test.gen_inputs()
    atol = 1e-3 if dtype == torch.float16 else 2e-3
    rtol = 1e-5
    test.check(kernel, *inputs, atol=atol, rtol=rtol)
