import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.kernels.mamba import SsdStatePassingBwdKernel


def ssd_state_passing_bwd_ref(
    g_states_read_out_in: torch.Tensor,  # (B, C, H, P, N)  — direct gradient onto s_{c-1}
    dA_chunk_cumsum: torch.Tensor,      # (B, H, C)         — per-chunk log-decay: log alpha_c
    states_before_chunk: torch.Tensor,  # (B, C, H, P, N)   — s_{c-1} saved in forward
    g_final_state_in: torch.Tensor | None,  # (B, H, P, N) or None
    has_initial_states: bool,
    return_states: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """
    Math-spec reference for _state_passing_bwd.

    Corresponds to: ssd_state_passing._state_passing_bwd_kernel
    Official file:  mamba_ssm/ops/triton/ssd_state_passing.py

    Forward state-passing recurrence (what this kernel inverts):
        s_c = alpha_c * s_{c-1} + u_c     for c = 0, ..., C-1
        where alpha_c = exp(dA_chunk_cumsum[b, h, c])

    The kernel stores s_{c-1} in ``states_before_chunk`` (index 0 = s_{-1}).

    Backward (official kernel scans c = C-1 down to 0):

        For c = C-1 down to 0:
            g_u[c]      = g_sc           (gradient wrt the local contribution u_c)
            g_dA[c]     = alpha_c * <g_sc, s_{c-1}>
                          (zero if c == 0 and not has_initial_states)
            g_{s_{c-1}} = g_read_out[c] + alpha_c * g_sc

        After loop: g_{s_{-1}} is saved as g_initial_states if has_initial_states.

    Mathematical note:
        g_dA[c] = exp(dA_chunk_cumsum[c]) * sum_{p,n}(g_sc[p,n] * s_{c-1}[p,n])
    This is a direct translation of the kernel lines:
        ddA = tl.sum(out * dstates) * scale
        dstates = scale * dstates + d_out

    Contract assumption:
        ``dA_chunk_cumsum`` contains per-chunk scalar values (one per (b,h,c) triple),
        NOT the within-chunk position-level cumsum used by chunk_scan / chunk_state kernels.

    Returns 4 values matching kernel output order:
        g_u_out              : (B, C, H, P, N)  float32
        g_dA_out             : (B, H, C)         float32
        g_initial_states_out : (B, H, P, N) or None
        states_out           : (B, C, H, P, N) float32 or None
    """
    B, C, H, P, N = g_states_read_out_in.shape
    dev = g_states_read_out_in.device

    g_u  = torch.zeros(B, C, H, P, N, dtype=torch.float32, device=dev)
    g_dA = torch.zeros(B, H, C,       dtype=torch.float32, device=dev)

    # Initialise the running gradient with the final-state gradient (if provided)
    if g_final_state_in is not None:
        g_sc = g_final_state_in.float().clone()   # (B, H, P, N)
    else:
        g_sc = torch.zeros(B, H, P, N, dtype=torch.float32, device=dev)

    # --------------------------------------------------------
    # Reverse scan: c = C-1 down to 0  (matches kernel loop direction)
    # --------------------------------------------------------
    for c in range(C - 1, -1, -1):
        s_prev  = states_before_chunk[:, c].float()        # (B, H, P, N)  = s_{c-1}
        alpha_c = torch.exp(dA_chunk_cumsum[:, :, c])      # (B, H)

        # g_u[c] = g_sc  (gradient flows back through the u_c additive branch)
        g_u[:, c] = g_sc

        # g_dA[c] = alpha_c * <g_sc, s_{c-1}>_pn
        # Official kernel: ddA = tl.sum(out * dstates) * scale
        #   where out = s_{c-1}, dstates = g_sc, scale = alpha_c
        # Zero at c=0 when there is no initial state (s_{-1} = 0, not differentiable)
        if c > 0 or has_initial_states:
            g_dA[:, :, c] = alpha_c * (g_sc * s_prev).sum(dim=(-2, -1))  # sum over p, n

        # Propagate gradient to s_{c-1}:
        #   g_{s_{c-1}} = g_read_out[c] + alpha_c * g_sc
        # Official kernel: dstates = scale * dstates + d_out
        g_sc = (
            g_states_read_out_in[:, c].float()
            + alpha_c[:, :, None, None] * g_sc
        )

    # After loop: g_sc holds g_{s_{-1}} = gradient w.r.t. initial states
    g_init = g_sc if has_initial_states else None
    s_out  = states_before_chunk.float() if return_states else None

    return g_u, g_dA, g_init, s_out


class SsdStatePassingBwdFixture(FixtureBase):
    PARAMS = [
        ("batch, num_chunks, n_heads, d_head, d_state, dtype, "
         "has_final_state_grad, has_initial_states, return_states, tune", [
            # smoke: baseline, no optional features
            pytest.param(1, 4, 4, 32, 32, torch.float16,
                         False, False, False, False, marks=pytest.mark.smoke),
            # with final-state grad
            pytest.param(1, 4, 4, 32, 32, torch.float16,
                         True, False, False, False, marks=pytest.mark.full),
            # with initial states (enables g_dA[0] and g_initial_states_out)
            pytest.param(1, 4, 4, 32, 32, torch.float16,
                         False, True, False, False, marks=pytest.mark.full),
            # return_states pass-through
            pytest.param(1, 4, 4, 32, 32, torch.float16,
                         False, False, True, False, marks=pytest.mark.full),
            # all flags on
            pytest.param(1, 4, 4, 32, 32, torch.float16,
                         True, True, True, False, marks=pytest.mark.full),
            # larger shape, bfloat16
            pytest.param(2, 8, 8, 64, 64, torch.bfloat16,
                         False, False, False, False, marks=pytest.mark.full),
            pytest.param(2, 8, 8, 64, 64, torch.bfloat16,
                         True, True, False, False, marks=pytest.mark.full),
        ]),
    ]


class SsdStatePassingBwdTest(TestBase):
    def __init__(
        self,
        batch: int,
        num_chunks: int,
        n_heads: int,
        d_head: int,
        d_state: int,
        dtype: torch.dtype,
        has_final_state_grad: bool,
        has_initial_states: bool,
        return_states: bool,
    ):
        self.batch = batch
        self.num_chunks = num_chunks
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_state = d_state
        self.dtype = dtype
        self.has_final_state_grad = has_final_state_grad
        self.has_initial_states = has_initial_states
        self.return_states = return_states

    def gen_inputs(self):
        B, C, H, P, N = (
            self.batch, self.num_chunks, self.n_heads, self.d_head, self.d_state,
        )
        dev = "cuda"

        # Direct read_out gradients onto s_{c-1}: float32 (output of chunk_scan bwd)
        g_read_out = torch.randn(B, C, H, P, N, dtype=torch.float32, device=dev) * 0.1

        # dA_chunk_cumsum: [B, H, C], dA <= 0
        dA_chunk_cumsum = -torch.rand(B, H, C, dtype=torch.float32, device=dev)

        # states_before_chunk stored in forward dtype (as saved by fwd kernel)
        states = torch.randn(B, C, H, P, N, dtype=self.dtype, device=dev) * 0.1

        g_final = None
        if self.has_final_state_grad:
            g_final = torch.randn(B, H, P, N, dtype=torch.float32, device=dev) * 0.1

        return g_read_out, dA_chunk_cumsum, states, g_final

    def ref_program(self, g_read_out, dA_chunk_cumsum, states, g_final):
        return ssd_state_passing_bwd_ref(
            g_read_out, dA_chunk_cumsum, states, g_final,
            has_initial_states=self.has_initial_states,
            return_states=self.return_states,
        )


@SsdStatePassingBwdFixture
def test_ssd_state_passing_bwd(
    batch, num_chunks, n_heads, d_head, d_state, dtype,
    has_final_state_grad, has_initial_states, return_states, tune,
):
    test = SsdStatePassingBwdTest(
        batch, num_chunks, n_heads, d_head, d_state, dtype,
        has_final_state_grad, has_initial_states, return_states,
    )
    kernel = SsdStatePassingBwdKernel(
        batch, num_chunks, n_heads, d_head, d_state,
        has_final_state_grad=has_final_state_grad,
        has_initial_states=has_initial_states,
        return_states=return_states,
        dtype=dtype,
        tune=tune,
    )
    inputs = test.gen_inputs()
    atol = 1e-3 if dtype == torch.float16 else 2e-3
    rtol = 1e-5
    # ref returns None for unused outputs; TestBase.check skips None refs
    test.check(kernel, *inputs, atol=atol, rtol=rtol)
