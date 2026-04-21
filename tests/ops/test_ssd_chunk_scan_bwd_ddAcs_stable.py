import pytest
import torch
import torch.nn.functional as F


def torch_ref_ssd_chunk_scan_bwd_ddAcs_stable(
    d_out,        # [B, C, H, Q, P]  — upstream gradient (seqlen position l)
    x,           # [B, C, H, Q, P]  — input activations
    dt,          # [B, C, H, Q]     — post-processed dt (already clamped/softplussed)
    dA_cumsum,   # [B, C, H, Q]     — A * dt cumsum; dA_cumsum[t] = sum_{i<=t} A*dt[i]
    CB,          # [B, C, G, Q, Q]  — CB product, causal lower-triangular
):
    """
    Math-spec reference for K10 / _chunk_scan_bwd_ddAcs_stable.

    Corresponds to: ssd_chunk_scan._chunk_scan_bwd_ddAcs_stable kernel
    Official file:  mamba_ssm/ops/triton/ssd_chunk_scan.py

    Forward local path (within-chunk diagonal contribution):
        out_local[b,c,h,l,p] = sum_{s<=l} CB[b,c,g(h),l,s]
                                           * exp(min(dA_cumsum[l] - dA_cumsum[s], 0))
                                           * dt[s]
                                           * x[b,c,h,s,p]

    Backward math — pairwise weight:
        W[b,c,h,l,s] = (sum_p d_out[l,p] * x[s,p])   <- inner product over head-dim
                        * CB[b,c,g(h),l,s]
                        * dt[s]
                        * exp(min(dA_cumsum[l] - dA_cumsum[s], 0))
                        * 1[l > s]                    <- strict lower-triangle (causal)

    ddA contribution:
        The official kernel computes, for each position t, the sum of W[l,s] over
        all pairs (l, s) such that s < t <= l  (i.e. the interval (s, l] contains t).

        Direct mathematical definition:
            ddA[b,c,h,t] = sum_{l,s: s < t <= l} W[b,c,h,l,s]

        Equivalently (using prefix-sum / interval membership):
            ddA[b,c,h,t] = sum_l (sum_{s < t} W[b,c,h,l,s])  for l >= t

        The official kernel implements this via a running rowsum + cumsum trick.
        This reference expresses it directly using an explicit interval mask to
        avoid imitating kernel scheduling structure.

    Contract assumption:
        ``dA_cumsum`` is the within-chunk cumulative sum (not the per-chunk end-value).
        It is NOT the log-decay of full chunks passed to _chunk_cumsum_bwd.
    """
    B, C, H, Q, P = d_out.shape
    _, _, G, _, _ = CB.shape
    assert H % G == 0
    heads_per_group = H // G

    # --------------------------------------------------------
    # Expand CB from group-stride [B,C,G,Q,Q] to head-stride [B,C,H,Q,Q].
    # Correct group-to-head mapping: head h belongs to group h // heads_per_group.
    # --------------------------------------------------------
    # group index for each head: [H]
    h_idx      = torch.arange(H, device=d_out.device)
    group_idx  = h_idx // heads_per_group            # [H]

    # select CB per head: CB_h[b,c,h,l,s] = CB[b,c,group_idx[h],l,s]
    CB_f = CB.float()                                # [B, C, G, Q, Q]
    CB_h = CB_f[:, :, group_idx, :, :]              # [B, C, H, Q, Q]

    d_out_f = d_out.float()                            # [B, C, H, Q, P]
    x_f    = x.float()                               # [B, C, H, Q, P]
    dt_f   = dt.float()                              # [B, C, H, Q]
    a      = dA_cumsum.float()                       # [B, C, H, Q]

    # --------------------------------------------------------
    # Pairwise inner product over head dimension P:
    #   pair_base[b,c,h,l,s] = sum_p d_out[b,c,h,l,p] * x[b,c,h,s,p]
    # This is a direct einsum translation of the kernel's tl.dot(d_out, x) step.
    # --------------------------------------------------------
    pair_base = torch.einsum("bchlp,bchsp->bchls", d_out_f, x_f)  # [B, C, H, Q, Q]

    # --------------------------------------------------------
    # Stable decay: exp(min(dA_cumsum[l] - dA_cumsum[s], 0))
    # Official kernel: tl.exp(tl.minimum(dA_cs_m - dA_cs_n, 0.0))
    # --------------------------------------------------------
    a_l = a.unsqueeze(-1)   # [B, C, H, Q, 1]   indexed as l
    a_s = a.unsqueeze(-2)   # [B, C, H, 1, Q]   indexed as s
    decay = torch.exp(torch.minimum(a_l - a_s, torch.zeros_like(a_l - a_s)))  # [B,C,H,Q,Q]

    # dt lives on the source position s  (official kernel: dt_n = load(dt_ptr + offs_n))
    dt_s = dt_f.unsqueeze(-2)  # [B, C, H, 1, Q]  broadcast over l

    # --------------------------------------------------------
    # Pairwise backward weight W[b,c,h,l,s]:
    #   W = pair_base * CB_h * dt_s * decay * strict_lower_triangle
    # The official kernel stores acc then applies:
    #   mask = offs_m >= start_n + offs_n + 1   (strict lower-triangle: l > s)
    # --------------------------------------------------------
    W = pair_base * CB_h * dt_s * decay            # [B, C, H, Q, Q]

    # Strict lower-triangle mask: keep only l > s
    l_pos = torch.arange(Q, device=d_out.device)
    s_pos = torch.arange(Q, device=d_out.device)
    strict_lower = (l_pos[:, None] > s_pos[None, :])  # [Q, Q]
    W = W * strict_lower[None, None, None, :, :]       # broadcast over [B,C,H]

    # --------------------------------------------------------
    # Interval-folding to get ddA[t]:
    #   ddA[b,c,h,t] = sum_{l,s: s < t <= l} W[b,c,h,l,s]
    #
    # Expressed with an explicit 3-way interval mask [l, s, t]:
    #   interval_mask[l, s, t] = 1  iff  s < t  and  t <= l
    # --------------------------------------------------------
    t_idx = torch.arange(Q, device=d_out.device)    # [Q]
    # interval_mask[l, s, t]: l-axis=row, s-axis=col, t-axis=depth
    in_interval = (
        (s_pos[None, :, None] < t_idx[None, None, :]) &   # s < t
        (t_idx[None, None, :] <= l_pos[:, None, None])     # t <= l
    )  # [Q, Q, Q]

    # ddA[b,c,h,t] = sum_{l,s} W[b,c,h,l,s] * in_interval[l,s,t]
    ddA_out = torch.einsum("bchls,lst->bcht", W, in_interval.float())  # [B,C,H,Q]

    return ddA_out


def make_monotonic_dA_cumsum(B, C, H, Q, device):
    # negative increments -> a_l - a_s is usually <= 0 for l >= s
    steps = -F.softplus(torch.randn(B, C, H, Q, device=device, dtype=torch.float32))
    return torch.cumsum(steps, dim=-1)


@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float16, 3e-2, 3e-2),
        (torch.bfloat16, 5e-2, 5e-2),
    ],
)
def test_ssd_chunk_scan_bwd_ddAcs_stable(dtype, atol, rtol):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    torch.manual_seed(0)
    device = "cuda"

    B = 2
    C = 3
    H = 4
    G = 2
    Q = 8
    P = 16

    assert H % G == 0

    d_out = torch.randn(B, C, H, Q, P, device=device, dtype=dtype)
    x = torch.randn(B, C, H, Q, P, device=device, dtype=dtype)
    dt = torch.rand(B, C, H, Q, device=device, dtype=torch.float32)
    dA_cumsum = make_monotonic_dA_cumsum(B, C, H, Q, device)
    CB = torch.randn(B, C, G, Q, Q, device=device, dtype=dtype)

    ref = torch_ref_ssd_chunk_scan_bwd_ddAcs_stable(
        d_out=d_out,
        x=x,
        dt=dt,
        dA_cumsum=dA_cumsum,
        CB=CB,
    )

    actual = torch.zeros((B, C, H, Q), device=device, dtype=torch.float32)

    # 替换成你的真实调用
    # ssd_chunk_scan_bwd_ddAcs_stable(
    #     d_out, x, dt, dA_cumsum, CB, actual
    # )

    raise RuntimeError("Replace the kernel call with your actual implementation, then compare actual vs ref.")

    # torch.testing.assert_close(actual.float(), ref, atol=atol, rtol=rtol)
