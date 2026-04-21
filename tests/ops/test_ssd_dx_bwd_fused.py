import torch
import torch.nn.functional as F


def chunk_scan_chunk_state_bwd_dx_ref(
    x,         # [B, S, H, P]     input dtype    S = C*L seqlen-fused
    dt,        # [B, H, C, L]     input dtype    already post-processed (clamped/softplussed)
    dA_cumsum, # [B, H, C, L]     float32        within-chunk cumulative sum of A*dt
    B_in,      # [B, S, G_or_H, N] input dtype
    CB,        # [B, C, G, L, L]  input dtype    causal; G = n_groups
    d_out,      # [B, S, H, P]     input dtype
    dstates,   # [B, C, H, P, N]  float32        gradient w.r.t. chunk states
):
    """
    Math-spec reference for _chunk_scan_chunk_state_bwd_dx (K4 / fused dx).

    Corresponds to: ssd_combined._chunk_scan_chunk_state_bwd_dx_kernel
    Official file:  mamba_ssm/ops/triton/ssd_combined.py

    Forward pass (output accumulation for position m in chunk c):
        acc[m] = (state-path)  +  (local-path)

    State path:
        acc_state[m, p] = sum_n B[m, g(h), n] * dstates[c, h, p, n]
                          * exp(min(dA_end - dA_cumsum[m], 0))
        where dA_end = dA_cumsum[c, last_valid_position]

    Local/diagonal path (CB-weighted sum over l >= m within chunk):
        acc_local[m, p] = sum_{l>=m} CB[c, g(h), l, m]
                          * exp(min(dA_cumsum[l] - dA_cumsum[m], 0))
                          * d_out[c*L+l, h, p]

    Combined:
        acc[m, p] = acc_state[m, p] + acc_local[m, p]

    Outputs:
        dx[m, p]          = acc[m, p] * dt[m]
        ddt_partial[m]    = sum_p acc[m, p] * x[m, p]

    Contract assumptions:
    - B_in can be either group-indexed (dim=G) or head-indexed (dim=H).
    - CB is the lower-triangle: CB[l, m] is zero for l < m.
    - dA_cumsum is the within-chunk (position-level) cumsum, not the per-chunk scalar.
    - dt is already post-processed (softplus + clamp applied in _chunk_cumsum_fwd).
    - D and dD terms are excluded from this reference (handled separately).
    - seq_idx (multi-sequence batching) is excluded.
    """
    B, S, H, P = x.shape
    _, H2, C, L = dt.shape
    assert H2 == H
    assert dA_cumsum.shape == (B, H, C, L)
    assert d_out.shape == (B, S, H, P)
    B2, S2, GH, N = B_in.shape
    assert B2 == B and S2 == S
    assert CB.shape[:2] == (B, C)
    G = CB.shape[2]
    assert CB.shape == (B, C, G, L, L)
    assert dstates.shape == (B, C, H, P, N)
    assert H % G == 0
    hpg = H // G

    total_len = C * L

    # --------------------------------------------------------
    # Pad seqlen-fused tensors to C*L if S < C*L, then chunk-view
    # Official kernel: chunk_size_limit = min(chunk_size, seqlen - pid_c*chunk_size)
    # --------------------------------------------------------
    if total_len > S:
        pad = total_len - S
        x     = F.pad(x,    (0, 0, 0, 0, 0, pad))
        d_out  = F.pad(d_out, (0, 0, 0, 0, 0, pad))
        B_in  = F.pad(B_in, (0, 0, 0, 0, 0, pad))

    # [B, C, L, H, P] -> [B, C, H, L, P]
    x_c    = x.view(B, C, L, H, P).permute(0, 1, 3, 2, 4).float()     # [B, C, H, L, P]
    d_out_c = d_out.view(B, C, L, H, P).permute(0, 1, 3, 2, 4).float()  # [B, C, H, L, P]

    # B_in chunked: handle both G and H as third dimension
    B_raw = B_in.view(B, C, L, GH, N).float()  # [B, C, L, GH, N]
    if GH == G:
        # group-owned: expand to head-owned [B, C, H, L, N]
        B_c = (B_raw.unsqueeze(3)                               # [B, C, L, 1, G, N]
                    .expand(B, C, L, hpg, G, N)
                    .reshape(B, C, L, H, N)
                    .permute(0, 1, 3, 2, 4))                    # [B, C, H, L, N]
    elif GH == H:
        B_c = B_raw.permute(0, 1, 3, 2, 4)                     # [B, C, H, L, N]
    else:
        raise ValueError(f"B_in third dim must be G={G} or H={H}, got {GH}")

    # CB: [B, C, G, L, L] -> expand to [B, C, H, L, L] (indexed as [l, m])
    CB_h = (CB.unsqueeze(3)
               .expand(B, C, G, hpg, L, L)
               .reshape(B, C, H, L, L)
               .float())

    # [B, H, C, L] -> [B, C, H, L]
    a      = dA_cumsum.permute(0, 2, 1, 3).float()   # [B, C, H, L]
    dt_c   = dt.permute(0, 2, 1, 3).float()          # [B, C, H, L]

    # --------------------------------------------------------
    # Valid-position mask: position m in chunk c valid iff c*L + m < S
    # --------------------------------------------------------
    c_start = torch.arange(C, device=x.device) * L           # [C]
    valid_len = (S - c_start).clamp(min=0, max=L)             # [C]
    m_pos = torch.arange(L, device=x.device)                  # [L]
    valid = (m_pos[None, :] < valid_len[:, None])             # [C, L]
    valid = valid[None, :, None, :]                            # [1, C, 1, L]

    # --------------------------------------------------------
    # dA_end: the dA_cumsum value at the last valid position in each chunk.
    # Official kernel: dA_cs_last = tl.load(dA_cumsum_ptr + (chunk_size-1)*stride)
    # (The kernel uses chunk_size-1 unconditionally; we use the actual last valid pos.)
    # --------------------------------------------------------
    end_idx = (valid_len.long() - 1).clamp(min=0)             # [C]
    dA_end  = a.gather(
        dim=-1,
        index=end_idx[None, :, None, None].expand(B, C, H, 1)
    ).squeeze(-1)                                              # [B, C, H]

    # --------------------------------------------------------
    # State path:
    #   acc_state[b,c,h,m,p] = sum_n B[b,c,h,m,n] * dstates[b,c,h,p,n]
    #                           * exp(min(dA_end[c] - dA_cumsum[m], 0))
    # Official kernel: acc += tl.dot(b, dstates) * scale[:, None]
    # --------------------------------------------------------
    state_decay = torch.exp(
        torch.minimum(dA_end[..., None] - a, torch.zeros_like(a))
    )                                                           # [B, C, H, L]

    # einsum: [B,C,H,L,N] x [B,C,H,P,N] -> [B,C,H,L,P]
    acc_state = torch.einsum("bchln,bchpn->bchlp", B_c, dstates.float())
    acc_state = acc_state * state_decay[..., None]             # scale by decay
    acc_state = acc_state.masked_fill(~valid[..., None], 0.0)  # zero invalid m

    # --------------------------------------------------------
    # Local/diagonal path:
    #   acc_local[b,c,h,m,p] = sum_{l: l>=m, valid} CB[b,c,h,l,m]
    #                           * exp(min(dA_cumsum[l] - dA_cumsum[m], 0))
    #                           * d_out[b,c,h,l,p]
    # Official kernel: cb *= tl.exp(tl.minimum(dA_cs_k - dA_cs_m, 0.0))
    #                  mask = (k + offs_k >= offs_m)
    #                  acc += tl.dot(cb, d_out)
    # --------------------------------------------------------
    a_l = a.unsqueeze(-1)   # [B, C, H, L, 1]   l-indexed
    a_m = a.unsqueeze(-2)   # [B, C, H, 1, L]   m-indexed
    local_decay = torch.exp(
        torch.minimum(a_l - a_m, torch.zeros_like(a_l - a_m))
    )  # [B, C, H, L, L]  — causal l>=m (official stable decay)

    # Causal mask: keep only l >= m  (lower-triangle of [l, m] matrix)
    causal = torch.tril(torch.ones(L, L, device=x.device, dtype=torch.bool))  # [l, m]
    # Both l and m must be valid
    valid_l = valid[..., :, None]   # [1, C, 1, L, 1]
    valid_m = valid[..., None, :]   # [1, C, 1, 1, L]

    # local_coeff[b,c,h,l,m] = CB[b,c,h,l,m] * decay * causal * valid
    local_coeff = CB_h * local_decay
    local_coeff = local_coeff.masked_fill(
        ~(causal[None, None, None, :, :] & valid_l & valid_m),
        0.0,
    )

    # einsum: sum over l  [B,C,H,l,m] x [B,C,H,l,p] -> [B,C,H,m,p]
    acc_local = torch.einsum("bchlm,bchlp->bchmp", local_coeff, d_out_c)

    # --------------------------------------------------------
    # Merge and compute outputs
    # --------------------------------------------------------
    acc = acc_state + acc_local                                  # [B, C, H, L, P]

    # dx[m, p] = acc[m, p] * dt[m]
    dx_c = acc * dt_c[..., None]                                 # [B, C, H, L, P]
    # ddt_partial[m] = sum_p acc[m, p] * x[m, p]
    ddt_c = (acc * x_c).sum(dim=-1)                              # [B, C, H, L]

    # Zero invalid positions
    dx_c  = dx_c.masked_fill(~valid[..., None], 0.0)
    ddt_c = ddt_c.masked_fill(~valid, 0.0)

    # Map back to seqlen-fused layouts
    dx_out  = dx_c.permute(0, 1, 3, 2, 4).reshape(B, total_len, H, P)[:, :S]
    ddt_out = ddt_c.permute(0, 2, 1, 3)                          # [B, H, C, L]

    return dx_out, ddt_out
