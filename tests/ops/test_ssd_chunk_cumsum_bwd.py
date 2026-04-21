import torch
import torch.nn.functional as F


def chunk_cumsum_bwd_ref(
    ddA,          # [batch, nheads, nchunks, chunk_size]  — final merged ddA
    ddt_out,      # [batch, nheads, nchunks, chunk_size]
    dt,           # [batch, seqlen, nheads]               (official name: dt_in)
    A,            # [nheads]
    dt_bias=None, # [nheads] or None
    dt_softplus=False,
    dt_limit=(0.0, float("inf")),
    ddt=None,     # optional preallocated output, same shape as dt
):
    """
    Math-spec reference for official _chunk_cumsum_bwd.

    Corresponds to: ssd_chunk_state._chunk_cumsum_bwd kernel
    Official file:  mamba_ssm/ops/triton/ssd_chunk_state.py

    Contract assumption
    -------------------
    ``ddA`` is the **already-merged** ddA gradient arriving at the
    _chunk_cumsum_bwd boundary.  No reverse-cumsum is performed here; that
    merging happens upstream in _chunk_scan_bwd_ddAcs_stable / _chunk_state_bwd.

    Forward pass (what this kernel inverts):
        dt_pre  = dt_in  [+ dt_bias]
        dt_sp   = softplus(dt_pre)   if dt_softplus else dt_pre
                  (official stable branch: identity for dt_pre > 20)
        dt_work = clamp(dt_sp, dt_min, dt_max)
        dA_raw  = A * dt_work          (per-position, per-head)
        dA_cumsum[t] = cumsum_t(dA_raw)

    Backward math:
        ddt_work[t] = ddA[t] * A   +   ddt_out[t]
        dA          = sum_{b,c,t} ddA[b,h,c,t] * dt_work[b,h,c,t]   (per head h)

    Then chain-rule through clamp, softplus, and optional dt_bias:
        ddt_sp  = ddt_work  (zeroed where clamped)
        ddt_pre = ddt_sp * sigmoid(dt_pre)  if dt_softplus and dt_pre <= 20
                  ddt_sp                    otherwise
        ddt_bias = sum_{b,c,t} ddt_pre[b,h,c,t]

    Output layout:
        ddt      : [B, S, H]   seqlen-fused (matches dt input layout)
        dA       : [H]
        ddt_bias : [H]  or None
    """
    B, H, C, T = ddA.shape
    assert ddt_out.shape == (B, H, C, T)
    assert dt.shape[0] == B and dt.shape[2] == H
    assert A.shape == (H,)

    S = dt.shape[1]
    total_len = C * T
    assert total_len >= S

    dt_min, dt_max = dt_limit

    # --------------------------------------------------------
    # All arithmetic in float32
    # --------------------------------------------------------
    ddA_f     = ddA.float()
    ddt_out_f = ddt_out.float()
    A_f       = A.float()
    dt_f      = dt.float()
    dt_bias_f = dt_bias.float() if dt_bias is not None else None

    # --------------------------------------------------------
    # Reshape dt from seqlen-fused [B, S, H] to chunk-local [B, H, C, T].
    # Pad if S < C*T (incomplete last chunk).
    # Official kernel: chunk_size_limit = min(chunk_size, seqlen - c*chunk_size)
    # --------------------------------------------------------
    if total_len > S:
        dt_padded = F.pad(dt_f, (0, 0, 0, total_len - S))  # [B, C*T, H]
    else:
        dt_padded = dt_f
    dt_chunk = dt_padded.view(B, C, T, H).permute(0, 3, 1, 2)  # [B, H, C, T]

    # Valid-position mask: position t in chunk c is valid iff c*T + t < S
    c_idx = torch.arange(C, device=dt.device)
    t_idx = torch.arange(T, device=dt.device)
    valid_len  = (S - c_idx * T).clamp(min=0, max=T)                    # [C]
    valid_mask = t_idx[None, None, None, :] < valid_len[None, None, :, None]  # [1,1,C,T]

    # Mask out invalid positions in the incoming gradients
    ddA_f     = ddA_f.masked_fill(~valid_mask, 0.0)
    ddt_out_f = ddt_out_f.masked_fill(~valid_mask, 0.0)

    # --------------------------------------------------------
    # Rebuild forward-side dt_work (same computation as forward pass)
    # --------------------------------------------------------
    dt_pre = dt_chunk
    if dt_bias_f is not None:
        dt_pre = dt_pre + dt_bias_f[None, :, None, None]  # [B, H, C, T]

    if dt_softplus:
        # Official stable softplus: identity for dt_pre > 20
        dt_sp = torch.where(dt_pre <= 20.0, F.softplus(dt_pre), dt_pre)
    else:
        dt_sp = dt_pre

    clamp_mask = (dt_sp < dt_min) | (dt_sp > dt_max)
    dt_work = dt_sp.clamp(min=dt_min, max=dt_max).masked_fill(~valid_mask, 0.0)

    # --------------------------------------------------------
    # Core backward math (direct translation of kernel lines 144–162):
    #   ddt_work[b,h,c,t] = ddA[b,h,c,t] * A[h]  +  ddt_out[b,h,c,t]
    #   dA[h]              = sum_{b,c,t} ddA[b,h,c,t] * dt_work[b,h,c,t]
    # --------------------------------------------------------
    ddt_work = ddA_f * A_f[None, :, None, None] + ddt_out_f  # [B, H, C, T]
    dA       = torch.einsum("bhct,bhct->h", ddA_f, dt_work)  # [H]

    # --------------------------------------------------------
    # Chain rule through clamp: zero gradient where clamped
    # --------------------------------------------------------
    ddt_sp = ddt_work.masked_fill(clamp_mask, 0.0)

    # --------------------------------------------------------
    # Chain rule through softplus (official stable derivative):
    #   d(softplus)/d(x) = sigmoid(x)   for x <= 20
    #   d(softplus)/d(x) = 1            for x > 20  (identity branch)
    # --------------------------------------------------------
    if dt_softplus:
        sp_grad = torch.where(dt_pre <= 20.0, torch.sigmoid(dt_pre), torch.ones_like(dt_pre))
        ddt_pre = ddt_sp * sp_grad
    else:
        ddt_pre = ddt_sp

    # Mask invalid positions in the output gradient
    ddt_pre = ddt_pre.masked_fill(~valid_mask, 0.0)

    # --------------------------------------------------------
    # dt_bias gradient: sum over all valid (b, c, t) positions
    # --------------------------------------------------------
    ddt_bias_out = torch.einsum("bhct->h", ddt_pre) if dt_bias_f is not None else None

    # --------------------------------------------------------
    # Map ddt back to seqlen-fused layout [B, S, H]:
    #   [B, H, C, T] -> permute -> [B, C, T, H] -> reshape -> [B, C*T, H]
    #   then truncate to the actual seqlen S
    # --------------------------------------------------------
    ddt_full = ddt_pre.permute(0, 2, 3, 1).reshape(B, total_len, H)[:, :S]  # [B, S, H]

    if ddt is None:
        ddt = ddt_full.contiguous()
    else:
        assert ddt.shape == dt.shape
        ddt.copy_(ddt_full)

    return ddt, dA, ddt_bias_out
