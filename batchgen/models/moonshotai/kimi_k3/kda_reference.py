# ---------------------------------------------------------------------------- #
#  BatchGen — Kimi-K3                                                           #
#  copyright (c) EfficientMoE team 2025                                         #
#  Licensed under the Apache License, Version 2.0                               #
# ---------------------------------------------------------------------------- #
"""Pure-torch KDA (Kimi Delta Attention) reference core — PARITY ORACLE ONLY.

This module is never a serving path.  It exists because the K3 reference code
(``modeling_kimi_linear.py``) has NO pure-torch KDA: it delegates the entire
recurrence to fla's triton ``chunk_kda``, so CPU parity tests need fla's own
torch reference math, vendored here.

Provenance (verbatim except where noted).  CORRECTED 2026-08-05: an earlier
header credited these files to fla-core 0.4.2.  That was wrong and it is the
documentation error that produced a wrong version pin downstream — the md5s
below are fla-core **0.5.2**'s.  0.4.2's ``naive.py`` is a DIFFERENT file
(md5 420284a3…, no HV/G generalization) and its chunk kernel does not apply
``sigmoid(beta)`` at all.  Re-vendor from 0.5.2 or newer only.
  * ``naive_recurrent_kda`` / ``naive_chunk_kda`` — flash-linear-attention
    ``fla/ops/kda/naive.py`` (fla-core 0.5.2 == git 2501ac83 for this file;
    md5 8dd3c35ed9a5f6af0bc7cedf9045f88d), MIT license, (c) 2023-2026
    Songlin Yang, Yu Zhang, Zhiyuan Li.
  * ``naive_kda_gate`` / ``naive_kda_lowerbound_gate`` — ``fla/ops/kda/gate.py``
    lines 27-70 (md5 8a30f4e20450fd24eb2028debfa73778), triton imports stripped.
    The lower-bound branch (K3's, keyed on ``lower_bound is not None``) is
    identical in 0.4.2 and 0.5.2; only the beta convention moved.
  * ``l2norm_ref`` — transcription of ``fla/modules/l2norm.py:105``
    (``rstd = 1/sqrt(SUM(x^2) + eps)`` — SUM, not mean; default eps 1e-6,
    l2norm.py:148-150).

``kda_reference_prefill`` composes them in the order documented by
``ChunkKDAFunction.forward`` (fla/ops/kda/chunk.py:55-100) for the flag set the
K3 oracle passes (modeling_kimi_linear.py:609-627):

    l2norm(q), l2norm(k)                (use_qk_l2norm_in_kernel=True)
    beta = sigmoid(beta_raw)            (applied unconditionally in-kernel)
    g    = LOWER-BOUND gate             (safe_gate=True because
                                         gate_lower_bound=-5.0 is set;
                                         NOT the softplus form)
    o    = naive_recurrent_kda(...)     (scale = K**-0.5, the kernel default)

The composition itself is validated against the real ``chunk_kda`` in the
staged GPU test (tests/gpu/test_kimi_k3_kda_fla_parity.py, Part A) — the
mutation discipline applies to this reference too.

Prefill-only: no initial_state, no cu_seqlens, no conv/recurrent cache.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange

__all__ = [
    "l2norm_ref",
    "naive_kda_gate",
    "naive_kda_lowerbound_gate",
    "naive_recurrent_kda",
    "naive_chunk_kda",
    "kda_reference_prefill",
]

#: fla/modules/l2norm.py:148-150 — the kernel default the oracle runs with.
L2NORM_EPS = 1e-6


def l2norm_ref(x: torch.Tensor, eps: float = L2NORM_EPS) -> torch.Tensor:
    """fla l2norm forward: ``x / sqrt(SUM(x^2, -1) + eps)`` in fp32.

    Note the SUM (fla/modules/l2norm.py:105) — this is not an RMS norm.
    Returns fp32 (the kernel pipeline keeps q/k in fp32 from here on).
    """
    x32 = x.float()
    return x32 * torch.rsqrt(x32.pow(2).sum(-1, keepdim=True) + eps)


def naive_kda_gate(
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None = None,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """fla/ops/kda/gate.py:27-55 (verbatim).  The softplus form — NOT what K3
    runs in prefill (K3 sets gate_lower_bound=-5.0, selecting the lower-bound
    form below).  Kept for the GPU cross-check of the flag wiring."""
    H, _ = g.shape[-2:]
    g = g.float()
    if dt_bias is not None:
        g = g + dt_bias.view(H, -1)
    g = (-A_log.view(H, 1).float().exp() * F.softplus(g.float())).to(output_dtype)
    return g


def naive_kda_lowerbound_gate(
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None = None,
    lower_bound: float = -5.0,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """fla/ops/kda/gate.py:58-70 (verbatim).  THE K3 gate:
    ``g = lower_bound * sigmoid(exp(A_log).view(H,1) * (g + dt_bias.view(H,K)))``
    (triton confirms at gate.py:124)."""
    H, _ = g.shape[-2:]
    g = g.float()
    if dt_bias is not None:
        g = g + dt_bias.view(H, -1)
    g = lower_bound * F.sigmoid(A_log.view(H, 1).exp() * g)
    return g.to(output_dtype)


def naive_recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
):
    """fla/ops/kda/naive.py:12-66 (verbatim).

    q,k: [B,T,H,K]; v: [B,T,HV,V]; g: LOG-space decay [B,T,HV,K];
    beta: POST-sigmoid [B,T,HV].  All-fp32 internally, o returned in v.dtype.
    State S is [B,HV,K,V] ([K,V]-major — the kernel's transpose_state_layout
    returns [V,K]-major; transpose before comparing states).
    """
    dtype = v.dtype
    B, T, H, K, HV, V = *q.shape, v.shape[2], v.shape[-1]
    G = HV // H
    if scale is None:
        scale = K ** -0.5

    q, k, v, g, beta = map(lambda x: x.to(torch.float), [q, k, v, g, beta])
    q = q.repeat_interleave(G, dim=2) * scale   # [B, T, HV, K]
    k = k.repeat_interleave(G, dim=2)           # [B, T, HV, K]

    S = k.new_zeros(B, HV, K, V).to(q)
    if initial_state is not None:
        S += initial_state
    o = torch.zeros_like(v)
    for i in range(0, T):
        q_i, k_i, v_i, g_i, b_i = q[:, i], k[:, i], v[:, i], g[:, i], beta[:, i]
        S = S * g_i[..., None].exp()
        S = S + torch.einsum('b h k, b h v -> b h k v', b_i[..., None] * k_i,
                             v_i - (k_i[..., None] * S).sum(-2))
        o[:, i] = torch.einsum('b h k, b h k v -> b h v', q_i, S)
    if not output_final_state:
        S = None
    return o.to(dtype), S


def naive_chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
):
    """fla/ops/kda/naive.py:69-166 (verbatim).  Requires T % chunk_size == 0
    (naive.py:112) — use the recurrent form for odd T.  Kept for the GPU-stage
    chunk-vs-recurrent self-consistency check."""
    dtype = v.dtype
    B, T, H, K, HV, V = *q.shape, v.shape[2], v.shape[-1]
    G = HV // H
    BT = chunk_size
    NT = T // BT
    if scale is None:
        scale = K ** -0.5
    assert T % BT == 0

    q, k = [rearrange(x, 'b (n c) h ... -> b h n c ...', c=BT).to(torch.float) for x in [q, k]]
    v, g, beta = [rearrange(x, 'b (n c) h ... -> b h n c ...', c=BT).to(torch.float)
                  for x in [v, g, beta]]
    q = q.repeat_interleave(G, dim=1) * scale  # [B, HV, NT, BT, K]
    k = k.repeat_interleave(G, dim=1)          # [B, HV, NT, BT, K]
    g = g.cumsum(-2)

    mask = torch.triu(torch.ones(BT, BT, dtype=torch.bool, device=q.device), diagonal=0)

    A = torch.zeros(*g.shape[:-1], BT, dtype=torch.float, device=q.device)
    for i in range(BT):
        k_i = k[..., i, :]
        g_i = g[..., i:i+1, :]
        A[..., i] = torch.einsum('... c d, ... d -> ... c', k * (g - g_i).exp(), k_i)
    A = A * beta[..., None]

    A = -A.masked_fill(mask, 0)
    for i in range(1, BT):
        A[..., i, :i] = A[..., i, :i].clone() + (
            A[..., i, :, None].clone() * A[..., :, :i].clone()).sum(-2)
    A = (A + torch.eye(BT, dtype=torch.float, device=q.device)) * beta[..., None, :]

    w = A @ (g.exp() * k)
    u = A @ v

    S = k.new_zeros(B, HV, K, V).to(q)
    if initial_state is not None:
        S += initial_state
    o = torch.zeros_like(v)
    mask = torch.triu(torch.ones(BT, BT, dtype=torch.bool, device=q.device), diagonal=1)
    for i in range(0, NT):
        q_i = q[:, :, i]
        k_i = k[:, :, i]
        u_i = u[:, :, i]
        g_i = g[:, :, i]
        w_i = w[:, :, i]
        Aqk = torch.zeros(B, HV, BT, BT, dtype=torch.float, device=q.device)
        for j in range(BT):
            k_j = k[:, :, i, j]
            g_j = g[:, :, i, j:j+1, :]
            Aqk[..., j] = torch.einsum('... c d, ... d -> ... c', q_i * (g_i - g_j).exp(), k_j)
        Aqk = Aqk.masked_fill(mask, 0)
        v_i = u_i - w_i @ S
        o[:, :, i] = (q_i * g_i.exp()) @ S + Aqk @ v_i
        S = S * rearrange(g_i[:, :, -1].exp(), 'b h k -> b h k 1')
        S += rearrange((g_i[:, :, -1:] - g_i).exp() * k_i, 'b h c k -> b h k c') @ v_i
    if not output_final_state:
        S = None
    return rearrange(o, 'b h n c d -> b (n c) h d').to(dtype), S


def kda_reference_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_raw: torch.Tensor,
    beta_raw: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float | None,
    use_qk_l2norm: bool = True,
    scale: float | None = None,
) -> torch.Tensor:
    """Pure-torch equivalent of the oracle's prefill ``chunk_kda`` call.

    Inputs mirror the kernel contract (fla/ops/kda/chunk.py:145-200):
    q,k,v [B,T,H,K/V]; g_raw RAW pre-decay [B,T,H,K]; beta_raw RAW logits
    [B,T,H]; A_log fp32 [H] (pass the [:num_heads] slice of the padded
    checkpoint buffer, never the full [128]); dt_bias fp32 [H*K].
    Returns o [B,T,H,V] in v.dtype.
    """
    if A_log.shape[-1] != q.shape[2]:
        raise ValueError(
            "A_log must be the [:num_heads] slice ([{}] given, {} heads). The "
            "checkpoint buffer is zero-padded to 128 and must be sliced by the "
            "caller.".format(A_log.shape[-1], q.shape[2]))
    if use_qk_l2norm:
        q = l2norm_ref(q)
        k = l2norm_ref(k)
    beta = beta_raw.float().sigmoid()
    if lower_bound is not None:
        g = naive_kda_lowerbound_gate(g_raw, A_log, dt_bias, lower_bound=lower_bound)
    else:
        g = naive_kda_gate(g_raw, A_log, dt_bias)
    o, _ = naive_recurrent_kda(q, k, v, g, beta, scale=scale,
                               initial_state=None, output_final_state=False)
    return o
