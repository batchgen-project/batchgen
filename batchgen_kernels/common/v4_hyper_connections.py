# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                          #
# ---------------------------------------------------------------------------- #

import torch
import torch.nn.functional as F


def hc_split(
    mixes: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    hc_mult: int,
    sinkhorn_iters: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split HC mixes into pre/post gates and Sinkhorn-normalized comb."""
    pre = torch.sigmoid(mixes[..., :hc_mult] * scale[0] + base[:hc_mult]) + eps
    post = 2 * torch.sigmoid(
        mixes[..., hc_mult : 2 * hc_mult] * scale[1]
        + base[hc_mult : 2 * hc_mult]
    )
    comb_base = base[2 * hc_mult :].view(hc_mult, hc_mult)
    comb = mixes[..., 2 * hc_mult :].view(*mixes.shape[:-1], hc_mult, hc_mult)
    comb = torch.softmax(comb * scale[2] + comb_base, dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(max(int(sinkhorn_iters) - 1, 0)):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb


def hc_pre(
    hidden_states: torch.Tensor,
    fn_weight: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    hc_mult: int,
    sinkhorn_iters: int,
    hc_eps: float,
    rms_norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply HC mixing on flattened states and reduce branches by pre gates."""
    shape = hidden_states.shape
    flat = hidden_states.flatten(2).float()
    rsqrt = torch.rsqrt(flat.square().mean(-1, keepdim=True) + rms_norm_eps)
    mixes = F.linear(flat, fn_weight) * rsqrt
    pre, post, comb = hc_split(
        mixes, scale, base, hc_mult, sinkhorn_iters, hc_eps
    )
    reduced = torch.sum(pre.unsqueeze(-1) * flat.view(shape), dim=2)
    return reduced.to(hidden_states.dtype), post, comb


def hc_post(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct HC outputs from branch activations and residual mixing."""
    return (
        post.unsqueeze(-1) * hidden_states.unsqueeze(-2)
        + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
    ).to(hidden_states.dtype)
