# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                          #
# ---------------------------------------------------------------------------- #

from __future__ import annotations

import torch
import torch.nn.functional as F


def sqrtsoftplus_topk(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    bias: torch.Tensor,
    topk: int = 6,
    route_scale: float = 1.0,
    norm_topk_prob: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute DeepSeek V4 sqrtsoftplus top-k routing weights and indices."""
    scores = F.linear(hidden_states.float(), gate_weight.float())
    scores = F.softplus(scores).sqrt()
    select_scores = scores + bias.float().unsqueeze(0)
    topk_indices = torch.topk(select_scores, k=topk, dim=-1)[1]
    topk_weights = scores.gather(-1, topk_indices)
    if norm_topk_prob:
        topk_weights = topk_weights / (
            topk_weights.sum(dim=-1, keepdim=True) + 1e-20
        )
    return topk_weights * route_scale, topk_indices
