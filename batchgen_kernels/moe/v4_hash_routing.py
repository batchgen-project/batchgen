# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                          #
# ---------------------------------------------------------------------------- #

from __future__ import annotations

import torch
import torch.nn.functional as F


def hash_routing(
    input_ids: torch.Tensor | None,
    tid2eid: torch.Tensor,
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    topk: int = 6,
    route_scale: float = 1.0,
    score_func: str = "sqrtsoftplus",
    norm_topk_prob: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = F.linear(hidden_states.float(), gate_weight.float())
    if score_func == "softmax":
        scores = scores.softmax(dim=-1)
    elif score_func == "sigmoid":
        scores = scores.sigmoid()
    elif score_func == "sqrtsoftplus":
        scores = F.softplus(scores).sqrt()
    else:
        raise ValueError(f"Unsupported V4 gate score function: {score_func}")

    raw_scores = scores
    if input_ids is None:
        topk_indices = torch.topk(scores, k=topk, dim=-1)[1]
    else:
        topk_indices = tid2eid[input_ids].long()

    topk_weights = raw_scores.gather(-1, topk_indices)
    if score_func != "softmax" and norm_topk_prob:
        topk_weights = topk_weights / (
            topk_weights.sum(dim=-1, keepdim=True) + 1e-20
        )
    return topk_weights * route_scale, topk_indices
