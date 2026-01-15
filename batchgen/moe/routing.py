# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                          #
#                                                                               #
#  licensed under the apache license, version 2.0 (the "license");              #
#  you may not use this file except in compliance with the license.             #
#                                                                               #
#  you may obtain a copy of the license at                                      #
#                                                                               #
#                  http://www.apache.org/licenses/license-2.0                   #
#                                                                               #
#  unless required by applicable law or agreed to in writing, software          #
#  distributed under the license is distributed on an "as is" basis,            #
#  without warranties or conditions of any kind, either express or implied.     #
#  see the license for the specific language governing permissions and          #
#  limitations under the license.                                               #
# ---------------------------------------------------------------------------- #

"""Top-k expert routing for MoE layers.

This module provides routing functions for selecting which experts process
each token. It implements standard top-k routing with softmax normalization.

Usage:
    # Compute routing
    topk_indices, topk_weights = moe_routing(
        hidden_states,
        gate_weight,
        gate_bias,
        experts_per_token=4,
    )

    # Use with token dispatcher
    dispatch_result = dispatcher.dispatch(hidden_states, topk_indices, topk_weights)
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def moe_routing(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    gate_bias: Optional[torch.Tensor] = None,
    experts_per_token: int = 4,
    normalize_weights: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Top-k expert routing with softmax normalization.

    Computes gate logits and selects top-k experts per token with
    softmax-normalized routing weights.

    Args:
        hidden_states: Input tokens. Shape: [num_tokens, hidden_size]
        gate_weight: Gate projection weight. Shape: [hidden_size, num_experts]
            or [num_experts, hidden_size] (will be transposed if needed)
        gate_bias: Optional gate bias. Shape: [num_experts]
        experts_per_token: Number of experts to select per token (k).
        normalize_weights: If True, apply softmax to top-k logits.

    Returns:
        topk_indices: Selected expert indices. Shape: [num_tokens, k]
        topk_weights: Routing weights for selected experts. Shape: [num_tokens, k]
    """
    # Compute gate logits
    # Handle both [hidden, experts] and [experts, hidden] layouts
    if gate_weight.shape[0] == hidden_states.shape[-1]:
        logits = hidden_states @ gate_weight  # [num_tokens, num_experts]
    else:
        logits = hidden_states @ gate_weight.T

    if gate_bias is not None:
        logits = logits + gate_bias

    # Top-k selection
    topk_weights, topk_indices = torch.topk(
        logits, k=experts_per_token, dim=-1
    )

    # Softmax normalize over selected experts
    if normalize_weights:
        topk_weights = F.softmax(topk_weights, dim=-1)

    return topk_indices, topk_weights


def moe_routing_with_auxiliary_loss(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    gate_bias: Optional[torch.Tensor] = None,
    experts_per_token: int = 4,
    num_experts: int = 128,
    aux_loss_coef: float = 0.01,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Top-k routing with load balancing auxiliary loss.

    Computes routing and an auxiliary loss to encourage balanced
    expert utilization across tokens.

    Args:
        hidden_states: Input tokens. Shape: [num_tokens, hidden_size]
        gate_weight: Gate projection weight.
        gate_bias: Optional gate bias.
        experts_per_token: Number of experts to select (k).
        num_experts: Total number of experts.
        aux_loss_coef: Coefficient for auxiliary loss.

    Returns:
        topk_indices: Selected expert indices. Shape: [num_tokens, k]
        topk_weights: Routing weights. Shape: [num_tokens, k]
        aux_loss: Load balancing auxiliary loss (scalar).
    """
    num_tokens = hidden_states.shape[0]

    # Compute gate logits
    if gate_weight.shape[0] == hidden_states.shape[-1]:
        logits = hidden_states @ gate_weight
    else:
        logits = hidden_states @ gate_weight.T

    if gate_bias is not None:
        logits = logits + gate_bias

    # Routing probabilities (full softmax for loss computation)
    routing_probs = F.softmax(logits, dim=-1)  # [num_tokens, num_experts]

    # Top-k selection
    topk_weights, topk_indices = torch.topk(
        logits, k=experts_per_token, dim=-1
    )

    # Normalize selected weights
    topk_weights = F.softmax(topk_weights, dim=-1)

    # Auxiliary loss: encourage uniform expert distribution
    # tokens_per_expert: fraction of tokens routed to each expert
    # Average routing probability per expert
    expert_mask = torch.zeros_like(routing_probs)
    expert_mask.scatter_(1, topk_indices, 1.0)

    tokens_per_expert = expert_mask.sum(dim=0) / num_tokens  # [num_experts]
    router_prob_per_expert = routing_probs.mean(dim=0)  # [num_experts]

    # Load balancing loss (encourages uniform distribution)
    aux_loss = aux_loss_coef * num_experts * (
        tokens_per_expert * router_prob_per_expert
    ).sum()

    return topk_indices, topk_weights, aux_loss


class MoERouter(torch.nn.Module):
    """Trainable MoE router module.

    Encapsulates gate parameters and routing logic.

    Args:
        hidden_size: Input hidden dimension.
        num_experts: Number of experts.
        experts_per_token: Number of experts to select per token.
        aux_loss_coef: Auxiliary loss coefficient (0 to disable).
        bias: Whether to include gate bias.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        experts_per_token: int = 4,
        aux_loss_coef: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.experts_per_token = experts_per_token
        self.aux_loss_coef = aux_loss_coef

        self.gate = torch.nn.Linear(hidden_size, num_experts, bias=bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Compute routing for input tokens.

        Args:
            hidden_states: Input. Shape: [num_tokens, hidden_size] or
                [batch, seq_len, hidden_size]

        Returns:
            topk_indices: Expert indices. Shape: [..., k]
            topk_weights: Routing weights. Shape: [..., k]
            aux_loss: Auxiliary loss if aux_loss_coef > 0, else None.
        """
        original_shape = hidden_states.shape
        if hidden_states.dim() == 3:
            batch, seq_len, _ = hidden_states.shape
            hidden_states = hidden_states.view(-1, self.hidden_size)
        else:
            batch, seq_len = None, None

        if self.aux_loss_coef > 0:
            topk_indices, topk_weights, aux_loss = moe_routing_with_auxiliary_loss(
                hidden_states,
                self.gate.weight.T,
                self.gate.bias,
                self.experts_per_token,
                self.num_experts,
                self.aux_loss_coef,
            )
        else:
            topk_indices, topk_weights = moe_routing(
                hidden_states,
                self.gate.weight.T,
                self.gate.bias,
                self.experts_per_token,
            )
            aux_loss = None

        # Restore shape if needed
        if batch is not None:
            topk_indices = topk_indices.view(batch, seq_len, self.experts_per_token)
            topk_weights = topk_weights.view(batch, seq_len, self.experts_per_token)

        return topk_indices, topk_weights, aux_loss


def compute_expert_load_stats(
    topk_indices: torch.Tensor,
    num_experts: int,
) -> dict:
    """Compute expert load statistics for debugging/monitoring.

    Args:
        topk_indices: Selected expert indices. Shape: [num_tokens, k]
        num_experts: Total number of experts.

    Returns:
        Dictionary with load statistics:
        - counts: Tokens per expert
        - mean: Average tokens per expert
        - std: Standard deviation
        - min/max: Min/max tokens per expert
        - utilization: Fraction of experts used
    """
    flat_indices = topk_indices.view(-1)
    counts = torch.bincount(flat_indices, minlength=num_experts).float()

    return {
        "counts": counts,
        "mean": counts.mean().item(),
        "std": counts.std().item(),
        "min": counts.min().item(),
        "max": counts.max().item(),
        "utilization": (counts > 0).sum().item() / num_experts,
    }
