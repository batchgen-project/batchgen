"""Lazy-loading Python wrappers for MGN (MoE General Native) CUDA kernels.

Provides the same API as the original mgn_kernel Python package, but loads
the pre-compiled extension via batchgen_kernels.load_extension().

Usage:
    from batchgen_kernels.common.mgn import (
        moe_fused_gate, expert_bincount, fused_moe_token_dispatch,
        compact_expert_data, fused_rmsnorm,
    )
"""

from typing import List, Tuple

import torch

_mgn_ext = None


def _get_mgn():
    global _mgn_ext
    if _mgn_ext is None:
        from batchgen_kernels import load_extension
        _mgn_ext = load_extension("batchgen_kernels.common._C_mgn_ops")
    return _mgn_ext


# ---------------------------------------------------------------------------
# MoE routing / gating
# ---------------------------------------------------------------------------

def moe_fused_gate(
    input_tensor: torch.Tensor,
    bias: torch.Tensor,
    num_expert_group: int,
    topk_group: int,
    topk: int,
    num_fused_shared_experts: int = 0,
    routed_scaling_factor: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused kernel to select top-k experts in a hierarchical 2-level manner.

    Args:
        input_tensor: (num_rows, num_experts) - Expert scores per token.
        bias: (num_experts,) - Bias vector added to expert scores.
        num_expert_group: Number of expert groups.
        topk_group: Number of top groups to select.
        topk: Number of top experts to select across selected groups.
        num_fused_shared_experts: Number of shared experts (unused, kept for API compat).
        routed_scaling_factor: Scaling factor for weight normalization.

    Returns:
        (topk_indices, topk_weights) each of shape (num_rows, topk).
    """
    ext = _get_mgn()
    n_routed_experts = input_tensor.size(1)
    return ext.moe_fused_gate(
        input_tensor,
        bias,
        num_expert_group,
        topk_group,
        n_routed_experts,
        topk,
        routed_scaling_factor,
    )


def expert_bincount(
    eid: torch.Tensor,
    routed_expert_start_idx: int,
    experts_per_rank: int,
    device: torch.device,
) -> List[torch.Tensor]:
    """Count the number of tokens routed to each expert.

    Args:
        eid: Tensor of expert IDs for each token.
        routed_expert_start_idx: Starting index for routed experts.
        experts_per_rank: Number of experts per rank.
        device: Device to place the output tensor.

    Returns:
        [group_size, activated_group_idx, group_start_indices, num_active_experts]
    """
    ext = _get_mgn()
    return ext.expert_bincount(eid, routed_expert_start_idx, experts_per_rank, device)


def fused_moe_token_dispatch(
    global_x: torch.Tensor,
    topk_idx: torch.Tensor,
    token_idx: torch.Tensor,
    topk_pos: torch.Tensor,
    routed_expert_start_idx: int,
    routed_expert_end_idx: int,
) -> List[torch.Tensor]:
    """Dispatch tokens to their respective experts based on top-k indices.

    Args:
        global_x: (num_tokens, hidden_size) input tensor.
        topk_idx: Indices of the top-k experts for each token.
        token_idx: Indices of the tokens.
        topk_pos: Positions of the top-k experts.
        routed_expert_start_idx: Starting index for routed experts.
        routed_expert_end_idx: Ending index for routed experts.

    Returns:
        [output_x, output_eids, output_token_idx, output_topk_pos,
         expert_counts, expert_offsets]
    """
    ext = _get_mgn()
    return ext.fused_moe_token_dispatch(
        global_x, topk_idx, token_idx, topk_pos,
        routed_expert_start_idx, routed_expert_end_idx,
    )


def compact_expert_data(
    expert_counts: torch.Tensor,
) -> List[torch.Tensor]:
    """Compact expert data by removing experts with zero counts.

    Args:
        expert_counts: Tensor containing counts of tokens per expert.

    Returns:
        [group_size, activated_group_idx, group_start_indices, num_active_experts]
    """
    ext = _get_mgn()
    return ext.compact_expert_data(expert_counts)


# ---------------------------------------------------------------------------
# Elementwise
# ---------------------------------------------------------------------------

def fused_rmsnorm(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Fused RMS normalization kernel.

    Args:
        input: Input tensor of shape (..., hidden_size).
        weight: Weight tensor of shape (hidden_size,).
        eps: Small value to avoid division by zero.

    Returns:
        Normalized output tensor.
    """
    ext = _get_mgn()
    return ext.fused_rmsnorm(input, weight, eps)
