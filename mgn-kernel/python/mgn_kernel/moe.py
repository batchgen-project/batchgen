import torch


def moe_fused_gate(
    input_tensor,
    bias,
    num_expert_group,
    topk_group,
    topk,
    num_fused_shared_experts=0,
    routed_scaling_factor=0,
):
    """
    Fused kernel to select top-k experts in a hierarchical 2-level manner.

    Inputs:
        input_tensor:           (num_rows, num_experts) - Expert scores per token.
        bias:                   (num_experts,) - Bias vector added to expert scores.
        num_expert_group:       int - Number of expert groups to split experts into.
        topk_group:             int - Number of top groups to select based on group score.
        topk:                   int - Number of top experts to select across selected groups.
        num_fused_shared_experts: int - Number of shared experts to include (appended at the end).
        routed_scaling_factor: float - Scaling factor applied to shared expert weights.

    Requirements:
        - num_experts must be a power of 2.
        - num_experts must be divisible by num_expert_group.
        - num_experts / num_expert_group must be ≤ 32.

    Returns:
        output:    (num_rows, topk)        - Selected expert scores after softmax.
        indices:   (num_rows, topk)        - Indices of selected experts (including shared if any).

    Notes:
        - If num_fused_shared_experts > 0, they occupy the last column(s) in output/indices.
        - Shared expert weights are set to the sum of selected scores divided by routed_scaling_factor.
    """
    return torch.ops.mgn_kernel.moe_fused_gate.default(
        input_tensor,
        bias,
        num_expert_group,
        topk_group,
        topk,
        num_fused_shared_experts,
        routed_scaling_factor,
    )


def expert_bincount(
    eid: torch.Tensor,
    routed_expert_start_idx: int,
    experts_per_rank: int,
    device: torch.device,
):
    """
    Count the number of tokens routed to each expert.
    Args:
        eid (torch.Tensor): Tensor of expert IDs for each token.
        routed_expert_start_idx (int): Starting index for routed experts.
        experts_per_rank (int): Number of experts per rank.
        device (torch.device): Device to place the output tensor.
    """
    return torch.ops.mgn_kernel.expert_bincount.default(
        eid, routed_expert_start_idx, experts_per_rank, device
    )


def fused_moe_token_dispatch(
    global_x: torch.Tensor,
    topk_idx: torch.Tensor,
    token_idx: torch.Tensor,
    topk_pos: torch.Tensor,
    routed_expert_start_idx: int,
    routed_expert_end_idx: int,
):
    """
    Dispatch tokens to their respective experts based on top-k indices.

    Args:
        global_x (torch.Tensor): Global input tensor of shape (num_tokens, hidden_size).
        topk_idx (torch.Tensor): Indices of the top-k experts for each token.
        token_idx (torch.Tensor): Indices of the tokens.
        topk_pos (torch.Tensor): Positions of the top-k experts.
        routed_expert_start_idx (int): Starting index for routed experts.
        routed_expert_end_idx (int): Ending index for routed experts.
    """
    return torch.ops.mgn_kernel.fused_moe_token_dispatch.default(
        global_x,
        topk_idx,
        token_idx,
        topk_pos,
        routed_expert_start_idx,
        routed_expert_end_idx,
    )
