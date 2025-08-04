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
    # This fused kernel function is used to select topk expert in a hierarchical 2-layer fashion
    # it split group of expert into num_expert_group, and use top2 expert weight sum in each group
    # as the group weight to select expert groups and then select topk experts within the selected groups
    # the #experts is decided by the input tensor shape and we currently only support power of 2 #experts
    # and #experts should be divisible by num_expert_group. #expert/num_expert_group <= 32 is limited for now.
    # for non-supported case, we suggest to use the biased_grouped_topk func in sglang.srt.layers.moe.topk
    # num_fused_shared_experts: if > 0, the last several experts will be replaced with shared experts
    # routed_scaling_factor: if > 0, the shared experts will be scaled by this factor
    return torch.ops.mgn_kernel.moe_fused_gate.default(
        input_tensor,
        bias,
        num_expert_group,
        topk_group,
        topk,
        num_fused_shared_experts,
        routed_scaling_factor,
    )