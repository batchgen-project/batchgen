import torch
from mgn_kernel import moe_fused_gate

seq_length = 8
num_experts = 16
num_expert_group = 4
topk_group = 1
topk = 2
num_fused_shared_experts = 1

# ----------------------------
torch.manual_seed(42)
dtype = torch.bfloat16

# input
tensor = torch.rand((seq_length, num_experts), dtype=dtype, device="cuda")

# bias: expert bias
bias = torch.rand((num_experts,), dtype=dtype, device="cuda")

# ----------------------------
# call moe_fused_gate
output, indices = moe_fused_gate(
    tensor,
    bias,
    num_expert_group=num_expert_group,
    topk_group=topk_group,
    topk=topk + num_fused_shared_experts,  # 加上共享 expert
    num_fused_shared_experts=num_fused_shared_experts,
    routed_scaling_factor=2.5,
)

# ----------------------------
# print
print("Input Tensor Shape:", tensor.shape)
print("Bias Shape:", bias.shape)
print("Output Shape:", output.shape)
print("Indices Shape:", indices.shape)

print("Output:\n", output)
print("Indices:\n", indices)