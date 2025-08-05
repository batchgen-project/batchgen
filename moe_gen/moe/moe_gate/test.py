import torch
import torch.nn.functional as F
import fused_moe_gate  # Import our custom module

# Original PyTorch function for reference
def original_moe_gate_forward(hidden_states, weight, e_score_correction_bias,
                              n_group, topk_group, n_routed_experts, top_k,
                              routed_scaling_factor):
    bsz, seq_len, h = hidden_states.shape
    hidden_states = hidden_states.view(-1, h)
    logits = F.linear(
        hidden_states.type(torch.float32), weight.type(torch.float32), None
    )
    scores = logits.sigmoid()

    scores_for_choice = scores.view(bsz * seq_len, -1) + e_score_correction_bias.unsqueeze(0).to(torch.float32)
    group_scores = (
        scores_for_choice.view(bsz * seq_len, n_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
    )
    group_idx = torch.topk(
        group_scores, k=topk_group, dim=-1, sorted=False
    )[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(
            bsz * seq_len, n_group, n_routed_experts // n_group
        )
        .reshape(bsz * seq_len, -1)
    )
    tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))
    
    # Get unsorted indices from topk
    _, topk_idx_unsorted = torch.topk(
        tmp_scores, k=top_k, dim=-1, sorted=False
    )
    
    # Gather the weights corresponding to the unsorted indices
    topk_weight_unsorted = scores.gather(1, topk_idx_unsorted)

    denominator = topk_weight_unsorted.sum(dim=-1, keepdim=True) + 1e-20
    topk_weight_scaled = (topk_weight_unsorted / denominator) * routed_scaling_factor

    return topk_idx_unsorted, topk_weight_scaled.to(hidden_states.dtype)


# --- Test Parameters ---
BSZ = 4
SEQ_LEN = 1024
H = 128
N_ROUTED_EXPERTS = 64
N_GROUP = 4
TOPK_GROUP = 2
TOP_K = 4
ROUTED_SCALING_FACTOR = 1.0

# --- Create Tensors ---
device = 'cuda'
dtype = torch.bfloat16

hidden_states = torch.randn(BSZ, SEQ_LEN, H, device=device, dtype=dtype)
weight = torch.randn(N_ROUTED_EXPERTS, H, device=device, dtype=dtype)
e_score_correction_bias = torch.randn(N_ROUTED_EXPERTS, device=device, dtype=dtype)

# --- Run Original PyTorch Function ---
print("Running original PyTorch function...")
torch_topk_idx_unsorted, torch_topk_weight = original_moe_gate_forward(
    hidden_states, weight, e_score_correction_bias,
    N_GROUP, TOPK_GROUP, N_ROUTED_EXPERTS, TOP_K, ROUTED_SCALING_FACTOR
)
print("Done.")

# --- Run Fused CUDA Kernel ---
print("\nRunning fused CUDA kernel...")
hidden_states_2d = hidden_states.view(-1, H)
cuda_topk_idx_unsorted, cuda_topk_weight = fused_moe_gate.forward(
    hidden_states_2d, weight, e_score_correction_bias,
    N_GROUP, TOPK_GROUP, N_ROUTED_EXPERTS, TOP_K, ROUTED_SCALING_FACTOR
)
print("Done.")

# --- Verification ---
print("\nVerifying results...")

# To compare correctly, we must sort the indices from BOTH implementations
# and then gather the weights in that sorted order.
torch_topk_idx_sorted, _ = torch.sort(torch_topk_idx_unsorted, dim=-1)
cuda_topk_idx_sorted, _ = torch.sort(cuda_topk_idx_unsorted, dim=-1)

# Check indices
idx_match = torch.all(torch_topk_idx_sorted == cuda_topk_idx_sorted)
print(f"Sorted Top-K Indices Match: {idx_match}")
if not idx_match:
    print("Mismatched sorted indices found. This indicates a functional difference.")
    # Find the first mismatch for debugging
    mismatch_mask = (torch_topk_idx_sorted != cuda_topk_idx_sorted)
    first_mismatch_row = torch.where(mismatch_mask.any(dim=1))[0][0].item()
    print(f"First mismatch at token index: {first_mismatch_row}")
    print("PyTorch sorted indices:", torch_topk_idx_sorted[first_mismatch_row])
    print("CUDA sorted indices:   ", cuda_topk_idx_sorted[first_mismatch_row])


# Now, gather the weights using the UNSORTED indices and compare them
# after sorting by the indices. This is a robust way to check.
# Create a common sorted order for weights
torch_weight_sorted = torch.gather(torch_topk_weight, 1, torch.argsort(torch_topk_idx_unsorted, dim=-1))
cuda_weight_sorted = torch.gather(cuda_topk_weight, 1, torch.argsort(cuda_topk_idx_unsorted, dim=-1))


weights_match = torch.allclose(torch_weight_sorted.float(), cuda_weight_sorted.float(), atol=1e-3, rtol=1e-2)
print(f"Sorted Top-K Weights Match: {weights_match}")
if not weights_match:
    print("\nMismatched sorted weights found.")
    diff = torch.abs(torch_weight_sorted.float() - cuda_weight_sorted.float())
    max_diff, _ = torch.max(diff, dim=1)
    first_mismatch_row = torch.argmax(max_diff).item()
    
    print(f"First significant weight mismatch at token index: {first_mismatch_row}")
    print("PyTorch sorted weights:", torch_weight_sorted[first_mismatch_row].float())
    print("CUDA sorted weights:   ", cuda_weight_sorted[first_mismatch_row].float())
    print(f"Max absolute difference: {torch.max(diff)}")

if idx_match and weights_match:
    print("\nSUCCESS: The outputs of the fused CUDA kernel and the PyTorch function are consistent.")