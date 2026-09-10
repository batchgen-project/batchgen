/*
 * GPT-OSS-120B MoE Routing CUDA Kernels.
 *
 * Declarations for gate, dispatch, and reduce CUDA kernels.
 * These are CUDA equivalents of the Triton routing kernels
 * for benchmarking comparison.
 */

#pragma once

#include <torch/extension.h>
#include <vector>

// Gate: fused top-k selection + softmax
// Input:  router_logits [N, E] FP32
// Output: topk_indices [N, K] int32, topk_weights [N, K] FP32
std::vector<torch::Tensor> gate_topk_softmax_cuda(
    torch::Tensor router_logits,
    int k,
    torch::Tensor topk_indices,   // optional pre-allocated
    torch::Tensor topk_weights,   // optional pre-allocated
    int64_t num_valid_tokens = -1 // -1 = process all N
);

// Dispatch: count+prefix_sum + gather (2 fused kernels)
// Input:  x [N, H] BF16, topk_indices [N, K] int32
// Output: dispatched_x, expert_counts, expert_offsets, topk_pos
std::vector<torch::Tensor> dispatch_count_gather_cuda(
    torch::Tensor x,
    torch::Tensor topk_indices,
    int64_t expert_start,
    int64_t num_local_experts,
    // Pre-allocated outputs (optional)
    torch::Tensor expert_counts,
    torch::Tensor expert_offsets,
    torch::Tensor expert_counters,
    torch::Tensor dispatched_x,
    torch::Tensor topk_pos,
    int64_t num_valid_tokens = -1  // -1 = process all N
);

// Router epilogue: fused BF16 bias add + BF16→FP32 cast
// Input:  logits [N, E] BF16, bias [E] BF16 (or empty)
// Output: output [N, E] FP32
void router_bias_cast_cuda(
    torch::Tensor logits,
    torch::Tensor bias,
    torch::Tensor output
);

// Gate: fused sigmoid + top-k + normalize + scale (K2.5, K3)
// Input:  router_logits [N, E] FP32 (row stride may exceed E),
//         e_score_correction [E] FP32
// Output: topk_indices [N, K] int32, topk_weights [N, K] FP32
std::vector<torch::Tensor> gate_sigmoid_topk_cuda(
    torch::Tensor router_logits,
    torch::Tensor e_score_correction,
    int k,
    float routed_scaling_factor,
    torch::Tensor topk_indices,       // optional pre-allocated
    torch::Tensor topk_weights,       // optional pre-allocated
    torch::Tensor num_valid_tokens,   // optional device int32 scalar; rows beyond
                                      // it get idx=-1 / weight=0. An undefined or
                                      // empty tensor means "all rows"; the Python
                                      // binding exposes this as c10::optional.
    torch::Tensor latent_out,         // optional [N, L] BF16 pre-allocated: K3's
                                      // latent suffix of the same fused GEMM row,
                                      // cast in place. Undefined/empty disables it.
    int64_t latent_offset             // column where the latent suffix starts
);

// GLM-5 router GEMM: BF16 hidden x BF16 weight^T -> FP32 logits.
// Uses device-side rank_token_counts for graph-stable rank-major padding masks.
torch::Tensor glm5_router_gemm_cuda(
    torch::Tensor hidden_states,   // [N, H] BF16
    torch::Tensor router_weight,   // [E, H] BF16
    torch::Tensor rank_token_counts, // [world_size] int64 or empty
    torch::Tensor output,          // [N, E] FP32 pre-allocated (optional)
    int64_t bucket_size,
    int64_t world_size
);

// Fused Gate: WGMMA GEMM + bias + TopK + Softmax (SM90a, 2 kernels)
// Requires FusedGateContext for cached weight transpose + TMA descriptors.
int64_t create_fused_gate_context(
    torch::Tensor router_weight,    // [E, K_dim] BF16 (nn.Linear weight)
    torch::Tensor router_bias,      // [E] BF16 (or empty)
    int topk
);

void destroy_fused_gate_context(int64_t ctx_ptr);

// Pre-create TMA descriptor for input buffer (call before CUDA graph capture)
void fused_gate_warmup(int64_t ctx_ptr, torch::Tensor hidden_states);

std::vector<torch::Tensor> fused_gate_forward(
    int64_t ctx_ptr,
    torch::Tensor hidden_states,     // [N, K_dim] BF16
    torch::Tensor logits,            // [N, E] FP32 pre-allocated (optional)
    torch::Tensor topk_indices,      // [N, topk] int32 pre-allocated (optional)
    torch::Tensor topk_weights,      // [N, topk] FP32 pre-allocated (optional)
    int64_t num_valid_tokens = -1
);

// Reduce: weighted scatter-add
// Input:  expert_output [max_disp, H] BF16, topk_pos [N*K] int32,
//         topk_weights [N, K] FP32
// Output: output [N, H] BF16
torch::Tensor reduce_weighted_scatter_cuda(
    torch::Tensor expert_output,
    torch::Tensor topk_pos,
    torch::Tensor topk_weights,
    int64_t N,
    int64_t H,
    int64_t K,
    torch::Tensor output,  // optional pre-allocated
    int64_t num_valid_tokens = -1  // -1 = process all N
);
