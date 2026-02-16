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
    torch::Tensor topk_weights    // optional pre-allocated
);

// Dispatch: count + prefix_sum + gather
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
    torch::Tensor topk_pos
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
    torch::Tensor output  // optional pre-allocated
);
