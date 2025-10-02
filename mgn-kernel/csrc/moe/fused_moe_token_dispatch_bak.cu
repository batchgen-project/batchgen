#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <vector>
#include <torch/all.h>

// Macro for dispatching including bfloat16 support
#define DISPATCH_FLOAT_AND_HALF_AND_BF16(TYPE, NAME, ...)                      \
    [&] {                                                                      \
        const auto& the_type = TYPE;                                           \
        at::ScalarType _st = the_type.scalar_type();                           \
        switch (_st) {                                                         \
            case at::ScalarType::Double: {                                     \
                using scalar_t = double;                                       \
                return __VA_ARGS__();                                          \
            }                                                                  \
            case at::ScalarType::Float: {                                      \
                using scalar_t = float;                                        \
                return __VA_ARGS__();                                          \
            }                                                                  \
            case at::ScalarType::Half: {                                       \
                using scalar_t = at::Half;                                     \
                return __VA_ARGS__();                                          \
            }                                                                  \
            case at::ScalarType::BFloat16: {                                   \
                using scalar_t = at::BFloat16;                                 \
                return __VA_ARGS__();                                          \
            }                                                                  \
            default:                                                           \
                AT_ERROR(#NAME, " not implemented for '", toString(_st), "'"); \
        }                                                                      \
    }()

template <typename scalar_t>
__global__ void fused_moe_token_dispatch_kernel(
    const scalar_t* __restrict__ global_x,  // [num_tokens, hidden_size]
    const int32_t* __restrict__ topk_idx,   // [num_tokens, K]
    scalar_t* __restrict__ output_x,  // Output: local tokens grouped by expert
    int32_t* __restrict__ output_eids,  // Output: expert IDs for each token
    int32_t* __restrict__ output_token_idx,  // Output: original token indices
    int32_t* __restrict__ output_topk_pos,   // Output: position in topk
    int32_t* __restrict__ expert_counters,   // Atomic counters per local expert
    int32_t* __restrict__ expert_offsets,    // Prefix sum for write positions
    int32_t num_tokens, int32_t hidden_size, int32_t K,
    int32_t routed_expert_start_idx, int32_t routed_expert_end_idx) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    // Each thread processes multiple (token, k) pairs
    for (int idx = tid; idx < num_tokens * K; idx += stride) {
        int token_id = idx / K;  // Which original token
        int topk_pos = idx % K;  // Position in topk

        int expert_id = topk_idx[token_id * K + topk_pos];

        // Only process if expert is local
        if (expert_id >= routed_expert_start_idx &&
            expert_id < routed_expert_end_idx) {
            // Atomic increment to get write position within this expert's
            // section
            int local_expert_id = expert_id - routed_expert_start_idx;
            int relative_pos = atomicAdd(&expert_counters[local_expert_id], 1);
            int write_pos = expert_offsets[local_expert_id] + relative_pos;

            // Copy token data to output position
            for (int h = 0; h < hidden_size; h++) {
                output_x[write_pos * hidden_size + h] =
                    global_x[token_id * hidden_size + h];
            }

            // Store metadata
            output_eids[write_pos] = expert_id;
            output_token_idx[write_pos] = token_id;
            output_topk_pos[write_pos] = topk_pos;
        }
    }
}

// Count how many tokens are assigned to each local expert
template <typename scalar_t>
__global__ void count_local_expert_tokens_kernel(
    const int32_t* __restrict__ topk_idx,  // [num_tokens, K]
    int32_t* __restrict__ expert_counts,   // Output: count per local expert
    int32_t num_tokens, int32_t K, int32_t routed_expert_start_idx,
    int32_t routed_expert_end_idx) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int idx = tid; idx < num_tokens * K; idx += stride) {
        int expert_id = topk_idx[idx];

        if (expert_id >= routed_expert_start_idx &&
            expert_id < routed_expert_end_idx) {
            int local_expert_id = expert_id - routed_expert_start_idx;
            atomicAdd(&expert_counts[local_expert_id], 1);
        }
    }
}

std::vector<torch::Tensor> fused_moe_token_dispatch_cuda(
    torch::Tensor global_x,  // [num_tokens, hidden_size]
    torch::Tensor topk_idx,  // [num_tokens, K]
    torch::Tensor
        token_idx,  // [num_tokens * K] - original implementation artifact
    torch::Tensor
        topk_pos,  // [num_tokens * K] - original implementation artifact
    int64_t routed_expert_start_idx, int64_t routed_expert_end_idx) {
    const auto num_tokens = global_x.size(0);
    const auto hidden_size = global_x.size(1);
    const auto K = topk_idx.size(1);
    const auto num_local_experts =
        routed_expert_end_idx - routed_expert_start_idx;

    // Validate inputs
    TORCH_CHECK(global_x.is_cuda(), "global_x must be a CUDA tensor");
    TORCH_CHECK(topk_idx.is_cuda(), "topk_idx must be a CUDA tensor");
    TORCH_CHECK(global_x.dtype() == torch::kFloat32 ||
                    global_x.dtype() == torch::kFloat16 ||
                    global_x.dtype() == torch::kBFloat16,
                "global_x must be float32, float16, or bfloat16");
    TORCH_CHECK(topk_idx.dtype() == torch::kInt32, "topk_idx must be int32");

    // Step 1: Count tokens per local expert
    auto expert_counts = torch::zeros(
        {num_local_experts},
        torch::TensorOptions().dtype(torch::kInt32).device(global_x.device()));

    const int threads = 256;
    const int blocks = (num_tokens * K + threads - 1) / threads;

    DISPATCH_FLOAT_AND_HALF_AND_BF16(
        global_x, "count_local_expert_tokens", [&] {
            count_local_expert_tokens_kernel<scalar_t><<<blocks, threads>>>(
                topk_idx.data_ptr<int32_t>(), expert_counts.data_ptr<int32_t>(),
                num_tokens, K, static_cast<int32_t>(routed_expert_start_idx), static_cast<int32_t>(routed_expert_end_idx));
        });

    // Step 2: Compute prefix sum to get expert offsets
    auto expert_offsets = torch::zeros(
        {num_local_experts + 1},
        torch::TensorOptions().dtype(torch::kInt32).device(global_x.device()));

    // Copy counts to CPU for prefix sum (could be optimized with GPU scan)
    auto expert_counts_cpu = expert_counts.cpu();
    auto expert_offsets_cpu = expert_offsets.cpu();

    int total_local_tokens = 0;
    for (int i = 0; i < num_local_experts; i++) {
        expert_offsets_cpu[i] = total_local_tokens;
        total_local_tokens += expert_counts_cpu[i].item<int>();
    }
    expert_offsets_cpu[num_local_experts] = total_local_tokens;

    expert_offsets = expert_offsets_cpu.to(global_x.device());

    // Step 3: Allocate output tensors
    auto output_x = torch::empty({total_local_tokens, hidden_size},
                                 torch::TensorOptions()
                                     .dtype(global_x.dtype())
                                     .device(global_x.device()));
    auto output_eids = torch::empty(
        {total_local_tokens},
        torch::TensorOptions().dtype(torch::kInt32).device(global_x.device()));
    auto output_token_idx = torch::empty(
        {total_local_tokens},
        torch::TensorOptions().dtype(torch::kInt32).device(global_x.device()));
    auto output_topk_pos = torch::empty(
        {total_local_tokens},
        torch::TensorOptions().dtype(torch::kInt32).device(global_x.device()));

    // Reset expert counters for the main kernel
    expert_counts.zero_();

    // Step 4: Launch the main dispatch kernel
    DISPATCH_FLOAT_AND_HALF_AND_BF16(global_x, "fused_moe_token_dispatch", [&] {
        fused_moe_token_dispatch_kernel<scalar_t><<<blocks, threads>>>(
            global_x.data_ptr<scalar_t>(), topk_idx.data_ptr<int32_t>(),
            output_x.data_ptr<scalar_t>(), output_eids.data_ptr<int32_t>(),
            output_token_idx.data_ptr<int32_t>(),
            output_topk_pos.data_ptr<int32_t>(),
            expert_counts.data_ptr<int32_t>(),
            expert_offsets.data_ptr<int32_t>(), num_tokens, hidden_size, K,
            static_cast<int32_t>(routed_expert_start_idx), static_cast<int32_t>(routed_expert_end_idx));
    });

    cudaDeviceSynchronize();

    return {output_x, output_eids, output_token_idx, output_topk_pos,
            expert_counts};
}
