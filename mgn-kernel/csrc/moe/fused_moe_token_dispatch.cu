#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <vector>
#include <torch/extension.h>
#include <cub/cub.cuh>

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

// Warp size constant
constexpr int WARP_SIZE = 32;

/**
 * OPTIMIZED DISPATCH KERNEL
 * 
 * Key improvements:
 * 1. Cooperative warp-level token copying for coalesced memory access
 * 2. Each warp processes one (token, k) pair together
 * 3. Vectorized loads/stores when possible
 */
template <typename scalar_t, int THREADS_PER_TOKEN = 32>
__global__ void fused_moe_token_dispatch_kernel_optimized(
    const scalar_t* __restrict__ global_x,  // [num_tokens, hidden_size]
    const int32_t* __restrict__ topk_idx,   // [num_tokens, K]
    scalar_t* __restrict__ output_x,  // Output: local tokens
    int32_t* __restrict__ output_eids,  // Output: expert IDs
    int32_t* __restrict__ output_token_idx,  // Output: original token indices
    int32_t* __restrict__ output_topk_pos,   // Output: position in topk
    int32_t* __restrict__ expert_counters,   // Atomic counters per local expert
    const int32_t* __restrict__ expert_offsets,  // Prefix sum for write positions
    int32_t num_tokens, int32_t hidden_size, int32_t K,
    int32_t routed_expert_start_idx, int32_t routed_expert_end_idx) {
    
    // Each warp processes one (token, k) pair
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int num_warps = (blockDim.x * gridDim.x) / WARP_SIZE;
    
    for (int idx = warp_id; idx < num_tokens * K; idx += num_warps) {
        int token_id = idx / K;
        int topk_pos = idx % K;
        
        int expert_id = topk_idx[token_id * K + topk_pos];
        
        // Only process if expert is local
        if (expert_id >= routed_expert_start_idx && 
            expert_id < routed_expert_end_idx) {
            
            int local_expert_id = expert_id - routed_expert_start_idx;
            
            // Only one thread per warp does the atomic operation
            int write_pos;
            if (lane_id == 0) {
                int relative_pos = atomicAdd(&expert_counters[local_expert_id], 1);
                write_pos = expert_offsets[local_expert_id] + relative_pos;
            }
            // Broadcast write position to all threads in warp
            write_pos = __shfl_sync(0xffffffff, write_pos, 0);
            
            // Cooperative copying: each thread in warp copies a portion
            // This ensures coalesced memory access
            const scalar_t* src = global_x + token_id * hidden_size;
            scalar_t* dst = output_x + write_pos * hidden_size;
            
            // Each thread copies multiple elements with stride
            for (int h = lane_id; h < hidden_size; h += WARP_SIZE) {
                dst[h] = src[h];
            }
            
            // Only one thread writes metadata
            if (lane_id == 0) {
                output_eids[write_pos] = expert_id;
                output_token_idx[write_pos] = token_id;
                output_topk_pos[write_pos] = topk_pos;
            }
        }
    }
}

/**
 * Vectorized version for large hidden sizes (uses float4 for better throughput)
 * Requires hidden_size to be divisible by 4 (or 8 for larger types)
 */
template <typename scalar_t>
__global__ void fused_moe_token_dispatch_kernel_vectorized(
    const scalar_t* __restrict__ global_x,
    const int32_t* __restrict__ topk_idx,
    scalar_t* __restrict__ output_x,
    int32_t* __restrict__ output_eids,
    int32_t* __restrict__ output_token_idx,
    int32_t* __restrict__ output_topk_pos,
    int32_t* __restrict__ expert_counters,
    const int32_t* __restrict__ expert_offsets,
    int32_t num_tokens, int32_t hidden_size, int32_t K,
    int32_t routed_expert_start_idx, int32_t routed_expert_end_idx) {
    
    constexpr int VEC_SIZE = sizeof(float4) / sizeof(scalar_t);
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int num_warps = (blockDim.x * gridDim.x) / WARP_SIZE;
    const int hidden_size_vec = hidden_size / VEC_SIZE;
    
    for (int idx = warp_id; idx < num_tokens * K; idx += num_warps) {
        int token_id = idx / K;
        int topk_pos = idx % K;
        
        int expert_id = topk_idx[token_id * K + topk_pos];
        
        if (expert_id >= routed_expert_start_idx && 
            expert_id < routed_expert_end_idx) {
            
            int local_expert_id = expert_id - routed_expert_start_idx;
            
            int write_pos;
            if (lane_id == 0) {
                int relative_pos = atomicAdd(&expert_counters[local_expert_id], 1);
                write_pos = expert_offsets[local_expert_id] + relative_pos;
            }
            write_pos = __shfl_sync(0xffffffff, write_pos, 0);
            
            // Vectorized copying
            const float4* src = reinterpret_cast<const float4*>(
                global_x + token_id * hidden_size);
            float4* dst = reinterpret_cast<float4*>(
                output_x + write_pos * hidden_size);
            
            for (int h = lane_id; h < hidden_size_vec; h += WARP_SIZE) {
                dst[h] = src[h];
            }
            
            if (lane_id == 0) {
                output_eids[write_pos] = expert_id;
                output_token_idx[write_pos] = token_id;
                output_topk_pos[write_pos] = topk_pos;
            }
        }
    }
}

// Count kernel (unchanged, already efficient)
template <typename scalar_t>
__global__ void count_local_expert_tokens_kernel(
    const int32_t* __restrict__ topk_idx,
    int32_t* __restrict__ expert_counts,
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

/**
 * OPTIMIZED VERSION: No CPU-GPU sync!
 * 
 * Key changes:
 * 1. Pre-allocate outputs with upper bound (num_tokens * K)
 * 2. No blocking cudaMemcpy to get total count
 * 3. Return actual count as a tensor for async access if needed
 */
std::vector<torch::Tensor> fused_moe_token_dispatch_cuda(
    torch::Tensor global_x,  // [num_tokens, hidden_size]
    torch::Tensor topk_idx,  // [num_tokens, K]
    torch::Tensor token_idx,  // Unused artifact
    torch::Tensor topk_pos,   // Unused artifact
    int64_t routed_expert_start_idx, int64_t routed_expert_end_idx,
    bool use_vectorized = false) {
    
    const auto num_tokens = global_x.size(0);
    const auto hidden_size = global_x.size(1);
    const auto K = topk_idx.size(1);
    const auto num_local_experts = routed_expert_end_idx - routed_expert_start_idx;
    
    // Validate inputs
    TORCH_CHECK(global_x.is_cuda(), "global_x must be a CUDA tensor");
    TORCH_CHECK(topk_idx.is_cuda(), "topk_idx must be a CUDA tensor");
    TORCH_CHECK(global_x.is_contiguous(), "global_x must be contiguous");
    TORCH_CHECK(topk_idx.is_contiguous(), "topk_idx must be contiguous");
    
    // Step 1: Count tokens per local expert
    auto expert_counts = torch::zeros(
        {num_local_experts},
        torch::TensorOptions().dtype(torch::kInt32).device(global_x.device()));
    
    const int threads = 256;
    const int blocks = std::min(65535, (int)((num_tokens * K + threads - 1) / threads));
    
    DISPATCH_FLOAT_AND_HALF_AND_BF16(
        global_x, "count_local_expert_tokens", [&] {
            count_local_expert_tokens_kernel<scalar_t><<<blocks, threads>>>(
                topk_idx.data_ptr<int32_t>(), expert_counts.data_ptr<int32_t>(),
                num_tokens, K, static_cast<int32_t>(routed_expert_start_idx), 
                static_cast<int32_t>(routed_expert_end_idx));
        });
    
    // Step 2: Compute prefix sum (no sync needed!)
    auto expert_offsets = torch::empty(
        {num_local_experts + 1},
        torch::TensorOptions().dtype(torch::kInt32).device(global_x.device()));
    
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    
    cub::DeviceScan::ExclusiveSum(
        d_temp_storage, temp_storage_bytes,
        expert_counts.data_ptr<int32_t>(),
        expert_offsets.data_ptr<int32_t>(),
        num_local_experts + 1
    );
    
    auto temp_storage = torch::empty(
        {static_cast<int64_t>(temp_storage_bytes)},
        torch::TensorOptions().dtype(torch::kUInt8).device(global_x.device()));
    d_temp_storage = temp_storage.data_ptr<uint8_t>();
    
    cub::DeviceScan::ExclusiveSum(
        d_temp_storage, temp_storage_bytes,
        expert_counts.data_ptr<int32_t>(),
        expert_offsets.data_ptr<int32_t>(),
        num_local_experts + 1
    );
    
    // Step 3: Pre-allocate with UPPER BOUND - no sync needed!
    const int64_t max_local_tokens = num_tokens * K;
    
    auto output_x = torch::empty({max_local_tokens, hidden_size},
                                 torch::TensorOptions()
                                     .dtype(global_x.dtype())
                                     .device(global_x.device()));
    auto output_eids = torch::empty(
        {max_local_tokens},
        torch::TensorOptions().dtype(torch::kInt32).device(global_x.device()));
    auto output_token_idx = torch::empty(
        {max_local_tokens},
        torch::TensorOptions().dtype(torch::kInt32).device(global_x.device()));
    auto output_topk_pos = torch::empty(
        {max_local_tokens},
        torch::TensorOptions().dtype(torch::kInt32).device(global_x.device()));
    
    // Reset counters for dispatch kernel
    expert_counts.zero_();
    
    // Step 4: Launch OPTIMIZED dispatch kernel
    // Use warp-sized thread blocks (multiple of 32) for better efficiency
    const int opt_threads = 256;  // 8 warps per block
    const int opt_blocks = std::min(65535, 
                                    (int)((num_tokens * K * WARP_SIZE + opt_threads - 1) / opt_threads));
    
    DISPATCH_FLOAT_AND_HALF_AND_BF16(global_x, "fused_moe_token_dispatch", [&] {
        if (use_vectorized && hidden_size % 4 == 0) {
            fused_moe_token_dispatch_kernel_vectorized<scalar_t><<<opt_blocks, opt_threads>>>(
                global_x.data_ptr<scalar_t>(), topk_idx.data_ptr<int32_t>(),
                output_x.data_ptr<scalar_t>(), output_eids.data_ptr<int32_t>(),
                output_token_idx.data_ptr<int32_t>(),
                output_topk_pos.data_ptr<int32_t>(),
                expert_counts.data_ptr<int32_t>(),
                expert_offsets.data_ptr<int32_t>(), 
                num_tokens, hidden_size, K,
                static_cast<int32_t>(routed_expert_start_idx), 
                static_cast<int32_t>(routed_expert_end_idx));
        } else {
            fused_moe_token_dispatch_kernel_optimized<scalar_t, 32><<<opt_blocks, opt_threads>>>(
                global_x.data_ptr<scalar_t>(), topk_idx.data_ptr<int32_t>(),
                output_x.data_ptr<scalar_t>(), output_eids.data_ptr<int32_t>(),
                output_token_idx.data_ptr<int32_t>(),
                output_topk_pos.data_ptr<int32_t>(),
                expert_counts.data_ptr<int32_t>(),
                expert_offsets.data_ptr<int32_t>(), 
                num_tokens, hidden_size, K,
                static_cast<int32_t>(routed_expert_start_idx), 
                static_cast<int32_t>(routed_expert_end_idx));
        }
    });
    
    // Optional: If you need exact-sized tensors, slice them
    // For now, return with upper bound and let caller handle
    // You can get actual size from expert_offsets[-1] on device if needed
    
    return {output_x, output_eids, output_token_idx, output_topk_pos, expert_counts};
}

// Binding for PyTorch
// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//     m.def("fused_moe_token_dispatch_optimized", &fused_moe_token_dispatch_cuda_optimized,
//           "Optimized MoE Token Dispatch (CUDA)");
// }
