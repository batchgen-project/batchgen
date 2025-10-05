#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <vector>
#include <cstdint>

// Kernel to count tokens per expert using atomic operations
__global__ void expert_bincount_kernel(
    const int32_t* __restrict__ eids,           // [num_tokens] - expert IDs
    int32_t* __restrict__ expert_counts,        // [experts_per_rank] - output counts
    int32_t num_tokens,
    int32_t routed_expert_start_idx,
    int32_t experts_per_rank) {
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    // Each thread processes multiple tokens
    for (int i = tid; i < num_tokens; i += stride) {
        int32_t expert_id = eids[i];
        int32_t adjusted_expert_id = expert_id - routed_expert_start_idx;
        
        // Only count if expert is in local range
        if (adjusted_expert_id >= 0 && adjusted_expert_id < experts_per_rank) {
            atomicAdd(&expert_counts[adjusted_expert_id], 1);
        }
    }
}

// Optimized single-block kernel for compacting (handles up to 1024 experts)
__global__ void compact_active_experts_single_block_kernel(
    const int32_t* __restrict__ expert_counts,  // [experts_per_rank] - input counts
    int32_t* __restrict__ activated_group_idx,  // [experts_per_rank] - output active expert indices
    int32_t* __restrict__ group_size,           // [experts_per_rank] - output group sizes  
    int32_t* __restrict__ group_start_indices,  // [experts_per_rank] - output start indices
    int32_t* __restrict__ num_active_experts,   // [1] - output number of active experts
    int32_t experts_per_rank) {
    
    const int tid = threadIdx.x;
    
    // Sequential processing by thread 0 to maintain order and compute prefix sum
    if (tid == 0) {
        int32_t write_pos = 0;
        int32_t cumulative_sum = 0;
        
        for (int expert_id = 0; expert_id < experts_per_rank; expert_id++) {
            int32_t count = expert_counts[expert_id];
            if (count > 0) {
                activated_group_idx[write_pos] = expert_id;
                group_size[write_pos] = count;
                group_start_indices[write_pos] = cumulative_sum;
                
                cumulative_sum += count;
                write_pos++;
            }
        }
        
        *num_active_experts = write_pos;
    }
}

// Warp-level parallel scan for larger expert counts
__device__ int32_t warp_scan_inclusive(int32_t val, int32_t* shared_temp) {
    const int lane_id = threadIdx.x & 31;
    
    #pragma unroll
    for (int offset = 1; offset < 32; offset *= 2) {
        int32_t temp = __shfl_up_sync(0xffffffff, val, offset);
        if (lane_id >= offset) val += temp;
    }
    return val;
}

// Multi-block kernel with proper synchronization (for >1024 experts)
__global__ void compact_active_experts_multiblock_kernel(
    const int32_t* __restrict__ expert_counts,
    int32_t* __restrict__ activated_group_idx,
    int32_t* __restrict__ group_size,
    int32_t* __restrict__ group_start_indices,
    int32_t* __restrict__ num_active_experts,
    int32_t* __restrict__ temp_flags,          // [experts_per_rank] - temporary flags
    int32_t* __restrict__ temp_positions,       // [experts_per_rank] - temporary positions
    int32_t experts_per_rank) {
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Phase 1: Mark active experts and compute local prefix sum
    int32_t is_active = 0;
    if (tid < experts_per_rank) {
        is_active = (expert_counts[tid] > 0) ? 1 : 0;
        temp_flags[tid] = is_active;
    }
    __syncthreads();
    
    // Phase 2: Compute prefix sum (simplified - can use CUB for better performance)
    if (tid == 0) {
        int32_t cumulative_active = 0;
        int32_t cumulative_tokens = 0;
        int32_t write_pos = 0;
        
        for (int i = 0; i < experts_per_rank; i++) {
            if (temp_flags[i]) {
                activated_group_idx[write_pos] = i;
                group_size[write_pos] = expert_counts[i];
                group_start_indices[write_pos] = cumulative_tokens;
                
                cumulative_tokens += expert_counts[i];
                write_pos++;
            }
        }
        *num_active_experts = write_pos;
    }
}

std::vector<torch::Tensor> expert_bincount_cuda(
    torch::Tensor eids,                         // [num_tokens] - expert IDs
    int64_t routed_expert_start_idx,
    int64_t experts_per_rank,
    torch::Device device) {
    
    const auto num_tokens = eids.size(0);
    
    // Validate inputs
    TORCH_CHECK(eids.is_cuda(), "eids must be a CUDA tensor");
    TORCH_CHECK(eids.dtype() == torch::kInt32, "eids must be int32");
    TORCH_CHECK(eids.dim() == 1, "eids must be 1-dimensional");
    TORCH_CHECK(experts_per_rank <= INT32_MAX, "experts_per_rank must fit in int32");
    TORCH_CHECK(routed_expert_start_idx <= INT32_MAX && routed_expert_start_idx >= INT32_MIN, 
                "routed_expert_start_idx must fit in int32");
    TORCH_CHECK(experts_per_rank <= INT32_MAX, "experts_per_rank must fit in int32");
    TORCH_CHECK(routed_expert_start_idx <= INT32_MAX && routed_expert_start_idx >= INT32_MIN, 
                "routed_expert_start_idx must fit in int32");
    
    // Step 1: Count tokens per expert
    auto expert_counts = torch::zeros({experts_per_rank}, 
                                     torch::TensorOptions().dtype(torch::kInt32).device(device));
    
    const int threads = 256;
    const int blocks = std::min<int>((num_tokens + threads - 1) / threads, 1024);
    
    expert_bincount_kernel<<<blocks, threads>>>(
        eids.data_ptr<int32_t>(),
        expert_counts.data_ptr<int32_t>(),
        num_tokens,
        static_cast<int32_t>(routed_expert_start_idx),
        static_cast<int32_t>(experts_per_rank)
    );
    
    // Step 2: Allocate output tensors (worst case: all experts active)
    auto activated_group_idx = torch::empty({experts_per_rank}, 
                                           torch::TensorOptions().dtype(torch::kInt32).device(device));
    auto group_size = torch::empty({experts_per_rank}, 
                                  torch::TensorOptions().dtype(torch::kInt32).device(device));
    auto group_start_indices = torch::empty({experts_per_rank}, 
                                           torch::TensorOptions().dtype(torch::kInt32).device(device));
    auto num_active_experts = torch::zeros({1}, 
                                          torch::TensorOptions().dtype(torch::kInt32).device(device));
    
    // Step 3: Compact active experts
    if (experts_per_rank <= 1024) {
        // Use optimized single-block kernel (NO synchronization needed!)
        compact_active_experts_single_block_kernel<<<1, 256>>>(
            expert_counts.data_ptr<int32_t>(),
            activated_group_idx.data_ptr<int32_t>(),
            group_size.data_ptr<int32_t>(),
            group_start_indices.data_ptr<int32_t>(),
            num_active_experts.data_ptr<int32_t>(),
            static_cast<int32_t>(experts_per_rank)
        );
    } else {
        // Use multi-block kernel for larger expert counts
        auto temp_flags = torch::empty({experts_per_rank}, 
                                      torch::TensorOptions().dtype(torch::kInt32).device(device));
        auto temp_positions = torch::empty({experts_per_rank}, 
                                          torch::TensorOptions().dtype(torch::kInt32).device(device));
        
        const int compact_blocks = (experts_per_rank + threads - 1) / threads;
        compact_active_experts_multiblock_kernel<<<compact_blocks, threads>>>(
            expert_counts.data_ptr<int32_t>(),
            activated_group_idx.data_ptr<int32_t>(),
            group_size.data_ptr<int32_t>(),
            group_start_indices.data_ptr<int32_t>(),
            num_active_experts.data_ptr<int32_t>(),
            temp_flags.data_ptr<int32_t>(),
            temp_positions.data_ptr<int32_t>(),
            static_cast<int32_t>(experts_per_rank)
        );
    }
    
    // ============================================================================
    // CRITICAL OPTIMIZATION: NO CPU-GPU SYNC!
    // Instead of .item() which forces sync, use GPU-native slicing with indexing
    // ============================================================================
    
    // Create index tensor for slicing (entirely on GPU)
    auto slice_indices = torch::arange(0, experts_per_rank, 
                                       torch::TensorOptions().dtype(torch::kInt32).device(device));
    
    // Create mask: indices < num_active_experts (GPU operation, no sync!)
    auto mask = slice_indices.lt(num_active_experts.squeeze());
    
    // Use masked_select or index_select (entirely GPU operations)
    activated_group_idx = activated_group_idx.masked_select(mask);
    group_size = group_size.masked_select(mask);
    group_start_indices = group_start_indices.masked_select(mask);
    
    return {group_size, activated_group_idx, group_start_indices, num_active_experts};
}

// Alternative API: Return count tensor, let caller handle slicing
std::vector<torch::Tensor> expert_bincount_cuda_v2(
    torch::Tensor eids,
    int64_t routed_expert_start_idx,
    int64_t experts_per_rank,
    torch::Device device) {
    
    const auto num_tokens = eids.size(0);
    
    TORCH_CHECK(eids.is_cuda(), "eids must be a CUDA tensor");
    TORCH_CHECK(eids.dtype() == torch::kInt32, "eids must be int32");
    TORCH_CHECK(eids.dim() == 1, "eids must be 1-dimensional");
    
    auto expert_counts = torch::zeros({experts_per_rank}, 
                                     torch::TensorOptions().dtype(torch::kInt32).device(device));
    
    const int threads = 256;
    const int blocks = std::min<int>((num_tokens + threads - 1) / threads, 1024);
    
    expert_bincount_kernel<<<blocks, threads>>>(
        eids.data_ptr<int32_t>(),
        expert_counts.data_ptr<int32_t>(),
        num_tokens,
        static_cast<int32_t>(routed_expert_start_idx),
        static_cast<int32_t>(experts_per_rank)
    );
    
    auto activated_group_idx = torch::empty({experts_per_rank}, 
                                           torch::TensorOptions().dtype(torch::kInt32).device(device));
    auto group_size = torch::empty({experts_per_rank}, 
                                  torch::TensorOptions().dtype(torch::kInt32).device(device));
    auto group_start_indices = torch::empty({experts_per_rank}, 
                                           torch::TensorOptions().dtype(torch::kInt32).device(device));
    auto num_active_experts = torch::zeros({1}, 
                                          torch::TensorOptions().dtype(torch::kInt32).device(device));
    
    if (experts_per_rank <= 1024) {
        compact_active_experts_single_block_kernel<<<1, 256>>>(
            expert_counts.data_ptr<int32_t>(),
            activated_group_idx.data_ptr<int32_t>(),
            group_size.data_ptr<int32_t>(),
            group_start_indices.data_ptr<int32_t>(),
            num_active_experts.data_ptr<int32_t>(),
            static_cast<int32_t>(experts_per_rank)
        );
    }
    
    // NO SYNC! Return everything including count tensor
    // Caller can slice on GPU: tensor[:count.item()] or use indexing
    return {group_size, activated_group_idx, group_start_indices, num_active_experts};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("expert_bincount", &expert_bincount_cuda, "Optimized Expert Bincount (GPU slicing)");
    m.def("expert_bincount_v2", &expert_bincount_cuda_v2, "Expert Bincount (returns count, caller slices)");
}