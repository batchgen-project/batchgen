#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <vector>
#include <torch/all.h>

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

// Kernel to compact active experts and compute start indices
__global__ void compact_active_experts_kernel(
    const int32_t* __restrict__ expert_counts,  // [experts_per_rank] - input counts
    int32_t* __restrict__ activated_group_idx,  // [num_active] - output active expert indices
    int32_t* __restrict__ group_size,           // [num_active] - output group sizes
    int32_t* __restrict__ group_start_indices,  // [num_active] - output start indices
    int32_t* __restrict__ num_active_experts,   // [1] - output number of active experts
    int32_t experts_per_rank) {
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Use shared memory for reduction
    __shared__ int32_t shared_active_count;
    __shared__ int32_t shared_write_pos;
    
    if (threadIdx.x == 0) {
        shared_active_count = 0;
        shared_write_pos = 0;
    }
    __syncthreads();
    
    // First pass: count active experts
    if (tid < experts_per_rank) {
        if (expert_counts[tid] > 0) {
            atomicAdd(&shared_active_count, 1);
        }
    }
    __syncthreads();
    
    // Write the total count
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *num_active_experts = shared_active_count;
    }
    
    // Second pass: compact and compute start indices
    int32_t cumulative_sum = 0;
    for (int expert_id = 0; expert_id < experts_per_rank; expert_id++) {
        if (expert_counts[expert_id] > 0) {
            int write_pos = atomicAdd(&shared_write_pos, 1);
            
            if (tid == 0) {  // Only thread 0 writes to avoid race conditions
                activated_group_idx[write_pos] = expert_id;
                group_size[write_pos] = expert_counts[expert_id];
                group_start_indices[write_pos] = cumulative_sum;
                cumulative_sum += expert_counts[expert_id];
            }
        }
    }
}

// Optimized single-block kernel for small expert counts (more efficient)
__global__ void compact_active_experts_single_block_kernel(
    const int32_t* __restrict__ expert_counts,  // [experts_per_rank] - input counts
    int32_t* __restrict__ activated_group_idx,  // [num_active] - output active expert indices
    int32_t* __restrict__ group_size,           // [num_active] - output group sizes  
    int32_t* __restrict__ group_start_indices,  // [num_active] - output start indices
    int32_t* __restrict__ num_active_experts,   // [1] - output number of active experts
    int32_t experts_per_rank) {
    
    const int tid = threadIdx.x;
    
    __shared__ int32_t shared_write_pos;
    __shared__ int32_t shared_prefix_sum;
    
    if (tid == 0) {
        shared_write_pos = 0;
        shared_prefix_sum = 0;
    }
    __syncthreads();
    
    // Sequential processing to maintain order and compute prefix sum correctly
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
    
    // Step 1: Count tokens per expert
    auto expert_counts = torch::zeros({experts_per_rank}, 
                                     torch::TensorOptions().dtype(torch::kInt32).device(device));
    
    const int threads = 256;
    const int blocks = (num_tokens + threads - 1) / threads;
    
    expert_bincount_kernel<<<blocks, threads>>>(
        eids.data_ptr<int32_t>(),
        expert_counts.data_ptr<int32_t>(),
        num_tokens,
        static_cast<int32_t>(routed_expert_start_idx),
        static_cast<int32_t>(experts_per_rank)
    );
    
    // Step 2: Allocate maximum possible output tensors
    auto activated_group_idx = torch::empty({experts_per_rank}, 
                                           torch::TensorOptions().dtype(torch::kInt32).device(device));
    auto group_size = torch::empty({experts_per_rank}, 
                                  torch::TensorOptions().dtype(torch::kInt32).device(device));
    auto group_start_indices = torch::empty({experts_per_rank}, 
                                           torch::TensorOptions().dtype(torch::kInt32).device(device));
    auto num_active_experts = torch::zeros({1}, 
                                          torch::TensorOptions().dtype(torch::kInt32).device(device));
    
    // Step 3: Compact active experts and compute start indices
    if (experts_per_rank <= 256) {
        // Use optimized single-block kernel for small expert counts
        compact_active_experts_single_block_kernel<<<1, 256>>>(
            expert_counts.data_ptr<int32_t>(),
            activated_group_idx.data_ptr<int32_t>(),
            group_size.data_ptr<int32_t>(),
            group_start_indices.data_ptr<int32_t>(),
            num_active_experts.data_ptr<int32_t>(),
            experts_per_rank
        );
    } else {
        // Use multi-block kernel for larger expert counts
        const int compact_blocks = (experts_per_rank + threads - 1) / threads;
        compact_active_experts_kernel<<<compact_blocks, threads>>>(
            expert_counts.data_ptr<int32_t>(),
            activated_group_idx.data_ptr<int32_t>(),
            group_size.data_ptr<int32_t>(),
            group_start_indices.data_ptr<int32_t>(),
            num_active_experts.data_ptr<int32_t>(),
            experts_per_rank
        );
    }
    
    cudaDeviceSynchronize();
    
    // Step 4: Resize output tensors to actual number of active experts
    int32_t actual_active_experts = num_active_experts.item<int32_t>();
    
    if (actual_active_experts > 0) {
        activated_group_idx = activated_group_idx.narrow(0, 0, actual_active_experts);
        group_size = group_size.narrow(0, 0, actual_active_experts);
        group_start_indices = group_start_indices.narrow(0, 0, actual_active_experts);
    } else {
        // No active experts - return empty tensors
        activated_group_idx = torch::empty({0}, torch::TensorOptions().dtype(torch::kInt32).device(device));
        group_size = torch::empty({0}, torch::TensorOptions().dtype(torch::kInt32).device(device));
        group_start_indices = torch::empty({0}, torch::TensorOptions().dtype(torch::kInt32).device(device));
    }
    
    return {group_size, activated_group_idx, group_start_indices};
}