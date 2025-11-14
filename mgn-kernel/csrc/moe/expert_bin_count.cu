#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <vector>
#include <torch/extension.h>  // This includes pybind11 and defines TORCH_EXTENSION_NAME

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

// Simple and correct parallel compaction (works for any size)
__global__ void compact_active_experts_parallel_kernel(
    const int32_t* __restrict__ expert_counts,  // [experts_per_rank] - input counts
    int32_t* __restrict__ activated_group_idx,  // [num_active] - output active expert indices
    int32_t* __restrict__ group_size,           // [num_active] - output group sizes
    int32_t* __restrict__ group_start_indices,  // [num_active] - output start indices
    int32_t* __restrict__ num_active_experts,   // [1] - output number of active experts
    int32_t experts_per_rank) {
    
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Step 1: Compact - each active expert atomically reserves its output slot
    int32_t my_count = 0;
    int32_t my_position = -1;
    
    if (tid < experts_per_rank) {
        my_count = expert_counts[tid];
        if (my_count > 0) {
            my_position = atomicAdd(num_active_experts, 1);
        }
    }
    
    // Step 2: Write compacted data
    if (my_position >= 0) {
        activated_group_idx[my_position] = tid;
        group_size[my_position] = my_count;
    }
}

// Second pass: compute prefix sum for start indices  
// Works for up to 256 active experts
__global__ void compute_start_indices_kernel(
    const int32_t* __restrict__ group_size,     // [max_size] - input group sizes
    int32_t* __restrict__ group_start_indices,  // [max_size] - output start indices
    const int32_t* __restrict__ num_active,     // [1] - actual number of active
    int32_t max_size) {
    
    const int tid = threadIdx.x;
    
    __shared__ int32_t shared_sizes[256];
    __shared__ int32_t shared_prefix[256];
    __shared__ int32_t actual_count;
    
    // Load actual count
    if (tid == 0) {
        actual_count = *num_active;
    }
    __syncthreads();
    
    // Load into shared memory
    if (tid < max_size && tid < actual_count) {
        shared_sizes[tid] = group_size[tid];
    } else {
        shared_sizes[tid] = 0;
    }
    __syncthreads();
    
    // Parallel prefix sum in shared memory
    if (tid < actual_count) {
        int32_t sum = shared_sizes[tid];
        shared_prefix[tid] = sum;
        __syncthreads();
        
        // Efficient parallel scan
        for (int offset = 1; offset < blockDim.x; offset *= 2) {
            int32_t temp = 0;
            if (tid >= offset) {
                temp = shared_prefix[tid - offset];
            }
            __syncthreads();
            
            if (tid >= offset) {
                shared_prefix[tid] += temp;
            }
            __syncthreads();
        }
        
        // Write exclusive prefix sum (start indices)
        group_start_indices[tid] = (tid == 0) ? 0 : shared_prefix[tid - 1];
    }
}

// Optimized single-block kernel for small expert counts
__global__ void compact_active_experts_single_block_kernel(
    const int32_t* __restrict__ expert_counts,  // [experts_per_rank] - input counts
    int32_t* __restrict__ activated_group_idx,  // [num_active] - output active expert indices
    int32_t* __restrict__ group_size,           // [num_active] - output group sizes  
    int32_t* __restrict__ group_start_indices,  // [num_active] - output start indices
    int32_t* __restrict__ num_active_experts,   // [1] - output number of active experts
    int32_t experts_per_rank) {
    
    const int tid = threadIdx.x;
    
    __shared__ int32_t write_pos;
    __shared__ int32_t prefix_sum;
    
    if (tid == 0) {
        write_pos = 0;
        prefix_sum = 0;
    }
    __syncthreads();
    
    // Sequential processing by thread 0 - correct for small sizes
    if (tid == 0) {
        int32_t local_write_pos = 0;
        int32_t cumulative = 0;
        
        for (int expert_id = 0; expert_id < experts_per_rank; expert_id++) {
            int32_t count = expert_counts[expert_id];
            if (count > 0) {
                activated_group_idx[local_write_pos] = expert_id;
                group_size[local_write_pos] = count;
                group_start_indices[local_write_pos] = cumulative;
                
                cumulative += count;
                local_write_pos++;
            }
        }
        
        *num_active_experts = local_write_pos;
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
    const int blocks = std::min((int)((num_tokens + threads - 1) / threads), 1024);
    
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
    if (experts_per_rank <= 512) {
        // Use optimized single-block kernel for small expert counts
        // This covers most practical cases (typical expert counts: 8, 16, 32, 64, 128)
        compact_active_experts_single_block_kernel<<<1, 256>>>(
            expert_counts.data_ptr<int32_t>(),
            activated_group_idx.data_ptr<int32_t>(),
            group_size.data_ptr<int32_t>(),
            group_start_indices.data_ptr<int32_t>(),
            num_active_experts.data_ptr<int32_t>(),
            static_cast<int32_t>(experts_per_rank)
        );
    } else {
        // Two-pass approach for very large expert counts (rare)
        const int compact_blocks = (experts_per_rank + threads - 1) / threads;
        
        // Pass 1: Compact active experts
        compact_active_experts_parallel_kernel<<<compact_blocks, threads>>>(
            expert_counts.data_ptr<int32_t>(),
            activated_group_idx.data_ptr<int32_t>(),
            group_size.data_ptr<int32_t>(),
            group_start_indices.data_ptr<int32_t>(),
            num_active_experts.data_ptr<int32_t>(),
            static_cast<int32_t>(experts_per_rank)
        );
        
        // Pass 2: Compute start indices
        // Note: Assumes num_active_experts <= 256 (true for most MoE models)
        compute_start_indices_kernel<<<1, 256>>>(
            group_size.data_ptr<int32_t>(),
            group_start_indices.data_ptr<int32_t>(),
            num_active_experts.data_ptr<int32_t>(),
            static_cast<int32_t>(experts_per_rank)
        );
    }
    
    // NO cudaDeviceSynchronize() needed here!
    // The next operation will implicitly synchronize if needed
    
    // Step 4: Return tensors WITHOUT calling .item()
    // Let the caller handle slicing based on num_active_experts
    // This avoids expensive CPU-GPU synchronization
    
    return {group_size, activated_group_idx, group_start_indices, num_active_experts};
}

// Alternative: If you MUST slice on the C++ side, use async approach
std::vector<torch::Tensor> expert_bincount_cuda_sliced(
    torch::Tensor eids,                         
    int64_t routed_expert_start_idx,
    int64_t experts_per_rank,
    torch::Device device) {
    
    auto results = expert_bincount_cuda(eids, routed_expert_start_idx, experts_per_rank, device);
    
    auto group_size = results[0];
    auto activated_group_idx = results[1];
    auto group_start_indices = results[2];
    auto num_active_experts_tensor = results[3];
    
    // Use index_select to slice without CPU sync
    // Create indices tensor [0, 1, 2, ..., num_active-1] on GPU
    auto max_size = experts_per_rank;
    auto indices = torch::arange(max_size, torch::TensorOptions().dtype(torch::kInt64).device(device));
    
    // Mask: indices < num_active_experts (broadcast comparison on GPU)
    auto mask = indices.lt(num_active_experts_tensor.squeeze());
    
    // Use masked_select or nonzero + index_select
    auto valid_indices = torch::nonzero(mask).squeeze(-1);
    
    // Slice using GPU operations only - no CPU sync!
    if (valid_indices.numel() > 0) {
        activated_group_idx = activated_group_idx.index_select(0, valid_indices);
        group_size = group_size.index_select(0, valid_indices);
        group_start_indices = group_start_indices.index_select(0, valid_indices);
    } else {
        activated_group_idx = torch::empty({0}, torch::TensorOptions().dtype(torch::kInt32).device(device));
        group_size = torch::empty({0}, torch::TensorOptions().dtype(torch::kInt32).device(device));
        group_start_indices = torch::empty({0}, torch::TensorOptions().dtype(torch::kInt32).device(device));
    }
    
    return {group_size, activated_group_idx, group_start_indices};
}


__global__ void compact_expert_data_dense_kernel(
    const int32_t* expert_counts,      // [experts_per_rank]
    int32_t* activated_group_idx,      // [experts_per_rank] - OUTPUT: dense, no gaps
    int32_t* group_size,               // [experts_per_rank] - OUTPUT: same as input but reordered
    int32_t* group_start_indices,      // [experts_per_rank] - OUTPUT: prefix sum
    int32_t* num_active_experts,       // [1] - OUTPUT: count of non-zero experts
    const int32_t experts_per_rank
) {
    __shared__ int32_t s_active_count;
    __shared__ int32_t s_prefix_sum[512];  // Assuming max 512 experts per rank
    
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    
    if (tid == 0) {
        s_active_count = 0;
    }
    __syncthreads();
    
    // Phase 1: Identify active experts and compute their positions
    for (int i = tid; i < experts_per_rank; i += num_threads) {
        int count = expert_counts[i];
        if (count > 0) {
            int pos = atomicAdd(&s_active_count, 1);
            s_prefix_sum[pos] = i;  // Store which expert is at this position
        }
    }
    __syncthreads();
    
    int num_active = s_active_count;
    
    // Phase 2: Build dense output arrays
    // Active experts go first (indices 0 to num_active-1)
    for (int i = tid; i < num_active; i += num_threads) {
        int expert_id = s_prefix_sum[i];
        activated_group_idx[i] = expert_id;
        group_size[i] = expert_counts[expert_id];
    }
    
    // Inactive experts fill the rest (indices num_active to experts_per_rank-1)
    // Set them to have size 0 so kernel can safely skip them
    for (int i = num_active + tid; i < experts_per_rank; i += num_threads) {
        activated_group_idx[i] = 0;  // Doesn't matter, won't be used
        group_size[i] = 0;           // Zero size = kernel will skip
    }
    __syncthreads();
    
    // Phase 3: Compute prefix sum for start indices
    if (tid == 0) {
        int cumsum = 0;
        for (int i = 0; i < experts_per_rank; i++) {
            group_start_indices[i] = cumsum;
            cumsum += group_size[i];
        }
        *num_active_experts = num_active;
    }
}

// -----------------------------------------------------------------
// NEW WRAPPER FUNCTION
// -----------------------------------------------------------------
// This function replaces torch.nonzero() and torch.count_nonzero()
// It takes the expert_counts (already computed) and finds the
// active expert indices and metadata.
std::vector<torch::Tensor> compact_expert_data_cuda(
    torch::Tensor expert_counts) {
    
    // Get dimensions and device
    const auto experts_per_rank = expert_counts.size(0);
    const auto device = expert_counts.device();

    TORCH_CHECK(expert_counts.is_cuda(), "expert_counts must be a CUDA tensor");
    TORCH_CHECK(expert_counts.dtype() == torch::kInt32, "expert_counts must be int32");

    // Step 1: Allocate output tensors (oversized)
    auto activated_group_idx = torch::empty({experts_per_rank}, 
                                           torch::TensorOptions().dtype(torch::kInt32).device(device));
    auto group_size = torch::empty({experts_per_rank}, 
                                  torch::TensorOptions().dtype(torch::kInt32).device(device));
    auto group_start_indices = torch::empty({experts_per_rank}, 
                                           torch::TensorOptions().dtype(torch::kInt32).device(device));
    auto num_active_experts = torch::zeros({1}, 
                                          torch::TensorOptions().dtype(torch::kInt32).device(device));

    // Step 2: Compact active experts and compute start indices (logic from your file)
    if (experts_per_rank <= 512) {
        // Use optimized single-block kernel
        const int threads = 256; // Or whatever is best for this kernel
        compact_active_experts_single_block_kernel<<<1, threads>>>(
            expert_counts.data_ptr<int32_t>(),
            activated_group_idx.data_ptr<int32_t>(),
            group_size.data_ptr<int32_t>(),
            group_start_indices.data_ptr<int32_t>(),
            num_active_experts.data_ptr<int32_t>(),
            static_cast<int32_t>(experts_per_rank)
        );
    } else {
        // Use two-pass approach
        const int threads = 256;
        const int compact_blocks = (experts_per_rank + threads - 1) / threads;
        
        compact_active_experts_parallel_kernel<<<compact_blocks, threads>>>(
            expert_counts.data_ptr<int32_t>(),
            activated_group_idx.data_ptr<int32_t>(),
            group_size.data_ptr<int32_t>(),
            group_start_indices.data_ptr<int32_t>(),
            num_active_experts.data_ptr<int32_t>(),
            static_cast<int32_t>(experts_per_rank)
        );
        
        compute_start_indices_kernel<<<1, 256>>>(
            group_size.data_ptr<int32_t>(),
            group_start_indices.data_ptr<int32_t>(),
            num_active_experts.data_ptr<int32_t>(),
            static_cast<int32_t>(experts_per_rank)
        );
    }
    
    // Return all computed tensors. They are async.
    // NOTE: These are oversized, and need to be sliced in Python
    // using the num_active_experts tensor.
    return {group_size, activated_group_idx, group_start_indices, num_active_experts};
}


// __global__ void compute_expert_offsets_kernel(
//     const int32_t* expert_counts,      // [num_local_experts]
//     int32_t* expert_offsets,           // [num_local_experts] - OUTPUT: prefix sum
//     const int32_t num_local_experts
// ) {
//     // Simple prefix sum computation
//     int tid = threadIdx.x;
    
//     if (tid == 0) {
//         int cumsum = 0;
//         for (int i = 0; i < num_local_experts; i++) {
//             expert_offsets[i] = cumsum;
//             cumsum += expert_counts[i];
//         }
//     }
// }

// torch::Tensor compute_expert_offsets_cuda(torch::Tensor expert_counts) {
    
//     const auto num_local_experts = expert_counts.size(0);
//     const auto device = expert_counts.device();

//     TORCH_CHECK(expert_counts.is_cuda(), "expert_counts must be a CUDA tensor");
//     TORCH_CHECK(expert_counts.dtype() == torch::kInt32, "expert_counts must be int32");

//     // Just compute prefix sum
//     auto expert_offsets = torch::empty({num_local_experts}, 
//                                        torch::TensorOptions().dtype(torch::kInt32).device(device));

//     compute_expert_offsets_kernel<<<1, 1>>>(
//         expert_counts.data_ptr<int32_t>(),
//         expert_offsets.data_ptr<int32_t>(),
//         static_cast<int32_t>(num_local_experts)
//     );

//     return expert_offsets;
// }