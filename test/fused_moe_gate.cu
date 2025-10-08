#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

constexpr int WARP_SIZE = 32;

// Warp-level primitives
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Block-level max reduction with index tracking
__device__ void block_argmax(float val, int idx, float* max_val, int* max_idx, 
                             float* smem_vals, int* smem_idxs) {
    const int tid = threadIdx.x;
    const int lane = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    
    // Warp-level reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        float other_val = __shfl_down_sync(0xffffffff, val, offset);
        int other_idx = __shfl_down_sync(0xffffffff, idx, offset);
        if (other_val > val) {
            val = other_val;
            idx = other_idx;
        }
    }
    
    // First thread in each warp writes to shared memory
    if (lane == 0) {
        smem_vals[warp_id] = val;
        smem_idxs[warp_id] = idx;
    }
    __syncthreads();
    
    // Final reduction in first warp
    if (warp_id == 0) {
        val = (tid < blockDim.x / WARP_SIZE) ? smem_vals[tid] : -1e9f;
        idx = (tid < blockDim.x / WARP_SIZE) ? smem_idxs[tid] : -1;
        
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            float other_val = __shfl_down_sync(0xffffffff, val, offset);
            int other_idx = __shfl_down_sync(0xffffffff, idx, offset);
            if (other_val > val) {
                val = other_val;
                idx = other_idx;
            }
        }
        
        if (lane == 0) {
            *max_val = val;
            *max_idx = idx;
        }
    }
    __syncthreads();
}

// Fully parallel MoE gate kernel
template<int TOP_K, int TOPK_GROUP, int THREADS>
__global__ void parallel_moe_gate_kernel(
    const float* __restrict__ scores,           // [n, e]
    const float* __restrict__ e_score_bias,     // [e]
    int* __restrict__ topk_indices,             // [n, TOP_K]
    float* __restrict__ topk_weights,           // [n, TOP_K]
    const int n, const int e,
    const int n_group, const int experts_per_group,
    const float routed_scaling_factor) {
    
    const int token_idx = blockIdx.x;
    if (token_idx >= n) return;
    
    const int tid = threadIdx.x;
    
    // Shared memory layout
    extern __shared__ float smem[];
    float* s_scores = smem;                      // [e]
    float* s_group_scores = &smem[e];            // [n_group]
    int* s_selected_groups = (int*)&smem[e + n_group];  // [TOPK_GROUP]
    int* s_selected_experts = &s_selected_groups[TOPK_GROUP];  // [TOP_K]
    float* s_reduction_vals = (float*)&s_selected_experts[TOP_K];  // [THREADS/32]
    int* s_reduction_idxs = (int*)&s_reduction_vals[THREADS/32];   // [THREADS/32]
    
    const float* token_scores = &scores[token_idx * e];
    
    // =====================================================================
    // STEP 1: Load scores + bias (PARALLEL)
    // =====================================================================
    for (int i = tid; i < e; i += THREADS) {
        s_scores[i] = token_scores[i] + e_score_bias[i];
    }
    __syncthreads();
    
    // =====================================================================
    // STEP 2: Compute group scores (PARALLEL)
    // =====================================================================
    for (int g = tid; g < n_group; g += THREADS) {
        const int start = g * experts_per_group;
        
        // Find top-2 in group
        float max1 = -1e9f, max2 = -1e9f;
        #pragma unroll 4
        for (int i = 0; i < experts_per_group; i++) {
            float val = s_scores[start + i];
            if (val > max1) {
                max2 = max1;
                max1 = val;
            } else if (val > max2) {
                max2 = val;
            }
        }
        s_group_scores[g] = max1 + max2;
    }
    __syncthreads();
    
    // =====================================================================
    // STEP 3: Select top TOPK_GROUP groups (PARALLEL with successive argmax)
    // =====================================================================
    for (int k = 0; k < TOPK_GROUP; k++) {
        // Each thread finds best group among its subset
        float my_best_val = -1e9f;
        int my_best_idx = -1;
        
        for (int g = tid; g < n_group; g += THREADS) {
            // Skip already selected groups
            bool already_selected = false;
            for (int i = 0; i < k; i++) {
                if (s_selected_groups[i] == g) {
                    already_selected = true;
                    break;
                }
            }
            
            if (!already_selected && s_group_scores[g] > my_best_val) {
                my_best_val = s_group_scores[g];
                my_best_idx = g;
            }
        }
        
        // Block-level argmax to find global best
        float best_val;
        int best_idx;
        block_argmax(my_best_val, my_best_idx, &best_val, &best_idx,
                    s_reduction_vals, s_reduction_idxs);
        
        if (tid == 0) {
            s_selected_groups[k] = best_idx;
        }
        __syncthreads();
    }
    
    // =====================================================================
    // STEP 4: Find top TOP_K experts in selected groups (PARALLEL)
    // =====================================================================
    // Mark selected experts with their scores, unselected with -inf
    for (int i = tid; i < e; i += THREADS) {
        int expert_group = i / experts_per_group;
        bool in_selected_group = false;
        
        #pragma unroll
        for (int k = 0; k < TOPK_GROUP; k++) {
            if (s_selected_groups[k] == expert_group) {
                in_selected_group = true;
                break;
            }
        }
        
        if (!in_selected_group) {
            s_scores[i] = -1e9f;  // Mask out
        }
    }
    __syncthreads();
    
    // Now find top-k using successive argmax
    for (int k = 0; k < TOP_K; k++) {
        float my_best_val = -1e9f;
        int my_best_idx = -1;
        
        for (int i = tid; i < e; i += THREADS) {
            // Skip already selected
            bool already_selected = false;
            for (int j = 0; j < k; j++) {
                if (s_selected_experts[j] == i) {
                    already_selected = true;
                    break;
                }
            }
            
            if (!already_selected && s_scores[i] > my_best_val) {
                my_best_val = s_scores[i];
                my_best_idx = i;
            }
        }
        
        float best_val;
        int best_idx;
        block_argmax(my_best_val, my_best_idx, &best_val, &best_idx,
                    s_reduction_vals, s_reduction_idxs);
        
        if (tid == 0) {
            s_selected_experts[k] = best_idx;
        }
        __syncthreads();
    }
    
    // =====================================================================
    // STEP 5: Compute weights and normalize (PARALLEL reduction for sum)
    // =====================================================================
    // Reload original scores (without bias) for selected experts
    if (tid == 0) {
        float orig_scores[16];
        float sum = 0.0f;
        
        #pragma unroll
        for (int k = 0; k < TOP_K; k++) {
            int expert_idx = s_selected_experts[k];
            float orig_score = token_scores[expert_idx];  // Original score without bias
            orig_scores[k] = orig_score;
            sum += orig_score;
        }
        
        // Normalize and write output
        const float scale = routed_scaling_factor / (sum + 1e-20f);
        #pragma unroll
        for (int k = 0; k < TOP_K; k++) {
            topk_indices[token_idx * TOP_K + k] = s_selected_experts[k];
            topk_weights[token_idx * TOP_K + k] = orig_scores[k] * scale;
        }
    }
}

// Launcher
std::vector<torch::Tensor> parallel_moe_gate_forward(
    torch::Tensor scores,                    // [n, n_routed_experts] - already sigmoid'd
    torch::Tensor e_score_correction_bias,   // [n_routed_experts]
    int64_t n_group,
    int64_t topk_group,
    int64_t n_routed_experts,
    int64_t top_k,
    double routed_scaling_factor) {
    
    const auto n = scores.size(0);
    const auto e = n_routed_experts;
    
    // Verify tensor dimensions match
    TORCH_CHECK(scores.size(1) == n_routed_experts, 
                "scores dimension 1 (", scores.size(1), ") must match n_routed_experts (", n_routed_experts, ")");
    TORCH_CHECK(e_score_correction_bias.size(0) == n_routed_experts,
                "e_score_correction_bias size (", e_score_correction_bias.size(0), ") must match n_routed_experts (", n_routed_experts, ")");
    TORCH_CHECK(e % n_group == 0,
                "n_routed_experts (", e, ") must be divisible by n_group (", n_group, ")");
    
    const auto experts_per_group = e / n_group;
    
    scores = scores.contiguous().to(torch::kFloat32);
    e_score_correction_bias = e_score_correction_bias.contiguous().to(torch::kFloat32);
    
    auto device = scores.device();
    auto topk_indices = torch::empty({n, top_k},
        torch::TensorOptions().dtype(torch::kInt32).device(device));
    auto topk_weights = torch::empty({n, top_k},
        torch::TensorOptions().dtype(torch::kFloat32).device(device));
    
    const int threads = 256;
    const int blocks = n;
    
    // Shared memory: scores + group_scores + selected_groups + selected_experts + reduction buffers
    const size_t smem_size = (e + n_group) * sizeof(float) + 
                             (topk_group + top_k) * sizeof(int) +
                             (threads / 32) * (sizeof(float) + sizeof(int));
    
    #define LAUNCH_KERNEL(TK, TG) \
        parallel_moe_gate_kernel<TK, TG, threads><<<blocks, threads, smem_size>>>( \
            scores.data_ptr<float>(), \
            e_score_correction_bias.data_ptr<float>(), \
            topk_indices.data_ptr<int>(), \
            topk_weights.data_ptr<float>(), \
            n, e, n_group, experts_per_group, \
            static_cast<float>(routed_scaling_factor))
    
    if (top_k == 2 && topk_group == 2) {
        LAUNCH_KERNEL(2, 2);
    } else if (top_k == 2 && topk_group == 4) {
        LAUNCH_KERNEL(2, 4);
    } else if (top_k == 4 && topk_group == 2) {
        LAUNCH_KERNEL(4, 2);
    } else if (top_k == 4 && topk_group == 4) {
        LAUNCH_KERNEL(4, 4);
    } else if (top_k == 6 && topk_group == 4) {
        LAUNCH_KERNEL(6, 4);
    } else if (top_k == 8 && topk_group == 4) {
        LAUNCH_KERNEL(8, 4);
    } else {
        AT_ERROR("Unsupported top_k=", top_k, " topk_group=", topk_group);
    }
    
    #undef LAUNCH_KERNEL
    
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    
    return {topk_indices.to(torch::kInt64), topk_weights};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &parallel_moe_gate_forward, "Parallel MoE Gate Forward");
}