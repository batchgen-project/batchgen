#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

constexpr int WARP_SIZE = 32;

// Warp-level reduction primitives
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Fast top-k selection using register-based heap
template<int K>
__device__ void topk_insert(float val, int idx, float* topk_vals, int* topk_idxs) {
    // Check if val should be in top-k
    if (val <= topk_vals[K-1]) return;
    
    // Find insertion position
    int pos = K - 1;
    #pragma unroll
    for (int i = 0; i < K - 1; i++) {
        if (val > topk_vals[i]) {
            pos = i;
            break;
        }
    }
    
    // Shift and insert
    #pragma unroll
    for (int i = K - 1; i > pos; i--) {
        topk_vals[i] = topk_vals[i-1];
        topk_idxs[i] = topk_idxs[i-1];
    }
    topk_vals[pos] = val;
    topk_idxs[pos] = idx;
}

// Main fused kernel - processes one token per block
template<int TOP_K, int TOPK_GROUP>
__global__ void fused_moe_gate_kernel(
    const float* __restrict__ hidden_states,
    const float* __restrict__ weight,
    const float* __restrict__ e_score_bias,
    int* __restrict__ topk_indices,
    float* __restrict__ topk_weights,
    const int n, const int h, const int e,
    const int n_group, const int experts_per_group,
    const float routed_scaling_factor) {
    
    const int token_idx = blockIdx.x;
    if (token_idx >= n) return;
    
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;
    
    extern __shared__ float smem[];
    float* s_scores = smem;  // [e]
    float* s_group_scores = &smem[e];  // [n_group]
    
    const float* hs = &hidden_states[token_idx * h];
    
    // Step 1: Compute all expert scores using all threads
    for (int expert_idx = tid; expert_idx < e; expert_idx += num_threads) {
        const float* w = &weight[expert_idx * h];
        
        // Dot product with vectorized loads
        float logit = 0.0f;
        #pragma unroll 4
        for (int i = 0; i < h; i++) {
            logit += hs[i] * w[i];
        }
        
        // Sigmoid + bias
        float score = 1.0f / (1.0f + __expf(-logit));
        s_scores[expert_idx] = score + e_score_bias[expert_idx];
    }
    __syncthreads();
    
    // Step 2: Compute group scores (each warp handles groups)
    if (tid < n_group) {
        const int group_start = tid * experts_per_group;
        const float* group_scores_ptr = &s_scores[group_start];
        
        float max1 = -1e9f, max2 = -1e9f;
        #pragma unroll
        for (int i = 0; i < experts_per_group; i++) {
            float val = group_scores_ptr[i];
            if (val > max1) {
                max2 = max1;
                max1 = val;
            } else if (val > max2) {
                max2 = val;
            }
        }
        s_group_scores[tid] = max1 + max2;
    }
    __syncthreads();
    
    // Step 3: Select top groups (single thread with registers)
    __shared__ int s_selected_groups[16];  // Max we'll ever need
    if (tid == 0) {
        float group_vals[16];
        int group_idxs[16];
        
        #pragma unroll
        for (int i = 0; i < TOPK_GROUP; i++) {
            group_vals[i] = -1e9f;
            group_idxs[i] = -1;
        }
        
        // Find top-k groups
        for (int g = 0; g < n_group; g++) {
            topk_insert<TOPK_GROUP>(s_group_scores[g], g, group_vals, group_idxs);
        }
        
        // Store selected groups
        #pragma unroll
        for (int i = 0; i < TOPK_GROUP; i++) {
            s_selected_groups[i] = group_idxs[i];
        }
    }
    __syncthreads();
    
    // Step 4: Mask scores and find top-k experts
    // Each thread maintains its own top-k candidates
    float my_topk_vals[16];  // Max top_k we support
    int my_topk_idxs[16];
    
    if (tid == 0) {
        #pragma unroll
        for (int i = 0; i < TOP_K; i++) {
            my_topk_vals[i] = -1e9f;
            my_topk_idxs[i] = -1;
        }
        
        // Check all experts in selected groups
        #pragma unroll
        for (int g = 0; g < TOPK_GROUP; g++) {
            int group_idx = s_selected_groups[g];
            int group_start = group_idx * experts_per_group;
            
            for (int i = 0; i < experts_per_group; i++) {
                int expert_idx = group_start + i;
                topk_insert<TOP_K>(s_scores[expert_idx], expert_idx, my_topk_vals, my_topk_idxs);
            }
        }
        
        // Step 5: Get original scores (without bias) and normalize
        float sum = 1e-20f;
        float orig_scores[16];
        
        #pragma unroll
        for (int i = 0; i < TOP_K; i++) {
            int expert_idx = my_topk_idxs[i];
            float orig_score = s_scores[expert_idx] - e_score_bias[expert_idx];
            orig_scores[i] = orig_score;
            sum += orig_score;
        }
        
        // Write final output
        const float inv_sum = routed_scaling_factor / sum;
        #pragma unroll
        for (int i = 0; i < TOP_K; i++) {
            topk_weights[token_idx * TOP_K + i] = orig_scores[i] * inv_sum;
            topk_indices[token_idx * TOP_K + i] = my_topk_idxs[i];
        }
    }
}

// Launcher function to handle different top_k values
std::vector<torch::Tensor> fused_moe_gate_forward(
    torch::Tensor hidden_states,
    torch::Tensor weight,
    torch::Tensor e_score_correction_bias,
    int64_t n_group,
    int64_t topk_group,
    int64_t n_routed_experts,
    int64_t top_k,
    double routed_scaling_factor) {
    
    const auto bsz = hidden_states.size(0);
    const auto seq_len = hidden_states.size(1);
    const auto h = hidden_states.size(2);
    const auto n = bsz * seq_len;
    const auto e = n_routed_experts;
    const auto experts_per_group = e / n_group;
    
    hidden_states = hidden_states.view({-1, h}).contiguous().to(torch::kFloat32);
    weight = weight.contiguous().to(torch::kFloat32);
    e_score_correction_bias = e_score_correction_bias.contiguous().to(torch::kFloat32);
    
    auto device = hidden_states.device();
    auto topk_indices = torch::empty({n, top_k}, 
        torch::TensorOptions().dtype(torch::kInt32).device(device));
    auto topk_weights = torch::empty({n, top_k}, 
        torch::TensorOptions().dtype(torch::kFloat32).device(device));
    
    const int threads = 256;
    const int blocks = n;
    const size_t smem_size = (e + n_group) * sizeof(float);
    
    // Template dispatch based on top_k and topk_group values
    #define LAUNCH_KERNEL(TK, TG) \
        fused_moe_gate_kernel<TK, TG><<<blocks, threads, smem_size>>>( \
            hidden_states.data_ptr<float>(), \
            weight.data_ptr<float>(), \
            e_score_correction_bias.data_ptr<float>(), \
            topk_indices.data_ptr<int>(), \
            topk_weights.data_ptr<float>(), \
            n, h, e, n_group, experts_per_group, \
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
        AT_ERROR("Unsupported top_k=", top_k, " topk_group=", topk_group, 
                 ". Supported: (2,2), (2,4), (4,2), (4,4), (6,4), (8,4)");
    }
    
    #undef LAUNCH_KERNEL
    
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    
    return {topk_indices.to(torch::kInt64), topk_weights};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fused_moe_gate_forward, "Fused MoE Gate Forward");
}