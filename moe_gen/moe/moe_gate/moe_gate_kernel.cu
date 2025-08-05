// #include "moe_gate_launcher.h"
// #include <cuda_bf16.h>
// #include <cmath>
// #include <ATen/cuda/CUDAContext.h>
// #include <ATen/cuda/Exceptions.h>

// // --- Device-level helper functions ---
// __device__ inline float sigmoidf(float x) {
//     return 1.0f / (1.0f + expf(-x));
// }

// template <int BLOCK_THREADS>
// __device__ void block_bitonic_sort(KeyValuePair* data, int thread_id) {
//     for (int k = 2; k <= BLOCK_THREADS; k <<= 1) {
//         for (int j = k >> 1; j > 0; j >>= 1) {
//             int ixj = thread_id ^ j;
//             if (ixj > thread_id) {
//                 bool ascending = ((thread_id & k) == 0);
//                 KeyValuePair thread_data = data[thread_id];
//                 KeyValuePair other_data = data[ixj];
//                 if ((thread_data.value < other_data.value) == ascending) {
//                     data[thread_id] = other_data;
//                     data[ixj] = thread_data;
//                 }
//             }
//             __syncthreads();
//         }
//     }
// }


// template<int BLOCK_THREADS>
// __global__ void compiled_moe_gate_forward_kernel(
//     const __nv_bfloat16* __restrict__ hidden_states,
//     const __nv_bfloat16* __restrict__ weight,
//     const __nv_bfloat16* __restrict__ e_score_correction_bias,
//     long* __restrict__ topk_idx_out,
//     __nv_bfloat16* __restrict__ topk_weight_out,
//     int bsz_seq_len, int h,
//     int n_group, int topk_group, int n_routed_experts, int top_k,
//     float routed_scaling_factor, int experts_per_group) {

//     const int token_idx = blockIdx.x;
//     const int thread_id = threadIdx.x;

//     // --- FINAL, ROBUST SHARED MEMORY PARTITIONING ---
//     extern __shared__ char sh_mem_base[];

//     // This layout is explicit and guaranteed not to overlap.
//     // 1. Buffer for scores with bias (`scores_for_choice`).
//     float* sh_biased_scores = (float*)sh_mem_base;
//     // 2. Buffer for pristine scores (`logits.sigmoid()`). Preserved until the end.
//     float* sh_original_scores = sh_biased_scores + n_routed_experts;
//     // 3. General purpose buffer for sorting.
//     KeyValuePair* sh_sort_buffer = (KeyValuePair*)(sh_original_scores + n_routed_experts);


//     // --- 1. Compute Gating Scores ---
//     if (thread_id < n_routed_experts) {
//         float logit = 0.0f;
//         const __nv_bfloat16* hs_ptr = hidden_states + token_idx * h;
//         const __nv_bfloat16* w_ptr = weight + thread_id * h;
//         for (int i = 0; i < h; ++i) {
//             logit += __bfloat162float(hs_ptr[i]) * __bfloat162float(w_ptr[i]);
//         }
//         float score = sigmoidf(logit);
//         sh_original_scores[thread_id] = score; // This is now pristine.
//         sh_biased_scores[thread_id] = score + __bfloat162float(e_score_correction_bias[thread_id]);
//     }
//     __syncthreads();

//     // --- 2. Grouped Top-2 and Sum ---
//     // Use the sort buffer temporarily to store group scores.
//     if (thread_id < n_group) {
//         float top1 = -INFINITY, top2 = -INFINITY;
//         int group_start_idx = thread_id * experts_per_group;
//         for (int i = 0; i < experts_per_group; ++i) {
//             float val = sh_biased_scores[group_start_idx + i];
//             if (val > top1) { top2 = top1; top1 = val; } else if (val > top2) { top2 = val; }
//         }
//         sh_sort_buffer[thread_id] = {top1 + top2, thread_id};
//     }
//     __syncthreads();

//     // --- 3. Select Top-k Groups ---
//     if (thread_id == 0) { // Single-threaded insertion sort on the small group array.
//         for (int i = 1; i < n_group; i++) {
//             KeyValuePair key = sh_sort_buffer[i];
//             int j = i - 1;
//             while (j >= 0 && sh_sort_buffer[j].value < key.value) { sh_sort_buffer[j + 1] = sh_sort_buffer[j]; j--; }
//             sh_sort_buffer[j + 1] = key;
//         }
//     }
//     __syncthreads();

//     // --- 4. Create and Apply Score Mask ---
//     bool is_in_top_group = false;
//     if (thread_id < n_routed_experts) {
//         int group_of_this_expert = thread_id / experts_per_group;
//         for(int i = 0; i < topk_group; ++i) {
//             if (sh_sort_buffer[i].index == group_of_this_expert) {
//                 is_in_top_group = true;
//                 break;
//             }
//         }
//     }
//     __syncthreads();

//     // --- 5. Final Top-k Expert Selection ---
//     if (thread_id < n_routed_experts) {
//         sh_sort_buffer[thread_id] = {is_in_top_group ? sh_biased_scores[thread_id] : -INFINITY, thread_id};
//     } else if (thread_id < BLOCK_THREADS) {
//         sh_sort_buffer[thread_id] = {-INFINITY, -1};
//     }
//     __syncthreads();

//     block_bitonic_sort<BLOCK_THREADS>(sh_sort_buffer, thread_id);

//     // --- 6. Gather Weights, Normalize and Scale ---
//     // Use sh_biased_scores as a temp buffer for the final weights before reduction.
//     float* sh_final_weights = sh_biased_scores;
//     if (thread_id < top_k) {
//         int expert_idx = sh_sort_buffer[thread_id].index;
//         sh_final_weights[thread_id] = (expert_idx != -1) ? sh_original_scores[expert_idx] : 0.0f;
//     }
//     __syncthreads();

//     if (thread_id >= top_k && thread_id < BLOCK_THREADS) {
//         sh_final_weights[thread_id] = 0.0f;
//     }
//     __syncthreads();

//     // In-place reduction on sh_final_weights.
//     for (int s = BLOCK_THREADS / 2; s > 0; s >>= 1) {
//         if (thread_id < s) sh_final_weights[thread_id] += sh_final_weights[thread_id + s];
//         __syncthreads();
//     }

//     // --- 7. Write to Global Memory ---
//     if (thread_id < top_k) {
//         float denominator = sh_final_weights[0] + 1e-20f;
//         int original_expert_idx = sh_sort_buffer[thread_id].index;
//         // The weight was already gathered and stored in sh_final_weights[thread_id] before reduction.
//         // We re-gather here from the pristine sh_original_scores to get the clean value.
//         float gathered_weight = (original_expert_idx != -1) ? sh_original_scores[original_expert_idx] : 0.0f;
//         float final_weight = (gathered_weight / denominator) * routed_scaling_factor;

//         topk_idx_out[token_idx * top_k + thread_id] = original_expert_idx;
//         topk_weight_out[token_idx * top_k + thread_id] = __float2bfloat16(final_weight);
//     }
// }

// // --- Launcher Function ---
// std::vector<torch::Tensor> launch_moe_gate_forward_kernel(
//     torch::Tensor hidden_states,
//     torch::Tensor weight,
//     torch::Tensor e_score_correction_bias,
//     int64_t n_group,
//     int64_t topk_group,
//     int64_t n_routed_experts,
//     int64_t top_k,
//     double routed_scaling_factor) {

//     TORCH_CHECK(hidden_states.is_cuda(), "hidden_states must be a CUDA tensor");
//     TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
//     TORCH_CHECK(hidden_states.dtype() == torch::kBFloat16, "Inputs must be BFloat16");
//     TORCH_CHECK(n_group > 0 && (n_routed_experts % n_group == 0), "n_routed_experts must be divisible by n_group");

//     const int bsz_seq_len = hidden_states.size(0);
//     const int h = hidden_states.size(1);

//     auto topk_idx = torch::empty({bsz_seq_len, top_k}, torch::dtype(torch::kInt64).device(hidden_states.device()));
//     auto topk_weight = torch::empty({bsz_seq_len, top_k}, hidden_states.options());

//     const int BLOCK_THREADS = 256;
//     TORCH_CHECK(n_routed_experts <= BLOCK_THREADS, "n_routed_experts must be <= BLOCK_THREADS");
//     const int experts_per_group = n_routed_experts / n_group;

//     // --- FINAL, ROBUST SHARED MEMORY SIZE CALCULATION ---
//     int sh_mem_size = (sizeof(float) * n_routed_experts) * 2 // sh_biased_scores + sh_original_scores
//                     + sizeof(KeyValuePair) * BLOCK_THREADS;  // sh_sort_buffer

//     dim3 grid(bsz_seq_len);
//     dim3 block(BLOCK_THREADS);

//     compiled_moe_gate_forward_kernel<BLOCK_THREADS><<<grid, block, sh_mem_size, at::cuda::getCurrentCUDAStream()>>>(
//         (const __nv_bfloat16*)hidden_states.data_ptr(),
//         (const __nv_bfloat16*)weight.data_ptr(),
//         (const __nv_bfloat16*)e_score_correction_bias.data_ptr(),
//         topk_idx.data_ptr<long>(),
//         (__nv_bfloat16*)topk_weight.data_ptr(),
//         bsz_seq_len, h,
//         n_group, topk_group, n_routed_experts, top_k,
//         (float)routed_scaling_factor,
//         experts_per_group
//     );

//     AT_CUDA_CHECK(cudaGetLastError());
//     return {topk_idx, topk_weight};
// }

#include "moe_gate_launcher.h"
#include <cuda_bf16.h>
#include <cmath>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>

// If not already defined in the header
struct KeyValuePair {
    float value;
    int   index;
};

// --- Device-level helper functions ---
__device__ inline float sigmoidf(float x) {
    return 1.0f / (1.0f + expf(-x));
}

template <int BLOCK_THREADS>
__device__ void block_bitonic_sort(KeyValuePair* data, int thread_id) {
    // bitonic sort on BLOCK_THREADS items
    #pragma unroll
    for (int k = 2; k <= BLOCK_THREADS; k <<= 1) {
        #pragma unroll
        for (int j = k >> 1; j > 0; j >>= 1) {
            int ixj = thread_id ^ j;
            if (ixj > thread_id) {
                bool ascending = ((thread_id & k) == 0);
                KeyValuePair a = data[thread_id];
                KeyValuePair b = data[ixj];
                if ((a.value < b.value) == ascending) {
                    data[thread_id] = b;
                    data[ixj] = a;
                }
            }
            __syncthreads();
        }
    }
}

template<int BLOCK_THREADS>
__global__ void compiled_moe_gate_forward_kernel(
    const __nv_bfloat16* __restrict__ hidden_states,
    const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ e_score_correction_bias,
    long* __restrict__ topk_idx_out,
    __nv_bfloat16* __restrict__ topk_weight_out,
    int bsz_seq_len, int h,
    int n_group, int topk_group, int n_routed_experts, int top_k,
    float routed_scaling_factor, int experts_per_group) {

    const int token_idx = blockIdx.x;
    const int thread_id = threadIdx.x;

    // --- Robust shared memory partitioning (no overlap) ---
    extern __shared__ __align__(16) unsigned char sh_mem_base[];
    unsigned char* base = sh_mem_base;

    // floats
    float* sh_biased_scores   = reinterpret_cast<float*>(base);
    float* sh_original_scores = sh_biased_scores + n_routed_experts;
    float* sh_final_weights   = sh_original_scores + n_routed_experts;         // size: BLOCK_THREADS floats

    // Align KeyValuePair buffer
    constexpr size_t kv_align = alignof(KeyValuePair);
    unsigned char* sort_buf_bytes = reinterpret_cast<unsigned char*>(sh_final_weights + BLOCK_THREADS);
    size_t sort_off = (reinterpret_cast<size_t>(sort_buf_bytes) + (kv_align - 1)) & ~(kv_align - 1);
    KeyValuePair* sh_sort_buffer = reinterpret_cast<KeyValuePair*>(sort_off);

    // --- 1. Compute gating scores ---
    if (thread_id < n_routed_experts) {
        float logit = 0.0f;
        const __nv_bfloat16* hs_ptr = hidden_states + token_idx * h;
        const __nv_bfloat16* w_ptr  = weight + thread_id * h;
        for (int i = 0; i < h; ++i) {
            logit += __bfloat162float(hs_ptr[i]) * __bfloat162float(w_ptr[i]);
        }
        float score = sigmoidf(logit);
        sh_original_scores[thread_id] = score; // pristine
        sh_biased_scores[thread_id]   = score + __bfloat162float(e_score_correction_bias[thread_id]);
    }
    __syncthreads();

    // --- 2. Grouped Top-2 and sum ---
    if (thread_id < n_group) {
        float top1 = -INFINITY, top2 = -INFINITY;
        int group_start_idx = thread_id * experts_per_group;
        for (int i = 0; i < experts_per_group; ++i) {
            float v = sh_biased_scores[group_start_idx + i];
            if (v > top1) { top2 = top1; top1 = v; }
            else if (v > top2) { top2 = v; }
        }
        sh_sort_buffer[thread_id] = {top1 + top2, thread_id};
    }
    __syncthreads();

    // --- 3. Select top-k groups (insertion sort on n_group entries) ---
    if (thread_id == 0) {
        for (int i = 1; i < n_group; ++i) {
            KeyValuePair key = sh_sort_buffer[i];
            int j = i - 1;
            while (j >= 0 && sh_sort_buffer[j].value < key.value) {
                sh_sort_buffer[j + 1] = sh_sort_buffer[j];
                --j;
            }
            sh_sort_buffer[j + 1] = key;
        }
    }
    __syncthreads();

    // --- 4. Build mask: experts in top-k groups
    bool is_in_top_group = false;
    if (thread_id < n_routed_experts) {
        int group_of_expert = thread_id / experts_per_group;
        #pragma unroll
        for (int i = 0; i < 8; ++i) { // supports up to topk_group<=8 comfortably; loop will short-circuit
            if (i < topk_group && sh_sort_buffer[i].index == group_of_expert) {
                is_in_top_group = true;
                break;
            }
        }
    }
    __syncthreads();

    // --- 5. Prepare expert scores for global top-k and sort ---
    if (thread_id < n_routed_experts) {
        sh_sort_buffer[thread_id] = {is_in_top_group ? sh_biased_scores[thread_id] : -INFINITY, thread_id};
    } else if (thread_id < BLOCK_THREADS) {
        sh_sort_buffer[thread_id] = {-INFINITY, -1};
    }
    __syncthreads();

    block_bitonic_sort<BLOCK_THREADS>(sh_sort_buffer, thread_id);
    __syncthreads();

    // --- 6. Gather top-k pristine scores into a dedicated buffer and reduce ---
    if (thread_id < top_k) {
        int expert_idx = sh_sort_buffer[thread_id].index;
        sh_final_weights[thread_id] = (expert_idx >= 0) ? sh_original_scores[expert_idx] : 0.0f;
    }
    __syncthreads();

    // zero the rest of the reduction buffer safely (now it does NOT overlap sort buffer)
    if (thread_id >= top_k && thread_id < BLOCK_THREADS) {
        sh_final_weights[thread_id] = 0.0f;
    }
    __syncthreads();

    // reduce BLOCK_THREADS floats in-place
    for (int s = BLOCK_THREADS / 2; s > 0; s >>= 1) {
        if (thread_id < s) {
            sh_final_weights[thread_id] += sh_final_weights[thread_id + s];
        }
        __syncthreads();
    }

    // --- 7. Write outputs ---
    if (thread_id < top_k) {
        float denom = sh_final_weights[0] + 1e-20f;
        int expert_idx = sh_sort_buffer[thread_id].index;
        float w = (expert_idx >= 0) ? sh_original_scores[expert_idx] : 0.0f;
        float final_w = (w / denom) * routed_scaling_factor;

        topk_idx_out[token_idx * top_k + thread_id]    = static_cast<long>(expert_idx);
        topk_weight_out[token_idx * top_k + thread_id] = __float2bfloat16(final_w);
    }
}

// --- Launcher ---
std::vector<torch::Tensor> launch_moe_gate_forward_kernel(
    torch::Tensor hidden_states,            // [B*S, H] bfloat16
    torch::Tensor weight,                   // [N_EXPERTS, H] bfloat16
    torch::Tensor e_score_correction_bias,  // [N_EXPERTS] bfloat16
    int64_t n_group,
    int64_t topk_group,
    int64_t n_routed_experts,
    int64_t top_k,
    double routed_scaling_factor) {

    TORCH_CHECK(hidden_states.is_cuda(), "hidden_states must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(e_score_correction_bias.is_cuda(), "bias must be CUDA");
    TORCH_CHECK(hidden_states.dtype() == torch::kBFloat16, "Inputs must be BFloat16");
    TORCH_CHECK(n_group > 0 && (n_routed_experts % n_group == 0), "n_routed_experts must be divisible by n_group");

    const int bsz_seq_len = static_cast<int>(hidden_states.size(0));
    const int h = static_cast<int>(hidden_states.size(1));

    auto topk_idx    = torch::empty({bsz_seq_len, top_k}, torch::dtype(torch::kInt64).device(hidden_states.device()));
    auto topk_weight = torch::empty({bsz_seq_len, top_k}, hidden_states.options());

    constexpr int BLOCK_THREADS = 256;
    TORCH_CHECK(n_routed_experts <= BLOCK_THREADS, "n_routed_experts must be <= BLOCK_THREADS");
    const int experts_per_group = static_cast<int>(n_routed_experts / n_group);

    // shared memory: biased + original + final_weights(BLOCK_THREADS) + sort buffer(BLOCK_THREADS KeyValuePair) + small alignment slack
    size_t sh_mem_size =
        sizeof(float) * (2 * n_routed_experts + BLOCK_THREADS) +
        sizeof(KeyValuePair) * BLOCK_THREADS +
        alignof(KeyValuePair);

    dim3 grid(bsz_seq_len);
    dim3 block(BLOCK_THREADS);

    compiled_moe_gate_forward_kernel<BLOCK_THREADS><<<grid, block, sh_mem_size, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(hidden_states.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(e_score_correction_bias.data_ptr()),
        topk_idx.data_ptr<long>(),
        reinterpret_cast<__nv_bfloat16*>(topk_weight.data_ptr()),
        bsz_seq_len, h,
        static_cast<int>(n_group), static_cast<int>(topk_group),
        static_cast<int>(n_routed_experts), static_cast<int>(top_k),
        static_cast<float>(routed_scaling_factor),
        experts_per_group
    );
    AT_CUDA_CHECK(cudaGetLastError());
    return {topk_idx, topk_weight};
}
