/*
 * K2.5 Gate Kernel: Fused Sigmoid + Top-K + Normalize + Scale (CUDA).
 *
 * Algorithm (K2.5 noaux_tc with n_group=1):
 *   1. sigmoid(logits) → scores [N, E]
 *   2. scores + e_score_correction → biased_scores [N, E]
 *   3. topk(biased_scores, k=8) → topk_idx [N, 8]
 *   4. gather(scores, topk_idx) → topk_weight [N, 8]  (raw sigmoid, NOT biased)
 *   5. normalize: topk_weight /= sum(topk_weight)
 *   6. scale: topk_weight *= routed_scaling_factor
 *
 * One block per token, 256 threads. Iterative argmax with warp-shuffle reduction.
 *
 * Input:  router_logits [N, E] FP32, e_score_correction [E] FP32
 * Output: topk_indices [N, K] int32, topk_weights [N, K] FP32
 *
 * K2.5: E=384, top_k=8, scoring_func=sigmoid, routed_scaling_factor=2.5
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <float.h>

#define WARP_SIZE_GST 32
#define MAX_EXPERTS_GST 512   // headroom beyond K2.5's 384
#define BLOCK_THREADS_GST 256

namespace {  // anonymous namespace to avoid symbol collision with gate_topk_softmax.cu

__device__ __forceinline__ void warp_reduce_argmax_sigmoid(float& val, int& idx) {
    #pragma unroll
    for (int offset = WARP_SIZE_GST / 2; offset > 0; offset /= 2) {
        float other_val = __shfl_down_sync(0xffffffff, val, offset);
        int other_idx = __shfl_down_sync(0xffffffff, idx, offset);
        if (other_val > val || (other_val == val && other_idx < idx)) {
            val = other_val;
            idx = other_idx;
        }
    }
}

template <int K>
__global__ void gate_sigmoid_topk_kernel(
    const float* __restrict__ router_logits,       // [N, E] FP32
    const float* __restrict__ e_score_correction,  // [E] FP32 bias for selection
    int32_t* __restrict__ topk_indices,            // [N, K] output
    float* __restrict__ topk_weights,              // [N, K] output
    float routed_scaling_factor,
    int N, int E
) {
    const int token_id = blockIdx.x;
    if (token_id >= N) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE_GST;
    const int lane_id = tid % WARP_SIZE_GST;
    const int num_warps = BLOCK_THREADS_GST / WARP_SIZE_GST;

    // Shared memory: scores (for gather) + biased scores (for top-K selection)
    __shared__ float s_scores[MAX_EXPERTS_GST];    // sigmoid(logits) — raw scores
    __shared__ float s_biased[MAX_EXPERTS_GST];    // sigmoid(logits) + bias — for selection
    __shared__ float s_warp_vals[8];
    __shared__ int s_warp_idxs[8];
    __shared__ float s_topk_vals[8];               // K <= 8
    __shared__ int s_topk_idxs[8];

    // Step 1: Load logits, compute sigmoid, add bias → shared memory
    const float* token_logits = router_logits + token_id * E;
    for (int i = tid; i < E; i += BLOCK_THREADS_GST) {
        float logit = token_logits[i];
        float score = 1.0f / (1.0f + expf(-logit));  // sigmoid
        s_scores[i] = score;
        s_biased[i] = score + e_score_correction[i];
    }
    __syncthreads();

    // Step 2: Iterative argmax on s_biased to find top-K
    #pragma unroll
    for (int k = 0; k < K; k++) {
        float local_max = -FLT_MAX;
        int local_idx = -1;
        for (int i = tid; i < E; i += BLOCK_THREADS_GST) {
            float v = s_biased[i];
            if (v > local_max || (v == local_max && i < local_idx)) {
                local_max = v;
                local_idx = i;
            }
        }

        // Warp-level reduction
        warp_reduce_argmax_sigmoid(local_max, local_idx);

        if (lane_id == 0) {
            s_warp_vals[warp_id] = local_max;
            s_warp_idxs[warp_id] = local_idx;
        }
        __syncthreads();

        // Final reduction in first warp
        if (warp_id == 0) {
            float val = (lane_id < num_warps) ? s_warp_vals[lane_id] : -FLT_MAX;
            int idx = (lane_id < num_warps) ? s_warp_idxs[lane_id] : -1;
            warp_reduce_argmax_sigmoid(val, idx);

            if (lane_id == 0) {
                s_topk_vals[k] = val;
                s_topk_idxs[k] = idx;
                s_biased[idx] = -FLT_MAX;  // mask out selected
            }
        }
        __syncthreads();
    }

    // Step 3: Gather raw sigmoid scores, normalize, scale (thread 0)
    if (tid == 0) {
        float raw_weights[8];
        float sum_w = 0.0f;

        #pragma unroll
        for (int k = 0; k < K; k++) {
            raw_weights[k] = s_scores[s_topk_idxs[k]];
            sum_w += raw_weights[k];
        }

        float inv_sum = 1.0f / (sum_w + 1e-20f);
        int32_t* out_idx = topk_indices + token_id * K;
        float* out_w = topk_weights + token_id * K;

        #pragma unroll
        for (int k = 0; k < K; k++) {
            out_idx[k] = s_topk_idxs[k];
            out_w[k] = raw_weights[k] * inv_sum * routed_scaling_factor;
        }
    }
}

}  // anonymous namespace


// ──────────────────────────────────────────────────────────────────────────────
// Python wrapper
// ──────────────────────────────────────────────────────────────────────────────

std::vector<torch::Tensor> gate_sigmoid_topk_cuda(
    torch::Tensor router_logits,
    torch::Tensor e_score_correction,
    int k,
    float routed_scaling_factor,
    torch::Tensor topk_indices,
    torch::Tensor topk_weights
) {
    TORCH_CHECK(router_logits.is_cuda(), "router_logits must be CUDA tensor");
    TORCH_CHECK(router_logits.dtype() == torch::kFloat32, "router_logits must be FP32");
    TORCH_CHECK(e_score_correction.is_cuda() && e_score_correction.dtype() == torch::kFloat32);

    const int N = router_logits.size(0);
    const int E = router_logits.size(1);
    TORCH_CHECK(E <= MAX_EXPERTS_GST, "E=", E, " exceeds MAX_EXPERTS=", MAX_EXPERTS_GST);
    TORCH_CHECK(e_score_correction.size(0) == E);

    if (!topk_indices.defined() || topk_indices.numel() == 0) {
        topk_indices = torch::empty({N, k}, torch::dtype(torch::kInt32).device(router_logits.device()));
    }
    if (!topk_weights.defined() || topk_weights.numel() == 0) {
        topk_weights = torch::empty({N, k}, torch::dtype(torch::kFloat32).device(router_logits.device()));
    }

    dim3 grid(N);
    dim3 block(BLOCK_THREADS_GST);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    switch (k) {
        case 2:
            gate_sigmoid_topk_kernel<2><<<grid, block, 0, stream>>>(
                router_logits.data_ptr<float>(), e_score_correction.data_ptr<float>(),
                topk_indices.data_ptr<int32_t>(), topk_weights.data_ptr<float>(),
                routed_scaling_factor, N, E);
            break;
        case 4:
            gate_sigmoid_topk_kernel<4><<<grid, block, 0, stream>>>(
                router_logits.data_ptr<float>(), e_score_correction.data_ptr<float>(),
                topk_indices.data_ptr<int32_t>(), topk_weights.data_ptr<float>(),
                routed_scaling_factor, N, E);
            break;
        case 8:
            gate_sigmoid_topk_kernel<8><<<grid, block, 0, stream>>>(
                router_logits.data_ptr<float>(), e_score_correction.data_ptr<float>(),
                topk_indices.data_ptr<int32_t>(), topk_weights.data_ptr<float>(),
                routed_scaling_factor, N, E);
            break;
        default:
            TORCH_CHECK(false, "Unsupported k=", k, ". Supported: 2, 4, 8");
    }

    return {topk_indices, topk_weights};
}
