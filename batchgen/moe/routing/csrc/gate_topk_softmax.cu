/*
 * GPT-OSS-120B Gate Kernel: Fused Top-K + Softmax (CUDA).
 *
 * One block per token, 256 threads.
 * Iterative argmax with warp-shuffle reduction + numerically stable softmax.
 *
 * Input:  router_logits [N, E] FP32 (E=128)
 * Output: topk_indices [N, K] int32, topk_weights [N, K] FP32
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <float.h>

#define WARP_SIZE 32
#define MAX_EXPERTS 128
#define BLOCK_THREADS 256


// ──────────────────────────────────────────────────────────────────────────────
// Warp-level argmax reduction
// ──────────────────────────────────────────────────────────────────────────────

__device__ __forceinline__ void warp_reduce_argmax(float& val, int& idx) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        float other_val = __shfl_down_sync(0xffffffff, val, offset);
        int other_idx = __shfl_down_sync(0xffffffff, idx, offset);
        if (other_val > val || (other_val == val && other_idx < idx)) {
            val = other_val;
            idx = other_idx;
        }
    }
}


// ──────────────────────────────────────────────────────────────────────────────
// Gate kernel: top-K selection + softmax
// ──────────────────────────────────────────────────────────────────────────────

template <int K>
__global__ void gate_topk_softmax_kernel(
    const float* __restrict__ router_logits,  // [N, E]
    int32_t* __restrict__ topk_indices,       // [N, K]
    float* __restrict__ topk_weights,         // [N, K]
    int N, int E
) {
    const int token_id = blockIdx.x;
    if (token_id >= N) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int num_warps = BLOCK_THREADS / WARP_SIZE;

    // Shared memory for logits + reduction
    __shared__ float s_logits[MAX_EXPERTS];
    __shared__ float s_warp_vals[8];  // max 8 warps with 256 threads
    __shared__ int s_warp_idxs[8];
    __shared__ float s_topk_vals[8];  // K <= 8
    __shared__ int s_topk_idxs[8];

    // Load logits to shared memory (coalesced)
    const float* token_logits = router_logits + token_id * E;
    for (int i = tid; i < E; i += BLOCK_THREADS) {
        s_logits[i] = token_logits[i];
    }
    __syncthreads();

    // Iterative argmax: find top-K by successive max + masking
    #pragma unroll
    for (int k = 0; k < K; k++) {
        // Each thread scans its subset
        float local_max = -FLT_MAX;
        int local_idx = -1;
        for (int i = tid; i < E; i += BLOCK_THREADS) {
            float v = s_logits[i];
            if (v > local_max || (v == local_max && i < local_idx)) {
                local_max = v;
                local_idx = i;
            }
        }

        // Warp-level reduction
        warp_reduce_argmax(local_max, local_idx);

        // Lane 0 of each warp writes to smem
        if (lane_id == 0) {
            s_warp_vals[warp_id] = local_max;
            s_warp_idxs[warp_id] = local_idx;
        }
        __syncthreads();

        // Final reduction in first warp
        if (warp_id == 0) {
            float val = (lane_id < num_warps) ? s_warp_vals[lane_id] : -FLT_MAX;
            int idx = (lane_id < num_warps) ? s_warp_idxs[lane_id] : -1;
            warp_reduce_argmax(val, idx);

            if (lane_id == 0) {
                s_topk_vals[k] = val;
                s_topk_idxs[k] = idx;
                // Mask out selected expert
                s_logits[idx] = -FLT_MAX;
            }
        }
        __syncthreads();
    }

    // Numerically stable softmax over K selected values (thread 0 only)
    if (tid == 0) {
        float max_val = s_topk_vals[0];
        #pragma unroll
        for (int k = 1; k < K; k++) {
            max_val = fmaxf(max_val, s_topk_vals[k]);
        }

        float sum_exp = 0.0f;
        float exp_vals[8];
        #pragma unroll
        for (int k = 0; k < K; k++) {
            exp_vals[k] = expf(s_topk_vals[k] - max_val);
            sum_exp += exp_vals[k];
        }

        float inv_sum = 1.0f / sum_exp;
        int32_t* out_idx = topk_indices + token_id * K;
        float* out_w = topk_weights + token_id * K;
        #pragma unroll
        for (int k = 0; k < K; k++) {
            out_idx[k] = s_topk_idxs[k];
            out_w[k] = exp_vals[k] * inv_sum;
        }
    }
}


// ──────────────────────────────────────────────────────────────────────────────
// Python wrapper
// ──────────────────────────────────────────────────────────────────────────────

std::vector<torch::Tensor> gate_topk_softmax_cuda(
    torch::Tensor router_logits,
    int k,
    torch::Tensor topk_indices,
    torch::Tensor topk_weights,
    int64_t num_valid_tokens
) {
    TORCH_CHECK(router_logits.is_cuda(), "router_logits must be CUDA tensor");
    TORCH_CHECK(router_logits.dtype() == torch::kFloat32, "router_logits must be FP32");

    const int N = router_logits.size(0);
    const int E = router_logits.size(1);

    // Effective token count for CUDA graph compatibility
    const int N_eff = (num_valid_tokens > 0 && num_valid_tokens < N)
                      ? static_cast<int>(num_valid_tokens) : N;

    // Allocate outputs if not pre-allocated
    if (!topk_indices.defined() || topk_indices.numel() == 0) {
        topk_indices = torch::empty({N, k}, torch::dtype(torch::kInt32).device(router_logits.device()));
    }
    if (!topk_weights.defined() || topk_weights.numel() == 0) {
        topk_weights = torch::empty({N, k}, torch::dtype(torch::kFloat32).device(router_logits.device()));
    }

    // Launch kernel on current CUDA stream (required for CUDA graph capture)
    dim3 grid(N_eff);  // only process valid tokens
    dim3 block(BLOCK_THREADS);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Dispatch K at compile time for best unrolling
    switch (k) {
        case 2:
            gate_topk_softmax_kernel<2><<<grid, block, 0, stream>>>(
                router_logits.data_ptr<float>(),
                topk_indices.data_ptr<int32_t>(),
                topk_weights.data_ptr<float>(),
                N, E);
            break;
        case 4:
            gate_topk_softmax_kernel<4><<<grid, block, 0, stream>>>(
                router_logits.data_ptr<float>(),
                topk_indices.data_ptr<int32_t>(),
                topk_weights.data_ptr<float>(),
                N, E);
            break;
        case 8:
            gate_topk_softmax_kernel<8><<<grid, block, 0, stream>>>(
                router_logits.data_ptr<float>(),
                topk_indices.data_ptr<int32_t>(),
                topk_weights.data_ptr<float>(),
                N, E);
            break;
        default:
            TORCH_CHECK(false, "Unsupported k=", k, ". Supported: 2, 4, 8");
    }

    return {topk_indices, topk_weights};
}
