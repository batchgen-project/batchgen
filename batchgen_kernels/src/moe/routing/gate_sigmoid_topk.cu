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
 * Input:  router_logits [N, E] FP32 (row stride may exceed E), e_score_correction [E] FP32
 * Output: topk_indices [N, K] int32, topk_weights [N, K] FP32
 *
 * K2.5: E=384, top_k=8, scoring_func=sigmoid, routed_scaling_factor=2.5
 * K3:   E=896, top_k=16 — the router logits are the leading columns of a fused
 *       router/down-projection GEMM output, hence the row-stride support, and
 *       the valid-row count is a device-resident scalar so a CUDA graph can
 *       vary the live token count across replays without a host read.
 *
 * Optional latent epilogue (K3 only): the same fused GEMM row also carries the
 * latent down-projection columns at ``latent_offset``.  When ``latent_out`` is
 * given, this kernel casts that suffix to BF16 while it is already streaming
 * the row, which removes the separate strided FP32->BF16 contiguous copy from
 * the decode graph.  Padding rows write zeros, matching the zero rows the fused
 * GEMM produces for a zero-padded activation.
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <c10/cuda/CUDAStream.h>
#include <float.h>

#define WARP_SIZE_GST 32
#define MAX_EXPERTS_SMALL_GST 512   // headroom beyond K2.5's 384
#define MAX_EXPERTS_GST 1024        // headroom beyond K3's 896
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

template <int K, int MAX_E>
__global__ void gate_sigmoid_topk_kernel(
    const float* __restrict__ router_logits,       // [N, E] FP32, row stride logits_stride
    const float* __restrict__ e_score_correction,  // [E] FP32 bias for selection
    int32_t* __restrict__ topk_indices,            // [N, K] output
    float* __restrict__ topk_weights,              // [N, K] output
    const int32_t* __restrict__ num_valid_tokens,  // device scalar, or nullptr for "all N"
    __nv_bfloat16* __restrict__ latent_out,        // [N, L] output, or nullptr to skip
    float routed_scaling_factor,
    int N, int E, int logits_stride,
    int latent_offset, int latent_size
) {
    const int token_id = blockIdx.x;
    if (token_id >= N) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE_GST;
    const int lane_id = tid % WARP_SIZE_GST;
    const int num_warps = BLOCK_THREADS_GST / WARP_SIZE_GST;

    // Padding rows: the live row count is only known on the device, so the grid
    // always covers the full static bucket.  The whole block takes this branch
    // together, so the early return never straddles a __syncthreads() below.
    if (num_valid_tokens != nullptr && token_id >= *num_valid_tokens) {
        if (tid == 0) {
            int32_t* out_idx = topk_indices + token_id * K;
            float* out_w = topk_weights + token_id * K;
            #pragma unroll
            for (int k = 0; k < K; k++) {
                out_idx[k] = -1;
                out_w[k] = 0.0f;
            }
        }
        // Deterministic padding latent: the zero-padded activation makes the
        // bias-free fused GEMM produce zero rows anyway.  Use canonical +0 so
        // downstream EP gather sees numerically the same padding as before.
        if (latent_out != nullptr) {
            __nv_bfloat16* out_latent = latent_out + (size_t)token_id * latent_size;
            for (int i = tid; i < latent_size; i += BLOCK_THREADS_GST) {
                out_latent[i] = __float2bfloat16(0.0f);
            }
        }
        return;
    }

    // Shared memory: scores (for gather) + biased scores (for top-K selection)
    __shared__ float s_scores[MAX_E];    // sigmoid(logits) — raw scores
    __shared__ float s_biased[MAX_E];    // sigmoid(logits) + bias — for selection
    __shared__ float s_warp_vals[8];
    __shared__ int s_warp_idxs[8];
    __shared__ float s_topk_vals[K];
    __shared__ int s_topk_idxs[K];

    // Step 1: Load logits, compute sigmoid, add bias → shared memory
    const float* token_logits = router_logits + (size_t)token_id * logits_stride;
    for (int i = tid; i < E; i += BLOCK_THREADS_GST) {
        float logit = token_logits[i];
        float score = 1.0f / (1.0f + expf(-logit));  // sigmoid
        s_scores[i] = score;
        s_biased[i] = score + e_score_correction[i];
    }

    // Step 1b: cast the latent suffix of the same row while it is hot.  This is
    // independent of the top-K reduction below and touches no shared memory.
    if (latent_out != nullptr) {
        const float* token_latent = token_logits + latent_offset;
        __nv_bfloat16* out_latent = latent_out + (size_t)token_id * latent_size;
        for (int i = tid; i < latent_size; i += BLOCK_THREADS_GST) {
            out_latent[i] = __float2bfloat16(token_latent[i]);
        }
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
        float raw_weights[K];
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

// MAX_E is a template parameter so that adding K3's 896-expert support does not
// grow the shared-memory footprint (and shrink occupancy) of the E<=512 callers.
template <int MAX_E>
void launch_gate_sigmoid_topk(
    int k,
    const float* router_logits,
    const float* e_score_correction,
    int32_t* topk_indices,
    float* topk_weights,
    const int32_t* num_valid_tokens,
    __nv_bfloat16* latent_out,
    float routed_scaling_factor,
    int N, int E, int logits_stride,
    int latent_offset, int latent_size,
    cudaStream_t stream
) {
    dim3 grid(N);
    dim3 block(BLOCK_THREADS_GST);

#define GST_LAUNCH(KK)                                                       \
    gate_sigmoid_topk_kernel<KK, MAX_E><<<grid, block, 0, stream>>>(          \
        router_logits, e_score_correction, topk_indices, topk_weights,        \
        num_valid_tokens, latent_out, routed_scaling_factor, N, E,            \
        logits_stride, latent_offset, latent_size)

    switch (k) {
        case 2:  GST_LAUNCH(2);  break;
        case 4:  GST_LAUNCH(4);  break;
        case 8:  GST_LAUNCH(8);  break;
        case 16: GST_LAUNCH(16); break;
        default:
            TORCH_CHECK(false, "Unsupported k=", k, ". Supported: 2, 4, 8, 16");
    }
#undef GST_LAUNCH
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
    torch::Tensor topk_weights,
    torch::Tensor num_valid_tokens,
    torch::Tensor latent_out,
    int64_t latent_offset
) {
    TORCH_CHECK(router_logits.is_cuda(), "router_logits must be CUDA tensor");
    TORCH_CHECK(router_logits.dtype() == torch::kFloat32, "router_logits must be FP32");
    TORCH_CHECK(router_logits.dim() == 2, "router_logits must be 2-D");
    // Rows may be strided (a fused GEMM's leading columns), but each row must be
    // contiguous so a thread block can stride over experts.
    TORCH_CHECK(router_logits.stride(1) == 1, "router_logits rows must be contiguous");
    TORCH_CHECK(e_score_correction.is_cuda() && e_score_correction.dtype() == torch::kFloat32);
    TORCH_CHECK(e_score_correction.device() == router_logits.device(),
                "e_score_correction must be on the router_logits device");

    const int N = router_logits.size(0);
    const int E = router_logits.size(1);
    const int logits_stride = static_cast<int>(router_logits.stride(0));
    TORCH_CHECK(E <= MAX_EXPERTS_GST, "E=", E, " exceeds MAX_EXPERTS=", MAX_EXPERTS_GST);
    // The iterative argmax masks one expert per round, so fewer experts than
    // routes would leave a round with no candidate to select.
    TORCH_CHECK(E >= k, "E=", E, " must be at least k=", k);
    TORCH_CHECK(e_score_correction.size(0) == E);

    if (!topk_indices.defined() || topk_indices.numel() == 0) {
        topk_indices = torch::empty({N, k}, torch::dtype(torch::kInt32).device(router_logits.device()));
    }
    if (!topk_weights.defined() || topk_weights.numel() == 0) {
        topk_weights = torch::empty({N, k}, torch::dtype(torch::kFloat32).device(router_logits.device()));
    }

    // A device-resident scalar keeps the live row count off the host, so one
    // captured graph can be replayed at different valid-token counts.
    const int32_t* valid_ptr = nullptr;
    if (num_valid_tokens.defined() && num_valid_tokens.numel() > 0) {
        TORCH_CHECK(num_valid_tokens.is_cuda(), "num_valid_tokens must be a CUDA tensor");
        TORCH_CHECK(num_valid_tokens.dtype() == torch::kInt32,
                    "num_valid_tokens must be int32");
        TORCH_CHECK(num_valid_tokens.numel() == 1,
                    "num_valid_tokens must contain one element");
        TORCH_CHECK(num_valid_tokens.device() == router_logits.device(),
                    "num_valid_tokens must be on the router_logits device");
        valid_ptr = num_valid_tokens.data_ptr<int32_t>();
    }

    // Optional latent epilogue: ``router_logits`` is the leading-expert view of
    // a wider fused GEMM output, so the latent columns live in the same rows at
    // ``latent_offset``.  Everything is validated against the row stride so a
    // caller that passes a genuinely contiguous [N, E] buffer fails closed
    // instead of reading the next row.
    __nv_bfloat16* latent_ptr = nullptr;
    int latent_off = 0;
    int latent_size = 0;
    if (latent_out.defined() && latent_out.numel() > 0) {
        TORCH_CHECK(latent_out.is_cuda(), "latent_out must be a CUDA tensor");
        TORCH_CHECK(latent_out.dtype() == torch::kBFloat16, "latent_out must be BF16");
        TORCH_CHECK(latent_out.dim() == 2, "latent_out must be 2-D");
        TORCH_CHECK(latent_out.is_contiguous(), "latent_out must be contiguous");
        TORCH_CHECK(latent_out.device() == router_logits.device(),
                    "latent_out must be on the router_logits device");
        TORCH_CHECK(latent_out.size(0) == N,
                    "latent_out must have ", N, " rows, got ", latent_out.size(0));
        // Bound the int64 offset before narrowing so a wild value cannot wrap
        // into a plausible-looking int.
        TORCH_CHECK(latent_offset >= 0 && latent_offset <= logits_stride,
                    "latent_offset=", latent_offset,
                    " is outside the router_logits row stride ", logits_stride);
        latent_off = static_cast<int>(latent_offset);
        latent_size = static_cast<int>(latent_out.size(1));
        TORCH_CHECK(latent_off >= E,
                    "latent_offset=", latent_off,
                    " must start at or after the ", E, " router columns");
        TORCH_CHECK(latent_off + latent_size <= logits_stride,
                    "latent columns [", latent_off, ", ", latent_off + latent_size,
                    ") exceed the router_logits row stride ", logits_stride);
        latent_ptr = reinterpret_cast<__nv_bfloat16*>(latent_out.data_ptr());
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (E <= MAX_EXPERTS_SMALL_GST) {
        launch_gate_sigmoid_topk<MAX_EXPERTS_SMALL_GST>(
            k, router_logits.data_ptr<float>(), e_score_correction.data_ptr<float>(),
            topk_indices.data_ptr<int32_t>(), topk_weights.data_ptr<float>(),
            valid_ptr, latent_ptr, routed_scaling_factor, N, E, logits_stride,
            latent_off, latent_size, stream);
    } else {
        launch_gate_sigmoid_topk<MAX_EXPERTS_GST>(
            k, router_logits.data_ptr<float>(), e_score_correction.data_ptr<float>(),
            topk_indices.data_ptr<int32_t>(), topk_weights.data_ptr<float>(),
            valid_ptr, latent_ptr, routed_scaling_factor, N, E, logits_stride,
            latent_off, latent_size, stream);
    }

    return {topk_indices, topk_weights};
}
