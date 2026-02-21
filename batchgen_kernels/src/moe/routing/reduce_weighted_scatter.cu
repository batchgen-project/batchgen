/*
 * GPT-OSS-120B Reduce Kernel: Weighted Scatter-Add (CUDA).
 *
 * For each original token, iterate K=4 topk slots.
 * If topk_pos >= 0 (local expert), load expert output, multiply by weight, accumulate.
 * FP32 accumulation, BF16 output.
 *
 * Grid: (N, ceil(H / BLOCK_H))
 * Threads: BLOCK_H (256)
 * Each thread handles one hidden element per token.
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <c10/cuda/CUDAStream.h>

#define BLOCK_H 256


// ──────────────────────────────────────────────────────────────────────────────
// Reduce kernel: weighted scatter-add
// ──────────────────────────────────────────────────────────────────────────────

template <int K>
__global__ void reduce_weighted_scatter_kernel(
    const __nv_bfloat16* __restrict__ expert_output,  // [max_disp, H]
    const int32_t* __restrict__ topk_pos,             // [N * K]
    const float* __restrict__ topk_weights,           // [N, K]
    __nv_bfloat16* __restrict__ output,               // [N, H]
    int N, int H
) {
    const int token_idx = blockIdx.x;
    const int h_offset = blockIdx.y * BLOCK_H + threadIdx.x;

    if (token_idx >= N || h_offset >= H) return;

    // Load topk_pos and weights for this token into registers
    int32_t pos[K];
    float w[K];
    const int topk_base = token_idx * K;

    #pragma unroll
    for (int k = 0; k < K; k++) {
        pos[k] = topk_pos[topk_base + k];
        w[k] = topk_weights[topk_base + k];
    }

    // FP32 accumulation
    float acc = 0.0f;

    #pragma unroll
    for (int k = 0; k < K; k++) {
        if (pos[k] >= 0) {
            float val = __bfloat162float(
                expert_output[(int64_t)pos[k] * H + h_offset]);
            acc += val * w[k];
        }
    }

    // Store BF16 result
    output[(int64_t)token_idx * H + h_offset] = __float2bfloat16(acc);
}


// ──────────────────────────────────────────────────────────────────────────────
// Python wrapper
// ──────────────────────────────────────────────────────────────────────────────

torch::Tensor reduce_weighted_scatter_cuda(
    torch::Tensor expert_output,
    torch::Tensor topk_pos,
    torch::Tensor topk_weights,
    int64_t N,
    int64_t H,
    int64_t K,
    torch::Tensor output,
    int64_t num_valid_tokens
) {
    TORCH_CHECK(expert_output.is_cuda(), "expert_output must be CUDA tensor");
    TORCH_CHECK(expert_output.dtype() == torch::kBFloat16, "expert_output must be BF16");
    auto device = expert_output.device();

    if (H == 0) H = expert_output.size(1);

    // Effective token count for CUDA graph compatibility
    const int64_t N_eff = (num_valid_tokens > 0 && num_valid_tokens < N)
                          ? num_valid_tokens : N;

    // Allocate output if not pre-allocated
    if (!output.defined() || output.numel() == 0) {
        output = torch::empty({N, H}, torch::dtype(torch::kBFloat16).device(device));
    }

    // Launch kernel on current CUDA stream (required for CUDA graph capture)
    dim3 grid(N_eff, (H + BLOCK_H - 1) / BLOCK_H);
    dim3 block(BLOCK_H);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Dispatch K at compile time
    switch (K) {
        case 2:
            reduce_weighted_scatter_kernel<2><<<grid, block, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(expert_output.data_ptr()),
                topk_pos.data_ptr<int32_t>(),
                topk_weights.data_ptr<float>(),
                reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
                N, H);
            break;
        case 4:
            reduce_weighted_scatter_kernel<4><<<grid, block, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(expert_output.data_ptr()),
                topk_pos.data_ptr<int32_t>(),
                topk_weights.data_ptr<float>(),
                reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
                N, H);
            break;
        case 8:
            reduce_weighted_scatter_kernel<8><<<grid, block, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(expert_output.data_ptr()),
                topk_pos.data_ptr<int32_t>(),
                topk_weights.data_ptr<float>(),
                reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
                N, H);
            break;
        default:
            TORCH_CHECK(false, "Unsupported K=", K, ". Supported: 2, 4, 8");
    }

    return output;
}
