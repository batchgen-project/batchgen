/*
 * Fused Router Epilogue: BF16 bias add + BF16→FP32 cast.
 *
 * Eliminates 2 kernel launches per MoE layer by fusing:
 *   router_logits.add_(bias)   →  elementwise kernel
 *   router_f32.copy_(logits)   →  dtype cast kernel
 * into a single pass:
 *   router_f32[i] = (float)(router_logits[i] + bias[i % E])
 *
 * Input:  router_logits [N, E] BF16 (from cuBLAS matmul), bias [E] BF16
 * Output: router_f32 [N, E] FP32 (for gate_topk_softmax)
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <c10/cuda/CUDAStream.h>

__global__ void router_bias_cast_kernel(
    const __nv_bfloat16* __restrict__ logits,  // [N, E] bf16
    const __nv_bfloat16* __restrict__ bias,    // [E] bf16 (nullptr if no bias)
    float* __restrict__ output,                // [N, E] f32
    int N, int E
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * E;
    if (idx >= total) return;

    float val = __bfloat162float(logits[idx]);
    if (bias != nullptr) {
        val += __bfloat162float(bias[idx % E]);
    }
    output[idx] = val;
}


void router_bias_cast_cuda(
    torch::Tensor logits,       // [N, E] bf16
    torch::Tensor bias,         // [E] bf16 or empty
    torch::Tensor output        // [N, E] f32
) {
    TORCH_CHECK(logits.is_cuda(), "logits must be CUDA");
    TORCH_CHECK(output.is_cuda(), "output must be CUDA");

    const int N = logits.size(0);
    const int E = logits.size(1);
    const int total = N * E;

    const __nv_bfloat16* bias_ptr = nullptr;
    if (bias.defined() && bias.numel() > 0) {
        bias_ptr = reinterpret_cast<const __nv_bfloat16*>(bias.data_ptr());
    }

    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    router_bias_cast_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr()),
        bias_ptr,
        output.data_ptr<float>(),
        N, E
    );
}
