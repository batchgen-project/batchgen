"""3D dispatch scatter + reduce kernels for strided MoE buffer layout.

Ported from GLM-5-FP8 layer_opt_pipeline (fp8_wgmma_pipeline.py).

Buffer layout: [E_local, max_tokens_padded, H] (3D strided).
Each expert e owns rows [e * mtp, (e+1) * mtp) in the flat [E*mtp, H] buffer.

dispatch_scatter_3d:
    Routes tokens from flat [G, H] into 3D [E*mtp, H] layout.
    Two-stage: count tokens per expert, then scatter with atomic counters.
    topk_pos stores absolute strided positions for reduce.

reduce_weighted_scatter:
    Weighted sum from 3D output back to flat [G, H] using topk_pos indices.
    FP32 accumulation, BF16 output. Template-specialized for K=2,4,8.
"""

import os
import logging

import torch
from torch.utils.cpp_extension import load_inline


_dispatch_reduce_module = None


CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <c10/cuda/CUDAStream.h>

#define WARP_SIZE 32

// ============================================================================
// dispatch_scatter_3d: Route tokens from flat [G, H] into 3D [E, mtp, H]
// ============================================================================
__global__ void count_tokens_3d_kernel(
    const int32_t* __restrict__ topk_indices,
    int32_t* __restrict__ expert_counts,
    int32_t* __restrict__ topk_pos,
    int NK, int expert_start, int E_local
) {
    extern __shared__ int32_t s_counts[];
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    for (int i = tid; i < E_local; i += stride) s_counts[i] = 0;
    __syncthreads();

    for (int i = tid; i < NK; i += stride) {
        topk_pos[i] = -1;
        int eid = topk_indices[i];
        int local_id = eid - expert_start;
        if (local_id >= 0 && local_id < E_local)
            atomicAdd(&s_counts[local_id], 1);
    }
    __syncthreads();

    for (int i = tid; i < E_local; i += stride)
        expert_counts[i] = s_counts[i];
}

__global__ void scatter_tokens_3d_kernel(
    const __nv_bfloat16* __restrict__ x,
    const int32_t* __restrict__ topk_indices,
    int32_t* __restrict__ expert_counters,
    __nv_bfloat16* __restrict__ act_buffer,
    int32_t* __restrict__ topk_pos,
    int NK, int H, int K,
    int expert_start, int E_local,
    int max_tokens_padded
) {
    const int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int warp_id = global_tid / WARP_SIZE;
    const int lane_id = global_tid % WARP_SIZE;

    if (warp_id >= NK) return;

    const int itopk = warp_id;
    const int token_id = itopk / K;
    const int eid = topk_indices[itopk];
    const int local_expert = eid - expert_start;

    if (local_expert < 0 || local_expert >= E_local) return;

    int write_pos;
    if (lane_id == 0) {
        int relative_pos = atomicAdd(&expert_counters[local_expert], 1);
        write_pos = local_expert * max_tokens_padded + relative_pos;
        topk_pos[itopk] = write_pos;
    }
    write_pos = __shfl_sync(0xffffffff, write_pos, 0);

    const int vec_size = 8;
    const int vec_count = H / vec_size;
    const int remainder = H % vec_size;

    const float4* src = reinterpret_cast<const float4*>(x + (int64_t)token_id * H);
    float4* dst = reinterpret_cast<float4*>(act_buffer + (int64_t)write_pos * H);

    for (int v = lane_id; v < vec_count; v += WARP_SIZE)
        dst[v] = src[v];

    if (remainder > 0 && lane_id == 0) {
        const __nv_bfloat16* src_r = x + (int64_t)token_id * H + vec_count * vec_size;
        __nv_bfloat16* dst_r = act_buffer + (int64_t)write_pos * H + vec_count * vec_size;
        for (int i = 0; i < remainder; i++) dst_r[i] = src_r[i];
    }
}

std::vector<torch::Tensor> dispatch_scatter_3d(
    torch::Tensor x,
    torch::Tensor topk_indices,
    torch::Tensor act_buffer,
    int64_t expert_start,
    int64_t num_local_experts,
    int64_t max_tokens_padded,
    torch::Tensor expert_counts,
    torch::Tensor expert_counters,
    torch::Tensor topk_pos
) {
    const int N = topk_indices.size(0);
    const int K = topk_indices.size(1);
    const int H = x.size(1);
    const int NK = N * K;
    const int E_local = num_local_experts;

    expert_counts.zero_();
    expert_counters.zero_();

    auto flat_indices = topk_indices.reshape({-1}).contiguous();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    {
        int threads = 256;
        int blocks = 1;
        int smem_bytes = E_local * sizeof(int32_t);
        count_tokens_3d_kernel<<<blocks, threads, smem_bytes, stream>>>(
            flat_indices.data_ptr<int32_t>(),
            expert_counts.data_ptr<int32_t>(),
            topk_pos.data_ptr<int32_t>(),
            NK, expert_start, E_local);
    }

    {
        int total_threads = NK * WARP_SIZE;
        int threads_per_block = 256;
        int blocks = (total_threads + threads_per_block - 1) / threads_per_block;
        scatter_tokens_3d_kernel<<<blocks, threads_per_block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
            flat_indices.data_ptr<int32_t>(),
            expert_counters.data_ptr<int32_t>(),
            reinterpret_cast<__nv_bfloat16*>(act_buffer.data_ptr()),
            topk_pos.data_ptr<int32_t>(),
            NK, H, K, expert_start, E_local, max_tokens_padded);
    }

    return {expert_counts, topk_pos};
}

// ============================================================================
// reduce_weighted_scatter: Weighted sum from 3D output back to flat [G, H]
// ============================================================================
#define BLOCK_H 256

template <int K>
__global__ void reduce_weighted_scatter_kernel(
    const __nv_bfloat16* __restrict__ expert_output,
    const int32_t* __restrict__ topk_pos,
    const float* __restrict__ topk_weights,
    __nv_bfloat16* __restrict__ output,
    int N, int H
) {
    const int token_idx = blockIdx.x;
    const int h_offset = blockIdx.y * BLOCK_H + threadIdx.x;
    if (token_idx >= N || h_offset >= H) return;

    int32_t pos[K];
    float w[K];
    const int topk_base = token_idx * K;
    #pragma unroll
    for (int k = 0; k < K; k++) {
        pos[k] = topk_pos[topk_base + k];
        w[k] = topk_weights[topk_base + k];
    }

    float acc = 0.0f;
    #pragma unroll
    for (int k = 0; k < K; k++) {
        if (pos[k] >= 0) {
            float val = __bfloat162float(expert_output[(int64_t)pos[k] * H + h_offset]);
            acc += val * w[k];
        }
    }
    output[(int64_t)token_idx * H + h_offset] = __float2bfloat16(acc);
}

torch::Tensor reduce_weighted_scatter(
    torch::Tensor expert_output, torch::Tensor topk_pos,
    torch::Tensor topk_weights, int64_t N, int64_t H, int64_t K,
    torch::Tensor output
) {
    auto device = expert_output.device();
    if (!output.defined() || output.numel() == 0)
        output = torch::zeros({N, H}, torch::dtype(torch::kBFloat16).device(device));

    dim3 grid(N, (H + BLOCK_H - 1) / BLOCK_H);
    dim3 block(BLOCK_H);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    switch (K) {
        case 2: reduce_weighted_scatter_kernel<2><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(expert_output.data_ptr()),
            topk_pos.data_ptr<int32_t>(), topk_weights.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), N, H); break;
        case 4: reduce_weighted_scatter_kernel<4><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(expert_output.data_ptr()),
            topk_pos.data_ptr<int32_t>(), topk_weights.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), N, H); break;
        case 8: reduce_weighted_scatter_kernel<8><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(expert_output.data_ptr()),
            topk_pos.data_ptr<int32_t>(), topk_weights.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), N, H); break;
        default: TORCH_CHECK(false, "Unsupported K=", K);
    }
    return output;
}
'''

CPP_SOURCE = r'''
#include <torch/extension.h>

std::vector<torch::Tensor> dispatch_scatter_3d(
    torch::Tensor x, torch::Tensor topk_indices,
    torch::Tensor act_buffer, int64_t expert_start,
    int64_t num_local_experts, int64_t max_tokens_padded,
    torch::Tensor expert_counts, torch::Tensor expert_counters,
    torch::Tensor topk_pos);

torch::Tensor reduce_weighted_scatter(
    torch::Tensor expert_output, torch::Tensor topk_pos,
    torch::Tensor topk_weights, int64_t N, int64_t H, int64_t K,
    torch::Tensor output);
'''


def _load_dispatch_reduce_module():
    """Build and cache the dispatch_scatter_3d + reduce_weighted_scatter module."""
    global _dispatch_reduce_module
    if _dispatch_reduce_module is not None:
        return _dispatch_reduce_module

    logging.info("[dispatch_scatter_3d] Building CUDA kernels...")
    os.environ.setdefault("MAX_JOBS", "8")
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0a")

    _dispatch_reduce_module = load_inline(
        name="dispatch_reduce_3d",
        cpp_sources=[CPP_SOURCE],
        cuda_sources=[CUDA_SOURCE],
        functions=["dispatch_scatter_3d", "reduce_weighted_scatter"],
        extra_cuda_cflags=[
            "-O3", "-arch=sm_90a", "--ptxas-options=-v", "-lineinfo",
        ],
        verbose=False,
    )
    logging.info("[dispatch_scatter_3d] Build complete.")
    return _dispatch_reduce_module


def dispatch_scatter_3d(
    x: torch.Tensor,
    topk_indices: torch.Tensor,
    act_buffer: torch.Tensor,
    expert_start: int,
    num_local_experts: int,
    max_tokens_padded: int,
    expert_counts: torch.Tensor,
    expert_counters: torch.Tensor,
    topk_pos: torch.Tensor,
):
    """Route tokens from flat [G, H] into 3D strided [E*mtp, H] buffer.

    Args:
        x: Input tokens [G, H] BF16
        topk_indices: Expert assignments [G, K] int32
        act_buffer: Pre-allocated 3D buffer [E_local * mtp, H] BF16
        expert_start: Global index of first local expert
        num_local_experts: Number of local experts
        max_tokens_padded: Stride per expert (mtp)
        expert_counts: Pre-allocated [E_local] int32 (zeroed internally)
        expert_counters: Pre-allocated [E_local] int32 (zeroed internally)
        topk_pos: Pre-allocated [G*K] int32 (set to strided positions)

    Returns:
        (expert_counts, topk_pos) — expert_counts[e] = tokens routed to expert e,
        topk_pos[i] = absolute row index in act_buffer (or -1 if non-local)
    """
    mod = _load_dispatch_reduce_module()
    return mod.dispatch_scatter_3d(
        x, topk_indices, act_buffer,
        expert_start, num_local_experts, max_tokens_padded,
        expert_counts, expert_counters, topk_pos,
    )


def reduce_weighted_scatter(
    expert_output: torch.Tensor,
    topk_pos: torch.Tensor,
    topk_weights: torch.Tensor,
    N: int,
    H: int,
    K: int,
    output: torch.Tensor = None,
) -> torch.Tensor:
    """Weighted sum from 3D expert output back to flat [N, H].

    Args:
        expert_output: 3D strided buffer [E*mtp, H] BF16
        topk_pos: Strided positions [N*K] int32
        topk_weights: Routing weights [N, K] FP32
        N: Number of original tokens
        H: Hidden dimension
        K: Top-k value
        output: Pre-allocated output [N, H] BF16 (optional)

    Returns:
        output [N, H] BF16
    """
    mod = _load_dispatch_reduce_module()
    if output is None:
        output = torch.zeros(N, H, dtype=torch.bfloat16, device=expert_output.device)
    return mod.reduce_weighted_scatter(
        expert_output, topk_pos, topk_weights, N, H, K, output,
    )
