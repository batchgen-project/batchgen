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

// Compact-offsets support: BLOCK_M(=64)-aligned exclusive prefix sum of per-expert
// counts. Expert e's window = [offsets[e], offsets[e] + round_up(counts[e], 64)) —
// sized to actual routing, so ANY skew fits with <=63 pad rows per expert. E is tiny
// (384): a single serial thread is negligible and keeps this graph-capturable.
__global__ void compute_aligned_offsets_kernel(
    const int32_t* __restrict__ expert_counts,
    int32_t* __restrict__ expert_offsets,
    int E_local, int align
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int acc = 0;
        for (int e = 0; e < E_local; e++) {
            expert_offsets[e] = acc;
            acc += ((expert_counts[e] + align - 1) / align) * align;
        }
    }
}

__global__ void scatter_tokens_3d_kernel(
    const __nv_bfloat16* __restrict__ x,
    const int32_t* __restrict__ topk_indices,
    int32_t* __restrict__ expert_counters,
    __nv_bfloat16* __restrict__ act_buffer,
    int32_t* __restrict__ topk_pos,
    int NK, int H, int K,
    int expert_start, int E_local,
    int max_tokens_padded,
    int32_t* __restrict__ overflow_flag,
    const int32_t* __restrict__ expert_offsets
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
        if (expert_offsets != nullptr) {
            // Compact layout: window sized to round_up(count) — overflow impossible.
            write_pos = expert_offsets[local_expert] + relative_pos;
            topk_pos[itopk] = write_pos;
        } else if (relative_pos >= max_tokens_padded) {
            // Strided overflow: a hot expert exceeded its mtp stride window. Writing
            // would corrupt the NEXT expert's rows (OOB for expert E_local-1). Mark the
            // slot skipped (-1; the weighted reduce already treats -1 as zero
            // contribution) and raise the sticky flag for the host-side redo policy.
            write_pos = -1;
            topk_pos[itopk] = -1;
            if (overflow_flag != nullptr) *overflow_flag = 1;
        } else {
            write_pos = local_expert * max_tokens_padded + relative_pos;
            topk_pos[itopk] = write_pos;
        }
    }
    write_pos = __shfl_sync(0xffffffff, write_pos, 0);
    if (write_pos < 0) return;  // whole warp skips the vector write on overflow

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
    torch::Tensor topk_pos,
    c10::optional<torch::Tensor> overflow_flag,
    c10::optional<torch::Tensor> expert_offsets
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

    // Compact-offsets mode: BLOCK_M(64)-aligned prefix sum of the just-computed counts,
    // entirely on-GPU (no host sync); the scatter then writes at offsets[e]+slot.
    int32_t* offsets_ptr = nullptr;
    if (expert_offsets.has_value() && expert_offsets->defined() && expert_offsets->numel() > 0) {
        offsets_ptr = expert_offsets->data_ptr<int32_t>();
        compute_aligned_offsets_kernel<<<1, 1, 0, stream>>>(
            expert_counts.data_ptr<int32_t>(), offsets_ptr, E_local, 64);
    }

    {
        int total_threads = NK * WARP_SIZE;
        int threads_per_block = 256;
        int blocks = (total_threads + threads_per_block - 1) / threads_per_block;
        int32_t* overflow_ptr =
            (overflow_flag.has_value() && overflow_flag->defined() && overflow_flag->numel() > 0)
            ? overflow_flag->data_ptr<int32_t>() : nullptr;
        scatter_tokens_3d_kernel<<<blocks, threads_per_block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
            flat_indices.data_ptr<int32_t>(),
            expert_counters.data_ptr<int32_t>(),
            reinterpret_cast<__nv_bfloat16*>(act_buffer.data_ptr()),
            topk_pos.data_ptr<int32_t>(),
            NK, H, K, expert_start, E_local, max_tokens_padded,
            overflow_ptr, offsets_ptr);
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dispatch_scatter_3d", &dispatch_scatter_3d,
          "3D dispatch scatter for strided MoE buffer layout",
          py::arg("x"), py::arg("topk_indices"), py::arg("act_buffer"),
          py::arg("expert_start"), py::arg("num_local_experts"),
          py::arg("max_tokens_padded"), py::arg("expert_counts"),
          py::arg("expert_counters"), py::arg("topk_pos"),
          // Optional [1] int32 sticky overflow flag; None/omitted = no flag
          // (decode call sites unchanged — overflow silently skips as before,
          // now WITHOUT corrupting the next expert's rows).
          py::arg("overflow_flag") = py::none(),
          // Optional [E] int32 output: filled with 64-aligned prefix-sum bases and
          // used as the COMPACT write layout (skew-proof); None = strided e*mtp.
          py::arg("expert_offsets") = py::none());
    m.def("reduce_weighted_scatter", &reduce_weighted_scatter,
          "Weighted reduce scatter from 3D to flat layout");
}
