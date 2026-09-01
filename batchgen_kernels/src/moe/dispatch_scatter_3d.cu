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
// dispatch_scatter_ragged: Route tokens from flat [G, H] into the COMPACT
// ragged buffer.
//
// Expert e owns rows [cu_seqlens[e], cu_seqlens[e] + counts[e]) and
// cu_seqlens[e+1] = cu_seqlens[e] + round_up(counts[e], row_align).
//
// row_align exists only so that the FP8 blockwise grouped GEMM can address
// x_scale in the same row space (scale tile index = cu_seqlens[e] / TileM);
// it must be a multiple of the largest supported TileM (64). The alignment
// holes are never written and never read: the GEMM's per-expert TMA extent is
// counts[e], so loads past it are hardware zero-filled and stores past it are
// clipped.
//
// Total rows are bounded by NK + E_local * (row_align - 1), a host constant —
// checked below, never silently regrown.
// ============================================================================
__global__ void count_tokens_ragged_kernel(
    const int32_t* __restrict__ topk_indices,
    int32_t* __restrict__ expert_counts,
    int32_t* __restrict__ cu_seqlens,
    int32_t* __restrict__ topk_pos,
    int NK, int expert_start, int E_local, int row_align
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

    // Single-thread exclusive scan of the aligned counts. E_local is <= 64 on
    // every shipped config and this kernel is a single block already launch
    // latency bound, so a serial scan is cheaper than a cooperative one.
    if (tid == 0) {
        int acc = 0;
        for (int e = 0; e < E_local; e++) {
            cu_seqlens[e] = acc;
            acc += ((s_counts[e] + row_align - 1) / row_align) * row_align;
        }
        cu_seqlens[E_local] = acc;
    }
}

__global__ void scatter_tokens_ragged_kernel(
    const __nv_bfloat16* __restrict__ x,
    const int32_t* __restrict__ topk_indices,
    int32_t* __restrict__ expert_counters,
    __nv_bfloat16* __restrict__ act_buffer,
    const int32_t* __restrict__ cu_seqlens,
    int32_t* __restrict__ topk_pos,
    int NK, int H, int K,
    int expert_start, int E_local
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
        write_pos = cu_seqlens[local_expert] + relative_pos;
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

std::vector<torch::Tensor> dispatch_scatter_ragged(
    torch::Tensor x,
    torch::Tensor topk_indices,
    torch::Tensor act_buffer,
    int64_t expert_start,
    int64_t num_local_experts,
    int64_t row_align,
    torch::Tensor expert_counts,
    torch::Tensor expert_counters,
    torch::Tensor cu_seqlens,
    torch::Tensor topk_pos
) {
    const int N = topk_indices.size(0);
    const int K = topk_indices.size(1);
    const int H = x.size(1);
    const int NK = N * K;
    const int E_local = num_local_experts;

    TORCH_CHECK(row_align > 0 && row_align % 64 == 0,
                "row_align must be a positive multiple of 64 (largest grouped-GEMM TileM), got ",
                row_align);
    TORCH_CHECK(cu_seqlens.size(0) == E_local + 1,
                "cu_seqlens must be [E_local+1] = [", E_local + 1, "], got ", cu_seqlens.size(0));
    TORCH_CHECK(cu_seqlens.dtype() == torch::kInt32, "cu_seqlens must be int32");
    TORCH_CHECK(topk_pos.numel() >= NK,
                "topk_pos too small: need ", NK, ", got ", topk_pos.numel());
    // Hard static bound — the compact layout has no per-expert capacity to
    // overflow, only a total. sum_e round_up(count_e, row_align)
    //   <= sum_e count_e + E*(row_align-1) <= NK + E*(row_align-1).
    const int64_t max_rows = (int64_t)NK + (int64_t)E_local * (row_align - 1);
    TORCH_CHECK(act_buffer.size(0) >= max_rows,
                "ragged MoE buffer too small: need >= ", max_rows,
                " rows (NK=", NK, ", E=", E_local, ", align=", row_align,
                "), got ", act_buffer.size(0));

    expert_counts.zero_();
    expert_counters.zero_();

    auto flat_indices = topk_indices.reshape({-1}).contiguous();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    {
        int threads = 256;
        int blocks = 1;
        int smem_bytes = E_local * sizeof(int32_t);
        count_tokens_ragged_kernel<<<blocks, threads, smem_bytes, stream>>>(
            flat_indices.data_ptr<int32_t>(),
            expert_counts.data_ptr<int32_t>(),
            cu_seqlens.data_ptr<int32_t>(),
            topk_pos.data_ptr<int32_t>(),
            NK, expert_start, E_local, (int)row_align);
    }

    {
        int total_threads = NK * WARP_SIZE;
        int threads_per_block = 256;
        int blocks = (total_threads + threads_per_block - 1) / threads_per_block;
        scatter_tokens_ragged_kernel<<<blocks, threads_per_block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
            flat_indices.data_ptr<int32_t>(),
            expert_counters.data_ptr<int32_t>(),
            reinterpret_cast<__nv_bfloat16*>(act_buffer.data_ptr()),
            cu_seqlens.data_ptr<int32_t>(),
            topk_pos.data_ptr<int32_t>(),
            NK, H, K, expert_start, E_local);
    }

    return {expert_counts, cu_seqlens, topk_pos};
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

template <int K>
__global__ void reduce_weighted_scatter_bf16_ordered_kernel(
    const __nv_bfloat16* __restrict__ expert_output,
    const int32_t* __restrict__ topk_pos,
    const int32_t* __restrict__ topk_indices,
    const float* __restrict__ topk_weights,
    __nv_bfloat16* __restrict__ output,
    int N, int H
) {
    const int token_idx = blockIdx.x;
    const int h_offset = blockIdx.y * BLOCK_H + threadIdx.x;
    if (token_idx >= N || h_offset >= H) return;

    int32_t expert[K];
    int32_t pos[K];
    float weight[K];
    const int topk_base = token_idx * K;
    #pragma unroll
    for (int k = 0; k < K; ++k) {
        expert[k] = topk_indices[topk_base + k];
        pos[k] = topk_pos[topk_base + k];
        weight[k] = topk_weights[topk_base + k];
    }

    // The reference prefill visits experts in ascending expert-id order and
    // rounds both each weighted contribution and each index_add_ update to
    // BF16. Preserve that order while computing all top-K slots in one kernel.
    #pragma unroll
    for (int i = 1; i < K; ++i) {
        int32_t expert_i = expert[i];
        int32_t pos_i = pos[i];
        float weight_i = weight[i];
        int j = i - 1;
        while (j >= 0 && expert[j] > expert_i) {
            expert[j + 1] = expert[j];
            pos[j + 1] = pos[j];
            weight[j + 1] = weight[j];
            --j;
        }
        expert[j + 1] = expert_i;
        pos[j + 1] = pos_i;
        weight[j + 1] = weight_i;
    }

    __nv_bfloat16 acc = __float2bfloat16(0.0f);
    #pragma unroll
    for (int k = 0; k < K; ++k) {
        if (pos[k] >= 0) {
            __nv_bfloat16 value =
                expert_output[(int64_t)pos[k] * H + h_offset];
            __nv_bfloat16 w_bf16 = __float2bfloat16(weight[k]);
            __nv_bfloat16 weighted = __float2bfloat16(
                __bfloat162float(value) * __bfloat162float(w_bf16));
            acc = __float2bfloat16(
                __bfloat162float(acc) + __bfloat162float(weighted));
        }
    }
    output[(int64_t)token_idx * H + h_offset] = acc;
}

torch::Tensor reduce_weighted_scatter_bf16_ordered(
    torch::Tensor expert_output, torch::Tensor topk_pos,
    torch::Tensor topk_indices, torch::Tensor topk_weights,
    int64_t N, int64_t H, int64_t K, torch::Tensor output
) {
    TORCH_CHECK(topk_indices.scalar_type() == torch::kInt32,
                "topk_indices must be int32");
    if (!output.defined() || output.numel() == 0) {
        output = torch::empty(
            {N, H},
            torch::dtype(torch::kBFloat16).device(expert_output.device()));
    }
    dim3 grid(N, (H + BLOCK_H - 1) / BLOCK_H);
    dim3 block(BLOCK_H);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    switch (K) {
        case 2: reduce_weighted_scatter_bf16_ordered_kernel<2><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(expert_output.data_ptr()),
            topk_pos.data_ptr<int32_t>(), topk_indices.data_ptr<int32_t>(),
            topk_weights.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), N, H); break;
        case 4: reduce_weighted_scatter_bf16_ordered_kernel<4><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(expert_output.data_ptr()),
            topk_pos.data_ptr<int32_t>(), topk_indices.data_ptr<int32_t>(),
            topk_weights.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), N, H); break;
        case 8: reduce_weighted_scatter_bf16_ordered_kernel<8><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(expert_output.data_ptr()),
            topk_pos.data_ptr<int32_t>(), topk_indices.data_ptr<int32_t>(),
            topk_weights.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), N, H); break;
        default: TORCH_CHECK(false, "Unsupported K=", K);
    }
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dispatch_scatter_3d", &dispatch_scatter_3d,
          "3D dispatch scatter for strided MoE buffer layout");
    m.def("dispatch_scatter_ragged", &dispatch_scatter_ragged,
          "Compact ragged dispatch scatter (row-aligned per-expert segments, "
          "returns device cu_seqlens)");
    m.def("reduce_weighted_scatter", &reduce_weighted_scatter,
          "Weighted reduce scatter from 3D to flat layout");
    m.def("reduce_weighted_scatter_bf16_ordered",
          &reduce_weighted_scatter_bf16_ordered,
          "BF16 expert-order reduce scatter matching GLM-5 prefill semantics");
}
