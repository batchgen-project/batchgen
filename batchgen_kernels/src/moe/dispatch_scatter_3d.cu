#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

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

// K3 keeps the routed-expert reduction in FP32 until after the EP
// reduce-scatter.  The existing BF16-output kernel cannot be used here: its
// downcast would happen before ranks combine their disjoint expert subsets.
// Keep the K=16 path separate so the decode graph has one fixed-shape launch
// and writes directly into its preallocated FP32 reduction buffer.
__global__ void reduce_weighted_scatter_fp32_k16_kernel(
    const __nv_bfloat16* __restrict__ expert_output,
    const int32_t* __restrict__ topk_pos,
    const float* __restrict__ topk_weights,
    float* __restrict__ output,
    int N, int H
) {
    const int token_idx = blockIdx.x;
    const int h_offset = blockIdx.y * BLOCK_H + threadIdx.x;

    __shared__ int32_t shared_pos[16];
    __shared__ float shared_weights[16];
    if (threadIdx.x < 16) {
        const int topk_base = token_idx * 16;
        shared_pos[threadIdx.x] = topk_pos[topk_base + threadIdx.x];
        shared_weights[threadIdx.x] = topk_weights[topk_base + threadIdx.x];
    }
    __syncthreads();

    if (token_idx >= N || h_offset >= H) return;

    float acc = 0.0f;
#pragma unroll
    for (int k = 0; k < 16; k++) {
        const int32_t pos = shared_pos[k];
        if (pos >= 0) {
            const float value = __bfloat162float(
                expert_output[(int64_t)pos * H + h_offset]);
            acc += value * shared_weights[k];
        }
    }
    output[(int64_t)token_idx * H + h_offset] = acc;
}

// Ordered BF16 combine.  The result for each token is a function only of the
// token's routed (expert id, row, weight) set on this rank, independent of
// slot order when expert ids are unique:
//   acc = +0
//   for slots in stable ascending expert-id order, skipping pos < 0:
//     acc = bf16(fadd_rn(acc, bf16(fmul_rn(x[pos], bf16(w)))))
// Every product and every partial sum is rounded to BF16, so the sequence can
// be replayed bitwise by a reference that follows the same order.
//
// One 256-thread block per token.  Thread 0 stable-sorts the token's K slots
// into shared memory; after the sync each thread owns the 8-column groups
// v = tid, tid + 256, ... and moves them with 16-byte loads/stores.
constexpr int kOrderedReduceThreads = 256;
constexpr int kOrderedReduceVec = 8;

template <int K>
__global__ void __launch_bounds__(kOrderedReduceThreads)
reduce_weighted_scatter_bf16_ordered_kernel(
    const __nv_bfloat16* __restrict__ expert_output,
    const int32_t* __restrict__ topk_pos,
    const int32_t* __restrict__ topk_indices,
    const float* __restrict__ topk_weights,
    __nv_bfloat16* __restrict__ output,
    int H
) {
    const int token_idx = blockIdx.x;
    const int tid = threadIdx.x;

    __shared__ int32_t shared_pos[K];
    __shared__ float shared_weights[K];
    if (tid == 0) {
        const int64_t base = (int64_t)token_idx * K;
        int32_t eid[K];
        int32_t pos[K];
        float w[K];
        // Insertion sort; strict '>' keeps equal expert ids in slot order.
        for (int k = 0; k < K; k++) {
            const int32_t e = topk_indices[base + k];
            int j = k;
            while (j > 0 && eid[j - 1] > e) {
                eid[j] = eid[j - 1];
                pos[j] = pos[j - 1];
                w[j] = w[j - 1];
                j--;
            }
            eid[j] = e;
            pos[j] = topk_pos[base + k];
            w[j] = __bfloat162float(__float2bfloat16(topk_weights[base + k]));
        }
        for (int k = 0; k < K; k++) {
            shared_pos[k] = pos[k];
            shared_weights[k] = w[k];
        }
    }
    __syncthreads();

    const int vec_count = H / kOrderedReduceVec;
    const int64_t out_base = (int64_t)token_idx * H;
    for (int v = tid; v < vec_count; v += blockDim.x) {
        const int64_t col = (int64_t)v * kOrderedReduceVec;
        __nv_bfloat16 acc[kOrderedReduceVec];
#pragma unroll
        for (int i = 0; i < kOrderedReduceVec; i++) {
            acc[i] = __float2bfloat16(0.0f);
        }

#pragma unroll
        for (int k = 0; k < K; k++) {
            const int32_t pos = shared_pos[k];
            if (pos >= 0) {
                const uint4 raw = *reinterpret_cast<const uint4*>(
                    expert_output + (int64_t)pos * H + col);
                const __nv_bfloat16* x = reinterpret_cast<const __nv_bfloat16*>(&raw);
                const float w = shared_weights[k];
#pragma unroll
                for (int i = 0; i < kOrderedReduceVec; i++) {
                    const __nv_bfloat16 prod = __float2bfloat16(
                        __fmul_rn(__bfloat162float(x[i]), w));
                    acc[i] = __float2bfloat16(
                        __fadd_rn(__bfloat162float(acc[i]), __bfloat162float(prod)));
                }
            }
        }
        uint4 out_raw;
        __nv_bfloat16* out_values = reinterpret_cast<__nv_bfloat16*>(&out_raw);
#pragma unroll
        for (int i = 0; i < kOrderedReduceVec; i++) out_values[i] = acc[i];
        *reinterpret_cast<uint4*>(output + out_base + col) = out_raw;
    }
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

torch::Tensor reduce_weighted_scatter_fp32(
    torch::Tensor expert_output, torch::Tensor topk_pos,
    torch::Tensor topk_weights, int64_t N, int64_t H, int64_t K,
    torch::Tensor output
) {
    TORCH_CHECK(N > 0 && H > 0, "N and H must be positive");
    TORCH_CHECK(K == 16, "K3 FP32 weighted combine requires K=16, got ", K);
    TORCH_CHECK(expert_output.is_cuda() && topk_pos.is_cuda() &&
                topk_weights.is_cuda() && output.is_cuda(),
                "K3 FP32 weighted combine requires CUDA tensors");
    TORCH_CHECK(expert_output.scalar_type() == at::kBFloat16,
                "expert_output must be BF16");
    TORCH_CHECK(topk_pos.scalar_type() == at::kInt,
                "topk_pos must be int32");
    TORCH_CHECK(topk_weights.scalar_type() == at::kFloat,
                "topk_weights must be float32");
    TORCH_CHECK(output.scalar_type() == at::kFloat,
                "output must be float32");
    TORCH_CHECK(expert_output.is_contiguous() && topk_pos.is_contiguous() &&
                topk_weights.is_contiguous() && output.is_contiguous(),
                "K3 FP32 weighted combine requires contiguous tensors");
    TORCH_CHECK(topk_pos.numel() >= N * K,
                "topk_pos is smaller than N*K");
    TORCH_CHECK(topk_weights.numel() >= N * K,
                "topk_weights is smaller than N*K");
    TORCH_CHECK(output.dim() == 2 && output.size(0) >= N &&
                output.size(1) >= H, "output must have shape [N, H]");
    TORCH_CHECK(expert_output.dim() == 2 && expert_output.size(1) == H,
                "expert_output must have shape [rows, H]");

    dim3 grid(static_cast<unsigned int>(N),
              static_cast<unsigned int>((H + BLOCK_H - 1) / BLOCK_H));
    dim3 block(BLOCK_H);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    reduce_weighted_scatter_fp32_k16_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(expert_output.data_ptr()),
        topk_pos.data_ptr<int32_t>(), topk_weights.data_ptr<float>(),
        output.data_ptr<float>(), static_cast<int>(N), static_cast<int>(H));
    return output;
}

torch::Tensor reduce_weighted_scatter_bf16_ordered(
    torch::Tensor expert_output, torch::Tensor topk_pos,
    torch::Tensor topk_indices, torch::Tensor topk_weights,
    int64_t N, int64_t H, int64_t K, torch::Tensor output
) {
    TORCH_CHECK(N > 0 && H > 0, "N and H must be positive");
    TORCH_CHECK(K == 2 || K == 4 || K == 8,
                "ordered BF16 combine supports K=2,4,8, got ", K);
    TORCH_CHECK(H % 8 == 0, "ordered BF16 combine requires H % 8 == 0, got H=", H);
    TORCH_CHECK(expert_output.is_cuda(), "expert_output must be a CUDA tensor");
    const auto device = expert_output.device();
    TORCH_CHECK(topk_pos.device() == device && topk_indices.device() == device &&
                topk_weights.device() == device && output.device() == device,
                "ordered BF16 combine requires all tensors on ", device);
    TORCH_CHECK(expert_output.scalar_type() == at::kBFloat16,
                "expert_output must be BF16");
    TORCH_CHECK(topk_pos.scalar_type() == at::kInt, "topk_pos must be int32");
    TORCH_CHECK(topk_indices.scalar_type() == at::kInt,
                "topk_indices must be int32");
    TORCH_CHECK(topk_weights.scalar_type() == at::kFloat,
                "topk_weights must be float32");
    TORCH_CHECK(output.scalar_type() == at::kBFloat16, "output must be BF16");
    TORCH_CHECK(expert_output.is_contiguous() && topk_pos.is_contiguous() &&
                topk_indices.is_contiguous() && topk_weights.is_contiguous() &&
                output.is_contiguous(),
                "ordered BF16 combine requires contiguous tensors");
    TORCH_CHECK(expert_output.dim() == 2 && expert_output.size(1) == H,
                "expert_output must have shape [rows, H]");
    // dispatch_scatter_ragged returns topk_pos as a flat [N*K] buffer, while
    // focused callers may use [N, K]. Both are the same contiguous ABI.
    TORCH_CHECK(topk_pos.numel() == N * K,
                "topk_pos must contain N*K elements");
    TORCH_CHECK(topk_indices.dim() == 2 && topk_indices.size(0) == N &&
                topk_indices.size(1) == K, "topk_indices must have shape [N, K]");
    TORCH_CHECK(topk_weights.dim() == 2 && topk_weights.size(0) == N &&
                topk_weights.size(1) == K, "topk_weights must have shape [N, K]");
    TORCH_CHECK(output.dim() == 2 && output.size(0) == N && output.size(1) == H,
                "output must have shape [N, H]");
    // The kernel moves 8 BF16 columns per 16-byte load/store.
    TORCH_CHECK(reinterpret_cast<uintptr_t>(expert_output.data_ptr()) % 16 == 0,
                "expert_output data must be 16-byte aligned");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(output.data_ptr()) % 16 == 0,
                "output data must be 16-byte aligned");

    const c10::cuda::CUDAGuard guard(device);
    dim3 grid(static_cast<unsigned int>(N));
    dim3 block(kOrderedReduceThreads);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    switch (K) {
        case 2: reduce_weighted_scatter_bf16_ordered_kernel<2><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(expert_output.data_ptr()),
            topk_pos.data_ptr<int32_t>(), topk_indices.data_ptr<int32_t>(),
            topk_weights.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
            static_cast<int>(H)); break;
        case 4: reduce_weighted_scatter_bf16_ordered_kernel<4><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(expert_output.data_ptr()),
            topk_pos.data_ptr<int32_t>(), topk_indices.data_ptr<int32_t>(),
            topk_weights.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
            static_cast<int>(H)); break;
        case 8: reduce_weighted_scatter_bf16_ordered_kernel<8><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(expert_output.data_ptr()),
            topk_pos.data_ptr<int32_t>(), topk_indices.data_ptr<int32_t>(),
            topk_weights.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
            static_cast<int>(H)); break;
    }
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dispatch_scatter_3d", &dispatch_scatter_3d,
          "3D dispatch scatter for strided MoE buffer layout");
    m.def("reduce_weighted_scatter", &reduce_weighted_scatter,
          "Weighted reduce scatter from 3D to flat layout");
    m.def("reduce_weighted_scatter_fp32", &reduce_weighted_scatter_fp32,
          "K3 K=16 weighted reduction with FP32 output");
    m.def("reduce_weighted_scatter_bf16_ordered",
          &reduce_weighted_scatter_bf16_ordered,
          "Expert-id-ordered weighted reduction with BF16 rounding per step");
}
