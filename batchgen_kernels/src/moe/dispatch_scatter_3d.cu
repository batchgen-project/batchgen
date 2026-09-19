#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <algorithm>

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
// dispatch_scatter_ragged: Route tokens from flat [G, H] into compact ragged
// rows. Local expert e owns rows [cu_seqlens[e], cu_seqlens[e + 1]) with
// cu_seqlens[e + 1] - cu_seqlens[e] = align64(expert_counts[e]). Counts and
// offsets stay on device; launch geometry depends only on host shapes
// (N, K, E_local), so the call is CUDA-graph capturable.
// ============================================================================
#define RAGGED_ROW_ALIGN 64
static constexpr int64_t kRaggedInt32Max = 2147483647;

__global__ void count_tokens_ragged_kernel(
    const int32_t* __restrict__ topk_indices,
    int32_t* __restrict__ expert_counts,
    int32_t* __restrict__ expert_counters,
    int32_t* __restrict__ cu_seqlens,
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
        const int64_t local_id = (int64_t)topk_indices[i] - expert_start;
        if (local_id >= 0 && local_id < E_local)
            atomicAdd(&s_counts[local_id], 1);
    }
    __syncthreads();

    for (int i = tid; i < E_local; i += stride) {
        expert_counts[i] = s_counts[i];
        expert_counters[i] = 0;
    }

    // Serial 64-aligned prefix sum on one thread (E_local is small). The host
    // capacity check bounds the total below act_buffer rows <= INT32_MAX.
    if (tid == 0) {
        int64_t offset = 0;
        for (int e = 0; e < E_local; e++) {
            cu_seqlens[e] = (int32_t)offset;
            offset += ((int64_t)s_counts[e] + RAGGED_ROW_ALIGN - 1) &
                      ~(int64_t)(RAGGED_ROW_ALIGN - 1);
        }
        cu_seqlens[E_local] = (int32_t)offset;
    }
}

__global__ void scatter_tokens_ragged_kernel(
    const __nv_bfloat16* __restrict__ x,
    const int32_t* __restrict__ topk_indices,
    const int32_t* __restrict__ cu_seqlens,
    int32_t* __restrict__ expert_counters,
    __nv_bfloat16* __restrict__ act_buffer,
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
    const int64_t local_expert = (int64_t)topk_indices[itopk] - expert_start;

    if (local_expert < 0 || local_expert >= E_local) return;

    int write_pos;
    if (lane_id == 0) {
        const int relative_pos = atomicAdd(&expert_counters[local_expert], 1);
        write_pos = (int)((int64_t)cu_seqlens[local_expert] + relative_pos);
        topk_pos[itopk] = write_pos;
    }
    write_pos = __shfl_sync(0xffffffff, write_pos, 0);

    if ((H & 7) == 0) {
        const int vec_count = H / 8;
        const float4* src = reinterpret_cast<const float4*>(x + (int64_t)token_id * H);
        float4* dst = reinterpret_cast<float4*>(act_buffer + (int64_t)write_pos * H);
        for (int v = lane_id; v < vec_count; v += WARP_SIZE)
            dst[v] = src[v];
    } else {
        // A row stride that is not a multiple of 16 bytes makes float4 loads
        // misaligned after the first row. Preserve arbitrary hidden sizes with
        // a scalar tail path instead of vectorizing from an unaligned address.
        const __nv_bfloat16* src = x + (int64_t)token_id * H;
        __nv_bfloat16* dst = act_buffer + (int64_t)write_pos * H;
        for (int h = lane_id; h < H; h += WARP_SIZE)
            dst[h] = src[h];
    }
}

// Worst-case rows needed by dispatch_scatter_ragged: every nonzero local
// expert may add up to 63 pad rows, and the aligned total is a multiple of 64.
static int64_t ragged_capacity_rows(
    int64_t num_tokens, int64_t top_k, int64_t num_local_experts
) {
    TORCH_CHECK(num_tokens >= 0 && num_tokens <= kRaggedInt32Max,
                "num_tokens out of range: ", num_tokens);
    TORCH_CHECK(top_k > 0 && top_k <= kRaggedInt32Max, "top_k out of range: ", top_k);
    TORCH_CHECK(num_local_experts > 0 && num_local_experts <= kRaggedInt32Max,
                "num_local_experts out of range: ", num_local_experts);
    const int64_t nk = num_tokens * top_k;
    const int64_t rows = nk + (RAGGED_ROW_ALIGN - 1) * std::min(num_local_experts, nk);
    return rows / RAGGED_ROW_ALIGN * RAGGED_ROW_ALIGN;
}

static void check_ragged_cuda_tensor(
    const torch::Tensor& t, const torch::Device& device, const char* name
) {
    TORCH_CHECK(t.defined() && t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.device() == device, name, " must be on ", device, ", got ", t.device());
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

std::vector<torch::Tensor> dispatch_scatter_ragged(
    torch::Tensor x,
    torch::Tensor topk_indices,
    torch::Tensor act_buffer,
    int64_t expert_start,
    int64_t num_local_experts,
    torch::Tensor expert_counts,
    torch::Tensor expert_counters,
    torch::Tensor topk_pos,
    torch::Tensor cu_seqlens
) {
    TORCH_CHECK(x.defined() && x.is_cuda(), "x must be a CUDA tensor");
    const auto device = x.device();
    check_ragged_cuda_tensor(x, device, "x");
    check_ragged_cuda_tensor(topk_indices, device, "topk_indices");
    check_ragged_cuda_tensor(act_buffer, device, "act_buffer");
    check_ragged_cuda_tensor(expert_counts, device, "expert_counts");
    check_ragged_cuda_tensor(expert_counters, device, "expert_counters");
    check_ragged_cuda_tensor(topk_pos, device, "topk_pos");
    check_ragged_cuda_tensor(cu_seqlens, device, "cu_seqlens");

    TORCH_CHECK(x.dim() == 2 && x.scalar_type() == at::kBFloat16,
                "x must be BF16 [N, H]");
    TORCH_CHECK(topk_indices.dim() == 2 && topk_indices.scalar_type() == at::kInt &&
                topk_indices.size(0) == x.size(0),
                "topk_indices must be int32 [N, K]");
    TORCH_CHECK(act_buffer.dim() == 2 && act_buffer.scalar_type() == at::kBFloat16 &&
                act_buffer.size(1) == x.size(1),
                "act_buffer must be BF16 [rows, H]");

    const int64_t N = x.size(0);
    const int64_t H = x.size(1);
    const int64_t K = topk_indices.size(1);
    const int64_t E = num_local_experts;
    TORCH_CHECK(H > 0 && H <= kRaggedInt32Max, "H out of range: ", H);
    TORCH_CHECK(expert_start >= 0 && expert_start + E <= kRaggedInt32Max,
                "expert_start out of range: ", expert_start);

    const int64_t capacity = ragged_capacity_rows(N, K, E);
    const int64_t NK = N * K;
    TORCH_CHECK(NK * WARP_SIZE + 256 <= kRaggedInt32Max,
                "N*K too large for scatter launch: ", NK);
    TORCH_CHECK(act_buffer.size(0) >= capacity && act_buffer.size(0) <= kRaggedInt32Max,
                "act_buffer rows (", act_buffer.size(0), ") must be in [", capacity,
                ", INT32_MAX]");

    TORCH_CHECK(expert_counts.dim() == 1 && expert_counts.scalar_type() == at::kInt &&
                expert_counts.size(0) == E, "expert_counts must be int32 [E_local]");
    TORCH_CHECK(expert_counters.dim() == 1 && expert_counters.scalar_type() == at::kInt &&
                expert_counters.size(0) == E, "expert_counters must be int32 [E_local]");
    TORCH_CHECK(cu_seqlens.dim() == 1 && cu_seqlens.scalar_type() == at::kInt &&
                cu_seqlens.size(0) == E + 1, "cu_seqlens must be int32 [E_local + 1]");
    TORCH_CHECK(topk_pos.dim() == 1 && topk_pos.scalar_type() == at::kInt &&
                topk_pos.size(0) == NK, "topk_pos must be int32 [N * K]");

    const c10::cuda::CUDAGuard device_guard(device);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    {
        int threads = 256;
        int blocks = 1;
        int smem_bytes = static_cast<int>(E * sizeof(int32_t));
        count_tokens_ragged_kernel<<<blocks, threads, smem_bytes, stream>>>(
            topk_indices.data_ptr<int32_t>(),
            expert_counts.data_ptr<int32_t>(),
            expert_counters.data_ptr<int32_t>(),
            cu_seqlens.data_ptr<int32_t>(),
            topk_pos.data_ptr<int32_t>(),
            static_cast<int>(NK), static_cast<int>(expert_start), static_cast<int>(E));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    if (NK > 0) {
        const int64_t total_threads = NK * WARP_SIZE;
        int threads_per_block = 256;
        int blocks = static_cast<int>((total_threads + threads_per_block - 1) / threads_per_block);
        scatter_tokens_ragged_kernel<<<blocks, threads_per_block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
            topk_indices.data_ptr<int32_t>(),
            cu_seqlens.data_ptr<int32_t>(),
            expert_counters.data_ptr<int32_t>(),
            reinterpret_cast<__nv_bfloat16*>(act_buffer.data_ptr()),
            topk_pos.data_ptr<int32_t>(),
            static_cast<int>(NK), static_cast<int>(H), static_cast<int>(K),
            static_cast<int>(expert_start), static_cast<int>(E));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dispatch_scatter_3d", &dispatch_scatter_3d,
          "3D dispatch scatter for strided MoE buffer layout");
    m.def("dispatch_scatter_ragged", &dispatch_scatter_ragged,
          "Compact ragged dispatch scatter with 64-aligned device cu_seqlens");
    m.def("reduce_weighted_scatter", &reduce_weighted_scatter,
          "Weighted reduce scatter from 3D to flat layout");
    m.def("reduce_weighted_scatter_fp32", &reduce_weighted_scatter_fp32,
          "K3 K=16 weighted reduction with FP32 output");
}
