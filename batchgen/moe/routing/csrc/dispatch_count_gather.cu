/*
 * GPT-OSS-120B Dispatch Kernel: Count + Prefix Sum + Gather (CUDA).
 *
 * 3 sub-kernels following hpc-ops count_and_gather architecture:
 *   A. count_tokens_kernel:    shared-memory atomics for counting + init topk_pos=-1
 *   B. prefix_sum_kernel:      sequential scan (E_local <= 128, negligible)
 *   C. gather_tokens_kernel:   warp-cooperative vectorized copy
 *
 * Key advantages over Triton version:
 *   - Shared memory atomics (vs Triton global atomics)
 *   - Warp-cooperative float4 copy (32 threads copy one token row)
 *   - __shfl_sync for position broadcast
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <c10/cuda/CUDAStream.h>

#define WARP_SIZE 32


// ──────────────────────────────────────────────────────────────────────────────
// Sub-kernel A: Count tokens per local expert + init topk_pos = -1
// ──────────────────────────────────────────────────────────────────────────────

__global__ void count_tokens_kernel(
    const int32_t* __restrict__ topk_indices,  // [NK] flat
    int32_t* __restrict__ expert_counts,       // [E_local] output
    int32_t* __restrict__ topk_pos,            // [NK] output
    int NK,
    int expert_start,
    int E_local
) {
    // Shared memory for counts (much faster than global atomics)
    extern __shared__ int32_t s_counts[];

    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    // Zero shared counts
    for (int i = tid; i < E_local; i += stride) {
        s_counts[i] = 0;
    }
    __syncthreads();

    // Count + init topk_pos
    for (int i = tid; i < NK; i += stride) {
        topk_pos[i] = -1;  // sentinel for non-local

        int eid = topk_indices[i];
        int local_id = eid - expert_start;
        if (local_id >= 0 && local_id < E_local) {
            atomicAdd(&s_counts[local_id], 1);
        }
    }
    __syncthreads();

    // Write back to global memory
    for (int i = tid; i < E_local; i += stride) {
        expert_counts[i] = s_counts[i];
    }
}


// ──────────────────────────────────────────────────────────────────────────────
// Sub-kernel B: Exclusive prefix sum (sequential, E_local is small)
// ──────────────────────────────────────────────────────────────────────────────

__global__ void prefix_sum_kernel(
    const int32_t* __restrict__ expert_counts,
    int32_t* __restrict__ expert_offsets,
    int E_local
) {
    if (threadIdx.x > 0) return;

    int32_t cumsum = 0;
    expert_offsets[0] = 0;
    for (int e = 0; e < E_local; e++) {
        cumsum += expert_counts[e];
        expert_offsets[e + 1] = cumsum;
    }
}


// ──────────────────────────────────────────────────────────────────────────────
// Sub-kernel C: Gather tokens to expert-sorted flat layout
// ──────────────────────────────────────────────────────────────────────────────

__global__ void gather_tokens_kernel(
    const __nv_bfloat16* __restrict__ x,           // [N, H]
    const int32_t* __restrict__ topk_indices,       // [NK] flat
    const int32_t* __restrict__ expert_offsets,     // [E_local + 1]
    int32_t* __restrict__ expert_counters,          // [E_local] scratch (zero-initialized)
    __nv_bfloat16* __restrict__ dispatched_x,       // [max_disp, H]
    int32_t* __restrict__ topk_pos,                 // [NK]
    int NK, int H, int K,
    int expert_start, int E_local
) {
    // One warp per topk assignment
    const int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int warp_id = global_tid / WARP_SIZE;
    const int lane_id = global_tid % WARP_SIZE;

    if (warp_id >= NK) return;

    const int itopk = warp_id;
    const int token_id = itopk / K;
    const int eid = topk_indices[itopk];
    const int local_expert = eid - expert_start;

    // Skip non-local experts (all lanes return together)
    if (local_expert < 0 || local_expert >= E_local) return;

    // Lane 0: atomically claim a position within expert's segment
    int write_pos;
    if (lane_id == 0) {
        int relative_pos = atomicAdd(&expert_counters[local_expert], 1);
        write_pos = expert_offsets[local_expert] + relative_pos;
        topk_pos[itopk] = write_pos;
    }
    // Broadcast write_pos to all lanes in the warp
    write_pos = __shfl_sync(0xffffffff, write_pos, 0);

    // Warp-cooperative vectorized copy: x[token_id, :] -> dispatched_x[write_pos, :]
    // Use float4 (128-bit) loads: 8 BF16 values per float4
    const int vec_size = 8;  // BF16 values per float4
    const int vec_count = H / vec_size;
    const int remainder = H % vec_size;

    const float4* src = reinterpret_cast<const float4*>(x + (int64_t)token_id * H);
    float4* dst = reinterpret_cast<float4*>(dispatched_x + (int64_t)write_pos * H);

    // Each of 32 lanes copies different float4 chunks
    for (int v = lane_id; v < vec_count; v += WARP_SIZE) {
        dst[v] = src[v];
    }

    // Handle remainder (H=2880: 2880/8=360, no remainder)
    if (remainder > 0 && lane_id == 0) {
        const __nv_bfloat16* src_r = x + (int64_t)token_id * H + vec_count * vec_size;
        __nv_bfloat16* dst_r = dispatched_x + (int64_t)write_pos * H + vec_count * vec_size;
        for (int i = 0; i < remainder; i++) {
            dst_r[i] = src_r[i];
        }
    }
}


// ──────────────────────────────────────────────────────────────────────────────
// Python wrapper: launches all 3 sub-kernels
// ──────────────────────────────────────────────────────────────────────────────

std::vector<torch::Tensor> dispatch_count_gather_cuda(
    torch::Tensor x,
    torch::Tensor topk_indices,
    int64_t expert_start,
    int64_t num_local_experts,
    torch::Tensor expert_counts,
    torch::Tensor expert_offsets,
    torch::Tensor expert_counters,
    torch::Tensor dispatched_x,
    torch::Tensor topk_pos
) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kBFloat16, "x must be BF16");

    const int N = topk_indices.size(0);
    const int K = topk_indices.size(1);
    const int H = x.size(1);
    const int NK = N * K;
    const int E_local = num_local_experts;
    auto device = x.device();

    // Allocate outputs if not pre-allocated
    if (!expert_counts.defined() || expert_counts.numel() == 0) {
        expert_counts = torch::zeros({E_local}, torch::dtype(torch::kInt32).device(device));
    } else {
        expert_counts.zero_();
    }

    if (!expert_offsets.defined() || expert_offsets.numel() == 0) {
        expert_offsets = torch::empty({E_local + 1}, torch::dtype(torch::kInt32).device(device));
    }

    if (!expert_counters.defined() || expert_counters.numel() == 0) {
        expert_counters = torch::zeros({E_local}, torch::dtype(torch::kInt32).device(device));
    } else {
        expert_counters.zero_();
    }

    if (!topk_pos.defined() || topk_pos.numel() == 0) {
        topk_pos = torch::full({NK}, -1, torch::dtype(torch::kInt32).device(device));
    }

    if (!dispatched_x.defined() || dispatched_x.numel() == 0) {
        dispatched_x = torch::empty({NK, H}, torch::dtype(torch::kBFloat16).device(device));
    }

    // Flat view of topk_indices
    auto flat_indices = topk_indices.reshape({-1}).contiguous();

    // All kernels launch on current CUDA stream (required for CUDA graph capture)
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // ── Sub-kernel A: Count ──
    {
        int threads = 256;
        int blocks = 1;  // For decode NK <= 256, single block suffices
        int smem_bytes = E_local * sizeof(int32_t);
        count_tokens_kernel<<<blocks, threads, smem_bytes, stream>>>(
            flat_indices.data_ptr<int32_t>(),
            expert_counts.data_ptr<int32_t>(),
            topk_pos.data_ptr<int32_t>(),
            NK, expert_start, E_local
        );
    }

    // ── Sub-kernel B: Prefix sum ──
    {
        prefix_sum_kernel<<<1, 1, 0, stream>>>(
            expert_counts.data_ptr<int32_t>(),
            expert_offsets.data_ptr<int32_t>(),
            E_local
        );
    }

    // ── Sub-kernel C: Gather ──
    {
        // One warp per topk assignment
        int total_threads = NK * WARP_SIZE;
        int threads_per_block = 256;
        int blocks = (total_threads + threads_per_block - 1) / threads_per_block;

        gather_tokens_kernel<<<blocks, threads_per_block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
            flat_indices.data_ptr<int32_t>(),
            expert_offsets.data_ptr<int32_t>(),
            expert_counters.data_ptr<int32_t>(),
            reinterpret_cast<__nv_bfloat16*>(dispatched_x.data_ptr()),
            topk_pos.data_ptr<int32_t>(),
            NK, H, K, expert_start, E_local
        );
    }

    return {dispatched_x, expert_counts, expert_offsets, topk_pos};
}
