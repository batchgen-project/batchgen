/*
 * GLM-5 router GEMM for CUDA graph decode.
 *
 * Computes BF16 hidden states x BF16 router weight^T into FP32 logits with a
 * per-row accumulation order that is independent of the captured bucket M.
 * Grid is (E, ceil(N/ROWS_PER_BLOCK)): each block computes a tile of rows for
 * one expert. Row tiling only partitions independent outputs — the K-order
 * FMA chain per output element is unchanged, so results are bit-identical to
 * the previous serial-row kernel while wall time stops scaling O(N).
 * Rank-major graph padding is masked by device-side rank_token_counts, so graph
 * replay can keep a fixed launch while changing valid rows without host sync.
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <c10/cuda/CUDAStream.h>
#include <climits>

namespace {

constexpr int ROUTER_THREADS = 256;
constexpr int ROUTER_WARP_SIZE = 32;
constexpr int ROUTER_ROWS_PER_BLOCK = 4;

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = ROUTER_WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void glm5_router_gemm_kernel(
    const __nv_bfloat16* __restrict__ hidden_states,  // [N, H]
    const __nv_bfloat16* __restrict__ router_weight,  // [E, H]
    const int64_t* __restrict__ rank_token_counts,    // [world_size] or nullptr
    float* __restrict__ output,                       // [N, E]
    int N,
    int H,
    int E,
    int bucket_size,
    int world_size
) {
    const int expert = blockIdx.x;
    if (expert >= E) {
        return;
    }
    const int row_begin = blockIdx.y * ROUTER_ROWS_PER_BLOCK;
    const int row_end = min(row_begin + ROUTER_ROWS_PER_BLOCK, N);

    const int tid = threadIdx.x;
    const int lane = tid % ROUTER_WARP_SIZE;
    const int warp = tid / ROUTER_WARP_SIZE;
    constexpr int NUM_WARPS = ROUTER_THREADS / ROUTER_WARP_SIZE;
    __shared__ float warp_sums[NUM_WARPS];

    const __nv_bfloat16* w = router_weight + expert * H;

    for (int row = row_begin; row < row_end; ++row) {
        bool valid = true;
        if (rank_token_counts != nullptr) {
            valid = false;
            if (bucket_size > 0) {
                const int rank = row / bucket_size;
                const int local_pos = row - rank * bucket_size;
                if (rank >= 0 && rank < world_size) {
                    valid = local_pos < static_cast<int>(rank_token_counts[rank]);
                }
            }
        }

        if (!valid) {
            if (tid == 0) {
                output[row * E + expert] = 0.0f;
            }
            continue;
        }

        const __nv_bfloat16* x = hidden_states + row * H;
        float acc = 0.0f;

        for (int k = tid; k < H; k += ROUTER_THREADS) {
            acc = fmaf(__bfloat162float(x[k]), __bfloat162float(w[k]), acc);
        }

        acc = warp_reduce_sum(acc);
        if (lane == 0) {
            warp_sums[warp] = acc;
        }
        __syncthreads();

        if (warp == 0) {
            float sum = (lane < NUM_WARPS) ? warp_sums[lane] : 0.0f;
            sum = warp_reduce_sum(sum);
            if (lane == 0) {
                output[row * E + expert] = sum;
            }
        }
        __syncthreads();
    }
}

}  // namespace

torch::Tensor glm5_router_gemm_cuda(
    torch::Tensor hidden_states,
    torch::Tensor router_weight,
    torch::Tensor rank_token_counts,
    torch::Tensor output,
    int64_t bucket_size,
    int64_t world_size
) {
    TORCH_CHECK(hidden_states.is_cuda(), "hidden_states must be a CUDA tensor");
    TORCH_CHECK(router_weight.is_cuda(), "router_weight must be a CUDA tensor");
    TORCH_CHECK(hidden_states.dtype() == torch::kBFloat16, "hidden_states must be BF16");
    TORCH_CHECK(router_weight.dtype() == torch::kBFloat16, "router_weight must be BF16");
    TORCH_CHECK(hidden_states.dim() == 2, "hidden_states must be rank-2 [N, H]");
    TORCH_CHECK(router_weight.dim() == 2, "router_weight must be rank-2 [E, H]");
    TORCH_CHECK(hidden_states.is_contiguous(), "hidden_states must be contiguous");
    TORCH_CHECK(router_weight.is_contiguous(), "router_weight must be contiguous");

    const int64_t N64 = hidden_states.size(0);
    const int64_t H64 = hidden_states.size(1);
    const int64_t E64 = router_weight.size(0);
    TORCH_CHECK(router_weight.size(1) == H64,
                "router_weight hidden dim mismatch: got ", router_weight.size(1),
                ", expected ", H64);
    TORCH_CHECK(N64 <= static_cast<int64_t>(INT_MAX), "N too large for router kernel");
    TORCH_CHECK(H64 <= static_cast<int64_t>(INT_MAX), "H too large for router kernel");
    TORCH_CHECK(E64 <= static_cast<int64_t>(INT_MAX), "E too large for router kernel");

    if (!output.defined() || output.numel() == 0) {
        output = torch::empty({N64, E64}, torch::dtype(torch::kFloat32).device(hidden_states.device()));
    }
    TORCH_CHECK(output.is_cuda(), "output must be a CUDA tensor");
    TORCH_CHECK(output.dtype() == torch::kFloat32, "output must be FP32");
    TORCH_CHECK(output.dim() == 2, "output must be rank-2 [N, E]");
    TORCH_CHECK(output.size(0) == N64 && output.size(1) == E64,
                "output shape mismatch: got [", output.size(0), ", ", output.size(1),
                "], expected [", N64, ", ", E64, "]");
    TORCH_CHECK(output.is_contiguous(), "output must be contiguous");

    const int N = static_cast<int>(N64);
    const int H = static_cast<int>(H64);
    const int E = static_cast<int>(E64);

    const int64_t* rank_counts_ptr = nullptr;
    if (rank_token_counts.defined() && rank_token_counts.numel() > 0) {
        TORCH_CHECK(rank_token_counts.is_cuda(), "rank_token_counts must be CUDA when provided");
        TORCH_CHECK(rank_token_counts.dtype() == torch::kInt64, "rank_token_counts must be int64");
        TORCH_CHECK(rank_token_counts.is_contiguous(), "rank_token_counts must be contiguous");
        TORCH_CHECK(world_size > 0, "world_size must be positive when rank_token_counts is provided");
        TORCH_CHECK(rank_token_counts.numel() >= world_size,
                    "rank_token_counts length ", rank_token_counts.numel(),
                    " is smaller than world_size ", world_size);
        TORCH_CHECK(bucket_size > 0, "bucket_size must be positive when rank_token_counts is provided");
        rank_counts_ptr = rank_token_counts.data_ptr<int64_t>();
    }

    if (N == 0 || E == 0) {
        return output;
    }

    const dim3 grid(E, (N + ROUTER_ROWS_PER_BLOCK - 1) / ROUTER_ROWS_PER_BLOCK);
    const dim3 block(ROUTER_THREADS);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    glm5_router_gemm_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(hidden_states.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(router_weight.data_ptr()),
        rank_counts_ptr,
        output.data_ptr<float>(),
        N,
        H,
        E,
        static_cast<int>(bucket_size),
        static_cast<int>(world_size));
    return output;
}
