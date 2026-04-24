/******************************************************************************
 * Fused interleaved RoPE + Hadamard transform kernel.
 * Specialized for dim=128, bf16 (GLM-5 indexer K path).
 *
 * Based on Dao-AILab fast_hadamard_transform.
 * For dim=128 bf16: 16 threads, 8 elements/thread, 1 chunk.
 * Threads 0-7 hold first 64 dims (RoPE applied), threads 8-15 hold rest.
 * Interleaved RoPE pairs (0,1),(2,3),... are within each thread's registers.
 ******************************************************************************/

#include <c10/util/BFloat16.h>
#include <c10/cuda/CUDAException.h>
#include <ATen/cuda/CUDAContext.h>

#include "fast_hadamard_transform_common.h"

// dim=128, bf16: kNThreads=16, kNElts=8, kNChunks=1
// log2(8)=3 thread-level stages, log2(16)=4 warp-level stages = 7 total = log2(128)
// kNWarps=1 (16 < 32), no smem exchange needed

struct FusedRopeHadamardParams {
    using index_t = int64_t;
    void *__restrict__ x_ptr;       // [batch, 128] input (after LayerNorm), bf16
    void *__restrict__ out_ptr;     // [batch, 128] output, bf16
    const float *__restrict__ cos_ptr; // [max_seq, 64] cos cache (first 32 used)
    const float *__restrict__ sin_ptr; // [max_seq, 64] sin cache (first 32 used)
    const int64_t *__restrict__ pos_ptr; // [batch] position indices
    int batch;
    int cos_stride;     // stride of cos dim-0 (= 64)
    float scale;        // 1/sqrt(128)
};

__global__ __launch_bounds__(16)
void fused_rope_hadamard_kernel(FusedRopeHadamardParams params) {
    constexpr int kNElts = 8;
    constexpr int kNChunks = 1;
    constexpr int kLogNElts = 3;   // log2(8)
    constexpr int kWarpSize = 16;
    constexpr int kLogWarpSize = 4; // log2(16)
    using input_t = at::BFloat16;

    const int batch_id = blockIdx.x;
    input_t *x = reinterpret_cast<input_t *>(params.x_ptr) + batch_id * 128;
    input_t *out = reinterpret_cast<input_t *>(params.out_ptr) + batch_id * 128;
    const int tid = threadIdx.x;

    // 1. Load 8 bf16 elements -> float
    float x_vals[kNChunks][kNElts];
    load_input<kNChunks, kNElts, input_t>(x, x_vals, 128);

    // 2. Apply interleaved RoPE (threads 0-7 only, first 64 dims)
    if (tid < 8) {
        int64_t pos = params.pos_ptr[batch_id];
        // Thread tid holds elements [tid*8..tid*8+7]
        // Interleaved pairs: (0,1),(2,3),(4,5),(6,7) within the 8 elements
        // cos/sin index for pair p: tid * 4 + p (maps to first 32 of cos_cached)
        #pragma unroll
        for (int p = 0; p < 4; p++) {
            int cos_idx = tid * 4 + p;
            float c = params.cos_ptr[pos * params.cos_stride + cos_idx];
            float s = params.sin_ptr[pos * params.cos_stride + cos_idx];
            float v0 = x_vals[0][2 * p];
            float v1 = x_vals[0][2 * p + 1];
            x_vals[0][2 * p]     = v0 * c - v1 * s;
            x_vals[0][2 * p + 1] = v1 * c + v0 * s;
        }
    }

    // 3. Hadamard butterfly: 3 thread-level + 4 warp-level stages
    hadamard_mult_thread<kLogNElts, kNChunks>(x_vals);
    hadamard_mult_warp<kLogWarpSize, 0, kNChunks, kNElts>(x_vals);
    // kNWarps=1, kNChunks=1: no smem exchange, no transposed stage

    // 4. Scale and store
    store_output<kNChunks, kNElts, input_t>(out, x_vals, 128, params.scale);
}

void fused_rope_hadamard_launch(FusedRopeHadamardParams &params, cudaStream_t stream) {
    dim3 grid(params.batch);
    fused_rope_hadamard_kernel<<<grid, 16, 0, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
