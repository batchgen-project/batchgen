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
#include <cuda_fp8.h>

#include "fast_hadamard_transform_common.h"

// Finite maximum of e4m3. bf16 represents it exactly, so clamping a bf16 value
// to it is lossless and the subsequent cast is always in range.
constexpr float kQFp8Max = 448.0f;

// dim=128, bf16: kNThreads=16, kNElts=8, kNChunks=1
// log2(8)=3 thread-level stages, log2(16)=4 warp-level stages = 7 total = log2(128)
// kNWarps=1 (16 < 32), no smem exchange needed

struct FusedRopeHadamardParams {
    using index_t = int64_t;
    void *__restrict__ x_ptr;       // [batch, 128] input (after LayerNorm), bf16
    void *__restrict__ out_ptr;     // [batch, 128] output, bf16
    void *__restrict__ out_fp8_ptr; // [batch, 128] optional e4m3 output, may be null
    const float *__restrict__ cos_ptr; // [max_seq, 64] cos cache (first 32 used)
    const float *__restrict__ sin_ptr; // [max_seq, 64] sin cache (first 32 used)
    const int64_t *__restrict__ pos_ptr; // [batch] position indices
    int batch;
    int cos_stride;     // stride of cos dim-0 (= 64)
    float scale;        // 1/sqrt(128)
};

// Saturating bf16 -> e4m3 cast that reproduces, byte for byte, the two ATen
// launches the Q scorer used to make: ``Tensor.clamp(-448, 448)`` on the bf16
// RoPE+Hadamard output, then ``copy_`` into a float8_e4m3fn tensor.
//
//   * finite inputs: the clamp pins |v| <= 448, so the cast is strictly in
//     range and both torch's ``fp8e4m3fn_from_fp32_value`` and SATFINITE round
//     to nearest even on the same value -> identical byte.
//   * +/-Inf and huge magnitudes: the clamp maps them to +/-448 first, which is
//     exactly representable, so neither path can reach torch's out-of-range
//     NaN encoding.
//   * NaN: ATen's clamp passes NaN through untouched (it special-cases NaN
//     rather than using min/max), and torch's cast then emits
//     ``sign | 0x7f``. SATFINITE drops the sign on a NaN, so the sign bit is
//     reattached here instead of relying on the hardware convert.
//
// There is no Q scale channel to apply: fp8_paged_mqa_logits reads q as plain
// e4m3 (the per-token fp32 scales in the paged cache cover K only).
__device__ __forceinline__ uint8_t clamped_bf16_to_e4m3(float v) {
    if (v != v) {
        return static_cast<uint8_t>(((__float_as_uint(v) >> 24) & 0x80u) | 0x7fu);
    }
    v = fminf(fmaxf(v, -kQFp8Max), kQFp8Max);
    return static_cast<uint8_t>(__nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3));
}

// bf16 store identical to ``store_output``, plus the optional e4m3 store. The
// e4m3 value is derived from the ROUNDED bf16 result, not from the fp32
// accumulator, because the ATen pair it replaces read the bf16 output tensor.
template <int kNChunks, int kNElts>
inline __device__ void store_output_bf16_and_e4m3(
    at::BFloat16 *out,
    uint8_t *out_fp8,
    float out_vals[kNChunks][kNElts],
    int dim,
    float scale
) {
    using vec_t = typename BytesToType<sizeof(at::BFloat16) * kNElts>::Type;
    using fp8_vec_t = typename BytesToType<sizeof(uint8_t) * kNElts>::Type;
    at::BFloat16 out_vals_store[kNChunks][kNElts];
    #pragma unroll
    for (int c = 0; c < kNChunks; ++c) {
        #pragma unroll
        for (int i = 0; i < kNElts; ++i) { out_vals_store[c][i] = out_vals[c][i] * scale; }
    }
    #pragma unroll
    for (int c = 0; c < kNChunks; ++c) {
        if ((c * blockDim.x + threadIdx.x) * kNElts < dim) {
            reinterpret_cast<vec_t *>(out)[c * blockDim.x + threadIdx.x] =
                reinterpret_cast<const vec_t *>(out_vals_store)[c];
        }
    }
    if (out_fp8 == nullptr) { return; }
    uint8_t fp8_store[kNChunks][kNElts];
    #pragma unroll
    for (int c = 0; c < kNChunks; ++c) {
        #pragma unroll
        for (int i = 0; i < kNElts; ++i) {
            fp8_store[c][i] = clamped_bf16_to_e4m3(float(out_vals_store[c][i]));
        }
    }
    #pragma unroll
    for (int c = 0; c < kNChunks; ++c) {
        if ((c * blockDim.x + threadIdx.x) * kNElts < dim) {
            reinterpret_cast<fp8_vec_t *>(out_fp8)[c * blockDim.x + threadIdx.x] =
                reinterpret_cast<const fp8_vec_t *>(fp8_store)[c];
        }
    }
}

// ``kApplyHadamard`` mirrors upstream's ``kHadamard`` template parameter on
// ``sglang::fused_q_indexer_rope_hadamard_quant``: both instantiations are
// built, and the caller picks the one its route needs. V3.2 drops the
// incoherence rotation on BOTH q and k -- H is orthonormal, so the pair
// cancels out of the indexer logit q @ k^T in exact arithmetic and only the
// fp8 quantization error moves.
template <bool kApplyHadamard>
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
    uint8_t *out_fp8 = params.out_fp8_ptr == nullptr
        ? nullptr
        : reinterpret_cast<uint8_t *>(params.out_fp8_ptr) + batch_id * 128;
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

    // 3. Hadamard butterfly: 3 thread-level + 4 warp-level stages.
    // ``params.scale`` is the 1/sqrt(128) normalization of H, so it belongs to
    // the rotation: with kApplyHadamard=false this is a pure interleaved-RoPE
    // pass and the store must not scale.
    float store_scale = 1.0f;
    if constexpr (kApplyHadamard) {
        hadamard_mult_thread<kLogNElts, kNChunks>(x_vals);
        hadamard_mult_warp<kLogWarpSize, 0, kNChunks, kNElts>(x_vals);
        // kNWarps=1, kNChunks=1: no smem exchange, no transposed stage
        store_scale = params.scale;
    }

    // 4. Scale and store (bf16, plus the saturating e4m3 copy when requested)
    store_output_bf16_and_e4m3<kNChunks, kNElts>(out, out_fp8, x_vals, 128, store_scale);
}

void fused_rope_hadamard_launch(FusedRopeHadamardParams &params,
                                bool apply_hadamard, cudaStream_t stream) {
    dim3 grid(params.batch);
    if (apply_hadamard) {
        fused_rope_hadamard_kernel<true><<<grid, 16, 0, stream>>>(params);
    } else {
        fused_rope_hadamard_kernel<false><<<grid, 16, 0, stream>>>(params);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ---------------------------------------------------------------------------
// Strided-input version (kernel version management: the contiguous kernel and
// launch above are unchanged and stay the entry point for every existing
// caller).
//
// The merged [q_b | wq_b] decode GEMM writes the indexer q as the trailing
// column slice of a wider [rows, q_b_cols + heads * 128] buffer, i.e. a
// [rows, heads, 128] tensor with strides (x_row_stride, 128, 1). That layout
// is not expressible as one flat [rows * heads, 128] row pitch, so the
// contiguous kernel's caller would have to reshape -- which silently
// allocates and copies. This version reads the source in place: flat row
// ``r`` of the launch is head ``r % heads`` of source row ``r / heads``.
// Everything after the load -- RoPE, the optional rotation, the bf16 store
// and the saturating e4m3 epilogue -- is the same code as above, and the
// outputs stay contiguous [rows * heads, 128].
// ---------------------------------------------------------------------------
template <bool kApplyHadamard>
__global__ __launch_bounds__(16)
void fused_rope_hadamard_strided_kernel(FusedRopeHadamardParams params,
                                        int64_t x_row_stride,
                                        int x_heads_per_row) {
    constexpr int kNElts = 8;
    constexpr int kNChunks = 1;
    constexpr int kLogNElts = 3;   // log2(8)
    constexpr int kWarpSize = 16;
    constexpr int kLogWarpSize = 4; // log2(16)
    using input_t = at::BFloat16;

    const int batch_id = blockIdx.x;
    const int64_t x_row = batch_id / x_heads_per_row;
    const int64_t x_head = batch_id - x_row * x_heads_per_row;
    input_t *x = reinterpret_cast<input_t *>(params.x_ptr)
        + x_row * x_row_stride + x_head * 128;
    input_t *out = reinterpret_cast<input_t *>(params.out_ptr) + batch_id * 128;
    uint8_t *out_fp8 = params.out_fp8_ptr == nullptr
        ? nullptr
        : reinterpret_cast<uint8_t *>(params.out_fp8_ptr) + batch_id * 128;
    const int tid = threadIdx.x;

    float x_vals[kNChunks][kNElts];
    load_input<kNChunks, kNElts, input_t>(x, x_vals, 128);

    if (tid < 8) {
        int64_t pos = params.pos_ptr[batch_id];
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

    float store_scale = 1.0f;
    if constexpr (kApplyHadamard) {
        hadamard_mult_thread<kLogNElts, kNChunks>(x_vals);
        hadamard_mult_warp<kLogWarpSize, 0, kNChunks, kNElts>(x_vals);
        store_scale = params.scale;
    }

    store_output_bf16_and_e4m3<kNChunks, kNElts>(out, out_fp8, x_vals, 128, store_scale);
}

void fused_rope_hadamard_strided_launch(FusedRopeHadamardParams &params,
                                        int64_t x_row_stride,
                                        int x_heads_per_row,
                                        bool apply_hadamard,
                                        cudaStream_t stream) {
    dim3 grid(params.batch);
    if (apply_hadamard) {
        fused_rope_hadamard_strided_kernel<true><<<grid, 16, 0, stream>>>(
            params, x_row_stride, x_heads_per_row);
    } else {
        fused_rope_hadamard_strided_kernel<false><<<grid, 16, 0, stream>>>(
            params, x_row_stride, x_heads_per_row);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
