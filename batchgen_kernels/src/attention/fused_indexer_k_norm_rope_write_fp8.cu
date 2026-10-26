/* Copyright 2025 SGLang Team. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ==========================================================================*/

// DSA indexer K chain in ONE launch: LayerNorm -> interleaved RoPE ->
// (optional) 128-point Hadamard -> per-token e4m3 quantization -> page-split
// paged store.
//
// Ported from sgl-project/sglang @ e00930c5489053f26d86b179cee0d087f846acbb
// (tag v0.5.21, the stack the comparison run measured):
//
//   * warp-per-token LayerNorm + rope-first schedule, the fp8 epilogue and the
//     "round through bf16 before taking the amax" rule from
//     python/sglang/kernels/jit/csrc/deepseek_v32/indexer_k.cuh
//     (sglang::fused_k_indexer_norm_rope_store);
//   * the 2 register + 5 shuffle Hadamard butterfly from
//     python/sglang/kernels/jit/csrc/deepseek_v4/main_norm_rope.cuh
//     (sglang::fused_q_indexer_rope_hadamard_quant, kHadamard=true).
//
// Three deliberate adaptations to BatchGen's contract:
//
//   1. Upstream's v3.2 kernel has NO Hadamard: it is instantiated with
//      kHadamard=false on BOTH q and k, which is logit-preserving in exact
//      arithmetic (H orthonormal, applied to both operands) and only trades
//      fp8 quantization accuracy. BatchGen now follows that on this
//      page-split FP8 route, so `apply_hadamard` is False in the model and
//      the Q index path drops the rotation with it. Both instantiations stay
//      compiled: the bf16 AoS indexer profiles (GLM-5/5.1/5.3) still rotate,
//      and the caller resolves which arm it needs from the cache it writes
//      (`cache_applies_hadamard` in
//      batchgen_kernels/attention/dsa/indexer_k_quant.py).
//   2. Upstream addresses the cache with a flat `out_cache_loc` (page =
//      loc >> page_bits). BatchGen resolves the physical page through the GPU
//      page table, with the same row gating as
//      `fused_indexer_k_write_fp8.cu`: rows at or past `num_valid_tokens` and
//      the -1 slot sentinel (CUDA graph bucket padding) write nothing.
//   3. The amax starts at 0 and folds in with `fmaxf`, which ignores a NaN
//      operand, so a stale padded row degenerates to the amax floor instead of
//      poisoning the scale. That is `fused_indexer_k_write_fp8.cu`'s documented
//      contract and the host-append sources keep being written for every row.
//
// The page layout, the amax floor, the reciprocal MULTIPLY and the SATFINITE
// cast are byte-identical to `fused_indexer_k_write_fp8.cu`, so a page written
// by this kernel is indistinguishable from one written by the un-fused chain
// up to the LayerNorm/RoPE/Hadamard arithmetic itself.

#include <torch/extension.h>

#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace {

constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 4;
constexpr int kBlockThreads = kWarpSize * kWarpsPerBlock;
constexpr int kHeadDim = 128;
constexpr int kRopeDim = 64;
constexpr int kVecSize = kHeadDim / kWarpSize;        // 4 elements per lane
constexpr int kRopeLanes = kRopeDim / kVecSize;       // lanes 0..15 own dims 0..63
constexpr int kScaleBytesPerToken = 4;
constexpr float kAmaxFloor = 1e-4f;
constexpr float kFp8Max = 448.0f;
constexpr unsigned kFullMask = 0xffffffffu;

static_assert(kHeadDim == kWarpSize * kVecSize, "one warp must own one token");
static_assert(kRopeDim == kWarpSize * 2, "two rope pairs per lane");

struct alignas(8) Bf16x4 {
    __nv_bfloat16 v[kVecSize];
};

struct alignas(4) Fp8x4 {
    uint8_t v[kVecSize];
};

__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
    for (int off = kWarpSize / 2; off >= 1; off >>= 1) {
        val += __shfl_xor_sync(kFullMask, val, off);
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
#pragma unroll
    for (int off = kWarpSize / 2; off >= 1; off >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(kFullMask, val, off));
    }
    return val;
}

// LayerNorm gamma/beta reach the kernel in the dtype the module holds them in:
// `nn.LayerNorm(128)` is bf16 after the model cast, fp32 before it. Loaded
// element-wise rather than vectorized: 128 parameter values are L1-hot and
// shared by every warp, and a scalar load cannot trip on a sliced tensor's
// alignment.
__device__ __forceinline__ float norm_param_value(
    const __nv_bfloat16* __restrict__ base, int index) {
    return __bfloat162float(base[index]);
}

__device__ __forceinline__ float norm_param_value(
    const float* __restrict__ base, int index) {
    return base[index];
}

template <typename ParamT>
__device__ __forceinline__ void load_norm_params(
    const ParamT* __restrict__ base, int lane, float out[kVecSize]) {
#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
        out[i] = norm_param_value(base, lane * kVecSize + i);
    }
}

// 128-point normalized Hadamard over one warp: 2 register stages mix the two
// low bits of the element index, 5 shuffle stages mix the lane bits. Natural
// (Sylvester) ordering, i.e. sign(e, f) = (-1)^popcount(e & f), the same
// transform `hadamard_mult_thread` + `hadamard_mult_warp` apply over the
// 16-thread x 8-element split -- only the summation order differs.
//
// The full shuffle mask is safe because the early return below is warp-uniform
// (`work_id` is constant across a warp), so every lane of a live warp
// participates.
__device__ __forceinline__ void hadamard_128(float data[kVecSize], int lane,
                                             float scale) {
    {
        const float a0 = data[0], a1 = data[1], a2 = data[2], a3 = data[3];
        data[0] = a0 + a1;
        data[1] = a0 - a1;
        data[2] = a2 + a3;
        data[3] = a2 - a3;
    }
    {
        const float a0 = data[0], a1 = data[1], a2 = data[2], a3 = data[3];
        data[0] = a0 + a2;
        data[1] = a1 + a3;
        data[2] = a0 - a2;
        data[3] = a1 - a3;
    }
#pragma unroll
    for (int mask = 1; mask < kWarpSize; mask <<= 1) {
#pragma unroll
        for (int i = 0; i < kVecSize; ++i) {
            const float other = __shfl_xor_sync(kFullMask, data[i], mask);
            data[i] = (lane & mask) ? (other - data[i]) : (data[i] + other);
        }
    }
#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
        data[i] *= scale;
    }
}

template <typename ParamT, typename PosT, bool kApplyHadamard>
__global__ __launch_bounds__(kBlockThreads, 16)
void fused_indexer_k_norm_rope_write_fp8_kernel(
    const __nv_bfloat16* __restrict__ k_raw,       // [N, 128] wk output
    const ParamT* __restrict__ ln_weight,          // [128]
    const ParamT* __restrict__ ln_bias,            // [128]
    const float* __restrict__ cos_cache,           // [max_seq, cos_stride]
    const float* __restrict__ sin_cache,           // [max_seq, cos_stride]
    const PosT* __restrict__ positions,            // [N]
    const int32_t* __restrict__ page_table,        // [num_slots, table_cols]
    const int32_t* __restrict__ slot_indices,      // [N]
    const int32_t* __restrict__ token_indices,     // [N]
    const int32_t* __restrict__ num_valid_tokens,  // [1], may be nullptr
    uint8_t* __restrict__ cache,                   // [num_pages, page_bytes]
    uint8_t* __restrict__ k_fp8_out,               // [N, 128], may be nullptr
    float* __restrict__ scales_out,                // [N], may be nullptr
    int num_tokens,
    int page_size,
    int page_table_cols,
    int num_pages,
    int cos_stride,
    float ln_eps,
    float hadamard_scale) {
    const int warp_id = threadIdx.x / kWarpSize;
    const int lane = threadIdx.x % kWarpSize;
    const int token = blockIdx.x * kWarpsPerBlock + warp_id;
    // Warp-uniform: every surviving warp keeps all 32 lanes, which is what the
    // full-mask shuffles below require.
    if (token >= num_tokens) {
        return;
    }

    float data[kVecSize];
    float gamma[kVecSize];
    float beta[kVecSize];

    // ---- LayerNorm ------------------------------------------------------
    {
        const Bf16x4 row = reinterpret_cast<const Bf16x4*>(
            k_raw + static_cast<long long>(token) * kHeadDim)[lane];
        load_norm_params<ParamT>(ln_weight, lane, gamma);
        load_norm_params<ParamT>(ln_bias, lane, beta);

        float sum = 0.0f;
#pragma unroll
        for (int i = 0; i < kVecSize; ++i) {
            data[i] = __bfloat162float(row.v[i]);
            sum += data[i];
        }
        const float mean = warp_reduce_sum(sum) / kHeadDim;

        float var = 0.0f;
#pragma unroll
        for (int i = 0; i < kVecSize; ++i) {
            const float centered = data[i] - mean;
            var += centered * centered;
        }
        const float inv_std = rsqrtf(warp_reduce_sum(var) / kHeadDim + ln_eps);

#pragma unroll
        for (int i = 0; i < kVecSize; ++i) {
            data[i] = (data[i] - mean) * inv_std * gamma[i] + beta[i];
        }
    }

    // The un-fused chain materializes the LayerNorm result as bf16 before the
    // RoPE kernel reads it, so the rounding happens here too.
#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
        data[i] = __bfloat162float(__float2bfloat16_rn(data[i]));
    }

    // ---- Interleaved RoPE on the leading 64 dims ------------------------
    // Lane L owns elements 4L..4L+3, i.e. the interleaved (real, imag) pairs
    // 2L and 2L+1; the cos/sin tables are indexed by that pair index.
    if (lane < kRopeLanes) {
        const long long pos = static_cast<long long>(positions[token]);
        const float* cos_row = cos_cache + pos * cos_stride;
        const float* sin_row = sin_cache + pos * cos_stride;
#pragma unroll
        for (int p = 0; p < 2; ++p) {
            const int pair = lane * 2 + p;
            const float c = cos_row[pair];
            const float s = sin_row[pair];
            const float v0 = data[2 * p];
            const float v1 = data[2 * p + 1];
            data[2 * p] = v0 * c - v1 * s;
            data[2 * p + 1] = v1 * c + v0 * s;
        }
    }

    // ---- Hadamard ------------------------------------------------------
    if constexpr (kApplyHadamard) {
        hadamard_128(data, lane, hadamard_scale);
    }

    // ---- fp8 epilogue: round to bf16, then quantize --------------------
    // The amax must be taken on the ROUNDED values: the chain this replaces
    // stored a bf16 tensor and the separate writer quantized that tensor.
#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
        data[i] = __bfloat162float(__float2bfloat16_rn(data[i]));
    }

    float local_max = 0.0f;
#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
        local_max = fmaxf(local_max, fabsf(data[i]));
    }
    const float scale = fmaxf(warp_reduce_max(local_max), kAmaxFloor) / kFp8Max;
    const float inv = 1.0f / scale;

    Fp8x4 quantized;
#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
        quantized.v[i] = __nv_cvt_float_to_fp8(data[i] * inv, __NV_SATFINITE,
                                               __NV_E4M3);
    }

    // Host-append sources are row-local and written for EVERY row of the
    // launch, exactly as the un-fused writer wrote them: which rows reach the
    // host is decided by the sequence set the append is called with.
    if (k_fp8_out != nullptr) {
        reinterpret_cast<Fp8x4*>(
            k_fp8_out + static_cast<long long>(token) * kHeadDim)[lane] =
            quantized;
    }
    if (scales_out != nullptr && lane == 0) {
        scales_out[token] = scale;
    }

    // ---- Page-split paged store ----------------------------------------
    if (num_valid_tokens != nullptr && token >= *num_valid_tokens) {
        return;
    }
    const int slot = slot_indices[token];
    if (slot < 0) {
        return;
    }
    const int token_index = token_indices[token];
    const int page_slot = token_index / page_size;
    const int offset = token_index - page_slot * page_size;
    if (page_slot >= page_table_cols) {
        return;
    }
    const int page =
        page_table[static_cast<long long>(slot) * page_table_cols + page_slot];
    // An unallocated page-table entry (-1) for an otherwise valid row is a
    // caller bug; the row is dropped rather than written through a negative
    // page base.
    if (page < 0 || page >= num_pages) {
        return;
    }

    const long long page_bytes =
        static_cast<long long>(page_size) * (kHeadDim + kScaleBytesPerToken);
    uint8_t* page_base = cache + page * page_bytes;
    reinterpret_cast<Fp8x4*>(page_base +
                             static_cast<long long>(offset) * kHeadDim)[lane] =
        quantized;
    if (lane == 0) {
        uint8_t* sdst = page_base +
                        static_cast<long long>(page_size) * kHeadDim +
                        static_cast<long long>(offset) * kScaleBytesPerToken;
        *reinterpret_cast<float*>(sdst) = scale;
    }
}

void CheckCudaInt32(const torch::Tensor& t, const char* name,
                    const torch::Tensor& ref) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.device() == ref.device(), name,
                " must share the device of k_raw");
    TORCH_CHECK(t.scalar_type() == torch::kInt32, name, " must be int32");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

void CheckRopeTable(const torch::Tensor& t, const char* name,
                    const torch::Tensor& ref) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.device() == ref.device(), name,
                " must share the device of k_raw");
    TORCH_CHECK(t.scalar_type() == torch::kFloat32, name, " must be float32");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.dim() == 2, name, " must be [max_seq, rope_dim]");
}

}  // namespace

void fused_indexer_k_norm_rope_write_fp8(
    torch::Tensor k_raw,         // [N, 128] bf16, the wk GEMM output
    torch::Tensor ln_weight,     // [128] bf16 or fp32
    torch::Tensor ln_bias,       // [128] bf16 or fp32
    torch::Tensor cos_cache,     // [max_seq, rope_dim] fp32
    torch::Tensor sin_cache,     // [max_seq, rope_dim] fp32
    torch::Tensor positions,     // [N] int32 or int64
    torch::Tensor page_table,    // [num_slots, table_cols] int32
    torch::Tensor slot_indices,  // [N] int32
    torch::Tensor token_indices, // [N] int32
    torch::Tensor cache,         // [num_pages, page_bytes] 1-byte
    int64_t page_size_tokens,
    double ln_eps,
    bool apply_hadamard,
    c10::optional<torch::Tensor> k_fp8_out,
    c10::optional<torch::Tensor> scales_out,
    c10::optional<torch::Tensor> num_valid_tokens) {
    TORCH_CHECK(k_raw.is_cuda(), "k_raw must be a CUDA tensor");
    TORCH_CHECK(k_raw.scalar_type() == torch::kBFloat16,
                "k_raw must be bfloat16");
    TORCH_CHECK(k_raw.is_contiguous(), "k_raw must be contiguous");
    TORCH_CHECK(k_raw.dim() == 2, "k_raw must be [N, 128]");
    TORCH_CHECK(k_raw.size(1) == kHeadDim,
                "the fused indexer-K norm/rope writer is specialized for a "
                "head dim of ", kHeadDim, ", got ", k_raw.size(1));

    const int num_tokens = static_cast<int>(k_raw.size(0));
    const int page_size = static_cast<int>(page_size_tokens);
    TORCH_CHECK(page_size > 0, "page_size_tokens must be positive");

    TORCH_CHECK(ln_weight.is_cuda() && ln_bias.is_cuda(),
                "ln_weight and ln_bias must be CUDA tensors");
    TORCH_CHECK(ln_weight.device() == k_raw.device() &&
                    ln_bias.device() == k_raw.device(),
                "ln_weight and ln_bias must share the device of k_raw");
    TORCH_CHECK(ln_weight.scalar_type() == ln_bias.scalar_type(),
                "ln_weight and ln_bias must share a dtype, got ",
                ln_weight.scalar_type(), " and ", ln_bias.scalar_type());
    TORCH_CHECK(ln_weight.scalar_type() == torch::kBFloat16 ||
                    ln_weight.scalar_type() == torch::kFloat32,
                "the LayerNorm parameters must be bfloat16 or float32, got ",
                ln_weight.scalar_type());
    TORCH_CHECK(ln_weight.is_contiguous() && ln_bias.is_contiguous(),
                "ln_weight and ln_bias must be contiguous");
    TORCH_CHECK(ln_weight.numel() == kHeadDim && ln_bias.numel() == kHeadDim,
                "ln_weight and ln_bias must hold ", kHeadDim, " elements");

    CheckRopeTable(cos_cache, "cos_cache", k_raw);
    CheckRopeTable(sin_cache, "sin_cache", k_raw);
    TORCH_CHECK(cos_cache.size(1) == sin_cache.size(1) &&
                    cos_cache.size(0) == sin_cache.size(0),
                "cos_cache and sin_cache must have the same shape");
    const int cos_stride = static_cast<int>(cos_cache.size(1));
    TORCH_CHECK(cos_stride >= kRopeDim / 2,
                "the rope tables must hold at least ", kRopeDim / 2,
                " interleaved pairs per position, got ", cos_stride);

    TORCH_CHECK(positions.is_cuda() && positions.device() == k_raw.device(),
                "positions must be a CUDA tensor on k_raw's device");
    TORCH_CHECK(positions.is_contiguous(), "positions must be contiguous");
    TORCH_CHECK(positions.scalar_type() == torch::kInt32 ||
                    positions.scalar_type() == torch::kInt64,
                "positions must be int32 or int64, got ",
                positions.scalar_type());
    TORCH_CHECK(positions.numel() == num_tokens,
                "positions must hold one entry per token");

    CheckCudaInt32(page_table, "page_table", k_raw);
    CheckCudaInt32(slot_indices, "slot_indices", k_raw);
    CheckCudaInt32(token_indices, "token_indices", k_raw);
    TORCH_CHECK(page_table.dim() == 2,
                "page_table must be [num_slots, max_pages_per_sequence]");
    TORCH_CHECK(slot_indices.size(0) == num_tokens,
                "slot_indices must hold one entry per token");
    TORCH_CHECK(token_indices.size(0) == num_tokens,
                "token_indices must hold one entry per token");

    TORCH_CHECK(cache.is_cuda(), "cache must be a CUDA tensor");
    TORCH_CHECK(cache.device() == k_raw.device(),
                "cache must share the device of k_raw");
    TORCH_CHECK(cache.is_contiguous(), "cache must be contiguous");
    TORCH_CHECK(cache.element_size() == 1,
                "cache must have a 1-byte element type");
    TORCH_CHECK(cache.dim() == 2, "cache must be [num_pages, page_bytes]");
    const int64_t expected_page_bytes =
        static_cast<int64_t>(page_size) * (kHeadDim + kScaleBytesPerToken);
    TORCH_CHECK(cache.size(1) == expected_page_bytes,
                "cache page bytes ", cache.size(1), " do not match page_size=",
                page_size, " and data_dim=", kHeadDim);
    const int num_pages = static_cast<int>(cache.size(0));

    uint8_t* k_fp8_ptr = nullptr;
    if (k_fp8_out.has_value()) {
        const torch::Tensor& out = *k_fp8_out;
        TORCH_CHECK(out.is_cuda() && out.device() == k_raw.device(),
                    "k_fp8_out must be a CUDA tensor on k_raw's device");
        TORCH_CHECK(out.is_contiguous(), "k_fp8_out must be contiguous");
        TORCH_CHECK(out.element_size() == 1,
                    "k_fp8_out must have a 1-byte element type");
        TORCH_CHECK(out.numel() == static_cast<int64_t>(num_tokens) * kHeadDim,
                    "k_fp8_out must hold N * ", kHeadDim, " elements");
        k_fp8_ptr = reinterpret_cast<uint8_t*>(out.data_ptr());
    }
    float* scales_ptr = nullptr;
    if (scales_out.has_value()) {
        const torch::Tensor& out = *scales_out;
        TORCH_CHECK(out.is_cuda() && out.device() == k_raw.device(),
                    "scales_out must be a CUDA tensor on k_raw's device");
        TORCH_CHECK(out.is_contiguous(), "scales_out must be contiguous");
        TORCH_CHECK(out.scalar_type() == torch::kFloat32,
                    "scales_out must be float32");
        TORCH_CHECK(out.numel() == num_tokens,
                    "scales_out must hold one scale per token");
        scales_ptr = out.data_ptr<float>();
    }
    const int32_t* valid_ptr = nullptr;
    if (num_valid_tokens.has_value()) {
        const torch::Tensor& nv = *num_valid_tokens;
        CheckCudaInt32(nv, "num_valid_tokens", k_raw);
        TORCH_CHECK(nv.numel() == 1,
                    "num_valid_tokens must contain one element");
        valid_ptr = nv.data_ptr<int32_t>();
    }

    if (num_tokens == 0) {
        return;
    }

    // Same value the un-fused chain passed down as ``k.shape[-1] ** -0.5``:
    // computed in double, then rounded once to float.
    const float hadamard_scale =
        static_cast<float>(1.0 / std::sqrt(static_cast<double>(kHeadDim)));
    const int num_blocks = (num_tokens + kWarpsPerBlock - 1) / kWarpsPerBlock;
    auto stream = at::cuda::getCurrentCUDAStream();

#define BATCHGEN_LAUNCH_K_NORM_ROPE(ParamT, PosT, HAD)                        \
    fused_indexer_k_norm_rope_write_fp8_kernel<ParamT, PosT, HAD>             \
        <<<num_blocks, kBlockThreads, 0, stream>>>(                           \
            reinterpret_cast<const __nv_bfloat16*>(k_raw.data_ptr()),         \
            reinterpret_cast<const ParamT*>(ln_weight.data_ptr()),            \
            reinterpret_cast<const ParamT*>(ln_bias.data_ptr()),              \
            cos_cache.data_ptr<float>(), sin_cache.data_ptr<float>(),         \
            reinterpret_cast<const PosT*>(positions.data_ptr()),              \
            page_table.data_ptr<int32_t>(), slot_indices.data_ptr<int32_t>(), \
            token_indices.data_ptr<int32_t>(), valid_ptr,                     \
            reinterpret_cast<uint8_t*>(cache.data_ptr()), k_fp8_ptr,          \
            scales_ptr, num_tokens, page_size,                                \
            static_cast<int>(page_table.size(1)), num_pages, cos_stride,      \
            static_cast<float>(ln_eps), hadamard_scale)

#define BATCHGEN_DISPATCH_K_NORM_ROPE_POS(ParamT, HAD)                        \
    do {                                                                      \
        if (positions.scalar_type() == torch::kInt32) {                       \
            BATCHGEN_LAUNCH_K_NORM_ROPE(ParamT, int32_t, HAD);                \
        } else {                                                              \
            BATCHGEN_LAUNCH_K_NORM_ROPE(ParamT, int64_t, HAD);                \
        }                                                                     \
    } while (0)

#define BATCHGEN_DISPATCH_K_NORM_ROPE(HAD)                                    \
    do {                                                                      \
        if (ln_weight.scalar_type() == torch::kBFloat16) {                    \
            BATCHGEN_DISPATCH_K_NORM_ROPE_POS(__nv_bfloat16, HAD);            \
        } else {                                                              \
            BATCHGEN_DISPATCH_K_NORM_ROPE_POS(float, HAD);                    \
        }                                                                     \
    } while (0)

    if (apply_hadamard) {
        BATCHGEN_DISPATCH_K_NORM_ROPE(true);
    } else {
        BATCHGEN_DISPATCH_K_NORM_ROPE(false);
    }

#undef BATCHGEN_DISPATCH_K_NORM_ROPE
#undef BATCHGEN_DISPATCH_K_NORM_ROPE_POS
#undef BATCHGEN_LAUNCH_K_NORM_ROPE
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_indexer_k_norm_rope_write_fp8",
          &fused_indexer_k_norm_rope_write_fp8,
          "LayerNorm + interleaved RoPE + optional Hadamard + e4m3 quantize + "
          "page-split paged write of the DSA indexer K cache, in one launch",
          py::arg("k_raw"), py::arg("ln_weight"), py::arg("ln_bias"),
          py::arg("cos_cache"), py::arg("sin_cache"), py::arg("positions"),
          py::arg("page_table"), py::arg("slot_indices"),
          py::arg("token_indices"), py::arg("cache"),
          py::arg("page_size_tokens"), py::arg("ln_eps"),
          py::arg("apply_hadamard"),
          py::arg("k_fp8_out") = c10::nullopt,
          py::arg("scales_out") = c10::nullopt,
          py::arg("num_valid_tokens") = c10::nullopt);
}
