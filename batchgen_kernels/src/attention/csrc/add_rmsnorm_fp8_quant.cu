#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>
#include <vector>

// ============================================================================
// Fused add + RMSNorm + per-token 128-block FP8 (e4m3) cast, hidden_size 6144.
//
// NEW KERNEL VERSION. It does NOT replace anything:
// `src/attention/csrc/rmsnorm.cu`'s `add_rmsnorm_bf16_6144_kernel` stays in the
// tree, stays compiled, and stays selectable (the GLM-5 switch
// `GLM5_NORM_QUANT_FUSE = False` restores it, and every non-GLM caller of
// `add_rmsnorm_forward` keeps using it unconditionally).
//
// WHY: three of the GLM-5 decode step's per-token blocked FP8 casts read a
// tensor a norm kernel wrote one launch earlier. The norm already holds the
// normalized row in registers when it stores it, so the cast is a second full
// HBM round trip and a second launch for data that was already resident. This
// kernel stores the FP8 bytes and the per-128-block FP32 scales from those
// registers, and the standalone quantize is not launched.
//
// ---------------------------------------------------------------------------
// BYTE IDENTITY — the two halves, and why each is bit-for-bit the old pair
// ---------------------------------------------------------------------------
//
// (1) The norm half is `add_rmsnorm_bf16_6144_kernel` COPIED VERBATIM: the same
//     BFloat16x8 vector type, the same 768-thread / 8-values-per-thread
//     partition, the same `fmaf(sum, sum, sum_sq)` accumulation, the same
//     24-warp shuffle reduction tree (`block_reduce_sum_24`), the same
//     `rsqrtf(total / 6144 + eps)`, the same `row_valid` PREDICATE treatment of
//     `row_valid_ptr`, the same re-read of the BF16-rounded residual as the
//     normalized numerator, and the same `val * s_inv_rms * scale` expression
//     order. Two multiplies cannot be contracted into an FMA (contraction needs
//     an add), and this translation unit is compiled with the same flags as
//     rmsnorm.cu -- in particular WITHOUT `--use_fast_math` -- so `rsqrtf` and
//     the multiplies emit the same instructions. Therefore `normed_out` and the
//     updated `residual` are the same bytes the unfused kernel writes.
//
// (2) The quantize half reproduces `_per_token_blocked_quantize_1d_kernel`
//     (batchgen_kernels/triton/fp8_quantize.py) statement for statement:
//
//       Triton                                      | here
//       --------------------------------------------+-----------------------------
//       q_bf16 = tl.load(...); q_float = .to(f32)   | __bfloat162float(normed_bf16)
//       amax = tl.max(tl.abs(q_float), axis=1)      | fabsf + fmaxf over the 128
//                                                   |   columns of one block
//       amax = tl.maximum(amax, 1.52587890625e-05)  | fmaxf(amax, kFp8MinNormal)
//       scale = tl.maximum(amax * (1/448), 1e-12)   | fmaxf(__fmul_rn(amax,
//                                                   |   kFp8SafeMaxInv), kEpsilon)
//       q = q_float / scale                         | __fdiv_rn(q, scale)
//       q = tl.minimum(q, 448.0)                    | fminf(q, 448.0f)
//       q = tl.maximum(q, -448.0)                   | fmaxf(q, -448.0f)
//       q = where(|q| < 1e30, q, 0.0)               | (fabsf(q) < 1e30f) ? q : 0
//       tl.store(fp8_ptr, q)                        | __nv_cvt_float_to_fp8(
//                                                   |   ..., __NV_SATFINITE,
//                                                   |   __NV_E4M3)
//       tl.store(scale_ptr, scale)                  | plain f32 store
//
//     The operand identities behind that table:
//
//     * THE QUANTIZED VALUE IS THE BF16-ROUNDED ONE. The standalone kernel
//       loaded `normed_out` back from global memory, i.e. the value after
//       `__float2bfloat16_rn`. This kernel converts the same `__nv_bfloat16` it
//       is about to store back to float and quantizes THAT, not the unrounded
//       FP32 accumulator. Quantizing the accumulator would be a different (and
//       arguably better) number -- and a byte mismatch.
//     * amax IS ORDER-INDEPENDENT. `max` is exactly associative and
//       commutative on floats, so reducing a 128-column block across 16 lanes
//       of a warp gives bit-identically what Triton's `tl.max(..., axis=1)`
//       gives over the same 128 values, whatever tree either uses. (`fmaxf` and
//       Triton's default `tl.maximum` both lower to `max.f32`, which returns the
//       non-NaN operand, so even NaN inputs agree -- the local fold is seeded
//       with the thread's first value, not with 0, so an all-NaN block yields
//       NaN on both sides rather than 0 on one of them.)
//     * `kFp8SafeMaxInv` is `static_cast<float>(1.0 / 448.0)`: a DOUBLE divide
//       folded at compile time and then rounded to float, which is exactly how
//       Triton materializes the `tl.constexpr` `1.0 / 448.0` written in Python.
//       `1.0f / 448.0f` would be a single-precision divide and is not
//       guaranteed to be the same float.
//     * `__fdiv_rn` / `__fmul_rn` pin IEEE round-to-nearest for the two
//       operations whose result a compiler flag could otherwise change
//       (`--use_fast_math` turns `/` into a reciprocal multiply). Triton emits
//       `div.rn.f32` / `mul.f32` here. Pinning them keeps this kernel
//       byte-identical even if someone later adds fast-math to its flags.
//     * The clamps are `fminf`/`fmaxf`, which lower to `min.f32`/`max.f32`,
//       the same instructions `tl.minimum`/`tl.maximum` lower to -- including
//       the "return the non-NaN operand" behaviour the `|q| < 1e30` guard
//       below then depends on.
//     * On sm_90 `__nv_cvt_float_to_fp8(..., __NV_SATFINITE, __NV_E4M3)` lowers
//       to `cvt.rn.satfinite.e4m3x2.f32`, which is what Triton emits for a
//       store of an FP32 tile through an FP8E4M3FN pointer. Saturation is
//       unreachable anyway: the value is already clamped into [-448, 448].
//       (The sm_80 image of this extension takes the header's software
//       conversion instead. It is the same round-to-nearest-even result, but
//       the bitwise gate is only claimed for sm_90, the only arch GLM-5 decode
//       runs on.)
//
// ---------------------------------------------------------------------------
// THE TWO VALIDITY GATES ARE SEPARATE, BECAUSE THE CALL SITES DISAGREE
// ---------------------------------------------------------------------------
// The GLM-5 layer segment calls `cuda_add_rmsnorm` WITHOUT `num_valid_tokens`
// (every row is normalized, padding rows included), while the DSA segment's
// `act_quant` of that output passes it (padding rows get exact-zero FP8 bytes
// and a 1e-12 scale, which is what the masked DeepGEMM and the indexer's dense
// wk GEMM rely on). Folding those into one gate would change one of the two
// outputs, so this kernel takes them separately:
//
//   norm_num_valid_ptr   -- skip the norm/residual work for this row entirely,
//                           exactly as the unfused kernel's early return does.
//   quant_num_valid_ptr  -- this row's FP8 bytes are exact zeros and its
//                           scales are 1e-12, exactly as the standalone
//                           quantize's `HAS_VALID_TOKENS` branch writes them.
//
// A row that is norm-skipped but quant-valid would have had the standalone
// quantize read whatever stale bytes `normed_out` carried, which is not a
// defined value to reproduce. The host function refuses that combination
// instead of inventing one.
// ============================================================================

namespace {

constexpr int kHiddenSize = 6144;
constexpr int kVectorWidth = 8;
constexpr int kVectorThreads = kHiddenSize / kVectorWidth;   // 768
constexpr int kVectorWarps = kVectorThreads / 32;            // 24
constexpr int kQuantBlock = 128;                             // fp8 scale block
constexpr int kNumQuantBlocks = kHiddenSize / kQuantBlock;   // 48
// Threads whose 8 columns fall inside one 128-column scale block. 16 threads
// per block, so one warp spans exactly two blocks and the per-block amax is a
// 16-lane butterfly inside the warp -- no shared memory, no __syncthreads.
constexpr int kThreadsPerQuantBlock = kQuantBlock / kVectorWidth;  // 16

// `_per_token_blocked_quantize_1d_kernel`'s constants, with the SAME spelling
// of each value (see the byte-identity note above for kFp8SafeMaxInv).
constexpr float kFp8SafeMax = 448.0f;
constexpr float kFp8SafeMaxInv = static_cast<float>(1.0 / 448.0);
constexpr float kFp8MinNormal = 1.52587890625e-05f;
constexpr float kEpsilon = 1e-12f;
constexpr float kFiniteBound = 1e30f;

struct alignas(16) BFloat16x8 {
    __nv_bfloat16 values[kVectorWidth];
};

static_assert(sizeof(BFloat16x8) == 16, "BFloat16x8 must be 16 bytes");
static_assert(sizeof(at::BFloat16) == sizeof(__nv_bfloat16),
              "at::BFloat16 and __nv_bfloat16 must have the same size");

// 8 e4m3 bytes = one 8-byte store, and a thread's 8 columns start at column
// 8 * threadIdx.x, so the store is naturally 8-byte aligned.
struct alignas(8) Fp8x8 {
    __nv_fp8_storage_t values[kVectorWidth];
};

static_assert(sizeof(Fp8x8) == 8, "Fp8x8 must be 8 bytes");
static_assert(sizeof(at::Float8_e4m3fn) == sizeof(__nv_fp8_storage_t),
              "at::Float8_e4m3fn and __nv_fp8_storage_t must have the same size");

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Verbatim from rmsnorm.cu: the reduction tree IS part of the numeric contract.
__device__ __forceinline__ float block_reduce_sum_24(
    float val,
    float* warp_sums)
{
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    val = warp_reduce_sum(val);
    if (lane == 0) {
        warp_sums[warp_id] = val;
    }
    __syncthreads();

    if (warp_id == 0) {
        val = (lane < kVectorWarps) ? warp_sums[lane] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane == 0) {
            warp_sums[0] = val;
        }
    }
    __syncthreads();
    return warp_sums[0];
}

// Per-128-block amax across the 16 lanes that cover that block. XOR offsets
// 1, 2, 4, 8 never flip lane bit 4, so the butterfly stays inside the 16-lane
// half of the warp that owns one block, and every lane ends with that block's
// amax (the quantize below needs it on all 16).
__device__ __forceinline__ float quant_block_amax(float val) {
    #pragma unroll
    for (int offset = 1; offset < kThreadsPerQuantBlock; offset <<= 1) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}

__global__ void add_rmsnorm_bf16_6144_fp8_quant_kernel(
    __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ hidden,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ normed_out,
    __nv_fp8_storage_t* __restrict__ quant_out,
    float* __restrict__ quant_scale,
    int64_t scale_stride_m,
    int64_t scale_stride_b,
    float eps,
    const int* __restrict__ norm_num_valid_ptr,
    const int* __restrict__ quant_num_valid_ptr,
    const __nv_bfloat16* __restrict__ row_valid_ptr)
{
    const int row = blockIdx.x;

    // Both gates are per-ROW and one block owns one row, so every branch below
    // is uniform across the block -- the __syncthreads in block_reduce_sum_24
    // is never reached by only part of the block.
    const bool quant_skip =
        (quant_num_valid_ptr != nullptr) && (row >= *quant_num_valid_ptr);
    const bool norm_skip =
        (norm_num_valid_ptr != nullptr) && (row >= *norm_num_valid_ptr);

    const int64_t vector_index =
        static_cast<int64_t>(row) * kVectorThreads + threadIdx.x;

    if (quant_skip) {
        // The standalone quantize's HAS_VALID_TOKENS branch, byte for byte:
        // exact-zero FP8 bytes over the whole row, and 1e-12 in every scale.
        Fp8x8 zeros;
        #pragma unroll
        for (int i = 0; i < kVectorWidth; ++i) {
            // 0.0f -> e4m3 0x00. Written through the same conversion the valid
            // path uses, so "exact zero" means the same byte in both branches.
            zeros.values[i] = __nv_cvt_float_to_fp8(
                0.0f, __NV_SATFINITE, __NV_E4M3);
        }
        reinterpret_cast<Fp8x8*>(quant_out)[vector_index] = zeros;
        if (threadIdx.x < kNumQuantBlocks) {
            quant_scale[static_cast<int64_t>(row) * scale_stride_m
                        + static_cast<int64_t>(threadIdx.x) * scale_stride_b] =
                kEpsilon;
        }
    }

    if (norm_skip) {
        return;
    }

    // ---- The norm, verbatim from add_rmsnorm_bf16_6144_kernel -------------
    // Row-select gate on `hidden`. The mask entry is read as a PREDICATE and
    // the invalid branch stores the literal 0.0f -- it is never multiplied in.
    // A multiply would be wrong, not merely slower: `hidden` rows that the
    // caller marks invalid may hold non-finite bytes, and bf16 NaN * 0 is NaN,
    // so a masked row would keep polluting the residual stream.
    const bool row_valid =
        (row_valid_ptr == nullptr) || (__bfloat162float(row_valid_ptr[row]) != 0.0f);

    __shared__ float warp_sums[kVectorWarps];
    __shared__ float s_inv_rms;

    auto* residual_vectors = reinterpret_cast<BFloat16x8*>(residual);
    const auto* hidden_vectors = reinterpret_cast<const BFloat16x8*>(hidden);
    const auto* weight_vectors = reinterpret_cast<const BFloat16x8*>(weight);
    auto* output_vectors = reinterpret_cast<BFloat16x8*>(normed_out);

    BFloat16x8 residual_vector = residual_vectors[vector_index];
    BFloat16x8 hidden_vector = hidden_vectors[vector_index];
    // Matches the scalar fallback: the reduction uses the unrounded float sum
    // while the normalized numerator re-reads the BF16-rounded residual.
    BFloat16x8 rounded_sum;
    float sum_sq = 0.0f;
    #pragma unroll
    for (int i = 0; i < kVectorWidth; ++i) {
        float h = row_valid ? __bfloat162float(hidden_vector.values[i]) : 0.0f;
        float sum = __bfloat162float(residual_vector.values[i]) + h;
        rounded_sum.values[i] = __float2bfloat16_rn(sum);
        sum_sq = fmaf(sum, sum, sum_sq);
    }
    residual_vectors[vector_index] = rounded_sum;

    float total = block_reduce_sum_24(sum_sq, warp_sums);
    if (threadIdx.x == 0) {
        s_inv_rms = rsqrtf(total / kHiddenSize + eps);
    }
    __syncthreads();

    BFloat16x8 weight_vector = weight_vectors[threadIdx.x];
    BFloat16x8 output_vector;
    #pragma unroll
    for (int i = 0; i < kVectorWidth; ++i) {
        float val = __bfloat162float(rounded_sum.values[i]);
        float scale = __bfloat162float(weight_vector.values[i]);
        output_vector.values[i] = __float2bfloat16_rn(val * s_inv_rms * scale);
    }
    output_vectors[vector_index] = output_vector;

    if (quant_skip) {
        // Already written above; the BF16 half still had to run.
        return;
    }

    // ---- The FP8 epilogue -------------------------------------------------
    // `_per_token_blocked_quantize_1d_kernel`, over the values this kernel just
    // stored -- read back from REGISTERS, through the same BF16 rounding the
    // standalone kernel would have loaded from memory.
    float normed[kVectorWidth];
    #pragma unroll
    for (int i = 0; i < kVectorWidth; ++i) {
        normed[i] = __bfloat162float(output_vector.values[i]);
    }
    // Seeded with the thread's FIRST value (not 0.0f) so an all-NaN 128-block
    // reduces to NaN here as it does in Triton, which has no 0 seed.
    float amax = fabsf(normed[0]);
    #pragma unroll
    for (int i = 1; i < kVectorWidth; ++i) {
        amax = fmaxf(amax, fabsf(normed[i]));
    }
    amax = quant_block_amax(amax);
    amax = fmaxf(amax, kFp8MinNormal);
    // SGLang formula: scale = amax * (1 / 448).
    const float scale = fmaxf(__fmul_rn(amax, kFp8SafeMaxInv), kEpsilon);

    Fp8x8 quantized;
    #pragma unroll
    for (int i = 0; i < kVectorWidth; ++i) {
        float q = __fdiv_rn(normed[i], scale);
        q = fminf(q, kFp8SafeMax);
        q = fmaxf(q, -kFp8SafeMax);
        const bool is_finite = fabsf(q) < kFiniteBound;
        q = is_finite ? q : 0.0f;
        quantized.values[i] = __nv_cvt_float_to_fp8(q, __NV_SATFINITE, __NV_E4M3);
    }
    reinterpret_cast<Fp8x8*>(quant_out)[vector_index] = quantized;

    // One thread per 128-block writes that block's scale. Lane 0 of each
    // 16-lane group is the thread whose column range starts the block.
    if (threadIdx.x % kThreadsPerQuantBlock == 0) {
        const int block_id = threadIdx.x / kThreadsPerQuantBlock;
        quant_scale[static_cast<int64_t>(row) * scale_stride_m
                    + static_cast<int64_t>(block_id) * scale_stride_b] = scale;
    }
}

inline bool is_aligned_16(const void* ptr) {
    return reinterpret_cast<std::uintptr_t>(ptr) % 16 == 0;
}

void check_num_valid(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.scalar_type() == torch::kInt32,
                name, " must be int32, got ", t.scalar_type());
    TORCH_CHECK(t.numel() == 1,
                name, " must contain one element, got ", t.numel());
}

}  // namespace

// Fused add + RMSNorm + per-token 128-block FP8 cast.
//
// Returns {normed_out, residual, quant_out, quant_scale}. `normed_out` and
// `residual` are bit-for-bit `add_rmsnorm_forward`'s; `quant_out` and
// `quant_scale` are bit-for-bit what `per_token_blocked_quantize_bf16_to_fp8_1d`
// would have produced from `normed_out`.
//
// Unsupported configurations FAIL LOUD. There is deliberately no scalar or
// non-6144 fallback: a fallback would have to be a second fused kernel, and a
// silent reroute to the unfused pair would make the eliminated launch reappear
// while the switch still claimed it was gone.
std::vector<torch::Tensor> add_rmsnorm_fp8_quant_forward(
    torch::Tensor residual,
    torch::Tensor hidden,
    torch::Tensor weight,
    double eps,
    c10::optional<torch::Tensor> norm_num_valid_tokens,
    c10::optional<torch::Tensor> quant_num_valid_tokens,
    c10::optional<torch::Tensor> row_valid_mask,
    bool scale_tma_aligned)
{
    TORCH_CHECK(residual.is_cuda(), "residual must be CUDA");
    TORCH_CHECK(hidden.is_cuda(), "hidden must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(residual.scalar_type() == at::ScalarType::BFloat16,
                "fused add+RMSNorm+FP8 requires bf16 residual, got ",
                residual.scalar_type());
    TORCH_CHECK(hidden.scalar_type() == at::ScalarType::BFloat16,
                "fused add+RMSNorm+FP8 requires bf16 hidden, got ",
                hidden.scalar_type());
    TORCH_CHECK(weight.scalar_type() == at::ScalarType::BFloat16,
                "fused add+RMSNorm+FP8 requires bf16 weight, got ",
                weight.scalar_type());

    residual = residual.contiguous();
    hidden = hidden.contiguous();
    weight = weight.contiguous();

    auto sizes = residual.sizes();
    const int hidden_size = sizes[sizes.size() - 1];
    TORCH_CHECK(hidden_size == kHiddenSize,
                "fused add+RMSNorm+FP8 is the hidden-size-", kHiddenSize,
                " version; got hidden_size=", hidden_size,
                ". Select GLM5_NORM_QUANT_FUSE=False for other widths.");
    TORCH_CHECK(residual.numel() == hidden.numel(),
                "residual and hidden must have the same number of elements (",
                residual.numel(), " vs ", hidden.numel(), ")");
    TORCH_CHECK(weight.numel() == kHiddenSize,
                "weight must hold ", kHiddenSize, " elements, got ",
                weight.numel());

    const int64_t num_rows = residual.numel() / hidden_size;

    auto normed_out = torch::empty_like(residual);
    auto quant_out = torch::empty(
        {num_rows, static_cast<int64_t>(kHiddenSize)},
        residual.options().dtype(torch::kFloat8_e4m3fn));

    // Scale layout, exactly as `per_token_blocked_quantize_bf16_to_fp8_1d`
    // allocates it. The TMA-aligned form is an MN-major view of a contiguous
    // [num_blocks, align4(M)] block, which is the layout DeepGEMM consumes
    // without a repack -- it is NOT contiguous, hence explicit strides below.
    torch::Tensor quant_scale;
    if (scale_tma_aligned) {
        const int64_t aligned_m = ((num_rows + 3) / 4) * 4;
        quant_scale = torch::empty(
            {static_cast<int64_t>(kNumQuantBlocks), aligned_m},
            residual.options().dtype(torch::kFloat32))
            .transpose(0, 1)
            .slice(0, 0, num_rows);
    } else {
        quant_scale = torch::empty(
            {num_rows, static_cast<int64_t>(kNumQuantBlocks)},
            residual.options().dtype(torch::kFloat32));
    }

    if (num_rows == 0) {
        return {normed_out, residual, quant_out, quant_scale};
    }

    const int* norm_num_valid_ptr = nullptr;
    if (norm_num_valid_tokens.has_value() && norm_num_valid_tokens->defined()) {
        check_num_valid(*norm_num_valid_tokens, "norm_num_valid_tokens");
        norm_num_valid_ptr = norm_num_valid_tokens->data_ptr<int>();
    }
    const int* quant_num_valid_ptr = nullptr;
    if (quant_num_valid_tokens.has_value() && quant_num_valid_tokens->defined()) {
        check_num_valid(*quant_num_valid_tokens, "quant_num_valid_tokens");
        quant_num_valid_ptr = quant_num_valid_tokens->data_ptr<int>();
    }
    // A norm-skipped row's `normed_out` is whatever the allocator handed over,
    // so a quantize of it has no defined result to be identical to. Refuse the
    // combination rather than produce one.
    TORCH_CHECK(norm_num_valid_ptr == nullptr
                || norm_num_valid_ptr == quant_num_valid_ptr,
                "norm_num_valid_tokens requires quant_num_valid_tokens to be "
                "the SAME tensor: a row the norm skips has undefined "
                "normed_out bytes, so its FP8 output cannot be reproduced");

    // `row_valid_mask` folds a caller's per-row gate on `hidden` into this
    // kernel. It is validated hard rather than silently ignored: a dropped
    // mask would let invalid rows into the residual stream, which is exactly
    // the failure this argument exists to prevent.
    const bool has_row_mask = row_valid_mask.has_value() && row_valid_mask->defined();
    torch::Tensor row_mask;
    if (has_row_mask) {
        row_mask = *row_valid_mask;
        TORCH_CHECK(row_mask.is_cuda(), "row_valid_mask must be CUDA");
        TORCH_CHECK(row_mask.scalar_type() == residual.scalar_type(),
                    "row_valid_mask dtype must match residual dtype");
        TORCH_CHECK(row_mask.numel() == num_rows,
                    "row_valid_mask must have one entry per row (expected ",
                    num_rows, ", got ", row_mask.numel(), ")");
        row_mask = row_mask.contiguous();
    }

    // The vector path is the ONLY path here (see the fail-loud note above).
    TORCH_CHECK(is_aligned_16(residual.data_ptr<at::BFloat16>())
                && is_aligned_16(hidden.data_ptr<at::BFloat16>())
                && is_aligned_16(weight.data_ptr<at::BFloat16>())
                && is_aligned_16(normed_out.data_ptr<at::BFloat16>()),
                "fused add+RMSNorm+FP8 requires 16-byte aligned bf16 operands");

    const int64_t scale_stride_m = quant_scale.stride(0);
    const int64_t scale_stride_b = quant_scale.stride(1);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    add_rmsnorm_bf16_6144_fp8_quant_kernel<<<
        static_cast<int>(num_rows), kVectorThreads, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(residual.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(hidden.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(normed_out.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_fp8_storage_t*>(
            quant_out.data_ptr<at::Float8_e4m3fn>()),
        quant_scale.data_ptr<float>(),
        scale_stride_m,
        scale_stride_b,
        static_cast<float>(eps),
        norm_num_valid_ptr,
        quant_num_valid_ptr,
        has_row_mask
            ? reinterpret_cast<const __nv_bfloat16*>(
                  row_mask.data_ptr<at::BFloat16>())
            : nullptr);

    return {normed_out, residual, quant_out, quant_scale};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add_rmsnorm_fp8_quant_forward", &add_rmsnorm_fp8_quant_forward,
          "Fused Add + RMSNorm + per-token 128-block FP8 e4m3 cast (CUDA, "
          "hidden_size 6144)",
          py::arg("residual"),
          py::arg("hidden"),
          py::arg("weight"),
          py::arg("eps"),
          py::arg("norm_num_valid_tokens") = c10::nullopt,
          py::arg("quant_num_valid_tokens") = c10::nullopt,
          py::arg("row_valid_mask") = c10::nullopt,
          py::arg("scale_tma_aligned") = false);
}
