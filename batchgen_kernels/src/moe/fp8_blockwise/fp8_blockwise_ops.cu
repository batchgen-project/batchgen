// BatchGen — FP8 Blockwise MoE Pipeline Operations
// CUDA kernels for 3D-sparse FP8 quantization and fused SiLU activation.
// All kernels operate on [E, mtp, dim] layout with tokens_per_expert masking.
//
// Kernels:
// 1. act_quant_3d       — BF16→FP8 blockwise quantization (128-element blocks)
// 2. silu_mul_3d        — SiLU(gate) × up (vectorized BF16)
// 3. fused_silu_quant_3d — SiLU(gate) × up + FP8 quantization (fuses S1 epilogue + S3 input)
// 4. act_quant_ragged   — act_quant_3d arithmetic on compact ragged [max_rows, K] rows

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <algorithm>
#include <cstdint>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <pybind11/pybind11.h>

#define FP8_MAX_VAL 448.0f
#define QUANT_EPS 1e-12f
#define BLOCK_SIZE_QUANT 128

// ============================================================================
// Kernel 1: act_quant_3d — 3D-sparse FP8 blockwise quantization
// Grid: (E,)   Block: 128 threads (4 warps)
// Each CTA processes one expert, loops over valid tokens only.
// Each warp handles one K-block (128 elements) at a time, warps stride.
// ============================================================================
// v2: grid = (E, mtp), one CTA per (expert, token).
// v1 (grid=(E,) with serial outer token loop) was launch-limited at
// E=16 → only 16 CTAs on H20's ~132 SMs, ~12% occupancy, ~594 µs per
// call at K=6144. v2 saturates the GPU with E×mtp=2048 CTAs and
// runs in ~22 µs (27× faster, validated in
// tests/kernels/test_act_quant_3d_v2.py bit-exact vs v1).
__global__ void act_quant_3d_kernel(
    const __nv_bfloat16* __restrict__ x,   // [E, mtp, K]
    uint8_t* __restrict__ y,                // [E, mtp, K] FP8
    float* __restrict__ scale,              // [E, mtp, num_k_blocks]
    const int32_t* __restrict__ tokens_per_expert,  // [E]
    int mtp, int K, int num_k_blocks
) {
    const int expert = blockIdx.x;
    const int token  = blockIdx.y;
    const int valid_tokens = tokens_per_expert[expert];
    if (token >= valid_tokens) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int num_warps = blockDim.x / 32;

    const __nv_bfloat16* x_row =
        x + ((int64_t)expert * mtp + token) * K;
    uint8_t* y_row =
        y + ((int64_t)expert * mtp + token) * K;
    float* scale_row =
        scale + ((int64_t)expert * mtp + token) * num_k_blocks;

    for (int kb = warp_id; kb < num_k_blocks; kb += num_warps) {
        int col_base = kb * BLOCK_SIZE_QUANT;

        float vals[4];
        float local_max = 0.0f;

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int col = col_base + lane_id * 4 + i;
            if (col < K) {
                vals[i] = __bfloat162float(x_row[col]);
            } else {
                vals[i] = 0.0f;
            }
            local_max = fmaxf(local_max, fabsf(vals[i]));
        }

        #pragma unroll
        for (int offset = 16; offset >= 1; offset >>= 1) {
            float other = __shfl_xor_sync(0xffffffff, local_max, offset);
            local_max = fmaxf(local_max, other);
        }

        // Use the FP32 op sequence (`scale = amax * (1/448)`,
        // `scaled = x / scale`) so the CUDA kernel produces byte-exact
        // FP8 output. Previously we did
        //     s = fmax(local_max, eps) / FP8_MAX_VAL;  // FP32 division
        //     inv_s = 1.0f / s;                        // FP32 reciprocal
        //     scaled = vals[i] * inv_s;                // FP32 mul
        // which introduces an extra FP32 rounding step and flips the
        // final FP8 cast on ~0.1% of elements.
        constexpr float FP8_MAX_VAL_INV = 1.0f / FP8_MAX_VAL;
        float s = fmaxf(local_max, QUANT_EPS) * FP8_MAX_VAL_INV;

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int col = col_base + lane_id * 4 + i;
            if (col < K) {
                float scaled = vals[i] / s;
                scaled = fmaxf(fminf(scaled, FP8_MAX_VAL), -FP8_MAX_VAL);
                y_row[col] = __nv_cvt_float_to_fp8(scaled, __NV_SATFINITE, __NV_E4M3);
            }
        }

        if (lane_id == 0) {
            scale_row[kb] = s;
        }
    }
}

// ============================================================================
// Kernel 1b: act_quant_ragged — compact ragged FP8 blockwise quantization
// Grid: max(one resident wave, ceil(max_rows / 16)), capped at max_rows
// Block: 128 threads (4 warps)
// Expert e owns rows [cu_seqlens[e], cu_seqlens[e] + align64(seqlens[e])).
// Valid rows use the exact act_quant_3d arithmetic. Pad rows inside that span
// are quantized as zero input (y = 0, finite eps scale), so stale buffer rows
// never reach GEMM tiles. Rows outside every span are left untouched.
// Scale layout is the grouped GEMM x_scale layout [num_k_blocks, max_rows].
//
// Launch shape. The grid no longer has one block per row of the padded
// capacity: it is one resident wave (sized from the device), or one block per
// 16 rows of capacity when that is larger, and never depends on the live
// counts, which are known only on device -- so a CUDA graph captured once
// replays any seqlens. Blocks walk the concatenation of every expert's span
// (clipped to [cu_seqlens[0], cu_seqlens[E]) and [0, max_rows)) round-robin,
// so the rows visited are the span rows, not the capacity. In the
// uniform-stride DeepEP layout cu_seqlens[E] == max_rows, so walking
// [cu_seqlens[0], cu_seqlens[E]) row by row would still visit every padded row.
//
// Write set (unchanged): row r is written iff r in [0, max_rows) and
// r in [cu_seqlens[0], cu_seqlens[E]) and, with o = the binary-search owner
// of r, r - cu_seqlens[o] < align64(seqlens[o]). Every such r lies in owner
// o's span, so it is visited while walking expert o; a row visited while
// walking any other expert fails the owner check and is skipped. Each written
// row is therefore produced exactly once, by the same per-row code.
// ============================================================================
#define RAGGED_ROW_ALIGN 64
#define RAGGED_QUANT_THREADS 128
#define RAGGED_ROWS_PER_BLOCK 16

// One row of act_quant_ragged: the act_quant_3d arithmetic on a valid row,
// zero input on a pad row. Writes y[row, 0:K) and scale[0:num_k_blocks, row].
// kVec (K % 4 == 0, x 8-byte and y 4-byte aligned): each lane's 4 columns are
// moved with one 8-byte load and one 4-byte store instead of 4 + 4 scalar
// accesses. The values, the FP32 op sequence and the bytes are unchanged; a
// lane's quad is either fully inside [0, K) or fully outside it.
template <bool kVec>
__device__ __forceinline__ void act_quant_ragged_row(
    const __nv_bfloat16* __restrict__ x,
    uint8_t* __restrict__ y,
    float* __restrict__ scale,
    int64_t row, bool is_valid, int max_rows, int K, int num_k_blocks
) {
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    constexpr int num_warps = RAGGED_QUANT_THREADS / 32;

    const __nv_bfloat16* x_row = x + (int64_t)row * K;
    uint8_t* y_row = y + (int64_t)row * K;

    if (!is_valid) {
        // A pad row quantizes all-zero input: every block max is 0, so the
        // scale is the eps scale below (the same FP32 expression the general
        // path evaluates) and every byte is FP8(+0 / s) = 0x00. Store those
        // results directly; x is not read, exactly as before.
        constexpr float FP8_MAX_VAL_INV = 1.0f / FP8_MAX_VAL;
        const float s_pad = fmaxf(0.0f, QUANT_EPS) * FP8_MAX_VAL_INV;
        for (int kb = warp_id; kb < num_k_blocks; kb += num_warps) {
            const int col0 = kb * BLOCK_SIZE_QUANT + lane_id * 4;
            if constexpr (kVec) {
                if (col0 < K) *reinterpret_cast<uint32_t*>(y_row + col0) = 0u;
            } else {
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    if (col0 + i < K) y_row[col0 + i] = 0;
                }
            }
            if (lane_id == 0) {
                scale[(int64_t)kb * max_rows + row] = s_pad;
            }
        }
        return;
    }

    for (int kb = warp_id; kb < num_k_blocks; kb += num_warps) {
        int col_base = kb * BLOCK_SIZE_QUANT;
        const int col0 = col_base + lane_id * 4;

        float vals[4];
        float local_max = 0.0f;

        if constexpr (kVec) {
            if (col0 < K) {
                const uint2 raw = *reinterpret_cast<const uint2*>(x_row + col0);
                const __nv_bfloat16* v = reinterpret_cast<const __nv_bfloat16*>(&raw);
                #pragma unroll
                for (int i = 0; i < 4; i++) vals[i] = __bfloat162float(v[i]);
            } else {
                #pragma unroll
                for (int i = 0; i < 4; i++) vals[i] = 0.0f;
            }
            #pragma unroll
            for (int i = 0; i < 4; i++) local_max = fmaxf(local_max, fabsf(vals[i]));
        } else {
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int col = col0 + i;
                if (col < K) {
                    vals[i] = __bfloat162float(x_row[col]);
                } else {
                    vals[i] = 0.0f;
                }
                local_max = fmaxf(local_max, fabsf(vals[i]));
            }
        }

        #pragma unroll
        for (int offset = 16; offset >= 1; offset >>= 1) {
            float other = __shfl_xor_sync(0xffffffff, local_max, offset);
            local_max = fmaxf(local_max, other);
        }

        // Same FP32 op sequence as act_quant_3d (byte-exact on valid rows).
        constexpr float FP8_MAX_VAL_INV = 1.0f / FP8_MAX_VAL;
        float s = fmaxf(local_max, QUANT_EPS) * FP8_MAX_VAL_INV;

        if constexpr (kVec) {
            if (col0 < K) {
                uint32_t packed = 0;
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    float scaled = vals[i] / s;
                    scaled = fmaxf(fminf(scaled, FP8_MAX_VAL), -FP8_MAX_VAL);
                    packed |= (uint32_t)__nv_cvt_float_to_fp8(scaled, __NV_SATFINITE, __NV_E4M3)
                              << (8 * i);
                }
                *reinterpret_cast<uint32_t*>(y_row + col0) = packed;
            }
        } else {
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int col = col0 + i;
                if (col < K) {
                    float scaled = vals[i] / s;
                    scaled = fmaxf(fminf(scaled, FP8_MAX_VAL), -FP8_MAX_VAL);
                    y_row[col] = __nv_cvt_float_to_fp8(scaled, __NV_SATFINITE, __NV_E4M3);
                }
            }
        }

        if (lane_id == 0) {
            scale[(int64_t)kb * max_rows + row] = s;
        }
    }
}

// 2048 resident threads per SM / 128 per block: ask for full residency
// (caps registers at 32 per thread; the old one-block-per-row kernel used 25).
template <bool kVec>
__global__ void __launch_bounds__(RAGGED_QUANT_THREADS, 2048 / RAGGED_QUANT_THREADS)
act_quant_ragged_kernel(
    const __nv_bfloat16* __restrict__ x,            // [max_rows, K]
    uint8_t* __restrict__ y,                         // [max_rows, K] FP8
    float* __restrict__ scale,                       // [num_k_blocks, max_rows]
    const int32_t* __restrict__ seqlens,             // [E]
    const int32_t* __restrict__ cu_seqlens,          // [E + 1]
    int E, int max_rows, int K, int num_k_blocks
) {
    // Rows the original one-block-per-row grid could reach: blockIdx.x in
    // [0, max_rows), minus every row outside [cu_seqlens[0], cu_seqlens[E]).
    const int64_t cu_first = cu_seqlens[0];
    const int64_t cu_last = cu_seqlens[E];
    const int64_t row_begin = cu_first > 0 ? cu_first : 0;
    const int64_t row_end = cu_last < max_rows ? cu_last : (int64_t)max_rows;
    if (row_begin >= row_end) return;

    const int tid = threadIdx.x;
    const int lane = tid % 32;
    const int warp = tid / 32;
    constexpr int kWarps = RAGGED_QUANT_THREADS / 32;
    const int64_t grid = gridDim.x;

    // One chunk of up to RAGGED_QUANT_THREADS experts at a time: thread t
    // loads expert c0 + t's clipped span [r_begin, r_end) (relative to its
    // cu_seqlens start), and a block-wide inclusive scan of the span lengths
    // places that chunk's rows in the concatenation of all clipped spans.
    // Block b takes concatenation positions b, b + grid, b + 2 * grid, ...
    __shared__ int64_t s_incl[RAGGED_QUANT_THREADS];
    __shared__ int64_t s_rbegin[RAGGED_QUANT_THREADS];
    __shared__ int64_t s_start[RAGGED_QUANT_THREADS];
    __shared__ int64_t s_warp_sum[kWarps];

    int64_t base = 0;  // concatenation position of this chunk's first row
    for (int c0 = 0; c0 < E; c0 += RAGGED_QUANT_THREADS) {
        const int e = c0 + tid;
        int64_t len = 0, r_begin = 0, start = 0;
        if (e < E) {
            start = cu_seqlens[e];
            const int64_t valid_e = seqlens[e];
            const int64_t span_e = (valid_e + RAGGED_ROW_ALIGN - 1) /
                                   RAGGED_ROW_ALIGN * RAGGED_ROW_ALIGN;
            r_begin = row_begin > start ? row_begin - start : 0;
            const int64_t r_end = span_e < row_end - start ? span_e : row_end - start;
            len = r_end > r_begin ? r_end - r_begin : 0;
        }
        int64_t incl = len;
        #pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
            const int64_t up = __shfl_up_sync(0xffffffff, incl, off);
            if (lane >= off) incl += up;
        }
        if (lane == 31) s_warp_sum[warp] = incl;
        __syncthreads();
        for (int w = 0; w < warp; ++w) incl += s_warp_sum[w];
        s_incl[tid] = incl;
        s_rbegin[tid] = r_begin;
        s_start[tid] = start;
        __syncthreads();

        const int n = E - c0 < RAGGED_QUANT_THREADS ? E - c0 : RAGGED_QUANT_THREADS;
        const int64_t chunk_rows = s_incl[n - 1];
        int64_t v = ((int64_t)blockIdx.x - base) % grid;
        if (v < 0) v += grid;
        for (; v < chunk_rows; v += grid) {
            // Chunk-local expert j holding position v: first j with s_incl[j] > v.
            int lo_j = 0, hi_j = n - 1;
            while (lo_j < hi_j) {
                const int mid = (lo_j + hi_j) / 2;
                if (s_incl[mid] > v) hi_j = mid;
                else lo_j = mid + 1;
            }
            const int64_t excl = lo_j ? s_incl[lo_j - 1] : 0;
            const int64_t row = s_start[lo_j] + s_rbegin[lo_j] + (v - excl);
            // Rightmost expert whose span starts at or before this row; an
            // empty expert shares its start with the next one, which then owns
            // the row. Identical search to the one-block-per-row kernel.
            int lo = 0;
            int hi = E - 1;
            while (lo < hi) {
                const int mid = (lo + hi + 1) / 2;
                if (cu_seqlens[mid] <= row) lo = mid;
                else hi = mid - 1;
            }
            if (lo != c0 + lo_j) continue;  // owned (and written) by expert lo
            const int64_t rel = row - cu_seqlens[lo];
            const int64_t valid_tokens = seqlens[lo];
            const int64_t span = (valid_tokens + RAGGED_ROW_ALIGN - 1) /
                                 RAGGED_ROW_ALIGN * RAGGED_ROW_ALIGN;
            if (rel >= span) continue;
            act_quant_ragged_row<kVec>(x, y, scale, row, rel < valid_tokens,
                                       max_rows, K, num_k_blocks);
        }
        base += chunk_rows;
        __syncthreads();  // s_* are rewritten by the next chunk
    }
}

// act_quant_ragged grid: one full wave of resident blocks, or one block per
// 16 rows of capacity when that is larger. Decode with E local experts fills
// ~E x 64 span rows (2048 at E=32) of a capacity of at most 16 waves (32768
// rows in the DeepEP layout), so it runs as a single wave with about one row
// per block and no early-exit blocks. A large, densely filled buffer (grouped
// prefill: 147200 rows, up to 131072 live) keeps enough blocks for the
// hardware scheduler to balance the rows; one block per row would be 16x
// more. The grid depends only on max_rows and the device, never on the live
// counts, so a captured graph replays any seqlens correctly.
template <bool kVec>
static int act_quant_ragged_grid(const torch::Device& device, int64_t max_rows) {
    static const int blocks_per_sm = [] {
        int n = 0;
        C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &n, act_quant_ragged_kernel<kVec>, RAGGED_QUANT_THREADS, 0));
        TORCH_CHECK(n > 0, "act_quant_ragged_kernel cannot be resident on this device");
        return n;
    }();
    const int64_t sms = at::cuda::getDeviceProperties(device.index())->multiProcessorCount;
    const int64_t wave = sms * blocks_per_sm;
    const int64_t per_capacity = (max_rows + RAGGED_ROWS_PER_BLOCK - 1) / RAGGED_ROWS_PER_BLOCK;
    return static_cast<int>(std::min<int64_t>(std::max<int64_t>(wave, per_capacity), max_rows));
}

// ============================================================================
// Kernel 2: silu_mul_3d — 3D-sparse SiLU×gate
// Grid: (E,)   Block: 256 threads
// ============================================================================
__global__ void silu_mul_3d_kernel(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    __nv_bfloat16* __restrict__ output,
    const int32_t* __restrict__ tokens_per_expert,
    int mtp, int N
) {
    const int expert = blockIdx.x;
    const int valid_tokens = tokens_per_expert[expert];
    if (valid_tokens == 0) return;

    const int tid = threadIdx.x;
    const int64_t expert_offset = (int64_t)expert * mtp * N;

    for (int m = 0; m < valid_tokens; m++) {
        const int64_t row_offset = expert_offset + (int64_t)m * N;

        for (int i = tid * 8; i < N; i += blockDim.x * 8) {
            if (i + 7 < N) {
                uint4 g_vec = *reinterpret_cast<const uint4*>(&gate[row_offset + i]);
                uint4 u_vec = *reinterpret_cast<const uint4*>(&up[row_offset + i]);

                __nv_bfloat16* g_arr = reinterpret_cast<__nv_bfloat16*>(&g_vec);
                __nv_bfloat16* u_arr = reinterpret_cast<__nv_bfloat16*>(&u_vec);

                uint4 out_vec;
                __nv_bfloat16* o_arr = reinterpret_cast<__nv_bfloat16*>(&out_vec);

                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    float g = __bfloat162float(g_arr[j]);
                    float u = __bfloat162float(u_arr[j]);
                    float silu_g = g / (1.0f + expf(-g));
                    o_arr[j] = __float2bfloat16(silu_g * u);
                }

                *reinterpret_cast<uint4*>(&output[row_offset + i]) = out_vec;
            } else {
                for (int j = 0; j < 8 && (i + j) < N; j++) {
                    float g = __bfloat162float(gate[row_offset + i + j]);
                    float u = __bfloat162float(up[row_offset + i + j]);
                    float silu_g = g / (1.0f + expf(-g));
                    output[row_offset + i + j] = __float2bfloat16(silu_g * u);
                }
            }
        }
    }
}

// ============================================================================
// Kernel 3: fused_silu_quant_3d — SiLU×gate + FP8 quantization
// Grid: (E,)   Block: 128 threads (4 warps)
// Fuses silu_mul + act_quant: eliminates intermediate BF16 buffer.
// ============================================================================
__global__ void fused_silu_quant_3d_kernel(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    uint8_t* __restrict__ y,
    float* __restrict__ scale,
    const int32_t* __restrict__ tokens_per_expert,
    int mtp, int N, int num_n_blocks
) {
    const int expert = blockIdx.x;
    const int valid_tokens = tokens_per_expert[expert];
    if (valid_tokens == 0) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int num_warps = blockDim.x / 32;

    const int64_t expert_offset = (int64_t)expert * mtp * N;
    float* scale_base = scale + (int64_t)expert * mtp * num_n_blocks;

    for (int m = 0; m < valid_tokens; m++) {
        const int64_t row_offset = expert_offset + (int64_t)m * N;
        float* scale_row = scale_base + (int64_t)m * num_n_blocks;

        for (int kb = warp_id; kb < num_n_blocks; kb += num_warps) {
            int col_base = kb * BLOCK_SIZE_QUANT;

            float vals[4];
            float local_max = 0.0f;

            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int col = col_base + lane_id * 4 + i;
                if (col < N) {
                    float g = __bfloat162float(gate[row_offset + col]);
                    float u = __bfloat162float(up[row_offset + col]);
                    float silu_g = g / (1.0f + expf(-g));
                    vals[i] = silu_g * u;
                } else {
                    vals[i] = 0.0f;
                }
                local_max = fmaxf(local_max, fabsf(vals[i]));
            }

            #pragma unroll
            for (int offset = 16; offset >= 1; offset >>= 1) {
                float other = __shfl_xor_sync(0xffffffff, local_max, offset);
                local_max = fmaxf(local_max, other);
            }

            float s = fmaxf(local_max, QUANT_EPS) / FP8_MAX_VAL;
            float inv_s = 1.0f / s;

            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int col = col_base + lane_id * 4 + i;
                if (col < N) {
                    float scaled = vals[i] * inv_s;
                    scaled = fmaxf(fminf(scaled, FP8_MAX_VAL), -FP8_MAX_VAL);
                    y[row_offset + col] = __nv_cvt_float_to_fp8(scaled, __NV_SATFINITE, __NV_E4M3);
                }
            }

            if (lane_id == 0) {
                scale_row[kb] = s;
            }
        }
    }
}

// ============================================================================
// C++ Wrappers
// ============================================================================

std::tuple<torch::Tensor, torch::Tensor> act_quant_3d(
    torch::Tensor x,
    torch::Tensor tokens_per_expert
) {
    TORCH_CHECK(x.dim() == 3, "x must be 3D [E, mtp, K]");
    TORCH_CHECK(x.dtype() == torch::kBFloat16, "x must be BF16");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

    int E = x.size(0);
    int mtp = x.size(1);
    int K = x.size(2);
    int num_k_blocks = (K + BLOCK_SIZE_QUANT - 1) / BLOCK_SIZE_QUANT;

    auto y = torch::empty({E, mtp, K}, torch::dtype(torch::kUInt8).device(x.device()));
    auto scale = torch::empty({E, mtp, num_k_blocks}, torch::dtype(torch::kFloat32).device(x.device()));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    // 2D grid: one CTA per (expert, token). Padded tokens early-return
    // inside the kernel so no compute wasted on expert_count < mtp.
    dim3 grid(E, mtp);
    act_quant_3d_kernel<<<grid, 128, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        y.data_ptr<uint8_t>(),
        scale.data_ptr<float>(),
        tokens_per_expert.data_ptr<int32_t>(),
        mtp, K, num_k_blocks);

    return std::make_tuple(y, scale);
}

static void check_ragged_cuda_tensor(
    const torch::Tensor& t, const torch::Device& device, const char* name
) {
    TORCH_CHECK(t.defined() && t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.device() == device, name, " must be on ", device, ", got ", t.device());
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

// Quantizes caller-owned compact ragged rows in place of y/scale; no
// allocations. Device seqlens/cu_seqlens values are never read on the host.
std::tuple<torch::Tensor, torch::Tensor> act_quant_ragged(
    torch::Tensor x,
    torch::Tensor seqlens,
    torch::Tensor cu_seqlens,
    torch::Tensor y,
    torch::Tensor scale
) {
    constexpr int64_t kInt32Max = 2147483647;
    TORCH_CHECK(x.defined() && x.is_cuda(), "x must be a CUDA tensor");
    const auto device = x.device();
    check_ragged_cuda_tensor(x, device, "x");
    check_ragged_cuda_tensor(seqlens, device, "seqlens");
    check_ragged_cuda_tensor(cu_seqlens, device, "cu_seqlens");
    check_ragged_cuda_tensor(y, device, "y");
    check_ragged_cuda_tensor(scale, device, "scale");

    TORCH_CHECK(x.dim() == 2 && x.scalar_type() == torch::kBFloat16,
                "x must be BF16 [max_rows, K]");
    const int64_t max_rows = x.size(0);
    const int64_t K = x.size(1);
    TORCH_CHECK(K > 0 && K <= kInt32Max && max_rows <= kInt32Max,
                "x shape out of range: [", max_rows, ", ", K, "]");
    const int64_t num_k_blocks = (K + BLOCK_SIZE_QUANT - 1) / BLOCK_SIZE_QUANT;

    TORCH_CHECK(seqlens.dim() == 1 && seqlens.scalar_type() == torch::kInt32,
                "seqlens must be int32 [E]");
    const int64_t E = seqlens.size(0);
    TORCH_CHECK(E > 0 && E < kInt32Max, "seqlens must be non-empty");
    TORCH_CHECK(cu_seqlens.dim() == 1 && cu_seqlens.scalar_type() == torch::kInt32 &&
                cu_seqlens.size(0) == E + 1, "cu_seqlens must be int32 [E + 1]");
    TORCH_CHECK(y.dim() == 2 &&
                (y.scalar_type() == torch::kUInt8 ||
                 y.scalar_type() == c10::ScalarType::Float8_e4m3fn) &&
                y.size(0) == max_rows && y.size(1) == K,
                "y must be uint8/float8_e4m3fn [max_rows, K]");
    TORCH_CHECK(scale.dim() == 2 && scale.scalar_type() == torch::kFloat32 &&
                scale.size(0) == num_k_blocks && scale.size(1) == max_rows,
                "scale must be float32 [ceil(K/128), max_rows]");

    if (max_rows == 0) return std::make_tuple(y, scale);

    const c10::cuda::CUDAGuard device_guard(device);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const bool vec = K % 4 == 0 &&
                     reinterpret_cast<uintptr_t>(x.data_ptr()) % 8 == 0 &&
                     reinterpret_cast<uintptr_t>(y.data_ptr()) % 4 == 0;
    auto launch = [&](auto kernel, int grid) {
        kernel<<<grid, RAGGED_QUANT_THREADS, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
            reinterpret_cast<uint8_t*>(y.data_ptr()),
            scale.data_ptr<float>(),
            seqlens.data_ptr<int32_t>(),
            cu_seqlens.data_ptr<int32_t>(),
            static_cast<int>(E), static_cast<int>(max_rows),
            static_cast<int>(K), static_cast<int>(num_k_blocks));
    };
    if (vec) {
        launch(act_quant_ragged_kernel<true>, act_quant_ragged_grid<true>(device, max_rows));
    } else {
        launch(act_quant_ragged_kernel<false>, act_quant_ragged_grid<false>(device, max_rows));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return std::make_tuple(y, scale);
}

torch::Tensor silu_mul_3d(
    torch::Tensor gate,
    torch::Tensor up,
    torch::Tensor tokens_per_expert
) {
    TORCH_CHECK(gate.dim() == 3 && up.dim() == 3, "Inputs must be 3D");
    TORCH_CHECK(gate.sizes() == up.sizes(), "Shape mismatch");
    TORCH_CHECK(gate.dtype() == torch::kBFloat16, "Must be BF16");

    int E = gate.size(0);
    int mtp = gate.size(1);
    int N = gate.size(2);

    auto output = torch::empty_like(gate);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    silu_mul_3d_kernel<<<E, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(gate.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(up.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        tokens_per_expert.data_ptr<int32_t>(),
        mtp, N);

    return output;
}

std::tuple<torch::Tensor, torch::Tensor> fused_silu_quant_3d(
    torch::Tensor gate,
    torch::Tensor up,
    torch::Tensor tokens_per_expert
) {
    TORCH_CHECK(gate.dim() == 3 && up.dim() == 3, "Inputs must be 3D");
    TORCH_CHECK(gate.sizes() == up.sizes(), "Shape mismatch");

    int E = gate.size(0);
    int mtp = gate.size(1);
    int N = gate.size(2);
    int num_n_blocks = (N + BLOCK_SIZE_QUANT - 1) / BLOCK_SIZE_QUANT;

    auto y = torch::empty({E, mtp, N}, torch::dtype(torch::kUInt8).device(gate.device()));
    auto scale = torch::empty({E, mtp, num_n_blocks}, torch::dtype(torch::kFloat32).device(gate.device()));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    fused_silu_quant_3d_kernel<<<E, 128, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(gate.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(up.data_ptr()),
        y.data_ptr<uint8_t>(),
        scale.data_ptr<float>(),
        tokens_per_expert.data_ptr<int32_t>(),
        mtp, N, num_n_blocks);

    return std::make_tuple(y, scale);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("act_quant_3d", &act_quant_3d,
          "FP8 blockwise quantization on 3D [E, mtp, K] layout");
    m.def("act_quant_ragged", &act_quant_ragged,
          "FP8 blockwise quantization on compact ragged rows (caller-owned outputs)");
    m.def("silu_mul_3d", &silu_mul_3d,
          "SiLU(gate) * up on 3D [E, mtp, N] layout");
    m.def("fused_silu_quant_3d", &fused_silu_quant_3d,
          "Fused SiLU(gate) * up + FP8 quantization on 3D layout");
}
