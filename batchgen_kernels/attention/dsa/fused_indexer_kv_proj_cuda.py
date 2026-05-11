"""Fused Indexer KV Projection — CUDA WGMMA (H20 SM90a)

Version: v3 — Single-WG TMA-both + separate act_quant
Hypothesis: v2 2-WG producer/math pipeline has too much overhead for tiny M.
            Single-WG with TMA for both A (pre-quantized) and B should be faster.
Result: POSITIVE — 1.47-1.62× over Torch, 1.12-2.09× over Triton (42-47µs)

Pipeline: hidden_states [B, 6144] BF16 → act_quant [B, 6144] FP8 + scale
          → WGMMA kernel (TMA both A+B) → [B, 128] BF16
          → RMSNorm [B, 128] BF16

Dimensions (GLM-5 indexer):
  M = B ≤ 64 (decode batch, padded to BLOCK_M=64)
  K = 6144 (hidden_size)
  N = 128 (index_head_dim)
  num_k_blocks = 48 (6144 / 128)

Architecture: 1 warpgroup (128 threads) per CTA
  TMA loads both A (act FP8) and B (weight FP8) with 128B swizzle
  WGMMA m64n32k32 FP8×FP8→FP32 + post-scale from quant scales
  4-stage pipeline for K dimension

Grid: (cdiv(B, 64), cdiv(N, 32)) = typically (1, 4)
"""

import os
import torch
from torch.utils.cpp_extension import load_inline

_module_cache = {}

_BLOCK_M = 64
_BLOCK_N = 32
_BLOCK_K = 128
_NUM_STAGES = 4

CUDA_SOURCE = r'''
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

// ============================================================================
// Constants
// ============================================================================
constexpr int BLOCK_M = 64;
constexpr int BLOCK_N = 32;
constexpr int BLOCK_K = 128;
constexpr int WARP_SIZE = 32;
constexpr int THREADS = 128;  // 1 warpgroup
#ifndef NUM_STAGES
#define NUM_STAGES 4
#endif

constexpr int SCALE_BLOCK_N = 32;
constexpr float FP8_MAX_VAL = 448.0f;

// SMEM per stage: A tile + B tile
constexpr int TILE_BYTES_A = BLOCK_M * BLOCK_K;      // 8192
constexpr int TILE_BYTES_B = BLOCK_K * BLOCK_N;      // 4096
constexpr int STAGE_BYTES = TILE_BYTES_A + TILE_BYTES_B;  // 12288
constexpr int PIPELINE_BYTES = NUM_STAGES * STAGE_BYTES;
// Barriers: full + empty
constexpr int BARRIER_BYTES = 2 * NUM_STAGES * sizeof(uint64_t);
// Epilogue: BF16 output tile
constexpr int EPILOGUE_BYTES = BLOCK_M * BLOCK_N * sizeof(__nv_bfloat16);
// Weight scale + act scale
constexpr int MAX_K_BLOCKS = 64;
constexpr int WSCALE_BYTES = MAX_K_BLOCKS * sizeof(float);
constexpr int ASCALE_BYTES = BLOCK_M * sizeof(float);

constexpr int TOTAL_SMEM_BYTES = PIPELINE_BYTES + BARRIER_BYTES +
    EPILOGUE_BYTES + WSCALE_BYTES + ASCALE_BYTES;

constexpr int WGMMA_K = 32;
constexpr int TILES_K = BLOCK_K / WGMMA_K;  // 4
constexpr int WGMMA_NUM_ACCUM = 16;

// ============================================================================
// GMMA Descriptor
// ============================================================================
union GmmaDescriptor {
    uint64_t desc_;
    struct {
        uint16_t start_address_: 14, : 2;
        uint16_t leading_byte_offset_: 14, : 2;
        uint16_t stride_byte_offset_: 14, : 2;
        uint8_t : 1, base_offset_: 3, : 4;
        uint8_t : 6, layout_type_: 2;
    } bitfield;
};

template <class T>
__device__ __forceinline__ GmmaDescriptor make_smem_desc(T* smem_ptr) {
    GmmaDescriptor desc;
    desc.desc_ = 0;
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    desc.bitfield.start_address_ = addr >> 4;
    desc.bitfield.layout_type_ = 1;  // 128B swizzle
    desc.bitfield.leading_byte_offset_ = 0;
    desc.bitfield.stride_byte_offset_ = 1024 >> 4;
    desc.bitfield.base_offset_ = 0;
    return desc;
}

// ============================================================================
// WGMMA m64n32k32 FP8×FP8→FP32 SS
// ============================================================================
__device__ __forceinline__ void wgmma_m64n32k32_fp8_ss(
    uint64_t const& desc_a, uint64_t const& desc_b,
    float& d00, float& d01, float& d02, float& d03,
    float& d04, float& d05, float& d06, float& d07,
    float& d08, float& d09, float& d10, float& d11,
    float& d12, float& d13, float& d14, float& d15,
    int scale_D
) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %18, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n32k32.f32.e4m3.e4m3 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        " %8,  %9,  %10, %11, %12, %13, %14, %15}, "
        " %16, %17, p, 1, 1;\n"
        "}\n"
        : "+f"(d00), "+f"(d01), "+f"(d02), "+f"(d03),
          "+f"(d04), "+f"(d05), "+f"(d06), "+f"(d07),
          "+f"(d08), "+f"(d09), "+f"(d10), "+f"(d11),
          "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15)
        : "l"(desc_a), "l"(desc_b), "r"(scale_D));
}

__device__ __forceinline__ void wgmma_fp8_ss(
    uint64_t const& desc_a, uint64_t const& desc_b,
    float* accum, int scale_D
) {
    wgmma_m64n32k32_fp8_ss(
        desc_a, desc_b,
        accum[0],  accum[1],  accum[2],  accum[3],
        accum[4],  accum[5],  accum[6],  accum[7],
        accum[8],  accum[9],  accum[10], accum[11],
        accum[12], accum[13], accum[14], accum[15],
        scale_D);
}

// ============================================================================
// Warpgroup helpers
// ============================================================================
__device__ __forceinline__ void warpgroup_arrive() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void warpgroup_commit_batch() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}
template <int N_>
__device__ __forceinline__ void warpgroup_wait() {
    static_assert(N_ >= 0 && N_ <= 7);
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N_) : "memory");
}
__device__ __forceinline__ void warpgroup_fence_operand(float& reg) {
    asm volatile("" : "+f"(reg) :: "memory");
}

// ============================================================================
// Barrier helpers
// ============================================================================
__device__ __forceinline__ void mbarrier_init(uint64_t* bar, int count) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(addr), "r"(count));
}
__device__ __forceinline__ void mbarrier_arrive(uint64_t* bar) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.arrive.shared.b64 _, [%0];" :: "r"(addr) : "memory");
}
__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t* bar, int tx_count) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile(
        "mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;"
        :: "r"(addr), "r"(tx_count) : "memory");
}
__device__ __forceinline__ void mbarrier_wait_parity(uint64_t* bar, int phase) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile(
        "{\n"
        ".reg .pred P;\n"
        "WAIT:\n"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
        "@!P bra WAIT;\n"
        "}\n"
        :: "r"(addr), "r"(phase) : "memory");
}

// ============================================================================
// TMA helpers
// ============================================================================
__device__ __forceinline__ void tma_load_2d(
    const CUtensorMap* desc, uint64_t* bar, void* smem_ptr,
    int coord_0, int coord_1
) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    uint32_t bar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%2, %3}], [%4];"
        :: "r"(smem_addr), "l"(desc), "r"(coord_0), "r"(coord_1), "r"(bar_addr)
        : "memory");
}

// ============================================================================
// SMEM helpers
// ============================================================================
__device__ __forceinline__ void st_shared(float* addr, float val) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(addr));
    asm volatile("st.shared.f32 [%0], %1;" :: "r"(smem_addr), "f"(val) : "memory");
}
__device__ __forceinline__ float ld_shared(const float* addr) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(addr));
    float val;
    asm volatile("ld.shared.f32 %0, [%1];" : "=f"(val) : "r"(smem_addr));
    return val;
}

// ============================================================================
// TMA descriptor creation
// ============================================================================
static auto get_cuTensorMapEncodeTiled() {
    void* handle;
    cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &handle, cudaEnableDefault);
    typedef CUresult (*FuncPtr)(
        CUtensorMap*, CUtensorMapDataType, cuuint32_t,
        void*, const cuuint64_t*, const cuuint64_t*,
        const cuuint32_t*, const cuuint32_t*,
        CUtensorMapInterleave, CUtensorMapSwizzle,
        CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
    return reinterpret_cast<FuncPtr>(handle);
}

CUtensorMap make_2d_tma_desc_fp8(
    const uint8_t* global_address, int outer_dim, int inner_dim,
    int box_outer, int box_inner,
    decltype(get_cuTensorMapEncodeTiled()) encode_func
) {
    CUtensorMap tensor_map;
    uint64_t gmem_dim[2] = {(uint64_t)inner_dim, (uint64_t)outer_dim};
    uint64_t global_stride[1] = {(uint64_t)(inner_dim * 1)};
    uint32_t smem_dim[2] = {(uint32_t)box_inner, (uint32_t)box_outer};
    uint32_t elem_strides[2] = {1, 1};
    auto result = encode_func(
        &tensor_map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2,
        const_cast<void*>((const void*)global_address),
        gmem_dim, global_stride, smem_dim, elem_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (result != CUDA_SUCCESS)
        throw std::runtime_error("cuTensorMapEncodeTiled failed");
    return tensor_map;
}

// ============================================================================
// Activation quantization kernel (BF16 → FP8 + per-row scale)
// ============================================================================
__global__ void act_quant_kernel(
    const __nv_bfloat16* __restrict__ X,  // [B, K] BF16
    __nv_fp8_e4m3* __restrict__ X_fp8,    // [B, K] FP8
    float* __restrict__ X_scale,          // [B] FP32
    const int* __restrict__ num_valid_tokens,  // optional device scalar
    int B, int K, bool has_valid_tokens
) {
    const int row = blockIdx.x;
    if (row >= B) return;

    const int tid = threadIdx.x;
    __nv_fp8_e4m3* out_ptr = X_fp8 + row * K;
    if (has_valid_tokens && row >= num_valid_tokens[0]) {
        for (int i = tid; i < K; i += blockDim.x) {
            reinterpret_cast<uint8_t*>(out_ptr)[i] =
                __nv_cvt_float_to_fp8(0.0f, __NV_SATFINITE, __NV_E4M3);
        }
        if (tid == 0) X_scale[row] = 1e-12f;
        return;
    }

    const __nv_bfloat16* row_ptr = X + row * K;

    // Find absmax across row
    float local_max = 0.0f;
    for (int i = tid; i < K; i += blockDim.x) {
        float v = __bfloat162float(row_ptr[i]);
        local_max = fmaxf(local_max, fabsf(v));
    }

    // Warp reduce
    for (int offset = 16; offset > 0; offset >>= 1)
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, offset));

    // Cross-warp reduce
    __shared__ float warp_max[8];
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    if (lane_id == 0) warp_max[warp_id] = local_max;
    __syncthreads();

    float global_max = 0.0f;
    if (tid < blockDim.x / 32) global_max = warp_max[tid];
    for (int offset = 4; offset > 0; offset >>= 1)
        global_max = fmaxf(global_max, __shfl_xor_sync(0xff, global_max, offset));
    global_max = __shfl_sync(0xffffffff, global_max, 0);

    __shared__ float shared_scale;
    if (tid == 0) {
        float scale = fmaxf(global_max, 1e-12f) / 448.0f;
        shared_scale = scale;
        X_scale[row] = scale;
    }
    __syncthreads();
    float inv_scale = 1.0f / shared_scale;

    // Quantize
    for (int i = tid; i < K; i += blockDim.x) {
        float v = __bfloat162float(row_ptr[i]) * inv_scale;
        v = fmaxf(fminf(v, 448.0f), -448.0f);
        reinterpret_cast<uint8_t*>(out_ptr)[i] =
            __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
    }
}

__device__ __forceinline__ int clamp_valid_m(
    const int* __restrict__ num_valid_m,
    bool has_valid_m,
    int B
) {
    int valid_m = B;
    if (has_valid_m) valid_m = num_valid_m[0];
    if (valid_m < 0) valid_m = 0;
    if (valid_m > B) valid_m = B;
    return valid_m;
}

__device__ __forceinline__ void zero_output_tile(
    __nv_bfloat16* __restrict__ OUT,
    int B,
    int N,
    int m_start,
    int n_start,
    int tid
) {
    const int total_elems = BLOCK_M * BLOCK_N;
    const __nv_bfloat16 zero = __float2bfloat16(0.0f);
    for (int idx = tid; idx < total_elems; idx += THREADS) {
        const int row = idx / BLOCK_N;
        const int col = idx % BLOCK_N;
        const int m_global = m_start + row;
        const int n_global = n_start + col;
        if (m_global < B && n_global < N) {
            OUT[m_global * N + n_global] = zero;
        }
    }
}

// ============================================================================
// Single-WG WGMMA Kernel — TMA both A and B
// ============================================================================
__global__ void __launch_bounds__(THREADS, 1)
indexer_kv_proj_kernel(
    const CUtensorMap* __restrict__ a_desc,    // Act TMA descriptor
    const CUtensorMap* __restrict__ w_desc,    // Weight TMA descriptor
    const float* __restrict__ w_scale,         // [N/32, num_k_blocks] FP32
    const float* __restrict__ a_scale,         // [B] FP32 per-row act scale
    __nv_bfloat16* __restrict__ OUT,           // [B, N] BF16 output
    const int* __restrict__ num_valid_m,       // optional [1] int32 device scalar
    bool has_valid_m,
    int B, int K, int N
) {
    const int tid = threadIdx.x;
    const int m_tile = blockIdx.x;
    const int n_tile = blockIdx.y;
    const int m_start = m_tile * BLOCK_M;
    if (m_start >= B) return;
    const int n_start = n_tile * BLOCK_N;
    const int num_k_blocks = (K + BLOCK_K - 1) / BLOCK_K;
    const int valid_m = clamp_valid_m(num_valid_m, has_valid_m, B);
    if (m_start >= valid_m) {
        zero_output_tile(OUT, B, N, m_start, n_start, tid);
        return;
    }

    // ── SMEM layout ──
    extern __shared__ __align__(128) char smem_buf[];
    uint64_t* full_barriers = reinterpret_cast<uint64_t*>(
        smem_buf + PIPELINE_BYTES);
    uint64_t* empty_barriers = full_barriers + NUM_STAGES;
    __nv_bfloat16* smem_out = reinterpret_cast<__nv_bfloat16*>(
        smem_buf + PIPELINE_BYTES + BARRIER_BYTES);
    float* smem_wscale = reinterpret_cast<float*>(
        smem_buf + PIPELINE_BYTES + BARRIER_BYTES + EPILOGUE_BYTES);
    float* smem_ascale = reinterpret_cast<float*>(
        smem_buf + PIPELINE_BYTES + BARRIER_BYTES + EPILOGUE_BYTES + WSCALE_BYTES);

    // ── Init barriers ──
    if (tid == 0) {
        for (int s = 0; s < NUM_STAGES; s++) {
            mbarrier_init(&full_barriers[s], 1);   // TMA arrivals only
            mbarrier_init(&empty_barriers[s], 1);   // math done
        }
    }
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    __syncthreads();

    if (tid == 0)
        asm volatile("prefetch.tensormap [%0];" :: "l"(a_desc) : "memory");

    // ── Pre-load scales ──
    const int n_scale_idx = n_start / SCALE_BLOCK_N;
    for (int i = tid; i < num_k_blocks; i += THREADS) {
        st_shared(&smem_wscale[i], w_scale[n_scale_idx * num_k_blocks + i]);
    }
    // Load per-row activation scales
    for (int i = tid; i < BLOCK_M; i += THREADS) {
        float s = (m_start + i < valid_m) ? a_scale[m_start + i] : 0.0f;
        st_shared(&smem_ascale[i], s);
    }
    __syncthreads();

    // ── Pipeline: issue first NUM_STAGES TMA loads ──
    int slot = 0;
    for (int s = 0; s < NUM_STAGES && s < num_k_blocks; s++) {
        if (tid == 0) {
            mbarrier_arrive_expect_tx(&full_barriers[s],
                                       TILE_BYTES_A + TILE_BYTES_B);
            uint8_t* cur_smem_a = reinterpret_cast<uint8_t*>(
                smem_buf + s * STAGE_BYTES);
            uint8_t* cur_smem_b = cur_smem_a + TILE_BYTES_A;
            tma_load_2d(a_desc, &full_barriers[s], cur_smem_a,
                       s * BLOCK_K, m_start);
            tma_load_2d(w_desc, &full_barriers[s], cur_smem_b,
                       s * BLOCK_K, n_start);
        }
    }

    const int warp_in_wg = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    float accum[WGMMA_NUM_ACCUM];
    float result[WGMMA_NUM_ACCUM];
    #pragma unroll
    for (int i = 0; i < WGMMA_NUM_ACCUM; i++) result[i] = 0.0f;

    const int m_row_in_tile_0 = warp_in_wg * 16 + (lane_id / 4);
    const int m_row_in_tile_1 = m_row_in_tile_0 + 8;
    const bool valid_0 = (m_start + m_row_in_tile_0 < valid_m);
    const bool valid_1 = (m_start + m_row_in_tile_1 < valid_m);

    slot = 0;
    int full_phase = 0;
    int prefetch_kb = NUM_STAGES;

    for (int kb = 0; kb < num_k_blocks; kb++) {
        float ws = ld_shared(&smem_wscale[kb]);

        mbarrier_wait_parity(&full_barriers[slot], full_phase);

        // Read activation scales for this tile's rows
        float xs0 = valid_0 ? ld_shared(&smem_ascale[m_row_in_tile_0]) : 0.0f;
        float xs1 = valid_1 ? ld_shared(&smem_ascale[m_row_in_tile_1]) : 0.0f;

        uint8_t* cur_smem_a = reinterpret_cast<uint8_t*>(
            smem_buf + slot * STAGE_BYTES);
        uint8_t* cur_smem_b = cur_smem_a + TILE_BYTES_A;

        #pragma unroll
        for (int i = 0; i < WGMMA_NUM_ACCUM; i++)
            warpgroup_fence_operand(accum[i]);

        warpgroup_arrive();

        #pragma unroll
        for (int t = 0; t < TILES_K; t++) {
            GmmaDescriptor da = make_smem_desc(cur_smem_a + t * WGMMA_K);
            GmmaDescriptor db = make_smem_desc(cur_smem_b + t * WGMMA_K);
            wgmma_fp8_ss(da.desc_, db.desc_, accum, (t == 0) ? 0 : 1);
        }

        warpgroup_commit_batch();

        #pragma unroll
        for (int i = 0; i < WGMMA_NUM_ACCUM; i++)
            warpgroup_fence_operand(accum[i]);

        warpgroup_wait<0>();

        // Prefetch next stage
        if (prefetch_kb < num_k_blocks && tid == 0) {
            mbarrier_arrive_expect_tx(&full_barriers[slot],
                                       TILE_BYTES_A + TILE_BYTES_B);
            uint8_t* next_smem_a = reinterpret_cast<uint8_t*>(
                smem_buf + slot * STAGE_BYTES);
            uint8_t* next_smem_b = next_smem_a + TILE_BYTES_A;
            tma_load_2d(a_desc, &full_barriers[slot], next_smem_a,
                       prefetch_kb * BLOCK_K, m_start);
            tma_load_2d(w_desc, &full_barriers[slot], next_smem_b,
                       prefetch_kb * BLOCK_K, n_start);
            prefetch_kb++;
        }

        // Scale multiply and accumulate
        float s0 = xs0 * ws;
        float s1 = xs1 * ws;

        #pragma unroll
        for (int i = 0; i < WGMMA_NUM_ACCUM; i++) {
            int m_half = (i % 4) / 2;
            float scale = (m_half == 0) ? s0 : s1;
            result[i] += accum[i] * scale;
        }

        slot++;
        if (slot == NUM_STAGES) { slot = 0; full_phase ^= 1; }
    }

    // ════════════════════════════════════════════════════════════════
    // Epilogue: STSM → coalesced global store
    // ════════════════════════════════════════════════════════════════
    #pragma unroll
    for (int i = 0; i < WGMMA_NUM_ACCUM / 8; i++) {
        uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(
            smem_out + (warp_in_wg * 16 + lane_id % 16) * BLOCK_N
                     + i * 16 + 8 * (lane_id / 16)));
        uint32_t r0, r1, r2, r3;
        asm volatile("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(r0) : "f"(result[i*8+0]), "f"(result[i*8+1]));
        asm volatile("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(r1) : "f"(result[i*8+2]), "f"(result[i*8+3]));
        asm volatile("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(r2) : "f"(result[i*8+4]), "f"(result[i*8+5]));
        asm volatile("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(r3) : "f"(result[i*8+6]), "f"(result[i*8+7]));
        asm volatile(
            "stmatrix.sync.aligned.x4.m8n8.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(smem_addr), "r"(r0), "r"(r1), "r"(r2), "r"(r3)
            : "memory");
    }
    __syncthreads();

    // Coalesced global store
    const int bf16_2_per_row = BLOCK_N / 2;
    const int total_bf16_2 = BLOCK_M * bf16_2_per_row;
    for (int idx = tid; idx < total_bf16_2; idx += THREADS) {
        const int row = idx / bf16_2_per_row;
        const int col2 = idx % bf16_2_per_row;
        const int n_global = n_start + col2 * 2;
        const int m_global = m_start + row;

        if (m_global < B && n_global + 1 < N) {
            if (m_global < valid_m) {
                __nv_bfloat162 val = *reinterpret_cast<__nv_bfloat162*>(
                    &smem_out[row * BLOCK_N + col2 * 2]);
                *reinterpret_cast<__nv_bfloat162*>(
                    &OUT[m_global * N + n_global]) = val;
            } else {
                OUT[m_global * N + n_global] = __float2bfloat16(0.0f);
                OUT[m_global * N + n_global + 1] = __float2bfloat16(0.0f);
            }
        } else if (m_global < B && n_global < N) {
            OUT[m_global * N + n_global] = (m_global < valid_m)
                ? smem_out[row * BLOCK_N + col2 * 2]
                : __float2bfloat16(0.0f);
        }
    }
}

// ============================================================================
// RMSNorm kernel — tiny [B, N=128] elementwise
// ============================================================================
__global__ void rmsnorm_kernel(
    const __nv_bfloat16* __restrict__ X,
    const __nv_bfloat16* __restrict__ gamma,
    __nv_bfloat16* __restrict__ OUT,
    int B, int N, float eps
) {
    const int row = blockIdx.x;
    if (row >= B) return;

    const int tid = threadIdx.x;
    float val = 0.0f;
    if (tid < N) val = __bfloat162float(X[row * N + tid]);

    float sq = val * val;
    for (int offset = 16; offset > 0; offset >>= 1)
        sq += __shfl_xor_sync(0xffffffff, sq, offset);

    __shared__ float warp_sums[4];
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    if (lane_id == 0) warp_sums[warp_id] = sq;
    __syncthreads();

    float total_sq = 0.0f;
    if (tid < 4) total_sq = warp_sums[tid];
    for (int offset = 2; offset > 0; offset >>= 1)
        total_sq += __shfl_xor_sync(0xf, total_sq, offset);
    total_sq = __shfl_sync(0xffffffff, total_sq, 0);

    __shared__ float shared_rrms;
    if (tid == 0) {
        float mean_sq = total_sq / (float)N;
        shared_rrms = rsqrtf(mean_sq + eps);
    }
    __syncthreads();
    float rrms = shared_rrms;

    if (tid < N) {
        float g = __bfloat162float(gamma[tid]);
        OUT[row * N + tid] = __float2bfloat16(val * rrms * g);
    }
}

// ============================================================================
// Host functions
// ============================================================================
// Pinned host staging buffer for async TMA descriptor copies (128 bytes)
static CUtensorMap* pinned_tma_stage = nullptr;

static CUtensorMap* get_pinned_tma_stage() {
    if (!pinned_tma_stage) {
        cudaHostAlloc(&pinned_tma_stage, sizeof(CUtensorMap), cudaHostAllocDefault);
    }
    return pinned_tma_stage;
}

torch::Tensor create_tma_desc(
    torch::Tensor data,  // [outer, inner] FP8
    int outer_dim, int inner_dim,
    int box_outer, int box_inner
) {
    at::cuda::CUDAGuard device_guard{data.device()};
    static auto encode_func = get_cuTensorMapEncodeTiled();
    CUtensorMap* stage = get_pinned_tma_stage();
    *stage = make_2d_tma_desc_fp8(
        reinterpret_cast<const uint8_t*>(data.data_ptr()),
        outer_dim, inner_dim, box_outer, box_inner, encode_func);
    int desc_size = sizeof(CUtensorMap);
    auto desc_dev = torch::empty({desc_size},
        torch::dtype(torch::kUInt8).device(data.device()));
    cudaMemcpyAsync(desc_dev.data_ptr(), stage, desc_size,
                     cudaMemcpyHostToDevice, at::cuda::getCurrentCUDAStream());
    return desc_dev;
}

void run_act_quant(
    torch::Tensor X,         // [B, K] BF16
    torch::Tensor X_fp8,     // [B, K] FP8 (output)
    torch::Tensor X_scale    // [B] FP32 (output)
) {
    at::cuda::CUDAGuard device_guard{X.device()};
    int B = X.size(0);
    int K = X.size(1);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    act_quant_kernel<<<B, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(X.data_ptr()),
        reinterpret_cast<__nv_fp8_e4m3*>(X_fp8.data_ptr()),
        X_scale.data_ptr<float>(),
        nullptr,
        B, K, false);
}

void run_act_quant_valid(
    torch::Tensor X,         // [B, K] BF16
    torch::Tensor X_fp8,     // [B, K] FP8 (output)
    torch::Tensor X_scale,   // [B] FP32 (output)
    torch::Tensor num_valid_tokens  // [1] int32 device scalar
) {
    TORCH_CHECK(num_valid_tokens.dtype() == torch::kInt32, "num_valid_tokens must be int32");
    TORCH_CHECK(num_valid_tokens.numel() == 1, "num_valid_tokens must contain one element");
    at::cuda::CUDAGuard device_guard{X.device()};
    int B = X.size(0);
    int K = X.size(1);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    act_quant_kernel<<<B, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(X.data_ptr()),
        reinterpret_cast<__nv_fp8_e4m3*>(X_fp8.data_ptr()),
        X_scale.data_ptr<float>(),
        num_valid_tokens.data_ptr<int>(),
        B, K, true);
}

torch::Tensor indexer_kv_proj_forward(
    torch::Tensor a_tma_desc,
    torch::Tensor w_tma_desc,
    torch::Tensor w_scale,
    torch::Tensor a_scale,
    torch::Tensor rmsnorm_weight,
    int B, int N, int K, float eps
) {
    at::cuda::CUDAGuard device_guard{a_tma_desc.device()};
    auto OUT_gemm = torch::empty({B, N}, torch::dtype(torch::kBFloat16).device(torch::kCUDA));

    const CUtensorMap* a_desc = reinterpret_cast<const CUtensorMap*>(a_tma_desc.data_ptr());
    const CUtensorMap* w_desc = reinterpret_cast<const CUtensorMap*>(w_tma_desc.data_ptr());
    int num_m_tiles = (B + BLOCK_M - 1) / BLOCK_M;
    int num_n_tiles = (N + BLOCK_N - 1) / BLOCK_N;
    dim3 grid(num_m_tiles, num_n_tiles);

    constexpr int smem_bytes = TOTAL_SMEM_BYTES;
    cudaFuncSetAttribute(indexer_kv_proj_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    indexer_kv_proj_kernel<<<grid, THREADS, smem_bytes, stream>>>(
        a_desc, w_desc,
        w_scale.data_ptr<float>(),
        a_scale.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(OUT_gemm.data_ptr()),
        nullptr,
        false,
        B, K, N);

    // RMSNorm
    auto OUT = torch::empty({B, N}, torch::dtype(torch::kBFloat16).device(torch::kCUDA));
    rmsnorm_kernel<<<B, 128, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(OUT_gemm.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(rmsnorm_weight.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(OUT.data_ptr()),
        B, N, eps);

    return OUT;
}

void indexer_kv_proj_forward_out(
    torch::Tensor a_tma_desc,
    torch::Tensor w_tma_desc,
    torch::Tensor w_scale,
    torch::Tensor a_scale,
    torch::Tensor rmsnorm_weight,
    torch::Tensor OUT_gemm,
    torch::Tensor OUT,
    int B, int N, int K, float eps
) {
    at::cuda::CUDAGuard device_guard{OUT.device()};
    const CUtensorMap* a_desc = reinterpret_cast<const CUtensorMap*>(a_tma_desc.data_ptr());
    const CUtensorMap* w_desc = reinterpret_cast<const CUtensorMap*>(w_tma_desc.data_ptr());
    int num_m_tiles = (B + BLOCK_M - 1) / BLOCK_M;
    int num_n_tiles = (N + BLOCK_N - 1) / BLOCK_N;
    dim3 grid(num_m_tiles, num_n_tiles);

    constexpr int smem_bytes = TOTAL_SMEM_BYTES;
    cudaFuncSetAttribute(indexer_kv_proj_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    indexer_kv_proj_kernel<<<grid, THREADS, smem_bytes, stream>>>(
        a_desc, w_desc,
        w_scale.data_ptr<float>(),
        a_scale.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(OUT_gemm.data_ptr()),
        nullptr,
        false,
        B, K, N);

    rmsnorm_kernel<<<B, 128, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(OUT_gemm.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(rmsnorm_weight.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(OUT.data_ptr()),
        B, N, eps);
}

torch::Tensor indexer_kv_proj_gemm_only(
    torch::Tensor a_tma_desc,
    torch::Tensor w_tma_desc,
    torch::Tensor w_scale,
    torch::Tensor a_scale,
    int B, int N, int K
) {
    at::cuda::CUDAGuard device_guard{a_tma_desc.device()};
    auto OUT = torch::empty({B, N}, torch::dtype(torch::kBFloat16).device(torch::kCUDA));

    const CUtensorMap* a_d = reinterpret_cast<const CUtensorMap*>(a_tma_desc.data_ptr());
    const CUtensorMap* w_d = reinterpret_cast<const CUtensorMap*>(w_tma_desc.data_ptr());
    int num_m_tiles = (B + BLOCK_M - 1) / BLOCK_M;
    int num_n_tiles = (N + BLOCK_N - 1) / BLOCK_N;
    dim3 grid(num_m_tiles, num_n_tiles);

    constexpr int smem_bytes = TOTAL_SMEM_BYTES;
    cudaFuncSetAttribute(indexer_kv_proj_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    indexer_kv_proj_kernel<<<grid, THREADS, smem_bytes, stream>>>(
        a_d, w_d,
        w_scale.data_ptr<float>(),
        a_scale.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(OUT.data_ptr()),
        nullptr,
        false,
        B, K, N);

    return OUT;
}

void indexer_kv_proj_gemm_only_out(
    torch::Tensor a_tma_desc,
    torch::Tensor w_tma_desc,
    torch::Tensor w_scale,
    torch::Tensor a_scale,
    torch::Tensor OUT,
    int B, int N, int K
) {
    at::cuda::CUDAGuard device_guard{OUT.device()};
    const CUtensorMap* a_d = reinterpret_cast<const CUtensorMap*>(a_tma_desc.data_ptr());
    const CUtensorMap* w_d = reinterpret_cast<const CUtensorMap*>(w_tma_desc.data_ptr());
    int num_m_tiles = (B + BLOCK_M - 1) / BLOCK_M;
    int num_n_tiles = (N + BLOCK_N - 1) / BLOCK_N;
    dim3 grid(num_m_tiles, num_n_tiles);

    constexpr int smem_bytes = TOTAL_SMEM_BYTES;
    cudaFuncSetAttribute(indexer_kv_proj_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    indexer_kv_proj_kernel<<<grid, THREADS, smem_bytes, stream>>>(
        a_d, w_d,
        w_scale.data_ptr<float>(),
        a_scale.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(OUT.data_ptr()),
        nullptr,
        false,
        B, K, N);
}

void indexer_kv_proj_gemm_only_valid_m_out(
    torch::Tensor a_tma_desc,
    torch::Tensor w_tma_desc,
    torch::Tensor w_scale,
    torch::Tensor a_scale,
    torch::Tensor OUT,
    torch::Tensor num_valid_m,
    int B, int N, int K
) {
    TORCH_CHECK(num_valid_m.is_cuda(), "num_valid_m must be a CUDA tensor");
    TORCH_CHECK(num_valid_m.dtype() == torch::kInt32, "num_valid_m must be int32");
    TORCH_CHECK(num_valid_m.numel() == 1, "num_valid_m must contain one element");
    at::cuda::CUDAGuard device_guard{OUT.device()};
    const CUtensorMap* a_d = reinterpret_cast<const CUtensorMap*>(a_tma_desc.data_ptr());
    const CUtensorMap* w_d = reinterpret_cast<const CUtensorMap*>(w_tma_desc.data_ptr());
    int num_m_tiles = (B + BLOCK_M - 1) / BLOCK_M;
    int num_n_tiles = (N + BLOCK_N - 1) / BLOCK_N;
    dim3 grid(num_m_tiles, num_n_tiles);

    constexpr int smem_bytes = TOTAL_SMEM_BYTES;
    cudaFuncSetAttribute(indexer_kv_proj_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    indexer_kv_proj_kernel<<<grid, THREADS, smem_bytes, stream>>>(
        a_d, w_d,
        w_scale.data_ptr<float>(),
        a_scale.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(OUT.data_ptr()),
        num_valid_m.data_ptr<int>(),
        true,
        B, K, N);
}
''';

CPP_SOURCE = r'''
#include <torch/extension.h>

torch::Tensor create_tma_desc(
    torch::Tensor data, int outer_dim, int inner_dim,
    int box_outer, int box_inner);

void run_act_quant(
    torch::Tensor X, torch::Tensor X_fp8, torch::Tensor X_scale);

void run_act_quant_valid(
    torch::Tensor X, torch::Tensor X_fp8, torch::Tensor X_scale,
    torch::Tensor num_valid_tokens);

torch::Tensor indexer_kv_proj_forward(
    torch::Tensor a_tma_desc,
    torch::Tensor w_tma_desc,
    torch::Tensor w_scale,
    torch::Tensor a_scale,
    torch::Tensor rmsnorm_weight,
    int B, int N, int K, float eps);

void indexer_kv_proj_forward_out(
    torch::Tensor a_tma_desc,
    torch::Tensor w_tma_desc,
    torch::Tensor w_scale,
    torch::Tensor a_scale,
    torch::Tensor rmsnorm_weight,
    torch::Tensor OUT_gemm,
    torch::Tensor OUT,
    int B, int N, int K, float eps);

torch::Tensor indexer_kv_proj_gemm_only(
    torch::Tensor a_tma_desc,
    torch::Tensor w_tma_desc,
    torch::Tensor w_scale,
    torch::Tensor a_scale,
    int B, int N, int K);

void indexer_kv_proj_gemm_only_out(
    torch::Tensor a_tma_desc,
    torch::Tensor w_tma_desc,
    torch::Tensor w_scale,
    torch::Tensor a_scale,
    torch::Tensor OUT,
    int B, int N, int K);

void indexer_kv_proj_gemm_only_valid_m_out(
    torch::Tensor a_tma_desc,
    torch::Tensor w_tma_desc,
    torch::Tensor w_scale,
    torch::Tensor a_scale,
    torch::Tensor OUT,
    torch::Tensor num_valid_m,
    int B, int N, int K);
''';


# ──────────────────────────────────────────────────────────────────────────────
# Build
# ──────────────────────────────────────────────────────────────────────────────

def build_module(num_stages=4):
    cache_key = ("indexer_kv_v3", num_stages)
    if cache_key in _module_cache:
        return _module_cache[cache_key]

    print(f"Building indexer KV proj CUDA kernel v3 (NUM_STAGES={num_stages})...")
    os.environ["MAX_JOBS"] = "8"
    os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"

    module = load_inline(
        name=f"indexer_kv_proj_v3_s{num_stages}",
        cpp_sources=[CPP_SOURCE],
        cuda_sources=[CUDA_SOURCE],
        functions=[
            "create_tma_desc",
            "run_act_quant",
            "run_act_quant_valid",
            "indexer_kv_proj_forward",
            "indexer_kv_proj_forward_out",
            "indexer_kv_proj_gemm_only",
            "indexer_kv_proj_gemm_only_out",
            "indexer_kv_proj_gemm_only_valid_m_out",
        ],
        extra_cuda_cflags=[
            "-O3", "-arch=sm_90a", "--ptxas-options=-v", "-lineinfo",
            f"-DNUM_STAGES={num_stages}",
        ],
        verbose=True,
    )
    print(f"Build complete (indexer_kv_proj v3, NUM_STAGES={num_stages}).")
    _module_cache[cache_key] = module
    return module


# ──────────────────────────────────────────────────────────────────────────────
# Weight preparation
# ──────────────────────────────────────────────────────────────────────────────

class FP8IndexerWeightsCUDA:
    """Pre-quantized indexer wk weights for CUDA WGMMA kernel."""

    def __init__(self, wk_weight_bf16: torch.Tensor, module, block_k: int = 128):
        N, K = wk_weight_bf16.shape
        fp8_max = 448.0
        num_k_blocks = (K + block_k - 1) // block_k

        self.w_fp8 = torch.empty(N, K, dtype=torch.float8_e4m3fn,
                                 device=wk_weight_bf16.device)
        self.w_scale = torch.empty(N // 32, num_k_blocks, dtype=torch.float32,
                                   device=wk_weight_bf16.device)

        for n_tile in range(N // 32):
            n_start = n_tile * 32
            n_end = n_start + 32
            for kb in range(num_k_blocks):
                k_start = kb * block_k
                k_end = min(k_start + block_k, K)
                block = wk_weight_bf16[n_start:n_end, k_start:k_end].float()
                absmax = block.abs().amax()
                scale = absmax.clamp(min=1e-12) / fp8_max
                self.w_fp8[n_start:n_end, k_start:k_end] = (block / scale).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
                self.w_scale[n_tile, kb] = scale

        self.block_k = block_k
        self.N = N
        self.K = K

        # Create TMA descriptor for weight [N, K] with box (32, 128)
        self.tma_desc = module.create_tma_desc(self.w_fp8, N, K, 32, 128)


# ──────────────────────────────────────────────────────────────────────────────
# High-level API
# ──────────────────────────────────────────────────────────────────────────────

def cuda_wk_proj_rmsnorm(
    hidden_states: torch.Tensor,
    weights: FP8IndexerWeightsCUDA,
    rmsnorm_weight: torch.Tensor,
    module,
    eps: float = 1e-6,
) -> torch.Tensor:
    hidden_states = hidden_states.contiguous()
    B, K = hidden_states.shape
    N = weights.N

    # Act quant
    x_fp8 = torch.empty(B, K, dtype=torch.float8_e4m3fn, device=hidden_states.device)
    x_scale = torch.empty(B, dtype=torch.float32, device=hidden_states.device)
    module.run_act_quant(hidden_states, x_fp8, x_scale)

    # Create TMA desc for activation — need padded to BLOCK_M=64
    B_padded = max(B, 64)
    if B < 64:
        x_fp8_padded = torch.zeros(B_padded, K, dtype=torch.float8_e4m3fn, device=x_fp8.device)
        x_fp8_padded[:B] = x_fp8
        x_fp8 = x_fp8_padded
    a_tma_desc = module.create_tma_desc(x_fp8, B_padded, K, 64, 128)

    return module.indexer_kv_proj_forward(
        a_tma_desc, weights.tma_desc,
        weights.w_scale, x_scale,
        rmsnorm_weight,
        B, N, K, eps,
    )


def cuda_wk_proj_rmsnorm_out(
    hidden_states: torch.Tensor,
    weights: FP8IndexerWeightsCUDA,
    rmsnorm_weight: torch.Tensor,
    module,
    x_fp8_padded: torch.Tensor,
    x_scale: torch.Tensor,
    a_tma_desc: torch.Tensor,
    out_gemm: torch.Tensor,
    out: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    if not hidden_states.is_contiguous():
        raise ValueError("hidden_states must be contiguous for graph-captured WK projection")
    B, K = hidden_states.shape
    N = weights.N
    _validate_projection_out_buffers(B, K, N, x_fp8_padded, x_scale, out)
    if out_gemm.shape != (B, N) or out_gemm.dtype != torch.bfloat16:
        raise ValueError(f"out_gemm must be BF16 with shape {(B, N)}, got {out_gemm.shape} {out_gemm.dtype}")

    module.run_act_quant(hidden_states, x_fp8_padded[:B], x_scale)
    module.indexer_kv_proj_forward_out(
        a_tma_desc,
        weights.tma_desc,
        weights.w_scale,
        x_scale,
        rmsnorm_weight,
        out_gemm,
        out,
        B,
        N,
        K,
        eps,
    )
    return out


def cuda_wk_proj_gemm_only(
    hidden_states: torch.Tensor,
    weights: FP8IndexerWeightsCUDA,
    module,
) -> torch.Tensor:
    hidden_states = hidden_states.contiguous()
    B, K = hidden_states.shape
    N = weights.N

    x_fp8 = torch.empty(B, K, dtype=torch.float8_e4m3fn, device=hidden_states.device)
    x_scale = torch.empty(B, dtype=torch.float32, device=hidden_states.device)
    module.run_act_quant(hidden_states, x_fp8, x_scale)

    B_padded = max(B, 64)
    if B < 64:
        x_fp8_padded = torch.zeros(B_padded, K, dtype=torch.float8_e4m3fn, device=x_fp8.device)
        x_fp8_padded[:B] = x_fp8
        x_fp8 = x_fp8_padded
    a_tma_desc = module.create_tma_desc(x_fp8, B_padded, K, 64, 128)

    return module.indexer_kv_proj_gemm_only(
        a_tma_desc, weights.tma_desc,
        weights.w_scale, x_scale,
        B, N, K,
    )


def cuda_wk_proj_gemm_only_out(
    hidden_states: torch.Tensor,
    weights: FP8IndexerWeightsCUDA,
    module,
    x_fp8_padded: torch.Tensor,
    x_scale: torch.Tensor,
    a_tma_desc: torch.Tensor,
    out: torch.Tensor,
    num_valid_tokens: torch.Tensor | None = None,
) -> torch.Tensor:
    if not hidden_states.is_contiguous():
        raise ValueError("hidden_states must be contiguous for graph-captured WK projection")
    B, K = hidden_states.shape
    N = weights.N
    _validate_projection_out_buffers(B, K, N, x_fp8_padded, x_scale, out)

    if num_valid_tokens is None:
        module.run_act_quant(hidden_states, x_fp8_padded[:B], x_scale)
    else:
        module.run_act_quant_valid(hidden_states, x_fp8_padded[:B], x_scale, num_valid_tokens)
    if num_valid_tokens is None:
        module.indexer_kv_proj_gemm_only_out(
            a_tma_desc,
            weights.tma_desc,
            weights.w_scale,
            x_scale,
            out,
            B,
            N,
            K,
        )
    else:
        module.indexer_kv_proj_gemm_only_valid_m_out(
            a_tma_desc,
            weights.tma_desc,
            weights.w_scale,
            x_scale,
            out,
            num_valid_tokens,
            B,
            N,
            K,
        )
    return out


def make_fp8_activation_scratch(
    batch_size: int,
    hidden_size: int,
    module,
    *,
    device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    padded_batch = max(batch_size, _BLOCK_M)
    x_fp8_padded = torch.empty(
        padded_batch,
        hidden_size,
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    x_scale = torch.empty(batch_size, dtype=torch.float32, device=device)
    a_tma_desc = module.create_tma_desc(
        x_fp8_padded,
        padded_batch,
        hidden_size,
        _BLOCK_M,
        _BLOCK_K,
    )
    return x_fp8_padded, x_scale, a_tma_desc


def _validate_projection_out_buffers(
    B: int,
    K: int,
    N: int,
    x_fp8_padded: torch.Tensor,
    x_scale: torch.Tensor,
    out: torch.Tensor,
) -> None:
    padded_batch = max(B, _BLOCK_M)
    if x_fp8_padded.shape != (padded_batch, K) or x_fp8_padded.dtype != torch.float8_e4m3fn:
        raise ValueError(
            f"x_fp8_padded must be FP8 with shape {(padded_batch, K)}, "
            f"got {x_fp8_padded.shape} {x_fp8_padded.dtype}"
        )
    if x_scale.shape != (B,) or x_scale.dtype != torch.float32:
        raise ValueError(f"x_scale must be FP32 with shape {(B,)}, got {x_scale.shape} {x_scale.dtype}")
    if out.shape != (B, N) or out.dtype != torch.bfloat16:
        raise ValueError(f"out must be BF16 with shape {(B, N)}, got {out.shape} {out.dtype}")


# ──────────────────────────────────────────────────────────────────────────────
# Reference
# ──────────────────────────────────────────────────────────────────────────────

def wk_proj_rmsnorm_reference(
    hidden_states: torch.Tensor,
    wk_weight: torch.Tensor,
    rmsnorm_weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    k = torch.nn.functional.linear(hidden_states, wk_weight)
    variance = k.float().pow(2).mean(-1, keepdim=True)
    k_normed = k * torch.rsqrt(variance + eps)
    return (k_normed * rmsnorm_weight).to(torch.bfloat16)
