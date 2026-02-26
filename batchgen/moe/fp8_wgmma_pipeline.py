"""
FP8 WGMMA MoE Pipeline — Production integration for GLM-5-FP8.

Replaces the Triton FP8 grouped GEMM pipeline with WGMMA kernels:
  dispatch_scatter_3d → act_quant_3d → v8c_fused_gateup → fused_silu_quant_3d → v8b_down → reduce

All CUDA kernels are inlined and compiled via load_inline on first use.
Enable with: BATCHGEN_USE_WGMMA_FP8=1

Kernels:
  - v8b: FP8 WGMMA m64n32k32 grouped GEMM (down projection)
  - v8c: Fused gate+up FP8 WGMMA (two projections in one launch)
  - act_quant_3d: Per-block absmax FP8 quantization with transposed scale output
  - fused_silu_quant_3d: SiLU×gate + FP8 quant with transposed scale output
  - dispatch_scatter_3d: Route tokens from flat [G, H] into 3D [E, mtp, H]
  - reduce_weighted_scatter: Weighted sum from 3D output back to flat [G, H]
"""

import os
import torch
from torch.utils.cpp_extension import load_inline

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
BLOCK_M = 64
BLOCK_N = 32
BLOCK_K = 128
SCALE_BLOCK_K = 128
QUANT_BLOCK = 128
DEFAULT_MTP = 4096

# ══════════════════════════════════════════════════════════════════════════════
# CUDA Source: WGMMA Kernels (v8b + v8c combined)
# ══════════════════════════════════════════════════════════════════════════════

WGMMA_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cstdint>
#include <utility>
#include <ATen/cuda/CUDAContext.h>

// ============================================================================
// Configuration — m64n32k32
// ============================================================================
#define WGMMA_M 64
#define BLOCK_M 64
#define BLOCK_N 32
#define BLOCK_K 128
#define WGMMA_K 32
#define TILES_K (BLOCK_K / WGMMA_K)
#define WGMMA_NUM_ACCUM 16
#define TOTAL_THREADS 256
#define PRODUCER_THREADS 128
#define WARP_SIZE 32
#ifndef NUM_STAGES
#define NUM_STAGES 4
#endif
#define TILE_BYTES_A (WGMMA_M * BLOCK_K)
#define TILE_BYTES_B (BLOCK_N * BLOCK_K)
#define TILE_BYTES_XSCALE (BLOCK_M * sizeof(float))
#define SCALE_BLOCK_K 128
#define SCALE_BLOCK_N 128

#define PIPELINE_AB_BYTES (NUM_STAGES * (TILE_BYTES_A + TILE_BYTES_B))
#define PIPELINE_XSCALE_BYTES (NUM_STAGES * TILE_BYTES_XSCALE)
#define BARRIER_BYTES  (2 * NUM_STAGES * sizeof(uint64_t))
#define EPILOGUE_BYTES (BLOCK_M * BLOCK_N * 2)

#define MAX_WSCALE_FLOATS 128
#define WSCALE_SMEM_BYTES (MAX_WSCALE_FLOATS * 4)

#define TOTAL_SMEM_BYTES (PIPELINE_AB_BYTES + PIPELINE_XSCALE_BYTES + BARRIER_BYTES + EPILOGUE_BYTES + WSCALE_SMEM_BYTES)

// ============================================================================
// GmmaDescriptor — 128B swizzle
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
    desc.bitfield.layout_type_ = 1;
    desc.bitfield.leading_byte_offset_ = 0;
    desc.bitfield.stride_byte_offset_ = 1024 >> 4;
    desc.bitfield.base_offset_ = 0;
    return desc;
}

// ============================================================================
// Warpgroup synchronization
// ============================================================================
__device__ __forceinline__ void warpgroup_arrive() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void warpgroup_commit_batch() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}
template <int N>
__device__ __forceinline__ void warpgroup_wait() {
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N) : "memory");
}
__device__ __forceinline__ void warpgroup_fence_operand(float& reg) {
    asm volatile("" : "+f"(reg) :: "memory");
}

// ============================================================================
// WGMMA m64n32k32 FP8×FP8→FP32 SS
// ============================================================================
__device__ __forceinline__ void wgmma_m64n32k32_f32_e4m3_e4m3_ss(
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
    uint64_t desc_a, uint64_t desc_b, float* accum, int scale_D
) {
    wgmma_m64n32k32_f32_e4m3_e4m3_ss(
        desc_a, desc_b,
        accum[0],  accum[1],  accum[2],  accum[3],
        accum[4],  accum[5],  accum[6],  accum[7],
        accum[8],  accum[9],  accum[10], accum[11],
        accum[12], accum[13], accum[14], accum[15],
        scale_D);
}

// ============================================================================
// TMA helpers
// ============================================================================
__device__ __forceinline__ void tma_load_2d(
    const void* desc, uint64_t* mbar, void* smem_ptr,
    int32_t coord_0, int32_t coord_1
) {
    uint64_t desc_addr = reinterpret_cast<uint64_t>(desc);
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    uint32_t mbar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%3, %4}], [%2];"
        :: "r"(smem_addr), "l"(desc_addr), "r"(mbar_addr),
           "r"(coord_0), "r"(coord_1)
        : "memory");
}

// ============================================================================
// mbarrier helpers
// ============================================================================
__device__ __forceinline__ void mbarrier_init(uint64_t* mbar, uint32_t count) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                 :: "r"(smem_addr), "r"(count));
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx(
    uint64_t* mbar, uint32_t tx_bytes
) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    asm volatile(
        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        :: "r"(smem_addr), "r"(tx_bytes));
}

__device__ __forceinline__ void mbarrier_arrive(uint64_t* mbar) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"
                 :: "r"(smem_addr));
}

__device__ __forceinline__ void mbarrier_wait_parity(
    uint64_t* mbar, uint32_t phase_parity
) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    asm volatile(
        "{\n"
        ".reg .pred P;\n"
        "WGMMA_MBAR_WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n"
        "@P bra WGMMA_MBAR_DONE_%=;\n"
        "bra WGMMA_MBAR_WAIT_%=;\n"
        "WGMMA_MBAR_DONE_%=:\n"
        "}\n"
        :: "r"(smem_addr), "r"(phase_parity));
}

// ============================================================================
// SMEM load/store helpers
// ============================================================================
__device__ __forceinline__ float ld_shared(const float* smem_ptr) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    float result;
    asm volatile("ld.shared.f32 %0, [%1];" : "=f"(result) : "r"(smem_addr));
    return result;
}

__device__ __forceinline__ void st_shared(float* smem_ptr, float val) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("st.shared.f32 [%0], %1;" :: "r"(smem_addr), "f"(val));
}

// ============================================================================
// TMA descriptor creation
// ============================================================================
static PFN_cuTensorMapEncodeTiled get_cuTensorMapEncodeTiled() {
    cudaDriverEntryPointQueryResult driver_status;
    void* ptr = nullptr;
#if CUDA_VERSION >= 12050
    cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &ptr, 12000,
                                     cudaEnableDefault, &driver_status);
#else
    cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &ptr,
                            cudaEnableDefault, &driver_status);
#endif
    if (driver_status != cudaDriverEntryPointSuccess)
        throw std::runtime_error("Failed to get cuTensorMapEncodeTiled");
    return reinterpret_cast<PFN_cuTensorMapEncodeTiled>(ptr);
}

static CUtensorMap make_2d_tma_desc_fp8(
    const void* global_address,
    uint64_t gmem_rows, uint64_t gmem_cols,
    uint32_t smem_rows, uint32_t smem_cols,
    PFN_cuTensorMapEncodeTiled encode_func
) {
    CUtensorMap tensor_map = {};
    uint64_t gmem_dim[2] = {gmem_cols, gmem_rows};
    uint64_t global_stride[1] = {gmem_cols * 1};
    uint32_t smem_dim[2] = {smem_cols, smem_rows};
    uint32_t elem_strides[2] = {1, 1};

    auto result = encode_func(
        &tensor_map,
        CU_TENSOR_MAP_DATA_TYPE_UINT8,
        2,
        const_cast<void*>(global_address),
        gmem_dim,
        global_stride,
        smem_dim,
        elem_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (result != CUDA_SUCCESS) {
        char msg[256];
        snprintf(msg, sizeof(msg),
                 "cuTensorMapEncodeTiled failed: error=%d, gmem=[%lu,%lu], smem=[%u,%u]",
                 (int)result, gmem_dim[0], gmem_dim[1], smem_dim[0], smem_dim[1]);
        throw std::runtime_error(msg);
    }
    return tensor_map;
}

// ============================================================================
// v8b Kernel — FP8 WGMMA m64n32k32 (down projection)
// ============================================================================
__global__ void __launch_bounds__(TOTAL_THREADS, 4)
grouped_fp8_moe_gemm_v8b_kernel(
    const CUtensorMap* __restrict__ a_descs,
    const CUtensorMap* __restrict__ b_descs,
    const CUtensorMap* __restrict__ xscale_descs,
    const int32_t* __restrict__ tokens_per_expert,
    const int64_t* __restrict__ w_scale_ptrs,
    int64_t w_scale_stride_n,
    __nv_bfloat16* __restrict__ C,
    int64_t stride_c_m,
    int max_tokens_padded, int N, int K, int num_experts
) {
    const int tid = threadIdx.x;
    const int wg_id = tid / 128;
    const int wg_tid = tid % 128;

    const int expert_idx = blockIdx.x;
    const int n_tile     = blockIdx.y;
    const int m_tile     = blockIdx.z;

    const int m_expert = tokens_per_expert[expert_idx];
    const int m_start_in_expert = m_tile * BLOCK_M;
    if (m_start_in_expert >= m_expert) return;

    const int m_global_start = expert_idx * max_tokens_padded + m_start_in_expert;
    const int m_expert_end = expert_idx * max_tokens_padded + m_expert;

    const int n_start = n_tile * BLOCK_N;
    const int num_k_blocks = (K + BLOCK_K - 1) / BLOCK_K;

    const float* w_scale_base = reinterpret_cast<const float*>(w_scale_ptrs[expert_idx]);

    extern __shared__ __align__(128) char smem_buf[];

    float* smem_xscale_base = reinterpret_cast<float*>(smem_buf + PIPELINE_AB_BYTES);
    uint64_t* full_barriers  = reinterpret_cast<uint64_t*>(
        smem_buf + PIPELINE_AB_BYTES + PIPELINE_XSCALE_BYTES);
    uint64_t* empty_barriers = full_barriers + NUM_STAGES;
    __nv_bfloat16* smem_out  = reinterpret_cast<__nv_bfloat16*>(
        smem_buf + PIPELINE_AB_BYTES + PIPELINE_XSCALE_BYTES + BARRIER_BYTES);
    float* smem_wscale = reinterpret_cast<float*>(
        smem_buf + PIPELINE_AB_BYTES + PIPELINE_XSCALE_BYTES + BARRIER_BYTES + EPILOGUE_BYTES);

    if (tid == 0) {
        for (int s = 0; s < NUM_STAGES; s++) {
            mbarrier_init(&full_barriers[s], 1);
            mbarrier_init(&empty_barriers[s], 1);
        }
    }
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    __syncthreads();

    if (tid == 0) {
        asm volatile("prefetch.tensormap [%0];" :: "l"(&a_descs[expert_idx]) : "memory");
        asm volatile("prefetch.tensormap [%0];" :: "l"(&b_descs[expert_idx]) : "memory");
        asm volatile("prefetch.tensormap [%0];" :: "l"(&xscale_descs[expert_idx]) : "memory");
    }

    {
        const int n_scale_idx = n_start / SCALE_BLOCK_N;
        for (int i = tid; i < num_k_blocks; i += TOTAL_THREADS) {
            st_shared(&smem_wscale[i],
                       w_scale_base[n_scale_idx * w_scale_stride_n + i]);
        }
    }
    __syncthreads();

    int slot = 0, empty_phase = 1, full_phase = 0;

    if (wg_id == 0) {
        for (int kb = 0; kb < num_k_blocks; kb++) {
            mbarrier_wait_parity(&empty_barriers[slot], empty_phase);

            if (wg_tid == 0) {
                mbarrier_arrive_expect_tx(&full_barriers[slot],
                    TILE_BYTES_A + TILE_BYTES_B + TILE_BYTES_XSCALE);

                uint8_t* cur_smem_a = reinterpret_cast<uint8_t*>(
                    smem_buf + slot * (TILE_BYTES_A + TILE_BYTES_B));
                uint8_t* cur_smem_b = cur_smem_a + TILE_BYTES_A;
                float* cur_smem_xscale = smem_xscale_base + slot * BLOCK_M;

                tma_load_2d(&a_descs[expert_idx], &full_barriers[slot],
                            cur_smem_a, kb * BLOCK_K, m_tile * BLOCK_M);
                tma_load_2d(&b_descs[expert_idx], &full_barriers[slot],
                            cur_smem_b, kb * BLOCK_K, n_start);
                tma_load_2d(&xscale_descs[expert_idx], &full_barriers[slot],
                            cur_smem_xscale, m_tile * BLOCK_M, kb);
            }

            slot++;
            if (slot == NUM_STAGES) { slot = 0; empty_phase ^= 1; }
        }
    } else {
        const int warp_in_wg = wg_tid / WARP_SIZE;
        const int lane_id = wg_tid % WARP_SIZE;

        float accum[WGMMA_NUM_ACCUM];
        float result[WGMMA_NUM_ACCUM];
        #pragma unroll
        for (int i = 0; i < WGMMA_NUM_ACCUM; i++) result[i] = 0.0f;

        const int m_row_in_tile_0 = warp_in_wg * 16 + (lane_id / 4);
        const int m_row_in_tile_1 = m_row_in_tile_0 + 8;
        const int m_global_0 = m_global_start + m_row_in_tile_0;
        const int m_global_1 = m_global_start + m_row_in_tile_1;
        const bool valid_0 = (m_global_0 < m_expert_end);
        const bool valid_1 = (m_global_1 < m_expert_end);

        for (int kb = 0; kb < num_k_blocks; kb++) {
            float ws = ld_shared(&smem_wscale[kb]);
            mbarrier_wait_parity(&full_barriers[slot], full_phase);

            float* cur_smem_xscale = smem_xscale_base + slot * BLOCK_M;
            float xs0 = valid_0 ? ld_shared(&cur_smem_xscale[m_row_in_tile_0]) : 0.0f;
            float xs1 = valid_1 ? ld_shared(&cur_smem_xscale[m_row_in_tile_1]) : 0.0f;

            uint8_t* cur_smem_a = reinterpret_cast<uint8_t*>(
                smem_buf + slot * (TILE_BYTES_A + TILE_BYTES_B));
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

            if (wg_tid == 0) {
                mbarrier_arrive(&empty_barriers[slot]);
            }

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

        // Epilogue — STSM + coalesced global stores
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

        asm volatile("bar.sync 1, 128;" ::: "memory");

        const int bf16_2_per_row = BLOCK_N / 2;
        const int total_bf16_2 = BLOCK_M * bf16_2_per_row;
        for (int idx = wg_tid; idx < total_bf16_2; idx += 128) {
            const int row = idx / bf16_2_per_row;
            const int col2 = idx % bf16_2_per_row;
            const int n_global = n_start + col2 * 2;
            const int m_global = m_global_start + row;

            if (m_global < m_expert_end && n_global + 1 < N) {
                __nv_bfloat162 val = *reinterpret_cast<__nv_bfloat162*>(
                    &smem_out[row * BLOCK_N + col2 * 2]);
                *reinterpret_cast<__nv_bfloat162*>(
                    &C[m_global * stride_c_m + n_global]) = val;
            } else if (m_global < m_expert_end && n_global < N) {
                C[m_global * stride_c_m + n_global] =
                    smem_out[row * BLOCK_N + col2 * 2];
            }
        }
    }
}

// ============================================================================
// v8c Kernel — Fused gate+up WGMMA (doubled N-tiles, gate/up selection)
// ============================================================================
__global__ void __launch_bounds__(TOTAL_THREADS, 4)
grouped_fp8_moe_gemm_v8c_kernel(
    const CUtensorMap* __restrict__ a_descs,
    const CUtensorMap* __restrict__ b_descs_gate,
    const CUtensorMap* __restrict__ b_descs_up,
    const CUtensorMap* __restrict__ xscale_descs,
    const int32_t* __restrict__ tokens_per_expert,
    const int64_t* __restrict__ w_scale_ptrs_gate,
    const int64_t* __restrict__ w_scale_ptrs_up,
    int64_t w_scale_stride_n,
    __nv_bfloat16* __restrict__ C_gate,
    __nv_bfloat16* __restrict__ C_up,
    int64_t stride_c_gate_m,
    int64_t stride_c_up_m,
    int max_tokens_padded,
    int N_gate, int N_up, int K,
    int num_n_tiles_gate,
    int num_experts
) {
    const int tid = threadIdx.x;
    const int wg_id = tid / 128;
    const int wg_tid = tid % 128;

    const int expert_idx = blockIdx.x;
    const int n_tile_global = blockIdx.y;
    const int m_tile     = blockIdx.z;

    const bool is_up = (n_tile_global >= num_n_tiles_gate);
    const CUtensorMap* my_b_descs = is_up ? b_descs_up : b_descs_gate;
    const int64_t* my_w_scale_ptrs = is_up ? w_scale_ptrs_up : w_scale_ptrs_gate;
    __nv_bfloat16* my_C = is_up ? C_up : C_gate;
    const int64_t my_stride_c_m = is_up ? stride_c_up_m : stride_c_gate_m;
    const int my_N = is_up ? N_up : N_gate;
    const int n_tile = is_up ? (n_tile_global - num_n_tiles_gate) : n_tile_global;

    const int m_expert = tokens_per_expert[expert_idx];
    const int m_start_in_expert = m_tile * BLOCK_M;
    if (m_start_in_expert >= m_expert) return;

    const int m_global_start = expert_idx * max_tokens_padded + m_start_in_expert;
    const int m_expert_end = expert_idx * max_tokens_padded + m_expert;

    const int n_start = n_tile * BLOCK_N;
    const int num_k_blocks = (K + BLOCK_K - 1) / BLOCK_K;

    const float* w_scale_base = reinterpret_cast<const float*>(my_w_scale_ptrs[expert_idx]);

    extern __shared__ __align__(128) char smem_buf[];

    float* smem_xscale_base = reinterpret_cast<float*>(smem_buf + PIPELINE_AB_BYTES);
    uint64_t* full_barriers  = reinterpret_cast<uint64_t*>(
        smem_buf + PIPELINE_AB_BYTES + PIPELINE_XSCALE_BYTES);
    uint64_t* empty_barriers = full_barriers + NUM_STAGES;
    __nv_bfloat16* smem_out  = reinterpret_cast<__nv_bfloat16*>(
        smem_buf + PIPELINE_AB_BYTES + PIPELINE_XSCALE_BYTES + BARRIER_BYTES);
    float* smem_wscale = reinterpret_cast<float*>(
        smem_buf + PIPELINE_AB_BYTES + PIPELINE_XSCALE_BYTES + BARRIER_BYTES + EPILOGUE_BYTES);

    if (tid == 0) {
        for (int s = 0; s < NUM_STAGES; s++) {
            mbarrier_init(&full_barriers[s], 1);
            mbarrier_init(&empty_barriers[s], 1);
        }
    }
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    __syncthreads();

    if (tid == 0) {
        asm volatile("prefetch.tensormap [%0];" :: "l"(&a_descs[expert_idx]) : "memory");
        asm volatile("prefetch.tensormap [%0];" :: "l"(&my_b_descs[expert_idx]) : "memory");
        asm volatile("prefetch.tensormap [%0];" :: "l"(&xscale_descs[expert_idx]) : "memory");
    }

    {
        const int n_scale_idx = n_start / SCALE_BLOCK_N;
        for (int i = tid; i < num_k_blocks; i += TOTAL_THREADS) {
            st_shared(&smem_wscale[i],
                       w_scale_base[n_scale_idx * w_scale_stride_n + i]);
        }
    }
    __syncthreads();

    int slot = 0, empty_phase = 1, full_phase = 0;

    if (wg_id == 0) {
        for (int kb = 0; kb < num_k_blocks; kb++) {
            mbarrier_wait_parity(&empty_barriers[slot], empty_phase);

            if (wg_tid == 0) {
                mbarrier_arrive_expect_tx(&full_barriers[slot],
                    TILE_BYTES_A + TILE_BYTES_B + TILE_BYTES_XSCALE);

                uint8_t* cur_smem_a = reinterpret_cast<uint8_t*>(
                    smem_buf + slot * (TILE_BYTES_A + TILE_BYTES_B));
                uint8_t* cur_smem_b = cur_smem_a + TILE_BYTES_A;
                float* cur_smem_xscale = smem_xscale_base + slot * BLOCK_M;

                tma_load_2d(&a_descs[expert_idx], &full_barriers[slot],
                            cur_smem_a, kb * BLOCK_K, m_tile * BLOCK_M);
                tma_load_2d(&my_b_descs[expert_idx], &full_barriers[slot],
                            cur_smem_b, kb * BLOCK_K, n_start);
                tma_load_2d(&xscale_descs[expert_idx], &full_barriers[slot],
                            cur_smem_xscale, m_tile * BLOCK_M, kb);
            }

            slot++;
            if (slot == NUM_STAGES) { slot = 0; empty_phase ^= 1; }
        }
    } else {
        const int warp_in_wg = wg_tid / WARP_SIZE;
        const int lane_id = wg_tid % WARP_SIZE;

        float accum[WGMMA_NUM_ACCUM];
        float result[WGMMA_NUM_ACCUM];
        #pragma unroll
        for (int i = 0; i < WGMMA_NUM_ACCUM; i++) result[i] = 0.0f;

        const int m_row_in_tile_0 = warp_in_wg * 16 + (lane_id / 4);
        const int m_row_in_tile_1 = m_row_in_tile_0 + 8;
        const int m_global_0 = m_global_start + m_row_in_tile_0;
        const int m_global_1 = m_global_start + m_row_in_tile_1;
        const bool valid_0 = (m_global_0 < m_expert_end);
        const bool valid_1 = (m_global_1 < m_expert_end);

        for (int kb = 0; kb < num_k_blocks; kb++) {
            float ws = ld_shared(&smem_wscale[kb]);
            mbarrier_wait_parity(&full_barriers[slot], full_phase);

            float* cur_smem_xscale = smem_xscale_base + slot * BLOCK_M;
            float xs0 = valid_0 ? ld_shared(&cur_smem_xscale[m_row_in_tile_0]) : 0.0f;
            float xs1 = valid_1 ? ld_shared(&cur_smem_xscale[m_row_in_tile_1]) : 0.0f;

            uint8_t* cur_smem_a = reinterpret_cast<uint8_t*>(
                smem_buf + slot * (TILE_BYTES_A + TILE_BYTES_B));
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

            if (wg_tid == 0) {
                mbarrier_arrive(&empty_barriers[slot]);
            }

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

        // Epilogue — STSM + coalesced global stores
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

        asm volatile("bar.sync 1, 128;" ::: "memory");

        const int bf16_2_per_row = BLOCK_N / 2;
        const int total_bf16_2 = BLOCK_M * bf16_2_per_row;
        for (int idx = wg_tid; idx < total_bf16_2; idx += 128) {
            const int row = idx / bf16_2_per_row;
            const int col2 = idx % bf16_2_per_row;
            const int n_global = n_start + col2 * 2;
            const int m_global = m_global_start + row;

            if (m_global < m_expert_end && n_global + 1 < my_N) {
                __nv_bfloat162 val = *reinterpret_cast<__nv_bfloat162*>(
                    &smem_out[row * BLOCK_N + col2 * 2]);
                *reinterpret_cast<__nv_bfloat162*>(
                    &my_C[m_global * my_stride_c_m + n_global]) = val;
            } else if (m_global < m_expert_end && n_global < my_N) {
                my_C[m_global * my_stride_c_m + n_global] =
                    smem_out[row * BLOCK_N + col2 * 2];
            }
        }
    }
}

// ============================================================================
// TMA descriptor creation — v8b (3E: A + B + xscale)
// ============================================================================
torch::Tensor create_tma_descriptors_v8b(
    torch::Tensor act_buffer,
    torch::Tensor weight_ptrs,
    torch::Tensor xscale_buffer,
    int N, int max_tokens_padded
) {
    const int num_experts = act_buffer.size(0);
    const int K = act_buffer.size(2);
    const int num_k_blocks = K / SCALE_BLOCK_K;

    static auto encode_func = get_cuTensorMapEncodeTiled();

    const int desc_size = sizeof(CUtensorMap);
    std::vector<CUtensorMap> all_descs(3 * num_experts);

    const auto wp_cpu = weight_ptrs.cpu();
    const int64_t* wp = wp_cpu.data_ptr<int64_t>();

    for (int e = 0; e < num_experts; e++) {
        const uint8_t* a_base = reinterpret_cast<const uint8_t*>(act_buffer.data_ptr())
                                + (int64_t)e * max_tokens_padded * K;
        all_descs[e] = make_2d_tma_desc_fp8(
            a_base, max_tokens_padded, K, BLOCK_M, BLOCK_K, encode_func);

        const uint8_t* b_base = reinterpret_cast<const uint8_t*>(wp[e]);
        all_descs[num_experts + e] = make_2d_tma_desc_fp8(
            b_base, N, K, BLOCK_N, BLOCK_K, encode_func);

        const float* xs_base = reinterpret_cast<const float*>(xscale_buffer.data_ptr())
                               + (int64_t)e * max_tokens_padded;
        int64_t total_cols = xscale_buffer.size(1);

        {
            CUtensorMap tensor_map = {};
            uint64_t gmem_dim[2] = {(uint64_t)max_tokens_padded, (uint64_t)num_k_blocks};
            uint64_t global_stride[1] = {(uint64_t)total_cols * sizeof(float)};
            uint32_t smem_dim[2] = {BLOCK_M, 1};
            uint32_t elem_strides[2] = {1, 1};

            auto result = encode_func(
                &tensor_map,
                CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
                2,
                const_cast<float*>(xs_base),
                gmem_dim,
                global_stride,
                smem_dim,
                elem_strides,
                CU_TENSOR_MAP_INTERLEAVE_NONE,
                CU_TENSOR_MAP_SWIZZLE_NONE,
                CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
            if (result != CUDA_SUCCESS)
                throw std::runtime_error("xscale TMA desc creation failed (v8b)");
            all_descs[2 * num_experts + e] = tensor_map;
        }
    }

    auto descs_dev = torch::empty({3 * num_experts * desc_size},
                                   torch::dtype(torch::kUInt8).device(act_buffer.device()));
    cudaMemcpy(descs_dev.data_ptr(), all_descs.data(),
               3 * num_experts * desc_size, cudaMemcpyHostToDevice);
    return descs_dev;
}

// ============================================================================
// TMA descriptor creation — v8c (4E: A + B_gate + B_up + xscale)
// ============================================================================
torch::Tensor create_tma_descriptors_v8c(
    torch::Tensor act_buffer,
    torch::Tensor weight_ptrs_gate,
    torch::Tensor weight_ptrs_up,
    torch::Tensor xscale_buffer,
    int N_gate, int N_up, int max_tokens_padded
) {
    const int num_experts = act_buffer.size(0);
    const int K = act_buffer.size(2);
    const int num_k_blocks = K / SCALE_BLOCK_K;

    static auto encode_func = get_cuTensorMapEncodeTiled();

    const int desc_size = sizeof(CUtensorMap);
    std::vector<CUtensorMap> all_descs(4 * num_experts);

    const auto wp_gate_cpu = weight_ptrs_gate.cpu();
    const int64_t* wp_gate = wp_gate_cpu.data_ptr<int64_t>();
    const auto wp_up_cpu = weight_ptrs_up.cpu();
    const int64_t* wp_up = wp_up_cpu.data_ptr<int64_t>();

    for (int e = 0; e < num_experts; e++) {
        const uint8_t* a_base = reinterpret_cast<const uint8_t*>(act_buffer.data_ptr())
                                + (int64_t)e * max_tokens_padded * K;
        all_descs[e] = make_2d_tma_desc_fp8(
            a_base, max_tokens_padded, K, BLOCK_M, BLOCK_K, encode_func);

        const uint8_t* bg_base = reinterpret_cast<const uint8_t*>(wp_gate[e]);
        all_descs[num_experts + e] = make_2d_tma_desc_fp8(
            bg_base, N_gate, K, BLOCK_N, BLOCK_K, encode_func);

        const uint8_t* bu_base = reinterpret_cast<const uint8_t*>(wp_up[e]);
        all_descs[2 * num_experts + e] = make_2d_tma_desc_fp8(
            bu_base, N_up, K, BLOCK_N, BLOCK_K, encode_func);

        const float* xs_base = reinterpret_cast<const float*>(xscale_buffer.data_ptr())
                               + (int64_t)e * max_tokens_padded;
        int64_t total_cols = xscale_buffer.size(1);

        {
            CUtensorMap tensor_map = {};
            uint64_t gmem_dim[2] = {(uint64_t)max_tokens_padded, (uint64_t)num_k_blocks};
            uint64_t global_stride[1] = {(uint64_t)total_cols * sizeof(float)};
            uint32_t smem_dim[2] = {BLOCK_M, 1};
            uint32_t elem_strides[2] = {1, 1};

            auto result = encode_func(
                &tensor_map,
                CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
                2,
                const_cast<float*>(xs_base),
                gmem_dim,
                global_stride,
                smem_dim,
                elem_strides,
                CU_TENSOR_MAP_INTERLEAVE_NONE,
                CU_TENSOR_MAP_SWIZZLE_NONE,
                CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
            if (result != CUDA_SUCCESS)
                throw std::runtime_error("xscale TMA desc creation failed (v8c)");
            all_descs[3 * num_experts + e] = tensor_map;
        }
    }

    auto opts = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);
    auto desc_tensor = torch::empty({(int64_t)(4 * num_experts * desc_size)}, opts);
    cudaMemcpy(desc_tensor.data_ptr(), all_descs.data(),
               4 * num_experts * desc_size, cudaMemcpyHostToDevice);
    return desc_tensor;
}

// ============================================================================
// Forward wrappers
// ============================================================================
torch::Tensor grouped_fp8_moe_gemm_v8b(
    torch::Tensor tma_descs,
    torch::Tensor tokens_per_expert,
    torch::Tensor w_scale_ptrs,
    int64_t w_scale_stride_n,
    torch::Tensor C,
    int max_tokens_padded, int N, int K, int num_experts
) {
    const int desc_size = sizeof(CUtensorMap);
    const CUtensorMap* a_descs = reinterpret_cast<const CUtensorMap*>(tma_descs.data_ptr());
    const CUtensorMap* b_descs = a_descs + num_experts;
    const CUtensorMap* xscale_descs = b_descs + num_experts;

    const int num_n_tiles = (N + BLOCK_N - 1) / BLOCK_N;
    const int max_m_tiles = (max_tokens_padded + BLOCK_M - 1) / BLOCK_M;
    dim3 grid(num_experts, num_n_tiles, max_m_tiles);
    dim3 block(TOTAL_THREADS);

    constexpr int smem_bytes = TOTAL_SMEM_BYTES;

    cudaFuncSetAttribute(grouped_fp8_moe_gemm_v8b_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    grouped_fp8_moe_gemm_v8b_kernel<<<grid, block, smem_bytes, stream>>>(
        a_descs, b_descs, xscale_descs,
        tokens_per_expert.data_ptr<int32_t>(),
        w_scale_ptrs.data_ptr<int64_t>(),
        w_scale_stride_n,
        reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
        C.stride(0),
        max_tokens_padded, N, K, num_experts);

    return C;
}

std::vector<torch::Tensor> grouped_fp8_moe_gemm_v8c(
    torch::Tensor tma_descs,
    torch::Tensor tokens_per_expert,
    torch::Tensor w_scale_ptrs_gate,
    torch::Tensor w_scale_ptrs_up,
    int64_t w_scale_stride_n,
    torch::Tensor C_gate,
    torch::Tensor C_up,
    int max_tokens_padded,
    int N_gate, int N_up, int K,
    int num_experts
) {
    const int desc_size = sizeof(CUtensorMap);
    const CUtensorMap* a_descs = reinterpret_cast<const CUtensorMap*>(tma_descs.data_ptr());
    const CUtensorMap* b_descs_gate = a_descs + num_experts;
    const CUtensorMap* b_descs_up = b_descs_gate + num_experts;
    const CUtensorMap* xscale_descs = b_descs_up + num_experts;

    const int num_n_tiles_gate = (N_gate + BLOCK_N - 1) / BLOCK_N;
    const int num_n_tiles_up = (N_up + BLOCK_N - 1) / BLOCK_N;
    const int max_m_tiles = (max_tokens_padded + BLOCK_M - 1) / BLOCK_M;
    dim3 grid(num_experts, num_n_tiles_gate + num_n_tiles_up, max_m_tiles);
    dim3 block(TOTAL_THREADS);

    constexpr int smem_bytes = TOTAL_SMEM_BYTES;

    cudaFuncSetAttribute(grouped_fp8_moe_gemm_v8c_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    grouped_fp8_moe_gemm_v8c_kernel<<<grid, block, smem_bytes, stream>>>(
        a_descs, b_descs_gate, b_descs_up, xscale_descs,
        tokens_per_expert.data_ptr<int32_t>(),
        w_scale_ptrs_gate.data_ptr<int64_t>(),
        w_scale_ptrs_up.data_ptr<int64_t>(),
        w_scale_stride_n,
        reinterpret_cast<__nv_bfloat16*>(C_gate.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(C_up.data_ptr()),
        C_gate.stride(0),
        C_up.stride(0),
        max_tokens_padded, N_gate, N_up, K,
        num_n_tiles_gate,
        num_experts);

    return {C_gate, C_up};
}
''';

WGMMA_CPP_SOURCE = r'''
#include <torch/extension.h>

torch::Tensor create_tma_descriptors_v8b(
    torch::Tensor act_buffer, torch::Tensor weight_ptrs,
    torch::Tensor xscale_buffer, int N, int max_tokens_padded);

torch::Tensor create_tma_descriptors_v8c(
    torch::Tensor act_buffer, torch::Tensor weight_ptrs_gate,
    torch::Tensor weight_ptrs_up, torch::Tensor xscale_buffer,
    int N_gate, int N_up, int max_tokens_padded);

torch::Tensor grouped_fp8_moe_gemm_v8b(
    torch::Tensor tma_descs, torch::Tensor tokens_per_expert,
    torch::Tensor w_scale_ptrs, int64_t w_scale_stride_n,
    torch::Tensor C, int max_tokens_padded, int N, int K, int num_experts);

std::vector<torch::Tensor> grouped_fp8_moe_gemm_v8c(
    torch::Tensor tma_descs, torch::Tensor tokens_per_expert,
    torch::Tensor w_scale_ptrs_gate, torch::Tensor w_scale_ptrs_up,
    int64_t w_scale_stride_n, torch::Tensor C_gate, torch::Tensor C_up,
    int max_tokens_padded, int N_gate, int N_up, int K, int num_experts);
''';

# ══════════════════════════════════════════════════════════════════════════════
# CUDA Source: Fast kernels (act_quant_3d, fused_silu_quant_3d)
# Modified: scale output in transposed [nk, E*mtp] layout
# ══════════════════════════════════════════════════════════════════════════════

FAST_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <ATen/cuda/CUDAContext.h>

#define FP8_MAX_VAL 448.0f
#define QUANT_EPS 1e-12f
#define BLOCK_SIZE_QUANT 128

// ============================================================================
// act_quant_3d — FP8 quantization with transposed scale output
// Input:  x [E, mtp, K] BF16
// Output: y [E, mtp, K] uint8 (FP8), scale_t [num_k_blocks, E*mtp] F32
// ============================================================================
__global__ void act_quant_3d_kernel(
    const __nv_bfloat16* __restrict__ x,
    uint8_t* __restrict__ y,
    float* __restrict__ scale_t,      // [num_k_blocks, E * mtp] — transposed
    const int32_t* __restrict__ tokens_per_expert,
    int E, int mtp, int K, int num_k_blocks
) {
    const int expert = blockIdx.x;
    const int valid_tokens = tokens_per_expert[expert];
    if (valid_tokens == 0) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int num_warps = blockDim.x / 32;

    const __nv_bfloat16* x_expert = x + (int64_t)expert * mtp * K;
    uint8_t* y_expert = y + (int64_t)expert * mtp * K;
    const int64_t total_m = (int64_t)E * mtp;

    for (int m = 0; m < valid_tokens; m++) {
        const __nv_bfloat16* x_row = x_expert + (int64_t)m * K;
        uint8_t* y_row = y_expert + (int64_t)m * K;
        const int64_t m_global = (int64_t)expert * mtp + m;

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

            float s = fmaxf(local_max, QUANT_EPS) / FP8_MAX_VAL;
            float inv_s = 1.0f / s;

            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int col = col_base + lane_id * 4 + i;
                if (col < K) {
                    float scaled = vals[i] * inv_s;
                    scaled = fmaxf(fminf(scaled, FP8_MAX_VAL), -FP8_MAX_VAL);
                    y_row[col] = __nv_cvt_float_to_fp8(scaled, __NV_SATFINITE, __NV_E4M3);
                }
            }

            // Store scale in transposed layout: scale_t[kb, m_global]
            if (lane_id == 0) {
                scale_t[(int64_t)kb * total_m + m_global] = s;
            }
        }
    }
}

// ============================================================================
// fused_silu_quant_3d — SiLU×gate + FP8 quant with transposed scale output
// Input:  gate [E, mtp, N] BF16, up [E, mtp, N] BF16
// Output: y [E, mtp, N] uint8 (FP8), scale_t [num_n_blocks, E*mtp] F32
// ============================================================================
__global__ void fused_silu_quant_3d_kernel(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    uint8_t* __restrict__ y,
    float* __restrict__ scale_t,      // [num_n_blocks, E * mtp] — transposed
    const int32_t* __restrict__ tokens_per_expert,
    int E, int mtp, int N, int num_n_blocks
) {
    const int expert = blockIdx.x;
    const int valid_tokens = tokens_per_expert[expert];
    if (valid_tokens == 0) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int num_warps = blockDim.x / 32;

    const int64_t expert_offset = (int64_t)expert * mtp * N;
    const int64_t total_m = (int64_t)E * mtp;

    for (int m = 0; m < valid_tokens; m++) {
        const int64_t row_offset = expert_offset + (int64_t)m * N;
        const int64_t m_global = (int64_t)expert * mtp + m;

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

            // Store scale in transposed layout: scale_t[kb, m_global]
            if (lane_id == 0) {
                scale_t[(int64_t)kb * total_m + m_global] = s;
            }
        }
    }
}

// ============================================================================
// C++ wrappers
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
    // Transposed scale: [num_k_blocks, E * mtp]
    auto scale_t = torch::empty({num_k_blocks, E * mtp},
                                 torch::dtype(torch::kFloat32).device(x.device()));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    act_quant_3d_kernel<<<E, 128, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        y.data_ptr<uint8_t>(),
        scale_t.data_ptr<float>(),
        tokens_per_expert.data_ptr<int32_t>(),
        E, mtp, K, num_k_blocks);

    return std::make_tuple(y, scale_t);
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
    // Transposed scale: [num_n_blocks, E * mtp]
    auto scale_t = torch::empty({num_n_blocks, E * mtp},
                                 torch::dtype(torch::kFloat32).device(gate.device()));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    fused_silu_quant_3d_kernel<<<E, 128, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(gate.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(up.data_ptr()),
        y.data_ptr<uint8_t>(),
        scale_t.data_ptr<float>(),
        tokens_per_expert.data_ptr<int32_t>(),
        E, mtp, N, num_n_blocks);

    return std::make_tuple(y, scale_t);
}
''';

FAST_CPP_SOURCE = r'''
#include <torch/extension.h>

std::tuple<torch::Tensor, torch::Tensor> act_quant_3d(
    torch::Tensor x, torch::Tensor tokens_per_expert);

std::tuple<torch::Tensor, torch::Tensor> fused_silu_quant_3d(
    torch::Tensor gate, torch::Tensor up, torch::Tensor tokens_per_expert);
''';

# ══════════════════════════════════════════════════════════════════════════════
# CUDA Source: Dispatch scatter 3D + Reduce weighted scatter
# ══════════════════════════════════════════════════════════════════════════════

DISPATCH_REDUCE_CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <c10/cuda/CUDAStream.h>

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
''';

DISPATCH_REDUCE_CPP_SOURCE = r'''
#include <torch/extension.h>

std::vector<torch::Tensor> dispatch_scatter_3d(
    torch::Tensor x, torch::Tensor topk_indices,
    torch::Tensor act_buffer, int64_t expert_start,
    int64_t num_local_experts, int64_t max_tokens_padded,
    torch::Tensor expert_counts, torch::Tensor expert_counters,
    torch::Tensor topk_pos);

torch::Tensor reduce_weighted_scatter(
    torch::Tensor expert_output, torch::Tensor topk_pos,
    torch::Tensor topk_weights, int64_t N, int64_t H, int64_t K,
    torch::Tensor output);
''';


# ══════════════════════════════════════════════════════════════════════════════
# Module builders
# ══════════════════════════════════════════════════════════════════════════════

_module_cache = {}


def _build_wgmma_module():
    if "wgmma" in _module_cache:
        return _module_cache["wgmma"]

    print("[WGMMA Pipeline] Building WGMMA kernels (v8b + v8c)...")
    os.environ["MAX_JOBS"] = "8"
    os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"

    module = load_inline(
        name="wgmma_fp8_pipeline",
        cpp_sources=[WGMMA_CPP_SOURCE],
        cuda_sources=[WGMMA_CUDA_SOURCE],
        functions=[
            "create_tma_descriptors_v8b", "grouped_fp8_moe_gemm_v8b",
            "create_tma_descriptors_v8c", "grouped_fp8_moe_gemm_v8c",
        ],
        extra_cuda_cflags=[
            "-O3", "-arch=sm_90a", "--ptxas-options=-v", "-lineinfo",
            "-DNUM_STAGES=4",
        ],
        verbose=True,
    )
    print("[WGMMA Pipeline] WGMMA build complete.")
    _module_cache["wgmma"] = module
    return module


def _build_fast_module():
    if "fast" in _module_cache:
        return _module_cache["fast"]

    print("[WGMMA Pipeline] Building fast kernels (act_quant_3d, fused_silu_quant_3d)...")
    os.environ["MAX_JOBS"] = "8"
    os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"

    module = load_inline(
        name="wgmma_fast_kernels",
        cpp_sources=[FAST_CPP_SOURCE],
        cuda_sources=[FAST_CUDA_SOURCE],
        functions=["act_quant_3d", "fused_silu_quant_3d"],
        extra_cuda_cflags=[
            "-O3", "-arch=sm_90a", "--ptxas-options=-v", "-lineinfo",
        ],
        verbose=True,
    )
    print("[WGMMA Pipeline] Fast kernels build complete.")
    _module_cache["fast"] = module
    return module


def _build_dispatch_reduce_module():
    if "dispatch_reduce" in _module_cache:
        return _module_cache["dispatch_reduce"]

    print("[WGMMA Pipeline] Building dispatch + reduce kernels...")
    os.environ["MAX_JOBS"] = "8"
    os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"

    module = load_inline(
        name="wgmma_dispatch_reduce",
        cpp_sources=[DISPATCH_REDUCE_CPP_SOURCE],
        cuda_sources=[DISPATCH_REDUCE_CUDA_SOURCE],
        functions=["dispatch_scatter_3d", "reduce_weighted_scatter"],
        extra_cuda_cflags=[
            "-O3", "-arch=sm_90a", "--ptxas-options=-v", "-lineinfo",
        ],
        verbose=True,
    )
    print("[WGMMA Pipeline] Dispatch + reduce build complete.")
    _module_cache["dispatch_reduce"] = module
    return module


def build_all_modules():
    """Build all CUDA modules. Call once during init()."""
    wgmma_mod = _build_wgmma_module()
    fast_mod = _build_fast_module()
    dr_mod = _build_dispatch_reduce_module()
    return wgmma_mod, fast_mod, dr_mod


# ══════════════════════════════════════════════════════════════════════════════
# WGMMAMoEBuffers — Pre-allocated buffers for the full pipeline
# ══════════════════════════════════════════════════════════════════════════════

class WGMMAMoEBuffers:
    """Manages all pre-allocated buffers for the WGMMA MoE pipeline.

    Shared across all MoE layers — buffers are reused, only weight pointers
    differ per layer (passed at forward time).
    """

    def __init__(self, wgmma_mod, fast_mod, dr_mod,
                 E_local, mtp, H, N,
                 gate_weights_list, gate_scales_list,
                 up_weights_list, up_scales_list,
                 down_weights_list, down_scales_list,
                 expert_start, top_k, num_global_tokens,
                 device="cuda"):
        self.wgmma_mod = wgmma_mod
        self.fast_mod = fast_mod
        self.dr_mod = dr_mod
        self.E = E_local
        self.H = H
        self.N = N
        self.expert_start = expert_start
        self.top_k = top_k
        self.device = device

        self.mtp = mtp
        mtp_padded = ((mtp + BLOCK_M - 1) // BLOCK_M) * BLOCK_M
        self.mtp_padded = mtp_padded

        num_k_h = H // QUANT_BLOCK   # H//128
        num_k_n = N // QUANT_BLOCK   # N//128

        # ── Activation buffers ──
        self.act_buf_bf16 = torch.zeros(
            E_local, mtp_padded, H, dtype=torch.bfloat16, device=device)
        self.act_buf_fp8 = torch.zeros(
            E_local, mtp_padded, H, dtype=torch.uint8, device=device)
        self.act_scale_t = torch.zeros(
            num_k_h, E_local * mtp_padded, dtype=torch.float32, device=device)

        # ── Intermediate buffers ──
        self.gate_out = torch.zeros(
            E_local * mtp_padded, N, dtype=torch.bfloat16, device=device)
        self.up_out = torch.zeros(
            E_local * mtp_padded, N, dtype=torch.bfloat16, device=device)
        self.inter_fp8 = torch.zeros(
            E_local, mtp_padded, N, dtype=torch.uint8, device=device)
        self.inter_scale_t = torch.zeros(
            num_k_n, E_local * mtp_padded, dtype=torch.float32, device=device)

        # ── Output buffer ──
        self.down_out = torch.zeros(
            E_local * mtp_padded, H, dtype=torch.bfloat16, device=device)

        # ── Dispatch/reduce buffers ──
        NK = num_global_tokens * top_k
        self.expert_counts = torch.zeros(E_local, dtype=torch.int32, device=device)
        self.expert_counters = torch.zeros(E_local, dtype=torch.int32, device=device)
        self.topk_pos = torch.full((NK,), -1, dtype=torch.int32, device=device)
        self.num_global_tokens = num_global_tokens

        # ── Weight pointers (per layer — stored as list of per-layer dicts) ──
        # For now, store the first layer's weights. Multi-layer support via
        # register_layer_weights() and select_layer().
        self._layer_weights = {}
        self._current_layer_id = None

        # Register the initial layer
        self.register_layer_weights(
            0, gate_weights_list, gate_scales_list,
            up_weights_list, up_scales_list,
            down_weights_list, down_scales_list)

        # ── TMA descriptors (created once, invalidated on resize) ──
        self._create_tma_descriptors()

    def register_layer_weights(self, layer_id,
                               gate_weights, gate_scales,
                               up_weights, up_scales,
                               down_weights, down_scales):
        """Register weight pointers for one MoE layer."""
        dev = self.device
        self._layer_weights[layer_id] = {
            "gate_weight_ptrs": torch.tensor(
                [w.data_ptr() for w in gate_weights], dtype=torch.int64, device=dev),
            "up_weight_ptrs": torch.tensor(
                [w.data_ptr() for w in up_weights], dtype=torch.int64, device=dev),
            "down_weight_ptrs": torch.tensor(
                [w.data_ptr() for w in down_weights], dtype=torch.int64, device=dev),
            "gate_scale_ptrs": torch.tensor(
                [s.data_ptr() for s in gate_scales], dtype=torch.int64, device=dev),
            "up_scale_ptrs": torch.tensor(
                [s.data_ptr() for s in up_scales], dtype=torch.int64, device=dev),
            "down_scale_ptrs": torch.tensor(
                [s.data_ptr() for s in down_scales], dtype=torch.int64, device=dev),
            # Keep references alive
            "_refs": (gate_weights, gate_scales, up_weights, up_scales,
                      down_weights, down_scales),
        }

    def _create_tma_descriptors(self, layer_id=0):
        """Create TMA descriptors for v8c (stage 1) and v8b (stage 2).

        Must be called each time the layer changes (weight pointers differ).
        """
        lw = self._layer_weights[layer_id]

        # v8c: 4E descriptors (A + B_gate + B_up + xscale)
        self.tma_v8c = self.wgmma_mod.create_tma_descriptors_v8c(
            self.act_buf_fp8, lw["gate_weight_ptrs"], lw["up_weight_ptrs"],
            self.act_scale_t, self.N, self.N, self.mtp_padded)

        # v8b: 3E descriptors (A + B_down + xscale)
        self.tma_v8b = self.wgmma_mod.create_tma_descriptors_v8b(
            self.inter_fp8, lw["down_weight_ptrs"],
            self.inter_scale_t, self.H, self.mtp_padded)

        self._current_tma_layer_id = layer_id

    def _maybe_resize(self, max_tokens_needed):
        """Resize buffers if max tokens per expert exceeds current mtp."""
        if max_tokens_needed <= self.mtp:
            return
        print(f"[WGMMA Pipeline] Resizing buffers: mtp {self.mtp} → {max_tokens_needed}")
        self.mtp = max_tokens_needed
        mtp_padded = ((max_tokens_needed + BLOCK_M - 1) // BLOCK_M) * BLOCK_M
        self.mtp_padded = mtp_padded

        E = self.E
        H = self.H
        N = self.N
        dev = self.device
        num_k_h = H // QUANT_BLOCK
        num_k_n = N // QUANT_BLOCK

        self.act_buf_bf16 = torch.zeros(E, mtp_padded, H, dtype=torch.bfloat16, device=dev)
        self.act_buf_fp8 = torch.zeros(E, mtp_padded, H, dtype=torch.uint8, device=dev)
        self.act_scale_t = torch.zeros(num_k_h, E * mtp_padded, dtype=torch.float32, device=dev)
        self.gate_out = torch.zeros(E * mtp_padded, N, dtype=torch.bfloat16, device=dev)
        self.up_out = torch.zeros(E * mtp_padded, N, dtype=torch.bfloat16, device=dev)
        self.inter_fp8 = torch.zeros(E, mtp_padded, N, dtype=torch.uint8, device=dev)
        self.inter_scale_t = torch.zeros(num_k_n, E * mtp_padded, dtype=torch.float32, device=dev)
        self.down_out = torch.zeros(E * mtp_padded, H, dtype=torch.bfloat16, device=dev)

        # Recreate TMA descriptors (they encode buffer dimensions)
        self._create_tma_descriptors()

    def _resize_topk_pos_if_needed(self, num_global_tokens):
        """Resize topk_pos if global token count changed."""
        NK = num_global_tokens * self.top_k
        if self.topk_pos.numel() < NK:
            self.topk_pos = torch.full((NK,), -1, dtype=torch.int32, device=self.device)
            self.num_global_tokens = num_global_tokens

    def forward(self, layer_id, global_x, topk_idx, topk_weight):
        """Full WGMMA MoE pipeline forward.

        Args:
            layer_id: Which MoE layer (for weight selection)
            global_x: [G, H] BF16 — all-gathered input
            topk_idx: [G, K] INT32 — expert indices from gating
            topk_weight: [G, K] F32 — routing weights from gating

        Returns:
            global_results: [G, H] BF16 — weighted expert output
        """
        G = global_x.shape[0]
        K_topk = topk_idx.shape[1]
        E = self.E
        H = self.H
        N = self.N
        mtp = self.mtp_padded

        lw = self._layer_weights[layer_id]

        # Recreate TMA descriptors if layer changed (weight pointers differ)
        if getattr(self, '_current_tma_layer_id', None) != layer_id:
            self._create_tma_descriptors(layer_id)

        # Ensure topk_pos is large enough
        self._resize_topk_pos_if_needed(G)

        # ── Step 1: Dispatch scatter 3D ──
        self.act_buf_bf16.zero_()
        self.topk_pos.fill_(-1)

        expert_counts, topk_pos = self.dr_mod.dispatch_scatter_3d(
            global_x, topk_idx.to(torch.int32), self.act_buf_bf16,
            self.expert_start, E, mtp,
            self.expert_counts, self.expert_counters, self.topk_pos)

        tpe = expert_counts.clone()
        max_tpe = expert_counts.max().item()

        if max_tpe == 0:
            return torch.zeros(G, H, dtype=torch.bfloat16, device=self.device)

        # Check if resize needed
        self._maybe_resize(max_tpe)

        # ── Step 2: act_quant_3d (transposed scale output) ──
        act_fp8, act_scale_t = self.fast_mod.act_quant_3d(
            self.act_buf_bf16, tpe)

        # Copy into pre-allocated buffers (for TMA)
        self.act_buf_fp8.copy_(act_fp8)
        self.act_scale_t.copy_(act_scale_t)

        # ── Step 3: v8c fused gate+up WGMMA ──
        num_k_h = H // QUANT_BLOCK
        self.wgmma_mod.grouped_fp8_moe_gemm_v8c(
            self.tma_v8c, tpe,
            lw["gate_scale_ptrs"], lw["up_scale_ptrs"],
            num_k_h,  # w_scale_stride_n
            self.gate_out, self.up_out,
            mtp, N, N, H, E)

        # ── Step 4: fused_silu_quant_3d (transposed scale output) ──
        gate_3d = self.gate_out.view(E, mtp, N)
        up_3d = self.up_out.view(E, mtp, N)
        inter_fp8, inter_scale_t = self.fast_mod.fused_silu_quant_3d(
            gate_3d, up_3d, tpe)

        self.inter_fp8.copy_(inter_fp8)
        self.inter_scale_t.copy_(inter_scale_t)

        # ── Step 5: v8b down WGMMA ──
        num_k_n = N // QUANT_BLOCK
        self.wgmma_mod.grouped_fp8_moe_gemm_v8b(
            self.tma_v8b, tpe,
            lw["down_scale_ptrs"],
            num_k_n,  # w_scale_stride_n
            self.down_out,
            mtp, H, N, E)

        # ── Step 6: Reduce weighted scatter ──
        global_results = torch.zeros(G, H, dtype=torch.bfloat16, device=self.device)
        topk_weights_flat = topk_weight.reshape(-1)
        self.dr_mod.reduce_weighted_scatter(
            self.down_out, topk_pos, topk_weights_flat,
            G, H, K_topk, global_results)

        return global_results
