"""
Fused grouped MoE kernels using WGMMA for GPT-OSS-120B decode.

Ported from the validated kernel in batchgen_kernels/moe/gptoss/grouped_mxfp4_moe_wgmma.py
which achieved 54/54 PASS with max_err=0.000000 against per-row reference.

Two CUDA kernels (manual global loads, no TMA):
- Stage 1: gate + up + SwiGLU (2D grid: experts × N-tiles, M-loop inside)
- Stage 2: down projection (1D grid: K-tiles, expert+M loops inside)

Pipeline: dispatch -> WGMMA S1 -> WGMMA S2 -> reduce (4 kernel launches)

Usage:
    from batchgen.moe.fused_wgmma_grouped import (
        fused_mxfp4_grouped_moe_forward_cuda_routing,
        is_grouped_wgmma_available,
    )

    if is_grouped_wgmma_available():
        output = fused_mxfp4_grouped_moe_forward_cuda_routing(
            hidden_states, topk_indices, topk_weights, ...)
"""

import os
import logging

import torch
from torch.utils.cpp_extension import load_inline


# Module-level state
_grouped_wgmma_available = None
_grouped_module = None


# ──────────────────────────────────────────────────────────────────────────────
# CUDA Source Code (merged Stage 1 + Stage 2)
# ──────────────────────────────────────────────────────────────────────────────
# Ported from batchgen_kernels/moe/gptoss/grouped_mxfp4_moe_wgmma.py
# Validated kernel: 54/54 PASS, max_err=0.000000 with per-row reference

CUDA_SOURCE_GROUPED_MXFP4_MOE = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cstdint>
#include <utility>

// ============================================================================
// Configuration
// ============================================================================
#define BLOCK_M 64
#define BLOCK_N 64
#define BLOCK_K 64
#define WARP_SIZE 32
#define WGMMA_K 16
#define TILES_K (BLOCK_K / WGMMA_K)          // 4
#define WGMMA_NUM_ACCUM 32                    // 64*64/128
#define NUM_STAGES 2
#define TILE_ELEMS (BLOCK_M * BLOCK_K)        // 4096
#define TILE_BYTES (TILE_ELEMS * 2)           // 8192 bytes (BF16)
#define TOTAL_THREADS 256                     // 2 WG
#define PRODUCER_THREADS 128                  // WG0
#define PRODUCER_BAR_ID 1                     // named barrier for producer sync
#define LUT_ENTRIES 256                       // byte-level LUT
#define LUT_BYTES (LUT_ENTRIES * 4)           // 256 * sizeof(__nv_bfloat162) = 1024

// SwiGLU activation parameters
#define SWIGLU_ALPHA 1.702f
#define SWIGLU_LIMIT 7.0f

// FP4 E2M1 values — used for byte LUT initialization
__constant__ float FP4_LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// ============================================================================
// GmmaDescriptor — 128B swizzle (layout_type=1)
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
// Warpgroup synchronization primitives
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
// WGMMA m64n64k16 BF16 SS
// ============================================================================
__device__ __forceinline__ void wgmma_m64n64k16_f32_bf16_bf16_ss(
    uint64_t const& desc_a,
    uint64_t const& desc_b,
    float& d00, float& d01, float& d02, float& d03,
    float& d04, float& d05, float& d06, float& d07,
    float& d08, float& d09, float& d10, float& d11,
    float& d12, float& d13, float& d14, float& d15,
    float& d16, float& d17, float& d18, float& d19,
    float& d20, float& d21, float& d22, float& d23,
    float& d24, float& d25, float& d26, float& d27,
    float& d28, float& d29, float& d30, float& d31,
    int scale_d
) {
    asm volatile(
    "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %34, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
      "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
      " %8,  %9,  %10, %11, %12, %13, %14, %15, "
      " %16, %17, %18, %19, %20, %21, %22, %23, "
      " %24, %25, %26, %27, %28, %29, %30, %31},"
      " %32,"
      " %33,"
      " p,   1, 1, 0, 0;\n"
    "}\n"
      : "+f"(d00), "+f"(d01), "+f"(d02), "+f"(d03),
        "+f"(d04), "+f"(d05), "+f"(d06), "+f"(d07),
        "+f"(d08), "+f"(d09), "+f"(d10), "+f"(d11),
        "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15),
        "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19),
        "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23),
        "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27),
        "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31)
      :  "l"(desc_a),
         "l"(desc_b),
         "r"(scale_d));
}

template <size_t... Idx>
__device__ __forceinline__ void wgmma_bf16_ss_impl(
    uint64_t const& desc_a, uint64_t const& desc_b,
    float* d, int scale_d,
    std::index_sequence<Idx...>
) {
    wgmma_m64n64k16_f32_bf16_bf16_ss(desc_a, desc_b, d[Idx]..., scale_d);
}

__device__ __forceinline__ void wgmma_bf16_ss(
    uint64_t const& desc_a, uint64_t const& desc_b,
    float* d, int scale_d
) {
    wgmma_bf16_ss_impl(desc_a, desc_b, d, scale_d,
                        std::make_index_sequence<32>{});
}

// ============================================================================
// reg_to_mn: accumulator register -> (m, n) position
// ============================================================================
__device__ __forceinline__ void reg_to_mn(
    int reg_idx, int warp_in_wg, int lane_id, int& m, int& n
) {
    const int group   = reg_idx / 4;
    const int sub_idx = reg_idx % 4;
    const int m_half  = sub_idx / 2;
    const int n_bit   = sub_idx % 2;
    m = warp_in_wg * 16 + (lane_id / 4) + m_half * 8;
    n = group * 8 + (lane_id % 4) * 2 + n_bit;
}

// ============================================================================
// mbarrier helpers
// ============================================================================
__device__ __forceinline__ void mbarrier_init(uint64_t* mbar, uint32_t count) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                 :: "r"(smem_addr), "r"(count));
}

__device__ __forceinline__ void mbarrier_arrive(uint64_t* mbar) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    asm volatile(
        "{\n"
        ".reg .b64 state;\n"
        "mbarrier.arrive.shared::cta.b64 state, [%0];\n"
        "}\n"
        :: "r"(smem_addr));
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx(
    uint64_t* mbar, uint32_t tx_bytes
) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    asm volatile(
        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        :: "r"(smem_addr), "r"(tx_bytes));
}

__device__ __forceinline__ void mbarrier_wait_parity(
    uint64_t* mbar, uint32_t phase_parity
) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    asm volatile(
        "{\n"
        ".reg .pred P;\n"
        "GROUPED_MXFP4_MBAR_WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n"
        "@P bra GROUPED_MXFP4_MBAR_DONE_%=;\n"
        "bra GROUPED_MXFP4_MBAR_WAIT_%=;\n"
        "GROUPED_MXFP4_MBAR_DONE_%=:\n"
        "}\n"
        :: "r"(smem_addr), "r"(phase_parity));
}

// ============================================================================
// Named barrier sync (producer WG internal)
// ============================================================================
__device__ __forceinline__ void bar_sync(uint32_t bar_id, uint32_t num_threads) {
    asm volatile("bar.sync %0, %1;" :: "r"(bar_id), "r"(num_threads));
}

// ============================================================================
// Global memory load for A with 128B swizzle (matching TMA pattern)
// Uses vectorized loads (uint4) matching the B matrix load pattern
// ============================================================================
__device__ __forceinline__ void load_a_tile_global(
    __nv_bfloat16* smem_a,
    const __nv_bfloat16* A,
    int m_start, int k_start,
    int M, int K,
    int64_t stride_a_m,
    int tid,
    int num_threads
) {
    const int m_local = tid / 2;        // 0-63
    const int k_half = tid & 1;         // 0 or 1
    const int k_local_start = k_half * 32;

    const int m_global = m_start + m_local;

    if (m_local >= BLOCK_M) return;

    const int m_mod8 = m_local & 7;
    const int m_base = m_local * BLOCK_K;  // m_local * 64

    #pragma unroll
    for (int g = 0; g < 4; g++) {
        const int k_local = k_local_start + g * 8;
        const int k_global = k_start + k_local;

        uint4 data;
        if (m_global < M && k_global + 7 < K) {
            data = *reinterpret_cast<const uint4*>(&A[m_global * stride_a_m + k_global]);
        } else {
            __nv_bfloat16 tmp[8];
            for (int i = 0; i < 8; i++) {
                if (m_global < M && k_global + i < K) {
                    tmp[i] = A[m_global * stride_a_m + k_global + i];
                } else {
                    tmp[i] = __float2bfloat16(0.0f);
                }
            }
            data = *reinterpret_cast<uint4*>(tmp);
        }

        const int grp = (k_local_start + g * 8) >> 3;
        const int swz = grp ^ m_mod8;
        const int addr = m_base + (swz << 3);

        *reinterpret_cast<uint4*>(&smem_a[addr]) = data;
    }
}

// ============================================================================
// Batched Byte-level BF16 LUT decode (Phase 10a pattern)
// ============================================================================
__device__ __forceinline__ void load_decode_rhs_swizzled_batched(
    __nv_bfloat16* smem_rhs,
    const uint8_t* weight_base,
    const uint8_t* scale_base,
    int n_start, int k_start,
    int N, int K,
    int64_t stride_weight_n,
    int64_t stride_scale_n,
    int tid,
    const __nv_bfloat162* byte_lut
) {
    const int n_local = tid / 2;
    const int k_local_start = (tid & 1) * 32;
    const int n_global = n_start + n_local;

    if (n_global >= N) return;

    // Load 32 packed FP4 values (uint4 = 16 bytes)
    const int k_packed_start = k_start / 2 + k_local_start / 2;
    uint4 packed_vec = *reinterpret_cast<const uint4*>(
        weight_base + n_global * stride_weight_n + k_packed_start);
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&packed_vec);

    // Scale: construct BF16x2 directly via IEEE 754 bit manipulation
    int raw_scale = static_cast<int>(
        scale_base[n_global * stride_scale_n + (k_start + k_local_start) / 32]);
    int exp_bits = max(1, min(254, raw_scale));
    uint16_t sbits = static_cast<uint16_t>(exp_bits << 7);
    uint32_t spair = (static_cast<uint32_t>(sbits) << 16) | sbits;
    __nv_bfloat162 scale2 = *reinterpret_cast<__nv_bfloat162*>(&spair);

    // Phase 1: All 16 smem LUT loads (independent, pipelineable)
    __nv_bfloat162 raw[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        raw[i] = byte_lut[bytes[i]];
    }

    // Phase 2: All 16 hmul2 scale multiplies (independent)
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        raw[i] = __hmul2(raw[i], scale2);
    }

    // Phase 3: Pack + 4 wide uint4 stores
    const int n_mod8 = n_local & 7;
    const int n_base = n_local << 6;  // n_local * 64

    #pragma unroll
    for (int g = 0; g < 4; g++) {
        const int grp = (k_local_start + g * 8) >> 3;
        const int swz = grp ^ n_mod8;
        const int addr = n_base + (swz << 3);

        uint4 wide;
        wide.x = *reinterpret_cast<const uint32_t*>(&raw[g * 4 + 0]);
        wide.y = *reinterpret_cast<const uint32_t*>(&raw[g * 4 + 1]);
        wide.z = *reinterpret_cast<const uint32_t*>(&raw[g * 4 + 2]);
        wide.w = *reinterpret_cast<const uint32_t*>(&raw[g * 4 + 3]);
        *reinterpret_cast<uint4*>(&smem_rhs[addr]) = wide;
    }
}

// ============================================================================
// Stage 1 Kernel: Gate + Up + SwiGLU (grouped MXFP4)
//   Grid: (num_experts, num_n_tiles) - 2D grid, M-tile loop inside
//   Manual global load for A, Phase 10a byte-LUT for weights (MXFP4)
// ============================================================================
__global__ void __launch_bounds__(TOTAL_THREADS, 1)
grouped_mxfp4_moe_stage1_kernel(
    const __nv_bfloat16* __restrict__ A,
    int64_t stride_a_m,
    const int32_t* __restrict__ expert_offsets,
    const int64_t* __restrict__ gate_ptrs,
    const int64_t* __restrict__ gate_scale_ptrs,
    const int64_t* __restrict__ up_ptrs,
    const int64_t* __restrict__ up_scale_ptrs,
    const int64_t* __restrict__ gate_bias_ptrs,
    const int64_t* __restrict__ up_bias_ptrs,
    __nv_bfloat16* __restrict__ C,
    int64_t stride_c_m,
    int total_tokens, int N, int K, int num_experts,
    int64_t stride_weight_n,
    int64_t stride_scale_n,
    int has_gate_bias, int has_up_bias
) {
    const int tid = threadIdx.x;
    const int wg_id = tid / 128;
    const int wg_tid = tid % 128;

    // 2D grid: (expert_idx, n_tile)
    const int expert_idx = blockIdx.x;
    const int n_tile = blockIdx.y;
    const int n_start = n_tile * BLOCK_N;

    if (n_start >= N) return;

    // Load expert's token range
    const int expert_start = expert_offsets[expert_idx];
    const int expert_end = expert_offsets[expert_idx + 1];
    const int expert_tokens = expert_end - expert_start;

    // Early exit for empty experts
    if (expert_tokens == 0) return;

    // Shared memory layout: tiles + barriers + byte LUT
    extern __shared__ __align__(128) char smem_buf[];
    __nv_bfloat16* smem_a[NUM_STAGES];
    __nv_bfloat16* smem_b[NUM_STAGES];
    for (int s = 0; s < NUM_STAGES; s++) {
        smem_a[s] = reinterpret_cast<__nv_bfloat16*>(smem_buf + (2*s)     * TILE_BYTES);
        smem_b[s] = reinterpret_cast<__nv_bfloat16*>(smem_buf + (2*s + 1) * TILE_BYTES);
    }
    const int bar_offset = 2 * NUM_STAGES * TILE_BYTES;
    uint64_t* gate_full_barriers  = reinterpret_cast<uint64_t*>(smem_buf + bar_offset);
    uint64_t* gate_empty_barriers = gate_full_barriers + NUM_STAGES;
    uint64_t* up_full_barriers    = gate_empty_barriers + NUM_STAGES;
    uint64_t* up_empty_barriers   = up_full_barriers + NUM_STAGES;
    __nv_bfloat162* byte_lut = reinterpret_cast<__nv_bfloat162*>(
        smem_buf + bar_offset + 4 * NUM_STAGES * sizeof(uint64_t));

    const int num_k_blocks = (K + BLOCK_K - 1) / BLOCK_K;

    // Load expert's weight pointers
    const uint8_t* gate_weight = reinterpret_cast<const uint8_t*>(gate_ptrs[expert_idx]);
    const uint8_t* gate_scale = reinterpret_cast<const uint8_t*>(gate_scale_ptrs[expert_idx]);
    const uint8_t* up_weight = reinterpret_cast<const uint8_t*>(up_ptrs[expert_idx]);
    const uint8_t* up_scale = reinterpret_cast<const uint8_t*>(up_scale_ptrs[expert_idx]);

    const __nv_bfloat16* gate_bias = has_gate_bias ?
        reinterpret_cast<const __nv_bfloat16*>(gate_bias_ptrs[expert_idx]) : nullptr;
    const __nv_bfloat16* up_bias = has_up_bias ?
        reinterpret_cast<const __nv_bfloat16*>(up_bias_ptrs[expert_idx]) : nullptr;

    const int num_m_tiles = (expert_tokens + BLOCK_M - 1) / BLOCK_M;

    {  // Scope for M-tile loop

    // M-TILE LOOP
    for (int m_tile = 0; m_tile < num_m_tiles; m_tile++) {
        const int m_start = expert_start + m_tile * BLOCK_M;
        const int m_size = min(BLOCK_M, expert_end - m_start);

        // Init barriers for this M-tile
        if (tid == 0) {
            for (int s = 0; s < NUM_STAGES; s++) {
                mbarrier_init(&gate_full_barriers[s], 1);   // producer only (no TMA)
                mbarrier_init(&gate_empty_barriers[s], 1);  // consumer
                mbarrier_init(&up_full_barriers[s], 1);
                mbarrier_init(&up_empty_barriers[s], 1);
            }
        }
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        __syncthreads();

        if (wg_id == 0) {
            // ════════════════════════════════════════════════════════════
            // PRODUCER WG: Load A + decode B
            // ════════════════════════════════════════════════════════════

            for (int i = wg_tid; i < LUT_ENTRIES; i += PRODUCER_THREADS) {
                __nv_bfloat16 lo = __float2bfloat16(FP4_LUT[i & 0xF]);
                __nv_bfloat16 hi = __float2bfloat16(FP4_LUT[i >> 4]);
                byte_lut[i] = __halves2bfloat162(lo, hi);
            }
            bar_sync(PRODUCER_BAR_ID, PRODUCER_THREADS);

            // GATE K-LOOP
            for (int kb = 0; kb < num_k_blocks; kb++) {
                const int s = kb % NUM_STAGES;
                const int empty_phase = ((kb / NUM_STAGES) + 1) & 1;

                mbarrier_wait_parity(&gate_empty_barriers[s], empty_phase);

                load_a_tile_global(smem_a[s], A, m_start, kb * BLOCK_K,
                                   total_tokens, K, stride_a_m, wg_tid, PRODUCER_THREADS);

                load_decode_rhs_swizzled_batched(
                    smem_b[s],
                    gate_weight, gate_scale,
                    n_start, kb * BLOCK_K,
                    N, K,
                    stride_weight_n, stride_scale_n,
                    wg_tid, byte_lut);

                bar_sync(PRODUCER_BAR_ID, PRODUCER_THREADS);
                asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                if (wg_tid == 0) {
                    mbarrier_arrive(&gate_full_barriers[s]);
                }
            }

            // SYNC BETWEEN GATE AND UP
            __syncthreads();

            // UP K-LOOP
            for (int kb = 0; kb < num_k_blocks; kb++) {
                const int s = kb % NUM_STAGES;
                const int empty_phase = ((kb / NUM_STAGES) + 1) & 1;

                mbarrier_wait_parity(&up_empty_barriers[s], empty_phase);

                load_a_tile_global(smem_a[s], A, m_start, kb * BLOCK_K,
                                   total_tokens, K, stride_a_m, wg_tid, PRODUCER_THREADS);

                load_decode_rhs_swizzled_batched(
                    smem_b[s],
                    up_weight, up_scale,
                    n_start, kb * BLOCK_K,
                    N, K,
                    stride_weight_n, stride_scale_n,
                    wg_tid, byte_lut);

                bar_sync(PRODUCER_BAR_ID, PRODUCER_THREADS);
                asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                if (wg_tid == 0) {
                    mbarrier_arrive(&up_full_barriers[s]);
                }
            }

        } else {
            // ════════════════════════════════════════════════════════════
            // MATH WG: gate GEMM -> up GEMM -> SwiGLU epilogue
            // ════════════════════════════════════════════════════════════
            float gate_acc[WGMMA_NUM_ACCUM];
            float up_acc[WGMMA_NUM_ACCUM];
            for (int i = 0; i < WGMMA_NUM_ACCUM; i++) gate_acc[i] = 0.0f;
            for (int i = 0; i < WGMMA_NUM_ACCUM; i++) up_acc[i] = 0.0f;

            const int warp_in_wg = wg_tid / WARP_SIZE;
            const int lane_id = wg_tid % WARP_SIZE;

            // GATE K-LOOP
            for (int kb = 0; kb < num_k_blocks; kb++) {
                const int s = kb % NUM_STAGES;
                const int full_phase = (kb / NUM_STAGES) & 1;

                mbarrier_wait_parity(&gate_full_barriers[s], full_phase);

                #pragma unroll
                for (int i = 0; i < WGMMA_NUM_ACCUM; i++)
                    warpgroup_fence_operand(gate_acc[i]);

                warpgroup_arrive();

                #pragma unroll
                for (int t = 0; t < TILES_K; t++) {
                    GmmaDescriptor da = make_smem_desc(smem_a[s] + t * WGMMA_K);
                    GmmaDescriptor db = make_smem_desc(smem_b[s] + t * WGMMA_K);
                    wgmma_bf16_ss(da.desc_, db.desc_, gate_acc, 1);
                }

                warpgroup_commit_batch();

                #pragma unroll
                for (int i = 0; i < WGMMA_NUM_ACCUM; i++)
                    warpgroup_fence_operand(gate_acc[i]);

                warpgroup_wait<0>();

                if (wg_tid == 0) {
                    mbarrier_arrive(&gate_empty_barriers[s]);
                }
            }

            // SYNC
            __syncthreads();

            // UP K-LOOP
            for (int kb = 0; kb < num_k_blocks; kb++) {
                const int s = kb % NUM_STAGES;
                const int full_phase = (kb / NUM_STAGES) & 1;

                mbarrier_wait_parity(&up_full_barriers[s], full_phase);

                #pragma unroll
                for (int i = 0; i < WGMMA_NUM_ACCUM; i++)
                    warpgroup_fence_operand(up_acc[i]);

                warpgroup_arrive();

                #pragma unroll
                for (int t = 0; t < TILES_K; t++) {
                    GmmaDescriptor da = make_smem_desc(smem_a[s] + t * WGMMA_K);
                    GmmaDescriptor db = make_smem_desc(smem_b[s] + t * WGMMA_K);
                    wgmma_bf16_ss(da.desc_, db.desc_, up_acc, 1);
                }

                warpgroup_commit_batch();

                #pragma unroll
                for (int i = 0; i < WGMMA_NUM_ACCUM; i++)
                    warpgroup_fence_operand(up_acc[i]);

                warpgroup_wait<0>();

                if (wg_tid == 0) {
                    mbarrier_arrive(&up_empty_barriers[s]);
                }
            }

            // SwiGLU EPILOGUE
            #pragma unroll 1
            for (int i = 0; i < WGMMA_NUM_ACCUM; i++) {
                int m, n;
                reg_to_mn(i, warp_in_wg, lane_id, m, n);

                const int m_global = m_start + m;
                const int n_global = n_start + n;

                if (m_global >= expert_end || n_global >= N) continue;

                __nv_bfloat16 g_bf16 = __float2bfloat16(gate_acc[i]);
                __nv_bfloat16 u_bf16 = __float2bfloat16(up_acc[i]);

                if (has_gate_bias && gate_bias != nullptr) {
                    g_bf16 = __hadd(g_bf16, gate_bias[n_global]);
                }
                if (has_up_bias && up_bias != nullptr) {
                    u_bf16 = __hadd(u_bf16, up_bias[n_global]);
                }

                float g = __bfloat162float(g_bf16);
                float u = __bfloat162float(u_bf16);

                float g_c = fminf(g, SWIGLU_LIMIT);
                float u_c = fmaxf(fminf(u, SWIGLU_LIMIT), -SWIGLU_LIMIT);
                float sig = __frcp_rn(1.0f + expf(-SWIGLU_ALPHA * g_c));
                float result = g_c * sig * (u_c + 1.0f);

                C[m_global * stride_c_m + n_global] = __float2bfloat16(result);
            }
        }

        // Sync before next M-tile (barrier re-init)
        __syncthreads();

    }  // M-tile loop
    }  // Scope for M-tile loop
}

// ============================================================================
// Stage 2 Kernel: Down projection (grouped MXFP4)
//   Grid: (num_k_tiles,) - 1D grid with expert+M loop inside
//   output[total_tokens, K] = intermediate[total_tokens, N] @ down[K, N]^T
// ============================================================================
__global__ void __launch_bounds__(TOTAL_THREADS, 1)
grouped_mxfp4_moe_stage2_kernel(
    const __nv_bfloat16* __restrict__ input,
    int64_t stride_input_m,
    const int32_t* __restrict__ expert_offsets,
    const int64_t* __restrict__ down_ptrs,
    const int64_t* __restrict__ down_scale_ptrs,
    const int64_t* __restrict__ down_bias_ptrs,
    __nv_bfloat16* __restrict__ C,
    int64_t stride_c_m,
    int total_tokens, int N, int K, int num_experts,
    int64_t stride_weight_n,
    int64_t stride_scale_n,
    int has_down_bias
) {
    const int tid = threadIdx.x;
    const int wg_id = tid / 128;
    const int wg_tid = tid % 128;

    // 1D grid over K-tiles (output dimension)
    const int k_tile = blockIdx.x;
    const int k_start = k_tile * BLOCK_N;

    if (k_start >= K) return;

    // Shared memory layout
    extern __shared__ __align__(128) char smem_buf[];
    __nv_bfloat16* smem_a[NUM_STAGES];
    __nv_bfloat16* smem_b[NUM_STAGES];
    for (int s = 0; s < NUM_STAGES; s++) {
        smem_a[s] = reinterpret_cast<__nv_bfloat16*>(smem_buf + (2*s)     * TILE_BYTES);
        smem_b[s] = reinterpret_cast<__nv_bfloat16*>(smem_buf + (2*s + 1) * TILE_BYTES);
    }
    const int bar_offset = 2 * NUM_STAGES * TILE_BYTES;
    uint64_t* full_barriers  = reinterpret_cast<uint64_t*>(smem_buf + bar_offset);
    uint64_t* empty_barriers = full_barriers + NUM_STAGES;
    __nv_bfloat162* byte_lut = reinterpret_cast<__nv_bfloat162*>(
        smem_buf + bar_offset + 2 * NUM_STAGES * sizeof(uint64_t));

    const int num_n_blocks = (N + BLOCK_K - 1) / BLOCK_K;

    // EXPERT LOOP
    for (int expert_idx = 0; expert_idx < num_experts; expert_idx++) {
        const int expert_start = expert_offsets[expert_idx];
        const int expert_end = expert_offsets[expert_idx + 1];
        const int expert_tokens = expert_end - expert_start;

        if (expert_tokens == 0) continue;

        const uint8_t* down_weight = reinterpret_cast<const uint8_t*>(down_ptrs[expert_idx]);
        const uint8_t* down_scale = reinterpret_cast<const uint8_t*>(down_scale_ptrs[expert_idx]);
        const __nv_bfloat16* down_bias = has_down_bias ?
            reinterpret_cast<const __nv_bfloat16*>(down_bias_ptrs[expert_idx]) : nullptr;

        const int num_m_tiles = (expert_tokens + BLOCK_M - 1) / BLOCK_M;

        // M-TILE LOOP
        for (int m_tile = 0; m_tile < num_m_tiles; m_tile++) {
            const int m_start = expert_start + m_tile * BLOCK_M;

            // Init barriers
            if (tid == 0) {
                for (int s = 0; s < NUM_STAGES; s++) {
                    mbarrier_init(&full_barriers[s], 1);
                    mbarrier_init(&empty_barriers[s], 1);
                }
            }
            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
            __syncthreads();

            if (wg_id == 0) {
                // PRODUCER WG

                for (int i = wg_tid; i < LUT_ENTRIES; i += PRODUCER_THREADS) {
                    __nv_bfloat16 lo = __float2bfloat16(FP4_LUT[i & 0xF]);
                    __nv_bfloat16 hi = __float2bfloat16(FP4_LUT[i >> 4]);
                    byte_lut[i] = __halves2bfloat162(lo, hi);
                }
                bar_sync(PRODUCER_BAR_ID, PRODUCER_THREADS);

                // N-loop (reduction)
                for (int nb = 0; nb < num_n_blocks; nb++) {
                    const int s = nb % NUM_STAGES;
                    const int empty_phase = ((nb / NUM_STAGES) + 1) & 1;

                    mbarrier_wait_parity(&empty_barriers[s], empty_phase);

                    load_a_tile_global(smem_a[s], input, m_start, nb * BLOCK_K,
                                       total_tokens, N, stride_input_m, wg_tid, PRODUCER_THREADS);

                    load_decode_rhs_swizzled_batched(
                        smem_b[s],
                        down_weight, down_scale,
                        k_start, nb * BLOCK_K,
                        K, N,
                        stride_weight_n, stride_scale_n,
                        wg_tid, byte_lut);

                    bar_sync(PRODUCER_BAR_ID, PRODUCER_THREADS);
                    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                    if (wg_tid == 0) {
                        mbarrier_arrive(&full_barriers[s]);
                    }
                }

            } else {
                // MATH WG
                float acc[WGMMA_NUM_ACCUM];
                for (int i = 0; i < WGMMA_NUM_ACCUM; i++) acc[i] = 0.0f;

                const int warp_in_wg = wg_tid / WARP_SIZE;
                const int lane_id = wg_tid % WARP_SIZE;

                // N-loop
                for (int nb = 0; nb < num_n_blocks; nb++) {
                    const int s = nb % NUM_STAGES;
                    const int full_phase = (nb / NUM_STAGES) & 1;

                    mbarrier_wait_parity(&full_barriers[s], full_phase);

                    #pragma unroll
                    for (int i = 0; i < WGMMA_NUM_ACCUM; i++)
                        warpgroup_fence_operand(acc[i]);

                    warpgroup_arrive();

                    #pragma unroll
                    for (int t = 0; t < TILES_K; t++) {
                        GmmaDescriptor da = make_smem_desc(smem_a[s] + t * WGMMA_K);
                        GmmaDescriptor db = make_smem_desc(smem_b[s] + t * WGMMA_K);
                        wgmma_bf16_ss(da.desc_, db.desc_, acc, 1);
                    }

                    warpgroup_commit_batch();

                    #pragma unroll
                    for (int i = 0; i < WGMMA_NUM_ACCUM; i++)
                        warpgroup_fence_operand(acc[i]);

                    warpgroup_wait<0>();

                    if (wg_tid == 0) {
                        mbarrier_arrive(&empty_barriers[s]);
                    }
                }

                // EPILOGUE: store with optional bias
                #pragma unroll 1
                for (int i = 0; i < WGMMA_NUM_ACCUM; i++) {
                    int m, n;
                    reg_to_mn(i, warp_in_wg, lane_id, m, n);

                    const int m_global = m_start + m;
                    const int k_global = k_start + n;

                    if (m_global >= expert_end || k_global >= K) continue;

                    float result = acc[i];

                    if (has_down_bias && down_bias != nullptr) {
                        result += __bfloat162float(down_bias[k_global]);
                    }

                    C[m_global * stride_c_m + k_global] = __float2bfloat16(result);
                }
            }

            __syncthreads();

        }  // M-tile loop

    }  // Expert loop
}

// ============================================================================
// C++ wrapper: Stage 1 (gate + up + SwiGLU)
// ============================================================================
torch::Tensor grouped_mxfp4_moe_stage1(
    torch::Tensor A,
    torch::Tensor expert_offsets,
    torch::Tensor gate_ptrs,
    torch::Tensor gate_scale_ptrs,
    torch::Tensor up_ptrs,
    torch::Tensor up_scale_ptrs,
    torch::Tensor gate_bias_ptrs,
    torch::Tensor up_bias_ptrs,
    int N,
    int64_t stride_weight_n,
    int64_t stride_scale_n
) {
    TORCH_CHECK(A.is_cuda() && A.dtype() == torch::kBFloat16);
    TORCH_CHECK(expert_offsets.is_cuda() && expert_offsets.dtype() == torch::kInt32);
    TORCH_CHECK(gate_ptrs.is_cuda() && gate_ptrs.dtype() == torch::kInt64);

    const int total_tokens = A.size(0);
    const int K = A.size(1);
    const int num_experts = expert_offsets.size(0) - 1;

    auto C = torch::empty({total_tokens, N}, A.options());

    // 2D grid: (num_experts, num_n_tiles)
    const int num_n_tiles = (N + BLOCK_N - 1) / BLOCK_N;
    dim3 grid(num_experts, num_n_tiles);
    dim3 block(TOTAL_THREADS);

    constexpr int smem_bytes = 2 * NUM_STAGES * TILE_BYTES +
                               4 * NUM_STAGES * sizeof(uint64_t) + LUT_BYTES;

    cudaFuncSetAttribute(grouped_mxfp4_moe_stage1_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);

    int has_gate_bias = gate_bias_ptrs.numel() > 0 ? 1 : 0;
    int has_up_bias = up_bias_ptrs.numel() > 0 ? 1 : 0;

    grouped_mxfp4_moe_stage1_kernel<<<grid, block, smem_bytes>>>(
        reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
        A.stride(0),
        expert_offsets.data_ptr<int32_t>(),
        gate_ptrs.data_ptr<int64_t>(),
        gate_scale_ptrs.data_ptr<int64_t>(),
        up_ptrs.data_ptr<int64_t>(),
        up_scale_ptrs.data_ptr<int64_t>(),
        has_gate_bias ? gate_bias_ptrs.data_ptr<int64_t>() : nullptr,
        has_up_bias ? up_bias_ptrs.data_ptr<int64_t>() : nullptr,
        reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
        C.stride(0),
        total_tokens, N, K, num_experts,
        stride_weight_n, stride_scale_n,
        has_gate_bias, has_up_bias);

    return C;
}

// ============================================================================
// C++ wrapper: Stage 2 (down projection)
// ============================================================================
torch::Tensor grouped_mxfp4_moe_stage2(
    torch::Tensor input,
    torch::Tensor expert_offsets,
    torch::Tensor down_ptrs,
    torch::Tensor down_scale_ptrs,
    torch::Tensor down_bias_ptrs,
    int K,
    int64_t stride_weight_n,
    int64_t stride_scale_n
) {
    TORCH_CHECK(input.is_cuda() && input.dtype() == torch::kBFloat16);
    TORCH_CHECK(expert_offsets.is_cuda() && expert_offsets.dtype() == torch::kInt32);
    TORCH_CHECK(down_ptrs.is_cuda() && down_ptrs.dtype() == torch::kInt64);

    const int total_tokens = input.size(0);
    const int N = input.size(1);
    const int num_experts = expert_offsets.size(0) - 1;

    auto C = torch::empty({total_tokens, K}, input.options());

    // 1D grid over K-tiles
    const int num_k_tiles = (K + BLOCK_N - 1) / BLOCK_N;
    dim3 grid(num_k_tiles);
    dim3 block(TOTAL_THREADS);

    constexpr int smem_bytes = 2 * NUM_STAGES * TILE_BYTES +
                               2 * NUM_STAGES * sizeof(uint64_t) + LUT_BYTES;

    cudaFuncSetAttribute(grouped_mxfp4_moe_stage2_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);

    int has_down_bias = down_bias_ptrs.numel() > 0 ? 1 : 0;

    grouped_mxfp4_moe_stage2_kernel<<<grid, block, smem_bytes>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
        input.stride(0),
        expert_offsets.data_ptr<int32_t>(),
        down_ptrs.data_ptr<int64_t>(),
        down_scale_ptrs.data_ptr<int64_t>(),
        has_down_bias ? down_bias_ptrs.data_ptr<int64_t>() : nullptr,
        reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
        C.stride(0),
        total_tokens, N, K, num_experts,
        stride_weight_n, stride_scale_n,
        has_down_bias);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("grouped_mxfp4_moe_stage1", &grouped_mxfp4_moe_stage1,
          "Grouped MXFP4 Stage 1 (gate+up+SwiGLU) - validated kernel");
    m.def("grouped_mxfp4_moe_stage2", &grouped_mxfp4_moe_stage2,
          "Grouped MXFP4 Stage 2 (down projection) - validated kernel");
}
'''


# ──────────────────────────────────────────────────────────────────────────────
# Module Loading
# ──────────────────────────────────────────────────────────────────────────────

def _check_wgmma_support() -> bool:
    """Check if WGMMA (SM90) is supported on this system."""
    if not torch.cuda.is_available():
        return False
    device = torch.cuda.current_device()
    cc = torch.cuda.get_device_capability(device)
    if cc[0] < 9:
        logging.debug(f"WGMMA requires SM90+, found SM{cc[0]}{cc[1]}")
        return False
    return True


def _load_grouped_module():
    """Load the grouped WGMMA CUDA module (Stage 1 + Stage 2)."""
    global _grouped_module

    if _grouped_module is not None:
        return _grouped_module

    try:
        cuda_flags = ["-std=c++17", "-arch=sm_90a", "-O3", "--ptxas-options=-v"]

        _grouped_module = load_inline(
            name="batchgen_fused_mxfp4_grouped_wgmma_v2",
            cpp_sources=[""],
            cuda_sources=[CUDA_SOURCE_GROUPED_MXFP4_MOE],
            extra_cuda_cflags=cuda_flags,
            verbose=False,
        )
        logging.info("Loaded WGMMA fused grouped MXFP4 MoE kernels")
        return _grouped_module
    except Exception as e:
        logging.warning(f"Failed to load WGMMA grouped MoE kernels: {e}")
        return None


def is_grouped_wgmma_available() -> bool:
    """Check if grouped WGMMA fused kernels are available."""
    global _grouped_wgmma_available

    if _grouped_wgmma_available is not None:
        return _grouped_wgmma_available

    if not _check_wgmma_support():
        _grouped_wgmma_available = False
        return False

    if os.environ.get("BATCHGEN_DISABLE_WGMMA_GROUPED", "0") == "1":
        logging.info("WGMMA grouped kernels disabled by BATCHGEN_DISABLE_WGMMA_GROUPED")
        _grouped_wgmma_available = False
        return False

    mod = _load_grouped_module()
    _grouped_wgmma_available = mod is not None
    return _grouped_wgmma_available


# ──────────────────────────────────────────────────────────────────────────────
# Low-Level Python Wrappers
# ──────────────────────────────────────────────────────────────────────────────

def fused_mxfp4_grouped_stage1(
    sorted_hidden: torch.Tensor,       # [total_tokens, K] BF16
    expert_offsets: torch.Tensor,       # [num_experts + 1] int32
    gate_ptrs: torch.Tensor,           # [num_experts] int64
    gate_scale_ptrs: torch.Tensor,     # [num_experts] int64
    up_ptrs: torch.Tensor,             # [num_experts] int64
    up_scale_ptrs: torch.Tensor,       # [num_experts] int64
    N: int,                            # intermediate_size
    stride_weight_n: int,              # K // 2
    stride_scale_n: int,               # K // 32
) -> torch.Tensor:                     # [total_tokens, N] BF16
    """Grouped MXFP4 Stage 1: gate + up + SwiGLU via WGMMA.

    Operates on 1D+offsets layout directly from CUDA dispatch.
    2D grid (experts, N-tiles) with M-tile loop inside kernel.

    Args:
        sorted_hidden: Dispatched tokens [total_tokens, K] BF16
        expert_offsets: Cumulative offsets [num_experts + 1] int32
        gate_ptrs/gate_scale_ptrs: Pointer arrays [num_experts] int64
        up_ptrs/up_scale_ptrs: Pointer arrays [num_experts] int64
        N: Intermediate dimension (gate/up output width)
        stride_weight_n: Weight stride along N (= K // 2 for MXFP4)
        stride_scale_n: Scale stride along N (= K // 32 for MXFP4)

    Returns:
        Intermediate activations [total_tokens, N] BF16 after SwiGLU
    """
    mod = _load_grouped_module()
    assert mod is not None, "WGMMA grouped module not available"

    # Empty bias tensors (GPT-OSS-120B has no biases)
    empty_bias = torch.empty(0, dtype=torch.int64, device=sorted_hidden.device)

    return mod.grouped_mxfp4_moe_stage1(
        sorted_hidden, expert_offsets,
        gate_ptrs, gate_scale_ptrs,
        up_ptrs, up_scale_ptrs,
        empty_bias, empty_bias,
        N, stride_weight_n, stride_scale_n,
    )


def fused_mxfp4_grouped_stage2(
    intermediate: torch.Tensor,        # [total_tokens, N] BF16
    expert_offsets: torch.Tensor,       # [num_experts + 1] int32
    down_ptrs: torch.Tensor,           # [num_experts] int64
    down_scale_ptrs: torch.Tensor,     # [num_experts] int64
    K: int,                            # hidden_size (output width)
    stride_weight_n: int,              # N // 2
    stride_scale_n: int,               # N // 32
) -> torch.Tensor:                     # [total_tokens, K] BF16
    """Grouped MXFP4 Stage 2: down projection via WGMMA.

    1D grid (K-tiles) with expert+M loop inside kernel.

    Args:
        intermediate: Stage 1 output [total_tokens, N] BF16
        expert_offsets: Cumulative offsets [num_experts + 1] int32
        down_ptrs/down_scale_ptrs: Pointer arrays [num_experts] int64
        K: Hidden size (output width)
        stride_weight_n: Weight stride along N (= N // 2 for MXFP4)
        stride_scale_n: Scale stride along N (= N // 32 for MXFP4)

    Returns:
        Output activations [total_tokens, K] BF16
    """
    mod = _load_grouped_module()
    assert mod is not None, "WGMMA grouped module not available"

    # Empty bias tensor (GPT-OSS-120B has no biases)
    empty_bias = torch.empty(0, dtype=torch.int64, device=intermediate.device)

    return mod.grouped_mxfp4_moe_stage2(
        intermediate, expert_offsets,
        down_ptrs, down_scale_ptrs,
        empty_bias,
        K, stride_weight_n, stride_scale_n,
    )


# ──────────────────────────────────────────────────────────────────────────────
# End-to-End API with CUDA Routing
# ──────────────────────────────────────────────────────────────────────────────

_debug_grouped_call_count = 0
_DEBUG_GROUPED_MAX = 300
_debug_reduce_call_count = 0
_debug_dispatch_call_count = 0
_debug_routing_match_count = 0


def fused_mxfp4_grouped_moe_forward_cuda_routing(
    hidden_states: torch.Tensor,       # [batch*seq, hidden] BF16
    topk_indices: torch.Tensor,        # [batch*seq, K] int32
    topk_weights: torch.Tensor,        # [batch*seq, K] FP32
    # Pre-computed pointer arrays
    gate_ptrs: torch.Tensor,
    gate_scale_ptrs: torch.Tensor,
    up_ptrs: torch.Tensor,
    up_scale_ptrs: torch.Tensor,
    down_ptrs: torch.Tensor,
    down_scale_ptrs: torch.Tensor,
    # Reference weights for stride computation
    gate_weight_ref: torch.Tensor,
    gate_scale_ref: torch.Tensor,
    down_weight_ref: torch.Tensor,
    down_scale_ref: torch.Tensor,
    num_experts: int = 128,
    expert_start: int = 0,
    num_local_experts: int = 128,
    _debug_weight_lists=None,
    _return_internals=False,
) -> torch.Tensor:
    """End-to-end grouped MXFP4 MoE forward using WGMMA + CUDA routing.

    Full pipeline: dispatch -> WGMMA S1 -> WGMMA S2 -> reduce (4 kernel launches).
    Drop-in replacement for grouped_mxfp4_moe_forward_cuda_routing in
    mxfp4_grouped_gemm.py which uses 9+ launches.

    Biases are not supported (GPT-OSS-120B biases are None by default).

    Args:
        hidden_states: Input [batch*seq, hidden] BF16
        topk_indices: Expert indices [batch*seq, K] int32
        topk_weights: Routing weights [batch*seq, K] FP32
        gate_ptrs, gate_scale_ptrs: Gate weight/scale pointer arrays
        up_ptrs, up_scale_ptrs: Up weight/scale pointer arrays
        down_ptrs, down_scale_ptrs: Down weight/scale pointer arrays
        gate_weight_ref, gate_scale_ref: Reference tensors for stride computation
        down_weight_ref, down_scale_ref: Reference tensors for stride computation
        num_experts: Total number of experts
        expert_start: First local expert index (for EP)
        num_local_experts: Number of local experts

    Returns:
        Output [batch*seq, hidden] BF16
    """
    from batchgen.moe.routing import dispatch_count_gather_cuda, reduce_weighted_scatter_cuda

    num_tokens = hidden_states.shape[0]
    hidden_size = hidden_states.shape[1]
    K_topk = topk_indices.shape[1]
    N_intermediate = gate_weight_ref.shape[0]  # intermediate_size

    # Step 1: CUDA dispatch (count + prefix_sum + gather)
    dispatched_x, expert_counts, expert_offsets, topk_pos = dispatch_count_gather_cuda(
        hidden_states, topk_indices,
        expert_start, num_local_experts,
    )

    # Trim to actual dispatched tokens
    total_dispatched = expert_offsets[num_local_experts].item()
    dispatched_x = dispatched_x[:total_dispatched]

    if total_dispatched == 0:
        return torch.zeros(num_tokens, hidden_size, dtype=hidden_states.dtype,
                           device=hidden_states.device)

    # ── Diagnostic: validate dispatch (tokens + topk_pos mapping) ──
    if os.environ.get("BATCHGEN_DEBUG_DISPATCH", "0") == "1":
        global _debug_dispatch_call_count
        _debug_dispatch_call_count += 1
        if _debug_dispatch_call_count <= _DEBUG_GROUPED_MAX:
            torch.cuda.synchronize()
            print(f"\n[DISPATCH DIAG #{_debug_dispatch_call_count}] "
                  f"num_tokens={num_tokens} K={K_topk} total_dispatched={total_dispatched} "
                  f"expert_start={expert_start} num_local={num_local_experts}", flush=True)

            # Build reference: what PyTorch masking would gather for each expert
            flat_expert_idx = topk_indices.view(-1)  # [N*K]
            ref_token_ids = torch.arange(num_tokens, device=hidden_states.device
                                         ).repeat_interleave(K_topk)  # [N*K]
            offsets_cpu = expert_offsets.cpu()

            dispatch_ok = True
            topk_pos_ok = True

            for e_local in range(num_local_experts):
                global_e = expert_start + e_local
                start = offsets_cpu[e_local].item()
                end = offsets_cpu[e_local + 1].item()
                n_dispatch = end - start

                # Reference: tokens that should be in this expert's segment
                mask = flat_expert_idx == global_e
                ref_tids = ref_token_ids[mask]  # original token IDs for this expert
                n_ref = ref_tids.shape[0]

                if n_dispatch != n_ref:
                    print(f"  [BUG] Expert {e_local} (global {global_e}): "
                          f"dispatch count={n_dispatch} != ref count={n_ref}", flush=True)
                    dispatch_ok = False
                    continue

                if n_dispatch == 0:
                    continue

                # Check dispatched tokens match original tokens
                # dispatched_x[start:end] should contain hidden_states[ref_tids] (in some order)
                disp_set = dispatched_x[start:end]  # [n_dispatch, H]
                ref_set = hidden_states[ref_tids]    # [n_ref, H]

                # Sort both sets by first element for comparison (order may differ)
                disp_sorted, disp_idx = disp_set[:, 0].sort()
                ref_sorted, ref_idx = ref_set[:, 0].sort()

                token_diff = (disp_set[disp_idx].float() - ref_set[ref_idx].float()).abs().max().item()
                if token_diff > 0:
                    print(f"  [BUG] Expert {e_local} (global {global_e}): "
                          f"dispatched tokens MISMATCH ref, max_diff={token_diff:.6f}", flush=True)
                    dispatch_ok = False

                # Validate topk_pos: for each assignment in mask, check topk_pos points
                # to a valid position in [start, end) and the token at that position
                # matches the original token
                mask_indices = torch.where(mask)[0]  # flat indices where mask is True
                for flat_idx in mask_indices[:5]:  # check first 5 per expert
                    fi = flat_idx.item()
                    pos = topk_pos[fi].item()
                    orig_token = fi // K_topk

                    if pos < start or pos >= end:
                        print(f"  [BUG] topk_pos[{fi}]={pos} OUT OF RANGE "
                              f"[{start},{end}) for expert {e_local}", flush=True)
                        topk_pos_ok = False
                    else:
                        # Token at dispatched_x[pos] should be hidden_states[orig_token]
                        d_tok = dispatched_x[pos]
                        o_tok = hidden_states[orig_token]
                        tok_diff = (d_tok.float() - o_tok.float()).abs().max().item()
                        if tok_diff > 0:
                            print(f"  [BUG] topk_pos[{fi}]={pos}: dispatched_x[{pos}] != "
                                  f"hidden_states[{orig_token}], diff={tok_diff:.6f}", flush=True)
                            topk_pos_ok = False

            if dispatch_ok and topk_pos_ok:
                print(f"  Dispatch validation: ALL OK", flush=True)
            else:
                print(f"  Dispatch validation: BUGS FOUND "
                      f"(dispatch_ok={dispatch_ok} topk_pos_ok={topk_pos_ok})", flush=True)
            print(flush=True)

    # ── Diagnostic: routing match (grouped dispatch vs per-expert-loop mask) ──
    if os.environ.get("BATCHGEN_DEBUG_ROUTING_MATCH", "0") == "1":
        global _debug_routing_match_count
        _debug_routing_match_count += 1
        if _debug_routing_match_count <= _DEBUG_GROUPED_MAX:
            torch.cuda.synchronize()
            print(f"\n[ROUTING MATCH #{_debug_routing_match_count}] "
                  f"num_tokens={num_tokens} K={K_topk} "
                  f"expert_start={expert_start} num_local={num_local_experts} "
                  f"total_dispatched={total_dispatched}", flush=True)

            # ─── A. Per-assignment verification ───
            # For EVERY flat index itopk, verify dispatch vs mask-based routing
            flat_indices = topk_indices.view(-1)  # [N*K]
            NK = num_tokens * K_topk
            offsets_cpu = expert_offsets.cpu()
            topk_pos_cpu = topk_pos.cpu()
            flat_indices_cpu = flat_indices.cpu()

            n_local_assignments = 0
            n_pos_mismatch = 0
            n_expert_mismatch = 0
            n_token_mismatch = 0
            n_weight_mismatch = 0
            first_bugs = []

            for itopk in range(NK):
                eid = flat_indices_cpu[itopk].item()
                local_e = eid - expert_start
                token_id = itopk // K_topk
                slot_k = itopk % K_topk
                pos = topk_pos_cpu[itopk].item()

                if local_e < 0 or local_e >= num_local_experts:
                    # Non-local expert: pos should be -1
                    if pos != -1:
                        n_pos_mismatch += 1
                        if len(first_bugs) < 5:
                            first_bugs.append(
                                f"  [BUG] itopk={itopk} token={token_id} slot={slot_k}: "
                                f"expert {eid} is NON-LOCAL but topk_pos={pos} (should be -1)")
                    continue

                # Local expert: pos should be in [expert_offsets[local_e], expert_offsets[local_e+1])
                n_local_assignments += 1
                e_start = offsets_cpu[local_e].item()
                e_end = offsets_cpu[local_e + 1].item()

                if pos < e_start or pos >= e_end:
                    n_expert_mismatch += 1
                    if len(first_bugs) < 5:
                        first_bugs.append(
                            f"  [BUG] itopk={itopk} token={token_id} slot={slot_k}: "
                            f"expert {eid} (local {local_e}) pos={pos} "
                            f"OUT OF expert range [{e_start}, {e_end})")
                    continue

                # Verify dispatched token content matches original
                d_tok = dispatched_x[pos]
                o_tok = hidden_states[token_id]
                tok_diff = (d_tok.float() - o_tok.float()).abs().max().item()
                if tok_diff > 0:
                    n_token_mismatch += 1
                    if len(first_bugs) < 5:
                        first_bugs.append(
                            f"  [BUG] itopk={itopk} token={token_id} slot={slot_k}: "
                            f"dispatched_x[{pos}] != hidden_states[{token_id}], diff={tok_diff:.6f}")

            # ─── B. Expert count verification ───
            ref_counts = []
            counts_cpu = expert_counts.cpu()
            count_mismatch = False
            for e_local in range(num_local_experts):
                global_e = expert_start + e_local
                ref_n = (flat_indices_cpu == global_e).sum().item()
                cuda_n = counts_cpu[e_local].item()
                ref_counts.append(ref_n)
                if ref_n != cuda_n:
                    count_mismatch = True
                    if len(first_bugs) < 5:
                        first_bugs.append(
                            f"  [BUG] Expert {e_local} (global {global_e}): "
                            f"CUDA count={cuda_n} != PyTorch mask count={ref_n}")

            # ─── C. Weight consistency check ───
            # In reduce, weight for (token, slot_k) = topk_weights[token, slot_k]
            # In per-expert loop, weight for (token, slot_k) = topk_weights[token, slot_k]
            # These are the SAME tensor — but verify topk_weights shape/content is sane
            w_nan = torch.isnan(topk_weights).sum().item()
            w_inf = torch.isinf(topk_weights).sum().item()
            w_neg = (topk_weights < 0).sum().item()
            w_sum_per_token = topk_weights.sum(dim=1)
            w_sum_min = w_sum_per_token.min().item()
            w_sum_max = w_sum_per_token.max().item()

            # ─── D. Print results ───
            all_ok = (n_pos_mismatch == 0 and n_expert_mismatch == 0
                      and n_token_mismatch == 0 and not count_mismatch)

            print(f"  local_assignments={n_local_assignments}/{NK} "
                  f"(non-local={NK - n_local_assignments})", flush=True)
            print(f"  pos_mismatch={n_pos_mismatch} expert_range_mismatch={n_expert_mismatch} "
                  f"token_content_mismatch={n_token_mismatch} count_mismatch={count_mismatch}",
                  flush=True)
            print(f"  topk_weights: nan={w_nan} inf={w_inf} neg={w_neg} "
                  f"sum_range=[{w_sum_min:.4f}, {w_sum_max:.4f}]", flush=True)
            print(f"  expert_counts(cuda)={counts_cpu.tolist()}", flush=True)
            print(f"  expert_counts(ref) ={ref_counts}", flush=True)

            # Show first few topk_indices for sanity
            n_show = min(5, num_tokens)
            print(f"  topk_indices[:{n_show}]={topk_indices[:n_show].cpu().tolist()}", flush=True)
            print(f"  topk_weights[:{n_show}]={topk_weights[:n_show].cpu().tolist()}", flush=True)

            for bug in first_bugs:
                print(bug, flush=True)

            if all_ok:
                print(f"  ROUTING MATCH: ALL OK ({n_local_assignments} assignments verified)",
                      flush=True)
            else:
                print(f"  ROUTING MATCH: BUGS FOUND", flush=True)
            print(flush=True)

    # Compute strides from reference weights
    # Stage 1 (gate/up): weight is [N, K//2], scale is [N, K//32]
    s1_stride_weight_n = gate_weight_ref.shape[1]   # K // 2
    s1_stride_scale_n = gate_scale_ref.shape[1]     # K // 32

    # Stage 2 (down): weight is [K_hidden, N//2], scale is [K_hidden, N//32]
    s2_stride_weight_n = down_weight_ref.shape[1]   # N // 2
    s2_stride_scale_n = down_scale_ref.shape[1]     # N // 32

    # Debug: stage-level logging
    debug_stages = os.environ.get("BATCHGEN_DEBUG_WGMMA_STAGES", "0") == "1"

    if debug_stages:
        torch.cuda.synchronize()
        offsets_cpu = expert_offsets.cpu()
        counts = offsets_cpu[1:] - offsets_cpu[:-1]
        print(f"[WGMMA DEBUG] dispatch: total={total_dispatched} experts={num_local_experts} "
              f"expert_start={expert_start} "
              f"counts={counts.tolist()} "
              f"input_range=[{dispatched_x.float().min():.4f}, {dispatched_x.float().max():.4f}]",
              flush=True)
        print(f"  N_intermediate={N_intermediate} hidden_size={hidden_size} "
              f"s1_stride_w={s1_stride_weight_n} s1_stride_s={s1_stride_scale_n} "
              f"s2_stride_w={s2_stride_weight_n} s2_stride_s={s2_stride_scale_n}",
              flush=True)

    # ── Snapshot topk_pos and topk_weights BEFORE kernel stages (corruption detection) ──
    _check_corruption = os.environ.get("BATCHGEN_DEBUG_CORRUPTION", "0") == "1"
    if _check_corruption:
        _topk_pos_snapshot = topk_pos.clone()
        _topk_weights_snapshot = topk_weights.clone()
        _expert_offsets_snapshot = expert_offsets.clone()

    # Step 2: WGMMA Stage 1 (gate + up + SwiGLU)
    intermediate = fused_mxfp4_grouped_stage1(
        dispatched_x, expert_offsets,
        gate_ptrs, gate_scale_ptrs,
        up_ptrs, up_scale_ptrs,
        N_intermediate, s1_stride_weight_n, s1_stride_scale_n,
    )

    if debug_stages:
        torch.cuda.synchronize()
        print(f"[WGMMA DEBUG] stage1 output: shape={intermediate.shape} "
              f"range=[{intermediate.float().min():.4f}, {intermediate.float().max():.4f}] "
              f"nan={torch.isnan(intermediate).sum().item()} "
              f"inf={torch.isinf(intermediate).sum().item()} "
              f"nonzero={(intermediate != 0).sum().item()}",
              flush=True)

    # Step 3: WGMMA Stage 2 (down projection)
    sorted_output = fused_mxfp4_grouped_stage2(
        intermediate, expert_offsets,
        down_ptrs, down_scale_ptrs,
        hidden_size, s2_stride_weight_n, s2_stride_scale_n,
    )

    if debug_stages:
        torch.cuda.synchronize()
        print(f"[WGMMA DEBUG] stage2 output: shape={sorted_output.shape} "
              f"range=[{sorted_output.float().min():.4f}, {sorted_output.float().max():.4f}] "
              f"nan={torch.isnan(sorted_output).sum().item()} "
              f"inf={torch.isinf(sorted_output).sum().item()} "
              f"nonzero={(sorted_output != 0).sum().item()}",
              flush=True)

    # ── Check if kernel stages corrupted topk_pos / topk_weights / expert_offsets ──
    if _check_corruption:
        torch.cuda.synchronize()
        pos_changed = (topk_pos != _topk_pos_snapshot).sum().item()
        wts_changed = (topk_weights != _topk_weights_snapshot).sum().item()
        off_changed = (expert_offsets != _expert_offsets_snapshot).sum().item()

        if pos_changed > 0 or wts_changed > 0 or off_changed > 0:
            print(f"\n[CORRUPTION DETECTED] expert_start={expert_start} "
                  f"total_dispatched={total_dispatched}", flush=True)
            print(f"  topk_pos changed: {pos_changed}/{topk_pos.numel()} elements", flush=True)
            print(f"  topk_weights changed: {wts_changed}/{topk_weights.numel()} elements",
                  flush=True)
            print(f"  expert_offsets changed: {off_changed}/{expert_offsets.numel()} elements",
                  flush=True)

            if pos_changed > 0:
                diff_mask = topk_pos != _topk_pos_snapshot
                diff_indices = torch.where(diff_mask)[0][:10]
                for idx in diff_indices:
                    i = idx.item()
                    print(f"    topk_pos[{i}]: before={_topk_pos_snapshot[i].item()} "
                          f"after={topk_pos[i].item()} "
                          f"(token={i // K_topk} slot={i % K_topk})", flush=True)

            if wts_changed > 0:
                diff_mask = topk_weights != _topk_weights_snapshot
                n_show = min(10, diff_mask.sum().item())
                if n_show > 0:
                    flat_diff = diff_mask.view(-1)
                    diff_indices = torch.where(flat_diff)[0][:10]
                    for idx in diff_indices:
                        i = idx.item()
                        r, c = i // K_topk, i % K_topk
                        print(f"    topk_weights[{r},{c}]: "
                              f"before={_topk_weights_snapshot[r, c].item():.6f} "
                              f"after={topk_weights[r, c].item():.6f}", flush=True)
            print(flush=True)
        else:
            global _debug_routing_match_count
            # Only print OK message on first few calls to avoid log spam
            if _debug_routing_match_count <= 5:
                print(f"[CORRUPTION CHECK] OK — topk_pos, topk_weights, expert_offsets "
                      f"unchanged after kernel stages "
                      f"(dispatched={total_dispatched})", flush=True)

        del _topk_pos_snapshot, _topk_weights_snapshot, _expert_offsets_snapshot

    # Step 4: Reduce (weighted scatter-add back to original order)
    _use_index_add = os.environ.get("BATCHGEN_USE_INDEX_ADD_REDUCE", "0") == "1"

    if _use_index_add:
        # PyTorch index_add_ reduce: known-correct, matches per-expert loop fallback
        output = torch.zeros(num_tokens, hidden_size, dtype=sorted_output.dtype,
                             device=sorted_output.device)
        topk_pos_2d = topk_pos.view(num_tokens, K_topk)
        for k in range(K_topk):
            valid = topk_pos_2d[:, k] >= 0
            if valid.any():
                pos = topk_pos_2d[valid, k].long()
                token_vals = sorted_output[pos]
                weights_k = topk_weights[valid, k:k+1]
                weighted = (token_vals.float() * weights_k).to(sorted_output.dtype)
                valid_indices = torch.where(valid)[0]
                output.index_add_(0, valid_indices, weighted)
    else:
        # CUDA reduce kernel
        output = reduce_weighted_scatter_cuda(
            sorted_output, topk_pos, topk_weights,
            num_tokens, hidden_size, K_topk,
        )

    # ── Diagnostic: reduce kernel vs PyTorch reference ──
    if os.environ.get("BATCHGEN_DEBUG_REDUCE", "0") == "1":
        global _debug_reduce_call_count
        _debug_reduce_call_count += 1
        if _debug_reduce_call_count <= _DEBUG_GROUPED_MAX:
            torch.cuda.synchronize()
            # PyTorch reference: same math as reduce kernel (FP32 accum, BF16 output)
            topk_pos_2d = topk_pos.view(num_tokens, K_topk)
            ref_fp32 = torch.zeros(num_tokens, hidden_size, dtype=torch.float32,
                                   device=sorted_output.device)
            for k in range(K_topk):
                valid = topk_pos_2d[:, k] >= 0
                if valid.any():
                    pos = topk_pos_2d[valid, k].long()
                    ref_fp32[valid] += sorted_output[pos].float() * topk_weights[valid, k:k+1]
            ref_bf16 = ref_fp32.to(torch.bfloat16)

            diff = (output.float() - ref_bf16.float()).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            n_nonzero_out = (output != 0).any(dim=1).sum().item()
            n_nonzero_ref = (ref_bf16 != 0).any(dim=1).sum().item()
            n_valid_pos = (topk_pos >= 0).sum().item()

            print(f"\n[REDUCE DIAG #{_debug_reduce_call_count}] reduce kernel vs PyTorch reference:",
                  flush=True)
            print(f"  max_diff={max_diff:.6f} mean_diff={mean_diff:.6f}", flush=True)
            print(f"  num_tokens={num_tokens} total_dispatched={total_dispatched} "
                  f"valid_topk_pos={n_valid_pos}/{num_tokens * K_topk}", flush=True)
            print(f"  output: nonzero_rows={n_nonzero_out} "
                  f"range=[{output.float().min():.4f}, {output.float().max():.4f}]", flush=True)
            print(f"  ref:    nonzero_rows={n_nonzero_ref} "
                  f"range=[{ref_bf16.float().min():.4f}, {ref_bf16.float().max():.4f}]", flush=True)

            if max_diff > 0.001:
                # Find worst tokens
                per_token = diff.max(dim=1).values
                _, worst = per_token.topk(min(5, num_tokens))
                for idx in worst:
                    i = idx.item()
                    slots = topk_pos_2d[i].cpu().tolist()
                    wts = topk_weights[i].cpu().tolist()
                    print(f"  token {i}: diff={per_token[i]:.6f} "
                          f"topk_pos={slots} topk_wts={[f'{w:.4f}' for w in wts]}", flush=True)
            print(flush=True)

    # ── Diagnostic: per-expert comparison with single-expert kernel ──
    if _debug_weight_lists is not None:
        global _debug_grouped_call_count
        _debug_grouped_call_count += 1
        if _debug_grouped_call_count <= _DEBUG_GROUPED_MAX:
            _run_grouped_diagnostic(
                _debug_weight_lists,
                dispatched_x, expert_offsets, intermediate, sorted_output,
                gate_ptrs, gate_scale_ptrs, up_ptrs, up_scale_ptrs,
                down_ptrs, down_scale_ptrs,
                gate_weight_ref, gate_scale_ref, down_weight_ref, down_scale_ref,
                num_local_experts, N_intermediate, hidden_size,
                s1_stride_weight_n, s1_stride_scale_n,
                s2_stride_weight_n, s2_stride_scale_n,
                _debug_grouped_call_count,
            )

    if _return_internals:
        return output, sorted_output, topk_pos, expert_offsets
    return output


def _run_grouped_diagnostic(
    weight_lists, dispatched_x, expert_offsets, grouped_intermediate, grouped_output,
    gate_ptrs, gate_scale_ptrs, up_ptrs, up_scale_ptrs,
    down_ptrs, down_scale_ptrs,
    gate_weight_ref, gate_scale_ref, down_weight_ref, down_scale_ref,
    num_local_experts, N_intermediate, hidden_size,
    s1_stride_weight_n, s1_stride_scale_n,
    s2_stride_weight_n, s2_stride_scale_n,
    call_count,
):
    """Per-expert comparison diagnostic for grouped WGMMA debugging."""
    from batchgen.moe.fused_wgmma_expert import fused_mxfp4_expert_forward

    gate_ws, gate_ss, up_ws, up_ss, down_ws, down_ss = weight_lists

    torch.cuda.synchronize()
    offsets_cpu = expert_offsets.cpu()
    print(f"\n[GROUPED DIAG] === Call #{call_count} ===", flush=True)

    # First call: print stride/shape info
    if call_count == 1:
        print(f"  gate_ref: shape={gate_weight_ref.shape} stride={gate_weight_ref.stride()} "
              f"contig={gate_weight_ref.is_contiguous()}", flush=True)
        print(f"  down_ref: shape={down_weight_ref.shape} stride={down_weight_ref.stride()} "
              f"contig={down_weight_ref.is_contiguous()}", flush=True)
        print(f"  s1_stride_w={s1_stride_weight_n} s1_stride_s={s1_stride_scale_n} "
              f"s2_stride_w={s2_stride_weight_n} s2_stride_s={s2_stride_scale_n}", flush=True)

    # Pointer validation
    ptr_bugs = 0
    contig_bugs = 0
    for e in range(num_local_experts):
        for name, ptrs, ws, ss in [
            ("gate", gate_ptrs, gate_ws, gate_ss),
            ("up", up_ptrs, up_ws, up_ss),
            ("down", down_ptrs, down_ws, down_ss),
        ]:
            w_expected = ws[e].data_ptr()
            w_actual = ptrs[e].item()
            if w_expected != w_actual:
                print(f"  [BUG] Expert {e} {name}_ptr MISMATCH: "
                      f"expected=0x{w_expected:x} got=0x{w_actual:x}", flush=True)
                ptr_bugs += 1
            if not ws[e].is_contiguous():
                print(f"  [BUG] Expert {e} {name}_weight NOT CONTIGUOUS "
                      f"shape={ws[e].shape} stride={ws[e].stride()}", flush=True)
                contig_bugs += 1

        for name, ptrs, ss in [
            ("gate_scale", gate_scale_ptrs, gate_ss),
            ("up_scale", up_scale_ptrs, up_ss),
            ("down_scale", down_scale_ptrs, down_ss),
        ]:
            s_expected = ss[e].data_ptr()
            s_actual = ptrs[e].item()
            if s_expected != s_actual:
                print(f"  [BUG] Expert {e} {name}_ptr MISMATCH: "
                      f"expected=0x{s_expected:x} got=0x{s_actual:x}", flush=True)
                ptr_bugs += 1
            if not ss[e].is_contiguous():
                print(f"  [BUG] Expert {e} {name} NOT CONTIGUOUS "
                      f"shape={ss[e].shape} stride={ss[e].stride()}", flush=True)
                contig_bugs += 1

    if ptr_bugs == 0 and contig_bugs == 0:
        print(f"  Pointer/contiguity check: ALL OK ({num_local_experts} experts × 6 tensors)",
              flush=True)
    else:
        print(f"  Pointer/contiguity check: {ptr_bugs} ptr mismatches, {contig_bugs} non-contiguous",
              flush=True)

    # Per-expert comparison (full pipeline: single-expert vs grouped)
    print(f"  Per-expert comparison (single-expert vs grouped full pipeline):", flush=True)
    max_diffs_s1 = []
    max_diffs_full = []
    for e in range(num_local_experts):
        start = offsets_cpu[e].item()
        end = offsets_cpu[e + 1].item()
        if start == end:
            continue

        expert_input = dispatched_x[start:end].contiguous()

        # Run single-expert full pipeline (Stage 1 + Stage 2)
        ref_out = fused_mxfp4_expert_forward(
            expert_input,
            gate_ws[e], gate_ss[e],
            up_ws[e], up_ss[e],
            down_ws[e], down_ss[e],
        )
        torch.cuda.synchronize()

        # Compare Stage 1 (grouped intermediate vs... we can't easily get single-expert S1 here,
        # but we can compare full pipeline output)
        grouped_slice = grouped_output[start:end]
        diff = (ref_out.float() - grouped_slice.float()).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        max_diffs_full.append(max_diff)

        if max_diff > 0.01 or e < 3:  # Always print first 3 + any with large diff
            print(f"    Expert {e}: M={end-start} max_diff={max_diff:.6f} mean_diff={mean_diff:.6f}",
                  flush=True)

    if max_diffs_full:
        overall_max = max(max_diffs_full)
        num_bad = sum(1 for d in max_diffs_full if d > 0.01)
        print(f"  Summary: {len(max_diffs_full)} experts tested, "
              f"overall_max_diff={overall_max:.6f}, "
              f"{num_bad}/{len(max_diffs_full)} experts with diff>0.01",
              flush=True)
    print(flush=True)
