"""
Fused grouped MoE kernels using WGMMA for GPT-OSS-120B decode.

Provides grouped MXFP4 WGMMA kernels that operate on 1D+offsets layout
directly from CUDA dispatch, eliminating the 3D reshape overhead of the
Triton grouped path.

Two CUDA kernels:
- Stage 1: gate projection + up projection + SwiGLU activation (fused)
- Stage 2: down projection

Pipeline: dispatch -> WGMMA S1 -> WGMMA S2 -> reduce (4 kernel launches)
vs Triton: dispatch -> reshape -> gate_gemm -> up_gemm -> swiglu -> down_gemm -> gather -> reduce (9+)

Performance (E=16, K=7168, N=14336, decode M=1-64):
- Stage 1: 4.0-4.1x over for-loop baseline
- Stage 2: 4.8-4.9x over for-loop baseline
- Full pipeline: 3.6-3.7x over for-loop baseline

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
# Ported from batchgen_kernels/moe/gptoss/grouped_bf16_moe_wgmma.py
# CUDA_SOURCE_GROUPED_MXFP4_PHASE2A (Stage 1) + CUDA_SOURCE_GROUPED_MXFP4_STAGE2 (Stage 2)

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
// TMA load 2D
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
// Named barrier sync (producer WG internal)
// ============================================================================
__device__ __forceinline__ void bar_sync(uint32_t bar_id, uint32_t num_threads) {
    asm volatile("bar.sync %0, %1;" :: "r"(bar_id), "r"(num_threads));
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
//   Output = SwiGLU(A @ dequant(Gate).T, A @ dequant(Up).T)
//   Grid: (num_experts, num_n_tiles, max_m_tiles)
//   TMA: desc_a for activations (BF16), Phase 10a byte-LUT for weights (MXFP4)
// ============================================================================
__global__ void __launch_bounds__(TOTAL_THREADS, 1)
grouped_mxfp4_stage1_tma_kernel(
    const __grid_constant__ CUtensorMap desc_a,
    const int64_t* __restrict__ gate_ptrs,
    const int64_t* __restrict__ gate_scale_ptrs,
    const int64_t* __restrict__ up_ptrs,
    const int64_t* __restrict__ up_scale_ptrs,
    const int32_t* __restrict__ expert_offsets,
    __nv_bfloat16* __restrict__ C,
    int64_t stride_c_m,
    int total_tokens, int N, int K, int num_experts,
    int64_t stride_weight_n,
    int64_t stride_scale_n
) {
    const int tid = threadIdx.x;
    const int wg_id = tid / 128;
    const int wg_tid = tid % 128;

    // 3D grid: (expert_idx, n_tile, m_tile)
    const int expert_idx = blockIdx.x;
    const int n_tile = blockIdx.y;
    const int m_tile = blockIdx.z;
    const int n_start = n_tile * BLOCK_N;

    if (n_start >= N) return;

    // Load expert's token range
    const int expert_start = expert_offsets[expert_idx];
    const int expert_end = expert_offsets[expert_idx + 1];
    const int expert_tokens = expert_end - expert_start;

    const int num_m_tiles = (expert_tokens + BLOCK_M - 1) / BLOCK_M;
    if (m_tile >= num_m_tiles) return;

    const int m_start = expert_start + m_tile * BLOCK_M;
    const int m_size = min(BLOCK_M, expert_end - m_start);

    // Load expert's weight pointers
    const uint8_t* gate_weight = reinterpret_cast<const uint8_t*>(gate_ptrs[expert_idx]);
    const uint8_t* gate_scale = reinterpret_cast<const uint8_t*>(gate_scale_ptrs[expert_idx]);
    const uint8_t* up_weight = reinterpret_cast<const uint8_t*>(up_ptrs[expert_idx]);
    const uint8_t* up_scale = reinterpret_cast<const uint8_t*>(up_scale_ptrs[expert_idx]);

    // Shared memory layout: 2 stages × 2 tiles (A + B) + 4 barrier sets + LUT
    extern __shared__ __align__(128) char smem_buf[];
    __nv_bfloat16* smem_a[NUM_STAGES];
    __nv_bfloat16* smem_b[NUM_STAGES];
    for (int s = 0; s < NUM_STAGES; s++) {
        smem_a[s] = reinterpret_cast<__nv_bfloat16*>(smem_buf + (2*s)     * TILE_BYTES);
        smem_b[s] = reinterpret_cast<__nv_bfloat16*>(smem_buf + (2*s + 1) * TILE_BYTES);
    }
    const int bar_offset = 2 * NUM_STAGES * TILE_BYTES;
    // Separate barrier sets for gate and up (no re-init needed)
    uint64_t* gate_full_barriers  = reinterpret_cast<uint64_t*>(smem_buf + bar_offset);
    uint64_t* gate_empty_barriers = gate_full_barriers + NUM_STAGES;
    uint64_t* up_full_barriers    = gate_empty_barriers + NUM_STAGES;
    uint64_t* up_empty_barriers   = up_full_barriers + NUM_STAGES;
    __nv_bfloat162* byte_lut = reinterpret_cast<__nv_bfloat162*>(
        smem_buf + bar_offset + 4 * NUM_STAGES * sizeof(uint64_t));

    const int num_k_blocks = (K + BLOCK_K - 1) / BLOCK_K;

    // Init all barriers at kernel start
    if (tid == 0) {
        for (int s = 0; s < NUM_STAGES; s++) {
            mbarrier_init(&gate_full_barriers[s], 2);   // TMA + decode
            mbarrier_init(&gate_empty_barriers[s], 1);  // consumer
            mbarrier_init(&up_full_barriers[s], 2);     // TMA + decode
            mbarrier_init(&up_empty_barriers[s], 1);    // consumer
        }
    }
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    __syncthreads();

    if (wg_id == 0) {
        // ════════════════════════════════════════════════════════════════
        // PRODUCER WG: TMA A + batched byte-LUT decode B (gate, then up)
        // ════════════════════════════════════════════════════════════════

        // Initialize byte-level LUT
        for (int i = wg_tid; i < LUT_ENTRIES; i += PRODUCER_THREADS) {
            __nv_bfloat16 lo = __float2bfloat16(FP4_LUT[i & 0xF]);
            __nv_bfloat16 hi = __float2bfloat16(FP4_LUT[i >> 4]);
            byte_lut[i] = __halves2bfloat162(lo, hi);
        }
        bar_sync(PRODUCER_BAR_ID, PRODUCER_THREADS);

        if (wg_tid == 0) {
            asm volatile("prefetch.tensormap [%0];" :: "l"(&desc_a) : "memory");
        }

        // GATE K-LOOP
        for (int kb = 0; kb < num_k_blocks; kb++) {
            const int s = kb % NUM_STAGES;
            const int empty_phase = ((kb / NUM_STAGES) + 1) & 1;

            mbarrier_wait_parity(&gate_empty_barriers[s], empty_phase);

            if (wg_tid == 0) {
                mbarrier_arrive_expect_tx(&gate_full_barriers[s], TILE_BYTES);
                tma_load_2d(&desc_a, &gate_full_barriers[s], smem_a[s],
                            kb * BLOCK_K, m_start);
            }

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

        // SYNC BETWEEN GATE AND UP K-LOOPS
        __syncthreads();

        // UP K-LOOP
        for (int kb = 0; kb < num_k_blocks; kb++) {
            const int s = kb % NUM_STAGES;
            const int empty_phase = ((kb / NUM_STAGES) + 1) & 1;

            mbarrier_wait_parity(&up_empty_barriers[s], empty_phase);

            if (wg_tid == 0) {
                mbarrier_arrive_expect_tx(&up_full_barriers[s], TILE_BYTES);
                tma_load_2d(&desc_a, &up_full_barriers[s], smem_a[s],
                            kb * BLOCK_K, m_start);
            }

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
        // ════════════════════════════════════════════════════════════════
        // MATH WG: gate GEMM -> up GEMM -> SwiGLU epilogue
        // ════════════════════════════════════════════════════════════════
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

        // SYNC BETWEEN GATE AND UP K-LOOPS
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
        #pragma unroll
        for (int i = 0; i < WGMMA_NUM_ACCUM; i++) {
            int m, n;
            reg_to_mn(i, warp_in_wg, lane_id, m, n);
            const int m_global = m_start + m;
            const int n_global = n_start + n;

            if (m < m_size && n_global < N) {
                float g = __bfloat162float(__float2bfloat16(gate_acc[i]));
                float u = __bfloat162float(__float2bfloat16(up_acc[i]));

                g = fminf(g, SWIGLU_LIMIT);
                u = fmaxf(fminf(u, SWIGLU_LIMIT), -SWIGLU_LIMIT);

                float sig = __frcp_rn(1.0f + expf(-SWIGLU_ALPHA * g));
                float result = g * sig * (u + 1.0f);

                C[m_global * stride_c_m + n_global] = __float2bfloat16(result);
            }
        }
    }
}

// ============================================================================
// Stage 2 Kernel: Down projection (grouped MXFP4)
//   output[total_tokens, K] = intermediate[total_tokens, N] x dequant(B_down)[K, N]^T
//   Grid: (num_experts, num_k_tiles, max_m_tiles)
// ============================================================================
__global__ void __launch_bounds__(TOTAL_THREADS, 1)
grouped_mxfp4_stage2_tma_kernel(
    const __grid_constant__ CUtensorMap desc_input,
    const int64_t* __restrict__ down_ptrs,
    const int64_t* __restrict__ down_scale_ptrs,
    const int32_t* __restrict__ expert_offsets,
    __nv_bfloat16* __restrict__ C,
    int64_t stride_c_m,
    int total_tokens, int N, int K, int num_experts,
    int64_t stride_weight_n,
    int64_t stride_scale_n
) {
    const int tid = threadIdx.x;
    const int wg_id = tid / 128;
    const int wg_tid = tid % 128;

    // 3D grid: (expert_idx, k_tile, m_tile)
    const int expert_idx = blockIdx.x;
    const int k_tile = blockIdx.y;
    const int m_tile = blockIdx.z;
    const int k_start = k_tile * BLOCK_N;  // output K dimension

    if (k_start >= K) return;

    // Load expert's token range
    const int expert_start = expert_offsets[expert_idx];
    const int expert_end = expert_offsets[expert_idx + 1];
    const int expert_tokens = expert_end - expert_start;

    const int num_m_tiles = (expert_tokens + BLOCK_M - 1) / BLOCK_M;
    if (m_tile >= num_m_tiles) return;

    const int m_start = expert_start + m_tile * BLOCK_M;
    const int m_size = min(BLOCK_M, expert_end - m_start);

    // Load expert's weight pointers
    const uint8_t* down_weight = reinterpret_cast<const uint8_t*>(down_ptrs[expert_idx]);
    const uint8_t* down_scale = reinterpret_cast<const uint8_t*>(down_scale_ptrs[expert_idx]);

    // Shared memory layout: 2 stages x 2 tiles + 2 barrier sets + LUT
    extern __shared__ __align__(128) char smem_buf[];
    __nv_bfloat16* smem_a[NUM_STAGES];  // input tile
    __nv_bfloat16* smem_b[NUM_STAGES];  // B_down decoded
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

    // Init barriers
    if (tid == 0) {
        for (int s = 0; s < NUM_STAGES; s++) {
            mbarrier_init(&full_barriers[s], 2);   // TMA + decode
            mbarrier_init(&empty_barriers[s], 1);  // consumer
        }
    }
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    __syncthreads();

    if (wg_id == 0) {
        // ════════════════════════════════════════════════════════════════
        // PRODUCER WG: TMA input + batched byte-LUT decode B_down
        // ════════════════════════════════════════════════════════════════

        // Initialize byte-level LUT
        for (int i = wg_tid; i < LUT_ENTRIES; i += PRODUCER_THREADS) {
            __nv_bfloat16 lo = __float2bfloat16(FP4_LUT[i & 0xF]);
            __nv_bfloat16 hi = __float2bfloat16(FP4_LUT[i >> 4]);
            byte_lut[i] = __halves2bfloat162(lo, hi);
        }
        bar_sync(PRODUCER_BAR_ID, PRODUCER_THREADS);

        if (wg_tid == 0) {
            asm volatile("prefetch.tensormap [%0];" :: "l"(&desc_input) : "memory");
        }

        // N-REDUCTION LOOP
        for (int nb = 0; nb < num_n_blocks; nb++) {
            const int s = nb % NUM_STAGES;
            const int empty_phase = ((nb / NUM_STAGES) + 1) & 1;

            mbarrier_wait_parity(&empty_barriers[s], empty_phase);

            // TMA load input tile [BLOCK_M, BLOCK_K] from [total_tokens, N]
            if (wg_tid == 0) {
                mbarrier_arrive_expect_tx(&full_barriers[s], TILE_BYTES);
                tma_load_2d(&desc_input, &full_barriers[s], smem_a[s],
                            nb * BLOCK_K, m_start);
            }

            // Dequant B_down tile
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
        // ════════════════════════════════════════════════════════════════
        // MATH WG: WGMMA compute + store epilogue
        // ════════════════════════════════════════════════════════════════
        float acc[WGMMA_NUM_ACCUM];
        for (int i = 0; i < WGMMA_NUM_ACCUM; i++) acc[i] = 0.0f;

        const int warp_in_wg = wg_tid / WARP_SIZE;
        const int lane_id = wg_tid % WARP_SIZE;

        // N-REDUCTION LOOP
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

        // EPILOGUE: FP32 -> BF16 store (no SwiGLU, no bias)
        #pragma unroll 1
        for (int i = 0; i < WGMMA_NUM_ACCUM; i++) {
            int m, n;
            reg_to_mn(i, warp_in_wg, lane_id, m, n);
            const int m_global = m_start + m;
            const int k_global = k_start + n;

            if (m < m_size && k_global < K) {
                C[m_global * stride_c_m + k_global] = __float2bfloat16(acc[i]);
            }
        }
    }
}

// ============================================================================
// Host utilities: TMA descriptor creation
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

static CUtensorMap make_2d_tma_desc_bf16(
    __nv_bfloat16* global_address,
    uint64_t gmem_rows, uint64_t gmem_cols,
    uint32_t smem_rows, uint32_t smem_cols,
    PFN_cuTensorMapEncodeTiled encode_func
) {
    CUtensorMap tensor_map = {};
    uint64_t gmem_dim[2] = {gmem_cols, gmem_rows};
    uint64_t global_stride[1] = {gmem_cols * sizeof(__nv_bfloat16)};
    uint32_t smem_dim[2] = {smem_cols, smem_rows};
    uint32_t elem_strides[2] = {1, 1};

    auto result = encode_func(
        &tensor_map,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2,
        global_address,
        gmem_dim,
        global_stride,
        smem_dim,
        elem_strides,
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA);
    if (result != CUDA_SUCCESS)
        throw std::runtime_error("cuTensorMapEncodeTiled failed");
    return tensor_map;
}

// ============================================================================
// C++ wrapper: Stage 1 (gate + up + SwiGLU)
// ============================================================================
torch::Tensor grouped_mxfp4_stage1_tma(
    torch::Tensor A,
    torch::Tensor expert_offsets,
    torch::Tensor gate_ptrs,
    torch::Tensor gate_scale_ptrs,
    torch::Tensor up_ptrs,
    torch::Tensor up_scale_ptrs,
    int N,
    int64_t stride_weight_n,
    int64_t stride_scale_n,
    int max_m_tiles
) {
    TORCH_CHECK(A.is_cuda() && A.dtype() == torch::kBFloat16);
    TORCH_CHECK(expert_offsets.is_cuda() && expert_offsets.dtype() == torch::kInt32);
    TORCH_CHECK(gate_ptrs.is_cuda() && gate_ptrs.dtype() == torch::kInt64);

    const int total_tokens = A.size(0);
    const int K = A.size(1);
    const int num_experts = expert_offsets.size(0) - 1;

    auto C = torch::empty({total_tokens, N}, A.options());

    auto encode_func = get_cuTensorMapEncodeTiled();

    CUtensorMap desc_a = make_2d_tma_desc_bf16(
        reinterpret_cast<__nv_bfloat16*>(A.data_ptr()),
        total_tokens, K, BLOCK_M, BLOCK_K, encode_func);

    const int num_n_tiles = (N + BLOCK_N - 1) / BLOCK_N;
    dim3 grid(num_experts, num_n_tiles, max_m_tiles);
    dim3 block(TOTAL_THREADS);

    constexpr int smem_bytes = 2 * NUM_STAGES * TILE_BYTES +
                               4 * NUM_STAGES * sizeof(uint64_t) + LUT_BYTES;

    cudaFuncSetAttribute(grouped_mxfp4_stage1_tma_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);

    grouped_mxfp4_stage1_tma_kernel<<<grid, block, smem_bytes>>>(
        desc_a,
        gate_ptrs.data_ptr<int64_t>(),
        gate_scale_ptrs.data_ptr<int64_t>(),
        up_ptrs.data_ptr<int64_t>(),
        up_scale_ptrs.data_ptr<int64_t>(),
        expert_offsets.data_ptr<int32_t>(),
        reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
        C.stride(0),
        total_tokens, N, K, num_experts,
        stride_weight_n, stride_scale_n);

    return C;
}

// ============================================================================
// C++ wrapper: Stage 2 (down projection)
// ============================================================================
torch::Tensor grouped_mxfp4_stage2_tma(
    torch::Tensor input,
    torch::Tensor expert_offsets,
    torch::Tensor down_ptrs,
    torch::Tensor down_scale_ptrs,
    int K,
    int64_t stride_weight_n,
    int64_t stride_scale_n,
    int max_m_tiles
) {
    TORCH_CHECK(input.is_cuda() && input.dtype() == torch::kBFloat16);
    TORCH_CHECK(expert_offsets.is_cuda() && expert_offsets.dtype() == torch::kInt32);
    TORCH_CHECK(down_ptrs.is_cuda() && down_ptrs.dtype() == torch::kInt64);

    const int total_tokens = input.size(0);
    const int N = input.size(1);
    const int num_experts = expert_offsets.size(0) - 1;

    auto C = torch::empty({total_tokens, K}, input.options());

    auto encode_func = get_cuTensorMapEncodeTiled();

    // TMA descriptor for input [total_tokens, N], tile [BLOCK_M, BLOCK_K]
    CUtensorMap desc_input = make_2d_tma_desc_bf16(
        reinterpret_cast<__nv_bfloat16*>(input.data_ptr()),
        total_tokens, N, BLOCK_M, BLOCK_K, encode_func);

    const int num_k_tiles = (K + BLOCK_N - 1) / BLOCK_N;
    dim3 grid(num_experts, num_k_tiles, max_m_tiles);
    dim3 block(TOTAL_THREADS);

    constexpr int smem_bytes = 2 * NUM_STAGES * TILE_BYTES +
                               2 * NUM_STAGES * sizeof(uint64_t) + LUT_BYTES;

    cudaFuncSetAttribute(grouped_mxfp4_stage2_tma_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);

    grouped_mxfp4_stage2_tma_kernel<<<grid, block, smem_bytes>>>(
        desc_input,
        down_ptrs.data_ptr<int64_t>(),
        down_scale_ptrs.data_ptr<int64_t>(),
        expert_offsets.data_ptr<int32_t>(),
        reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
        C.stride(0),
        total_tokens, N, K, num_experts,
        stride_weight_n, stride_scale_n);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("grouped_mxfp4_stage1_tma", &grouped_mxfp4_stage1_tma,
          "Grouped MXFP4 Stage 1 (gate+up+SwiGLU) with TMA");
    m.def("grouped_mxfp4_stage2_tma", &grouped_mxfp4_stage2_tma,
          "Grouped MXFP4 Stage 2 (down projection) with TMA");
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
            name="batchgen_fused_mxfp4_grouped_wgmma",
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
    No 3D reshape needed.

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

    num_experts = expert_offsets.shape[0] - 1
    total_tokens = sorted_hidden.shape[0]

    # Compute max_m_tiles from expert_offsets
    # Each expert may have different token counts; max_m_tiles covers the largest
    if total_tokens > 0:
        max_m_tiles = (total_tokens + 63) // 64  # conservative upper bound
    else:
        max_m_tiles = 1

    return mod.grouped_mxfp4_stage1_tma(
        sorted_hidden, expert_offsets,
        gate_ptrs, gate_scale_ptrs,
        up_ptrs, up_scale_ptrs,
        N, stride_weight_n, stride_scale_n, max_m_tiles,
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

    total_tokens = intermediate.shape[0]

    if total_tokens > 0:
        max_m_tiles = (total_tokens + 63) // 64
    else:
        max_m_tiles = 1

    return mod.grouped_mxfp4_stage2_tma(
        intermediate, expert_offsets,
        down_ptrs, down_scale_ptrs,
        K, stride_weight_n, stride_scale_n, max_m_tiles,
    )


# ──────────────────────────────────────────────────────────────────────────────
# End-to-End API with CUDA Routing
# ──────────────────────────────────────────────────────────────────────────────

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

    # Compute strides from reference weights
    # Stage 1 (gate/up): weight is [N, K//2], scale is [N, K//32]
    s1_stride_weight_n = gate_weight_ref.shape[1]   # K // 2
    s1_stride_scale_n = gate_scale_ref.shape[1]     # K // 32

    # Stage 2 (down): weight is [K_hidden, N//2], scale is [K_hidden, N//32]
    s2_stride_weight_n = down_weight_ref.shape[1]   # N // 2
    s2_stride_scale_n = down_scale_ref.shape[1]     # N // 32

    # Step 2: WGMMA Stage 1 (gate + up + SwiGLU)
    intermediate = fused_mxfp4_grouped_stage1(
        dispatched_x, expert_offsets,
        gate_ptrs, gate_scale_ptrs,
        up_ptrs, up_scale_ptrs,
        N_intermediate, s1_stride_weight_n, s1_stride_scale_n,
    )

    # Step 3: WGMMA Stage 2 (down projection)
    sorted_output = fused_mxfp4_grouped_stage2(
        intermediate, expert_offsets,
        down_ptrs, down_scale_ptrs,
        hidden_size, s2_stride_weight_n, s2_stride_scale_n,
    )

    # Step 4: CUDA reduce (weighted scatter-add back to original order)
    output = reduce_weighted_scatter_cuda(
        sorted_output, topk_pos, topk_weights,
        num_tokens, hidden_size, K_topk,
    )

    return output
