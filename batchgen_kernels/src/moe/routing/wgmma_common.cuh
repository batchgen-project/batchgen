/*
 * Shared WGMMA infrastructure for fused MoE routing kernels (SM90a).
 *
 * Provides: GmmaDescriptor, WGMMA m64n64k16 BF16 PTX, mbarrier helpers,
 * TMA load 2D, reg_to_mn mapping, make_2d_tma_desc_bf16.
 *
 * Extracted from BatchGen/batchgen/moe/fused_wgmma_grouped.py infrastructure.
 */

#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cstdint>
#include <utility>
#include <stdexcept>

// CUDA 13 retains the versioned driver-entry typedef but drops the legacy
// unversioned alias used by CUDA 12.x headers.
#if CUDA_VERSION >= 13000
using PFN_cuTensorMapEncodeTiled = PFN_cuTensorMapEncodeTiled_v12000;
#endif

// ============================================================================
// Configuration
// ============================================================================
#define BLOCK_M 64
#define BLOCK_N 64
#define BLOCK_K 64
#define WGMMA_K 16
#define TILES_K (BLOCK_K / WGMMA_K)          // 4
#define WGMMA_NUM_ACCUM 32                    // 64*64/128
#define NUM_STAGES 2
#define TILE_BYTES_A (BLOCK_M * BLOCK_K * 2)  // 8192 bytes (BF16)
#define TILE_BYTES_B (BLOCK_K * BLOCK_N * 2)  // 8192 bytes (BF16)
#define TOTAL_THREADS 256                     // 2 warp groups
#define PRODUCER_THREADS 128                  // WG0
#define PRODUCER_BAR_ID 1

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

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
// WGMMA m64n64k16 BF16×BF16→FP32 (SS)
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
// reg_to_mn: accumulator register → (m, n) position
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
        "FUSED_GATE_MBAR_WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n"
        "@P bra FUSED_GATE_MBAR_DONE_%=;\n"
        "bra FUSED_GATE_MBAR_WAIT_%=;\n"
        "FUSED_GATE_MBAR_DONE_%=:\n"
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
// Named barrier sync
// ============================================================================
__device__ __forceinline__ void bar_sync(uint32_t bar_id, uint32_t num_threads) {
    asm volatile("bar.sync %0, %1;" :: "r"(bar_id), "r"(num_threads));
}

// ============================================================================
// Warp-level argmax reduction (for top-K)
// ============================================================================
__device__ __forceinline__ void warp_reduce_argmax(float& val, int& idx) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        float other_val = __shfl_down_sync(0xffffffff, val, offset);
        int other_idx = __shfl_down_sync(0xffffffff, idx, offset);
        if (other_val > val || (other_val == val && other_idx < idx)) {
            val = other_val;
            idx = other_idx;
        }
    }
}

// ============================================================================
// Host: TMA descriptor creation
// ============================================================================
static inline PFN_cuTensorMapEncodeTiled get_cuTensorMapEncodeTiled() {
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

static inline CUtensorMap make_2d_tma_desc_bf16(
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
