// tcgen05 MMA + TMEM Primitives — Shared utilities for SM100+/SM120 kernels
// Source: batchgen_kernel_dev/templates/sm100a/tcgen05_mma.cuh
// Requires: -arch=sm_100a or -arch=sm_120 (or higher), -lcuda
//
// tcgen05 is the native tensor core ISA for SM100+ (B200, RTX PRO 6000, etc.).
// Key differences from WGMMA:
//   - Accumulator lives in Tensor Memory (TMEM), not registers
//   - Requires explicit alloc/dealloc of TMEM columns
//   - 128 threads (4 warps), single CTA group
//   - elect.sync selects one thread per warp for MMA/TMA operations
//   - 128B swizzle for SMEM descriptors (same as WGMMA)

#pragma once
#include <cuda_runtime.h>
#include <cstdint>

// ── elect.sync — single elected thread per warp ──

__device__ __forceinline__
uint32_t elect_sync() {
    uint32_t pred = 0;
    asm volatile(
        "{\n\t"
        ".reg .pred %%px;\n\t"
        "elect.sync _|%%px, %1;\n\t"
        "@%%px mov.s32 %0, 1;\n\t"
        "}"
        : "+r"(pred) : "r"(0xFFFFFFFF));
    return pred;
}

// ── mbarrier primitives (SM100 uses same mbarrier as SM90) ──

__device__ __forceinline__
void mbarrier_init(int mbar_addr, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                :: "r"(mbar_addr), "r"(count));
}

__device__ __forceinline__
void mbarrier_wait(int mbar_addr, int phase) {
    uint32_t ticks = 0x989680;  // ~10M cycles timeout
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE;\n\t"
        "bra.uni LAB_WAIT;\n\t"
        "DONE:\n\t"
        "}"
        :: "r"(mbar_addr), "r"(phase), "r"(ticks));
}

__device__ __forceinline__
void mbarrier_arrive_expect_tx(int mbar_addr, int bytes) {
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
        :: "r"(mbar_addr), "r"(bytes) : "memory");
}

// ── TMA 2D load (same PTX as SM90, different encoding flags) ──

__device__ __forceinline__
void tma_2d_gmem2smem(int dst_smem, const void *tmap_ptr,
                       int coord_x, int coord_y, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%2, %3}], [%4];"
        :: "r"(dst_smem), "l"(tmap_ptr), "r"(coord_x), "r"(coord_y),
           "r"(mbar_addr) : "memory");
}

// ── SMEM descriptor for tcgen05 (128B swizzle) ──

__device__ __forceinline__
constexpr uint64_t desc_encode(uint64_t x) {
    return (x & 0x3FFFFULL) >> 4ULL;
}

// Build SMEM descriptor for tcgen05 MMA. SBO = stride between (MMA_M, 8) blocks.
// For 128B swizzle with BF16: SBO = 8 * 128 = 1024 bytes.
__device__ __forceinline__
uint64_t make_tcgen05_desc(int smem_addr) {
    const int SBO = 8 * 128;
    return desc_encode(smem_addr) | (desc_encode(SBO) << 32ULL)
           | (1ULL << 46ULL) | (2ULL << 61ULL);
}

// ── tcgen05 MMA (BF16×BF16→FP32, CTA group 1) ──
// Accumulator in TMEM at taddr. enable_input_d: 0=overwrite, 1=accumulate.

__device__ __forceinline__
void tcgen05_mma_bf16(int taddr, uint64_t a_desc, uint64_t b_desc,
                      uint32_t i_desc, int enable_input_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t"
        "}"
        :: "r"(taddr), "l"(a_desc), "l"(b_desc), "r"(i_desc),
           "r"(enable_input_d));
}

// ── Instruction descriptor builder ──
// FP32 accum, BF16 A, BF16 B, with BLOCK_M and BLOCK_N encoded.

__device__ __forceinline__
constexpr uint32_t make_idesc_bf16(int BLOCK_M, int BLOCK_N) {
    return (1U << 4U)                        // dtype = FP32
         | (1U << 7U)                        // atype = BF16
         | (1U << 10U)                       // btype = BF16
         | ((uint32_t)BLOCK_N >> 3U << 17U)  // MMA_N
         | ((uint32_t)BLOCK_M >> 4U << 24U); // MMA_M
}

// ── tcgen05 commit (signal MMA completion via mbarrier) ──

__device__ __forceinline__
void tcgen05_commit(int mbar_addr) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];"
        :: "r"(mbar_addr) : "memory");
}

// ── tcgen05 fence (required after sync before TMEM access) ──

__device__ __forceinline__
void tcgen05_fence_after_sync() {
    asm volatile("tcgen05.fence::after_thread_sync;");
}

// ── TMEM alloc / dealloc ──

// Allocate TMEM columns. Result address written to smem_addr_ptr.
// Must be called by warp 1 (convention: warp 0 does mbarrier init).
__device__ __forceinline__
void tcgen05_alloc(int smem_result_addr, int num_columns) {
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
        :: "r"(smem_result_addr), "r"(num_columns));
}

// Deallocate TMEM columns. Call from warp 0 after epilogue.
__device__ __forceinline__
void tcgen05_dealloc(int taddr, int num_columns) {
    asm volatile(
        "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
        :: "r"(taddr), "r"(num_columns));
}

// ── TMEM load (TMEM → registers) ──
// Loads 8 FP32 values from TMEM. Used in epilogue (TMEM → convert → GMEM).
// addr = taddr + ((warp_id * 32) << 16) + (col_group * 8)

__device__ __forceinline__
void tcgen05_load_8(int addr, float (&tmp)[8]) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x8.b32 "
        "{%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
        : "=f"(tmp[0]), "=f"(tmp[1]), "=f"(tmp[2]), "=f"(tmp[3]),
          "=f"(tmp[4]), "=f"(tmp[5]), "=f"(tmp[6]), "=f"(tmp[7])
        : "r"(addr));
    asm volatile("tcgen05.wait::ld.sync.aligned;");
}
