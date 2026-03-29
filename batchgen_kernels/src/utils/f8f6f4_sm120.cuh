// SM120 F8F6F4 MMA + TMA Primitives — Shared utilities for SM120 kernels
// Source: CUTLASS mma_sm120.hpp analysis + batchgen_kernel_dev ISA research
// Requires: -arch=sm_120, CUDA >= 13.0
//
// SM120 uses mma.sync.aligned.kind::f8f6f4 (register-based, warp-synchronous).
// This is architecturally similar to SM80 mma.sync, NOT SM100 tcgen05.
// Key properties:
//   - Accumulator lives in REGISTERS (not TMEM like SM100)
//   - 32 threads per warp, all threads participate in MMA
//   - No elect.sync needed (standard warp-synchronous)
//   - TMA: SM90-style cp.async.bulk.tensor, same 128B swizzle
//   - LDSM: SM100A ld.matrix variants available

#pragma once
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cstdint>

// ── mma.sync.aligned FP8 E4M3×E4M3→FP32 (16×8×32) ──
// Tile: 16 rows × 8 cols, K=32 (32 FP8 values along reduction)
// A: 4×uint32 = 128 bits = 16 FP8 values per thread (row-major, 16M × 32K)
// B: 2×uint32 = 64 bits = 8 FP8 values per thread (col-major, 8N × 32K)
// C/D: 4×float = 128 bits per thread (16M × 8N, 4 elements per thread)

__device__ __forceinline__
void mma_f8f6f4_e4m3_16x8x32(
    float &d0, float &d1, float &d2, float &d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float c0, float c1, float c2, float c3)
{
    asm volatile(
        "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
           "r"(b0),  "r"(b1),
           "f"(c0),  "f"(c1),  "f"(c2),  "f"(c3));
}

// ── Block-scaled FP8 E4M3×E4M3→FP32 with UE8M0 scale factors (16×8×32) ──
// Same as above but with per-block scale factors (FP8 blockwise quantization)

__device__ __forceinline__
void mma_mxf8f6f4_e4m3_blockscale_16x8x32(
    float &d0, float &d1, float &d2, float &d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float c0, float c1, float c2, float c3,
    uint32_t sfa, uint32_t sfb)
{
    asm volatile(
        "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13},"
        "{%14},"
        "{%15};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
           "r"(b0),  "r"(b1),
           "f"(c0),  "f"(c1),  "f"(c2),  "f"(c3),
           "r"(sfa), "r"(sfb));
}

// ── FP8 E5M2×E4M3→FP32 (common for mixed precision: activations E5M2, weights E4M3) ──

__device__ __forceinline__
void mma_f8f6f4_e5m2_e4m3_16x8x32(
    float &d0, float &d1, float &d2, float &d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float c0, float c1, float c2, float c3)
{
    asm volatile(
        "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e5m2.e4m3.f32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
           "r"(b0),  "r"(b1),
           "f"(c0),  "f"(c1),  "f"(c2),  "f"(c3));
}

// ── TMA 2D load (same as SM90, works on SM120) ──

__device__ __forceinline__
void tma_2d_gmem2smem(int dst_smem, const void *tmap_ptr,
                       int coord_x, int coord_y, int mbar_addr)
{
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%2, %3}], [%4];"
        :: "r"(dst_smem), "l"(tmap_ptr), "r"(coord_x), "r"(coord_y),
           "r"(mbar_addr) : "memory");
}

// ── mbarrier primitives (SM90 API, works on SM120) ──

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

// ── LDSM (ld.matrix) for SM120 ──
// SM120 supports both SM75 and SM100A LDSM variants.
//
// For FP8 data (8-bit elements), CUTLASS uses SM75 ldmatrix.b16:
//   ldmatrix treats pairs of FP8 values as 16-bit elements.
//   This is the same approach used for SM80 INT8 GEMM (m16n8k32).
//
// For FP4 data (4-bit elements), SM120 uses SM100A ldmatrix.b8x16:
//   ldmatrix.sync.aligned.m8n16.x{1,2,4}.shared.b8x16.b4x16_p64

// SM75 ldmatrix x4 (FP8: pairs of 8-bit values treated as 16-bit elements)
__device__ __forceinline__
void ldsm_x4(uint32_t &r0, uint32_t &r1, uint32_t &r2, uint32_t &r3,
             int smem_addr)
{
    asm volatile(
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(smem_addr));
}

// SM75 ldmatrix x2 transposed (for B operand)
__device__ __forceinline__
void ldsm_x2_trans(uint32_t &r0, uint32_t &r1, int smem_addr)
{
    asm volatile(
        "ldmatrix.sync.aligned.x2.m8n8.trans.shared.b16 "
        "{%0, %1}, [%2];"
        : "=r"(r0), "=r"(r1)
        : "r"(smem_addr));
}

// SM100A ldmatrix for FP8 (16×16 tile, transposed, with byte rearrangement)
// Returns 2×uint32 (8 FP8 values)
__device__ __forceinline__
void ldsm_b8_x1_trans(uint32_t &r0, uint32_t &r1, int smem_addr)
{
    uint32_t tmp0, tmp1;
    asm volatile(
        "ldmatrix.sync.aligned.m16n16.x1.trans.shared.b8 "
        "{%0, %1}, [%2];"
        : "=r"(tmp0), "=r"(tmp1)
        : "r"(smem_addr));
    // Rearrange bytes to match MMA operand layout (CUTLASS pattern)
    unsigned char *t0 = reinterpret_cast<unsigned char*>(&tmp0);
    unsigned char *t1 = reinterpret_cast<unsigned char*>(&tmp1);
    unsigned char d0[4] = {t0[0], t0[1], t1[0], t1[1]};
    unsigned char d1[4] = {t0[2], t0[3], t1[2], t1[3]};
    r0 = *reinterpret_cast<uint32_t*>(d0);
    r1 = *reinterpret_cast<uint32_t*>(d1);
}

// SM100A ldmatrix for FP8 (16×16 tile, transposed, 4×uint32 output)
__device__ __forceinline__
void ldsm_b8_x2_trans(uint32_t &r0, uint32_t &r1, uint32_t &r2, uint32_t &r3,
                      int smem_addr)
{
    uint32_t tmp0, tmp1, tmp2, tmp3;
    asm volatile(
        "ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(tmp0), "=r"(tmp1), "=r"(tmp2), "=r"(tmp3)
        : "r"(smem_addr));
    unsigned char *t0 = reinterpret_cast<unsigned char*>(&tmp0);
    unsigned char *t1 = reinterpret_cast<unsigned char*>(&tmp1);
    unsigned char *t2 = reinterpret_cast<unsigned char*>(&tmp2);
    unsigned char *t3 = reinterpret_cast<unsigned char*>(&tmp3);
    unsigned char d0[4] = {t0[0], t0[1], t1[0], t1[1]};
    unsigned char d1[4] = {t0[2], t0[3], t1[2], t1[3]};
    unsigned char d2[4] = {t2[0], t2[1], t3[0], t3[1]};
    unsigned char d3[4] = {t2[2], t2[3], t3[2], t3[3]};
    r0 = *reinterpret_cast<uint32_t*>(d0);
    r1 = *reinterpret_cast<uint32_t*>(d1);
    r2 = *reinterpret_cast<uint32_t*>(d2);
    r3 = *reinterpret_cast<uint32_t*>(d3);
}
