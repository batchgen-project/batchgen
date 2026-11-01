/*
 * Grouped Marlin MXFP4 (E2M1 weights, E8M0 group-32 scales), version 3:
 * swap-AB wgmma GEMM for sm_90a.
 *
 * Included by marlin_grouped_gemm.cu after marlin_grouped_mxfp4_v2.cuh; not a
 * standalone header. Everything lives in namespace marlin_v3. The E2M1 decode
 * and the folded E8M0 scales are marlin_v2's, so the MMA inputs are
 * bit-identical to v2 (and v1), and v2's precondition carries over: every
 * E8M0 scale byte e satisfies 1 <= e <= 128
 * (batchgen.moe.marlin_grouped_moe.check_mxfp4_scales_v2_compatible).
 *
 * Swap-AB: the dequantised weights are the wgmma A operand (rows = output
 * features, from registers) and the activations are the B operand (N =
 * tokens, K-major from 128B-swizzled shared memory), so the MMA N width
 * follows the rows of the expert: 8 for up to 8 rows, else the row count
 * rounded up to 16, at most NCAP.
 *
 * Marlin weights need no repack. The int32 at Marlin position
 * [k16 row, 128*ch + 4*lane + j] of the [K/16, 2N] tensor decodes
 * (marlin_v2::dequant_e2m1_raw_x8) to b0 = (k 2t..2t+1 | 2t+8..2t+9, column
 * 64*ch + 16*j + g) and b1 = the same at column + 8, with g = lane/4,
 * t = lane%4. Read with rows = features, {b0[0], b1[0], b0[1], b1[1]} is the
 * m16k16 A fragment of rows 16*j + {g, g+8}: one int32 per lane is one
 * complete 16-feature x 16-K fragment, as in v1's mma_trans_op.
 *
 * CTA: two warpgroups. Fused S1: warpgroup 0 streams gate (w1) and
 * warpgroup 1 streams up (w3) over the same 64*JW output columns. Grouped:
 * each warpgroup owns 64*JW columns. Warp w of a warpgroup takes Marlin chunk
 * w % JW of the warpgroup tile and A fragments JW*(w / JW) .. +JW-1, one
 * wgmma per fragment. Rows are processed in chunks of NCAP tokens.
 * grid = column tiles x matrices x chunks; chunk CTAs past the expert's row
 * count exit before touching shared memory. An optional expert order
 * (ExpertOrder, descending row count) puts the heavy experts' CTAs first.
 *
 * ASMEM (variant 3): the same fragments are stored to a per-warpgroup ring of
 * A tiles in shared memory and wgmma reads A through a descriptor (SS form).
 * In the RS form the dequant defines wgmma input registers while earlier
 * groups are in flight, which makes ptxas serialise the wgmma pipeline; in
 * the SS form no register operand of an in-flight wgmma is written. Tile
 * layout (64 rows x 16 K bf16, K-major, no swizzle, 8x8 core matrices of
 * 8 rows x 16 B stored contiguously): element (m, k) at byte
 * 256*(m/8) + 128*(k/8) + 16*(m%8) + 2*(k%8), i.e. LBO (next core matrix
 * along K) = 128 B and SBO (next 8-row group along M) = 256 B. Lane
 * (g, t) of warp w stores a0/a1/a2/a3 at 512*w + 4*lane + 0/256/128/384;
 * each of these four stores covers 128 contiguous bytes per warp.
 *
 * MBAR (variants 4 and 5): variant 0's tiling, MMA inputs and K order with
 * the ring filled by the async proxy and synchronised by mbarriers instead of
 * cp.async and a CTA barrier per stage. Thread 0 is the producer: per stage
 * one arrive.expect_tx on the slot's full barrier, 8 weight and 4 scale bulk
 * copies (one contiguous Marlin row segment per warpgroup and 16-K row or
 * scale group) and NB/8 TMA boxes of 8 activation rows x 64 K. The tensor map
 * uses SWIZZLE_128B, which writes the same layout the cp.async path builds by
 * hand; rows past the activation tensor are zero filled, rows of the next
 * expert are loaded and never stored. Each warpgroup releases a slot through
 * one lane after the wgmma group that last read it has retired, so the slot
 * arithmetic of the cp.async pipeline is kept and only the barrier type
 * changes. Variant 5 additionally pins the live range of the A register set
 * around its rewrite (empty asm "+r" fences after the wait and before
 * wgmma.fence).
 */
#pragma once

#include <cuda.h>  // CUtensorMap

namespace marlin_v3 {

constexpr int THREADS = 256;
constexpr int KR = 4;                          // 16-K rows per stage (64 K)
constexpr int SMEM_BUDGET_BYTES = 112 * 1024;  // two CTAs per SM
constexpr int MAX_STAGES = 8;
constexpr int MIN_STAGES = 4;
// The 128B swizzle atom is 8 rows x 128 B; dynamic shared memory is only
// guaranteed 16 B aligned, so the ring is aligned by hand.
constexpr int SMEM_ALIGN = 1024;

__host__ __device__ constexpr int cmax(int a, int b) { return a > b ? a : b; }

// Named barriers 1 and 2 (warpgroup 0 / 1, 128 threads) publish the A tiles
// of the ASMEM path; barrier 0 is __syncthreads.
constexpr int A_BAR_ID0 = 1;

// MBAR: activation TMA box (64 K x 8 rows of bf16 = one 1024 B swizzle atom)
// and the refill lag: stage s refills the slot of stage s - REFILL_LAG with
// stage s + STAGES - REFILL_LAG.
constexpr int ACT_BOX_K = 16 * KR;
constexpr int ACT_BOX_ROWS = 8;
constexpr int REFILL_LAG = 2;

template <int JW, int NCAP, bool FUSED, bool ASMEM = false>
struct Cfg {
  static_assert(JW == 2 || JW == 4, "JW must be 2 or 4");
  static_assert(NCAP % 16 == 0 && NCAP >= 16 && NCAP <= 64, "NCAP in {16..64}, multiple of 16");

  static constexpr int WG_COLS = 64 * JW;                 // output columns per warpgroup
  static constexpr int CTA_COLS = FUSED ? WG_COLS : 2 * WG_COLS;
  static constexpr int CHUNKS = WG_COLS / 64;             // Marlin 64-column chunks per warpgroup
  // A-fragment register sets in flight; wgmma.wait_group ASETS-1 before each
  // rewrite. UNROLL stages per loop body keep step % ASETS compile-time.
  static constexpr int ASETS = JW == 2 ? 3 : 2;
  static constexpr int UNROLL = ASETS == 3 ? 3 : 1;

  // Per stage: activations NCAP rows x 64 K bf16 (row = 128 B, swizzled),
  // weights 2 warpgroups x KR rows x WG_COLS*16 nibbles, scales 2 warpgroups
  // x 2 groups x WG_COLS bytes. Each ring is contiguous; the activation ring
  // comes first so every stage starts on a swizzle atom.
  static constexpr int ACT_STAGE = NCAP * 128;
  static constexpr int W_ROW = 8 * WG_COLS;
  static constexpr int W_WG = KR * W_ROW;
  static constexpr int W_STAGE = 2 * W_WG;
  static constexpr int S_ROW = WG_COLS;
  static constexpr int S_WG = 2 * S_ROW;
  static constexpr int S_STAGE = 2 * S_WG;
  static constexpr int STAGE_BYTES = ACT_STAGE + W_STAGE + S_STAGE;

  static constexpr int W_COPIES = W_STAGE / (16 * THREADS);  // int4 copies per thread
  static constexpr int S_VECS = S_STAGE / 16;

  // ASMEM A ring after the cp.async ring: per warpgroup ASETS sets of JW
  // tiles (one per wgmma), each 64 rows x 16 K bf16 in the no-swizzle
  // K-major layout described at the top of the file.
  static constexpr int A_TILE = 64 * 16 * 2;
  static constexpr int A_SET = JW * A_TILE;
  static constexpr int A_WG = ASMEM ? ASETS * A_SET : 0;
  static constexpr int A_RING = 2 * A_WG;

  // Output staging (aliases the ring after the main loop): bf16 rows of
  // CTA_COLS + 8; fused S1 keeps a gate tile and an up tile.
  static constexpr int C_STRIDE = CTA_COLS + 8;
  static constexpr int C_BYTES = (FUSED ? 2 : 1) * NCAP * C_STRIDE * 2;

  static_assert(ACT_STAGE % SMEM_ALIGN == 0, "activation stages must stay atom aligned");
  static_assert(W_STAGE % (16 * THREADS) == 0, "whole weight copies per thread");
  static_assert(S_VECS <= THREADS, "one scale copy per thread");
  static_assert((4 * UNROLL) % ASETS == 0, "register-set index must repeat per loop body");
  static_assert(STAGE_BYTES % 128 == 0, "A ring must start on a core-matrix boundary");
};

// Bytes from the aligned base to the end of the region the output staging
// aliases (rings, then staging); MBAR keeps its barriers after it.
template <int JW, int NCAP, bool FUSED, bool ASMEM = false>
__host__ __device__ constexpr int aliased_bytes(int stages) {
  using C = Cfg<JW, NCAP, FUSED, ASMEM>;
  return cmax(stages * C::STAGE_BYTES + C::A_RING, C::C_BYTES);
}

// MBAR: one full and one empty 8-byte mbarrier per ring slot.
template <int JW, int NCAP, bool FUSED, bool ASMEM = false, bool MBAR = false>
__host__ __device__ constexpr int smem_bytes(int stages) {
  return SMEM_ALIGN + aliased_bytes<JW, NCAP, FUSED, ASMEM>(stages) + (MBAR ? 16 * stages : 0);
}

template <int JW, int NCAP, bool FUSED, bool ASMEM = false, bool MBAR = false>
__host__ __device__ constexpr int pick_stages() {
  int s = MAX_STAGES;
  while (s > MIN_STAGES && smem_bytes<JW, NCAP, FUSED, ASMEM, MBAR>(s) > SMEM_BUDGET_BYTES) --s;
  return s;
}

// ---------------------------------------------------------------------------
// sm_90a primitives
// ---------------------------------------------------------------------------

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(src));
}

// src_bytes == 0 writes 16 zero bytes without reading global memory.
__device__ __forceinline__ void cp_async16_zfill(uint32_t dst, const void* src, int src_bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
               :: "r"(dst), "l"(src), "r"(src_bytes));
}

// cp.async writes through the generic proxy; wgmma reads its shared-memory
// operand through the async proxy.
__device__ __forceinline__ void fence_proxy_async() {
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void wgmma_fence() {
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void wgmma_commit() {
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

template <int N>
__device__ __forceinline__ void wgmma_wait() {
  asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N) : "memory");
}

__device__ __forceinline__ void fence_operand(float& r) {
  asm volatile("" : "+f"(r) :: "memory");
}

// Shared-memory matrix descriptor, 128B swizzle, K-major (same encoding as
// make_smem_desc in grouped_mxfp4_wgmma.cu): start address >> 4 in bits
// 0..13, stride byte offset 1024 >> 4 (next 8-row group) in bits 32..45,
// layout type 1 in bits 62..63. The start address may advance by 32 B per
// 16-K step inside the atom.
__device__ __forceinline__ uint64_t make_desc_sw128(uint32_t addr) {
  return static_cast<uint64_t>((addr & 0x3FFFF) >> 4) |
         (static_cast<uint64_t>(1024 >> 4) << 32) |
         (static_cast<uint64_t>(1) << 62);
}

// Shared-memory matrix descriptor, no swizzle (layout type 0 in bits
// 62..63), K-major: start address >> 4 in bits 0..13, leading byte offset
// (next 8x16 B core matrix along K) 128 >> 4 in bits 16..29, stride byte
// offset (next 8-row group along M) 256 >> 4 in bits 32..45, base offset 0
// (only swizzled layouts use it).
__device__ __forceinline__ uint64_t make_desc_kmajor_none(uint32_t addr) {
  return static_cast<uint64_t>((addr & 0x3FFFF) >> 4) |
         (static_cast<uint64_t>(128 >> 4) << 16) |
         (static_cast<uint64_t>(256 >> 4) << 32);
}

__device__ __forceinline__ void st_shared_b32(uint32_t addr, uint32_t v) {
  asm volatile("st.shared.b32 [%0], %1;\n" :: "r"(addr), "r"(v) : "memory");
}

// Named barrier over `threads` threads (a multiple of 32); id != 0.
template <int THREADS_>
__device__ __forceinline__ void named_bar_sync(int id) {
  asm volatile("bar.sync %0, %1;\n" :: "r"(id), "n"(THREADS_) : "memory");
}

// ---- mbarrier / bulk-copy primitives (MBAR path); addresses are shared::cta.

__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(bar), "r"(count) : "memory");
}

// Makes the initialised barriers visible to the async proxy (complete_tx).
__device__ __forceinline__ void fence_mbarrier_init() {
  asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}

// Release arrive (default .release.cta semantics).
__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
  asm volatile(
      "{\n.reg .b64 state;\n"
      "mbarrier.arrive.shared::cta.b64 state, [%0];\n}\n"
      :: "r"(bar) : "memory");
}

__device__ __forceinline__ void mbar_arrive_expect_tx(uint32_t bar, uint32_t tx_bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
               :: "r"(bar), "r"(tx_bytes) : "memory");
}

// Acquire wait until the phase with parity `parity` has completed (a fresh
// barrier counts the phase before phase 0, parity 1, as completed).
__device__ __forceinline__ bool mbar_try_wait(uint32_t bar, uint32_t parity) {
  uint32_t done;
  asm volatile(
      "{\n.reg .pred p;\n"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
      "selp.u32 %0, 1, 0, p;\n}\n"
      : "=r"(done) : "r"(bar), "r"(parity) : "memory");
  return done != 0;
}

__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
  while (!mbar_try_wait(bar, parity)) {
  }
}

// Contiguous global -> shared bulk copy; bytes, src and dst 16 B aligned.
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes,
                                         uint32_t bar) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes "
      "[%0], [%1], %2, [%3];\n"
      :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}

// 2D TMA tile load: c0 = innermost (K element), c1 = row.
__device__ __forceinline__ void tma_load_2d(uint32_t dst, const CUtensorMap* map, int c0, int c1,
                                            uint32_t bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes "
      "[%0], [%1, {%2, %3}], [%4];\n"
      :: "r"(dst), "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(bar)
      : "memory");
}

__device__ __forceinline__ void prefetch_tensormap(const CUtensorMap* map) {
  asm volatile("prefetch.tensormap [%0];\n" :: "l"(reinterpret_cast<uint64_t>(map)) : "memory");
}

#define V3_D4(i) "+f"(d[i]), "+f"(d[i + 1]), "+f"(d[i + 2]), "+f"(d[i + 3])

// D[64 x NB] += A[64 x 16] (registers, bf16) * B[16 x NB] (shared, K-major).
template <int NB>
__device__ __forceinline__ void wgmma_rs(const uint32_t* a, uint64_t desc, float* d) {
  if constexpr (NB == 8) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %9, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, p, 1, 1, 0;\n}\n"
        : V3_D4(0)
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(1));
  } else if constexpr (NB == 16) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %13, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7}, "
        "{%8, %9, %10, %11}, %12, p, 1, 1, 0;\n}\n"
        : V3_D4(0), V3_D4(4)
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(1));
  } else if constexpr (NB == 32) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %21, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, "
        "%8, %9, %10, %11, %12, %13, %14, %15}, "
        "{%16, %17, %18, %19}, %20, p, 1, 1, 0;\n}\n"
        : V3_D4(0), V3_D4(4), V3_D4(8), V3_D4(12)
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(1));
  } else if constexpr (NB == 48) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %29, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n48k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, "
        "%8, %9, %10, %11, %12, %13, %14, %15, "
        "%16, %17, %18, %19, %20, %21, %22, %23}, "
        "{%24, %25, %26, %27}, %28, p, 1, 1, 0;\n}\n"
        : V3_D4(0), V3_D4(4), V3_D4(8), V3_D4(12), V3_D4(16), V3_D4(20)
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(1));
  } else {
    static_assert(NB == 64, "wgmma N width");
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %37, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, "
        "%8, %9, %10, %11, %12, %13, %14, %15, "
        "%16, %17, %18, %19, %20, %21, %22, %23, "
        "%24, %25, %26, %27, %28, %29, %30, %31}, "
        "{%32, %33, %34, %35}, %36, p, 1, 1, 0;\n}\n"
        : V3_D4(0), V3_D4(4), V3_D4(8), V3_D4(12),
          V3_D4(16), V3_D4(20), V3_D4(24), V3_D4(28)
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(1));
  }
}

// D[64 x NB] += A[64 x 16] (shared, K-major) * B[16 x NB] (shared, K-major).
// Operands: d, a-desc, b-desc, scale-d, imm-scale-a, imm-scale-b,
// imm-trans-a, imm-trans-b.
template <int NB>
__device__ __forceinline__ void wgmma_ss(uint64_t desc_a, uint64_t desc_b, float* d) {
  if constexpr (NB == 8) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %6, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3}, %4, %5, p, 1, 1, 0, 0;\n}\n"
        : V3_D4(0)
        : "l"(desc_a), "l"(desc_b), "r"(1));
  } else if constexpr (NB == 16) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %10, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7}, "
        "%8, %9, p, 1, 1, 0, 0;\n}\n"
        : V3_D4(0), V3_D4(4)
        : "l"(desc_a), "l"(desc_b), "r"(1));
  } else if constexpr (NB == 32) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %18, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, "
        "%8, %9, %10, %11, %12, %13, %14, %15}, "
        "%16, %17, p, 1, 1, 0, 0;\n}\n"
        : V3_D4(0), V3_D4(4), V3_D4(8), V3_D4(12)
        : "l"(desc_a), "l"(desc_b), "r"(1));
  } else if constexpr (NB == 48) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %26, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n48k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, "
        "%8, %9, %10, %11, %12, %13, %14, %15, "
        "%16, %17, %18, %19, %20, %21, %22, %23}, "
        "%24, %25, p, 1, 1, 0, 0;\n}\n"
        : V3_D4(0), V3_D4(4), V3_D4(8), V3_D4(12), V3_D4(16), V3_D4(20)
        : "l"(desc_a), "l"(desc_b), "r"(1));
  } else {
    static_assert(NB == 64, "wgmma N width");
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %34, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, "
        "%8, %9, %10, %11, %12, %13, %14, %15, "
        "%16, %17, %18, %19, %20, %21, %22, %23, "
        "%24, %25, %26, %27, %28, %29, %30, %31}, "
        "%32, %33, p, 1, 1, 0, 0;\n}\n"
        : V3_D4(0), V3_D4(4), V3_D4(8), V3_D4(12),
          V3_D4(16), V3_D4(20), V3_D4(24), V3_D4(28)
        : "l"(desc_a), "l"(desc_b), "r"(1));
  }
}

#undef V3_D4

// ---------------------------------------------------------------------------
// Tile body
// ---------------------------------------------------------------------------

struct TileArgs {
  const int4* A;
  const int4* W0;      // warpgroup 0 weights (fused: gate)
  const int4* W1;      // warpgroup 1 weights (fused: up; else == W0)
  const uint8_t* S0;   // E8M0 scales matching W0
  const uint8_t* S1;
  int4* C;             // expert output, row 0 = the expert's first row
  int prob_n, prob_k;
  int token0;          // A row of this chunk's first token
  int row0;            // first row of this chunk within the expert
  int rows_c;          // rows in this chunk, 1..NCAP
  int colbase;         // first output column of the CTA
  const CUtensorMap* act_map;  // MBAR only: activation [rows, K] bf16 tensor map
};

template <int JW, int NCAP, int STAGES, bool FUSED, bool ASMEM, bool MBAR, bool LFENCE, int NB>
__device__ __forceinline__ void run_tile(const TileArgs& ta, uint8_t* smem) {
  using C = Cfg<JW, NCAP, FUSED, ASMEM>;
  static_assert(NB == 8 || (NB % 16 == 0 && NB <= NCAP), "N width");
  static_assert(!(ASMEM && (MBAR || LFENCE)), "MBAR and LFENCE build on the RS form");
  static_assert(!MBAR || (REFILL_LAG >= 2 && REFILL_LAG <= STAGES - 2),
                "refill lag: the stage after the current one must already be in flight");
  static_assert(!MBAR || NB % ACT_BOX_ROWS == 0, "whole activation boxes");
  constexpr int NACC = NB / 2;                       // accumulators per thread per wgmma
  constexpr int ASETS = C::ASETS;
  constexpr int UNROLL = C::UNROLL;
  constexpr int A_COPIES = (NB * 8 + THREADS - 1) / THREADS;

  const int tid = threadIdx.x;
  const int wg = tid / 128;
  const int warp = (tid / 32) % 4;
  const int lane = tid % 32;
  const int ch = warp % C::CHUNKS;
  const int jbase = JW * (warp / C::CHUNKS);

  const int k_tiles = ta.prob_k / (16 * KR);
  const uint32_t act_u32 = smem_u32(smem);
  const uint32_t w_u32 = act_u32 + STAGES * C::ACT_STAGE;
  const uint32_t s_u32 = w_u32 + STAGES * C::W_STAGE;
  const uint32_t a_ring_u32 = s_u32 + STAGES * C::S_STAGE;  // ASMEM only
  const uint8_t* w_ring = smem + STAGES * C::ACT_STAGE;
  const uint8_t* s_ring = w_ring + STAGES * C::W_STAGE;
  // MBAR: full barrier of slot b at bar_u32 + 8*b, empty at + 8*(STAGES + b),
  // after the region the epilogue staging aliases.
  const uint32_t bar_u32 = act_u32 + aliased_bytes<JW, NCAP, FUSED, ASMEM>(STAGES);
  static_assert(aliased_bytes<JW, NCAP, FUSED, ASMEM>(STAGES) % 8 == 0, "mbarrier alignment");

  // ---- Global -> shared copy assignment, fixed per thread (cp.async path;
  // unused, hence eliminated, under MBAR) ----
  // Weights: smem [wg][kr][WG_COLS/2 int4] filled linearly; global int4
  // (16*s + kr) row of the Marlin tensor, columns wg_col/2 + q.
  const int w_row_int4 = ta.prob_n / 2;
  const int w_stage_int4 = KR * w_row_int4;
  const int4* w_src[C::W_COPIES];
#pragma unroll
  for (int i = 0; i < C::W_COPIES; i++) {
    const int linear = tid + THREADS * i;
    const int wgi = linear / (C::W_WG / 16);
    const int rem = linear % (C::W_WG / 16);
    const int kr = rem / (C::W_ROW / 16);
    const int q = rem % (C::W_ROW / 16);
    const int wg_col = ta.colbase + (FUSED ? 0 : C::WG_COLS * wgi);
    w_src[i] = (wgi ? ta.W1 : ta.W0) + kr * w_row_int4 + wg_col / 2 + q;
  }

  // Scales: smem [wg][group][WG_COLS bytes]; one 16-byte copy per thread.
  const bool s_on = tid < C::S_VECS;
  const uint8_t* s_src;
  {
    const int byte = 16 * (s_on ? tid : 0);
    const int wgi = byte / C::S_WG;
    const int rem = byte % C::S_WG;
    const int gi = rem / C::S_ROW;
    const int wg_col = ta.colbase + (FUSED ? 0 : C::WG_COLS * wgi);
    s_src = (wgi ? ta.S1 : ta.S0) + gi * ta.prob_n + wg_col + rem % C::S_ROW;
  }
  const int s_stage_bytes = 2 * ta.prob_n;

  // Activations: rows 0..NB-1 of the chunk, 8 x 16 B per 64-K row, 16-byte
  // chunk c of row t at 128*t + 16*(c ^ (t % 8)). Rows >= rows_c are zero.
  const int a_row_int4 = ta.prob_k / 8;
  const int4* a_src[A_COPIES];
  uint32_t a_dst[A_COPIES];
  int a_bytes[A_COPIES];
  bool a_on[A_COPIES];
#pragma unroll
  for (int i = 0; i < A_COPIES; i++) {
    const int linear = tid + THREADS * i;
    const int t = linear / 8;
    const int c = linear % 8;
    const bool valid = t < ta.rows_c;
    a_on[i] = linear < NB * 8;
    a_bytes[i] = valid ? 16 : 0;
    a_src[i] = ta.A + static_cast<int64_t>(ta.token0 + (valid ? t : 0)) * a_row_int4 + c;
    a_dst[i] = 128 * t + 16 * (c ^ (t & 7));
  }

  auto fetch = [&](int s, int b) {
    if (s < k_tiles) {
      const uint32_t wdst = w_u32 + b * C::W_STAGE + 16 * tid;
#pragma unroll
      for (int i = 0; i < C::W_COPIES; i++)
        cp_async16(wdst + 16 * THREADS * i, w_src[i] + s * w_stage_int4);
      if (s_on)
        cp_async16(s_u32 + b * C::S_STAGE + 16 * tid, s_src + s * s_stage_bytes);
      const uint32_t adst = act_u32 + b * C::ACT_STAGE;
#pragma unroll
      for (int i = 0; i < A_COPIES; i++)
        if (a_on[i]) cp_async16_zfill(adst + a_dst[i], a_src[i] + 8 * s, a_bytes[i]);
    }
    cp_async_fence();
  };

  // MBAR producer (thread 0 only): stage f into slot b, every copy completing
  // on full[b]. Per warpgroup: KR weight segments of W_ROW bytes (Marlin row
  // KR*f + kr, byte offset 8*wg_col) and two scale segments of S_ROW bytes
  // (scale row 2*f + gi, offset wg_col); smem layout as the cp.async path.
  // The TMA box count is fixed by NB, so the transaction byte count is too.
  constexpr uint32_t TX_BYTES = C::W_STAGE + C::S_STAGE + NB * 128;
  auto issue = [&](int f, int b) {
    const uint32_t full_b = bar_u32 + 8 * b;
    mbar_arrive_expect_tx(full_b, TX_BYTES);
#pragma unroll
    for (int wgi = 0; wgi < 2; wgi++) {
      const int wg_col = ta.colbase + (FUSED ? 0 : C::WG_COLS * wgi);
      const int4* wsrc = (wgi ? ta.W1 : ta.W0) + KR * f * w_row_int4 + wg_col / 2;
      const uint32_t wdst = w_u32 + b * C::W_STAGE + wgi * C::W_WG;
#pragma unroll
      for (int kr = 0; kr < KR; kr++)
        bulk_g2s(wdst + kr * C::W_ROW, wsrc + kr * w_row_int4, C::W_ROW, full_b);
      const uint8_t* ssrc = (wgi ? ta.S1 : ta.S0) + 2 * f * ta.prob_n + wg_col;
      const uint32_t sdst = s_u32 + b * C::S_STAGE + wgi * C::S_WG;
#pragma unroll
      for (int gi = 0; gi < 2; gi++)
        bulk_g2s(sdst + gi * C::S_ROW, ssrc + gi * ta.prob_n, C::S_ROW, full_b);
    }
    const uint32_t adst = act_u32 + b * C::ACT_STAGE;
#pragma unroll
    for (int i = 0; i < NB / ACT_BOX_ROWS; i++)
      tma_load_2d(adst + ACT_BOX_ROWS * 128 * i, ta.act_map, ACT_BOX_K * f,
                  ta.token0 + ACT_BOX_ROWS * i, full_b);
  };

  // ---- Shared -> register reads of this lane ----
  const int w_rd = wg * C::W_WG + 16 * (32 * ch + lane) + 4 * jbase;
  const int s_rd = wg * C::S_WG + 64 * ch + 8 * (lane / 4) + 2 * jbase;

  auto load_w = [&](uint32_t* dst, int b, int kr) {
    const uint8_t* p = w_ring + b * C::W_STAGE + kr * C::W_ROW + w_rd;
    if constexpr (JW == 2) {
      const uint2 v = *reinterpret_cast<const uint2*>(p);
      dst[0] = v.x;
      dst[1] = v.y;
    } else {
      const uint4 v = *reinterpret_cast<const uint4*>(p);
      dst[0] = v.x;
      dst[1] = v.y;
      dst[2] = v.z;
      dst[3] = v.w;
    }
  };

  // Byte 2j / 2j+1 of the lane's scale bytes scale rows g / g+8 of fragment
  // jbase + j; the 2^126 E2M1 rebias is folded in as in v2.
  auto load_s = [&](uint32_t* dst, int b, int gi) {
    const uint8_t* p = s_ring + b * C::S_STAGE + gi * C::S_ROW + s_rd;
    if constexpr (JW == 2) {
      dst[0] = *reinterpret_cast<const uint32_t*>(p) + marlin_v2::SCALE_FOLD;
    } else {
      const uint2 v = *reinterpret_cast<const uint2*>(p);
      dst[0] = v.x + marlin_v2::SCALE_FOLD;
      dst[1] = v.y + marlin_v2::SCALE_FOLD;
    }
  };

  uint32_t wq[2][JW];          // weight words of the current / next 16-K step
  uint32_t sw[2][JW / 2];      // folded scales of scale group 0 / 1 of a stage
  uint32_t afr[ASETS][JW][4];  // RS only: A fragments, one set per in-flight wgmma group
  float acc[JW][NACC];

  // ASMEM: this lane's a0 address in tile 0 of set 0 of the warpgroup's A
  // ring (warp rows 16*warp.., see the layout at the top of the file).
  const uint32_t a_st = a_ring_u32 + wg * C::A_WG + 512 * warp + 4 * lane;

  auto dequant = [&](int set, int wsel, int ssel) {
#pragma unroll
    for (int i = 0; i < JW; i++) {
      FragB b0, b1;
      marlin_v2::dequant_e2m1_raw_x8(wq[wsel][i], b0, b1);
      const uint2 sv = (JW == 2) ? make_uint2(sw[ssel][0], 0u)
                                 : make_uint2(sw[ssel][0], sw[ssel][JW / 2 - 1]);
      FragS fs;
      *reinterpret_cast<uint32_t*>(&fs) = marlin_v2::folded_scale_pair(sv, i);
      scale_op(b0, fs, 0);
      scale_op(b1, fs, 1);
      const uint32_t* r0 = reinterpret_cast<const uint32_t*>(&b0);
      const uint32_t* r1 = reinterpret_cast<const uint32_t*>(&b1);
      if constexpr (ASMEM) {
        const uint32_t p = a_st + set * C::A_SET + i * C::A_TILE;
        st_shared_b32(p, r0[0]);        // a0: row g,     k 2t..2t+1
        st_shared_b32(p + 256, r1[0]);  // a1: row g + 8, k 2t..2t+1
        st_shared_b32(p + 128, r0[1]);  // a2: row g,     k 2t+8..2t+9
        st_shared_b32(p + 384, r1[1]);  // a3: row g + 8, k 2t+8..2t+9
      } else {
        afr[set][i][0] = r0[0];
        afr[set][i][1] = r1[0];
        afr[set][i][2] = r0[1];
        afr[set][i][3] = r1[1];
      }
    }
  };

  auto fence_acc = [&]() {
#pragma unroll
    for (int i = 0; i < JW; i++)
#pragma unroll
      for (int r = 0; r < NACC; r++) fence_operand(acc[i][r]);
  };

  // LFENCE: liveness fence on A register set `set` (compile-time index).
  auto fence_aset = [&](int set) {
#pragma unroll
    for (int i = 0; i < JW; i++)
#pragma unroll
      for (int j = 0; j < 4; j++) asm volatile("" : "+r"(afr[set][i][j]) :: "memory");
  };

#pragma unroll
  for (int i = 0; i < JW; i++)
#pragma unroll
    for (int r = 0; r < NACC; r++) acc[i][r] = 0.0f;
  fence_acc();
  if constexpr (LFENCE) {
    // The fences read every set before its first definition.
#pragma unroll
    for (int a = 0; a < ASETS; a++)
#pragma unroll
      for (int i = 0; i < JW; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) afr[a][i][j] = 0u;
  }

  const uint64_t desc0 = make_desc_sw128(act_u32);
  const uint64_t desc_a0 = make_desc_kmajor_none(a_ring_u32 + wg * C::A_WG);  // ASMEM only

  // ---- Pipeline ----
  // Stage s lives in ring slot s % STAGES. The refill issued at step t == 2
  // of stage s targets stage s + STAGES - 2, i.e. the slot of stage s - 2:
  // every thread's wgmma groups on that slot finished at its wait before the
  // barrier of stage s - 1, so no warpgroup can still read it. The barrier
  // of stage s publishes stage s + 1, which step t == 3 starts reading.
  //
  // MBAR keeps the slot arithmetic with per-slot barriers. full[b] (count 1:
  // the producer's arrive.expect_tx) completes phase j when stage j*STAGES + b
  // has landed; consumers wait parity (s / STAGES) & 1. empty[b] (count 2: one
  // lane per warpgroup) completes phase j when both warpgroups released stage
  // j*STAGES + b; the fill of stage f waits parity ((f / STAGES) & 1) ^ 1,
  // which passes at once on a fresh barrier for the first fill of a slot. At
  // step 2 of stage s >= 1 each warpgroup releases stage s - 1 (its last
  // wgmma group retired at the wait just above, its last ld.shared fed that
  // group), and the producer refills the slot of stage s - REFILL_LAG with
  // stage s + STAGES - REFILL_LAG. The empty wait also bounds the skew: no
  // warpgroup can release a slot's next phase before the other released the
  // current one, because the stage in between is not issued until then.
  if constexpr (MBAR) {
    if (tid == 0) {
#pragma unroll
      for (int b = 0; b < STAGES; b++) {
        mbar_init(bar_u32 + 8 * b, 1);
        mbar_init(bar_u32 + 8 * (STAGES + b), 2);
      }
      fence_mbarrier_init();
      prefetch_tensormap(ta.act_map);
    }
    __syncthreads();
    if (tid == 0) {
#pragma unroll 1
      for (int p = 0; p < STAGES - REFILL_LAG && p < k_tiles; p++) issue(p, p);
    }
    __syncwarp();
    if (k_tiles > 0) mbar_wait(bar_u32, 0);
  } else {
#pragma unroll
    for (int p = 0; p < STAGES - 2; p++) fetch(p, p);
    cp_async_wait<STAGES - 3>();
    fence_proxy_async();
    __syncthreads();
  }
  load_w(wq[0], 0, 0);
  load_s(sw[0], 0, 0);

  int buf = 0;
  for (int s0 = 0; s0 < k_tiles; s0 += UNROLL) {
#pragma unroll
    for (int u = 0; u < UNROLL; u++) {
      const int s = s0 + u;
      if (UNROLL > 1 && s >= k_tiles) break;
      const int nbuf = buf + 1 == STAGES ? 0 : buf + 1;
#pragma unroll
      for (int t = 0; t < KR; t++) {
        const int set = (KR * u + t) % ASETS;
        if (t < KR - 1) {
          load_w(wq[(t + 1) & 1], buf, t + 1);
        } else {
          load_w(wq[0], nbuf, 0);
        }
        if (t == 1) load_s(sw[1], buf, 1);
        if (t == KR - 1) load_s(sw[0], nbuf, 0);

        // Frees A set `set` (read by the group of step - ASETS; registers,
        // or for ASMEM the shared tiles, of which each warp rewrites only
        // its own 16 rows) and, at the stage barrier, the ring slot
        // refilled below.
        wgmma_wait<ASETS - 1>();
        if constexpr (LFENCE) fence_aset(set);
        if (t == 2) {
          if constexpr (MBAR) {
            if (s >= 1 && tid % 128 == 0)
              mbar_arrive(bar_u32 + 8 * (STAGES + (buf == 0 ? STAGES - 1 : buf - 1)));
            if (tid == 0) {
              const int f = s + STAGES - REFILL_LAG;
              if (f < k_tiles) {
                int fb = buf + STAGES - REFILL_LAG;
                if (fb >= STAGES) fb -= STAGES;
                mbar_wait(bar_u32 + 8 * (STAGES + fb), ((f / STAGES) & 1) ^ 1);
                issue(f, fb);
              }
            }
            __syncwarp();
            if (s + 1 < k_tiles) mbar_wait(bar_u32 + 8 * nbuf, ((s + 1) / STAGES) & 1);
          } else {
            int fb = buf + STAGES - 2;
            if (fb >= STAGES) fb -= STAGES;
            fetch(s + STAGES - 2, fb);
            cp_async_wait<STAGES - 3>();
            fence_proxy_async();
            __syncthreads();
          }
        }

        dequant(set, t & 1, t >> 1);
        const uint64_t desc = desc0 + static_cast<uint64_t>((buf * C::ACT_STAGE + 32 * t) >> 4);
        if constexpr (ASMEM) {
          // Generic-proxy stores -> async-proxy wgmma reads: proxy fence,
          // then all four warps' rows of the tiles before any warp issues.
          fence_proxy_async();
          named_bar_sync<128>(A_BAR_ID0 + wg);
        }
        if constexpr (LFENCE) fence_aset(set);
        // No accumulator fence inside the loop: touching acc while groups are
        // in flight makes ptxas serialise every wgmma.
        wgmma_fence();
        if constexpr (ASMEM) {
#pragma unroll
          for (int i = 0; i < JW; i++)
            wgmma_ss<NB>(desc_a0 + static_cast<uint64_t>((set * C::A_SET + i * C::A_TILE) >> 4),
                         desc, acc[i]);
        } else {
#pragma unroll
          for (int i = 0; i < JW; i++) wgmma_rs<NB>(afr[set][i], desc, acc[i]);
        }
        wgmma_commit();
      }
      buf = nbuf;
    }
  }
  wgmma_wait<0>();
  fence_acc();
  // MBAR: every issued stage was waited on by every thread, so no async-proxy
  // write is outstanding. The CTA barrier keeps the other warpgroup's wgmma
  // reads of the activation ring ahead of the staging stores below.
  if constexpr (!MBAR) cp_async_wait<0>();
  __syncthreads();

  // ---- Epilogue: bf16 staging (v1's rounding point), then row write-back ----
  // Accumulator r of wgmma i: token 8*(r/4) + 2*(lane%4) + r%2, feature
  // (warpgroup tile) 64*ch + 16*(jbase+i) + lane/4 + 8*((r/2)%2).
  scalar_t* stg = reinterpret_cast<scalar_t*>(smem);
  {
    scalar_t* tile = stg + (FUSED ? wg * NCAP * C::C_STRIDE : 0);
    const int col_off = FUSED ? 0 : wg * C::WG_COLS;
#pragma unroll
    for (int i = 0; i < JW; i++) {
      const int fbase = col_off + 64 * ch + 16 * (jbase + i) + lane / 4;
#pragma unroll
      for (int r = 0; r < NACC; r++) {
        const int token = 8 * (r / 4) + 2 * (lane % 4) + (r % 2);
        const int feat = fbase + 8 * ((r / 2) % 2);
        tile[token * C::C_STRIDE + feat] = float2num(acc[i][r]);
      }
    }
  }
  __syncthreads();

  constexpr int ROW_VECS = C::CTA_COLS / 8;
  constexpr int ROWS_PER_PASS = THREADS / ROW_VECS;
  const int vc = tid % ROW_VECS;
  const int c_stride = ta.prob_n / 8;
  int4* out = ta.C + static_cast<int64_t>(ta.row0) * c_stride + ta.colbase / 8 + vc;
  for (int r = tid / ROW_VECS; r < ta.rows_c; r += ROWS_PER_PASS) {
    const int4 v = reinterpret_cast<const int4*>(stg + r * C::C_STRIDE)[vc];
    if constexpr (FUSED) {
      const int4 uv = reinterpret_cast<const int4*>(stg + (NCAP + r) * C::C_STRIDE)[vc];
      const scalar_t* g_ptr = reinterpret_cast<const scalar_t*>(&v);
      const scalar_t* u_ptr = reinterpret_cast<const scalar_t*>(&uv);
      int4 res;
      scalar_t* r_ptr = reinterpret_cast<scalar_t*>(&res);
#pragma unroll
      for (int e = 0; e < 8; e++)
        r_ptr[e] = float2num(act_gate_mul<Act::SITU>(num2float(g_ptr[e]), num2float(u_ptr[e])));
      out[static_cast<int64_t>(r) * c_stride] = res;
    } else {
      out[static_cast<int64_t>(r) * c_stride] = v;
    }
  }
}

template <int JW, int NCAP, int STAGES, bool FUSED, bool ASMEM, bool MBAR = false,
          bool LFENCE = false>
__device__ __forceinline__ void grouped_body(
    const int4* __restrict__ A,
    const CUtensorMap* act_map,                 // MBAR only (else nullptr)
    const int4* const* __restrict__ B_ptrs,     // FUSED: gate (w1)
    const int4* const* __restrict__ B2_ptrs,    // FUSED: up (w3); else unused
    int4* const* __restrict__ C_ptrs,
    const int4* const* __restrict__ s_ptrs,     // FUSED: gate scales
    const int4* const* __restrict__ s2_ptrs,    // FUSED: up scales; else unused
    const int* __restrict__ expert_starts,
    const int* __restrict__ expert_counts,
    const int* __restrict__ order,              // optional matrix order
    int num_experts, int prob_n, int prob_k, int n_col_tiles, int max_chunks) {
  using C = Cfg<JW, NCAP, FUSED, ASMEM>;
  static_assert(STAGES >= MIN_STAGES, "pipeline depth");
  static_assert(smem_bytes<JW, NCAP, FUSED, ASMEM, MBAR>(STAGES) <= SMEM_BUDGET_BYTES,
                "shared memory exceeds the two-CTA budget");

  const int tile = blockIdx.x % n_col_tiles;
  const int item = blockIdx.x / n_col_tiles;
  const int rank = item / max_chunks;
  const int chunk = item % max_chunks;
  const int matrix = order ? order[rank] : rank;
  const int expert = matrix % num_experts;
  const int count = expert_counts[expert];
  const int row0 = NCAP * chunk;
  if (row0 >= count) return;

  extern __shared__ __align__(16) uint8_t marlin_v3_smem[];
  const uint32_t raw = smem_u32(marlin_v3_smem);
  uint8_t* smem = marlin_v3_smem + ((SMEM_ALIGN - (raw & (SMEM_ALIGN - 1))) & (SMEM_ALIGN - 1));

  TileArgs ta;
  ta.A = A;
  ta.W0 = B_ptrs[matrix];
  ta.W1 = FUSED ? B2_ptrs[matrix] : ta.W0;
  ta.S0 = reinterpret_cast<const uint8_t*>(s_ptrs[matrix]);
  ta.S1 = FUSED ? reinterpret_cast<const uint8_t*>(s2_ptrs[matrix]) : ta.S0;
  ta.C = C_ptrs[matrix];
  ta.prob_n = prob_n;
  ta.prob_k = prob_k;
  ta.token0 = expert_starts[expert] + row0;
  ta.row0 = row0;
  ta.rows_c = min(NCAP, count - row0);
  ta.colbase = C::CTA_COLS * tile;
  ta.act_map = act_map;

  const int rows_c = ta.rows_c;
  if (rows_c <= 8) {
    run_tile<JW, NCAP, STAGES, FUSED, ASMEM, MBAR, LFENCE, 8>(ta, smem);
  } else if (rows_c <= 16) {
    run_tile<JW, NCAP, STAGES, FUSED, ASMEM, MBAR, LFENCE, 16>(ta, smem);
  } else if (rows_c <= 32) {
    if constexpr (NCAP >= 32) run_tile<JW, NCAP, STAGES, FUSED, ASMEM, MBAR, LFENCE, 32>(ta, smem);
  } else if (rows_c <= 48) {
    if constexpr (NCAP >= 48) run_tile<JW, NCAP, STAGES, FUSED, ASMEM, MBAR, LFENCE, 48>(ta, smem);
  } else {
    if constexpr (NCAP >= 64) run_tile<JW, NCAP, STAGES, FUSED, ASMEM, MBAR, LFENCE, 64>(ta, smem);
  }
}

// Fused S1: SiTU(x @ gate^T, x @ up^T), each branch rounded to bf16 first.
template <int JW, int NCAP, int STAGES, bool ASMEM>
__global__ void __launch_bounds__(THREADS, 2) GroupedS1(
    const int4* __restrict__ A,
    const int4* const* __restrict__ gate_B_ptrs,
    const int4* const* __restrict__ up_B_ptrs,
    int4* const* __restrict__ C_ptrs,
    const int4* const* __restrict__ gate_scales_ptrs,
    const int4* const* __restrict__ up_scales_ptrs,
    const int* __restrict__ expert_starts,
    const int* __restrict__ expert_counts,
    const int* __restrict__ order,
    int num_experts, int prob_n, int prob_k, int n_col_tiles, int max_chunks) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  grouped_body<JW, NCAP, STAGES, true, ASMEM>(
      A, nullptr, gate_B_ptrs, up_B_ptrs, C_ptrs, gate_scales_ptrs, up_scales_ptrs,
      expert_starts, expert_counts, order, num_experts, prob_n, prob_k,
      n_col_tiles, max_chunks);
#endif
}

// S3 / standalone grouped GEMM. Pointer arrays are indexed by matrix
// (num_matrices >= num_experts); rows come from expert matrix % num_experts.
template <int JW, int NCAP, int STAGES, bool ASMEM>
__global__ void __launch_bounds__(THREADS, 2) Grouped(
    const int4* __restrict__ A,
    const int4* const* __restrict__ B_ptrs,
    int4* const* __restrict__ C_ptrs,
    const int4* const* __restrict__ scales_ptrs,
    const int* __restrict__ expert_starts,
    const int* __restrict__ expert_counts,
    const int* __restrict__ order,
    int num_experts, int prob_n, int prob_k, int n_col_tiles, int max_chunks) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  grouped_body<JW, NCAP, STAGES, false, ASMEM>(
      A, nullptr, B_ptrs, nullptr, C_ptrs, scales_ptrs, nullptr,
      expert_starts, expert_counts, order, num_experts, prob_n, prob_k,
      n_col_tiles, max_chunks);
#endif
}

// MBAR variants of GroupedS1 / Grouped (variant 4: LFENCE = false, variant 5:
// LFENCE = true). act_map: 2D bf16 tensor map over the activation tensor
// {K, rows}, box {ACT_BOX_K, ACT_BOX_ROWS}, SWIZZLE_128B, zero OOB fill.
template <int JW, int NCAP, int STAGES, bool LFENCE>
__global__ void __launch_bounds__(THREADS, 2) GroupedS1Mbar(
    const __grid_constant__ CUtensorMap act_map,
    const int4* __restrict__ A,
    const int4* const* __restrict__ gate_B_ptrs,
    const int4* const* __restrict__ up_B_ptrs,
    int4* const* __restrict__ C_ptrs,
    const int4* const* __restrict__ gate_scales_ptrs,
    const int4* const* __restrict__ up_scales_ptrs,
    const int* __restrict__ expert_starts,
    const int* __restrict__ expert_counts,
    const int* __restrict__ order,
    int num_experts, int prob_n, int prob_k, int n_col_tiles, int max_chunks) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  grouped_body<JW, NCAP, STAGES, true, false, true, LFENCE>(
      A, &act_map, gate_B_ptrs, up_B_ptrs, C_ptrs, gate_scales_ptrs, up_scales_ptrs,
      expert_starts, expert_counts, order, num_experts, prob_n, prob_k,
      n_col_tiles, max_chunks);
#endif
}

template <int JW, int NCAP, int STAGES, bool LFENCE>
__global__ void __launch_bounds__(THREADS, 2) GroupedMbar(
    const __grid_constant__ CUtensorMap act_map,
    const int4* __restrict__ A,
    const int4* const* __restrict__ B_ptrs,
    int4* const* __restrict__ C_ptrs,
    const int4* const* __restrict__ scales_ptrs,
    const int* __restrict__ expert_starts,
    const int* __restrict__ expert_counts,
    const int* __restrict__ order,
    int num_experts, int prob_n, int prob_k, int n_col_tiles, int max_chunks) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  grouped_body<JW, NCAP, STAGES, false, false, true, LFENCE>(
      A, &act_map, B_ptrs, nullptr, C_ptrs, scales_ptrs, nullptr,
      expert_starts, expert_counts, order, num_experts, prob_n, prob_k,
      n_col_tiles, max_chunks);
#endif
}

// order[rank] = expert with the rank-th largest row count, ties by index.
// One CTA; n <= 12288 (counts staged in dynamic shared memory).
__global__ void __launch_bounds__(THREADS) ExpertOrder(
    const int* __restrict__ counts, int* __restrict__ order, int n) {
  extern __shared__ int marlin_v3_order_counts[];
  for (int i = threadIdx.x; i < n; i += blockDim.x)
    marlin_v3_order_counts[i] = counts[i];
  __syncthreads();
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const int ci = marlin_v3_order_counts[i];
    int rank = 0;
    for (int j = 0; j < n; j++) {
      const int cj = marlin_v3_order_counts[j];
      rank += (cj > ci) || (cj == ci && j < i);
    }
    order[rank] = i;
  }
}

}  // namespace marlin_v3
