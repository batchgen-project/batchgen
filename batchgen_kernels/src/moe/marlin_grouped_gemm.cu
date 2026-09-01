/*
 * Grouped Marlin W4A16 — Production kernel with two variants:
 *
 * MarlinGrouped_M8  (v12c): MBLOCK=8, mma_trans, ldsm<2>. 80 regs, 32% occ. Decode M<=8.
 * MarlinGrouped_M16 (v14):  MBLOCK=16, standard mma, ldsm<4>. CTA M-tiling for any M.
 *                           130 regs, 12.5% occ. 2.6× faster than WGMMA at all M values.
 *
 * Both use GROUP_BLOCKS=2 (gs=32, K2.5 native quantization).
 * M16 uses CTA-level M-tiling: grid = num_matrices × max_m_tiles × n_tiles.
 * Each CTA processes 16 rows; early-exits if m_start >= expert_counts[expert].
 */

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

// ============================================================================
// Types and helpers
// ============================================================================

template <typename T, int n>
struct Vec {
  T elems[n];
  __device__ T& operator[](int i) { return elems[i]; }
};

using I4 = Vec<int, 4>;

__host__ __device__ constexpr int div_ceil(int a, int b) {
  return (a + b - 1) / b;
}

// Compute dtype selection
#if defined(USE_BF16_COMPUTE)
using scalar_t = nv_bfloat16;
using scalar_t2 = nv_bfloat162;
using FragA = Vec<nv_bfloat162, 4>;
using FragB = Vec<nv_bfloat162, 2>;
using FragC = Vec<float, 4>;
using FragS = Vec<nv_bfloat162, 1>;
__device__ inline float num2float(scalar_t x) { return __bfloat162float(x); }
__device__ inline scalar_t2 num2num2(scalar_t x) { return __bfloat162bfloat162(x); }
__device__ inline scalar_t2 nums2num2(scalar_t x1, scalar_t x2) { return __halves2bfloat162(x1, x2); }
__host__ __device__ inline scalar_t float2num(float x) { return __float2bfloat16(x); }
#else
using scalar_t = half;
using scalar_t2 = half2;
using FragA = Vec<half2, 4>;
using FragB = Vec<half2, 2>;
using FragC = Vec<float, 4>;
using FragS = Vec<half2, 1>;
__device__ inline float num2float(scalar_t x) { return __half2float(x); }
__device__ inline scalar_t2 num2num2(scalar_t x) { return __half2half2(x); }
__device__ inline scalar_t2 nums2num2(scalar_t x1, scalar_t x2) { return __halves2half2(x1, x2); }
__host__ __device__ inline scalar_t float2num(float x) { return __float2half(x); }
#endif

// cp.async helpers
__device__ inline void cp_async4_pred(void* smem_ptr, const void* glob_ptr, bool pred = true) {
  const int BYTES = 16;
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %0, 0;\n @p cp.async.cg.shared.global [%1], [%2], %3;\n}\n"
      :: "r"((int)pred), "r"(smem), "l"(glob_ptr), "n"(BYTES));
}

__device__ inline void cp_async4(void* smem_ptr, const void* glob_ptr) {
  const int BYTES = 16;
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("{\n cp.async.cg.shared.global [%0], [%1], %2;\n}\n"
      :: "r"(smem), "l"(glob_ptr), "n"(BYTES));
}

__device__ inline void cp_async8(void* smem_ptr, const void* glob_ptr) {
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  // PTX permits 8-byte copies only for the .ca cache operator; .cg is 16B.
  asm volatile("{\n cp.async.ca.shared.global [%0], [%1], 8;\n}\n"
      :: "r"(smem), "l"(glob_ptr));
}

__device__ inline void cp_async_fence() { asm volatile("cp.async.commit_group;\n" ::); }

template <int n>
__device__ inline void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(n)); }

template <int lut>
__device__ inline int lop3(int a, int b, int c) {
  int res;
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n" : "=r"(res) : "r"(a), "r"(b), "r"(c), "n"(lut));
  return res;
}

// ============================================================================
// Dequant, MMA, Scale
// ============================================================================

__device__ inline void dequant_u4b8(int q, scalar_t2* frag_b) {
#if defined(USE_BF16_COMPUTE)
  // BF16 dequant: shift-based nibble extraction
  static constexpr uint32_t MASK = 0x000f000f;
  static constexpr uint32_t EX = 0x43004300;
  int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, MASK, EX);
  q >>= 4;
  int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, MASK, EX);
  // Subtract bias via PTX sub.bf16x2 (avoids __hsub2 operator= issues)
  static constexpr uint32_t SUB = 0x43084308;
  uint32_t res_lo, res_hi;
  asm("sub.rn.bf16x2 %0, %1, %2;\n" : "=r"(res_lo) : "r"((uint32_t)lo), "r"(SUB));
  asm("sub.rn.bf16x2 %0, %1, %2;\n" : "=r"(res_hi) : "r"((uint32_t)hi), "r"(SUB));
  reinterpret_cast<uint32_t*>(frag_b)[0] = res_lo;
  reinterpret_cast<uint32_t*>(frag_b)[1] = res_hi;
#else
  // FP16 dequant: dual-mask extraction
  static constexpr int LO = 0x000f000f;
  static constexpr int HI = 0x00f000f0;
  static constexpr int EX = 0x64006400;
  int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, LO, EX);
  int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, HI, EX);
  static constexpr int SUB = 0x64086408;
  static constexpr int MUL = 0x2c002c00;
  static constexpr int ADD = 0xd480d480;
  frag_b[0] = __hsub2(*reinterpret_cast<half2*>(&lo), *reinterpret_cast<const half2*>(&SUB));
  frag_b[1] = __hfma2(*reinterpret_cast<half2*>(&hi), *reinterpret_cast<const half2*>(&MUL),
                       *reinterpret_cast<const half2*>(&ADD));
#endif
}

// ----------------------------------------------------------------------------
// MXFP4 (E2M1) dequant — Kimi-K3 (task #34).
//
// E2M1 nibble is SIGN-MAGNITUDE: bit3 = sign, bits2:0 = eem index into
// {0, 0.5, 1, 1.5, 2, 3, 4, 6}. The additive magic-number trick used by
// dequant_u4b8 (0x4300 bias / sub 0x4308) is intrinsically uint4-zero-point-8
// and CANNOT be parameterized to produce sign-magnitude E2M1 — this is a full
// replacement of the decode, with the identical register contract
// (int q in; frag_b[0] = lanes from nibbles at bits[3:0]/[19:16],
//  frag_b[1] = lanes from bits[7:4]/[23:20]; caller does q >> 8 for the rest).
//
// Recipe (branch-free): plant eem at bf16 bits [8:6] (low 2 exponent bits +
// mantissa MSB), sign at bit 15, then rebias with one mul by 2^126:
//   eem=0        -> bits 0x0000 -> 0.0                (x 2^126 = 0)
//   eem=1        -> bits 0x0040 = 2^-127 (subnormal)  (x 2^126 = 0.5)
//   eem=2e+m,e>0 -> (1+m/2) * 2^(e-127)               (x 2^126 = {1,1.5,2,3,4,6})
// Exact for all 8 magnitudes. NOTE the eem=1 path transits a bf16 SUBNORMAL
// input to mul.rn.bf16x2 (no .ftz variant exists for bf16x2 on sm_90) — the
// GPU parity suite is deliberately +-0.5-heavy to pin this; the fallback if a
// device flushes it is a 2x PRMT byte-LUT (hi {00,3F,3F,3F,40,40,40,40},
// lo {00,00,80,C0,00,40,80,C0}) + sign OR.
//
// Scales are handled by fetch_scale_vec/load_scale_vec: Marlin-ordered E8M0
// bytes are copied asynchronously and expanded to exact BF16 powers of two
// when their shared-memory vector is consumed, so scale_op remains unchanged.
// Residual: scale byte 0x01 (2^-126) times a +-0.5 code
// makes the scale_op PRODUCT itself a bf16 subnormal — unreachable for K3
// (observed scale floor is 112, and repack hard-fails only 0x00/0xFF), but
// revisit this if the legal scale window is ever widened toward the bottom.
// ----------------------------------------------------------------------------
__device__ inline void dequant_e2m1(int q, scalar_t2* frag_b) {
#if defined(USE_BF16_COMPUTE)
  static constexpr uint32_t EEM = 0x00070007;      // magnitude bits per lane
  static constexpr uint32_t SGN = 0x00080008;      // sign bit per lane
  static constexpr uint32_t REBIAS = 0x7E807E80;   // bf16x2 {2^126, 2^126}
  uint32_t uq = (uint32_t)q;
  uint32_t lo = ((uq & EEM) << 6) | ((uq & SGN) << 12);
  uint32_t uq_hi = uq >> 4;
  uint32_t hi = ((uq_hi & EEM) << 6) | ((uq_hi & SGN) << 12);
  uint32_t res_lo, res_hi;
  asm("mul.rn.bf16x2 %0, %1, %2;\n" : "=r"(res_lo) : "r"(lo), "r"(REBIAS));
  asm("mul.rn.bf16x2 %0, %1, %2;\n" : "=r"(res_hi) : "r"(hi), "r"(REBIAS));
  reinterpret_cast<uint32_t*>(frag_b)[0] = res_lo;
  reinterpret_cast<uint32_t*>(frag_b)[1] = res_hi;
#else
  // Both build registrations (setup.py + _jit_registry.py) define
  // USE_BF16_COMPUTE; an FP16 build of the MXFP4 path was never validated and
  // must fail loudly rather than ship an untested decode.
#error "Marlin MXFP4 (E2M1) dequant requires USE_BF16_COMPUTE"
#endif
}

// ----------------------------------------------------------------------------
// Compile-time functors: weight codec + fused-epilogue activation.
// The U4B8/SILU instantiations must stay semantically identical to the
// pre-template K2.5 production kernels (if-constexpr resolves at compile time;
// verify with -Xptxas -v / SASS diff on the GPU stage).
// ----------------------------------------------------------------------------
enum class WCodec { U4B8, E2M1 };
enum class Act { SILU, SITU };

// K3 stores Marlin-ordered E8M0 scale bytes directly. Keep the established
// BF16 shared-memory/fragment contract by expanding each consumed 8-byte
// vector after its asynchronous global-to-shared copy. Byte e represents
// 2^(e-127), whose exact BF16 encoding is uint16(e) << 7.
__device__ inline int4 load_e8m0x8_as_bf16x8(const uint8_t* src) {
  uint64_t packed = *reinterpret_cast<const uint64_t*>(src);
  uint32_t lo = static_cast<uint32_t>(packed);
  uint32_t hi = static_cast<uint32_t>(packed >> 32);
  int4 out;
  uint32_t* words = reinterpret_cast<uint32_t*>(&out);
  words[0] = __byte_perm(lo, 0, 0x4140) << 7;
  words[1] = __byte_perm(lo, 0, 0x4342) << 7;
  words[2] = __byte_perm(hi, 0, 0x4140) << 7;
  words[3] = __byte_perm(hi, 0, 0x4342) << 7;
  return out;
}

template <WCodec CODEC>
__device__ inline void fetch_scale_vec(
    int4* sh_stage, int sh_logical_vec,
    const int4* global_scales, int global_logical_vec) {
  if constexpr (CODEC == WCodec::E2M1) {
    cp_async8(
        reinterpret_cast<uint8_t*>(sh_stage) + sh_logical_vec * 8,
        reinterpret_cast<const uint8_t*>(global_scales) +
            global_logical_vec * 8);
  } else {
    cp_async4(&sh_stage[sh_logical_vec], &global_scales[global_logical_vec]);
  }
}

template <WCodec CODEC>
__device__ inline int4 load_scale_vec(
    const int4* sh_stage, int sh_logical_vec) {
  if constexpr (CODEC == WCodec::E2M1) {
    return load_e8m0x8_as_bf16x8(
        reinterpret_cast<const uint8_t*>(sh_stage) + sh_logical_vec * 8);
  } else {
    return sh_stage[sh_logical_vec];
  }
}

template <WCodec CODEC>
__device__ inline void dequant_w4(int q, scalar_t2* frag_b) {
  if constexpr (CODEC == WCodec::E2M1) {
    dequant_e2m1(q, frag_b);
  } else {
    dequant_u4b8(q, frag_b);
  }
}

// Fused S1 epilogue: combine gate-branch value g (pass 1 / w1) with
// linear-branch value u (pass 2 / w3). fp32 scalar, per output element.
template <Act ACT>
__device__ inline float act_gate_mul(float g, float u) {
  if constexpr (ACT == Act::SITU) {
    // Kimi-K3 SiTU (modeling_kimi_linear.py:75-82; config beta=4.0,
    // linear_beta=25.0). fp32 interior, tanh-soft-clamped both branches:
    //   situ_a = 4 * tanh(g/4) * sigmoid(g)      in (-0.2698, 4)
    //   u_c    = 25 * tanh(u/25)                 in (-25, 25)
    // NOTE branch order is SILENT if swapped — pinned by the GPU mutation
    // test. Under --use_fast_math tanhf/__expf lower to SFU approximations
    // (~2^-11 rel err), inside the 1.6e-2 parity gate; de-fast-math this
    // epilogue only if parity fails.
    float situ_a = 4.0f * tanhf(0.25f * g) * (1.0f / (1.0f + __expf(-g)));
    float u_c = 25.0f * tanhf(0.04f * u);
    return situ_a * u_c;
  } else {
    return g / (1.0f + __expf(-g)) * u;
  }
}

// mma_trans: swaps A and B operand positions for m_block_size_8
// B fragments go into the 4-register A operand slot (interleaved b0, b1)
// A fragment goes into the 2-register B operand slot (ldsm<2>)
__device__ inline void mma_trans_op(
    const FragA& a_frag, const FragB& frag_b, const FragB& frag_b2, FragC& frag_c) {
  const uint32_t* a = reinterpret_cast<const uint32_t*>(&a_frag);
  const uint32_t* b = reinterpret_cast<const uint32_t*>(&frag_b);
  const uint32_t* b2 = reinterpret_cast<const uint32_t*>(&frag_b2);
  float* c = reinterpret_cast<float*>(&frag_c);
#if defined(USE_BF16_COMPUTE)
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
      : "r"(b[0]), "r"(b2[0]), "r"(b[1]), "r"(b2[1]),
        "r"(a[0]), "r"(a[1]),
        "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
#else
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
      : "r"(b[0]), "r"(b2[0]), "r"(b[1]), "r"(b2[1]),
        "r"(a[0]), "r"(a[1]),
        "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
#endif
}

// ldsm<2> for m_block_size_8 (loads 2 fragments instead of 4)
__device__ inline void ldsm2(FragA& frag_a, const void* smem_ptr) {
  uint32_t* a = reinterpret_cast<uint32_t*>(&frag_a);
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(a[0]), "=r"(a[1]) : "r"(smem));
}

// Standard mma for M16: A in 4-register slot, B in 2-register slot
__device__ inline void mma_op(const FragA& a_frag, const FragB& frag_b, FragC& frag_c) {
  const uint32_t* a = reinterpret_cast<const uint32_t*>(&a_frag);
  const uint32_t* b = reinterpret_cast<const uint32_t*>(&frag_b);
  float* c = reinterpret_cast<float*>(&frag_c);
#if defined(USE_BF16_COMPUTE)
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
        "r"(b[0]), "r"(b[1]), "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
#else
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
        "r"(b[0]), "r"(b[1]), "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
#endif
}

// ldsm<4> for M16 (loads 4 fragments)
__device__ inline void ldsm4(FragA& frag_a, const void* smem_ptr) {
  uint32_t* a = reinterpret_cast<uint32_t*>(&frag_a);
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(smem));
}

__device__ inline void scale_op(FragB& frag_b, FragS& frag_s, int i) {
  scalar_t2 s = num2num2(reinterpret_cast<scalar_t*>(&frag_s)[i]);
  frag_b[0] = __hmul2(frag_b[0], s);
  frag_b[1] = __hmul2(frag_b[1], s);
}

// ============================================================================
// Hardcoded constants
// ============================================================================

static constexpr int THREADS = 256;
static constexpr int TM = 1;    // thread_m_blocks
static constexpr int TN = 16;   // thread_n_blocks (256 output cols)
static constexpr int TK = 8;    // thread_k_blocks (128 K per stage)
static constexpr int STAGES = 4;
static constexpr int GROUP_BLOCKS = 2;  // group_size=32, block=16 (K2.5 native gs)
static constexpr int PACK = 8;          // 32/4
static constexpr int MBLOCK = 8;        // m_block_size_8 = true

// Derived
static constexpr int a_sh_stride = 16 * TK / 8;     // 16
static constexpr int a_gl_rd_delta_o = a_sh_stride;  // 16
static constexpr int a_sh_wr_delta = a_sh_stride * (THREADS / a_gl_rd_delta_o);  // 256
static constexpr int a_sh_rd_delta_o = 2 * ((THREADS / 32) / (TN / 4));  // 4
static constexpr int a_sh_rd_delta_i = a_sh_stride * 16;  // 256
static constexpr int a_sh_stage = a_sh_stride * MBLOCK;   // 256
static constexpr int a_sh_wr_iters = div_ceil(a_sh_stage, a_sh_wr_delta);  // 1

static constexpr int b_sh_stride = ((TN * 16) * 16 / PACK) / 4;  // 128
static constexpr int b_sh_stride_threads = b_sh_stride;  // 128 (b_thread_vecs=1)
static constexpr int b_sh_wr_delta = THREADS;  // 256
static constexpr int b_sh_stage = b_sh_stride * TK;  // 1024
static constexpr int b_sh_wr_iters = b_sh_stage / b_sh_wr_delta;  // 4

static constexpr int s_sh_stride = 16 * TN / 8;  // 32
static constexpr int s_tb_groups = TK / GROUP_BLOCKS;  // 4 (scale groups per pipeline stage)
static constexpr int s_sh_stage = s_tb_groups * s_sh_stride;  // 128

static constexpr int k_iter_size = TK * 16 / b_sh_wr_iters;  // 32 (K elements per k-iteration)

static constexpr int sh_red_size = (2 * TN + 1) * 16 * TM;  // 528
static constexpr int sh_b_size = STAGES * b_sh_stage;        // 4096
static constexpr int sh_s_size = STAGES * s_sh_stage;        // 512


// ============================================================================
// Simplified kernel: one CTA = one (expert, N-tile), full K-reduction
// ============================================================================

__global__ void MarlinSimple(
    const int4* __restrict__ A,
    const int4* const* __restrict__ B_ptrs,
    int4* const* __restrict__ C_ptrs,
    const int4* const* __restrict__ scales_ptrs,
    const int* __restrict__ expert_starts,     // [E] row offset per expert (stride-based)
    const int* __restrict__ expert_counts,     // [E] actual token count per expert
    int num_experts,
    int prob_n, int prob_k, int lda,
    int n_tiles_per_expert)
{
  // Dispatch
  int matrix_idx = blockIdx.x / n_tiles_per_expert;
  int tile_idx = blockIdx.x % n_tiles_per_expert;
  int expert_idx = matrix_idx % num_experts;

  int prob_m = expert_counts[expert_idx];
  if (prob_m <= 0) return;
  int token_start = expert_starts[expert_idx];

  const int4* A_ptr = A + token_start * (lda / 8);
  const int4* B = B_ptrs[matrix_idx];
  int4* C = C_ptrs[matrix_idx];
  const int4* scales_ptr = scales_ptrs[matrix_idx];

  int k_tiles = prob_k / 16 / TK;  // 56 for K=7168

  // A indices
  int a_gl_stride = lda / 8;
  int a_gl_rd_delta_i = a_gl_stride * (THREADS / a_gl_rd_delta_o);
  int a_gl_rd = a_gl_stride * (threadIdx.x / a_gl_rd_delta_o) + (threadIdx.x % a_gl_rd_delta_o);

  int a_sh_wr = a_sh_stride * (threadIdx.x / a_gl_rd_delta_o) + (threadIdx.x % a_gl_rd_delta_o);
  // m_block_size_8: different A shared read pattern (8 rows instead of 16)
  int a_sh_rd = a_sh_stride * ((threadIdx.x % 32) % 8) + (threadIdx.x % 32) / 8;
  a_sh_rd += 2 * ((threadIdx.x / 32) / (TN / 4));

  // B indices
  int b_gl_stride = 16 * prob_n / (PACK * 4);
  int b_gl_rd_delta_o = b_gl_stride * TK;
  int b_gl_rd_delta_i = b_gl_stride * (THREADS / b_sh_stride_threads);
  int b_gl_rd = b_gl_stride * (threadIdx.x / b_sh_stride_threads) + (threadIdx.x % b_sh_stride_threads);
  b_gl_rd += b_sh_stride * tile_idx;  // offset to our N-tile

  // Scale indices
  int s_gl_stride = prob_n / 8;
  int s_gl_rd = s_sh_stride * tile_idx + threadIdx.x;
  bool s_sh_wr_pred = threadIdx.x < s_sh_stride;
  int s_sh_rd = 8 * ((threadIdx.x / 32) % (TN / 4)) + (threadIdx.x % 32) / 4;

  // A predicates
  bool a_sh_wr_pred[a_sh_wr_iters];
#pragma unroll
  for (int i = 0; i < a_sh_wr_iters; i++)
    a_sh_wr_pred[i] = a_sh_wr_delta * i + a_sh_wr < a_sh_stride * prob_m;

  // XOR transform for A shared memory
  auto transform_a = [&](int i) {
    int row = i / a_gl_rd_delta_o;
    return a_gl_rd_delta_o * row + (i % a_gl_rd_delta_o) ^ (row % 8);
  };

  int a_sh_wr_trans[a_sh_wr_iters];
#pragma unroll
  for (int i = 0; i < a_sh_wr_iters; i++)
    a_sh_wr_trans[i] = transform_a(a_sh_wr_delta * i + a_sh_wr);

  int a_sh_rd_trans[b_sh_wr_iters][TM];
#pragma unroll
  for (int i = 0; i < b_sh_wr_iters; i++)
    for (int j = 0; j < TM; j++)
      a_sh_rd_trans[i][j] = transform_a(a_sh_rd_delta_o * i + a_sh_rd_delta_i * j + a_sh_rd);

  // B pointers
  const int4* B_ptr[b_sh_wr_iters];
#pragma unroll
  for (int i = 0; i < b_sh_wr_iters; i++)
    B_ptr[i] = B + b_gl_rd_delta_i * i + b_gl_rd;

  // Shared memory
  extern __shared__ int4 sh[];
  int4* sh_b = sh;
  int4* sh_red = sh;
  int4* sh_s = sh + (sh_red_size > sh_b_size ? sh_red_size : sh_b_size);
  int4* sh_a = sh_s + sh_s_size;

  // Registers
  FragA frag_a[2][TM];
  I4 frag_b_quant[2][1];
  FragC frag_c[TM][4][2];
  FragS frag_s[2][4];

  // Zero accumulators
#pragma unroll
  for (int i = 0; i < TM * 4 * 2 * 4; i++)
    reinterpret_cast<float*>(frag_c)[i] = 0;

  // ---- Pipeline: fetch A, B, scales to SMEM ----
  auto fetch_to_shared = [&](int pipe, int k_off, bool pred) {
    if (pred) {
      int4* sh_a_stage = sh_a + a_sh_stage * pipe;
#pragma unroll
      for (int i = 0; i < a_sh_wr_iters; i++)
        cp_async4_pred(&sh_a_stage[a_sh_wr_trans[i]],
                       &A_ptr[a_gl_rd_delta_i * i + a_gl_rd + a_gl_rd_delta_o * k_off],
                       a_sh_wr_pred[i]);

      int4* sh_b_stage = sh_b + b_sh_stage * pipe;
#pragma unroll
      for (int i = 0; i < b_sh_wr_iters; i++) {
        cp_async4(&sh_b_stage[b_sh_wr_delta * i + threadIdx.x], B_ptr[i]);
        B_ptr[i] += b_gl_rd_delta_o;
      }

      // Scales (GROUP_BLOCKS=2: load s_tb_groups=4 scale rows per stage)
      int4* sh_s_stage = sh_s + s_sh_stage * pipe;
#pragma unroll
      for (int i = 0; i < s_tb_groups; i++) {
        if (s_sh_wr_pred)
          cp_async4(&sh_s_stage[i * s_sh_stride + threadIdx.x], &scales_ptr[s_gl_rd]);
        s_gl_rd += s_gl_stride;
      }
    }
    cp_async_fence();
  };

  // ---- Startup: fill pipeline ----
#pragma unroll
  for (int i = 0; i < STAGES - 1; i++)
    fetch_to_shared(i, i, i < k_tiles);

  cp_async_wait<STAGES - 2>();
  __syncthreads();

  // Load first registers
  auto load_regs = [&](int k, int pipe) {
    int4* sh_a_stage = sh_a + a_sh_stage * pipe;
#pragma unroll
    for (int i = 0; i < TM; i++)
      ldsm2(frag_a[k % 2][i], &sh_a_stage[a_sh_rd_trans[k % b_sh_wr_iters][i]]);
    int4* sh_b_stage = sh_b + b_sh_stage * pipe;
    frag_b_quant[k % 2][0] = *reinterpret_cast<I4*>(
        &sh_b_stage[b_sh_wr_delta * (k % b_sh_wr_iters) + threadIdx.x]);
  };

  auto load_scales = [&](int k, int pipe) {
    // GROUP_BLOCKS=2: compute which scale group this k-iteration belongs to
    int cur_k = k_iter_size * (k % b_sh_wr_iters);
    int k_blocks = cur_k / 16;
    int cur_group_id = k_blocks / GROUP_BLOCKS;

    int4* sh_s_stage = sh_s + s_sh_stage * (pipe % STAGES);
    reinterpret_cast<int4*>(&frag_s[k % 2])[0] =
        sh_s_stage[s_sh_rd + cur_group_id * s_sh_stride];
  };

  auto matmul = [&](int k) {
    int k2 = k % 2;
#pragma unroll
    for (int j = 0; j < 4; j++) {
      FragB frag_b0, frag_b1;
      int b_quant_0 = frag_b_quant[k2][0][j];
      int b_quant_1 = b_quant_0 >> 8;
      dequant_u4b8(b_quant_0, reinterpret_cast<scalar_t2*>(&frag_b0));
      dequant_u4b8(b_quant_1, reinterpret_cast<scalar_t2*>(&frag_b1));
      scale_op(frag_b0, frag_s[k2][j], 0);
      scale_op(frag_b1, frag_s[k2][j], 1);
#pragma unroll
      for (int i = 0; i < TM; i++) {
        // m_block_size_8: single mma_trans combines both b0 and b1
        mma_trans_op(frag_a[k2][i], frag_b0, frag_b1, frag_c[i][j][0]);
      }
    }
  };

  // Load first register tile
  load_regs(0, 0);
  load_scales(0, 0);
  a_gl_rd += a_gl_rd_delta_o * (STAGES - 1);

  // ---- Main loop: process all K-tiles ----
  int slice_iters = k_tiles;
  while (slice_iters) {
#pragma unroll
    for (int pipe = 0; pipe < STAGES;) {
#pragma unroll
      for (int k = 0; k < b_sh_wr_iters; k++) {
        load_regs(k + 1, pipe % STAGES);
        load_scales(k + 1, pipe);
        if (k == b_sh_wr_iters - 2) {
          fetch_to_shared((pipe + STAGES - 1) % STAGES, pipe, slice_iters >= STAGES);
          pipe++;
          cp_async_wait<STAGES - 2>();
          __syncthreads();
        }
        matmul(k);
      }
      slice_iters--;
      if (slice_iters == 0) break;
    }
    a_gl_rd += a_gl_rd_delta_o * STAGES;
  }
  cp_async_wait<0>();

  // ---- Thread-block reduce ----
  {
    constexpr int red_off = THREADS / b_sh_stride_threads / 2;  // 1
    if constexpr (red_off >= 1) {
      auto red_idx = threadIdx.x / b_sh_stride_threads;
      constexpr int red_sh_stride = b_sh_stride_threads * 4 * 2;
      constexpr int red_sh_delta = b_sh_stride_threads;
      int red_sh_rd = red_sh_stride * (threadIdx.x / b_sh_stride_threads) +
                      (threadIdx.x % b_sh_stride_threads);
#pragma unroll
      for (int m = 0; m < TM; m++) {
#pragma unroll
        for (int i = red_off; i > 0; i /= 2) {
          if (i <= red_idx && red_idx < 2 * i) {
#pragma unroll
            for (int j = 0; j < 4 * 2; j += 2) {  // m_block_size_8: step by 2
              int red_sh_wr = red_sh_delta * j + (red_sh_rd - red_sh_stride * i);
              if (i < red_off) {
                float* c_rd = reinterpret_cast<float*>(&sh_red[red_sh_delta * j + red_sh_rd]);
                float* c_wr = reinterpret_cast<float*>(&sh_red[red_sh_wr]);
#pragma unroll
                for (int kk = 0; kk < 4; kk++)
                  reinterpret_cast<FragC*>(frag_c)[4 * 2 * m + j][kk] += c_rd[kk] + c_wr[kk];
              }
              sh_red[red_sh_wr] = reinterpret_cast<int4*>(&frag_c)[4 * 2 * m + j];
            }
          }
          __syncthreads();
        }
        if (red_idx == 0) {
#pragma unroll
          for (int i = 0; i < 4 * 2; i += 2) {  // m_block_size_8: step by 2
            float* c_rd = reinterpret_cast<float*>(&sh_red[red_sh_delta * i + red_sh_rd]);
#pragma unroll
            for (int j = 0; j < 4; j++)
              reinterpret_cast<FragC*>(frag_c)[4 * 2 * m + i][j] += c_rd[j];
          }
        }
        __syncthreads();
      }
    }
  }

  // ---- Write result (m_block_size_8 layout) ----
  {
    int c_gl_stride = prob_n / 8;
    constexpr int c_sh_stride = 2 * TN + 1;
    int c_gl_wr_delta = c_gl_stride * (THREADS / (2 * TN));
    constexpr int c_sh_rd_delta = c_sh_stride * (THREADS / (2 * TN));

    int c_gl_wr = c_gl_stride * (threadIdx.x / (2 * TN)) + (threadIdx.x % (2 * TN));
    c_gl_wr += (2 * TN) * tile_idx;
    // m_block_size_8: different SMEM write layout
    int c_sh_wr = (8 * c_sh_stride) * ((threadIdx.x % 32) % 4 * 2) + (threadIdx.x % 32) / 4;
    c_sh_wr += 64 * (threadIdx.x / 32);
    int c_sh_rd = c_sh_stride * (threadIdx.x / (2 * TN)) + (threadIdx.x % (2 * TN));
    int c_gl_wr_end = c_gl_stride * prob_m;

    // m_block_size_8: per-element write (not per-pair)
    auto write_m8 = [&](int idx, float c0, float c1) {
      scalar_t2 res = nums2num2(float2num(c0), float2num(c1));
      ((scalar_t*)sh_red)[idx] = res.x;
      ((scalar_t*)sh_red)[idx + 8 * c_sh_stride] = res.y;
    };

    if (threadIdx.x / 32 < TN / 4) {
#pragma unroll
      for (int i = 0; i < TM; i++) {
#pragma unroll
        for (int j = 0; j < 4; j++) {
          int wr = c_sh_wr + 16 * j;
          write_m8(wr, frag_c[i][j][0][0], frag_c[i][j][0][1]);
          write_m8(wr + 8, frag_c[i][j][0][2], frag_c[i][j][0][3]);
        }
        c_sh_wr += 16 * (4 * c_sh_stride);
      }
    }
    __syncthreads();

    // m_block_size_8: fewer rows to write (8 instead of 16)
#pragma unroll
    for (int i = 0; i < div_ceil(8 * TM, THREADS / (2 * TN)); i++) {
      if (c_gl_wr < c_gl_wr_end) {
        C[c_gl_wr] = sh_red[c_sh_rd];
        c_gl_wr += c_gl_wr_delta;
        c_sh_rd += c_sh_rd_delta;
      }
    }
  }
}

// ============================================================================
// M16 kernel: MBLOCK=16, standard mma, CTA M-tiling for any M
// ============================================================================

// M16-specific constants (override M8 MBLOCK-dependent values)
static constexpr int M16_MBLOCK = 16;
static constexpr int m16_a_sh_stage = a_sh_stride * M16_MBLOCK;         // 256
static constexpr int m16_a_sh_wr_iters = div_ceil(m16_a_sh_stage, a_sh_wr_delta);  // 1
static constexpr int m16_sh_a_size = STAGES * m16_a_sh_stage;           // 1024
// Total SMEM M16: max(528, 4096) + 512 + 1024 = 5632 int4 = 90112 bytes
// Total SMEM M16_S1 (fused): + 528 for gate result = 6160 int4 = 98560 bytes
static constexpr int sh_gate_size = sh_red_size;  // 528 int4 — stores gate BF16 result

// ============================================================================
// Fused S1 kernel: gate+up+activation in single kernel, no temp buffer.
// Templated on weight codec (U4B8 = K2.5 INT4, E2M1 = K3 MXFP4) and epilogue
// activation (SILU = K2.5, SITU = K3). <U4B8, SILU> is the production K2.5
// instantiation and must stay bit-identical to the pre-template kernel.
// ============================================================================

template <WCodec CODEC, Act ACT>
__global__ void MarlinGrouped_M16_S1(
    const int4* __restrict__ A,
    const int4* const* __restrict__ gate_B_ptrs,     // [E] gate weight ptrs
    const int4* const* __restrict__ up_B_ptrs,       // [E] up weight ptrs
    int4* const* __restrict__ C_ptrs,                // [E] output ptrs (into intermediate)
    const int4* const* __restrict__ gate_scales_ptrs, // [E] gate scale ptrs
    const int4* const* __restrict__ up_scales_ptrs,   // [E] up scale ptrs
    const int* __restrict__ expert_starts,
    const int* __restrict__ expert_counts,
    int num_experts,
    int prob_n, int prob_k, int lda,
    int n_tiles_per_expert,
    int max_m_tiles)
{
  // CTA dispatch: grid = E × max_m_tiles × n_tiles
  int linear_idx = blockIdx.x;
  int expert_idx = linear_idx / (max_m_tiles * n_tiles_per_expert);
  int remainder = linear_idx % (max_m_tiles * n_tiles_per_expert);
  int m_tile_idx = remainder / n_tiles_per_expert;
  int tile_idx = remainder % n_tiles_per_expert;

  int expert_m = expert_counts[expert_idx];
  int m_start = m_tile_idx * M16_MBLOCK;
  if (m_start >= expert_m) return;

  int prob_m = min(M16_MBLOCK, expert_m - m_start);
  int token_start = expert_starts[expert_idx] + m_start;

  const int4* A_ptr = A + token_start * (lda / 8);
  int c_gl_stride_out = prob_n / 8;
  int4* C = C_ptrs[expert_idx] + m_start * c_gl_stride_out;

  int k_tiles = prob_k / 16 / TK;

  // A indices (constant across gate/up passes — A is the same)
  int a_gl_stride = lda / 8;
  int a_gl_rd_delta_i = a_gl_stride * (THREADS / a_gl_rd_delta_o);
  int a_gl_rd_base = a_gl_stride * (threadIdx.x / a_gl_rd_delta_o) + (threadIdx.x % a_gl_rd_delta_o);

  int a_sh_wr = a_sh_stride * (threadIdx.x / a_gl_rd_delta_o) + (threadIdx.x % a_gl_rd_delta_o);
  int a_sh_rd = a_sh_stride * ((threadIdx.x % 32) % 16) + (threadIdx.x % 32) / 16;
  a_sh_rd += 2 * ((threadIdx.x / 32) / (TN / 4));

  // B indices (N-tile offset, reused for both passes)
  int b_gl_stride = 16 * prob_n / (PACK * 4);
  int b_gl_rd_delta_o = b_gl_stride * TK;
  int b_gl_rd_delta_i = b_gl_stride * (THREADS / b_sh_stride_threads);
  int b_gl_rd_base = b_gl_stride * (threadIdx.x / b_sh_stride_threads) + (threadIdx.x % b_sh_stride_threads);
  b_gl_rd_base += b_sh_stride * tile_idx;

  // Scale indices
  int s_gl_stride = prob_n / 8;
  int s_gl_rd_base = s_sh_stride * tile_idx + threadIdx.x;
  bool s_sh_wr_pred = threadIdx.x < s_sh_stride;
  int s_sh_rd = 8 * ((threadIdx.x / 32) % (TN / 4)) + (threadIdx.x % 32) / 4;

  // A predicates
  bool a_sh_wr_pred[m16_a_sh_wr_iters];
#pragma unroll
  for (int i = 0; i < m16_a_sh_wr_iters; i++)
    a_sh_wr_pred[i] = a_sh_wr_delta * i + a_sh_wr < a_sh_stride * prob_m;

  // XOR transform for A
  auto transform_a = [&](int i) {
    int row = i / a_gl_rd_delta_o;
    return a_gl_rd_delta_o * row + (i % a_gl_rd_delta_o) ^ (row % 8);
  };

  int a_sh_wr_trans[m16_a_sh_wr_iters];
#pragma unroll
  for (int i = 0; i < m16_a_sh_wr_iters; i++)
    a_sh_wr_trans[i] = transform_a(a_sh_wr_delta * i + a_sh_wr);

  int a_sh_rd_trans[b_sh_wr_iters][TM];
#pragma unroll
  for (int i = 0; i < b_sh_wr_iters; i++)
    for (int j = 0; j < TM; j++)
      a_sh_rd_trans[i][j] = transform_a(a_sh_rd_delta_o * i + a_sh_rd_delta_i * j + a_sh_rd);

  // Shared memory layout (with gate result area)
  extern __shared__ int4 sh[];
  int4* sh_b = sh;
  int4* sh_red = sh;
  int4* sh_s = sh + (sh_red_size > sh_b_size ? sh_red_size : sh_b_size);
  int4* sh_a = sh_s + sh_s_size;
  int4* sh_gate = sh_a + m16_sh_a_size;  // gate BF16 result (528 int4 = 8.4 KB)

  // Registers
  FragA frag_a[2][TM];
  I4 frag_b_quant[2][1];
  FragC frag_c[TM][4][2];
  FragS frag_s[2][4];

  // ---- Lambdas for pipeline ops (capture mutable state via references) ----
  // These reference B_ptr, scales_ptr, s_gl_rd, a_gl_rd which are reset between passes
  const int4* B_ptr_arr[b_sh_wr_iters];
  const int4* cur_scales_ptr;
  int s_gl_rd;
  int a_gl_rd;

  auto fetch_to_shared = [&](int pipe, int k_off, bool pred) {
    if (pred) {
      int4* sh_a_stage = sh_a + m16_a_sh_stage * pipe;
#pragma unroll
      for (int i = 0; i < m16_a_sh_wr_iters; i++)
        cp_async4_pred(&sh_a_stage[a_sh_wr_trans[i]],
                       &A_ptr[a_gl_rd_delta_i * i + a_gl_rd + a_gl_rd_delta_o * k_off],
                       a_sh_wr_pred[i]);
      int4* sh_b_stage = sh_b + b_sh_stage * pipe;
#pragma unroll
      for (int i = 0; i < b_sh_wr_iters; i++) {
        cp_async4(&sh_b_stage[b_sh_wr_delta * i + threadIdx.x], B_ptr_arr[i]);
        B_ptr_arr[i] += b_gl_rd_delta_o;
      }
      int4* sh_s_stage = sh_s + s_sh_stage * pipe;
#pragma unroll
      for (int i = 0; i < s_tb_groups; i++) {
        if (s_sh_wr_pred) {
          fetch_scale_vec<CODEC>(
              sh_s_stage, i * s_sh_stride + threadIdx.x,
              cur_scales_ptr, s_gl_rd);
        }
        s_gl_rd += s_gl_stride;
      }
    }
    cp_async_fence();
  };

  auto load_regs = [&](int k, int pipe) {
    int4* sh_a_stage = sh_a + m16_a_sh_stage * pipe;
#pragma unroll
    for (int i = 0; i < TM; i++)
      ldsm4(frag_a[k % 2][i], &sh_a_stage[a_sh_rd_trans[k % b_sh_wr_iters][i]]);
    int4* sh_b_stage = sh_b + b_sh_stage * pipe;
    frag_b_quant[k % 2][0] = *reinterpret_cast<I4*>(
        &sh_b_stage[b_sh_wr_delta * (k % b_sh_wr_iters) + threadIdx.x]);
  };

  auto load_scales = [&](int k, int pipe) {
    int cur_k = k_iter_size * (k % b_sh_wr_iters);
    int k_blocks = cur_k / 16;
    int cur_group_id = k_blocks / GROUP_BLOCKS;
    int4* sh_s_stage = sh_s + s_sh_stage * (pipe % STAGES);
    reinterpret_cast<int4*>(&frag_s[k % 2])[0] = load_scale_vec<CODEC>(
        sh_s_stage, s_sh_rd + cur_group_id * s_sh_stride);
  };

  auto matmul = [&](int k) {
    int k2 = k % 2;
#pragma unroll
    for (int j = 0; j < 4; j++) {
      FragB frag_b0, frag_b1;
      int b_quant_0 = frag_b_quant[k2][0][j];
      int b_quant_1 = b_quant_0 >> 8;
      dequant_w4<CODEC>(b_quant_0, reinterpret_cast<scalar_t2*>(&frag_b0));
      dequant_w4<CODEC>(b_quant_1, reinterpret_cast<scalar_t2*>(&frag_b1));
      scale_op(frag_b0, frag_s[k2][j], 0);
      scale_op(frag_b1, frag_s[k2][j], 1);
#pragma unroll
      for (int i = 0; i < TM; i++) {
        mma_op(frag_a[k2][i], frag_b0, frag_c[i][j][0]);
        mma_op(frag_a[k2][i], frag_b1, frag_c[i][j][1]);
      }
    }
  };

  auto run_reduction = [&]() {
    constexpr int red_off = THREADS / b_sh_stride_threads / 2;
    if constexpr (red_off >= 1) {
      auto red_idx = threadIdx.x / b_sh_stride_threads;
      constexpr int red_sh_stride = b_sh_stride_threads * 4 * 2;
      constexpr int red_sh_delta = b_sh_stride_threads;
      int red_sh_rd = red_sh_stride * (threadIdx.x / b_sh_stride_threads) +
                      (threadIdx.x % b_sh_stride_threads);
#pragma unroll
      for (int m = 0; m < TM; m++) {
#pragma unroll
        for (int i = red_off; i > 0; i /= 2) {
          if (i <= red_idx && red_idx < 2 * i) {
#pragma unroll
            for (int j = 0; j < 4 * 2; j++) {
              int red_sh_wr = red_sh_delta * j + (red_sh_rd - red_sh_stride * i);
              if (i < red_off) {
                float* c_rd = reinterpret_cast<float*>(&sh_red[red_sh_delta * j + red_sh_rd]);
                float* c_wr = reinterpret_cast<float*>(&sh_red[red_sh_wr]);
#pragma unroll
                for (int kk = 0; kk < 4; kk++)
                  reinterpret_cast<FragC*>(frag_c)[4 * 2 * m + j][kk] += c_rd[kk] + c_wr[kk];
              }
              sh_red[red_sh_wr] = reinterpret_cast<int4*>(&frag_c)[4 * 2 * m + j];
            }
          }
          __syncthreads();
        }
        if (red_idx == 0) {
#pragma unroll
          for (int i = 0; i < 4 * 2; i++) {
            float* c_rd = reinterpret_cast<float*>(&sh_red[red_sh_delta * i + red_sh_rd]);
#pragma unroll
            for (int j = 0; j < 4; j++)
              reinterpret_cast<FragC*>(frag_c)[4 * 2 * m + i][j] += c_rd[j];
          }
        }
        __syncthreads();
      }
    }
  };

  auto write_result_to_smem = [&](int4* dst) {
    // Write reduced result from frag_c → sh_red, then copy to dst
    constexpr int c_sh_stride = 2 * TN + 1;
    int c_sh_wr = (4 * c_sh_stride) * ((threadIdx.x % 32) / 4) + (threadIdx.x % 32) % 4;
    c_sh_wr += 32 * (threadIdx.x / 32);

    if (threadIdx.x / 32 < TN / 4) {
#pragma unroll
      for (int i = 0; i < TM; i++) {
#pragma unroll
        for (int j = 0; j < 4; j++) {
          int wr = c_sh_wr + 8 * j;
          scalar_t2 r0 = nums2num2(float2num(frag_c[i][j][0][0]), float2num(frag_c[i][j][0][1]));
          scalar_t2 r1 = nums2num2(float2num(frag_c[i][j][0][2]), float2num(frag_c[i][j][0][3]));
          scalar_t2 r2 = nums2num2(float2num(frag_c[i][j][1][0]), float2num(frag_c[i][j][1][1]));
          scalar_t2 r3 = nums2num2(float2num(frag_c[i][j][1][2]), float2num(frag_c[i][j][1][3]));
          ((scalar_t2*)sh_red)[wr + (4 * c_sh_stride) * 0 + 0] = r0;
          ((scalar_t2*)sh_red)[wr + (4 * c_sh_stride) * 8 + 0] = r1;
          ((scalar_t2*)sh_red)[wr + (4 * c_sh_stride) * 0 + 4] = r2;
          ((scalar_t2*)sh_red)[wr + (4 * c_sh_stride) * 8 + 4] = r3;
        }
        c_sh_wr += 16 * (4 * c_sh_stride);
      }
    }
    __syncthreads();

    // Copy sh_red → dst (gate result area)
#pragma unroll
    for (int i = threadIdx.x; i < sh_red_size; i += THREADS)
      dst[i] = sh_red[i];
    __syncthreads();
  };

  auto run_pipeline = [&]() {
    // Fill pipeline
#pragma unroll
    for (int i = 0; i < STAGES - 1; i++)
      fetch_to_shared(i, i, i < k_tiles);
    cp_async_wait<STAGES - 2>();
    __syncthreads();

    // Load first registers
    load_regs(0, 0);
    load_scales(0, 0);
    a_gl_rd += a_gl_rd_delta_o * (STAGES - 1);

    // Main loop
    int slice_iters = k_tiles;
    while (slice_iters) {
#pragma unroll
      for (int pipe = 0; pipe < STAGES;) {
#pragma unroll
        for (int k = 0; k < b_sh_wr_iters; k++) {
          load_regs(k + 1, pipe % STAGES);
          load_scales(k + 1, pipe);
          if (k == b_sh_wr_iters - 2) {
            fetch_to_shared((pipe + STAGES - 1) % STAGES, pipe, slice_iters >= STAGES);
            pipe++;
            cp_async_wait<STAGES - 2>();
            __syncthreads();
          }
          matmul(k);
        }
        slice_iters--;
        if (slice_iters == 0) break;
      }
      a_gl_rd += a_gl_rd_delta_o * STAGES;
    }
    cp_async_wait<0>();
  };

  // ================================================================
  // PASS 1: Gate K-reduction
  // ================================================================
  {
    const int4* B = gate_B_ptrs[expert_idx];
    cur_scales_ptr = gate_scales_ptrs[expert_idx];
    s_gl_rd = s_gl_rd_base;
    a_gl_rd = a_gl_rd_base;
#pragma unroll
    for (int i = 0; i < b_sh_wr_iters; i++)
      B_ptr_arr[i] = B + b_gl_rd_delta_i * i + b_gl_rd_base;

    // Zero accumulators
#pragma unroll
    for (int i = 0; i < TM * 4 * 2 * 4; i++)
      reinterpret_cast<float*>(frag_c)[i] = 0;

    run_pipeline();
    run_reduction();
    write_result_to_smem(sh_gate);  // gate BF16 → sh_gate
  }

  // ================================================================
  // PASS 2: Up K-reduction
  // ================================================================
  {
    const int4* B = up_B_ptrs[expert_idx];
    cur_scales_ptr = up_scales_ptrs[expert_idx];
    s_gl_rd = s_gl_rd_base;
    a_gl_rd = a_gl_rd_base;
#pragma unroll
    for (int i = 0; i < b_sh_wr_iters; i++)
      B_ptr_arr[i] = B + b_gl_rd_delta_i * i + b_gl_rd_base;

    // Zero accumulators
#pragma unroll
    for (int i = 0; i < TM * 4 * 2 * 4; i++)
      reinterpret_cast<float*>(frag_c)[i] = 0;

    run_pipeline();
    run_reduction();

    // Write up result to sh_red (standard write-back pattern)
    constexpr int c_sh_stride = 2 * TN + 1;
    int c_sh_wr = (4 * c_sh_stride) * ((threadIdx.x % 32) / 4) + (threadIdx.x % 32) % 4;
    c_sh_wr += 32 * (threadIdx.x / 32);

    if (threadIdx.x / 32 < TN / 4) {
#pragma unroll
      for (int i = 0; i < TM; i++) {
#pragma unroll
        for (int j = 0; j < 4; j++) {
          int wr = c_sh_wr + 8 * j;
          scalar_t2 r0 = nums2num2(float2num(frag_c[i][j][0][0]), float2num(frag_c[i][j][0][1]));
          scalar_t2 r1 = nums2num2(float2num(frag_c[i][j][0][2]), float2num(frag_c[i][j][0][3]));
          scalar_t2 r2 = nums2num2(float2num(frag_c[i][j][1][0]), float2num(frag_c[i][j][1][1]));
          scalar_t2 r3 = nums2num2(float2num(frag_c[i][j][1][2]), float2num(frag_c[i][j][1][3]));
          ((scalar_t2*)sh_red)[wr + (4 * c_sh_stride) * 0 + 0] = r0;
          ((scalar_t2*)sh_red)[wr + (4 * c_sh_stride) * 8 + 0] = r1;
          ((scalar_t2*)sh_red)[wr + (4 * c_sh_stride) * 0 + 4] = r2;
          ((scalar_t2*)sh_red)[wr + (4 * c_sh_stride) * 8 + 4] = r3;
        }
        c_sh_wr += 16 * (4 * c_sh_stride);
      }
    }
    __syncthreads();
  }

  // ================================================================
  // FUSED WRITE-BACK: act(gate, up) → output C
  // (SILU: SiLU(gate) * up — K2.5; SITU: Kimi-K3, see act_gate_mul)
  // ================================================================
  {
    int c_gl_stride = prob_n / 8;
    constexpr int c_sh_stride = 2 * TN + 1;
    int c_gl_wr_delta = c_gl_stride * (THREADS / (2 * TN));
    constexpr int c_sh_rd_delta = c_sh_stride * (THREADS / (2 * TN));

    int c_gl_wr = c_gl_stride * (threadIdx.x / (2 * TN)) + (threadIdx.x % (2 * TN));
    c_gl_wr += (2 * TN) * tile_idx;
    int c_sh_rd = c_sh_stride * (threadIdx.x / (2 * TN)) + (threadIdx.x % (2 * TN));
    int c_gl_wr_end = c_gl_stride * prob_m;

    // Read gate from sh_gate, up from sh_red (same layout), fuse SiLU
#pragma unroll
    for (int i = 0; i < div_ceil(16 * TM, THREADS / (2 * TN)); i++) {
      if (c_gl_wr < c_gl_wr_end) {
        // Read 8 BF16 elements each from gate and up
        int4 gate_chunk = sh_gate[c_sh_rd];
        int4 up_chunk = sh_red[c_sh_rd];
        scalar_t* g_ptr = reinterpret_cast<scalar_t*>(&gate_chunk);
        scalar_t* u_ptr = reinterpret_cast<scalar_t*>(&up_chunk);
        int4 result;
        scalar_t* r_ptr = reinterpret_cast<scalar_t*>(&result);
#pragma unroll
        for (int k = 0; k < 8; k++) {
          float g = num2float(g_ptr[k]);
          float u = num2float(u_ptr[k]);
          r_ptr[k] = float2num(act_gate_mul<ACT>(g, u));
        }
        C[c_gl_wr] = result;
        c_gl_wr += c_gl_wr_delta;
        c_sh_rd += c_sh_rd_delta;
      }
    }
  }
}

// ============================================================================
// M16 kernel (for S2 and standalone use)
// Templated on weight codec; <U4B8> is the production K2.5 instantiation.
// ============================================================================

template <WCodec CODEC>
__global__ void MarlinGrouped_M16(
    const int4* __restrict__ A,
    const int4* const* __restrict__ B_ptrs,
    int4* const* __restrict__ C_ptrs,
    const int4* const* __restrict__ scales_ptrs,
    const int* __restrict__ expert_starts,
    const int* __restrict__ expert_counts,
    int num_experts,
    int prob_n, int prob_k, int lda,
    int n_tiles_per_expert,
    int max_m_tiles)
{
  // CTA M-tiling dispatch:
  // blockIdx.x = matrix_idx * (max_m_tiles * n_tiles) + m_tile_idx * n_tiles + tile_idx
  int linear_idx = blockIdx.x;
  int matrix_idx = linear_idx / (max_m_tiles * n_tiles_per_expert);
  int remainder = linear_idx % (max_m_tiles * n_tiles_per_expert);
  int m_tile_idx = remainder / n_tiles_per_expert;
  int tile_idx = remainder % n_tiles_per_expert;
  int expert_idx = matrix_idx % num_experts;

  int expert_m = expert_counts[expert_idx];
  int m_start = m_tile_idx * M16_MBLOCK;
  if (m_start >= expert_m) return;  // GPU-side dispatch: skip unused M-tiles

  int prob_m = min(M16_MBLOCK, expert_m - m_start);
  int token_start = expert_starts[expert_idx] + m_start;

  const int4* A_ptr = A + token_start * (lda / 8);
  const int4* B = B_ptrs[matrix_idx];
  int c_gl_stride_m16 = prob_n / 8;
  int4* C = C_ptrs[matrix_idx] + m_start * c_gl_stride_m16;  // offset by m_start rows
  const int4* scales_ptr = scales_ptrs[matrix_idx];

  int k_tiles = prob_k / 16 / TK;

  // A indices (M16: standard pattern)
  int a_gl_stride = lda / 8;
  int a_gl_rd_delta_i = a_gl_stride * (THREADS / a_gl_rd_delta_o);
  int a_gl_rd = a_gl_stride * (threadIdx.x / a_gl_rd_delta_o) + (threadIdx.x % a_gl_rd_delta_o);

  int a_sh_wr = a_sh_stride * (threadIdx.x / a_gl_rd_delta_o) + (threadIdx.x % a_gl_rd_delta_o);
  // M16: a_sh_rd uses %16 pattern (not %8 like M8)
  int a_sh_rd = a_sh_stride * ((threadIdx.x % 32) % 16) + (threadIdx.x % 32) / 16;
  a_sh_rd += 2 * ((threadIdx.x / 32) / (TN / 4));

  // B indices (identical to M8)
  int b_gl_stride = 16 * prob_n / (PACK * 4);
  int b_gl_rd_delta_o = b_gl_stride * TK;
  int b_gl_rd_delta_i = b_gl_stride * (THREADS / b_sh_stride_threads);
  int b_gl_rd = b_gl_stride * (threadIdx.x / b_sh_stride_threads) + (threadIdx.x % b_sh_stride_threads);
  b_gl_rd += b_sh_stride * tile_idx;

  // Scale indices (identical to M8)
  int s_gl_stride = prob_n / 8;
  int s_gl_rd = s_sh_stride * tile_idx + threadIdx.x;
  bool s_sh_wr_pred = threadIdx.x < s_sh_stride;
  int s_sh_rd = 8 * ((threadIdx.x / 32) % (TN / 4)) + (threadIdx.x % 32) / 4;

  // A predicates
  bool a_sh_wr_pred[m16_a_sh_wr_iters];
#pragma unroll
  for (int i = 0; i < m16_a_sh_wr_iters; i++)
    a_sh_wr_pred[i] = a_sh_wr_delta * i + a_sh_wr < a_sh_stride * prob_m;

  // XOR transform for A shared memory
  auto transform_a = [&](int i) {
    int row = i / a_gl_rd_delta_o;
    return a_gl_rd_delta_o * row + (i % a_gl_rd_delta_o) ^ (row % 8);
  };

  int a_sh_wr_trans[m16_a_sh_wr_iters];
#pragma unroll
  for (int i = 0; i < m16_a_sh_wr_iters; i++)
    a_sh_wr_trans[i] = transform_a(a_sh_wr_delta * i + a_sh_wr);

  int a_sh_rd_trans[b_sh_wr_iters][TM];
#pragma unroll
  for (int i = 0; i < b_sh_wr_iters; i++)
    for (int j = 0; j < TM; j++)
      a_sh_rd_trans[i][j] = transform_a(a_sh_rd_delta_o * i + a_sh_rd_delta_i * j + a_sh_rd);

  // B pointers
  const int4* B_ptr[b_sh_wr_iters];
#pragma unroll
  for (int i = 0; i < b_sh_wr_iters; i++)
    B_ptr[i] = B + b_gl_rd_delta_i * i + b_gl_rd;

  // Shared memory (M16: larger a_sh area)
  extern __shared__ int4 sh[];
  int4* sh_b = sh;
  int4* sh_red = sh;
  int4* sh_s = sh + (sh_red_size > sh_b_size ? sh_red_size : sh_b_size);
  int4* sh_a = sh_s + sh_s_size;

  // Registers
  FragA frag_a[2][TM];
  I4 frag_b_quant[2][1];
  FragC frag_c[TM][4][2];
  FragS frag_s[2][4];

  // Zero accumulators
#pragma unroll
  for (int i = 0; i < TM * 4 * 2 * 4; i++)
    reinterpret_cast<float*>(frag_c)[i] = 0;

  // ---- Pipeline: fetch A, B, scales to SMEM ----
  auto fetch_to_shared = [&](int pipe, int k_off, bool pred) {
    if (pred) {
      int4* sh_a_stage = sh_a + m16_a_sh_stage * pipe;  // M16 stage size
#pragma unroll
      for (int i = 0; i < m16_a_sh_wr_iters; i++)
        cp_async4_pred(&sh_a_stage[a_sh_wr_trans[i]],
                       &A_ptr[a_gl_rd_delta_i * i + a_gl_rd + a_gl_rd_delta_o * k_off],
                       a_sh_wr_pred[i]);

      int4* sh_b_stage = sh_b + b_sh_stage * pipe;
#pragma unroll
      for (int i = 0; i < b_sh_wr_iters; i++) {
        cp_async4(&sh_b_stage[b_sh_wr_delta * i + threadIdx.x], B_ptr[i]);
        B_ptr[i] += b_gl_rd_delta_o;
      }

      // Scales (GROUP_BLOCKS=2: load s_tb_groups=4 scale rows per stage)
      int4* sh_s_stage = sh_s + s_sh_stage * pipe;
#pragma unroll
      for (int i = 0; i < s_tb_groups; i++) {
        if (s_sh_wr_pred) {
          fetch_scale_vec<CODEC>(
              sh_s_stage, i * s_sh_stride + threadIdx.x,
              scales_ptr, s_gl_rd);
        }
        s_gl_rd += s_gl_stride;
      }
    }
    cp_async_fence();
  };

  // ---- Startup: fill pipeline ----
#pragma unroll
  for (int i = 0; i < STAGES - 1; i++)
    fetch_to_shared(i, i, i < k_tiles);

  cp_async_wait<STAGES - 2>();
  __syncthreads();

  // Load first registers (M16: uses ldsm4 instead of ldsm2)
  auto load_regs = [&](int k, int pipe) {
    int4* sh_a_stage = sh_a + m16_a_sh_stage * pipe;  // M16 stage size
#pragma unroll
    for (int i = 0; i < TM; i++)
      ldsm4(frag_a[k % 2][i], &sh_a_stage[a_sh_rd_trans[k % b_sh_wr_iters][i]]);
    int4* sh_b_stage = sh_b + b_sh_stage * pipe;
    frag_b_quant[k % 2][0] = *reinterpret_cast<I4*>(
        &sh_b_stage[b_sh_wr_delta * (k % b_sh_wr_iters) + threadIdx.x]);
  };

  auto load_scales = [&](int k, int pipe) {
    // GROUP_BLOCKS=2: compute which scale group this k-iteration belongs to
    int cur_k = k_iter_size * (k % b_sh_wr_iters);
    int k_blocks = cur_k / 16;
    int cur_group_id = k_blocks / GROUP_BLOCKS;

    int4* sh_s_stage = sh_s + s_sh_stage * (pipe % STAGES);
    reinterpret_cast<int4*>(&frag_s[k % 2])[0] = load_scale_vec<CODEC>(
        sh_s_stage, s_sh_rd + cur_group_id * s_sh_stride);
  };

  // M16: standard mma_op with two calls per j (separate b0, b1)
  auto matmul = [&](int k) {
    int k2 = k % 2;
#pragma unroll
    for (int j = 0; j < 4; j++) {
      FragB frag_b0, frag_b1;
      int b_quant_0 = frag_b_quant[k2][0][j];
      int b_quant_1 = b_quant_0 >> 8;
      dequant_w4<CODEC>(b_quant_0, reinterpret_cast<scalar_t2*>(&frag_b0));
      dequant_w4<CODEC>(b_quant_1, reinterpret_cast<scalar_t2*>(&frag_b1));
      scale_op(frag_b0, frag_s[k2][j], 0);
      scale_op(frag_b1, frag_s[k2][j], 1);
#pragma unroll
      for (int i = 0; i < TM; i++) {
        mma_op(frag_a[k2][i], frag_b0, frag_c[i][j][0]);
        mma_op(frag_a[k2][i], frag_b1, frag_c[i][j][1]);
      }
    }
  };

  // Load first register tile
  load_regs(0, 0);
  load_scales(0, 0);
  a_gl_rd += a_gl_rd_delta_o * (STAGES - 1);

  // ---- Main loop: process all K-tiles ----
  int slice_iters = k_tiles;
  while (slice_iters) {
#pragma unroll
    for (int pipe = 0; pipe < STAGES;) {
#pragma unroll
      for (int k = 0; k < b_sh_wr_iters; k++) {
        load_regs(k + 1, pipe % STAGES);
        load_scales(k + 1, pipe);
        if (k == b_sh_wr_iters - 2) {
          fetch_to_shared((pipe + STAGES - 1) % STAGES, pipe, slice_iters >= STAGES);
          pipe++;
          cp_async_wait<STAGES - 2>();
          __syncthreads();
        }
        matmul(k);
      }
      slice_iters--;
      if (slice_iters == 0) break;
    }
    a_gl_rd += a_gl_rd_delta_o * STAGES;
  }
  cp_async_wait<0>();

  // ---- Thread-block reduce (M16: step by 1, not 2) ----
  {
    constexpr int red_off = THREADS / b_sh_stride_threads / 2;
    if constexpr (red_off >= 1) {
      auto red_idx = threadIdx.x / b_sh_stride_threads;
      constexpr int red_sh_stride = b_sh_stride_threads * 4 * 2;
      constexpr int red_sh_delta = b_sh_stride_threads;
      int red_sh_rd = red_sh_stride * (threadIdx.x / b_sh_stride_threads) +
                      (threadIdx.x % b_sh_stride_threads);
#pragma unroll
      for (int m = 0; m < TM; m++) {
#pragma unroll
        for (int i = red_off; i > 0; i /= 2) {
          if (i <= red_idx && red_idx < 2 * i) {
#pragma unroll
            for (int j = 0; j < 4 * 2; j++) {  // M16: step by 1 (both [0] and [1])
              int red_sh_wr = red_sh_delta * j + (red_sh_rd - red_sh_stride * i);
              if (i < red_off) {
                float* c_rd = reinterpret_cast<float*>(&sh_red[red_sh_delta * j + red_sh_rd]);
                float* c_wr = reinterpret_cast<float*>(&sh_red[red_sh_wr]);
#pragma unroll
                for (int kk = 0; kk < 4; kk++)
                  reinterpret_cast<FragC*>(frag_c)[4 * 2 * m + j][kk] += c_rd[kk] + c_wr[kk];
              }
              sh_red[red_sh_wr] = reinterpret_cast<int4*>(&frag_c)[4 * 2 * m + j];
            }
          }
          __syncthreads();
        }
        if (red_idx == 0) {
#pragma unroll
          for (int i = 0; i < 4 * 2; i++) {  // M16: step by 1
            float* c_rd = reinterpret_cast<float*>(&sh_red[red_sh_delta * i + red_sh_rd]);
#pragma unroll
            for (int j = 0; j < 4; j++)
              reinterpret_cast<FragC*>(frag_c)[4 * 2 * m + i][j] += c_rd[j];
          }
        }
        __syncthreads();
      }
    }
  }

  // ---- Write result (M16: standard 16-row layout) ----
  {
    int c_gl_stride = prob_n / 8;
    constexpr int c_sh_stride = 2 * TN + 1;
    int c_gl_wr_delta = c_gl_stride * (THREADS / (2 * TN));
    constexpr int c_sh_rd_delta = c_sh_stride * (THREADS / (2 * TN));

    int c_gl_wr = c_gl_stride * (threadIdx.x / (2 * TN)) + (threadIdx.x % (2 * TN));
    c_gl_wr += (2 * TN) * tile_idx;
    // M16: standard SMEM write pattern (not m_block_size_8)
    int c_sh_wr = (4 * c_sh_stride) * ((threadIdx.x % 32) / 4) + (threadIdx.x % 32) % 4;
    c_sh_wr += 32 * (threadIdx.x / 32);
    int c_sh_rd = c_sh_stride * (threadIdx.x / (2 * TN)) + (threadIdx.x % (2 * TN));
    int c_gl_wr_end = c_gl_stride * prob_m;

    if (threadIdx.x / 32 < TN / 4) {
#pragma unroll
      for (int i = 0; i < TM; i++) {
#pragma unroll
        for (int j = 0; j < 4; j++) {
          int wr = c_sh_wr + 8 * j;
          scalar_t2 r0 = nums2num2(float2num(frag_c[i][j][0][0]), float2num(frag_c[i][j][0][1]));
          scalar_t2 r1 = nums2num2(float2num(frag_c[i][j][0][2]), float2num(frag_c[i][j][0][3]));
          scalar_t2 r2 = nums2num2(float2num(frag_c[i][j][1][0]), float2num(frag_c[i][j][1][1]));
          scalar_t2 r3 = nums2num2(float2num(frag_c[i][j][1][2]), float2num(frag_c[i][j][1][3]));
          ((scalar_t2*)sh_red)[wr + (4 * c_sh_stride) * 0 + 0] = r0;
          ((scalar_t2*)sh_red)[wr + (4 * c_sh_stride) * 8 + 0] = r1;
          ((scalar_t2*)sh_red)[wr + (4 * c_sh_stride) * 0 + 4] = r2;
          ((scalar_t2*)sh_red)[wr + (4 * c_sh_stride) * 8 + 4] = r3;
        }
        c_sh_wr += 16 * (4 * c_sh_stride);
      }
    }
    __syncthreads();

    // M16: 16 rows to write (not 8)
#pragma unroll
    for (int i = 0; i < div_ceil(16 * TM, THREADS / (2 * TN)); i++) {
      if (c_gl_wr < c_gl_wr_end) {
        C[c_gl_wr] = sh_red[c_sh_rd];
        c_gl_wr += c_gl_wr_delta;
        c_sh_rd += c_sh_rd_delta;
      }
    }
  }
}

// ============================================================================
// SiLU kernel
// ============================================================================

__global__ void silu_mul_kernel(
    const scalar_t* __restrict__ gate, const scalar_t* __restrict__ up,
    scalar_t* __restrict__ out, int n)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float g = num2float(gate[idx]);
    float u = num2float(up[idx]);
    out[idx] = float2num(g / (1.0f + __expf(-g)) * u);
  }
}

// Stride-aware SiLU: reads from compact gate/up (compact_stride per expert),
// writes to mtp-strided output (output_stride per expert).
__global__ void silu_mul_scatter_kernel(
    const scalar_t* __restrict__ gate, const scalar_t* __restrict__ up,
    scalar_t* __restrict__ out,
    const int* __restrict__ expert_counts,
    int num_experts, int compact_stride, int output_stride, int N)
{
  // Each thread handles one element: (expert, row, col)
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total_compact = num_experts * compact_stride * N;
  if (tid >= total_compact) return;

  int col = tid % N;
  int row_global = tid / N;
  int expert = row_global / compact_stride;
  int row_local = row_global % compact_stride;

  if (expert >= num_experts) return;
  if (row_local >= expert_counts[expert]) return;

  int src_idx = expert * compact_stride * N + row_local * N + col;
  int dst_idx = expert * output_stride * N + row_local * N + col;

  float g = num2float(gate[src_idx]);
  float u = num2float(up[src_idx]);
  out[dst_idx] = float2num(g / (1.0f + __expf(-g)) * u);
}

// Dual-stride SiLU: gate from mtp-stride buffer (in-place), up from compact-stride buffer.
// Used when gate output writes directly to intermediate (mtp stride) and
// up output writes to a separate compact buffer.
__global__ void silu_mul_dual_stride_kernel(
    scalar_t* __restrict__ gate_inplace,    // intermediate[E*gate_stride, N] (read gate, write result)
    const scalar_t* __restrict__ up,        // up_buf[E*up_stride, N]
    const int* __restrict__ expert_counts,
    int num_experts, int gate_stride, int up_stride, int N)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total = num_experts * up_stride * N;
  if (tid >= total) return;

  int col = tid % N;
  int row_global = tid / N;
  int expert = row_global / up_stride;
  int row_local = row_global % up_stride;

  if (expert >= num_experts) return;
  if (row_local >= expert_counts[expert]) return;

  int gate_idx = expert * gate_stride * N + row_local * N + col;
  int up_idx = tid;  // linear in up_buf layout

  float g = num2float(gate_inplace[gate_idx]);
  float u = num2float(up[up_idx]);
  gate_inplace[gate_idx] = float2num(g / (1.0f + __expf(-g)) * u);
}

// ============================================================================
// Launchers
// ============================================================================

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

void grouped_marlin_gemm(
    torch::Tensor A, torch::Tensor B_ptrs, torch::Tensor C_ptrs,
    torch::Tensor scales_ptrs,
    torch::Tensor expert_starts, torch::Tensor expert_counts,
    int num_experts, int prob_n, int prob_k,
    torch::Tensor workspace, int num_matrices, int n_tiles)
{
    auto stream = at::cuda::getCurrentCUDAStream();

    // SMEM: max(sh_red=528, sh_b=4096) + sh_s=512 + 4*a_sh_stage=512 = 5120 int4 = 81920 bytes
    constexpr int smem_bytes = ((5120 * 16) + 1023) / 1024 * 1024;  // 82944 bytes
    cudaFuncSetAttribute((void*)MarlinSimple, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);

    int total_ctas = n_tiles * num_matrices;
    MarlinSimple<<<total_ctas, 256, smem_bytes, stream>>>(
        reinterpret_cast<const int4*>(A.data_ptr()),
        reinterpret_cast<const int4* const*>(B_ptrs.data_ptr()),
        reinterpret_cast<int4* const*>(C_ptrs.data_ptr()),
        reinterpret_cast<const int4* const*>(scales_ptrs.data_ptr()),
        expert_starts.data_ptr<int>(),
        expert_counts.data_ptr<int>(),
        num_experts, prob_n, prob_k, prob_k,
        n_tiles);
}

void silu_mul(torch::Tensor gate, torch::Tensor up, torch::Tensor out)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    int n = gate.numel();
    silu_mul_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
        reinterpret_cast<const scalar_t*>(gate.data_ptr()),
        reinterpret_cast<const scalar_t*>(up.data_ptr()),
        reinterpret_cast<scalar_t*>(out.data_ptr()), n);
}

void silu_mul_scatter(
    torch::Tensor gate, torch::Tensor up, torch::Tensor out,
    torch::Tensor expert_counts,
    int num_experts, int compact_stride, int output_stride, int N)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    int total = num_experts * compact_stride * N;
    silu_mul_scatter_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
        reinterpret_cast<const scalar_t*>(gate.data_ptr()),
        reinterpret_cast<const scalar_t*>(up.data_ptr()),
        reinterpret_cast<scalar_t*>(out.data_ptr()),
        expert_counts.data_ptr<int>(),
        num_experts, compact_stride, output_stride, N);
}

template <WCodec CODEC, Act ACT>
static void launch_m16_s1(
    torch::Tensor& A,
    torch::Tensor& gate_B_ptrs, torch::Tensor& up_B_ptrs,
    torch::Tensor& C_ptrs,
    torch::Tensor& gate_scales_ptrs, torch::Tensor& up_scales_ptrs,
    torch::Tensor& expert_starts, torch::Tensor& expert_counts,
    int num_experts, int prob_n, int prob_k,
    int n_tiles, int max_m_tiles)
{
    auto stream = at::cuda::getCurrentCUDAStream();

    // SMEM: M16 base (90112) + gate result (528 * 16 = 8448) = 98560 bytes
    constexpr int smem_bytes = 98560;
    cudaFuncSetAttribute((void*)MarlinGrouped_M16_S1<CODEC, ACT>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);

    int total_ctas = n_tiles * max_m_tiles * num_experts;
    MarlinGrouped_M16_S1<CODEC, ACT><<<total_ctas, 256, smem_bytes, stream>>>(
        reinterpret_cast<const int4*>(A.data_ptr()),
        reinterpret_cast<const int4* const*>(gate_B_ptrs.data_ptr()),
        reinterpret_cast<const int4* const*>(up_B_ptrs.data_ptr()),
        reinterpret_cast<int4* const*>(C_ptrs.data_ptr()),
        reinterpret_cast<const int4* const*>(gate_scales_ptrs.data_ptr()),
        reinterpret_cast<const int4* const*>(up_scales_ptrs.data_ptr()),
        expert_starts.data_ptr<int>(),
        expert_counts.data_ptr<int>(),
        num_experts, prob_n, prob_k, prob_k,
        n_tiles, max_m_tiles);
}

void grouped_marlin_gemm_m16_s1(
    torch::Tensor A,
    torch::Tensor gate_B_ptrs, torch::Tensor up_B_ptrs,
    torch::Tensor C_ptrs,
    torch::Tensor gate_scales_ptrs, torch::Tensor up_scales_ptrs,
    torch::Tensor expert_starts, torch::Tensor expert_counts,
    int num_experts, int prob_n, int prob_k,
    torch::Tensor workspace, int n_tiles, int max_m_tiles)
{
    launch_m16_s1<WCodec::U4B8, Act::SILU>(
        A, gate_B_ptrs, up_B_ptrs, C_ptrs, gate_scales_ptrs, up_scales_ptrs,
        expert_starts, expert_counts, num_experts, prob_n, prob_k,
        n_tiles, max_m_tiles);
}

// ----------------------------------------------------------------------------
// K3 MXFP4 entry-point hard-fail checks. The established integration pattern
// (K2.5: kimi_k25/model.py calls the pybind symbols directly) bypasses the
// python wrappers entirely, so the C++ entries must hard-fail on malformed
// metadata themselves: distinct symbols stop codec mixups, these TORCH_CHECKs
// stop everything else that is visible host-side. INT4 entries are left
// byte-identical to production on purpose.
// ----------------------------------------------------------------------------
static void check_mxfp4_ptr_array(const torch::Tensor& t, const char* name,
                                  int64_t len) {
    TORCH_CHECK(t.scalar_type() == at::kLong && t.is_cuda() && t.numel() == len,
        "mxfp4 launch: ", name, " must be int64 CUDA [", len, "], got ",
        t.scalar_type(), " numel=", t.numel());
}

static void check_mxfp4_counts(const torch::Tensor& t, const char* name,
                               int64_t len) {
    TORCH_CHECK(t.scalar_type() == at::kInt && t.is_cuda() && t.numel() == len,
        "mxfp4 launch: ", name, " must be int32 CUDA [", len, "], got ",
        t.scalar_type(), " numel=", t.numel());
}

static void check_mxfp4_common(const torch::Tensor& A,
                               const torch::Tensor& expert_starts,
                               const torch::Tensor& expert_counts,
                               int num_experts, int prob_n, int prob_k,
                               int n_tiles, int max_m_tiles) {
    TORCH_CHECK(A.scalar_type() == at::kBFloat16 && A.is_cuda()
                && A.is_contiguous() && A.size(-1) == prob_k,
        "mxfp4 launch: A must be contiguous bf16 CUDA [*, ", prob_k,
        "], got ", A.scalar_type(), " last dim ", A.size(-1));
    TORCH_CHECK(prob_n % 256 == 0 && prob_k % 128 == 0,
        "mxfp4 launch: prob_n%256==0 and prob_k%128==0 required, got prob_n=",
        prob_n, " prob_k=", prob_k);
    TORCH_CHECK(n_tiles == prob_n / 256,
        "mxfp4 launch: n_tiles=", n_tiles, " != prob_n/256=", prob_n / 256);
    TORCH_CHECK(num_experts >= 1 && max_m_tiles >= 1,
        "mxfp4 launch: num_experts=", num_experts, " max_m_tiles=",
        max_m_tiles, " must both be >= 1");
    check_mxfp4_counts(expert_starts, "expert_starts", num_experts);
    check_mxfp4_counts(expert_counts, "expert_counts", num_experts);
}

// K3 MXFP4 fused S1: E2M1 weight decode + SiTU epilogue. Separate entry point
// on purpose — pointer arrays are opaque to the kernel, so distinct pybind
// symbols are the enforceable seam preventing INT4 entries from silently
// consuming E2M1 codes ((q-8)*s on E2M1 codes is finite plausible garbage).
void grouped_marlin_gemm_m16_s1_mxfp4_situ(
    torch::Tensor A,
    torch::Tensor gate_B_ptrs, torch::Tensor up_B_ptrs,
    torch::Tensor C_ptrs,
    torch::Tensor gate_scales_ptrs, torch::Tensor up_scales_ptrs,
    torch::Tensor expert_starts, torch::Tensor expert_counts,
    int num_experts, int prob_n, int prob_k,
    torch::Tensor workspace, int n_tiles, int max_m_tiles)
{
    check_mxfp4_common(A, expert_starts, expert_counts,
                       num_experts, prob_n, prob_k, n_tiles, max_m_tiles);
    check_mxfp4_ptr_array(gate_B_ptrs, "gate_B_ptrs", num_experts);
    check_mxfp4_ptr_array(up_B_ptrs, "up_B_ptrs", num_experts);
    check_mxfp4_ptr_array(C_ptrs, "C_ptrs", num_experts);
    check_mxfp4_ptr_array(gate_scales_ptrs, "gate_scales_ptrs", num_experts);
    check_mxfp4_ptr_array(up_scales_ptrs, "up_scales_ptrs", num_experts);
    launch_m16_s1<WCodec::E2M1, Act::SITU>(
        A, gate_B_ptrs, up_B_ptrs, C_ptrs, gate_scales_ptrs, up_scales_ptrs,
        expert_starts, expert_counts, num_experts, prob_n, prob_k,
        n_tiles, max_m_tiles);
}

template <WCodec CODEC>
static void launch_m16(
    torch::Tensor& A, torch::Tensor& B_ptrs, torch::Tensor& C_ptrs,
    torch::Tensor& scales_ptrs,
    torch::Tensor& expert_starts, torch::Tensor& expert_counts,
    int num_experts, int prob_n, int prob_k,
    int num_matrices, int n_tiles, int max_m_tiles)
{
    auto stream = at::cuda::getCurrentCUDAStream();

    // SMEM: max(528, 4096) + 512 + 1024 = 5632 int4 = 90112 bytes
    constexpr int smem_bytes = 90112;
    cudaFuncSetAttribute((void*)MarlinGrouped_M16<CODEC>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);

    int total_ctas = n_tiles * max_m_tiles * num_matrices;
    MarlinGrouped_M16<CODEC><<<total_ctas, 256, smem_bytes, stream>>>(
        reinterpret_cast<const int4*>(A.data_ptr()),
        reinterpret_cast<const int4* const*>(B_ptrs.data_ptr()),
        reinterpret_cast<int4* const*>(C_ptrs.data_ptr()),
        reinterpret_cast<const int4* const*>(scales_ptrs.data_ptr()),
        expert_starts.data_ptr<int>(),
        expert_counts.data_ptr<int>(),
        num_experts, prob_n, prob_k, prob_k,
        n_tiles, max_m_tiles);
}

void grouped_marlin_gemm_m16(
    torch::Tensor A, torch::Tensor B_ptrs, torch::Tensor C_ptrs,
    torch::Tensor scales_ptrs,
    torch::Tensor expert_starts, torch::Tensor expert_counts,
    int num_experts, int prob_n, int prob_k,
    torch::Tensor workspace, int num_matrices, int n_tiles,
    int max_m_tiles)
{
    launch_m16<WCodec::U4B8>(
        A, B_ptrs, C_ptrs, scales_ptrs, expert_starts, expert_counts,
        num_experts, prob_n, prob_k, num_matrices, n_tiles, max_m_tiles);
}

// K3 MXFP4 M16 (S3 down projection / standalone): E2M1 weight decode.
void grouped_marlin_gemm_m16_mxfp4(
    torch::Tensor A, torch::Tensor B_ptrs, torch::Tensor C_ptrs,
    torch::Tensor scales_ptrs,
    torch::Tensor expert_starts, torch::Tensor expert_counts,
    int num_experts, int prob_n, int prob_k,
    torch::Tensor workspace, int num_matrices, int n_tiles,
    int max_m_tiles)
{
    check_mxfp4_common(A, expert_starts, expert_counts,
                       num_experts, prob_n, prob_k, n_tiles, max_m_tiles);
    TORCH_CHECK(num_matrices >= num_experts,
        "mxfp4 launch: num_matrices=", num_matrices, " < num_experts=",
        num_experts);
    check_mxfp4_ptr_array(B_ptrs, "B_ptrs", num_matrices);
    check_mxfp4_ptr_array(C_ptrs, "C_ptrs", num_matrices);
    check_mxfp4_ptr_array(scales_ptrs, "scales_ptrs", num_matrices);
    launch_m16<WCodec::E2M1>(
        A, B_ptrs, C_ptrs, scales_ptrs, expert_starts, expert_counts,
        num_experts, prob_n, prob_k, num_matrices, n_tiles, max_m_tiles);
}

void silu_mul_dual_stride(
    torch::Tensor gate_inplace, torch::Tensor up,
    torch::Tensor expert_counts,
    int num_experts, int gate_stride, int up_stride, int N)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    int total = num_experts * up_stride * N;
    silu_mul_dual_stride_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
        reinterpret_cast<scalar_t*>(gate_inplace.data_ptr()),
        reinterpret_cast<const scalar_t*>(up.data_ptr()),
        expert_counts.data_ptr<int>(),
        num_experts, gate_stride, up_stride, N);
}

// ============================================================================
// PyBind11 module
// ============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("grouped_marlin_gemm", &grouped_marlin_gemm, "Marlin M8 grouped GEMM");
    m.def("grouped_marlin_gemm_m16", &grouped_marlin_gemm_m16, "Marlin M16 grouped GEMM with CTA M-tiling");
    m.def("grouped_marlin_gemm_m16_s1", &grouped_marlin_gemm_m16_s1, "Marlin M16 fused S1 (gate+up+SiLU)");
    m.def("grouped_marlin_gemm_m16_mxfp4", &grouped_marlin_gemm_m16_mxfp4,
          "Marlin M16 grouped GEMM, MXFP4 (E2M1) weights — Kimi-K3");
    m.def("grouped_marlin_gemm_m16_s1_mxfp4_situ", &grouped_marlin_gemm_m16_s1_mxfp4_situ,
          "Marlin M16 fused S1, MXFP4 (E2M1) weights + SiTU epilogue — Kimi-K3");
    m.def("silu_mul", &silu_mul, "Element-wise SiLU(gate) * up");
    m.def("silu_mul_scatter", &silu_mul_scatter, "SiLU with expert_counts scatter");
    m.def("silu_mul_dual_stride", &silu_mul_dual_stride, "SiLU with dual-stride layout");
}
