/*
 * Version: v12c — Grouped Marlin with m_block_size_8 for decode
 * Hypothesis: m_block_size_8 halves A SMEM + uses mma_trans → 2 blocks/SM → 2× occupancy
 * Result: PENDING
 *
 * Uses mma_trans for M<=8 (swaps A/B operands, ldsm<2> for A).
 * Assumes: M <= 8 per expert (decode case).
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
static constexpr int GROUP_BLOCKS = 8;  // group_size=128, block=16
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
static constexpr int s_sh_stage = s_sh_stride;    // 32 (group_blocks >= TK)

static constexpr int sh_red_size = (2 * TN + 1) * 16 * TM;  // 528
static constexpr int sh_b_size = STAGES * b_sh_stage;        // 4096
static constexpr int sh_s_size = STAGES * s_sh_stage;        // 128


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

      // Scales (group_blocks >= TK: load every stage)
      int4* sh_s_stage = sh_s + s_sh_stage * pipe;
      if (s_sh_wr_pred)
        cp_async4(&sh_s_stage[threadIdx.x], &scales_ptr[s_gl_rd]);
      s_gl_rd += s_gl_stride;
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

  auto load_scales = [&](int k, int full_pipe) {
    int pipe = full_pipe % STAGES;
    if (k % b_sh_wr_iters == 0) {
      int4* sh_s_stage = sh_s + s_sh_stage * pipe;
      reinterpret_cast<int4*>(&frag_s[k % 2])[0] = sh_s_stage[s_sh_rd];
    } else {
      reinterpret_cast<int4*>(&frag_s[1])[0] = reinterpret_cast<int4*>(&frag_s[0])[0];
    }
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

    constexpr int smem_bytes = ((4736 * 16) + 1023) / 1024 * 1024;  // 76800 bytes
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
