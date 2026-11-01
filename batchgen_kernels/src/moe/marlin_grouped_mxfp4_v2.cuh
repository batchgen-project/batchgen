/*
 * Grouped Marlin MXFP4 (E2M1 weights, E8M0 group-32 scales) — version 2.
 *
 * Included by marlin_grouped_gemm.cu after its shared helpers (Vec/I4,
 * FragA/B/C/S, cp_async*, lop3, ldsm4, mma_op, scale_op, act_gate_mul,
 * div_ceil, float2num/num2float); not a standalone header. Everything lives in
 * namespace marlin_v2 so the v1 kernels and their global constants are
 * untouched. Only the E2M1 codec and the SiTU epilogue are implemented.
 *
 * Differences from MarlinGrouped_M16{,_S1}<WCodec::E2M1>:
 *   - TM in {1, 2} 16-row m-blocks per CTA tile: each dequantised and scaled
 *     B fragment feeds TM MMAs, so an expert with up to 16*TM rows streams
 *     and dequantises its weights once.
 *   - TN in {16, 8} 16-column n-blocks per CTA. TN=16 keeps the v1 warp
 *     layout (4 warps along N x 2 K-slices); TN=8 uses 2 warps along N x 4
 *     K-slices with a 4-way shared-memory K reduction.
 *   - grid = num_matrices x (prob_n / (16*TN)); each CTA walks its expert's
 *     rows in super-tiles of 16*TM, so the grid does not depend on the
 *     worst-case per-expert row count.
 *   - The E2M1 2^126 rebias is folded into the E8M0 scale expansion. MMA
 *     inputs are bit-identical to v1's: v1 computes (raw * 2^126) * 2^(e-127)
 *     with two exact power-of-two multiplies, v2 computes raw * 2^(e-1) with
 *     one, and both round the same real value once.
 *
 * Preconditions beyond v1 (callers validate once at weight-build time with
 * batchgen.moe.marlin_grouped_moe.check_mxfp4_scales_v2_compatible):
 *   - every E8M0 scale byte e satisfies 1 <= e <= 128. The folded bf16 scale
 *     is (e + 126) << 7, which reaches the Inf/NaN exponent for e >= 129.
 *     (e == 0 is already rejected by the repack.)
 */
#pragma once

namespace marlin_v2 {

constexpr int THREADS = 256;
constexpr int TK = 8;                 // 16-K blocks per pipeline stage (128 K)
constexpr int GROUP_BLOCKS = 2;       // MXFP4 group = 32 K = 2 blocks
constexpr int GROUPS_PER_STAGE = TK / GROUP_BLOCKS;
constexpr int SMEM_BUDGET_BYTES = 112 * 1024;  // two CTAs per SM
constexpr int MAX_STAGES = 8;

__host__ __device__ constexpr int cmax(int a, int b) { return a > b ? a : b; }

// Tile geometry. Marlin's weight and scale permutations operate on 16x64
// tiles, so each warp owns 64 output columns (4 x n16 MMA column pairs) and
// one 16-K slice per k-iteration; WARPS_N * KSLICES == 8 warps.
template <int TM, int TN>
struct Layout {
  static_assert(TM == 1 || TM == 2, "TM must be 1 or 2");
  static_assert(TN == 16 || TN == 8, "TN must be 16 or 8");

  static constexpr int WARPS = THREADS / 32;
  static constexpr int WARPS_N = TN / 4;
  static constexpr int KSLICES = WARPS / WARPS_N;   // 2 (TN=16) or 4 (TN=8)
  static constexpr int K_ITERS = TK / KSLICES;      // k-iterations per stage

  // A: 16*TM rows x 128 K per stage, 16 int4 per row, one int4 per thread
  // per 16 rows.
  static constexpr int a_sh_stride = 16 * TK / 8;
  static constexpr int a_gl_rd_delta_o = a_sh_stride;
  static constexpr int a_sh_wr_delta = a_sh_stride * (THREADS / a_gl_rd_delta_o);
  static constexpr int a_sh_rd_delta_o = 2 * KSLICES;     // int4 per k-iteration
  static constexpr int a_sh_rd_delta_i = a_sh_stride * 16;  // one m-block
  static constexpr int a_sh_stage = a_sh_stride * 16 * TM;
  static constexpr int a_sh_wr_iters = a_sh_stage / a_sh_wr_delta;

  // B: per 16-K block the tile holds 16*TN columns = 8*TN int4.
  static constexpr int b_sh_stride = 8 * TN;
  static constexpr int b_sh_stride_threads = b_sh_stride;
  static constexpr int b_sh_wr_delta = THREADS;
  static constexpr int b_sh_stage = b_sh_stride * TK;
  static constexpr int b_sh_wr_iters = b_sh_stage / b_sh_wr_delta;

  // Scales stay raw E8M0 bytes in shared memory: one group row of the tile is
  // 16*TN bytes = 2*TN 8-byte vectors; a stage holds GROUPS_PER_STAGE rows.
  static constexpr int s_sh_stride = 2 * TN;                   // 8-byte vectors
  static constexpr int s_sh_vecs = GROUPS_PER_STAGE * s_sh_stride;
  static constexpr int s_sh_stage = s_sh_vecs * 8 / 16;        // int4

  // Output staging: 16*TM rows of 2*TN int4 plus one pad int4 per row.
  static constexpr int c_sh_stride = 2 * TN + 1;
  static constexpr int c_sh_size = c_sh_stride * 16 * TM;
  static constexpr int c_rows_per_pass = THREADS / (2 * TN);

  // K-slice reduction tree: KSLICES/2 slots of one FragC[4][2] per thread.
  static constexpr int red_off = KSLICES / 2;
  static constexpr int red_sh_stride = b_sh_stride_threads * 4 * 2;
  static constexpr int red_size = red_off * red_sh_stride;
  static constexpr int sh_red_size = cmax(red_size, c_sh_size);

  static_assert(a_sh_wr_iters == TM, "one A copy per thread per m-block");
  static_assert(b_sh_wr_iters == K_ITERS, "one B int4 per thread per k-iteration");
  static_assert(THREADS / b_sh_stride_threads == KSLICES,
                "B rows read per k-iteration must match the warp K-slices");
  static_assert(K_ITERS >= 2, "pipeline issues its fetch at k == K_ITERS - 2");
  static_assert(KSLICES % GROUP_BLOCKS == 0,
                "a k-iteration must start on a scale-group boundary");
  static_assert(s_sh_vecs <= THREADS, "one scale vector copy per thread");
  static_assert((16 * TM) % c_rows_per_pass == 0, "whole output passes");
};

// Dynamic shared memory per CTA, shared by the kernel layout and launcher.
// Layout: [max(B stages, reduction/output staging)] [scale stages]
//         [A stages] [gate staging (fused S1 only)].
template <int TM, int TN, bool FUSED>
__host__ __device__ constexpr int smem_int4(int stages) {
  using L = Layout<TM, TN>;
  return cmax(stages * L::b_sh_stage, L::sh_red_size) + stages * L::s_sh_stage +
         stages * L::a_sh_stage + (FUSED ? L::c_sh_size : 0);
}

template <int TM, int TN, bool FUSED>
__host__ __device__ constexpr int smem_bytes(int stages) {
  return 16 * smem_int4<TM, TN, FUSED>(stages);
}

// Deepest pipeline that fits the two-CTA budget, capped at MAX_STAGES.
template <int TM, int TN, bool FUSED>
__host__ __device__ constexpr int pick_stages() {
  int s = MAX_STAGES;
  while (s > 3 && smem_bytes<TM, TN, FUSED>(s) > SMEM_BUDGET_BYTES) --s;
  return s;
}

__device__ __forceinline__ int transform_a(int i) {
  // XOR swizzle of the 16-int4 A rows (same as v1's transform_a).
  const int row = i / 16;
  return 16 * row + ((i % 16) ^ (row % 8));
}

// Two bf16 lanes in raw E2M1 position: magnitude bits 8:6 / 24:22 taken from
// `mag`, sign bit 15 / 31 taken from `sgn`, everything else cleared.
__device__ __forceinline__ uint32_t e2m1_lanes(uint32_t mag, uint32_t sgn) {
  // lop3 0xD8 = (c ? b : a) with c = the sign-bit mask.
  const uint32_t t = static_cast<uint32_t>(
      lop3<0xD8>(static_cast<int>(mag), static_cast<int>(sgn),
                 static_cast<int>(0x80008000u)));
  return t & 0x81C081C0u;
}

// Eight E2M1 nibbles of one Marlin int32 -> two B fragments in raw position
// (value * 2^-126, no rebias). Lane order matches v1's
// dequant_e2m1(q) -> frag_b0 and dequant_e2m1(q >> 8) -> frag_b1.
// eem == 1 (+-0.5) yields the bf16 subnormal 0x0040, exactly as v1.
__device__ __forceinline__ void dequant_e2m1_raw_x8(uint32_t q, FragB& b0, FragB& b1) {
  uint32_t* r0 = reinterpret_cast<uint32_t*>(&b0);
  uint32_t* r1 = reinterpret_cast<uint32_t*>(&b1);
  r0[0] = e2m1_lanes(q << 6, q << 12);  // nibbles at bits 3:0 / 19:16
  r0[1] = e2m1_lanes(q << 2, q << 8);   // bits 7:4 / 23:20
  r1[0] = e2m1_lanes(q >> 2, q << 4);   // bits 11:8 / 27:24
  r1[1] = e2m1_lanes(q >> 6, q);        // bits 15:12 / 31:28
}

// Adding 126 to every byte (no carry for e <= 129) turns E8M0 byte e into the
// exponent field of 2^(e-1), the scale with the E2M1 rebias folded in.
constexpr uint32_t SCALE_FOLD = 0x7E7E7E7Eu;

// bf16x2 folded scales for column pair j (0..3) from 8 folded bytes.
__device__ __forceinline__ uint32_t folded_scale_pair(uint2 s, int j) {
  const uint32_t w = (j < 2) ? s.x : s.y;
  return __byte_perm(w, 0, (j & 1) ? 0x4342 : 0x4140) << 7;
}

template <int TM, int TN, int STAGES, bool FUSED>
__device__ __forceinline__ void grouped_body(
    const int4* __restrict__ A,
    const int4* const* __restrict__ B_ptrs,     // FUSED: gate (w1)
    const int4* const* __restrict__ B2_ptrs,    // FUSED: up (w3); else unused
    int4* const* __restrict__ C_ptrs,
    const int4* const* __restrict__ s_ptrs,     // FUSED: gate scales
    const int4* const* __restrict__ s2_ptrs,    // FUSED: up scales; else unused
    const int* __restrict__ expert_starts,
    const int* __restrict__ expert_counts,
    int num_experts, int prob_n, int prob_k) {
  using L = Layout<TM, TN>;
  static_assert(STAGES >= 3, "pipeline depth");
  static_assert(smem_bytes<TM, TN, FUSED>(STAGES) <= SMEM_BUDGET_BYTES,
                "shared memory exceeds the two-CTA budget");

  const int n_tiles = prob_n / (16 * TN);
  const int matrix_idx = blockIdx.x / n_tiles;
  const int tile_idx = blockIdx.x % n_tiles;
  const int expert_idx = matrix_idx % num_experts;

  const int expert_m = expert_counts[expert_idx];
  if (expert_m <= 0) return;
  const int token_base = expert_starts[expert_idx];

  const int k_tiles = prob_k / (16 * TK);
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int kslice = warp / L::WARPS_N;   // == threadIdx.x / b_sh_stride_threads
  const int warp_n = warp % L::WARPS_N;

  // ---- A indices ----
  const int a_gl_stride = prob_k / 8;
  const int a_gl_rd_delta_i = a_gl_stride * (THREADS / L::a_gl_rd_delta_o);
  const int a_gl_rd_base = a_gl_stride * (threadIdx.x / L::a_gl_rd_delta_o) +
                           threadIdx.x % L::a_gl_rd_delta_o;
  const int a_sh_wr = L::a_sh_stride * (threadIdx.x / L::a_gl_rd_delta_o) +
                      threadIdx.x % L::a_gl_rd_delta_o;
  // Offsets of whole 16-row blocks keep row % 8, so the swizzle commutes with
  // the m-block offset (a_sh_wr_delta, a_sh_rd_delta_i).
  const int a_sh_wr_trans = transform_a(a_sh_wr);
  const int a_sh_rd = L::a_sh_stride * (lane % 16) + lane / 16 + 2 * kslice;
  int a_sh_rd_trans[L::K_ITERS];
#pragma unroll
  for (int i = 0; i < L::K_ITERS; i++)
    a_sh_rd_trans[i] = transform_a(L::a_sh_rd_delta_o * i + a_sh_rd);

  // ---- B indices (int4 units; 16*prob_n/(8*4) int4 per 16-K block row) ----
  const int b_gl_stride = prob_n / 2;
  const int b_gl_rd_delta_o = b_gl_stride * TK;
  const int b_gl_rd_delta_i = b_gl_stride * L::KSLICES;
  const int b_gl_rd_base = b_gl_stride * (threadIdx.x / L::b_sh_stride_threads) +
                           threadIdx.x % L::b_sh_stride_threads +
                           L::b_sh_stride * tile_idx;

  // ---- Scale indices (8-byte vectors of E8M0 bytes) ----
  // Thread t < s_sh_vecs copies vector (t % s_sh_stride) of group row
  // (t / s_sh_stride) of each stage.
  const int s_gl_stride = prob_n / 8;
  const bool s_sh_wr_pred = threadIdx.x < L::s_sh_vecs;
  const int s_gl_rd_base = s_gl_stride * (threadIdx.x / L::s_sh_stride) +
                           L::s_sh_stride * tile_idx + threadIdx.x % L::s_sh_stride;
  // Scale group of a warp's 16-K slice in k-iteration k of a stage:
  // (k * KSLICES + kslice) / GROUP_BLOCKS = k * (KSLICES / GROUP_BLOCKS)
  // + kslice / GROUP_BLOCKS. The kslice term is folded in here; with four
  // K-slices omitting it would read the wrong group for slices 2 and 3.
  const int s_sh_rd = 8 * warp_n + lane / 4 + (kslice / GROUP_BLOCKS) * L::s_sh_stride;

  // ---- Shared memory ----
  extern __shared__ int4 sh[];
  int4* sh_b = sh;
  int4* sh_red = sh;
  int4* sh_s = sh + cmax(STAGES * L::b_sh_stage, L::sh_red_size);
  int4* sh_a = sh_s + STAGES * L::s_sh_stage;
  int4* sh_gate = sh_a + STAGES * L::a_sh_stage;  // FUSED only

  // ---- Registers ----
  FragA frag_a[2][TM];
  I4 frag_b_quant[2];
  uint2 frag_s_raw[2];  // 8 folded scale bytes per k-iteration
  FragC frag_c[TM][4][2];

  // Per super-tile / per pass state referenced by the pipeline lambdas.
  const int4* A_ptr = A;
  const int4* B_base = nullptr;
  const int4* s_base = nullptr;
  bool a_sh_wr_pred[TM];
  int a_gl_rd = 0, b_gl_rd = 0, s_gl_rd = 0;

  auto fetch_to_shared = [&](int pipe, int k_off, bool pred) {
    if (pred) {
      int4* sh_a_stage = sh_a + L::a_sh_stage * pipe;
#pragma unroll
      for (int i = 0; i < L::a_sh_wr_iters; i++)
        cp_async4_pred(&sh_a_stage[a_sh_wr_trans + L::a_sh_wr_delta * i],
                       &A_ptr[a_gl_rd_delta_i * i + a_gl_rd + L::a_gl_rd_delta_o * k_off],
                       a_sh_wr_pred[i]);
      int4* sh_b_stage = sh_b + L::b_sh_stage * pipe;
#pragma unroll
      for (int i = 0; i < L::b_sh_wr_iters; i++)
        cp_async4(&sh_b_stage[L::b_sh_wr_delta * i + threadIdx.x],
                  B_base + (b_gl_rd + b_gl_rd_delta_i * i));
      b_gl_rd += b_gl_rd_delta_o;
      if (s_sh_wr_pred) {
        cp_async8(reinterpret_cast<uint8_t*>(sh_s + L::s_sh_stage * pipe) + 8 * threadIdx.x,
                  reinterpret_cast<const uint8_t*>(s_base) + 8 * s_gl_rd);
      }
      s_gl_rd += GROUPS_PER_STAGE * s_gl_stride;
    }
    cp_async_fence();
  };

  auto load_regs = [&](int k, int pipe) {
    int4* sh_a_stage = sh_a + L::a_sh_stage * (pipe % STAGES);
#pragma unroll
    for (int i = 0; i < TM; i++)
      ldsm4(frag_a[k % 2][i],
            &sh_a_stage[a_sh_rd_trans[k % L::K_ITERS] + L::a_sh_rd_delta_i * i]);
    int4* sh_b_stage = sh_b + L::b_sh_stage * (pipe % STAGES);
    frag_b_quant[k % 2] = *reinterpret_cast<I4*>(
        &sh_b_stage[L::b_sh_wr_delta * (k % L::K_ITERS) + threadIdx.x]);
  };

  auto load_scales = [&](int k, int pipe) {
    const int group = (k % L::K_ITERS) * (L::KSLICES / GROUP_BLOCKS);
    const uint8_t* src = reinterpret_cast<const uint8_t*>(sh_s + L::s_sh_stage * (pipe % STAGES)) +
                         8 * (s_sh_rd + group * L::s_sh_stride);
    uint2 raw = *reinterpret_cast<const uint2*>(src);
    raw.x += SCALE_FOLD;
    raw.y += SCALE_FOLD;
    frag_s_raw[k % 2] = raw;
  };

  auto matmul = [&](int k) {
    const int k2 = k % 2;
#pragma unroll
    for (int j = 0; j < 4; j++) {
      FragB frag_b0, frag_b1;
      dequant_e2m1_raw_x8(static_cast<uint32_t>(frag_b_quant[k2][j]), frag_b0, frag_b1);
      FragS frag_s;
      *reinterpret_cast<uint32_t*>(&frag_s) = folded_scale_pair(frag_s_raw[k2], j);
      scale_op(frag_b0, frag_s, 0);
      scale_op(frag_b1, frag_s, 1);
#pragma unroll
      for (int i = 0; i < TM; i++) {
        mma_op(frag_a[k2][i], frag_b0, frag_c[i][j][0]);
        mma_op(frag_a[k2][i], frag_b1, frag_c[i][j][1]);
      }
    }
  };

  // Same issue order as v1's Marlin pipeline: the fetch into the stage
  // consumed last happens at k == K_ITERS - 2, after every thread's final
  // register load from that stage and before the barrier that publishes the
  // next stage.
  auto run_pipeline = [&]() {
#pragma unroll
    for (int i = 0; i < STAGES - 1; i++)
      fetch_to_shared(i, i, i < k_tiles);
    cp_async_wait<STAGES - 2>();
    __syncthreads();

    load_regs(0, 0);
    load_scales(0, 0);
    a_gl_rd += L::a_gl_rd_delta_o * (STAGES - 1);

    int slice_iters = k_tiles;
    while (slice_iters) {
#pragma unroll
      for (int pipe = 0; pipe < STAGES;) {
#pragma unroll
        for (int k = 0; k < L::K_ITERS; k++) {
          load_regs(k + 1, pipe);
          load_scales(k + 1, pipe);
          if (k == L::K_ITERS - 2) {
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
      a_gl_rd += L::a_gl_rd_delta_o * STAGES;
    }
    cp_async_wait<0>();
  };

  // K-slice reduction into the kslice-0 warps (Marlin thread_block_reduce;
  // red_off == 1 for TN=16 is exactly v1's two-slice reduction).
  auto reduce = [&]() {
    constexpr int red_sh_delta = L::b_sh_stride_threads;
    const int red_idx = threadIdx.x / L::b_sh_stride_threads;
    const int red_sh_rd = L::red_sh_stride * red_idx + threadIdx.x % L::b_sh_stride_threads;
#pragma unroll
    for (int m = 0; m < TM; m++) {
#pragma unroll
      for (int i = L::red_off; i > 0; i /= 2) {
        if (i <= red_idx && red_idx < 2 * i) {
#pragma unroll
          for (int j = 0; j < 4 * 2; j++) {
            const int red_sh_wr = red_sh_delta * j + (red_sh_rd - L::red_sh_stride * i);
            if (i < L::red_off) {
              const float* c_rd = reinterpret_cast<const float*>(&sh_red[red_sh_delta * j + red_sh_rd]);
              const float* c_wr = reinterpret_cast<const float*>(&sh_red[red_sh_wr]);
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
          const float* c_rd = reinterpret_cast<const float*>(&sh_red[red_sh_delta * i + red_sh_rd]);
#pragma unroll
          for (int j = 0; j < 4; j++)
            reinterpret_cast<FragC*>(frag_c)[4 * 2 * m + i][j] += c_rd[j];
        }
      }
      __syncthreads();
    }
  };

  // Round the reduced accumulators to bf16 into the padded staging layout
  // (v1's write-back pattern; only the kslice-0 warps hold full sums).
  auto store_c = [&](int4* dst) {
    if (warp < L::WARPS_N) {
      int c_sh_wr = (4 * L::c_sh_stride) * (lane / 4) + lane % 4 + 32 * warp;
#pragma unroll
      for (int i = 0; i < TM; i++) {
#pragma unroll
        for (int j = 0; j < 4; j++) {
          const int wr = c_sh_wr + 8 * j;
          scalar_t2 r0 = nums2num2(float2num(frag_c[i][j][0][0]), float2num(frag_c[i][j][0][1]));
          scalar_t2 r1 = nums2num2(float2num(frag_c[i][j][0][2]), float2num(frag_c[i][j][0][3]));
          scalar_t2 r2 = nums2num2(float2num(frag_c[i][j][1][0]), float2num(frag_c[i][j][1][1]));
          scalar_t2 r3 = nums2num2(float2num(frag_c[i][j][1][2]), float2num(frag_c[i][j][1][3]));
          reinterpret_cast<scalar_t2*>(dst)[wr + (4 * L::c_sh_stride) * 0 + 0] = r0;
          reinterpret_cast<scalar_t2*>(dst)[wr + (4 * L::c_sh_stride) * 8 + 0] = r1;
          reinterpret_cast<scalar_t2*>(dst)[wr + (4 * L::c_sh_stride) * 0 + 4] = r2;
          reinterpret_cast<scalar_t2*>(dst)[wr + (4 * L::c_sh_stride) * 8 + 4] = r3;
        }
        c_sh_wr += 16 * (4 * L::c_sh_stride);
      }
    }
  };

  constexpr int NUM_PASSES = FUSED ? 2 : 1;
  const int c_gl_stride = prob_n / 8;

  for (int m0 = 0; m0 < expert_m; m0 += 16 * TM) {
    const int prob_m = min(16 * TM, expert_m - m0);
    A_ptr = A + (token_base + m0) * a_gl_stride;
#pragma unroll
    for (int i = 0; i < TM; i++)
      a_sh_wr_pred[i] = L::a_sh_wr_delta * i + a_sh_wr < L::a_sh_stride * prob_m;

    // Pass 0: gate (FUSED) or the only GEMM; pass 1: up. One loop body keeps
    // a single copy of the unrolled pipeline.
#pragma unroll 1
    for (int pass = 0; pass < NUM_PASSES; pass++) {
      B_base = (FUSED && pass) ? B2_ptrs[matrix_idx] : B_ptrs[matrix_idx];
      s_base = (FUSED && pass) ? s2_ptrs[matrix_idx] : s_ptrs[matrix_idx];
      a_gl_rd = a_gl_rd_base;
      b_gl_rd = b_gl_rd_base;
      s_gl_rd = s_gl_rd_base;
#pragma unroll
      for (int i = 0; i < TM * 4 * 2 * 4; i++)
        reinterpret_cast<float*>(frag_c)[i] = 0;

      run_pipeline();
      reduce();
      // The gate staging buffer is disjoint from the pipeline buffers, so the
      // up pass may refill them while the gate result sits there.
      store_c((FUSED && pass == 0) ? sh_gate : sh_red);
      __syncthreads();
    }

    // ---- Write-back of rows < prob_m of this super-tile ----
    int4* C = C_ptrs[matrix_idx] + m0 * c_gl_stride;
    int c_gl_wr = c_gl_stride * (threadIdx.x / (2 * TN)) + threadIdx.x % (2 * TN) +
                  (2 * TN) * tile_idx;
    int c_sh_rd = L::c_sh_stride * (threadIdx.x / (2 * TN)) + threadIdx.x % (2 * TN);
    const int c_gl_wr_end = c_gl_stride * prob_m;
#pragma unroll
    for (int i = 0; i < div_ceil(16 * TM, L::c_rows_per_pass); i++) {
      if (c_gl_wr < c_gl_wr_end) {
        if constexpr (FUSED) {
          int4 gate_chunk = sh_gate[c_sh_rd];
          int4 up_chunk = sh_red[c_sh_rd];
          const scalar_t* g_ptr = reinterpret_cast<const scalar_t*>(&gate_chunk);
          const scalar_t* u_ptr = reinterpret_cast<const scalar_t*>(&up_chunk);
          int4 result;
          scalar_t* r_ptr = reinterpret_cast<scalar_t*>(&result);
#pragma unroll
          for (int e = 0; e < 8; e++)
            r_ptr[e] = float2num(act_gate_mul<Act::SITU>(num2float(g_ptr[e]), num2float(u_ptr[e])));
          C[c_gl_wr] = result;
        } else {
          C[c_gl_wr] = sh_red[c_sh_rd];
        }
        c_gl_wr += c_gl_stride * L::c_rows_per_pass;
        c_sh_rd += L::c_sh_stride * L::c_rows_per_pass;
      }
    }
    // The next super-tile refills the pipeline buffers that alias sh_red and
    // rewrites sh_gate.
    __syncthreads();
  }
}

// Fused S1: SiTU(x @ gate^T, x @ up^T) with v1's rounding points (each branch
// rounded to bf16 before the fp32 epilogue).
template <int TM, int TN, int STAGES>
__global__ void __launch_bounds__(THREADS, 2) GroupedS1(
    const int4* __restrict__ A,
    const int4* const* __restrict__ gate_B_ptrs,
    const int4* const* __restrict__ up_B_ptrs,
    int4* const* __restrict__ C_ptrs,
    const int4* const* __restrict__ gate_scales_ptrs,
    const int4* const* __restrict__ up_scales_ptrs,
    const int* __restrict__ expert_starts,
    const int* __restrict__ expert_counts,
    int num_experts, int prob_n, int prob_k) {
  grouped_body<TM, TN, STAGES, true>(
      A, gate_B_ptrs, up_B_ptrs, C_ptrs, gate_scales_ptrs, up_scales_ptrs,
      expert_starts, expert_counts, num_experts, prob_n, prob_k);
}

// S3 / standalone grouped GEMM. Pointer arrays are indexed by matrix
// (num_matrices >= num_experts); rows come from expert matrix % num_experts.
template <int TM, int TN, int STAGES>
__global__ void __launch_bounds__(THREADS, 2) Grouped(
    const int4* __restrict__ A,
    const int4* const* __restrict__ B_ptrs,
    int4* const* __restrict__ C_ptrs,
    const int4* const* __restrict__ scales_ptrs,
    const int* __restrict__ expert_starts,
    const int* __restrict__ expert_counts,
    int num_experts, int prob_n, int prob_k) {
  grouped_body<TM, TN, STAGES, false>(
      A, B_ptrs, nullptr, C_ptrs, scales_ptrs, nullptr,
      expert_starts, expert_counts, num_experts, prob_n, prob_k);
}

}  // namespace marlin_v2
