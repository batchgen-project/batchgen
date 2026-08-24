// BatchGen — FP8 Blockwise Fused S1 Kernel: gate GEMM + up GEMM + SiLU
// Two-phase persistent CuTe kernel, forked from fp8_blockwise_gemm_kernel.cuh.
// Phase 1: gate K-loop → gate result to SMEM (aliases shm_c).
// Phase 2: up K-loop → SiLU(gate)*up epilogue → TMA store.

#ifndef BATCHGEN_FP8_BLOCKWISE_S1_KERNEL_CUH_
#define BATCHGEN_FP8_BLOCKWISE_S1_KERNEL_CUH_

#include <cuda.h>
#include <stdio.h>

#include "cute/tensor.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/arch/reg_reconfig.h"
#include "src/moe/fp8_blockwise/fp8_blockwise_utils.cuh"

namespace batchgen {
namespace moe {
namespace kernels {

// ============================================================================
// Fused S1 kernel: gate GEMM + up GEMM + SiLU in single persistent launch.
// Config supplies the math-WG topology; one additional TMA loader WG is required.
// Dual weight TMA descriptors: tma_b_gate/tma_b_up, tma_bs_gate/tma_bs_up.
// Phase transition via __syncthreads + barrier reinitialization.
// Warpgroup barriers use IDs 2/3 to avoid conflict with __syncthreads (barrier 0).
// ============================================================================
template <typename Config, typename TmaA, typename TmaB, typename TmaC,
          typename TmaAS, typename TmaBS, bool IsLoopH, int kBlockThreads,
          int kMinBlocksPerSM = 1>
__global__ void __launch_bounds__(kBlockThreads, kMinBlocksPerSM)
    fp8_blockwise_fused_s1_kernel(
        const __grid_constant__ TmaB tma_b_gate,
        const __grid_constant__ TmaB tma_b_up,
        const __grid_constant__ TmaAS tma_as,
        const __grid_constant__ TmaBS tma_bs_gate,
        const __grid_constant__ TmaBS tma_bs_up,
        cute::TmaDescriptor *td_xy, int *seqlens_ptr,
        const int *cu_seqlens_ptr,
        float *xscale_ptr,
        int *tiles_ptr, int *cu_tiles_ptr,
        int num_group, int m, int n, int k,
        int m_pad,
        int num_block_n, int num_block_k, int num_block_k_pad4,
        cutlass::FastDivmod flat_divider) {
  using namespace cute;  // NOLINT

  using Tin = typename Config::Tin;
  using Tout = typename Config::Tout;
  using TS = typename Config::TS;
  using TiledMma = typename Config::TiledMma;
  using SLayoutA = typename Config::SLayoutX;
  using SLayoutB = typename Config::SLayoutW;
  using SLayoutCT = typename Config::SLayoutY;
  using SLayoutAS = typename Config::SLayoutXS;
  using SLayoutBS = typename Config::SLayoutWS;

  constexpr int kTileM = Config::kTileM;
  constexpr int kTileN = Config::kTileN;
  constexpr int kTileK = Config::kTileK;
  constexpr int kStage = Config::kStage;
  static_assert(Config::kWeightScaleBlockN % kTileN == 0,
                "TileN must divide the 128-row weight-scale block");
  constexpr int kWeightScaleTilesPerBlock = Config::kWeightScaleBlockN / kTileN;

  int idx = threadIdx.x;
  int iwarp = __shfl_sync(0xFFFFFFFF, idx / 32, 0);
  int elected = cute::elect_one_sync();
  bool is_leader_in_block = (iwarp == 0) && elected;
  bool is_leader_in_warpgroup = ((iwarp % 4) == 0) && elected;

  __shared__ uint64_t writable[kStage];
  __shared__ uint64_t readable[kStage];

  extern __shared__ uint8_t shm_data[] alignas(128);
  auto *shm_a = reinterpret_cast<Tin *>(shm_data);
  auto *shm_b = shm_a + cosize(SLayoutA{});
  auto *shm_c = reinterpret_cast<Tout *>(shm_b + cosize(SLayoutB{}));
  auto *shm_as = reinterpret_cast<float *>(shm_c + cosize(SLayoutCT{}));
  auto *shm_bs = reinterpret_cast<float *>(shm_as + cosize(SLayoutAS{}));
  int *shm_tiles = reinterpret_cast<int *>(shm_bs + cosize(SLayoutBS{}));

  // shm_gate aliases shm_c: gate written end of phase 1, consumed in phase 2
  // epilogue before R2S writes to shm_c. Same layout: SLayoutCT = (kTileN, kTileM).
  auto *shm_gate = shm_c;

  TmaA tma_a;
  TmaC tma_c;

  int num_total_warps = blockDim.x / 32;
  for (int i = iwarp; i < num_group * 2; i += num_total_warps) {
    tma_descriptor_fence_acquire(td_xy + i);
  }

  auto sA = make_tensor(make_smem_ptr(shm_a), SLayoutA{});
  auto sB = make_tensor(make_smem_ptr(shm_b), SLayoutB{});
  auto sAS = make_tensor(make_smem_ptr(shm_as), SLayoutAS{});
  auto sBS = make_tensor(make_smem_ptr(shm_bs), SLayoutBS{});

  // TMA partitions — activations + x_scale (shared between phases)
  auto gA = tma_a.get_tma_tensor(make_shape(m, k));
  auto gAS = tma_as.get_tma_tensor(make_shape(num_block_k, m_pad));
  auto btma_a = tma_a.get_slice(0);
  auto btma_as = tma_as.get_slice(0);
  auto tAg = btma_a.partition_S(gA);
  auto tAs = btma_a.partition_D(sA);
  auto tASg = btma_as.partition_S(gAS);
  auto tASs = btma_as.partition_D(sAS);

  // Gate weight TMA partitions
  auto gB_gate = tma_b_gate.get_tma_tensor(make_shape(n, k, num_group));
  auto gBS_gate = tma_bs_gate.get_tma_tensor(make_shape(num_block_n, num_block_k_pad4, num_group));
  auto btma_b_gate = tma_b_gate.get_slice(0);
  auto btma_bs_gate = tma_bs_gate.get_slice(0);
  auto tBg_gate = btma_b_gate.partition_S(gB_gate);
  auto tBs_gate = btma_b_gate.partition_D(sB);
  auto tBSg_gate = btma_bs_gate.partition_S(gBS_gate);
  auto tBSs_gate = btma_bs_gate.partition_D(sBS);

  // Up weight TMA partitions (same SMEM targets, different global sources)
  auto gB_up = tma_b_up.get_tma_tensor(make_shape(n, k, num_group));
  auto gBS_up = tma_bs_up.get_tma_tensor(make_shape(num_block_n, num_block_k_pad4, num_group));
  auto btma_b_up = tma_b_up.get_slice(0);
  auto btma_bs_up = tma_bs_up.get_slice(0);
  auto tBg_up = btma_b_up.partition_S(gB_up);
  auto tBs_up = btma_b_up.partition_D(sB);
  auto tBSg_up = btma_bs_up.partition_S(gBS_up);
  auto tBSs_up = btma_bs_up.partition_D(sBS);

  // Output layout (for MMA fragment sizing)
  auto gC = make_tensor(make_gmem_ptr(static_cast<Tout *>(nullptr)),
                        make_shape(Int<kTileN>{}, Int<kTileM>{}),
                        make_stride(Int<kTileM>{}, Int<1>{}));

  int num_tile_n = size<1>(tBg_gate);

  if (is_leader_in_block) {
#pragma unroll
    for (int i = 0; i < kStage; ++i) {
      initialize_barrier(readable[i], 1);
      initialize_barrier(writable[i], size(TiledMma{}) / 128);
    }
  }

  int total_m = cu_tiles_ptr[num_group];
  if (total_m <= 0) return;

  if constexpr (IsLoopH) {
    for (int i = idx; i < num_group; i += blockDim.x) shm_tiles[i] = tiles_ptr[i];
  } else {
    for (int i = idx; i < (num_group + 1); i += blockDim.x) shm_tiles[i] = cu_tiles_ptr[i];
  }

  __syncthreads();

  constexpr int kNumThreads = size(TiledMma{});
  static_assert(kBlockThreads == kNumThreads + 128,
                "block must contain the math threads plus one loader warpgroup");
  static_assert(kMinBlocksPerSM == 1 || (kMinBlocksPerSM == 2 && kBlockThreads <= 256),
                "minBlocksPerSM=2 is only valid for blocks of at most 256 threads");

  // ========================================================================
  // LOADER WARPGROUP (after the Config-selected math threads)
  // ========================================================================
  if (idx >= kNumThreads) {
    cutlass::arch::warpgroup_reg_dealloc<32>();
    int loader_idx = idx - kNumThreads;

    constexpr int kTransactionBytes =
        sizeof(Tin) * (kTileM + kTileN) * kTileK + (kTileM + 4) * sizeof(float);

    int loader_iwarp = __shfl_sync(0xFFFFFFFF, loader_idx / 32, 0);
    int is_leader_in_load = ((loader_iwarp == 0) && elected);

    int ntile_k = size<2>(tAg);

    // All loader threads iterate tile loop (for __syncthreads pairing)
    int pipe_phase = 1;
    int ismem_write = 0;
    int iblock = blockIdx.x;
    int igroup = 0, sum_tile_m = 0;
    int itile_m, itile_n;

    while (true) {
      if constexpr (IsLoopH) {
        get_next_tile_horizon(shm_tiles, iblock, num_group, igroup, itile_m,
                              itile_n, sum_tile_m, flat_divider);
        if (igroup < 0) break;
      } else {
        get_next_tile_vert(shm_tiles, iblock, num_group, igroup, itile_m,
                           itile_n, total_m);
        if (itile_n >= num_tile_n) break;
      }
      iblock += gridDim.x;
      // x_scale shares x's row space (see fp8_blockwise_gemm_kernel.cuh header):
      // expert igroup's first scale tile is cu_seqlens[igroup] / kTileM.
      int scale_tile_base = cu_seqlens_ptr[igroup] / kTileM;
      int weight_scale_tile_n = itile_n / kWeightScaleTilesPerBlock;

      // ──── PHASE 1: Gate weight K-loop (leader only) ────
      if (is_leader_in_load) {
        auto *td_x = td_xy + igroup * 2;
#pragma unroll 1
        for (int itile_k = 0; itile_k < ntile_k; ++itile_k) {
          wait_barrier(writable[ismem_write], pipe_phase);
          cute::copy(tma_a.with(td_x, readable[ismem_write]),
                     tAg(_, itile_m, itile_k), tAs(_, 0, 0, ismem_write));
          cute::copy(tma_b_gate.with(readable[ismem_write]),
                     tBg_gate(_, itile_n, itile_k, igroup),
                     tBs_gate(_, 0, 0, ismem_write));
          cute::copy(tma_as.with(readable[ismem_write]),
                     tASg(_, itile_k, scale_tile_base + itile_m),
                     tASs(_, ismem_write, 0));
          cute::copy(tma_bs_gate.with(readable[ismem_write]),
                     tBSg_gate(_, weight_scale_tile_n, itile_k / 4, igroup),
                     tBSs_gate(_, ismem_write, 0));
          set_barrier_transaction_bytes(readable[ismem_write], kTransactionBytes);
          ++ismem_write;
          if (ismem_write == kStage) { ismem_write = 0; pipe_phase ^= 1; }
        }
      }

      // Phase transition: all math and loader threads sync.
      __syncthreads();
      __syncthreads();  // Wait for barrier reinit by block leader (math branch)

      // Reset pipeline state for phase 2
      ismem_write = 0;
      pipe_phase = 1;

      // ──── PHASE 2: Up weight K-loop (leader only) ────
      if (is_leader_in_load) {
        auto *td_x = td_xy + igroup * 2;
#pragma unroll 1
        for (int itile_k = 0; itile_k < ntile_k; ++itile_k) {
          wait_barrier(writable[ismem_write], pipe_phase);
          cute::copy(tma_a.with(td_x, readable[ismem_write]),
                     tAg(_, itile_m, itile_k), tAs(_, 0, 0, ismem_write));
          cute::copy(tma_b_up.with(readable[ismem_write]),
                     tBg_up(_, itile_n, itile_k, igroup),
                     tBs_up(_, 0, 0, ismem_write));
          cute::copy(tma_as.with(readable[ismem_write]),
                     tASg(_, itile_k, scale_tile_base + itile_m),
                     tASs(_, ismem_write, 0));
          cute::copy(tma_bs_up.with(readable[ismem_write]),
                     tBSg_up(_, weight_scale_tile_n, itile_k / 4, igroup),
                     tBSs_up(_, ismem_write, 0));
          set_barrier_transaction_bytes(readable[ismem_write], kTransactionBytes);
          ++ismem_write;
          if (ismem_write == kStage) { ismem_write = 0; pipe_phase ^= 1; }
        }
      }
    }

  // ========================================================================
  // MATH WARPGROUPS
  // ========================================================================
  } else {
    cutlass::arch::warpgroup_reg_alloc<168>();

    int idx_in_warpgroup = idx % 128;
    int iwarpgroup = idx / 128;
    int iwarp_in_warpgroup = idx_in_warpgroup / 32;
    int elected_idx_in_warpgroup = ((iwarp_in_warpgroup == 0) && elected);

    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_slice(idx);
    auto tBs4r = thr_mma.partition_A(sB);
    auto tAs4r = thr_mma.partition_B(sA);
    auto tBr = thr_mma.make_fragment_A(tBs4r);
    auto tAr = thr_mma.make_fragment_B(tAs4r);
    auto tCr = thr_mma.partition_fragment_C(gC);
    auto tCr_mn = retile_fragment(tCr);
    constexpr int kM = size<0>(tCr_mn);
    constexpr int kN = size<1>(tCr_mn);

    auto gI = make_identity_tensor(gC.shape());
    auto tI = thr_mma.partition_C(gI);
    auto tI_mn = retile_fragment(tI);

    int ismem_read = 0;
    int phase = 0;
    int iblock = blockIdx.x;
    int igroup = 0, sum_tile_m = 0;
    int itile_m, itile_n;

    while (true) {
      if constexpr (IsLoopH) {
        get_next_tile_horizon(shm_tiles, iblock, num_group, igroup, itile_m,
                              itile_n, sum_tile_m, flat_divider);
        if (igroup < 0) break;
      } else {
        get_next_tile_vert(shm_tiles, iblock, num_group, igroup, itile_m,
                           itile_n, total_m);
        if (itile_n >= num_tile_n) break;
      }
      iblock += gridDim.x;

      int ntile_k = size<2>(tAg);

      // ──── PHASE 1: Gate K-loop ────
      {
        auto tDr = make_tensor_like(tCr);
        clear(tDr);

#pragma unroll 1
        for (int itile_k = 0; itile_k < ntile_k; ++itile_k) {
          wait_barrier(readable[ismem_read], phase);

          float tCS[kN];
          float wscale = sBS(ismem_read, itile_k % 4);
#pragma unroll
          for (int in = 0; in < kN; in++) {
            tCS[in] = sAS(ismem_read, get<1>(tI_mn(0, in))) * wscale;
          }

          tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
          warpgroup_fence_operand(tCr);
          warpgroup_arrive();
#pragma unroll
          for (int ik = 0; ik < size<2>(tAr); ++ik) {
            cute::gemm(tiled_mma, tBr(_, _, ik, ismem_read),
                       tAr(_, _, ik, ismem_read), tCr(_, _, _));
            tiled_mma.accumulate_ = GMMA::ScaleOut::One;
          }
          warpgroup_commit_batch();
          warpgroup_wait<0>();
          warpgroup_fence_operand(tCr);

          auto tDr_mn = retile_fragment(tDr);
#pragma unroll
          for (int in = 0; in < kN; in++) {
            float yscale = tCS[in];
#pragma unroll
            for (int im = 0; im < kM; im++) {
              tDr_mn(im, in) = tCr_mn(im, in) * yscale + tDr_mn(im, in);
            }
          }

          if (elected_idx_in_warpgroup) {
            arrive_barrier(writable[ismem_read]);
          }
          ++ismem_read;
          if (ismem_read == kStage) { phase ^= 1; ismem_read = 0; }
        }

        // Gate to SMEM: FP32 → BF16 → R2S
        auto tCrh_gate = make_tensor_like<cute::bfloat16_t>(tCr);
#pragma unroll
        for (int i = 0; i < size(tCr); ++i) {
          tCrh_gate(i) = (Tout)(tDr(i));
        }

        auto sGate = make_tensor(make_smem_ptr(reinterpret_cast<Tout *>(shm_gate)),
                                 SLayoutCT{});
        using R2SCopyAtomC = Copy_Atom<cute::SM90_U16x8_STSM_T, Tout>;
        auto tiled_copy_gate = make_tiled_copy_C(R2SCopyAtomC{}, tiled_mma);
        auto thr_copy_gate = tiled_copy_gate.get_slice(idx);
        auto tGr4s = thr_copy_gate.retile_S(tCrh_gate);
        auto tGs4r = thr_copy_gate.partition_D(sGate);

        // Wait for previous tile's TMA store (shm_gate aliases shm_c)
        tma_store_wait<0>();
        syncwarpgroup(iwarpgroup + 2);
        cute::copy(tiled_copy_gate, tGr4s, tGs4r);
        syncwarpgroup(iwarpgroup + 2);
      }

      // Phase transition: all math and loader threads sync.
      __syncthreads();

      // Reinitialize barriers for phase 2
      if (is_leader_in_block) {
#pragma unroll
        for (int i = 0; i < kStage; ++i) {
          initialize_barrier(readable[i], 1);
          initialize_barrier(writable[i], size(TiledMma{}) / 128);
        }
      }
      __syncthreads();

      // Reset pipeline state for phase 2
      ismem_read = 0;
      phase = 0;

      // ──── PHASE 2: Up K-loop ────
      {
        auto tDr = make_tensor_like(tCr);
        clear(tDr);

#pragma unroll 1
        for (int itile_k = 0; itile_k < ntile_k; ++itile_k) {
          wait_barrier(readable[ismem_read], phase);

          float tCS[kN];
          float wscale = sBS(ismem_read, itile_k % 4);
#pragma unroll
          for (int in = 0; in < kN; in++) {
            tCS[in] = sAS(ismem_read, get<1>(tI_mn(0, in))) * wscale;
          }

          tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
          warpgroup_fence_operand(tCr);
          warpgroup_arrive();
#pragma unroll
          for (int ik = 0; ik < size<2>(tAr); ++ik) {
            cute::gemm(tiled_mma, tBr(_, _, ik, ismem_read),
                       tAr(_, _, ik, ismem_read), tCr(_, _, _));
            tiled_mma.accumulate_ = GMMA::ScaleOut::One;
          }
          warpgroup_commit_batch();
          warpgroup_wait<0>();
          warpgroup_fence_operand(tCr);

          auto tDr_mn = retile_fragment(tDr);
#pragma unroll
          for (int in = 0; in < kN; in++) {
            float yscale = tCS[in];
#pragma unroll
            for (int im = 0; im < kM; im++) {
              tDr_mn(im, in) = tCr_mn(im, in) * yscale + tDr_mn(im, in);
            }
          }

          if (elected_idx_in_warpgroup) {
            arrive_barrier(writable[ismem_read]);
          }
          ++ismem_read;
          if (ismem_read == kStage) { phase ^= 1; ismem_read = 0; }
        }

        // SiLU Fused Epilogue: load gate from SMEM, compute SiLU(gate)*up
        {
          auto sGate = make_tensor(
              make_smem_ptr(reinterpret_cast<Tout *>(shm_gate)), SLayoutCT{});
          auto tDr_mn = retile_fragment(tDr);

          // tI_mn: get<0>=N coord, get<1>=M coord. SLayoutCT=(kTileN, kTileM).
#pragma unroll
          for (int in = 0; in < kN; in++) {
#pragma unroll
            for (int im = 0; im < kM; im++) {
              auto coord = tI_mn(im, in);
              auto gate_bf16 = sGate(get<0>(coord), get<1>(coord));
              float gate_val =
                  __bfloat162float(reinterpret_cast<const __nv_bfloat16 &>(gate_bf16));
              float sigmoid = 1.0f / (1.0f + __expf(-gate_val));
              tDr_mn(im, in) = (gate_val * sigmoid) * tDr_mn(im, in);
            }
          }

          // Standard epilogue: FP32 → BF16 → R2S → TMA store
          auto tCrh = make_tensor_like<cute::bfloat16_t>(tCr);
#pragma unroll
          for (int i = 0; i < size(tCr); ++i) {
            tCrh(i) = (Tout)(tDr(i));
          }

          auto sCT = make_tensor(make_smem_ptr(reinterpret_cast<Tout *>(shm_c)),
                                 SLayoutCT{});
          using R2SCopyAtomC = Copy_Atom<cute::SM90_U16x8_STSM_T, Tout>;
          auto tiled_copy_c = make_tiled_copy_C(R2SCopyAtomC{}, tiled_mma);
          auto thr_copy_c = tiled_copy_c.get_slice(idx);
          auto tCr4s = thr_copy_c.retile_S(tCrh);
          auto tCs4r = thr_copy_c.partition_D(sCT);

          syncwarpgroup(iwarpgroup + 2);
          cute::copy(tiled_copy_c, tCr4s, tCs4r);
          syncwarpgroup(iwarpgroup + 2);
          cute::tma_store_fence();

          if (is_leader_in_warpgroup) {
            auto gD = tma_c.get_tma_tensor(make_shape(n, m));
            auto btma_c = tma_c.get_slice(0);
            auto tDs = btma_c.partition_S(sCT);
            auto tDg = btma_c.partition_D(gD);
            auto *td_y = td_xy + igroup * 2 + 1;
            cute::copy(tma_c.with(td_y), tDs(_, iwarpgroup, Int<0>{}),
                       tDg(_, itile_n * Config::kWarpgroupM + iwarpgroup, itile_m));
            tma_store_arrive();
          }
        }
      }
    }
  }
}

}  // namespace kernels
}  // namespace moe
}  // namespace batchgen

#endif  // BATCHGEN_FP8_BLOCKWISE_S1_KERNEL_CUH_
