// BatchGen — FP8 Blockwise Grouped GEMM Kernel
// Persistent 3-WG CuTe kernel: 2 math WGs (TiledMma N-split) + 1 TMA loader WG.
// Adaptive TileM (16/32/64), TileN=128, TileK=128, 8-stage TMA pipeline.
// x/y/x_scale all live in one row space described by cu_seqlens; the layout may
// be uniform-stride padded ([E, mtp, dim]) or compact ragged, the only contract
// being that every cu_seqlens[e] is a multiple of TileM.

#ifndef BATCHGEN_FP8_BLOCKWISE_GEMM_KERNEL_CUH_
#define BATCHGEN_FP8_BLOCKWISE_GEMM_KERNEL_CUH_

#include <cuda.h>
#include <stdio.h>

#include <cub/cub.cuh>

#include "cute/tensor.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/arch/reg_reconfig.h"
#include "src/moe/fp8_blockwise/fp8_blockwise_utils.cuh"

namespace batchgen {
namespace moe {

namespace kernels {

// ============================================================================
// Tile scheduling: horizontal scan through expert tiles
// ============================================================================
__device__ __forceinline__ void get_next_tile_horizon(const int *tiles_ptr, int iblock,
                                                      int num_group, int &igroup, int &itile_m,
                                                      int &itile_n, int &sum_tile_m,
                                                      cutlass::FastDivmod flat_divider) {
  int num_tile_m, itile_m_total;
  flat_divider(itile_m_total, itile_n, iblock);
  for (int i = igroup; i < num_group; i++) {
    num_tile_m = tiles_ptr[i];
    sum_tile_m += num_tile_m;
    if (itile_m_total < sum_tile_m) {
      igroup = i;
      sum_tile_m = sum_tile_m - num_tile_m;
      itile_m = itile_m_total - sum_tile_m;
      return;
    }
  }
  igroup = -1;
}

__device__ __forceinline__ void get_next_tile_vert(const int *cu_tiles_ptr, int iblock,
                                                   int num_group, int &igroup, int &itile_m,
                                                   int &itile_n, int total_m) {
  int itile_m_total = iblock % total_m;
  itile_n = iblock / total_m;
  int left = 0;
  int right = num_group;
  while (left <= right) {
    int mid = left + (right - left) / 2;
    if (cu_tiles_ptr[mid] > itile_m_total) {
      right = mid - 1;
    } else {
      left = mid + 1;
    }
  }
  itile_m = itile_m_total - cu_tiles_ptr[right];
  igroup = right;
}

// ============================================================================
// Pre-launch kernel: update per-expert TMA descriptors + compute tile counts
// ============================================================================
template <typename Tin, typename Tout, typename TmaX, typename TmaY, int kTileM,
          int kGroupPerThread, int kThreadPerBlock>
__global__ void update_expert_tma(const vec_t<cute::TmaDescriptor, 2> td_xy,
                                  cute::TmaDescriptor *tma_xy, const Tin *x_ptr, const Tout *y_ptr,
                                  const int *seqlens_ptr, const int *cu_seqlens_ptr,
                                  int *tiles_ptr, int *cu_tiles_ptr, int num_group, int m, int n,
                                  int k) {
  using namespace cute;  // NOLINT

  int idx = threadIdx.x;
  int igroup = blockIdx.x;

  if (igroup == num_group) {
    // Last block: compute tile counts + prefix sum
    int tiles[kGroupPerThread];
#pragma unroll
    for (int i = 0; i < kGroupPerThread; i++) {
      int ig = idx * kGroupPerThread + i;
      if (ig < num_group) {
        tiles[i] = (seqlens_ptr[ig] + kTileM - 1) / kTileM;
        tiles_ptr[ig] = tiles[i];
      } else {
        tiles[i] = 0;
      }
    }

    using BlockScan = cub::BlockScan<int, kThreadPerBlock>;
    __shared__ typename BlockScan::TempStorage temp_storage;
    int block_aggregate;
    BlockScan(temp_storage).ExclusiveSum(tiles, tiles, block_aggregate);

#pragma unroll
    for (int i = 0; i < kGroupPerThread; i++) {
      int ig = idx * kGroupPerThread + i;
      if (ig < num_group) {
        cu_tiles_ptr[ig] = tiles[i];
      }
    }
    if (idx == 0) {
      cu_tiles_ptr[num_group] = block_aggregate;
    }

  } else {
    // Per-expert blocks: update TMA descriptors for X and Y
    __shared__ cute::TmaDescriptor smem_tma_desc[2];

    int num_seq = seqlens_ptr[igroup];
    int cu_seqlen = cu_seqlens_ptr[igroup];
    // x_scale is addressed in the SAME row space as x (tile index
    // cu_seqlens[e] / kTileM), so every expert segment start must be a
    // multiple of kTileM. Both supported layouts satisfy this: the padded
    // layout has cu_seqlens[e] = e * mtp with mtp a multiple of 64, and the
    // compact ragged layout aligns segment starts up to 64. Fail loudly
    // rather than silently reading a neighbour expert's scales.
    if (idx == 0 && (cu_seqlen % kTileM) != 0) {
      printf("[fp8_blockwise] FATAL: cu_seqlens[%d]=%d is not a multiple of "
             "TileM=%d; x_scale tile indexing requires aligned expert segment "
             "starts\n",
             igroup, cu_seqlen, kTileM);
      __trap();
    }
    auto *x_ibatch_ptr = x_ptr + cu_seqlen * k;
    auto *y_ibatch_ptr = y_ptr + cu_seqlen * n;

    if (idx < 2) {
      smem_tma_desc[idx] = td_xy[idx];
    }
    __syncwarp();

    // X (activations)
    if (idx == 0) {
      auto gX = make_tensor(make_gmem_ptr(x_ibatch_ptr), make_shape(num_seq, k),
                            make_stride(k, Int<1>{}));
      update_tma_gtensor<TmaX>(smem_tma_desc[idx], gX);
    }

    // Y (output)
    if (idx == 1) {
      auto gY = make_tensor(make_gmem_ptr(y_ibatch_ptr), make_shape(n, num_seq),
                            make_stride(Int<1>{}, n));
      update_tma_gtensor<TmaY>(smem_tma_desc[idx], gY);
    }

#pragma unroll
    for (int i = 0; i < 2; i++) {
      __syncwarp();
      if (cute::elect_one_sync()) {
        cute::tma_desc_commit_group();
        cute::tma_desc_wait_group();
      }
      tma_descriptor_cp_fence_release(tma_xy + igroup * 2 + i, smem_tma_desc[i]);
    }
  }
}

// Pointer-array variant used by streamed expert offload. X/Y still share one
// compact row space; W/WS are independent core-engine ring-slot allocations.
// The preparation stays fully on device: one block patches and publishes four
// TMA descriptors per expert while the final block computes tile prefixes.
template <typename Tin, typename Tout, typename TS, typename TmaX, typename TmaY,
          typename TmaW, typename TmaWS, int kTileM, int kGroupPerThread,
          int kThreadPerBlock>
__global__ void update_expert_tma_ptrs(
    const vec_t<cute::TmaDescriptor, 4> td_xyww,
    cute::TmaDescriptor *tma_xyww, const Tin *x_ptr, const Tout *y_ptr,
    const int64_t *weight_ptrs, const int64_t *w_scale_ptrs,
    const int *seqlens_ptr, const int *cu_seqlens_ptr,
    int *tiles_ptr, int *cu_tiles_ptr, int num_group, int m, int n, int k,
    int num_block_n, int num_block_k_pad4) {
  using namespace cute;  // NOLINT

  int idx = threadIdx.x;
  int igroup = blockIdx.x;

  if (igroup == num_group) {
    int tiles[kGroupPerThread];
#pragma unroll
    for (int i = 0; i < kGroupPerThread; i++) {
      int ig = idx * kGroupPerThread + i;
      if (ig < num_group) {
        tiles[i] = (seqlens_ptr[ig] + kTileM - 1) / kTileM;
        tiles_ptr[ig] = tiles[i];
      } else {
        tiles[i] = 0;
      }
    }

    using BlockScan = cub::BlockScan<int, kThreadPerBlock>;
    __shared__ typename BlockScan::TempStorage temp_storage;
    int block_aggregate;
    BlockScan(temp_storage).ExclusiveSum(tiles, tiles, block_aggregate);

#pragma unroll
    for (int i = 0; i < kGroupPerThread; i++) {
      int ig = idx * kGroupPerThread + i;
      if (ig < num_group) cu_tiles_ptr[ig] = tiles[i];
    }
    if (idx == 0) cu_tiles_ptr[num_group] = block_aggregate;
    return;
  }

  __shared__ cute::TmaDescriptor smem_tma_desc[4];
  int num_seq = seqlens_ptr[igroup];
  int cu_seqlen = cu_seqlens_ptr[igroup];
  if (idx == 0 && (cu_seqlen % kTileM) != 0) {
    printf("[fp8_blockwise_ptrs] FATAL: cu_seqlens[%d]=%d is not a "
           "multiple of TileM=%d\n", igroup, cu_seqlen, kTileM);
    __trap();
  }

  if (idx < 4) smem_tma_desc[idx] = td_xyww[idx];
  __syncwarp();

  if (idx == 0) {
    auto gX = make_tensor(make_gmem_ptr(x_ptr + cu_seqlen * k),
                          make_shape(num_seq, k), make_stride(k, Int<1>{}));
    update_tma_gtensor<TmaX>(smem_tma_desc[0], gX);
  } else if (idx == 1) {
    auto gY = make_tensor(make_gmem_ptr(y_ptr + cu_seqlen * n),
                          make_shape(n, num_seq), make_stride(Int<1>{}, n));
    update_tma_gtensor<TmaY>(smem_tma_desc[1], gY);
  } else if (idx == 2) {
    auto *w_ptr = reinterpret_cast<const Tin *>(weight_ptrs[igroup]);
    auto gW = make_tensor(make_gmem_ptr(w_ptr), make_shape(n, k, Int<1>{}),
                          make_stride(k, Int<1>{}, n * k));
    update_tma_gtensor<TmaW>(smem_tma_desc[2], gW);
  } else if (idx == 3) {
    auto *ws_ptr = reinterpret_cast<const TS *>(w_scale_ptrs[igroup]);
    auto gWS = make_tensor(
        make_gmem_ptr(ws_ptr),
        make_shape(num_block_n, num_block_k_pad4, Int<1>{}),
        make_stride(num_block_k_pad4, Int<1>{},
                    num_block_n * num_block_k_pad4));
    update_tma_gtensor<TmaWS>(smem_tma_desc[3], gWS);
  }

#pragma unroll
  for (int i = 0; i < 4; i++) {
    __syncwarp();
    if (cute::elect_one_sync()) {
      cute::tma_desc_commit_group();
      cute::tma_desc_wait_group();
    }
    tma_descriptor_cp_fence_release(tma_xyww + igroup * 4 + i,
                                    smem_tma_desc[i]);
  }
}

// ============================================================================
// Main persistent kernel: FP8 blockwise grouped GEMM
// 384 threads: 2 math WGs (256 threads) + 1 loader WG (128 threads)
// x_scale is indexed in x's row space via cu_seqlens (see file header).
// ============================================================================
template <typename Config, typename TmaA, typename TmaB, typename TmaC, typename TmaAS,
          typename TmaBS, bool IsLoopH, bool UsePointerWeights = false>
__global__ void __launch_bounds__(384, 1)
    fp8_blockwise_grouped_gemm_kernel(const __grid_constant__ TmaB tma_b,
                                      const __grid_constant__ TmaAS tma_as,
                                      const __grid_constant__ TmaBS tma_bs,
                                      cute::TmaDescriptor *td_xy, int *seqlens_ptr,
                                      const int *cu_seqlens_ptr,
                                      float *xscale_ptr, float *wscale_ptr,
                                      int *tiles_ptr, int *cu_tiles_ptr,
                                      int num_group, int m, int n, int k,
                                      int m_pad,
                                      int num_block_n, int num_block_k,
                                      int num_block_k_pad4,
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

  TmaA tma_a;
  TmaC tma_c;

  int num_total_warps = blockDim.x / 32;
  constexpr int kDescPerGroup = UsePointerWeights ? 4 : 2;
  for (int i = iwarp; i < num_group * kDescPerGroup; i += num_total_warps) {
    tma_descriptor_fence_acquire(td_xy + i);
  }

  auto sA = make_tensor(make_smem_ptr(shm_a), SLayoutA{});
  auto sB = make_tensor(make_smem_ptr(shm_b), SLayoutB{});
  auto sAS = make_tensor(make_smem_ptr(shm_as), SLayoutAS{});
  auto sBS = make_tensor(make_smem_ptr(shm_bs), SLayoutBS{});

  auto gA = tma_a.get_tma_tensor(make_shape(m, k));
  int weight_groups = UsePointerWeights ? 1 : num_group;
  auto gB = tma_b.get_tma_tensor(make_shape(n, k, weight_groups));
  auto gC =
      make_tensor(make_gmem_ptr(static_cast<Tout *>(nullptr)),
                  make_shape(Int<kTileN>{}, Int<kTileM>{}), make_stride(Int<kTileM>{}, Int<1>{}));
  auto gAS = tma_as.get_tma_tensor(make_shape(num_block_k, m_pad));
  auto gBS = tma_bs.get_tma_tensor(
      make_shape(num_block_n, num_block_k_pad4, weight_groups));

  auto btma_a = tma_a.get_slice(0);
  auto btma_b = tma_b.get_slice(0);
  auto btma_as = tma_as.get_slice(0);
  auto btma_bs = tma_bs.get_slice(0);

  auto tAg = btma_a.partition_S(gA);
  auto tAs = btma_a.partition_D(sA);
  auto tBg = btma_b.partition_S(gB);
  auto tBs = btma_b.partition_D(sB);
  auto tASg = btma_as.partition_S(gAS);
  auto tASs = btma_as.partition_D(sAS);
  auto tBSg = btma_bs.partition_S(gBS);
  auto tBSs = btma_bs.partition_D(sBS);

  int num_tile_n = size<1>(tBg);

  if (is_leader_in_block) {
#pragma unroll
    for (int i = 0; i < kStage; ++i) {
      initialize_barrier(readable[i], 1);
      initialize_barrier(writable[i], size(TiledMma{}) / 128);
    }
  }

  int total_m = cu_tiles_ptr[num_group];
  if (total_m <= 0) {
    return;
  }

  if constexpr (IsLoopH) {
    for (int i = idx; i < num_group; i += blockDim.x) {
      shm_tiles[i] = tiles_ptr[i];
    }
  } else {
    for (int i = idx; i < (num_group + 1); i += blockDim.x) {
      shm_tiles[i] = cu_tiles_ptr[i];
    }
  }

  __syncthreads();

  constexpr int kNumThreads = size(TiledMma{});
  // Loader warpgroup
  if (idx >= kNumThreads) {
    cutlass::arch::warpgroup_reg_dealloc<32>();
    idx -= kNumThreads;
    constexpr int kTransactionBytes =
        sizeof(Tin) * (kTileM + kTileN) * kTileK + (kTileM + 4) * sizeof(float);

    int iwarp = __shfl_sync(0xFFFFFFFF, idx / 32, 0);
    int is_leader_in_load = ((iwarp == 0) && elected);

    if (is_leader_in_load) {
      int phase = 1;
      int ismem_write = __shfl_sync(0xFFFFFFFF, 0, 0);
      int iblock = blockIdx.x;
      int ntile_k = size<2>(tAg);

      int igroup = 0;
      int sum_tile_m = 0;
      int itile_m, itile_n;
      while (true) {
        if constexpr (IsLoopH) {
          get_next_tile_horizon(shm_tiles, iblock, num_group, igroup, itile_m, itile_n, sum_tile_m,
                                flat_divider);
          if (igroup < 0) break;
        } else {
          get_next_tile_vert(shm_tiles, iblock, num_group, igroup, itile_m, itile_n, total_m);
          if (itile_n >= num_tile_n) break;
        }

        iblock += gridDim.x;
        auto *td_group = td_xy + igroup * kDescPerGroup;
        auto *td_x = td_group;
        // x_scale shares x's row space: expert igroup's first M-tile of scales
        // sits at row cu_seqlens[igroup], i.e. tile cu_seqlens[igroup]/kTileM.
        // Works for both the padded (cu_seqlens[e] = e*mtp) and the compact
        // ragged (64-aligned segment starts) layouts.
        int scale_tile_base = cu_seqlens_ptr[igroup] / kTileM;

#pragma unroll 1
        for (int itile_k = 0; itile_k < ntile_k; ++itile_k) {
          wait_barrier(writable[ismem_write], phase);

          cute::copy(tma_a.with(td_x, readable[ismem_write]), tAg(_, itile_m, itile_k),
                     tAs(_, 0, 0, ismem_write));
          if constexpr (UsePointerWeights) {
            cute::copy(tma_b.with(td_group + 2, readable[ismem_write]),
                       tBg(_, itile_n, itile_k, Int<0>{}),
                       tBs(_, 0, 0, ismem_write));
          } else {
            cute::copy(tma_b.with(readable[ismem_write]),
                       tBg(_, itile_n, itile_k, igroup),
                       tBs(_, 0, 0, ismem_write));
          }

          cute::copy(tma_as.with(readable[ismem_write]),
                     tASg(_, itile_k, scale_tile_base + itile_m), tASs(_, ismem_write, 0));
          if constexpr (UsePointerWeights) {
            cute::copy(tma_bs.with(td_group + 3, readable[ismem_write]),
                       tBSg(_, itile_n, itile_k / 4, Int<0>{}),
                       tBSs(_, ismem_write, 0));
          } else {
            cute::copy(tma_bs.with(readable[ismem_write]),
                       tBSg(_, itile_n, itile_k / 4, igroup),
                       tBSs(_, ismem_write, 0));
          }

          set_barrier_transaction_bytes(readable[ismem_write], kTransactionBytes);

          ++ismem_write;
          if (ismem_write == kStage) { ismem_write = 0; phase ^= 1; }
        }
      }
    }

  } else {
    // Math warpgroups
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
    int igroup = 0;
    int sum_tile_m = 0;
    int itile_m, itile_n;
    while (true) {
      if constexpr (IsLoopH) {
        get_next_tile_horizon(shm_tiles, iblock, num_group, igroup, itile_m, itile_n, sum_tile_m,
                              flat_divider);
        if (igroup < 0) break;
      } else {
        get_next_tile_vert(shm_tiles, iblock, num_group, igroup, itile_m, itile_n, total_m);
        if (itile_n >= num_tile_n) break;
      }

      iblock += gridDim.x;

      auto tDr = make_tensor_like(tCr);
      clear(tDr);

      int ntile_k = size<2>(tAg);
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
          cute::gemm(tiled_mma, tBr(_, _, ik, ismem_read), tAr(_, _, ik, ismem_read), tCr(_, _, _));
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

      // Epilogue: FP32→BF16 + TMA store
      auto tCrh = make_tensor_like<cute::bfloat16_t>(tCr);
#pragma unroll
      for (int i = 0; i < size(tCr); ++i) {
        tCrh(i) = (Tout)(tDr(i));
      }

      auto sCT = make_tensor(make_smem_ptr(reinterpret_cast<Tout *>(shm_c)), SLayoutCT{});
      using R2SCopyAtomC = Copy_Atom<cute::SM90_U16x8_STSM_T, Tout>;
      auto tiled_copy_c = make_tiled_copy_C(R2SCopyAtomC{}, tiled_mma);
      auto thr_copy_c = tiled_copy_c.get_slice(idx);

      auto tCr4s = thr_copy_c.retile_S(tCrh);
      auto tCs4r = thr_copy_c.partition_D(sCT);

      tma_store_wait<0>();
      syncwarpgroup(iwarpgroup);

      cute::copy(tiled_copy_c, tCr4s, tCs4r);
      syncwarpgroup(iwarpgroup);
      cute::tma_store_fence();

      if (is_leader_in_warpgroup) {
        auto gD = tma_c.get_tma_tensor(make_shape(n, m));
        auto btma_c = tma_c.get_slice(0);
        auto tDs = btma_c.partition_S(sCT);
        auto tDg = btma_c.partition_D(gD);
        auto *td_y = td_xy + igroup * kDescPerGroup + 1;
        cute::copy(tma_c.with(td_y), tDs(_, iwarpgroup, Int<0>{}),
                   tDg(_, itile_n * 2 + iwarpgroup, itile_m));
        tma_store_arrive();
      }
    }
  }
}

}  // namespace kernels
}  // namespace moe
}  // namespace batchgen

#endif  // BATCHGEN_FP8_BLOCKWISE_GEMM_KERNEL_CUH_
