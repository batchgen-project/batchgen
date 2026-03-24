// BatchGen — FP8 Blockwise Grouped GEMM Launcher
// Dispatches to correct TileM variant, launches persistent CuTe kernel.

#include <cuda.h>
#include <stdio.h>

#include <cub/cub.cuh>

#include "cute/tensor.hpp"
#include "src/moe/fp8_blockwise/fp8_blockwise_gemm_config.h"
#include "src/moe/fp8_blockwise/fp8_blockwise_gemm_kernel.cuh"
#include "src/moe/fp8_blockwise/fp8_blockwise_utils.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>
#include <torch/library.h>

namespace batchgen {
namespace moe {

// ============================================================================
// Launcher: configures TMA, dispatches kernel
// ============================================================================
template <int kTileM, int kTileN, int kTileK, int kTileS, int kStage, int kWarpgroupM,
          int kWarpgroupN, int kSwizzleX, int kSwizzleW, int kSwizzleY>
void launch_fp8_blockwise_gemm(void *y_ptr, const void *x_ptr, const void *w_ptr,
                                const void *seqlens_ptr, const void *cu_seqlens_ptr,
                                const void *xscale_ptr, const void *wscale_ptr, void *tmas_ptr,
                                void *tiles_ptr, void *cu_tiles_ptr, int num_group, int m,
                                int n, int k, int m_pad, int num_block_k_pad4, bool update_tma,
                                cudaStream_t stream) {
  using namespace cute;  // NOLINT

  using Tin = cute::float_e4m3_t;
  using Tout = cute::bfloat16_t;
  using TS = float;

  int num_block_k = k / kTileK;
  int num_block_n = n / kTileN;

  // mtp_tiles: uniform stride for x_scale indexing in our reserved buffer layout
  int mtp_tiles = m_pad / (kTileM * num_group);

  auto X = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(x_ptr)), make_shape(m, k),
                       make_stride(k, Int<1>{}));
  auto W = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(w_ptr)),
                       make_shape(n, k, num_group), make_stride(k, Int<1>{}, n * k));
  auto Y = make_tensor(make_gmem_ptr(reinterpret_cast<Tout *>(y_ptr)), make_shape(n, m),
                       make_stride(Int<1>{}, n));
  auto XS = make_tensor(make_gmem_ptr(reinterpret_cast<const TS *>(xscale_ptr)),
                        make_shape(num_block_k, m_pad), make_stride(m_pad, Int<1>{}));
  auto WS = make_tensor(make_gmem_ptr(reinterpret_cast<const TS *>(wscale_ptr)),
                        make_shape(num_block_n, num_block_k_pad4, num_group),
                        make_stride(num_block_k_pad4, Int<1>{}, num_block_n * num_block_k_pad4));

  using Config =
      Fp8BlockwiseGemmConfig<Tin, Tout, TS, kTileM, kTileN, kTileK, kTileS, kStage,
                              kWarpgroupM, kWarpgroupN, kSwizzleX, kSwizzleW, kSwizzleY>;
  Config config;
  auto [tma_x, tma_w, tma_y, tma_xs, tma_ws] = config.get_tma(X, W, Y, XS, WS);

  auto *tma_xy = static_cast<cute::TmaDescriptor *>(tmas_ptr);

  // Pre-launch: update per-expert TMA descriptors + tile counts
  if (update_tma) {
    vec_t<cute::TmaDescriptor, 2> td_xy{
        *tma_x.get_tma_descriptor(),
        *tma_y.get_tma_descriptor(),
    };

    constexpr int kGroupPerThread = 8;
    constexpr int kThreadPerBlock = 32;
    kernels::update_expert_tma<Tin, Tout, decltype(tma_x), decltype(tma_y), kTileM,
                                kGroupPerThread, kThreadPerBlock>
        <<<num_group + 1, kThreadPerBlock, 0, stream>>>(
            td_xy, tma_xy, (const Tin *)x_ptr, (const Tout *)y_ptr, (const int *)seqlens_ptr,
            (const int *)cu_seqlens_ptr, (int *)tiles_ptr, (int *)cu_tiles_ptr, num_group, m, n, k);
  }

  // Main kernel
  {
    int num_tile_n = (n + kTileN - 1) / kTileN;
    cutlass::FastDivmod flat_divider(num_tile_n);

    dim3 block(384);
    dim3 grid(get_sm_count());

    int shm_seq = sizeof(int) * (num_group + 1);
    int shm_size = config.get_shm_size() + shm_seq;

    if (k <= 1024 || n <= 1024) {
      constexpr bool IsLoopH = true;
      auto kernel =
          kernels::fp8_blockwise_grouped_gemm_kernel<decltype(config), decltype(tma_x),
                                                      decltype(tma_w), decltype(tma_y),
                                                      decltype(tma_xs), decltype(tma_ws), IsLoopH>;
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
      kernel<<<grid, block, shm_size, stream>>>(
          tma_w, tma_xs, tma_ws, tma_xy, (int *)seqlens_ptr, (float *)xscale_ptr,
          (float *)wscale_ptr, (int *)tiles_ptr, (int *)cu_tiles_ptr, num_group, m, n, k, m_pad,
          mtp_tiles, num_block_n, num_block_k, num_block_k_pad4, flat_divider);
    } else {
      constexpr bool IsLoopH = false;
      auto kernel =
          kernels::fp8_blockwise_grouped_gemm_kernel<decltype(config), decltype(tma_x),
                                                      decltype(tma_w), decltype(tma_y),
                                                      decltype(tma_xs), decltype(tma_ws), IsLoopH>;
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
      kernel<<<grid, block, shm_size, stream>>>(
          tma_w, tma_xs, tma_ws, tma_xy, (int *)seqlens_ptr, (float *)xscale_ptr,
          (float *)wscale_ptr, (int *)tiles_ptr, (int *)cu_tiles_ptr, num_group, m, n, k, m_pad,
          mtp_tiles, num_block_n, num_block_k, num_block_k_pad4, flat_divider);
    }
  }
}

// ============================================================================
// TileM dispatch based on average tokens per expert
// ============================================================================
void fp8_blockwise_grouped_gemm_async(void *y_ptr, const void *x_ptr, const void *w_ptr,
                                       const void *seqlens_ptr, const void *cu_seqlens_ptr,
                                       const void *xscale_ptr, const void *wscale_ptr, void *tmas_ptr,
                                       void *tiles_ptr, void *cu_tiles_ptr, int num_group, int m,
                                       int n, int k, int m_pad, int num_block_k_pad4,
                                       int num_seq_per_group_avg, bool update_tma,
                                       cudaStream_t stream) {
  constexpr int kTileN = 128;
  constexpr int kTileK = 128;
  constexpr int kTileS = 64;
  constexpr int kWarpgroupM = 2;
  constexpr int kWarpgroupN = 1;
  constexpr int kSwizzleX = 128;
  constexpr int kSwizzleW = 128;
  constexpr int kSwizzleY = 64;

  if (num_seq_per_group_avg <= 16) {
    constexpr int kTileM = 16;
    constexpr int kStage = 8;
    launch_fp8_blockwise_gemm<kTileM, kTileN, kTileK, kTileS, kStage, kWarpgroupM,
                               kWarpgroupN, kSwizzleX, kSwizzleW, kSwizzleY>(
        y_ptr, x_ptr, w_ptr, seqlens_ptr, cu_seqlens_ptr, xscale_ptr, wscale_ptr, tmas_ptr,
        tiles_ptr, cu_tiles_ptr, num_group, m, n, k, m_pad, num_block_k_pad4, update_tma, stream);
  } else if (num_seq_per_group_avg <= 32) {
    constexpr int kTileM = 32;
    constexpr int kStage = 8;
    launch_fp8_blockwise_gemm<kTileM, kTileN, kTileK, kTileS, kStage, kWarpgroupM,
                               kWarpgroupN, kSwizzleX, kSwizzleW, kSwizzleY>(
        y_ptr, x_ptr, w_ptr, seqlens_ptr, cu_seqlens_ptr, xscale_ptr, wscale_ptr, tmas_ptr,
        tiles_ptr, cu_tiles_ptr, num_group, m, n, k, m_pad, num_block_k_pad4, update_tma, stream);
  } else {
    // TileM=48 skipped: mtp (multiple of 64) not divisible by 48.
    // Use TileM=64 for avg > 32.
    constexpr int kTileM = 64;
    constexpr int kStage = 8;
    launch_fp8_blockwise_gemm<kTileM, kTileN, kTileK, kTileS, kStage, kWarpgroupM,
                               kWarpgroupN, kSwizzleX, kSwizzleW, kSwizzleY>(
        y_ptr, x_ptr, w_ptr, seqlens_ptr, cu_seqlens_ptr, xscale_ptr, wscale_ptr, tmas_ptr,
        tiles_ptr, cu_tiles_ptr, num_group, m, n, k, m_pad, num_block_k_pad4, update_tma, stream);
  }
}

// ============================================================================
// PyTorch entry point
// ============================================================================
torch::Tensor fp8_blockwise_grouped_gemm(
    const torch::Tensor &x, const torch::Tensor &weight, const torch::Tensor &seqlens,
    const torch::Tensor &cu_seqlens, const torch::Tensor &x_scale, const torch::Tensor &w_scale,
    const int64_t num_seq_per_group_avg, std::optional<torch::Tensor> output,
    std::optional<torch::Tensor> tma_desc) {
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
  TORCH_CHECK(x.device().is_cuda(), "x must be on CUDA");
  TORCH_CHECK(weight.device().is_cuda(), "weight must be on CUDA");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
  TORCH_CHECK(seqlens.size(0) == weight.size(0), "seqlens and weight num_group mismatch");
  TORCH_CHECK(x.size(1) == weight.size(2), "x and weight K mismatch");
  TORCH_CHECK(w_scale.size(2) % 4 == 0, "w_scale K-dim must be multiple of 4");

  int m = x.size(0);
  int k = x.size(1);
  int n = weight.size(1);
  int m_pad = x_scale.size(1);
  int num_block_k_pad4 = w_scale.size(2);
  int num_group = seqlens.size(0);

  auto options = x.options();
  torch::Tensor y;
  if (output.has_value()) {
    y = output.value();
  } else {
    y = torch::empty({m, n}, options.dtype(torch::kBFloat16));
  }

  torch::Tensor tmas;
  bool update_tma = true;
  if (tma_desc.has_value()) {
    tmas = tma_desc.value();
    update_tma = false;
  } else {
    tmas = torch::empty({num_group * 2, 128}, options);
  }

  torch::Tensor tiles = torch::empty({num_group}, options.dtype(torch::kInt32));
  torch::Tensor cu_tiles = torch::empty({num_group + 1}, options.dtype(torch::kInt32));

  fp8_blockwise_grouped_gemm_async(
      y.mutable_data_ptr(), x.const_data_ptr(), weight.const_data_ptr(),
      seqlens.const_data_ptr(), cu_seqlens.const_data_ptr(),
      x_scale.const_data_ptr(), w_scale.const_data_ptr(),
      tmas.mutable_data_ptr(), tiles.mutable_data_ptr(), cu_tiles.mutable_data_ptr(),
      num_group, m, n, k, m_pad, num_block_k_pad4,
      num_seq_per_group_avg, update_tma, stream);

  return y;
}

TORCH_LIBRARY_FRAGMENT(batchgen_kernels, m) {
  m.def(
      "fp8_blockwise_grouped_gemm(Tensor x, Tensor weight, Tensor seqlens, Tensor cu_seqlens, "
      "Tensor xscale, Tensor wscale, int num_seq_per_group_avg, Tensor? output, "
      "Tensor? tma_desc) -> (Tensor)");
  m.impl("fp8_blockwise_grouped_gemm", torch::kCUDA,
         &batchgen::moe::fp8_blockwise_grouped_gemm);
}

}  // namespace moe
}  // namespace batchgen
