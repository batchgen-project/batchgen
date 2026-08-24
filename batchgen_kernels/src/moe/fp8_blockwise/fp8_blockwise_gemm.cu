// BatchGen — FP8 Blockwise Grouped GEMM Launcher
// Dispatches to correct TileM variant, launches persistent CuTe kernel.

#include <cuda.h>
#include <stdio.h>

#include <cub/cub.cuh>

#include "cute/tensor.hpp"
#include "src/moe/fp8_blockwise/fp8_blockwise_gemm_config.h"
#include "src/moe/fp8_blockwise/fp8_blockwise_gemm_kernel.cuh"
#include "src/moe/fp8_blockwise/fp8_blockwise_s1_kernel.cuh"
#include "src/moe/fp8_blockwise/fp8_blockwise_utils.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>
#include <torch/extension.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;

namespace batchgen {
namespace moe {

// ============================================================================
// Launcher: configures TMA, dispatches kernel
// ============================================================================
template <int kTileM, int kTileN, int kTileK, int kTileS, int kStage, int kWarpgroupM,
          int kWarpgroupN, int kSwizzleX, int kSwizzleW, int kSwizzleY,
          int kBlockThreads = 384, int kGridMultiplier = 1>
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
  using Config =
      Fp8BlockwiseGemmConfig<Tin, Tout, TS, kTileM, kTileN, kTileK, kTileS, kStage,
                              kWarpgroupM, kWarpgroupN, kSwizzleX, kSwizzleW, kSwizzleY>;

  int num_block_k = k / kTileK;
  int num_block_n = n / Config::kWeightScaleBlockN;

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

    dim3 block(kBlockThreads);
    dim3 grid(kGridMultiplier * get_sm_count());

    int shm_seq = sizeof(int) * (num_group + 1);
    int shm_size = config.get_shm_size() + shm_seq;

    if (k <= 1024 || n <= 1024) {
      constexpr bool IsLoopH = true;
      auto kernel =
          kernels::fp8_blockwise_grouped_gemm_kernel<decltype(config), decltype(tma_x),
                                                      decltype(tma_w), decltype(tma_y),
                                                      decltype(tma_xs), decltype(tma_ws), IsLoopH,
                                                      kBlockThreads>;
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
      kernel<<<grid, block, shm_size, stream>>>(
          tma_w, tma_xs, tma_ws, tma_xy, (int *)seqlens_ptr, (const int *)cu_seqlens_ptr,
          (float *)xscale_ptr, (float *)wscale_ptr, (int *)tiles_ptr, (int *)cu_tiles_ptr,
          num_group, m, n, k, m_pad, num_block_n, num_block_k, num_block_k_pad4, flat_divider);
    } else {
      constexpr bool IsLoopH = false;
      auto kernel =
          kernels::fp8_blockwise_grouped_gemm_kernel<decltype(config), decltype(tma_x),
                                                      decltype(tma_w), decltype(tma_y),
                                                      decltype(tma_xs), decltype(tma_ws), IsLoopH,
                                                      kBlockThreads>;
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
      kernel<<<grid, block, shm_size, stream>>>(
          tma_w, tma_xs, tma_ws, tma_xy, (int *)seqlens_ptr, (const int *)cu_seqlens_ptr,
          (float *)xscale_ptr, (float *)wscale_ptr, (int *)tiles_ptr, (int *)cu_tiles_ptr,
          num_group, m, n, k, m_pad, num_block_n, num_block_k, num_block_k_pad4, flat_divider);
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

// Opt-in occupancy probe. Production dispatch above remains 3-WG/TileN128/stage8.
void fp8_blockwise_grouped_gemm_high_occ_async(
    void *y_ptr, const void *x_ptr, const void *w_ptr,
    const void *seqlens_ptr, const void *cu_seqlens_ptr,
    const void *xscale_ptr, const void *wscale_ptr, void *tmas_ptr,
    void *tiles_ptr, void *cu_tiles_ptr, int num_group, int m,
    int n, int k, int m_pad, int num_block_k_pad4,
    int num_seq_per_group_avg, bool update_tma, cudaStream_t stream) {
  constexpr int kTileN = 64;
  constexpr int kTileK = 128;
  constexpr int kTileS = 64;
  constexpr int kStage = 4;
  constexpr int kWarpgroupM = 1;
  constexpr int kWarpgroupN = 1;
  constexpr int kSwizzleX = 128;
  constexpr int kSwizzleW = 128;
  constexpr int kSwizzleY = 64;
  constexpr int kBlockThreads = 256;
  constexpr int kGridMultiplier = 2;

  if (num_seq_per_group_avg <= 16) {
    launch_fp8_blockwise_gemm<16, kTileN, kTileK, kTileS, kStage, kWarpgroupM,
                               kWarpgroupN, kSwizzleX, kSwizzleW, kSwizzleY,
                               kBlockThreads, kGridMultiplier>(
        y_ptr, x_ptr, w_ptr, seqlens_ptr, cu_seqlens_ptr, xscale_ptr, wscale_ptr, tmas_ptr,
        tiles_ptr, cu_tiles_ptr, num_group, m, n, k, m_pad, num_block_k_pad4, update_tma, stream);
  } else if (num_seq_per_group_avg <= 32) {
    launch_fp8_blockwise_gemm<32, kTileN, kTileK, kTileS, kStage, kWarpgroupM,
                               kWarpgroupN, kSwizzleX, kSwizzleW, kSwizzleY,
                               kBlockThreads, kGridMultiplier>(
        y_ptr, x_ptr, w_ptr, seqlens_ptr, cu_seqlens_ptr, xscale_ptr, wscale_ptr, tmas_ptr,
        tiles_ptr, cu_tiles_ptr, num_group, m, n, k, m_pad, num_block_k_pad4, update_tma, stream);
  } else {
    launch_fp8_blockwise_gemm<64, kTileN, kTileK, kTileS, kStage, kWarpgroupM,
                               kWarpgroupN, kSwizzleX, kSwizzleW, kSwizzleY,
                               kBlockThreads, kGridMultiplier>(
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

torch::Tensor fp8_blockwise_grouped_gemm_high_occ(
    const torch::Tensor &x, const torch::Tensor &weight, const torch::Tensor &seqlens,
    const torch::Tensor &cu_seqlens, const torch::Tensor &x_scale, const torch::Tensor &w_scale,
    const int64_t num_seq_per_group_avg, std::optional<torch::Tensor> output) {
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
  TORCH_CHECK(n % 128 == 0, "weight N-dim must be a multiple of the 128-row scale block");

  auto options = x.options();
  torch::Tensor y;
  if (output.has_value()) {
    y = output.value();
  } else {
    y = torch::empty({m, n}, options.dtype(torch::kBFloat16));
  }

  torch::Tensor tmas = torch::empty({num_group * 2, 128}, options);
  torch::Tensor tiles = torch::empty({num_group}, options.dtype(torch::kInt32));
  torch::Tensor cu_tiles = torch::empty({num_group + 1}, options.dtype(torch::kInt32));

  fp8_blockwise_grouped_gemm_high_occ_async(
      y.mutable_data_ptr(), x.const_data_ptr(), weight.const_data_ptr(),
      seqlens.const_data_ptr(), cu_seqlens.const_data_ptr(),
      x_scale.const_data_ptr(), w_scale.const_data_ptr(),
      tmas.mutable_data_ptr(), tiles.mutable_data_ptr(), cu_tiles.mutable_data_ptr(),
      num_group, m, n, k, m_pad, num_block_k_pad4,
      num_seq_per_group_avg, true, stream);

  return y;
}

// ============================================================================
// Fused S1 Launcher: gate GEMM + up GEMM + SiLU in single kernel
// ============================================================================
template <int kTileM, int kTileN, int kTileK, int kTileS, int kStage,
          int kWarpgroupM, int kWarpgroupN,
          int kSwizzleX, int kSwizzleW, int kSwizzleY,
          int kBlockThreads = 384, int kGridMultiplier = 1>
void launch_fp8_blockwise_fused_s1(
    void *y_ptr, const void *x_ptr,
    const void *gate_w_ptr, const void *up_w_ptr,
    const void *seqlens_ptr, const void *cu_seqlens_ptr,
    const void *xscale_ptr,
    const void *gate_wscale_ptr, const void *up_wscale_ptr,
    void *tmas_ptr, void *tiles_ptr, void *cu_tiles_ptr,
    int num_group, int m, int n, int k, int m_pad,
    int num_block_k_pad4, bool update_tma, cudaStream_t stream) {
  using namespace cute;  // NOLINT
  using Tin = cute::float_e4m3_t;
  using Tout = cute::bfloat16_t;
  using TS = float;
  using Config = Fp8BlockwiseGemmConfig<Tin, Tout, TS, kTileM, kTileN, kTileK, kTileS, kStage,
                                         kWarpgroupM, kWarpgroupN, kSwizzleX, kSwizzleW, kSwizzleY>;

  int num_block_k = k / kTileK;
  int num_block_n = n / Config::kWeightScaleBlockN;

  auto X = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(x_ptr)),
                       make_shape(m, k), make_stride(k, Int<1>{}));
  auto W_gate = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(gate_w_ptr)),
                            make_shape(n, k, num_group), make_stride(k, Int<1>{}, n * k));
  auto W_up = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(up_w_ptr)),
                          make_shape(n, k, num_group), make_stride(k, Int<1>{}, n * k));
  auto Y = make_tensor(make_gmem_ptr(reinterpret_cast<Tout *>(y_ptr)),
                       make_shape(n, m), make_stride(Int<1>{}, n));
  auto XS = make_tensor(make_gmem_ptr(reinterpret_cast<const TS *>(xscale_ptr)),
                        make_shape(num_block_k, m_pad), make_stride(m_pad, Int<1>{}));
  auto WS_gate = make_tensor(make_gmem_ptr(reinterpret_cast<const TS *>(gate_wscale_ptr)),
                             make_shape(num_block_n, num_block_k_pad4, num_group),
                             make_stride(num_block_k_pad4, Int<1>{}, num_block_n * num_block_k_pad4));
  auto WS_up = make_tensor(make_gmem_ptr(reinterpret_cast<const TS *>(up_wscale_ptr)),
                           make_shape(num_block_n, num_block_k_pad4, num_group),
                           make_stride(num_block_k_pad4, Int<1>{}, num_block_n * num_block_k_pad4));

  Config config;

  auto [tma_x, tma_w_gate, tma_y, tma_xs, tma_ws_gate] = config.get_tma(X, W_gate, Y, XS, WS_gate);
  auto tma_w_up = make_tma_copy(SM90_TMA_LOAD{}, W_up, take<0, 2>(typename Config::SLayoutW{}));
  auto tma_ws_up = make_tma_copy(SM90_TMA_LOAD{}, WS_up, typename Config::CopyBoxWS{});

  auto *tma_xy = static_cast<cute::TmaDescriptor *>(tmas_ptr);

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
            td_xy, tma_xy, (const Tin *)x_ptr, (const Tout *)y_ptr,
            (const int *)seqlens_ptr, (const int *)cu_seqlens_ptr,
            (int *)tiles_ptr, (int *)cu_tiles_ptr, num_group, m, n, k);
  }

  {
    int num_tile_n = (n + kTileN - 1) / kTileN;
    cutlass::FastDivmod flat_divider(num_tile_n);
    dim3 block(kBlockThreads);
    dim3 grid(kGridMultiplier * get_sm_count());
    int shm_seq = sizeof(int) * (num_group + 1);
    int shm_size = config.get_shm_size() + shm_seq;

    if (k <= 1024 || n <= 1024) {
      constexpr bool IsLoopH = true;
      auto kernel = kernels::fp8_blockwise_fused_s1_kernel<
          decltype(config), decltype(tma_x), decltype(tma_w_gate), decltype(tma_y),
          decltype(tma_xs), decltype(tma_ws_gate), IsLoopH, kBlockThreads>;
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
      kernel<<<grid, block, shm_size, stream>>>(
          tma_w_gate, tma_w_up, tma_xs, tma_ws_gate, tma_ws_up,
          tma_xy, (int *)seqlens_ptr, (const int *)cu_seqlens_ptr, (float *)xscale_ptr,
          (int *)tiles_ptr, (int *)cu_tiles_ptr,
          num_group, m, n, k, m_pad,
          num_block_n, num_block_k, num_block_k_pad4, flat_divider);
    } else {
      constexpr bool IsLoopH = false;
      auto kernel = kernels::fp8_blockwise_fused_s1_kernel<
          decltype(config), decltype(tma_x), decltype(tma_w_gate), decltype(tma_y),
          decltype(tma_xs), decltype(tma_ws_gate), IsLoopH, kBlockThreads>;
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
      kernel<<<grid, block, shm_size, stream>>>(
          tma_w_gate, tma_w_up, tma_xs, tma_ws_gate, tma_ws_up,
          tma_xy, (int *)seqlens_ptr, (const int *)cu_seqlens_ptr, (float *)xscale_ptr,
          (int *)tiles_ptr, (int *)cu_tiles_ptr,
          num_group, m, n, k, m_pad,
          num_block_n, num_block_k, num_block_k_pad4, flat_divider);
    }
  }
}

void fp8_blockwise_fused_s1_async(
    void *y_ptr, const void *x_ptr,
    const void *gate_w_ptr, const void *up_w_ptr,
    const void *seqlens_ptr, const void *cu_seqlens_ptr,
    const void *xscale_ptr,
    const void *gate_wscale_ptr, const void *up_wscale_ptr,
    void *tmas_ptr, void *tiles_ptr, void *cu_tiles_ptr,
    int num_group, int m, int n, int k, int m_pad,
    int num_block_k_pad4, int num_seq_per_group_avg,
    bool update_tma, cudaStream_t stream) {
  constexpr int kTileN = 128, kTileK = 128, kTileS = 64;
  constexpr int kWarpgroupM = 2, kWarpgroupN = 1;
  constexpr int kSwizzleX = 128, kSwizzleW = 128, kSwizzleY = 64;

  if (num_seq_per_group_avg <= 16) {
    launch_fp8_blockwise_fused_s1<16, kTileN, kTileK, kTileS, 8, kWarpgroupM, kWarpgroupN, kSwizzleX, kSwizzleW, kSwizzleY>(
        y_ptr, x_ptr, gate_w_ptr, up_w_ptr, seqlens_ptr, cu_seqlens_ptr,
        xscale_ptr, gate_wscale_ptr, up_wscale_ptr, tmas_ptr, tiles_ptr,
        cu_tiles_ptr, num_group, m, n, k, m_pad, num_block_k_pad4,
        update_tma, stream);
  } else if (num_seq_per_group_avg <= 32) {
    launch_fp8_blockwise_fused_s1<32, kTileN, kTileK, kTileS, 8, kWarpgroupM, kWarpgroupN, kSwizzleX, kSwizzleW, kSwizzleY>(
        y_ptr, x_ptr, gate_w_ptr, up_w_ptr, seqlens_ptr, cu_seqlens_ptr,
        xscale_ptr, gate_wscale_ptr, up_wscale_ptr, tmas_ptr, tiles_ptr,
        cu_tiles_ptr, num_group, m, n, k, m_pad, num_block_k_pad4,
        update_tma, stream);
  } else {
    launch_fp8_blockwise_fused_s1<64, kTileN, kTileK, kTileS, 8, kWarpgroupM, kWarpgroupN, kSwizzleX, kSwizzleW, kSwizzleY>(
        y_ptr, x_ptr, gate_w_ptr, up_w_ptr, seqlens_ptr, cu_seqlens_ptr,
        xscale_ptr, gate_wscale_ptr, up_wscale_ptr, tmas_ptr, tiles_ptr,
        cu_tiles_ptr, num_group, m, n, k, m_pad, num_block_k_pad4,
        update_tma, stream);
  }
}

// Opt-in occupancy probe. Production fused S1 dispatch above is unchanged.
void fp8_blockwise_fused_s1_high_occ_async(
    void *y_ptr, const void *x_ptr,
    const void *gate_w_ptr, const void *up_w_ptr,
    const void *seqlens_ptr, const void *cu_seqlens_ptr,
    const void *xscale_ptr,
    const void *gate_wscale_ptr, const void *up_wscale_ptr,
    void *tmas_ptr, void *tiles_ptr, void *cu_tiles_ptr,
    int num_group, int m, int n, int k, int m_pad,
    int num_block_k_pad4, int num_seq_per_group_avg,
    bool update_tma, cudaStream_t stream) {
  constexpr int kTileN = 64, kTileK = 128, kTileS = 64, kStage = 4;
  constexpr int kWarpgroupM = 1, kWarpgroupN = 1;
  constexpr int kSwizzleX = 128, kSwizzleW = 128, kSwizzleY = 64;
  constexpr int kBlockThreads = 256, kGridMultiplier = 2;

  if (num_seq_per_group_avg <= 16) {
    launch_fp8_blockwise_fused_s1<
        16, kTileN, kTileK, kTileS, kStage, kWarpgroupM, kWarpgroupN,
        kSwizzleX, kSwizzleW, kSwizzleY, kBlockThreads, kGridMultiplier>(
        y_ptr, x_ptr, gate_w_ptr, up_w_ptr, seqlens_ptr, cu_seqlens_ptr,
        xscale_ptr, gate_wscale_ptr, up_wscale_ptr, tmas_ptr, tiles_ptr,
        cu_tiles_ptr, num_group, m, n, k, m_pad, num_block_k_pad4,
        update_tma, stream);
  } else if (num_seq_per_group_avg <= 32) {
    launch_fp8_blockwise_fused_s1<
        32, kTileN, kTileK, kTileS, kStage, kWarpgroupM, kWarpgroupN,
        kSwizzleX, kSwizzleW, kSwizzleY, kBlockThreads, kGridMultiplier>(
        y_ptr, x_ptr, gate_w_ptr, up_w_ptr, seqlens_ptr, cu_seqlens_ptr,
        xscale_ptr, gate_wscale_ptr, up_wscale_ptr, tmas_ptr, tiles_ptr,
        cu_tiles_ptr, num_group, m, n, k, m_pad, num_block_k_pad4,
        update_tma, stream);
  } else {
    launch_fp8_blockwise_fused_s1<
        64, kTileN, kTileK, kTileS, kStage, kWarpgroupM, kWarpgroupN,
        kSwizzleX, kSwizzleW, kSwizzleY, kBlockThreads, kGridMultiplier>(
        y_ptr, x_ptr, gate_w_ptr, up_w_ptr, seqlens_ptr, cu_seqlens_ptr,
        xscale_ptr, gate_wscale_ptr, up_wscale_ptr, tmas_ptr, tiles_ptr,
        cu_tiles_ptr, num_group, m, n, k, m_pad, num_block_k_pad4,
        update_tma, stream);
  }
}

// ============================================================================
// Fused S1 PyTorch entry point
// ============================================================================
torch::Tensor fp8_blockwise_fused_s1(
    const torch::Tensor &x,
    const torch::Tensor &gate_weight, const torch::Tensor &up_weight,
    const torch::Tensor &seqlens, const torch::Tensor &cu_seqlens,
    const torch::Tensor &x_scale,
    const torch::Tensor &gate_w_scale, const torch::Tensor &up_w_scale,
    const int64_t num_seq_per_group_avg,
    std::optional<torch::Tensor> output) {
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
  TORCH_CHECK(x.device().is_cuda(), "x must be on CUDA");
  TORCH_CHECK(gate_weight.is_contiguous() && up_weight.is_contiguous(),
              "gate/up weights must be contiguous");
  TORCH_CHECK(gate_weight.sizes() == up_weight.sizes(),
              "gate and up weight shape mismatch");

  int m = x.size(0);
  int k = x.size(1);
  int n = gate_weight.size(1);
  int m_pad = x_scale.size(1);
  int num_block_k_pad4 = gate_w_scale.size(2);
  int num_group = seqlens.size(0);

  auto options = x.options();
  torch::Tensor y;
  if (output.has_value()) {
    y = output.value();
  } else {
    y = torch::empty({m, n}, options.dtype(torch::kBFloat16));
  }

  torch::Tensor tmas = torch::empty({num_group * 2, 128}, options);
  torch::Tensor tiles = torch::empty({num_group}, options.dtype(torch::kInt32));
  torch::Tensor cu_tiles = torch::empty({num_group + 1}, options.dtype(torch::kInt32));

  fp8_blockwise_fused_s1_async(
      y.mutable_data_ptr(), x.const_data_ptr(),
      gate_weight.const_data_ptr(), up_weight.const_data_ptr(),
      seqlens.const_data_ptr(), cu_seqlens.const_data_ptr(),
      x_scale.const_data_ptr(),
      gate_w_scale.const_data_ptr(), up_w_scale.const_data_ptr(),
      tmas.mutable_data_ptr(), tiles.mutable_data_ptr(), cu_tiles.mutable_data_ptr(),
      num_group, m, n, k, m_pad, num_block_k_pad4,
      num_seq_per_group_avg, true, stream);

  return y;
}

torch::Tensor fp8_blockwise_fused_s1_high_occ(
    const torch::Tensor &x,
    const torch::Tensor &gate_weight, const torch::Tensor &up_weight,
    const torch::Tensor &seqlens, const torch::Tensor &cu_seqlens,
    const torch::Tensor &x_scale,
    const torch::Tensor &gate_w_scale, const torch::Tensor &up_w_scale,
    const int64_t num_seq_per_group_avg,
    std::optional<torch::Tensor> output) {
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
  TORCH_CHECK(x.device().is_cuda(), "x must be on CUDA");
  TORCH_CHECK(gate_weight.is_contiguous() && up_weight.is_contiguous(),
              "gate/up weights must be contiguous");
  TORCH_CHECK(gate_weight.sizes() == up_weight.sizes(),
              "gate and up weight shape mismatch");

  int m = x.size(0);
  int k = x.size(1);
  int n = gate_weight.size(1);
  int m_pad = x_scale.size(1);
  int num_block_k_pad4 = gate_w_scale.size(2);
  int num_group = seqlens.size(0);
  TORCH_CHECK(n % 128 == 0, "weight N-dim must be a multiple of the 128-row scale block");

  auto options = x.options();
  torch::Tensor y;
  if (output.has_value()) {
    y = output.value();
  } else {
    y = torch::empty({m, n}, options.dtype(torch::kBFloat16));
  }

  torch::Tensor tmas = torch::empty({num_group * 2, 128}, options);
  torch::Tensor tiles = torch::empty({num_group}, options.dtype(torch::kInt32));
  torch::Tensor cu_tiles = torch::empty({num_group + 1}, options.dtype(torch::kInt32));

  fp8_blockwise_fused_s1_high_occ_async(
      y.mutable_data_ptr(), x.const_data_ptr(),
      gate_weight.const_data_ptr(), up_weight.const_data_ptr(),
      seqlens.const_data_ptr(), cu_seqlens.const_data_ptr(),
      x_scale.const_data_ptr(),
      gate_w_scale.const_data_ptr(), up_w_scale.const_data_ptr(),
      tmas.mutable_data_ptr(), tiles.mutable_data_ptr(), cu_tiles.mutable_data_ptr(),
      num_group, m, n, k, m_pad, num_block_k_pad4,
      num_seq_per_group_avg, true, stream);

  return y;
}

}  // close namespace moe
}  // close namespace batchgen

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fp8_blockwise_grouped_gemm",
        &batchgen::moe::fp8_blockwise_grouped_gemm,
        "FP8 blockwise grouped GEMM (CuTe persistent 3-WG, adaptive TileM)",
        py::arg("x"), py::arg("weight"), py::arg("seqlens"), py::arg("cu_seqlens"),
        py::arg("x_scale"), py::arg("w_scale"), py::arg("num_seq_per_group_avg"),
        py::arg("output") = py::none(), py::arg("tma_desc") = py::none());
  m.def("fp8_blockwise_grouped_gemm_high_occ",
        &batchgen::moe::fp8_blockwise_grouped_gemm_high_occ,
        "Opt-in FP8 grouped GEMM probe (1 math WG + 1 loader WG, TileN64, stage4)",
        py::arg("x"), py::arg("weight"), py::arg("seqlens"), py::arg("cu_seqlens"),
        py::arg("x_scale"), py::arg("w_scale"), py::arg("num_seq_per_group_avg"),
        py::arg("output") = py::none());
  m.def("fp8_blockwise_s3_high_occ",
        &batchgen::moe::fp8_blockwise_grouped_gemm_high_occ,
        "Opt-in FP8 S3 probe (1 math WG + 1 loader WG, TileN64, stage4)",
        py::arg("x"), py::arg("weight"), py::arg("seqlens"), py::arg("cu_seqlens"),
        py::arg("x_scale"), py::arg("w_scale"), py::arg("num_seq_per_group_avg"),
        py::arg("output") = py::none());
  m.def("fp8_blockwise_fused_s1",
        &batchgen::moe::fp8_blockwise_fused_s1,
        "FP8 blockwise fused S1: gate+up+SiLU (CuTe persistent 3-WG, v19)",
        py::arg("x"), py::arg("gate_weight"), py::arg("up_weight"),
        py::arg("seqlens"), py::arg("cu_seqlens"),
        py::arg("x_scale"), py::arg("gate_w_scale"), py::arg("up_w_scale"),
        py::arg("num_seq_per_group_avg"), py::arg("output") = py::none());
  m.def("fp8_blockwise_fused_s1_high_occ",
        &batchgen::moe::fp8_blockwise_fused_s1_high_occ,
        "Opt-in FP8 fused S1 probe (1 math WG + 1 loader WG, TileN64, stage4)",
        py::arg("x"), py::arg("gate_weight"), py::arg("up_weight"),
        py::arg("seqlens"), py::arg("cu_seqlens"),
        py::arg("x_scale"), py::arg("gate_w_scale"), py::arg("up_w_scale"),
        py::arg("num_seq_per_group_avg"), py::arg("output") = py::none());
}
