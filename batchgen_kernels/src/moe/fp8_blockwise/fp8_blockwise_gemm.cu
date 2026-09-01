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
          tma_w, tma_xs, tma_ws, tma_xy, (int *)seqlens_ptr, (const int *)cu_seqlens_ptr,
          (float *)xscale_ptr, (float *)wscale_ptr, (int *)tiles_ptr, (int *)cu_tiles_ptr,
          num_group, m, n, k, m_pad, num_block_n, num_block_k, num_block_k_pad4, flat_divider);
    } else {
      constexpr bool IsLoopH = false;
      auto kernel =
          kernels::fp8_blockwise_grouped_gemm_kernel<decltype(config), decltype(tma_x),
                                                      decltype(tma_w), decltype(tma_y),
                                                      decltype(tma_xs), decltype(tma_ws), IsLoopH>;
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
      kernel<<<grid, block, shm_size, stream>>>(
          tma_w, tma_xs, tma_ws, tma_xy, (int *)seqlens_ptr, (const int *)cu_seqlens_ptr,
          (float *)xscale_ptr, (float *)wscale_ptr, (int *)tiles_ptr, (int *)cu_tiles_ptr,
          num_group, m, n, k, m_pad, num_block_n, num_block_k, num_block_k_pad4, flat_divider);
    }
  }
}

// Streamed-offload variant: expert weights and scales live at independent ring
// slot addresses. The prototype pointers are used only to construct base TMA
// descriptors; update_expert_tma_ptrs patches every live address on device.
template <int kTileM, int kTileN, int kTileK, int kTileS, int kStage,
          int kWarpgroupM, int kWarpgroupN, int kSwizzleX, int kSwizzleW,
          int kSwizzleY>
void launch_fp8_blockwise_gemm_ptrs(
    void *y_ptr, const void *x_ptr, const void *w_prototype_ptr,
    const void *weight_ptrs_ptr, const void *seqlens_ptr,
    const void *cu_seqlens_ptr, const void *xscale_ptr,
    const void *wscale_prototype_ptr, const void *wscale_ptrs_ptr,
    void *tmas_ptr, void *tiles_ptr, void *cu_tiles_ptr, int num_group, int m,
    int n, int k, int m_pad, int num_block_k_pad4, cudaStream_t stream) {
  using namespace cute;  // NOLINT

  using Tin = cute::float_e4m3_t;
  using Tout = cute::bfloat16_t;
  using TS = float;

  int num_block_k = k / kTileK;
  int num_block_n = n / kTileN;

  auto X = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(x_ptr)),
                       make_shape(m, k), make_stride(k, Int<1>{}));
  auto W = make_tensor(
      make_gmem_ptr(reinterpret_cast<const Tin *>(w_prototype_ptr)),
      make_shape(n, k, Int<1>{}), make_stride(k, Int<1>{}, n * k));
  auto Y = make_tensor(make_gmem_ptr(reinterpret_cast<Tout *>(y_ptr)),
                       make_shape(n, m), make_stride(Int<1>{}, n));
  auto XS = make_tensor(make_gmem_ptr(reinterpret_cast<const TS *>(xscale_ptr)),
                        make_shape(num_block_k, m_pad),
                        make_stride(m_pad, Int<1>{}));
  auto WS = make_tensor(
      make_gmem_ptr(reinterpret_cast<const TS *>(wscale_prototype_ptr)),
      make_shape(num_block_n, num_block_k_pad4, Int<1>{}),
      make_stride(num_block_k_pad4, Int<1>{},
                  num_block_n * num_block_k_pad4));

  using Config = Fp8BlockwiseGemmConfig<
      Tin, Tout, TS, kTileM, kTileN, kTileK, kTileS, kStage, kWarpgroupM,
      kWarpgroupN, kSwizzleX, kSwizzleW, kSwizzleY>;
  Config config;
  auto [tma_x, tma_w, tma_y, tma_xs, tma_ws] =
      config.get_tma(X, W, Y, XS, WS);

  auto *tma_xyww = static_cast<cute::TmaDescriptor *>(tmas_ptr);
  vec_t<cute::TmaDescriptor, 4> td_xyww{
      *tma_x.get_tma_descriptor(), *tma_y.get_tma_descriptor(),
      *tma_w.get_tma_descriptor(), *tma_ws.get_tma_descriptor()};
  constexpr int kGroupPerThread = 8;
  constexpr int kThreadPerBlock = 32;
  kernels::update_expert_tma_ptrs<
      Tin, Tout, TS, decltype(tma_x), decltype(tma_y), decltype(tma_w),
      decltype(tma_ws), kTileM, kGroupPerThread, kThreadPerBlock>
      <<<num_group + 1, kThreadPerBlock, 0, stream>>>(
          td_xyww, tma_xyww, reinterpret_cast<const Tin *>(x_ptr),
          reinterpret_cast<const Tout *>(y_ptr),
          reinterpret_cast<const int64_t *>(weight_ptrs_ptr),
          reinterpret_cast<const int64_t *>(wscale_ptrs_ptr),
          reinterpret_cast<const int *>(seqlens_ptr),
          reinterpret_cast<const int *>(cu_seqlens_ptr),
          reinterpret_cast<int *>(tiles_ptr),
          reinterpret_cast<int *>(cu_tiles_ptr), num_group, m, n, k,
          num_block_n, num_block_k_pad4);

  int num_tile_n = (n + kTileN - 1) / kTileN;
  cutlass::FastDivmod flat_divider(num_tile_n);
  dim3 block(384);
  dim3 grid(get_sm_count());
  int shm_seq = sizeof(int) * (num_group + 1);
  int shm_size = config.get_shm_size() + shm_seq;

  if (k <= 1024 || n <= 1024) {
    constexpr bool IsLoopH = true;
    constexpr bool UsePointerWeights = true;
    auto kernel = kernels::fp8_blockwise_grouped_gemm_kernel<
        decltype(config), decltype(tma_x), decltype(tma_w), decltype(tma_y),
        decltype(tma_xs), decltype(tma_ws), IsLoopH, UsePointerWeights>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         shm_size);
    kernel<<<grid, block, shm_size, stream>>>(
        tma_w, tma_xs, tma_ws, tma_xyww,
        reinterpret_cast<int *>(const_cast<void *>(seqlens_ptr)),
        reinterpret_cast<const int *>(cu_seqlens_ptr),
        reinterpret_cast<float *>(const_cast<void *>(xscale_ptr)),
        reinterpret_cast<float *>(const_cast<void *>(wscale_prototype_ptr)),
        reinterpret_cast<int *>(tiles_ptr), reinterpret_cast<int *>(cu_tiles_ptr),
        num_group, m, n, k, m_pad, num_block_n, num_block_k,
        num_block_k_pad4, flat_divider);
  } else {
    constexpr bool IsLoopH = false;
    constexpr bool UsePointerWeights = true;
    auto kernel = kernels::fp8_blockwise_grouped_gemm_kernel<
        decltype(config), decltype(tma_x), decltype(tma_w), decltype(tma_y),
        decltype(tma_xs), decltype(tma_ws), IsLoopH, UsePointerWeights>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         shm_size);
    kernel<<<grid, block, shm_size, stream>>>(
        tma_w, tma_xs, tma_ws, tma_xyww,
        reinterpret_cast<int *>(const_cast<void *>(seqlens_ptr)),
        reinterpret_cast<const int *>(cu_seqlens_ptr),
        reinterpret_cast<float *>(const_cast<void *>(xscale_ptr)),
        reinterpret_cast<float *>(const_cast<void *>(wscale_prototype_ptr)),
        reinterpret_cast<int *>(tiles_ptr), reinterpret_cast<int *>(cu_tiles_ptr),
        num_group, m, n, k, m_pad, num_block_n, num_block_k,
        num_block_k_pad4, flat_divider);
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

void fp8_blockwise_grouped_gemm_ptrs_async(
    void *y_ptr, const void *x_ptr, const void *w_prototype_ptr,
    const void *weight_ptrs_ptr, const void *seqlens_ptr,
    const void *cu_seqlens_ptr, const void *xscale_ptr,
    const void *wscale_prototype_ptr, const void *wscale_ptrs_ptr,
    void *tmas_ptr, void *tiles_ptr, void *cu_tiles_ptr, int num_group, int m,
    int n, int k, int m_pad, int num_block_k_pad4,
    int num_seq_per_group_avg, cudaStream_t stream) {
  constexpr int kTileN = 128;
  constexpr int kTileK = 128;
  constexpr int kTileS = 64;
  constexpr int kWarpgroupM = 2;
  constexpr int kWarpgroupN = 1;
  constexpr int kSwizzleX = 128;
  constexpr int kSwizzleW = 128;
  constexpr int kSwizzleY = 64;

  if (num_seq_per_group_avg <= 16) {
    launch_fp8_blockwise_gemm_ptrs<16, kTileN, kTileK, kTileS, 8,
        kWarpgroupM, kWarpgroupN, kSwizzleX, kSwizzleW, kSwizzleY>(
        y_ptr, x_ptr, w_prototype_ptr, weight_ptrs_ptr, seqlens_ptr,
        cu_seqlens_ptr, xscale_ptr, wscale_prototype_ptr, wscale_ptrs_ptr,
        tmas_ptr, tiles_ptr, cu_tiles_ptr, num_group, m, n, k, m_pad,
        num_block_k_pad4, stream);
  } else if (num_seq_per_group_avg <= 32) {
    launch_fp8_blockwise_gemm_ptrs<32, kTileN, kTileK, kTileS, 8,
        kWarpgroupM, kWarpgroupN, kSwizzleX, kSwizzleW, kSwizzleY>(
        y_ptr, x_ptr, w_prototype_ptr, weight_ptrs_ptr, seqlens_ptr,
        cu_seqlens_ptr, xscale_ptr, wscale_prototype_ptr, wscale_ptrs_ptr,
        tmas_ptr, tiles_ptr, cu_tiles_ptr, num_group, m, n, k, m_pad,
        num_block_k_pad4, stream);
  } else {
    launch_fp8_blockwise_gemm_ptrs<64, kTileN, kTileK, kTileS, 8,
        kWarpgroupM, kWarpgroupN, kSwizzleX, kSwizzleW, kSwizzleY>(
        y_ptr, x_ptr, w_prototype_ptr, weight_ptrs_ptr, seqlens_ptr,
        cu_seqlens_ptr, xscale_ptr, wscale_prototype_ptr, wscale_ptrs_ptr,
        tmas_ptr, tiles_ptr, cu_tiles_ptr, num_group, m, n, k, m_pad,
        num_block_k_pad4, stream);
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

torch::Tensor fp8_blockwise_grouped_gemm_ptrs(
    const torch::Tensor &x, const torch::Tensor &weight_prototype,
    const torch::Tensor &weight_ptrs, const torch::Tensor &seqlens,
    const torch::Tensor &cu_seqlens, const torch::Tensor &x_scale,
    const torch::Tensor &w_scale_prototype, const torch::Tensor &w_scale_ptrs,
    const int64_t num_seq_per_group_avg,
    std::optional<torch::Tensor> output) {
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
  TORCH_CHECK(x.device().is_cuda(), "x must be on CUDA");
  TORCH_CHECK(weight_prototype.device().is_cuda(),
              "weight_prototype must be on CUDA");
  TORCH_CHECK(weight_ptrs.device().is_cuda() &&
                  weight_ptrs.scalar_type() == torch::kInt64,
              "weight_ptrs must be CUDA int64");
  TORCH_CHECK(w_scale_ptrs.device().is_cuda() &&
                  w_scale_ptrs.scalar_type() == torch::kInt64,
              "w_scale_ptrs must be CUDA int64");
  TORCH_CHECK(x.is_contiguous() && weight_prototype.is_contiguous() &&
                  w_scale_prototype.is_contiguous(),
              "x and prototype tensors must be contiguous");
  TORCH_CHECK(weight_prototype.dim() == 2,
              "weight_prototype must be [N,K]");
  TORCH_CHECK(w_scale_prototype.dim() == 2,
              "w_scale_prototype must be [N/128,K/128_pad4]");

  int m = x.size(0);
  int k = x.size(1);
  int n = weight_prototype.size(0);
  int m_pad = x_scale.size(1);
  int num_block_k_pad4 = w_scale_prototype.size(1);
  int num_group = seqlens.size(0);
  TORCH_CHECK(weight_prototype.size(1) == k,
              "x and weight_prototype K mismatch");
  TORCH_CHECK(weight_ptrs.numel() == num_group &&
                  w_scale_ptrs.numel() == num_group,
              "pointer arrays and seqlens num_group mismatch");
  TORCH_CHECK(w_scale_prototype.size(0) == n / 128,
              "weight scale N-block dimension mismatch");
  TORCH_CHECK(num_block_k_pad4 % 4 == 0,
              "w_scale K-block dimension must be padded to a multiple of 4");

  torch::Tensor y = output.has_value()
                        ? output.value()
                        : torch::empty({m, n}, x.options().dtype(torch::kBFloat16));
  torch::Tensor tmas = torch::empty({num_group * 4, 128}, x.options());
  torch::Tensor tiles =
      torch::empty({num_group}, x.options().dtype(torch::kInt32));
  torch::Tensor cu_tiles =
      torch::empty({num_group + 1}, x.options().dtype(torch::kInt32));

  fp8_blockwise_grouped_gemm_ptrs_async(
      y.mutable_data_ptr(), x.const_data_ptr(),
      weight_prototype.const_data_ptr(), weight_ptrs.const_data_ptr(),
      seqlens.const_data_ptr(), cu_seqlens.const_data_ptr(),
      x_scale.const_data_ptr(), w_scale_prototype.const_data_ptr(),
      w_scale_ptrs.const_data_ptr(), tmas.mutable_data_ptr(),
      tiles.mutable_data_ptr(), cu_tiles.mutable_data_ptr(), num_group, m, n, k,
      m_pad, num_block_k_pad4, num_seq_per_group_avg, stream);
  return y;
}

// ============================================================================
// Fused S1 Launcher: gate GEMM + up GEMM + SiLU in single kernel
// ============================================================================
template <int kTileM, int kTileN, int kTileK, int kTileS, int kStage,
          int kWarpgroupM, int kWarpgroupN,
          int kSwizzleX, int kSwizzleW, int kSwizzleY>
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

  int num_block_k = k / kTileK;
  int num_block_n = n / kTileN;

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

  using Config = Fp8BlockwiseGemmConfig<Tin, Tout, TS, kTileM, kTileN, kTileK, kTileS, kStage,
                                         kWarpgroupM, kWarpgroupN, kSwizzleX, kSwizzleW, kSwizzleY>;
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
    dim3 block(384);
    dim3 grid(get_sm_count());
    int shm_seq = sizeof(int) * (num_group + 1);
    int shm_size = config.get_shm_size() + shm_seq;

    if (k <= 1024 || n <= 1024) {
      constexpr bool IsLoopH = true;
      auto kernel = kernels::fp8_blockwise_fused_s1_kernel<
          decltype(config), decltype(tma_x), decltype(tma_w_gate), decltype(tma_y),
          decltype(tma_xs), decltype(tma_ws_gate), IsLoopH>;
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
          decltype(tma_xs), decltype(tma_ws_gate), IsLoopH>;
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

template <int kTileM, int kTileN, int kTileK, int kTileS, int kStage,
          int kWarpgroupM, int kWarpgroupN, int kSwizzleX, int kSwizzleW,
          int kSwizzleY>
void launch_fp8_blockwise_fused_s1_ptrs(
    void *y_ptr, const void *x_ptr, const void *gate_w_prototype_ptr,
    const void *gate_weight_ptrs_ptr, const void *up_w_prototype_ptr,
    const void *up_weight_ptrs_ptr, const void *seqlens_ptr,
    const void *cu_seqlens_ptr, const void *xscale_ptr,
    const void *gate_wscale_prototype_ptr, const void *gate_scale_ptrs_ptr,
    const void *up_wscale_prototype_ptr, const void *up_scale_ptrs_ptr,
    void *tmas_ptr, void *tiles_ptr, void *cu_tiles_ptr, int num_group, int m,
    int n, int k, int m_pad, int num_block_k_pad4, cudaStream_t stream) {
  using namespace cute;  // NOLINT
  using Tin = cute::float_e4m3_t;
  using Tout = cute::bfloat16_t;
  using TS = float;

  int num_block_k = k / kTileK;
  int num_block_n = n / kTileN;
  auto X = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(x_ptr)),
                       make_shape(m, k), make_stride(k, Int<1>{}));
  auto W_gate = make_tensor(
      make_gmem_ptr(reinterpret_cast<const Tin *>(gate_w_prototype_ptr)),
      make_shape(n, k, Int<1>{}), make_stride(k, Int<1>{}, n * k));
  auto W_up = make_tensor(
      make_gmem_ptr(reinterpret_cast<const Tin *>(up_w_prototype_ptr)),
      make_shape(n, k, Int<1>{}), make_stride(k, Int<1>{}, n * k));
  auto Y = make_tensor(make_gmem_ptr(reinterpret_cast<Tout *>(y_ptr)),
                       make_shape(n, m), make_stride(Int<1>{}, n));
  auto XS = make_tensor(make_gmem_ptr(reinterpret_cast<const TS *>(xscale_ptr)),
                        make_shape(num_block_k, m_pad),
                        make_stride(m_pad, Int<1>{}));
  auto WS_gate = make_tensor(
      make_gmem_ptr(reinterpret_cast<const TS *>(gate_wscale_prototype_ptr)),
      make_shape(num_block_n, num_block_k_pad4, Int<1>{}),
      make_stride(num_block_k_pad4, Int<1>{},
                  num_block_n * num_block_k_pad4));
  auto WS_up = make_tensor(
      make_gmem_ptr(reinterpret_cast<const TS *>(up_wscale_prototype_ptr)),
      make_shape(num_block_n, num_block_k_pad4, Int<1>{}),
      make_stride(num_block_k_pad4, Int<1>{},
                  num_block_n * num_block_k_pad4));

  using Config = Fp8BlockwiseGemmConfig<
      Tin, Tout, TS, kTileM, kTileN, kTileK, kTileS, kStage, kWarpgroupM,
      kWarpgroupN, kSwizzleX, kSwizzleW, kSwizzleY>;
  Config config;
  auto [tma_x, tma_w_gate, tma_y, tma_xs, tma_ws_gate] =
      config.get_tma(X, W_gate, Y, XS, WS_gate);
  auto tma_w_up = make_tma_copy(SM90_TMA_LOAD{}, W_up,
                                take<0, 2>(typename Config::SLayoutW{}));
  auto tma_ws_up =
      make_tma_copy(SM90_TMA_LOAD{}, WS_up, typename Config::CopyBoxWS{});

  auto *tma_s1 = static_cast<cute::TmaDescriptor *>(tmas_ptr);
  vec_t<cute::TmaDescriptor, 6> td_s1{
      *tma_x.get_tma_descriptor(),       *tma_y.get_tma_descriptor(),
      *tma_w_gate.get_tma_descriptor(),  *tma_ws_gate.get_tma_descriptor(),
      *tma_w_up.get_tma_descriptor(),    *tma_ws_up.get_tma_descriptor()};
  constexpr int kGroupPerThread = 8;
  constexpr int kThreadPerBlock = 32;
  kernels::update_expert_tma_s1_ptrs<
      Tin, Tout, TS, decltype(tma_x), decltype(tma_y), decltype(tma_w_gate),
      decltype(tma_ws_gate), kTileM, kGroupPerThread, kThreadPerBlock>
      <<<num_group + 1, kThreadPerBlock, 0, stream>>>(
          td_s1, tma_s1, reinterpret_cast<const Tin *>(x_ptr),
          reinterpret_cast<const Tout *>(y_ptr),
          reinterpret_cast<const int64_t *>(gate_weight_ptrs_ptr),
          reinterpret_cast<const int64_t *>(gate_scale_ptrs_ptr),
          reinterpret_cast<const int64_t *>(up_weight_ptrs_ptr),
          reinterpret_cast<const int64_t *>(up_scale_ptrs_ptr),
          reinterpret_cast<const int *>(seqlens_ptr),
          reinterpret_cast<const int *>(cu_seqlens_ptr),
          reinterpret_cast<int *>(tiles_ptr),
          reinterpret_cast<int *>(cu_tiles_ptr), num_group, m, n, k,
          num_block_n, num_block_k_pad4);

  int num_tile_n = (n + kTileN - 1) / kTileN;
  cutlass::FastDivmod flat_divider(num_tile_n);
  dim3 block(384);
  dim3 grid(get_sm_count());
  int shm_seq = sizeof(int) * (num_group + 1);
  int shm_size = config.get_shm_size() + shm_seq;

  if (k <= 1024 || n <= 1024) {
    constexpr bool IsLoopH = true;
    constexpr bool UsePointerWeights = true;
    auto kernel = kernels::fp8_blockwise_fused_s1_kernel<
        decltype(config), decltype(tma_x), decltype(tma_w_gate),
        decltype(tma_y), decltype(tma_xs), decltype(tma_ws_gate), IsLoopH,
        UsePointerWeights>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         shm_size);
    kernel<<<grid, block, shm_size, stream>>>(
        tma_w_gate, tma_w_up, tma_xs, tma_ws_gate, tma_ws_up, tma_s1,
        reinterpret_cast<int *>(const_cast<void *>(seqlens_ptr)),
        reinterpret_cast<const int *>(cu_seqlens_ptr),
        reinterpret_cast<float *>(const_cast<void *>(xscale_ptr)),
        reinterpret_cast<int *>(tiles_ptr), reinterpret_cast<int *>(cu_tiles_ptr),
        num_group, m, n, k, m_pad, num_block_n, num_block_k,
        num_block_k_pad4, flat_divider);
  } else {
    constexpr bool IsLoopH = false;
    constexpr bool UsePointerWeights = true;
    auto kernel = kernels::fp8_blockwise_fused_s1_kernel<
        decltype(config), decltype(tma_x), decltype(tma_w_gate),
        decltype(tma_y), decltype(tma_xs), decltype(tma_ws_gate), IsLoopH,
        UsePointerWeights>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         shm_size);
    kernel<<<grid, block, shm_size, stream>>>(
        tma_w_gate, tma_w_up, tma_xs, tma_ws_gate, tma_ws_up, tma_s1,
        reinterpret_cast<int *>(const_cast<void *>(seqlens_ptr)),
        reinterpret_cast<const int *>(cu_seqlens_ptr),
        reinterpret_cast<float *>(const_cast<void *>(xscale_ptr)),
        reinterpret_cast<int *>(tiles_ptr), reinterpret_cast<int *>(cu_tiles_ptr),
        num_group, m, n, k, m_pad, num_block_n, num_block_k,
        num_block_k_pad4, flat_divider);
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

void fp8_blockwise_fused_s1_ptrs_async(
    void *y_ptr, const void *x_ptr, const void *gate_w_prototype_ptr,
    const void *gate_weight_ptrs_ptr, const void *up_w_prototype_ptr,
    const void *up_weight_ptrs_ptr, const void *seqlens_ptr,
    const void *cu_seqlens_ptr, const void *xscale_ptr,
    const void *gate_wscale_prototype_ptr, const void *gate_scale_ptrs_ptr,
    const void *up_wscale_prototype_ptr, const void *up_scale_ptrs_ptr,
    void *tmas_ptr, void *tiles_ptr, void *cu_tiles_ptr, int num_group, int m,
    int n, int k, int m_pad, int num_block_k_pad4,
    int num_seq_per_group_avg, cudaStream_t stream) {
  constexpr int kTileN = 128, kTileK = 128, kTileS = 64;
  constexpr int kWarpgroupM = 2, kWarpgroupN = 1;
  constexpr int kSwizzleX = 128, kSwizzleW = 128, kSwizzleY = 64;
#define LAUNCH_PTR_S1(TM)                                                       \
  launch_fp8_blockwise_fused_s1_ptrs<TM, kTileN, kTileK, kTileS, 8,            \
      kWarpgroupM, kWarpgroupN, kSwizzleX, kSwizzleW, kSwizzleY>(              \
      y_ptr, x_ptr, gate_w_prototype_ptr, gate_weight_ptrs_ptr,                 \
      up_w_prototype_ptr, up_weight_ptrs_ptr, seqlens_ptr, cu_seqlens_ptr,      \
      xscale_ptr, gate_wscale_prototype_ptr, gate_scale_ptrs_ptr,               \
      up_wscale_prototype_ptr, up_scale_ptrs_ptr, tmas_ptr, tiles_ptr,          \
      cu_tiles_ptr, num_group, m, n, k, m_pad, num_block_k_pad4, stream)
  if (num_seq_per_group_avg <= 16) {
    LAUNCH_PTR_S1(16);
  } else if (num_seq_per_group_avg <= 32) {
    LAUNCH_PTR_S1(32);
  } else {
    LAUNCH_PTR_S1(64);
  }
#undef LAUNCH_PTR_S1
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

torch::Tensor fp8_blockwise_fused_s1_ptrs(
    const torch::Tensor &x, const torch::Tensor &gate_weight_prototype,
    const torch::Tensor &gate_weight_ptrs,
    const torch::Tensor &up_weight_prototype,
    const torch::Tensor &up_weight_ptrs, const torch::Tensor &seqlens,
    const torch::Tensor &cu_seqlens, const torch::Tensor &x_scale,
    const torch::Tensor &gate_w_scale_prototype,
    const torch::Tensor &gate_w_scale_ptrs,
    const torch::Tensor &up_w_scale_prototype,
    const torch::Tensor &up_w_scale_ptrs,
    const int64_t num_seq_per_group_avg,
    std::optional<torch::Tensor> output) {
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
  TORCH_CHECK(x.device().is_cuda(), "x must be on CUDA");
  TORCH_CHECK(gate_weight_prototype.device().is_cuda() &&
                  up_weight_prototype.device().is_cuda(),
              "gate/up weight prototypes must be on CUDA");
  TORCH_CHECK(gate_weight_prototype.is_contiguous() &&
                  up_weight_prototype.is_contiguous() &&
                  gate_w_scale_prototype.is_contiguous() &&
                  up_w_scale_prototype.is_contiguous(),
              "prototype tensors must be contiguous");
  TORCH_CHECK(gate_weight_prototype.sizes() == up_weight_prototype.sizes(),
              "gate/up weight prototype shape mismatch");
  TORCH_CHECK(gate_w_scale_prototype.sizes() ==
                  up_w_scale_prototype.sizes(),
              "gate/up scale prototype shape mismatch");
  for (const auto *ptrs : {&gate_weight_ptrs, &up_weight_ptrs,
                           &gate_w_scale_ptrs, &up_w_scale_ptrs}) {
    TORCH_CHECK(ptrs->device().is_cuda() &&
                    ptrs->scalar_type() == torch::kInt64,
                "all S1 pointer arrays must be CUDA int64");
  }

  int m = x.size(0);
  int k = x.size(1);
  int n = gate_weight_prototype.size(0);
  int m_pad = x_scale.size(1);
  int num_block_k_pad4 = gate_w_scale_prototype.size(1);
  int num_group = seqlens.size(0);
  TORCH_CHECK(gate_weight_prototype.dim() == 2 &&
                  gate_weight_prototype.size(1) == k,
              "gate/up weight prototypes must be [N,K]");
  TORCH_CHECK(gate_w_scale_prototype.dim() == 2 &&
                  gate_w_scale_prototype.size(0) == n / 128 &&
                  num_block_k_pad4 % 4 == 0,
              "gate/up scale prototype shape mismatch");
  TORCH_CHECK(gate_weight_ptrs.numel() == num_group &&
                  up_weight_ptrs.numel() == num_group &&
                  gate_w_scale_ptrs.numel() == num_group &&
                  up_w_scale_ptrs.numel() == num_group,
              "S1 pointer arrays and seqlens num_group mismatch");

  torch::Tensor y = output.has_value()
                        ? output.value()
                        : torch::empty({m, n}, x.options().dtype(torch::kBFloat16));
  torch::Tensor tmas = torch::empty({num_group * 6, 128}, x.options());
  torch::Tensor tiles =
      torch::empty({num_group}, x.options().dtype(torch::kInt32));
  torch::Tensor cu_tiles =
      torch::empty({num_group + 1}, x.options().dtype(torch::kInt32));
  fp8_blockwise_fused_s1_ptrs_async(
      y.mutable_data_ptr(), x.const_data_ptr(),
      gate_weight_prototype.const_data_ptr(), gate_weight_ptrs.const_data_ptr(),
      up_weight_prototype.const_data_ptr(), up_weight_ptrs.const_data_ptr(),
      seqlens.const_data_ptr(), cu_seqlens.const_data_ptr(),
      x_scale.const_data_ptr(), gate_w_scale_prototype.const_data_ptr(),
      gate_w_scale_ptrs.const_data_ptr(), up_w_scale_prototype.const_data_ptr(),
      up_w_scale_ptrs.const_data_ptr(), tmas.mutable_data_ptr(),
      tiles.mutable_data_ptr(), cu_tiles.mutable_data_ptr(), num_group, m, n, k,
      m_pad, num_block_k_pad4, num_seq_per_group_avg, stream);
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
  m.def("fp8_blockwise_grouped_gemm_ptrs",
        &batchgen::moe::fp8_blockwise_grouped_gemm_ptrs,
        "FP8 blockwise grouped GEMM over independent weight pointer arrays",
        py::arg("x"), py::arg("weight_prototype"), py::arg("weight_ptrs"),
        py::arg("seqlens"), py::arg("cu_seqlens"), py::arg("x_scale"),
        py::arg("w_scale_prototype"), py::arg("w_scale_ptrs"),
        py::arg("num_seq_per_group_avg"), py::arg("output") = py::none());
  m.def("fp8_blockwise_fused_s1",
        &batchgen::moe::fp8_blockwise_fused_s1,
        "FP8 blockwise fused S1: gate+up+SiLU (CuTe persistent 3-WG, v19)",
        py::arg("x"), py::arg("gate_weight"), py::arg("up_weight"),
        py::arg("seqlens"), py::arg("cu_seqlens"),
        py::arg("x_scale"), py::arg("gate_w_scale"), py::arg("up_w_scale"),
        py::arg("num_seq_per_group_avg"), py::arg("output") = py::none());
  m.def("fp8_blockwise_fused_s1_ptrs",
        &batchgen::moe::fp8_blockwise_fused_s1_ptrs,
        "FP8 fused S1 over independent gate/up weight pointer arrays",
        py::arg("x"), py::arg("gate_weight_prototype"),
        py::arg("gate_weight_ptrs"), py::arg("up_weight_prototype"),
        py::arg("up_weight_ptrs"), py::arg("seqlens"),
        py::arg("cu_seqlens"), py::arg("x_scale"),
        py::arg("gate_w_scale_prototype"), py::arg("gate_w_scale_ptrs"),
        py::arg("up_w_scale_prototype"), py::arg("up_w_scale_ptrs"),
        py::arg("num_seq_per_group_avg"), py::arg("output") = py::none());
}
