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
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cstdint>
#include <limits>
#include <torch/all.h>
#include <torch/extension.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;

namespace batchgen {
namespace moe {

constexpr int kTmaGroupsPerThread = 8;
constexpr int kTmaThreadsPerBlock = 32;
constexpr int kMaxTmaGroups = kTmaGroupsPerThread * kTmaThreadsPerBlock;

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

  // x_scale tiles are addressed per expert as cu_seqlens[e] / kTileM (device
  // side, alignment trapped in the kernel); the column space must tile evenly.
  TORCH_CHECK(m_pad % kTileM == 0, "x_scale columns (m_pad=", m_pad,
              ") must be a multiple of TileM=", kTileM);

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

    kernels::update_expert_tma<Tin, Tout, decltype(tma_x), decltype(tma_y), kTileM,
                                kTmaGroupsPerThread, kTmaThreadsPerBlock>
        <<<num_group + 1, kTmaThreadsPerBlock, 0, stream>>>(
            td_xy, tma_xy, (const Tin *)x_ptr, (const Tout *)y_ptr, (const int *)seqlens_ptr,
            (const int *)cu_seqlens_ptr, (int *)tiles_ptr, (int *)cu_tiles_ptr, num_group, m, n, k);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  // Main kernel
  {
    int num_tile_n = (n + kTileN - 1) / kTileN;
    cutlass::FastDivmod flat_divider(num_tile_n);

    dim3 block(384);
    dim3 grid(get_sm_count());

    // shm_tiles [num_group + 1] followed by shm_xs_tile [num_group].
    int shm_seq = sizeof(int) * (2 * num_group + 1);
    int shm_size = config.get_shm_size() + shm_seq;

    if (k <= 1024 || n <= 1024) {
      constexpr bool IsLoopH = true;
      auto kernel =
          kernels::fp8_blockwise_grouped_gemm_kernel<decltype(config), decltype(tma_x),
                                                      decltype(tma_w), decltype(tma_y),
                                                      decltype(tma_xs), decltype(tma_ws), IsLoopH>;
      C10_CUDA_CHECK(
          cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size));
      kernel<<<grid, block, shm_size, stream>>>(
          tma_w, tma_xs, tma_ws, tma_xy, (int *)seqlens_ptr, (int *)cu_seqlens_ptr,
          (float *)xscale_ptr, (float *)wscale_ptr, (int *)tiles_ptr, (int *)cu_tiles_ptr,
          num_group, m, n, k, m_pad, num_block_n, num_block_k, num_block_k_pad4, flat_divider);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
      constexpr bool IsLoopH = false;
      auto kernel =
          kernels::fp8_blockwise_grouped_gemm_kernel<decltype(config), decltype(tma_x),
                                                      decltype(tma_w), decltype(tma_y),
                                                      decltype(tma_xs), decltype(tma_ws), IsLoopH>;
      C10_CUDA_CHECK(
          cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size));
      kernel<<<grid, block, shm_size, stream>>>(
          tma_w, tma_xs, tma_ws, tma_xy, (int *)seqlens_ptr, (int *)cu_seqlens_ptr,
          (float *)xscale_ptr, (float *)wscale_ptr, (int *)tiles_ptr, (int *)cu_tiles_ptr,
          num_group, m, n, k, m_pad, num_block_n, num_block_k, num_block_k_pad4, flat_divider);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
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
  TORCH_CHECK(m_pad % kTileM == 0, "x_scale columns (m_pad=", m_pad,
              ") must be a multiple of TileM=", kTileM);

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
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  int num_tile_n = (n + kTileN - 1) / kTileN;
  cutlass::FastDivmod flat_divider(num_tile_n);
  dim3 block(384);
  dim3 grid(get_sm_count());
  // shm_tiles [num_group + 1] followed by shm_xs_tile [num_group].
  int shm_seq = sizeof(int) * (2 * num_group + 1);
  int shm_size = config.get_shm_size() + shm_seq;

  if (k <= 1024 || n <= 1024) {
    constexpr bool IsLoopH = true;
    constexpr bool UsePointerWeights = true;
    auto kernel = kernels::fp8_blockwise_grouped_gemm_kernel<
        decltype(config), decltype(tma_x), decltype(tma_w), decltype(tma_y),
        decltype(tma_xs), decltype(tma_ws), IsLoopH, UsePointerWeights>;
    C10_CUDA_CHECK(
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size));
    kernel<<<grid, block, shm_size, stream>>>(
        tma_w, tma_xs, tma_ws, tma_xyww,
        reinterpret_cast<int *>(const_cast<void *>(seqlens_ptr)),
        reinterpret_cast<const int *>(cu_seqlens_ptr),
        reinterpret_cast<float *>(const_cast<void *>(xscale_ptr)),
        reinterpret_cast<float *>(const_cast<void *>(wscale_prototype_ptr)),
        reinterpret_cast<int *>(tiles_ptr), reinterpret_cast<int *>(cu_tiles_ptr),
        num_group, m, n, k, m_pad, num_block_n, num_block_k,
        num_block_k_pad4, flat_divider);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    constexpr bool IsLoopH = false;
    constexpr bool UsePointerWeights = true;
    auto kernel = kernels::fp8_blockwise_grouped_gemm_kernel<
        decltype(config), decltype(tma_x), decltype(tma_w), decltype(tma_y),
        decltype(tma_xs), decltype(tma_ws), IsLoopH, UsePointerWeights>;
    C10_CUDA_CHECK(
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size));
    kernel<<<grid, block, shm_size, stream>>>(
        tma_w, tma_xs, tma_ws, tma_xyww,
        reinterpret_cast<int *>(const_cast<void *>(seqlens_ptr)),
        reinterpret_cast<const int *>(cu_seqlens_ptr),
        reinterpret_cast<float *>(const_cast<void *>(xscale_ptr)),
        reinterpret_cast<float *>(const_cast<void *>(wscale_prototype_ptr)),
        reinterpret_cast<int *>(tiles_ptr), reinterpret_cast<int *>(cu_tiles_ptr),
        num_group, m, n, k, m_pad, num_block_n, num_block_k,
        num_block_k_pad4, flat_divider);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
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
// Host boundary checks shared by the grouped GEMM and fused S1 entry points.
// Only host metadata is inspected: device seqlens/cu_seqlens values are never
// dereferenced here; their layout is validated inside the kernels.
// ============================================================================
namespace {

constexpr int64_t kInt32Max = std::numeric_limits<int32_t>::max();
constexpr int64_t kQuantBlock = 128;  // TileN == TileK == quantization block

void check_cuda_tensor(const torch::Tensor &t, const torch::Device &device, const char *name) {
  TORCH_CHECK(t.defined() && t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(t.device() == device, name, " must be on ", device, ", got ", t.device());
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

struct GroupedGemmGeometry {
  int m, n, k, m_pad, num_block_k_pad4, num_group;
};

GroupedGemmGeometry check_grouped_gemm_inputs(const torch::Tensor &x, const torch::Tensor &weight,
                                              const torch::Tensor &seqlens,
                                              const torch::Tensor &cu_seqlens,
                                              const torch::Tensor &x_scale,
                                              const torch::Tensor &w_scale,
                                              const std::optional<torch::Tensor> &output) {
  TORCH_CHECK(x.defined() && x.is_cuda(), "x must be a CUDA tensor");
  const auto device = x.device();
  check_cuda_tensor(x, device, "x");
  check_cuda_tensor(weight, device, "weight");
  check_cuda_tensor(seqlens, device, "seqlens");
  check_cuda_tensor(cu_seqlens, device, "cu_seqlens");
  check_cuda_tensor(x_scale, device, "x_scale");
  check_cuda_tensor(w_scale, device, "w_scale");

  TORCH_CHECK(x.dim() == 2 && x.scalar_type() == c10::ScalarType::Float8_e4m3fn,
              "x must be float8_e4m3fn [m, K]");
  TORCH_CHECK(weight.dim() == 3 && weight.scalar_type() == c10::ScalarType::Float8_e4m3fn,
              "weight must be float8_e4m3fn [G, N, K]");
  TORCH_CHECK(seqlens.dim() == 1 && seqlens.scalar_type() == torch::kInt32,
              "seqlens must be int32 [G]");
  TORCH_CHECK(cu_seqlens.dim() == 1 && cu_seqlens.scalar_type() == torch::kInt32,
              "cu_seqlens must be int32 [G + 1]");
  TORCH_CHECK(x_scale.dim() == 2 && x_scale.scalar_type() == torch::kFloat32,
              "x_scale must be float32 [K/128, m_pad]");
  TORCH_CHECK(w_scale.dim() == 3 && w_scale.scalar_type() == torch::kFloat32,
              "w_scale must be float32 [G, N/128, pad4(K/128)]");

  const int64_t m = x.size(0);
  const int64_t k = x.size(1);
  const int64_t num_group = weight.size(0);
  const int64_t n = weight.size(1);
  TORCH_CHECK(num_group > 0 && num_group <= kMaxTmaGroups,
              "num_group must be in [1, ", kMaxTmaGroups, "], got ", num_group);
  TORCH_CHECK(weight.size(2) == k, "x and weight K mismatch");
  TORCH_CHECK(k > 0 && k % kQuantBlock == 0 && n > 0 && n % kQuantBlock == 0,
              "K (", k, ") and N (", n, ") must be positive multiples of 128");
  TORCH_CHECK(seqlens.size(0) == num_group, "seqlens and weight num_group mismatch");
  TORCH_CHECK(cu_seqlens.size(0) == num_group + 1,
              "cu_seqlens must have num_group + 1 entries, got ", cu_seqlens.size(0));

  const int64_t num_block_k = k / kQuantBlock;
  const int64_t num_block_n = n / kQuantBlock;
  const int64_t num_block_k_pad4 = (num_block_k + 3) / 4 * 4;
  TORCH_CHECK(x_scale.size(0) == num_block_k, "x_scale must have K/128 = ", num_block_k,
              " rows, got ", x_scale.size(0));
  const int64_t m_pad = x_scale.size(1);
  TORCH_CHECK(m > 0 && m_pad > 0 && m_pad <= m, "x_scale columns (m_pad=", m_pad,
              ") must be in [1, x rows=", m, "]");
  TORCH_CHECK(w_scale.size(0) == num_group && w_scale.size(1) == num_block_n &&
                  w_scale.size(2) == num_block_k_pad4,
              "w_scale must be [G, N/128, pad4(K/128)] = [", num_group, ", ", num_block_n, ", ",
              num_block_k_pad4, "]");
  // Values narrowed to int below: m, and the int strides n*k and
  // num_block_n*num_block_k_pad4. Row offsets are computed in int64.
  TORCH_CHECK(m <= kInt32Max && n * k <= kInt32Max &&
                  num_block_n * num_block_k_pad4 <= kInt32Max,
              "grouped GEMM shape products overflow int32");

  if (output.has_value()) {
    const auto &y = output.value();
    check_cuda_tensor(y, device, "output");
    TORCH_CHECK(y.dim() == 2 && y.scalar_type() == torch::kBFloat16 && y.size(0) == m &&
                    y.size(1) == n,
                "output must be BF16 [m, N] = [", m, ", ", n, "]");
  }

  return {static_cast<int>(m),     static_cast<int>(n),
          static_cast<int>(k),     static_cast<int>(m_pad),
          static_cast<int>(num_block_k_pad4), static_cast<int>(num_group)};
}

GroupedGemmGeometry check_grouped_gemm_ptr_inputs(
    const torch::Tensor &x, const torch::Tensor &weight_prototype,
    const torch::Tensor &weight_ptrs, const torch::Tensor &seqlens,
    const torch::Tensor &cu_seqlens, const torch::Tensor &x_scale,
    const torch::Tensor &w_scale_prototype, const torch::Tensor &w_scale_ptrs,
    const std::optional<torch::Tensor> &output) {
  TORCH_CHECK(x.defined() && x.is_cuda(), "x must be a CUDA tensor");
  const auto device = x.device();
  check_cuda_tensor(x, device, "x");
  check_cuda_tensor(weight_prototype, device, "weight_prototype");
  check_cuda_tensor(weight_ptrs, device, "weight_ptrs");
  check_cuda_tensor(seqlens, device, "seqlens");
  check_cuda_tensor(cu_seqlens, device, "cu_seqlens");
  check_cuda_tensor(x_scale, device, "x_scale");
  check_cuda_tensor(w_scale_prototype, device, "w_scale_prototype");
  check_cuda_tensor(w_scale_ptrs, device, "w_scale_ptrs");

  TORCH_CHECK(x.dim() == 2 && x.scalar_type() == c10::ScalarType::Float8_e4m3fn,
              "x must be float8_e4m3fn [m, K]");
  TORCH_CHECK(weight_prototype.dim() == 2 &&
                  weight_prototype.scalar_type() == c10::ScalarType::Float8_e4m3fn,
              "weight_prototype must be float8_e4m3fn [N, K]");
  TORCH_CHECK(weight_ptrs.dim() == 1 && weight_ptrs.scalar_type() == torch::kInt64,
              "weight_ptrs must be int64 [G]");
  TORCH_CHECK(seqlens.dim() == 1 && seqlens.scalar_type() == torch::kInt32,
              "seqlens must be int32 [G]");
  TORCH_CHECK(cu_seqlens.dim() == 1 && cu_seqlens.scalar_type() == torch::kInt32,
              "cu_seqlens must be int32 [G + 1]");
  TORCH_CHECK(x_scale.dim() == 2 && x_scale.scalar_type() == torch::kFloat32,
              "x_scale must be float32 [K/128, m_pad]");
  TORCH_CHECK(w_scale_prototype.dim() == 2 &&
                  w_scale_prototype.scalar_type() == torch::kFloat32,
              "w_scale_prototype must be float32 [N/128, pad4(K/128)]");
  TORCH_CHECK(w_scale_ptrs.dim() == 1 && w_scale_ptrs.scalar_type() == torch::kInt64,
              "w_scale_ptrs must be int64 [G]");

  const int64_t m = x.size(0);
  const int64_t k = x.size(1);
  const int64_t n = weight_prototype.size(0);
  const int64_t num_group = seqlens.size(0);
  TORCH_CHECK(num_group > 0 && num_group <= kMaxTmaGroups,
              "num_group must be in [1, ", kMaxTmaGroups, "], got ", num_group);
  TORCH_CHECK(weight_prototype.size(1) == k, "x and weight_prototype K mismatch");
  TORCH_CHECK(k > 0 && k % kQuantBlock == 0 && n > 0 && n % kQuantBlock == 0,
              "K (", k, ") and N (", n, ") must be positive multiples of 128");
  TORCH_CHECK(weight_ptrs.numel() == num_group &&
                  w_scale_ptrs.numel() == num_group,
              "pointer arrays and seqlens num_group mismatch");
  TORCH_CHECK(cu_seqlens.size(0) == num_group + 1,
              "cu_seqlens must have num_group + 1 entries, got ", cu_seqlens.size(0));

  const int64_t num_block_k = k / kQuantBlock;
  const int64_t num_block_n = n / kQuantBlock;
  const int64_t num_block_k_pad4 = (num_block_k + 3) / 4 * 4;
  TORCH_CHECK(x_scale.size(0) == num_block_k, "x_scale must have K/128 = ",
              num_block_k, " rows, got ", x_scale.size(0));
  const int64_t m_pad = x_scale.size(1);
  TORCH_CHECK(m > 0 && m_pad > 0 && m_pad <= m, "x_scale columns (m_pad=", m_pad,
              ") must be in [1, x rows=", m, "]");
  TORCH_CHECK(w_scale_prototype.size(0) == num_block_n &&
                  w_scale_prototype.size(1) == num_block_k_pad4,
              "w_scale_prototype must be [N/128, pad4(K/128)] = [",
              num_block_n, ", ", num_block_k_pad4, "]");
  TORCH_CHECK(m <= kInt32Max && n * k <= kInt32Max &&
                  num_block_n * num_block_k_pad4 <= kInt32Max,
              "grouped GEMM shape products overflow int32");

  if (output.has_value()) {
    const auto &y = output.value();
    check_cuda_tensor(y, device, "output");
    TORCH_CHECK(y.dim() == 2 && y.scalar_type() == torch::kBFloat16 &&
                    y.size(0) == m && y.size(1) == n,
                "output must be BF16 [m, N] = [", m, ", ", n, "]");
  }

  return {static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
          static_cast<int>(m_pad), static_cast<int>(num_block_k_pad4),
          static_cast<int>(num_group)};
}

torch::Tensor prepare_byte_workspace(
    const std::optional<torch::Tensor> &workspace, const torch::Tensor &like,
    int64_t rows, const char *name) {
  if (!workspace.has_value()) {
    return torch::empty({rows, 128}, like.options());
  }
  const auto &value = workspace.value();
  check_cuda_tensor(value, like.device(), name);
  TORCH_CHECK(value.dim() == 2 && value.element_size() == 1 &&
                  value.size(0) >= rows && value.size(1) == 128,
              name, " must be 1-byte [at least ", rows, ", 128] storage");
  TORCH_CHECK(reinterpret_cast<std::uintptr_t>(value.data_ptr()) % 64 == 0,
              name, " data pointer must be 64-byte aligned");
  return value;
}

torch::Tensor prepare_int_workspace(
    const std::optional<torch::Tensor> &workspace, const torch::Tensor &like,
    int64_t size, const char *name) {
  if (!workspace.has_value()) {
    return torch::empty({size}, like.options().dtype(torch::kInt32));
  }
  const auto &value = workspace.value();
  check_cuda_tensor(value, like.device(), name);
  TORCH_CHECK(value.dim() == 1 && value.scalar_type() == torch::kInt32 &&
                  value.numel() >= size,
              name, " must be int32 [at least ", size, "]");
  return value;
}

// The X/Y descriptor array of one non-pointer grouped GEMM: [2 * G, 128] bytes.
void check_xy_tma_desc(const torch::Tensor &td, const torch::Device &device, int num_group,
                       const char *name) {
  check_cuda_tensor(td, device, name);
  TORCH_CHECK(td.dim() == 2 && td.element_size() == 1 && td.size(0) == 2 * num_group &&
                  td.size(1) == 128,
              name, " must be a 1-byte [2 * num_group, 128] tensor");
  TORCH_CHECK(reinterpret_cast<std::uintptr_t>(td.data_ptr()) % 64 == 0,
              name, " data pointer must be 64-byte aligned");
}

// update_tma = false: the GEMM consumes descriptors and tile tables written
// earlier on this stream (fp8_blockwise_prepare_s1_s3_tma), so all three must
// be the caller's; a fresh allocation would hold no descriptors at all.
void check_prepared_tma(bool update_tma, const std::optional<torch::Tensor> &tma_desc,
                        const std::optional<torch::Tensor> &tiles,
                        const std::optional<torch::Tensor> &cu_tiles, const char *op) {
  if (update_tma) return;
  TORCH_CHECK(tma_desc.has_value() && tiles.has_value() && cu_tiles.has_value(), op,
              ": update_tma=false requires the prepared tma_desc, tiles and cu_tiles");
}

}  // namespace

// ============================================================================
// PyTorch entry point
// ============================================================================
torch::Tensor fp8_blockwise_grouped_gemm(
    const torch::Tensor &x, const torch::Tensor &weight, const torch::Tensor &seqlens,
    const torch::Tensor &cu_seqlens, const torch::Tensor &x_scale, const torch::Tensor &w_scale,
    const int64_t num_seq_per_group_avg, std::optional<torch::Tensor> output,
    std::optional<torch::Tensor> tma_desc, std::optional<torch::Tensor> tiles_workspace,
    std::optional<torch::Tensor> cu_tiles_workspace, bool update_tma) {
  const auto geom =
      check_grouped_gemm_inputs(x, weight, seqlens, cu_seqlens, x_scale, w_scale, output);
  const int m = geom.m;
  const int k = geom.k;
  const int n = geom.n;
  const int m_pad = geom.m_pad;
  const int num_block_k_pad4 = geom.num_block_k_pad4;
  const int num_group = geom.num_group;

  check_prepared_tma(update_tma, tma_desc, tiles_workspace, cu_tiles_workspace,
                     "fp8_blockwise_grouped_gemm");
  if (tma_desc.has_value()) {
    check_xy_tma_desc(tma_desc.value(), x.device(), num_group, "tma_desc");
  }

  const c10::cuda::CUDAGuard device_guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device());

  auto options = x.options();
  torch::Tensor y;
  if (output.has_value()) {
    y = output.value();
  } else {
    y = torch::empty({m, n}, options.dtype(torch::kBFloat16));
  }

  torch::Tensor tmas;
  if (tma_desc.has_value()) {
    tmas = tma_desc.value();
  } else {
    tmas = torch::empty({num_group * 2, 128}, options);
  }

  torch::Tensor tiles = prepare_int_workspace(tiles_workspace, x, num_group, "tiles");
  torch::Tensor cu_tiles = prepare_int_workspace(cu_tiles_workspace, x, num_group + 1, "cu_tiles");

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
    std::optional<torch::Tensor> output,
    std::optional<torch::Tensor> tma_desc,
    std::optional<torch::Tensor> tiles_workspace,
    std::optional<torch::Tensor> cu_tiles_workspace) {
  const auto geom = check_grouped_gemm_ptr_inputs(
      x, weight_prototype, weight_ptrs, seqlens, cu_seqlens, x_scale,
      w_scale_prototype, w_scale_ptrs, output);
  const int m = geom.m;
  const int n = geom.n;
  const int k = geom.k;
  const int m_pad = geom.m_pad;
  const int num_block_k_pad4 = geom.num_block_k_pad4;
  const int num_group = geom.num_group;

  const c10::cuda::CUDAGuard device_guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
  torch::Tensor y = output.has_value()
                        ? output.value()
                        : torch::empty({m, n}, x.options().dtype(torch::kBFloat16));
  torch::Tensor tmas =
      prepare_byte_workspace(tma_desc, x, num_group * 4, "tma_desc");
  torch::Tensor tiles =
      prepare_int_workspace(tiles_workspace, x, num_group, "tiles");
  torch::Tensor cu_tiles =
      prepare_int_workspace(cu_tiles_workspace, x, num_group + 1, "cu_tiles");

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
  TORCH_CHECK(m_pad % kTileM == 0, "x_scale columns (m_pad=", m_pad,
              ") must be a multiple of TileM=", kTileM);
  TORCH_CHECK(m_pad % kTileM == 0, "x_scale columns (m_pad=", m_pad,
              ") must be a multiple of TileM=", kTileM);

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
    kernels::update_expert_tma<Tin, Tout, decltype(tma_x), decltype(tma_y), kTileM,
                                kTmaGroupsPerThread, kTmaThreadsPerBlock>
        <<<num_group + 1, kTmaThreadsPerBlock, 0, stream>>>(
            td_xy, tma_xy, (const Tin *)x_ptr, (const Tout *)y_ptr,
            (const int *)seqlens_ptr, (const int *)cu_seqlens_ptr,
            (int *)tiles_ptr, (int *)cu_tiles_ptr, num_group, m, n, k);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  {
    int num_tile_n = (n + kTileN - 1) / kTileN;
    cutlass::FastDivmod flat_divider(num_tile_n);
    dim3 block(384);
    dim3 grid(get_sm_count());
    // shm_tiles [num_group + 1] followed by shm_xs_tile [num_group].
    int shm_seq = sizeof(int) * (2 * num_group + 1);
    int shm_size = config.get_shm_size() + shm_seq;

    if (k <= 1024 || n <= 1024) {
      constexpr bool IsLoopH = true;
      auto kernel = kernels::fp8_blockwise_fused_s1_kernel<
          decltype(config), decltype(tma_x), decltype(tma_w_gate), decltype(tma_y),
          decltype(tma_xs), decltype(tma_ws_gate), IsLoopH>;
      C10_CUDA_CHECK(
          cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size));
      kernel<<<grid, block, shm_size, stream>>>(
          tma_w_gate, tma_w_up, tma_xs, tma_ws_gate, tma_ws_up,
          tma_xy, (int *)seqlens_ptr, (int *)cu_seqlens_ptr, (float *)xscale_ptr,
          (int *)tiles_ptr, (int *)cu_tiles_ptr,
          num_group, m, n, k, m_pad,
          num_block_n, num_block_k, num_block_k_pad4, flat_divider);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
      constexpr bool IsLoopH = false;
      auto kernel = kernels::fp8_blockwise_fused_s1_kernel<
          decltype(config), decltype(tma_x), decltype(tma_w_gate), decltype(tma_y),
          decltype(tma_xs), decltype(tma_ws_gate), IsLoopH>;
      C10_CUDA_CHECK(
          cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size));
      kernel<<<grid, block, shm_size, stream>>>(
          tma_w_gate, tma_w_up, tma_xs, tma_ws_gate, tma_ws_up,
          tma_xy, (int *)seqlens_ptr, (int *)cu_seqlens_ptr, (float *)xscale_ptr,
          (int *)tiles_ptr, (int *)cu_tiles_ptr,
          num_group, m, n, k, m_pad,
          num_block_n, num_block_k, num_block_k_pad4, flat_divider);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
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
  TORCH_CHECK(m_pad % kTileM == 0, "x_scale columns (m_pad=", m_pad,
              ") must be a multiple of TileM=", kTileM);
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
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  int num_tile_n = (n + kTileN - 1) / kTileN;
  cutlass::FastDivmod flat_divider(num_tile_n);
  dim3 block(384);
  dim3 grid(get_sm_count());
  // shm_tiles [num_group + 1] followed by shm_xs_tile [num_group].
  int shm_seq = sizeof(int) * (2 * num_group + 1);
  int shm_size = config.get_shm_size() + shm_seq;

  if (k <= 1024 || n <= 1024) {
    constexpr bool IsLoopH = true;
    constexpr bool UsePointerWeights = true;
    auto kernel = kernels::fp8_blockwise_fused_s1_kernel<
        decltype(config), decltype(tma_x), decltype(tma_w_gate),
        decltype(tma_y), decltype(tma_xs), decltype(tma_ws_gate), IsLoopH,
        UsePointerWeights>;
    C10_CUDA_CHECK(
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size));
    kernel<<<grid, block, shm_size, stream>>>(
        tma_w_gate, tma_w_up, tma_xs, tma_ws_gate, tma_ws_up, tma_s1,
        reinterpret_cast<int *>(const_cast<void *>(seqlens_ptr)),
        reinterpret_cast<const int *>(cu_seqlens_ptr),
        reinterpret_cast<float *>(const_cast<void *>(xscale_ptr)),
        reinterpret_cast<int *>(tiles_ptr), reinterpret_cast<int *>(cu_tiles_ptr),
        num_group, m, n, k, m_pad, num_block_n, num_block_k,
        num_block_k_pad4, flat_divider);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    constexpr bool IsLoopH = false;
    constexpr bool UsePointerWeights = true;
    auto kernel = kernels::fp8_blockwise_fused_s1_kernel<
        decltype(config), decltype(tma_x), decltype(tma_w_gate),
        decltype(tma_y), decltype(tma_xs), decltype(tma_ws_gate), IsLoopH,
        UsePointerWeights>;
    C10_CUDA_CHECK(
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size));
    kernel<<<grid, block, shm_size, stream>>>(
        tma_w_gate, tma_w_up, tma_xs, tma_ws_gate, tma_ws_up, tma_s1,
        reinterpret_cast<int *>(const_cast<void *>(seqlens_ptr)),
        reinterpret_cast<const int *>(cu_seqlens_ptr),
        reinterpret_cast<float *>(const_cast<void *>(xscale_ptr)),
        reinterpret_cast<int *>(tiles_ptr), reinterpret_cast<int *>(cu_tiles_ptr),
        num_group, m, n, k, m_pad, num_block_n, num_block_k,
        num_block_k_pad4, flat_divider);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
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
// One-launch descriptor preparation for a fused S1 + grouped S3 pair that
// shares seqlens / cu_seqlens / TileM. Writes the per-expert X/Y descriptors of
// BOTH GEMMs plus one tiles / cu_tiles, which the two GEMMs then consume with
// update_tma = false. The Config template arguments are the ones
// fp8_blockwise_fused_s1_async and fp8_blockwise_grouped_gemm_async
// instantiate, so the prototypes are the ones their own launches build.
// ============================================================================
template <int kTileM, int kTileN, int kTileK, int kTileS, int kStage, int kWarpgroupM,
          int kWarpgroupN, int kSwizzleX, int kSwizzleW, int kSwizzleY>
void launch_prepare_s1_s3_tma(const void *x1_ptr, void *y1_ptr, const void *x3_ptr,
                              void *y3_ptr, const void *seqlens_ptr,
                              const void *cu_seqlens_ptr, void *tma_s1_ptr, void *tma_s3_ptr,
                              void *tiles_ptr, void *cu_tiles_ptr, int num_group, int m,
                              int n1, int k1, int n3, int k3, cudaStream_t stream) {
  using namespace cute;  // NOLINT
  using Tin = cute::float_e4m3_t;
  using Tout = cute::bfloat16_t;
  using TS = float;

  using Config = Fp8BlockwiseGemmConfig<Tin, Tout, TS, kTileM, kTileN, kTileK, kTileS, kStage,
                                         kWarpgroupM, kWarpgroupN, kSwizzleX, kSwizzleW,
                                         kSwizzleY>;
  Config config;

  auto X1 = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(x1_ptr)), make_shape(m, k1),
                        make_stride(k1, Int<1>{}));
  auto Y1 = make_tensor(make_gmem_ptr(reinterpret_cast<Tout *>(y1_ptr)), make_shape(n1, m),
                        make_stride(Int<1>{}, n1));
  auto X3 = make_tensor(make_gmem_ptr(reinterpret_cast<const Tin *>(x3_ptr)), make_shape(m, k3),
                        make_stride(k3, Int<1>{}));
  auto Y3 = make_tensor(make_gmem_ptr(reinterpret_cast<Tout *>(y3_ptr)), make_shape(n3, m),
                        make_stride(Int<1>{}, n3));
  auto [tma_x1, tma_y1] = config.get_tma_xy(X1, Y1);
  auto [tma_x3, tma_y3] = config.get_tma_xy(X3, Y3);

  vec_t<cute::TmaDescriptor, 4> td_proto{
      *tma_x1.get_tma_descriptor(),
      *tma_y1.get_tma_descriptor(),
      *tma_x3.get_tma_descriptor(),
      *tma_y3.get_tma_descriptor(),
  };
  kernels::update_expert_tma_s1_s3<Tin, Tout, decltype(tma_x1), decltype(tma_y1),
                                   decltype(tma_x3), decltype(tma_y3), kTileM,
                                   kTmaGroupsPerThread, kTmaThreadsPerBlock>
      <<<num_group + 1, kTmaThreadsPerBlock, 0, stream>>>(
          td_proto, static_cast<cute::TmaDescriptor *>(tma_s1_ptr),
          static_cast<cute::TmaDescriptor *>(tma_s3_ptr), (const Tin *)x1_ptr,
          (const Tout *)y1_ptr, (const Tin *)x3_ptr, (const Tout *)y3_ptr,
          (const int *)seqlens_ptr, (const int *)cu_seqlens_ptr, (int *)tiles_ptr,
          (int *)cu_tiles_ptr, num_group, n1, k1, n3, k3);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void fp8_blockwise_prepare_s1_s3_tma_async(const void *x1_ptr, void *y1_ptr, const void *x3_ptr,
                                           void *y3_ptr, const void *seqlens_ptr,
                                           const void *cu_seqlens_ptr, void *tma_s1_ptr,
                                           void *tma_s3_ptr, void *tiles_ptr, void *cu_tiles_ptr,
                                           int num_group, int m, int n1, int k1, int n3, int k3,
                                           int num_seq_per_group_avg, cudaStream_t stream) {
  constexpr int kTileN = 128, kTileK = 128, kTileS = 64;
  constexpr int kWarpgroupM = 2, kWarpgroupN = 1;
  constexpr int kSwizzleX = 128, kSwizzleW = 128, kSwizzleY = 64;
#define LAUNCH_PREPARE_S1_S3(TM)                                                    \
  launch_prepare_s1_s3_tma<TM, kTileN, kTileK, kTileS, 8, kWarpgroupM, kWarpgroupN, \
                           kSwizzleX, kSwizzleW, kSwizzleY>(                        \
      x1_ptr, y1_ptr, x3_ptr, y3_ptr, seqlens_ptr, cu_seqlens_ptr, tma_s1_ptr,      \
      tma_s3_ptr, tiles_ptr, cu_tiles_ptr, num_group, m, n1, k1, n3, k3, stream)
  // Same TileM selection as fp8_blockwise_fused_s1_async and
  // fp8_blockwise_grouped_gemm_async (avg 33..48 already falls to 64 there).
  if (num_seq_per_group_avg <= 16) {
    LAUNCH_PREPARE_S1_S3(16);
  } else if (num_seq_per_group_avg <= 32) {
    LAUNCH_PREPARE_S1_S3(32);
  } else {
    LAUNCH_PREPARE_S1_S3(64);
  }
#undef LAUNCH_PREPARE_S1_S3
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
    std::optional<torch::Tensor> output,
    std::optional<torch::Tensor> tma_desc,
    std::optional<torch::Tensor> tiles_workspace,
    std::optional<torch::Tensor> cu_tiles_workspace,
    bool update_tma) {
  const auto geom = check_grouped_gemm_inputs(x, gate_weight, seqlens, cu_seqlens, x_scale,
                                              gate_w_scale, output);
  check_cuda_tensor(up_weight, x.device(), "up_weight");
  check_cuda_tensor(up_w_scale, x.device(), "up_w_scale");
  TORCH_CHECK(up_weight.scalar_type() == gate_weight.scalar_type() &&
                  up_weight.sizes() == gate_weight.sizes(),
              "gate and up weight dtype/shape mismatch");
  TORCH_CHECK(up_w_scale.scalar_type() == gate_w_scale.scalar_type() &&
                  up_w_scale.sizes() == gate_w_scale.sizes(),
              "gate and up weight scale dtype/shape mismatch");

  const int m = geom.m;
  const int k = geom.k;
  const int n = geom.n;
  const int m_pad = geom.m_pad;
  const int num_block_k_pad4 = geom.num_block_k_pad4;
  const int num_group = geom.num_group;

  check_prepared_tma(update_tma, tma_desc, tiles_workspace, cu_tiles_workspace,
                     "fp8_blockwise_fused_s1");
  if (tma_desc.has_value()) {
    check_xy_tma_desc(tma_desc.value(), x.device(), num_group, "tma_desc");
  }

  const c10::cuda::CUDAGuard device_guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device());

  auto options = x.options();
  torch::Tensor y;
  if (output.has_value()) {
    y = output.value();
  } else {
    y = torch::empty({m, n}, options.dtype(torch::kBFloat16));
  }

  torch::Tensor tmas =
      tma_desc.has_value() ? tma_desc.value() : torch::empty({num_group * 2, 128}, options);
  torch::Tensor tiles = prepare_int_workspace(tiles_workspace, x, num_group, "tiles");
  torch::Tensor cu_tiles = prepare_int_workspace(cu_tiles_workspace, x, num_group + 1, "cu_tiles");

  fp8_blockwise_fused_s1_async(
      y.mutable_data_ptr(), x.const_data_ptr(),
      gate_weight.const_data_ptr(), up_weight.const_data_ptr(),
      seqlens.const_data_ptr(), cu_seqlens.const_data_ptr(),
      x_scale.const_data_ptr(),
      gate_w_scale.const_data_ptr(), up_w_scale.const_data_ptr(),
      tmas.mutable_data_ptr(), tiles.mutable_data_ptr(), cu_tiles.mutable_data_ptr(),
      num_group, m, n, k, m_pad, num_block_k_pad4,
      num_seq_per_group_avg, update_tma, stream);

  return y;
}

// ============================================================================
// Shared descriptor preparation for a fused S1 + grouped S3 pair (one launch
// instead of one update_expert_tma per GEMM). Requirements the pair must meet,
// checked here on host metadata: one row space m for all four buffers, the
// same seqlens / cu_seqlens, and the same num_seq_per_group_avg (=> TileM) as
// the two GEMM calls that later consume the result with update_tma = false.
// ============================================================================
void fp8_blockwise_prepare_s1_s3_tma(
    const torch::Tensor &s1_x, const torch::Tensor &s1_output,
    const torch::Tensor &s3_x, const torch::Tensor &s3_output,
    const torch::Tensor &seqlens, const torch::Tensor &cu_seqlens,
    const int64_t num_seq_per_group_avg,
    const torch::Tensor &s1_tma_desc, const torch::Tensor &s3_tma_desc,
    const torch::Tensor &tiles, const torch::Tensor &cu_tiles) {
  TORCH_CHECK(s1_x.defined() && s1_x.is_cuda(), "s1_x must be a CUDA tensor");
  const auto device = s1_x.device();
  check_cuda_tensor(s1_x, device, "s1_x");
  check_cuda_tensor(s1_output, device, "s1_output");
  check_cuda_tensor(s3_x, device, "s3_x");
  check_cuda_tensor(s3_output, device, "s3_output");
  check_cuda_tensor(seqlens, device, "seqlens");
  check_cuda_tensor(cu_seqlens, device, "cu_seqlens");

  TORCH_CHECK(s1_x.dim() == 2 && s1_x.scalar_type() == c10::ScalarType::Float8_e4m3fn,
              "s1_x must be float8_e4m3fn [m, K1]");
  TORCH_CHECK(s3_x.dim() == 2 && s3_x.scalar_type() == c10::ScalarType::Float8_e4m3fn,
              "s3_x must be float8_e4m3fn [m, K3]");
  TORCH_CHECK(s1_output.dim() == 2 && s1_output.scalar_type() == torch::kBFloat16,
              "s1_output must be BF16 [m, N1]");
  TORCH_CHECK(s3_output.dim() == 2 && s3_output.scalar_type() == torch::kBFloat16,
              "s3_output must be BF16 [m, N3]");
  TORCH_CHECK(seqlens.dim() == 1 && seqlens.scalar_type() == torch::kInt32,
              "seqlens must be int32 [G]");
  const int64_t num_group = seqlens.size(0);
  TORCH_CHECK(num_group > 0 && num_group <= kMaxTmaGroups,
              "num_group must be in [1, ", kMaxTmaGroups, "], got ", num_group);
  TORCH_CHECK(cu_seqlens.dim() == 1 && cu_seqlens.scalar_type() == torch::kInt32 &&
                  cu_seqlens.size(0) == num_group + 1,
              "cu_seqlens must be int32 [G + 1]");

  const int64_t m = s1_x.size(0);
  const int64_t k1 = s1_x.size(1);
  const int64_t n1 = s1_output.size(1);
  const int64_t k3 = s3_x.size(1);
  const int64_t n3 = s3_output.size(1);
  TORCH_CHECK(m > 0 && s1_output.size(0) == m && s3_x.size(0) == m && s3_output.size(0) == m,
              "s1_x, s1_output, s3_x and s3_output must share the row space m=", m);
  for (int64_t d : {k1, n1, k3, n3}) {
    TORCH_CHECK(d > 0 && d % kQuantBlock == 0,
                "K1/N1/K3/N3 must be positive multiples of 128, got ", k1, "/", n1, "/", k3,
                "/", n3);
  }
  TORCH_CHECK(m <= kInt32Max && n1 * k1 <= kInt32Max && n3 * k3 <= kInt32Max,
              "prepare_s1_s3_tma shape products overflow int32");

  check_xy_tma_desc(s1_tma_desc, device, static_cast<int>(num_group), "s1_tma_desc");
  check_xy_tma_desc(s3_tma_desc, device, static_cast<int>(num_group), "s3_tma_desc");
  const auto s1_lo = reinterpret_cast<std::uintptr_t>(s1_tma_desc.data_ptr());
  const auto s3_lo = reinterpret_cast<std::uintptr_t>(s3_tma_desc.data_ptr());
  const auto desc_bytes = static_cast<std::uintptr_t>(2 * num_group * 128);
  TORCH_CHECK(s1_lo + desc_bytes <= s3_lo || s3_lo + desc_bytes <= s1_lo,
              "s1_tma_desc and s3_tma_desc must not overlap");
  check_cuda_tensor(tiles, device, "tiles");
  check_cuda_tensor(cu_tiles, device, "cu_tiles");
  TORCH_CHECK(tiles.dim() == 1 && tiles.scalar_type() == torch::kInt32 &&
                  tiles.numel() >= num_group,
              "tiles must be int32 [at least ", num_group, "]");
  TORCH_CHECK(cu_tiles.dim() == 1 && cu_tiles.scalar_type() == torch::kInt32 &&
                  cu_tiles.numel() >= num_group + 1,
              "cu_tiles must be int32 [at least ", num_group + 1, "]");

  const c10::cuda::CUDAGuard device_guard(device);
  auto stream = at::cuda::getCurrentCUDAStream(s1_x.get_device());
  fp8_blockwise_prepare_s1_s3_tma_async(
      s1_x.const_data_ptr(), s1_output.mutable_data_ptr(), s3_x.const_data_ptr(),
      s3_output.mutable_data_ptr(), seqlens.const_data_ptr(), cu_seqlens.const_data_ptr(),
      s1_tma_desc.mutable_data_ptr(), s3_tma_desc.mutable_data_ptr(), tiles.mutable_data_ptr(),
      cu_tiles.mutable_data_ptr(), static_cast<int>(num_group), static_cast<int>(m),
      static_cast<int>(n1), static_cast<int>(k1), static_cast<int>(n3), static_cast<int>(k3),
      static_cast<int>(num_seq_per_group_avg), stream);
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
    std::optional<torch::Tensor> output,
    std::optional<torch::Tensor> tma_desc,
    std::optional<torch::Tensor> tiles_workspace,
    std::optional<torch::Tensor> cu_tiles_workspace) {
  const auto geom = check_grouped_gemm_ptr_inputs(
      x, gate_weight_prototype, gate_weight_ptrs, seqlens, cu_seqlens,
      x_scale, gate_w_scale_prototype, gate_w_scale_ptrs, output);
  const auto up_geom = check_grouped_gemm_ptr_inputs(
      x, up_weight_prototype, up_weight_ptrs, seqlens, cu_seqlens,
      x_scale, up_w_scale_prototype, up_w_scale_ptrs, output);
  TORCH_CHECK(up_geom.m == geom.m && up_geom.n == geom.n &&
                  up_geom.k == geom.k && up_geom.m_pad == geom.m_pad &&
                  up_geom.num_block_k_pad4 == geom.num_block_k_pad4 &&
                  up_geom.num_group == geom.num_group,
              "gate/up pointer geometry mismatch");

  const int m = geom.m;
  const int n = geom.n;
  const int k = geom.k;
  const int m_pad = geom.m_pad;
  const int num_block_k_pad4 = geom.num_block_k_pad4;
  const int num_group = geom.num_group;

  const c10::cuda::CUDAGuard device_guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
  torch::Tensor y = output.has_value()
                        ? output.value()
                        : torch::empty({m, n}, x.options().dtype(torch::kBFloat16));
  torch::Tensor tmas =
      prepare_byte_workspace(tma_desc, x, num_group * 6, "tma_desc");
  torch::Tensor tiles =
      prepare_int_workspace(tiles_workspace, x, num_group, "tiles");
  torch::Tensor cu_tiles =
      prepare_int_workspace(cu_tiles_workspace, x, num_group + 1, "cu_tiles");
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
        py::arg("output") = py::none(), py::arg("tma_desc") = py::none(),
        py::arg("tiles") = py::none(), py::arg("cu_tiles") = py::none(),
        py::arg("update_tma") = true);
  m.def("fp8_blockwise_grouped_gemm_ptrs",
        &batchgen::moe::fp8_blockwise_grouped_gemm_ptrs,
        "FP8 blockwise grouped GEMM over independent weight pointer arrays",
        py::arg("x"), py::arg("weight_prototype"), py::arg("weight_ptrs"),
        py::arg("seqlens"), py::arg("cu_seqlens"), py::arg("x_scale"),
        py::arg("w_scale_prototype"), py::arg("w_scale_ptrs"),
        py::arg("num_seq_per_group_avg"), py::arg("output") = py::none(),
        py::arg("tma_desc") = py::none(), py::arg("tiles") = py::none(),
        py::arg("cu_tiles") = py::none());
  m.def("fp8_blockwise_fused_s1",
        &batchgen::moe::fp8_blockwise_fused_s1,
        "FP8 blockwise fused S1: gate+up+SiLU (CuTe persistent 3-WG, v19)",
        py::arg("x"), py::arg("gate_weight"), py::arg("up_weight"),
        py::arg("seqlens"), py::arg("cu_seqlens"),
        py::arg("x_scale"), py::arg("gate_w_scale"), py::arg("up_w_scale"),
        py::arg("num_seq_per_group_avg"), py::arg("output") = py::none(),
        py::arg("tma_desc") = py::none(), py::arg("tiles") = py::none(),
        py::arg("cu_tiles") = py::none(), py::arg("update_tma") = true);
  m.def("fp8_blockwise_prepare_s1_s3_tma",
        &batchgen::moe::fp8_blockwise_prepare_s1_s3_tma,
        "One launch writing the per-expert X/Y TMA descriptors of a fused S1 and a grouped "
        "S3 GEMM plus their shared tiles / cu_tiles (consumed with update_tma=False)",
        py::arg("s1_x"), py::arg("s1_output"), py::arg("s3_x"), py::arg("s3_output"),
        py::arg("seqlens"), py::arg("cu_seqlens"), py::arg("num_seq_per_group_avg"),
        py::arg("s1_tma_desc"), py::arg("s3_tma_desc"), py::arg("tiles"),
        py::arg("cu_tiles"));
  m.def("fp8_blockwise_fused_s1_ptrs",
        &batchgen::moe::fp8_blockwise_fused_s1_ptrs,
        "FP8 fused S1 over independent gate/up weight pointer arrays",
        py::arg("x"), py::arg("gate_weight_prototype"),
        py::arg("gate_weight_ptrs"), py::arg("up_weight_prototype"),
        py::arg("up_weight_ptrs"), py::arg("seqlens"),
        py::arg("cu_seqlens"), py::arg("x_scale"),
        py::arg("gate_w_scale_prototype"), py::arg("gate_w_scale_ptrs"),
        py::arg("up_w_scale_prototype"), py::arg("up_w_scale_ptrs"),
        py::arg("num_seq_per_group_avg"), py::arg("output") = py::none(),
        py::arg("tma_desc") = py::none(), py::arg("tiles") = py::none(),
        py::arg("cu_tiles") = py::none());
}
