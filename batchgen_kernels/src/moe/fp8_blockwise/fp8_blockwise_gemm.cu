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

}  // namespace

// ============================================================================
// PyTorch entry point
// ============================================================================
torch::Tensor fp8_blockwise_grouped_gemm(
    const torch::Tensor &x, const torch::Tensor &weight, const torch::Tensor &seqlens,
    const torch::Tensor &cu_seqlens, const torch::Tensor &x_scale, const torch::Tensor &w_scale,
    const int64_t num_seq_per_group_avg, std::optional<torch::Tensor> output,
    std::optional<torch::Tensor> tma_desc) {
  const auto geom =
      check_grouped_gemm_inputs(x, weight, seqlens, cu_seqlens, x_scale, w_scale, output);
  const int m = geom.m;
  const int k = geom.k;
  const int n = geom.n;
  const int m_pad = geom.m_pad;
  const int num_block_k_pad4 = geom.num_block_k_pad4;
  const int num_group = geom.num_group;

  if (tma_desc.has_value()) {
    const auto &td = tma_desc.value();
    check_cuda_tensor(td, x.device(), "tma_desc");
    TORCH_CHECK(td.dim() == 2 && td.element_size() == 1 && td.size(0) == 2 * num_group &&
                    td.size(1) == 128,
                "tma_desc must be a 1-byte [2 * num_group, 128] tensor");
    TORCH_CHECK(reinterpret_cast<std::uintptr_t>(td.data_ptr()) % 64 == 0,
                "tma_desc data pointer must be 64-byte aligned");
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

  torch::Tensor tiles = torch::empty({num_group}, options.dtype(torch::kInt32));
  torch::Tensor cu_tiles = torch::empty({num_group + 1}, options.dtype(torch::kInt32));

  fp8_blockwise_grouped_gemm_async(
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

  const c10::cuda::CUDAGuard device_guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device());

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

}  // close namespace moe
}  // close namespace batchgen

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fp8_blockwise_grouped_gemm",
        &batchgen::moe::fp8_blockwise_grouped_gemm,
        "FP8 blockwise grouped GEMM (CuTe persistent 3-WG, adaptive TileM)",
        py::arg("x"), py::arg("weight"), py::arg("seqlens"), py::arg("cu_seqlens"),
        py::arg("x_scale"), py::arg("w_scale"), py::arg("num_seq_per_group_avg"),
        py::arg("output") = py::none(), py::arg("tma_desc") = py::none());
  m.def("fp8_blockwise_fused_s1",
        &batchgen::moe::fp8_blockwise_fused_s1,
        "FP8 blockwise fused S1: gate+up+SiLU (CuTe persistent 3-WG, v19)",
        py::arg("x"), py::arg("gate_weight"), py::arg("up_weight"),
        py::arg("seqlens"), py::arg("cu_seqlens"),
        py::arg("x_scale"), py::arg("gate_w_scale"), py::arg("up_w_scale"),
        py::arg("num_seq_per_group_avg"), py::arg("output") = py::none());
}
