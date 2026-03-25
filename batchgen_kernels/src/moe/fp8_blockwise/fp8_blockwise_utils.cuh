// BatchGen — FP8 Blockwise GEMM Utility Functions
// TMA descriptor management, vectorized load/store, synchronization primitives.

#ifndef BATCHGEN_FP8_BLOCKWISE_UTILS_CUH_
#define BATCHGEN_FP8_BLOCKWISE_UTILS_CUH_

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include "cute/tensor.hpp"

namespace batchgen {
namespace moe {

// ============================
//    Vectorized Load/Store
// ============================
template <typename T, int N>
struct vec_t {
  T data[N];
  using type = T;
  static constexpr int kNum = N;
  __device__ __forceinline__ constexpr T &operator[](int idx) { return data[idx]; }
  __device__ __forceinline__ constexpr const T &operator[](int idx) const { return data[idx]; }
};

template <typename T, int N>
__device__ __forceinline__ constexpr auto load(const void *ptr) {
  using V = vec_t<T, N>;
  V v;
  constexpr int kBytes = sizeof(T) * N;
  static_assert(kBytes == 1 || kBytes == 2 || kBytes == 4 || kBytes == 8 || kBytes == 16,
                "unsupported load size");
  if constexpr (kBytes == 16) {
    *reinterpret_cast<uint4 *>(&v) = *reinterpret_cast<const uint4 *>(ptr);
  } else if constexpr (kBytes == 8) {
    *reinterpret_cast<uint64_t *>(&v) = *reinterpret_cast<const uint64_t *>(ptr);
  } else if constexpr (kBytes == 4) {
    *reinterpret_cast<uint32_t *>(&v) = *reinterpret_cast<const uint32_t *>(ptr);
  } else if constexpr (kBytes == 2) {
    *reinterpret_cast<uint16_t *>(&v) = *reinterpret_cast<const uint16_t *>(ptr);
  } else {
    *reinterpret_cast<uint8_t *>(&v) = *reinterpret_cast<const uint8_t *>(ptr);
  }
  return v;
}

template <typename T, int N>
__device__ __forceinline__ constexpr void store(void *ptr, const vec_t<T, N> &v) {
  constexpr int kBytes = sizeof(T) * N;
  static_assert(kBytes == 1 || kBytes == 2 || kBytes == 4 || kBytes == 8 || kBytes == 16,
                "unsupported store size");
  if constexpr (kBytes == 16) {
    *reinterpret_cast<uint4 *>(ptr) = *reinterpret_cast<const uint4 *>(&v);
  } else if constexpr (kBytes == 8) {
    *reinterpret_cast<uint64_t *>(ptr) = *reinterpret_cast<const uint64_t *>(&v);
  } else if constexpr (kBytes == 4) {
    *reinterpret_cast<uint32_t *>(ptr) = *reinterpret_cast<const uint32_t *>(&v);
  } else if constexpr (kBytes == 2) {
    *reinterpret_cast<uint16_t *>(ptr) = *reinterpret_cast<const uint16_t *>(&v);
  } else {
    *reinterpret_cast<uint8_t *>(ptr) = *reinterpret_cast<const uint8_t *>(&v);
  }
}

// ============================
//    TMA Descriptor Helpers
// ============================
__device__ __forceinline__ void tma_descriptor_replace_shapes_in_shared_mem(
    cute::TmaDescriptor &smem_desc, cute::array<uint32_t, 5> const &prob_shape) {
#if (__CUDACC_VER_MAJOR__ > 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 3))
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  uint32_t smem_int_desc = cute::cast_smem_ptr_to_uint(&smem_desc);
  uint64_t const smem_int64_desc = 0;
  asm volatile("cvt.u64.u32 %0, %1;" ::"l"(smem_int64_desc), "r"(smem_int_desc));
  asm volatile(
      "tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%0], 0, %1;" ::"l"(smem_int64_desc),
      "r"(prob_shape[0]));
  asm volatile(
      "tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%0], 1, %1;" ::"l"(smem_int64_desc),
      "r"(prob_shape[1]));
  asm volatile(
      "tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%0], 2, %1;" ::"l"(smem_int64_desc),
      "r"(prob_shape[2]));
  asm volatile(
      "tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%0], 3, %1;" ::"l"(smem_int64_desc),
      "r"(prob_shape[3]));
  asm volatile(
      "tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%0], 4, %1;" ::"l"(smem_int64_desc),
      "r"(prob_shape[4]));
#endif
#endif
}

template <typename Tma, typename GTensor, bool kUpdateShape = true>
__device__ __forceinline__ void update_tma_gtensor(cute::TmaDescriptor &smem_tma_desc,
                                                   const GTensor &gtensor) {
  cute::array<uint32_t, 5> shape{1, 1, 1, 1, 1};
  cute::array<uint64_t, 5> stride{0, 0, 0, 0, 0};
  cute::detail::fill_tma_gmem_shape_stride(Tma{}, gtensor, shape, stride);
  const void *gmem_ptr = gtensor.data().get();
  cute::tma_descriptor_replace_addr_in_shared_mem(smem_tma_desc, gmem_ptr);
  if constexpr (kUpdateShape) {
    tma_descriptor_replace_shapes_in_shared_mem(smem_tma_desc, shape);
  }
}

// ============================
//    Synchronization
// ============================
__device__ __forceinline__ void syncwarpgroup(int barrier_id) {
  asm volatile("barrier.cta.sync %0, 128;\n" ::"r"(barrier_id) : "memory");
}

// ============================
//    Device Information
// ============================
inline int get_sm_count() {
  static int sm_count = -1;
  if (sm_count == -1) {
    int device;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
  }
  return sm_count;
}

// ============================
//    Fragment Retiling
// ============================
template <typename Tensor>
__device__ __forceinline__ constexpr auto retile_fragment(Tensor &&tensor) {
  using namespace cute;  // NOLINT
  constexpr int R = decltype(tensor.layout())::rank;
  static_assert(R == 3, "only rank 3 fragment supported");
  auto thr_vmk = flatten(select<0>(tensor.layout()));
  auto tile_mk = select<1, 2>(tensor.layout());
  auto m_layout =
      coalesce(make_layout(make_shape(get<1>(thr_vmk.shape()), get<0>(tile_mk.shape())),
                           make_stride(get<1>(thr_vmk.stride()), get<0>(tile_mk.stride()))));
  auto k_layout = coalesce(make_layout(
      make_shape(get<0>(thr_vmk.shape()), get<2>(thr_vmk.shape()), get<1>(tile_mk.shape())),
      make_stride(get<0>(thr_vmk.stride()), get<2>(thr_vmk.stride()), get<1>(tile_mk.stride()))));
  return make_tensor(static_cast<Tensor &&>(tensor).data(), make_layout(m_layout, k_layout));
}

}  // namespace moe
}  // namespace batchgen

#endif  // BATCHGEN_FP8_BLOCKWISE_UTILS_CUH_
