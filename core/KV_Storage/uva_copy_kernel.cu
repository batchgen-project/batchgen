#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "../utils.h"

namespace batchgen::kv::worker_detail {
namespace {

// each block copies one page. page_idx = blockIdx.x
// src_ptrs / dst_ptrs are device-resident arrays of byte pointers. Either side
// may point to GPU memory or UVA-mapped pinned host memory, so the same kernel
// is used for host->device page loads and device->host decode appends.
//
__device__ void UvaPageCopyKernelImpl(uint8_t** src_ptrs, uint8_t** dst_ptrs,
                                      size_t page_size_bytes, int num_pages) {
    const int page_idx = blockIdx.x;
    if (page_idx >= num_pages) {
        return;
    }

    uint8_t* src = src_ptrs[page_idx];  // UVA-mapped host memory pointer
    uint8_t* dst = dst_ptrs[page_idx];  // device global memory pointer

    const auto src_addr = reinterpret_cast<uintptr_t>(src);
    const auto dst_addr = reinterpret_cast<uintptr_t>(dst);
    const bool can_vectorize =
        (src_addr % alignof(uint4) == 0) && (dst_addr % alignof(uint4) == 0);

    if (can_vectorize) {
        const size_t num_words = page_size_bytes / sizeof(uint4);
        const size_t vector_bytes = num_words * sizeof(uint4);
        uint4* src_vec = reinterpret_cast<uint4*>(src);
        uint4* dst_vec = reinterpret_cast<uint4*>(dst);

        for (size_t i = threadIdx.x; i < num_words; i += blockDim.x) {
            dst_vec[i] = src_vec[i];
        }
        for (size_t i = vector_bytes + threadIdx.x; i < page_size_bytes;
             i += blockDim.x) {
            dst[i] = src[i];
        }
        return;
    }

    for (size_t i = threadIdx.x; i < page_size_bytes; i += blockDim.x) {
        dst[i] = src[i];
    }
}

__global__ void UvaPageCopyKernel(uint8_t** src_ptrs, uint8_t** dst_ptrs,
                                  size_t page_size_bytes, int num_pages) {
    UvaPageCopyKernelImpl(src_ptrs, dst_ptrs, page_size_bytes, num_pages);
}

}  // namespace

void LaunchUvaPageCopyKernel(uint8_t** src_ptrs, uint8_t** dst_ptrs,
                             std::size_t page_size_bytes, int num_pages,
                             cudaStream_t stream) {
    if (src_ptrs == nullptr || dst_ptrs == nullptr || num_pages <= 0 ||
        page_size_bytes == 0) {
        return;
    }
    constexpr int kThreadsPerBlock = 256;
    const dim3 grid(static_cast<unsigned int>(num_pages));
    const dim3 block(kThreadsPerBlock);
    UvaPageCopyKernel<<<grid, block, 0, stream>>>(src_ptrs, dst_ptrs,
                                                  page_size_bytes, num_pages);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace batchgen::kv::worker_detail
