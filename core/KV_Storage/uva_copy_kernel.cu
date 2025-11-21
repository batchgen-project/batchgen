#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "../utils.h"

namespace batchgen::kv::worker_detail {
namespace {

// each block copies one page. page_idx = blockIdx.x
// src_ptrs:
//     A device-resident array of uint8_t* pointers.
//     Each pointer refers to **host pinned memory** (allocated with
//     cudaHostAlloc or registered with cudaHostRegister), that is **UVA-mapped
//     into the device address space**. Because of UVA (Unified Virtual
//     Addressing), these host pointers are valid device pointers as well, so
//     the GPU can directly load from them using global memory instructions
//     (zero-copy access).
//
// dst_ptrs:
//     A device-resident array of uint8_t* pointers to normal GPU global memory.
//
// Why this works (important UVA explanation):
//     - Pinned host memory is registered with the CUDA driver.
//     - On UVA-enabled systems (all modern GPUs), the driver maps this host
//       memory into the GPU's virtual address space.
//     - Therefore, the CPU pointer returned by cudaHostAlloc is already a
//       **device-accessible virtual address**.
//     - The GPU can dereference these pointers directly inside the kernel,
//       performing PCIe reads with no need for cudaMemcpyAsync.
//
// This kernel performs a simple vectorized copy (using uint4 loads/stores)
// from host memory → device memory without using cudaMemcpyAsync.
//
__device__ void UvaPageCopyKernelImpl(uint8_t** src_ptrs, uint8_t** dst_ptrs,
                                      size_t page_size_bytes, int num_pages) {
    const int page_idx = blockIdx.x;
    if (page_idx >= num_pages) {
        return;
    }

    uint8_t* src = src_ptrs[page_idx];  // UVA-mapped host memory pointer
    uint8_t* dst = dst_ptrs[page_idx];  // device global memory pointer

    const int num_words = static_cast<int>(page_size_bytes / sizeof(uint4));
    uint4* src_vec = reinterpret_cast<uint4*>(src);
    uint4* dst_vec = reinterpret_cast<uint4*>(dst);

    for (int i = threadIdx.x; i < num_words; i += blockDim.x) {
        dst_vec[i] = src_vec[i];  // GPU performs a PCIe read here
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