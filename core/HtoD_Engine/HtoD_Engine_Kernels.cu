#include <cuda_runtime.h>
#include <stdint.h>

__global__ void batched_page_copy_kernel(
    uint8_t** src_ptrs, 
    uint8_t** dst_ptrs, 
    size_t page_size, 
    int num_pages) 
{
    // Get the page this block is responsible for
    int page_idx = blockIdx.x;
    if (page_idx >= num_pages) {
        return;
    }

    // Get the source and destination pointers for this specific page
    uint8_t* src_page = src_ptrs[page_idx];
    uint8_t* dst_page = dst_ptrs[page_idx];

    // Calculate number of 16-byte words to copy (using uint4)
    int num_words = page_size / sizeof(uint4);

    // Cast to uint4* for 16-byte vectorized loads/stores
    uint4* src_page_vec = (uint4*)src_page;
    uint4* dst_page_vec = (uint4*)dst_page;

    // Parallel copy within the block
    // Each thread copies multiple words in a strided loop
    for (int i = threadIdx.x; i < num_words; i += blockDim.x) {
        dst_page_vec[i] = src_page_vec[i];
    }
}