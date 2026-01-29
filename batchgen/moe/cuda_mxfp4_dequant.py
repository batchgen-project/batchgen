"""CUDA kernel for MXFP4 batch dequantization with shared memory LUT.

This kernel uses shared memory to store the FP4 E2M1 lookup table, enabling
single-cycle lookups instead of the 5-6 tl.where() conditionals in Triton.

Expected performance: ~8-10ms vs 28ms (Triton E2M1) for GPT-OSS-120B weights.

Usage:
    from batchgen.moe.cuda_mxfp4_dequant import batch_mxfp4_dequant_cuda

    batch_mxfp4_dequant_cuda(weight_ptrs, scale_ptrs, output, weights[0], scales[0])
"""

import torch
from torch.utils.cpp_extension import load_inline
import os
import time

# CUDA source for batch MXFP4 dequantization
CUDA_SOURCE = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

// FP4 E2M1 lookup table (16 values, positive and negative)
// Values: 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
__constant__ float FP4_LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// Block configuration
#define BLOCK_N 64
#define BLOCK_K 32  // Must match MXFP4 scale block size
#define THREADS_PER_BLOCK 256

// Fast ldexp using bit manipulation
__device__ __forceinline__ float fast_ldexp(float x, int exp) {
    // Clamp exponent to valid range
    exp = max(min(exp, 127), -126);
    // Create 2^exp by setting IEEE 754 exponent bits
    int exp_bits = (exp + 127) << 23;
    float power_of_2 = __int_as_float(exp_bits);
    return x * power_of_2;
}

// Convert float to bfloat16
__device__ __forceinline__ __nv_bfloat16 float_to_bf16(float x) {
    return __float2bfloat16(x);
}

/*
 * Batch MXFP4 dequantization kernel with shared memory LUT
 *
 * Grid: (num_experts, cdiv(N, BLOCK_N), cdiv(K, BLOCK_K))
 * Block: THREADS_PER_BLOCK threads
 *
 * Each thread block processes a [BLOCK_N, BLOCK_K] tile of one expert's weights.
 * Shared memory holds the FP4 LUT for single-cycle lookups.
 */
__global__ void batch_mxfp4_dequant_kernel(
    const int64_t* __restrict__ packed_ptrs,    // [num_experts] pointers to packed FP4
    const int64_t* __restrict__ scale_ptrs,     // [num_experts] pointers to scales
    __nv_bfloat16* __restrict__ output,         // [num_experts, N, K] output
    const int N,
    const int K,
    const int K_packed,                          // K // 2
    const int K_scale,                           // K // 32
    const int64_t stride_packed_n,               // Stride for packed weights
    const int64_t stride_packed_k,
    const int64_t stride_scale_n,                // Stride for scales
    const int64_t stride_scale_k,
    const int64_t stride_out_e,                  // Stride for output
    const int64_t stride_out_n,
    const int64_t stride_out_k
) {
    // Load FP4 LUT into shared memory (only first 16 threads)
    __shared__ float lut[16];
    if (threadIdx.x < 16) {
        lut[threadIdx.x] = FP4_LUT[threadIdx.x];
    }
    __syncthreads();

    // Get expert and tile indices
    const int expert_idx = blockIdx.x;
    const int n_block = blockIdx.y;
    const int k_block = blockIdx.z;

    // Base pointers for this expert
    const uint8_t* packed_base = reinterpret_cast<const uint8_t*>(packed_ptrs[expert_idx]);
    const uint8_t* scale_base = reinterpret_cast<const uint8_t*>(scale_ptrs[expert_idx]);

    // Each thread processes multiple elements in the tile
    // Tile size: [BLOCK_N, BLOCK_K] = [64, 32] = 2048 elements
    // With 256 threads, each thread processes 8 elements
    const int elements_per_tile = BLOCK_N * BLOCK_K;
    const int elements_per_thread = (elements_per_tile + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    for (int elem = 0; elem < elements_per_thread; elem++) {
        int linear_idx = threadIdx.x + elem * THREADS_PER_BLOCK;
        if (linear_idx >= elements_per_tile) break;

        // Convert linear index to (n_local, k_local) within tile
        int n_local = linear_idx / BLOCK_K;  // [0, BLOCK_N)
        int k_local = linear_idx % BLOCK_K;  // [0, BLOCK_K)

        // Global indices
        int n_global = n_block * BLOCK_N + n_local;
        int k_global = k_block * BLOCK_K + k_local;

        // Bounds check
        if (n_global >= N || k_global >= K) continue;

        // Load scale (one scale per 32 K values, so use k_block)
        int scale_idx = n_global * stride_scale_n + k_block * stride_scale_k;
        int scale_raw = static_cast<int>(scale_base[scale_idx]) - 127;

        // Load packed FP4 byte
        // Each byte contains 2 FP4 values: low nibble (even K), high nibble (odd K)
        int k_packed_idx = k_global / 2;
        int packed_idx = n_global * stride_packed_n + k_packed_idx * stride_packed_k;
        uint8_t packed_byte = packed_base[packed_idx];

        // Extract correct nibble based on even/odd K index
        int fp4_idx;
        if (k_global % 2 == 0) {
            fp4_idx = packed_byte & 0x0F;  // Low nibble for even K
        } else {
            fp4_idx = (packed_byte >> 4) & 0x0F;  // High nibble for odd K
        }

        // Lookup in shared memory (single cycle!)
        float val = lut[fp4_idx];

        // Apply scale: val * 2^scale
        float scaled_val = fast_ldexp(val, scale_raw);

        // Write output as bfloat16
        int64_t out_idx = static_cast<int64_t>(expert_idx) * stride_out_e +
                          static_cast<int64_t>(n_global) * stride_out_n +
                          static_cast<int64_t>(k_global) * stride_out_k;
        output[out_idx] = float_to_bf16(scaled_val);
    }
}


/*
 * Vectorized version: each thread processes 4 packed bytes (8 FP4 values)
 * This improves memory coalescing and instruction-level parallelism.
 */
__global__ void batch_mxfp4_dequant_kernel_vec4(
    const int64_t* __restrict__ packed_ptrs,
    const int64_t* __restrict__ scale_ptrs,
    __nv_bfloat16* __restrict__ output,
    const int N,
    const int K,
    const int K_packed,
    const int K_scale,
    const int64_t stride_packed_n,
    const int64_t stride_packed_k,
    const int64_t stride_scale_n,
    const int64_t stride_scale_k,
    const int64_t stride_out_e,
    const int64_t stride_out_n,
    const int64_t stride_out_k
) {
    // Load FP4 LUT into shared memory
    __shared__ float lut[16];
    if (threadIdx.x < 16) {
        lut[threadIdx.x] = FP4_LUT[threadIdx.x];
    }
    __syncthreads();

    const int expert_idx = blockIdx.x;
    const int n_block = blockIdx.y;
    const int k_block = blockIdx.z;

    const uint8_t* packed_base = reinterpret_cast<const uint8_t*>(packed_ptrs[expert_idx]);
    const uint8_t* scale_base = reinterpret_cast<const uint8_t*>(scale_ptrs[expert_idx]);

    // Each thread processes 8 output values (4 packed bytes)
    // Tile: [BLOCK_N, BLOCK_K] = [64, 32] = 2048 values
    // 2048 / 8 = 256 units of work, one per thread
    const int k_start = k_block * BLOCK_K;

    // Thread assignment: process 8 consecutive K values for one N
    const int work_units = (BLOCK_N * BLOCK_K) / 8;  // 256
    const int unit_idx = threadIdx.x;

    if (unit_idx >= work_units) return;

    // Each unit covers 8 K values for one N row
    // k_local ranges [0, BLOCK_K) in steps of 8
    const int n_local = (unit_idx * 8) / BLOCK_K;
    const int k_local_base = (unit_idx * 8) % BLOCK_K;

    const int n_global = n_block * BLOCK_N + n_local;
    if (n_global >= N) return;

    // Load scale (same for all 8 K values since BLOCK_K=32 matches scale block)
    const int scale_idx = n_global * stride_scale_n + k_block * stride_scale_k;
    const int scale_raw = static_cast<int>(scale_base[scale_idx]) - 127;

    // Load 4 packed bytes (strided if stride_packed_k != 1)
    const int k_packed_base = (k_start + k_local_base) / 2;

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        const int k_packed_idx = k_packed_base + i;
        const int k_global_lo = k_start + k_local_base + i * 2;
        const int k_global_hi = k_global_lo + 1;

        if (k_global_hi >= K) break;

        // Load packed byte
        const int packed_idx = n_global * stride_packed_n + k_packed_idx * stride_packed_k;
        const uint8_t packed_byte = packed_base[packed_idx];

        // Unpack and lookup
        const int fp4_lo = packed_byte & 0x0F;
        const int fp4_hi = (packed_byte >> 4) & 0x0F;

        const float val_lo = lut[fp4_lo];
        const float val_hi = lut[fp4_hi];

        // Apply scale
        const float scaled_lo = fast_ldexp(val_lo, scale_raw);
        const float scaled_hi = fast_ldexp(val_hi, scale_raw);

        // Write outputs
        const int64_t out_idx_lo = static_cast<int64_t>(expert_idx) * stride_out_e +
                                   static_cast<int64_t>(n_global) * stride_out_n +
                                   static_cast<int64_t>(k_global_lo) * stride_out_k;
        const int64_t out_idx_hi = out_idx_lo + stride_out_k;

        output[out_idx_lo] = float_to_bf16(scaled_lo);
        output[out_idx_hi] = float_to_bf16(scaled_hi);
    }
}


/*
 * Optimized version: Row-major processing for better memory coalescing
 * Each warp processes consecutive K elements across multiple N rows
 */
__global__ void batch_mxfp4_dequant_kernel_coalesced(
    const int64_t* __restrict__ packed_ptrs,
    const int64_t* __restrict__ scale_ptrs,
    __nv_bfloat16* __restrict__ output,
    const int N,
    const int K,
    const int64_t stride_packed_n,
    const int64_t stride_packed_k,
    const int64_t stride_scale_n,
    const int64_t stride_scale_k,
    const int64_t stride_out_e,
    const int64_t stride_out_n,
    const int64_t stride_out_k
) {
    // Load FP4 LUT into shared memory
    __shared__ float lut[16];
    if (threadIdx.x < 16) {
        lut[threadIdx.x] = FP4_LUT[threadIdx.x];
    }
    __syncthreads();

    const int expert_idx = blockIdx.x;
    const int n_global = blockIdx.y * blockDim.y + threadIdx.y;
    const int k_block = blockIdx.z;

    if (n_global >= N) return;

    const uint8_t* packed_base = reinterpret_cast<const uint8_t*>(packed_ptrs[expert_idx]);
    const uint8_t* scale_base = reinterpret_cast<const uint8_t*>(scale_ptrs[expert_idx]);

    // Load scale for this N row and K block
    const int scale_idx = n_global * stride_scale_n + k_block * stride_scale_k;
    const int scale_raw = static_cast<int>(scale_base[scale_idx]) - 127;

    // Each thread in x-dimension processes 2 consecutive K values (one packed byte)
    const int k_local = threadIdx.x * 2;  // 0, 2, 4, ..., 30
    const int k_global = k_block * BLOCK_K + k_local;

    if (k_global + 1 >= K) return;

    // Load packed byte
    const int k_packed_idx = k_global / 2;
    const int packed_idx = n_global * stride_packed_n + k_packed_idx * stride_packed_k;
    const uint8_t packed_byte = packed_base[packed_idx];

    // Unpack and lookup
    const float val_lo = lut[packed_byte & 0x0F];
    const float val_hi = lut[(packed_byte >> 4) & 0x0F];

    // Apply scale
    const float scaled_lo = fast_ldexp(val_lo, scale_raw);
    const float scaled_hi = fast_ldexp(val_hi, scale_raw);

    // Write outputs (consecutive in K dimension for coalescing)
    const int64_t out_base = static_cast<int64_t>(expert_idx) * stride_out_e +
                             static_cast<int64_t>(n_global) * stride_out_n +
                             static_cast<int64_t>(k_global) * stride_out_k;

    output[out_base] = float_to_bf16(scaled_lo);
    output[out_base + stride_out_k] = float_to_bf16(scaled_hi);
}


void batch_mxfp4_dequant_cuda_impl(
    torch::Tensor packed_ptrs,     // [num_experts] int64
    torch::Tensor scale_ptrs,      // [num_experts] int64
    torch::Tensor output,          // [num_experts, N, K] bfloat16
    int64_t stride_packed_n,
    int64_t stride_packed_k,
    int64_t stride_scale_n,
    int64_t stride_scale_k,
    int kernel_version             // 0=basic, 1=vec4, 2=coalesced
) {
    const int num_experts = packed_ptrs.size(0);
    const int N = output.size(1);
    const int K = output.size(2);
    const int K_packed = K / 2;
    const int K_scale = K / 32;

    // Output strides
    const int64_t stride_out_e = output.stride(0);
    const int64_t stride_out_n = output.stride(1);
    const int64_t stride_out_k = output.stride(2);

    if (kernel_version == 0) {
        // Basic kernel
        dim3 grid(num_experts, (N + BLOCK_N - 1) / BLOCK_N, (K + BLOCK_K - 1) / BLOCK_K);
        dim3 block(THREADS_PER_BLOCK);

        batch_mxfp4_dequant_kernel<<<grid, block>>>(
            packed_ptrs.data_ptr<int64_t>(),
            scale_ptrs.data_ptr<int64_t>(),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
            N, K, K_packed, K_scale,
            stride_packed_n, stride_packed_k,
            stride_scale_n, stride_scale_k,
            stride_out_e, stride_out_n, stride_out_k
        );
    } else if (kernel_version == 1) {
        // Vectorized kernel (4 bytes = 8 values per thread)
        dim3 grid(num_experts, (N + BLOCK_N - 1) / BLOCK_N, (K + BLOCK_K - 1) / BLOCK_K);
        dim3 block(THREADS_PER_BLOCK);

        batch_mxfp4_dequant_kernel_vec4<<<grid, block>>>(
            packed_ptrs.data_ptr<int64_t>(),
            scale_ptrs.data_ptr<int64_t>(),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
            N, K, K_packed, K_scale,
            stride_packed_n, stride_packed_k,
            stride_scale_n, stride_scale_k,
            stride_out_e, stride_out_n, stride_out_k
        );
    } else {
        // Coalesced kernel: 16 threads in x for K (16*2=32), 16 threads in y for N
        dim3 grid(num_experts, (N + 15) / 16, (K + BLOCK_K - 1) / BLOCK_K);
        dim3 block(16, 16);  // 256 threads: 16 for K, 16 for N

        batch_mxfp4_dequant_kernel_coalesced<<<grid, block>>>(
            packed_ptrs.data_ptr<int64_t>(),
            scale_ptrs.data_ptr<int64_t>(),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
            N, K,
            stride_packed_n, stride_packed_k,
            stride_scale_n, stride_scale_k,
            stride_out_e, stride_out_n, stride_out_k
        );
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("batch_mxfp4_dequant_cuda_impl", &batch_mxfp4_dequant_cuda_impl,
          "Batch MXFP4 dequantization with shared memory LUT");
}
'''

# Compile and cache the CUDA extension
_cuda_module = None

def clear_cuda_cache():
    """Clear the cached CUDA extension to force recompilation."""
    import shutil
    import glob
    cache_dir = os.path.expanduser("~/.cache/torch_extensions")
    for path in glob.glob(os.path.join(cache_dir, "*", "cuda_mxfp4_dequant")):
        try:
            shutil.rmtree(path)
            print(f"Cleared cache: {path}")
        except Exception as e:
            print(f"Failed to clear {path}: {e}")

def _get_cuda_module():
    """Lazy-load and compile the CUDA module."""
    global _cuda_module
    if _cuda_module is None:
        # First try to clear any stale cache
        clear_cuda_cache()

        # Use cuda_sources only with PYBIND11_MODULE defined inside
        # No cpp_sources or functions parameter needed
        _cuda_module = load_inline(
            name='cuda_mxfp4_dequant',
            cpp_sources=[],  # Empty - everything is in CUDA source
            cuda_sources=[CUDA_SOURCE],
            extra_cuda_cflags=['-O3', '--use_fast_math', '-lineinfo'],
            verbose=os.environ.get('CUDA_DEBUG', '0') == '1',
        )
    return _cuda_module


def batch_mxfp4_dequant_cuda(
    packed_ptrs: torch.Tensor,    # [num_experts] int64
    scale_ptrs: torch.Tensor,     # [num_experts] int64
    output: torch.Tensor,         # [num_experts, N, K] BF16
    packed_ref: torch.Tensor,     # Reference tensor for strides [N, K//2]
    scale_ref: torch.Tensor,      # Reference tensor for strides [N, K//32]
    kernel_version: int = 2,      # 0=basic, 1=vec4, 2=coalesced (default)
) -> None:
    """Batch dequantize all experts' MXFP4 weights using CUDA with shared memory LUT.

    Args:
        packed_ptrs: Pointer array to packed FP4 weights [num_experts]
        scale_ptrs: Pointer array to scales [num_experts]
        output: Pre-allocated output buffer [num_experts, N, K] BF16
        packed_ref: Reference weight tensor for computing strides
        scale_ref: Reference scale tensor for computing strides
        kernel_version: 0=basic, 1=vec4, 2=coalesced (best for most cases)
    """
    cuda_mod = _get_cuda_module()
    cuda_mod.batch_mxfp4_dequant_cuda_impl(
        packed_ptrs,
        scale_ptrs,
        output,
        packed_ref.stride(0),
        packed_ref.stride(1),
        scale_ref.stride(0),
        scale_ref.stride(1),
        kernel_version,
    )


def benchmark_cuda_dequant(
    num_experts: int = 128,
    N: int = 13824,
    K: int = 5120,
    device: str = "cuda",
    warmup_iters: int = 5,
    bench_iters: int = 20,
) -> dict:
    """Benchmark CUDA batch dequantization kernel.

    Returns dict with timing for each kernel version.
    """
    # Create test weights
    weights = [torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=device)
               for _ in range(num_experts)]
    scales = [torch.randint(120, 134, (N, K // 32), dtype=torch.uint8, device=device)
              for _ in range(num_experts)]

    # Create pointer arrays
    weight_ptrs = torch.tensor([w.data_ptr() for w in weights],
                               dtype=torch.int64, device=device)
    scale_ptrs = torch.tensor([s.data_ptr() for s in scales],
                              dtype=torch.int64, device=device)

    # Allocate output
    output = torch.empty(num_experts, N, K, dtype=torch.bfloat16, device=device)

    results = {}
    kernel_names = ['basic', 'vec4', 'coalesced']

    for version, name in enumerate(kernel_names):
        # Warmup
        for _ in range(warmup_iters):
            batch_mxfp4_dequant_cuda(weight_ptrs, scale_ptrs, output,
                                     weights[0], scales[0], kernel_version=version)
        torch.cuda.synchronize()

        # Benchmark
        start = time.perf_counter()
        for _ in range(bench_iters):
            batch_mxfp4_dequant_cuda(weight_ptrs, scale_ptrs, output,
                                     weights[0], scales[0], kernel_version=version)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / bench_iters * 1000

        results[name] = elapsed

    # Calculate memory bandwidth
    total_bytes = num_experts * (N * K // 2 + N * K // 32 + N * K * 2)  # packed + scales + output
    best_time = min(results.values())
    bandwidth_gbps = (total_bytes / 1e9) / (best_time / 1000)

    results['best_time_ms'] = best_time
    results['bandwidth_gbps'] = bandwidth_gbps

    return results


if __name__ == "__main__":
    print("CUDA MXFP4 Batch Dequantization Benchmark")
    print("=" * 60)

    # Compile the CUDA extension
    print("Compiling CUDA kernel...")
    _get_cuda_module()
    print("Compilation successful!")

    # Run benchmark
    print("\nRunning benchmark (GPT-OSS-120B dimensions)...")
    results = benchmark_cuda_dequant()

    print(f"\nResults:")
    print(f"  Basic kernel:     {results['basic']:.3f} ms")
    print(f"  Vec4 kernel:      {results['vec4']:.3f} ms")
    print(f"  Coalesced kernel: {results['coalesced']:.3f} ms")
    print(f"\nBest time: {results['best_time_ms']:.3f} ms")
    print(f"Effective bandwidth: {results['bandwidth_gbps']:.1f} GB/s")
