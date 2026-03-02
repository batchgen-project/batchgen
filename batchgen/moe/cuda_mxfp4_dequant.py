"""CUDA kernel for MXFP4 batch dequantization with shared memory LUT.

This kernel uses shared memory to store the FP4 E2M1 lookup table, enabling
single-cycle lookups instead of the 5-6 tl.where() conditionals in Triton.

Expected performance: ~8-10ms vs 28ms (Triton E2M1) for GPT-OSS-120B weights.

Usage:
    from batchgen.moe.cuda_mxfp4_dequant import batch_mxfp4_dequant_cuda

    batch_mxfp4_dequant_cuda(weight_ptrs, scale_ptrs, output, weights[0], scales[0])
"""

import torch

_cuda_module = None


def _get_cuda_module():
    """Lazy-load the pre-compiled MXFP4 dequant CUDA module."""
    global _cuda_module
    if _cuda_module is None:
        import batchgen_kernels
        _cuda_module = batchgen_kernels.load_extension("batchgen_kernels.moe._C_mxfp4_dequant")
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
