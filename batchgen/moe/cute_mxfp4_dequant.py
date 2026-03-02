"""CuTe-inspired CUDA kernel for MXFP4 batch dequantization.

This kernel implements CuTe-style optimizations for MXFP4 dequantization:
1. Vectorized 128-bit loads (uint4) for packed FP4 data
2. Vectorized 128-bit stores for BF16 output
3. Shared memory LUT for single-cycle FP4 decode (no ALU)
4. Warp-level scale broadcast using __shfl_sync
5. Swizzled shared memory to eliminate bank conflicts

Target: 50%+ HBM utilization (~10 ms) vs Triton v7's 29% (19.8 ms)

Usage:
    from batchgen.moe.cute_mxfp4_dequant import batch_mxfp4_dequant_cute

    batch_mxfp4_dequant_cute(weight_ptrs, scale_ptrs, output, weights[0], scales[0])
"""

import torch

# Pre-compiled at pip install time, loaded lazily on first use
_cute_module = None


def _get_cute_module():
    """Return the pre-compiled CuTe-style CUDA module (lazy load)."""
    global _cute_module
    if _cute_module is None:
        import batchgen_kernels
        _cute_module = batchgen_kernels.load_extension("batchgen_kernels.moe._C_mxfp4_dequant_cute")
    return _cute_module


def mxfp4_dequant_single_expert_cute(
    packed: torch.Tensor,         # [N, K//2] uint8 packed FP4
    scales: torch.Tensor,         # [K//32, N] uint8 scales (K-major)
    output: torch.Tensor,         # [N, K] BF16 output buffer
    kernel_version: int = 0,      # 0=simple, 1=swizzle
) -> None:
    """CuTe-style dequantize a single expert's MXFP4 weights.

    This is the per-expert dequant function for prefill, matching the user's requirement:
    "dequant an expert, call the matmul" - sequential per-expert processing.

    Args:
        packed: Packed FP4 weights [N, K//2] uint8 (row-major, 2 FP4 per byte)
        scales: K-major scales [K//32, N] uint8 (one scale per 32 K values)
        output: Pre-allocated output buffer [N, K] BF16
        kernel_version: 0=simple vectorized, 1=swizzled smem

    Note:
        The output buffer should be reused across projections to minimize memory.
        Example: ~142 MB buffer for GPT-OSS-120B (max of gate/up/down dimensions)
    """
    assert packed.dtype == torch.uint8, f"packed must be uint8, got {packed.dtype}"
    assert scales.dtype == torch.uint8, f"scales must be uint8, got {scales.dtype}"
    assert output.dtype == torch.bfloat16, f"output must be bfloat16, got {output.dtype}"
    assert packed.is_contiguous(), "packed must be contiguous"
    assert scales.is_contiguous(), "scales must be contiguous"
    assert output.is_contiguous(), "output must be contiguous"

    N, K_half = packed.shape
    K = K_half * 2
    assert output.shape == (N, K), f"output shape mismatch: expected ({N}, {K}), got {output.shape}"
    assert scales.shape == (K // 32, N), f"scales shape mismatch: expected ({K // 32}, {N}), got {scales.shape}"

    # Wrap as single-element pointer arrays to reuse the batch kernel
    packed_ptr = torch.tensor([packed.data_ptr()], dtype=torch.int64, device=packed.device)
    scale_ptr = torch.tensor([scales.data_ptr()], dtype=torch.int64, device=scales.device)

    # Reshape output to [1, N, K] for the batch kernel
    output_3d = output.unsqueeze(0)

    cute_mod = _get_cute_module()
    cute_mod.batch_mxfp4_dequant_cute_impl(
        packed_ptr,
        scale_ptr,
        output_3d,
        packed.stride(0),  # stride_packed_n
        packed.stride(1),  # stride_packed_k
        scales.stride(0),  # stride_scale_k (K-major)
        scales.stride(1),  # stride_scale_n
        kernel_version,
    )


def batch_mxfp4_dequant_cute(
    packed_ptrs: torch.Tensor,    # [num_experts] int64
    scale_ptrs: torch.Tensor,     # [num_experts] int64
    output: torch.Tensor,         # [num_experts, N, K] BF16
    packed_ref: torch.Tensor,     # Reference tensor for strides [N, K//2]
    scale_ref: torch.Tensor,      # Reference tensor for strides [K//32, N] (K-major!)
    kernel_version: int = 0,      # 0=simple, 1=swizzle
) -> None:
    """CuTe-style batch dequantize all experts' MXFP4 weights.

    Key differences from Triton:
    - Vectorized 128-bit loads/stores
    - Shared memory LUT (single-cycle FP4 decode, no ALU)
    - Hardware ldexp intrinsic
    - K-major scale layout for coalesced access

    Args:
        packed_ptrs: Pointer array to packed FP4 weights [num_experts]
        scale_ptrs: Pointer array to K-major transposed scales [num_experts]
        output: Pre-allocated output buffer [num_experts, N, K] BF16
        packed_ref: Reference weight tensor for computing strides [N, K//2]
        scale_ref: Reference K-major scale tensor for strides [K//32, N]
        kernel_version: 0=simple vectorized, 1=swizzled smem
    """
    cute_mod = _get_cute_module()
    cute_mod.batch_mxfp4_dequant_cute_impl(
        packed_ptrs,
        scale_ptrs,
        output,
        packed_ref.stride(0),
        packed_ref.stride(1),
        scale_ref.stride(0),  # K-major: stride along K//32 dimension
        scale_ref.stride(1),  # stride along N dimension
        kernel_version,
    )


if __name__ == "__main__":
    import time

    print("CuTe-Style MXFP4 Batch Dequantization")
    print("=" * 60)

    # Compile
    print("Compiling CuTe-style CUDA kernel...")
    _get_cute_module()
    print("Compilation successful!")

    # Test parameters (GPT-OSS-120B)
    num_experts = 128
    N = 13824
    K = 5120
    device = "cuda"

    # Create test data
    weights = [torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=device)
               for _ in range(num_experts)]
    # K-major scales: [K//32, N]
    scales = [torch.randint(120, 134, (K // 32, N), dtype=torch.uint8, device=device)
              for _ in range(num_experts)]

    # Pointer arrays
    weight_ptrs = torch.tensor([w.data_ptr() for w in weights],
                               dtype=torch.int64, device=device)
    scale_ptrs = torch.tensor([s.data_ptr() for s in scales],
                              dtype=torch.int64, device=device)

    # Output buffer
    output = torch.empty(num_experts, N, K, dtype=torch.bfloat16, device=device)

    # Benchmark
    kernel_names = ['simple', 'swizzle']
    warmup_iters = 5
    bench_iters = 20

    for version, name in enumerate(kernel_names):
        # Warmup
        for _ in range(warmup_iters):
            batch_mxfp4_dequant_cute(weight_ptrs, scale_ptrs, output,
                                     weights[0], scales[0], kernel_version=version)
        torch.cuda.synchronize()

        # Benchmark
        start = time.perf_counter()
        for _ in range(bench_iters):
            batch_mxfp4_dequant_cute(weight_ptrs, scale_ptrs, output,
                                     weights[0], scales[0], kernel_version=version)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) / bench_iters * 1000

        # Calculate bandwidth
        total_bytes = num_experts * (N * K // 2 + N * K // 32 + N * K * 2)
        bandwidth_gbps = (total_bytes / 1e9) / (elapsed_ms / 1000)
        hbm_pct = bandwidth_gbps / 4000 * 100  # H20 peak = 4 TB/s

        print(f"\n{name} kernel:")
        print(f"  Time: {elapsed_ms:.3f} ms")
        print(f"  Bandwidth: {bandwidth_gbps:.1f} GB/s ({hbm_pct:.1f}% HBM)")
