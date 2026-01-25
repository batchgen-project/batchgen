#!/usr/bin/env python3
"""Sanity check and benchmark for fused MXFP4 GEMM Triton kernel.

This script verifies that the Triton kernel produces correct results
compared to the PyTorch reference implementation, and benchmarks
the speedup.

Usage:
    python test/triton_kernels/test_fused_mxfp4_gemm.py

    # Run only sanity check:
    python test/triton_kernels/test_fused_mxfp4_gemm.py --sanity-only

    # Run only benchmark:
    python test/triton_kernels/test_fused_mxfp4_gemm.py --benchmark-only
"""

import argparse
import sys
import time
from typing import Tuple

import torch

# Add BatchGen to path
sys.path.insert(0, "/Users/andrew/Desktop/MS application/Documentations/MoE-Gen/BatchGen")


def generate_mxfp4_weights(
    N: int,
    K: int,
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate random MXFP4 weights for testing.

    Args:
        N: Output features
        K: Input features
        device: Target device

    Returns:
        Tuple of (packed_weights, scales)
        - packed_weights: [N, K//2] uint8
        - scales: [N, K//32] uint8
    """
    # Generate random packed FP4 values
    # Each byte has 2 FP4 values (4 bits each, values 0-15)
    K_packed = K // 2
    packed = torch.randint(0, 256, (N, K_packed), dtype=torch.uint8, device=device)

    # Generate random scales (uint8)
    # Scale exponent = scale_uint8 - 127
    # Use range 120-134 for exponents around -7 to +7, giving reasonable magnitudes
    K_scales = K // 32
    scales = torch.randint(120, 134, (N, K_scales), dtype=torch.uint8, device=device)

    return packed, scales


def reference_mxfp4_gemm(
    x: torch.Tensor,
    weight_packed: torch.Tensor,
    weight_scales: torch.Tensor,
    bias: torch.Tensor = None,
) -> torch.Tensor:
    """Reference implementation: dequantize + matmul (unfused PyTorch).

    This is the slow but correct reference implementation.
    """
    from batchgen.quantization.mxfp4 import mxfp4_dequantize

    # Dequantize to BF16
    weight_bf16 = mxfp4_dequantize(weight_packed, weight_scales, dtype=torch.bfloat16)

    # Reshape x to 2D
    x_2d = x.view(-1, x.shape[-1])

    # Standard matmul
    output = torch.mm(x_2d, weight_bf16.T)

    if bias is not None:
        output = output + bias

    # Reshape back
    output = output.view(*x.shape[:-1], -1)

    return output


def sanity_check_single_gemm(
    M: int = 32,
    N: int = 2880,
    K: int = 2880,
    rtol: float = 1e-2,
) -> bool:
    """Test single GEMM correctness: Triton vs PyTorch reference.

    Args:
        M: Batch size (number of tokens)
        N: Output features
        K: Input features
        rtol: Relative tolerance for comparison

    Returns:
        True if test passes, False otherwise
    """
    from batchgen.triton_kernels import fused_mxfp4_gemm

    print(f"\n=== Sanity Check: Single GEMM (M={M}, N={N}, K={K}) ===")

    device = "cuda"

    # Generate test inputs
    x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    weight_packed, weight_scales = generate_mxfp4_weights(N, K, device)
    bias = torch.randn(N, dtype=torch.bfloat16, device=device)

    # Reference implementation (unfused)
    torch.cuda.synchronize()
    ref_output = reference_mxfp4_gemm(x, weight_packed, weight_scales, bias)
    torch.cuda.synchronize()

    # Triton implementation (fused)
    torch.cuda.synchronize()
    triton_output = fused_mxfp4_gemm(x, weight_packed, weight_scales, bias)
    torch.cuda.synchronize()

    # Compare outputs using relative error
    # BF16 has ~3 decimal digits of precision, so rtol=1e-2 is reasonable
    ref_abs = ref_output.abs()
    diff_abs = (ref_output - triton_output).abs()

    # Compute relative error where ref is non-zero
    # Use max(|ref|, epsilon) to avoid division by zero
    epsilon = 1e-6
    rel_error = diff_abs / torch.clamp(ref_abs, min=epsilon)

    max_rel_error = rel_error.max().item()
    mean_rel_error = rel_error.mean().item()
    max_abs_diff = diff_abs.max().item()
    ref_std = ref_output.float().std().item()
    ref_max = ref_output.abs().max().item()

    print(f"Reference output: std={ref_std:.4f}, max_abs={ref_max:.4f}")
    print(f"Max absolute diff: {max_abs_diff:.6f}")
    print(f"Max relative error: {max_rel_error:.6f} ({max_rel_error*100:.4f}%)")
    print(f"Mean relative error: {mean_rel_error:.6f} ({mean_rel_error*100:.4f}%)")

    # Check with relative tolerance
    passed = max_rel_error < rtol

    if passed:
        print(f"✓ PASSED: Max relative error {max_rel_error:.6f} < rtol={rtol}")
        return True
    else:
        print(f"✗ FAILED: Max relative error {max_rel_error:.6f} >= rtol={rtol}")
        # Print some sample values for debugging
        print(f"Reference[:5,:5]:\n{ref_output[:5,:5]}")
        print(f"Triton[:5,:5]:\n{triton_output[:5,:5]}")
        # Find worst case
        worst_idx = torch.argmax(rel_error.view(-1)).item()
        worst_i, worst_j = worst_idx // N, worst_idx % N
        print(f"Worst element at [{worst_i}, {worst_j}]: ref={ref_output[worst_i, worst_j]:.6f}, "
              f"triton={triton_output[worst_i, worst_j]:.6f}")
        return False


def openai_swiglu(gate: torch.Tensor, up: torch.Tensor, alpha: float = 1.702, limit: float = 7.0) -> torch.Tensor:
    """OpenAI's SwiGLU activation: gate * sigmoid(alpha * gate) * (up + 1)"""
    gate_clamped = gate.clamp(max=limit)
    up_clamped = up.clamp(min=-limit, max=limit)
    glu = gate_clamped * torch.sigmoid(alpha * gate_clamped)
    return glu * (up_clamped + 1)


def sanity_check_mlp_forward(
    M: int = 32,
    hidden_size: int = 2880,
    intermediate_size: int = 2880,
    rtol: float = 5e-2,  # 5% relative tolerance - MLP has 3 GEMMs + activation
) -> bool:
    """Test full MLP forward: Triton vs PyTorch reference.

    This tests gate, up, down projections with SwiGLU activation.
    MLP accumulates more error (3 GEMMs + nonlinear activation), so use looser tolerance.
    """
    from batchgen.triton_kernels import fused_mxfp4_mlp_forward

    print(f"\n=== Sanity Check: Full MLP (M={M}, hidden={hidden_size}, intermediate={intermediate_size}) ===")

    device = "cuda"

    # Generate test inputs
    x = torch.randn(M, hidden_size, dtype=torch.bfloat16, device=device)

    # Generate MLP weights
    gate_packed, gate_scales = generate_mxfp4_weights(intermediate_size, hidden_size, device)
    up_packed, up_scales = generate_mxfp4_weights(intermediate_size, hidden_size, device)
    down_packed, down_scales = generate_mxfp4_weights(hidden_size, intermediate_size, device)

    gate_bias = torch.randn(intermediate_size, dtype=torch.bfloat16, device=device)
    up_bias = torch.randn(intermediate_size, dtype=torch.bfloat16, device=device)
    down_bias = torch.randn(hidden_size, dtype=torch.bfloat16, device=device)

    # Reference implementation (unfused)
    torch.cuda.synchronize()
    ref_gate = reference_mxfp4_gemm(x, gate_packed, gate_scales, gate_bias)
    ref_up = reference_mxfp4_gemm(x, up_packed, up_scales, up_bias)
    ref_intermediate = openai_swiglu(ref_gate, ref_up, alpha=1.702, limit=7.0)
    ref_output = reference_mxfp4_gemm(ref_intermediate, down_packed, down_scales, down_bias)
    torch.cuda.synchronize()

    # Triton implementation (fused)
    torch.cuda.synchronize()
    triton_output = fused_mxfp4_mlp_forward(
        x,
        gate_packed, gate_scales, gate_bias,
        up_packed, up_scales, up_bias,
        down_packed, down_scales, down_bias,
        alpha=1.702,
        limit=7.0,
    )
    torch.cuda.synchronize()

    # Compare using relative error
    ref_abs = ref_output.abs()
    diff_abs = (ref_output - triton_output).abs()
    epsilon = 1e-6
    rel_error = diff_abs / torch.clamp(ref_abs, min=epsilon)

    max_rel_error = rel_error.max().item()
    mean_rel_error = rel_error.mean().item()
    max_abs_diff = diff_abs.max().item()
    ref_std = ref_output.float().std().item()
    ref_max = ref_output.abs().max().item()

    print(f"Reference output: std={ref_std:.4f}, max_abs={ref_max:.4f}")
    print(f"Max absolute diff: {max_abs_diff:.6f}")
    print(f"Max relative error: {max_rel_error:.6f} ({max_rel_error*100:.4f}%)")
    print(f"Mean relative error: {mean_rel_error:.6f} ({mean_rel_error*100:.4f}%)")

    passed = max_rel_error < rtol

    if passed:
        print(f"✓ PASSED: Max relative error {max_rel_error:.6f} < rtol={rtol}")
        return True
    else:
        print(f"✗ FAILED: Max relative error {max_rel_error:.6f} >= rtol={rtol}")
        return False


def benchmark_single_gemm(
    M: int = 32,
    N: int = 2880,
    K: int = 2880,
    warmup_iters: int = 10,
    bench_iters: int = 100,
) -> Tuple[float, float, float]:
    """Benchmark single GEMM: Triton vs PyTorch reference.

    Returns:
        Tuple of (reference_ms, triton_ms, speedup)
    """
    from batchgen.triton_kernels import fused_mxfp4_gemm

    print(f"\n=== Benchmark: Single GEMM (M={M}, N={N}, K={K}) ===")

    device = "cuda"

    # Generate test inputs
    x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    weight_packed, weight_scales = generate_mxfp4_weights(N, K, device)
    bias = torch.randn(N, dtype=torch.bfloat16, device=device)

    # Warmup - Reference
    for _ in range(warmup_iters):
        _ = reference_mxfp4_gemm(x, weight_packed, weight_scales, bias)
    torch.cuda.synchronize()

    # Benchmark - Reference
    start = time.perf_counter()
    for _ in range(bench_iters):
        _ = reference_mxfp4_gemm(x, weight_packed, weight_scales, bias)
    torch.cuda.synchronize()
    ref_time = (time.perf_counter() - start) / bench_iters * 1000

    # Warmup - Triton
    for _ in range(warmup_iters):
        _ = fused_mxfp4_gemm(x, weight_packed, weight_scales, bias)
    torch.cuda.synchronize()

    # Benchmark - Triton
    start = time.perf_counter()
    for _ in range(bench_iters):
        _ = fused_mxfp4_gemm(x, weight_packed, weight_scales, bias)
    torch.cuda.synchronize()
    triton_time = (time.perf_counter() - start) / bench_iters * 1000

    speedup = ref_time / triton_time

    print(f"Reference (unfused): {ref_time:.3f} ms")
    print(f"Triton (fused):      {triton_time:.3f} ms")
    print(f"Speedup:             {speedup:.2f}x")

    return ref_time, triton_time, speedup


def benchmark_mlp_forward(
    M: int = 32,
    hidden_size: int = 2880,
    intermediate_size: int = 2880,
    warmup_iters: int = 10,
    bench_iters: int = 100,
) -> Tuple[float, float, float]:
    """Benchmark full MLP: Triton vs PyTorch reference.

    Returns:
        Tuple of (reference_ms, triton_ms, speedup)
    """
    from batchgen.triton_kernels import fused_mxfp4_mlp_forward

    print(f"\n=== Benchmark: Full MLP (M={M}, hidden={hidden_size}, intermediate={intermediate_size}) ===")

    device = "cuda"

    # Generate test inputs
    x = torch.randn(M, hidden_size, dtype=torch.bfloat16, device=device)

    # Generate MLP weights
    gate_packed, gate_scales = generate_mxfp4_weights(intermediate_size, hidden_size, device)
    up_packed, up_scales = generate_mxfp4_weights(intermediate_size, hidden_size, device)
    down_packed, down_scales = generate_mxfp4_weights(hidden_size, intermediate_size, device)

    gate_bias = torch.randn(intermediate_size, dtype=torch.bfloat16, device=device)
    up_bias = torch.randn(intermediate_size, dtype=torch.bfloat16, device=device)
    down_bias = torch.randn(hidden_size, dtype=torch.bfloat16, device=device)

    def reference_mlp():
        ref_gate = reference_mxfp4_gemm(x, gate_packed, gate_scales, gate_bias)
        ref_up = reference_mxfp4_gemm(x, up_packed, up_scales, up_bias)
        ref_intermediate = openai_swiglu(ref_gate, ref_up, alpha=1.702, limit=7.0)
        return reference_mxfp4_gemm(ref_intermediate, down_packed, down_scales, down_bias)

    def triton_mlp():
        return fused_mxfp4_mlp_forward(
            x,
            gate_packed, gate_scales, gate_bias,
            up_packed, up_scales, up_bias,
            down_packed, down_scales, down_bias,
            alpha=1.702,
            limit=7.0,
        )

    # Warmup - Reference
    for _ in range(warmup_iters):
        _ = reference_mlp()
    torch.cuda.synchronize()

    # Benchmark - Reference
    start = time.perf_counter()
    for _ in range(bench_iters):
        _ = reference_mlp()
    torch.cuda.synchronize()
    ref_time = (time.perf_counter() - start) / bench_iters * 1000

    # Warmup - Triton
    for _ in range(warmup_iters):
        _ = triton_mlp()
    torch.cuda.synchronize()

    # Benchmark - Triton
    start = time.perf_counter()
    for _ in range(bench_iters):
        _ = triton_mlp()
    torch.cuda.synchronize()
    triton_time = (time.perf_counter() - start) / bench_iters * 1000

    speedup = ref_time / triton_time

    print(f"Reference (unfused): {ref_time:.3f} ms")
    print(f"Triton (fused):      {triton_time:.3f} ms")
    print(f"Speedup:             {speedup:.2f}x")

    return ref_time, triton_time, speedup


def run_sanity_checks() -> bool:
    """Run all sanity checks.

    Returns:
        True if all tests pass.
    """
    print("\n" + "=" * 60)
    print("SANITY CHECKS")
    print("=" * 60)

    all_passed = True

    # Test various sizes for single GEMM
    # Using 1% relative tolerance for single GEMM (BF16 has ~0.4% precision)
    sizes = [
        (1, 2880, 2880),     # Single token
        (32, 2880, 2880),    # Small batch (decode)
        (128, 2880, 2880),   # Medium batch
        (512, 2880, 2880),   # Large batch (prefill)
    ]

    for M, N, K in sizes:
        passed = sanity_check_single_gemm(M, N, K, rtol=0.01)
        all_passed = all_passed and passed

    # Test full MLP (3 GEMMs + activation - use 5% tolerance)
    for M in [1, 32, 128]:
        passed = sanity_check_mlp_forward(M, 2880, 2880, rtol=0.05)
        all_passed = all_passed and passed

    return all_passed


def run_benchmarks():
    """Run all benchmarks."""
    print("\n" + "=" * 60)
    print("BENCHMARKS")
    print("=" * 60)

    results = []

    # Benchmark various sizes
    sizes = [
        (1, 2880, 2880),     # Single token (decode)
        (32, 2880, 2880),    # Small batch
        (128, 2880, 2880),   # Medium batch
        (512, 2880, 2880),   # Large batch (prefill)
    ]

    for M, N, K in sizes:
        ref_ms, triton_ms, speedup = benchmark_single_gemm(M, N, K)
        results.append(("GEMM", M, N, K, ref_ms, triton_ms, speedup))

    # Benchmark full MLP
    for M in [1, 32, 128, 512]:
        ref_ms, triton_ms, speedup = benchmark_mlp_forward(M, 2880, 2880)
        results.append(("MLP", M, 2880, 2880, ref_ms, triton_ms, speedup))

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Type':<6} {'M':>6} {'N':>6} {'K':>6} {'Ref(ms)':>10} {'Triton(ms)':>12} {'Speedup':>8}")
    print("-" * 60)
    for type_, M, N, K, ref_ms, triton_ms, speedup in results:
        print(f"{type_:<6} {M:>6} {N:>6} {K:>6} {ref_ms:>10.3f} {triton_ms:>12.3f} {speedup:>7.2f}x")


def main():
    parser = argparse.ArgumentParser(description="Test fused MXFP4 GEMM Triton kernel")
    parser.add_argument("--sanity-only", action="store_true", help="Run only sanity checks")
    parser.add_argument("--benchmark-only", action="store_true", help="Run only benchmarks")
    args = parser.parse_args()

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This test requires a GPU.")
        sys.exit(1)

    print(f"Using device: {torch.cuda.get_device_name(0)}")

    if args.benchmark_only:
        run_benchmarks()
    elif args.sanity_only:
        passed = run_sanity_checks()
        sys.exit(0 if passed else 1)
    else:
        # Run both
        passed = run_sanity_checks()
        if passed:
            run_benchmarks()
        else:
            print("\nSkipping benchmarks due to sanity check failures.")
            sys.exit(1)


if __name__ == "__main__":
    main()
