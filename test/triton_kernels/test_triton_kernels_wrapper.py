#!/usr/bin/env python3
"""Sanity check for triton_kernels MXFP4 GEMM wrapper.

Compares triton_kernels.matmul output against PyTorch reference.

Usage:
    python test/triton_kernels/test_triton_kernels_wrapper.py
"""

import sys
sys.path.insert(0, "/Users/andrew/Desktop/MS application/Documentations/MoE-Gen/BatchGen")

import torch
from batchgen.quantization.mxfp4 import mxfp4_dequantize

# Check if triton_kernels is available
try:
    from batchgen.triton_kernels.triton_kernels_mxfp4_gemm import (
        triton_kernels_mxfp4_gemm,
        TRITON_KERNELS_AVAILABLE
    )
except ImportError:
    TRITON_KERNELS_AVAILABLE = False


def reference_gemm(x, packed, scales, bias=None):
    """PyTorch reference: dequant then matmul."""
    weight_bf16 = mxfp4_dequantize(packed, scales, dtype=torch.bfloat16)
    output = torch.mm(x, weight_bf16.T)
    if bias is not None:
        output = output + bias
    return output


def test_triton_kernels_single_gemm():
    """Test triton_kernels GEMM matches reference."""
    print("=" * 60)
    print("TEST 1: triton_kernels Single GEMM (small)")
    print("=" * 60)

    if not TRITON_KERNELS_AVAILABLE:
        print("SKIPPED: triton_kernels not installed")
        print("Install with: pip install -e /path/to/triton/python/triton_kernels")
        return None

    device = "cuda"
    M, N, K = 4, 64, 128
    K_packed = K // 2
    K_scales = K // 32

    # Generate test data
    x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    packed = torch.randint(0, 256, (N, K_packed), dtype=torch.uint8, device=device)
    scales = torch.randint(120, 134, (N, K_scales), dtype=torch.uint8, device=device)
    bias = torch.randn(N, dtype=torch.bfloat16, device=device)

    print(f"Shapes: x={x.shape}, weight={packed.shape}, scales={scales.shape}, bias={bias.shape}")

    # Reference
    ref_output = reference_gemm(x, packed, scales, bias)

    # triton_kernels
    try:
        tk_output = triton_kernels_mxfp4_gemm(x, packed, scales, bias)
    except Exception as e:
        print(f"ERROR: triton_kernels_mxfp4_gemm failed: {e}")
        return False

    # Compare
    diff = (ref_output.float() - tk_output.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    # Relative error
    ref_abs = ref_output.abs().clamp(min=1e-6)
    rel_error = (diff / ref_abs).max().item()

    print(f"Output shape: {ref_output.shape}")
    print(f"Max abs diff: {max_diff:.6f}")
    print(f"Mean abs diff: {mean_diff:.6f}")
    print(f"Max relative error: {rel_error:.6f} ({rel_error*100:.4f}%)")

    print(f"\nFirst row comparison:")
    print(f"  Ref[:8]:    {ref_output[0,:8].tolist()}")
    print(f"  TK[:8]:     {tk_output[0,:8].tolist()}")

    if rel_error < 0.02:  # 2% tolerance for triton_kernels
        print("\n✓ PASSED: triton_kernels matches within 2% relative tolerance")
        return True
    else:
        print(f"\n✗ FAILED: triton_kernels differs beyond 2% relative tolerance")
        return False


def test_triton_kernels_no_bias():
    """Test without bias."""
    print("\n" + "=" * 60)
    print("TEST 2: triton_kernels GEMM (no bias)")
    print("=" * 60)

    if not TRITON_KERNELS_AVAILABLE:
        print("SKIPPED: triton_kernels not installed")
        return None

    device = "cuda"
    M, N, K = 8, 128, 256
    K_packed = K // 2
    K_scales = K // 32

    x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    packed = torch.randint(0, 256, (N, K_packed), dtype=torch.uint8, device=device)
    scales = torch.randint(120, 134, (N, K_scales), dtype=torch.uint8, device=device)

    print(f"Shapes: x={x.shape}, weight={packed.shape}, scales={scales.shape}")

    ref_output = reference_gemm(x, packed, scales, bias=None)

    try:
        tk_output = triton_kernels_mxfp4_gemm(x, packed, scales, bias=None)
    except Exception as e:
        print(f"ERROR: triton_kernels_mxfp4_gemm failed: {e}")
        return False

    diff = (ref_output.float() - tk_output.float()).abs()
    rel_error = (diff / ref_output.abs().clamp(min=1e-6)).max().item()

    print(f"Max relative error: {rel_error:.6f} ({rel_error*100:.4f}%)")

    if rel_error < 0.02:
        print("✓ PASSED")
        return True
    else:
        print("✗ FAILED")
        return False


def test_triton_kernels_large_gemm():
    """Test with larger dimensions (more realistic for LLMs)."""
    print("\n" + "=" * 60)
    print("TEST 3: triton_kernels Large GEMM (LLM-sized)")
    print("=" * 60)

    if not TRITON_KERNELS_AVAILABLE:
        print("SKIPPED: triton_kernels not installed")
        return None

    device = "cuda"
    # GPT-OSS-120B dimensions: hidden=2880, intermediate=5760
    M = 32  # Batch size
    N = 2880  # hidden_size (for down projection)
    K = 2880  # intermediate_size / 2 (after SwiGLU)

    K_packed = K // 2
    K_scales = K // 32

    x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    packed = torch.randint(0, 256, (N, K_packed), dtype=torch.uint8, device=device)
    scales = torch.randint(120, 134, (N, K_scales), dtype=torch.uint8, device=device)
    bias = torch.randn(N, dtype=torch.bfloat16, device=device)

    print(f"Shapes: x={x.shape}, weight={packed.shape}, scales={scales.shape}")

    ref_output = reference_gemm(x, packed, scales, bias)

    try:
        tk_output = triton_kernels_mxfp4_gemm(x, packed, scales, bias)
    except Exception as e:
        print(f"ERROR: triton_kernels_mxfp4_gemm failed: {e}")
        return False

    diff = (ref_output.float() - tk_output.float()).abs()
    max_diff = diff.max().item()
    rel_error = (diff / ref_output.abs().clamp(min=1e-6)).max().item()

    print(f"Max abs diff: {max_diff:.6f}")
    print(f"Max relative error: {rel_error:.6f} ({rel_error*100:.4f}%)")

    if rel_error < 0.02:
        print("✓ PASSED")
        return True
    else:
        print("✗ FAILED")
        return False


def test_triton_kernels_3d_weight():
    """Test with 3D weight tensor format [N, K//32, 16]."""
    print("\n" + "=" * 60)
    print("TEST 4: triton_kernels 3D Weight Format")
    print("=" * 60)

    if not TRITON_KERNELS_AVAILABLE:
        print("SKIPPED: triton_kernels not installed")
        return None

    device = "cuda"
    M, N, K = 4, 64, 128
    K_scales = K // 32  # 4 groups
    bytes_per_group = 16  # K//2 / (K//32) = 16 bytes per scale group

    x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    # 3D format: [N, num_groups, bytes_per_group]
    packed_3d = torch.randint(0, 256, (N, K_scales, bytes_per_group), dtype=torch.uint8, device=device)
    scales = torch.randint(120, 134, (N, K_scales), dtype=torch.uint8, device=device)

    print(f"Shapes: x={x.shape}, weight_3d={packed_3d.shape}, scales={scales.shape}")

    # Reference: flatten 3D to 2D
    packed_2d = packed_3d.view(N, K // 2)
    ref_output = reference_gemm(x, packed_2d, scales, bias=None)

    # triton_kernels wrapper should handle 3D internally
    try:
        tk_output = triton_kernels_mxfp4_gemm(x, packed_3d, scales, bias=None)
    except Exception as e:
        print(f"ERROR: triton_kernels_mxfp4_gemm failed with 3D input: {e}")
        return False

    diff = (ref_output.float() - tk_output.float()).abs()
    rel_error = (diff / ref_output.abs().clamp(min=1e-6)).max().item()

    print(f"Max relative error: {rel_error:.6f} ({rel_error*100:.4f}%)")

    if rel_error < 0.02:
        print("✓ PASSED")
        return True
    else:
        print("✗ FAILED")
        return False


def test_performance_comparison():
    """Compare performance of triton_kernels vs reference."""
    print("\n" + "=" * 60)
    print("TEST 5: Performance Comparison")
    print("=" * 60)

    if not TRITON_KERNELS_AVAILABLE:
        print("SKIPPED: triton_kernels not installed")
        return None

    device = "cuda"
    M, N, K = 32, 2880, 2880

    x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    packed = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=device)
    scales = torch.randint(120, 134, (N, K // 32), dtype=torch.uint8, device=device)
    bias = torch.randn(N, dtype=torch.bfloat16, device=device)

    # Warmup
    for _ in range(5):
        _ = reference_gemm(x, packed, scales, bias)
        _ = triton_kernels_mxfp4_gemm(x, packed, scales, bias)

    torch.cuda.synchronize()

    # Time reference (dequant + matmul)
    import time
    n_iters = 20

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_iters):
        _ = reference_gemm(x, packed, scales, bias)
    torch.cuda.synchronize()
    ref_time = (time.time() - start) / n_iters * 1000

    # Time triton_kernels
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_iters):
        _ = triton_kernels_mxfp4_gemm(x, packed, scales, bias)
    torch.cuda.synchronize()
    tk_time = (time.time() - start) / n_iters * 1000

    print(f"Reference (dequant + matmul): {ref_time:.3f} ms")
    print(f"triton_kernels:               {tk_time:.3f} ms")
    print(f"Speedup: {ref_time / tk_time:.2f}x")

    # No pass/fail for performance test
    return True


def main():
    if not torch.cuda.is_available():
        print("ERROR: CUDA required")
        return

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"triton_kernels available: {TRITON_KERNELS_AVAILABLE}")
    print()

    results = []
    results.append(("Single GEMM (small)", test_triton_kernels_single_gemm()))
    results.append(("No bias", test_triton_kernels_no_bias()))
    results.append(("Large GEMM", test_triton_kernels_large_gemm()))
    results.append(("3D weight format", test_triton_kernels_3d_weight()))
    results.append(("Performance", test_performance_comparison()))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = 0
    failed = 0
    skipped = 0

    for name, result in results:
        if result is None:
            status = "SKIPPED"
            skipped += 1
        elif result:
            status = "PASSED"
            passed += 1
        else:
            status = "FAILED"
            failed += 1
        print(f"  {name}: {status}")

    print()
    if failed == 0 and passed > 0:
        print(f"ALL {passed} TESTS PASSED")
    elif skipped == len(results):
        print("ALL TESTS SKIPPED (triton_kernels not installed)")
    else:
        print(f"RESULTS: {passed} passed, {failed} failed, {skipped} skipped")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
