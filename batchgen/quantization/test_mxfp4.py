"""Unit tests for MXFP4 dequantization.

Tests verify correctness against the OpenAI reference implementation.
"""

import torch
import pytest


def test_fp4_lookup_values():
    """Test that FP4 lookup table has correct values."""
    from mxfp4 import FP4_LOOKUP_TABLE

    expected = torch.tensor([
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
    ], dtype=torch.float32)

    assert torch.allclose(FP4_LOOKUP_TABLE, expected), \
        f"FP4 lookup table mismatch: {FP4_LOOKUP_TABLE} vs {expected}"


def test_unpack_nibbles():
    """Test unpacking of nibbles from packed bytes."""
    from mxfp4 import mxfp4_dequantize_reference

    # Create packed bytes with known values
    # 0x12 -> low=2 (1.0), high=1 (0.5)
    # 0x34 -> low=4 (2.0), high=3 (1.5)
    packed = torch.tensor([0x12, 0x34], dtype=torch.uint8)

    # Scale = 127 means exponent = 0, so values unchanged
    scales = torch.tensor([127], dtype=torch.uint8)

    result = mxfp4_dequantize_reference(packed, scales, dtype=torch.float32)

    # Expected: [1.0, 0.5, 2.0, 1.5] (interleaved low/high)
    expected = torch.tensor([1.0, 0.5, 2.0, 1.5], dtype=torch.float32)

    assert torch.allclose(result, expected), \
        f"Unpack mismatch: {result} vs {expected}"


def test_scale_application():
    """Test that scales are correctly applied as power-of-2 exponents."""
    from mxfp4 import mxfp4_dequantize_reference

    # All values = 1.0 (index 2)
    # 0x22 -> low=2 (1.0), high=2 (1.0)
    packed = torch.tensor([0x22] * 16, dtype=torch.uint8)  # 32 values (16 bytes)

    # Scale = 128 means exponent = 1, so values * 2
    scales = torch.tensor([128], dtype=torch.uint8)

    result = mxfp4_dequantize_reference(packed, scales, dtype=torch.float32)

    # All values should be 1.0 * 2^1 = 2.0
    expected = torch.full((32,), 2.0, dtype=torch.float32)

    assert torch.allclose(result, expected), \
        f"Scale application mismatch: {result} vs {expected}"


def test_scale_negative_exponent():
    """Test negative exponents (scale < 127)."""
    from mxfp4 import mxfp4_dequantize_reference

    # All values = 2.0 (index 4)
    # 0x44 -> low=4 (2.0), high=4 (2.0)
    packed = torch.tensor([0x44] * 16, dtype=torch.uint8)

    # Scale = 126 means exponent = -1, so values * 0.5
    scales = torch.tensor([126], dtype=torch.uint8)

    result = mxfp4_dequantize_reference(packed, scales, dtype=torch.float32)

    # All values should be 2.0 * 2^(-1) = 1.0
    expected = torch.full((32,), 1.0, dtype=torch.float32)

    assert torch.allclose(result, expected), \
        f"Negative exponent mismatch: {result} vs {expected}"


def test_negative_fp4_values():
    """Test negative FP4 values (indices 8-15)."""
    from mxfp4 import mxfp4_dequantize_reference

    # 0x9A -> low=10 (-1.0), high=9 (-0.5)
    packed = torch.tensor([0x9A] * 16, dtype=torch.uint8)

    # Neutral scale
    scales = torch.tensor([127], dtype=torch.uint8)

    result = mxfp4_dequantize_reference(packed, scales, dtype=torch.float32)

    # Values should alternate: -1.0, -0.5, -1.0, -0.5, ...
    expected = torch.tensor([-1.0, -0.5] * 16, dtype=torch.float32)

    assert torch.allclose(result, expected), \
        f"Negative values mismatch: {result} vs {expected}"


def test_multiple_scales():
    """Test with multiple scale blocks."""
    from mxfp4 import mxfp4_dequantize_reference

    # Two blocks of 32 values each (16 bytes each = 32 bytes total)
    # First block: all 1.0 with scale 128 (exp=1) -> 2.0
    # Second block: all 1.0 with scale 126 (exp=-1) -> 0.5
    packed = torch.tensor([0x22] * 32, dtype=torch.uint8)
    scales = torch.tensor([128, 126], dtype=torch.uint8)

    result = mxfp4_dequantize_reference(packed, scales, dtype=torch.float32)

    expected = torch.cat([
        torch.full((32,), 2.0, dtype=torch.float32),
        torch.full((32,), 0.5, dtype=torch.float32)
    ])

    assert torch.allclose(result, expected), \
        f"Multiple scales mismatch: {result} vs {expected}"


def test_2d_input():
    """Test with 2D input (batch dimension)."""
    from mxfp4 import mxfp4_dequantize_reference

    # 2 rows, each with 32 values
    packed = torch.tensor([
        [0x22] * 16,  # Row 0: all 1.0
        [0x44] * 16,  # Row 1: all 2.0
    ], dtype=torch.uint8)

    scales = torch.tensor([
        [127],  # Row 0: exp=0
        [128],  # Row 1: exp=1
    ], dtype=torch.uint8)

    result = mxfp4_dequantize_reference(packed, scales, dtype=torch.float32)

    expected = torch.stack([
        torch.full((32,), 1.0, dtype=torch.float32),   # 1.0 * 2^0
        torch.full((32,), 4.0, dtype=torch.float32),   # 2.0 * 2^1
    ])

    assert torch.allclose(result, expected), \
        f"2D input mismatch: {result} vs {expected}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_triton_vs_reference():
    """Test that Triton implementation matches reference."""
    from mxfp4 import mxfp4_dequantize_reference, mxfp4_dequantize_triton

    # Random test data
    torch.manual_seed(42)
    M, K = 64, 256
    packed = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device="cuda")
    scales = torch.randint(100, 154, (M, K // 32), dtype=torch.uint8, device="cuda")

    ref_result = mxfp4_dequantize_reference(packed, scales, dtype=torch.bfloat16)
    triton_result = mxfp4_dequantize_triton(packed, scales, dtype=torch.bfloat16)

    # Allow small tolerance for BF16
    assert torch.allclose(ref_result, triton_result, atol=1e-2, rtol=1e-2), \
        f"Triton vs reference mismatch. Max diff: {(ref_result - triton_result).abs().max()}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_dequantize_performance():
    """Benchmark dequantization performance."""
    from mxfp4 import mxfp4_dequantize

    # Simulate GPT-OSS-120B expert weights: ~8MB per expert
    M, K = 2880, 8640  # hidden_size x intermediate_size
    packed = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device="cuda")
    scales = torch.randint(100, 154, (M, K // 32), dtype=torch.uint8, device="cuda")

    # Warmup
    for _ in range(3):
        _ = mxfp4_dequantize(packed, scales)

    # Benchmark
    torch.cuda.synchronize()
    import time
    start = time.perf_counter()
    n_iters = 100
    for _ in range(n_iters):
        result = mxfp4_dequantize(packed, scales)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    ms_per_iter = elapsed / n_iters * 1000
    gb_per_sec = (M * K * 2) / (elapsed / n_iters) / 1e9  # BF16 = 2 bytes

    print(f"\nMXFP4 Dequantization Benchmark:")
    print(f"  Shape: {M} x {K}")
    print(f"  Time: {ms_per_iter:.3f} ms")
    print(f"  Throughput: {gb_per_sec:.2f} GB/s")


if __name__ == "__main__":
    print("Running MXFP4 dequantization tests...")

    # Run CPU tests
    test_fp4_lookup_values()
    print("  [PASS] FP4 lookup values")

    test_unpack_nibbles()
    print("  [PASS] Unpack nibbles")

    test_scale_application()
    print("  [PASS] Scale application")

    test_scale_negative_exponent()
    print("  [PASS] Negative exponent")

    test_negative_fp4_values()
    print("  [PASS] Negative FP4 values")

    test_multiple_scales()
    print("  [PASS] Multiple scales")

    test_2d_input()
    print("  [PASS] 2D input")

    # Run GPU tests if available
    if torch.cuda.is_available():
        test_triton_vs_reference()
        print("  [PASS] Triton vs reference")

        test_dequantize_performance()
    else:
        print("  [SKIP] GPU tests (CUDA not available)")

    print("\nAll tests passed!")
