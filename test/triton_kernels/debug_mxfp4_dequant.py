#!/usr/bin/env python3
"""Diagnostic script to debug MXFP4 dequantization.

This script isolates the dequantization step to find where
the discrepancy between Triton and PyTorch occurs.
"""

import sys
sys.path.insert(0, "/Users/andrew/Desktop/MS application/Documentations/MoE-Gen/BatchGen")

import torch
import triton
import triton.language as tl


# FP4 lookup table (same as reference)
FP4_TABLE = torch.tensor([
    +0.0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=torch.float32)


def reference_dequant_single_block(packed_bytes: torch.Tensor, scale_uint8: int) -> torch.Tensor:
    """Reference: dequantize a single block of 16 packed bytes with one scale.

    Args:
        packed_bytes: [16] uint8 tensor (16 bytes = 32 FP4 values)
        scale_uint8: Single uint8 scale value

    Returns:
        [32] BF16 tensor of dequantized values
    """
    device = packed_bytes.device
    lut = FP4_TABLE.to(device)

    # Unpack nibbles
    idx_lo = (packed_bytes & 0x0F).long()  # [16] - even positions
    idx_hi = (packed_bytes >> 4).long()     # [16] - odd positions

    # Lookup FP4 values
    val_lo = lut[idx_lo]  # [16] float32
    val_hi = lut[idx_hi]  # [16] float32

    # Interleave: even positions get low nibble, odd get high nibble
    output = torch.zeros(32, dtype=torch.float32, device=device)
    output[0::2] = val_lo  # positions 0, 2, 4, ..., 30
    output[1::2] = val_hi  # positions 1, 3, 5, ..., 31

    # Apply scale: exponent = scale_uint8 - 127
    exponent = int(scale_uint8) - 127
    output = torch.ldexp(output, torch.tensor(exponent, device=device))

    return output.to(torch.bfloat16)


@triton.jit
def fp4_lookup_triton(idx):
    """Triton FP4 lookup."""
    sign = tl.where(idx >= 8, -1.0, 1.0)
    mag_idx = idx & 0x07

    mag = tl.where(mag_idx == 0, 0.0, 0.0)
    mag = tl.where(mag_idx == 1, 0.5, mag)
    mag = tl.where(mag_idx == 2, 1.0, mag)
    mag = tl.where(mag_idx == 3, 1.5, mag)
    mag = tl.where(mag_idx == 4, 2.0, mag)
    mag = tl.where(mag_idx == 5, 3.0, mag)
    mag = tl.where(mag_idx == 6, 4.0, mag)
    mag = tl.where(mag_idx == 7, 6.0, mag)

    return (sign * mag).to(tl.float32)


@triton.jit
def ldexp_triton(mantissa, exponent):
    """Triton ldexp."""
    exp_clamped = tl.minimum(tl.maximum(exponent, -126), 127)
    exp_bits = (exp_clamped + 127).to(tl.int32) << 23
    power_of_2 = exp_bits.to(tl.float32, bitcast=True)
    return mantissa * power_of_2


@triton.jit
def dequant_debug_kernel(
    packed_ptr,
    scale_value,  # single int32 value
    output_ptr,
    BLOCK_SIZE: tl.constexpr,  # 16
):
    """Debug kernel: dequantize 16 bytes with one scale."""
    offs = tl.arange(0, BLOCK_SIZE)

    # Load packed bytes
    packed = tl.load(packed_ptr + offs).to(tl.uint8)

    # Extract nibbles
    idx_lo = (packed & 0x0F).to(tl.int32)
    idx_hi = ((packed >> 4) & 0x0F).to(tl.int32)

    # Lookup FP4 values
    val_lo = fp4_lookup_triton(idx_lo)  # [16] float32
    val_hi = fp4_lookup_triton(idx_hi)  # [16] float32

    # Apply ldexp
    exponent = scale_value - 127
    val_lo_scaled = ldexp_triton(val_lo, exponent)
    val_hi_scaled = ldexp_triton(val_hi, exponent)

    # Store interleaved: even positions get lo, odd get hi
    # Output positions: 0, 2, 4, ... for lo; 1, 3, 5, ... for hi
    out_offs_lo = offs * 2      # [0, 2, 4, ..., 30]
    out_offs_hi = offs * 2 + 1  # [1, 3, 5, ..., 31]

    tl.store(output_ptr + out_offs_lo, val_lo_scaled.to(tl.bfloat16))
    tl.store(output_ptr + out_offs_hi, val_hi_scaled.to(tl.bfloat16))


def triton_dequant_single_block(packed_bytes: torch.Tensor, scale_uint8: int) -> torch.Tensor:
    """Triton: dequantize a single block of 16 packed bytes with one scale."""
    device = packed_bytes.device
    output = torch.empty(32, dtype=torch.bfloat16, device=device)

    dequant_debug_kernel[(1,)](
        packed_bytes,
        scale_uint8,
        output,
        BLOCK_SIZE=16,
    )

    return output


def test_single_block():
    """Test dequantization of a single block."""
    print("=" * 60)
    print("TEST 1: Single block dequantization (16 bytes, 1 scale)")
    print("=" * 60)

    device = "cuda"

    # Test with specific values
    # Packed: bytes 0-15 with known values
    packed = torch.tensor([
        0x10,  # low=0 (0.0), high=1 (0.5)
        0x32,  # low=2 (1.0), high=3 (1.5)
        0x54,  # low=4 (2.0), high=5 (3.0)
        0x76,  # low=6 (4.0), high=7 (6.0)
        0x98,  # low=8 (-0.0), high=9 (-0.5)
        0xBA,  # low=10 (-1.0), high=11 (-1.5)
        0xDC,  # low=12 (-2.0), high=13 (-3.0)
        0xFE,  # low=14 (-4.0), high=15 (-6.0)
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    ], dtype=torch.uint8, device=device)

    scale = 127  # exponent = 0, so 2^0 = 1 (no scaling)

    ref_output = reference_dequant_single_block(packed, scale)
    triton_output = triton_dequant_single_block(packed, scale)

    print(f"Scale: {scale} (exponent = {scale - 127})")
    print(f"\nReference output (first 16 values):")
    print(f"  {ref_output[:16].tolist()}")
    print(f"\nTriton output (first 16 values):")
    print(f"  {triton_output[:16].tolist()}")

    diff = (ref_output.float() - triton_output.float()).abs()
    max_diff = diff.max().item()

    print(f"\nMax absolute diff: {max_diff}")

    if max_diff < 1e-5:
        print("✓ PASSED: Single block dequantization matches")
        return True
    else:
        print("✗ FAILED: Single block dequantization differs")
        print(f"\nDiff per element: {diff.tolist()}")
        return False


def test_random_blocks():
    """Test with random packed values and scales."""
    print("\n" + "=" * 60)
    print("TEST 2: Random blocks (various scales)")
    print("=" * 60)

    device = "cuda"
    all_passed = True

    for scale in [100, 120, 127, 134, 150]:
        exponent = scale - 127
        print(f"\n--- Scale={scale}, exponent={exponent}, 2^exp={2**exponent:.6g} ---")

        # Random packed bytes
        packed = torch.randint(0, 256, (16,), dtype=torch.uint8, device=device)

        ref_output = reference_dequant_single_block(packed, scale)
        triton_output = triton_dequant_single_block(packed, scale)

        diff = (ref_output.float() - triton_output.float()).abs()
        max_diff = diff.max().item()

        # Compute relative error
        ref_abs = ref_output.float().abs()
        rel_error = (diff / ref_abs.clamp(min=1e-10)).max().item()

        print(f"Max abs diff: {max_diff:.6g}, Max rel error: {rel_error:.6g}")

        if max_diff < 1e-5:
            print("✓ PASSED")
        else:
            print("✗ FAILED")
            all_passed = False
            # Print first few values for debugging
            print(f"Packed bytes: {packed[:8].tolist()}")
            print(f"Ref[:8]:    {ref_output[:8].tolist()}")
            print(f"Triton[:8]: {triton_output[:8].tolist()}")

    return all_passed


def test_full_dequant():
    """Test full tensor dequantization matching the test setup."""
    print("\n" + "=" * 60)
    print("TEST 3: Full tensor dequantization (N=64, K=128)")
    print("=" * 60)

    from batchgen.quantization.mxfp4 import mxfp4_dequantize

    device = "cuda"
    N, K = 64, 128
    K_packed = K // 2
    K_scales = K // 32

    # Generate test data
    packed = torch.randint(0, 256, (N, K_packed), dtype=torch.uint8, device=device)
    scales = torch.randint(120, 134, (N, K_scales), dtype=torch.uint8, device=device)

    # Reference dequantization
    ref_output = mxfp4_dequantize(packed, scales, dtype=torch.bfloat16)

    print(f"packed shape: {packed.shape}")
    print(f"scales shape: {scales.shape}")
    print(f"ref_output shape: {ref_output.shape}")
    print(f"ref_output stats: min={ref_output.min():.4f}, max={ref_output.max():.4f}, mean={ref_output.mean():.4f}")

    # Now test: manually dequant first row and compare
    print("\n--- Manual verification of first row ---")

    # First K block (K=0..31) of first row (N=0)
    packed_block = packed[0, 0:16]  # 16 bytes for K positions 0-31
    scale_value = scales[0, 0].item()  # Scale for K block 0

    manual_output = reference_dequant_single_block(packed_block, scale_value)
    ref_first_block = ref_output[0, 0:32]

    diff = (manual_output.float() - ref_first_block.float()).abs()
    max_diff = diff.max().item()

    print(f"First K block (K=0..31) of row 0:")
    print(f"  Scale: {scale_value} (exp={scale_value-127})")
    print(f"  Manual[:8]:    {manual_output[:8].tolist()}")
    print(f"  Reference[:8]: {ref_first_block[:8].tolist()}")
    print(f"  Max diff: {max_diff}")

    if max_diff < 1e-5:
        print("✓ Manual verification passed - reference dequant is correct")
    else:
        print("✗ Manual verification failed - check reference implementation")
        return False

    return True


def test_gemm_isolated():
    """Test GEMM with pre-dequantized weights to isolate GEMM from dequant."""
    print("\n" + "=" * 60)
    print("TEST 4: Isolated GEMM (pre-dequantized weights)")
    print("=" * 60)

    from batchgen.quantization.mxfp4 import mxfp4_dequantize
    from batchgen.triton_kernels import fused_mxfp4_gemm

    device = "cuda"
    M, N, K = 4, 64, 128  # Small sizes for debugging
    K_packed = K // 2
    K_scales = K // 32

    # Generate test data
    x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    packed = torch.randint(0, 256, (N, K_packed), dtype=torch.uint8, device=device)
    scales = torch.randint(125, 130, (N, K_scales), dtype=torch.uint8, device=device)  # Small exponents
    bias = torch.randn(N, dtype=torch.bfloat16, device=device)

    # Reference: dequant then matmul
    weight_bf16 = mxfp4_dequantize(packed, scales, dtype=torch.bfloat16)
    ref_output = torch.mm(x, weight_bf16.T) + bias

    # Triton: fused dequant + matmul
    triton_output = fused_mxfp4_gemm(x, packed, scales, bias)

    # Compare
    diff = (ref_output.float() - triton_output.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    # Relative error
    ref_abs = ref_output.abs()
    rel_error = diff / ref_abs.clamp(min=1e-6)
    max_rel = rel_error.max().item()

    print(f"Shapes: x={x.shape}, weight={weight_bf16.shape}, output={ref_output.shape}")
    print(f"Output stats: ref_std={ref_output.float().std():.4f}, max_abs={ref_output.abs().max():.4f}")
    print(f"Max abs diff: {max_diff:.6f}")
    print(f"Mean abs diff: {mean_diff:.6f}")
    print(f"Max relative error: {max_rel:.6f} ({max_rel*100:.4f}%)")

    # Print some values for debugging
    print(f"\nFirst row comparison:")
    print(f"  Ref[0,:8]:    {ref_output[0,:8].tolist()}")
    print(f"  Triton[0,:8]: {triton_output[0,:8].tolist()}")

    if max_rel < 0.01:  # 1% relative tolerance
        print("\n✓ PASSED: GEMM output matches within 1% relative tolerance")
        return True
    else:
        print(f"\n✗ FAILED: GEMM output differs beyond 1% relative tolerance")

        # Find worst element
        flat_rel = rel_error.view(-1)
        worst_idx = flat_rel.argmax().item()
        worst_i, worst_j = worst_idx // N, worst_idx % N
        print(f"Worst element at [{worst_i}, {worst_j}]:")
        print(f"  Ref: {ref_output[worst_i, worst_j].item():.6f}")
        print(f"  Triton: {triton_output[worst_i, worst_j].item():.6f}")
        print(f"  Rel error: {rel_error[worst_i, worst_j].item():.6f}")

        return False


def main():
    if not torch.cuda.is_available():
        print("ERROR: CUDA required")
        return

    print(f"Device: {torch.cuda.get_device_name(0)}")

    passed = True
    passed = test_single_block() and passed
    passed = test_random_blocks() and passed
    passed = test_full_dequant() and passed
    passed = test_gemm_isolated() and passed

    print("\n" + "=" * 60)
    if passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED - See above for details")
    print("=" * 60)


if __name__ == "__main__":
    main()
