# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                          #
#                                                                               #
#  licensed under the apache license, version 2.0 (the "license");              #
#  you may not use this file except in compliance with the license.             #
#                                                                               #
#  you may obtain a copy of the license at                                      #
#                                                                               #
#                  http://www.apache.org/licenses/license-2.0                   #
#                                                                               #
#  unless required by applicable law or agreed to in writing, software          #
#  distributed under the license is distributed on an "as is" basis,            #
#  without warranties or conditions of any kind, either express or implied.     #
#  see the license for the specific language governing permissions and          #
#  limitations under the license.                                               #
# ---------------------------------------------------------------------------- #

"""Test INT4 W4A16 dequantization matches compressed-tensors reference.

This test compares BatchGen's INT4 dequantization (batchgen/quantization/int4.py)
against the compressed-tensors library's pack-quantized format — the format used
by Kimi K2.5 checkpoints.

INT4 Format (compressed-tensors pack-quantized, symmetric):
- Group size: 32 INT4 values per scale
- Packing: 2 INT4 values per uint8 byte (low nibble first, high nibble second)
- Scales: bf16, one per group of 32 elements
- Symmetric: no zero-point
- Tensor names: .weight_packed (uint8), .weight_scale (bf16)

Sign encoding (compressed-tensors standard — OFFSET):
- Unsigned nibble [0, 15] → signed value = nibble - 8
- Range: [-8, +7]
- nibble 0 → -8, nibble 8 → 0, nibble 15 → +7

Dequantization: (nibble - 8) * bf16_scale → bf16 weight

Usage:
    python -m pytest tests/test_int4_dequantization.py -v
    or
    python tests/test_int4_dequantization.py
"""

import sys
import os

import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================ #
# Compressed-Tensors Reference Implementation
# ============================================================================ #
# Ported from compressed-tensors library (pack_quantized.py).
# This is the ground truth for K2.5 checkpoint INT4 dequantization.
#
# Key: uses OFFSET encoding (nibble - 8), not two's complement.
# Source: https://github.com/neuralmagic/compressed-tensors
#   compressed_tensors/quantization/lifecycle/forward.py
#   compressed_tensors/compressors/pack_quantized.py
# ============================================================================ #

INT4_GROUP_SIZE = 32


def compressed_tensors_int4_dequantize(packed_uint8, scales, group_size=32):
    """Reference INT4 dequant using offset encoding (compressed-tensors standard).

    This is the authoritative ground truth for K2.5 checkpoint weights.
    The compressed-tensors library packs signed INT4 values by adding an offset
    of 8 before storing as unsigned nibbles. Dequantization subtracts 8 to
    recover the signed value, then multiplies by the group scale.

    Args:
        packed_uint8: [M, K//2] uint8 — 2 nibbles per byte
                      low nibble (bits 0-3) = first value (even positions)
                      high nibble (bits 4-7) = second value (odd positions)
        scales: [M, K//group_size] bf16 — one scale per group of 32 elements

    Returns:
        [M, K] float32 — dequantized weights
    """
    lo = (packed_uint8 & 0x0F).to(torch.int32)
    hi = (packed_uint8 >> 4).to(torch.int32)

    M, K_half = packed_uint8.shape
    K = K_half * 2

    unpacked = torch.empty(M, K, dtype=torch.int32)
    unpacked[:, 0::2] = lo
    unpacked[:, 1::2] = hi

    # Offset encoding: subtract 8 to recover signed value
    # nibble 0 → -8, nibble 8 → 0, nibble 15 → +7
    unpacked = unpacked - 8

    # Apply group scales
    n_groups = scales.shape[-1]
    unpacked_grouped = unpacked.to(torch.float32).view(M, n_groups, group_size)
    scales_expanded = scales.to(torch.float32).unsqueeze(-1)  # [M, G, 1]

    return (unpacked_grouped * scales_expanded).view(M, K)


# ============================================================================ #
# Test Utilities
# ============================================================================ #

def pack_int4(lo_nibble, hi_nibble):
    """Pack two INT4 nibbles into one uint8 byte.

    Args:
        lo_nibble: unsigned nibble [0, 15] for even position
        hi_nibble: unsigned nibble [0, 15] for odd position

    Returns:
        uint8 value with lo in bits[0:4] and hi in bits[4:8]
    """
    return ((hi_nibble & 0x0F) << 4) | (lo_nibble & 0x0F)


def create_synthetic_int4_data(M, K, device="cpu"):
    """Create synthetic INT4 packed data for testing.

    Args:
        M: Number of rows (output features)
        K: Number of columns (input features, must be divisible by 32)

    Returns:
        (packed, scales) tuple:
        - packed: [M, K//2] uint8 — random nibble values
        - scales: [M, K//32] bf16 — random scales in [0.001, 0.1]
    """
    assert K % 32 == 0, f"K must be divisible by 32, got {K}"

    packed = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device=device)

    # Scales in reasonable range for neural network weights
    scales = torch.rand(M, K // 32, dtype=torch.float32, device=device) * 0.099 + 0.001
    scales = scales.to(torch.bfloat16)

    return packed, scales


# ============================================================================ #
# Test Functions
# ============================================================================ #

def test_sign_encoding_all_nibbles():
    """Test all 16 nibble values with scale=1.0 produce correct signed values.

    Under offset encoding (compressed-tensors standard):
        nibble 0 → -8, nibble 1 → -7, ..., nibble 7 → -1,
        nibble 8 → 0, nibble 9 → +1, ..., nibble 15 → +7
    """
    # Pack all 16 nibble values: each byte has (lo, hi) = (2i, 2i+1)
    # 8 bytes → 16 nibbles covering 0..15
    packed_bytes = []
    for i in range(8):
        lo = 2 * i        # 0, 2, 4, 6, 8, 10, 12, 14
        hi = 2 * i + 1    # 1, 3, 5, 7, 9, 11, 13, 15
        packed_bytes.append(pack_int4(lo, hi))

    # Need K divisible by 32: pad to 16 bytes (32 elements)
    while len(packed_bytes) < 16:
        packed_bytes.append(pack_int4(8, 8))  # nibble 8 → 0 (padding)

    packed = torch.tensor([packed_bytes], dtype=torch.uint8)  # [1, 16]
    scales = torch.tensor([[1.0]], dtype=torch.bfloat16)      # [1, 1] — one group of 32

    out = compressed_tensors_int4_dequantize(packed, scales, group_size=32)
    out_flat = out.flatten().tolist()

    # Expected: positions 0,2,4,...,14 = lo nibbles 0,2,4,...,14
    #           positions 1,3,5,...,15 = hi nibbles 1,3,5,...,15
    # All offset by -8
    expected_signed = [n - 8 for n in range(16)]  # [-8, -7, ..., -1, 0, 1, ..., 7]

    # The output interleaves: out[0]=lo[0]=nibble0, out[1]=hi[0]=nibble1, ...
    # So out[2k] = nibble(2k) - 8, out[2k+1] = nibble(2k+1) - 8
    # With our packing: out[0..15] = [0-8, 1-8, 2-8, ..., 15-8] = [-8, -7, ..., 7]
    for i in range(16):
        expected = expected_signed[i]
        actual = out_flat[i]
        assert abs(expected - actual) < 1e-5, (
            f"Nibble {i}: expected {expected} (nibble={i}, offset encoding), got {actual}"
        )

    print("PASS: All 16 nibble values produce correct signed values under offset encoding")


def test_nibble_ordering():
    """Test that low nibble goes to even positions and high nibble to odd positions."""
    # lo=3, hi=12 → packed byte = (12 << 4) | 3 = 195
    # After offset encoding: lo: 3-8 = -5, hi: 12-8 = +4
    byte_val = pack_int4(3, 12)
    assert byte_val == 195, f"Packing error: expected 195, got {byte_val}"

    # Fill 16 bytes with same value
    packed = torch.full((1, 16), byte_val, dtype=torch.uint8)
    scales = torch.tensor([[1.0]], dtype=torch.bfloat16)

    out = compressed_tensors_int4_dequantize(packed, scales, group_size=32)
    out_flat = out.flatten().tolist()

    for i in range(32):
        if i % 2 == 0:
            expected = -5.0  # lo nibble: 3 - 8
        else:
            expected = 4.0   # hi nibble: 12 - 8

        assert abs(out_flat[i] - expected) < 1e-5, (
            f"Position {i}: expected {expected}, got {out_flat[i]}"
        )

    print("PASS: Nibble ordering correct (lo=even positions, hi=odd positions)")


def test_known_pattern_dequantization():
    """Test 16 bytes encoding nibbles 0-15 with scale=1.0 → expected signed sequence."""
    # Each byte packs two consecutive nibbles
    # Byte 0: (lo=0, hi=1), Byte 1: (lo=2, hi=3), ..., Byte 7: (lo=14, hi=15)
    packed_bytes = []
    for i in range(8):
        packed_bytes.append(pack_int4(2 * i, 2 * i + 1))

    # Pad to 16 bytes
    for i in range(8):
        packed_bytes.append(pack_int4(2 * i, 2 * i + 1))

    packed = torch.tensor([packed_bytes], dtype=torch.uint8)
    scales = torch.tensor([[1.0]], dtype=torch.bfloat16)

    out = compressed_tensors_int4_dequantize(packed, scales, group_size=32)
    out_flat = out.flatten().tolist()

    # First 16 values: nibbles 0..15 offset by -8 → [-8, -7, ..., +7]
    # Second 16 values: same pattern repeated
    expected = [n - 8 for n in range(16)] * 2

    for i in range(32):
        assert abs(out_flat[i] - expected[i]) < 1e-5, (
            f"Position {i}: expected {expected[i]}, got {out_flat[i]}"
        )

    print("PASS: Known pattern dequantization produces correct signed sequence")


def test_scale_application():
    """Test scale multiplication at 0.5, 1.0, 2.0, 4.0."""
    # Use nibble=15 → signed value = +7 (under offset encoding)
    # lo=15, hi=15 → byte = (15 << 4) | 15 = 255
    packed = torch.full((1, 16), 255, dtype=torch.uint8)

    test_scales = [0.5, 1.0, 2.0, 4.0]

    for scale_val in test_scales:
        scales = torch.tensor([[scale_val]], dtype=torch.bfloat16)
        out = compressed_tensors_int4_dequantize(packed, scales, group_size=32)

        expected = 7.0 * scale_val  # nibble 15 → +7, then * scale
        actual = out[0, 0].item()

        # bf16 scale introduces small rounding
        assert abs(actual - expected) < abs(expected) * 0.02 + 1e-5, (
            f"Scale {scale_val}: expected {expected}, got {actual}"
        )

    print("PASS: Scale multiplication is correct at 0.5, 1.0, 2.0, 4.0")


def test_negative_values():
    """Test nibbles encoding negative values dequantize correctly."""
    # Under offset encoding, nibbles 0-7 are negative:
    #   nibble 0 → -8, nibble 1 → -7, ..., nibble 7 → -1

    # Pack nibble 0 in both positions: byte = (0 << 4) | 0 = 0
    packed = torch.full((1, 16), 0, dtype=torch.uint8)
    scales = torch.tensor([[1.0]], dtype=torch.bfloat16)

    out = compressed_tensors_int4_dequantize(packed, scales, group_size=32)

    # All values should be -8.0 (nibble 0 → 0 - 8 = -8)
    for i in range(32):
        assert abs(out[0, i].item() - (-8.0)) < 1e-5, (
            f"Position {i}: expected -8.0 for nibble=0, got {out[0, i].item()}"
        )

    # Pack nibble 7 in both positions: byte = (7 << 4) | 7 = 119
    packed = torch.full((1, 16), pack_int4(7, 7), dtype=torch.uint8)
    out = compressed_tensors_int4_dequantize(packed, scales, group_size=32)

    # All values should be -1.0 (nibble 7 → 7 - 8 = -1)
    for i in range(32):
        assert abs(out[0, i].item() - (-1.0)) < 1e-5, (
            f"Position {i}: expected -1.0 for nibble=7, got {out[0, i].item()}"
        )

    print("PASS: Negative nibble values dequantize correctly under offset encoding")


def test_zero_nibble():
    """Test that nibble=0 under offset encoding gives -8*scale, NOT 0.

    This is a critical difference from two's complement where nibble 0 → 0.
    Under offset encoding: nibble 0 → 0 - 8 = -8.
    """
    packed = torch.full((1, 16), 0, dtype=torch.uint8)  # All nibbles = 0
    scales = torch.tensor([[2.0]], dtype=torch.bfloat16)

    out = compressed_tensors_int4_dequantize(packed, scales, group_size=32)

    expected = -8.0 * 2.0  # -16.0
    actual = out[0, 0].item()

    assert abs(actual - expected) < 0.1, (
        f"Nibble=0 with scale=2.0: expected {expected}, got {actual}"
    )

    print("PASS: Nibble=0 correctly gives -8*scale under offset encoding")


def test_group_boundary():
    """Test two groups with different scales → boundary values use correct scale."""
    # K=64 → 2 groups of 32, need 32 packed bytes
    # Group 0 (bytes 0-15): all nibble=15 → signed +7
    # Group 1 (bytes 16-31): all nibble=8 → signed 0
    packed_bytes = [pack_int4(15, 15)] * 16 + [pack_int4(8, 8)] * 16
    packed = torch.tensor([packed_bytes], dtype=torch.uint8)  # [1, 32]

    # Two different scales: group 0 = 3.0, group 1 = 5.0
    scales = torch.tensor([[3.0, 5.0]], dtype=torch.bfloat16)  # [1, 2]

    out = compressed_tensors_int4_dequantize(packed, scales, group_size=32)
    out_flat = out.flatten().tolist()

    # Group 0 (positions 0-31): +7 * 3.0 = 21.0
    for i in range(32):
        expected = 7.0 * 3.0  # 21.0
        assert abs(out_flat[i] - expected) < 0.5, (
            f"Group 0, position {i}: expected {expected}, got {out_flat[i]}"
        )

    # Group 1 (positions 32-63): 0 * 5.0 = 0.0
    for i in range(32, 64):
        expected = 0.0 * 5.0  # 0.0
        assert abs(out_flat[i] - expected) < 1e-5, (
            f"Group 1, position {i}: expected {expected}, got {out_flat[i]}"
        )

    print("PASS: Group boundary correctly uses different scales per group")


def test_batchgen_vs_reference():
    """Compare BatchGen int4_dequantize_reference() against compressed-tensors reference.

    This is the PRIMARY SANITY CHECK. If BatchGen uses a different sign encoding
    than compressed-tensors (the format K2.5 checkpoints are stored in), this test
    will fail — which is the correct outcome (catching the encoding mismatch).

    Known issue: BatchGen uses two's complement (nibble>=8 → nibble-16),
    compressed-tensors uses offset (nibble - 8). These differ for all nibble values.
    """
    from batchgen.quantization.int4 import int4_dequantize_reference

    test_shapes = [
        (1, 32),      # Minimal: 1 row, 1 group
        (1, 64),      # 1 row, 2 groups
        (4, 128),     # Multiple rows
        (16, 256),    # Larger
        (64, 1024),   # Realistic small
    ]

    all_match = True

    for M, K in test_shapes:
        packed, scales = create_synthetic_int4_data(M, K)

        # Ground truth: compressed-tensors offset encoding
        ref_out = compressed_tensors_int4_dequantize(packed, scales, group_size=32)

        # BatchGen implementation
        batchgen_out = int4_dequantize_reference(packed, scales, dtype=torch.bfloat16)
        batchgen_out = batchgen_out.to(torch.float32)

        # Compare
        max_diff = (ref_out - batchgen_out).abs().max().item()
        mean_diff = (ref_out - batchgen_out).abs().mean().item()

        matches = max_diff < 0.01  # Allow small bf16 rounding
        status = "MATCH" if matches else "MISMATCH"
        all_match = all_match and matches

        print(f"  Shape [{M}, {K}]: {status} (max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f})")

    if all_match:
        print("PASS: BatchGen matches compressed-tensors reference for all shapes")
    else:
        # This is the expected outcome if the encoding mismatch exists
        print("FAIL: BatchGen does NOT match compressed-tensors reference")
        print("  → This confirms the sign encoding mismatch:")
        print("    BatchGen uses two's complement (nibble>=8 → nibble-16)")
        print("    compressed-tensors uses offset (nibble - 8)")
        print("  → BatchGen int4.py needs to be updated to use offset encoding")

        # Demonstrate the difference on a single example
        packed_demo = torch.tensor([[pack_int4(0, 3)] * 16], dtype=torch.uint8)
        scales_demo = torch.tensor([[1.0]], dtype=torch.bfloat16)

        ref = compressed_tensors_int4_dequantize(packed_demo, scales_demo)
        bg = int4_dequantize_reference(packed_demo, scales_demo).float()

        print(f"\n  Example: nibbles (lo=0, hi=3), scale=1.0")
        print(f"    compressed-tensors: lo={ref[0,0].item()}, hi={ref[0,1].item()}")
        print(f"    BatchGen:           lo={bg[0,0].item()}, hi={bg[0,1].item()}")
        print(f"    Expected (offset):  lo={0-8}, hi={3-8}")

        raise AssertionError(
            "BatchGen INT4 dequant does not match compressed-tensors reference. "
            "Sign encoding mismatch: two's complement vs offset."
        )


def test_expert_weight_shapes():
    """Test with K2.5 expert dimensions: packed [2048, 3584] → unpacked [2048, 7168].

    K2.5 routed expert shapes:
    - gate_proj / up_proj: [2048, 7168] → packed [2048, 3584], scales [2048, 224]
    - down_proj: [7168, 2048] → packed [7168, 1024], scales [7168, 64]
    """
    expert_shapes = [
        ("gate/up_proj", 2048, 7168),  # [intermediate, hidden]
        ("down_proj", 7168, 2048),      # [hidden, intermediate]
    ]

    for name, M, K in expert_shapes:
        packed, scales = create_synthetic_int4_data(M, K)

        out = compressed_tensors_int4_dequantize(packed, scales, group_size=32)

        assert out.shape == (M, K), f"{name}: expected shape ({M}, {K}), got {out.shape}"
        assert not torch.isnan(out).any(), f"{name}: output contains NaN"
        assert not torch.isinf(out).any(), f"{name}: output contains Inf"

        # Verify scale dimensions
        assert scales.shape == (M, K // 32), (
            f"{name}: expected scales ({M}, {K//32}), got {scales.shape}"
        )

        print(f"  {name}: [{M}, {K//2}] packed → [{M}, {K}] unpacked, "
              f"scales [{M}, {K//32}] — OK")

    print("PASS: K2.5 expert weight shapes are correct")


def test_gpu_triton_vs_reference():
    """Test Triton kernel vs CPU reference on same data.

    Skipped if CUDA is not available (run on remote machine).
    """
    if not torch.cuda.is_available():
        print("SKIP: CUDA not available (run on remote machine)")
        return

    from batchgen.quantization.int4 import int4_dequantize_reference, int4_dequantize_triton

    test_shapes = [
        (1, 32),
        (4, 128),
        (64, 1024),
        (256, 7168),   # K2.5 hidden dim
    ]

    for M, K in test_shapes:
        packed, scales = create_synthetic_int4_data(M, K, device="cuda")

        # CPU reference
        packed_cpu = packed.cpu()
        scales_cpu = scales.cpu()
        ref_out = int4_dequantize_reference(packed_cpu, scales_cpu, dtype=torch.bfloat16)

        # GPU Triton
        gpu_out = int4_dequantize_triton(packed, scales, dtype=torch.bfloat16)

        max_diff = (ref_out.float() - gpu_out.cpu().float()).abs().max().item()
        print(f"  Shape [{M}, {K}]: max_diff={max_diff:.6f}")

        assert max_diff < 0.01, (
            f"Triton vs reference mismatch at [{M}, {K}]: max_diff={max_diff}"
        )

    print("PASS: Triton kernel matches CPU reference for all shapes")


def test_checkpoint_weights():
    """Test with actual K2.5 checkpoint if available.

    Skipped if K2.5 checkpoint path is not set.
    Set KIMI_K25_CHECKPOINT_PATH environment variable to test with real weights.
    """
    checkpoint_path = os.environ.get("KIMI_K25_CHECKPOINT_PATH")

    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        print("SKIP: K2.5 checkpoint not available")
        print("  Set KIMI_K25_CHECKPOINT_PATH environment variable to test with real weights")
        return

    from safetensors import safe_open

    print(f"\nTesting with checkpoint: {checkpoint_path}")

    # Find safetensor files
    safetensor_files = sorted([
        os.path.join(checkpoint_path, f)
        for f in os.listdir(checkpoint_path)
        if f.endswith(".safetensors")
    ])

    if not safetensor_files:
        print("  No safetensor files found")
        return

    # Try to load first layer's expert 0 gate_proj
    packed_name = "model.layers.3.mlp.experts.0.gate_proj.weight_packed"
    scales_name = "model.layers.3.mlp.experts.0.gate_proj.weight_scale"

    packed = None
    scales = None

    for sf_file in safetensor_files:
        with safe_open(sf_file, framework="pt", device="cpu") as f:
            keys = f.keys()
            if packed_name in keys:
                packed = f.get_tensor(packed_name)
            if scales_name in keys:
                scales = f.get_tensor(scales_name)
        if packed is not None and scales is not None:
            break

    if packed is None or scales is None:
        print(f"  Could not find {packed_name}")
        print("  Trying alternate tensor names...")
        # Try listing available expert tensors
        for sf_file in safetensor_files[:3]:
            with safe_open(sf_file, framework="pt", device="cpu") as f:
                expert_keys = [k for k in f.keys() if "weight_packed" in k]
                if expert_keys:
                    print(f"  Found in {os.path.basename(sf_file)}: {expert_keys[:3]}...")
        return

    print(f"  packed shape: {packed.shape}, dtype: {packed.dtype}")
    print(f"  scales shape: {scales.shape}, dtype: {scales.dtype}")

    # Ensure uint8 and bf16
    assert packed.dtype == torch.uint8, f"Expected uint8, got {packed.dtype}"
    assert scales.dtype == torch.bfloat16, f"Expected bf16, got {scales.dtype}"

    # Run compressed-tensors reference
    ref_out = compressed_tensors_int4_dequantize(packed, scales, group_size=32)

    print(f"  output shape: {ref_out.shape}")
    print(f"  output range: [{ref_out.min().item():.4f}, {ref_out.max().item():.4f}]")
    print(f"  output mean: {ref_out.mean().item():.6f}, std: {ref_out.std().item():.6f}")

    assert not torch.isnan(ref_out).any(), "Checkpoint dequant produced NaN"
    assert not torch.isinf(ref_out).any(), "Checkpoint dequant produced Inf"

    print("PASS: Checkpoint weight dequantization produces valid output")


# ============================================================================ #
# Main
# ============================================================================ #

def run_all_tests():
    """Run all INT4 dequantization tests."""
    print("=" * 60)
    print("INT4 W4A16 Dequantization Test Suite")
    print("=" * 60)

    tests = [
        ("Sign Encoding (All 16 Nibbles)", test_sign_encoding_all_nibbles),
        ("Nibble Ordering", test_nibble_ordering),
        ("Known Pattern Dequantization", test_known_pattern_dequantization),
        ("Scale Application", test_scale_application),
        ("Negative Values", test_negative_values),
        ("Zero Nibble (offset vs two's complement)", test_zero_nibble),
        ("Group Boundary", test_group_boundary),
        ("BatchGen vs compressed-tensors Reference", test_batchgen_vs_reference),
        ("K2.5 Expert Weight Shapes", test_expert_weight_shapes),
        ("GPU Triton vs Reference", test_gpu_triton_vs_reference),
        ("Checkpoint Weights", test_checkpoint_weights),
    ]

    passed = 0
    failed = 0
    skipped = 0

    for name, test_fn in tests:
        print(f"\n{'=' * 60}")
        print(f"Test: {name}")
        print("-" * 60)
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"FAIL: {e}")
            failed += 1
        except Exception as e:
            err_str = str(e)
            if "SKIP" in err_str:
                skipped += 1
            else:
                print(f"ERROR: {e}")
                import traceback
                traceback.print_exc()
                failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
