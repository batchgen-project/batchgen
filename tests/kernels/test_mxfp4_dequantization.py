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

"""Test MXFP4 dequantization matches OpenAI reference implementation.

This test compares BatchGen's MXFP4 dequantization with the exact OpenAI
implementation from gpt-oss/gpt_oss/torch/weights.py.

MXFP4 Format:
- 32 FP4 values per block (16 bytes packed, 2 values per byte)
- 1 scale (uint8) per block, exponent = scale - 127
- FP4 lookup table: [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
- Dequantization: ldexp(fp4_value, exponent) = fp4_value * 2^exponent

Usage:
    python -m pytest tests/test_mxfp4_dequantization.py -v
    or
    python tests/test_mxfp4_dequantization.py
"""

import math
import sys
import os

import torch
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================ #
# OpenAI Reference Implementation (from gpt-oss/gpt_oss/torch/weights.py)
# ============================================================================ #

BYTES_PER_BLOCK = 16  # 32 FP4 numbers packed in 16 bytes

FP4_VALUES = [
    +0.0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
]


def openai_mxfp4_dequantize(
    blocks: torch.Tensor,
    scales: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """OpenAI reference MXFP4 dequantization (from gpt-oss/gpt_oss/torch/weights.py).

    Args:
        blocks: Packed FP4 values as uint8 [..., G, B] where B=16 (bytes per block)
        scales: Scale factors as uint8 [..., G] (one per block of 32 FP4 values)
        dtype: Output dtype (default: bfloat16)

    Returns:
        Dequantized tensor [..., G * B * 2] = [..., G * 32]
    """
    # Validate shapes
    assert blocks.shape[:-1] == scales.shape, (
        f"{blocks.shape=} does not match {scales.shape=}"
    )

    # Convert scales: uint8 - 127 -> exponent
    scales = scales.to(torch.int32) - 127

    lut = torch.tensor(FP4_VALUES, dtype=dtype, device=blocks.device)

    *prefix_shape, G, B = blocks.shape
    rows_total = math.prod(prefix_shape) * G

    blocks = blocks.reshape(rows_total, B)
    scales = scales.reshape(rows_total, 1)

    out = torch.empty(rows_total, B * 2, dtype=dtype, device=blocks.device)

    # Extract nibbles and lookup
    idx_lo = (blocks & 0x0F).to(torch.long)
    idx_hi = (blocks >> 4).to(torch.long)

    out[:, 0::2] = lut[idx_lo]  # Even indices get low nibble
    out[:, 1::2] = lut[idx_hi]  # Odd indices get high nibble

    # Apply scale: value * 2^exponent
    torch.ldexp(out, scales, out=out)

    return out.reshape(*prefix_shape, G, B * 2).view(*prefix_shape, G * B * 2)


# ============================================================================ #
# Test Utilities
# ============================================================================ #

def create_synthetic_mxfp4_data(
    num_rows: int,
    num_blocks_per_row: int = 1,
    device: str = "cpu",
) -> tuple:
    """Create synthetic MXFP4 packed data for testing.

    Args:
        num_rows: Number of rows
        num_blocks_per_row: Number of 32-element blocks per row
        device: Device to create tensors on

    Returns:
        (blocks, scales) tuple:
        - blocks: [num_rows, num_blocks_per_row, 16] uint8
        - scales: [num_rows, num_blocks_per_row] uint8
    """
    # Each block is 16 bytes (32 FP4 values)
    blocks = torch.randint(
        0, 256,
        (num_rows, num_blocks_per_row, BYTES_PER_BLOCK),
        dtype=torch.uint8,
        device=device
    )

    # Scales: reasonable exponent range around 127 (exponent 0)
    # Range [117, 137] gives exponents [-10, +10]
    scales = torch.randint(
        117, 138,
        (num_rows, num_blocks_per_row),
        dtype=torch.uint8,
        device=device
    )

    return blocks, scales


def create_known_pattern_data(device: str = "cpu") -> tuple:
    """Create MXFP4 data with known values for verification.

    Returns blocks containing all 16 FP4 indices and scale=127 (exponent=0).
    """
    # Create a block with indices 0-15 repeated
    # Each byte packs 2 FP4 values: low nibble (even), high nibble (odd)
    # Byte i contains: low=2i, high=2i+1
    packed_bytes = []
    for i in range(16):
        lo = (2 * i) % 16      # Even positions: 0, 2, 4, ..., 14, 0, 2, ...
        hi = (2 * i + 1) % 16  # Odd positions: 1, 3, 5, ..., 15, 1, 3, ...
        packed_bytes.append((hi << 4) | lo)

    blocks = torch.tensor([packed_bytes], dtype=torch.uint8, device=device)
    blocks = blocks.unsqueeze(0)  # [1, 1, 16]

    scales = torch.tensor([[127]], dtype=torch.uint8, device=device)  # exponent=0

    return blocks, scales


# ============================================================================ #
# Test Functions
# ============================================================================ #

def test_fp4_lookup_table():
    """Verify FP4 lookup table matches expected values."""
    expected = [
        +0.0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ]

    # Check BatchGen implementation
    from batchgen.quantization.mxfp4 import FP4_LOOKUP_TABLE

    for i, (expected_val, actual_val) in enumerate(zip(expected, FP4_LOOKUP_TABLE.tolist())):
        assert expected_val == actual_val, f"FP4 index {i}: expected {expected_val}, got {actual_val}"

    print("PASS: FP4 lookup table matches reference")


def test_known_pattern_dequantization():
    """Test dequantization with known pattern produces expected values."""
    blocks, scales = create_known_pattern_data()

    # OpenAI reference
    ref_out = openai_mxfp4_dequantize(blocks, scales, dtype=torch.float32)

    # With scale=127 (exponent=0), output should be raw FP4 values
    # Pattern: [FP4[0], FP4[1], FP4[2], ..., FP4[15], FP4[0], FP4[1], ...]
    expected_indices = [(2*i) % 16 for i in range(16)] + [(2*i+1) % 16 for i in range(16)]
    # Wait, the interleaving is different. Let me recalculate.
    # Byte i contains: lo=(2i)%16, hi=(2i+1)%16
    # Output positions: out[0::2] = lo[0..15], out[1::2] = hi[0..15]
    # So: out[0]=lo[0], out[1]=hi[0], out[2]=lo[1], out[3]=hi[1], ...
    # With lo[i] = (2i)%16 and hi[i] = (2i+1)%16:
    # out[0]=0, out[1]=1, out[2]=2, out[3]=3, ..., out[30]=14, out[31]=15
    expected_fp4_indices = list(range(16)) * 2  # 0-15 twice
    expected_values = [FP4_VALUES[i] for i in expected_fp4_indices]

    # Actually the pattern is: out[2k] = lo[k], out[2k+1] = hi[k]
    # With our encoding: lo[k] = (2k)%16, hi[k] = (2k+1)%16
    # So out[2k] = FP4[(2k)%16], out[2k+1] = FP4[(2k+1)%16]
    # This gives out = [FP4[0], FP4[1], FP4[2], FP4[3], ..., FP4[14], FP4[15],
    #                   FP4[0], FP4[1], ..., FP4[14], FP4[15]]
    # = all 16 values, then repeated

    ref_out_flat = ref_out.flatten().tolist()

    print(f"Reference output (first 16): {ref_out_flat[:16]}")
    print(f"Expected values (first 16): {expected_values[:16]}")

    # Verify
    for i, (expected, actual) in enumerate(zip(expected_values, ref_out_flat)):
        assert abs(expected - actual) < 1e-6, f"Position {i}: expected {expected}, got {actual}"

    print("PASS: Known pattern dequantization produces correct values")


def test_batchgen_vs_openai_reference():
    """Compare BatchGen MXFP4 dequantization with OpenAI reference."""
    from batchgen.quantization.mxfp4 import mxfp4_dequantize_reference

    # Test with various shapes
    test_cases = [
        (1, 1),      # Single block
        (10, 1),     # Multiple rows, single block
        (10, 4),     # Multiple rows, multiple blocks
        (128, 90),   # Realistic expert shape (128 experts, ~2880 hidden / 32)
    ]

    for num_rows, num_blocks in test_cases:
        print(f"\nTesting shape: [{num_rows}, {num_blocks}, 16]...")

        blocks, scales = create_synthetic_mxfp4_data(num_rows, num_blocks)

        # OpenAI reference
        ref_out = openai_mxfp4_dequantize(blocks, scales, dtype=torch.float32)

        # BatchGen implementation
        # Note: BatchGen expects flat packed tensor [M, N//2] and scales [M, N//32]
        # where N is the number of FP4 elements
        # Reshape to match: blocks [num_rows, num_blocks * 16], scales [num_rows, num_blocks]
        blocks_flat = blocks.view(num_rows, num_blocks * BYTES_PER_BLOCK)
        scales_flat = scales.view(num_rows, num_blocks)

        batchgen_out = mxfp4_dequantize_reference(
            blocks_flat,
            scales_flat,
            dtype=torch.float32
        )

        # Compare
        ref_out_flat = ref_out.view(num_rows, -1)

        # Check shapes match
        assert ref_out_flat.shape == batchgen_out.shape, (
            f"Shape mismatch: ref={ref_out_flat.shape}, batchgen={batchgen_out.shape}"
        )

        # Check values match
        max_diff = (ref_out_flat - batchgen_out).abs().max().item()
        print(f"  Max difference: {max_diff}")

        assert max_diff < 1e-5, f"Values differ by {max_diff}"

        print(f"  PASS: Shapes {ref_out_flat.shape} match with max diff {max_diff:.2e}")

    print("\nPASS: BatchGen matches OpenAI reference for all test cases")


def test_scale_exponent_calculation():
    """Test that scale exponent calculation is correct."""
    # Scale=127 should give exponent=0 (no scaling)
    # Scale=128 should give exponent=1 (multiply by 2)
    # Scale=126 should give exponent=-1 (divide by 2)

    # Create single block with all indices = 2 (FP4 value = 1.0)
    # Byte with lo=2, hi=2: (2 << 4) | 2 = 34
    blocks = torch.full((1, 1, 16), 34, dtype=torch.uint8)

    test_scales = [
        (125, 0.25),   # 2^-2 = 0.25
        (126, 0.5),    # 2^-1 = 0.5
        (127, 1.0),    # 2^0 = 1.0
        (128, 2.0),    # 2^1 = 2.0
        (129, 4.0),    # 2^2 = 4.0
    ]

    for scale_val, expected_multiplier in test_scales:
        scales = torch.full((1, 1), scale_val, dtype=torch.uint8)

        out = openai_mxfp4_dequantize(blocks, scales, dtype=torch.float32)

        # FP4 value 1.0 * expected_multiplier
        expected = 1.0 * expected_multiplier
        actual = out[0, 0].item()

        assert abs(expected - actual) < 1e-6, (
            f"Scale {scale_val}: expected {expected}, got {actual}"
        )

    print("PASS: Scale exponent calculation is correct")


def test_nibble_ordering():
    """Test that nibble ordering (low/high) is correct."""
    # Create a byte with distinct low and high nibbles
    # lo=1 (FP4=0.5), hi=6 (FP4=4.0)
    # Packed byte: (6 << 4) | 1 = 97
    blocks = torch.full((1, 1, 16), 97, dtype=torch.uint8)
    scales = torch.full((1, 1), 127, dtype=torch.uint8)  # exponent=0

    out = openai_mxfp4_dequantize(blocks, scales, dtype=torch.float32)

    # Even positions should have FP4[1] = 0.5
    # Odd positions should have FP4[6] = 4.0
    for i in range(32):
        if i % 2 == 0:
            expected = 0.5  # low nibble
        else:
            expected = 4.0  # high nibble

        actual = out[0, i].item()
        assert abs(expected - actual) < 1e-6, (
            f"Position {i}: expected {expected}, got {actual}"
        )

    print("PASS: Nibble ordering is correct (low=even, high=odd)")


def test_negative_values():
    """Test that negative FP4 values are handled correctly."""
    # FP4 index 8-15 are negative values
    # Index 10 = -1.0
    # lo=10, hi=10 -> byte = (10 << 4) | 10 = 170
    blocks = torch.full((1, 1, 16), 170, dtype=torch.uint8)
    scales = torch.full((1, 1), 127, dtype=torch.uint8)

    out = openai_mxfp4_dequantize(blocks, scales, dtype=torch.float32)

    # All values should be -1.0
    for i in range(32):
        assert abs(out[0, i].item() - (-1.0)) < 1e-6, f"Position {i}: expected -1.0"

    print("PASS: Negative FP4 values handled correctly")


def test_gpu_if_available():
    """Test on GPU if available."""
    if not torch.cuda.is_available():
        print("SKIP: CUDA not available")
        return

    from batchgen.quantization.mxfp4 import mxfp4_dequantize

    device = "cuda"
    blocks, scales = create_synthetic_mxfp4_data(100, 10, device=device)

    # OpenAI reference
    ref_out = openai_mxfp4_dequantize(blocks, scales, dtype=torch.bfloat16)

    # BatchGen (may use Triton kernel on GPU)
    blocks_flat = blocks.view(100, 10 * 16)
    scales_flat = scales.view(100, 10)
    batchgen_out = mxfp4_dequantize(blocks_flat, scales_flat, dtype=torch.bfloat16)

    ref_out_flat = ref_out.view(100, -1)

    # BF16 has less precision, allow slightly larger tolerance
    max_diff = (ref_out_flat.float() - batchgen_out.float()).abs().max().item()
    print(f"GPU test max difference: {max_diff}")

    assert max_diff < 1e-2, f"GPU values differ by {max_diff}"

    print("PASS: GPU dequantization matches reference")


def test_expert_weight_shape():
    """Test with realistic GPT-OSS expert weight shapes.

    GPT-OSS expert shapes (after slicing):
    - gate_proj: [2880, 1440] packed -> [2880, 2880] unpacked
    - up_proj: [2880, 1440] packed -> [2880, 2880] unpacked
    - down_proj: [2880, 1440] packed -> [2880, 2880] unpacked

    Scales: [2880, 90] (2880 / 32 = 90)
    """
    from batchgen.quantization.mxfp4 import mxfp4_dequantize_reference

    # Realistic dimensions
    out_features = 2880
    in_features_packed = 1440  # 2880 / 2
    num_blocks = in_features_packed // 16  # 90 blocks per row

    print(f"\nTesting realistic expert shape: [{out_features}, {in_features_packed}]")
    print(f"  Blocks per row: {num_blocks}, Total FP4 elements per row: {num_blocks * 32}")

    blocks = torch.randint(0, 256, (out_features, num_blocks, 16), dtype=torch.uint8)
    scales = torch.randint(117, 138, (out_features, num_blocks), dtype=torch.uint8)

    # OpenAI reference
    ref_out = openai_mxfp4_dequantize(blocks, scales, dtype=torch.float32)

    # BatchGen
    blocks_flat = blocks.view(out_features, in_features_packed)
    batchgen_out = mxfp4_dequantize_reference(blocks_flat, scales, dtype=torch.float32)

    ref_out_flat = ref_out.view(out_features, -1)

    assert ref_out_flat.shape == batchgen_out.shape, (
        f"Shape mismatch: ref={ref_out_flat.shape}, batchgen={batchgen_out.shape}"
    )

    max_diff = (ref_out_flat - batchgen_out).abs().max().item()
    print(f"  Max difference: {max_diff}")

    assert max_diff < 1e-5, f"Values differ by {max_diff}"

    print("PASS: Realistic expert shape dequantization works correctly")


def test_with_checkpoint(checkpoint_path: str = None):
    """Test with actual GPT-OSS checkpoint if available.

    Args:
        checkpoint_path: Path to GPT-OSS checkpoint directory
    """
    if checkpoint_path is None:
        checkpoint_path = os.environ.get("GPT_OSS_CHECKPOINT_PATH")

    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        print("SKIP: Checkpoint path not provided or doesn't exist")
        print("  Set GPT_OSS_CHECKPOINT_PATH environment variable to test with real weights")
        return

    from safetensors import safe_open
    from batchgen.quantization.mxfp4 import mxfp4_dequantize_reference

    print(f"\nTesting with checkpoint: {checkpoint_path}")

    # Find safetensor files
    safetensor_files = [
        os.path.join(checkpoint_path, f)
        for f in os.listdir(checkpoint_path)
        if f.endswith(".safetensors")
    ]

    if not safetensor_files:
        print("  No safetensor files found")
        return

    # Load first layer's mlp1 weights
    blocks_name = "block.0.mlp.mlp1_weight.blocks"
    scales_name = "block.0.mlp.mlp1_weight.scales"

    blocks = None
    scales = None

    for sf_file in safetensor_files:
        with safe_open(sf_file, framework="pt", device="cpu") as f:
            if blocks_name in f.keys():
                blocks = f.get_tensor(blocks_name)
            if scales_name in f.keys():
                scales = f.get_tensor(scales_name)

    if blocks is None or scales is None:
        print(f"  Could not find {blocks_name} or {scales_name}")
        return

    print(f"  blocks shape: {blocks.shape}")
    print(f"  scales shape: {scales.shape}")

    # OpenAI reference (use a subset for speed)
    subset_rows = min(128, blocks.shape[0])
    blocks_subset = blocks[:subset_rows]
    scales_subset = scales[:subset_rows]

    ref_out = openai_mxfp4_dequantize(blocks_subset, scales_subset, dtype=torch.float32)

    # BatchGen
    *prefix, G, B = blocks_subset.shape
    blocks_flat = blocks_subset.view(subset_rows, G * B)
    scales_flat = scales_subset.view(subset_rows, G)

    batchgen_out = mxfp4_dequantize_reference(blocks_flat, scales_flat, dtype=torch.float32)

    ref_out_flat = ref_out.view(subset_rows, -1)

    max_diff = (ref_out_flat - batchgen_out).abs().max().item()
    mean_diff = (ref_out_flat - batchgen_out).abs().mean().item()

    print(f"  Max difference: {max_diff}")
    print(f"  Mean difference: {mean_diff}")

    assert max_diff < 1e-5, f"Checkpoint values differ by {max_diff}"

    print("PASS: Checkpoint weight dequantization matches reference")


# ============================================================================ #
# Main
# ============================================================================ #

def run_all_tests():
    """Run all MXFP4 dequantization tests."""
    print("=" * 60)
    print("MXFP4 Dequantization Test Suite")
    print("=" * 60)

    tests = [
        ("FP4 Lookup Table", test_fp4_lookup_table),
        ("Scale Exponent Calculation", test_scale_exponent_calculation),
        ("Nibble Ordering", test_nibble_ordering),
        ("Negative Values", test_negative_values),
        ("Known Pattern", test_known_pattern_dequantization),
        ("BatchGen vs OpenAI Reference", test_batchgen_vs_openai_reference),
        ("Expert Weight Shape", test_expert_weight_shape),
        ("GPU (if available)", test_gpu_if_available),
        ("Checkpoint (if available)", test_with_checkpoint),
    ]

    passed = 0
    failed = 0
    skipped = 0

    for name, test_fn in tests:
        print(f"\n{'='*60}")
        print(f"Test: {name}")
        print("-" * 60)
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"FAIL: {e}")
            failed += 1
        except Exception as e:
            if "SKIP" in str(e) or "SKIP" in getattr(e, 'args', [''])[0] if e.args else '':
                skipped += 1
            else:
                print(f"ERROR: {e}")
                failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
