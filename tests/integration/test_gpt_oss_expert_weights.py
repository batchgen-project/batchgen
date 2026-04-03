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

"""Test GPT-OSS expert weight loading and slicing.

This test verifies:
1. MLP1 splitting into gate_proj + up_proj
2. Expert tensor slicing from stacked format [128, ...] -> individual experts
3. MXFP4 weight shapes match expected dimensions
4. End-to-end expert forward pass produces reasonable outputs

GPT-OSS MoE Format:
- mlp1: [128, 5760, packed_dim] where 5760 = 2 * intermediate_size
- mlp1 splits into gate_proj [2880, ...] and up_proj [2880, ...]
- mlp2: [128, 2880, packed_dim] -> down_proj
- Biases: [128, 5760] for mlp1, [128, 2880] for mlp2

Usage:
    python -m pytest tests/test_gpt_oss_expert_weights.py -v
    or
    python tests/test_gpt_oss_expert_weights.py
"""

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================ #
# Test Constants (GPT-OSS-120B dimensions)
# ============================================================================ #

HIDDEN_SIZE = 2880
INTERMEDIATE_SIZE = 2880
NUM_EXPERTS = 128
NUM_LAYERS = 36

# MXFP4 packing: 2 FP4 values per byte
PACKED_HIDDEN = HIDDEN_SIZE // 2  # 1440
SCALES_HIDDEN = HIDDEN_SIZE // 32  # 90

PACKED_INTERMEDIATE = INTERMEDIATE_SIZE // 2  # 1440
SCALES_INTERMEDIATE = INTERMEDIATE_SIZE // 32  # 90


# ============================================================================ #
# OpenAI Reference: SwiGLU Expert
# ============================================================================ #

class SwiGLUExpert(nn.Module):
    """Reference SwiGLU expert implementation (like GPT-OSS)."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """SwiGLU forward: clamp(silu(gate) * up) @ down."""
        gate = self.gate_proj(x)
        up = self.up_proj(x)

        # SwiGLU with clamping (GPT-OSS specific)
        hidden = torch.clamp(F.silu(gate) * up, -7.0, 7.0)
        return self.down_proj(hidden)


# ============================================================================ #
# Test Utilities
# ============================================================================ #

def create_synthetic_expert_weights_stacked():
    """Create synthetic stacked expert weights in OpenAI checkpoint format.

    Returns dict with:
    - mlp1_blocks: [128, 5760, packed_dim] - gate + up combined
    - mlp1_scales: [128, 5760, scales_dim]
    - mlp1_bias: [128, 5760]
    - mlp2_blocks: [128, 2880, packed_dim] - down_proj
    - mlp2_scales: [128, 2880, scales_dim]
    - mlp2_bias: [128, 2880]
    """
    # mlp1 contains both gate_proj and up_proj stacked
    mlp1_out_dim = INTERMEDIATE_SIZE * 2  # 5760

    # Packed dimensions (MXFP4: 2 values per byte)
    mlp1_packed_dim = PACKED_HIDDEN  # 1440
    mlp2_packed_dim = PACKED_INTERMEDIATE  # 1440

    # Scale dimensions (1 scale per 32 FP4 values)
    mlp1_scales_dim = SCALES_HIDDEN  # 90
    mlp2_scales_dim = SCALES_INTERMEDIATE  # 90

    return {
        "mlp1_blocks": torch.randint(0, 256, (NUM_EXPERTS, mlp1_out_dim, mlp1_packed_dim), dtype=torch.uint8),
        "mlp1_scales": torch.randint(117, 138, (NUM_EXPERTS, mlp1_out_dim, mlp1_scales_dim), dtype=torch.uint8),
        "mlp1_bias": torch.randn(NUM_EXPERTS, mlp1_out_dim, dtype=torch.bfloat16) * 0.01,
        "mlp2_blocks": torch.randint(0, 256, (NUM_EXPERTS, HIDDEN_SIZE, mlp2_packed_dim), dtype=torch.uint8),
        "mlp2_scales": torch.randint(117, 138, (NUM_EXPERTS, HIDDEN_SIZE, mlp2_scales_dim), dtype=torch.uint8),
        "mlp2_bias": torch.randn(NUM_EXPERTS, HIDDEN_SIZE, dtype=torch.bfloat16) * 0.01,
    }


def slice_expert_weights(stacked: dict, expert_idx: int) -> dict:
    """Slice individual expert weights from stacked format.

    This implements the logic from gpt_oss_parameter_server._convert_layer.
    """
    # Get expert's portion
    mlp1_blocks = stacked["mlp1_blocks"][expert_idx]  # [5760, packed_dim]
    mlp1_scales = stacked["mlp1_scales"][expert_idx]  # [5760, scales_dim]
    mlp1_bias = stacked["mlp1_bias"][expert_idx]      # [5760]

    mlp2_blocks = stacked["mlp2_blocks"][expert_idx]  # [2880, packed_dim]
    mlp2_scales = stacked["mlp2_scales"][expert_idx]  # [2880, scales_dim]
    mlp2_bias = stacked["mlp2_bias"][expert_idx]      # [2880]

    # Split mlp1 into gate_proj (first half) and up_proj (second half)
    gate_blocks = mlp1_blocks[:INTERMEDIATE_SIZE]     # [2880, packed_dim]
    gate_scales = mlp1_scales[:INTERMEDIATE_SIZE]     # [2880, scales_dim]
    gate_bias = mlp1_bias[:INTERMEDIATE_SIZE]         # [2880]

    up_blocks = mlp1_blocks[INTERMEDIATE_SIZE:]       # [2880, packed_dim]
    up_scales = mlp1_scales[INTERMEDIATE_SIZE:]       # [2880, scales_dim]
    up_bias = mlp1_bias[INTERMEDIATE_SIZE:]           # [2880]

    return {
        "gate_proj.weight": gate_blocks,
        "gate_proj.weight_scales": gate_scales,
        "gate_proj.bias": gate_bias,
        "up_proj.weight": up_blocks,
        "up_proj.weight_scales": up_scales,
        "up_proj.bias": up_bias,
        "down_proj.weight": mlp2_blocks,
        "down_proj.weight_scales": mlp2_scales,
        "down_proj.bias": mlp2_bias,
    }


def dequantize_expert_weights(sliced: dict) -> dict:
    """Dequantize MXFP4 expert weights to BF16."""
    from batchgen.quantization.mxfp4 import mxfp4_dequantize_reference

    result = {}

    for name in ["gate_proj", "up_proj", "down_proj"]:
        packed = sliced[f"{name}.weight"]
        scales = sliced[f"{name}.weight_scales"]
        bias = sliced[f"{name}.bias"]

        # Dequantize weight
        weight = mxfp4_dequantize_reference(packed, scales, dtype=torch.bfloat16)
        result[f"{name}.weight"] = weight
        result[f"{name}.bias"] = bias

    return result


# ============================================================================ #
# Test Functions
# ============================================================================ #

def test_stacked_weight_shapes():
    """Verify stacked weight shapes match GPT-OSS format."""
    stacked = create_synthetic_expert_weights_stacked()

    # mlp1: gate + up combined
    assert stacked["mlp1_blocks"].shape == (128, 5760, 1440), (
        f"mlp1_blocks shape: {stacked['mlp1_blocks'].shape}"
    )
    assert stacked["mlp1_scales"].shape == (128, 5760, 90), (
        f"mlp1_scales shape: {stacked['mlp1_scales'].shape}"
    )
    assert stacked["mlp1_bias"].shape == (128, 5760), (
        f"mlp1_bias shape: {stacked['mlp1_bias'].shape}"
    )

    # mlp2: down_proj
    assert stacked["mlp2_blocks"].shape == (128, 2880, 1440), (
        f"mlp2_blocks shape: {stacked['mlp2_blocks'].shape}"
    )
    assert stacked["mlp2_scales"].shape == (128, 2880, 90), (
        f"mlp2_scales shape: {stacked['mlp2_scales'].shape}"
    )
    assert stacked["mlp2_bias"].shape == (128, 2880), (
        f"mlp2_bias shape: {stacked['mlp2_bias'].shape}"
    )

    print("PASS: Stacked weight shapes are correct")


def test_expert_slicing():
    """Verify expert slicing produces correct shapes."""
    stacked = create_synthetic_expert_weights_stacked()

    for expert_idx in [0, 63, 127]:  # Test first, middle, last expert
        sliced = slice_expert_weights(stacked, expert_idx)

        # gate_proj
        assert sliced["gate_proj.weight"].shape == (2880, 1440), (
            f"Expert {expert_idx} gate_proj.weight: {sliced['gate_proj.weight'].shape}"
        )
        assert sliced["gate_proj.weight_scales"].shape == (2880, 90), (
            f"Expert {expert_idx} gate_proj.weight_scales: {sliced['gate_proj.weight_scales'].shape}"
        )
        assert sliced["gate_proj.bias"].shape == (2880,), (
            f"Expert {expert_idx} gate_proj.bias: {sliced['gate_proj.bias'].shape}"
        )

        # up_proj
        assert sliced["up_proj.weight"].shape == (2880, 1440), (
            f"Expert {expert_idx} up_proj.weight: {sliced['up_proj.weight'].shape}"
        )

        # down_proj
        assert sliced["down_proj.weight"].shape == (2880, 1440), (
            f"Expert {expert_idx} down_proj.weight: {sliced['down_proj.weight'].shape}"
        )

    print("PASS: Expert slicing produces correct shapes")


def test_mlp1_split_correctness():
    """Verify mlp1 split preserves data correctly."""
    stacked = create_synthetic_expert_weights_stacked()

    for expert_idx in [0, 42]:
        sliced = slice_expert_weights(stacked, expert_idx)

        # gate_proj should be first half of mlp1
        assert torch.equal(
            sliced["gate_proj.weight"],
            stacked["mlp1_blocks"][expert_idx, :INTERMEDIATE_SIZE]
        ), f"Expert {expert_idx}: gate_proj.weight mismatch"

        assert torch.equal(
            sliced["gate_proj.weight_scales"],
            stacked["mlp1_scales"][expert_idx, :INTERMEDIATE_SIZE]
        ), f"Expert {expert_idx}: gate_proj.weight_scales mismatch"

        # up_proj should be second half of mlp1
        assert torch.equal(
            sliced["up_proj.weight"],
            stacked["mlp1_blocks"][expert_idx, INTERMEDIATE_SIZE:]
        ), f"Expert {expert_idx}: up_proj.weight mismatch"

        # down_proj should be mlp2
        assert torch.equal(
            sliced["down_proj.weight"],
            stacked["mlp2_blocks"][expert_idx]
        ), f"Expert {expert_idx}: down_proj.weight mismatch"

    print("PASS: mlp1 split preserves data correctly")


def test_dequantized_weight_shapes():
    """Verify dequantized weights have correct shapes."""
    stacked = create_synthetic_expert_weights_stacked()
    sliced = slice_expert_weights(stacked, 0)
    dequant = dequantize_expert_weights(sliced)

    # After dequantization: packed [out, in//2] -> unpacked [out, in]
    assert dequant["gate_proj.weight"].shape == (2880, 2880), (
        f"gate_proj.weight dequantized: {dequant['gate_proj.weight'].shape}"
    )
    assert dequant["up_proj.weight"].shape == (2880, 2880), (
        f"up_proj.weight dequantized: {dequant['up_proj.weight'].shape}"
    )
    assert dequant["down_proj.weight"].shape == (2880, 2880), (
        f"down_proj.weight dequantized: {dequant['down_proj.weight'].shape}"
    )

    # Biases unchanged
    assert dequant["gate_proj.bias"].shape == (2880,)
    assert dequant["up_proj.bias"].shape == (2880,)
    assert dequant["down_proj.bias"].shape == (2880,)

    print("PASS: Dequantized weight shapes are correct")


def test_expert_forward_pass():
    """Test expert forward pass with dequantized weights."""
    stacked = create_synthetic_expert_weights_stacked()
    sliced = slice_expert_weights(stacked, 0)
    dequant = dequantize_expert_weights(sliced)

    # Create expert module
    expert = SwiGLUExpert(HIDDEN_SIZE, INTERMEDIATE_SIZE)

    # Load dequantized weights
    with torch.no_grad():
        expert.gate_proj.weight.copy_(dequant["gate_proj.weight"])
        expert.gate_proj.bias.copy_(dequant["gate_proj.bias"])
        expert.up_proj.weight.copy_(dequant["up_proj.weight"])
        expert.up_proj.bias.copy_(dequant["up_proj.bias"])
        expert.down_proj.weight.copy_(dequant["down_proj.weight"])
        expert.down_proj.bias.copy_(dequant["down_proj.bias"])

    # Forward pass
    x = torch.randn(4, HIDDEN_SIZE, dtype=torch.bfloat16)
    expert = expert.to(torch.bfloat16)

    output = expert(x)

    assert output.shape == (4, HIDDEN_SIZE), f"Output shape: {output.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN"
    assert not torch.isinf(output).any(), "Output contains Inf"

    print(f"  Input: {x.shape}, Output: {output.shape}")
    print(f"  Output range: [{output.min():.4f}, {output.max():.4f}]")

    print("PASS: Expert forward pass produces valid output")


def test_weight_key_naming():
    """Verify weight key names match expected BatchGen format."""
    stacked = create_synthetic_expert_weights_stacked()
    sliced = slice_expert_weights(stacked, 0)

    expected_keys = [
        "gate_proj.weight",
        "gate_proj.weight_scales",
        "gate_proj.bias",
        "up_proj.weight",
        "up_proj.weight_scales",
        "up_proj.bias",
        "down_proj.weight",
        "down_proj.weight_scales",
        "down_proj.bias",
    ]

    for key in expected_keys:
        assert key in sliced, f"Missing key: {key}"

    print("PASS: Weight key names match expected format")


def test_all_experts_unique():
    """Verify each expert gets unique sliced weights."""
    stacked = create_synthetic_expert_weights_stacked()

    # Slice a few experts
    expert_0 = slice_expert_weights(stacked, 0)
    expert_1 = slice_expert_weights(stacked, 1)

    # Weights should be different (very unlikely to be equal with random data)
    assert not torch.equal(expert_0["gate_proj.weight"], expert_1["gate_proj.weight"]), (
        "Expert 0 and 1 have identical gate_proj weights (should be different)"
    )

    print("PASS: Each expert has unique weights")


def test_with_checkpoint(checkpoint_path: str = None):
    """Test with actual GPT-OSS checkpoint if available."""
    if checkpoint_path is None:
        checkpoint_path = os.environ.get("GPT_OSS_CHECKPOINT_PATH")

    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        print("SKIP: Checkpoint path not provided or doesn't exist")
        print("  Set GPT_OSS_CHECKPOINT_PATH environment variable to test with real weights")
        return

    from safetensors import safe_open

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

    # Look for layer 0 MoE weights
    tensor_to_file = {}
    for sf_file in safetensor_files:
        with safe_open(sf_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor_to_file[key] = sf_file

    # Required tensors
    required = [
        "block.0.mlp.mlp1_weight.blocks",
        "block.0.mlp.mlp1_weight.scales",
        "block.0.mlp.mlp1_bias",
        "block.0.mlp.mlp2_weight.blocks",
        "block.0.mlp.mlp2_weight.scales",
        "block.0.mlp.mlp2_bias",
    ]

    missing = [t for t in required if t not in tensor_to_file]
    if missing:
        print(f"  Missing tensors: {missing}")
        return

    # Load tensors
    tensors = {}
    for name in required:
        with safe_open(tensor_to_file[name], framework="pt", device="cpu") as f:
            tensors[name] = f.get_tensor(name)

    stacked = {
        "mlp1_blocks": tensors["block.0.mlp.mlp1_weight.blocks"],
        "mlp1_scales": tensors["block.0.mlp.mlp1_weight.scales"],
        "mlp1_bias": tensors["block.0.mlp.mlp1_bias"],
        "mlp2_blocks": tensors["block.0.mlp.mlp2_weight.blocks"],
        "mlp2_scales": tensors["block.0.mlp.mlp2_weight.scales"],
        "mlp2_bias": tensors["block.0.mlp.mlp2_bias"],
    }

    print("  Loaded tensor shapes:")
    for name, tensor in stacked.items():
        print(f"    {name}: {tensor.shape}")

    # Test slicing
    sliced = slice_expert_weights(stacked, 0)
    print("  Sliced expert 0 shapes:")
    for name, tensor in sliced.items():
        print(f"    {name}: {tensor.shape}")

    # Test dequantization
    dequant = dequantize_expert_weights(sliced)
    print("  Dequantized shapes:")
    for name, tensor in dequant.items():
        print(f"    {name}: {tensor.shape}")

    # Test forward pass
    expert = SwiGLUExpert(HIDDEN_SIZE, INTERMEDIATE_SIZE).to(torch.bfloat16)

    with torch.no_grad():
        expert.gate_proj.weight.copy_(dequant["gate_proj.weight"])
        expert.gate_proj.bias.copy_(dequant["gate_proj.bias"])
        expert.up_proj.weight.copy_(dequant["up_proj.weight"])
        expert.up_proj.bias.copy_(dequant["up_proj.bias"])
        expert.down_proj.weight.copy_(dequant["down_proj.weight"])
        expert.down_proj.bias.copy_(dequant["down_proj.bias"])

    x = torch.randn(4, HIDDEN_SIZE, dtype=torch.bfloat16)
    output = expert(x)

    print(f"  Forward pass: input {x.shape} -> output {output.shape}")
    print(f"  Output range: [{output.min():.4f}, {output.max():.4f}]")

    assert not torch.isnan(output).any(), "Output contains NaN"
    assert not torch.isinf(output).any(), "Output contains Inf"

    print("PASS: Checkpoint expert weights work correctly")


# ============================================================================ #
# Main
# ============================================================================ #

def run_all_tests():
    """Run all expert weight tests."""
    print("=" * 60)
    print("GPT-OSS Expert Weight Test Suite")
    print("=" * 60)

    tests = [
        ("Stacked Weight Shapes", test_stacked_weight_shapes),
        ("Expert Slicing", test_expert_slicing),
        ("MLP1 Split Correctness", test_mlp1_split_correctness),
        ("Dequantized Weight Shapes", test_dequantized_weight_shapes),
        ("Weight Key Naming", test_weight_key_naming),
        ("All Experts Unique", test_all_experts_unique),
        ("Expert Forward Pass", test_expert_forward_pass),
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
            error_str = str(e)
            if "SKIP" in error_str:
                skipped += 1
            else:
                print(f"ERROR: {e}")
                import traceback
                traceback.print_exc()
                failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
