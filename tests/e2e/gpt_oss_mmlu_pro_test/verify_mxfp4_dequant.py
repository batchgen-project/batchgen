"""Verify MXFP4 dequantization by checking weight statistics."""

import os
import sys
from pathlib import Path

# Add gpt-oss reference repo to path
GPT_OSS_PATH = os.environ.get("GPT_OSS_PATH", "")
if not GPT_OSS_PATH:
    raise RuntimeError("Set GPT_OSS_PATH to the gpt-oss reference repo checkout.")
sys.path.insert(0, GPT_OSS_PATH)

import torch
from safetensors import safe_open

# MXFP4 constants from gpt-oss reference
FP4_VALUES = [
    +0.0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
]


def dequant_mxfp4_manual(blocks: torch.Tensor, scales: torch.Tensor, dtype=torch.bfloat16):
    """Manual MXFP4 dequantization for verification."""
    lut = torch.tensor(FP4_VALUES, dtype=dtype)

    # scales are uint8, need to convert to exponent: exp = scale - 127
    exp = scales.to(torch.int32) - 127

    # blocks are packed: each byte has 2 fp4 values (lo nibble, hi nibble)
    idx_lo = (blocks & 0x0F).to(torch.long)
    idx_hi = (blocks >> 4).to(torch.long)

    # Look up fp4 values
    val_lo = lut[idx_lo]
    val_hi = lut[idx_hi]

    # Interleave: [lo0, hi0, lo1, hi1, ...]
    # Shape: [..., G, B] -> [..., G, B*2]
    result = torch.stack([val_lo, val_hi], dim=-1)
    result = result.view(*result.shape[:-2], -1)

    # Apply scale (2^exp)
    result = torch.ldexp(result, exp.unsqueeze(-1).expand_as(result))

    return result


def main():
    checkpoint_path = os.environ.get("GPT_OSS_CHECKPOINT_PATH", "")
    if not checkpoint_path:
        raise RuntimeError("Set GPT_OSS_CHECKPOINT_PATH to the GPT-OSS-120B 'original' checkpoint dir.")

    # Find safetensor files
    safetensor_files = list(Path(checkpoint_path).glob("*.safetensors"))
    print(f"Found {len(safetensor_files)} safetensor files")

    if not safetensor_files:
        print("No safetensor files found!")
        return

    # Check what tensors are in the first file
    print(f"\nChecking {safetensor_files[0].name}...")
    with safe_open(safetensor_files[0], framework="pt", device="cpu") as f:
        keys = list(f.keys())
        print(f"Found {len(keys)} tensors")
        print(f"First 20 tensor names: {keys[:20]}")

        # Look for MLP weights
        mlp1_blocks = None
        mlp1_scales = None
        for key in keys:
            if "mlp1_weight.blocks" in key:
                mlp1_blocks = key
            if "mlp1_weight.scales" in key:
                mlp1_scales = key
            if mlp1_blocks and mlp1_scales:
                break

        if mlp1_blocks and mlp1_scales:
            print(f"\nFound MLP1 weight:")
            print(f"  blocks: {mlp1_blocks}")
            print(f"  scales: {mlp1_scales}")

            blocks = f.get_tensor(mlp1_blocks)
            scales = f.get_tensor(mlp1_scales)

            print(f"\nBlocks shape: {blocks.shape}, dtype: {blocks.dtype}")
            print(f"Scales shape: {scales.shape}, dtype: {scales.dtype}")

            # Check scale distribution
            scale_values = scales.flatten().to(torch.int32)
            print(f"\nScale value stats:")
            print(f"  min: {scale_values.min().item()}")
            print(f"  max: {scale_values.max().item()}")
            print(f"  mean: {scale_values.float().mean().item():.2f}")
            print(f"  unique values: {scale_values.unique().shape[0]}")

            # After subtracting 127 (the bias), what's the exponent range?
            exp_values = scale_values - 127
            print(f"\nExponent (scale - 127) stats:")
            print(f"  min: {exp_values.min().item()}")
            print(f"  max: {exp_values.max().item()}")
            print(f"  mean: {exp_values.float().mean().item():.2f}")

            # Dequantize a small portion
            print("\nDequantizing first expert, first 32 elements...")
            # blocks shape: [num_experts, G, B] where G is groups, B is bytes per group
            # For GPT-OSS: blocks is [128, G, 16] (16 bytes = 32 nibbles)
            # scales is [128, G]

            expert0_blocks = blocks[0]  # [G, B]
            expert0_scales = scales[0]  # [G]

            print(f"Expert 0 blocks shape: {expert0_blocks.shape}")
            print(f"Expert 0 scales shape: {expert0_scales.shape}")

            # Shape is [5760, 90, 16] for blocks, [5760, 90] for scales
            # 5760 = intermediate_size * 2 (rows)
            # 90 = hidden_size / 32 (groups, each group has 32 elements)
            # 16 = bytes per group (32 nibbles)

            # Take first row, first group
            row0_group0_blocks = expert0_blocks[0, 0]  # [16 bytes]
            row0_group0_scale = expert0_scales[0, 0]   # scalar

            print(f"\nRow 0, Group 0:")
            print(f"  blocks (16 bytes): {row0_group0_blocks.tolist()}")
            print(f"  scale: {row0_group0_scale.item()}")
            print(f"  exponent: {row0_group0_scale.item() - 127}")

            # Manual dequant of this group
            exp = row0_group0_scale.to(torch.int32).item() - 127

            values = []
            for byte in row0_group0_blocks:
                byte_val = byte.item()
                lo = byte_val & 0x0F
                hi = byte_val >> 4
                val_lo = FP4_VALUES[lo] * (2.0 ** exp)
                val_hi = FP4_VALUES[hi] * (2.0 ** exp)
                values.extend([val_lo, val_hi])

            print(f"\n  Dequantized values (first 32 elements of row 0):")
            print(f"  {values}")
            print(f"  min: {min(values):.6f}, max: {max(values):.6f}")

            # Now use the reference implementation's dequant
            print("\n\nUsing gpt-oss reference dequantization...")
            from gpt_oss.torch.weights import Checkpoint

            ckpt = Checkpoint(checkpoint_path, torch.device("cpu"))

            # Get a weight using the checkpoint API
            weight = ckpt.get("block.0.mlp.mlp1_weight")
            print(f"MLP1 weight shape: {weight.shape}, dtype: {weight.dtype}")
            print(f"MLP1 weight stats:")
            print(f"  min: {weight.min().item():.6f}")
            print(f"  max: {weight.max().item():.6f}")
            print(f"  mean: {weight.mean().item():.6f}")
            print(f"  std: {weight.std().item():.6f}")

            # Check for reasonable weight distribution
            # Typical neural network weights should have std around 0.01-0.1
            # and be roughly centered around 0
            if abs(weight.mean().item()) > 1.0:
                print("  WARNING: Mean is suspiciously large!")
            if weight.std().item() > 10.0:
                print("  WARNING: Std is suspiciously large!")
            if weight.std().item() < 0.001:
                print("  WARNING: Std is suspiciously small!")

            # Check first expert's first 64 values
            print(f"\nFirst expert, first 64 values:")
            print(f"  {weight[0, :64].tolist()}")

        else:
            print("No MLP1 weight blocks/scales found!")
            print("This might not be the original MXFP4 checkpoint.")
            print("\nAll tensor names:")
            for key in keys:
                print(f"  {key}")


if __name__ == "__main__":
    main()
