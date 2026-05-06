"""Dump all tensor metadata from GPT-OSS-120B checkpoint.

This script provides comprehensive knowledge of all tensors in the raw safetensors.
"""

import json
import os
from pathlib import Path
from collections import defaultdict

import torch
from safetensors import safe_open


def main():
    checkpoint_path = os.environ.get("GPT_OSS_CHECKPOINT_PATH", "")
    if not checkpoint_path:
        raise RuntimeError("Set GPT_OSS_CHECKPOINT_PATH to the GPT-OSS-120B 'original' checkpoint dir.")
    checkpoint_path = Path(checkpoint_path)

    # Load config
    config_path = checkpoint_path / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    print("=" * 80)
    print("MODEL CONFIG")
    print("=" * 80)
    for k, v in sorted(config.items()):
        print(f"  {k}: {v}")

    # Find all safetensor files
    safetensor_files = sorted(checkpoint_path.glob("*.safetensors"))
    print(f"\n{'=' * 80}")
    print(f"SAFETENSOR FILES: {len(safetensor_files)}")
    print("=" * 80)
    for f in safetensor_files:
        print(f"  {f.name}")

    # Collect all tensor info
    all_tensors = {}
    for sf_file in safetensor_files:
        with safe_open(sf_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                all_tensors[key] = {
                    "file": sf_file.name,
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "numel": tensor.numel(),
                }

    print(f"\n{'=' * 80}")
    print(f"TOTAL TENSORS: {len(all_tensors)}")
    print("=" * 80)

    # Group by category
    categories = defaultdict(list)
    for name, info in sorted(all_tensors.items()):
        if "embedding" in name or "unembedding" in name:
            categories["embedding"].append((name, info))
        elif "norm" in name:
            categories["norm"].append((name, info))
        elif "attn" in name:
            if "qkv" in name:
                categories["attn_qkv"].append((name, info))
            elif "out" in name:
                categories["attn_out"].append((name, info))
            elif "sinks" in name:
                categories["attn_sinks"].append((name, info))
            else:
                categories["attn_other"].append((name, info))
        elif "mlp" in name:
            if "gate" in name:
                categories["mlp_gate"].append((name, info))
            elif "mlp1" in name:
                if "blocks" in name:
                    categories["mlp1_blocks"].append((name, info))
                elif "scales" in name:
                    categories["mlp1_scales"].append((name, info))
                elif "bias" in name:
                    categories["mlp1_bias"].append((name, info))
                else:
                    categories["mlp1_other"].append((name, info))
            elif "mlp2" in name:
                if "blocks" in name:
                    categories["mlp2_blocks"].append((name, info))
                elif "scales" in name:
                    categories["mlp2_scales"].append((name, info))
                elif "bias" in name:
                    categories["mlp2_bias"].append((name, info))
                else:
                    categories["mlp2_other"].append((name, info))
            else:
                categories["mlp_other"].append((name, info))
        else:
            categories["other"].append((name, info))

    # Print by category
    for cat_name in sorted(categories.keys()):
        tensors = categories[cat_name]
        print(f"\n{'=' * 80}")
        print(f"CATEGORY: {cat_name} ({len(tensors)} tensors)")
        print("=" * 80)

        # Show first few examples
        for name, info in tensors[:3]:
            print(f"\n  {name}")
            print(f"    file:  {info['file']}")
            print(f"    shape: {info['shape']}")
            print(f"    dtype: {info['dtype']}")
            print(f"    numel: {info['numel']:,}")

        if len(tensors) > 3:
            print(f"\n  ... and {len(tensors) - 3} more")

        # Check if all tensors in category have same shape pattern
        shapes = set(tuple(info['shape']) for _, info in tensors)
        if len(shapes) == 1:
            print(f"\n  All {len(tensors)} tensors have shape: {list(shapes)[0]}")
        else:
            print(f"\n  Different shapes found: {len(shapes)}")
            for shape in sorted(shapes):
                count = sum(1 for _, info in tensors if tuple(info['shape']) == shape)
                print(f"    {list(shape)}: {count} tensors")

    # Detailed analysis of MLP weights (the key tensors for expert handling)
    print(f"\n{'=' * 80}")
    print("DETAILED MLP WEIGHT ANALYSIS")
    print("=" * 80)

    # MLP1 blocks/scales
    mlp1_blocks = [t for t in all_tensors.items() if "mlp1_weight.blocks" in t[0]]
    mlp1_scales = [t for t in all_tensors.items() if "mlp1_weight.scales" in t[0]]

    if mlp1_blocks:
        name, info = mlp1_blocks[0]
        print(f"\nMLP1 Weight Blocks (example: {name})")
        print(f"  Shape: {info['shape']}")
        print(f"  Interpretation:")
        shape = info['shape']
        if len(shape) == 4:
            print(f"    dim 0: {shape[0]} = num_experts")
            print(f"    dim 1: {shape[1]} = intermediate_size * 2 (gate + up interleaved)")
            print(f"    dim 2: {shape[2]} = hidden_size / 32 (number of MXFP4 groups)")
            print(f"    dim 3: {shape[3]} = 16 bytes per group (32 FP4 values)")
            print(f"  After dequant: [{shape[0]}, {shape[1]}, {shape[2] * shape[3] * 2}]")
            print(f"               = [{shape[0]}, {shape[1]}, {shape[2] * 32}]")
            print(f"  Expected: [num_experts, intermediate_size*2, hidden_size]")
            print(f"          = [128, 5760, 2880]")

    if mlp1_scales:
        name, info = mlp1_scales[0]
        print(f"\nMLP1 Weight Scales (example: {name})")
        print(f"  Shape: {info['shape']}")

    # MLP2 blocks/scales
    mlp2_blocks = [t for t in all_tensors.items() if "mlp2_weight.blocks" in t[0]]
    mlp2_scales = [t for t in all_tensors.items() if "mlp2_weight.scales" in t[0]]

    if mlp2_blocks:
        name, info = mlp2_blocks[0]
        print(f"\nMLP2 Weight Blocks (example: {name})")
        print(f"  Shape: {info['shape']}")
        shape = info['shape']
        if len(shape) == 4:
            print(f"  Interpretation:")
            print(f"    dim 0: {shape[0]} = num_experts")
            print(f"    dim 1: {shape[1]} = hidden_size")
            print(f"    dim 2: {shape[2]} = intermediate_size / 32 (number of MXFP4 groups)")
            print(f"    dim 3: {shape[3]} = 16 bytes per group")
            print(f"  After dequant: [{shape[0]}, {shape[1]}, {shape[2] * shape[3] * 2}]")
            print(f"               = [{shape[0]}, {shape[1]}, {shape[2] * 32}]")

    # MLP biases
    mlp1_bias = [t for t in all_tensors.items() if "mlp1_bias" in t[0] and "blocks" not in t[0] and "scales" not in t[0]]
    mlp2_bias = [t for t in all_tensors.items() if "mlp2_bias" in t[0] and "blocks" not in t[0] and "scales" not in t[0]]

    if mlp1_bias:
        name, info = mlp1_bias[0]
        print(f"\nMLP1 Bias (example: {name})")
        print(f"  Shape: {info['shape']}")
        print(f"  dtype: {info['dtype']}")

    if mlp2_bias:
        name, info = mlp2_bias[0]
        print(f"\nMLP2 Bias (example: {name})")
        print(f"  Shape: {info['shape']}")
        print(f"  dtype: {info['dtype']}")

    # Attention weights
    print(f"\n{'=' * 80}")
    print("ATTENTION WEIGHT ANALYSIS")
    print("=" * 80)

    qkv_weights = [t for t in all_tensors.items() if "qkv.weight" in t[0]]
    if qkv_weights:
        name, info = qkv_weights[0]
        print(f"\nQKV Weight (example: {name})")
        print(f"  Shape: {info['shape']}")
        print(f"  dtype: {info['dtype']}")
        shape = info['shape']
        if len(shape) == 2:
            print(f"  Interpretation:")
            print(f"    dim 0: {shape[0]} = (num_q_heads + 2*num_kv_heads) * head_dim")
            print(f"         = (64 + 2*8) * 64 = 80 * 64 = 5120")
            print(f"    dim 1: {shape[1]} = hidden_size = 2880")

    out_weights = [t for t in all_tensors.items() if "attn.out.weight" in t[0]]
    if out_weights:
        name, info = out_weights[0]
        print(f"\nAttention Out Weight (example: {name})")
        print(f"  Shape: {info['shape']}")
        print(f"  dtype: {info['dtype']}")

    sinks = [t for t in all_tensors.items() if "sinks" in t[0]]
    if sinks:
        name, info = sinks[0]
        print(f"\nSinks (example: {name})")
        print(f"  Shape: {info['shape']}")
        print(f"  dtype: {info['dtype']}")

    # Print sample values for key tensors
    print(f"\n{'=' * 80}")
    print("SAMPLE VALUES FROM KEY TENSORS")
    print("=" * 80)

    sf_file = safetensor_files[0]
    with safe_open(sf_file, framework="pt", device="cpu") as f:
        for key in list(f.keys())[:5]:
            tensor = f.get_tensor(key)
            print(f"\n{key}")
            print(f"  shape: {tensor.shape}, dtype: {tensor.dtype}")
            flat = tensor.flatten()
            if flat.dtype in [torch.float16, torch.bfloat16, torch.float32]:
                print(f"  first 10: {flat[:10].tolist()}")
                print(f"  stats: min={flat.min().item():.6f}, max={flat.max().item():.6f}, mean={flat.float().mean().item():.6f}")
            elif flat.dtype == torch.uint8:
                print(f"  first 10 (raw bytes): {flat[:10].tolist()}")
            else:
                print(f"  first 10: {flat[:10].tolist()}")


if __name__ == "__main__":
    main()
