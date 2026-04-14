"""Verify: Marlin checkpoint weights round-trip to original raw INT4.

Loads one expert's gate weights from both checkpoints:
1. Original (converted_ckpt_bak): raw INT4 [N, K//8]
2. Marlin (converted_ckpt): Marlin [K//16, N*2]

Transforms Marlin→raw and compares. Must be bit-identical.
"""
import json
import torch
import ctypes
import sys

def load_tensor_from_bin(bin_path, json_path, tensor_name):
    """Load a specific tensor from converted checkpoint."""
    with open(json_path) as f:
        meta = json.load(f)
    info = meta["state_dict"].get(tensor_name)
    if info is None:
        print(f"Tensor {tensor_name} not found in {json_path}")
        return None
    dtype_map = {"int32": torch.int32, "bfloat16": torch.bfloat16, "float16": torch.float16, "uint8": torch.uint8}
    dtype = dtype_map[info["dtype"]]
    shape = tuple(info["shape"])
    offset = info["offset"]
    byte_size = info["byte_size"]

    with open(bin_path, "rb") as f:
        f.seek(offset)
        data = f.read(byte_size)
    tensor = torch.frombuffer(bytearray(data), dtype=dtype).reshape(shape)
    return tensor

# Paths
marlin_dir = sys.argv[1] if len(sys.argv) > 1 else "/data3/tairan/workspace/models/moonshotai/Kimi-K2.5/converted_ckpt"
raw_dir = sys.argv[2] if len(sys.argv) > 2 else "/data3/tairan/workspace/models/moonshotai/Kimi-K2.5/converted_ckpt_bak"

# Use shard 2 (first shard with expert weights)
shard = "model-00002-of-000064"
marlin_bin = f"{marlin_dir}/{shard}.bin"
marlin_json = f"{marlin_dir}/{shard}.json"
raw_bin = f"{raw_dir}/{shard}.bin"
raw_json = f"{raw_dir}/{shard}.json"

# Find first expert gate weight
with open(marlin_json) as f:
    meta = json.load(f)
gate_keys = [k for k in meta["state_dict"] if "gate_proj.weight_packed" in k]
if not gate_keys:
    print("No gate weights in shard 2")
    sys.exit(1)

key = gate_keys[0]
print(f"Testing: {key}")

# Load from both checkpoints
marlin_packed = load_tensor_from_bin(marlin_bin, marlin_json, key)
raw_packed = load_tensor_from_bin(raw_bin, raw_json, key)

print(f"  Marlin shape: {marlin_packed.shape}, dtype: {marlin_packed.dtype}")
print(f"  Raw shape:    {raw_packed.shape}, dtype: {raw_packed.dtype}")

# Load scales
scale_key = key.replace("weight_packed", "weight_scale")
marlin_scale = load_tensor_from_bin(marlin_bin, marlin_json, scale_key)
raw_scale = load_tensor_from_bin(raw_bin, raw_json, scale_key)
print(f"  Marlin scale shape: {marlin_scale.shape}")
print(f"  Raw scale shape:    {raw_scale.shape}")

# Transform Marlin→raw on GPU
device = "cuda"
from batchgen.moe.marlin_transform import marlin_to_wgmma_fused_gpu
marlin_packed_gpu = marlin_packed.to(device)
marlin_scale_gpu = marlin_scale.to(device)

K = marlin_packed.shape[0] * 16
N = marlin_packed.shape[1] // 2
print(f"\n  K={K}, N={N}")

recovered_packed, recovered_scale = marlin_to_wgmma_fused_gpu(
    marlin_packed_gpu, marlin_scale_gpu, K, N)
torch.cuda.synchronize()

recovered_packed = recovered_packed.cpu()
recovered_scale = recovered_scale.cpu()

print(f"  Recovered shape: {recovered_packed.shape}")
print(f"  Recovered scale shape: {recovered_scale.shape}")

# Compare
w_match = torch.equal(raw_packed, recovered_packed)
s_match = torch.equal(raw_scale, recovered_scale)
print(f"\n  Weights bit-identical: {w_match}")
print(f"  Scales bit-identical:  {s_match}")

if not w_match:
    diff = (raw_packed != recovered_packed).sum().item()
    print(f"  Weight mismatches: {diff} / {raw_packed.numel()}")
    # Show first mismatch
    idx = (raw_packed != recovered_packed).nonzero()[0]
    print(f"  First mismatch at {idx.tolist()}: raw={raw_packed[idx[0], idx[1]].item():#010x}, "
          f"recovered={recovered_packed[idx[0], idx[1]].item():#010x}")
if not s_match:
    diff = (raw_scale != recovered_scale).sum().item()
    print(f"  Scale mismatches: {diff} / {raw_scale.numel()}")

status = "PASS" if (w_match and s_match) else "FAIL"
print(f"\n  [{status}]")
