"""Verify MLP1 weight slicing produces equivalent results to reference.

This test compares:
1. Reference: Fused MLP1 matmul followed by swiglu output split
2. BatchGen: Sliced weights (gate/up) with separate matmuls + swiglu

The MXFP4 weights in the checkpoint are stored INTERLEAVED:
- Row 0: gate weight row 0
- Row 1: up weight row 0
- Row 2: gate weight row 1
- Row 3: up weight row 1
- ...

Reference approach:
- MLP1 output = mlp1_weight @ input  # [5760, 2880] @ [2880] = [5760]
- swiglu splits output[::2] (gate) and output[1::2] (up)

BatchGen approach:
- gate_weight = mlp1_weight[::2]  # [2880, 2880]
- up_weight = mlp1_weight[1::2]   # [2880, 2880]
- gate_out = gate_weight @ input  # [2880]
- up_out = up_weight @ input      # [2880]
- swiglu(gate_out, up_out)

These should be mathematically equivalent.
"""

import sys
from pathlib import Path

# Add gpt-oss to path
GPT_OSS_PATH = "/data2/tairan/workspace/gpt-oss"
sys.path.insert(0, GPT_OSS_PATH)

import torch
from safetensors import safe_open


# FP4 lookup table (from gpt-oss reference)
FP4_VALUES = [
    +0.0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
]


def mxfp4_dequantize(packed: torch.Tensor, scales: torch.Tensor, dtype=torch.bfloat16):
    """Dequantize MXFP4 packed tensor (matches gpt-oss reference)."""
    device = packed.device
    lut = torch.tensor(FP4_VALUES, dtype=torch.float32, device=device)

    # Handle 3D packed: [rows, groups, bytes_per_group] -> [rows, groups * bytes_per_group]
    if packed.dim() == 3:
        packed = packed.view(packed.shape[0], -1)

    # Unpack nibbles
    idx_lo = (packed & 0x0F).to(torch.long)
    idx_hi = (packed >> 4).to(torch.long)

    # Lookup FP4 values
    val_lo = lut[idx_lo]
    val_hi = lut[idx_hi]

    # Interleave: [lo0, hi0, lo1, hi1, ...]
    unpacked = torch.empty(packed.shape[:-1] + (packed.shape[-1] * 2,),
                          dtype=torch.float32, device=device)
    unpacked[..., 0::2] = val_lo
    unpacked[..., 1::2] = val_hi

    # Apply scales
    exponents = scales.to(torch.int32) - 127
    exponents = exponents.clamp(min=-126, max=127)

    # Expand scales: each scale covers 32 elements
    n_elements = unpacked.shape[-1]
    n_blocks = scales.shape[-1]
    expanded_exp = exponents.unsqueeze(-1).expand(*exponents.shape, 32)
    expanded_exp = expanded_exp.reshape(*exponents.shape[:-1], n_blocks * 32)
    if expanded_exp.shape[-1] > n_elements:
        expanded_exp = expanded_exp[..., :n_elements]

    result = torch.ldexp(unpacked, expanded_exp)
    return result.to(dtype)


def reference_swiglu(x: torch.Tensor, alpha: float = 1.702, limit: float = 7.0):
    """Reference swiglu that splits interleaved output."""
    x_glu = x[..., ::2]
    x_linear = x[..., 1::2]
    x_glu = x_glu.clamp(max=limit)
    x_linear = x_linear.clamp(min=-limit, max=limit)
    out_glu = x_glu * torch.sigmoid(alpha * x_glu)
    return out_glu * (x_linear + 1)


def batchgen_swiglu(gate: torch.Tensor, up: torch.Tensor, alpha: float = 1.702, limit: float = 7.0):
    """BatchGen swiglu with separate gate and up tensors."""
    gate = gate.clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    out_glu = gate * torch.sigmoid(alpha * gate)
    return out_glu * (up + 1)


def main():
    checkpoint_path = "/data2/tairan/modelscope/hub/models/openai/gpt-oss-120b/original"

    # Find safetensor files
    safetensor_files = list(Path(checkpoint_path).glob("*.safetensors"))
    print(f"Found {len(safetensor_files)} safetensor files")

    if not safetensor_files:
        print("No safetensor files found!")
        return

    # Load MLP1 weights for one expert
    print(f"\nLoading MLP1 weights from {safetensor_files[0].name}...")
    with safe_open(safetensor_files[0], framework="pt", device="cpu") as f:
        keys = list(f.keys())

        # Find MLP1 blocks and scales for first expert
        mlp1_blocks_key = None
        mlp1_scales_key = None
        mlp1_bias_key = None

        for key in keys:
            if "block.0.mlp.mlp1_weight.blocks" in key:
                mlp1_blocks_key = key
            if "block.0.mlp.mlp1_weight.scales" in key:
                mlp1_scales_key = key
            if "block.0.mlp.mlp1_bias" in key:
                mlp1_bias_key = key

        if not mlp1_blocks_key or not mlp1_scales_key:
            print("MLP1 weight not found!")
            return

        mlp1_blocks = f.get_tensor(mlp1_blocks_key)  # [128, 5760, 90, 16]
        mlp1_scales = f.get_tensor(mlp1_scales_key)  # [128, 5760, 90]
        mlp1_bias = f.get_tensor(mlp1_bias_key) if mlp1_bias_key else None

        print(f"MLP1 blocks shape: {mlp1_blocks.shape}")
        print(f"MLP1 scales shape: {mlp1_scales.shape}")
        if mlp1_bias is not None:
            print(f"MLP1 bias shape: {mlp1_bias.shape}")

    # Extract first expert
    expert_idx = 0
    expert_blocks = mlp1_blocks[expert_idx]  # [5760, 90, 16]
    expert_scales = mlp1_scales[expert_idx]  # [5760, 90]
    expert_bias = mlp1_bias[expert_idx] if mlp1_bias is not None else None

    print(f"\n=== Expert {expert_idx} ===")
    print(f"Expert blocks: {expert_blocks.shape}")
    print(f"Expert scales: {expert_scales.shape}")

    # Dequantize full MLP1 weight (reference approach)
    print("\nDequantizing full MLP1 weight...")
    mlp1_weight_full = mxfp4_dequantize(expert_blocks, expert_scales)
    print(f"Full MLP1 weight shape: {mlp1_weight_full.shape}")
    print(f"  Expected: [5760, 2880] (intermediate_size*2, hidden_size)")

    # BatchGen approach: slice BEFORE dequantization
    print("\n=== BatchGen Slicing (BEFORE dequant) ===")
    gate_blocks = expert_blocks[::2].contiguous()   # Even rows [2880, 90, 16]
    gate_scales = expert_scales[::2].contiguous()   # Even rows [2880, 90]
    up_blocks = expert_blocks[1::2].contiguous()    # Odd rows [2880, 90, 16]
    up_scales = expert_scales[1::2].contiguous()    # Odd rows [2880, 90]

    print(f"gate_blocks: {gate_blocks.shape}")
    print(f"up_blocks: {up_blocks.shape}")

    # Dequantize sliced weights
    gate_weight = mxfp4_dequantize(gate_blocks, gate_scales)
    up_weight = mxfp4_dequantize(up_blocks, up_scales)
    print(f"gate_weight shape: {gate_weight.shape}")
    print(f"up_weight shape: {up_weight.shape}")

    # Verify sliced weights match full weight rows
    print("\n=== Verify Weight Equivalence ===")
    # Full weight row 0 should equal gate weight row 0
    match_gate_0 = torch.allclose(mlp1_weight_full[0], gate_weight[0], atol=1e-4)
    # Full weight row 1 should equal up weight row 0
    match_up_0 = torch.allclose(mlp1_weight_full[1], up_weight[0], atol=1e-4)
    # Full weight row 2 should equal gate weight row 1
    match_gate_1 = torch.allclose(mlp1_weight_full[2], gate_weight[1], atol=1e-4)
    # Full weight row 3 should equal up weight row 1
    match_up_1 = torch.allclose(mlp1_weight_full[3], up_weight[1], atol=1e-4)

    print(f"mlp1_weight_full[0] == gate_weight[0]: {match_gate_0}")
    print(f"mlp1_weight_full[1] == up_weight[0]: {match_up_0}")
    print(f"mlp1_weight_full[2] == gate_weight[1]: {match_gate_1}")
    print(f"mlp1_weight_full[3] == up_weight[1]: {match_up_1}")

    if not all([match_gate_0, match_up_0, match_gate_1, match_up_1]):
        print("\n*** WEIGHT SLICING MISMATCH DETECTED! ***")
        print("Debugging values:")
        print(f"  mlp1_weight_full[0, :8]: {mlp1_weight_full[0, :8].tolist()}")
        print(f"  gate_weight[0, :8]: {gate_weight[0, :8].tolist()}")
        print(f"  mlp1_weight_full[1, :8]: {mlp1_weight_full[1, :8].tolist()}")
        print(f"  up_weight[0, :8]: {up_weight[0, :8].tolist()}")
        return
    else:
        print("\n✓ Weight slicing is CORRECT!")

    # Test forward pass equivalence
    print("\n=== Test Forward Pass Equivalence ===")
    batch_size = 4
    hidden_size = 2880

    # Random input
    torch.manual_seed(42)
    x = torch.randn(batch_size, hidden_size, dtype=torch.bfloat16)
    print(f"Input shape: {x.shape}")

    # Reference approach: fused matmul + swiglu split
    print("\n--- Reference Approach ---")
    mlp1_out_ref = torch.mm(x.float(), mlp1_weight_full.T.float())  # [batch, 5760]
    if expert_bias is not None:
        mlp1_out_ref = mlp1_out_ref + expert_bias.float()
    swiglu_out_ref = reference_swiglu(mlp1_out_ref)  # [batch, 2880]
    print(f"MLP1 output shape: {mlp1_out_ref.shape}")
    print(f"SwiGLU output shape: {swiglu_out_ref.shape}")

    # BatchGen approach: separate matmuls + swiglu
    print("\n--- BatchGen Approach ---")
    gate_out = torch.mm(x.float(), gate_weight.T.float())  # [batch, 2880]
    up_out = torch.mm(x.float(), up_weight.T.float())      # [batch, 2880]
    if expert_bias is not None:
        gate_bias = expert_bias[::2]
        up_bias = expert_bias[1::2]
        gate_out = gate_out + gate_bias.float()
        up_out = up_out + up_bias.float()
    swiglu_out_bg = batchgen_swiglu(gate_out, up_out)
    print(f"Gate output shape: {gate_out.shape}")
    print(f"Up output shape: {up_out.shape}")
    print(f"SwiGLU output shape: {swiglu_out_bg.shape}")

    # Compare outputs
    print("\n=== Output Comparison ===")

    # Intermediate comparison
    gate_out_ref = mlp1_out_ref[..., ::2]  # Reference gate from interleaved output
    up_out_ref = mlp1_out_ref[..., 1::2]   # Reference up from interleaved output

    gate_match = torch.allclose(gate_out, gate_out_ref, atol=1e-3, rtol=1e-3)
    up_match = torch.allclose(up_out, up_out_ref, atol=1e-3, rtol=1e-3)

    print(f"Gate output match: {gate_match}")
    if not gate_match:
        diff = (gate_out - gate_out_ref).abs()
        print(f"  Max diff: {diff.max().item():.6f}")
        print(f"  Mean diff: {diff.mean().item():.6f}")

    print(f"Up output match: {up_match}")
    if not up_match:
        diff = (up_out - up_out_ref).abs()
        print(f"  Max diff: {diff.max().item():.6f}")
        print(f"  Mean diff: {diff.mean().item():.6f}")

    # Final swiglu output comparison
    swiglu_match = torch.allclose(swiglu_out_bg, swiglu_out_ref, atol=1e-3, rtol=1e-3)
    print(f"SwiGLU output match: {swiglu_match}")
    if not swiglu_match:
        diff = (swiglu_out_bg - swiglu_out_ref).abs()
        print(f"  Max diff: {diff.max().item():.6f}")
        print(f"  Mean diff: {diff.mean().item():.6f}")
        # Show sample values
        print(f"  Reference[0,:8]: {swiglu_out_ref[0,:8].tolist()}")
        print(f"  BatchGen[0,:8]: {swiglu_out_bg[0,:8].tolist()}")

    if gate_match and up_match and swiglu_match:
        print("\n✓✓✓ BatchGen MLP1 slicing is MATHEMATICALLY EQUIVALENT to reference! ✓✓✓")
        print("\nConclusion: The issue is NOT in MLP1 weight slicing.")
        print("Look elsewhere: KV cache management, attention, position encoding, etc.")
    else:
        print("\n*** MLP1 FORWARD PASS MISMATCH! ***")
        print("The slicing approach produces different results than the reference.")


if __name__ == "__main__":
    main()
