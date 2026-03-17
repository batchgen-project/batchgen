#!/usr/bin/env python3
"""Standalone test: Marlin weight prep + grouped GEMM + SiLU.

Tests the full pipeline without model integration:
1. Create synthetic K2.5-format INT4 weights
2. Convert to Marlin format
3. Run grouped GEMM
4. Compare against PyTorch reference

Usage (on H20):
    python -m batchgen.moe.test_marlin_standalone
"""

import torch
import numpy as np


def calc_diff(x, y):
    x, y = x.double(), y.double()
    denom = (x * x + y * y).sum()
    if denom == 0:
        return 0.0
    return (1 - 2 * (x * y).sum() / denom).item()


def create_k25_int4_weights(K, N, group_size=32, device="cuda"):
    """Create synthetic K2.5-format INT4 weights.

    Returns:
        packed: [N, K//8] int32 (K2.5 INT4 format)
        scales: [N, K//32] BF16
        w_ref_bf16: [K, N] BF16 dequantized reference for validation
    """
    # Generate random FP16 weights and quantize to INT4
    w_fp16 = torch.randn((N, K), dtype=torch.float16, device=device) * 0.1

    # Group quantize
    n_groups = K // group_size
    w_grouped = w_fp16.view(N, n_groups, group_size)

    # Symmetric quantization: scale = max(|max/7|, |min/-8|)
    max_val = w_grouped.max(dim=-1, keepdim=True).values
    min_val = w_grouped.min(dim=-1, keepdim=True).values
    scales = torch.max(max_val.abs() / 7.0, min_val.abs() / 8.0).clamp(min=1e-10)

    # Quantize to [0, 15]
    q = torch.round(w_grouped / scales).int() + 8
    q = torch.clamp(q, 0, 15)

    # Dequantized reference
    w_deq = ((q - 8).float() * scales.float()).view(N, K)
    w_ref_bf16 = w_deq.t().to(torch.bfloat16).contiguous()  # [K, N]

    # Pack into K2.5 int32 format: 8 nibbles per int32
    q_flat = q.view(N, K).int()
    K_div8 = K // 8
    packed = torch.zeros(N, K_div8, dtype=torch.int32, device=device)
    for i in range(8):
        packed |= (q_flat[:, i::8] & 0xF) << (i * 4)

    scales_out = scales.squeeze(-1).to(torch.bfloat16)  # [N, n_groups]
    return packed, scales_out, w_ref_bf16


def main():
    E, M, K, N = 4, 4, 7168, 2048
    device = "cuda"
    print(f"=== Marlin Standalone Test: E={E} M={M} K={K} N={N} ===\n")

    from batchgen.moe.marlin_weight_prep import convert_int4_to_marlin
    from batchgen.moe.marlin_grouped_moe import marlin_grouped_stage1

    # Create weights for each expert
    gate_packed_list, gate_scales_list, gate_refs = [], [], []
    up_packed_list, up_scales_list, up_refs = [], [], []

    print("Creating synthetic K2.5 INT4 weights...", end=" ", flush=True)
    for e in range(E):
        gp, gs, gr = create_k25_int4_weights(K, N, device=device)
        gate_packed_list.append(gp)
        gate_scales_list.append(gs)
        gate_refs.append(gr)

        up, us, ur = create_k25_int4_weights(K, N, device=device)
        up_packed_list.append(up)
        up_scales_list.append(us)
        up_refs.append(ur)
    print("done")

    # Convert to Marlin format
    print("Converting to Marlin format (gs=128)...", end=" ", flush=True)
    gate_marlin_qw, gate_marlin_s = [], []
    up_marlin_qw, up_marlin_s = [], []
    for e in range(E):
        gqw, gs = convert_int4_to_marlin(gate_packed_list[e], gate_scales_list[e], K, N)
        gate_marlin_qw.append(gqw)
        gate_marlin_s.append(gs)

        uqw, us = convert_int4_to_marlin(up_packed_list[e], up_scales_list[e], K, N)
        up_marlin_qw.append(uqw)
        up_marlin_s.append(us)
    print("done")

    # Build pointer arrays
    gate_ptrs = torch.tensor([w.data_ptr() for w in gate_marlin_qw], dtype=torch.int64, device=device)
    gate_scale_ptrs = torch.tensor([s.data_ptr() for s in gate_marlin_s], dtype=torch.int64, device=device)
    up_ptrs = torch.tensor([w.data_ptr() for w in up_marlin_qw], dtype=torch.int64, device=device)
    up_scale_ptrs = torch.tensor([s.data_ptr() for s in up_marlin_s], dtype=torch.int64, device=device)

    # Activations + expert offsets
    total_tokens = E * M
    A = torch.randn((total_tokens, K), dtype=torch.bfloat16, device=device) * 0.1
    expert_offsets = torch.arange(E + 1, dtype=torch.int32, device=device) * M

    # Workspace
    n_tiles = N // 256
    num_matrices = 2 * E
    workspace = torch.zeros(num_matrices * (n_tiles + 17), dtype=torch.int32, device=device)

    # Run Marlin S1
    print("Running Marlin grouped GEMM + SiLU...", end=" ", flush=True)
    intermediate = marlin_grouped_stage1(
        A, expert_offsets,
        gate_ptrs, gate_scale_ptrs, up_ptrs, up_scale_ptrs,
        N, K, workspace)
    torch.cuda.synchronize()
    print("done")

    # Reference: BF16 matmul + SiLU
    # Note: reference uses the ORIGINAL gs=32 dequant (gate_refs),
    # while kernel uses re-quantized gs=128 Marlin weights.
    # Double quantization (gs=32→fp16→gs=128) introduces extra error.
    print("\nValidation (double-quantized: gs=32→FP16→gs=128 vs gs=32 reference):")
    all_pass = True
    for e in range(E):
        a_e = A[e * M:(e + 1) * M]

        ref_gate = torch.mm(a_e, gate_refs[e])  # [M, N] BF16
        ref_up = torch.mm(a_e, up_refs[e])
        ref_silu = torch.nn.functional.silu(ref_gate.float()) * ref_up.float()
        ref_out = ref_silu.bfloat16()

        out_e = intermediate[e * M:(e + 1) * M]

        cd = calc_diff(out_e, ref_out)
        # Double quantization: expect calc_diff < 1e-2 (looser than single-quant 1e-3)
        status = "PASS" if cd < 1e-2 else "FAIL"
        if cd >= 1e-2:
            all_pass = False
        print(f"  E{e} calc_diff={cd:.2e} [{status}]")

    print(f"\nOverall: {'PASS' if all_pass else 'FAIL'}")
    print(f"Note: calc_diff > ~1e-3 expected due to double quantization (gs32→fp16→gs128)")


if __name__ == "__main__":
    main()
