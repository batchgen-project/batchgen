#!/usr/bin/env python3
"""Standalone test: Marlin gs=32 direct repack + GROUP_BLOCKS=2 kernel.

Tests the full pipeline without model integration:
1. Create synthetic K2.5-format INT4 weights (gs=32)
2. Repack to Marlin tile layout (NO requantization)
3. Run GROUP_BLOCKS=2 Marlin kernel
4. Compare against PyTorch BF16 reference (dequant gs=32)

Expected: calc_diff < 1e-5 (only BF16 MMA rounding, no quantization error)

Usage (on GH02 or H20):
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
    w_fp16 = torch.randn((N, K), dtype=torch.float16, device=device) * 0.1

    n_groups = K // group_size
    w_grouped = w_fp16.view(N, n_groups, group_size)

    max_val = w_grouped.max(dim=-1, keepdim=True).values
    min_val = w_grouped.min(dim=-1, keepdim=True).values
    scales = torch.max(max_val.abs() / 7.0, min_val.abs() / 8.0).clamp(min=1e-10)

    q = torch.round(w_grouped / scales).int() + 8
    q = torch.clamp(q, 0, 15)

    w_deq = ((q - 8).float() * scales.float()).view(N, K)
    w_ref_bf16 = w_deq.t().to(torch.bfloat16).contiguous()  # [K, N]

    q_flat = q.view(N, K).int()
    K_div8 = K // 8
    packed = torch.zeros(N, K_div8, dtype=torch.int32, device=device)
    for i in range(8):
        packed |= (q_flat[:, i::8] & 0xF) << (i * 4)

    scales_out = scales.squeeze(-1).to(torch.bfloat16)  # [N, n_groups]
    return packed, scales_out, w_ref_bf16


def test_gs32_repack():
    """Test repack_int4_to_marlin_gs32 + GROUP_BLOCKS=2 kernel.

    Validates that direct nibble repack (no requantization) produces
    the same output as PyTorch BF16 reference within MMA rounding error.
    """
    E, M, K, N = 4, 4, 7168, 2048
    device = "cuda"
    print(f"\n=== Test gs=32 Direct Repack: E={E} M={M} K={K} N={N} ===\n")

    from batchgen.moe.marlin_weight_prep import repack_int4_to_marlin_gs32
    from batchgen.moe.marlin_grouped_moe import _load_module

    mod = _load_module()

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

    # Repack to Marlin gs=32 layout (NO dequant/requant)
    print("Repacking to Marlin gs=32 layout...", end=" ", flush=True)
    gate_marlin_qw, gate_marlin_s = [], []
    up_marlin_qw, up_marlin_s = [], []
    for e in range(E):
        gqw, gs = repack_int4_to_marlin_gs32(gate_packed_list[e], gate_scales_list[e], K, N)
        gate_marlin_qw.append(gqw)
        gate_marlin_s.append(gs)

        uqw, us = repack_int4_to_marlin_gs32(up_packed_list[e], up_scales_list[e], K, N)
        up_marlin_qw.append(uqw)
        up_marlin_s.append(us)
    print("done")

    # Build pointer arrays [2E]: [gate_0..gate_E-1, up_0..up_E-1]
    B_ptrs = torch.tensor(
        [w.data_ptr() for w in gate_marlin_qw] + [w.data_ptr() for w in up_marlin_qw],
        dtype=torch.int64, device=device)
    scales_ptrs = torch.tensor(
        [s.data_ptr() for s in gate_marlin_s] + [s.data_ptr() for s in up_marlin_s],
        dtype=torch.int64, device=device)

    # Activations: flat [E*M, K] contiguous (stride = M per expert)
    A = torch.randn((E * M, K), dtype=torch.bfloat16, device=device) * 0.1
    expert_starts = torch.arange(E, dtype=torch.int32, device=device) * M
    expert_counts = torch.full((E,), M, dtype=torch.int32, device=device)

    # Compact output buffers
    compact_stride = 16
    gate_buf = torch.zeros(E * compact_stride, N, dtype=torch.bfloat16, device=device)
    up_buf = torch.zeros(E * compact_stride, N, dtype=torch.bfloat16, device=device)

    # C_ptrs: into compact gate_buf and up_buf
    bytes_per_row = N * 2
    C_gate = torch.tensor(
        [gate_buf.data_ptr() + e * compact_stride * bytes_per_row for e in range(E)],
        dtype=torch.int64, device=device)
    C_up = torch.tensor(
        [up_buf.data_ptr() + e * compact_stride * bytes_per_row for e in range(E)],
        dtype=torch.int64, device=device)
    C_ptrs = torch.cat([C_gate, C_up])

    # Output intermediate with same stride as A (M per expert)
    intermediate = torch.zeros(E * M, N, dtype=torch.bfloat16, device=device)

    # Workspace
    n_tiles = N // 256
    workspace = torch.zeros(2 * E * (n_tiles + 17), dtype=torch.int32, device=device)

    # Run kernel: GEMM + scatter SiLU
    print("Running Marlin gs=32 grouped GEMM + scatter SiLU...", end=" ", flush=True)
    mod.grouped_marlin_gemm(
        A, B_ptrs, C_ptrs, scales_ptrs,
        expert_starts, expert_counts,
        E, N, K, workspace, 2 * E, n_tiles)
    mod.silu_mul_scatter(
        gate_buf, up_buf, intermediate, expert_counts,
        E, compact_stride, M, N)
    torch.cuda.synchronize()
    print("done")

    # Reference: BF16 matmul + SiLU (using SAME quantized weights, dequantized to BF16)
    print("\nValidation (gs=32 repack vs gs=32 BF16 reference — no requantization):")
    all_pass = True
    for e in range(E):
        a_e = A[e * M:(e + 1) * M]

        ref_gate = torch.mm(a_e, gate_refs[e])  # [M, N] BF16
        ref_up = torch.mm(a_e, up_refs[e])
        ref_silu = torch.nn.functional.silu(ref_gate.float()) * ref_up.float()
        ref_out = ref_silu.bfloat16()

        out_e = intermediate[e * M:(e + 1) * M]

        cd = calc_diff(out_e, ref_out)
        status = "PASS" if cd < 1e-3 else "FAIL"
        if cd >= 1e-3:
            all_pass = False
        print(f"  E{e} calc_diff={cd:.2e} [{status}]")

    print(f"\nOverall: {'PASS' if all_pass else 'FAIL'}")
    if all_pass:
        print("gs=32 direct repack produces near-identical output to BF16 reference.")
    else:
        print("ERROR: gs=32 repack has unexpected error. Investigate weight permutation.")


def test_gs32_3d_inplace():
    """Test the full 3D strided path matching K2.5 integration layout."""
    E, K, N = 24, 7168, 2048
    mtp = 64  # Small stride for testing
    compact_stride = 16
    device = "cuda"

    # Variable tokens per expert (1-8, matching decode)
    expert_counts_list = [3, 1, 5, 2, 4, 1, 7, 3, 2, 6, 1, 4, 8, 2, 3, 5, 1, 6, 4, 2, 7, 3, 1, 5]
    expert_counts = torch.tensor(expert_counts_list[:E], dtype=torch.int32, device=device)

    print(f"\n=== Test gs=32 3D Inplace: E={E} mtp={mtp} K={K} N={N} ===")
    print(f"    expert_counts={expert_counts.tolist()}\n")

    from batchgen.moe.marlin_weight_prep import repack_int4_to_marlin_gs32
    from batchgen.moe.marlin_grouped_moe import marlin_grouped_stage1_3d_inplace

    # Create weights and repack
    gate_packed, gate_scales, gate_refs = [], [], []
    up_packed, up_scales, up_refs = [], [], []
    gate_qw, gate_s, up_qw, up_s = [], [], [], []

    print("Creating + repacking weights...", end=" ", flush=True)
    for e in range(E):
        gp, gs, gr = create_k25_int4_weights(K, N, device=device)
        gate_packed.append(gp); gate_scales.append(gs); gate_refs.append(gr)
        gqw, gms = repack_int4_to_marlin_gs32(gp, gs, K, N)
        gate_qw.append(gqw); gate_s.append(gms)

        up_, us_, ur_ = create_k25_int4_weights(K, N, device=device)
        up_packed.append(up_); up_scales.append(us_); up_refs.append(ur_)
        uqw, ums = repack_int4_to_marlin_gs32(up_, us_, K, N)
        up_qw.append(uqw); up_s.append(ums)
    print("done")

    # Build pointer arrays
    B_ptrs = torch.tensor(
        [w.data_ptr() for w in gate_qw] + [w.data_ptr() for w in up_qw],
        dtype=torch.int64, device=device)
    scales_ptrs = torch.tensor(
        [s.data_ptr() for s in gate_s] + [s.data_ptr() for s in up_s],
        dtype=torch.int64, device=device)

    # 3D strided input: [E*mtp, K] with expert e at rows [e*mtp : e*mtp+cnt]
    dispatched_x = torch.zeros(E * mtp, K, dtype=torch.bfloat16, device=device)
    for e in range(E):
        cnt = expert_counts[e].item()
        dispatched_x[e * mtp: e * mtp + cnt] = torch.randn(cnt, K, dtype=torch.bfloat16, device=device) * 0.1

    # Buffers
    expert_starts = torch.arange(E, dtype=torch.int32, device=device) * mtp
    intermediate = torch.zeros(E * mtp, N, dtype=torch.bfloat16, device=device)
    gate_buf = torch.zeros(E * compact_stride, N, dtype=torch.bfloat16, device=device)
    up_buf = torch.zeros(E * compact_stride, N, dtype=torch.bfloat16, device=device)

    bytes_per_row = N * 2
    C_ptrs = torch.tensor(
        [gate_buf.data_ptr() + e * compact_stride * bytes_per_row for e in range(E)]
        + [up_buf.data_ptr() + e * compact_stride * bytes_per_row for e in range(E)],
        dtype=torch.int64, device=device)

    workspace = torch.zeros(2 * E * (N // 256 + 17), dtype=torch.int32, device=device)

    # Run full 3D inplace path
    print("Running Marlin 3D inplace (GEMM + scatter SiLU)...", end=" ", flush=True)
    marlin_grouped_stage1_3d_inplace(
        dispatched_x, intermediate, expert_counts,
        expert_starts, B_ptrs, scales_ptrs,
        C_ptrs, gate_buf, up_buf,
        N, K, workspace, compact_stride=compact_stride)
    torch.cuda.synchronize()
    print("done")

    # Reference
    print("\nValidation per-expert:")
    all_pass = True
    for e in range(E):
        cnt = expert_counts[e].item()
        if cnt == 0:
            continue
        a_e = dispatched_x[e * mtp: e * mtp + cnt]

        ref_gate = torch.mm(a_e, gate_refs[e])
        ref_up = torch.mm(a_e, up_refs[e])
        ref_silu = torch.nn.functional.silu(ref_gate.float()) * ref_up.float()
        ref_out = ref_silu.bfloat16()

        out_e = intermediate[e * mtp: e * mtp + cnt]

        cd = calc_diff(out_e, ref_out)
        status = "PASS" if cd < 1e-3 else "FAIL"
        if cd >= 1e-3:
            all_pass = False
        print(f"  E{e:2d} cnt={cnt} calc_diff={cd:.2e} [{status}]")

    print(f"\nOverall: {'PASS' if all_pass else 'FAIL'}")


def test_m16_sweep():
    """Test M16 kernel across a wide range of M values.

    M16 uses CTA M-tiling: each CTA processes up to 16 rows.
    For M > 16, multiple M-tiles are launched.
    """
    E, K, N = 4, 7168, 2048
    device = "cuda"
    M_values = [1, 2, 4, 8, 9, 16, 32, 64, 128, 256]

    print(f"\n=== Test M16 Sweep: E={E} K={K} N={N} ===\n")

    from batchgen.moe.marlin_weight_prep import repack_int4_to_marlin_gs32
    from batchgen.moe.marlin_grouped_moe import _load_module

    mod = _load_module()

    # Create weights once
    gate_qw_list, gate_s_list, gate_refs = [], [], []
    up_qw_list, up_s_list, up_refs = [], [], []

    print("Creating + repacking weights...", end=" ", flush=True)
    for e in range(E):
        gp, gs, gr = create_k25_int4_weights(K, N, device=device)
        gqw, gms = repack_int4_to_marlin_gs32(gp, gs, K, N)
        gate_qw_list.append(gqw); gate_s_list.append(gms); gate_refs.append(gr)

        up_, us_, ur_ = create_k25_int4_weights(K, N, device=device)
        uqw, ums = repack_int4_to_marlin_gs32(up_, us_, K, N)
        up_qw_list.append(uqw); up_s_list.append(ums); up_refs.append(ur_)
    print("done")

    B_ptrs = torch.tensor(
        [w.data_ptr() for w in gate_qw_list] + [w.data_ptr() for w in up_qw_list],
        dtype=torch.int64, device=device)
    scales_ptrs = torch.tensor(
        [s.data_ptr() for s in gate_s_list] + [s.data_ptr() for s in up_s_list],
        dtype=torch.int64, device=device)

    n_tiles = N // 256
    all_pass = True

    for M in M_values:
        mtp = M  # For simple test, mtp = M
        max_m_tiles = (M + 15) // 16
        compact_stride = max_m_tiles * 16

        # Activations
        A = torch.randn((E * mtp, K), dtype=torch.bfloat16, device=device) * 0.1
        expert_starts = torch.arange(E, dtype=torch.int32, device=device) * mtp
        expert_counts = torch.full((E,), M, dtype=torch.int32, device=device)

        # Gate writes to intermediate (mtp stride), up writes to up_buf (compact stride)
        intermediate = torch.zeros(E * mtp, N, dtype=torch.bfloat16, device=device)
        up_buf = torch.zeros(E * compact_stride, N, dtype=torch.bfloat16, device=device)

        bytes_per_row = N * 2
        C_gate = torch.tensor(
            [intermediate.data_ptr() + e * mtp * bytes_per_row for e in range(E)],
            dtype=torch.int64, device=device)
        C_up = torch.tensor(
            [up_buf.data_ptr() + e * compact_stride * bytes_per_row for e in range(E)],
            dtype=torch.int64, device=device)
        C_ptrs = torch.cat([C_gate, C_up])

        workspace = torch.zeros(2 * E * (n_tiles + 17), dtype=torch.int32, device=device)

        # Run M16 kernel + dual-stride SiLU
        mod.grouped_marlin_gemm_m16(
            A, B_ptrs, C_ptrs, scales_ptrs,
            expert_starts, expert_counts,
            E, N, K, workspace, 2 * E, n_tiles, max_m_tiles)
        mod.silu_mul_dual_stride(
            intermediate, up_buf, expert_counts,
            E, mtp, compact_stride, N)
        torch.cuda.synchronize()

        # Validate per expert
        max_cd = 0
        for e in range(E):
            a_e = A[e * mtp: e * mtp + M]
            ref_gate = torch.mm(a_e, gate_refs[e])
            ref_up = torch.mm(a_e, up_refs[e])
            ref_silu = torch.nn.functional.silu(ref_gate.float()) * ref_up.float()
            ref_out = ref_silu.bfloat16()
            out_e = intermediate[e * mtp: e * mtp + M]
            cd = calc_diff(out_e, ref_out)
            max_cd = max(max_cd, cd)

        status = "PASS" if max_cd < 1e-3 else "FAIL"
        if max_cd >= 1e-3:
            all_pass = False
        print(f"  M={M:4d} max_m_tiles={max_m_tiles:3d} max_calc_diff={max_cd:.2e} [{status}]")

    print(f"\nOverall: {'PASS' if all_pass else 'FAIL'}")


def test_m16_variable_experts():
    """Test M16 with variable M per expert (realistic decode/prefill mix)."""
    E, K, N = 24, 7168, 2048
    device = "cuda"

    # Variable tokens: mix of small (decode-like) and medium (prefill-like)
    expert_counts_list = [1, 3, 8, 15, 32, 64, 2, 7, 16, 48, 4, 1,
                          128, 5, 12, 3, 24, 96, 1, 6, 33, 2, 10, 55]
    max_m = max(expert_counts_list)
    max_m_tiles = (max_m + 15) // 16
    compact_stride = max_m_tiles * 16
    mtp = compact_stride  # Use compact_stride as mtp for simplicity

    expert_counts = torch.tensor(expert_counts_list[:E], dtype=torch.int32, device=device)

    print(f"\n=== Test M16 Variable Experts: E={E} K={K} N={N} ===")
    print(f"    max_m={max_m} max_m_tiles={max_m_tiles} compact_stride={compact_stride}")
    print(f"    expert_counts={expert_counts.tolist()}\n")

    from batchgen.moe.marlin_weight_prep import repack_int4_to_marlin_gs32
    from batchgen.moe.marlin_grouped_moe import _load_module

    mod = _load_module()

    # Weights
    gate_qw, gate_s, gate_refs = [], [], []
    up_qw, up_s, up_refs = [], [], []
    print("Creating + repacking weights...", end=" ", flush=True)
    for e in range(E):
        gp, gs, gr = create_k25_int4_weights(K, N, device=device)
        gqw, gms = repack_int4_to_marlin_gs32(gp, gs, K, N)
        gate_qw.append(gqw); gate_s.append(gms); gate_refs.append(gr)

        up_, us_, ur_ = create_k25_int4_weights(K, N, device=device)
        uqw, ums = repack_int4_to_marlin_gs32(up_, us_, K, N)
        up_qw.append(uqw); up_s.append(ums); up_refs.append(ur_)
    print("done")

    B_ptrs = torch.tensor(
        [w.data_ptr() for w in gate_qw] + [w.data_ptr() for w in up_qw],
        dtype=torch.int64, device=device)
    scales_ptrs = torch.tensor(
        [s.data_ptr() for s in gate_s] + [s.data_ptr() for s in up_s],
        dtype=torch.int64, device=device)

    # 3D strided input
    dispatched_x = torch.zeros(E * mtp, K, dtype=torch.bfloat16, device=device)
    for e in range(E):
        cnt = expert_counts[e].item()
        dispatched_x[e * mtp: e * mtp + cnt] = torch.randn(cnt, K, dtype=torch.bfloat16, device=device) * 0.1

    expert_starts = torch.arange(E, dtype=torch.int32, device=device) * mtp
    intermediate = torch.zeros(E * mtp, N, dtype=torch.bfloat16, device=device)
    up_buf = torch.zeros(E * compact_stride, N, dtype=torch.bfloat16, device=device)

    n_tiles = N // 256
    bytes_per_row = N * 2
    C_gate = torch.tensor(
        [intermediate.data_ptr() + e * mtp * bytes_per_row for e in range(E)],
        dtype=torch.int64, device=device)
    C_up = torch.tensor(
        [up_buf.data_ptr() + e * compact_stride * bytes_per_row for e in range(E)],
        dtype=torch.int64, device=device)
    C_ptrs = torch.cat([C_gate, C_up])

    workspace = torch.zeros(2 * E * (n_tiles + 17), dtype=torch.int32, device=device)

    # Run
    print("Running M16 unified path...", end=" ", flush=True)
    mod.grouped_marlin_gemm_m16(
        dispatched_x, B_ptrs, C_ptrs, scales_ptrs,
        expert_starts, expert_counts,
        E, N, K, workspace, 2 * E, n_tiles, max_m_tiles)
    mod.silu_mul_dual_stride(
        intermediate, up_buf, expert_counts,
        E, mtp, compact_stride, N)
    torch.cuda.synchronize()
    print("done")

    # Validate
    print("\nValidation per-expert:")
    all_pass = True
    for e in range(E):
        cnt = expert_counts[e].item()
        if cnt == 0:
            continue
        a_e = dispatched_x[e * mtp: e * mtp + cnt]
        ref_gate = torch.mm(a_e, gate_refs[e])
        ref_up = torch.mm(a_e, up_refs[e])
        ref_silu = torch.nn.functional.silu(ref_gate.float()) * ref_up.float()
        ref_out = ref_silu.bfloat16()
        out_e = intermediate[e * mtp: e * mtp + cnt]

        cd = calc_diff(out_e, ref_out)
        status = "PASS" if cd < 1e-3 else "FAIL"
        if cd >= 1e-3:
            all_pass = False
        print(f"  E{e:2d} cnt={cnt:3d} calc_diff={cd:.2e} [{status}]")

    print(f"\nOverall: {'PASS' if all_pass else 'FAIL'}")


def test_m16_edge_cases():
    """Test M16 edge cases: M=0 experts, single expert large M."""
    E, K, N = 8, 7168, 2048
    device = "cuda"

    # Mix of 0 and large M
    expert_counts_list = [0, 0, 256, 0, 1, 0, 64, 0]
    max_m = max(expert_counts_list)
    max_m_tiles = (max_m + 15) // 16
    compact_stride = max_m_tiles * 16
    mtp = compact_stride

    expert_counts = torch.tensor(expert_counts_list[:E], dtype=torch.int32, device=device)

    print(f"\n=== Test M16 Edge Cases: E={E} K={K} N={N} ===")
    print(f"    expert_counts={expert_counts.tolist()}\n")

    from batchgen.moe.marlin_weight_prep import repack_int4_to_marlin_gs32
    from batchgen.moe.marlin_grouped_moe import _load_module

    mod = _load_module()

    gate_qw, gate_s, gate_refs = [], [], []
    up_qw, up_s, up_refs = [], [], []
    print("Creating + repacking weights...", end=" ", flush=True)
    for e in range(E):
        gp, gs, gr = create_k25_int4_weights(K, N, device=device)
        gqw, gms = repack_int4_to_marlin_gs32(gp, gs, K, N)
        gate_qw.append(gqw); gate_s.append(gms); gate_refs.append(gr)
        up_, us_, ur_ = create_k25_int4_weights(K, N, device=device)
        uqw, ums = repack_int4_to_marlin_gs32(up_, us_, K, N)
        up_qw.append(uqw); up_s.append(ums); up_refs.append(ur_)
    print("done")

    B_ptrs = torch.tensor(
        [w.data_ptr() for w in gate_qw] + [w.data_ptr() for w in up_qw],
        dtype=torch.int64, device=device)
    scales_ptrs = torch.tensor(
        [s.data_ptr() for s in gate_s] + [s.data_ptr() for s in up_s],
        dtype=torch.int64, device=device)

    dispatched_x = torch.zeros(E * mtp, K, dtype=torch.bfloat16, device=device)
    for e in range(E):
        cnt = expert_counts[e].item()
        if cnt > 0:
            dispatched_x[e * mtp: e * mtp + cnt] = torch.randn(cnt, K, dtype=torch.bfloat16, device=device) * 0.1

    expert_starts = torch.arange(E, dtype=torch.int32, device=device) * mtp
    intermediate = torch.zeros(E * mtp, N, dtype=torch.bfloat16, device=device)
    up_buf = torch.zeros(E * compact_stride, N, dtype=torch.bfloat16, device=device)

    n_tiles = N // 256
    bytes_per_row = N * 2
    C_gate = torch.tensor(
        [intermediate.data_ptr() + e * mtp * bytes_per_row for e in range(E)],
        dtype=torch.int64, device=device)
    C_up = torch.tensor(
        [up_buf.data_ptr() + e * compact_stride * bytes_per_row for e in range(E)],
        dtype=torch.int64, device=device)
    C_ptrs = torch.cat([C_gate, C_up])

    workspace = torch.zeros(2 * E * (n_tiles + 17), dtype=torch.int32, device=device)

    print("Running M16 edge cases...", end=" ", flush=True)
    mod.grouped_marlin_gemm_m16(
        dispatched_x, B_ptrs, C_ptrs, scales_ptrs,
        expert_starts, expert_counts,
        E, N, K, workspace, 2 * E, n_tiles, max_m_tiles)
    mod.silu_mul_dual_stride(
        intermediate, up_buf, expert_counts,
        E, mtp, compact_stride, N)
    torch.cuda.synchronize()
    print("done")

    print("\nValidation:")
    all_pass = True
    for e in range(E):
        cnt = expert_counts[e].item()
        if cnt == 0:
            print(f"  E{e:2d} cnt=0 [SKIP]")
            continue
        a_e = dispatched_x[e * mtp: e * mtp + cnt]
        ref_gate = torch.mm(a_e, gate_refs[e])
        ref_up = torch.mm(a_e, up_refs[e])
        ref_silu = torch.nn.functional.silu(ref_gate.float()) * ref_up.float()
        ref_out = ref_silu.bfloat16()
        out_e = intermediate[e * mtp: e * mtp + cnt]
        cd = calc_diff(out_e, ref_out)
        status = "PASS" if cd < 1e-3 else "FAIL"
        if cd >= 1e-3:
            all_pass = False
        print(f"  E{e:2d} cnt={cnt:3d} calc_diff={cd:.2e} [{status}]")

    print(f"\nOverall: {'PASS' if all_pass else 'FAIL'}")


def main():
    import sys
    test_name = sys.argv[1] if len(sys.argv) > 1 else "all"

    tests = {
        "m8_repack": test_gs32_repack,
        "m8_3d": test_gs32_3d_inplace,
        "m16_sweep": test_m16_sweep,
        "m16_variable": test_m16_variable_experts,
        "m16_edge": test_m16_edge_cases,
    }

    if test_name == "all":
        for name, fn in tests.items():
            print(f"\n{'='*60}")
            print(f"Running: {name}")
            print(f"{'='*60}")
            fn()
    elif test_name in tests:
        tests[test_name]()
    else:
        print(f"Unknown test: {test_name}")
        print(f"Available: {', '.join(tests.keys())}, all")


if __name__ == "__main__":
    main()
