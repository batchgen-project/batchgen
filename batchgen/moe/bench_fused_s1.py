#!/usr/bin/env python3
"""Benchmark: Fused S1 (gate+up+SiLU) vs non-fused (2E GEMM + SiLU) vs WGMMA ref.

CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=9.0a python -m batchgen.moe.bench_fused_s1
"""

import torch


def create_k25_int4_weights(K, N, group_size=32, device="cuda"):
    w_fp16 = torch.randn((N, K), dtype=torch.float16, device=device) * 0.1
    n_groups = K // group_size
    w_grouped = w_fp16.view(N, n_groups, group_size)
    max_val = w_grouped.max(dim=-1, keepdim=True).values
    min_val = w_grouped.min(dim=-1, keepdim=True).values
    scales = torch.max(max_val.abs() / 7.0, min_val.abs() / 8.0).clamp(min=1e-10)
    q = torch.round(w_grouped / scales).int() + 8
    q = torch.clamp(q, 0, 15)
    q_flat = q.view(N, K).int()
    packed = torch.zeros(N, K // 8, dtype=torch.int32, device=device)
    for i in range(8):
        packed |= (q_flat[:, i::8] & 0xF) << (i * 4)
    return packed, scales.squeeze(-1).to(torch.bfloat16)


def main():
    E, K, N = 24, 7168, 2048
    device = "cuda"
    M_values = [1, 4, 8, 16, 32, 64, 128]
    iters = 100

    print(f"Fused S1 Benchmark: E={E} K={K} N={N}")
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"Iterations: {iters}\n")

    from batchgen.moe.marlin_weight_prep import repack_int4_to_marlin_gs32
    from batchgen.moe.marlin_grouped_moe import _load_module
    mod = _load_module()

    n_tiles = N // 256

    # Create weights
    print("Creating + repacking weights...", end=" ", flush=True)
    gate_qw, gate_s, up_qw, up_s = [], [], [], []
    for e in range(E):
        gp, gs = create_k25_int4_weights(K, N, device=device)
        gqw, gms = repack_int4_to_marlin_gs32(gp, gs, K, N)
        gate_qw.append(gqw); gate_s.append(gms)
        up_, us_ = create_k25_int4_weights(K, N, device=device)
        uqw, ums = repack_int4_to_marlin_gs32(up_, us_, K, N)
        up_qw.append(uqw); up_s.append(ums)
    print("done")

    # Separate [E] arrays for fused kernel
    gate_B_ptrs = torch.tensor([w.data_ptr() for w in gate_qw], dtype=torch.int64, device=device)
    up_B_ptrs = torch.tensor([w.data_ptr() for w in up_qw], dtype=torch.int64, device=device)
    gate_scales_ptrs = torch.tensor([s.data_ptr() for s in gate_s], dtype=torch.int64, device=device)
    up_scales_ptrs = torch.tensor([s.data_ptr() for s in up_s], dtype=torch.int64, device=device)

    # Interleaved [2E] arrays for non-fused kernel
    B_ptrs_2E = torch.tensor(
        [w.data_ptr() for w in gate_qw] + [w.data_ptr() for w in up_qw],
        dtype=torch.int64, device=device)
    scales_ptrs_2E = torch.tensor(
        [s.data_ptr() for s in gate_s] + [s.data_ptr() for s in up_s],
        dtype=torch.int64, device=device)

    print(f"{'M':>5} | {'Fused S1 (us)':>14} | {'NonFused (us)':>14} | {'Fused CTAs':>10} | {'NonFused CTAs':>13}")
    print("-" * 70)

    for M in M_values:
        mtp = M
        max_m_tiles = (M + 15) // 16

        A = torch.randn((E * mtp, K), dtype=torch.bfloat16, device=device) * 0.1
        expert_starts = torch.arange(E, dtype=torch.int32, device=device) * mtp
        expert_counts = torch.full((E,), M, dtype=torch.int32, device=device)
        intermediate = torch.zeros(E * mtp, N, dtype=torch.bfloat16, device=device)

        bytes_per_row = N * 2

        # Fused: C_ptrs [E] into intermediate
        C_ptrs_fused = torch.tensor(
            [intermediate.data_ptr() + e * mtp * bytes_per_row for e in range(E)],
            dtype=torch.int64, device=device)
        workspace_fused = torch.zeros(E * (n_tiles + 17), dtype=torch.int32, device=device)

        # Non-fused: separate gate_buf + up_buf + C_ptrs [2E]
        compact_stride = max_m_tiles * 16
        up_buf = torch.zeros(E * compact_stride, N, dtype=torch.bfloat16, device=device)
        C_gate_nf = torch.tensor(
            [intermediate.data_ptr() + e * mtp * bytes_per_row for e in range(E)],
            dtype=torch.int64, device=device)
        C_up_nf = torch.tensor(
            [up_buf.data_ptr() + e * compact_stride * bytes_per_row for e in range(E)],
            dtype=torch.int64, device=device)
        C_ptrs_nf = torch.cat([C_gate_nf, C_up_nf])
        workspace_nf = torch.zeros(2 * E * (n_tiles + 17), dtype=torch.int32, device=device)

        fused_ctas = n_tiles * max_m_tiles * E
        nonfused_ctas = n_tiles * max_m_tiles * 2 * E

        # Bench fused S1
        for _ in range(10):
            mod.grouped_marlin_gemm_m16_s1(
                A, gate_B_ptrs, up_B_ptrs, C_ptrs_fused,
                gate_scales_ptrs, up_scales_ptrs,
                expert_starts, expert_counts,
                E, N, K, workspace_fused, n_tiles, max_m_tiles)
        s = torch.cuda.Event(enable_timing=True)
        e_ev = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            mod.grouped_marlin_gemm_m16_s1(
                A, gate_B_ptrs, up_B_ptrs, C_ptrs_fused,
                gate_scales_ptrs, up_scales_ptrs,
                expert_starts, expert_counts,
                E, N, K, workspace_fused, n_tiles, max_m_tiles)
        e_ev.record()
        torch.cuda.synchronize()
        fused_us = s.elapsed_time(e_ev) / iters * 1000

        # Bench non-fused (M16 GEMM + dual-stride SiLU)
        for _ in range(10):
            mod.grouped_marlin_gemm_m16(
                A, B_ptrs_2E, C_ptrs_nf, scales_ptrs_2E,
                expert_starts, expert_counts,
                E, N, K, workspace_nf, 2 * E, n_tiles, max_m_tiles)
            mod.silu_mul_dual_stride(intermediate, up_buf, expert_counts, E, mtp, compact_stride, N)
        s.record()
        for _ in range(iters):
            mod.grouped_marlin_gemm_m16(
                A, B_ptrs_2E, C_ptrs_nf, scales_ptrs_2E,
                expert_starts, expert_counts,
                E, N, K, workspace_nf, 2 * E, n_tiles, max_m_tiles)
            mod.silu_mul_dual_stride(intermediate, up_buf, expert_counts, E, mtp, compact_stride, N)
        e_ev.record()
        torch.cuda.synchronize()
        nonfused_us = s.elapsed_time(e_ev) / iters * 1000

        print(f"{M:5d} | {fused_us:14.1f} | {nonfused_us:14.1f} | {fused_ctas:10d} | {nonfused_ctas:13d}")

    print()
    print("Fused = single kernel (gate+up+SiLU). No temp buffer.")
    print("NonFused = M16 GEMM (2E matrices) + dual-stride SiLU kernel.")
    print("WGMMA S1 reference: ~815 us (from previous session)")


if __name__ == "__main__":
    main()
