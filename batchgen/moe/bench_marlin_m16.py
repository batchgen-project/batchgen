#!/usr/bin/env python3
"""Benchmark: M16 Marlin kernel timing across M sweep.

Uses CUDA events for accurate timing. Compares M8 vs M16 vs WGMMA.
Run on H20 with CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=9.0a.

Usage:
    CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=9.0a \
        python -m batchgen.moe.bench_marlin_m16
"""

import torch
import sys


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
    K_div8 = K // 8
    packed = torch.zeros(N, K_div8, dtype=torch.int32, device=device)
    for i in range(8):
        packed |= (q_flat[:, i::8] & 0xF) << (i * 4)
    scales_out = scales.squeeze(-1).to(torch.bfloat16)
    return packed, scales_out


def bench_m16_gemm_only(mod, A, B_ptrs, C_ptrs, scales_ptrs,
                         expert_starts, expert_counts, E, N, K,
                         workspace, n_tiles, max_m_tiles, iters=100):
    """Benchmark M16 GEMM kernel only (no SiLU)."""
    # Warmup
    for _ in range(10):
        mod.grouped_marlin_gemm_m16(
            A, B_ptrs, C_ptrs, scales_ptrs,
            expert_starts, expert_counts,
            E, N, K, workspace, 2 * E, n_tiles, max_m_tiles)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        mod.grouped_marlin_gemm_m16(
            A, B_ptrs, C_ptrs, scales_ptrs,
            expert_starts, expert_counts,
            E, N, K, workspace, 2 * E, n_tiles, max_m_tiles)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1000  # us


def bench_m8_gemm_only(mod, A, B_ptrs, C_ptrs, scales_ptrs,
                        expert_starts, expert_counts, E, N, K,
                        workspace, n_tiles, iters=100):
    """Benchmark M8 GEMM kernel only (no SiLU)."""
    for _ in range(10):
        mod.grouped_marlin_gemm(
            A, B_ptrs, C_ptrs, scales_ptrs,
            expert_starts, expert_counts,
            E, N, K, workspace, 2 * E, n_tiles)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        mod.grouped_marlin_gemm(
            A, B_ptrs, C_ptrs, scales_ptrs,
            expert_starts, expert_counts,
            E, N, K, workspace, 2 * E, n_tiles)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1000  # us


def main():
    E, K, N = 24, 7168, 2048  # K2.5 production shapes
    device = "cuda"
    M_values = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    iters = 100

    print(f"Marlin M16 Benchmark: E={E} K={K} N={N}")
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"Iterations: {iters}")
    print()

    from batchgen.moe.marlin_weight_prep import repack_int4_to_marlin_gs32
    from batchgen.moe.marlin_grouped_moe import _load_module

    mod = _load_module()
    n_tiles = N // 256

    # Create weights
    print("Creating + repacking weights for 24 experts...", end=" ", flush=True)
    gate_qw, gate_s, up_qw, up_s = [], [], [], []
    for e in range(E):
        gp, gs = create_k25_int4_weights(K, N, device=device)
        gqw, gms = repack_int4_to_marlin_gs32(gp, gs, K, N)
        gate_qw.append(gqw); gate_s.append(gms)
        up_, us_ = create_k25_int4_weights(K, N, device=device)
        uqw, ums = repack_int4_to_marlin_gs32(up_, us_, K, N)
        up_qw.append(uqw); up_s.append(ums)
    print("done")

    B_ptrs = torch.tensor(
        [w.data_ptr() for w in gate_qw] + [w.data_ptr() for w in up_qw],
        dtype=torch.int64, device=device)
    scales_ptrs = torch.tensor(
        [s.data_ptr() for s in gate_s] + [s.data_ptr() for s in up_s],
        dtype=torch.int64, device=device)

    workspace = torch.zeros(2 * E * (n_tiles + 17), dtype=torch.int32, device=device)

    # Weight data size per expert (for BW calculation)
    # INT4: K*N/2 bytes for weights + K/32*N*2 bytes for scales
    weight_bytes_per_expert = K * N // 2 + (K // 32) * N * 2
    total_weight_bytes = weight_bytes_per_expert * E * 2  # gate + up

    print(f"\n{'M':>5} | {'M8 (us)':>10} | {'M16 (us)':>10} | {'M16 CTAs':>10} | {'M16 BW(TB/s)':>12}")
    print("-" * 65)

    for M in M_values:
        mtp = M  # simple: mtp = M for benchmark
        max_m_tiles = (M + 15) // 16

        A = torch.randn((E * mtp, K), dtype=torch.bfloat16, device=device) * 0.1
        expert_starts = torch.arange(E, dtype=torch.int32, device=device) * mtp
        expert_counts = torch.full((E,), M, dtype=torch.int32, device=device)

        # M16 output buffers (gate → intermediate, up → up_buf)
        intermediate = torch.zeros(E * mtp, N, dtype=torch.bfloat16, device=device)
        compact_stride = max_m_tiles * 16
        up_buf = torch.zeros(E * compact_stride, N, dtype=torch.bfloat16, device=device)

        bytes_per_row = N * 2
        C_gate_m16 = torch.tensor(
            [intermediate.data_ptr() + e * mtp * bytes_per_row for e in range(E)],
            dtype=torch.int64, device=device)
        C_up_m16 = torch.tensor(
            [up_buf.data_ptr() + e * compact_stride * bytes_per_row for e in range(E)],
            dtype=torch.int64, device=device)
        C_ptrs_m16 = torch.cat([C_gate_m16, C_up_m16])

        # Bench M16
        m16_us = bench_m16_gemm_only(
            mod, A, B_ptrs, C_ptrs_m16, scales_ptrs,
            expert_starts, expert_counts, E, N, K,
            workspace, n_tiles, max_m_tiles, iters=iters)

        # Activation read bytes + output write bytes
        act_bytes = E * M * K * 2  # BF16
        out_bytes = E * M * N * 2 * 2  # gate + up, BF16
        total_bytes = total_weight_bytes + act_bytes + out_bytes
        bw_tb = total_bytes / (m16_us * 1e-6) / 1e12

        total_ctas_m16 = n_tiles * max_m_tiles * 2 * E

        # Bench M8 (only for M <= 8)
        m8_str = "N/A"
        if M <= 8:
            compact_stride_m8 = 16
            gate_buf_m8 = torch.zeros(E * compact_stride_m8, N, dtype=torch.bfloat16, device=device)
            up_buf_m8 = torch.zeros(E * compact_stride_m8, N, dtype=torch.bfloat16, device=device)
            C_gate_m8 = torch.tensor(
                [gate_buf_m8.data_ptr() + e * compact_stride_m8 * bytes_per_row for e in range(E)],
                dtype=torch.int64, device=device)
            C_up_m8 = torch.tensor(
                [up_buf_m8.data_ptr() + e * compact_stride_m8 * bytes_per_row for e in range(E)],
                dtype=torch.int64, device=device)
            C_ptrs_m8 = torch.cat([C_gate_m8, C_up_m8])

            m8_us = bench_m8_gemm_only(
                mod, A, B_ptrs, C_ptrs_m8, scales_ptrs,
                expert_starts, expert_counts, E, N, K,
                workspace, n_tiles, iters=iters)
            m8_str = f"{m8_us:10.1f}"

        print(f"{M:5d} | {m8_str:>10} | {m16_us:10.1f} | {total_ctas_m16:10d} | {bw_tb:12.2f}")

    print()
    print("Notes:")
    print("  - WGMMA S1 reference: ~815 us (from v12c session)")
    print("  - M8 only valid for M <= 8")
    print("  - BW includes weight read + activation read + output write")


if __name__ == "__main__":
    main()
