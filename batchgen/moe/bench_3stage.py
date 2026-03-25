#!/usr/bin/env python3
"""Benchmark: 3-stage Marlin MoE pipeline (S1 GEMM + S2 SiLU + S3 GEMM).

CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=9.0a python -m batchgen.moe.bench_3stage
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
    E, K, N = 24, 7168, 2048  # K2.5 production shapes
    device = "cuda"
    M_values = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    iters = 100

    print(f"3-Stage Marlin MoE Benchmark: E={E} K={K} N={N}")
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"Iterations: {iters}")
    print()

    from batchgen.moe.marlin_weight_prep import repack_int4_to_marlin_gs32
    from batchgen.moe.marlin_grouped_moe import _load_module
    mod = _load_module()

    n_tiles_s1 = N // 256   # 8
    n_tiles_s3 = K // 256   # 28

    # Create weights for all 3 projections
    print("Creating + repacking weights (gate+up+down)...", end=" ", flush=True)
    gate_qw, gate_s, up_qw, up_s, down_qw, down_s = [], [], [], [], [], []
    for e in range(E):
        gp, gs = create_k25_int4_weights(K, N, device=device)
        gqw, gms = repack_int4_to_marlin_gs32(gp, gs, K, N)
        gate_qw.append(gqw); gate_s.append(gms)

        up_, us_ = create_k25_int4_weights(K, N, device=device)
        uqw, ums = repack_int4_to_marlin_gs32(up_, us_, K, N)
        up_qw.append(uqw); up_s.append(ums)

        dp, ds = create_k25_int4_weights(N, K, device=device)  # down: input=N, output=K
        dqw, dms = repack_int4_to_marlin_gs32(dp, ds, N, K)
        down_qw.append(dqw); down_s.append(dms)
    print("done")

    # S1 pointer arrays [2E]: gate + up
    s1_B_ptrs = torch.tensor(
        [w.data_ptr() for w in gate_qw] + [w.data_ptr() for w in up_qw],
        dtype=torch.int64, device=device)
    s1_scales_ptrs = torch.tensor(
        [s.data_ptr() for s in gate_s] + [s.data_ptr() for s in up_s],
        dtype=torch.int64, device=device)

    # S3 pointer arrays [E]: down
    s3_B_ptrs = torch.tensor([w.data_ptr() for w in down_qw], dtype=torch.int64, device=device)
    s3_scales_ptrs = torch.tensor([s.data_ptr() for s in down_s], dtype=torch.int64, device=device)

    # Weight sizes for BW calculation
    # S1: 2 * E * (K*N/2 + K/32*N*2) bytes (gate + up)
    s1_weight_bytes = 2 * E * (K * N // 2 + (K // 32) * N * 2)
    # S3: E * (N*K/2 + N/32*K*2) bytes (down)
    s3_weight_bytes = E * (N * K // 2 + (N // 32) * K * 2)

    print(f"{'M':>5} | {'S1 gate+up':>11} | {'S2 SiLU':>9} | {'S3 down':>9} | {'Total':>9} | {'WGMMA ref':>10} | {'Speedup':>8}")
    print(f"{'':>5} | {'(us)':>11} | {'(us)':>9} | {'(us)':>9} | {'(us)':>9} | {'(us)':>10} | {'':>8}")
    print("-" * 80)

    for M in M_values:
        mtp = M
        max_m_tiles = (M + 15) // 16

        A = torch.randn((E * mtp, K), dtype=torch.bfloat16, device=device) * 0.1
        expert_starts = torch.arange(E, dtype=torch.int32, device=device) * mtp
        expert_counts = torch.full((E,), M, dtype=torch.int32, device=device)

        gate_buf = torch.zeros(E * mtp, N, dtype=torch.bfloat16, device=device)
        up_buf = torch.zeros(E * mtp, N, dtype=torch.bfloat16, device=device)
        intermediate = torch.zeros(E * mtp, N, dtype=torch.bfloat16, device=device)
        expert_out = torch.zeros(E * mtp, K, dtype=torch.bfloat16, device=device)

        bytes_per_row_N = N * 2
        bytes_per_row_K = K * 2

        s1_C_ptrs = torch.tensor(
            [gate_buf.data_ptr() + e * mtp * bytes_per_row_N for e in range(E)]
            + [up_buf.data_ptr() + e * mtp * bytes_per_row_N for e in range(E)],
            dtype=torch.int64, device=device)
        s3_C_ptrs = torch.tensor(
            [expert_out.data_ptr() + e * mtp * bytes_per_row_K for e in range(E)],
            dtype=torch.int64, device=device)

        s1_workspace = torch.zeros(2 * E * (n_tiles_s1 + 17), dtype=torch.int32, device=device)
        s3_workspace = torch.zeros(E * (n_tiles_s3 + 17), dtype=torch.int32, device=device)

        def run_s1():
            mod.grouped_marlin_gemm_m16(
                A, s1_B_ptrs, s1_C_ptrs, s1_scales_ptrs,
                expert_starts, expert_counts,
                E, N, K, s1_workspace, 2 * E, n_tiles_s1, max_m_tiles)

        def run_s2():
            mod.silu_mul(gate_buf, up_buf, intermediate)

        def run_s3():
            mod.grouped_marlin_gemm_m16(
                intermediate, s3_B_ptrs, s3_C_ptrs, s3_scales_ptrs,
                expert_starts, expert_counts,
                E, K, N, s3_workspace, E, n_tiles_s3, max_m_tiles)

        def run_all():
            run_s1()
            run_s2()
            run_s3()

        # Warmup
        for _ in range(10):
            run_all()

        # Bench S1
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            run_s1()
        e.record()
        torch.cuda.synchronize()
        s1_us = s.elapsed_time(e) / iters * 1000

        # Bench S2
        s.record()
        for _ in range(iters):
            run_s2()
        e.record()
        torch.cuda.synchronize()
        s2_us = s.elapsed_time(e) / iters * 1000

        # Bench S3
        s.record()
        for _ in range(iters):
            run_s3()
        e.record()
        torch.cuda.synchronize()
        s3_us = s.elapsed_time(e) / iters * 1000

        # Bench total
        s.record()
        for _ in range(iters):
            run_all()
        e.record()
        torch.cuda.synchronize()
        total_us = s.elapsed_time(e) / iters * 1000

        # WGMMA reference: S1=815us, S2 (down) ~similar, total ~1630us (from previous sessions)
        wgmma_total = 815 + 815  # approximate S1+S2 WGMMA reference
        speedup = wgmma_total / total_us if total_us > 0 else 0

        print(f"{M:5d} | {s1_us:11.1f} | {s2_us:9.1f} | {s3_us:9.1f} | {total_us:9.1f} | {wgmma_total:10d} | {speedup:7.2f}×")

    print()
    print("Notes:")
    print("  S1: gate+up GEMM (2E matrices, prob_k=7168, prob_n=2048, n_tiles=8)")
    print("  S2: SiLU(gate) * up element-wise")
    print("  S3: down GEMM (E matrices, prob_k=2048, prob_n=7168, n_tiles=28)")
    print("  WGMMA ref: ~815us S1 + ~815us S2 = ~1630us total (from previous sessions)")
    print(f"  Commit: see git log")


if __name__ == "__main__":
    main()
