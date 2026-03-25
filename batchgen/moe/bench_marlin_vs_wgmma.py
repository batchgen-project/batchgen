#!/usr/bin/env python3
"""Benchmark: 3-stage Marlin vs 2-stage WGMMA across M sweep.

Compares full MoE pipeline timing (S1+S2) for both kernel implementations.
WGMMA uses non-inplace API (allocates output, no TMA descriptor needed).

CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=9.0a python -m batchgen.moe.bench_marlin_vs_wgmma
"""

import torch


def create_k25_int4_weights_raw(K, N, group_size=32, device="cuda"):
    """Create K2.5 INT4 weights in RAW byte format (for WGMMA).
    Returns packed [N, K//2] uint8 + scales [N, K//32] BF16.
    """
    w_fp16 = torch.randn((N, K), dtype=torch.float16, device=device) * 0.1
    n_groups = K // group_size
    w_grouped = w_fp16.view(N, n_groups, group_size)
    max_val = w_grouped.max(dim=-1, keepdim=True).values
    min_val = w_grouped.min(dim=-1, keepdim=True).values
    scales = torch.max(max_val.abs() / 7.0, min_val.abs() / 8.0).clamp(min=1e-10)
    q = torch.round(w_grouped / scales).int() + 8
    q = torch.clamp(q, 0, 15)

    # Pack as uint8 (2 nibbles per byte) for WGMMA
    q_flat = q.view(N, K).int()
    K_div2 = K // 2
    packed_u8 = torch.zeros(N, K_div2, dtype=torch.uint8, device=device)
    for i in range(K_div2):
        packed_u8[:, i] = (q_flat[:, 2*i] & 0xF) | ((q_flat[:, 2*i+1] & 0xF) << 4)

    # Also pack as int32 for Marlin repack
    K_div8 = K // 8
    packed_i32 = torch.zeros(N, K_div8, dtype=torch.int32, device=device)
    for i in range(8):
        packed_i32 |= (q_flat[:, i::8] & 0xF) << (i * 4)

    scales_out = scales.squeeze(-1).to(torch.bfloat16)
    return packed_u8, packed_i32, scales_out


def main():
    E, K, N = 24, 7168, 2048
    device = "cuda"
    M_values = [1, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    iters = 100

    print(f"Marlin vs WGMMA Benchmark: E={E} K={K} N={N}")
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"Iterations: {iters}")
    print()

    from batchgen.moe.marlin_weight_prep import repack_int4_to_marlin_gs32
    from batchgen.moe.marlin_grouped_moe import _load_module as load_marlin
    from batchgen.moe.fused_int4_wgmma_grouped import _load_int4_grouped_module

    marlin_mod = load_marlin()
    wgmma_mod = _load_int4_grouped_module()
    if wgmma_mod is None:
        print("ERROR: WGMMA module not available")
        return

    n_tiles_s1 = N // 256
    n_tiles_s3 = K // 256

    # Create weights in both formats
    print("Creating weights (raw + Marlin)...", end=" ", flush=True)
    # WGMMA format: raw uint8 [N, K//2]
    gate_raw, up_raw, down_raw = [], [], []
    gate_scale, up_scale, down_scale = [], [], []
    # Marlin format: permuted int32
    gate_mqw, gate_ms, up_mqw, up_ms, down_mqw, down_ms = [], [], [], [], [], []

    for e in range(E):
        # Gate
        pu8, pi32, sc = create_k25_int4_weights_raw(K, N, device=device)
        gate_raw.append(pu8); gate_scale.append(sc)
        mqw, ms = repack_int4_to_marlin_gs32(pi32, sc, K, N)
        gate_mqw.append(mqw); gate_ms.append(ms)

        # Up
        pu8, pi32, sc = create_k25_int4_weights_raw(K, N, device=device)
        up_raw.append(pu8); up_scale.append(sc)
        mqw, ms = repack_int4_to_marlin_gs32(pi32, sc, K, N)
        up_mqw.append(mqw); up_ms.append(ms)

        # Down (input=N, output=K)
        pu8, pi32, sc = create_k25_int4_weights_raw(N, K, device=device)
        down_raw.append(pu8); down_scale.append(sc)
        mqw, ms = repack_int4_to_marlin_gs32(pi32, sc, N, K)
        down_mqw.append(mqw); down_ms.append(ms)
    print("done")

    # Marlin pointer arrays
    m_s1_B = torch.tensor(
        [w.data_ptr() for w in gate_mqw] + [w.data_ptr() for w in up_mqw],
        dtype=torch.int64, device=device)
    m_s1_S = torch.tensor(
        [s.data_ptr() for s in gate_ms] + [s.data_ptr() for s in up_ms],
        dtype=torch.int64, device=device)
    m_s3_B = torch.tensor([w.data_ptr() for w in down_mqw], dtype=torch.int64, device=device)
    m_s3_S = torch.tensor([s.data_ptr() for s in down_ms], dtype=torch.int64, device=device)

    # WGMMA pointer arrays
    w_gate_ptrs = torch.tensor([w.data_ptr() for w in gate_raw], dtype=torch.int64, device=device)
    w_gate_s = torch.tensor([s.data_ptr() for s in gate_scale], dtype=torch.int64, device=device)
    w_up_ptrs = torch.tensor([w.data_ptr() for w in up_raw], dtype=torch.int64, device=device)
    w_up_s = torch.tensor([s.data_ptr() for s in up_scale], dtype=torch.int64, device=device)
    w_down_ptrs = torch.tensor([w.data_ptr() for w in down_raw], dtype=torch.int64, device=device)
    w_down_s = torch.tensor([s.data_ptr() for s in down_scale], dtype=torch.int64, device=device)
    empty_bias = torch.empty(0, dtype=torch.int64, device=device)

    print(f"{'M':>5} | {'Marlin 3stg':>12} | {'WGMMA 2stg':>12} | {'Marlin/WGMMA':>12} | {'M S1':>8} | {'M S3':>8} | {'W S1':>8} | {'W S2':>8}")
    print(f"{'':>5} | {'total (us)':>12} | {'total (us)':>12} | {'':>12} | {'(us)':>8} | {'(us)':>8} | {'(us)':>8} | {'(us)':>8}")
    print("-" * 100)

    for M in M_values:
        mtp = M
        max_m_tiles_marlin = (M + 15) // 16
        max_m_tiles_wgmma = (M + 63) // 64

        A = torch.randn((E * mtp, K), dtype=torch.bfloat16, device=device) * 0.1
        expert_starts = torch.arange(E, dtype=torch.int32, device=device) * mtp
        expert_counts = torch.full((E,), M, dtype=torch.int32, device=device)

        # Marlin buffers
        gate_buf = torch.zeros(E * mtp, N, dtype=torch.bfloat16, device=device)
        up_buf_m = torch.zeros(E * mtp, N, dtype=torch.bfloat16, device=device)
        intermediate_m = torch.zeros(E * mtp, N, dtype=torch.bfloat16, device=device)
        expert_out_m = torch.zeros(E * mtp, K, dtype=torch.bfloat16, device=device)

        bpr_N = N * 2
        bpr_K = K * 2
        m_s1_C = torch.tensor(
            [gate_buf.data_ptr() + e * mtp * bpr_N for e in range(E)]
            + [up_buf_m.data_ptr() + e * mtp * bpr_N for e in range(E)],
            dtype=torch.int64, device=device)
        m_s3_C = torch.tensor(
            [expert_out_m.data_ptr() + e * mtp * bpr_K for e in range(E)],
            dtype=torch.int64, device=device)
        m_s1_ws = torch.zeros(2 * E * (n_tiles_s1 + 17), dtype=torch.int32, device=device)
        m_s3_ws = torch.zeros(E * (n_tiles_s3 + 17), dtype=torch.int32, device=device)

        def run_marlin_s1():
            marlin_mod.grouped_marlin_gemm_m16(
                A, m_s1_B, m_s1_C, m_s1_S,
                expert_starts, expert_counts,
                E, N, K, m_s1_ws, 2 * E, n_tiles_s1, max_m_tiles_marlin)

        def run_marlin_s2():
            marlin_mod.silu_mul(gate_buf, up_buf_m, intermediate_m)

        def run_marlin_s3():
            marlin_mod.grouped_marlin_gemm_m16(
                intermediate_m, m_s3_B, m_s3_C, m_s3_S,
                expert_starts, expert_counts,
                E, K, N, m_s3_ws, E, n_tiles_s3, max_m_tiles_marlin)

        def run_marlin_all():
            run_marlin_s1()
            run_marlin_s2()
            run_marlin_s3()

        # WGMMA uses non-inplace API (allocates internally, no TMA desc needed)
        stride_wn_gate = K // 2
        stride_sn_gate = K // 32
        stride_wn_down = N // 2
        stride_sn_down = N // 32

        def run_wgmma_s1():
            return wgmma_mod.grouped_int4_moe_stage1(
                A, expert_counts,
                w_gate_ptrs, w_gate_s,
                w_up_ptrs, w_up_s,
                empty_bias, empty_bias,
                N, stride_wn_gate, stride_sn_gate,
                max_m_tiles_wgmma, mtp)

        wgmma_intermediate = None

        def run_wgmma_s2():
            nonlocal wgmma_intermediate
            return wgmma_mod.grouped_int4_moe_stage2(
                wgmma_intermediate, expert_counts,
                w_down_ptrs, w_down_s,
                empty_bias,
                K, stride_wn_down, stride_sn_down,
                max_m_tiles_wgmma, mtp)

        def run_wgmma_all():
            nonlocal wgmma_intermediate
            wgmma_intermediate = run_wgmma_s1()
            return run_wgmma_s2()

        # Warmup
        for _ in range(10):
            run_marlin_all()
            run_wgmma_all()

        s = torch.cuda.Event(enable_timing=True)
        e_ev = torch.cuda.Event(enable_timing=True)

        # Bench Marlin S1
        s.record()
        for _ in range(iters):
            run_marlin_s1()
        e_ev.record(); torch.cuda.synchronize()
        m_s1_us = s.elapsed_time(e_ev) / iters * 1000

        # Bench Marlin S3
        s.record()
        for _ in range(iters):
            run_marlin_s3()
        e_ev.record(); torch.cuda.synchronize()
        m_s3_us = s.elapsed_time(e_ev) / iters * 1000

        # Bench Marlin total (S1+S2+S3)
        s.record()
        for _ in range(iters):
            run_marlin_all()
        e_ev.record(); torch.cuda.synchronize()
        m_total_us = s.elapsed_time(e_ev) / iters * 1000

        # Bench WGMMA S1
        s.record()
        for _ in range(iters):
            wgmma_intermediate = run_wgmma_s1()
        e_ev.record(); torch.cuda.synchronize()
        w_s1_us = s.elapsed_time(e_ev) / iters * 1000

        # Bench WGMMA S2
        s.record()
        for _ in range(iters):
            run_wgmma_s2()
        e_ev.record(); torch.cuda.synchronize()
        w_s2_us = s.elapsed_time(e_ev) / iters * 1000

        # Bench WGMMA total
        s.record()
        for _ in range(iters):
            run_wgmma_all()
        e_ev.record(); torch.cuda.synchronize()
        w_total_us = s.elapsed_time(e_ev) / iters * 1000

        ratio = m_total_us / w_total_us if w_total_us > 0 else float('inf')
        faster = "Marlin" if ratio < 1 else "WGMMA"
        speedup = 1/ratio if ratio < 1 else ratio
        label = f"{speedup:.2f}× {faster}"

        print(f"{M:5d} | {m_total_us:12.1f} | {w_total_us:12.1f} | {label:>12} | {m_s1_us:8.1f} | {m_s3_us:8.1f} | {w_s1_us:8.1f} | {w_s2_us:8.1f}")

    print()
    print("Marlin 3stg = S1 GEMM(gate+up) + S2 SiLU + S3 GEMM(down)")
    print("WGMMA 2stg = S1 fused(gate+up+SiLU) + S2 GEMM(down)")


if __name__ == "__main__":
    main()
