#!/usr/bin/env python3
"""Apples-to-apples TP-MoE bsz sweep: BatchGen full routed MoE vs SGLang fused_marlin_moe.

Unlike tp_marlin_roofline_bench.py (which times BG GEMM-ONLY vs SGL full-MoE — an
admitted apples-to-oranges ratio), this harness times the SAME full routed scope on
both, from identical (hidden, topk_ids, topk_weights) -> [num_global, H]:

  BG full  = dispatch_scatter_3d -> S1 -> S3 -> reduce_weighted_scatter
             (mirrors _forward_decode_tp_batchgen steps 5-8, INCLUDING the honest
              expert_counts.max().item() host sync that production pays).
  SGL full = one fused_marlin_moe(...) call.

Both read the SAME build_tp_weights slabs (w13 concat / w2). AllGather / AllReduce /
shared-expert are excluded on BOTH (identical envelope), so the ratio is fair.

Reports sgl_full / bg_full per batch size (>1 = BatchGen faster, rule #10) and prints
the CROSSOVER num_global -> that is the value to set for BATCHGEN_KIMI_TP_MOE_HYBRID_BSZ
(the hybrid then runs SGL below it, BatchGen at/above it).

Select the BatchGen kernel path with BATCHGEN_KIMI_TP_MARLIN_V3=1 (or _V2=1; default v1).
Correctness pre-req: tp_marlin_batchgen_parity.py (calc_diff<1e-3) on the same build.

GPU-only; needs a fresh batchgen_kernels build (V3 symbols were added here):
    CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=9.0a \
        BATCHGEN_KIMI_TP_MARLIN_V3=1 python test/tp_marlin_hybrid_sweep.py
"""
import os
import sys

# append (not insert(0)): keep the freshly-built site-packages batchgen_kernels.so
# ahead of the workspace SOURCE tree (which has a stale/no compiled .so).
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

# Reuse the roofline bench's weight builder + routing + constants (import only;
# its main() is __main__-guarded, so importing runs no benchmark).
from tp_marlin_roofline_bench import (              # noqa: E402
    build_tp_weights, make_routing,
    H, INTER_PR, E, TOP_K, WORLD_SIZE,
)
from tp_marlin_moe_parity import _load_fused_marlin_moe      # noqa: E402
from batchgen.moe.dispatch_scatter_3d import (               # noqa: E402
    dispatch_scatter_3d, reduce_weighted_scatter,
)

ITERS = 100
WARMUP = 10
MTP = int(os.getenv("BATCHGEN_KIMI_TP_MOE_MTP", "256"))     # fixed buffer stride
BATCH_SIZES = [1, 2, 4, 8, 16, 24, 32, 48, 64]              # per-rank; num_global = bs*ws


def cuda_time(fn):
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(ITERS):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / ITERS * 1000.0  # us


def main():
    if not torch.cuda.is_available():
        print("SKIP: CUDA required (Marlin GEMM + dispatch + sgl_kernel are GPU-only).")
        return 0
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)

    try:
        import batchgen_kernels.moe._C_marlin_grouped_gemm as mod
    except ImportError as ex:
        print(f"ERROR: cannot import the marlin grouped-GEMM extension: {ex}\n"
              "  -> REBUILD batchgen_kernels (pip install --force-reinstall).")
        return 2

    use_v3 = os.environ.get("BATCHGEN_KIMI_TP_MARLIN_V3", "0") == "1"
    use_v2 = os.environ.get("BATCHGEN_KIMI_TP_MARLIN_V2", "0") == "1"
    if use_v3 and not (hasattr(mod, "grouped_marlin_tp_s1_v3") and hasattr(mod, "grouped_marlin_tp_s3_v3")):
        print("ERROR: BATCHGEN_KIMI_TP_MARLIN_V3=1 but grouped_marlin_tp_s1_v3/s3_v3 missing "
              "— rebuild batchgen_kernels.")
        return 2
    if use_v2 and not (hasattr(mod, "grouped_marlin_tp_s1") and hasattr(mod, "grouped_marlin_tp_s3")):
        print("ERROR: BATCHGEN_KIMI_TP_MARLIN_V2=1 but grouped_marlin_tp_s1/s3 missing "
              "— rebuild batchgen_kernels.")
        return 2
    bg_path = "v3 (STAGES3-S1 + M8-S3)" if use_v3 else "v2 (MarlinTP)" if use_v2 else "v1 (m16)"

    print(f"TP-MoE hybrid sweep  H={H} ws={WORLD_SIZE} inter_pr={INTER_PR} E={E} top_k={TOP_K} "
          f"mtp={MTP}  BG-path={bg_path}")
    print(f"Device: {torch.cuda.get_device_name()}  iters={ITERS}")
    print("Building TP-MoE Marlin weights (384 experts)...", end=" ", flush=True)
    w13, w13_s, w2, w2_s = build_tp_weights(device)
    print("done")

    sgl_moe = _load_fused_marlin_moe()
    from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace
    sgl_ws = marlin_make_workspace(device, max_blocks_per_sm=4)

    n_tiles_s1 = (2 * INTER_PR) // 256
    n_tiles_s3 = H // 256

    # Per-expert weight/scale + buffer pointer arrays (all stable: weights and the
    # mtp-strided buffers below are allocated once).
    idx = torch.arange(E, dtype=torch.int64, device=device)
    s1_B = w13.data_ptr() + idx * (H // 16) * (4 * INTER_PR) * 4
    s1_S = w13_s.data_ptr() + idx * (H // 32) * (2 * INTER_PR) * 2
    s3_B = w2.data_ptr() + idx * (INTER_PR // 16) * (2 * H) * 4
    s3_S = w2_s.data_ptr() + idx * (INTER_PR // 32) * H * 2

    dispatched_x = torch.empty(E * MTP, H, dtype=torch.bfloat16, device=device)
    gateup = torch.empty(E * MTP, 2 * INTER_PR, dtype=torch.bfloat16, device=device)
    intermediate = torch.empty(E * MTP, INTER_PR, dtype=torch.bfloat16, device=device)
    expert_out = torch.empty(E * MTP, H, dtype=torch.bfloat16, device=device)
    s1_C = gateup.data_ptr() + idx * (MTP * 2 * INTER_PR * 2)
    s1f_C = intermediate.data_ptr() + idx * (MTP * INTER_PR * 2)
    s3_C = expert_out.data_ptr() + idx * (MTP * H * 2)
    expert_starts = torch.arange(E, dtype=torch.int32, device=device) * MTP
    s1_ws = torch.zeros(E * (n_tiles_s1 + 17), dtype=torch.int32, device=device)
    s3_ws = torch.zeros(E * (n_tiles_s3 + 17), dtype=torch.int32, device=device)
    ec_buf = torch.zeros(E, dtype=torch.int32, device=device)
    ecnt_buf = torch.zeros(E, dtype=torch.int32, device=device)

    hdr = f"{'bs':>3} {'numG':>5} {'M/e':>5} {'maxc':>5} | {'BG us':>8} {'SGL us':>8} | {'sgl/bg':>7}  winner"
    print(hdr)
    print("-" * len(hdr))

    crossover = None
    for bs in BATCH_SIZES:
        num_global = bs * WORLD_SIZE
        hidden, topk_ids, topk_weights = make_routing(num_global, device)
        topk_ids32 = topk_ids.to(torch.int32)
        counts = torch.bincount(topk_ids.reshape(-1), minlength=E).int()
        max_e = int(counts.max().item())
        m_per_e = counts.sum().item() / E
        if max_e > MTP:
            print(f"{bs:3d} {num_global:5d} SKIP: max per-expert count {max_e} > mtp {MTP} "
                  "(raise BATCHGEN_KIMI_TP_MOE_MTP).")
            continue
        topk_pos_buf = torch.empty(num_global * TOP_K, dtype=torch.int32, device=device)
        result_buffer = torch.empty(num_global, H, dtype=torch.bfloat16, device=device)

        def bg_full():
            expert_counts, topk_pos = dispatch_scatter_3d(
                hidden, topk_ids32, dispatched_x, 0, E, MTP,
                ec_buf, ecnt_buf, topk_pos_buf)
            # Honest production host sync (shrinks the down grid to actual occupancy).
            max_count = int(expert_counts.max().item())
            mmt = max(1, (max_count + 15) // 16)
            if use_v3:
                mod.grouped_marlin_tp_s1_v3(
                    dispatched_x, s1_B, s1f_C, s1_S, expert_starts, expert_counts,
                    E, 2 * INTER_PR, H, s1_ws, E, n_tiles_s1, mmt, INTER_PR)
                if max_count <= 8:
                    mod.grouped_marlin_tp_s3_v3(
                        intermediate, s3_B, s3_C, s3_S, expert_starts, expert_counts,
                        E, H, INTER_PR, s3_ws, E, n_tiles_s3, mmt, H)
                else:
                    mod.grouped_marlin_tp_s3(
                        intermediate, s3_B, s3_C, s3_S, expert_starts, expert_counts,
                        E, H, INTER_PR, s3_ws, E, n_tiles_s3, mmt)
            elif use_v2:
                mod.grouped_marlin_tp_s1(
                    dispatched_x, s1_B, s1f_C, s1_S, expert_starts, expert_counts,
                    E, 2 * INTER_PR, H, s1_ws, E, n_tiles_s1, mmt, INTER_PR)
                mod.grouped_marlin_tp_s3(
                    intermediate, s3_B, s3_C, s3_S, expert_starts, expert_counts,
                    E, H, INTER_PR, s3_ws, E, n_tiles_s3, mmt)
            else:
                mod.grouped_marlin_gemm_m16(
                    dispatched_x, s1_B, s1_C, s1_S, expert_starts, expert_counts,
                    E, 2 * INTER_PR, H, s1_ws, E, n_tiles_s1, mmt)
                mod.silu_mul_split(gateup, intermediate, expert_counts,
                                   E, mmt * 16, MTP, INTER_PR)
                mod.grouped_marlin_gemm_m16(
                    intermediate, s3_B, s3_C, s3_S, expert_starts, expert_counts,
                    E, H, INTER_PR, s3_ws, E, n_tiles_s3, mmt)
            reduce_weighted_scatter(
                expert_out, topk_pos, topk_weights, num_global, H, TOP_K, result_buffer)

        def sgl_full():
            return sgl_moe(
                hidden_states=hidden, w1=w13, w2=w2, w1_scale=w13_s, w2_scale=w2_s,
                gating_output=topk_weights, topk_weights=topk_weights, topk_ids=topk_ids32,
                global_num_experts=E, expert_map=None,
                g_idx1=None, g_idx2=None, sort_indices1=None, sort_indices2=None,
                w1_zeros=None, w2_zeros=None, workspace=sgl_ws,
                num_bits=4, is_k_full=True, inplace=False, routed_scaling_factor=None)

        for _ in range(WARMUP):
            bg_full(); sgl_full()
        torch.cuda.synchronize()

        bg_us = cuda_time(bg_full)
        sgl_us = cuda_time(sgl_full)
        ratio = sgl_us / bg_us
        winner = "BatchGen" if ratio >= 1.0 else "SGLang"
        if ratio >= 1.0 and crossover is None:
            crossover = num_global
        print(f"{bs:3d} {num_global:5d} {m_per_e:5.1f} {max_e:5d} | "
              f"{bg_us:8.1f} {sgl_us:8.1f} | {ratio:6.2f}x  {winner}")

    print()
    if crossover is not None:
        print(f"CROSSOVER: BatchGen ({bg_path}) first wins at num_global = {crossover}.")
        print(f"  -> set  BATCHGEN_KIMI_TP_MOE_HYBRID_BSZ={crossover}  "
              "(+ BATCHGEN_KIMI_TP_MOE_KERNEL=hybrid) so the hybrid runs SGL below it, "
              "BatchGen at/above it.")
    else:
        print("No crossover in the swept range: SGLang wins at every bsz "
              f"(BG path {bg_path}). Tune the kernel further or widen BATCH_SIZES.")
    print("ratio = sgl_full / bg_full (>1 = BatchGen faster). Both = full routed scope "
          "(dispatch+S1+S3+reduce vs fused_marlin_moe); AllGather/AllReduce excluded on both.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
