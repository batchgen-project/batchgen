#!/usr/bin/env python3
"""APPLES-TO-APPLES drop-in sweep: batchgen_fused_marlin_moe vs SGLang fused_marlin_moe.

Both are the FULL routed INT4 MoE over the SAME standard-marlin slabs and the SAME
tokens/topk, with the SAME moe_align compact dispatch — the only difference is the GEMM
(BatchGen grouped-marlin vs SGLang moe_wna16_marlin) plus BatchGen's explicit
gather/scatter glue (SGLang fuses the gather into its kernel; that glue is a real cost
of the drop-in and is included). This is the fair number for the hybrid threshold: the
num_global where BatchGen first beats SGLang -> BATCHGEN_KIMI_TP_MOE_HYBRID_BSZ.

Select the kernel variant with BATCHGEN_KIMI_TP_MARLIN_V3=1 (or _V2=1).
GPU-only:  BATCHGEN_KIMI_TP_MARLIN_V3=1 python test/tp_marlin_dropin_sweep.py
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from batchgen.moe.marlin_transform import raw_to_marlin_fused_gpu
from batchgen.moe.batchgen_fused_marlin_moe import batchgen_fused_marlin_moe
from tp_moe_repack_parity import build_raw_int4
from tp_marlin_moe_parity import _load_fused_marlin_moe

H = 7168
INTER_PR = 128
GROUP_SIZE = 32
E = 384
TOPK = 8
WS = 16
BATCH_SIZES = [1, 2, 4, 8, 16, 24, 32, 48, 64]   # per-rank; num_global = bs*WS
ITERS = 100


def _bench(fn, *a, **k):
    for _ in range(10):
        fn(*a, **k)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(ITERS):
        fn(*a, **k)
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / ITERS * 1e3   # us


def main():
    if not torch.cuda.is_available():
        print("SKIP: CUDA required."); return 0
    torch.cuda.set_device(0)
    dev = torch.device("cuda", 0)
    use_v3 = os.environ.get("BATCHGEN_KIMI_TP_MARLIN_V3", "0") == "1"
    use_v2 = os.environ.get("BATCHGEN_KIMI_TP_MARLIN_V2", "0") == "1"
    path = "v3" if use_v3 else "v2" if use_v2 else "v1"

    w13 = torch.empty(E, H // 16, 4 * INTER_PR, dtype=torch.int32, device=dev)
    w13_s = torch.empty(E, H // GROUP_SIZE, 2 * INTER_PR, dtype=torch.bfloat16, device=dev)
    w2 = torch.empty(E, INTER_PR // 16, 2 * H, dtype=torch.int32, device=dev)
    w2_s = torch.empty(E, INTER_PR // GROUP_SIZE, H, dtype=torch.bfloat16, device=dev)
    for e in range(E):
        g_raw, g_s = build_raw_int4(INTER_PR, H, dev, 100 + (e % 64) * 3)
        u_raw, u_s = build_raw_int4(INTER_PR, H, dev, 101 + (e % 64) * 3)
        d_raw, d_s = build_raw_int4(H, INTER_PR, dev, 102 + (e % 64) * 3)
        g_mw, g_ms = raw_to_marlin_fused_gpu(g_raw, g_s, H, INTER_PR)
        u_mw, u_ms = raw_to_marlin_fused_gpu(u_raw, u_s, H, INTER_PR)
        w13[e] = torch.cat([g_mw, u_mw], dim=1); w13_s[e] = torch.cat([g_ms, u_ms], dim=1)
        w2[e], w2_s[e] = raw_to_marlin_fused_gpu(d_raw, d_s, INTER_PR, H)

    fused_marlin_moe = _load_fused_marlin_moe()
    print(f"TP-MoE drop-in sweep  H={H} ws={WS} inter_pr={INTER_PR} E={E} topk={TOPK}  BG-path={path}")
    print(f"Device: {torch.cuda.get_device_name(0)}  iters={ITERS}")
    print(f"{'bs':>3} {'numG':>5} {'maxc':>4} | {'BG us':>8} {'SGL us':>8} | {'sgl/bg':>7}  winner")
    print("-" * 60)
    crossover = None
    for bs in BATCH_SIZES:
        numG = bs * WS
        g = torch.Generator(device=dev).manual_seed(7)
        hidden = (torch.randn(numG, H, device=dev, generator=g, dtype=torch.bfloat16) * 0.1)
        scores = torch.randn(numG, E, device=dev, generator=g)
        topk_w, topk_idx = torch.topk(torch.sigmoid(scores), TOPK, dim=1)
        topk_w = (topk_w / topk_w.sum(-1, keepdim=True) * 2.5).to(torch.float32)
        topk_idx = topk_idx.to(torch.int32)
        maxc = int(torch.bincount(topk_idx.reshape(-1).long(), minlength=E).max().item())

        sgl_us = _bench(
            fused_marlin_moe, hidden_states=hidden, w1=w13, w2=w2,
            w1_scale=w13_s, w2_scale=w2_s, gating_output=topk_w, topk_weights=topk_w,
            topk_ids=topk_idx, global_num_experts=E, expert_map=None,
            g_idx1=None, g_idx2=None, sort_indices1=None, sort_indices2=None,
            w1_zeros=None, w2_zeros=None, workspace=None, num_bits=4,
            is_k_full=True, inplace=False, routed_scaling_factor=None,
        )
        bg_us = _bench(
            batchgen_fused_marlin_moe, hidden, w13, w2, w13_s, w2_s,
            topk_idx, topk_w, INTER_PR, routed_scaling_factor=None,
        )
        ratio = sgl_us / bg_us
        win = "BatchGen" if ratio > 1.0 else "SGLang"
        if ratio > 1.0 and crossover is None:
            crossover = numG
        print(f"{bs:>3} {numG:>5} {maxc:>4} | {bg_us:>8.1f} {sgl_us:>8.1f} | {ratio:>6.2f}x  {win}")

    print()
    if crossover is not None:
        print(f"CROSSOVER: BatchGen ({path}) first wins at num_global = {crossover}.")
        print(f"  -> set BATCHGEN_KIMI_TP_MOE_HYBRID_BSZ={crossover}")
    else:
        print(f"No crossover in {BATCH_SIZES} — SGLang wins the whole sweep.")
    print("ratio = sgl_full / bg_full (>1 = BatchGen faster). Both = full routed scope, "
          "SAME moe_align dispatch; BG includes its explicit gather/scatter glue.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
