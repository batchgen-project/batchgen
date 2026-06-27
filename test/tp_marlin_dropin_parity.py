#!/usr/bin/env python3
"""Drop-in correctness gate: batchgen_fused_marlin_moe vs SGLang fused_marlin_moe.

This is the APPLES-TO-APPLES correctness test for replacing SGLang's TP-MoE kernel
with BatchGen's. Both functions take the SAME standard GPTQ-marlin slabs (BatchGen
stores byte-identical marlin) + the SAME tokens/topk, run the full routed INT4 MoE,
and must agree to calc_diff < 1e-3 (cosine, the dev-infra convention). It validates
the whole drop-in: moe_align compact dispatch -> gather -> BatchGen GEMM (S1/silu/S3)
-> weighted scatter-combine, against SGLang's fused_marlin_moe end to end.

Select the BatchGen kernel variant with BATCHGEN_KIMI_TP_MARLIN_V3=1 (or _V2=1).
GPU-only, needs rebuilt batchgen_kernels + sglang:
    BATCHGEN_KIMI_TP_MARLIN_V3=1 python test/tp_marlin_dropin_parity.py
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
E = 32
TOPK = 8
M = 128


def calc_diff(x, y):
    x, y = x.double(), y.double()
    denom = (x * x + y * y).sum()
    if denom == 0:
        return 0.0
    return (1 - 2 * (x * y).sum() / denom).item()


def main():
    if not torch.cuda.is_available():
        print("SKIP: CUDA required."); return 0
    torch.cuda.set_device(0)
    dev = torch.device("cuda", 0)

    use_v3 = os.environ.get("BATCHGEN_KIMI_TP_MARLIN_V3", "0") == "1"
    use_v2 = os.environ.get("BATCHGEN_KIMI_TP_MARLIN_V2", "0") == "1"
    path = "v3" if use_v3 else "v2" if use_v2 else "v1"
    print(f"Drop-in parity  H={H} inter_pr={INTER_PR} E={E} topk={TOPK} M={M}  BG-path={path}")

    # --- standard GPTQ-marlin slabs (identical bytes feed BOTH functions) ---
    w13 = torch.empty(E, H // 16, 4 * INTER_PR, dtype=torch.int32, device=dev)
    w13_s = torch.empty(E, H // GROUP_SIZE, 2 * INTER_PR, dtype=torch.bfloat16, device=dev)
    w2 = torch.empty(E, INTER_PR // 16, 2 * H, dtype=torch.int32, device=dev)
    w2_s = torch.empty(E, INTER_PR // GROUP_SIZE, H, dtype=torch.bfloat16, device=dev)
    for e in range(E):
        g_raw, g_s = build_raw_int4(INTER_PR, H, dev, 100 + e * 3)
        u_raw, u_s = build_raw_int4(INTER_PR, H, dev, 101 + e * 3)
        d_raw, d_s = build_raw_int4(H, INTER_PR, dev, 102 + e * 3)
        g_mw, g_ms = raw_to_marlin_fused_gpu(g_raw, g_s, H, INTER_PR)
        u_mw, u_ms = raw_to_marlin_fused_gpu(u_raw, u_s, H, INTER_PR)
        w13[e] = torch.cat([g_mw, u_mw], dim=1); w13_s[e] = torch.cat([g_ms, u_ms], dim=1)
        w2[e], w2_s[e] = raw_to_marlin_fused_gpu(d_raw, d_s, INTER_PR, H)

    fused_marlin_moe = _load_fused_marlin_moe()
    all_pass = True
    # M=8 -> max_count<=8 exercises the S3 mma_trans (v3) path; M=128 -> S3 fallback.
    for m in [8, 128]:
        g = torch.Generator(device=dev).manual_seed(7)
        hidden = (torch.randn(m, H, device=dev, generator=g, dtype=torch.bfloat16) * 0.1)
        scores = torch.randn(m, E, device=dev, generator=g)
        topk_w, topk_idx = torch.topk(torch.sigmoid(scores), TOPK, dim=1)
        topk_w = (topk_w / topk_w.sum(-1, keepdim=True) * 2.5).to(torch.float32)  # ×2.5 routed scale
        topk_idx = topk_idx.to(torch.int32)
        maxc = int(torch.bincount(topk_idx.reshape(-1).long(), minlength=E).max().item())

        ref = fused_marlin_moe(
            hidden_states=hidden, w1=w13, w2=w2, w1_scale=w13_s, w2_scale=w2_s,
            gating_output=topk_w, topk_weights=topk_w, topk_ids=topk_idx,
            global_num_experts=E, expert_map=None, g_idx1=None, g_idx2=None,
            sort_indices1=None, sort_indices2=None, w1_zeros=None, w2_zeros=None,
            workspace=None, num_bits=4, is_k_full=True, inplace=False,
            routed_scaling_factor=None,
        )
        out = batchgen_fused_marlin_moe(
            hidden, w13, w2, w13_s, w2_s, topk_idx, topk_w, INTER_PR,
            routed_scaling_factor=None,
        )
        torch.cuda.synchronize()

        cd = calc_diff(out, ref)
        rel = (out.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-9)
        ok = cd < 1e-3
        all_pass = all_pass and ok
        s3 = "S3=mma_trans(v3)" if (use_v3 and maxc <= 8) else "S3=fallback"
        print(f"M={m:3d} maxc={maxc:2d} {s3}: calc_diff={cd:.3e}  rel_l2={rel.item():.3e}  "
              f"[{'PASS' if ok else 'FAIL'}]  (ref_norm={ref.float().norm().item():.2f} "
              f"bg_norm={out.float().norm().item():.2f})")
    print(f"\n{'ALL PASS' if all_pass else 'FAILURES PRESENT'} (thr calc_diff<1e-3)")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
