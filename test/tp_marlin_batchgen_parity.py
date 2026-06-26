#!/usr/bin/env python3
"""GPU parity for the BatchGen-marlin TP-MoE kernel path (CORRECTNESS GATE).

The roofline bench only TIMES random data — it does not check output. This test
runs the exact kernel sequence of _forward_decode_tp_batchgen (minus dispatch/
combine):
    grouped_marlin_gemm_m16(w13=concat(gate,up), N=2*inter_pr) ->
    silu_mul_split(active=max_m_tiles*16) ->
    grouped_marlin_gemm_m16(w2=down, N=H)
on KNOWN int4 weights + tokens, and compares to a torch fp32 dequant reference
silu(x@Wg)*(x@Wu) @ Wd per expert.

CRITICAL: uses expert counts that include values > 16 (so max_m_tiles > 1) and
counts crossing a 16-row m-tile boundary, to catch the token-dropping failure
mode of the max_m_tiles / silu `active` optimizations (commit 53ea09e9). count=0
experts must contribute nothing.

GPU-only. Run on a node with the rebuilt batchgen_kernels:
    python test/tp_marlin_batchgen_parity.py
"""
import os
import sys

# append (not insert): prefer the built site-packages batchgen_kernels if the
# workspace source tree's .so is stale (see tp_marlin_roofline_bench.py).
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from batchgen.moe.marlin_transform import raw_to_marlin_fused_gpu
from batchgen.moe.marlin_weight_prep import _dequantize_k25_int4
from tp_moe_repack_parity import build_raw_int4  # raw int4 generator

H = 7168
INTER_PR = 128          # ws16 per-rank intermediate
GROUP_SIZE = 32
E = 8                   # experts
MTP = 64                # buffer stride (small; > max count)
# Per-expert token counts — deliberately span the 16-row m-tile boundary:
# 0 (empty), 1, 16 (exact tile), 17 (crosses to 2nd tile), 32 (2 full tiles),
# plus a few mid values. max(counts)=33 -> max_m_tiles=ceil(33/16)=3.
COUNTS = [0, 1, 16, 17, 32, 33, 5, 8]
assert len(COUNTS) == E and max(COUNTS) <= MTP


def deq(raw, s, K, N):
    return _dequantize_k25_int4(raw, s, K, N).float()   # [K, N]


def main():
    if not torch.cuda.is_available():
        print("SKIP: CUDA required."); return 0
    torch.cuda.set_device(0)
    dev = torch.device("cuda", 0)
    import batchgen_kernels.moe._C_marlin_grouped_gemm as mod
    if not hasattr(mod, "silu_mul_split"):
        print("ERROR: rebuild batchgen_kernels (silu_mul_split missing)."); return 2
    print(f"BatchGen-marlin TP parity  H={H} inter_pr={INTER_PR} E={E} mtp={MTP} counts={COUNTS}")

    counts = torch.tensor(COUNTS, dtype=torch.int32, device=dev)
    max_count = int(counts.max().item())
    max_m_tiles = max(1, (max_count + 15) // 16)
    print(f"  max_count={max_count}  max_m_tiles={max_m_tiles}  active={max_m_tiles*16}")

    # Per-expert int4 weights + fp32 dequant references.
    Wg, Wu, Wd = [], [], []          # Wg/Wu: [H, inter_pr]  Wd: [inter_pr, H]
    w13 = torch.empty(E, H // 16, 4 * INTER_PR, dtype=torch.int32, device=dev)
    w13_s = torch.empty(E, H // GROUP_SIZE, 2 * INTER_PR, dtype=torch.bfloat16, device=dev)
    w2 = torch.empty(E, INTER_PR // 16, 2 * H, dtype=torch.int32, device=dev)
    w2_s = torch.empty(E, INTER_PR // GROUP_SIZE, H, dtype=torch.bfloat16, device=dev)
    for e in range(E):
        g_raw, g_s = build_raw_int4(INTER_PR, H, dev, 100 + e * 3 + 0)   # [inter_pr, H//8]
        u_raw, u_s = build_raw_int4(INTER_PR, H, dev, 100 + e * 3 + 1)
        d_raw, d_s = build_raw_int4(H, INTER_PR, dev, 100 + e * 3 + 2)   # [H, inter_pr//8]
        Wg.append(deq(g_raw, g_s, H, INTER_PR)); Wu.append(deq(u_raw, u_s, H, INTER_PR))
        Wd.append(deq(d_raw, d_s, INTER_PR, H))
        # marlinize gate/up SEPARATELY, concat marlin cols (the proven w13 layout)
        g_mw, g_ms = raw_to_marlin_fused_gpu(g_raw, g_s, H, INTER_PR)
        u_mw, u_ms = raw_to_marlin_fused_gpu(u_raw, u_s, H, INTER_PR)
        w13[e] = torch.cat([g_mw, u_mw], dim=1); w13_s[e] = torch.cat([g_ms, u_ms], dim=1)
        w2[e], w2_s[e] = raw_to_marlin_fused_gpu(d_raw, d_s, INTER_PR, H)

    # Tokens: random hidden for each expert's active rows, in the mtp-strided buffer.
    g = torch.Generator(device=dev).manual_seed(7)
    dispatched_x = (torch.randn(E * MTP, H, device=dev, generator=g, dtype=torch.bfloat16) * 0.1)
    expert_starts = torch.arange(E, dtype=torch.int32, device=dev) * MTP

    # Buffers + pointer arrays (mirror _init_tp_batchgen_buffers).
    idx = torch.arange(E, dtype=torch.int64, device=dev)
    gateup = torch.empty(E * MTP, 2 * INTER_PR, dtype=torch.bfloat16, device=dev)
    intermediate = torch.empty(E * MTP, INTER_PR, dtype=torch.bfloat16, device=dev)
    expert_out = torch.empty(E * MTP, H, dtype=torch.bfloat16, device=dev)
    s1_B = w13.data_ptr() + idx * (H // 16) * (4 * INTER_PR) * 4
    s1_S = w13_s.data_ptr() + idx * (H // GROUP_SIZE) * (2 * INTER_PR) * 2
    s1_C = gateup.data_ptr() + idx * (MTP * 2 * INTER_PR * 2)
    s3_B = w2.data_ptr() + idx * (INTER_PR // 16) * (2 * H) * 4
    s3_S = w2_s.data_ptr() + idx * (INTER_PR // GROUP_SIZE) * H * 2
    s3_C = expert_out.data_ptr() + idx * (MTP * H * 2)
    n_tiles_s1 = (2 * INTER_PR) // 256 if (2 * INTER_PR) >= 256 else 1
    n_tiles_s3 = H // 256
    s1_ws = torch.zeros(E * (n_tiles_s1 + 17), dtype=torch.int32, device=dev)
    s3_ws = torch.zeros(E * (n_tiles_s3 + 17), dtype=torch.int32, device=dev)

    # Run the kernel path.
    mod.grouped_marlin_gemm_m16(dispatched_x, s1_B, s1_C, s1_S, expert_starts, counts,
                                E, 2 * INTER_PR, H, s1_ws, E, n_tiles_s1, max_m_tiles)
    mod.silu_mul_split(gateup, intermediate, counts, E, max_m_tiles * 16, MTP, INTER_PR)
    mod.grouped_marlin_gemm_m16(intermediate, s3_B, s3_C, s3_S, expert_starts, counts,
                                E, H, INTER_PR, s3_ws, E, n_tiles_s3, max_m_tiles)
    torch.cuda.synchronize()

    # fp32 reference over the active rows of every expert.
    max_abs = 0.0; worst = None
    for e in range(E):
        c = COUNTS[e]
        for t in range(c):
            x = dispatched_x[e * MTP + t].float()
            act = F.silu(x @ Wg[e]) * (x @ Wu[e])      # [inter_pr]
            ref = act @ Wd[e]                          # [H]
            got = expert_out[e * MTP + t].float()
            d = (ref - got).abs().max().item()
            if d > max_abs:
                max_abs = d; worst = (e, t, c)
    ok_active = max_abs < 1e-1     # int4 W4A16 over K=7168 + bf16 acts

    # Tokens beyond each expert's count must be untouched-or-irrelevant: explicitly
    # check an empty expert (count=0) produced no NaN/garbage that could leak.
    e0 = COUNTS.index(0)
    empty_clean = torch.isfinite(expert_out[e0 * MTP:(e0 + 1) * MTP]).all().item()

    print(f"  [{'PASS' if ok_active else 'FAIL'}] active-row parity: max_abs_diff={max_abs:.3e} "
          f"(worst expert,tok,count={worst})")
    print(f"  [{'PASS' if empty_clean else 'FAIL'}] count=0 expert output finite (no garbage)")
    passed = ok_active and empty_clean
    print("ALL PASS" if passed else "FAILURES PRESENT")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
