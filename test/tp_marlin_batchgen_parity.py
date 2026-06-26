#!/usr/bin/env python3
"""Correctness gate for the BatchGen-marlin TP-MoE kernel path — dev-infra standard.

Uses the batchgen_kernel_dev sanity convention (v12c_m8_marlin.py): the primary
metric is calc_diff (cosine-similarity) < 1e-3 against a BF16 reference
torch.mm(a, w_ref) where w_ref = the DEQUANTIZED int4 weight (matches the kernel's
bf16xbf16->fp32->bf16 path). bf16_elem_fail% is reported diagnostically (INT4
dequant exceeds per-element BF16 tol by design). A fp32 reference is WRONG here.

Runs the actual path: grouped_marlin_gemm_m16(w13=concat(gate,up), N=2*inter_pr)
-> silu_mul_split(active=max_m_tiles*16) -> grouped_marlin_gemm_m16(w2=down, N=H).
Expert counts span the 16-row m-tile boundary (0,1,16,17,32,33,...) to catch
token-dropping from the max_m_tiles / silu `active` optimizations (commit 53ea09e9).

GPU-only, needs rebuilt batchgen_kernels:  python test/tp_marlin_batchgen_parity.py
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from batchgen.moe.marlin_transform import raw_to_marlin_fused_gpu
from batchgen.moe.marlin_weight_prep import _dequantize_k25_int4
from tp_moe_repack_parity import build_raw_int4

H = 7168
INTER_PR = 128
GROUP_SIZE = 32
E = 8
MTP = 64
COUNTS = [0, 1, 16, 17, 32, 33, 5, 8]      # span the 16-row m-tile boundary
assert len(COUNTS) == E and max(COUNTS) <= MTP


def calc_diff(x, y):
    """DeepGEMM-style cosine metric (batchgen_kernel_dev convention)."""
    x, y = x.double(), y.double()
    denom = (x * x + y * y).sum()
    if denom == 0:
        return 0.0
    return (1 - 2 * (x * y).sum() / denom).item()


def wref(raw, s, K, N):
    return _dequantize_k25_int4(raw, s, K, N).to(torch.bfloat16)   # [K, N] bf16


def main():
    if not torch.cuda.is_available():
        print("SKIP: CUDA required."); return 0
    torch.cuda.set_device(0)
    dev = torch.device("cuda", 0)
    import batchgen_kernels.moe._C_marlin_grouped_gemm as mod
    if not hasattr(mod, "silu_mul_split"):
        print("ERROR: rebuild batchgen_kernels (silu_mul_split missing)."); return 2

    use_v2 = os.environ.get("BATCHGEN_KIMI_TP_MARLIN_V2", "0") == "1"
    if use_v2 and not (hasattr(mod, "grouped_marlin_tp_s1") and hasattr(mod, "grouped_marlin_tp_s3")):
        print("ERROR: BATCHGEN_KIMI_TP_MARLIN_V2=1 but grouped_marlin_tp_s1/s3 missing "
              "— rebuild batchgen_kernels."); return 2

    counts = torch.tensor(COUNTS, dtype=torch.int32, device=dev)
    max_m_tiles = max(1, (int(counts.max().item()) + 15) // 16)
    print(f"BatchGen-marlin TP parity (calc_diff<1e-3 vs BF16 ref)  "
          f"H={H} inter_pr={INTER_PR} E={E} counts={COUNTS} max_m_tiles={max_m_tiles}  "
          f"path={'v2 (MarlinTP fused-S1 + STAGES2-S3)' if use_v2 else 'v1 (m16 + silu_mul_split)'}")

    Wg, Wu, Wd = [], [], []
    w13 = torch.empty(E, H // 16, 4 * INTER_PR, dtype=torch.int32, device=dev)
    w13_s = torch.empty(E, H // GROUP_SIZE, 2 * INTER_PR, dtype=torch.bfloat16, device=dev)
    w2 = torch.empty(E, INTER_PR // 16, 2 * H, dtype=torch.int32, device=dev)
    w2_s = torch.empty(E, INTER_PR // GROUP_SIZE, H, dtype=torch.bfloat16, device=dev)
    for e in range(E):
        g_raw, g_s = build_raw_int4(INTER_PR, H, dev, 100 + e * 3)
        u_raw, u_s = build_raw_int4(INTER_PR, H, dev, 101 + e * 3)
        d_raw, d_s = build_raw_int4(H, INTER_PR, dev, 102 + e * 3)
        Wg.append(wref(g_raw, g_s, H, INTER_PR))      # [H, inter_pr]
        Wu.append(wref(u_raw, u_s, H, INTER_PR))
        Wd.append(wref(d_raw, d_s, INTER_PR, H))      # [inter_pr, H]
        g_mw, g_ms = raw_to_marlin_fused_gpu(g_raw, g_s, H, INTER_PR)
        u_mw, u_ms = raw_to_marlin_fused_gpu(u_raw, u_s, H, INTER_PR)
        w13[e] = torch.cat([g_mw, u_mw], dim=1); w13_s[e] = torch.cat([g_ms, u_ms], dim=1)
        w2[e], w2_s[e] = raw_to_marlin_fused_gpu(d_raw, d_s, INTER_PR, H)

    g = torch.Generator(device=dev).manual_seed(7)
    dispatched_x = (torch.randn(E * MTP, H, device=dev, generator=g, dtype=torch.bfloat16) * 0.1)
    expert_starts = torch.arange(E, dtype=torch.int32, device=dev) * MTP

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

    if use_v2:
        # v2: fused S1 writes SiLU(gate)*up DIRECTLY into intermediate (no gateup
        # round-trip, no silu_mul_split); S3 (down) uses the STAGES=2 kernel.
        s1f_C = intermediate.data_ptr() + idx * (MTP * INTER_PR * 2)
        mod.grouped_marlin_tp_s1(dispatched_x, s1_B, s1f_C, s1_S, expert_starts, counts,
                                 E, 2 * INTER_PR, H, s1_ws, E, n_tiles_s1, max_m_tiles, INTER_PR)
        mod.grouped_marlin_tp_s3(intermediate, s3_B, s3_C, s3_S, expert_starts, counts,
                                 E, H, INTER_PR, s3_ws, E, n_tiles_s3, max_m_tiles)
    else:
        mod.grouped_marlin_gemm_m16(dispatched_x, s1_B, s1_C, s1_S, expert_starts, counts,
                                    E, 2 * INTER_PR, H, s1_ws, E, n_tiles_s1, max_m_tiles)
        mod.silu_mul_split(gateup, intermediate, counts, E, max_m_tiles * 16, MTP, INTER_PR)
        mod.grouped_marlin_gemm_m16(intermediate, s3_B, s3_C, s3_S, expert_starts, counts,
                                    E, H, INTER_PR, s3_ws, E, n_tiles_s3, max_m_tiles)
    torch.cuda.synchronize()

    # BF16 reference (matches kernel precision) over every expert's active rows.
    all_pass = True
    worst_cd = 0.0
    for e in range(E):
        c = COUNTS[e]
        if c == 0:
            continue
        a = dispatched_x[e * MTP:e * MTP + c]                  # [c, H] bf16
        ref = (F.silu(torch.mm(a, Wg[e])) * torch.mm(a, Wu[e]))  # [c, inter_pr] bf16
        ref = torch.mm(ref, Wd[e])                             # [c, H] bf16
        out = expert_out[e * MTP:e * MTP + c]
        cd = calc_diff(out, ref)
        diff = (out.float() - ref.float()).abs()
        bf16_tol = 1e-5 + 1.6e-2 * ref.float().abs()
        bf16_fail = (diff > bf16_tol).float().mean().item() * 100
        ok = cd < 1e-3
        all_pass = all_pass and ok
        worst_cd = max(worst_cd, cd)
        print(f"  E{e} (count={c:2d}): calc_diff={cd:.2e}  bf16_elem_fail={bf16_fail:5.2f}% "
              f"[{'PASS' if ok else 'FAIL'}]")

    print(f"\n{'ALL PASS' if all_pass else 'FAILURES PRESENT'}  (worst calc_diff={worst_cd:.2e}, thr 1e-3)")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
