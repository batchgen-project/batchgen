#!/usr/bin/env python3
"""S1->S3 GEMM-only comparison: BatchGen grouped-marlin vs SGLang moe_wna16_marlin.

Isolates the GEMM (the two int4 marlin GEMMs + silu), moe_align PRECOMPUTED and
EXCLUDED on both sides, no final moe_sum combine. This answers: is BatchGen's GEMM
itself faster than SGLang's GEMM op?

CAVEAT (one-sided, stated honestly): SGLang's moe_wna16_marlin_gemm FUSES the gather
(reads hidden[sorted_token_ids] in the A-load) into the GEMM — inseparable. BatchGen's
GEMM runs on PRE-gathered compact buffers (gather is external, excluded here). So this
is SGL-GEMM(gather-fused) vs BG-GEMM(bare). If BG-bare can't beat SGL-fused here,
fusing gather/scatter into BG won't win either; if it can, fusing makes the drop-in win.

GPU-only:  BATCHGEN_KIMI_TP_MARLIN_V3=1 python test/tp_marlin_gemm_only.py
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from batchgen.moe.marlin_transform import raw_to_marlin_fused_gpu
from batchgen.moe.marlin_grouped_moe import _load_module
from tp_moe_repack_parity import build_raw_int4

H = 7168
INTER_PR = 128
GROUP_SIZE = 32
E = 384
TOPK = 8
WS = 16
ITERS = 100


def _t(fn):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(ITERS):
        fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / ITERS * 1e3


def main():
    torch.cuda.set_device(0)
    dev = torch.device("cuda", 0)
    use_v3 = os.environ.get("BATCHGEN_KIMI_TP_MARLIN_V3", "0") == "1"
    use_v2 = os.environ.get("BATCHGEN_KIMI_TP_MARLIN_V2", "0") == "1"
    path = "v3" if use_v3 else "v2" if use_v2 else "v1"
    mod = _load_module()
    from sgl_kernel import silu_and_mul
    from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
    from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import get_scalar_type
    st = get_scalar_type(4, False).id
    gemm_op = torch.ops.sgl_kernel.moe_wna16_marlin_gemm.default

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

    idx = torch.arange(E, dtype=torch.int64, device=dev)
    bg_s1_B = w13.data_ptr() + idx * (H // 16) * (4 * INTER_PR) * 4
    bg_s1_S = w13_s.data_ptr() + idx * (H // 32) * (2 * INTER_PR) * 2
    bg_s3_B = w2.data_ptr() + idx * (INTER_PR // 16) * (2 * H) * 4
    bg_s3_S = w2_s.data_ptr() + idx * (INTER_PR // 32) * H * 2
    n_tiles_s1 = (2 * INTER_PR) // 256 if (2 * INTER_PR) >= 256 else 1
    n_tiles_s3 = H // 256
    sms = torch.cuda.get_device_properties(dev).multi_processor_count

    print(f"S1->S3 GEMM-only  H={H} ws={WS} inter_pr={INTER_PR} E={E} topk={TOPK}  BG-path={path}")
    print(f"Device: {torch.cuda.get_device_name(0)}  iters={ITERS}  (moe_align excluded both sides)")
    print(f"{'bs':>3} {'numG':>5} {'maxc':>4} {'bsm':>3} | {'BG us':>8} {'SGL us':>8} | {'sgl/bg':>7}  winner")
    print("-" * 64)
    for bs in [1, 2, 4, 8, 16, 24, 32, 48, 64]:
        numG = bs * WS
        M = numG
        g = torch.Generator(device=dev).manual_seed(7)
        hidden = (torch.randn(M, H, device=dev, generator=g, dtype=torch.bfloat16) * 0.1)
        scores = torch.randn(M, E, device=dev, generator=g)
        topk_w, topk_idx = torch.topk(torch.sigmoid(scores), TOPK, dim=1)
        topk_w = (topk_w / topk_w.sum(-1, keepdim=True) * 2.5).to(torch.float32)
        topk_idx = topk_idx.to(torch.int32)
        maxc = int(torch.bincount(topk_idx.reshape(-1).long(), minlength=E).max().item())

        # ---- SGLang: its block_size_m selection + two GEMM ops + silu ----
        for bsm in [8, 16, 32, 48, 64]:
            if M * TOPK / E / bsm < 0.9:
                break
        s_ids, e_ids, npost = moe_align_block_size(topk_idx, bsm, E)
        ws_sz = min((max(2 * INTER_PR, H) // 64) * (s_ids.size(0) // bsm), sms * 4)
        sgl_ws = torch.zeros(ws_sz, dtype=torch.int32, device=dev)
        ic2 = torch.empty(M * TOPK, INTER_PR, dtype=torch.bfloat16, device=dev)
        ic13 = torch.empty(M * TOPK * max(2 * INTER_PR, H), dtype=torch.bfloat16, device=dev)
        ic1 = ic13[: M * TOPK * 2 * INTER_PR].view(-1, 2 * INTER_PR)
        ic3 = ic13[: M * TOPK * H].view(-1, H)

        def sgl():
            c1 = gemm_op(hidden, ic1, w13, None, w13_s, None, None, None, None, sgl_ws,
                         s_ids, e_ids, npost, topk_w, moe_block_size=bsm, top_k=TOPK,
                         mul_topk_weights=False, is_ep=False, b_q_type_id=st, size_m=M,
                         size_n=2 * INTER_PR, size_k=H, is_k_full=True, use_atomic_add=True,
                         use_fp32_reduce=True, is_zp_float=False)
            silu_and_mul(c1.view(-1, 2 * INTER_PR), ic2)
            gemm_op(ic2, ic3, w2, None, w2_s, None, None, None, None, sgl_ws,
                    s_ids, e_ids, npost, topk_w, moe_block_size=bsm, top_k=1,
                    mul_topk_weights=True, is_ep=False, b_q_type_id=st, size_m=M * TOPK,
                    size_n=H, size_k=INTER_PR, is_k_full=True, use_atomic_add=True,
                    use_fp32_reduce=True, is_zp_float=False)
        sgl_us = _t(sgl)

        # ---- BatchGen: pre-gathered compact buffers (block=16), bare S1/silu/S3 ----
        counts = torch.bincount(topk_idx.reshape(-1).long(), minlength=E).to(torch.int32)
        padded = ((counts + 15) // 16) * 16
        starts = torch.zeros(E, dtype=torch.int32, device=dev)
        starts[1:] = torch.cumsum(padded, 0)[:-1].to(torch.int32)
        nact = int(padded.sum().item())
        A = (torch.randn(max(nact, 16), H, device=dev, generator=g, dtype=torch.bfloat16) * 0.1)
        inter = torch.empty(A.shape[0], INTER_PR, dtype=torch.bfloat16, device=dev)
        out_c = torch.empty(A.shape[0], H, dtype=torch.bfloat16, device=dev)
        inter_C = inter.data_ptr() + starts.to(torch.int64) * (INTER_PR * 2)
        s3_C = out_c.data_ptr() + starts.to(torch.int64) * (H * 2)
        max_m_tiles = max(1, (maxc + 15) // 16)
        s1_ws = torch.zeros(E * (n_tiles_s1 + 17), dtype=torch.int32, device=dev)
        s3_ws = torch.zeros(E * (n_tiles_s3 + 17), dtype=torch.int32, device=dev)

        def bg():
            if use_v3:
                mod.grouped_marlin_tp_s1_v3(A, bg_s1_B, inter_C, bg_s1_S, starts, counts,
                                            E, 2 * INTER_PR, H, s1_ws, E, n_tiles_s1, max_m_tiles, INTER_PR)
                if maxc <= 8:
                    mod.grouped_marlin_tp_s3_v3(inter, bg_s3_B, s3_C, bg_s3_S, starts, counts,
                                                E, H, INTER_PR, s3_ws, E, n_tiles_s3, max_m_tiles, H)
                else:
                    mod.grouped_marlin_tp_s3(inter, bg_s3_B, s3_C, bg_s3_S, starts, counts,
                                             E, H, INTER_PR, s3_ws, E, n_tiles_s3, max_m_tiles)
            else:
                mod.grouped_marlin_tp_s1(A, bg_s1_B, inter_C, bg_s1_S, starts, counts,
                                         E, 2 * INTER_PR, H, s1_ws, E, n_tiles_s1, max_m_tiles, INTER_PR)
                mod.grouped_marlin_tp_s3(inter, bg_s3_B, s3_C, bg_s3_S, starts, counts,
                                         E, H, INTER_PR, s3_ws, E, n_tiles_s3, max_m_tiles)
        bg_us = _t(bg)

        ratio = sgl_us / bg_us
        win = "BatchGen" if ratio > 1.0 else "SGLang"
        print(f"{bs:>3} {numG:>5} {maxc:>4} {bsm:>3} | {bg_us:>8.1f} {sgl_us:>8.1f} | {ratio:>6.2f}x  {win}")
    print("\nBG = bare S1+silu+S3 on pre-gathered compact; SGL = 2x moe_wna16_marlin_gemm "
          "+ silu (gather fused in, moe_sum excluded). moe_align excluded both.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
