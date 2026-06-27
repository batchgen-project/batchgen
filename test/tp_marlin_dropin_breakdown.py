#!/usr/bin/env python3
"""Per-stage breakdown of batchgen_fused_marlin_moe — locate the drop-in's overhead.

The apples-to-apples sweep showed the drop-in is 3-5x slower than SGLang end-to-end
despite a competitive GEMM (roofline ~228us @ bs1 vs 945us total). This inlines the
exact stages and times each so we know whether the cost is moe_align, the explicit
gather (index_select), the GEMMs, or the scatter-combine (index_add) — i.e. what must
be fused into the kernel.

GPU-only:  BATCHGEN_KIMI_TP_MARLIN_V3=1 python test/tp_marlin_dropin_breakdown.py
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
_BLOCK = 16
ITERS = 50


def _t(fn):
    for _ in range(8):
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
    mod = _load_module()
    from sglang.srt.layers.moe.fused_moe_triton.moe_align_block_size import moe_align_block_size

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
    s1_B = w13.data_ptr() + idx * (H // 16) * (4 * INTER_PR) * 4
    s1_S = w13_s.data_ptr() + idx * (H // 32) * (2 * INTER_PR) * 2
    s3_B = w2.data_ptr() + idx * (INTER_PR // 16) * (2 * H) * 4
    s3_S = w2_s.data_ptr() + idx * (INTER_PR // 32) * H * 2
    n_tiles_s1 = (2 * INTER_PR) // 256 if (2 * INTER_PR) >= 256 else 1
    n_tiles_s3 = H // 256

    print(f"Drop-in stage breakdown  H={H} E={E} topk={TOPK} ws={WS}  (us, mean/{ITERS})")
    print(f"{'bs':>3} {'numG':>5} {'npost':>6} | {'align':>6} {'gather':>7} {'gemm':>7} "
          f"{'scatter':>8} | {'total':>7}")
    print("-" * 60)
    for bs in [1, 8, 32, 64]:
        numG = bs * WS
        g = torch.Generator(device=dev).manual_seed(7)
        hidden = (torch.randn(numG, H, device=dev, generator=g, dtype=torch.bfloat16) * 0.1)
        scores = torch.randn(numG, E, device=dev, generator=g)
        topk_w, topk_idx = torch.topk(torch.sigmoid(scores), TOPK, dim=1)
        topk_w = (topk_w / topk_w.sum(-1, keepdim=True) * 2.5).to(torch.float32)
        topk_idx = topk_idx.to(torch.int32)

        # precompute the parts that are routing-dependent (done once per decode anyway)
        sorted_ids, expert_ids, _ = moe_align_block_size(topk_idx, _BLOCK, E)
        num_post = sorted_ids.shape[0]
        counts = torch.bincount(topk_idx.reshape(-1).long(), minlength=E).to(torch.int32)
        padded = ((counts + _BLOCK - 1) // _BLOCK) * _BLOCK
        starts = torch.zeros(E, dtype=torch.int32, device=dev)
        starts[1:] = torch.cumsum(padded, 0)[:-1].to(torch.int32)
        Mtopk = numG * TOPK
        valid = sorted_ids < Mtopk
        flat = sorted_ids.clamp(0, Mtopk - 1)
        token_idx = (flat // TOPK).to(torch.int64)
        max_m_tiles = max(1, (int(counts.max().item()) + 15) // 16)
        s1_ws = torch.zeros(E * (n_tiles_s1 + 17), dtype=torch.int32, device=dev)
        s3_ws = torch.zeros(E * (n_tiles_s3 + 17), dtype=torch.int32, device=dev)
        inter = torch.empty(num_post, INTER_PR, dtype=torch.bfloat16, device=dev)
        out_c = torch.empty(num_post, H, dtype=torch.bfloat16, device=dev)
        inter_C = inter.data_ptr() + starts.to(torch.int64) * (INTER_PR * 2)
        s3_C = out_c.data_ptr() + starts.to(torch.int64) * (H * 2)
        w_row = topk_w.reshape(-1).index_select(0, flat).float() * valid.float()

        t_align = _t(lambda: moe_align_block_size(topk_idx, _BLOCK, E))
        t_gather = _t(lambda: hidden.index_select(0, token_idx) * valid.unsqueeze(1))
        A = hidden.index_select(0, token_idx) * valid.unsqueeze(1)

        def gemm():
            mod.grouped_marlin_tp_s1_v3(A, s1_B, inter_C, s1_S, starts, counts,
                                        E, 2 * INTER_PR, H, s1_ws, E, n_tiles_s1, max_m_tiles, INTER_PR)
            mod.grouped_marlin_tp_s3(inter, s3_B, s3_C, s3_S, starts, counts,
                                     E, H, INTER_PR, s3_ws, E, n_tiles_s3, max_m_tiles)
        t_gemm = _t(gemm)

        def scatter():
            out = torch.zeros(numG, H, dtype=torch.float32, device=dev)
            out.index_add_(0, token_idx, out_c.float() * w_row.unsqueeze(1))
            return out.bfloat16()
        t_scatter = _t(scatter)

        tot = t_align + t_gather + t_gemm + t_scatter
        print(f"{bs:>3} {numG:>5} {num_post:>6} | {t_align:>6.1f} {t_gather:>7.1f} "
              f"{t_gemm:>7.1f} {t_scatter:>8.1f} | {tot:>7.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
