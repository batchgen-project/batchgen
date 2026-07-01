#!/usr/bin/env python3
"""Stage 2 — INT4 MoE kernel A/B for COMPUTE-BOUND prefill (large tokens/expert).

Three arms compute the SAME MoE (silu(gate·x)·(up·x) → down) from the SAME INT4 weights,
one expert group per token (top_k=1, weight=1), so per-expert throughput is comparable:

  1. BatchGen-WGMMA   raw uint8  -> grouped_int4_moe_stage1 (gate|up|silu) + stage2 (down)
  2. BatchGen-marlin  marlin i32 -> grouped_marlin_gemm_m16 (gate|up) + silu_mul + m16 (down)
  3. SGL-marlin       marlin i32 -> fused_marlin_moe (SGLang moe_wna16_marlin, decode path)

Reports us, TFLOP/s = 2·E·M_e·(2·H·N + N·H)/t, and calc_diff = |out-ref|.mean()/|ref|.mean()
vs a bf16 dequant reference (gate <1e-3). M_e sweeps the prefill regime; the 64k-prefill
operating point is ~1365 tokens/expert (64000·top_k/384).

Run on a Hopper GPU (H20 node0 / GH02), from the BatchGen workspace:
    CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=9.0a \
        python -m batchgen.moe.bench_prefill_kernel_ab
"""

import torch

from batchgen.moe.bench_marlin_vs_wgmma import create_k25_int4_weights_raw
from batchgen.moe.marlin_weight_prep import repack_int4_to_marlin_gs32
from batchgen.moe.marlin_transform import raw_to_marlin_fused_gpu

H = 7168          # hidden
N = 2048          # expert intermediate
E = 128           # experts in the bench (grid occupancy; per-expert tput is E-independent)
GROUP_SIZE = 32
M_VALUES = [256, 512, 1024, 2048, 4096]   # tokens per expert; ~1365 = 64k operating point
ITERS = 30


def _dequant(packed_i32, scales, rows, cols):
    """INT4 packed [rows, cols//8] + scales [rows, cols//32] bf16 -> bf16 weight [rows, cols]."""
    q = torch.zeros(rows, cols, dtype=torch.float32, device=packed_i32.device)
    for i in range(8):
        q[:, i::8] = ((packed_i32 >> (i * 4)) & 0xF).float()
    scale_exp = scales.float().repeat_interleave(GROUP_SIZE, dim=1)
    return ((q - 8.0) * scale_exp).to(torch.bfloat16)


def _time(fn, iters=ITERS):
    for _ in range(5):
        fn()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1000.0  # us


def _calc_diff(out, ref):
    return ((out.float() - ref.float()).abs().mean() / ref.float().abs().mean().clamp(min=1e-9)).item()


def main():
    dev = "cuda"
    print(f"Prefill INT4 MoE kernel A/B | device={torch.cuda.get_device_name()} | iters={ITERS}")
    print(f"H={H} N={N} E={E} group_size={GROUP_SIZE}  (top_k=1, one expert per token)\n")

    from batchgen.moe.marlin_grouped_moe import _load_module as load_marlin
    from batchgen.moe.fused_int4_wgmma_grouped import _load_int4_grouped_module
    marlin_mod = load_marlin()
    wgmma_mod = _load_int4_grouped_module()

    # SGL fused_marlin_moe (mirror the decode path's lazy import + server-args seed).
    try:
        from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import fused_marlin_moe
        from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace
        from sglang.srt.server_args import (
            ServerArgs, get_global_server_args, set_global_server_args_for_scheduler)
        try:
            get_global_server_args()
        except ValueError:
            set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
        sgl_workspace = marlin_make_workspace(torch.device(dev), max_blocks_per_sm=4)
        have_sgl = True
    except Exception as ex:
        print(f"[warn] SGL fused_marlin_moe unavailable: {ex}")
        have_sgl = False

    print("Building weights (raw + marlin + SGL slabs + bf16 golden)...", flush=True)
    # Per-expert raw uint8 (WGMMA) + int32 (repack) + scales; dequant golden weights.
    gate_raw, up_raw, down_raw = [], [], []
    gate_sc, up_sc, down_sc = [], [], []
    gate_mqw, gate_ms, up_mqw, up_ms, down_mqw, down_ms = [], [], [], [], [], []
    gate_w, up_w, down_w = [], [], []   # bf16 golden weights [out, in]
    # SGL slabs
    w13 = torch.empty(E, H // 16, 4 * N, dtype=torch.int32, device=dev)
    w2 = torch.empty(E, N // 16, 2 * H, dtype=torch.int32, device=dev)
    w13s = torch.empty(E, H // GROUP_SIZE, 2 * N, dtype=torch.bfloat16, device=dev)
    w2s = torch.empty(E, N // GROUP_SIZE, H, dtype=torch.bfloat16, device=dev)

    for e in range(E):
        gpu8, gpi32, gs = create_k25_int4_weights_raw(H, N, device=dev)   # gate [N,H]
        upu8, upi32, us = create_k25_int4_weights_raw(H, N, device=dev)   # up   [N,H]
        dpu8, dpi32, ds = create_k25_int4_weights_raw(N, H, device=dev)   # down [H,N]
        gate_raw.append(gpu8); up_raw.append(upu8); down_raw.append(dpu8)
        gate_sc.append(gs); up_sc.append(us); down_sc.append(ds)

        gmw, gms = repack_int4_to_marlin_gs32(gpi32, gs, H, N)
        umw, ums = repack_int4_to_marlin_gs32(upi32, us, H, N)
        dmw, dms = repack_int4_to_marlin_gs32(dpi32, ds, N, H)
        gate_mqw.append(gmw); gate_ms.append(gms)
        up_mqw.append(umw); up_ms.append(ums)
        down_mqw.append(dmw); down_ms.append(dms)

        gate_w.append(_dequant(gpi32, gs, N, H))   # [N, H]
        up_w.append(_dequant(upi32, us, N, H))     # [N, H]
        down_w.append(_dequant(dpi32, ds, H, N))   # [H, N]

        # SGL slabs (no TP slice; inter_pr = N): concat(marlin(gate), marlin(up)).
        gmw2, gms2 = raw_to_marlin_fused_gpu(gpi32, gs, H, N)
        umw2, ums2 = raw_to_marlin_fused_gpu(upi32, us, H, N)
        dmw2, dms2 = raw_to_marlin_fused_gpu(dpi32, ds, N, H)
        w13[e] = torch.cat([gmw2, umw2], dim=1)
        w13s[e] = torch.cat([gms2, ums2], dim=1)
        w2[e] = dmw2
        w2s[e] = dms2
    print("done\n")

    # Pointer arrays (WGMMA + marlin) — built once, weights are static.
    empty_bias = torch.empty(0, dtype=torch.int64, device=dev)
    w_gate_p = torch.tensor([w.data_ptr() for w in gate_raw], dtype=torch.int64, device=dev)
    w_gate_s = torch.tensor([s.data_ptr() for s in gate_sc], dtype=torch.int64, device=dev)
    w_up_p = torch.tensor([w.data_ptr() for w in up_raw], dtype=torch.int64, device=dev)
    w_up_s = torch.tensor([s.data_ptr() for s in up_sc], dtype=torch.int64, device=dev)
    w_down_p = torch.tensor([w.data_ptr() for w in down_raw], dtype=torch.int64, device=dev)
    w_down_s = torch.tensor([s.data_ptr() for s in down_sc], dtype=torch.int64, device=dev)
    m_s1_B = torch.tensor([w.data_ptr() for w in gate_mqw] + [w.data_ptr() for w in up_mqw],
                          dtype=torch.int64, device=dev)
    m_s1_S = torch.tensor([s.data_ptr() for s in gate_ms] + [s.data_ptr() for s in up_ms],
                          dtype=torch.int64, device=dev)
    m_s3_B = torch.tensor([w.data_ptr() for w in down_mqw], dtype=torch.int64, device=dev)
    m_s3_S = torch.tensor([s.data_ptr() for s in down_ms], dtype=torch.int64, device=dev)
    n_tiles_s1 = N // 256
    n_tiles_s3 = H // 256
    stride_wn_gate, stride_sn_gate = H // 2, H // 32
    stride_wn_down, stride_sn_down = N // 2, N // 32

    hdr = (f"{'M/exp':>6} | {'WGMMA':>18} | {'BG-marlin':>18} | {'SGL-marlin':>18} | "
           f"{'best':>10}")
    print(hdr)
    print(f"{'':>6} | {'us   TFLOP/s  diff':>18} | {'us   TFLOP/s  diff':>18} | "
          f"{'us   TFLOP/s  diff':>18} |")
    print("-" * len(hdr))

    flops_per = lambda Me: 2.0 * E * Me * (2 * H * N + N * H)

    for Me in M_VALUES:
        M = E * Me
        A = torch.randn((M, H), dtype=torch.bfloat16, device=dev) * 0.1
        expert_counts = torch.full((E,), Me, dtype=torch.int32, device=dev)
        expert_starts = torch.arange(E, dtype=torch.int32, device=dev) * Me
        max_m_marlin = (Me + 15) // 16
        max_m_wgmma = (Me + 63) // 64

        # ---- bf16 golden (per-expert block) ----
        ref = torch.empty(M, H, dtype=torch.bfloat16, device=dev)
        for e in range(E):
            x = A[e * Me:(e + 1) * Me]
            g = x @ gate_w[e].t()
            u = x @ up_w[e].t()
            inter = torch.nn.functional.silu(g) * u
            ref[e * Me:(e + 1) * Me] = inter @ down_w[e].t()

        # ---- WGMMA ----
        wg_us = wg_diff = float('nan')
        try:
            def wgmma():
                itm = wgmma_mod.grouped_int4_moe_stage1(
                    A, expert_counts, w_gate_p, w_gate_s, w_up_p, w_up_s,
                    empty_bias, empty_bias, N, stride_wn_gate, stride_sn_gate, max_m_wgmma, Me)
                return wgmma_mod.grouped_int4_moe_stage2(
                    itm, expert_counts, w_down_p, w_down_s, empty_bias,
                    H, stride_wn_down, stride_sn_down, max_m_wgmma, Me)
            out = wgmma()
            wg_diff = _calc_diff(out, ref)
            wg_us = _time(wgmma)
        except Exception as ex:
            print(f"  [wgmma M={Me}] {ex}")

        # ---- BatchGen marlin ----
        bm_us = bm_diff = float('nan')
        try:
            gate_buf = torch.zeros(M, N, dtype=torch.bfloat16, device=dev)
            up_buf = torch.zeros(M, N, dtype=torch.bfloat16, device=dev)
            inter_m = torch.zeros(M, N, dtype=torch.bfloat16, device=dev)
            out_m = torch.zeros(M, H, dtype=torch.bfloat16, device=dev)
            bpr_N, bpr_K = N * 2, H * 2
            m_s1_C = torch.tensor(
                [gate_buf.data_ptr() + e * Me * bpr_N for e in range(E)]
                + [up_buf.data_ptr() + e * Me * bpr_N for e in range(E)],
                dtype=torch.int64, device=dev)
            m_s3_C = torch.tensor(
                [out_m.data_ptr() + e * Me * bpr_K for e in range(E)],
                dtype=torch.int64, device=dev)
            m_s1_ws = torch.zeros(2 * E * (n_tiles_s1 + 17), dtype=torch.int32, device=dev)
            m_s3_ws = torch.zeros(E * (n_tiles_s3 + 17), dtype=torch.int32, device=dev)

            def bmarlin():
                marlin_mod.grouped_marlin_gemm_m16(
                    A, m_s1_B, m_s1_C, m_s1_S, expert_starts, expert_counts,
                    E, N, H, m_s1_ws, 2 * E, n_tiles_s1, max_m_marlin)
                marlin_mod.silu_mul(gate_buf, up_buf, inter_m)
                marlin_mod.grouped_marlin_gemm_m16(
                    inter_m, m_s3_B, m_s3_C, m_s3_S, expert_starts, expert_counts,
                    E, H, N, m_s3_ws, E, n_tiles_s3, max_m_marlin)
            bmarlin()
            bm_diff = _calc_diff(out_m, ref)
            bm_us = _time(bmarlin)
        except Exception as ex:
            print(f"  [bg-marlin M={Me}] {ex}")

        # ---- SGL fused_marlin_moe ----
        sg_us = sg_diff = float('nan')
        if have_sgl:
            try:
                topk_ids = (torch.arange(M, device=dev) // Me).view(M, 1).to(torch.int32)
                topk_w = torch.ones(M, 1, dtype=torch.bfloat16, device=dev)
                def sgl():
                    return fused_marlin_moe(
                        hidden_states=A, w1=w13, w2=w2, w1_scale=w13s, w2_scale=w2s,
                        gating_output=topk_w, topk_weights=topk_w, topk_ids=topk_ids,
                        global_num_experts=E, expert_map=None,
                        g_idx1=None, g_idx2=None, sort_indices1=None, sort_indices2=None,
                        w1_zeros=None, w2_zeros=None, workspace=sgl_workspace,
                        num_bits=4, is_k_full=True, inplace=False, routed_scaling_factor=None)
                out = sgl()
                sg_diff = _calc_diff(out, ref)
                sg_us = _time(sgl)
            except Exception as ex:
                print(f"  [sgl M={Me}] {ex}")

        def tf(us):
            return flops_per(Me) / (us * 1e-6) / 1e12 if us == us and us > 0 else float('nan')
        cands = {"WGMMA": wg_us, "BG-marlin": bm_us, "SGL-marlin": sg_us}
        valid = {k: v for k, v in cands.items() if v == v}
        best = min(valid, key=valid.get) if valid else "-"
        print(f"{Me:6d} | {wg_us:6.0f} {tf(wg_us):6.1f} {wg_diff:6.4f} | "
              f"{bm_us:6.0f} {tf(bm_us):6.1f} {bm_diff:6.4f} | "
              f"{sg_us:6.0f} {tf(sg_us):6.1f} {sg_diff:6.4f} | {best:>10}")

    print("\nTFLOP/s = 2·E·M_e·(2·H·N + N·H) / t   (E={E}); 64k-prefill op point ~1365 tok/expert."
          .format(E=E))


if __name__ == "__main__":
    main()
