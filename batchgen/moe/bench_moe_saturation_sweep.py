#!/usr/bin/env python3
"""Saturation sweep — "how many tokens/expert saturate the device per grouped-GEMM launch".

Sweeps M_e (tokens per expert) and, for each of three INT4 W4A16 MoE kernels, reports
us + TF/s + calc_diff (vs a bf16-dequant golden, tol RTOL=1.6e-2) and the per-arm KNEE
(smallest M_e whose TF/s reaches >=95% of that arm's own max TF/s). Each arm is driven at
its PRODUCTION launch shape for Kimi-K2.5 on 16-rank parallelism:

  arm            kernel                         prod role              per-expert GEMM shape
  ---------------------------------------------------------------------------------------
  WGMMA          grouped_int4_moe_stage1/2      PREFILL / EP-fallback  E x N=2048 (wide)   BLOCK_M=64
  BG-marlin      grouped_marlin_gemm_m16        EP16 DECODE (default)  E x N=2048 (wide)   MBLOCK=16
  SGL-marlin     fused_marlin_moe (wna16)       TP16 DECODE (TP-MoE)   E x N=128  (narrow) adaptive m

WGMMA and BG-marlin are the SAME MoE math (silu(gate.x)*(up.x) -> down) at EP width N=2048.
SGL-marlin runs the TP slice N/16=128 over all experts (this is why its knee is far lower:
SGL fused_marlin_moe adaptively picks block_size_m in [8,16,32,48,64]).

EP<->TP token identity (top_k=8, 384 experts, 16 ranks, B = global concurrent decode tokens):
  M_e(EP) = 8B/384 = B/48   (rank owns 24 experts, each sees 8B/384 tokens)
  M_e(TP) = 8B/384 = B/48   (rank holds N/16 slice of ALL 384 experts, each sees 8B/384)
=> tokens-to-saturate B = 48 * knee_M_e for BOTH launch shapes (the algebra collapses; the
   real EP-vs-TP lever is the GEMM width 2048 vs 128, not the token count).

CUDA-Events timing only. Drop-in:
    CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=9.0a \
        python -m batchgen.moe.bench_moe_saturation_sweep
"""

import torch

from batchgen.moe.bench_marlin_vs_wgmma import create_k25_int4_weights_raw
from batchgen.moe.marlin_weight_prep import (
    repack_int4_to_marlin_gs32, get_weight_perm)

H = 7168          # hidden / K
N_EP = 2048       # EP full expert-intermediate width  (WGMMA + BG-marlin)
N_TP = 128        # TP slice width  = N_EP / 16          (SGL-marlin)
E = 128           # experts in the bench (grid occupancy; per-expert tput is E-independent)
GROUP_SIZE = 32
M_VALUES = [1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256, 512]
TOPK_RATIO = 48   # M_e = B / 48  ->  B = 48 * M_e  (tokens-to-saturate, both shapes)
ITERS = 30


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _dequant(packed_i32, scales, rows, cols):
    """INT4 packed [rows, cols//8] i32 + scales [rows, cols//32] bf16 -> bf16 [rows, cols]."""
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
    return ((out.float() - ref.float()).abs().mean()
            / ref.float().abs().mean().clamp(min=1e-9)).item()


def _nibbles_from_i32(packed_i32, rows_out, cols_in):
    """[rows_out, cols_in//8] i32 -> [rows_out, cols_in] i32 nibbles (0..15)."""
    nib = torch.empty(rows_out, cols_in, dtype=torch.int32, device=packed_i32.device)
    for i in range(8):
        nib[:, i::8] = (packed_i32 >> (i * 4)) & 0xF
    return nib


def _gptq_pack_kn(q_kn):
    """[K, N] nibbles 0..15 -> GPTQ-packed [K//8, N] i32 (8 consecutive K per int32)."""
    K, Ncol = q_kn.shape
    q = q_kn.reshape(K // 8, 8, Ncol)
    out = torch.zeros(K // 8, Ncol, dtype=torch.int32, device=q_kn.device)
    for i in range(8):
        out |= (q[:, i, :] & 0xF) << (4 * i)
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    dev = "cuda"
    print(f"MoE saturation sweep | device={torch.cuda.get_device_name()} | iters={ITERS}")
    print(f"H={H}  N_EP={N_EP}  N_TP={N_TP}  E={E}  gs={GROUP_SIZE}  (top_k=1, 1 expert/token)\n")

    from batchgen.moe.marlin_grouped_moe import _load_module as load_marlin
    from batchgen.moe.fused_int4_wgmma_grouped import _load_int4_grouped_module
    marlin_mod = load_marlin()
    wgmma_mod = _load_int4_grouped_module()

    # -------------------------------------------------------------------
    # SGL fused_marlin_moe: import + build TP-narrow (N=128) weights via
    # SGLang's OWN gptq_marlin_repack layout (the RUN_SGL blocker fix).
    # -------------------------------------------------------------------
    have_sgl = False
    sgl_err = ""
    try:
        from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import fused_marlin_moe
        from sglang.srt.layers.quantization.marlin_utils import (
            marlin_make_workspace, marlin_moe_permute_scales)
        try:
            from sglang.srt.layers.quantization.gptq import gptq_marlin_moe_repack
        except Exception:
            from sglang.srt.layers.quantization.marlin_utils import gptq_marlin_moe_repack
        from sglang.srt.server_args import (
            ServerArgs, get_global_server_args, set_global_server_args_for_scheduler)
        try:
            get_global_server_args()
        except Exception:
            set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
        sgl_workspace = marlin_make_workspace(torch.device(dev), max_blocks_per_sm=4)
        have_sgl = True
    except Exception as ex:
        sgl_err = f"import failed: {ex}"

    # -------------------------------------------------------------------
    # Build EP (N=2048) weights: WGMMA raw + BG-marlin i32 + bf16 golden.
    # (verbatim structure from bench_prefill_kernel_ab.py)
    # -------------------------------------------------------------------
    print("Building EP weights (raw + marlin + bf16 golden) ...", flush=True)
    gate_raw, up_raw, down_raw = [], [], []
    gate_sc, up_sc, down_sc = [], [], []
    gate_mqw, gate_ms, up_mqw, up_ms, down_mqw, down_ms = [], [], [], [], [], []
    gate_w, up_w, down_w = [], [], []          # bf16 golden [out, in]

    for _e in range(E):
        gpu8, gpi32, gs = create_k25_int4_weights_raw(H, N_EP, device=dev)   # gate [N_EP,H]
        upu8, upi32, us = create_k25_int4_weights_raw(H, N_EP, device=dev)   # up   [N_EP,H]
        dpu8, dpi32, ds = create_k25_int4_weights_raw(N_EP, H, device=dev)   # down [H,N_EP]
        gate_raw.append(gpu8); up_raw.append(upu8); down_raw.append(dpu8)
        gate_sc.append(gs); up_sc.append(us); down_sc.append(ds)

        gmw, gms = repack_int4_to_marlin_gs32(gpi32, gs, H, N_EP)
        umw, ums = repack_int4_to_marlin_gs32(upi32, us, H, N_EP)
        dmw, dms = repack_int4_to_marlin_gs32(dpi32, ds, N_EP, H)
        gate_mqw.append(gmw); gate_ms.append(gms)
        up_mqw.append(umw); up_ms.append(ums)
        down_mqw.append(dmw); down_ms.append(dms)

        gate_w.append(_dequant(gpi32, gs, N_EP, H))
        up_w.append(_dequant(upi32, us, N_EP, H))
        down_w.append(_dequant(dpi32, ds, H, N_EP))
    print("  EP done")

    # EP pointer arrays (WGMMA + BG-marlin) — static.
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
    n_tiles_s1 = N_EP // 256
    n_tiles_s3 = H // 256
    stride_wn_gate, stride_sn_gate = H // 2, H // 32
    stride_wn_down, stride_sn_down = N_EP // 2, N_EP // 32

    # -------------------------------------------------------------------
    # Build TP (N=128) weights for SGL fused_marlin_moe via SGLang repack.
    # -------------------------------------------------------------------
    tp_gate_w, tp_up_w, tp_down_w = [], [], []   # bf16 golden [out,in]
    if have_sgl:
        try:
            # per-expert GPTQ-packed nibbles + raw scales, then batch repack.
            w1_gptq_l, w2_gptq_l = [], []
            w1_s_l, w2_s_l = [], []
            for _e in range(E):
                gpu8, gpi32, gs = create_k25_int4_weights_raw(H, N_TP, device=dev)  # gate [N_TP,H]
                upu8, upi32, us = create_k25_int4_weights_raw(H, N_TP, device=dev)  # up   [N_TP,H]
                dpu8, dpi32, ds = create_k25_int4_weights_raw(N_TP, H, device=dev)  # down [H,N_TP]
                tp_gate_w.append(_dequant(gpi32, gs, N_TP, H))   # [N_TP,H]
                tp_up_w.append(_dequant(upi32, us, N_TP, H))     # [N_TP,H]
                tp_down_w.append(_dequant(dpi32, ds, H, N_TP))   # [H,N_TP]

                # nibbles [out,in] -> [in=K, out=N] -> GPTQ pack [K//8, N]
                g_kn = _nibbles_from_i32(gpi32, N_TP, H).t().contiguous()   # [H, N_TP]
                u_kn = _nibbles_from_i32(upi32, N_TP, H).t().contiguous()   # [H, N_TP]
                d_kn = _nibbles_from_i32(dpi32, H, N_TP).t().contiguous()   # [N_TP, H]
                w1_kn = torch.cat([g_kn, u_kn], dim=1)                      # [H, 2*N_TP]
                w1_gptq_l.append(_gptq_pack_kn(w1_kn))                      # [H//8, 2N_TP]
                w2_gptq_l.append(_gptq_pack_kn(d_kn))                       # [N_TP//8, H]

                # scales [out, in//gs] -> [in//gs, out]; concat gate|up along out.
                w1_s = torch.cat([gs.t().contiguous(), us.t().contiguous()], dim=1)  # [H//gs, 2N_TP]
                w1_s_l.append(w1_s.to(torch.bfloat16))
                w2_s_l.append(ds.t().contiguous().to(torch.bfloat16))               # [N_TP//gs, H]

            w1_gptq = torch.stack(w1_gptq_l, 0)   # [E, H//8, 2N_TP]
            w2_gptq = torch.stack(w2_gptq_l, 0)   # [E, N_TP//8, H]
            perm1 = torch.empty((E, 0), dtype=torch.int32, device=dev)
            perm2 = torch.empty((E, 0), dtype=torch.int32, device=dev)
            w13 = gptq_marlin_moe_repack(w1_gptq, perm1, size_k=H, size_n=2 * N_TP, num_bits=4)
            w2 = gptq_marlin_moe_repack(w2_gptq, perm2, size_k=N_TP, size_n=H, num_bits=4)
            w13s = marlin_moe_permute_scales(
                torch.stack(w1_s_l, 0), size_k=H, size_n=2 * N_TP, group_size=GROUP_SIZE)
            w2s = marlin_moe_permute_scales(
                torch.stack(w2_s_l, 0), size_k=N_TP, size_n=H, group_size=GROUP_SIZE)
            print("  TP/SGL weights built via gptq_marlin_moe_repack")
        except Exception as ex:
            have_sgl = False
            sgl_err = f"weight build failed: {ex}"

    if not have_sgl:
        print(f"[warn] SGL arm DISABLED — {sgl_err}")
    print()

    # -------------------------------------------------------------------
    # header
    # -------------------------------------------------------------------
    hdr = (f"{'M/exp':>6} | {'B':>7} | {'WGMMA(EP N=2048)':>22} | "
           f"{'BG-marlin(EP N=2048)':>22} | {'SGL-marlin(TP N=128)':>22}")
    print(hdr)
    print(f"{'':>6} | {'=48Me':>7} | {'us   TF/s   diff':>22} | "
          f"{'us   TF/s   diff':>22} | {'us   TF/s   diff':>22}")
    print("-" * len(hdr))

    flops_ep = lambda Me: 2.0 * E * Me * (2 * H * N_EP + N_EP * H)
    flops_tp = lambda Me: 2.0 * E * Me * (2 * H * N_TP + N_TP * H)

    tf_hist = {"WGMMA": {}, "BG-marlin": {}, "SGL-marlin": {}}

    for Me in M_VALUES:
        M = E * Me
        B = TOPK_RATIO * Me
        A = torch.randn((M, H), dtype=torch.bfloat16, device=dev) * 0.1
        expert_counts = torch.full((E,), Me, dtype=torch.int32, device=dev)
        expert_starts = torch.arange(E, dtype=torch.int32, device=dev) * Me
        max_m_marlin = (Me + 15) // 16
        max_m_wgmma = (Me + 63) // 64

        # ---- EP bf16 golden (shared WGMMA + BG-marlin) ----
        ref_ep = torch.empty(M, H, dtype=torch.bfloat16, device=dev)
        for e in range(E):
            x = A[e * Me:(e + 1) * Me]
            inter = torch.nn.functional.silu(x @ gate_w[e].t()) * (x @ up_w[e].t())
            ref_ep[e * Me:(e + 1) * Me] = inter @ down_w[e].t()

        # ---- WGMMA (EP) ----
        wg_us = wg_diff = float('nan')
        try:
            def wgmma():
                itm = wgmma_mod.grouped_int4_moe_stage1(
                    A, expert_counts, w_gate_p, w_gate_s, w_up_p, w_up_s,
                    empty_bias, empty_bias, N_EP, stride_wn_gate, stride_sn_gate,
                    max_m_wgmma, Me)
                return wgmma_mod.grouped_int4_moe_stage2(
                    itm, expert_counts, w_down_p, w_down_s, empty_bias,
                    H, stride_wn_down, stride_sn_down, max_m_wgmma, Me)
            wg_diff = _calc_diff(wgmma(), ref_ep)
            wg_us = _time(wgmma)
        except Exception as ex:
            print(f"  [wgmma Me={Me}] {ex}")

        # ---- BG-marlin (EP) ----
        bm_us = bm_diff = float('nan')
        try:
            gate_buf = torch.zeros(M, N_EP, dtype=torch.bfloat16, device=dev)
            up_buf = torch.zeros(M, N_EP, dtype=torch.bfloat16, device=dev)
            inter_m = torch.zeros(M, N_EP, dtype=torch.bfloat16, device=dev)
            out_m = torch.zeros(M, H, dtype=torch.bfloat16, device=dev)
            bpr_N, bpr_K = N_EP * 2, H * 2
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
                    E, N_EP, H, m_s1_ws, 2 * E, n_tiles_s1, max_m_marlin)
                marlin_mod.silu_mul(gate_buf, up_buf, inter_m)
                marlin_mod.grouped_marlin_gemm_m16(
                    inter_m, m_s3_B, m_s3_C, m_s3_S, expert_starts, expert_counts,
                    E, H, N_EP, m_s3_ws, E, n_tiles_s3, max_m_marlin)
            bmarlin()
            bm_diff = _calc_diff(out_m, ref_ep)
            bm_us = _time(bmarlin)
        except Exception as ex:
            print(f"  [bg-marlin Me={Me}] {ex}")

        # ---- SGL-marlin (TP narrow N=128) ----
        sg_us = sg_diff = float('nan')
        if have_sgl:
            try:
                ref_tp = torch.empty(M, H, dtype=torch.bfloat16, device=dev)
                for e in range(E):
                    x = A[e * Me:(e + 1) * Me]
                    inter = torch.nn.functional.silu(x @ tp_gate_w[e].t()) * (x @ tp_up_w[e].t())
                    ref_tp[e * Me:(e + 1) * Me] = inter @ tp_down_w[e].t()
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
                sg_diff = _calc_diff(sgl(), ref_tp)
                sg_us = _time(sgl)
                del ref_tp
            except Exception as ex:
                print(f"  [sgl Me={Me}] {ex}")

        def tf(us, ep):
            fp = flops_ep(Me) if ep else flops_tp(Me)
            return fp / (us * 1e-6) / 1e12 if us == us and us > 0 else float('nan')
        wg_tf, bm_tf, sg_tf = tf(wg_us, True), tf(bm_us, True), tf(sg_us, False)
        if wg_tf == wg_tf: tf_hist["WGMMA"][Me] = wg_tf
        if bm_tf == bm_tf: tf_hist["BG-marlin"][Me] = bm_tf
        if sg_tf == sg_tf: tf_hist["SGL-marlin"][Me] = sg_tf

        print(f"{Me:6d} | {B:7d} | {wg_us:6.0f} {wg_tf:6.1f} {wg_diff:7.4f} | "
              f"{bm_us:6.0f} {bm_tf:6.1f} {bm_diff:7.4f} | "
              f"{sg_us:6.0f} {sg_tf:6.1f} {sg_diff:7.4f}")
        del A, ref_ep
        torch.cuda.empty_cache()

    # -------------------------------------------------------------------
    # knee analysis
    # -------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("SATURATION KNEE (smallest M_e reaching >=95% of that arm's own max TF/s)")
    print("=" * 78)
    print(f"{'arm':>12} | {'max TF/s':>9} | {'knee M_e':>9} | "
          f"{'B=48*M_e':>9} | tokens-to-saturate (EP 24x2048  &  TP 384x128)")
    print("-" * 92)
    for arm, hist in tf_hist.items():
        if not hist:
            print(f"{arm:>12} | {'n/a':>9} | disabled")
            continue
        mx = max(hist.values())
        knee = min(m for m in sorted(hist) if hist[m] >= 0.95 * mx)
        Bsat = TOPK_RATIO * knee
        print(f"{arm:>12} | {mx:9.1f} | {knee:9d} | {Bsat:9d} | "
              f"EP: B~{Bsat} global tok  |  TP: B~{Bsat} global tok  (M_e=B/48 both)")
    print("\nNote: M_e=B/48 for EP and TP alike (8B/384) -> tokens-to-saturate is the same B;")
    print("the EP-vs-TP lever is GEMM width 2048(EP) vs 128(TP), not token count.")
    print("TF/s(EP) uses N=2048; TF/s(TP) uses N=128 (per-launch FLOP; 384*128 == 24*2048).")


if __name__ == "__main__":
    main()
