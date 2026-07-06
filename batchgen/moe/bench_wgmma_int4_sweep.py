#!/usr/bin/env python3
"""BatchGen WGMMA INT4 grouped MoE saturation sweep — dense 37-point grid, E=128.

Same shapes / FLOP / uniform routing / dense M_e grid as bench_sgl_int4_sweep.py and
bench_sgl_marlin_sweep.py, so the WGMMA arm is directly comparable to SGL-triton / SGL-marlin.
Full MoE: grouped_int4_moe_stage1 (gate|up|silu) + stage2 (down). FLOP = 2*E*M_e*(2HN + NH).
Golden = per-expert dequant-weight MoE; diff isolates kernel error.

Run:
    CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=9.0a \
        python -m batchgen.moe.bench_wgmma_int4_sweep
"""

import torch

from batchgen.moe.bench_marlin_vs_wgmma import create_k25_int4_weights_raw
from batchgen.moe.fused_int4_wgmma_grouped import _load_int4_grouped_module

H = 7168
N_INTER = 2048
E = 128
GROUP_SIZE = 32
TOPK_RATIO = 48
ITERS = 30

M_VALUES = [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120,
            128, 144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 384, 448, 512,
            640, 768, 1024, 1536, 2048]


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


def _dequant(packed_i32, scales, rows, cols):
    q = torch.zeros(rows, cols, dtype=torch.float32, device=packed_i32.device)
    for i in range(8):
        q[:, i::8] = ((packed_i32 >> (i * 4)) & 0xF).float()
    scale_exp = scales.float().repeat_interleave(GROUP_SIZE, dim=1)
    return ((q - 8.0) * scale_exp).to(torch.bfloat16)


def main():
    dev = "cuda"
    print(f"BatchGen WGMMA INT4 grouped MoE sweep | device={torch.cuda.get_device_name()} "
          f"| iters={ITERS}")
    print(f"H={H}  N_inter={N_INTER}  E={E}  gs={GROUP_SIZE}  top_k=1 (uniform block-diagonal)")
    print(f"FLOP = 2*E*M_e*(2*H*N + N*H)\n")

    mod = _load_int4_grouped_module()

    print("Building INT4 weights + bf16 golden ...", flush=True)
    gate_raw, up_raw, down_raw = [], [], []
    gate_sc, up_sc, down_sc = [], [], []
    gate_w, up_w, down_w = [], [], []   # bf16 golden [out, in]
    for _e in range(E):
        gpu8, gpi32, gs = create_k25_int4_weights_raw(H, N_INTER, device=dev)   # gate [N,H]
        upu8, upi32, us = create_k25_int4_weights_raw(H, N_INTER, device=dev)   # up   [N,H]
        dpu8, dpi32, ds = create_k25_int4_weights_raw(N_INTER, H, device=dev)   # down [H,N]
        gate_raw.append(gpu8); up_raw.append(upu8); down_raw.append(dpu8)
        gate_sc.append(gs); up_sc.append(us); down_sc.append(ds)
        gate_w.append(_dequant(gpi32, gs, N_INTER, H))   # [N, H]
        up_w.append(_dequant(upi32, us, N_INTER, H))     # [N, H]
        down_w.append(_dequant(dpi32, ds, H, N_INTER))   # [H, N]
    empty_bias = torch.empty(0, dtype=torch.int64, device=dev)
    wgp = torch.tensor([w.data_ptr() for w in gate_raw], dtype=torch.int64, device=dev)
    wgs = torch.tensor([s.data_ptr() for s in gate_sc], dtype=torch.int64, device=dev)
    wup = torch.tensor([w.data_ptr() for w in up_raw], dtype=torch.int64, device=dev)
    wus = torch.tensor([s.data_ptr() for s in up_sc], dtype=torch.int64, device=dev)
    wdp = torch.tensor([w.data_ptr() for w in down_raw], dtype=torch.int64, device=dev)
    wds = torch.tensor([s.data_ptr() for s in down_sc], dtype=torch.int64, device=dev)
    swn_g, ssn_g = H // 2, H // 32
    swn_d, ssn_d = N_INTER // 2, N_INTER // 32
    print("  done\n")

    hdr = f"{'M_e':>5} | {'B_glob':>7} | {'WGMMA (grouped int4):  us     TF/s     diff':>44}"
    print(hdr)
    print("-" * len(hdr))

    tf_hist = {}
    for Me in M_VALUES:
        M = E * Me
        B = TOPK_RATIO * Me
        A = torch.randn((M, H), dtype=torch.bfloat16, device=dev) * 0.1
        cnt = torch.full((E,), Me, dtype=torch.int32, device=dev)
        mmt = (Me + 63) // 64

        wg_us = wg_diff = float('nan')
        try:
            def wgmma():
                itm = mod.grouped_int4_moe_stage1(
                    A, cnt, wgp, wgs, wup, wus, empty_bias, empty_bias,
                    N_INTER, swn_g, ssn_g, mmt, Me)
                return mod.grouped_int4_moe_stage2(
                    itm, cnt, wdp, wds, empty_bias, H, swn_d, ssn_d, mmt, Me)
            out = wgmma()
            ref = torch.empty(M, H, dtype=torch.bfloat16, device=dev)
            for e in range(E):
                x = A[e * Me:(e + 1) * Me]
                inter = torch.nn.functional.silu(x @ gate_w[e].t()) * (x @ up_w[e].t())
                ref[e * Me:(e + 1) * Me] = (inter @ down_w[e].t()).to(torch.bfloat16)
            wg_diff = _calc_diff(out, ref)
            wg_us = _time(wgmma)
            del ref
        except Exception as ex:
            print(f"  [wgmma Me={Me}] {type(ex).__name__}: {ex}")

        flop = 2.0 * E * Me * (2 * H * N_INTER + N_INTER * H)
        wg_tf = flop / (wg_us * 1e-6) / 1e12 if wg_us == wg_us and wg_us > 0 else float('nan')
        if wg_tf == wg_tf:
            tf_hist[Me] = wg_tf
        print(f"{Me:5d} | {B:7d} | {wg_us:14.1f} {wg_tf:8.1f} {wg_diff:10.5f}")
        del A
        torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("SATURATION KNEE (smallest M_e reaching >=95% of own max TF/s)")
    print("=" * 60)
    if tf_hist:
        mx = max(tf_hist.values())
        knee = min(m for m in sorted(tf_hist) if tf_hist[m] >= 0.95 * mx)
        print(f"  WGMMA: max {mx:.1f} TF/s  knee M_e={knee}  B_global={TOPK_RATIO*knee}")
    print("FLOP=2*E*M_e*(2*H*N+N*H); vs BG-marlin ~84 / SGL-marlin 81.3 / SGL-triton 68.8 (untuned).")


if __name__ == "__main__":
    main()
