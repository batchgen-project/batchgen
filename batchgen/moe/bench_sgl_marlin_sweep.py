#!/usr/bin/env python3
"""SGLang fused_marlin_moe (INT4 fused-MMA) MoE saturation sweep — the KIMI PRODUCTION path.

BatchGen's Kimi-K2.5 TP-MoE decode calls `fused_marlin_moe` (model.py:1267), NOT the triton
dequant path. This benches it across an M_e sweep at the Kimi INT4 shapes, with the SAME FLOP
formula as the WGMMA / BG-marlin / SGL-triton arms so all four are directly comparable.

Weight build is the PRODUCTION-VALIDATED recipe from test/tp_marlin_moe_parity.py L1:
per expert, marlinize gate & up SEPARATELY via raw_to_marlin_fused_gpu then concat marlin cols
(marlin(concat) != concat(marlin)); down marlinized directly. This is the fix for the earlier
Kimi-sweep SGL-marlin arm that broke (diff 0.5) on a synthetic Python repack.

Uniform block-diagonal routing: token t -> expert t//M_e, top_k=1, weight 1.0.
FLOP = 2*E*M_e*(2*H*N + N*H).  Golden = per-expert dequant-weight MoE (isolates kernel error).

Run:
    CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=9.0a \
        python -m batchgen.moe.bench_sgl_marlin_sweep
"""

import torch
import torch.nn.functional as F

from batchgen.moe.marlin_weight_prep import _dequantize_k25_int4
from batchgen.moe.marlin_transform import raw_to_marlin_fused_gpu

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


def build_raw_int4(N, K, device, seed):
    """Raw K2.5 INT4 projection: raw_packed [N, K//8] int32 (8 nibbles/int32), scale [N,K//32] bf16."""
    g = torch.Generator(device=device).manual_seed(seed)
    q = torch.randint(0, 16, (N, K), dtype=torch.int32, device=device, generator=g)
    raw = torch.zeros(N, K // 8, dtype=torch.int32, device=device)
    for i in range(8):
        raw |= (q[:, i::8] & 0xF) << (i * 4)
    scale = (torch.rand(N, K // 32, device=device, generator=g) * 0.05 + 0.01).to(torch.bfloat16)
    return raw, scale


def main():
    dev = "cuda"
    print(f"SGLang fused_marlin_moe (INT4 fused-MMA) MoE sweep | device={torch.cuda.get_device_name()} "
          f"| iters={ITERS}")
    print(f"H={H}  N_inter={N_INTER}  E={E}  gs={GROUP_SIZE}  top_k=1 (uniform block-diagonal) | KIMI PATH")
    print(f"FLOP = 2*E*M_e*(2*H*N + N*H)\n")

    from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import fused_marlin_moe
    from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace
    from sglang.srt.server_args import (
        ServerArgs, get_global_server_args, set_global_server_args_for_scheduler)
    try:
        get_global_server_args()
    except Exception:
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
    workspace = marlin_make_workspace(torch.device(dev), max_blocks_per_sm=4)

    print("Building marlin weights + bf16 golden (raw_to_marlin_fused_gpu, production recipe) ...",
          flush=True)
    w13 = torch.empty(E, H // 16, 4 * N_INTER, dtype=torch.int32, device=dev)
    w13_s = torch.empty(E, H // GROUP_SIZE, 2 * N_INTER, dtype=torch.bfloat16, device=dev)
    w2 = torch.empty(E, N_INTER // 16, 2 * H, dtype=torch.int32, device=dev)
    w2_s = torch.empty(E, N_INTER // GROUP_SIZE, H, dtype=torch.bfloat16, device=dev)
    Wg_l, Wu_l, Wd_l = [], [], []   # bf16 golden weights: Wg/Wu [H, N_INTER], Wd [N_INTER, H]
    for e in range(E):
        gate_raw, gate_s = build_raw_int4(N_INTER, H, dev, 1000 + e * 3 + 0)
        up_raw, up_s = build_raw_int4(N_INTER, H, dev, 1000 + e * 3 + 1)
        down_raw, down_s = build_raw_int4(H, N_INTER, dev, 1000 + e * 3 + 2)
        g_mw, g_ms = raw_to_marlin_fused_gpu(gate_raw, gate_s, H, N_INTER)
        u_mw, u_ms = raw_to_marlin_fused_gpu(up_raw, up_s, H, N_INTER)
        w13[e] = torch.cat([g_mw, u_mw], dim=1)
        w13_s[e] = torch.cat([g_ms, u_ms], dim=1)
        w2[e], w2_s[e] = raw_to_marlin_fused_gpu(down_raw, down_s, N_INTER, H)
        Wg_l.append(_dequantize_k25_int4(gate_raw, gate_s, H, N_INTER).to(torch.bfloat16))  # [H,N]
        Wu_l.append(_dequantize_k25_int4(up_raw, up_s, H, N_INTER).to(torch.bfloat16))
        Wd_l.append(_dequantize_k25_int4(down_raw, down_s, N_INTER, H).to(torch.bfloat16))  # [N,H]
    print("  done\n")

    hdr = f"{'M_e':>5} | {'B_glob':>7} | {'SGL-marlin (fused_marlin_moe):  us     TF/s     diff':>52}"
    print(hdr)
    print("-" * len(hdr))

    tf_hist = {}
    for Me in M_VALUES:
        M = E * Me
        B = TOPK_RATIO * Me
        hidden = (torch.randn((M, H), dtype=torch.bfloat16, device=dev) * 0.1)
        topk_ids = (torch.arange(M, device=dev) // Me).view(M, 1).to(torch.int32)
        topk_w = torch.ones((M, 1), dtype=torch.float32, device=dev)

        sg_us = sg_diff = float('nan')
        try:
            def sgl():
                return fused_marlin_moe(
                    hidden_states=hidden, w1=w13, w2=w2, w1_scale=w13_s, w2_scale=w2_s,
                    gating_output=topk_w, topk_weights=topk_w, topk_ids=topk_ids,
                    global_num_experts=E, expert_map=None,
                    g_idx1=None, g_idx2=None, sort_indices1=None, sort_indices2=None,
                    w1_zeros=None, w2_zeros=None, workspace=workspace,
                    num_bits=4, is_k_full=True, inplace=False, routed_scaling_factor=None)
            out = sgl()
            ref = torch.empty(M, H, dtype=torch.bfloat16, device=dev)
            for e in range(E):
                x = hidden[e * Me:(e + 1) * Me]
                inter = F.silu(x @ Wg_l[e]) * (x @ Wu_l[e])   # [Me, N_INTER]
                ref[e * Me:(e + 1) * Me] = (inter @ Wd_l[e]).to(torch.bfloat16)
            sg_diff = _calc_diff(out, ref)
            sg_us = _time(sgl)
            del ref
        except Exception as ex:
            print(f"  [marlin Me={Me}] {type(ex).__name__}: {ex}")

        flop = 2.0 * E * Me * (2 * H * N_INTER + N_INTER * H)
        sg_tf = flop / (sg_us * 1e-6) / 1e12 if sg_us == sg_us and sg_us > 0 else float('nan')
        if sg_tf == sg_tf:
            tf_hist[Me] = sg_tf
        print(f"{Me:5d} | {B:7d} | {sg_us:14.1f} {sg_tf:8.1f} {sg_diff:10.5f}")
        del hidden, topk_ids, topk_w
        torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("SATURATION KNEE (smallest M_e reaching >=95% of own max TF/s)")
    print("=" * 60)
    if tf_hist:
        mx = max(tf_hist.values())
        knee = min(m for m in sorted(tf_hist) if tf_hist[m] >= 0.95 * mx)
        print(f"  SGL-marlin: max {mx:.1f} TF/s  knee M_e={knee}  B_global={TOPK_RATIO*knee}")
    print("FLOP=2*E*M_e*(2*H*N+N*H); compare WGMMA ~110 / BG-marlin ~84 / SGL-triton 68.8 (untuned).")


if __name__ == "__main__":
    main()
