#!/usr/bin/env python3
"""One-shot diagnostic: BatchGen raw_to_marlin vs SGLang canonical marlin.

The TP-marlin parity test fails (L1 max_abs_diff=1.5) with byte-identical SCALE
perms, so the WEIGHT marlin layout from BatchGen's raw_to_marlin_fused_gpu differs
from what SGLang's fused_marlin_moe kernel reads. This script pins the exact
difference in ONE run by comparing, for the SAME quantized weight:

  (1) get_weight_perm arrays           BatchGen vs SGLang
  (2) scale perm arrays                BatchGen vs SGLang
  (3) dequant round-trip               BatchGen raw == SGLang w_ref ?
  (4) packed marlin WEIGHT bytes       BatchGen vs SGLang  (+ permutation analysis)
  (5) marlin SCALE bytes               BatchGen vs SGLang
  (6) functional GEMM/MoE              SGLang-marlin (control) vs BatchGen-marlin vs ref

SGLang's marlin_quantize is pure-Python (get_weight_perm + marlin_weights +
marlin_permute_scales) and is the reference the kernel is tested against, so a
byte diff here IS the bug. Run on a GPU host:  python test/marlin_layout_diag.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

# Small but marlin-tile-valid dims (K%16, N%64, group 32).
H = 256       # "hidden" / contraction K for gate-up
N_INTER = 128  # "intermediate" / output N
GS = 32


def banner(t):
    print(f"\n{'='*70}\n{t}\n{'='*70}")


def cmp_arr(name, a, b):
    a = np.asarray(a); b = np.asarray(b)
    eq = a.shape == b.shape and np.array_equal(a, b)
    print(f"  [{'EQUAL' if eq else 'DIFF '}] {name}: shapes {a.shape} vs {b.shape}", end="")
    if not eq and a.shape == b.shape:
        nd = int((a != b).sum())
        print(f"  ({nd}/{a.size} differ; a[:8]={a.ravel()[:8].tolist()} b[:8]={b.ravel()[:8].tolist()})")
    else:
        print()
    return eq


def cmp_t(name, a, b):
    eq = a.shape == b.shape and torch.equal(a, b)
    extra = ""
    if not eq and a.shape == b.shape:
        diff = (a != b)
        extra = f"  ({int(diff.sum())}/{a.numel()} differ)"
        # is b a permutation of a along the last dim (per-row reorder)?
        try:
            sa = torch.sort(a.reshape(-1, a.shape[-1]), dim=-1).values
            sb = torch.sort(b.reshape(-1, b.shape[-1]), dim=-1).values
            if torch.equal(sa, sb):
                extra += " [same multiset per row → a COLUMN PERMUTATION]"
            elif torch.equal(torch.sort(a.flatten()).values, torch.sort(b.flatten()).values):
                extra += " [same global multiset → a GLOBAL PERMUTATION]"
            else:
                extra += " [different values → NOT a pure permutation]"
        except Exception as e:
            extra += f" [perm-check err: {e}]"
    print(f"  [{'EQUAL' if eq else 'DIFF '}] {name}: {tuple(a.shape)} vs {tuple(b.shape)}{extra}")
    return eq


def build_bg_raw_from_codes(q_w, s):
    """q_w [K,N] uint4 codes (0..15, bias-8), s [K//GS, N] -> BatchGen raw [N,K//8] int32, [N,K//GS] bf16."""
    K, N = q_w.shape
    codes = q_w.to(torch.int32).t().contiguous().view(N, K // 8, 8)  # [N, K//8, 8]
    raw = torch.zeros(N, K // 8, dtype=torch.int32, device=q_w.device)
    for i in range(8):
        raw |= (codes[:, :, i] & 0xF) << (i * 4)
    raw_s = s.t().contiguous().to(torch.bfloat16)  # [N, K//GS]
    return raw, raw_s


def quant_one(W, qtype, sgl):
    """Quantize W[K,N] both ways from the SAME codes. Returns dict."""
    gptq_quantize_weights = sgl["gptq_quantize_weights"]
    marlin_weights = sgl["marlin_weights"]
    marlin_permute_scales = sgl["marlin_permute_scales"]
    sgl_get_weight_perm = sgl["get_weight_perm"]
    from batchgen.moe.marlin_transform import raw_to_marlin_fused_gpu
    from batchgen.moe.marlin_weight_prep import _dequantize_k25_int4

    K, N = W.shape
    w_ref, q_w, s, g_idx, rand_perm = gptq_quantize_weights(W, qtype, GS, False, None)
    # SGLang canonical marlin (Python, kernel-correct)
    wp = sgl_get_weight_perm(4)
    sgl_mw = marlin_weights(q_w, K, N, 4, wp)
    sgl_ms = marlin_permute_scales(s, K, N, GS)
    # BatchGen marlin from the same codes
    bg_raw, bg_raw_s = build_bg_raw_from_codes(q_w, s)
    bg_deq = _dequantize_k25_int4(bg_raw, bg_raw_s, K, N).float()  # [K,N]
    bg_mw, bg_ms = raw_to_marlin_fused_gpu(bg_raw, bg_raw_s, K, N)
    return dict(w_ref=w_ref.to(W.device).float(), q_w=q_w, s=s,
                sgl_mw=sgl_mw.to(W.device), sgl_ms=sgl_ms.to(W.device),
                bg_mw=bg_mw, bg_ms=bg_ms, bg_deq=bg_deq)


def main():
    if not torch.cuda.is_available():
        print("SKIP: CUDA required.")
        return 0
    dev = torch.device("cuda")
    torch.manual_seed(0)

    # ---- imports ----
    from sglang.test.test_marlin_utils import get_weight_perm as sgl_gwp, marlin_weights
    from sglang.srt.layers.quantization.utils import gptq_quantize_weights
    from sglang.srt.layers.quantization.marlin_utils import marlin_permute_scales
    from sgl_kernel.scalar_type import scalar_types
    from batchgen.moe.marlin_weight_prep import get_weight_perm as bg_gwp, _get_scale_perms
    qtype = scalar_types.uint4b8
    sgl = dict(gptq_quantize_weights=gptq_quantize_weights, marlin_weights=marlin_weights,
               marlin_permute_scales=marlin_permute_scales, get_weight_perm=sgl_gwp)

    banner("(1) get_weight_perm arrays  BatchGen vs SGLang (num_bits=4)")
    cmp_arr("weight_perm", bg_gwp(4), sgl_gwp(4))

    banner("(2) scale perm arrays  BatchGen vs SGLang")
    bsp, bsps = _get_scale_perms()
    from sglang.srt.layers.quantization.marlin_utils import get_scale_perms as sgl_gsp
    ssp, ssps = sgl_gsp()
    cmp_arr("scale_perm", bsp, ssp)
    cmp_arr("scale_perm_single", bsps, ssps)

    W = (torch.randn(H, N_INTER, device=dev, dtype=torch.float16) * 0.1)
    r = quant_one(W, qtype, sgl)

    banner("(3) dequant round-trip: BatchGen raw == SGLang w_ref ?")
    d = (r["bg_deq"] - r["w_ref"]).abs().max().item()
    print(f"  max|bg_deq - sgl_w_ref| = {d:.3e}   [{'OK' if d < 1e-2 else 'MISMATCH'}]")

    banner("(4) packed marlin WEIGHT bytes  BatchGen vs SGLang")
    cmp_t("marlin_w", r["bg_mw"], r["sgl_mw"])

    banner("(5) marlin SCALE bytes  BatchGen vs SGLang")
    cmp_t("marlin_s", r["bg_ms"].to(r["sgl_ms"].dtype), r["sgl_ms"])

    # ---- (6) functional: 1-expert MoE, SGLang-marlin (control) vs BatchGen-marlin vs ref ----
    banner("(6) functional fused_marlin_moe  (E=1, top-1):  control(SGLang) vs BatchGen vs ref")
    try:
        from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import fused_marlin_moe
        from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace
        from sglang.srt.server_args import ServerArgs, get_global_server_args, set_global_server_args_for_scheduler
        try:
            get_global_server_args()
        except ValueError:
            set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
        ws = marlin_make_workspace(dev, max_blocks_per_sm=4)

        Wg = (torch.randn(H, N_INTER, device=dev, dtype=torch.float16) * 0.1)
        Wu = (torch.randn(H, N_INTER, device=dev, dtype=torch.float16) * 0.1)
        Wd = (torch.randn(N_INTER, H, device=dev, dtype=torch.float16) * 0.1)
        rg, ru, rd = (quant_one(w, qtype, sgl) for w in (Wg, Wu, Wd))

        M = 4
        x = (torch.randn(M, H, device=dev, dtype=torch.bfloat16) * 0.1)
        tw = torch.ones(M, 1, device=dev, dtype=torch.float32)
        ti = torch.zeros(M, 1, dtype=torch.int32, device=dev)

        def run(tag, g, u, d):
            w1 = torch.cat([g["%s_mw" % tag], u["%s_mw" % tag]], dim=1).unsqueeze(0).contiguous()
            w1s = torch.cat([g["%s_ms" % tag], u["%s_ms" % tag]], dim=1).unsqueeze(0).contiguous().to(torch.bfloat16)
            w2 = d["%s_mw" % tag].unsqueeze(0).contiguous()
            w2s = d["%s_ms" % tag].unsqueeze(0).contiguous().to(torch.bfloat16)
            return fused_marlin_moe(hidden_states=x, w1=w1, w2=w2, w1_scale=w1s, w2_scale=w2s,
                                    gating_output=tw, topk_weights=tw, topk_ids=ti,
                                    global_num_experts=1, expert_map=None, g_idx1=None, g_idx2=None,
                                    sort_indices1=None, sort_indices2=None, w1_zeros=None, w2_zeros=None,
                                    workspace=ws, num_bits=4, is_k_full=True, inplace=False,
                                    routed_scaling_factor=None).float()

        xf = x.float()
        act = torch.nn.functional.silu(xf @ rg["w_ref"]) * (xf @ ru["w_ref"])
        ref = act @ rd["w_ref"]
        out_sgl = run("sgl", rg, ru, rd)
        out_bg = run("bg", rg, ru, rd)
        print(f"  control SGLang-marlin vs ref : max_abs_diff={ (out_sgl-ref).abs().max().item():.3e}")
        print(f"  BatchGen-marlin   vs ref     : max_abs_diff={ (out_bg-ref).abs().max().item():.3e}")
        print(f"  BatchGen vs SGLang output    : max_abs_diff={ (out_bg-out_sgl).abs().max().item():.3e}")
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"  functional test error: {e}")

    print("\nDONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
