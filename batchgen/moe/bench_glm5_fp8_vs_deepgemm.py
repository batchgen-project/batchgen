#!/usr/bin/env python3
"""GLM-5-FP8 EP-decode grouped-GEMM saturation sweep: BatchGen vs DeepGEMM.

"How many tokens/expert saturate the device per grouped-FP8-GEMM launch, and how far is
BatchGen's hand-rolled FP8 blockwise kernel from DeepGEMM (the SGLang MoE path)?"

Two arms, driven at GLM-5-FP8 EP16 decode shapes (16 local experts at FULL width), on the
SAME per-expert FP8 weights + SAME bf16-dequant golden, across an M_e (tokens/expert) sweep:

  ARM 1  BatchGen  grouped_fp8_blockwise_gemm  (batchgen/moe/grouped_fp8_blockwise_moe.py ->
          batchgen_kernels.moe._C_fp8_blockwise_gemm; CuTe SM90 persistent 3-WG, TileN=TileK=128,
          adaptive TileM 16/32/64). Prior finding: ~39% of H20 FP8 peak (~115 TF/s).
  ARM 2  DeepGEMM  m_grouped_fp8_gemm_nt_masked  (deep_gemm; SGLang decode MoE uses this).
          Masked variant fits the EP uniform-per-expert layout with no token sorting.

We bench the GEMM STAGE (a single grouped FP8 GEMM), NOT the full silu(gate.x)*(up.x)->down
fusion, because that is the apples-to-apples primitive both libraries expose: DeepGEMM masked is
one grouped GEMM/call, and BatchGen's fused_s1 folds gate+up+SiLU into one launch (not comparable
per-GEMM). We sweep the TWO GLM-5 MoE GEMM shapes separately:
    gate_up (S1 one proj): K=H=6144,      N=N_inter=2048   -> [M_e,6144] x [2048,6144]^T
    down    (S3)         : K=N_inter=2048, N=H=6144         -> [M_e,2048] x [6144,2048]^T

Both operands NT (K contiguous), e4m3, 1x128 per-token act scale + 128x128 blockwise weight scale.

FLOP per grouped GEMM (valid tokens only, uniform M_e over E experts):
    FLOP = 2 * E * M_e * N * K          TF/s = FLOP / (us*1e-6) / 1e12
Full MoE (for reference, not benched here) would be 2*E*M_e*(2*H*N + N*H).

KNEE = smallest M_e whose TF/s reaches >=95% of that arm's own max TF/s (CUDA-Events timing).
GLM-5 EP16 token identity: top_k=8, E_total=256 -> M_e = 8*B_global/256 = B_global/32, so
tokens-to-saturate B_global = 32 * knee_M_e.

Run:
    CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=9.0a \
        python -m batchgen.moe.bench_glm5_fp8_vs_deepgemm
"""

import torch

# ------------------------------------------------------------------ #
# GLM-5-FP8 EP16 decode constants (config.json / test_glm5_decode_moe_kernels.py)
# ------------------------------------------------------------------ #
H = 6144            # hidden
N_INTER = 2048      # moe_intermediate_size (per-expert FFN width)
E = 16              # E_local: experts owned per rank on 16-way EP, FULL width
E_TOTAL = 256       # n_routed_experts (global)
TOPK = 8            # num_experts_per_tok
BLOCK = 128         # weight 128x128, act 1x128
FP8_MAX = 448.0
TOPK_RATIO = E_TOTAL // TOPK   # 32:  M_e = B_global / 32  ->  B_global = 32 * M_e
H20_FP8_PEAK_TFLOPS = 296.0

M_VALUES = [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120,
            128, 144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 384, 448, 512]
ITERS = 30

# The two GLM-5 MoE GEMM shapes:  (name, K=contraction, N=output_width)
GEMM_SHAPES = [
    ("gate_up (S1)", H, N_INTER),      # K=6144, N=2048
    ("down    (S3)", N_INTER, H),      # K=2048, N=6144
]


# ------------------------------------------------------------------ #
# harness (ported verbatim from Kimi bench_moe_saturation_sweep.py)
# ------------------------------------------------------------------ #
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


def _round_up(x, m):
    return ((x + m - 1) // m) * m


# ------------------------------------------------------------------ #
# synthetic FP8 weights: per-128x128 tile scale = amax/448 (multiply-to-dequant),
# shared by BOTH arms + the bf16 golden. (from test_glm5_fp8_gemm_vs_bf16.py:67)
# ------------------------------------------------------------------ #
def _build_expert_fp8(N, K, dev, seed):
    """One expert: (w_fp8[N,K] e4m3, w_scale[N/128,K/128] f32, w_bf16[N,K])."""
    g = torch.Generator(device=dev).manual_seed(seed)
    w_bf = torch.randn((N, K), dtype=torch.bfloat16, device=dev, generator=g) * 0.02
    w_f32 = w_bf.float()
    nb, kb = N // BLOCK, K // BLOCK
    # vectorized per-tile amax over [nb,128,kb,128]
    tiles = w_f32.reshape(nb, BLOCK, kb, BLOCK)
    amax = tiles.abs().amax(dim=(1, 3)).clamp(min=1e-12)          # [nb,kb]
    scale = amax / FP8_MAX                                        # [nb,kb]
    scale_full = scale.repeat_interleave(BLOCK, 0).repeat_interleave(BLOCK, 1)
    w_fp8 = (w_f32 / scale_full).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return w_fp8, scale, w_bf


def build_weights(N, K, dev):
    """Stack E experts -> (w3d[E,N,K] fp8, ws_raw[E,N/128,K/128] f32, w_bf[E,N,K] bf16)."""
    w_fp8_l, ws_l, wbf_l = [], [], []
    for e in range(E):
        wf, ws, wb = _build_expert_fp8(N, K, dev, seed=1000 + e)
        w_fp8_l.append(wf); ws_l.append(ws); wbf_l.append(wb)
    w3d = torch.stack(w_fp8_l, 0).contiguous()
    ws_raw = torch.stack(ws_l, 0).contiguous()            # [E, N/128, K/128]
    w_bf = torch.stack(wbf_l, 0).contiguous()             # [E, N, K] bf16 golden weight
    return w3d, ws_raw, w_bf


def pad_k4(ws_raw):
    """[E,N/128,K/128] -> [E,N/128,pad4(K/128)] zero-padded (BatchGen kernel checks %4==0)."""
    Eg, nb, kb = ws_raw.shape
    kb4 = _round_up(kb, 4)
    if kb4 == kb:
        return ws_raw.contiguous()
    out = torch.zeros(Eg, nb, kb4, dtype=ws_raw.dtype, device=ws_raw.device)
    out[:, :, :kb] = ws_raw
    return out.contiguous()


# ------------------------------------------------------------------ #
# main
# ------------------------------------------------------------------ #
def main():
    dev = "cuda"
    torch.manual_seed(0)
    print(f"GLM-5-FP8 EP-decode grouped-GEMM sweep | device={torch.cuda.get_device_name()} "
          f"| iters={ITERS}")
    print(f"H={H}  N_inter={N_INTER}  E_local={E}  top_k={TOPK}  E_total={E_TOTAL}  "
          f"block=128  e4m3 | FP8 peak ~{H20_FP8_PEAK_TFLOPS:.0f} TF/s")
    print(f"FLOP/GEMM = 2*E*M_e*N*K   |   B_global = {TOPK_RATIO}*M_e\n")

    # ---- ARM 1: BatchGen kernel + act quant ----
    from batchgen.moe.grouped_fp8_blockwise_moe import grouped_fp8_blockwise_gemm
    from batchgen_kernels.moe._C_fp8_blockwise_ops import act_quant_3d

    # ---- ARM 2: DeepGEMM (guarded) ----
    have_dg = False
    dg_err = ""
    try:
        import deep_gemm
        from deep_gemm.utils.math import per_token_cast_to_fp8
        have_dg = True
    except Exception as ex:
        dg_err = f"import failed: {ex}"
    if not have_dg:
        print(f"[warn] DeepGEMM arm DISABLED — {dg_err}\n")

    def dg_per_token_cast(x2d):
        # x2d [m,K] bf16 -> (fp8[m,K], sf[m,K/128]); H20 non-UE8M0 path
        try:
            return per_token_cast_to_fp8(x2d, False)
        except TypeError:
            return per_token_cast_to_fp8(x2d)

    # per-arm TF/s history, keyed per GEMM shape
    tf_hist = {}   # {shape_name: {"BatchGen":{Me:tf}, "DeepGEMM":{Me:tf}}}

    for shape_name, K, N in GEMM_SHAPES:
        print("=" * 96)
        print(f"GEMM: {shape_name}   K={K}  N={N}   (per-expert [M_e,{K}] x [{N},{K}]^T -> [M_e,{N}])")
        print("=" * 96)
        # weights + golden built ONCE per shape
        w3d, ws_raw, w_bf = build_weights(N, K, dev)
        ws_pad4 = pad_k4(ws_raw)   # BatchGen weight-scale layout

        hdr = (f"{'M_e':>5} | {'B_glob':>7} | "
               f"{'BatchGen: us   TF/s   diff':>30} | "
               f"{'DeepGEMM: us   TF/s   diff':>30}")
        print(hdr)
        print("-" * len(hdr))
        tf_hist[shape_name] = {"BatchGen": {}, "DeepGEMM": {}}

        for Me in M_VALUES:
            mtp = max(_round_up(Me, 64), 64)     # reserved-buffer stride (>=64, mult of 64)
            avg = Me                              # selects TileM (wrapper bumps 33..48 -> 64)
            seqlens = torch.full((E,), Me, dtype=torch.int32, device=dev)
            cu_seqlens = torch.arange(0, (E + 1) * mtp, mtp, dtype=torch.int32, device=dev)
            B = TOPK_RATIO * Me

            # shared bf16 activations (same tensor feeds both arms' quantizers)
            x_bf = torch.randn((E, mtp, K), dtype=torch.bfloat16, device=dev) * 0.3

            # ---- bf16 golden (per expert, valid rows only) ----
            ref = torch.empty(E, Me, N, dtype=torch.bfloat16, device=dev)
            for e in range(E):
                ref[e] = (x_bf[e, :Me].float() @ w_bf[e].float().t()).to(torch.bfloat16)
            ref_flat = ref.reshape(E * Me, N)

            # ---------- ARM 1: BatchGen ----------
            bg_us = bg_diff = float('nan')
            try:
                y_u8, x_scale_3d = act_quant_3d(x_bf, seqlens)       # [E,mtp,K],[E,mtp,K/128]
                x_fp8 = y_u8.view(E * mtp, K).view(torch.float8_e4m3fn)
                x_scale_t = x_scale_3d.view(E * mtp, -1).t().contiguous()   # [K/128, E*mtp]

                def bg():
                    return grouped_fp8_blockwise_gemm(
                        x_fp8, w3d, seqlens, cu_seqlens,
                        x_scale_t, ws_pad4, avg)
                out = bg().view(E, mtp, N)[:, :Me, :].reshape(E * Me, N)
                bg_diff = _calc_diff(out, ref_flat)
                bg_us = _time(bg)
            except Exception as ex:
                print(f"  [batchgen Me={Me}] {type(ex).__name__}: {ex}")

            # ---------- ARM 2: DeepGEMM (masked) ----------
            dg_us = dg_diff = float('nan')
            if have_dg:
                try:
                    x2d = x_bf.reshape(E * mtp, K)
                    a_fp8_2d, a_sf_2d = dg_per_token_cast(x2d)
                    a_fp8 = a_fp8_2d.view(E, mtp, K)
                    a_scale = a_sf_2d.view(E, mtp, K // BLOCK)
                    a_scale = deep_gemm.get_mn_major_tma_aligned_tensor(a_scale)
                    masked_m = torch.full((E,), Me, dtype=torch.int32, device=dev)
                    out_dg = torch.empty(E, mtp, N, dtype=torch.bfloat16, device=dev)
                    max_block_n = 160 if Me <= 64 else 256

                    def dg():
                        deep_gemm.m_grouped_fp8_gemm_nt_masked(
                            (a_fp8, a_scale), (w3d, ws_raw), out_dg,
                            masked_m, Me, max_block_n=max_block_n)
                        return out_dg
                    dg()
                    torch.cuda.synchronize()
                    out_dg_valid = out_dg[:, :Me, :].reshape(E * Me, N)
                    dg_diff = _calc_diff(out_dg_valid, ref_flat)
                    dg_us = _time(dg)
                except Exception as ex:
                    if not tf_hist[shape_name]["DeepGEMM"]:
                        print(f"  [deepgemm Me={Me}] {type(ex).__name__}: {ex}  (auto-skip)")

            # ---- TF/s ----
            flop = 2.0 * E * Me * N * K
            bg_tf = flop / (bg_us * 1e-6) / 1e12 if bg_us == bg_us and bg_us > 0 else float('nan')
            dg_tf = flop / (dg_us * 1e-6) / 1e12 if dg_us == dg_us and dg_us > 0 else float('nan')
            if bg_tf == bg_tf: tf_hist[shape_name]["BatchGen"][Me] = bg_tf
            if dg_tf == dg_tf: tf_hist[shape_name]["DeepGEMM"][Me] = dg_tf

            print(f"{Me:5d} | {B:7d} | "
                  f"{bg_us:8.1f} {bg_tf:6.1f} {bg_diff:8.5f}     | "
                  f"{dg_us:8.1f} {dg_tf:6.1f} {dg_diff:8.5f}")

            del x_bf, ref, ref_flat
            torch.cuda.empty_cache()
        print()

    # ------------------------------------------------------------------ #
    # knee analysis
    # ------------------------------------------------------------------ #
    print("=" * 96)
    print("SATURATION KNEE (smallest M_e reaching >=95% of that arm's own max TF/s)")
    print("=" * 96)
    print(f"{'GEMM':>14} | {'arm':>9} | {'max TF/s':>9} | {'% peak':>7} | "
          f"{'knee M_e':>9} | {'B_glob=32*Me':>13}")
    print("-" * 78)
    for shape_name, _K, _N in GEMM_SHAPES:
        for arm in ("BatchGen", "DeepGEMM"):
            hist = tf_hist[shape_name][arm]
            if not hist:
                print(f"{shape_name:>14} | {arm:>9} | {'n/a':>9} | {'-':>7} | {'disabled':>9} |")
                continue
            mx = max(hist.values())
            knee = min(m for m in sorted(hist) if hist[m] >= 0.95 * mx)
            pk = 100.0 * mx / H20_FP8_PEAK_TFLOPS
            print(f"{shape_name:>14} | {arm:>9} | {mx:9.1f} | {pk:6.1f}% | "
                  f"{knee:9d} | {TOPK_RATIO * knee:13d}")
    print("\nNote: TF/s = 2*E*M_e*N*K / us; M_e = B_global/32 (GLM-5 EP16, top_k=8, 256 experts).")
    print("calc_diff is vs a bf16-dequant golden; FP8 gate target < 1e-3 (do NOT loosen).")
    print("The BatchGen-vs-DeepGEMM max-TF/s ratio is the kernel gap this bench quantifies.")


if __name__ == "__main__":
    main()
