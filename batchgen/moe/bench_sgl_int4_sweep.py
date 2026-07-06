#!/usr/bin/env python3
"""SGLang INT4 W4A16 "dequant grouped GEMM" MoE saturation sweep (uniform routing).

Benchmarks SGLang's Triton `fused_moe(use_int4_w4a16=True)` — the path that dequantizes
INT4 weights to BF16 then runs the grouped GEMM — across an M_e (tokens/expert) sweep, at
the Kimi-K2.5 INT4 MoE shapes so the TF/s is directly comparable to the WGMMA / BG-marlin
arms from bench_moe_saturation_sweep.py (H=7168, N=2048, gs=32, top_k=1, full MoE).

Full MoE per expert: silu(gate·x)·(up·x) → down. Uniform block-diagonal routing: token t
→ expert t//M_e, top_k=1, weight=1. Every expert gets exactly M_e tokens.
FLOP = 2·E·M_e·(2·H·N + N·H)   TF/s = FLOP/(us*1e-6)/1e12   (same as the Kimi sweep).

Weight layout is SGLang's own (moe_wna16 create_weights):
  w1 (gate|up) [E, 2N, H//2] uint8   w1_scale [E, 2N, H//gs] bf16   (N-major scale)
  w2 (down)    [E, H,  N//2] uint8   w2_scale [E, H,  N//gs] bf16
INT4 nibble packing: byte j low nibble = k=2j (even), high = 2j+1; code = (nibble-8).
Golden uses the SAME dequantized INT4 weights, so calc_diff isolates kernel error (~1e-2),
not quant error.

Run:
    CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=9.0a \
        python -m batchgen.moe.bench_sgl_int4_sweep
"""

import torch

H = 7168
N_INTER = 2048
E = 128
GROUP_SIZE = 32
TOPK_RATIO = 48         # Kimi: E_total/top_k = 384/8; B_global = 48*M_e (for reference)
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


def _build_expert(N_out, K, dev, seed):
    """Return (u8_packed [N_out, K//2], scale [N_out, K//gs] bf16, w_bf [N_out, K] bf16 golden).

    Codes in [0,16); dequant weight = (code-8)*scale. u8 byte j: low nib=k=2j, high=2j+1.
    """
    g = torch.Generator(device=dev).manual_seed(seed)
    codes = torch.randint(0, 16, (N_out, K), dtype=torch.int32, device=dev, generator=g)
    scale = (torch.rand(N_out, K // GROUP_SIZE, device=dev, generator=g) * 0.05 + 0.01).to(torch.bfloat16)
    scale_full = scale.float().repeat_interleave(GROUP_SIZE, dim=1)        # [N_out, K]
    w_bf = ((codes.float() - 8.0) * scale_full).to(torch.bfloat16)         # dequant golden
    u8 = (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8)          # [N_out, K//2]
    return u8, scale, w_bf


def main():
    dev = "cuda"
    torch.manual_seed(0)
    print(f"SGLang INT4 (fused_moe use_int4_w4a16) MoE sweep | device={torch.cuda.get_device_name()} "
          f"| iters={ITERS}")
    print(f"H={H}  N_inter={N_INTER}  E={E}  gs={GROUP_SIZE}  top_k=1 (uniform block-diagonal)")
    print(f"FLOP = 2*E*M_e*(2*H*N + N*H)\n")

    from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_moe
    from sglang.srt.layers.moe.topk import StandardTopKOutput
    try:
        from sglang.srt.layers.moe import MoeRunnerConfig
    except Exception:
        from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig

    # SGLang's fused_moe reads global server args (autotune/config); init a dummy one.
    from sglang.srt.server_args import (
        ServerArgs, get_global_server_args, set_global_server_args_for_scheduler)
    try:
        get_global_server_args()
    except Exception:
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

    print("Building INT4 weights + bf16 golden ...", flush=True)
    w1_u8_l, w1_s_l, w1_bf_l = [], [], []   # gate|up: [2N, H]
    w2_u8_l, w2_s_l, w2_bf_l = [], [], []   # down:    [H,  N]
    for e in range(E):
        u8, sc, wbf = _build_expert(2 * N_INTER, H, dev, seed=100 + e)   # w13
        w1_u8_l.append(u8); w1_s_l.append(sc); w1_bf_l.append(wbf)
        u8d, scd, wbfd = _build_expert(H, N_INTER, dev, seed=5000 + e)   # w2
        w2_u8_l.append(u8d); w2_s_l.append(scd); w2_bf_l.append(wbfd)
    w1 = torch.stack(w1_u8_l, 0).contiguous()          # [E, 2N, H//2] uint8
    w1_scale = torch.stack(w1_s_l, 0).contiguous()     # [E, 2N, H//gs] bf16
    w2 = torch.stack(w2_u8_l, 0).contiguous()          # [E, H, N//2] uint8
    w2_scale = torch.stack(w2_s_l, 0).contiguous()     # [E, H, N//gs] bf16
    print("  done\n")

    cfg = MoeRunnerConfig(
        num_experts=E, num_local_experts=E, hidden_size=H,
        intermediate_size_per_partition=N_INTER, top_k=1,
        activation="silu", is_gated=True, params_dtype=torch.bfloat16, inplace=False)

    hdr = f"{'M_e':>5} | {'B_glob':>7} | {'SGL-int4 (fused_moe):  us     TF/s     diff':>44}"
    print(hdr)
    print("-" * len(hdr))

    tf_hist = {}
    for Me in M_VALUES:
        M = E * Me
        B = TOPK_RATIO * Me
        hidden = (torch.randn((M, H), dtype=torch.bfloat16, device=dev) * 0.1)
        topk_ids = (torch.arange(M, device=dev) // Me).view(M, 1).to(torch.int32)
        topk_w = torch.ones((M, 1), dtype=torch.float32, device=dev)
        router_logits = torch.zeros((M, E), dtype=torch.float32, device=dev)
        topk_output = StandardTopKOutput(topk_w, topk_ids, router_logits)

        sg_us = sg_diff = float('nan')
        try:
            def sgl():
                return fused_moe(
                    hidden, w1, w2, topk_output, cfg,
                    use_int4_w4a16=True, w1_scale=w1_scale, w2_scale=w2_scale,
                    block_shape=[0, GROUP_SIZE])
            out = sgl()
            # golden: per-expert dequant-weight MoE (isolates kernel error, not quant)
            ref = torch.empty(M, H, dtype=torch.bfloat16, device=dev)
            for e in range(E):
                x = hidden[e * Me:(e + 1) * Me]
                gu = x @ w1_bf_l[e].t()                     # [Me, 2N]
                gate, up = gu[:, :N_INTER], gu[:, N_INTER:]
                inter = torch.nn.functional.silu(gate) * up
                ref[e * Me:(e + 1) * Me] = (inter @ w2_bf_l[e].t()).to(torch.bfloat16)
            sg_diff = _calc_diff(out, ref)
            sg_us = _time(sgl)
            del ref
        except Exception as ex:
            print(f"  [sgl Me={Me}] {type(ex).__name__}: {ex}")

        flop = 2.0 * E * Me * (2 * H * N_INTER + N_INTER * H)
        sg_tf = flop / (sg_us * 1e-6) / 1e12 if sg_us == sg_us and sg_us > 0 else float('nan')
        if sg_tf == sg_tf:
            tf_hist[Me] = sg_tf
        print(f"{Me:5d} | {B:7d} | {sg_us:12.1f} {sg_tf:8.1f} {sg_diff:10.5f}")
        del hidden, topk_ids, topk_w, router_logits
        torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("SATURATION KNEE (smallest M_e reaching >=95% of own max TF/s)")
    print("=" * 60)
    if tf_hist:
        mx = max(tf_hist.values())
        knee = min(m for m in sorted(tf_hist) if tf_hist[m] >= 0.95 * mx)
        print(f"  SGL-int4: max {mx:.1f} TF/s  knee M_e={knee}  B_global={TOPK_RATIO*knee}")
    print("FLOP=2*E*M_e*(2*H*N+N*H); compare to Kimi WGMMA ~110 / BG-marlin ~84 TF/s (same shapes).")


if __name__ == "__main__":
    main()
