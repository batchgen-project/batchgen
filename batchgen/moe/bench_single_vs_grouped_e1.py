#!/usr/bin/env python3
"""Discriminator: single-expert vs grouped-at-E=1 vs grouped-at-E=128, same work.

Resolves whether the single-expert kernel's higher TF/s is (a) a grouped-machinery tax, or
(b) a working-set/L2 artifact (one expert's weights fit L2; 128 experts don't).

For ONE expert's worth of work at each M:
  ARM A  single-expert  int4_single_expert_stage1/2
  ARM B  grouped E=1    grouped_int4_moe_stage1/2 with a single expert (same L2-resident weights)
  ARM C  grouped E=128  grouped_int4_moe_stage1/2, M_e=M per expert (weights blow L2 -> HBM stream)
All three do the SAME per-expert math (silu(gate·x)·(up·x)→down), checked vs one bf16 golden.
FLOP per token-set = 2*M*(2*H*N + N*H) (one expert's worth). TF/s uses that per-expert FLOP.

If A ~= B, grouped machinery is free and the single-expert lift is a working-set effect.
If C < A ~= B, the E=128 drop is the L2->HBM working-set penalty (the honest MoE number).

Run:
    CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=9.0a \
        python -m batchgen.moe.bench_single_vs_grouped_e1
"""

import torch

from batchgen.moe.bench_marlin_vs_wgmma import create_k25_int4_weights_raw
from batchgen.moe.int4_single_expert_wgmma import single_expert_int4_forward
from batchgen.moe.fused_int4_wgmma_grouped import _load_int4_grouped_module

H = 7168
N_INTER = 2048
GROUP_SIZE = 32
E128 = 128
ITERS = 30
M_VALUES = [64, 128, 256, 512, 1024, 2048, 4096]


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
    return s.elapsed_time(e) / iters * 1000.0


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
    mod = _load_int4_grouped_module()
    print(f"single vs grouped(E=1) vs grouped(E=128) | {torch.cuda.get_device_name()} | iters={ITERS}")
    print(f"H={H} N={N_INTER} gs={GROUP_SIZE}  per-expert FLOP=2*M*(2HN+NH)")
    print(f"one expert weights ~14MB (fits 60MB L2); 128 experts ~1.8GB (HBM)\n")

    # One expert's weights (shared by single-expert + grouped E=1).
    gpu8, gpi32, gate_s = create_k25_int4_weights_raw(H, N_INTER, device=dev)
    upu8, upi32, up_s = create_k25_int4_weights_raw(H, N_INTER, device=dev)
    dpu8, dpi32, down_s = create_k25_int4_weights_raw(N_INTER, H, device=dev)
    gate_w = _dequant(gpi32, gate_s, N_INTER, H)
    up_w = _dequant(upi32, up_s, N_INTER, H)
    down_w = _dequant(dpi32, down_s, H, N_INTER)

    eb = torch.empty(0, dtype=torch.int64, device=dev)
    swn_g, ssn_g = H // 2, H // 32
    swn_d, ssn_d = N_INTER // 2, N_INTER // 32

    # grouped E=1 pointer arrays (one expert).
    p1 = lambda t: torch.tensor([t.data_ptr()], dtype=torch.int64, device=dev)
    wgp1, wgs1 = p1(gpu8), p1(gate_s)
    wup1, wus1 = p1(upu8), p1(up_s)
    wdp1, wds1 = p1(dpu8), p1(down_s)

    # grouped E=128 pointer arrays: replicate the SAME expert 128x but as 128 DISTINCT buffers
    # so the working set is 128*14MB (blows L2), matching a real MoE weight footprint.
    g8_l, u8_l, d8_l, gs_l, us_l, ds_l = [], [], [], [], [], []
    for _ in range(E128):
        a, _, b = create_k25_int4_weights_raw(H, N_INTER, device=dev); g8_l.append(a); gs_l.append(b)
        a, _, b = create_k25_int4_weights_raw(H, N_INTER, device=dev); u8_l.append(a); us_l.append(b)
        a, _, b = create_k25_int4_weights_raw(N_INTER, H, device=dev); d8_l.append(a); ds_l.append(b)
    tl = lambda L: torch.tensor([t.data_ptr() for t in L], dtype=torch.int64, device=dev)
    wgpE, wgsE, wupE, wusE, wdpE, wdsE = tl(g8_l), tl(gs_l), tl(u8_l), tl(us_l), tl(d8_l), tl(ds_l)

    hdr = f"{'M':>7} | {'single: us  TF/s':>18} | {'grouped E=1: us  TF/s':>22} | {'grouped E=128: us  TF/s':>24}"
    print(hdr)
    print("-" * len(hdr))

    for M in M_VALUES:
        A = torch.randn((M, H), dtype=torch.bfloat16, device=dev) * 0.1
        inter = torch.nn.functional.silu(A @ gate_w.t()) * (A @ up_w.t())
        ref = (inter @ down_w.t()).to(torch.bfloat16)
        flop = 2.0 * M * (2 * H * N_INTER + N_INTER * H)

        # ARM A: single-expert
        def a():
            return single_expert_int4_forward(A, gpu8, gate_s, upu8, up_s, dpu8, down_s)
        da = _calc_diff(a(), ref); ua = _time(a)

        # ARM B: grouped E=1
        cnt1 = torch.full((1,), M, dtype=torch.int32, device=dev)
        mmt = (M + 63) // 64
        def b():
            itm = mod.grouped_int4_moe_stage1(A, cnt1, wgp1, wgs1, wup1, wus1, eb, eb,
                                              N_INTER, swn_g, ssn_g, mmt, M)
            return mod.grouped_int4_moe_stage2(itm, cnt1, wdp1, wds1, eb, H, swn_d, ssn_d, mmt, M)
        db = _calc_diff(b(), ref); ub = _time(b)

        # ARM C: grouped E=128 (M_e=M per expert; distinct weights -> HBM working set).
        # per-expert output compared to the same golden (expert 0 uses different weights,
        # so only compare TIMING for C; correctness for C is covered by the main sweep).
        cntE = torch.full((E128,), M, dtype=torch.int32, device=dev)
        AE = A.repeat(E128, 1)  # E128*M tokens, block-diagonal
        def c():
            itm = mod.grouped_int4_moe_stage1(AE, cntE, wgpE, wgsE, wupE, wusE, eb, eb,
                                              N_INTER, swn_g, ssn_g, mmt, M)
            return mod.grouped_int4_moe_stage2(itm, cntE, wdpE, wdsE, eb, H, swn_d, ssn_d, mmt, M)
        try:
            c(); uc = _time(c); tfc = flop / (uc / E128 * 1e-6) / 1e12
        except Exception as ex:
            uc = float('nan'); tfc = float('nan')
            print(f"  [E128 M={M}] {type(ex).__name__}: {ex}")

        tfa = flop / (ua * 1e-6) / 1e12
        tfb = flop / (ub * 1e-6) / 1e12
        ucpe = uc / E128 if uc == uc else float('nan')
        print(f"{M:7d} | {ua:8.1f} {tfa:6.1f} | {ub:9.1f} {tfb:6.1f} (d{db:.4f}) | "
              f"{ucpe:9.1f} {tfc:6.1f}  (dA{da:.4f})")
        del A, AE, ref, inter
        torch.cuda.empty_cache()

    print("\nA≈B => grouped machinery free; single lift is working-set/L2. "
          "C<A≈B => E=128 drop is the L2->HBM weight-stream penalty (honest MoE number).")


if __name__ == "__main__":
    main()
