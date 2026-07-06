#!/usr/bin/env python3
"""Single-expert INT4 WGMMA vs grouped: does grouping lower tensor-core utilization?

Benches the SINGLE-expert INT4 kernel (int4_single_expert_stage1/2) at one expert over an M
sweep, same Kimi shapes (H=7168, N=2048, gs=32). FLOP = 2*M*(2*H*N + N*H) (one expert). TF/s
is directly comparable to the grouped WGMMA sweep's per-M_e TF/s (grouped peak ~110).

If the single-expert ceiling ~= grouped ceiling, grouping does NOT cost TC util (both are the
same dequant-bound per-CTA engine). If single-expert is meaningfully higher, grouping costs.

Run:
    CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=9.0a \
        python -m batchgen.moe.bench_single_expert_int4
"""

import torch

from batchgen.moe.bench_marlin_vs_wgmma import create_k25_int4_weights_raw
from batchgen.moe.int4_single_expert_wgmma import single_expert_int4_forward

H = 7168
N_INTER = 2048
GROUP_SIZE = 32
ITERS = 30

# One expert: sweep total tokens M. Include large M so the single-expert grid fills the GPU
# (small M under-fills the grid -> low TF/s that is a grid-occupancy artifact, not TC util).
M_VALUES = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]


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
    print(f"Single-expert INT4 WGMMA sweep | device={torch.cuda.get_device_name()} | iters={ITERS}")
    print(f"H={H}  N_inter={N_INTER}  gs={GROUP_SIZE}  ONE expert | FLOP = 2*M*(2*H*N + N*H)")
    print(f"(compare to grouped WGMMA peak ~110 TF/s at the same shapes)\n")

    gpu8, gpi32, gate_s = create_k25_int4_weights_raw(H, N_INTER, device=dev)   # gate [N,H]
    upu8, upi32, up_s = create_k25_int4_weights_raw(H, N_INTER, device=dev)     # up   [N,H]
    dpu8, dpi32, down_s = create_k25_int4_weights_raw(N_INTER, H, device=dev)   # down [H,N]
    gate_w = _dequant(gpi32, gate_s, N_INTER, H)   # [N,H]
    up_w = _dequant(upi32, up_s, N_INTER, H)
    down_w = _dequant(dpi32, down_s, H, N_INTER)   # [H,N]

    hdr = f"{'M':>7} | {'single-expert:  us     TF/s     diff':>38}"
    print(hdr)
    print("-" * len(hdr))

    tf_hist = {}
    for M in M_VALUES:
        A = torch.randn((M, H), dtype=torch.bfloat16, device=dev) * 0.1
        us = diff = float('nan')
        try:
            def fwd():
                return single_expert_int4_forward(
                    A, gpu8, gate_s, upu8, up_s, dpu8, down_s)
            out = fwd()
            inter = torch.nn.functional.silu(A @ gate_w.t()) * (A @ up_w.t())
            ref = (inter @ down_w.t()).to(torch.bfloat16)
            diff = _calc_diff(out, ref)
            us = _time(fwd)
        except Exception as ex:
            print(f"  [M={M}] {type(ex).__name__}: {ex}")

        flop = 2.0 * M * (2 * H * N_INTER + N_INTER * H)
        tf = flop / (us * 1e-6) / 1e12 if us == us and us > 0 else float('nan')
        if tf == tf:
            tf_hist[M] = tf
        print(f"{M:7d} | {us:12.1f} {tf:8.1f} {diff:10.5f}")
        del A
        torch.cuda.empty_cache()

    print("\n" + "=" * 56)
    if tf_hist:
        mx = max(tf_hist.values())
        argmx = max(tf_hist, key=tf_hist.get)
        print(f"single-expert peak: {mx:.1f} TF/s at M={argmx}")
        print(f"grouped WGMMA peak: ~110.6 TF/s (E=128).  Ratio single/grouped = {mx/110.6:.3f}")
    print("If ~1.0, grouping does NOT lower TC util (same dequant-bound engine).")


if __name__ == "__main__":
    main()
