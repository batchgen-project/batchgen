#!/usr/bin/env python3
"""Stage 1 — quantify the Kimi-K2.5 prefill marlin->raw conversion cost.

The prefill MoE path (model.py:_forward_prefill -> wrappers.py _transform_marlin_to_raw)
converts every expert's marlin-packed INT4 weights to the raw/wgmma layout on EVERY prefill
forward (the transform writes into a local dict, never back to the module, so it is not
cached). This bench measures that recurring cost:

  - full wrapper  marlin_to_wgmma_fused_gpu()   -> the real per-call cost (fused weight
                                                   kernel + eager scale-perm + H2D each call)
  - weights-only  mod.marlin_to_wgmma_transform  -> the pure CUDA kernel (~4 us)

The gap between the two is the eager scale-perm/H2D/launch overhead that the wrapper pays
per expert. Extrapolated to 384 experts x 60 MoE layers = the per-prefill-forward transform
tax, which recurs every microbatch.

Run on a Hopper GPU (H20 node0 / GH02), from the BatchGen workspace:
    CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=9.0a \
        python -m batchgen.moe.bench_prefill_conversion_cost
"""

import torch

from batchgen.moe.marlin_transform import (
    _get_inverse_scale_perm,
    _get_inverse_weight_perm,
    _load_transform_module,
    marlin_to_wgmma_fused_gpu,
)
from batchgen.moe.marlin_weight_prep import repack_int4_to_marlin_gs32

# Kimi-K2.5 MoE: hidden H=7168, expert intermediate N=2048, 384 routed experts, 60 MoE layers.
H = 7168
N = 2048
NUM_EXPERTS = 384
NUM_MOE_LAYERS = 60
GROUP_SIZE = 32
ITERS = 200


def _make_marlin(K, N, device):
    """Build one projection's marlin-packed INT4 weights + scales."""
    w = torch.randn((N, K), dtype=torch.float16, device=device) * 0.1
    wg = w.view(N, K // GROUP_SIZE, GROUP_SIZE)
    mx = wg.max(dim=-1, keepdim=True).values
    mn = wg.min(dim=-1, keepdim=True).values
    scales = torch.max(mx.abs() / 7.0, mn.abs() / 8.0).clamp(min=1e-10)
    q = torch.clamp(torch.round(wg / scales).int() + 8, 0, 15).view(N, K).int()
    raw_packed = torch.zeros(N, K // 8, dtype=torch.int32, device=device)
    for i in range(8):
        raw_packed |= (q[:, i::8] & 0xF) << (i * 4)
    raw_scales = scales.squeeze(-1).to(torch.bfloat16)
    return repack_int4_to_marlin_gs32(raw_packed, raw_scales, K, N)


def _time(fn, iters=ITERS):
    for _ in range(10):
        fn()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1000.0  # us


def main():
    device = "cuda"
    mod = _load_transform_module()
    print(f"Prefill marlin->raw conversion cost | device={torch.cuda.get_device_name()} "
          f"| iters={ITERS}")
    print(f"Kimi-K2.5: H={H} N={N} experts={NUM_EXPERTS} moe_layers={NUM_MOE_LAYERS} "
          f"group_size={GROUP_SIZE}\n")

    projections = [("gate", H, N), ("up", H, N), ("down", N, H)]

    print(f"{'proj':>5} | {'K':>5} {'N':>5} | {'full wrapper':>13} | {'weights-kernel':>14} | "
          f"{'scale+H2D gap':>13}")
    print("-" * 66)

    full_per_expert = 0.0
    kernel_per_expert = 0.0
    for name, K, Np in projections:
        marlin_qw, marlin_s = _make_marlin(K, Np, device)

        # Full wrapper — the ACTUAL per-forward path (fused kernel + eager scale-perm + H2D).
        full_us = _time(lambda: marlin_to_wgmma_fused_gpu(marlin_qw, marlin_s, K, Np))

        # Weights-only fused kernel (pre-allocated output + perm, no eager scale path).
        inv_perm = _get_inverse_weight_perm(4).to(device=device, dtype=torch.int32)
        out_packed = torch.empty(Np, K // 8, dtype=torch.int32, device=device)
        ker_us = _time(lambda: mod.marlin_to_wgmma_transform(marlin_qw, out_packed, inv_perm, K, Np))

        gap = full_us - ker_us
        full_per_expert += full_us
        kernel_per_expert += ker_us
        print(f"{name:>5} | {K:>5} {Np:>5} | {full_us:10.1f} us | {ker_us:11.1f} us | {gap:10.1f} us")

    total_transforms = NUM_EXPERTS * NUM_MOE_LAYERS
    full_forward_s = full_per_expert * total_transforms / 1e6
    kernel_forward_s = kernel_per_expert * total_transforms / 1e6

    print("-" * 66)
    print(f"\nPer expert (gate+up+down): full={full_per_expert:.1f} us | "
          f"kernel-only={kernel_per_expert:.1f} us")
    print(f"Per prefill forward = per-expert x {NUM_EXPERTS} experts x {NUM_MOE_LAYERS} layers "
          f"= {total_transforms:,} transforms:")
    print(f"    full wrapper (real):  {full_forward_s:6.2f} s  <-- recurs EVERY prefill microbatch")
    print(f"    kernel-only (ideal):  {kernel_forward_s:6.2f} s")
    print(f"    eager scale+H2D tax:  {full_forward_s - kernel_forward_s:6.2f} s "
          f"({(1 - kernel_forward_s / full_forward_s) * 100:.0f}% of the wrapper cost)")
    print("\nThe transform is NOT cached (wrappers.py:191 writes a local dict, never the module),")
    print("so this cost is paid on every prefill forward. A direct-marlin prefill path (no")
    print("conversion) eliminates it entirely at zero resident-memory cost (raw==marlin byte size).")


if __name__ == "__main__":
    main()
