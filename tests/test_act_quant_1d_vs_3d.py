"""Compare CUDA act_quant_3d vs Triton per_token_blocked_quantize_bf16_to_fp8_flat.

Checks:
  1. Bit-exactness of FP8 bytes and FP32 scales across M=1..100_000.
  2. Latency of each at the same M sweep.

Target shape: [M, K=6144], block_size=128 — matches GLM-5 hidden size.
Run on H20 (SM90a). Kernels require the batchgen_kernels CUDA extension
built for sm_90a.
"""
import argparse
import math
import statistics
import sys

import torch


def run_cuda_3d(x: torch.Tensor):
    """Wrap CUDA act_quant_3d: treats each row as an 'expert' with mtp=1."""
    from batchgen_kernels.moe._C_fp8_blockwise_ops import act_quant_3d
    M, K = x.shape
    x_3d = x.view(M, 1, K).contiguous()
    tpe = torch.ones(M, dtype=torch.int32, device=x.device)
    y_u8, s_fp32 = act_quant_3d(x_3d, tpe)
    # [M, 1, K] u8 -> [M, K] fp8
    y = y_u8.view(torch.float8_e4m3fn).view(M, K)
    # [M, 1, num_blocks] -> [M, num_blocks]
    s = s_fp32.view(M, s_fp32.size(-1))
    return y, s


def run_triton_1d(x: torch.Tensor):
    from batchgen_kernels.triton.fp8_quantize import per_token_blocked_quantize_bf16_to_fp8_flat
    return per_token_blocked_quantize_bf16_to_fp8_flat(x, block_size=128)


def bit_exact(y_a: torch.Tensor, s_a: torch.Tensor, y_b: torch.Tensor, s_b: torch.Tensor):
    assert y_a.shape == y_b.shape, f"y shape {y_a.shape} vs {y_b.shape}"
    assert s_a.shape == s_b.shape, f"s shape {s_a.shape} vs {s_b.shape}"
    # FP8 bytes must match bit-for-bit.
    ya_u8 = y_a.view(torch.uint8)
    yb_u8 = y_b.view(torch.uint8)
    y_eq = bool(torch.equal(ya_u8, yb_u8))
    # Scales: fp32 — compare via .equal() then fall back to small-tol report.
    s_eq = bool(torch.equal(s_a, s_b))
    if not s_eq:
        diff = (s_a - s_b).abs()
        max_abs = diff.max().item()
        rel = (diff / s_a.abs().clamp(min=1e-12)).max().item()
        del diff
    else:
        max_abs = 0.0
        rel = 0.0
    total = ya_u8.numel()
    if not y_eq:
        # Stream the mismatch count in row-chunks to keep peak memory
        # under ~100 MB even for M >= 65k at K=6144.
        ya = ya_u8.view(y_a.size(0), -1)
        yb = yb_u8.view(y_b.size(0), -1)
        CHUNK = 4096
        ne = 0
        for i in range(0, ya.size(0), CHUNK):
            ne += int((ya[i:i + CHUNK] != yb[i:i + CHUNK]).sum().item())
    else:
        ne = 0
    return y_eq, s_eq, ne, total, max_abs, rel


def bench(fn, x, warmup=5, iters=20):
    # Warmup
    for _ in range(warmup):
        fn(x)
    torch.cuda.synchronize()
    # Timed
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn(x)
        ends[i].record()
    torch.cuda.synchronize()
    times_ms = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    return {
        "median_ms": statistics.median(times_ms),
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "mean_ms": statistics.mean(times_ms),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms", type=str,
                    default="1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65535,65536,100000",
                    help="comma-separated list of M values")
    ap.add_argument("--k", type=int, default=6144)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--no-bench", action="store_true")
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available; bailing.", file=sys.stderr)
        return 1

    device = torch.device("cuda")
    torch.manual_seed(args.seed)

    ms = [int(m) for m in args.ms.split(",")]
    K = args.k

    # Probe availability.
    try:
        from batchgen_kernels.moe._C_fp8_blockwise_ops import act_quant_3d  # noqa: F401
    except ImportError as e:
        print(f"CUDA act_quant_3d not available: {e}", file=sys.stderr)
        return 2
    try:
        from batchgen_kernels.triton.fp8_quantize import per_token_blocked_quantize_bf16_to_fp8_flat  # noqa: F401
    except ImportError as e:
        print(f"Triton per_token_blocked_quantize_bf16_to_fp8_flat not available: {e}", file=sys.stderr)
        return 2

    # Print header
    print(f"# Device: {torch.cuda.get_device_name(device)} (cap {torch.cuda.get_device_capability(device)})")
    print(f"# K={K}, seed={args.seed}, warmup={args.warmup}, iters={args.iters}")
    print()
    print(f"{'M':>8}  {'bit-eq':>7}  {'fp8-diff':>10}  {'scale-eq':>8}  {'scale-max-abs':>14}  "
          f"{'cuda-3d(ms)':>12}  {'triton-1d(ms)':>13}  {'speedup':>8}")
    print("-" * 110)

    any_mismatch = False
    for M in ms:
        # Aggressively free leftovers from previous M so large cases don't OOM
        # when the GPU is shared with a running server (benches allocate
        # per-iter outputs; their lifetime is bounded by the iter but cache
        # pressure builds without an empty_cache between loop iterations).
        torch.cuda.empty_cache()

        x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
        # Correctness run (fresh outputs).
        y_cuda, s_cuda = run_cuda_3d(x)
        y_trit, s_trit = run_triton_1d(x)

        if not args.no_verify:
            y_eq, s_eq, ne, total, s_max_abs, s_rel = bit_exact(y_cuda, s_cuda, y_trit, s_trit)
            if not y_eq or not s_eq:
                any_mismatch = True
            fp8_diff_str = f"{ne}/{total}" if not y_eq else "0"
            s_eq_str = "Y" if s_eq else "N"
            bit_eq_str = "Y" if y_eq else "N"
            s_max_abs_str = f"{s_max_abs:.3e}" if not s_eq else "0"
        else:
            bit_eq_str = fp8_diff_str = s_eq_str = s_max_abs_str = "-"

        # Free correctness outputs before benching to reduce peak footprint.
        del y_cuda, s_cuda, y_trit, s_trit

        if not args.no_bench:
            t_cuda = bench(run_cuda_3d, x, args.warmup, args.iters)
            t_trit = bench(run_triton_1d, x, args.warmup, args.iters)
            speedup = t_trit["median_ms"] / t_cuda["median_ms"]  # >1 → CUDA faster
            cuda_str = f"{t_cuda['median_ms']:.4f}"
            trit_str = f"{t_trit['median_ms']:.4f}"
            sp_str = f"{speedup:.2f}x"
        else:
            cuda_str = trit_str = sp_str = "-"

        del x
        print(f"{M:>8}  {bit_eq_str:>7}  {fp8_diff_str:>10}  {s_eq_str:>8}  {s_max_abs_str:>14}  "
              f"{cuda_str:>12}  {trit_str:>13}  {sp_str:>8}", flush=True)

    print()
    print(f"# Summary: {'MISMATCH found' if any_mismatch else 'all bit-exact'}")
    return 0 if not any_mismatch else 3


if __name__ == "__main__":
    sys.exit(main())
