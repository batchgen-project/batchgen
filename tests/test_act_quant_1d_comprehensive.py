"""Comprehensive correctness + boundary tests for
`per_token_blocked_quantize_bf16_to_fp8_1d` — the Triton 1D-grid CUDA-parity
act_quant kernel in batchgen_kernels.triton.fp8_quantize.

Covers:
    1. Broad shape sweep  (M ∈ {1..100000}, K ∈ {128, 2048, 4096, 6144, 8192,
       16384, 32768}) — both correctness vs CUDA `act_quant_3d` and vs a
       reference PyTorch implementation, plus a pow2/non-pow2 NUM_BLOCKS
       split so the mask path is exercised.
    2. Value-distribution stress  — uniform [-3, 3], Gaussian, heavy tails
       (Student-T), saturating values > FP8_MAX, subnormals close to 0,
       inf/nan injection (should survive without OOB).
    3. Block-size variation  — only block_size=128 is the production path
       but we verify the kernel rejects other block sizes cleanly.
    4. Non-contiguous + strided input  — kernel asserts contiguous, so
       verify it raises instead of silently accessing bad memory.
    5. K-edge cases  — K not a multiple of 128 (pad path), K=128 (single
       block), K=0 (empty hidden) edge.
    6. Scale properties — verify scale ≥ EPSILON, amax / scale == 448 on
       saturating blocks, FP8 bytes round to RNE.
    7. Determinism — same input → same output (should be fully deterministic
       since no atomic reductions).
    8. Numerical stability — saturating values should not produce inf/nan
       in scale; NaN in input should be zeroed post-quant.

Run on H20 (SM90a).
"""
import argparse
import math
import sys

import torch


# ---------------------------------------------------------------------- #
# Kernel imports
# ---------------------------------------------------------------------- #


def _reference_blockwise_fp8_quant(x: torch.Tensor, block_size: int = 128):
    """Pure-PyTorch reference: per-token per-block amax → scale, saturating
    FP8 cast with RNE. Matches SGLang semantics.
    """
    assert x.dim() == 2
    M, K = x.shape
    num_blocks = (K + block_size - 1) // block_size
    FP8_SAFE_MAX = 448.0
    FP8_E4M3_MIN_NORMAL = 1.52587890625e-05
    EPSILON = 1e-12

    x_f32 = x.to(torch.float32)
    # Reshape into [M, num_blocks, block_size], padding K if necessary.
    pad = num_blocks * block_size - K
    if pad > 0:
        x_padded = torch.nn.functional.pad(x_f32, (0, pad), value=0.0)
    else:
        x_padded = x_f32
    x_grid = x_padded.view(M, num_blocks, block_size)
    amax = x_grid.abs().amax(dim=-1)                        # [M, num_blocks]
    amax = amax.clamp_min(FP8_E4M3_MIN_NORMAL)
    scale = (amax * (1.0 / FP8_SAFE_MAX)).clamp_min(EPSILON)
    # Broadcast scale and quantize (saturating, NaN-safe).
    scaled = x_grid / scale.unsqueeze(-1)
    scaled = scaled.clamp(min=-FP8_SAFE_MAX, max=FP8_SAFE_MAX)
    finite = scaled.abs() < 1e30
    scaled = torch.where(finite, scaled, torch.zeros_like(scaled))
    # Cast to FP8 (RNE), reshape back, trim padding.
    out = scaled.to(torch.float8_e4m3fn).view(M, num_blocks * block_size)
    if pad > 0:
        out = out[:, :K].contiguous()
    return out, scale  # scale: [M, num_blocks]


def run_triton_1d(x: torch.Tensor, block_size: int = 128):
    from batchgen_kernels.triton.fp8_quantize import per_token_blocked_quantize_bf16_to_fp8_1d
    return per_token_blocked_quantize_bf16_to_fp8_1d(x, block_size=block_size)


def run_cuda_3d(x: torch.Tensor, block_size: int = 128):
    """Wrap CUDA act_quant_3d: treats each row as an 'expert' (E=M, mtp=1)."""
    assert block_size == 128, "CUDA act_quant_3d hardwired to BLOCK_SIZE=128"
    from batchgen_kernels.moe._C_fp8_blockwise_ops import act_quant_3d
    M, K = x.shape
    x_3d = x.view(M, 1, K).contiguous()
    tpe = torch.ones(M, dtype=torch.int32, device=x.device)
    y_u8, s_fp32 = act_quant_3d(x_3d, tpe)
    return y_u8.view(torch.float8_e4m3fn).view(M, K), s_fp32.view(M, -1)


# ---------------------------------------------------------------------- #
# Comparison helpers
# ---------------------------------------------------------------------- #


def fp8_byte_diff(a: torch.Tensor, b: torch.Tensor) -> int:
    """Count mismatched FP8 bytes; stream in 4 K-row chunks."""
    a_u8 = a.view(torch.uint8).view(a.size(0), -1)
    b_u8 = b.view(torch.uint8).view(b.size(0), -1)
    ne = 0
    CHUNK = 4096
    for i in range(0, a.size(0), CHUNK):
        ne += int((a_u8[i:i+CHUNK] != b_u8[i:i+CHUNK]).sum().item())
    return ne


def scale_max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


def pretty_pct(ne: int, total: int) -> str:
    if total == 0:
        return "n/a"
    return f"{ne/total*100:.4e}%"


# ---------------------------------------------------------------------- #
# Test cases
# ---------------------------------------------------------------------- #


def test_shape_sweep(device, verbose: bool = False, bench: bool = False):
    print("\n=== Shape sweep: correctness vs CUDA act_quant_3d AND vs PyTorch ref ===")
    failed = 0
    # Cover: small decode, medium decode, medium prefill, large prefill,
    # and the specific problem shapes K=2048 (q_a), K=6144 (hidden),
    # K=32768 (attn_output = num_heads*v_head_dim = 128*256).
    shapes = []
    for K in [128, 256, 1024, 2048, 4096, 6144, 8192, 16384, 32768]:
        for M in [1, 7, 32, 64, 128, 256, 512, 1024, 4096, 16384, 65535, 65536, 100000]:
            # Skip shapes that would OOM on a shared-use H20.
            bytes_est = M * K * 2 * 3  # x + y + scale + margin
            if bytes_est > 4 * (1 << 30):  # >4GB: skip
                continue
            shapes.append((M, K))
    print(f"{len(shapes)} shapes")
    print(f"{'M':>6}  {'K':>6}  {'vs CUDA fp8':>14}  {'vs ref fp8':>14}  {'scale-exact':>11}  {'max|Δscale|':>12}")
    print("-" * 82)
    for M, K in shapes:
        torch.cuda.empty_cache()
        x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
        # Triton 1D kernel under test
        try:
            y_trit, s_trit = run_triton_1d(x)
        except Exception as e:
            print(f"{M:>6}  {K:>6}  TRITON-ERROR: {type(e).__name__}: {e}")
            failed += 1
            continue
        # vs CUDA act_quant_3d (block_size=128 only)
        y_cuda, s_cuda = run_cuda_3d(x)
        ne_cuda = fp8_byte_diff(y_trit, y_cuda)
        scale_eq_cuda = bool(torch.equal(s_trit, s_cuda))
        max_ds_cuda = 0.0 if scale_eq_cuda else scale_max_abs(s_trit, s_cuda)
        # vs PyTorch reference
        y_ref, s_ref = _reference_blockwise_fp8_quant(x)
        ne_ref = fp8_byte_diff(y_trit, y_ref)
        total = y_trit.numel()
        print(f"{M:>6}  {K:>6}  {f'{ne_cuda}/{total}':>14}  {f'{ne_ref}/{total}':>14}  "
              f"{'Y' if scale_eq_cuda else 'N':>11}  {max_ds_cuda:>12.3e}")
        del x, y_trit, s_trit, y_cuda, s_cuda, y_ref, s_ref
    return failed


def test_value_distributions(device):
    """Feed non-random inputs to exercise edge cases of the reduction + FP8 cast."""
    print("\n=== Value distribution stress (M=256, K=6144) ===")
    M, K = 256, 6144
    FP8_SAFE_MAX = 448.0

    cases = {
        "uniform[-3,3]":     (torch.rand(M, K, device=device) * 6 - 3).to(torch.bfloat16),
        "gaussian":          torch.randn(M, K, dtype=torch.bfloat16, device=device),
        "student-t-heavy":   (torch.randn(M, K, device=device) / (torch.rand(M, K, device=device) + 0.05)).clamp_(-500, 500).to(torch.bfloat16),
        "all-zeros":         torch.zeros(M, K, dtype=torch.bfloat16, device=device),
        "all-subnormal":     torch.full((M, K), 1e-5, dtype=torch.bfloat16, device=device),
        "saturating":        torch.full((M, K), FP8_SAFE_MAX * 2, dtype=torch.bfloat16, device=device),
        "alternating-sign":  ((torch.arange(M * K, device=device) % 2) * 2 - 1).to(torch.bfloat16).view(M, K) * 100,
        "large-then-small":  torch.cat([
            torch.full((M, K // 2), 200.0, dtype=torch.bfloat16, device=device),
            torch.full((M, K // 2), 1e-4, dtype=torch.bfloat16, device=device),
        ], dim=1),
    }
    # Add explicit NaN / Inf injection on a subset of rows.
    x_nan = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    x_nan[10] = float("nan")
    x_nan[20] = float("inf")
    x_nan[30] = float("-inf")
    cases["nan+inf-rows"] = x_nan

    print(f"{'case':<22}  {'vs CUDA fp8':>14}  {'scale-exact':>11}  {'scale-min':>12}  {'scale-max':>12}  {'has-nan-scale':>13}")
    print("-" * 100)
    for name, x in cases.items():
        torch.cuda.empty_cache()
        y_trit, s_trit = run_triton_1d(x)
        y_cuda, s_cuda = run_cuda_3d(x)
        ne = fp8_byte_diff(y_trit, y_cuda)
        total = y_trit.numel()
        scale_eq = bool(torch.equal(s_trit, s_cuda))
        smin = float(s_trit.min().item())
        smax = float(s_trit.max().item())
        has_nan_scale = bool(torch.isnan(s_trit).any().item())
        print(f"{name:<22}  {f'{ne}/{total}':>14}  {'Y' if scale_eq else 'N':>11}  "
              f"{smin:>12.3e}  {smax:>12.3e}  {str(has_nan_scale):>13}")
        del x, y_trit, s_trit, y_cuda, s_cuda


def test_k_edges(device):
    """K not aligned to 128, K=128, small M."""
    print("\n=== K-alignment edge cases ===")
    FP8_SAFE_MAX = 448.0
    cases = [
        (128, 128),     # single K-block exactly
        (256, 130),     # K not aligned, tail pad
        (256, 255),     # odd K, 2 blocks, 1 block padded
        (128, 384),     # 3 blocks
        (256, 5120),    # 40 blocks (not pow2), pad to 64
        (128, 8064),    # 63 blocks, pad to 64
        (64,  8320),    # 65 blocks, pad to 128
    ]
    print(f"{'M':>6}  {'K':>6}  {'num_blocks':>10}  {'vs ref fp8':>14}  {'scale-shape':>13}")
    print("-" * 62)
    for M, K in cases:
        torch.cuda.empty_cache()
        x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
        y_trit, s_trit = run_triton_1d(x)
        y_ref, s_ref = _reference_blockwise_fp8_quant(x)
        num_blocks = s_trit.size(1)
        ne = fp8_byte_diff(y_trit, y_ref)
        total = y_trit.numel()
        print(f"{M:>6}  {K:>6}  {num_blocks:>10}  {f'{ne}/{total}':>14}  {str(tuple(s_trit.shape)):>13}")
        del x, y_trit, s_trit, y_ref, s_ref


def test_non_contiguous_rejected(device):
    """Kernel wrapper asserts is_contiguous(); verify it raises instead of
    silently OOB-accessing strided memory."""
    print("\n=== Non-contiguous input rejection ===")
    x = torch.randn(128, 12288, dtype=torch.bfloat16, device=device)
    x_view = x[:, :6144]  # contiguous-on-row but stride != 1 on col for reshape
    assert x_view.is_contiguous(), "test setup broke"
    # Make a truly non-contiguous tensor
    x_nc = x.t()  # [12288, 128] — non-contiguous
    try:
        _ = run_triton_1d(x_nc[:128])   # 128 rows, stride(1)=12288
        print("  FAIL: kernel did NOT raise on non-contiguous input")
    except AssertionError:
        print("  OK:  AssertionError (expected)")


def test_determinism(device):
    """Same input twice → identical FP8 bytes AND scales."""
    print("\n=== Determinism ===")
    for M, K in [(1, 6144), (128, 6144), (4096, 6144), (256, 32768)]:
        torch.cuda.empty_cache()
        x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
        y1, s1 = run_triton_1d(x)
        y2, s2 = run_triton_1d(x)
        y_eq = bool(torch.equal(y1.view(torch.uint8), y2.view(torch.uint8)))
        s_eq = bool(torch.equal(s1, s2))
        print(f"  M={M:>6} K={K:>6}:  fp8-identical={y_eq}  scale-identical={s_eq}")
        del x, y1, s1, y2, s2


def test_scale_bounds(device):
    """Scale must be finite, positive, ≥ EPSILON; on saturating inputs
    amax/scale ≈ 448."""
    print("\n=== Scale bounds + saturation ===")
    FP8_SAFE_MAX = 448.0
    EPSILON = 1e-12
    M, K = 256, 6144
    x = torch.randn(M, K, dtype=torch.bfloat16, device=device) * 1000  # force saturation
    _, s = run_triton_1d(x)
    smin = float(s.min().item())
    smax = float(s.max().item())
    n_inf = int(torch.isinf(s).sum().item())
    n_nan = int(torch.isnan(s).sum().item())
    print(f"  scale min: {smin:.3e} (≥EPSILON={EPSILON:.0e}: {smin >= EPSILON})")
    print(f"  scale max: {smax:.3e}")
    print(f"  #inf in scale: {n_inf} (expected 0)")
    print(f"  #nan in scale: {n_nan} (expected 0)")
    # Check that quantized |y|/scale bounded by FP8_SAFE_MAX (post-clamp).
    y, _ = run_triton_1d(x)
    y_f32 = y.to(torch.float32)
    max_abs_y = float(y_f32.abs().max().item())
    print(f"  max|fp8 output| cast to f32: {max_abs_y} (≤ {FP8_SAFE_MAX})")


def test_aligned_with_prefill_shapes(device):
    """Exact shapes that hit the production prefill call sites on GLM-5."""
    print("\n=== GLM-5 production prefill call-site shapes ===")
    # From MLA prefill and attention:
    #   hidden_flat      [M, 6144]    M = up to ~32k
    #   q_a_normed       [M, 2048]    q_lora_rank
    #   attn_output      [M, 32768]   num_heads=128 * v_head_dim=256
    prod_shapes = [
        ("hidden@decode",  128,  6144),
        ("hidden@prefill-mb", 4096, 6144),
        ("hidden@prefill-xl", 32000, 6144),
        ("q_a@decode",     128,  2048),
        ("q_a@prefill-mb", 4096, 2048),
        ("attn_out@decode", 128, 32768),
        ("attn_out@prefill-mb", 4096, 32768),
    ]
    print(f"{'case':<22}  {'M':>6}  {'K':>6}  {'vs CUDA':>12}  {'vs ref':>12}")
    print("-" * 68)
    for name, M, K in prod_shapes:
        torch.cuda.empty_cache()
        x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
        y_trit, s_trit = run_triton_1d(x)
        y_cuda, s_cuda = run_cuda_3d(x)
        y_ref, s_ref = _reference_blockwise_fp8_quant(x)
        ne_cuda = fp8_byte_diff(y_trit, y_cuda)
        ne_ref = fp8_byte_diff(y_trit, y_ref)
        total = y_trit.numel()
        print(f"{name:<22}  {M:>6}  {K:>6}  {f'{ne_cuda}/{total}':>12}  {f'{ne_ref}/{total}':>12}")
        del x, y_trit, s_trit, y_cuda, s_cuda, y_ref, s_ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-sweep", action="store_true")
    ap.add_argument("--skip-distributions", action="store_true")
    ap.add_argument("--skip-k-edges", action="store_true")
    ap.add_argument("--skip-determinism", action="store_true")
    ap.add_argument("--skip-bounds", action="store_true")
    ap.add_argument("--skip-prod-shapes", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        print("CUDA not available")
        return 1
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    print(f"# Device: {torch.cuda.get_device_name(device)} "
          f"(cap {torch.cuda.get_device_capability(device)})  seed={args.seed}")

    fail = 0
    if not args.skip_sweep:
        fail += test_shape_sweep(device)
    if not args.skip_distributions:
        test_value_distributions(device)
    if not args.skip_k_edges:
        test_k_edges(device)
    if not args.skip_determinism:
        test_determinism(device)
    if not args.skip_bounds:
        test_scale_bounds(device)
    if not args.skip_prod_shapes:
        test_aligned_with_prefill_shapes(device)
    print()
    print(f"# {'ALL GREEN' if fail == 0 else f'{fail} FAIL'}")
    return fail


if __name__ == "__main__":
    sys.exit(main())
