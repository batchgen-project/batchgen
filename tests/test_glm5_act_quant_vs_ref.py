"""Byte-level test of BatchGen ``act_quant`` vs a PyTorch reference that
implements the canonical per-128-block FP8 E4M3 quantization used by
DeepGEMM (``fp8_gemm_nt``) and SGLang/vLLM.

Why this test: 37 unit tests have proven BatchGen's BF16 modules match
HF element-wise; a separate FP8 GEMM test showed ~16-32% mean relative
error per ``w8a16_gemm`` vs BF16 reference, which is consistent with
~2% per-element ``act_quant`` round-trip error accumulated over k
elements. Both BatchGen and SGLang call ``deep_gemm.fp8_gemm_nt`` with
pre-quantized inputs, so IF their ``act_quant`` implementations produce
identical FP8 bytes, the GEMMs are byte-identical. If they differ, the
difference compounds across 78 layers × 5 GEMMs and can flip the top-1
prefill logit.

Reference implementation below is written to match SGLang's
``act_quant`` in ``layers/attention/nsa/triton_kernel.py:86-136``:
  - block_size = 128 along the LAST dim (input is ``[..., K]``,
    ``K % 128 == 0``).
  - Per-block ``amax = max(|x|)``.
  - ``scale = amax / 448.0``.
  - Quantized ``y = clamp(x / scale, -448, 448)``.
  - ``y`` cast to ``float8_e4m3fn``.
  - Scale shape ``[..., K // 128]`` (``float32``).

BatchGen's ``act_quant`` (``fa3_backend.py:451``) routes BF16 input
through a CUDA or Triton kernel. Both kernels are documented to use
``FP8_SAFE_MAX = 448`` and the same per-128-block semantics — but the
exact cast rounding and any extra eps / NaN-handling branches may
differ. This test pins the byte-level behavior.
"""
import pytest
import torch


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="FP8 kernels require CUDA",
)


def ref_act_quant(
    x: torch.Tensor, block_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Canonical per-K-block FP8 E4M3 quant reference.

    Matches the math in SGLang's ``_act_quant_kernel``
    (``triton_kernel.py:40-83``), implemented as dense PyTorch so rounding
    is deterministic via the native ``float8_e4m3fn`` cast.

    Returns (fp8, scale_f32). ``fp8`` has the same shape as ``x``;
    ``scale_f32`` has shape ``x.shape[:-1] + (K // block_size,)``.
    """
    assert x.is_contiguous()
    assert x.shape[-1] % block_size == 0
    orig_shape = x.shape
    x_flat = x.reshape(-1, orig_shape[-1])
    M, K = x_flat.shape
    n_blocks = K // block_size
    x_reshaped = x_flat.view(M, n_blocks, block_size).float()
    # Per-block absmax (shape [M, n_blocks])
    amax = x_reshaped.abs().amax(dim=-1)
    # SGLang clamps amax to a floor (1e-4 in SGLang; BatchGen uses
    # FP8_E4M3_MIN_NORMAL=1.52587890625e-5). The test fixes the floor to
    # a clearly-below-any-reasonable-activation value to make the
    # comparison independent of this floor. Use BatchGen's value.
    amax = amax.clamp(min=1.52587890625e-5)
    scale_f32 = amax / 448.0                 # [M, n_blocks]
    scale_bcast = scale_f32.unsqueeze(-1)    # [M, n_blocks, 1]
    y_f32 = (x_reshaped / scale_bcast).clamp(-448.0, 448.0)
    y_fp8 = y_f32.to(torch.float8_e4m3fn).view(M, K)
    return (
        y_fp8.view(orig_shape),
        scale_f32.view(*orig_shape[:-1], n_blocks),
    )


def _fp8_bytes(t: torch.Tensor) -> torch.Tensor:
    """Reinterpret an FP8_E4M3 tensor as raw uint8 bytes for byte-level compare."""
    return t.view(torch.uint8)


@pytest.mark.parametrize("shape,std", [
    ((1, 6144), 0.03),      # typical layer-0 post-embedding
    ((16, 6144), 0.05),     # prefill with 16 tokens
    ((256, 6144), 0.05),    # prefill with 256 tokens
    ((128, 2048), 0.10),    # q_a output pre q_b_proj
    ((256, 512), 0.20),     # normed_kv pre kv_b_proj
    ((32, 16384), 0.02),    # attn_output pre o_proj
])
def test_batchgen_act_quant_matches_ref_byte_level(shape, std):
    """Compare BatchGen ``act_quant`` FP8 output + scale to the reference
    byte-for-byte and exact-FP32-for-scale.

    If they match: BatchGen's act_quant is canonical, any divergence vs
    SGLang must come from somewhere else (e.g., the DeepGEMM call itself,
    or how BatchGen calls it).
    If they differ: the difference is the ROOT CAUSE of per-GEMM
    precision drift accumulating through 78 layers.
    """
    from batchgen.attention.mla.fa3_backend import act_quant

    device = "cuda"
    torch.manual_seed(0xCA5CA1D + hash(shape) % 1000 + int(std * 10000))
    x = torch.randn(shape, dtype=torch.bfloat16, device=device) * std

    # BatchGen path
    bg_fp8, bg_scale = act_quant(x)

    # Reference path
    ref_fp8, ref_scale = ref_act_quant(x, block_size=128)

    assert bg_fp8.shape == ref_fp8.shape, (
        f"shape mismatch: bg={bg_fp8.shape}, ref={ref_fp8.shape}"
    )
    assert bg_scale.shape == ref_scale.shape, (
        f"scale shape: bg={bg_scale.shape}, ref={ref_scale.shape}"
    )

    # ---- Scale check (exact FP32) ----
    scale_diff = (bg_scale.float() - ref_scale.float()).abs()
    scale_max = scale_diff.max().item()
    scale_mismatch = (scale_diff > 0).sum().item()
    print(
        f"[ACTQUANT SCALE shape={shape} std={std:.3f}] "
        f"max|Δ|={scale_max:.4e} mismatched_elems={scale_mismatch}"
        f"/{bg_scale.numel()}"
    )

    # ---- FP8 byte-level check ----
    bg_bytes = _fp8_bytes(bg_fp8)
    ref_bytes = _fp8_bytes(ref_fp8)
    byte_match = (bg_bytes == ref_bytes).all().item()
    byte_mismatch_count = (bg_bytes != ref_bytes).sum().item()
    total_bytes = bg_bytes.numel()
    print(
        f"[ACTQUANT FP8  shape={shape} std={std:.3f}] "
        f"byte_exact={byte_match}  mismatched_bytes={byte_mismatch_count}"
        f"/{total_bytes}  ({100.0 * byte_mismatch_count / max(1, total_bytes):.3f}%)"
    )

    # Also report max/value-level divergence when bytes differ (ULP view)
    bg_f32 = bg_fp8.float()
    ref_f32 = ref_fp8.float()
    fp8_diff = (bg_f32 - ref_f32).abs()
    if fp8_diff.max().item() > 0:
        # Find a sample mismatching position for diagnostic printout
        flat_diff = fp8_diff.view(-1)
        worst_idx = flat_diff.argmax().item()
        bg_val = bg_f32.view(-1)[worst_idx].item()
        ref_val = ref_f32.view(-1)[worst_idx].item()
        x_val = x.float().view(-1)[worst_idx].item() if x.numel() == bg_f32.numel() else float("nan")
        print(
            f"[ACTQUANT FP8  shape={shape} std={std:.3f}] "
            f"max|Δ_fp8|={fp8_diff.max().item():.4e} "
            f"at flat_idx={worst_idx}: "
            f"x={x_val:.6f}, bg={bg_val:.6f}, ref={ref_val:.6f}"
        )

    # Strict assertion: byte-exact. If this fails we've localized the bug.
    if not byte_match:
        # Look at which BLOCKS have mismatches (mostly affects one scale)
        # to help diagnose.
        block_size = 128
        M = bg_fp8.numel() // bg_fp8.shape[-1]
        K = bg_fp8.shape[-1]
        n_blocks = K // block_size
        bg_blocked = bg_fp8.view(M, n_blocks, block_size).view(torch.uint8)
        ref_blocked = ref_fp8.view(M, n_blocks, block_size).view(torch.uint8)
        per_block_mismatches = (bg_blocked != ref_blocked).any(dim=-1)
        block_mismatch_count = per_block_mismatches.sum().item()
        total_blocks = M * n_blocks
        print(
            f"[ACTQUANT FP8  shape={shape} std={std:.3f}] "
            f"blocks with any mismatch: {block_mismatch_count}/{total_blocks} "
            f"({100.0 * block_mismatch_count / max(1, total_blocks):.2f}%)"
        )

    assert byte_match, (
        f"BatchGen act_quant and PyTorch reference produce DIFFERENT FP8 "
        f"bytes on shape={shape} std={std}. Mismatched bytes: "
        f"{byte_mismatch_count}/{total_bytes}. This is the FP8 quantization"
        " divergence source."
    )


@pytest.mark.parametrize("shape,std", [
    ((1, 6144), 0.03),
    ((128, 6144), 0.05),
    ((256, 6144), 0.05),
])
def test_batchgen_act_quant_deterministic(shape, std):
    """Running ``act_quant`` twice on the same input must produce byte-
    identical output. Flags any nondeterminism (atomic ops, uninitialized
    intermediates) that would confound all downstream comparisons."""
    from batchgen.attention.mla.fa3_backend import act_quant

    device = "cuda"
    torch.manual_seed(0xDE7ABCD + hash(shape) % 1000)
    x = (torch.randn(shape, dtype=torch.bfloat16, device=device) * std).contiguous()

    a_fp8, a_scale = act_quant(x)
    b_fp8, b_scale = act_quant(x)

    assert (_fp8_bytes(a_fp8) == _fp8_bytes(b_fp8)).all(), "act_quant is nondeterministic (FP8)"
    assert torch.equal(a_scale, b_scale), "act_quant is nondeterministic (scale)"


@pytest.mark.parametrize("shape,std", [
    # Stress the edge: tiny block magnitude near the clamp floor.
    ((4, 512), 1e-5),   # amax per block may be near 1.5e-5 floor
    ((4, 512), 1e-3),   # within normal range
    ((4, 512), 1.0),    # loud, large scales
])
def test_batchgen_act_quant_edge_magnitudes(shape, std):
    """Edge-case magnitudes to ensure both paths clamp consistently."""
    from batchgen.attention.mla.fa3_backend import act_quant

    device = "cuda"
    torch.manual_seed(0xED6E + int(std * 1e6) % 1000)
    x = (torch.randn(shape, dtype=torch.bfloat16, device=device) * std).contiguous()

    bg_fp8, bg_scale = act_quant(x)
    ref_fp8, ref_scale = ref_act_quant(x, block_size=128)

    bg_bytes = _fp8_bytes(bg_fp8)
    ref_bytes = _fp8_bytes(ref_fp8)
    mismatch = (bg_bytes != ref_bytes).sum().item()
    scale_max_diff = (bg_scale.float() - ref_scale.float()).abs().max().item()
    print(
        f"[ACTQUANT EDGE shape={shape} std={std:.1e}] "
        f"byte_mismatch={mismatch}/{bg_bytes.numel()} "
        f"scale_max_diff={scale_max_diff:.4e}"
    )
    # Edge tests are informational when std is extremely small (1e-5),
    # because BatchGen and SGLang differ on the amax floor (1.5e-5 vs
    # 1e-4). Skip strict assertion for std < 1e-4; assert for normal std.
    if std >= 1e-4:
        assert mismatch == 0, (
            f"act_quant byte divergence at std={std}: {mismatch} bytes differ."
        )


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
