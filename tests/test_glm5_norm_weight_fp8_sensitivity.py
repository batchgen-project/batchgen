"""Verify whether a RMSNorm weight change is observable downstream through
``act_quant → w8a16_gemm``.

**Hypothesis** (POIS, discussed extensively):
Per-128-block FP8 ``act_quant`` computes ``scale = amax/448``. If the
upstream RMSNorm weight is a **uniform scalar multiplier** ``c*ones``,
then every element of the normed activation is ``c × (same-with-w=ones)``,
so block-amax scales by ``c``, ``scale`` scales by ``c``, and the
FP8-quantized bytes are **identical** (only the scale differs by ``c``).
Downstream ``fp8_gemm_nt`` output then differs ONLY in a uniform ``c``
factor — which softmax absorbs in attention. That would explain why
"loading trained kv_a_layernorm (abs_mean ≈ 0.006)" was silent
end-to-end (L2 output byte-identical pre/post skeleton-load fix).

BUT: if the trained weight is **not** uniform (per-channel variation),
the per-block-amax is not proportional to a single ``c``, FP8 bytes
actually differ element-wise, and ``w8a16_gemm`` output is non-
proportionally different — which SHOULD have been visible in L2 if the
fix is on the hot path.

This test constructs a synthetic layer-0 Q chain (``norm → q_b_proj``
via ``w8a16_gemm``) and runs 5 configurations of the norm weight, then
compares:
  - Post-norm BF16 output.
  - ``act_quant`` FP8 bytes (byte-exact).
  - ``act_quant`` FP32 scale.
  - Final ``w8a16_gemm`` BF16 output.

Configurations:
  A: weight = ones(H)               (reference; default init).
  B: weight = 0.006 * ones(H)       (uniform small scalar; simulates
                                     kv_a abs_mean).
  C: weight = 0.006 * |randn(H)|    (per-channel positive small).
  D: weight = 0.006 * randn(H)      (per-channel mixed sign).
  E: real from /data2/models/zai-org/GLM-5-FP8/ (if accessible).

Expected outcomes:
  A vs B — uniform scalar:
    * post-norm: y_B ≈ 0.006 × y_A (bitwise equivalent under BF16 round).
    * FP8 bytes: IDENTICAL (block-max normalizes away the uniform c).
    * scale: scale_B ≈ 0.006 × scale_A.
    * w8a16_gemm out: out_B ≈ 0.006 × out_A (proportional).
  A vs C/D — per-channel:
    * FP8 bytes DIFFER (per-block amax no longer scales uniformly).
    * w8a16_gemm out not proportional.
  A vs E — real weights (whichever of B/C/D it resembles):
    * If real is ≈ uniform → A vs E uniform-scalar behavior.
    * If real has per-channel variation → A vs E per-channel behavior.

The critical assertion is whether the trained norm weights are
distinguishable from a uniform scalar in the final w8a16_gemm output.
That's what decides whether the silent skeleton-fix is explained by
absorption or indicates a dead branch elsewhere.
"""
import os
import pytest
import torch


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="FP8 kernels require CUDA",
)


def _build_synthetic_fp8_weight(n, k, device, seed):
    """Per-128-block FP8 weight + FP32 scale. Matches the helper in
    test_glm5_fp8_gemm_vs_bf16.py but inlined to keep this test standalone."""
    assert n % 128 == 0 and k % 128 == 0
    g = torch.Generator(device=device).manual_seed(seed)
    w_bf16 = torch.randn((n, k), dtype=torch.bfloat16, device=device, generator=g) * 0.02
    w_fp32 = w_bf16.float()
    n_b, k_b = n // 128, k // 128
    w_fp8 = torch.empty((n, k), dtype=torch.float8_e4m3fn, device=device)
    scale = torch.empty((n_b, k_b), dtype=torch.float32, device=device)
    FP8_MAX = 448.0
    for i in range(n_b):
        for j in range(k_b):
            tile = w_fp32[i * 128:(i + 1) * 128, j * 128:(j + 1) * 128]
            s = tile.abs().max().clamp(min=1e-12) / FP8_MAX
            scale[i, j] = s
            w_fp8[i * 128:(i + 1) * 128, j * 128:(j + 1) * 128] = (
                (tile / s).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
            )
    return w_fp8, scale


def _run_chain(x_bf16, norm, w_fp8, w_scale):
    """Returns (y_norm, y_fp8, y_scale, y_gemm)."""
    from batchgen.attention.mla.fa3_backend import act_quant, w8a16_gemm
    y_norm = norm(x_bf16)
    y_fp8, y_scale = act_quant(y_norm.contiguous())
    y_gemm = w8a16_gemm(w_fp8, w_scale, y_norm)
    return y_norm, y_fp8, y_scale, y_gemm


def _summary(label, y_a, y_b, c_expected=None):
    """FP32 diff summary. If c_expected is given, also measure proportional
    vs non-proportional fit."""
    diff = (y_a.float() - y_b.float()).abs()
    print(
        f"  [{label:18s}] shape={tuple(y_a.shape)} "
        f"max|Δ|={diff.max().item():.4e} mean|Δ|={diff.mean().item():.4e}"
    )
    if c_expected is not None:
        scaled = y_a.float() * c_expected
        prop_diff = (scaled - y_b.float()).abs()
        ref = scaled.abs().clamp(min=1e-6)
        print(
            f"    scaled-by-{c_expected:.4f}: max|Δ|={prop_diff.max().item():.4e} "
            f"mean|Δ|={prop_diff.mean().item():.4e} "
            f"max_rel={((prop_diff / ref).max().item()):.4e}"
        )


@pytest.mark.parametrize("M", [128, 256])  # both dispatch branches
def test_q_a_layernorm_fp8_sensitivity(M):
    """q_a_layernorm output (2048-dim) -> q_b_proj (FP8 gemm 16384<-2048)."""
    from batchgen.models.glm.glm5.model import Glm5RMSNorm

    device = "cuda"
    torch.manual_seed(0xFEED1 + M)

    HIDDEN_IN = 2048                       # q_lora_rank
    HIDDEN_OUT = 64 * 256                  # num_heads * q_head_dim = 16384
    RMS_EPS = 1e-5

    # Random "q_a output" feeding into norm then q_b_proj.
    x = torch.randn(M, HIDDEN_IN, dtype=torch.bfloat16, device=device) * 0.05

    # q_b_proj synthetic FP8 weight (16384x2048).
    qb_fp8, qb_scale = _build_synthetic_fp8_weight(
        HIDDEN_OUT, HIDDEN_IN, device=device, seed=0xABBA,
    )

    def make_norm(weight_tensor):
        n = Glm5RMSNorm(HIDDEN_IN, eps=RMS_EPS).to(device=device, dtype=torch.bfloat16)
        n.weight.data = weight_tensor.to(device=device, dtype=torch.bfloat16).contiguous()
        return n

    # --- Config A: ones (default init) ---
    w_A = torch.ones(HIDDEN_IN, dtype=torch.bfloat16, device=device)
    norm_A = make_norm(w_A)

    # --- Config B: 0.006 * ones (uniform scalar matching kv_a abs_mean) ---
    c_B = 0.006
    w_B = torch.full((HIDDEN_IN,), c_B, dtype=torch.bfloat16, device=device)
    norm_B = make_norm(w_B)

    # --- Config C: 0.006 * |randn| (per-channel positive small) ---
    g_C = torch.Generator(device=device).manual_seed(0xCCCC)
    w_C = (torch.randn(HIDDEN_IN, generator=g_C, device=device).abs() * 0.006).to(torch.bfloat16)
    norm_C = make_norm(w_C)

    # --- Config D: 0.006 * randn (per-channel mixed sign) ---
    g_D = torch.Generator(device=device).manual_seed(0xDDDD)
    w_D = (torch.randn(HIDDEN_IN, generator=g_D, device=device) * 0.006).to(torch.bfloat16)
    norm_D = make_norm(w_D)

    with torch.no_grad():
        y_A_norm, y_A_fp8, y_A_sc, y_A_out = _run_chain(x, norm_A, qb_fp8, qb_scale)
        y_B_norm, y_B_fp8, y_B_sc, y_B_out = _run_chain(x, norm_B, qb_fp8, qb_scale)
        y_C_norm, y_C_fp8, y_C_sc, y_C_out = _run_chain(x, norm_C, qb_fp8, qb_scale)
        y_D_norm, y_D_fp8, y_D_sc, y_D_out = _run_chain(x, norm_D, qb_fp8, qb_scale)

    # -------- Report --------
    print(f"\n=== q_a_layernorm sensitivity test, M={M} ===")

    print("A vs B (uniform scalar c=0.006):")
    _summary("post_norm", y_A_norm, y_B_norm, c_expected=c_B)
    # FP8 byte comparison
    bytes_eq = (y_A_fp8.view(torch.uint8) == y_B_fp8.view(torch.uint8)).all().item()
    byte_mismatch = (y_A_fp8.view(torch.uint8) != y_B_fp8.view(torch.uint8)).sum().item()
    print(
        f"  [act_quant fp8_byte] byte_exact={bytes_eq} mismatched={byte_mismatch}"
        f"/{y_A_fp8.numel()}"
    )
    _summary("act_quant scale", y_A_sc, y_B_sc, c_expected=c_B)
    _summary("w8a16_gemm out", y_A_out, y_B_out, c_expected=c_B)

    print("A vs C (per-channel positive small):")
    _summary("post_norm", y_A_norm, y_C_norm)
    bytes_eq_C = (y_A_fp8.view(torch.uint8) == y_C_fp8.view(torch.uint8)).all().item()
    byte_mismatch_C = (y_A_fp8.view(torch.uint8) != y_C_fp8.view(torch.uint8)).sum().item()
    print(
        f"  [act_quant fp8_byte] byte_exact={bytes_eq_C} mismatched={byte_mismatch_C}"
        f"/{y_A_fp8.numel()}"
    )
    _summary("w8a16_gemm out", y_A_out, y_C_out)

    print("A vs D (per-channel mixed sign):")
    _summary("post_norm", y_A_norm, y_D_norm)
    bytes_eq_D = (y_A_fp8.view(torch.uint8) == y_D_fp8.view(torch.uint8)).all().item()
    byte_mismatch_D = (y_A_fp8.view(torch.uint8) != y_D_fp8.view(torch.uint8)).sum().item()
    print(
        f"  [act_quant fp8_byte] byte_exact={bytes_eq_D} mismatched={byte_mismatch_D}"
        f"/{y_A_fp8.numel()}"
    )
    _summary("w8a16_gemm out", y_A_out, y_D_out)

    # Sanity:
    assert torch.isfinite(y_A_out).all()
    assert torch.isfinite(y_B_out).all()
    assert torch.isfinite(y_C_out).all()
    assert torch.isfinite(y_D_out).all()


@pytest.mark.skipif(
    not os.path.isdir("/data2/models/zai-org/GLM-5-FP8"),
    reason="real GLM-5-FP8 checkpoint not available",
)
def test_real_trained_norm_weights():
    """Load real q_a_layernorm and kv_a_layernorm from checkpoint and
    measure end-to-end chain vs config A (ones). Decides whether real
    weights are observably different from ones after the FP8 path."""
    import json
    from safetensors import safe_open
    from batchgen.models.glm.glm5.model import Glm5RMSNorm

    device = "cuda"
    torch.manual_seed(0xFEED77)
    CKPT = "/data2/models/zai-org/GLM-5-FP8"

    with open(f"{CKPT}/model.safetensors.index.json") as f:
        idx = json.load(f)["weight_map"]

    # Load q_a_layernorm.weight (hidden = q_lora_rank = 2048)
    key_q = "model.layers.0.self_attn.q_a_layernorm.weight"
    with safe_open(f"{CKPT}/{idx[key_q]}", framework="pt") as f:
        w_q = f.get_tensor(key_q).to(device=device)

    # Load kv_a_layernorm.weight (hidden = kv_lora_rank = 512)
    key_kv = "model.layers.0.self_attn.kv_a_layernorm.weight"
    with safe_open(f"{CKPT}/{idx[key_kv]}", framework="pt") as f:
        w_kv = f.get_tensor(key_kv).to(device=device)

    print(f"\n=== real GLM-5-FP8 norm weights, layer 0 ===")
    print(
        f"q_a_layernorm.weight: dtype={w_q.dtype} shape={tuple(w_q.shape)} "
        f"abs_mean={w_q.float().abs().mean().item():.4f} "
        f"std={w_q.float().std().item():.4f} "
        f"min={w_q.float().min().item():.4f} max={w_q.float().max().item():.4f} "
        f"first5={w_q.float()[:5].tolist()}"
    )
    print(
        f"kv_a_layernorm.weight: dtype={w_kv.dtype} shape={tuple(w_kv.shape)} "
        f"abs_mean={w_kv.float().abs().mean().item():.4f} "
        f"std={w_kv.float().std().item():.4f} "
        f"min={w_kv.float().min().item():.4f} max={w_kv.float().max().item():.4f} "
        f"first5={w_kv.float()[:5].tolist()}"
    )

    # Is the real weight near-uniform? Coefficient of variation (std/|mean|)
    q_cv = (w_q.float().std() / w_q.float().abs().mean().clamp(min=1e-9)).item()
    kv_cv = (w_kv.float().std() / w_kv.float().abs().mean().clamp(min=1e-9)).item()
    print(f"q_a coeff_of_variation={q_cv:.4f}, kv_a coeff_of_variation={kv_cv:.4f}")
    print(
        "  (CV < 0.1 → near-uniform → config B-like; "
        "CV > 0.3 → per-channel variation → config C/D-like)"
    )

    # ---- End-to-end comparison: q_a_layernorm chain ----
    M = 256
    HIDDEN_IN = 2048
    HIDDEN_OUT = 64 * 256
    RMS_EPS = 1e-5

    x = torch.randn(M, HIDDEN_IN, dtype=torch.bfloat16, device=device) * 0.05
    qb_fp8, qb_scale = _build_synthetic_fp8_weight(
        HIDDEN_OUT, HIDDEN_IN, device=device, seed=0x1234,
    )

    def make_norm(weight_tensor):
        n = Glm5RMSNorm(HIDDEN_IN, eps=RMS_EPS).to(device=device, dtype=torch.bfloat16)
        n.weight.data = weight_tensor.to(device=device, dtype=torch.bfloat16).contiguous()
        return n

    norm_A = make_norm(torch.ones(HIDDEN_IN, dtype=torch.bfloat16, device=device))
    norm_real = make_norm(w_q)

    with torch.no_grad():
        y_A_norm, y_A_fp8, y_A_sc, y_A_out = _run_chain(x, norm_A, qb_fp8, qb_scale)
        y_R_norm, y_R_fp8, y_R_sc, y_R_out = _run_chain(x, norm_real, qb_fp8, qb_scale)

    c_scalar = w_q.float().abs().mean().item()  # treat as if uniform c
    print(f"\nA vs REAL q_a_layernorm (c_scalar_estimate={c_scalar:.4f}):")
    _summary("post_norm", y_A_norm, y_R_norm)
    bytes_eq = (y_A_fp8.view(torch.uint8) == y_R_fp8.view(torch.uint8)).all().item()
    byte_mismatch = (y_A_fp8.view(torch.uint8) != y_R_fp8.view(torch.uint8)).sum().item()
    print(
        f"  [act_quant fp8_byte] byte_exact={bytes_eq} mismatched={byte_mismatch}"
        f"/{y_A_fp8.numel()} ({100*byte_mismatch/max(1,y_A_fp8.numel()):.3f}%)"
    )
    _summary("w8a16_gemm out", y_A_out, y_R_out, c_expected=c_scalar)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
