"""Quantify FP8 act_quant + DeepGEMM precision loss vs plain BF16 matmul.

Prefill GEMMs in BatchGen (``batchgen/attention/mla/fa3_backend.py``) go
through ``w8a16_gemm`` which:
  1. ``act_quant`` the activation (BF16) to FP8 E4M3 with per-128-block
     FP32 scales.
  2. Calls ``deep_gemm.fp8_gemm_nt(x_fp8, y_fp8, out)`` where ``y_fp8`` is
     the (pre-quantized) FP8 weight + blockwise FP32 scale.
  3. Returns BF16 output.

HF's reference pipeline uses plain ``nn.Linear`` (BF16 weight × BF16
input → BF16 output). The unit tests elsewhere in ``tests/`` have proven
every BatchGen module matches HF element-wise WHEN plain BF16 linears
are used. The FP8 GEMM path is the only code path that our unit tests
don't touch.

This test file quantifies the per-GEMM precision loss of
``w8a16_gemm`` vs a ground-truth BF16 matmul on dequantized weights:

  ref = F.linear(x_bf16, dequant(W_fp8, scale))  # BF16 matmul
  out = w8a16_gemm(W_fp8, scale, x_bf16)         # FP8 act_quant + DeepGEMM

If the FP8 quantization path drifts significantly from the BF16
reference on typical GLM-5 weight shapes, the cumulative 78-layer × 5
GEMM/layer composition would shift the prefill last-token logits enough
to flip ``The`` → ``#`` on seq 0 despite identical weights and input.

Requires CUDA + DeepGEMM installed (the standard BatchGen runtime).
"""
import math
import pytest
import torch


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="FP8 kernels require CUDA (SM90+ for DeepGEMM fp8_gemm_nt)",
)


# Realistic GLM-5-FP8 projection shapes, mirroring configuration_glm5.py:
#   hidden_size=6144, q_lora_rank=2048, kv_lora_rank=512, qk_rope_head_dim=64,
#   qk_nope_head_dim=192, v_head_dim=256, num_heads=64.
#
# Each entry is (name, weight_shape_as_nn_Linear_stores_it [out, in]).
# The linear's weight shape is [out_features, in_features]; dims must be
# multiples of 128 for blockwise FP8.
GLM5_PROJECTION_SHAPES = [
    # q_a_proj:      hidden -> q_lora_rank        (Linear: out=q_lora, in=hidden)
    ("q_a_proj",              2048, 6144),
    # q_b_proj:      q_lora -> num_heads*q_head_dim=64*256=16384
    ("q_b_proj",             16384, 2048),
    # kv_a_proj:     hidden -> kv_lora+qk_rope=576
    ("kv_a_proj_with_mqa",     576, 6144),
    # kv_b_proj:     kv_lora=512 -> num_heads*(qk_nope+v_head)=64*(192+256)=64*448=28672
    ("kv_b_proj",            28672, 512),
    # o_proj:        num_heads*v_head=64*256=16384 -> hidden=6144
    ("o_proj",                6144, 16384),
    # mlp.gate/up:   hidden -> intermediate_size=12288
    ("mlp.gate_proj",        12288, 6144),
    ("mlp.up_proj",          12288, 6144),
    # mlp.down:      intermediate -> hidden
    ("mlp.down_proj",         6144, 12288),
]


def _build_synthetic_fp8_weight(n: int, k: int, device: str, seed: int):
    """Build a synthetic FP8 weight + blockwise FP32 scale that dequantizes
    to a BF16 weight with realistic magnitude (std≈0.02 so the product
    ``W @ x`` for typical unit-variance x stays bounded).

    Returns ``(weight_fp8, weight_scale_inv_fp32, weight_bf16_dequantized)``.
    """
    assert n % 128 == 0 and k % 128 == 0, f"Shape ({n},{k}) must be multiples of 128"
    g = torch.Generator(device=device).manual_seed(seed)
    # Generate a ground-truth BF16 weight with typical transformer scale.
    weight_bf16 = torch.randn(
        (n, k), dtype=torch.bfloat16, device=device, generator=g
    ) * 0.02
    # Per-128-block quantization. For each 128×128 tile, compute a scale
    # = max(|tile|) / 448 (matches DeepGEMM blockwise FP8 E4M3 semantics).
    n_blocks = n // 128
    k_blocks = k // 128
    weight_fp32 = weight_bf16.float()
    weight_fp8 = torch.empty((n, k), dtype=torch.float8_e4m3fn, device=device)
    weight_scale_inv = torch.empty(
        (n_blocks, k_blocks), dtype=torch.float32, device=device
    )
    FP8_MAX = 448.0
    for i in range(n_blocks):
        for j in range(k_blocks):
            tile = weight_fp32[i * 128 : (i + 1) * 128, j * 128 : (j + 1) * 128]
            tile_max = tile.abs().max().clamp(min=1e-12)
            scale = tile_max / FP8_MAX
            weight_scale_inv[i, j] = scale
            tile_quant = (tile / scale).clamp(-FP8_MAX, FP8_MAX)
            weight_fp8[i * 128 : (i + 1) * 128, j * 128 : (j + 1) * 128] = \
                tile_quant.to(torch.float8_e4m3fn)
    return weight_fp8, weight_scale_inv, weight_bf16


def _dequant_fp8_to_bf16_ref(weight_fp8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Pure PyTorch reference for blockwise FP8 → BF16 dequantization."""
    n, k = weight_fp8.shape
    assert n % 128 == 0 and k % 128 == 0
    n_blocks, k_blocks = scale.shape
    assert n_blocks == n // 128 and k_blocks == k // 128
    w_f32 = weight_fp8.float()
    # Expand scale [n_blocks, k_blocks] -> [n, k]
    scale_full = scale.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
    return (w_f32 * scale_full).to(torch.bfloat16)


@pytest.mark.parametrize("name,n,k", GLM5_PROJECTION_SHAPES)
@pytest.mark.parametrize("m", [1, 16, 256, 1024])  # token-count stress
def test_fp8_gemm_vs_bf16_reference(name, n, k, m):
    """Element-wise compare w8a16_gemm FP8 path vs F.linear(x, W_bf16) ref.

    For each GLM-5 projection shape and each batch-of-tokens size:
      - Synthesize a well-scaled FP8 weight + FP32 blockwise scale.
      - Compute BF16 dequantized ground-truth weight.
      - Apply ``w8a16_gemm`` and plain BF16 matmul on the SAME BF16 input.
      - Report max|diff|, mean|diff|, and relative error.

    This test does not ASSERT a specific tolerance — it QUANTIFIES the
    precision loss of the FP8 path. Output is printed; use this to decide
    whether the FP8 path alone explains the observed prefill divergence.
    """
    from batchgen.attention.mla.fa3_backend import w8a16_gemm
    import torch.nn.functional as F

    device = "cuda"
    torch.manual_seed(0x51234 + hash(name) % 1000 + m)

    weight_fp8, weight_scale_inv, _ = _build_synthetic_fp8_weight(
        n=n, k=k, device=device, seed=0x9273 + hash(name) % 1000
    )
    # Dequantize ourselves (matches what the reference matmul sees).
    weight_bf16 = _dequant_fp8_to_bf16_ref(weight_fp8, weight_scale_inv)

    x = torch.randn((m, k), dtype=torch.bfloat16, device=device)

    # FP8 path (act_quant + DeepGEMM fp8_gemm_nt)
    out_fp8 = w8a16_gemm(weight_fp8, weight_scale_inv, x)

    # BF16 reference (matches HF plain nn.Linear with dequantized weights)
    out_ref = F.linear(x, weight_bf16)

    assert out_fp8.shape == out_ref.shape, (
        f"[{name}] shape mismatch: fp8={out_fp8.shape}, ref={out_ref.shape}"
    )

    # Compare
    diff = (out_fp8.float() - out_ref.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    abs_ref = out_ref.float().abs().clamp(min=1e-6)
    rel_err_max = (diff / abs_ref).max().item()
    rel_err_mean = (diff / abs_ref).mean().item()
    ref_max = out_ref.float().abs().max().item()
    ref_mean = out_ref.float().abs().mean().item()

    # Print-only diagnostic (always passes; use -s flag to see output)
    print(
        f"[FP8-vs-BF16 {name:22s} m={m:4d} n={n:5d} k={k:5d}] "
        f"max|Δ|={max_diff:.4e} mean|Δ|={mean_diff:.4e} "
        f"relmax={rel_err_max:.4e} relmean={rel_err_mean:.4e} "
        f"|ref|max={ref_max:.4e} |ref|mean={ref_mean:.4e}"
    )

    # Hard assert only on NaN / Inf — anything numeric is a measurement.
    assert torch.isfinite(out_fp8).all(), f"[{name}] FP8 path produced NaN/Inf"
    assert torch.isfinite(out_ref).all(), f"[{name}] BF16 ref has NaN/Inf (setup bug)"


@pytest.mark.parametrize("m", [1, 128, 1024])
def test_fp8_gemm_chain_composition(m):
    """Stress test: simulate the 5-GEMM-per-layer composition.

    Chain (layer 0 pass, simplified):
      hidden (6144)
       → q_a_proj → q_b_proj (Q)
       → kv_a_proj → kv_b_proj (KV)
       → o_proj (hidden)

    Compare FP8-path vs BF16-path output after 5 chained FP8 GEMMs. This
    quantifies compounded error that propagates through a full attention
    block's linear ops.
    """
    from batchgen.attention.mla.fa3_backend import w8a16_gemm
    import torch.nn.functional as F

    device = "cuda"
    torch.manual_seed(0xABCDE + m)

    hidden = 6144
    q_lora = 2048
    num_heads = 64
    q_head_dim = 256
    kv_lora = 512
    qk_rope = 64
    v_head = 256

    # Build a chain of 5 GEMMs (matching GLM-5 layer-0 linear ops)
    chain = [
        ("q_a_proj",            q_lora,                  hidden),
        ("q_b_proj",            num_heads * q_head_dim,  q_lora),
        ("kv_a_proj_with_mqa",  kv_lora + qk_rope,       hidden),
        # Skip kv_b for chain simplicity — just o_proj on pretend-attn-out
        ("o_proj",              hidden,                  num_heads * v_head),
    ]

    # Synthesize weights once; use same for both paths.
    weights = []
    for i, (name, n, k) in enumerate(chain):
        w_fp8, scale, _ = _build_synthetic_fp8_weight(n=n, k=k, device=device, seed=0x77 + i)
        w_bf16 = _dequant_fp8_to_bf16_ref(w_fp8, scale)
        weights.append((name, w_fp8, scale, w_bf16))

    # Starting input: hidden state at layer 0 (post-embedding BF16 ~0.03 range)
    x = torch.randn((m, hidden), dtype=torch.bfloat16, device=device) * 0.03
    fp8_x = x.clone()
    bf16_x = x.clone()

    for i, (name, w_fp8, scale, w_bf16) in enumerate(weights):
        # Each op's input needs dim-k to match this layer's k.
        # For q_a_proj + kv_a_proj + o_proj → input dim is `k` of that weight.
        # Chain q_a_proj(hidden) → q_b_proj(q_lora) → ... so after q_a_proj,
        # feed q_b_proj with the Q output. But we also do kv_a_proj
        # independently from the same hidden. For this chain test we just
        # run the 4 projections SEQUENTIALLY on synthetic state (not a
        # faithful GLM-5 forward; purpose is to measure FP8 precision
        # accumulation across multiple GEMMs).
        if fp8_x.shape[-1] != w_fp8.shape[-1]:
            # Pad/truncate to match — this is synthetic, not a real forward.
            new_k = w_fp8.shape[-1]
            fp8_x = fp8_x[..., :new_k] if fp8_x.shape[-1] > new_k else \
                torch.nn.functional.pad(fp8_x, (0, new_k - fp8_x.shape[-1]))
            bf16_x = bf16_x[..., :new_k] if bf16_x.shape[-1] > new_k else \
                torch.nn.functional.pad(bf16_x, (0, new_k - bf16_x.shape[-1]))
        fp8_x = w8a16_gemm(w_fp8, scale, fp8_x)
        bf16_x = F.linear(bf16_x, w_bf16)
        step_diff = (fp8_x.float() - bf16_x.float()).abs()
        print(
            f"  [chain step {i+1} {name:22s}] shape={tuple(fp8_x.shape)} "
            f"max|Δ|={step_diff.max().item():.4e} "
            f"mean|Δ|={step_diff.mean().item():.4e} "
            f"|ref|max={bf16_x.float().abs().max().item():.4e} "
            f"|ref|mean={bf16_x.float().abs().mean().item():.4e}"
        )
        assert torch.isfinite(fp8_x).all(), f"FP8 chain step {i+1} {name} NaN/Inf"


@pytest.mark.parametrize("shape_scale", [
    # (n, k, weight_std) — tests sensitivity to weight magnitude
    (2048, 6144, 0.02),    # typical transformer scale
    (2048, 6144, 0.01),    # tighter (sharper distribution)
    (2048, 6144, 0.05),    # looser
])
def test_fp8_act_quant_vs_bf16_precision_floor(shape_scale):
    """Probe the precision floor of act_quant alone (no matmul).

    Quantizes a BF16 activation to FP8 via ``act_quant``, then dequantizes
    back to BF16 (by multiplying with the returned scale). Reports how
    much the round-trip differs from the original — a lower bound on the
    per-activation precision loss that ``w8a16_gemm`` incurs.
    """
    from batchgen.attention.mla.fa3_backend import act_quant

    device = "cuda"
    n, k, std = shape_scale
    torch.manual_seed(0x123FEED + int(std * 10000))

    x = torch.randn((n, k), dtype=torch.bfloat16, device=device) * std
    x_fp8, x_scale = act_quant(x)
    # x_fp8: FP8 activation, shape [n, k]. x_scale: blockwise FP32 scales.
    # Dequantize: scale applies per 128-element block along last dim.
    # Shape of x_scale should be [n, k/128].
    assert x_scale.shape[-1] == k // 128, (
        f"act_quant scale shape unexpected: got {x_scale.shape}, "
        f"expected [..., {k // 128}] for k={k}"
    )
    # Expand scale to [n, k]
    x_fp8_f32 = x_fp8.float()
    x_scale_expanded = x_scale.repeat_interleave(128, dim=-1)
    x_roundtrip = (x_fp8_f32 * x_scale_expanded).to(torch.bfloat16)

    diff = (x.float() - x_roundtrip.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    abs_x = x.float().abs().clamp(min=1e-6)
    rel_err_max = (diff / abs_x).max().item()
    rel_err_mean = (diff / abs_x).mean().item()

    print(
        f"[act_quant round-trip n={n} k={k} std={std:.3f}] "
        f"max|Δ|={max_diff:.4e} mean|Δ|={mean_diff:.4e} "
        f"relmax={rel_err_max:.4e} relmean={rel_err_mean:.4e}"
    )

    assert torch.isfinite(x_fp8_f32).all(), "FP8 activation produced NaN"
    assert torch.isfinite(x_roundtrip).all(), "Round-trip produced NaN"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
