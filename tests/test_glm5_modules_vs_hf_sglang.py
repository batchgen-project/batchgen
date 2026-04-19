"""Incremental per-module prefill bisect: BatchGen vs HF vs SGLang.

Goal: find the first layer-0 prefill module whose output diverges between
BatchGen and the HF+SGLang reference when fed IDENTICAL BF16 inputs and
IDENTICAL BF16 dequantized weights from the same GLM-5-FP8 checkpoint.

Plan-of-record:
  /home/tairan/.claude/plans/velvet-popping-hopper.md
Prior trace notes + decisions:
  /home/tairan/workspace/glm5-debug/2026-04-19_prefill-divergence-trace.md

Run (on H20, inside the sglang conda env that also has batchgen):
  python3 -m pytest tests/test_glm5_modules_vs_hf_sglang.py -v -s

The test is built incrementally — Phase 0 foundation + Step 1 (embedding)
first, additional steps added as each prior one passes.
"""
from __future__ import annotations

import json
import math
import os
import sys

# Make the `batchgen` package importable when the test is invoked as a
# standalone script (`python3 tests/xxx.py`) rather than via pytest. Adds
# the repo root (`BatchGen/`) to sys.path so `from batchgen.foo import bar`
# resolves.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
try:
    import pytest
    _HAS_PYTEST = True
except ImportError:
    # Fallback shim so the file still parses without pytest installed.
    # The __main__ block drives tests directly in that case.
    _HAS_PYTEST = False
    class _PytestShim:
        mark = type("mark", (), {
            "skipif": lambda *a, **k: (lambda fn: fn),
        })()
        def fixture(*a, **k):
            def _deco(fn): return fn
            return _deco
    pytest = _PytestShim()  # type: ignore
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Optional


# ============================================================================
# Constants: checkpoint on H20
# ============================================================================

MODEL_DIR = Path("/data2/models/zai-org/GLM-5-FP8")
# Probe tokens — start tiny so shapes stay small: 16-token "sentence" whose
# IDs don't actually matter for correctness of the per-module test (we only
# need all engines to process the SAME IDs). We pick arbitrary legal IDs.
PROBE_TOKEN_IDS = torch.tensor(
    [[151, 262, 333, 4412, 8199, 23456, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]],
    dtype=torch.long,
)

# BF16 tolerance. RMSNorm / linear on BF16 can drift by ~1e-2 relative on
# larger magnitudes; ULP at unit scale is ~4e-3.
DEFAULT_RTOL = 1e-2
DEFAULT_ATOL = 1e-2


# ============================================================================
# Environment checks — the test needs CUDA, the real checkpoint on disk,
# and SGLang importable. Skip gracefully if any are missing.
# ============================================================================

_HAS_CUDA = torch.cuda.is_available()
_HAS_CKPT = MODEL_DIR.exists() and (MODEL_DIR / "model.safetensors.index.json").exists()

_HAS_SGLANG = True  # pure-Python inline — no sglang import needed


def block_quant_dequant(
    x_q_block: torch.Tensor,
    x_s: torch.Tensor,
    block_size,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Inlined copy of SGLang's `sglang.srt.layers.quantization.fp8_utils.block_quant_dequant`
    (20 lines, pure-PyTorch — no sgl_kernel / sglang-env dependency).

    Converts block-wise FP8 quant → dense BF16/FP32 tensor: repeat-interleave
    the per-block scale over (block_n, block_k) and cast.
    """
    block_n, block_k = block_size[0], block_size[1]
    *_, n, k = x_q_block.shape
    x_scale_repeat = x_s.repeat_interleave(block_n, dim=-2).repeat_interleave(
        block_k, dim=-1
    )
    x_scale_repeat = x_scale_repeat[..., :n, :k]
    return (x_q_block.to(torch.float32) * x_scale_repeat).to(dtype)

pytestmark = [
    pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required"),
    pytest.mark.skipif(not _HAS_CKPT, reason=f"GLM-5-FP8 checkpoint not at {MODEL_DIR}"),
]


# ============================================================================
# Phase 0: weight loader + input fixture
# ============================================================================

_safetensors_index_cache: Optional[Dict[str, str]] = None


def _get_weight_map() -> Dict[str, str]:
    """Load safetensors weight-map once."""
    global _safetensors_index_cache
    if _safetensors_index_cache is None:
        with open(MODEL_DIR / "model.safetensors.index.json") as f:
            _safetensors_index_cache = json.load(f)["weight_map"]
    return _safetensors_index_cache


def _load_tensor(name: str) -> torch.Tensor:
    """Pull a single named tensor from its shard."""
    from safetensors import safe_open
    wmap = _get_weight_map()
    shard = wmap[name]
    with safe_open(str(MODEL_DIR / shard), framework="pt") as f:
        return f.get_tensor(name)


def _dequant_fp8_weight(
    weight_name: str, block: int = 128
) -> torch.Tensor:
    """Load an FP8 blockwise weight + its scale, dequant to BF16.

    Matches SGLang's `block_quant_dequant(weight, scale, (128,128), bf16)`
    semantics. Returns BF16 tensor shaped as the original weight.
    """
    w_fp8 = _load_tensor(weight_name)
    scale_name = weight_name + "_scale_inv"
    if scale_name in _get_weight_map():
        scale = _load_tensor(scale_name).to(torch.float32)
        return block_quant_dequant(
            w_fp8, scale, (block, block), torch.bfloat16,
        )
    # Not FP8 (e.g. layer norms, embed_tokens): return as-is (BF16).
    return w_fp8.to(torch.bfloat16) if w_fp8.dtype != torch.bfloat16 else w_fp8


@pytest.fixture(scope="session")
def ckpt_cfg() -> Dict:
    """Load `config.json` so we know vocab_size, hidden_size, rms_norm_eps, etc."""
    with open(MODEL_DIR / "config.json") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def device() -> torch.device:
    return torch.device("cuda")


@pytest.fixture(scope="session")
def probe_ids(device) -> torch.Tensor:
    return PROBE_TOKEN_IDS.to(device)


# ============================================================================
# Step 1 — `embed_tokens` (layer 0 is trivial: just an Embedding lookup)
# ============================================================================

@pytest.fixture(scope="session")
def embed_weight(device) -> torch.Tensor:
    """Layer-0 embed_tokens weight — BF16 on disk per `modules_to_not_convert`."""
    w = _load_tensor("model.embed_tokens.weight").to(device)
    assert w.dtype == torch.bfloat16, f"expected BF16 embed_tokens, got {w.dtype}"
    return w


def _bg_embed(probe_ids, embed_weight, cfg, device):
    """BatchGen Glm5Model's embed_tokens is a plain nn.Embedding (model.py:1997).
    No need to import Glm5Model for this — `nn.Embedding` constructor is
    already identical on BatchGen's side. We just re-create it with the same
    args + same weight to isolate the lookup math."""
    emb = nn.Embedding(cfg["vocab_size"], cfg["hidden_size"], cfg["pad_token_id"]).to(device)
    with torch.no_grad():
        emb.weight.copy_(embed_weight)
    return emb(probe_ids).to(torch.bfloat16)


def _hf_embed(probe_ids, embed_weight, cfg, device):
    """HF GlmMoeDsaModel's embed_tokens is also nn.Embedding (modeling_glm_moe_dsa.py:738)."""
    emb = nn.Embedding(cfg["vocab_size"], cfg["hidden_size"], cfg["pad_token_id"]).to(device)
    with torch.no_grad():
        emb.weight.copy_(embed_weight)
    return emb(probe_ids).to(torch.bfloat16)


def _sgl_embed(probe_ids, embed_weight, cfg, device):
    """SGLang's glm4_moe.py uses VocabParallelEmbedding. At TP=1 its forward
    is a plain lookup identical to nn.Embedding."""
    emb = nn.Embedding(cfg["vocab_size"], cfg["hidden_size"], cfg["pad_token_id"]).to(device)
    with torch.no_grad():
        emb.weight.copy_(embed_weight)
    return emb(probe_ids).to(torch.bfloat16)


# ============================================================================
# Comparator: emit BatchGen-vs-HF / SGLang-vs-HF / BatchGen-vs-SGLang stats
# ============================================================================

def _diff_stats(a: torch.Tensor, b: torch.Tensor) -> Dict:
    af = a.detach().float()
    bf = b.detach().float()
    err = (af - bf).abs()
    ref = bf.abs().clamp(min=1e-6)
    max_abs = err.max().item()
    max_rel = (err / ref).max().item()
    mean_abs = err.mean().item()
    argmax = err.reshape(-1).argmax().item()
    return dict(max_abs=max_abs, max_rel=max_rel, mean_abs=mean_abs, argmax=argmax)


def _emit_verdict(module_name: str, bg, hf, sgl, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL):
    """Print the per-module verdict block + return (bg_matches_hf, sgl_matches_hf)."""
    bg_vs_hf = _diff_stats(bg, hf)
    sgl_vs_hf = _diff_stats(sgl, hf)
    bg_vs_sgl = _diff_stats(bg, sgl)
    bg_ok = bg_vs_hf["max_abs"] <= atol and bg_vs_hf["max_rel"] <= rtol
    sgl_ok = sgl_vs_hf["max_abs"] <= atol and sgl_vs_hf["max_rel"] <= rtol
    if bg_ok and sgl_ok:
        verdict = "match"
    elif bg_ok and not sgl_ok:
        verdict = "SGL_diverges"
    elif not bg_ok and sgl_ok:
        verdict = "BG_diverges"
    else:
        verdict = "both_diverge"
    print(f"\n=== Module: {module_name} ===")
    print(f"  shapes: bg={tuple(bg.shape)} hf={tuple(hf.shape)} sgl={tuple(sgl.shape)}")
    print(f"  BatchGen vs HF:    max_abs={bg_vs_hf['max_abs']:.4e}  "
          f"max_rel={bg_vs_hf['max_rel']:.4e}  mean={bg_vs_hf['mean_abs']:.4e}")
    print(f"  SGLang   vs HF:    max_abs={sgl_vs_hf['max_abs']:.4e}  "
          f"max_rel={sgl_vs_hf['max_rel']:.4e}  mean={sgl_vs_hf['mean_abs']:.4e}")
    print(f"  BatchGen vs SGLang:max_abs={bg_vs_sgl['max_abs']:.4e}  "
          f"max_rel={bg_vs_sgl['max_rel']:.4e}")
    print(f"  VERDICT: {verdict}")
    return bg_ok, sgl_ok, verdict


# ============================================================================
# Tests (one per module; later modules added incrementally)
# ============================================================================

def test_step1_embed_tokens(probe_ids, embed_weight, ckpt_cfg, device):
    """Step 1: embed_tokens. All three engines should produce identical output."""
    bg = _bg_embed(probe_ids, embed_weight, ckpt_cfg, device)
    hf = _hf_embed(probe_ids, embed_weight, ckpt_cfg, device)
    sgl = _sgl_embed(probe_ids, embed_weight, ckpt_cfg, device)
    bg_ok, sgl_ok, verdict = _emit_verdict("embed_tokens", bg, hf, sgl)
    # Embedding is a lookup — should be bit-exact.
    assert verdict == "match", f"unexpected verdict at embed_tokens: {verdict}"


# ============================================================================
# Step 2 — `input_layernorm` (Glm5RMSNorm / RMSNorm / GlmMoeDsaRMSNorm)
# ============================================================================

class _HfRMSNorm(nn.Module):
    """Inlined copy of HF's GlmMoeDsaRMSNorm (modeling_glm_moe_dsa.py:47-65)
    — cast-then-weight semantics: FP32 variance, cast hidden back to input
    dtype, then `weight * hidden`.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


def _bg_rmsnorm(x, weight_bf16, cfg, device):
    """BatchGen Glm5RMSNorm (model.py:145-192) wraps `F.rms_norm` — weight-then-cast."""
    norm = nn.Module().to(device)  # dummy — just call F.rms_norm directly
    eps = cfg["rms_norm_eps"]
    # F.rms_norm signature: (input, normalized_shape, weight=None, eps=None)
    out = F.rms_norm(x, (x.shape[-1],), weight=weight_bf16, eps=eps)
    return out.to(torch.bfloat16)


def _hf_rmsnorm(x, weight_bf16, cfg, device):
    eps = cfg["rms_norm_eps"]
    norm = _HfRMSNorm(x.shape[-1], eps=eps).to(device)
    with torch.no_grad():
        # HF's Parameter is FP32 by default — copy preserves FP32 storage
        # but values come from BF16 weight (exact BF16→FP32 upcast).
        norm.weight.copy_(weight_bf16.to(torch.float32))
    out = norm(x)
    return out.to(torch.bfloat16)


def _sgl_rmsnorm(x, weight_bf16, cfg, device):
    """Inlined copy of SGLang RMSNorm.forward_native (layernorm.py:287-331 in
    /home/tairan/workspace/refs/sglang/...). FP32 variance, `(x * weight).to(orig_dtype)`
    — weight-then-cast. Independent of the sgl_kernel runtime."""
    eps = cfg["rms_norm_eps"]
    orig_dtype = x.dtype
    if not x.is_contiguous():
        x = x.contiguous()
    # SGLang's Parameter defaults to FP32; same exact upcast from BF16.
    weight_fp32 = weight_bf16.to(torch.float32).to(device)
    x_fp32 = x.to(torch.float32)
    variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
    x_fp32 = x_fp32 * torch.rsqrt(variance + eps)
    # cast_x_before_out_mul=False default → (x * weight).to(orig_dtype)
    out = (x_fp32 * weight_fp32).to(orig_dtype)
    return out.to(torch.bfloat16)


# ============================================================================
# Step 3 — `q_a_proj` (first FP8 blockwise linear after input_layernorm)
# ============================================================================

def _bg_linear_fp8(x_bf16, weight_name, device):
    """BatchGen: run the w8a16_gemm *semantics* (act_quant + DeepGEMM
    FP8×FP8→BF16) on the raw FP8 weight + FP32 blockwise scale from disk.

    We call `deep_gemm.gemm_fp8_fp8_bf16_nt` directly because BatchGen's
    `w8a16_gemm` wrapper (fa3_backend.py:679) calls
    `deep_gemm.fp8_gemm_nt` which is a stale name in the installed
    deep_gemm — current API is `gemm_fp8_fp8_bf16_nt`. Math is identical
    (same C++ backend).

    Also uses BatchGen's chunked Triton quant (per_token_blocked_quantize_
    bf16_to_fp8_chunked) for the activation FP8 quant to mirror
    BatchGen's prefill hot path exactly.
    """
    import deep_gemm  # type: ignore
    from batchgen_kernels.triton.fp8_quantize import (  # type: ignore
        per_token_blocked_quantize_bf16_to_fp8_chunked,
    )
    w_fp8 = _load_tensor(weight_name).to(device)
    w_scale = _load_tensor(weight_name + "_scale_inv").to(device).to(torch.float32)
    # Flatten (1, L, K) → (L, K) for the quant / gemm.
    orig_shape = x_bf16.shape
    x_flat = x_bf16.reshape(-1, x_bf16.shape[-1])
    x_fp8, x_scale = per_token_blocked_quantize_bf16_to_fp8_chunked(
        x_flat, block_size=128, chunk_rows=8192,
    )
    m = x_flat.shape[0]
    n = w_fp8.shape[0]
    out = torch.empty((m, n), dtype=torch.bfloat16, device=device)
    deep_gemm.gemm_fp8_fp8_bf16_nt((x_fp8, x_scale), (w_fp8, w_scale), out)
    return out.view(*orig_shape[:-1], n).to(torch.bfloat16)


def _sgl_linear_fp8_via_dequant(x_bf16, weight_name, device):
    """SGLang's default GLM-5-on-Hopper path: block-dequant weight to BF16
    once at load time, then plain BF16 F.linear. This is the canonical
    reference (matches HF exactly since HF just does BF16 linear on BF16
    weights)."""
    w_bf16 = _dequant_fp8_weight(weight_name).to(device)
    return F.linear(x_bf16, w_bf16).to(torch.bfloat16)


def _hf_linear(x_bf16, weight_name, device):
    """HF ground truth: identical to SGL-dequant (same BF16 × BF16 linear)."""
    return _sgl_linear_fp8_via_dequant(x_bf16, weight_name, device)


def test_step3_q_a_proj(probe_ids, embed_weight, ckpt_cfg, device):
    """Step 3: layer-0 q_a_proj — the first FP8 block-quant linear. BG runs
    act_quant+fp8_gemm_nt; SGL and HF run BF16 F.linear on the block-dequant
    weight. Tolerance relaxed slightly to allow for FP8 activation-quant noise
    (~1e-3 beyond BF16 ULP)."""
    # Get x = input_layernorm(embed_tokens(probe_ids))
    x = _hf_embed(probe_ids, embed_weight, ckpt_cfg, device)
    w_ln = _load_tensor("model.layers.0.input_layernorm.weight").to(device)
    x = _hf_rmsnorm(x, w_ln, ckpt_cfg, device)  # shared reference

    # Run q_a_proj three ways.
    wname = "model.layers.0.self_attn.q_a_proj.weight"
    bg = _bg_linear_fp8(x, wname, device)
    sgl = _sgl_linear_fp8_via_dequant(x, wname, device)
    hf = _hf_linear(x, wname, device)

    # HF ≡ SGL here (same math). BG includes FP8 act-quant noise.
    bg_ok, sgl_ok, verdict = _emit_verdict(
        "q_a_proj", bg, hf, sgl,
        rtol=5e-2,  # BF16 GEMM (~1e-2) + FP8 act-quant (~1e-2) headroom
        atol=2e-2,
    )
    assert verdict in ("match",), (
        f"q_a_proj divergence beyond FP8-quant noise — bg_ok={bg_ok} "
        f"sgl_ok={sgl_ok} verdict={verdict}"
    )


def test_step2_input_layernorm(probe_ids, embed_weight, ckpt_cfg, device):
    """Step 2: layer-0 input_layernorm. Feed HF-side embedding output
    (same as BG / SGL since embed_tokens matches bit-exact — confirmed at
    step 1)."""
    # 1. Get the common "x" for all three engines — the embedding output.
    x = _hf_embed(probe_ids, embed_weight, ckpt_cfg, device)
    # 2. Load input_layernorm weight (BF16 on disk per modules_to_not_convert).
    w = _load_tensor("model.layers.0.input_layernorm.weight").to(device)
    assert w.dtype == torch.bfloat16, f"expected BF16 input_layernorm, got {w.dtype}"
    # 3. Run all three RMSNorms.
    bg = _bg_rmsnorm(x, w, ckpt_cfg, device)
    hf = _hf_rmsnorm(x, w, ckpt_cfg, device)
    sgl = _sgl_rmsnorm(x, w, ckpt_cfg, device)
    bg_ok, sgl_ok, verdict = _emit_verdict(
        "input_layernorm", bg, hf, sgl,
        rtol=1e-2, atol=1e-2,
    )
    assert verdict in ("match",), (
        f"RMSNorm divergence — bg_ok={bg_ok} sgl_ok={sgl_ok} verdict={verdict}"
    )


if __name__ == "__main__":
    # Standalone runner — avoids the pytest dependency so the script can
    # run inside any Python env that has torch + sglang (on H20 the sglang
    # conda env doesn't ship pytest). Each step_* test is invoked with
    # fixture values produced here.
    import logging
    logging.basicConfig(level=logging.INFO)
    if not _HAS_CUDA:
        print("CUDA unavailable — abort"); sys.exit(2)
    if not _HAS_CKPT:
        print(f"checkpoint not at {MODEL_DIR} — abort"); sys.exit(2)
    if not _HAS_SGLANG:
        print("sglang not importable — abort"); sys.exit(2)
    _device = torch.device("cuda")
    with open(MODEL_DIR / "config.json") as _f:
        _cfg = json.load(_f)
    _probe = PROBE_TOKEN_IDS.to(_device)
    _embed_w = _load_tensor("model.embed_tokens.weight").to(_device)
    # Step 1 — embed_tokens
    test_step1_embed_tokens(_probe, _embed_w, _cfg, _device)
    # Step 2 — input_layernorm
    test_step2_input_layernorm(_probe, _embed_w, _cfg, _device)
    # Step 3 — q_a_proj (first FP8 GEMM)
    test_step3_q_a_proj(_probe, _embed_w, _cfg, _device)
    print("\n[ALL TESTS PASSED]")
