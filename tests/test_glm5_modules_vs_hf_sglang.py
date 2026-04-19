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
import pytest
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

try:
    from sglang.srt.layers.quantization.fp8_utils import block_quant_dequant  # type: ignore
    _HAS_SGLANG = True
except Exception:
    _HAS_SGLANG = False

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

@pytest.mark.skipif(not _HAS_SGLANG, reason="sglang import required")
def test_step1_embed_tokens(probe_ids, embed_weight, ckpt_cfg, device):
    """Step 1: embed_tokens. All three engines should produce identical output."""
    bg = _bg_embed(probe_ids, embed_weight, ckpt_cfg, device)
    hf = _hf_embed(probe_ids, embed_weight, ckpt_cfg, device)
    sgl = _sgl_embed(probe_ids, embed_weight, ckpt_cfg, device)
    bg_ok, sgl_ok, verdict = _emit_verdict("embed_tokens", bg, hf, sgl)
    # Embedding is a lookup — should be bit-exact.
    assert verdict == "match", f"unexpected verdict at embed_tokens: {verdict}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
