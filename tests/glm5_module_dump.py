"""Cross-env per-module dump: run ONE engine's real-runtime module and
save output tensors to .pt. Designed to be invoked separately in each
conda env (batchgen / sglang / hf) so kernel-level divergences surface
— not just math-equivalent Python inlines.

Usage:
    /root/miniconda3/envs/batchgen/bin/python3 tests/glm5_module_dump.py \\
        --engine batchgen --step embed,input_ln,q_a_proj \\
        --out /tmp/bg_dump.pt

    /root/miniconda3/envs/sglang/bin/python3 tests/glm5_module_dump.py \\
        --engine sglang --step embed,input_ln,q_a_proj \\
        --out /tmp/sgl_dump.pt

    python3 tests/glm5_module_dump.py --compare /tmp/bg_dump.pt /tmp/sgl_dump.pt

The `--compare` path requires nothing but torch — loads both .pt and
emits verdict lines per module. The `--engine` paths require their
respective runtime: batchgen for bg, sglang for sgl, HF transformers
for hf.

Probe input is a FIXED 16-token tensor encoded in this file — not a
tokenizer output — so token IDs are bit-identical across all engines.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

MODEL_DIR = Path("/data2/models/zai-org/GLM-5-FP8")
# Fixed 16 token IDs — identical across all engines to eliminate tokenizer
# variance. These are arbitrary legal vocab IDs; content doesn't matter
# for correctness of per-module numerical comparison.
PROBE_TOKEN_IDS = torch.tensor(
    [[151, 262, 333, 4412, 8199, 23456, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]],
    dtype=torch.long,
)

ALL_STEPS = [
    "embed", "input_ln",
    "q_a_proj", "q_a_normed", "q_b_proj",
    "kv_a_proj", "kv_norm", "kv_b_proj",
]


# ============================================================================
# Shared weight loading (identical logic in every env — just reads .safetensors)
# ============================================================================

_weight_map_cache: Dict[str, str] = {}


def _load_weight_map() -> Dict[str, str]:
    global _weight_map_cache
    if not _weight_map_cache:
        with open(MODEL_DIR / "model.safetensors.index.json") as f:
            _weight_map_cache = json.load(f)["weight_map"]
    return _weight_map_cache


def _load_tensor(name: str) -> torch.Tensor:
    from safetensors import safe_open
    wmap = _load_weight_map()
    with safe_open(str(MODEL_DIR / wmap[name]), framework="pt") as f:
        return f.get_tensor(name)


def _load_config() -> Dict:
    with open(MODEL_DIR / "config.json") as f:
        return json.load(f)


# ============================================================================
# BatchGen engine — runs BatchGen's real runtime modules
# ============================================================================

def run_batchgen(steps: List[str], device, ref_dump: Dict = None) -> Dict[str, torch.Tensor]:
    """Invoke BatchGen's real layer-0 forward on the fixed probe input.
    Returns a dict {step_name: output_tensor_cpu} so compare can run in
    any env.

    If `ref_dump` is provided, each step reads its UPSTREAM input from
    that dump instead of recomputing — so divergence at step N tests ONLY
    that module's math, not accumulated noise from N-1 earlier modules.
    """
    out: Dict[str, torch.Tensor] = {}
    cfg = _load_config()
    probe = PROBE_TOKEN_IDS.to(device)

    def _upstream(name: str) -> torch.Tensor:
        """Return upstream-step tensor: from ref_dump if supplied, else
        from this run's own `out` dict (computed above)."""
        if ref_dump is not None and name in ref_dump:
            return ref_dump[name].to(device)
        if name in out:
            return out[name].to(device)
        raise RuntimeError(f"Upstream {name} not available — add to --step list")

    # Step 1 — embed_tokens (plain nn.Embedding in BatchGen — model.py:1997)
    if "embed" in steps:
        import torch.nn as nn
        ew = _load_tensor("model.embed_tokens.weight").to(device)
        emb = nn.Embedding(cfg["vocab_size"], cfg["hidden_size"], cfg["pad_token_id"]).to(device)
        with torch.no_grad():
            emb.weight.copy_(ew)
        out["embed"] = emb(probe).to(torch.bfloat16).cpu()

    # Step 2 — input_layernorm (Glm5RMSNorm wraps F.rms_norm — model.py:145-192)
    if "input_ln" in steps:
        import torch.nn.functional as F
        x = _upstream("embed")
        w = _load_tensor("model.layers.0.input_layernorm.weight").to(device)
        y = F.rms_norm(x, (x.shape[-1],), weight=w, eps=cfg["rms_norm_eps"])
        out["input_ln"] = y.to(torch.bfloat16).cpu()

    # Env var BATCHGEN_W8A16_DEQUANT=1 swaps the GEMM path to
    # `w8a16_gemm_dequant` (dequant weight to BF16 once, then plain BF16
    # matmul — identical to SGL's Hopper GLM-5 path). Matches the dispatch
    # in `fa3_backend.py:1259`.
    _bg_use_dequant = os.environ.get("BATCHGEN_W8A16_DEQUANT", "0") == "1"

    def _bg_gemm(w_fp8, w_scale, x):
        from batchgen.attention.mla.fa3_backend import (  # type: ignore
            w8a16_gemm, w8a16_gemm_dequant,
        )
        fn = w8a16_gemm_dequant if _bg_use_dequant else w8a16_gemm
        return fn(w_fp8, w_scale, x)

    # Step 3 — q_a_proj via BatchGen's real w8a16_gemm (fa3_backend.py:679)
    if "q_a_proj" in steps:
        x = _upstream("input_ln")
        wname = "model.layers.0.self_attn.q_a_proj.weight"
        w_fp8 = _load_tensor(wname).to(device)
        w_scale = _load_tensor(wname + "_scale_inv").to(device).to(torch.float32)
        y = _bg_gemm(w_fp8, w_scale, x).to(torch.bfloat16)
        out["q_a_proj"] = y.cpu()

    # Step 4 — q_a_layernorm via BatchGen's Glm5RMSNorm (F.rms_norm)
    if "q_a_normed" in steps:
        import torch.nn.functional as F
        x = _upstream("q_a_proj")
        w = _load_tensor("model.layers.0.self_attn.q_a_layernorm.weight").to(device)
        y = F.rms_norm(x, (x.shape[-1],), weight=w, eps=cfg["rms_norm_eps"])
        out["q_a_normed"] = y.to(torch.bfloat16).cpu()

    # Step 5 — q_b_proj via BatchGen's real w8a16_gemm (or w8a16_gemm_dequant
    # when BATCHGEN_W8A16_DEQUANT=1)
    if "q_b_proj" in steps:
        x = _upstream("q_a_normed")
        wname = "model.layers.0.self_attn.q_b_proj.weight"
        w_fp8 = _load_tensor(wname).to(device)
        w_scale = _load_tensor(wname + "_scale_inv").to(device).to(torch.float32)
        y = _bg_gemm(w_fp8, w_scale, x).to(torch.bfloat16)
        out["q_b_proj"] = y.cpu()

    # Step 6 — kv_a_proj_with_mqa — FP8 GEMM. Output is [..., kv_lora_rank +
    # qk_rope_head_dim] = [..., 512 + 64 = 576]. The downstream split into
    # (kv_a, k_pe) happens at step 7.
    if "kv_a_proj" in steps:
        x = _upstream("input_ln")
        wname = "model.layers.0.self_attn.kv_a_proj_with_mqa.weight"
        w_fp8 = _load_tensor(wname).to(device)
        w_scale = _load_tensor(wname + "_scale_inv").to(device).to(torch.float32)
        y = _bg_gemm(w_fp8, w_scale, x).to(torch.bfloat16)
        out["kv_a_proj"] = y.cpu()

    # Step 7 — kv_a_layernorm on the kv_a slice of kv_a_proj_out (first
    # kv_lora_rank=512 dims). BatchGen's Glm5RMSNorm → F.rms_norm.
    if "kv_norm" in steps:
        import torch.nn.functional as F
        kv_a_proj = _upstream("kv_a_proj")
        kv_lora_rank = cfg["kv_lora_rank"]  # 512
        kv_a = kv_a_proj[..., :kv_lora_rank].contiguous()
        w = _load_tensor("model.layers.0.self_attn.kv_a_layernorm.weight").to(device)
        y = F.rms_norm(kv_a, (kv_a.shape[-1],), weight=w, eps=cfg["rms_norm_eps"])
        out["kv_norm"] = y.to(torch.bfloat16).cpu()

    # Step 8 — kv_b_proj — expands kv_norm [..., 512] → [..., num_heads *
    # (qk_nope_head_dim + v_head_dim)]. BG uses w8a16_gemm.
    if "kv_b_proj" in steps:
        x = _upstream("kv_norm")
        wname = "model.layers.0.self_attn.kv_b_proj.weight"
        w_fp8 = _load_tensor(wname).to(device)
        w_scale = _load_tensor(wname + "_scale_inv").to(device).to(torch.float32)
        y = _bg_gemm(w_fp8, w_scale, x).to(torch.bfloat16)
        out["kv_b_proj"] = y.cpu()

    return out


def _fresh_embed(device, cfg, probe):
    import torch.nn as nn
    ew = _load_tensor("model.embed_tokens.weight").to(device)
    emb = nn.Embedding(cfg["vocab_size"], cfg["hidden_size"], cfg["pad_token_id"]).to(device)
    with torch.no_grad():
        emb.weight.copy_(ew)
    return emb(probe).to(torch.bfloat16)


def _fresh_input_ln(device, cfg, probe):
    import torch.nn.functional as F
    x = _fresh_embed(device, cfg, probe)
    w = _load_tensor("model.layers.0.input_layernorm.weight").to(device)
    return F.rms_norm(x, (x.shape[-1],), weight=w, eps=cfg["rms_norm_eps"]).to(torch.bfloat16)


# ============================================================================
# SGLang engine — runs SGLang's real runtime modules
# ============================================================================

def run_sglang(steps: List[str], device, ref_dump: Dict = None) -> Dict[str, torch.Tensor]:
    """Invoke SGLang's real layer-0 forward on the fixed probe. SGLang
    modules (VocabParallelEmbedding, RMSNorm, ColumnParallelLinear) are
    instantiated directly — TP=1, no forward_batch needed for these
    forward paths."""
    out: Dict[str, torch.Tensor] = {}
    cfg = _load_config()
    probe = PROBE_TOKEN_IDS.to(device)

    def _upstream(name: str) -> torch.Tensor:
        if ref_dump is not None and name in ref_dump:
            return ref_dump[name].to(device)
        if name in out:
            return out[name].to(device)
        raise RuntimeError(f"Upstream {name} not available — add to --step list")

    # Step 1 — embed_tokens via SGLang's VocabParallelEmbedding
    if "embed" in steps:
        import torch.nn as nn
        # VocabParallelEmbedding on TP=1 is math-identical to nn.Embedding;
        # use nn.Embedding with the SGLang-loaded weight to avoid forcing
        # the full SGLang distributed init here. This is the one place the
        # two paths converge trivially; for validation only.
        ew = _load_tensor("model.embed_tokens.weight").to(device)
        emb = nn.Embedding(cfg["vocab_size"], cfg["hidden_size"], cfg["pad_token_id"]).to(device)
        with torch.no_grad():
            emb.weight.copy_(ew)
        out["embed"] = emb(probe).to(torch.bfloat16).cpu()

    # Step 2 — input_layernorm via SGLang's real RMSNorm module
    # (layernorm.py:151-222; forward_cuda uses sgl_kernel rmsnorm).
    if "input_ln" in steps:
        from sglang.srt.layers.layernorm import RMSNorm as SglRMSNorm  # type: ignore
        x = _upstream("embed")
        w = _load_tensor("model.layers.0.input_layernorm.weight").to(device)
        assert w.dtype == torch.bfloat16
        # Construct SGLang's RMSNorm with weight_dtype=BF16 so the Parameter
        # matches disk and sgl_kernel sees the expected dtype. The default
        # (weight_dtype=None → FP32) would mismatch the disk layout.
        norm = SglRMSNorm(
            x.shape[-1], eps=cfg["rms_norm_eps"], weight_dtype=torch.bfloat16,
        ).to(device)
        with torch.no_grad():
            norm.weight.copy_(w)
        # Call forward() which dispatches to forward_cuda (real sgl_kernel)
        # on CUDA platform. Reshape for the kernel's 2D input requirement.
        orig_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1]).contiguous()
        y = norm(x_2d).reshape(*orig_shape).to(torch.bfloat16)
        out["input_ln"] = y.cpu()

    # Step 3 — q_a_proj via SGLang's REAL Fp8LinearMethod runtime path
    # (fp8.py:718-747). Block-quant FP8 weights → per-token-group FP8 act
    # quant → DeepGEMM FP8 matmul. NOT block_quant_dequant + F.linear —
    # that's only used for kv_b_proj's MLA absorb post-load.
    if "q_a_proj" in steps:
        from sglang.srt.layers.quantization.fp8_utils import (  # type: ignore
            deepgemm_w8a8_block_fp8_linear_with_fallback,
        )
        x = _upstream("input_ln")
        wname = "model.layers.0.self_attn.q_a_proj.weight"
        w_fp8 = _load_tensor(wname).to(device)
        w_scale = _load_tensor(wname + "_scale_inv").to(device).to(torch.float32)
        y = deepgemm_w8a8_block_fp8_linear_with_fallback(
            x, w_fp8, [128, 128], w_scale,
        ).to(torch.bfloat16)
        out["q_a_proj"] = y.cpu()

    # Step 4 — q_a_layernorm via SGLang's real RMSNorm (sgl_kernel rmsnorm).
    if "q_a_normed" in steps:
        from sglang.srt.layers.layernorm import RMSNorm as SglRMSNorm  # type: ignore
        x = _upstream("q_a_proj")
        w = _load_tensor("model.layers.0.self_attn.q_a_layernorm.weight").to(device)
        assert w.dtype == torch.bfloat16
        norm = SglRMSNorm(
            x.shape[-1], eps=cfg["rms_norm_eps"], weight_dtype=torch.bfloat16,
        ).to(device)
        with torch.no_grad():
            norm.weight.copy_(w)
        orig_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1]).contiguous()
        y = norm(x_2d).reshape(*orig_shape).to(torch.bfloat16)
        out["q_a_normed"] = y.cpu()

    # Step 5 — q_b_proj via SGLang's REAL runtime path (same as step 3).
    if "q_b_proj" in steps:
        from sglang.srt.layers.quantization.fp8_utils import (  # type: ignore
            deepgemm_w8a8_block_fp8_linear_with_fallback,
        )
        x = _upstream("q_a_normed")
        wname = "model.layers.0.self_attn.q_b_proj.weight"
        w_fp8 = _load_tensor(wname).to(device)
        w_scale = _load_tensor(wname + "_scale_inv").to(device).to(torch.float32)
        y = deepgemm_w8a8_block_fp8_linear_with_fallback(
            x, w_fp8, [128, 128], w_scale,
        ).to(torch.bfloat16)
        out["q_b_proj"] = y.cpu()

    # Step 6 — kv_a_proj_with_mqa via SGLang's real FP8 GEMM.
    if "kv_a_proj" in steps:
        from sglang.srt.layers.quantization.fp8_utils import (  # type: ignore
            deepgemm_w8a8_block_fp8_linear_with_fallback,
        )
        x = _upstream("input_ln")
        wname = "model.layers.0.self_attn.kv_a_proj_with_mqa.weight"
        w_fp8 = _load_tensor(wname).to(device)
        w_scale = _load_tensor(wname + "_scale_inv").to(device).to(torch.float32)
        y = deepgemm_w8a8_block_fp8_linear_with_fallback(
            x, w_fp8, [128, 128], w_scale,
        ).to(torch.bfloat16)
        out["kv_a_proj"] = y.cpu()

    # Step 7 — kv_a_layernorm (SGLang RMSNorm, same recipe as step 2/4).
    if "kv_norm" in steps:
        from sglang.srt.layers.layernorm import RMSNorm as SglRMSNorm  # type: ignore
        kv_a_proj = _upstream("kv_a_proj")
        kv_lora_rank = cfg["kv_lora_rank"]
        kv_a = kv_a_proj[..., :kv_lora_rank].contiguous()
        w = _load_tensor("model.layers.0.self_attn.kv_a_layernorm.weight").to(device)
        assert w.dtype == torch.bfloat16
        norm = SglRMSNorm(
            kv_a.shape[-1], eps=cfg["rms_norm_eps"], weight_dtype=torch.bfloat16,
        ).to(device)
        with torch.no_grad():
            norm.weight.copy_(w)
        orig_shape = kv_a.shape
        x_2d = kv_a.reshape(-1, kv_a.shape[-1]).contiguous()
        y = norm(x_2d).reshape(*orig_shape).to(torch.bfloat16)
        out["kv_norm"] = y.cpu()

    # Step 8 — kv_b_proj via SGLang's real FP8 GEMM.
    if "kv_b_proj" in steps:
        from sglang.srt.layers.quantization.fp8_utils import (  # type: ignore
            deepgemm_w8a8_block_fp8_linear_with_fallback,
        )
        x = _upstream("kv_norm")
        wname = "model.layers.0.self_attn.kv_b_proj.weight"
        w_fp8 = _load_tensor(wname).to(device)
        w_scale = _load_tensor(wname + "_scale_inv").to(device).to(torch.float32)
        y = deepgemm_w8a8_block_fp8_linear_with_fallback(
            x, w_fp8, [128, 128], w_scale,
        ).to(torch.bfloat16)
        out["kv_b_proj"] = y.cpu()

    return out


# ============================================================================
# HF engine — runs HF transformers' real modules
# ============================================================================

def run_hf(steps: List[str], device, ref_dump: Dict = None) -> Dict[str, torch.Tensor]:
    """Invoke HF transformers' real modules from `modeling_glm_moe_dsa.py`.
    Requires `transformers` importable in the env."""
    out: Dict[str, torch.Tensor] = {}
    cfg = _load_config()
    probe = PROBE_TOKEN_IDS.to(device)

    def _upstream(name: str) -> torch.Tensor:
        if ref_dump is not None and name in ref_dump:
            return ref_dump[name].to(device)
        if name in out:
            return out[name].to(device)
        raise RuntimeError(f"Upstream {name} not available — add to --step list")

    # Make refs/modeling_glm_moe_dsa.py importable by adding its dir.
    hf_ref_dir = "/home/tairan/workspace/refs"
    if hf_ref_dir not in sys.path:
        sys.path.insert(0, hf_ref_dir)
    # Try to import HF's GlmMoeDsa modules. If transformers isn't present,
    # fall back to the minimal inline class (still PURE-python).
    try:
        from modeling_glm_moe_dsa import GlmMoeDsaRMSNorm  # type: ignore
    except ImportError:
        import torch.nn as nn
        class GlmMoeDsaRMSNorm(nn.Module):
            def __init__(self, hidden_size, eps=1e-6):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(hidden_size))
                self.variance_epsilon = eps

            def forward(self, hidden_states):
                input_dtype = hidden_states.dtype
                hidden_states = hidden_states.to(torch.float32)
                variance = hidden_states.pow(2).mean(-1, keepdim=True)
                hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
                return self.weight * hidden_states.to(input_dtype)

    # Step 1 — embed_tokens (nn.Embedding)
    if "embed" in steps:
        import torch.nn as nn
        ew = _load_tensor("model.embed_tokens.weight").to(device)
        emb = nn.Embedding(cfg["vocab_size"], cfg["hidden_size"], cfg["pad_token_id"]).to(device)
        with torch.no_grad():
            emb.weight.copy_(ew)
        out["embed"] = emb(probe).to(torch.bfloat16).cpu()

    # Step 2 — input_layernorm via HF's real GlmMoeDsaRMSNorm class
    if "input_ln" in steps:
        x = _upstream("embed")
        w = _load_tensor("model.layers.0.input_layernorm.weight").to(device)
        norm = GlmMoeDsaRMSNorm(x.shape[-1], eps=cfg["rms_norm_eps"]).to(device)
        with torch.no_grad():
            norm.weight.copy_(w.to(torch.float32))
        y = norm(x).to(torch.bfloat16)
        out["input_ln"] = y.cpu()

    # Step 3 — q_a_proj via HF's BF16 linear on dequant weight
    if "q_a_proj" in steps:
        import torch.nn.functional as F
        x = _upstream("input_ln")
        # Manual block dequant (same formula as block_quant_dequant, 20 lines)
        wname = "model.layers.0.self_attn.q_a_proj.weight"
        w_fp8 = _load_tensor(wname).to(device)
        w_scale = _load_tensor(wname + "_scale_inv").to(device).to(torch.float32)
        block = 128
        w_scale_rep = w_scale.repeat_interleave(block, dim=-2).repeat_interleave(
            block, dim=-1
        )[..., : w_fp8.shape[-2], : w_fp8.shape[-1]]
        w_bf16 = (w_fp8.to(torch.float32) * w_scale_rep).to(torch.bfloat16)
        y = F.linear(x, w_bf16).to(torch.bfloat16)
        out["q_a_proj"] = y.cpu()

    # Step 4 — q_a_layernorm via HF's real GlmMoeDsaRMSNorm
    if "q_a_normed" in steps:
        x = _upstream("q_a_proj")
        w = _load_tensor("model.layers.0.self_attn.q_a_layernorm.weight").to(device)
        norm = GlmMoeDsaRMSNorm(x.shape[-1], eps=cfg["rms_norm_eps"]).to(device)
        with torch.no_grad():
            norm.weight.copy_(w.to(torch.float32))
        y = norm(x).to(torch.bfloat16)
        out["q_a_normed"] = y.cpu()

    # HF reference: plain BF16 linear on block-dequanted weight.
    # Pure-math "upper bound"; neither production engine hits this path at runtime.
    def _hf_linear(x, wname):
        import torch.nn.functional as F
        w_fp8 = _load_tensor(wname).to(device)
        w_scale = _load_tensor(wname + "_scale_inv").to(device).to(torch.float32)
        block = 128
        w_scale_rep = w_scale.repeat_interleave(block, dim=-2).repeat_interleave(
            block, dim=-1
        )[..., : w_fp8.shape[-2], : w_fp8.shape[-1]]
        w_bf16 = (w_fp8.to(torch.float32) * w_scale_rep).to(torch.bfloat16)
        return F.linear(x, w_bf16).to(torch.bfloat16)

    # Step 5 — q_b_proj (HF BF16 linear on dequant weight)
    if "q_b_proj" in steps:
        out["q_b_proj"] = _hf_linear(
            _upstream("q_a_normed"),
            "model.layers.0.self_attn.q_b_proj.weight",
        ).cpu()

    # Step 6 — kv_a_proj_with_mqa
    if "kv_a_proj" in steps:
        out["kv_a_proj"] = _hf_linear(
            _upstream("input_ln"),
            "model.layers.0.self_attn.kv_a_proj_with_mqa.weight",
        ).cpu()

    # Step 7 — kv_a_layernorm on the kv_a slice
    if "kv_norm" in steps:
        kv_a_proj = _upstream("kv_a_proj")
        kv_lora_rank = cfg["kv_lora_rank"]
        kv_a = kv_a_proj[..., :kv_lora_rank].contiguous()
        w = _load_tensor("model.layers.0.self_attn.kv_a_layernorm.weight").to(device)
        norm = GlmMoeDsaRMSNorm(kv_a.shape[-1], eps=cfg["rms_norm_eps"]).to(device)
        with torch.no_grad():
            norm.weight.copy_(w.to(torch.float32))
        y = norm(kv_a).to(torch.bfloat16)
        out["kv_norm"] = y.cpu()

    # Step 8 — kv_b_proj
    if "kv_b_proj" in steps:
        out["kv_b_proj"] = _hf_linear(
            _upstream("kv_norm"),
            "model.layers.0.self_attn.kv_b_proj.weight",
        ).cpu()

    return out


# ============================================================================
# Comparator — diffs two .pt dumps
# ============================================================================

def compare(paths: List[str]) -> int:
    dumps = {}
    for p in paths:
        d = torch.load(p, map_location="cpu")
        dumps[p] = d
    # Keys across all dumps
    all_steps = set()
    for d in dumps.values():
        all_steps.update(d.keys())
    all_steps = sorted(all_steps)
    for step in all_steps:
        print(f"\n=== Module: {step} ===")
        tensors = {p: d[step] for p, d in dumps.items() if step in d}
        shapes = {p: tuple(t.shape) for p, t in tensors.items()}
        print(f"  shapes: {shapes}")
        # Pair-wise diff
        keys = list(tensors.keys())
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a = tensors[keys[i]].float()
                b = tensors[keys[j]].float()
                err = (a - b).abs()
                max_abs = err.max().item()
                mean = err.mean().item()
                ref = b.abs().clamp(min=1e-3)
                max_rel_clean = (err / ref).max().item()
                allclose = torch.allclose(a, b, rtol=1e-2, atol=1e-2)
                tag_a = Path(keys[i]).stem.replace("_dump", "")
                tag_b = Path(keys[j]).stem.replace("_dump", "")
                print(f"  {tag_a:10s} vs {tag_b:10s}: max_abs={max_abs:.4e}  "
                      f"mean={mean:.4e}  max_rel(ref>1e-3)={max_rel_clean:.4e}  "
                      f"allclose(rtol=1e-2,atol=1e-2)={allclose}")
    return 0


# ============================================================================
# CLI
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["batchgen", "sglang", "hf"],
                    help="Engine to run (producer mode)")
    ap.add_argument("--step", default=",".join(ALL_STEPS),
                    help=f"Comma-separated step names (default: {ALL_STEPS})")
    ap.add_argument("--out", help="Output .pt path (producer mode)")
    ap.add_argument("--from-ref", metavar="REF.pt",
                    help="Feed each step its upstream input from this ref dump "
                         "(isolates module-level divergence — no cascading noise)")
    ap.add_argument("--compare", nargs="+", metavar="DUMP.pt",
                    help="Compare mode: diff two or more .pt dumps")
    args = ap.parse_args()

    if args.compare:
        return compare(args.compare)

    if not args.engine or not args.out:
        ap.error("producer mode needs --engine and --out")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    steps = args.step.split(",")
    ref_dump = None
    if args.from_ref:
        ref_dump = torch.load(args.from_ref, map_location="cpu")
        print(f"[{args.engine}] upstream ref: {args.from_ref} "
              f"(steps available: {list(ref_dump.keys())})")
    print(f"[{args.engine}] running steps={steps} on {device}...")
    if args.engine == "batchgen":
        out = run_batchgen(steps, device, ref_dump=ref_dump)
    elif args.engine == "sglang":
        out = run_sglang(steps, device, ref_dump=ref_dump)
    else:
        out = run_hf(steps, device, ref_dump=ref_dump)
    for k, v in out.items():
        print(f"  {k:12s} shape={tuple(v.shape)} dtype={v.dtype} "
              f"abs_mean={v.float().abs().mean().item():.4e}")
    torch.save(out, args.out)
    print(f"[{args.engine}] saved {len(out)} tensors to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
