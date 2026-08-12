# ---------------------------------------------------------------------------- #
#  BatchGen — Kimi-K3                                                           #
#  copyright (c) EfficientMoE team 2025                                         #
#  Licensed under the Apache License, Version 2.0                               #
# ---------------------------------------------------------------------------- #
"""Test harness for the Kimi-K3 M2 model (tests/test_kimi_k3_model.py).

Responsibilities:
  * file-path loading of the modules under test (never ``import batchgen`` —
    that JIT-builds the core engine; the ``tests/test_kimi_k3_tensor_map.py``
    pattern);
  * loading the vendored HF oracle (``tests/kimi_k3_oracle_assets/``) with the
    fla CPU shim installed and the attention implementation forced to eager;
  * the two synthetic shrunk configs (K3-SYN-25 primary, K3-SKEW-10
    ratio-degeneracy killer);
  * deterministic name-keyed weight seeding shared by both stacks;
  * the numeric gates (BF16 project gate / FP32 tight);
  * the verbatim reference form of ``_apply_attn_res`` (kept OUT of model.py
    on purpose — POIS decision 2 makes the lean form the production one);
  * the mutation registry (applied to OUR modules only, selected by the
    ``BATCHGEN_K3_MUTATION`` env var, driven by tests/mutation_check_kimi_k3.py).

CPU-parity caveat, stated up front: on CPU the fla shim backs BOTH stacks'
KDA recurrence with the same vendored torch core, so the recurrence interior
cancels in every parity test here; it is validated exclusively by the staged
GPU test (tests/gpu/test_kimi_k3_kda_fla_parity.py).

The same caveat extends one step further than it first appears.  The shim's
``ShortConvolution`` and ``FusedRMSNormGated`` are written independently of
model.py's ports, but both transcribe the SAME reading of fla — so a shared
misreading of fla would cancel on CPU exactly as the KDA interior does.
Those two modules are cross-validated against the real fla only by GPU Part
B.  Everything else — projections, gates, norms, router, MoE, AttnRes, the
layer map — is genuinely cross-validated here against the oracle's own code.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import types
import zlib
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

TESTS_DIR = Path(__file__).resolve().parent
ROOT = TESTS_DIR.parent
K3_PKG_DIR = ROOT / "batchgen" / "models" / "moonshotai" / "kimi_k3"
ORACLE_ASSETS_DIR = TESTS_DIR / "kimi_k3_oracle_assets"
REAL_CONFIG_JSON = K3_PKG_DIR / "assets" / "config.json"

GLOBAL_SEED = 20260805

# Byte pins (tests/test_kimi_k3_model.py::test_oracle_md5_pins).
MODELING_KIMI_LINEAR_MD5 = "4e3de36ab2a5de1232c05ce346a3426e"
CONFIGURATION_KIMI_K3_MD5 = "3165dde7cebe8471fdf43aa9890d5c02"

MUTATION_ENV = "BATCHGEN_K3_MUTATION"


# --------------------------------------------------------------------------- #
#  Module loading                                                              #
# --------------------------------------------------------------------------- #
def _load_by_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _stub_package(name: str, search_path: Path) -> None:
    if name in sys.modules:
        return
    package = types.ModuleType(name)
    package.__path__ = [str(search_path)]
    sys.modules[name] = package


class _OurModules:
    """Namespace over the loaded BatchGen K3 modules (config / model /
    kda_reference)."""

    def __init__(self, config_mod, model_mod, kda_ref_mod):
        self.config = config_mod
        self.model = model_mod
        self.kda_reference = kda_ref_mod


_OUR: _OurModules | None = None


def load_our_modules() -> _OurModules:
    """Load kimi_k3/{config,model,kda_reference}.py by file path, then apply
    the mutation named by $BATCHGEN_K3_MUTATION (if any) to OUR modules only.
    Memoized per process (mutations patch module state)."""
    global _OUR
    if _OUR is not None:
        return _OUR
    _stub_package("_k3m", K3_PKG_DIR)
    config_mod = _load_by_path("_k3m.config", K3_PKG_DIR / "config.py")
    kda_ref_mod = _load_by_path("_k3m.kda_reference", K3_PKG_DIR / "kda_reference.py")
    model_mod = _load_by_path("_k3m.model", K3_PKG_DIR / "model.py")
    _OUR = _OurModules(config_mod, model_mod, kda_ref_mod)

    mutation = os.environ.get(MUTATION_ENV, "").strip()
    if mutation:
        if mutation not in MUTATIONS:
            raise ValueError(
                "Unknown {}={!r}. Known mutations: {}".format(
                    MUTATION_ENV, mutation, sorted(MUTATIONS)))
        MUTATIONS[mutation].apply(_OUR)
    return _OUR


class _OracleModules:
    def __init__(self, configuration_mod, modeling_mod, shim_mod):
        self.configuration = configuration_mod
        self.modeling = modeling_mod
        self.shim = shim_mod


_ORACLE: _OracleModules | None = None


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def check_oracle_pins() -> None:
    got = _md5(ORACLE_ASSETS_DIR / "modeling_kimi_linear.py")
    if got != MODELING_KIMI_LINEAR_MD5:
        raise RuntimeError(
            "Vendored modeling_kimi_linear.py drifted from the checkpoint pin "
            "({} != {})".format(got, MODELING_KIMI_LINEAR_MD5))
    got = _md5(ORACLE_ASSETS_DIR / "configuration_kimi_k3.py")
    if got != CONFIGURATION_KIMI_K3_MD5:
        raise RuntimeError(
            "Vendored configuration_kimi_k3.py drifted from the checkpoint pin "
            "({} != {})".format(got, CONFIGURATION_KIMI_K3_MD5))


def load_oracle_modules() -> _OracleModules:
    """Install the fla CPU shim, then import the vendored oracle.

    HARD-FAILS (never skips) when the environment cannot run the oracle:
    the CPU parity suite is the M2 gate, a skip would silently green it.
    """
    global _ORACLE
    if _ORACLE is not None:
        return _ORACLE

    check_oracle_pins()

    try:
        import transformers
        from packaging import version as _v
    except ImportError as exc:
        raise RuntimeError(
            "The K3 CPU parity suite requires `transformers` (>=4.56) and "
            "`packaging`; install them — do not skip this suite") from exc
    if _v.parse(transformers.__version__) < _v.parse("4.56.0"):
        raise RuntimeError(
            "The K3 oracle asserts transformers >= 4.56.0 at import; found {}. "
            "Upgrade the environment — do not skip this suite"
            .format(transformers.__version__))
    try:
        import einops  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "The K3 oracle (and kda_reference) require `einops`; install it — "
            "do not skip this suite") from exc

    shim = _load_by_path("_k3_fla_cpu_shim", ORACLE_ASSETS_DIR / "fla_cpu_shim.py")
    shim.install(force=True)

    _stub_package("_k3_oracle", ORACLE_ASSETS_DIR)
    configuration = _load_by_path(
        "_k3_oracle.configuration_kimi_k3", ORACLE_ASSETS_DIR / "configuration_kimi_k3.py")
    modeling = _load_by_path(
        "_k3_oracle.modeling_kimi_linear", ORACLE_ASSETS_DIR / "modeling_kimi_linear.py")
    _ORACLE = _OracleModules(configuration, modeling, shim)
    return _ORACLE


# --------------------------------------------------------------------------- #
#  Synthetic shrunk configs                                                    #
# --------------------------------------------------------------------------- #
def syn25_config_dict() -> dict:
    """K3-SYN-25 — the primary synthetic config.  Structure audit vs the real
    93-layer model: 25 layers with the true 3:1 KDA/MLA rhythm AND the
    double-MLA tail (24 on-rhythm + 25, mirroring 92+93); layer 0 KDA + dense;
    attn_res_block_size 3 -> 9 boundary blocks (>= production's 8), boundaries
    landing on BOTH layer kinds (3 is coprime to the rhythm period 4) and a
    ragged final block (25 = 8*3 + 1, mirroring 93 = 7*12 + 9); MLA per-head
    geometry REAL (192/128/64/128, kv latent 576); LatentMoE at the real
    latent = hidden/2 ratio with a genuine top-16-of-64 minority selection."""
    full_attn = [4, 8, 12, 16, 20, 24, 25]          # 1-BASED, like the checkpoint
    kda = [i for i in range(1, 26) if i not in full_attn]
    text = {
        "vocab_size": 2048,
        "hidden_size": 448,                          # 7168 / 16
        "intermediate_size": 2112,                   # 33792 / 16
        "num_hidden_layers": 25,
        "rms_norm_eps": 1e-5,
        "num_attention_heads": 8,
        "num_key_value_heads": 8,
        "q_lora_rank": 128,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "mla_use_nope": True,
        "mla_use_output_gate": True,
        "linear_attn_config": {
            "kda_layers": kda,
            "full_attn_layers": full_attn,
            "num_heads": 8,
            "head_dim": 64,
            "short_conv_kernel_size": 4,
            "use_full_rank_gate": True,
            "gate_lower_bound": -5.0,
        },
        "num_experts": 64,
        "num_experts_per_token": 16,
        "moe_intermediate_size": 192,                # real ratio 3072/3584
        "num_shared_experts": 2,
        "first_k_dense_replace": 1,
        "moe_layer_freq": 1,
        "moe_renormalize": True,
        "moe_router_activation_func": "sigmoid",
        "routed_scaling_factor": 1.0,
        "num_expert_group": 1,
        "topk_group": 1,
        "use_grouped_topk": True,
        "topk_method": "noaux_tc",
        "routed_expert_hidden_size": 224,            # hidden / 2, real ratio
        "latent_moe_use_norm": True,
        "hidden_act": "situ",
        "activation_situ_beta": 4.0,
        "activation_situ_linear_beta": 25.0,
        "attn_res_block_size": 3,
        "num_nextn_predict_layers": 0,
        "max_position_embeddings": 4096,
        "initializer_range": 0.02,
        "tie_word_embeddings": False,
        "pad_token_id": 2047,
        "bos_token_id": 2044,
        "eos_token_id": 2045,
    }
    return {
        "model_type": "kimi_k3",
        "media_placeholder_token_id": 2040,
        "text_config": text,
    }


def skew10_config_dict() -> dict:
    """K3-SKEW-10 — every dimension pair production ties is decoupled, so a
    'derived the dim from the wrong source' bug cannot hide: latent != hidden/2,
    v_head_dim != qk_nope_head_dim, KDA head_dim != MLA dims, shared width 3x,
    rms_norm_eps != 1e-5 (proves eps is read from config while the MLA q_a/kv_a
    layernorms stay hardcoded 1e-6)."""
    full_attn = [4, 8, 10]                           # 0-idx MLA {3, 7, 9}
    kda = [i for i in range(1, 11) if i not in full_attn]
    text = {
        "vocab_size": 1024,
        "hidden_size": 320,
        "intermediate_size": 960,
        "num_hidden_layers": 10,
        "rms_norm_eps": 2e-5,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "q_lora_rank": 160,
        "kv_lora_rank": 256,
        "qk_nope_head_dim": 96,
        "qk_rope_head_dim": 32,
        "v_head_dim": 80,                            # != qk_nope_head_dim!
        "mla_use_nope": True,
        "mla_use_output_gate": True,
        "linear_attn_config": {
            "kda_layers": kda,
            "full_attn_layers": full_attn,
            "num_heads": 6,
            "head_dim": 32,
            "short_conv_kernel_size": 4,
            "use_full_rank_gate": True,
            "gate_lower_bound": -5.0,
        },
        "num_experts": 48,
        "num_experts_per_token": 16,
        "moe_intermediate_size": 112,
        "num_shared_experts": 3,
        "first_k_dense_replace": 1,
        "moe_layer_freq": 1,
        "moe_renormalize": True,
        "moe_router_activation_func": "sigmoid",
        "routed_scaling_factor": 1.0,
        "num_expert_group": 1,
        "topk_group": 1,
        "use_grouped_topk": True,
        "topk_method": "noaux_tc",
        "routed_expert_hidden_size": 192,            # != hidden/2 (160)
        "latent_moe_use_norm": True,
        "hidden_act": "situ",
        "activation_situ_beta": 4.0,
        "activation_situ_linear_beta": 25.0,
        "attn_res_block_size": 4,
        "num_nextn_predict_layers": 0,
        "max_position_embeddings": 4096,
        "initializer_range": 0.02,
        "tie_word_embeddings": False,
        "pad_token_id": 1023,
        "bos_token_id": 1020,
        "eos_token_id": 1021,
    }
    return {
        "model_type": "kimi_k3",
        "media_placeholder_token_id": 1010,
        "text_config": text,
    }


def real_config_dict() -> dict:
    with open(REAL_CONFIG_JSON) as f:
        return json.load(f)


def build_our_config(cfg_dict: dict):
    ours = load_our_modules()
    return ours.config.parse_k3_config(cfg_dict)


def build_oracle_config(cfg_dict: dict):
    om = load_oracle_modules()
    ocfg = om.configuration.KimiLinearConfig(**cfg_dict["text_config"])
    ocfg._attn_implementation = "eager"
    ocfg.use_cache = False
    return ocfg


# --------------------------------------------------------------------------- #
#  Deterministic name-keyed weight seeding                                     #
# --------------------------------------------------------------------------- #
def _generator(name: str) -> torch.Generator:
    seed = zlib.crc32("{}:{}".format(GLOBAL_SEED, name).encode()) & 0x7FFFFFFF
    return torch.Generator().manual_seed(int(seed))


def seeded_master(name: str, shape: Tuple[int, ...]) -> torch.Tensor:
    """fp32 master value for a parameter, keyed on its (full) name so both
    stacks get identical values independent of construction order.

    Distributions (test-plan 4.3): linears N(0, .02); norm gains U(.8, 1.2)
    (non-unit — catches gain-ordering bugs); res_proj N(0, H^-1/2);
    A_log log(U(1,16)); dt_bias N(0, .5); conv N(0, .25);
    e_score_correction_bias N(0, .05)."""
    g = _generator(name)
    t = torch.empty(*shape, dtype=torch.float32)
    if name.endswith("A_log"):
        return torch.log(t.uniform_(1.0, 16.0, generator=g))
    if name.endswith("dt_bias"):
        return t.normal_(0.0, 0.5, generator=g)
    if name.endswith(("q_conv1d.weight", "k_conv1d.weight", "v_conv1d.weight")):
        return t.normal_(0.0, 0.25, generator=g)
    if name.endswith("e_score_correction_bias"):
        return t.normal_(0.0, 0.05, generator=g)
    if name.endswith("res_proj.weight"):
        return t.normal_(0.0, shape[-1] ** -0.5, generator=g)
    if name.endswith("q_a_proj.weight"):
        # Seeded SMALLER than the generic 0.02 on purpose: it pushes the q_a
        # pre-norm variance down to ~0.01, where the hardcoded 1e-6 eps of
        # q_a_layernorm differs from a mutated 1e-5 by ~4e-4 relative — safely
        # above the 1e-6 fp32 gate after softmax attenuation.  At the generic
        # scale the eps flip lands right AT the gate (measured 2026-08-05).
        return t.normal_(0.0, 0.005, generator=g)
    parts = name.rsplit(".", 2)
    if len(shape) == 1 and len(parts) >= 2 and ("norm" in parts[-2]):
        # *.{...norm...}.weight — all RMSNorm gains (input/post/res/o_norm/
        # routed_expert_norm/q_a_layernorm/kv_a_layernorm/model.norm/...).
        return t.uniform_(0.8, 1.2, generator=g)
    return t.normal_(0.0, 0.02, generator=g)


def make_state_dicts(named_shapes: Dict[str, Tuple[int, ...]], kda_num_heads: int,
                     prefix: str = "") -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """(ours, oracle) fp32 state dicts from OUR parameter names/shapes.

    The single documented shape delta: ``A_log`` — ours is the padded
    checkpoint buffer (F32[a_log_padded_len], entries [:H] live, pad zero);
    the oracle allocates [H] (modeling_kimi_linear.py:520-521 cannot
    strict-load its own checkpoint — flag B)."""
    ours: Dict[str, torch.Tensor] = {}
    oracle: Dict[str, torch.Tensor] = {}
    for name, shape in named_shapes.items():
        master = seeded_master(prefix + name, shape)
        if name.endswith("A_log"):
            master = master.clone()
            master[kda_num_heads:] = 0.0
            oracle[name] = master[:kda_num_heads].clone()
        else:
            oracle[name] = master
        ours[name] = master
    return ours, oracle


def named_shapes_of(module: torch.nn.Module) -> Dict[str, Tuple[int, ...]]:
    return {name: tuple(p.shape) for name, p in module.named_parameters()}


def load_pair(our_module: torch.nn.Module, oracle_module: torch.nn.Module,
              kda_num_heads: int, dtype: torch.dtype, prefix: str = "") -> None:
    """Seed OUR module, mirror into the oracle module, apply the shared dtype
    policy (selective bf16 with the FP32 parameter set kept fp32 — flag G:
    both stacks must agree on this policy before any comparison), eval()."""
    ours_sd, oracle_sd = make_state_dicts(
        named_shapes_of(our_module), kda_num_heads, prefix=prefix)
    missing_o = set(oracle_sd) ^ {n for n, _ in oracle_module.named_parameters()}
    if missing_o:
        raise AssertionError(
            "our/oracle parameter names diverge (this is test_state_dict_key_parity's "
            "job to report properly): {}".format(sorted(missing_o)[:10]))
    our_module.load_state_dict(ours_sd, strict=True)
    oracle_module.load_state_dict(oracle_sd, strict=True)
    ours = load_our_modules()
    if dtype != torch.float32:
        ours.model.cast_model_to_inference_dtype(our_module, dtype)
        cast_selective(oracle_module, dtype)
    our_module.eval()
    oracle_module.eval()


def cast_selective(module: torch.nn.Module, dtype: torch.dtype) -> None:
    """The oracle-side mirror of cast_model_to_inference_dtype: everything to
    ``dtype`` except the checkpoint's FP32 set (same name suffixes)."""
    ours = load_our_modules()
    with torch.no_grad():
        for name, p in module.named_parameters():
            if name.endswith(tuple(ours.model.K3_FP32_PARAM_SUFFIXES)):
                continue
            p.data = p.data.to(dtype)


# --------------------------------------------------------------------------- #
#  Inputs                                                                      #
# --------------------------------------------------------------------------- #
def seeded_input(name: str, *shape: int, dtype: torch.dtype = torch.float32,
                 rms: float = 1.0) -> torch.Tensor:
    g = _generator("input:" + name)
    t = torch.empty(*shape, dtype=torch.float32).normal_(0.0, rms, generator=g)
    return t.to(dtype)


def seeded_token_ids(name: str, batch: int, seq_len: int, vocab: int,
                     media_token_id: int) -> torch.Tensor:
    """Uniform over the vocab EXCLUDING the media placeholder (the model
    hard-fails on it, by design)."""
    g = _generator("ids:" + name)
    ids = torch.randint(0, vocab - 1, (batch, seq_len), generator=g)
    ids[ids >= media_token_id] += 1
    return ids


def build_causal_mask(seq_len: int, dtype: torch.dtype) -> torch.Tensor:
    mask = torch.full((1, 1, seq_len, seq_len), float("-inf"), dtype=dtype)
    return mask.triu(1)


# --------------------------------------------------------------------------- #
#  Numeric gates                                                               #
# --------------------------------------------------------------------------- #
def assert_bf16_gate(actual: torch.Tensor, ref: torch.Tensor, what: str) -> None:
    """Project BF16 gate: no NaN/Inf; fail_frac(|a-r| > 1e-5 + 1.6e-2|r|) < 1e-4."""
    a = actual.float()
    r = ref.float()
    assert torch.isfinite(a).all(), "{}: non-finite values in output".format(what)
    tol = 1e-5 + 1.6e-2 * r.abs()
    fail_frac = ((a - r).abs() > tol).float().mean().item()
    assert fail_frac < 1e-4, "{}: fail_frac={:.3e} (gate 1e-4); max_abs={:.3e}".format(
        what, fail_frac, (a - r).abs().max().item())


def assert_kernel_err_ratio(actual: torch.Tensor, ref: torch.Tensor, what: str,
                            ratio: float = 5e-3) -> None:
    """fla's OWN acceptance metric, for comparing a chunked recurrent triton
    kernel against a naive torch recurrence.

    Metric: ``RMS(actual - ref) / RMS(ref)`` — scale-relative, not per-element
    relative.  Source: ``fla/utils/_testing.py::get_err_ratio`` +
    ``assert_close``, and ``fla/tests/ops/test_kda.py`` uses ``ratio=0.005``
    for exactly this comparison (their ``chunk_kda`` vs their
    ``naive_recurrent_kda``).  We adopt the library authors' bar rather than
    invent one.

    Why NOT ``assert_bf16_gate`` here: a per-element relative gate is the
    wrong instrument for a recurrence whose output distribution is heavily
    concentrated near zero.  Measured on the syn case (2026-08-05):
    output RMS 1.01e-2, max|ref| 1.5e-1; with fp32 inputs the per-element
    gate's failures are 100% concentrated at |ref| < 0.1*RMS — i.e. positions
    where the true value is a cancellation residue — while the RMS-relative
    error is 0.0018.  Switching inputs bf16 -> fp32 drops fail_frac 240x
    (5.6e-2 -> 2.3e-4) with an unchanged formula, which is the signature of
    rounding, not of wrong math.  The per-element gate stays in force for
    module-level outputs, where magnitudes are O(1) and it discriminates.

    This is a LOOSER-LOOKING but strictly better-targeted gate: it is a whole-
    tensor budget, so a real formula error (wrong gate branch, missing
    sigmoid, dropped l2norm) blows past 0.005 immediately — all four such
    mutations were verified to do so.
    """
    a = actual.float()
    r = ref.float()
    assert torch.isfinite(a).all(), "{}: non-finite values in output".format(what)
    err = (a - r).pow(2).mean().sqrt().item()
    base = r.pow(2).mean().sqrt().item()
    measured = err / (base + 1e-8)
    print("[ratio] {:52s} err_ratio={:.6f} (bar {:.3f})".format(
        what, measured, ratio))
    assert measured < ratio, (
        "{}: err_ratio={:.6f} exceeds fla's own bar {:.3f} for kernel-vs-"
        "reference (RMS-relative). max_abs={:.3e}, ref RMS={:.3e}".format(
            what, measured, ratio, (a - r).abs().max().item(), base))


def assert_fp32_tight(actual: torch.Tensor, ref: torch.Tensor, what: str,
                      tol: float = 1e-6) -> None:
    diff = (actual.double() - ref.double()).abs().max().item()
    assert diff < tol, "{}: max_abs_diff={:.3e} >= {:.1e}".format(what, diff, tol)


def topk_index_sets(topk_idx: torch.Tensor):
    return [frozenset(row.tolist()) for row in topk_idx]


# --------------------------------------------------------------------------- #
#  Verbatim reference mixer (ML:1075-1088 transcription)                       #
# --------------------------------------------------------------------------- #
def apply_attn_res_reference(prefix_sum: torch.Tensor, block_residual: torch.Tensor,
                             proj, norm) -> torch.Tensor:
    """The materializing reference form of the Block-Attention-Residual depth
    mixer.  Deliberately NOT in model.py (POIS decision 2: the memory-lean form
    is the production one); this transcription is checked against the oracle's
    own ``_apply_attn_res`` (test_attn_res_reference_fp32) and the lean form is
    gated against THIS at max_abs < 1e-6 (test_attn_res_lean_equiv)."""
    v = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
    v_float = v.float()
    variance = v_float.pow(2).mean(-1, keepdim=True)
    k = v_float * torch.rsqrt(variance + norm.variance_epsilon)
    score_weight = norm.weight.float() * proj.weight.squeeze(0).float()
    scores = (k * score_weight).sum(-1)
    probs = scores.softmax(-1).unsqueeze(1)
    hidden_states = torch.matmul(probs, v_float).squeeze(1)
    return hidden_states.to(v.dtype)


# --------------------------------------------------------------------------- #
#  Model / module builders                                                     #
# --------------------------------------------------------------------------- #
def build_pair_causallm(cfg_dict: dict, dtype: torch.dtype):
    """(our KimiK3ForCausalLM[reference backend], oracle KimiLinearForCausalLM)
    with identical seeded weights and the shared dtype policy."""
    ours = load_our_modules()
    om = load_oracle_modules()
    cfg = build_our_config(cfg_dict)
    our_model = ours.model.KimiK3ForCausalLM(cfg, kda_backend="reference")
    ocfg = build_oracle_config(cfg_dict)
    oracle_model = om.modeling.KimiLinearForCausalLM(ocfg)
    # The oracle force-sets flash_attention_2 during construction
    # (modeling_kimi_linear.py:1110-1117); flip it back AFTER construction.
    oracle_model.config._attn_implementation = "eager"
    oracle_model.model._use_flash_attention_2 = False
    load_pair(our_model, oracle_model, cfg.kda_num_heads, dtype)
    return our_model, oracle_model, cfg


def oracle_forward_logits(oracle_model, input_ids: torch.Tensor) -> torch.Tensor:
    out = oracle_model(input_ids=input_ids, use_cache=False)
    return out.logits


def build_pair_module(kind: str, cfg_dict: dict, dtype: torch.dtype,
                      layer_idx: int = 0):
    """Isolated module pair with identical seeded weights.
    kind in {'mla', 'kda', 'moe', 'layer'}."""
    ours = load_our_modules()
    om = load_oracle_modules()
    cfg = build_our_config(cfg_dict)
    ocfg = build_oracle_config(cfg_dict)
    if kind == "mla":
        our_mod = ours.model.KimiK3MLAAttention(cfg, layer_idx)
        oracle_mod = om.modeling.KimiMLAAttention(ocfg, layer_idx)
    elif kind == "kda":
        our_mod = ours.model.KimiK3KDAAttention(cfg, layer_idx, kda_backend="reference")
        oracle_mod = om.modeling.KimiDeltaAttention(ocfg, layer_idx)
    elif kind == "moe":
        our_mod = ours.model.KimiSparseMoeBlock(cfg)
        oracle_mod = om.modeling.KimiSparseMoeBlock(ocfg)
    elif kind == "layer":
        our_mod = ours.model.KimiK3DecoderLayer(cfg, layer_idx, kda_backend="reference")
        oracle_mod = om.modeling.KimiDecoderLayer(ocfg, layer_idx)
    else:
        raise ValueError(kind)
    load_pair(our_mod, oracle_mod, cfg.kda_num_heads, dtype,
              prefix="module:{}:{}:".format(kind, layer_idx))
    return our_mod, oracle_mod, cfg


# --------------------------------------------------------------------------- #
#  Mutation registry                                                           #
# --------------------------------------------------------------------------- #
class Mutation:
    def __init__(self, name: str, apply, must_red: str, note: str = ""):
        self.name = name
        self.apply = apply          # fn(_OurModules) -> None
        self.must_red = must_red    # pytest -k expression expected to FAIL
        self.note = note


MUTATIONS: Dict[str, Mutation] = {}


def _mutation(name: str, must_red: str, note: str = ""):
    def wrap(fn):
        MUTATIONS[name] = Mutation(name, fn, must_red, note)
        return fn
    return wrap


# ---- MLA -------------------------------------------------------------------
@_mutation("mla_q_a_eps_1e5", "test_mla_module_fp32")
def _m(ns):
    orig = ns.model.KimiK3MLAAttention.__init__

    def patched(self, config, layer_idx):
        orig(self, config, layer_idx)
        self.q_a_layernorm.variance_epsilon = 1e-5
    ns.model.KimiK3MLAAttention.__init__ = patched


@_mutation("mla_kv_a_eps_1e5", "test_mla_module_fp32")
def _m(ns):
    orig = ns.model.KimiK3MLAAttention.__init__

    def patched(self, config, layer_idx):
        orig(self, config, layer_idx)
        self.kv_a_layernorm.variance_epsilon = 1e-5
    ns.model.KimiK3MLAAttention.__init__ = patched


@_mutation("mla_gate_no_sigmoid", "test_mla_module_fp32")
def _m(ns):
    def patched(self, attn_output, hidden_states):
        return attn_output * self.g_proj(hidden_states)
    ns.model.KimiK3MLAAttention._apply_output_gate = patched


@_mutation("mla_gate_from_attn_out", "test_mla_module_fp32",
           "plausible-looking: gate computed from the attention output")
def _m(ns):
    def patched(self, attn_output, hidden_states):
        return attn_output * self.g_proj(attn_output).sigmoid()
    ns.model.KimiK3MLAAttention._apply_output_gate = patched


@_mutation("mla_gate_fp32_sigmoid", "test_mla_module_bf16_bitwise",
           "reference computes the sigmoid in bf16 (ML:470-472), not fp32")
def _m(ns):
    def patched(self, attn_output, hidden_states):
        g32 = torch.sigmoid(self.g_proj(hidden_states).float())
        return (attn_output.float() * g32).to(attn_output.dtype)
    ns.model.KimiK3MLAAttention._apply_output_gate = patched


@_mutation("mla_rope_applied", "test_mla_module_fp32",
           "K3 is NoPE: no rotary anywhere (ML:403/439-440)")
def _m(ns):
    def patched(q_pass, q_rot, k_pass, k_rot):
        # apply a rotary embedding on the 'rope' sub-dim
        T = q_rot.shape[-2]
        d = q_rot.shape[-1]
        pos = torch.arange(T, dtype=torch.float32)
        inv = 1.0 / (10000.0 ** (torch.arange(0, d, 2, dtype=torch.float32) / d))
        ang = torch.einsum("t,f->tf", pos, inv)
        cos = torch.cos(ang).repeat_interleave(2, -1)
        sin = torch.sin(ang).repeat_interleave(2, -1)

        def rot(x):
            x1 = x[..., 0::2]
            x2 = x[..., 1::2]
            xr = torch.stack((-x2, x1), dim=-1).flatten(-2)
            return (x.float() * cos + xr.float() * sin).to(x.dtype)
        q = torch.cat((q_pass, rot(q_rot)), dim=-1)
        k = torch.cat((k_pass, rot(k_rot)), dim=-1)
        return q, k
    ns.model._nope_join = patched


# ---- AttnRes body -----------------------------------------------------------
def _patched_layer_forward(ns, *, boundary_reset=True, mix_feeds_accumulator=False,
                           snapshot_post_mix=False):
    lean = ns.model._apply_attn_res_lean

    def forward(self, hidden_states, attention_mask, block_residual):
        batch_size, seq_len, hidden_size = hidden_states.shape
        prefix_sum = hidden_states
        if block_residual.shape[1] > 0:
            hidden_states = lean(
                prefix_sum.view(-1, hidden_size), block_residual,
                self.self_attention_res_proj, self.self_attention_res_norm,
            ).view(batch_size, seq_len, hidden_size)
            if mix_feeds_accumulator:
                prefix_sum = hidden_states
        if self.layer_idx % self.attn_res_block_size == 0:
            snap = hidden_states if snapshot_post_mix else prefix_sum
            block_residual = torch.cat(
                [block_residual, snap.view(-1, hidden_size).unsqueeze(1)], dim=1)
            if boundary_reset:
                prefix_sum = None
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self._run_attn(hidden_states, attention_mask)
        if prefix_sum is not None:
            prefix_sum = prefix_sum + hidden_states
        else:
            prefix_sum = hidden_states
        hidden_states = lean(
            prefix_sum.view(-1, hidden_size), block_residual,
            self.mlp_res_proj, self.mlp_res_norm,
        ).view(batch_size, seq_len, hidden_size)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self._run_ffn(hidden_states)
        prefix_sum = prefix_sum + hidden_states
        return prefix_sum, block_residual
    return forward


@_mutation("prefix_sum_sees_mix", "test_layer_skeleton or test_full_model_logits")
def _m(ns):
    ns.model.KimiK3DecoderLayer.forward = _patched_layer_forward(
        ns, mix_feeds_accumulator=True)


@_mutation("no_boundary_reset", "test_layer_skeleton or test_full_model_logits")
def _m(ns):
    ns.model.KimiK3DecoderLayer.forward = _patched_layer_forward(
        ns, boundary_reset=False)


@_mutation("post_mix_snapshot", "test_layer_skeleton or test_full_model_logits")
def _m(ns):
    ns.model.KimiK3DecoderLayer.forward = _patched_layer_forward(
        ns, snapshot_post_mix=True)


@_mutation("block_residual_carried", "test_forward_twice_identical",
           "simulates the scratch buffer not being re-zeroed across forwards")
def _m(ns):
    def patched(self, inputs_embeds):
        batch, seq_len, hidden = inputs_embeds.shape
        carry = getattr(self, "_k3_carry", None)
        if carry is not None and carry.shape[0] == batch * seq_len:
            return carry
        self._k3_carry = inputs_embeds.reshape(batch * seq_len, 1, hidden)
        return inputs_embeds.new_zeros(batch * seq_len, 0, hidden)
    ns.model.KimiK3Model._initial_block_residual = patched


@_mutation("res_fold_no_norm_weight", "test_attn_res_lean_equiv")
def _m(ns):
    def patched(proj, norm):
        return proj.weight.squeeze(0).float()
    ns.model._attn_res_score_weight = patched


@_mutation("output_mixer_wrong_params", "test_full_model_logits")
def _m(ns):
    lean = ns.model._apply_attn_res_lean

    def patched(self, hidden_states, block_residual):
        batch_size, seq_len, hidden_size = hidden_states.shape
        last = self.layers[-1]
        hidden_states = lean(
            hidden_states.view(-1, hidden_size), block_residual,
            last.mlp_res_proj, last.mlp_res_norm,
        ).view(batch_size, seq_len, hidden_size)
        return self.norm(hidden_states)
    ns.model.KimiK3Model._finalize = patched


@_mutation("norm_before_output_mix", "test_full_model_logits")
def _m(ns):
    lean = ns.model._apply_attn_res_lean

    def patched(self, hidden_states, block_residual):
        batch_size, seq_len, hidden_size = hidden_states.shape
        hidden_states = self.norm(hidden_states)
        return lean(
            hidden_states.view(-1, hidden_size), block_residual,
            self.output_attn_res_proj, self.output_attn_res_norm,
        ).view(batch_size, seq_len, hidden_size)
    ns.model.KimiK3Model._finalize = patched


@_mutation("lean_revert_reference", "test_attn_res_lean_no_materialization",
           "validates the memory detector itself")
def _m(ns):
    def patched(prefix_sum, block_residual, proj, norm, chunk_size=1024):
        return apply_attn_res_reference(prefix_sum, block_residual, proj, norm)
    ns.model._apply_attn_res_lean = patched


# ---- SiTU -------------------------------------------------------------------
@_mutation("situ_gate_clamp_dropped", "test_situ_bitexact")
def _m(ns):
    def patched(self, x):
        d = x.shape[-1] // 2
        gate = x[..., :d].to(torch.float32)
        up = x[..., d:].to(torch.float32)
        situ_a = gate * torch.sigmoid(gate)          # SiLU: tanh clamp dropped
        up = self.linear_beta * torch.tanh(up / self.linear_beta)
        return (situ_a * up).to(x.dtype)
    ns.model.SituAndMul.forward = patched


@_mutation("situ_linear_clamp_dropped", "test_situ_bitexact")
def _m(ns):
    def patched(self, x):
        d = x.shape[-1] // 2
        gate = x[..., :d].to(torch.float32)
        up = x[..., d:].to(torch.float32)
        situ_a = self.beta * torch.tanh(gate / self.beta) * torch.sigmoid(gate)
        return (situ_a * up).to(x.dtype)             # up clamp dropped
    ns.model.SituAndMul.forward = patched


@_mutation("situ_bf16_interior", "test_situ_bitexact")
def _m(ns):
    def patched(self, x):
        d = x.shape[-1] // 2
        gate = x[..., :d]
        up = x[..., d:]
        situ_a = self.beta * torch.tanh(gate / self.beta) * torch.sigmoid(gate)
        up = self.linear_beta * torch.tanh(up / self.linear_beta)
        return situ_a * up                            # computed in input dtype
    ns.model.SituAndMul.forward = patched


# ---- Router -----------------------------------------------------------------
def _patched_gate_forward(ns, *, bf16_logits=False, topk_prebias=False,
                          gather_postbias=False, no_renorm=False):
    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(
            hidden_states.type(torch.float32), self.weight.type(torch.float32), None)
        if bf16_logits:
            logits = logits.bfloat16().float()
        scores = logits.sigmoid()
        scores_for_choice = scores + self.e_score_correction_bias.unsqueeze(0)
        sel = scores if topk_prebias else scores_for_choice
        _, topk_idx = torch.topk(sel, k=self.top_k, dim=-1, sorted=False)
        src = scores_for_choice if gather_postbias else scores
        topk_weight = src.gather(1, topk_idx)
        if self.top_k > 1 and self.moe_renormalize and not no_renorm:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator
        topk_weight = topk_weight * self.routed_scaling_factor
        return topk_idx, topk_weight
    return forward


@_mutation("router_bf16", "test_router_unit")
def _m(ns):
    ns.model.KimiMoEGate.forward = _patched_gate_forward(ns, bf16_logits=True)


@_mutation("router_topk_prebias", "test_router_unit")
def _m(ns):
    ns.model.KimiMoEGate.forward = _patched_gate_forward(ns, topk_prebias=True)


@_mutation("router_gather_postbias", "test_router_unit")
def _m(ns):
    ns.model.KimiMoEGate.forward = _patched_gate_forward(ns, gather_postbias=True)


@_mutation("router_no_renorm", "test_router_unit")
def _m(ns):
    ns.model.KimiMoEGate.forward = _patched_gate_forward(ns, no_renorm=True)


# ---- MoE --------------------------------------------------------------------
@_mutation("combine_bf16", "test_moe_bf16_bitwise")
def _m(ns):
    def patched(new_x, topk_shape, topk_weight):
        return (
            new_x.view(*topk_shape, -1)
            .mul(topk_weight.unsqueeze(dim=-1).to(new_x.dtype))
            .sum(dim=1)
        )
    ns.model._moe_combine = patched


@_mutation("latent_norm_dropped", "test_latent_moe_fp32")
def _m(ns):
    orig_init = ns.model.KimiSparseMoeBlock.__init__

    def patched_init(self, config):
        orig_init(self, config)
        self.routed_expert_norm = torch.nn.Identity()
        # keep the parameter so state-dict load still succeeds
        self.routed_expert_norm.weight = torch.nn.Parameter(
            torch.ones(self.moe_hidden_size))
    ns.model.KimiSparseMoeBlock.__init__ = patched_init


@_mutation("latent_norm_per_expert", "test_latent_moe_fp32")
def _m(ns):
    orig_infer = ns.model.KimiSparseMoeBlock.moe_infer
    orig_forward = ns.model.KimiSparseMoeBlock.forward

    def patched_forward(self, hidden_states):
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        hidden_states = self.routed_expert_down_proj(hidden_states)
        y = orig_infer(self, hidden_states, topk_idx, topk_weight)
        # norm INSIDE the combine path (per-token pre-combine is equivalent to
        # per-expert here) — outer norm skipped
        y = self.routed_expert_up_proj(y)
        y = y.view(*orig_shape)
        return y + self.shared_experts(identity)

    def patched_infer(self, x, topk_ids, topk_weight):
        cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0).cpu().numpy()
        idxs = topk_ids.view(-1).argsort()
        sorted_tokens = x[idxs // topk_ids.shape[1]]
        outputs = []
        start_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            out = self.experts[i](sorted_tokens[start_idx:end_idx])
            outputs.append(self.routed_expert_norm(out))   # per-expert norm
            start_idx = end_idx
        outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)
        new_x = torch.empty_like(outs)
        new_x[idxs] = outs
        return ns.model._moe_combine(new_x, topk_ids.shape, topk_weight)

    ns.model.KimiSparseMoeBlock.forward = patched_forward
    ns.model.KimiSparseMoeBlock.moe_infer = patched_infer


@_mutation("latent_proj_skipped", "test_latent_moe_fp32",
           "feeds hidden-width tokens to latent-width experts: shape crash = red")
def _m(ns):
    orig_forward = ns.model.KimiSparseMoeBlock.forward

    def patched(self, hidden_states):
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        y = self.moe_infer(hidden_states, topk_idx, topk_weight)   # no down_proj
        y = self.routed_expert_norm(y)
        y = self.routed_expert_up_proj(y)
        y = y.view(*orig_shape)
        return y + self.shared_experts(identity)
    ns.model.KimiSparseMoeBlock.forward = patched


@_mutation("w1_w3_swapped", "test_latent_moe_fp32")
def _m(ns):
    def patched(self, hidden_states):
        gate_up = torch.cat([self.w3(hidden_states), self.w1(hidden_states)], dim=-1)
        return self.w2(self.act_fn(gate_up))
    ns.model.KimiBlockSparseMLP.forward = patched


@_mutation("shared_fed_latent_y", "test_latent_moe_fp32")
def _m(ns):
    def patched(self, hidden_states):
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        hidden_states = self.routed_expert_down_proj(hidden_states)
        y = self.moe_infer(hidden_states, topk_idx, topk_weight)
        y = self.routed_expert_norm(y)
        y = self.routed_expert_up_proj(y)
        y = y.view(*orig_shape)
        return y + self.shared_experts(y)          # y, not identity
    ns.model.KimiSparseMoeBlock.forward = patched


@_mutation("shared_width_single", "test_state_dict_key_parity")
def _m(ns):
    orig_init = ns.model.KimiSparseMoeBlock.__init__

    def patched_init(self, config):
        orig_init(self, config)
        self.shared_experts = ns.model.KimiMLP(
            config, intermediate_size=int(config.moe_intermediate_size))
    ns.model.KimiSparseMoeBlock.__init__ = patched_init


# ---- KDA --------------------------------------------------------------------
@_mutation("kda_no_l2norm_flag", "test_kda_module_fp32")
def _m(ns):
    ns.model.KimiK3KDAAttention.use_qk_l2norm = False


@_mutation("kda_softplus_gate", "test_kda_module_fp32",
           "the softplus gate form is NOT what K3 runs (gate_lower_bound=-5.0 "
           "selects the lower-bound form)")
def _m(ns):
    orig = ns.model.KimiK3KDAAttention.__init__

    def patched(self, config, layer_idx, kda_backend="fla_chunk"):
        orig(self, config, layer_idx, kda_backend=kda_backend)
        self.gate_lower_bound = None
    ns.model.KimiK3KDAAttention.__init__ = patched


@_mutation("kda_beta_presigmoid", "test_kda_module_fp32")
def _m(ns):
    def patched(self, hidden_states):
        return self.b_proj(hidden_states).float().sigmoid()
    ns.model.KimiK3KDAAttention._beta = patched


@_mutation("kda_conv_no_silu", "test_kda_module_fp32")
def _m(ns):
    def patched(self, x):
        seq_len = x.shape[1]
        y = F.conv1d(
            x.transpose(1, 2).float(), self.weight, bias=None,
            groups=self.hidden_size, padding=self.kernel_size - 1,
        )[..., :seq_len]
        return y.transpose(1, 2).to(x.dtype)
    ns.model.CausalConv1dSilu.forward = patched


@_mutation("kda_conv_noncausal", "test_kda_module_fp32")
def _m(ns):
    def patched(self, x):
        seq_len = x.shape[1]
        y = F.conv1d(
            x.transpose(1, 2).float(), self.weight, bias=None,
            groups=self.hidden_size, padding=self.kernel_size - 1,
        )[..., 1:seq_len + 1]                      # looks one step ahead
        return F.silu(y).transpose(1, 2).to(x.dtype)
    ns.model.CausalConv1dSilu.forward = patched


@_mutation("dt_bias_transposed", "test_kda_module_fp32")
def _m(ns):
    def patched(self):
        return self.dt_bias.view(self.num_heads, self.head_dim).t().reshape(-1)
    ns.model.KimiK3KDAAttention._dt_bias = patched


@_mutation("a_log_offset", "test_kda_module_fp32 or test_a_log_pad_poison")
def _m(ns):
    def patched(self):
        return self.A_log[1: self.num_heads + 1]
    ns.model.KimiK3KDAAttention._a_log = patched


# ---- Layer map / structure --------------------------------------------------
@_mutation("layer_map_0based", "test_layer_type_map")
def _m(ns):
    def patched(self, layer_idx):
        return layer_idx in self.linear_attn_config["kda_layers"]
    ns.config.KimiK3Config.is_kda_layer = patched


@_mutation("layer0_moe", "test_state_dict_key_parity")
def _m(ns):
    orig = ns.model.KimiK3DecoderLayer.__init__

    def patched(self, config, layer_idx, kda_backend="fla_chunk"):
        import copy
        cfg = copy.copy(config)
        cfg.first_k_dense_replace = 0
        orig(self, cfg, layer_idx, kda_backend=kda_backend)
    ns.model.KimiK3DecoderLayer.__init__ = patched


# ---- Hard-fail perimeter ----------------------------------------------------
@_mutation("hard_fail_removed_decode", "test_hard_fail_decode")
def _m(ns):
    def patched(self, past_key_values, attention_mask, position_ids):
        return None
    ns.model.KimiK3Model._guard_prefill_only = patched


@_mutation("hard_fail_removed_vision", "test_hard_fail_vision_token")
def _m(ns):
    def patched(self, input_ids):
        return None
    ns.model.KimiK3Model._guard_no_vision_tokens = patched


@_mutation("hard_fail_removed_unknown_config", "test_hard_fail_unknown_config_key")
def _m(ns):
    def patched(unknown, where):
        return None
    ns.config._reject_unknown_keys = patched
