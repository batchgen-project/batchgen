# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                          #
#                                                                               #
#  licensed under the apache license, version 2.0 (the "license");              #
# ---------------------------------------------------------------------------- #

"""Kimi-K3 checkpoint <-> BatchGen tensor-name / shape map.

Owns the three declarations that must agree with the artifact or the model loads
and returns confidently wrong logits:

  1. ``state_dict_name_map`` — checkpoint name -> (module_key, tensor_key), for
     the four streamed module rings.
  2. ``module_shapes`` / ``weight_dtypes`` / ``tensor_dtypes`` — the GPU ring-slot
     geometry.
  3. the skeleton + explicit-ignore partition of everything else.

:func:`reconcile_k3_checkpoint` proves offline, from the shard headers, that the
three partition the checkpoint exactly and that the declared bytes equal
``metadata.total_size``.  It is wired into the parameter server as a mandatory
startup gate, not merely into a test: BatchGen has never exercised
``module_shapes`` for ``attn`` / ``kda_attn`` / ``shared_expert`` (those rings are
resident in every shipped model), so K3 is the first consumer of three quarters
of that surface.

Facts below were read out of the released checkpoint's shard headers:

  * 497,220 index entries, ``metadata.total_size`` 1,560,860,324,864 B, 60
    distinct name templates (29 module + 19 skeleton + 12 ignored).
  * every text tensor is prefixed ``language_model.``; Kimi-Linear-48B has no
    prefix.
  * ``self_attn.g_proj.weight`` appears on BOTH layer kinds (93x): the KDA
    full-rank gate and the MLA output gate.  One handler serves both.
  * ``self_attn.A_log`` ships **F32[128]** — a per-head [96] vector zero-padded up
    to 128 (ACTIVATION_FLOW.md D2).  Deriving its length from ``num_heads``
    under-declares the buffer.
  * ``self_attn.{q,k,v}_conv1d.weight`` and ``self_attn.o_norm.weight`` ship
    **F32** in K3 (they are BF16 in the 48B).  Same names, different checkpoint,
    different dtype — hence the explicit ``tensor_dtypes`` overrides.
  * routed experts ship ``w{1,2,3}.weight_packed`` + ``.weight_scale`` only.
    There is no ``.weight`` for any routed expert.

Scope: PREFILL ONLY.  Vision is explicitly ignorable, not silently dropped.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import torch

from batchgen.models.weight_reconciler import (
    IgnoreRule,
    ReconcileSpec,
    ReconcileReport,
    read_index,
    read_safetensors_headers,
    reconcile,
)

from .mxfp4_layout import (
    routed_expert_module_shapes,
    routed_expert_tensor_dtypes,
    routed_expert_tensor_names,
    validate_quantization_config,
)


# --------------------------------------------------------------------------- #
#  Checkpoint-level constants (verified against the released index/shards)      #
# --------------------------------------------------------------------------- #

#: Every K3 text tensor carries this prefix; Kimi-Linear-48B has none.
K3_CKPT_PREFIX = "language_model."

#: ``self_attn.A_log`` is a per-head [num_heads] vector zero-padded up to this
#: length.  MUST NOT be derived from the KDA head count (96): the checkpoint
#: ships F32[128] and a [96] declaration under-copies by 128 B per KDA layer.
K3_A_LOG_PADDED_LEN = 128

K3_MODULE_TYPES = ("attn", "kda_attn", "shared_expert", "routed_expert")

#: Checkpoint roots deliberately not loaded, with the reason.  Not dropped
#: silently: :func:`reconcile_k3_checkpoint` counts every one and fails if the
#: set changes size.
K3_IGNORE_RULES = (
    IgnoreRule(
        prefix="vision_tower.",
        reason=(
            "MoonViT-V2 tower — vision is out of scope for prefill-only K3 "
            "(PREFILL_PLAN.md §2), 165 tensors. An image token in a request "
            "must be rejected at admission, not silently dropped."
        ),
    ),
    IgnoreRule(
        prefix="mm_projector.",
        reason="Vision->text projector, out of scope with the tower it feeds (3 tensors).",
    ),
)

#: Exact size of the ignored set in the released full-depth checkpoint. Pinned so
#: a revised vision tower fails loudly instead of quietly changing the shm
#: reservation.  A depth-truncated variant must pass its own values explicitly.
K3_IGNORED_TENSOR_COUNT = 168
K3_IGNORED_BYTES = 894_717_952

#: ``Parameter_Server.cpp:404-405`` rounds the per-file global offset up to 4 KiB.
_SHM_ALIGNMENT = 4096


# --------------------------------------------------------------------------- #
#  Module tensor keys                                                          #
#                                                                              #
#  CONTRACT: tensor_key == checkpoint suffix == model.py parameter name.       #
#  ``apply_weights`` (models/wrappers/base.py:163-178) silently skips any name  #
#  that is not a module parameter, so a rename on either side is a             #
#  silent-zeros bug.                                                           #
# --------------------------------------------------------------------------- #

#: NoPE-MLA, 24 layers.  Deltas vs the 48B: Q is factored through a LoRA triple
#: instead of a single ``q_proj``, and there is a sigmoid output gate ``g_proj``.
K3_MLA_TENSOR_NAMES: Tuple[str, ...] = (
    "q_a_proj.weight",
    "q_a_layernorm.weight",
    "q_b_proj.weight",
    "kv_a_proj_with_mqa.weight",
    "kv_a_layernorm.weight",
    "kv_b_proj.weight",
    "o_proj.weight",
    "g_proj.weight",
)

#: KDA linear attention, 69 layers.  Delta vs the 48B: the gate is a single
#: FULL-RANK ``g_proj`` instead of the ``g_a_proj``/``g_b_proj`` low-rank pair.
K3_KDA_TENSOR_NAMES: Tuple[str, ...] = (
    "q_proj.weight",
    "k_proj.weight",
    "v_proj.weight",
    "q_conv1d.weight",
    "k_conv1d.weight",
    "v_conv1d.weight",
    "A_log",
    "f_a_proj.weight",
    "f_b_proj.weight",
    "dt_bias",
    "b_proj.weight",
    "g_proj.weight",
    "o_norm.weight",
    "o_proj.weight",
)

#: One shared-expert MLP per MoE layer (``n_shared_experts`` is a width
#: multiplier on ONE module, not a module count).  Identical to the 48B.
K3_SHARED_EXPERT_TENSOR_NAMES: Tuple[str, ...] = (
    "gate_proj.weight",
    "up_proj.weight",
    "down_proj.weight",
)


# --------------------------------------------------------------------------- #
#  Skeleton templates (resident; NOT in the name map)                          #
#                                                                              #
#  Every checkpoint tensor absent from the name map is promoted to             #
#  ``skeleton_state_dict_`` by the C++ parameter server without a word          #
#  (Parameter_Server.cpp:357-397).  Listing them explicitly is what lets the    #
#  reconciler prove the partition is total instead of trusting that default.    #
#  Names are model.py parameter names; the checkpoint name is the same string   #
#  prefixed with ``language_model.``.                                          #
# --------------------------------------------------------------------------- #

_SKELETON_GLOBAL = (
    ("model.embed_tokens.weight", "vocab_hidden", torch.bfloat16),
    ("lm_head.weight", "vocab_hidden", torch.bfloat16),
    ("model.norm.weight", "hidden", torch.bfloat16),
    # Block Attention Residual final stage (model level, after all layers).
    ("model.output_attn_res_norm.weight", "hidden", torch.bfloat16),
    ("model.output_attn_res_proj.weight", "one_hidden", torch.bfloat16),
)

#: On every layer.  ``*_res_*`` are the Block Attention Residual depth-mixer
#: parameters — no 48B analogue.
_SKELETON_PER_LAYER = (
    ("input_layernorm.weight", "hidden", torch.bfloat16),
    ("post_attention_layernorm.weight", "hidden", torch.bfloat16),
    ("self_attention_res_norm.weight", "hidden", torch.bfloat16),
    ("self_attention_res_proj.weight", "one_hidden", torch.bfloat16),
    ("mlp_res_norm.weight", "hidden", torch.bfloat16),
    ("mlp_res_proj.weight", "one_hidden", torch.bfloat16),
)

#: On MoE layers only.  The LatentMoE projections sit on the TOKEN stream (once
#: per token, outside dispatch) and are 51.4 MB each — resident by design.
_SKELETON_PER_MOE_LAYER = (
    ("block_sparse_moe.gate.weight", "experts_hidden", torch.bfloat16),
    ("block_sparse_moe.gate.e_score_correction_bias", "experts", torch.float32),
    ("block_sparse_moe.routed_expert_down_proj.weight", "latent_hidden", torch.bfloat16),
    ("block_sparse_moe.routed_expert_up_proj.weight", "hidden_latent", torch.bfloat16),
    ("block_sparse_moe.routed_expert_norm.weight", "latent", torch.bfloat16),
)

#: Layer 0 only (``first_k_dense_replace == 1``): a KDA layer with a dense MLP
#: and no ``block_sparse_moe`` subtree at all.
_SKELETON_DENSE_MLP = (
    ("mlp.gate_proj.weight", "ffn_hidden", torch.bfloat16),
    ("mlp.up_proj.weight", "ffn_hidden", torch.bfloat16),
    ("mlp.down_proj.weight", "hidden_ffn", torch.bfloat16),
)


# --------------------------------------------------------------------------- #
#  Dims + config validation                                                    #
# --------------------------------------------------------------------------- #

def _k3_dims(cfg) -> Dict[str, int]:
    lac = getattr(cfg, "linear_attn_config", None) or {}
    dims = {
        "hidden": int(cfg.hidden_size),
        "vocab": int(cfg.vocab_size),
        "ffn": int(cfg.intermediate_size),
        "latent": int(cfg.routed_expert_hidden_size),
        "moe_ffn": int(cfg.moe_intermediate_size),
        "experts": int(cfg.n_routed_experts),
        "kv_lora": int(cfg.kv_lora_rank),
        "q_lora": int(cfg.q_lora_rank),
        "nope": int(cfg.qk_nope_head_dim),
        "rope": int(cfg.qk_rope_head_dim),
        "v_head": int(cfg.v_head_dim),
        "heads": int(cfg.num_attention_heads),
        "kda_heads": int(lac["num_heads"]),
        "kda_head_dim": int(lac["head_dim"]),
        "conv_w": int(lac.get("short_conv_kernel_size", 4)),
    }
    dims["q_head"] = dims["nope"] + dims["rope"]                    # 192
    dims["kda_proj"] = dims["kda_heads"] * dims["kda_head_dim"]     # 12288
    dims["compressed_kv"] = dims["kv_lora"] + dims["rope"]          # 576
    dims["shared_ffn"] = int(cfg.n_shared_experts) * dims["moe_ffn"]  # 6144
    return dims


def validate_k3_config(cfg) -> None:
    """Hard-fail on every K3 feature switch that can be silently dropped.

    ``KimiLinearConfig.from_hf_dict`` filters unknown keys (config.py:150-151)
    and every K3-only switch defaults to ``None``/``False``, so a key-name miss
    produces a model that loads, runs, and is wrong.  Every problem is collected
    and reported at once, because fixing them one raise at a time is how a
    config lands half-right.

    If the released ``config.json`` spells one of these keys differently, add the
    alias to ``_HF_ALIASES`` in ``config.py``.  Do NOT delete the check.
    """
    problems: List[str] = []

    if getattr(cfg, "model_type", None) != "kimi_k3":
        problems.append(
            "model_type is {!r}, expected 'kimi_k3'. The name-pattern registry "
            "shortcut (model_registry.py:232-238) returns "
            "CONFIG_REGISTRY['kimi_k3']() == KimiLinearConfig() with its 48B "
            "DEFAULTS, whose model_type field is the literal 'kimi_linear'. "
            "Build the config with KimiLinearConfig.from_json(<cache_dir>/"
            "config.json), which flattens text_config and stamps 'kimi_k3'."
            .format(getattr(cfg, "model_type", None))
        )

    lac = getattr(cfg, "linear_attn_config", None) or {}
    if not lac:
        problems.append(
            "linear_attn_config is absent — is_kda_layer() would return False "
            "for every layer and all layers would silently become MLA "
            "(config.py:172-175)."
        )
    else:
        kda_layers = lac.get("kda_layers")
        full_attn = lac.get("full_attn_layers")
        n_layers = int(getattr(cfg, "num_hidden_layers", 0) or 0)
        if not kda_layers:
            problems.append(
                "linear_attn_config['kda_layers'] is missing or empty. "
                "is_kda_layer() reads ONLY this key (config.py:172-175), so "
                "every layer would be treated as MLA and the KDA weights would "
                "have no destination. If the checkpoint declares the split as "
                "'full_attn_layers' only, derive kda_layers from it in "
                "config.py — explicitly, 1-indexed — rather than defaulting."
            )
        elif full_attn:
            expected = set(range(1, n_layers + 1))
            union = set(int(x) for x in kda_layers) | set(int(x) for x in full_attn)
            overlap = set(int(x) for x in kda_layers) & set(int(x) for x in full_attn)
            if overlap or union != expected:
                problems.append(
                    "linear_attn_config layer lists do not partition "
                    "1..{} (1-INDEXED, configuration_kimi_k3.py layer lists). "
                    "overlap={}, missing={}, extra={}. A 0-indexed read offsets "
                    "every attention layer by one and loads MLA weights into "
                    "KDA modules.".format(
                        n_layers, sorted(overlap)[:8],
                        sorted(expected - union)[:8], sorted(union - expected)[:8],
                    )
                )
        if not lac.get("use_full_rank_gate", False):
            problems.append(
                "linear_attn_config['use_full_rank_gate'] is not True, but the "
                "checkpoint ships a single full-rank self_attn.g_proj on every "
                "layer and no g_a_proj/g_b_proj anywhere."
            )

    if getattr(cfg, "q_lora_rank", None) is None:
        problems.append(
            "q_lora_rank is None — K3 MLA factors Q through "
            "q_a_proj/q_a_layernorm/q_b_proj (1536). A direct q_proj has no "
            "checkpoint tensor to load from."
        )
    if not getattr(cfg, "mla_use_output_gate", False):
        problems.append(
            "mla_use_output_gate is False, but the checkpoint ships "
            "self_attn.g_proj on every MLA layer."
        )
    if getattr(cfg, "routed_expert_hidden_size", None) is None:
        problems.append(
            "routed_expert_hidden_size is None — LatentMoE would be off and "
            "experts would run in hidden space (K=7168), but the checkpoint's "
            "expert K is 3584/3072."
        )
    if not getattr(cfg, "latent_moe_use_norm", False):
        problems.append(
            "latent_moe_use_norm is False, but the checkpoint ships "
            "block_sparse_moe.routed_expert_norm on every MoE layer."
        )
    if getattr(cfg, "attn_res_block_size", None) is None:
        problems.append(
            "attn_res_block_size is None — Block Attention Residuals would be "
            "off, but the checkpoint ships self_attention_res_* / mlp_res_* on "
            "every layer and output_attn_res_* at model level."
        )
    if getattr(cfg, "hidden_act", None) != "situ":
        problems.append(
            "hidden_act is {!r}, expected 'situ'.".format(
                getattr(cfg, "hidden_act", None))
        )
    if getattr(cfg, "activation_situ_beta", None) is None or \
            getattr(cfg, "activation_situ_linear_beta", None) is None:
        problems.append(
            "activation_situ_beta / activation_situ_linear_beta are absent — "
            "model.py falls back to SiLU via `beta or 1.0`, which also fires on "
            "a legitimate 0.0."
        )
    if int(getattr(cfg, "num_nextn_predict_layers", 0) or 0) != 0:
        problems.append(
            "num_nextn_predict_layers != 0 — MTP heads are unimplemented for "
            "K3 and the released checkpoint ships 0."
        )
    if int(getattr(cfg, "first_k_dense_replace", 0) or 0) < 1:
        problems.append(
            "first_k_dense_replace < 1, but the checkpoint ships "
            "layers.0.mlp.{gate,up,down}_proj — layer 0 is dense."
        )

    try:
        validate_quantization_config(getattr(cfg, "quantization_config", None))
    except Exception as exc:                       # noqa: BLE001 — re-reported below
        problems.append(str(exc))

    if problems:
        raise ValueError(
            "Kimi-K3 config is not loadable — {} problem(s):\n  - {}".format(
                len(problems), "\n  - ".join(problems))
        )

    if not cfg.is_kda_layer(0):
        raise ValueError(
            "Layer 0 must be KDA — the checkpoint ships layers.0.self_attn."
            "A_log and layers.0.self_attn.q_conv1d (and a dense layers.0.mlp)."
        )


# --------------------------------------------------------------------------- #
#  1. state_dict_name_map + weight_copy_task                                    #
# --------------------------------------------------------------------------- #

def build_k3_state_dict_name_map(cfg) -> Tuple[Dict[str, Dict[str, str]],
                                               Dict[str, List[str]]]:
    """Return ``(state_dict_name_map, weight_copy_task)`` for Kimi-K3.

    ``weight_copy_task`` lists module keys in layer-major ascending order — the
    order the forwards MUST consume them in, or the single-threaded producer
    stalls and ``get_weights`` throws after 2 s.  The queue the worker actually
    installs comes from the PSM's ``configure_prefill()``; this is the reference
    ordering that must reproduce.
    """
    validate_k3_config(cfg)
    prefix = K3_CKPT_PREFIX
    n_layers = int(cfg.num_hidden_layers)
    n_experts = int(cfg.n_routed_experts)
    first_moe = int(cfg.first_k_dense_replace)
    expert_names = routed_expert_tensor_names()

    name_map: Dict[str, Dict[str, str]] = {}
    task: Dict[str, List[str]] = {t: [] for t in K3_MODULE_TYPES}

    for layer in range(n_layers):
        # --- attention: one g_proj handler serves BOTH layer kinds ---
        if cfg.is_kda_layer(layer):
            module_key = "kda_attn_{}".format(layer)
            names, ring = K3_KDA_TENSOR_NAMES, "kda_attn"
        else:
            module_key = "attn_{}".format(layer)
            names, ring = K3_MLA_TENSOR_NAMES, "attn"
        for name in names:
            full = "{}model.layers.{}.self_attn.{}".format(prefix, layer, name)
            name_map[full] = {"module_key": module_key, "tensor_key": name}
        task[ring].append(module_key)

        # --- MoE (layer 0 is dense: no block_sparse_moe subtree at all) ---
        if layer < first_moe:
            continue

        for name in K3_SHARED_EXPERT_TENSOR_NAMES:
            full = "{}model.layers.{}.block_sparse_moe.shared_experts.{}".format(
                prefix, layer, name)
            name_map[full] = {
                "module_key": "shared_expert_{}".format(layer),
                "tensor_key": name,
            }
        task["shared_expert"].append("shared_expert_{}".format(layer))

        for expert in range(n_experts):
            module_key = "routed_expert_{}_{}".format(layer, expert)
            base = "{}model.layers.{}.block_sparse_moe.experts.{}.".format(
                prefix, layer, expert)
            for name in expert_names:
                name_map[base + name] = {
                    "module_key": module_key, "tensor_key": name,
                }
            task["routed_expert"].append(module_key)

    return name_map, task


def k3_skeleton_declaration(cfg) -> Dict[str, Tuple[List[int], torch.dtype]]:
    """Explicit skeleton allowlist: ckpt name -> (shape, dtype).

    The C++ parameter server keys ``skeleton_state_dict_`` by the CHECKPOINT
    name, so K3's entries carry the ``language_model.`` prefix while
    ``model.named_parameters()`` does not.  Use :func:`k3_skeleton_key` to bridge.
    """
    dims = _k3_dims(cfg)
    shape_of = {
        "hidden": [dims["hidden"]],
        "one_hidden": [1, dims["hidden"]],
        "vocab_hidden": [dims["vocab"], dims["hidden"]],
        "latent": [dims["latent"]],
        "experts": [dims["experts"]],
        "experts_hidden": [dims["experts"], dims["hidden"]],
        "latent_hidden": [dims["latent"], dims["hidden"]],
        "hidden_latent": [dims["hidden"], dims["latent"]],
        "ffn_hidden": [dims["ffn"], dims["hidden"]],
        "hidden_ffn": [dims["hidden"], dims["ffn"]],
    }
    out: Dict[str, Tuple[List[int], torch.dtype]] = {}

    def add(param_name: str, shape_key: str, dtype: torch.dtype) -> None:
        out[K3_CKPT_PREFIX + param_name] = (shape_of[shape_key], dtype)

    for param_name, shape_key, dtype in _SKELETON_GLOBAL:
        add(param_name, shape_key, dtype)
    for layer in range(int(cfg.num_hidden_layers)):
        for suffix, shape_key, dtype in _SKELETON_PER_LAYER:
            add("model.layers.{}.{}".format(layer, suffix), shape_key, dtype)
        if layer < int(cfg.first_k_dense_replace):
            for suffix, shape_key, dtype in _SKELETON_DENSE_MLP:
                add("model.layers.{}.{}".format(layer, suffix), shape_key, dtype)
        else:
            for suffix, shape_key, dtype in _SKELETON_PER_MOE_LAYER:
                add("model.layers.{}.{}".format(layer, suffix), shape_key, dtype)
    return out


def k3_skeleton_key(model_param_name: str) -> str:
    """``model.layers.3.mlp_res_norm.weight`` -> the K3 checkpoint name."""
    return K3_CKPT_PREFIX + model_param_name


# --------------------------------------------------------------------------- #
#  2. module_shapes / dtypes                                                    #
# --------------------------------------------------------------------------- #

def k3_module_shapes(cfg) -> Tuple[
    Dict[str, Dict[str, List[int]]],
    Dict[str, torch.dtype],
    Dict[str, Dict[str, torch.dtype]],
]:
    """Return ``(module_shapes, weight_dtypes, tensor_dtypes)`` for K3.

    Cross-checked byte-for-byte against the released index by
    :func:`reconcile_k3_checkpoint`.  A wrong entry does NOT fail on GPU:
    ``blocking_copy_`` (HtoD_Engine.cu:232-238) copies the source byte size into the
    destination with no bound check, so an under-declared buffer overruns into
    the neighbouring tensor of the same slot and an over-declared one keeps a
    stale tail.
    """
    dims = _k3_dims(cfg)
    hidden = dims["hidden"]

    module_shapes: Dict[str, Dict[str, List[int]]] = {
        # NoPE-MLA, 464,392,192 B/layer
        "attn": {
            "q_a_proj.weight": [dims["q_lora"], hidden],                  # [1536, 7168]
            "q_a_layernorm.weight": [dims["q_lora"]],                     # [1536]  eps 1e-6
            "q_b_proj.weight": [dims["heads"] * dims["q_head"],
                                dims["q_lora"]],                          # [18432, 1536]
            "kv_a_proj_with_mqa.weight": [dims["compressed_kv"], hidden],  # [576, 7168]
            "kv_a_layernorm.weight": [dims["kv_lora"]],                   # [512]   eps 1e-6
            "kv_b_proj.weight": [dims["heads"] * (dims["nope"] + dims["v_head"]),
                                 dims["kv_lora"]],                        # [24576, 512]
            "o_proj.weight": [hidden, dims["heads"] * dims["v_head"]],    # [7168, 12288]
            "g_proj.weight": [dims["heads"] * dims["v_head"], hidden],    # [12288, 7168]
        },
        # KDA, 887,160,832 B BF16 + 640,000 B F32 per layer
        "kda_attn": {
            "q_proj.weight": [dims["kda_proj"], hidden],                  # [12288, 7168]
            "k_proj.weight": [dims["kda_proj"], hidden],
            "v_proj.weight": [dims["kda_proj"], hidden],
            "q_conv1d.weight": [dims["kda_proj"], 1, dims["conv_w"]],     # F32 [12288,1,4]
            "k_conv1d.weight": [dims["kda_proj"], 1, dims["conv_w"]],
            "v_conv1d.weight": [dims["kda_proj"], 1, dims["conv_w"]],
            "A_log": [K3_A_LOG_PADDED_LEN],                               # F32 [128]
            "f_a_proj.weight": [dims["kda_head_dim"], hidden],            # [128, 7168]
            "f_b_proj.weight": [dims["kda_proj"], dims["kda_head_dim"]],  # [12288, 128]
            "dt_bias": [dims["kda_proj"]],                                # F32 [12288]
            "b_proj.weight": [dims["kda_heads"], hidden],                 # [96, 7168]
            "g_proj.weight": [dims["kda_proj"], hidden],                  # [12288, 7168]
            "o_norm.weight": [dims["kda_head_dim"]],                      # F32 [128]
            "o_proj.weight": [hidden, dims["kda_proj"]],                  # [7168, 12288]
        },
        # shared expert, 264,241,152 B/layer
        "shared_expert": {
            "gate_proj.weight": [dims["shared_ffn"], hidden],             # [6144, 7168]
            "up_proj.weight": [dims["shared_ffn"], hidden],               # [6144, 7168]
            "down_proj.weight": [hidden, dims["shared_ffn"]],             # [7168, 6144]
        },
        # MXFP4 routed expert, 17,547,264 B each
        "routed_expert": routed_expert_module_shapes(
            dims["moe_ffn"], dims["latent"]),
    }

    weight_dtypes: Dict[str, torch.dtype] = {
        "attn": torch.bfloat16,
        "kda_attn": torch.bfloat16,
        "shared_expert": torch.bfloat16,
        # Routed experts stream PACKED; dequant happens inside the GEMM.
        "routed_expert": torch.uint8,
    }

    # Per-tensor overrides. NOT cosmetic: the slot is sized from these, and K3's
    # conv1d/A_log/dt_bias/o_norm are F32 where the 48B's are BF16 — a BF16
    # declaration is a 2x overrun per conv tensor (196,608 B into 98,304 B).
    tensor_dtypes: Dict[str, Dict[str, torch.dtype]] = {
        "kda_attn": {
            "q_conv1d.weight": torch.float32,
            "k_conv1d.weight": torch.float32,
            "v_conv1d.weight": torch.float32,
            "A_log": torch.float32,
            "dt_bias": torch.float32,
            "o_norm.weight": torch.float32,
        },
        "routed_expert": routed_expert_tensor_dtypes(),
    }
    return module_shapes, weight_dtypes, tensor_dtypes


# --------------------------------------------------------------------------- #
#  3. SHM reservation — computed, never a constant                             #
# --------------------------------------------------------------------------- #

def k3_shm_byte_size(cache_dir: str) -> int:
    """Shm reservation for K3, from the checkpoint index. Cheap (one JSON read).

    ``metadata.total_size`` is the exact sum of tensor bytes; the parameter
    server additionally 4 KiB-aligns each converted file's global offset
    (Parameter_Server.cpp:404-405), so add one alignment unit per shard.

    Released K3: 1,560,860,324,864 + 96*4096 = 1,560,860,718,080 B = 1453.66 GiB
    — 13.2x the 110 GiB constant the 48B path uses, which would SIGBUS mid-load.
    """
    index_path = os.path.join(cache_dir or "", "model.safetensors.index.json")
    if not os.path.isfile(index_path):
        raise FileNotFoundError(
            "{} not found. The K3 shm reservation is computed from the "
            "checkpoint index; there is no default byte size and there must not "
            "be one.".format(index_path)
        )
    index = read_index(index_path)
    return index.total_bytes + index.num_shards * _SHM_ALIGNMENT


# --------------------------------------------------------------------------- #
#  4. Offline reconciliation gate                                              #
# --------------------------------------------------------------------------- #

def k3_reconcile_spec(cfg,
                      ignored_count: Optional[int] = K3_IGNORED_TENSOR_COUNT,
                      ignored_bytes: Optional[int] = K3_IGNORED_BYTES
                      ) -> ReconcileSpec:
    """Package K3's declarations for :func:`weight_reconciler.reconcile`.

    ``ignored_*`` default to the released full-depth checkpoint's pinned values.
    Both are depth-INDEPENDENT (the vision tower has no per-layer tensors), so a
    depth-truncated stage (K3-24L) passes unchanged **provided it keeps the 168
    ``vision_tower.*`` / ``mm_projector.*`` tensors** — 894,717,952 B, which is
    0.06 % of the checkpoint and the reason to keep them rather than plumb an
    override.  A stage that drops them must pass ``ignored_count=0,
    ignored_bytes=0`` explicitly (``reconcile_k3_checkpoint`` forwards both), so
    the change is visible in the call site rather than absorbed silently.
    ``ignored_bytes`` is used only in index mode; header mode measures it.
    """
    name_map, _ = build_k3_state_dict_name_map(cfg)
    module_shapes, weight_dtypes, tensor_dtypes = k3_module_shapes(cfg)
    return ReconcileSpec(
        name_map=name_map,
        module_shapes=module_shapes,
        weight_dtypes=weight_dtypes,
        tensor_dtypes=tensor_dtypes,
        skeleton=k3_skeleton_declaration(cfg),
        ignore_rules=K3_IGNORE_RULES,
        ignored_count=ignored_count,
        ignored_bytes=ignored_bytes,
    )


def reconcile_k3_checkpoint(cache_dir: str, cfg, use_shard_headers: bool = True,
                            **spec_kwargs: Any) -> ReconcileReport:
    """Prove the name map + skeleton + ignore list partition the checkpoint.

    ``use_shard_headers=True`` (DEFAULT) reads every shard's safetensors JSON
    header — a few hundred KB each, no weight byte — and compares shape AND
    dtype per tensor.  This is the only mode that catches a BYTE-NEUTRAL shape
    error: declaring ``o_proj.weight`` as [12288, 7168] instead of [7168, 12288]
    passes the aggregate byte total, sizes the ``torch::zeros`` slot correctly
    (GPU_Weight_Buffer.cpp:121-122), copies without complaint, and hands the
    GEMM a transposed weight.  Measured on the released checkpoint: 6.1 s.

    ``use_shard_headers=False`` reads only ``model.safetensors.index.json`` —
    names plus ``metadata.total_size``.  It catches every name-level defect and
    any error in the declared byte TOTAL, and needs no shards, but it is blind
    to a transpose.  Use it only where the shards are genuinely unavailable, and
    never as the production gate.
    """
    spec = k3_reconcile_spec(cfg, **spec_kwargs)
    if use_shard_headers:
        checkpoint = read_safetensors_headers(cache_dir)
    else:
        index_path = os.path.join(cache_dir, "model.safetensors.index.json")
        if not os.path.isfile(index_path):
            raise FileNotFoundError(
                "{} not found — the K3 name map cannot be verified and must "
                "not be used unverified.".format(index_path)
            )
        checkpoint = read_index(index_path)
    return reconcile(checkpoint, spec)


def load_k3_config(cache_dir: str):
    """Build a validated K3 config from ``<cache_dir>/config.json``.

    The only supported way to get a K3 config: ``load_config(<hf id>)`` takes the
    name-pattern shortcut and returns ``KimiLinearConfig()`` with 48B DEFAULTS
    (model_registry.py:232-238), which is a loadable, runnable, wrong model.
    """
    cfg_json = os.path.join(cache_dir or "", "config.json")
    if not os.path.isfile(cfg_json):
        raise FileNotFoundError(
            "Kimi-K3 requires an on-disk config.json under --cache-dir (looked "
            "for {}). Every K3 feature switch is Optional/None in the built-in "
            "defaults and a miss on any one of them loads a wrong model."
            .format(cfg_json)
        )
    # Absolute, not relative: this module is also loaded by file path (offline
    # tests, tooling), where a `from ..config import` escapes the top-level
    # package and raises ImportError.
    from batchgen.models.moonshotai.kimi_linear.config import KimiLinearConfig

    cfg = KimiLinearConfig.from_json(cfg_json)
    validate_k3_config(cfg)
    return cfg
