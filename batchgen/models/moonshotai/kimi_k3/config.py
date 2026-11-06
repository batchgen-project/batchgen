# ---------------------------------------------------------------------------- #
#  BatchGen — Kimi-K3                                                           #
#  Copyright (c) 2025-2026 BatchGen Team                                        #
#  Licensed under the Apache License, Version 2.0                               #
# ---------------------------------------------------------------------------- #
"""Strict Kimi-K3 / Kimi-K3.1 text-model configuration (M2, prefill-only).

Parses the checkpoint's ``config.json`` (the K3 form: everything under
``text_config``) into a flat, validated :class:`KimiK3Config`.

Two checkpoints share this parser, distinguished by the TOP-LEVEL
``model_type`` of the VLM wrapper:

  * ``kimi_k3``   — ``moonshotai/Kimi-K3``, no ``dsa_config``.
  * ``kimi_k31``  — ``moonshotai/Kimi-K3.1-Pro``: byte-identical architecture
    plus a DeepSeek-style sparse-attention indexer on the 24 full-attention
    layers, declared by ``text_config.dsa_config``.  Nothing else differs:
    the released indexes differ by exactly 120 tensors (5 patterns x 24
    layers, see ``kimi_linear/k3/tensor_map.py``), SiTU/KDA/MoE/MXFP4/MLA
    geometry and the tokenizer are identical.

The two are kept apart by a hard requirement rather than a default:
``dsa_config`` is REQUIRED for ``kimi_k31`` and REFUSED for ``kimi_k3``.  A
default would let a K3.1 checkpoint load as a dense K3 — loadable, runnable,
and silently attending to the whole context on every full-attention layer.

Design decisions (POIS ledger 2026-08-04, decision 1 — hard-fail everywhere):

  * UNKNOWN keys raise.  ``KimiLinearConfig.from_hf_dict``
    (``batchgen/models/moonshotai/kimi_linear/config.py:150-151``) silently
    filters unknown keys and silently defaults missing ones; that behavior is a
    correctness hazard for K3 (every K3 feature switch defaults to off) and is
    deliberately NOT reused here.  Keys are checked against two explicit
    allowlists: the architecture keys of the checkpoint's own
    ``configuration_kimi_k3.py`` (vendored at ``assets/configuration_kimi_k3.py``,
    md5 3165dde7cebe8471fdf43aa9890d5c02) and the HF bookkeeping keys
    ``transformers`` 4.56.2 injects into a serialized config.
  * REQUIRED keys raise when absent — no silent defaulting of load-bearing
    architecture fields.
  * Every K3 invariant this model build depends on is validated here, once, so
    ``model.py`` can consume the config without re-checking.

This module is intentionally free of ``batchgen`` imports so it can be loaded by
file path in offline tests (the ``tests/test_kimi_k3_tensor_map.py`` pattern)
without JIT-building the core engine.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# --------------------------------------------------------------------------- #
#  Constants                                                                   #
# --------------------------------------------------------------------------- #

#: ``self_attn.A_log`` ships F32[128] in the released checkpoint: a per-head
#: [num_heads] vector zero-padded up to 128 (ACTIVATION_FLOW.md D2,
#: kimi_linear/k3/tensor_map.py:K3_A_LOG_PADDED_LEN).  model.py registers the
#: padded buffer and consumes entries [:num_heads] only.
K3_A_LOG_PADDED_LEN = 128

#: Top-level ``media_placeholder_token_id`` of the released checkpoint.
K3_DEFAULT_MEDIA_PLACEHOLDER_TOKEN_ID = 163605


# Architecture keys of the checkpoint's own configuration class
# (assets/configuration_kimi_k3.py:10-58, KimiLinearConfig.__init__ signature).
_ARCH_KEYS = frozenset({
    "model_type",
    "vocab_size",
    "hidden_size",
    "head_dim",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "hidden_act",
    "initializer_range",
    "rms_norm_eps",
    "use_cache",
    "pad_token_id",
    "bos_token_id",
    "eos_token_id",
    "rope_theta",
    "rope_scaling",
    "tie_word_embeddings",
    "moe_intermediate_size",
    "moe_renormalize",
    "moe_router_activation_func",
    "num_experts",
    "num_experts_per_token",
    "num_shared_experts",
    "routed_scaling_factor",
    "first_k_dense_replace",
    "moe_layer_freq",
    "use_grouped_topk",
    "num_expert_group",
    "topk_group",
    "q_lora_rank",
    "kv_lora_rank",
    "qk_nope_head_dim",
    "qk_rope_head_dim",
    "v_head_dim",
    "mla_use_nope",
    "mla_use_output_gate",
    "num_nextn_predict_layers",
    "linear_attn_config",
    "attn_res_block_size",
    "latent_moe_use_norm",
    "activation_situ_beta",
    "activation_situ_linear_beta",
    "max_position_embeddings",
    "routed_expert_hidden_size",
    "topk_method",
    "quantization_config",
    # K3.1 only (absent from K3's config.json); see _DSA_KEYS.
    "dsa_config",
})

#: The eight keys of K3.1's ``text_config.dsa_config``.  The checkpoint's own
#: ``configuration_kimi_k3.py`` stores the block as an opaque
#: ``dsa_config: Optional[dict] = None`` and validates NOTHING in it (unlike
#: ``linear_attn_config``, which it asserts on), so this allowlist is the only
#: place the block is checked at all.  Keys outside it raise.
_DSA_KEYS = frozenset({
    "dsa_layers",
    "index_topk",
    "index_head_dim",
    "index_num_heads",
    "index_rope_dim",
    "index_k_norm_type",
    "index_topk_freq",
    "index_skip_topk_offset",
})

#: The only index-K normalisation this port implements.  K3.1 ships a
#: LayerNorm (``indexer.k_norm.weight`` AND ``indexer.k_norm.bias``); an RMS
#: variant would ship no bias, so the name is load-bearing for the weight map.
_DSA_K_NORM_LAYERNORM = "layernorm"

# HF bookkeeping keys `transformers` (4.56.2, the checkpoint's own
# transformers_version) serializes into every config.json.  Accepted and
# ignored EXPLICITLY — they carry generation defaults / framework metadata,
# not architecture.  Anything outside _ARCH_KEYS | _HF_BOOKKEEPING_KEYS raises.
_HF_BOOKKEEPING_KEYS = frozenset({
    "_name_or_path",
    "add_cross_attention",
    "architectures",
    "auto_map",
    "bad_words_ids",
    "begin_suppress_tokens",
    "chunk_size_feed_forward",
    "cross_attention_hidden_size",
    "decoder_start_token_id",
    "diversity_penalty",
    "do_sample",
    "dtype",
    "torch_dtype",
    "early_stopping",
    "encoder_no_repeat_ngram_size",
    "exponential_decay_length_penalty",
    "finetuning_task",
    "forced_bos_token_id",
    "forced_eos_token_id",
    "id2label",
    "is_decoder",
    "is_encoder_decoder",
    "label2id",
    "length_penalty",
    "max_length",
    "min_length",
    "no_repeat_ngram_size",
    "num_beam_groups",
    "num_beams",
    "num_return_sequences",
    "output_attentions",
    "output_hidden_states",
    "output_scores",
    "prefix",
    "problem_type",
    "pruned_heads",
    "remove_invalid_values",
    "repetition_penalty",
    "return_dict",
    "return_dict_in_generate",
    "sep_token_id",
    "suppress_tokens",
    "task_specific_params",
    "temperature",
    "tf_legacy_loss",
    "tie_encoder_decoder",
    "tokenizer_class",
    "top_k",
    "top_p",
    "torchscript",
    "transformers_version",
    "typical_p",
    "use_bfloat16",
})

_LINEAR_ATTN_KEYS = frozenset({
    "kda_layers",
    "full_attn_layers",
    "num_heads",
    "head_dim",
    "short_conv_kernel_size",
    "use_full_rank_gate",
    "gate_lower_bound",
})

# Top-level config.json keys of the K3 (VLM-wrapped) checkpoint.
_TOP_LEVEL_KEYS = frozenset({
    "architectures",
    "auto_map",
    "bos_token_id",
    "eos_token_id",
    "pad_token_id",
    "dtype",
    "torch_dtype",
    "ignore_index",
    "image_placeholder",
    "media_placeholder_token_id",
    "model_type",
    "text_config",
    "tie_word_embeddings",
    "transformers_version",
    "vision_config",
})


def _reject_unknown_keys(unknown: List[str], where: str) -> None:
    """Hard-fail on unknown config keys.  Factored out as the mutation seam for
    ``hard_fail_removed_unknown_config`` in the CPU test suite."""
    if unknown:
        raise ValueError(
            "Kimi-K3 config: unknown key(s) in {}: {}. Unknown keys are never "
            "silently dropped (POIS decision 1); if this key is legitimate, add "
            "it to the explicit allowlist in kimi_k3/config.py with a comment "
            "saying what consumes it.".format(where, sorted(unknown))
        )


def _require(d: Dict[str, Any], key: str, where: str) -> Any:
    if key not in d or d[key] is None:
        raise ValueError(
            "Kimi-K3 config: required key '{}' is missing (or null) in {}. "
            "Load-bearing architecture fields are never defaulted.".format(key, where)
        )
    return d[key]


def _require_present(d: Dict[str, Any], key: str, where: str) -> Any:
    """:func:`_require` for a key whose legitimate value may be falsy.

    ``index_rope_dim`` is 0 in the released K3.1 config; ``_require``'s
    ``is None`` test passes it, but a future truthiness-based read would not,
    so the presence-only check is explicit here.
    """
    if key not in d:
        raise ValueError(
            "Kimi-K3 config: required key '{}' is missing in {}. Load-bearing "
            "architecture fields are never defaulted.".format(key, where)
        )
    value = d[key]
    if value is None:
        raise ValueError(
            "Kimi-K3 config: required key '{}' in {} is null.".format(key, where)
        )
    return value


def dsa_layer_skips_topk(freq: int, offset: int, layer_idx: int) -> bool:
    """Whether a DSA layer reuses the PREVIOUS indexer layer's top-k indices.

    Same frequency/offset formula the repo already carries for GLM-5.2 at
    ``batchgen/models/glm/glm5/dsa_schedule.py:dsa_layer_skips_topk`` (ported
    there from the upstream reference schedule).  Restated here as a pure
    function of the two integers so this package keeps its no-cross-model-
    import property; ``tests/test_kimi_k31_config.py`` asserts the two agree
    layer-for-layer over the released K3.1 and GLM-5.2 parameters.

    ``freq == 1`` means every layer recomputes its own top-k and the offset is
    inert.  K3.1 ships ``index_topk_freq=1``, which is independently confirmed
    by its checkpoint: ALL 24 full-attention layers carry indexer weights, so
    no layer has a previous layer's selection to reuse.
    """
    if freq <= 0:
        raise ValueError("index_topk_freq must be positive, got {}".format(freq))
    if freq == 1:
        return False
    if offset <= 0:
        raise ValueError(
            "index_skip_topk_offset must be positive; offset <= 0 marks layer 0 "
            "as a reuse layer with no prior top-k to reuse (got {})".format(offset))
    return max(layer_idx - offset + 1, 0) % freq != 0


@dataclass(frozen=True)
class KimiDsaConfig:
    """Validated ``text_config.dsa_config`` of Kimi-K3.1.

    Released values: ``dsa_layers`` = the 24 full-attention layers (1-INDEXED,
    exactly ``linear_attn_config['full_attn_layers']``), ``index_topk`` 2048,
    ``index_head_dim`` 128, ``index_num_heads`` 32, ``index_rope_dim`` 0,
    ``index_k_norm_type`` "layernorm", ``index_topk_freq`` 1,
    ``index_skip_topk_offset`` 1.

    Every field is required; there are no defaults.  The indexer is the only
    thing that distinguishes K3.1 from K3, so a defaulted field here is a
    silently different model.
    """

    dsa_layers: tuple
    index_topk: int
    index_head_dim: int
    index_num_heads: int
    index_rope_dim: int
    index_k_norm_type: str
    index_topk_freq: int
    index_skip_topk_offset: int

    def is_dsa_layer(self, layer_idx: int) -> bool:
        """``dsa_layers`` is 1-INDEXED, like ``linear_attn_config``'s lists."""
        return (layer_idx + 1) in self.dsa_layers

    def skips_topk(self, layer_idx: int) -> bool:
        return dsa_layer_skips_topk(
            self.index_topk_freq, self.index_skip_topk_offset, layer_idx)

    @property
    def index_dim(self) -> int:
        """Q width per token: ``index_num_heads * index_head_dim`` (4096)."""
        return self.index_num_heads * self.index_head_dim


def parse_dsa_config(raw: Dict[str, Any], num_hidden_layers: int) -> KimiDsaConfig:
    """Parse and validate ``text_config.dsa_config``.  Unknown/missing raise."""
    if not isinstance(raw, dict):
        raise ValueError(
            "dsa_config must be an object, got {}".format(type(raw).__name__))
    _reject_unknown_keys([k for k in raw if k not in _DSA_KEYS], "dsa_config")
    where = "dsa_config"

    layers = tuple(int(x) for x in _require(raw, "dsa_layers", where))
    cfg = KimiDsaConfig(
        dsa_layers=layers,
        index_topk=int(_require(raw, "index_topk", where)),
        index_head_dim=int(_require(raw, "index_head_dim", where)),
        index_num_heads=int(_require(raw, "index_num_heads", where)),
        # `_require` rejects null but NOT 0, and 0 is this key's released
        # value, so it must be read with an explicit presence test instead.
        index_rope_dim=int(_require_present(raw, "index_rope_dim", where)),
        index_k_norm_type=str(_require(raw, "index_k_norm_type", where)),
        index_topk_freq=int(_require(raw, "index_topk_freq", where)),
        index_skip_topk_offset=int(_require(raw, "index_skip_topk_offset", where)),
    )

    problems: List[str] = []
    for name in ("index_topk", "index_head_dim", "index_num_heads"):
        if int(getattr(cfg, name)) <= 0:
            problems.append("dsa_config['{}'] must be positive".format(name))
    if not layers:
        problems.append("dsa_config['dsa_layers'] is empty — no layer would be sparse")
    out_of_range = sorted(x for x in layers if not 1 <= x <= num_hidden_layers)
    if out_of_range:
        problems.append(
            "dsa_config['dsa_layers'] entries {} fall outside 1..{} — the list "
            "is 1-INDEXED like linear_attn_config's".format(
                out_of_range[:8], num_hidden_layers))
    if len(set(layers)) != len(layers):
        problems.append("dsa_config['dsa_layers'] contains duplicates")
    if problems:
        raise ValueError(
            "Kimi-K3.1 dsa_config is invalid — {} problem(s):\n  - {}".format(
                len(problems), "\n  - ".join(problems)))

    # Separate raises: each names the unimplemented variant rather than
    # quietly running the implemented one over different weights.
    if cfg.index_rope_dim != 0:
        raise NotImplementedError(
            "dsa_config['index_rope_dim']={} — only the NoPE index path is "
            "implemented (K3.1 ships 0: no rotation is applied anywhere on the "
            "index path, so prefill-written and decode-scored index keys live "
            "in the same space without one)".format(cfg.index_rope_dim))
    if cfg.index_k_norm_type != _DSA_K_NORM_LAYERNORM:
        raise NotImplementedError(
            "dsa_config['index_k_norm_type']={!r} — only {!r} is implemented; "
            "K3.1 ships indexer.k_norm.weight AND indexer.k_norm.bias, which "
            "only a LayerNorm has".format(
                cfg.index_k_norm_type, _DSA_K_NORM_LAYERNORM))
    reuse = [idx for idx in range(num_hidden_layers)
             if cfg.is_dsa_layer(idx) and cfg.skips_topk(idx)]
    if reuse:
        raise NotImplementedError(
            "dsa_config schedule marks DSA layers {} as top-k REUSE layers "
            "(index_topk_freq={}, index_skip_topk_offset={}), but every K3.1 "
            "full-attention layer ships its own indexer weights and the reuse "
            "path is not implemented here".format(
                reuse[:8], cfg.index_topk_freq, cfg.index_skip_topk_offset))
    return cfg


@dataclass
class KimiK3Config:
    """Flat, validated Kimi-K3 text-model configuration.

    Built exclusively through :func:`parse_k3_config` /
    :func:`parse_k3_config_json`; direct construction skips key allowlisting
    (but not :meth:`validate`, which the model calls again defensively).
    """

    # core
    vocab_size: int = 0
    hidden_size: int = 0
    intermediate_size: int = 0          # dense (layer-0) FFN width
    num_hidden_layers: int = 0
    rms_norm_eps: float = 1e-5

    # MLA
    num_attention_heads: int = 0
    num_key_value_heads: int = 0
    q_lora_rank: Optional[int] = None
    kv_lora_rank: Optional[int] = None
    qk_nope_head_dim: Optional[int] = None
    qk_rope_head_dim: Optional[int] = None
    v_head_dim: Optional[int] = None
    mla_use_nope: bool = False
    mla_use_output_gate: bool = False

    # KDA
    linear_attn_config: Optional[Dict[str, Any]] = None

    # DSA (K3.1 only; None IS the K3 configuration and the dense-MLA gate)
    dsa_config: Optional[KimiDsaConfig] = None

    # MoE
    num_experts: Optional[int] = None
    num_experts_per_token: Optional[int] = None
    moe_intermediate_size: Optional[int] = None
    num_shared_experts: int = 0
    first_k_dense_replace: int = 0
    moe_layer_freq: int = 1
    moe_renormalize: bool = True
    moe_router_activation_func: str = "sigmoid"
    routed_scaling_factor: float = 1.0
    num_expert_group: int = 1
    topk_group: int = 1
    use_grouped_topk: bool = True
    topk_method: str = "noaux_tc"
    routed_expert_hidden_size: Optional[int] = None
    latent_moe_use_norm: bool = False

    # activation
    hidden_act: str = ""
    activation_situ_beta: Optional[float] = None
    activation_situ_linear_beta: Optional[float] = None

    # Block Attention Residuals
    attn_res_block_size: Optional[int] = None

    # misc
    num_nextn_predict_layers: int = 0
    max_position_embeddings: int = 0
    initializer_range: float = 0.02
    tie_word_embeddings: bool = False
    pad_token_id: Optional[int] = None
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    quantization_config: Optional[Dict[str, Any]] = None
    media_placeholder_token_id: int = K3_DEFAULT_MEDIA_PLACEHOLDER_TOKEN_ID
    a_log_padded_len: int = K3_A_LOG_PADDED_LEN
    model_type: str = "kimi_k3"
    raw_text_config: Dict[str, Any] = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------ #
    #  Hybrid-layer helpers (1-INDEXED lists, configuration_kimi_k3.py:152) #
    # ------------------------------------------------------------------ #
    def is_kda_layer(self, layer_idx: int) -> bool:
        return (layer_idx + 1) in self.linear_attn_config["kda_layers"]

    def is_mla_layer(self, layer_idx: int) -> bool:
        return not self.is_kda_layer(layer_idx)

    def is_dsa_layer(self, layer_idx: int) -> bool:
        """Whether this MLA layer runs the sparse (indexer-gated) path.

        False for every layer of K3 (``dsa_config is None``), which is what
        keeps the K3 build structurally identical to before this feature.
        """
        return self.dsa_config is not None and self.dsa_config.is_dsa_layer(layer_idx)

    @property
    def kda_num_heads(self) -> int:
        return int(self.linear_attn_config["num_heads"])

    @property
    def kda_head_dim(self) -> int:
        return int(self.linear_attn_config["head_dim"])

    @property
    def kda_conv_size(self) -> int:
        return int(self.linear_attn_config["short_conv_kernel_size"])

    @property
    def kda_use_full_rank_gate(self) -> bool:
        return bool(self.linear_attn_config["use_full_rank_gate"])

    @property
    def kda_gate_lower_bound(self) -> float:
        return float(self.linear_attn_config["gate_lower_bound"])

    @property
    def q_head_dim(self) -> int:
        return int(self.qk_nope_head_dim) + int(self.qk_rope_head_dim)

    # ------------------------------------------------------------------ #
    #  Invariants                                                          #
    # ------------------------------------------------------------------ #
    def validate(self) -> None:
        """Every K3 invariant this build depends on.  All problems reported at
        once — fixing them one raise at a time is how a config lands half-right
        (pattern from kimi_linear/k3/tensor_map.py:validate_k3_config)."""
        problems: List[str] = []

        for name in ("vocab_size", "hidden_size", "intermediate_size",
                     "num_hidden_layers", "num_attention_heads",
                     "num_key_value_heads", "max_position_embeddings"):
            if int(getattr(self, name) or 0) <= 0:
                problems.append("{} must be a positive integer".format(name))

        if self.hidden_act != "situ":
            problems.append(
                "hidden_act is {!r}, expected 'situ' (K3 uses SiTU on the dense "
                "layer, the shared experts, and every routed expert)".format(self.hidden_act))
        if self.activation_situ_beta is None or self.activation_situ_linear_beta is None:
            problems.append(
                "activation_situ_beta / activation_situ_linear_beta must both be "
                "set (K3: 4.0 / 25.0). model.py refuses the `beta or 1.0` "
                "fallback idiom of the reference code")

        # --- MLA ---
        if self.q_lora_rank is None:
            problems.append(
                "q_lora_rank is None — K3 MLA factors Q through "
                "q_a_proj/q_a_layernorm/q_b_proj; a direct q_proj has no "
                "checkpoint tensor")
        for name in ("kv_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim"):
            if getattr(self, name) is None:
                problems.append("{} must be set for K3 MLA".format(name))
        if not self.mla_use_nope:
            problems.append(
                "mla_use_nope is False — the rotary MLA path is deliberately "
                "unimplemented (K3 is NoPE; no rope module exists in the "
                "reference either, modeling_kimi_linear.py:403)")
        if not self.mla_use_output_gate:
            problems.append(
                "mla_use_output_gate is False, but the checkpoint ships "
                "self_attn.g_proj on every MLA layer")
        if self.num_key_value_heads != self.num_attention_heads:
            problems.append(
                "num_key_value_heads ({}) != num_attention_heads ({}). This "
                "port assumes the K3 property that repeat_kv is a no-op "
                "(modeling_kimi_linear.py:304-308 with 96/96 heads); grouped KV "
                "would need an explicit repeat_kv reintroduced".format(
                    self.num_key_value_heads, self.num_attention_heads))

        # --- KDA ---
        lac = self.linear_attn_config
        if not lac:
            problems.append("linear_attn_config is missing — no layer would be KDA")
        else:
            unknown = [k for k in lac if k not in _LINEAR_ATTN_KEYS]
            _reject_unknown_keys(unknown, "linear_attn_config")
            for key in ("kda_layers", "full_attn_layers", "num_heads", "head_dim",
                        "short_conv_kernel_size", "use_full_rank_gate", "gate_lower_bound"):
                if lac.get(key) is None:
                    problems.append("linear_attn_config['{}'] must be set".format(key))
            kda = lac.get("kda_layers") or []
            mla = lac.get("full_attn_layers") or []
            # `if kda and mla` would skip the partition check whenever one
            # list is empty — an all-KDA list WITH HOLES would then validate.
            # An empty full_attn_layers is legitimate (a pure-KDA stack), so
            # require only that at least one list is populated.
            if kda or mla:
                expected = set(range(1, self.num_hidden_layers + 1))
                union = set(map(int, kda)) | set(map(int, mla))
                overlap = set(map(int, kda)) & set(map(int, mla))
                if overlap or union != expected:
                    problems.append(
                        "linear_attn_config layer lists do not partition 1..{} "
                        "(1-INDEXED). overlap={}, missing={}, extra={}".format(
                            self.num_hidden_layers, sorted(overlap)[:8],
                            sorted(expected - union)[:8], sorted(union - expected)[:8]))
                elif 1 not in set(map(int, kda)):
                    problems.append(
                        "layer 0 (1-based layer 1) must be KDA — the checkpoint "
                        "ships layers.0.self_attn.A_log/q_conv1d and a dense "
                        "layers.0.mlp")
            if not lac.get("use_full_rank_gate", False):
                problems.append(
                    "linear_attn_config['use_full_rank_gate'] is not True, but "
                    "the checkpoint ships one full-rank self_attn.g_proj per KDA "
                    "layer and no g_a_proj/g_b_proj anywhere")
            if lac.get("num_heads") and int(lac["num_heads"]) > self.a_log_padded_len:
                problems.append(
                    "kda num_heads ({}) exceeds the A_log padded buffer length "
                    "({})".format(lac["num_heads"], self.a_log_padded_len))

        # --- DSA (K3.1) ---
        dsa = self.dsa_config
        if dsa is not None:
            if self.model_type != "kimi_k31":
                problems.append(
                    "dsa_config is set but model_type is {!r}; the indexer is "
                    "the K3.1 delta and must not be attached to another "
                    "identity".format(self.model_type))
            full_attn = set(map(int, (lac or {}).get("full_attn_layers") or []))
            if full_attn and set(dsa.dsa_layers) != full_attn:
                problems.append(
                    "dsa_config['dsa_layers'] != linear_attn_config"
                    "['full_attn_layers']: only={}, missing={}. The indexer "
                    "lives on the full-attention layers (the checkpoint ships "
                    "self_attn.indexer.* on exactly those 24 layers); a KDA "
                    "layer has no MLA attention to gate and no q_a to read."
                    .format(sorted(set(dsa.dsa_layers) - full_attn)[:8],
                            sorted(full_attn - set(dsa.dsa_layers))[:8]))
            if self.q_lora_rank is None:
                problems.append(
                    "dsa_config is set but q_lora_rank is None — the indexer's "
                    "wq_b reads the MLA q_a intermediate (shape "
                    "[index_num_heads*index_head_dim, q_lora_rank])")
        elif self.model_type == "kimi_k31":
            problems.append(
                "model_type is 'kimi_k31' but dsa_config is absent. The sparse "
                "indexer is the ONLY thing that distinguishes K3.1 from K3, so "
                "there is no default: without it the 24 full-attention layers "
                "would silently run dense over the whole context while the "
                "checkpoint's 120 indexer tensors had no destination.")

        # --- MoE ---
        for name in ("num_experts", "num_experts_per_token", "moe_intermediate_size",
                     "routed_expert_hidden_size"):
            if getattr(self, name) is None:
                problems.append("{} must be set for K3 LatentMoE".format(name))
        if not self.latent_moe_use_norm:
            problems.append(
                "latent_moe_use_norm is False, but the checkpoint ships "
                "block_sparse_moe.routed_expert_norm on every MoE layer")
        if int(self.num_shared_experts or 0) < 1:
            problems.append("num_shared_experts must be >= 1 (K3: 2)")
        if self.moe_router_activation_func != "sigmoid":
            problems.append(
                "moe_router_activation_func is {!r}; only 'sigmoid' (noaux_tc) "
                "is implemented for K3".format(self.moe_router_activation_func))
        if self.topk_method != "noaux_tc":
            problems.append(
                "topk_method is {!r}; only 'noaux_tc' is implemented".format(self.topk_method))
        if int(self.num_expert_group or 1) != 1 or int(self.topk_group or 1) != 1:
            problems.append(
                "num_expert_group/topk_group must be 1: K3's group-limited "
                "routing branch is provably dead (num_expert_group=1 fails the "
                "`>1` test, modeling_kimi_linear.py:724-746) and is deliberately "
                "NOT implemented")
        if int(self.moe_layer_freq or 1) != 1:
            problems.append("moe_layer_freq must be 1 (every layer >= first_k_dense_replace is MoE)")
        if int(self.first_k_dense_replace or 0) < 1:
            problems.append(
                "first_k_dense_replace < 1, but the checkpoint ships a dense "
                "layers.0.mlp.{gate,up,down}_proj")
        if int(self.first_k_dense_replace or 0) >= self.num_hidden_layers:
            problems.append("first_k_dense_replace >= num_hidden_layers: no MoE layer at all")

        # --- AttnRes ---
        if self.attn_res_block_size is None or int(self.attn_res_block_size) < 1:
            problems.append(
                "attn_res_block_size must be a positive integer — Block "
                "Attention Residuals are structural in K3 (the classic residual "
                "body is dead code and not implemented)")

        if problems:
            raise ValueError(
                "Kimi-K3 config is invalid — {} problem(s):\n  - {}".format(
                    len(problems), "\n  - ".join(problems)))

        # Separate raise: this one names the milestone that will implement it.
        if int(self.num_nextn_predict_layers or 0) != 0:
            raise NotImplementedError(
                "num_nextn_predict_layers={} — the K3 MTP head is not "
                "implemented (unscheduled; the released checkpoint ships 0)"
                .format(self.num_nextn_predict_layers))


# --------------------------------------------------------------------------- #
#  Parser                                                                      #
# --------------------------------------------------------------------------- #

#: Wrapper ``model_type`` values this parser accepts, and whether that
#: identity carries the DSA indexer.  An unlisted value raises: binding an
#: unknown Kimi wrapper to either of these graphs is exactly the silent
#: mis-load this parser exists to prevent.
_WRAPPER_MODEL_TYPES = {
    "kimi_k3": False,
    "kimi_k31": True,
}


def parse_k3_config(raw: Dict[str, Any]) -> KimiK3Config:
    """Parse the checkpoint's full ``config.json`` dict into a validated
    :class:`KimiK3Config`.

    Requires the K3 nesting: the text model lives under ``raw['text_config']``.
    Unknown keys — top-level, text-level, or inside ``linear_attn_config`` /
    ``dsa_config`` — raise (they are listed in the error).

    The TOP-LEVEL ``model_type`` selects the identity: ``kimi_k3`` (dense) or
    ``kimi_k31`` (DSA).  Both checkpoints declare ``text_config.model_type``
    as the literal ``"kimi_linear"``, so the text-level value cannot be used
    to tell them apart.
    """
    if not isinstance(raw, dict):
        raise ValueError("parse_k3_config expects the config.json dict, got {}".format(type(raw)))
    if "text_config" not in raw or not isinstance(raw["text_config"], dict):
        raise ValueError(
            "Kimi-K3 config.json must nest the text model under 'text_config' "
            "(the flat form is Kimi-Linear-48B, a different model). Top-level "
            "keys seen: {}".format(sorted(raw.keys())))

    _reject_unknown_keys([k for k in raw if k not in _TOP_LEVEL_KEYS],
                         "config.json (top level)")

    wrapper_type = str(_require(raw, "model_type", "top-level config"))
    if wrapper_type not in _WRAPPER_MODEL_TYPES:
        raise ValueError(
            "Kimi-K3 config: top-level model_type is {!r}; this parser serves "
            "{}. The wrapper identity selects the attention graph (dense vs "
            "DSA-gated), so an unrecognised one is never defaulted."
            .format(wrapper_type, sorted(_WRAPPER_MODEL_TYPES)))

    tc = raw["text_config"]
    _reject_unknown_keys(
        [k for k in tc if k not in _ARCH_KEYS and k not in _HF_BOOKKEEPING_KEYS],
        "config.json['text_config']")

    where = "text_config"
    num_layers = int(_require(tc, "num_hidden_layers", where))
    dsa_raw = tc.get("dsa_config")
    expects_dsa = _WRAPPER_MODEL_TYPES[wrapper_type]
    if dsa_raw is None and expects_dsa:
        raise ValueError(
            "Kimi-K3 config: model_type={!r} requires text_config.dsa_config "
            "(the sparse indexer is its only delta vs Kimi-K3) and it is "
            "absent.".format(wrapper_type))
    if dsa_raw is not None and not expects_dsa:
        raise ValueError(
            "Kimi-K3 config: model_type={!r} must not carry a dsa_config — the "
            "released Kimi-K3 checkpoint ships no indexer weights, so a model "
            "built with one would attend over an all-zero index space."
            .format(wrapper_type))
    dsa_cfg = None if dsa_raw is None else parse_dsa_config(dsa_raw, num_layers)
    cfg = KimiK3Config(
        vocab_size=int(_require(tc, "vocab_size", where)),
        hidden_size=int(_require(tc, "hidden_size", where)),
        intermediate_size=int(_require(tc, "intermediate_size", where)),
        num_hidden_layers=num_layers,
        rms_norm_eps=float(_require(tc, "rms_norm_eps", where)),
        num_attention_heads=int(_require(tc, "num_attention_heads", where)),
        num_key_value_heads=int(_require(tc, "num_key_value_heads", where)),
        q_lora_rank=int(_require(tc, "q_lora_rank", where)),
        kv_lora_rank=int(_require(tc, "kv_lora_rank", where)),
        qk_nope_head_dim=int(_require(tc, "qk_nope_head_dim", where)),
        qk_rope_head_dim=int(_require(tc, "qk_rope_head_dim", where)),
        v_head_dim=int(_require(tc, "v_head_dim", where)),
        mla_use_nope=bool(_require(tc, "mla_use_nope", where)),
        mla_use_output_gate=bool(_require(tc, "mla_use_output_gate", where)),
        linear_attn_config=dict(_require(tc, "linear_attn_config", where)),
        dsa_config=dsa_cfg,
        num_experts=int(_require(tc, "num_experts", where)),
        num_experts_per_token=int(_require(tc, "num_experts_per_token", where)),
        moe_intermediate_size=int(_require(tc, "moe_intermediate_size", where)),
        num_shared_experts=int(_require(tc, "num_shared_experts", where)),
        first_k_dense_replace=int(_require(tc, "first_k_dense_replace", where)),
        moe_layer_freq=int(tc.get("moe_layer_freq", 1)),
        moe_renormalize=bool(_require(tc, "moe_renormalize", where)),
        moe_router_activation_func=str(_require(tc, "moe_router_activation_func", where)),
        routed_scaling_factor=float(_require(tc, "routed_scaling_factor", where)),
        num_expert_group=int(_require(tc, "num_expert_group", where)),
        topk_group=int(_require(tc, "topk_group", where)),
        use_grouped_topk=bool(tc.get("use_grouped_topk", True)),
        topk_method=str(_require(tc, "topk_method", where)),
        routed_expert_hidden_size=int(_require(tc, "routed_expert_hidden_size", where)),
        latent_moe_use_norm=bool(_require(tc, "latent_moe_use_norm", where)),
        hidden_act=str(_require(tc, "hidden_act", where)),
        activation_situ_beta=float(_require(tc, "activation_situ_beta", where)),
        activation_situ_linear_beta=float(_require(tc, "activation_situ_linear_beta", where)),
        attn_res_block_size=int(_require(tc, "attn_res_block_size", where)),
        num_nextn_predict_layers=int(tc.get("num_nextn_predict_layers", 0) or 0),
        max_position_embeddings=int(_require(tc, "max_position_embeddings", where)),
        initializer_range=float(tc.get("initializer_range", 0.02)),
        tie_word_embeddings=bool(tc.get("tie_word_embeddings", False)),
        pad_token_id=tc.get("pad_token_id"),
        bos_token_id=tc.get("bos_token_id"),
        eos_token_id=tc.get("eos_token_id"),
        quantization_config=tc.get("quantization_config"),
        # No default: the vision hard-fail guards this id, and guarding a
        # GUESSED id is a silent fallback wearing a hard-fail's clothes. The
        # released checkpoint ships the key at top level.
        media_placeholder_token_id=int(
            _require(raw, "media_placeholder_token_id", "top-level config")),
        model_type=wrapper_type,
        raw_text_config=dict(tc),
    )
    cfg.validate()
    return cfg


def parse_k3_config_json(path: str) -> KimiK3Config:
    """Parse ``config.json`` from disk (e.g. the vendored ``assets/config.json``)."""
    with open(path, "r") as f:
        return parse_k3_config(json.load(f))
