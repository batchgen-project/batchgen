# ---------------------------------------------------------------------------- #
#  BatchGen                                                                     #
#  copyright (c) EfficientMoE team 2025                                         #
#  Licensed under the Apache License, Version 2.0                               #
# ---------------------------------------------------------------------------- #
"""Kimi-Linear / Kimi-K3 family configuration for BatchGen.

Covers both:
  * Kimi-Linear-48B-A3B (testbed): flat config.json, hidden-space MoE (SiLU),
    low-rank KDA gate, NoPE-MLA (no output gate), BF16.
  * Kimi-K3 (2.8T): config.json with everything under `text_config`; full-rank KDA
    gate, gated NoPE-MLA, LatentMoE (routed_expert_hidden_size) + SiTU + AttnRes,
    MXFP4 weights, 2 shared experts, 896 experts top-16.

Architecture (shared): a hybrid decoder stack where most layers are KDA
(Kimi Delta Attention — linear/recurrent, fixed per-seq state, no KV cache) and
1-in-4 layers are MLA (paged KV, NoPE). `linear_attn_config.kda_layers` lists the
KDA layers using **1-indexed** layer numbers (layer_idx + 1); every other layer is MLA.
"""

from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional
import json
from pathlib import Path

from batchgen.config.model_config import BaseModelConfig
from batchgen.config.model_registry import register_config


# HF config.json key -> BatchGen canonical dataclass field name.
_HF_ALIASES = {
    "num_experts": "n_routed_experts",
    "num_experts_per_token": "num_experts_per_tok",
    "num_shared_experts": "n_shared_experts",
    "num_expert_group": "n_group",
    "moe_router_activation_func": "scoring_func",
    "moe_renormalize": "norm_topk_prob",
}


def require_num_routed_experts(cfg) -> int:
    """Routed-expert count for a Kimi-Linear / Kimi-K3 config, or raise.

    K3's ``config.json`` declares the count as ``num_experts`` (896) and ships
    **no** ``n_routed_experts`` key; :data:`_HF_ALIASES` bridges the two, but
    only for a config that actually went through :meth:`from_hf_dict`.

    This replaces ``getattr(cfg, "n_routed_experts", 256) or 256``, which
    returns **256 for an 896-expert model** whenever the attribute is absent —
    and on :class:`batchgen.config.config.ModelConfig` it is *always* absent,
    since that class has no such field. Nothing raised: the EP shard layout,
    the routing pool and the expert copy task were all sized for the wrong
    model, and only the 48B's coincidental 256 hid it.

    Accepts either config class (``KimiLinearConfig`` has both fields,
    ``ModelConfig`` only ``num_local_experts``). There is no default.
    """
    for attr in ("n_routed_experts", "num_local_experts"):
        value = getattr(cfg, attr, None)
        if value:
            return int(value)
    raise RuntimeError(
        "Cannot determine the routed-expert count from a "
        f"{type(cfg).__name__}: neither 'n_routed_experts' nor "
        "'num_local_experts' is set. K3 declares it as 'num_experts' in "
        "config.json (aliased by KimiLinearConfig.from_hf_dict) — pass "
        "--cache-dir so that file is read. There is no default and there "
        "must not be one."
    )


@register_config("kimi_linear")
@register_config("kimi_k3")
@dataclass
class KimiLinearConfig(BaseModelConfig):
    """Kimi-Linear / Kimi-K3 hybrid (KDA + NoPE-MLA) MoE configuration."""

    # ==================== Identity ====================
    model_type: str = "kimi_linear"
    architectures: List[str] = field(default_factory=lambda: ["KimiLinearForCausalLM"])

    # ==================== Core ====================
    vocab_size: int = 163840
    hidden_size: int = 2304
    intermediate_size: int = 9216          # dense-layer FFN
    num_hidden_layers: int = 27

    # ==================== Attention (MLA layers are NoPE-MLA) ====================
    attention_type: str = "mla"
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    kv_lora_rank: Optional[int] = 512
    q_lora_rank: Optional[int] = None       # testbed: None (direct q_proj); K3: 1536
    qk_nope_head_dim: Optional[int] = 128
    qk_rope_head_dim: Optional[int] = 64
    v_head_dim: Optional[int] = 128
    compressed_kv_dim: Optional[int] = 576  # kv_lora_rank + qk_rope_head_dim
    mla_use_nope: bool = True               # NoPE: rope subdim carried but zeroed
    mla_use_output_gate: bool = False       # K3: True (sigmoid output gate on MLA)

    # ==================== KDA (linear attention) ====================
    # dict: {kda_layers:[1-indexed], full_attn_layers:[...], num_heads, head_dim,
    #        short_conv_kernel_size, use_full_rank_gate?, gate_lower_bound?}
    linear_attn_config: Optional[Dict[str, Any]] = None

    # ==================== MoE ====================
    num_local_experts: int = 256
    n_routed_experts: Optional[int] = 256
    n_shared_experts: int = 1
    num_experts_per_tok: int = 8
    first_k_dense_replace: int = 1
    moe_intermediate_size: Optional[int] = 1024
    moe_layer_freq: int = 1
    topk_method: str = "noaux_tc"
    n_group: Optional[int] = 1
    topk_group: Optional[int] = 1
    routed_scaling_factor: float = 2.446
    norm_topk_prob: bool = True
    scoring_func: str = "sigmoid"
    # LatentMoE (K3): experts operate in a compressed latent space.
    routed_expert_hidden_size: Optional[int] = None   # K3: 3584; testbed: None
    latent_moe_use_norm: bool = False

    # ==================== Activation ====================
    hidden_act: str = "silu"                 # K3: "situ"
    activation_situ_beta: Optional[float] = None
    activation_situ_linear_beta: Optional[float] = None

    # ==================== Block Attention Residuals (K3 only) ====================
    attn_res_block_size: Optional[int] = None   # K3: 12; testbed: None

    # ==================== Position ====================
    max_position_embeddings: int = 1048576
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict[str, Any]] = None

    # ==================== Norm ====================
    rms_norm_eps: float = 1e-5

    # ==================== Quantization ====================
    quantization: str = "none"               # K3: "mxfp4"
    quantization_config: Optional[Dict[str, Any]] = None

    # ==================== Tokenizer ====================
    bos_token_id: Optional[int] = 163584
    eos_token_id: Optional[int] = 163586
    pad_token_id: Optional[int] = 163839
    tie_word_embeddings: bool = False

    # ==================== Misc ====================
    num_nextn_predict_layers: int = 0
    ep_size: int = 1

    # ------------------------------------------------------------------ #
    #  Loading: handle K3's nested `text_config` + HF->canonical aliases
    # ------------------------------------------------------------------ #
    @classmethod
    def from_json(cls, path: str) -> "KimiLinearConfig":
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_hf_dict(data)

    @classmethod
    def from_hf_dict(cls, data: Dict[str, Any]) -> "KimiLinearConfig":
        top_model_type = data.get("model_type", "kimi_linear")
        # K3: the language-model fields live under text_config; flatten them up.
        merged: Dict[str, Any] = {}
        if "text_config" in data and isinstance(data["text_config"], dict):
            merged.update(data["text_config"])
            merged["model_type"] = "kimi_k3"
        else:
            merged.update(data)
            merged.setdefault("model_type", top_model_type)
        # apply HF -> canonical aliases (without clobbering an explicit canonical key)
        for hf_key, canon in _HF_ALIASES.items():
            if hf_key in merged and canon not in merged:
                merged[canon] = merged[hf_key]
        # num_experts also feeds num_local_experts
        if "num_experts" in merged:
            merged.setdefault("num_local_experts", merged["num_experts"])
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in merged.items() if k in known}
        return cls(**filtered)

    def __post_init__(self):
        # derived attention dims
        if self.qk_nope_head_dim is not None and self.qk_rope_head_dim is not None:
            self.head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        if self.kv_lora_rank is not None and self.qk_rope_head_dim is not None:
            self.compressed_kv_dim = self.kv_lora_rank + self.qk_rope_head_dim
        # SiTU implies act name
        if self.activation_situ_beta is not None and self.hidden_act != "situ":
            self.hidden_act = "situ"
        # quantization label from compressed-tensors mxfp4 config
        qc = self.quantization_config or {}
        fmt = (qc.get("format") or qc.get("quant_method") or "")
        if "mxfp4" in str(fmt).lower() or self.model_type == "kimi_k3":
            if self.quantization == "none" and self.quantization_config:
                self.quantization = "mxfp4"

    # ------------------------------------------------------------------ #
    #  Hybrid-layer helpers
    # ------------------------------------------------------------------ #
    def is_kda_layer(self, layer_idx: int) -> bool:
        """KDA layers are listed 1-indexed in linear_attn_config['kda_layers']."""
        lac = self.linear_attn_config
        return bool(lac) and (layer_idx + 1) in lac.get("kda_layers", [])

    def is_mla_layer(self, layer_idx: int) -> bool:
        return not self.is_kda_layer(layer_idx)

    @property
    def kda_num_heads(self) -> int:
        return int(self.linear_attn_config["num_heads"])

    @property
    def kda_head_dim(self) -> int:
        return int(self.linear_attn_config["head_dim"])

    @property
    def kda_conv_size(self) -> int:
        return int(self.linear_attn_config.get("short_conv_kernel_size", 4))

    @property
    def kda_use_full_rank_gate(self) -> bool:
        return bool(self.linear_attn_config.get("use_full_rank_gate", False))

    @property
    def kda_gate_lower_bound(self) -> Optional[float]:
        return self.linear_attn_config.get("gate_lower_bound", None)

    @property
    def use_latent_moe(self) -> bool:
        return self.routed_expert_hidden_size is not None

    @property
    def use_attn_residuals(self) -> bool:
        return self.attn_res_block_size is not None

    @property
    def num_kda_layers(self) -> int:
        return sum(self.is_kda_layer(i) for i in range(self.num_hidden_layers))

    @property
    def num_mla_layers(self) -> int:
        return self.num_hidden_layers - self.num_kda_layers

    @property
    def num_kv_heads(self) -> int:
        return self.num_key_value_heads
