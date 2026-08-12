# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                             #
# ---------------------------------------------------------------------------- #

"""GLM-5 model configuration for BatchGen.

GLM-5 is a 744B parameter MoE model from Zhipu AI with 40B active parameters.
Architecture: MLA + DSA (DeepSeek Sparse Attention) + 256 routed experts.

Key differences from DeepSeek-V3:
- 78 layers (vs 61), hidden_size=6144 (vs 7168)
- MLA: qk_nope_head_dim=192 (vs 128), v_head_dim=256 (vs 128), q_lora_rank=2048 (vs 1536)
- rope_interleave=True, rope_theta=1M (no YaRN scaling)
- n_group=1 (simple top-8 routing, no group-based selection)
- DSA indexer: 32 heads (vs 64 for V3.2)
- FP8: E4M3, dynamic activation, [128,128] block (same as V3)
"""

import logging
from dataclasses import dataclass, field, fields
from typing import Dict, Any, List, Optional

from batchgen.config.model_config import BaseModelConfig
from batchgen.config.model_registry import register_config

logger = logging.getLogger(__name__)


# Identity keys owned by the config class — deliberately NOT taken from HF so
# GLM-5.2 keeps its distinct model_type. Marked "consumed" so they are not
# reported as silently-dropped keys.
_HANDLED_IDENTITY_KEYS = {"model_type", "architectures"}

# HF source keys that MUST be present (and non-null) in a real checkpoint
# config.json. An absent/null key here means a truncated or wrong config; we
# fail loud rather than silently backfilling the GLM-5 dataclass defaults (which
# describe the 744B GLM-5 base model and would mis-size the engine).
# num_local_experts is satisfied by either "n_routed_experts" or
# "num_local_experts"; compressed_kv_dim is derived from kv_lora_rank +
# qk_rope_head_dim (both listed here).
_REQUIRED_HF_KEYS = (
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "qk_head_dim",
    "kv_lora_rank",
    "qk_rope_head_dim",
    "first_k_dense_replace",
)


@register_config("glm_moe_dsa")
@dataclass
class GLM5Config(BaseModelConfig):
    """GLM-5 (GlmMoeDsaForCausalLM) configuration."""

    # ==================== Identity ====================
    model_type: str = "glm_moe_dsa"
    architectures: List[str] = field(default_factory=lambda: ["GlmMoeDsaForCausalLM"])

    # ==================== Core Architecture ====================
    vocab_size: int = 154880
    hidden_size: int = 6144
    intermediate_size: int = 12288
    num_hidden_layers: int = 78

    # ==================== Attention: MLA ====================
    attention_type: str = "mla"
    num_attention_heads: int = 64
    num_key_value_heads: int = 64
    head_dim: int = 64
    qk_head_dim: int = 256
    attention_bias: bool = False
    attention_dropout: float = 0.0

    # MLA dimensions
    kv_lora_rank: int = 512
    q_lora_rank: int = 2048
    qk_nope_head_dim: int = 192
    qk_rope_head_dim: int = 64
    v_head_dim: int = 256
    compressed_kv_dim: int = 576  # kv_lora_rank + qk_rope_head_dim

    # ==================== DSA (DeepSeek Sparse Attention) ====================
    index_n_heads: int = 32
    index_head_dim: int = 128
    index_topk: int = 2048
    rope_interleave: bool = True
    indexer_rope_interleave: bool = True
    # Structurally disable the DSA indexer (no Glm5Indexer module constructed,
    # no aux KV cache, no WP2/WP4/WP5 kernels, pure einsum absorbed MLA in
    # decode). Set per-config to bypass DSA and run pure absorbed MLA.
    use_dense_mla: bool = False

    # ==================== Context: Full attention ====================
    sliding_window_size: Optional[int] = None

    # ==================== MoE ====================
    num_local_experts: int = 256
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    num_experts_per_tok: int = 8
    first_k_dense_replace: int = 3
    moe_intermediate_size: int = 2048
    moe_layer_freq: int = 1

    # Router
    topk_method: str = "noaux_tc"
    n_group: int = 1
    topk_group: int = 1
    routed_scaling_factor: float = 2.5
    norm_topk_prob: bool = True
    scoring_func: str = "sigmoid"

    # ==================== Position Encoding ====================
    max_position_embeddings: int = 202752
    rope_theta: float = 1000000.0
    rope_scaling: Optional[Dict[str, Any]] = None  # No YaRN

    # ==================== Normalization & Activation ====================
    rms_norm_eps: float = 1e-5
    hidden_act: str = "silu"

    # ==================== Quantization ====================
    quantization: str = "fp8"
    quantization_config: Dict[str, Any] = field(default_factory=lambda: {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "weight_block_size": [128, 128],
    })

    # ==================== Tokenizer ====================
    bos_token_id: Optional[int] = None
    eos_token_id: List[int] = field(default_factory=lambda: [154820, 154827, 154829])
    pad_token_id: int = 154820
    tie_word_embeddings: bool = False

    # ==================== Other ====================
    num_nextn_predict_layers: int = 1
    ep_size: int = 1

    @property
    def num_kv_heads(self) -> int:
        return self.num_key_value_heads

    # ------------------------------------------------------------------ #
    #  Checkpoint resolution (HF config.json -> rich config)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _rope_theta_from_hf(hf: Dict[str, Any]) -> Optional[float]:
        """Read rope_theta, handling GLM-5's flat key and GLM-5.2's nested
        ``rope_parameters`` block."""
        rope_params = hf.get("rope_parameters")
        if isinstance(rope_params, dict) and rope_params.get("rope_theta") is not None:
            return rope_params["rope_theta"]
        return hf.get("rope_theta")

    @staticmethod
    def _common_hf_kwargs(hf: Dict[str, Any]) -> Dict[str, Any]:
        """Explicit HF-key -> field mapping shared by GLM-5 and GLM-5.2.

        Only keys actually present in ``hf`` are emitted; everything else falls
        back to the dataclass default. Identity fields (model_type,
        architectures) are deliberately NOT taken from HF — they are owned by
        the config class so GLM-5.2 keeps its distinct model_type.

        Every HF key this mapping *references* (whether present or not) is added
        to ``hf["__consumed_keys__"]`` (a scratch set the callers pop back out),
        so ``from_hf`` can report checkpoint keys it saw but never mapped.
        """
        kwargs: Dict[str, Any] = {}
        consumed = hf.setdefault("__consumed_keys__", set())
        consumed.update(_HANDLED_IDENTITY_KEYS)
        # mlp_layer_types is not a scalar field but IS load-bearing — it is
        # cross-checked against first_k_dense_replace in _assert_mlp_layer_types,
        # so treat it as consumed rather than a silently-dropped key.
        consumed.add("mlp_layer_types")

        def put(field_name: str, hf_key: str) -> None:
            consumed.add(hf_key)
            if hf_key in hf and hf[hf_key] is not None:
                kwargs[field_name] = hf[hf_key]

        # Core architecture
        put("vocab_size", "vocab_size")
        put("hidden_size", "hidden_size")
        put("intermediate_size", "intermediate_size")
        put("num_hidden_layers", "num_hidden_layers")

        # Attention (MLA geometry). NOTE: head_dim and qk_head_dim are distinct
        # in HF (192 vs 256); we carry both faithfully. The engine projection
        # deliberately uses qk_head_dim, not head_dim.
        put("num_attention_heads", "num_attention_heads")
        put("num_key_value_heads", "num_key_value_heads")
        put("head_dim", "head_dim")
        put("qk_head_dim", "qk_head_dim")
        put("qk_nope_head_dim", "qk_nope_head_dim")
        put("qk_rope_head_dim", "qk_rope_head_dim")
        put("v_head_dim", "v_head_dim")
        put("kv_lora_rank", "kv_lora_rank")
        put("q_lora_rank", "q_lora_rank")
        put("attention_bias", "attention_bias")
        put("attention_dropout", "attention_dropout")

        # compressed_kv_dim is not stored in HF config.json — derive it from the
        # MLA dims (kv_lora_rank + qk_rope_head_dim) when both are available.
        kv_lora_rank = hf.get("kv_lora_rank")
        qk_rope_head_dim = hf.get("qk_rope_head_dim")
        if kv_lora_rank is not None and qk_rope_head_dim is not None:
            kwargs["compressed_kv_dim"] = kv_lora_rank + qk_rope_head_dim

        # DSA indexer
        put("index_n_heads", "index_n_heads")
        put("index_head_dim", "index_head_dim")
        put("index_topk", "index_topk")
        put("rope_interleave", "rope_interleave")
        put("indexer_rope_interleave", "indexer_rope_interleave")

        # MoE. HF stores n_routed_experts; the engine's num_local_experts mirrors
        # it (there is no separate num_local_experts key in GLM config.json).
        put("n_routed_experts", "n_routed_experts")
        consumed.add("num_local_experts")
        if hf.get("num_local_experts") is not None:
            kwargs["num_local_experts"] = hf["num_local_experts"]
        elif hf.get("n_routed_experts") is not None:
            kwargs["num_local_experts"] = hf["n_routed_experts"]
        put("n_shared_experts", "n_shared_experts")
        put("num_experts_per_tok", "num_experts_per_tok")
        put("first_k_dense_replace", "first_k_dense_replace")
        put("moe_intermediate_size", "moe_intermediate_size")
        put("moe_layer_freq", "moe_layer_freq")

        # Router / gating
        put("topk_method", "topk_method")
        put("n_group", "n_group")
        put("topk_group", "topk_group")
        put("routed_scaling_factor", "routed_scaling_factor")
        put("norm_topk_prob", "norm_topk_prob")
        put("scoring_func", "scoring_func")

        # Position encoding (rope_theta may be nested under rope_parameters).
        put("max_position_embeddings", "max_position_embeddings")
        consumed.update({"rope_theta", "rope_parameters"})
        rope_theta = GLM5Config._rope_theta_from_hf(hf)
        if rope_theta is not None:
            kwargs["rope_theta"] = rope_theta

        # Normalization & activation
        put("rms_norm_eps", "rms_norm_eps")
        put("hidden_act", "hidden_act")

        # Quantization
        consumed.add("quantization_config")
        if hf.get("quantization_config") is not None:
            kwargs["quantization_config"] = hf["quantization_config"]
            quant_method = hf["quantization_config"].get("quant_method")
            if quant_method:
                kwargs["quantization"] = quant_method

        # Tokenizer / embeddings
        put("bos_token_id", "bos_token_id")
        put("eos_token_id", "eos_token_id")
        put("pad_token_id", "pad_token_id")
        put("tie_word_embeddings", "tie_word_embeddings")

        # Other GLM-specific
        put("num_nextn_predict_layers", "num_nextn_predict_layers")
        put("ep_size", "ep_size")

        return kwargs

    # ------------------------------------------------------------------ #
    #  Missing-required / silently-dropped diagnostics (shared by GLM-5.x)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _assert_required_hf_keys(cls_name: str, hf: Dict[str, Any]) -> None:
        """Fail loud when a checkpoint config.json omits a required HF key.

        Guards against a truncated / wrong config.json being silently backfilled
        with the 744B GLM-5 dataclass defaults. ``num_local_experts`` counts as
        present if either ``n_routed_experts`` or ``num_local_experts`` is set.
        """
        missing = [
            key for key in _REQUIRED_HF_KEYS
            if hf.get(key) is None
        ]
        if hf.get("n_routed_experts") is None and hf.get("num_local_experts") is None:
            missing.append("n_routed_experts|num_local_experts")
        if missing:
            raise ValueError(
                f"{cls_name}.from_hf: checkpoint config.json is missing required "
                f"fields {missing}. Refusing to silently backfill GLM-5 defaults "
                f"for a truncated/incompatible config."
            )

    @staticmethod
    def _warn_dropped_hf_keys(cls_name: str, hf: Dict[str, Any], consumed: set) -> None:
        """WARN loudly for checkpoint keys the mapping never referenced.

        These are silently ignored today; surfacing them catches a checkpoint
        that carries structurally load-bearing state the resolver does not model
        (e.g. a non-contiguous ``mlp_layer_types`` layout).
        """
        # HF bookkeeping keys that are intentionally irrelevant to the engine.
        ignore = {"__consumed_keys__", "torch_dtype", "dtype", "transformers_version",
                  "use_cache", "initializer_range", "pretraining_tp", "moe_router_dtype",
                  "index_topk_pattern"}
        dropped = sorted(set(hf) - consumed - ignore)
        if dropped:
            logger.warning(
                "%s.from_hf: checkpoint config.json keys seen but not consumed by "
                "the mapping (silently ignored): %s. Verify none are structurally "
                "load-bearing for this checkpoint.",
                cls_name, dropped,
            )

    @staticmethod
    def _assert_mlp_layer_types(cls_name: str, hf: Dict[str, Any],
                                first_k_dense_replace: int) -> None:
        """Cross-check a per-layer dense/sparse pattern against the scalar
        ``first_k_dense_replace`` the engine relies on.

        ``is_dense_layer`` treats layers ``< first_k_dense_replace`` as dense and
        the rest as sparse (a contiguous dense prefix). If a checkpoint ships a
        non-contiguous ``mlp_layer_types`` the scalar cannot represent, fail loud
        rather than silently mis-modelling the MoE layout.
        """
        layer_types = hf.get("mlp_layer_types")
        if not isinstance(layer_types, list) or not layer_types:
            return
        expected = [
            "dense" if i < first_k_dense_replace else "sparse"
            for i in range(len(layer_types))
        ]
        if layer_types != expected:
            raise ValueError(
                f"{cls_name}.from_hf: mlp_layer_types is not a contiguous dense "
                f"prefix of length first_k_dense_replace={first_k_dense_replace}; "
                f"the engine's scalar first_k_dense_replace cannot represent this "
                f"layout. Got {layer_types!r}."
            )

    @classmethod
    def from_hf(cls, hf_dict: Dict[str, Any]) -> "GLM5Config":
        """Build a rich GLM-5 config from a HuggingFace ``config.json`` dict.

        Maps HF keys onto the config fields explicitly (no silent
        name-intersection). Fails loud when a required HF key is absent (rather
        than masking it with a family default) and WARNs on checkpoint keys the
        mapping never consumed.
        """
        cls._assert_required_hf_keys(cls.__name__, hf_dict)
        kwargs = cls._common_hf_kwargs(hf_dict)
        consumed = hf_dict.pop("__consumed_keys__", set())
        cls._warn_dropped_hf_keys(cls.__name__, hf_dict, consumed)
        cls._assert_mlp_layer_types(
            cls.__name__, hf_dict,
            kwargs.get("first_k_dense_replace", cls.first_k_dense_replace),
        )
        # Guard against passing keys the (possibly-subclassed) dataclass does
        # not declare.
        known = {f.name for f in fields(cls)}
        unknown = set(kwargs) - known
        if unknown:  # pragma: no cover - defensive
            logger.debug("Dropping HF keys not declared on %s: %s", cls.__name__, unknown)
            kwargs = {k: v for k, v in kwargs.items() if k in known}
        return cls(**kwargs)

    def validate(self) -> None:
        """Fail loud on missing required fields or self-inconsistency.

        This is the last line of defence before the rich config is projected
        into the engine's minimal ModelConfig.
        """
        required = {
            "model_type": self.model_type,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "num_local_experts": self.num_local_experts,
            "qk_head_dim": self.qk_head_dim,
            "kv_lora_rank": self.kv_lora_rank,
            "qk_rope_head_dim": self.qk_rope_head_dim,
            "compressed_kv_dim": self.compressed_kv_dim,
            "first_k_dense_replace": self.first_k_dense_replace,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(
                f"{type(self).__name__}.validate: missing required fields "
                f"{missing} (model_type={self.model_type!r}). Config resolution "
                f"produced an incomplete config; refusing to build the engine."
            )

        if self.num_key_value_heads > self.num_attention_heads:
            raise ValueError(
                f"{type(self).__name__}.validate: num_key_value_heads="
                f"{self.num_key_value_heads} > num_attention_heads="
                f"{self.num_attention_heads}."
            )

        expected_compressed = self.kv_lora_rank + self.qk_rope_head_dim
        if self.compressed_kv_dim != expected_compressed:
            raise ValueError(
                f"{type(self).__name__}.validate: compressed_kv_dim="
                f"{self.compressed_kv_dim} != kv_lora_rank + qk_rope_head_dim="
                f"{self.kv_lora_rank} + {self.qk_rope_head_dim} = "
                f"{expected_compressed}."
            )

        if self.first_k_dense_replace >= self.num_hidden_layers:
            raise ValueError(
                f"{type(self).__name__}.validate: first_k_dense_replace="
                f"{self.first_k_dense_replace} must be < num_hidden_layers="
                f"{self.num_hidden_layers}."
            )

        for name in ("num_hidden_layers", "num_attention_heads",
                     "num_local_experts", "qk_head_dim"):
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(
                    f"{type(self).__name__}.validate: {name}={value} must be > 0."
                )


@register_config("glm_moe_dsa_5_2")
@dataclass
class GLM52Config(GLM5Config):
    """GLM-5.2 (GlmMoeDsaForCausalLM) configuration.

    GLM-5.2 shares the glm_moe_dsa MODEL graph with GLM-5 but is given its own
    config identity (``model_type="glm_moe_dsa_5_2"``) so it can carry distinct
    DSA-indexer scheduling knobs and a longer native context window without
    perturbing GLM-5. Notable checkpoint differences vs GLM-5:

    - max_position_embeddings = 1,048,576 (vs 202,752)
    - rope_theta = 8,000,000, delivered nested under ``rope_parameters``
    - DSA indexer gains ``index_topk_freq`` / ``index_skip_topk_offset`` /
      ``index_share_for_mtp_iteration`` and a per-layer ``indexer_types`` list;
      GLM-5 lacks these.
    """

    # ==================== Identity ====================
    model_type: str = "glm_moe_dsa_5_2"

    # ==================== Position Encoding ====================
    max_position_embeddings: int = 1048576
    rope_theta: float = 8000000.0

    # ==================== DSA (GLM-5.2-only indexer scheduling) ====================
    index_topk_freq: int = 4
    index_skip_topk_offset: int = 3
    index_share_for_mtp_iteration: bool = True
    indexer_types: Optional[List[str]] = None

    @classmethod
    def from_hf(cls, hf_dict: Dict[str, Any]) -> "GLM52Config":
        """Build a rich GLM-5.2 config from a HuggingFace ``config.json`` dict.

        Extends the shared GLM mapping with the GLM-5.2-only indexer fields, and
        applies the same fail-loud / warn-on-drop diagnostics as GLM-5.
        """
        cls._assert_required_hf_keys(cls.__name__, hf_dict)
        kwargs = cls._common_hf_kwargs(hf_dict)
        consumed = hf_dict.get("__consumed_keys__", set())

        def put(field_name: str, hf_key: str) -> None:
            consumed.add(hf_key)
            if hf_key in hf_dict and hf_dict[hf_key] is not None:
                kwargs[field_name] = hf_dict[hf_key]

        put("index_topk_freq", "index_topk_freq")
        put("index_skip_topk_offset", "index_skip_topk_offset")
        put("index_share_for_mtp_iteration", "index_share_for_mtp_iteration")
        put("indexer_types", "indexer_types")

        consumed = hf_dict.pop("__consumed_keys__", set())
        cls._warn_dropped_hf_keys(cls.__name__, hf_dict, consumed)
        cls._assert_mlp_layer_types(
            cls.__name__, hf_dict,
            kwargs.get("first_k_dense_replace", cls.first_k_dense_replace),
        )

        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in kwargs.items() if k in known}
        return cls(**kwargs)


