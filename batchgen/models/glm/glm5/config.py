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

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

from batchgen.config.model_config import BaseModelConfig
from batchgen.config.model_registry import register_config


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
    # decode). Used to bisect whether output degradation lives in DSA or in
    # the core MLA/MoE stack. Can also be toggled via env var
    # BATCHGEN_GLM5_USE_DENSE_MLA=1 (propagated in batchgen_worker.py).
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

    def __post_init__(self):
        # Honor env var BATCHGEN_GLM5_USE_DENSE_MLA=1 even when the config
        # dict didn't set it, so the flag can be flipped at launch time
        # without editing config files. Matches the BATCHGEN_GLM5_* knob
        # convention (FORCE_DENSE_MLA, DISABLE_DUAL_KV, DSA_DIAG).
        import os as _os_glm5cfg
        if _os_glm5cfg.environ.get("BATCHGEN_GLM5_USE_DENSE_MLA", "0") == "1":
            self.use_dense_mla = True
