# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                             #
#                                                                               #
#  licensed under the apache license, version 2.0 (the "license");              #
#  you may not use this file except in compliance with the license.             #
#                                                                               #
#  you may obtain a copy of the license at                                      #
#                                                                               #
#                  http://www.apache.org/licenses/license-2.0                   #
#                                                                               #
#  unless required by applicable law or agreed to in writing, software          #
#  distributed under the license is distributed on an "as is" basis,            #
#  without warranties or conditions of any kind, either express or implied.     #
#  see the license for the specific language governing permissions and          #
#  limitations under the license.                                               #
# ---------------------------------------------------------------------------- #

"""GPT-OSS-120B model configuration for BatchGen.

This module provides the config class for GPT-OSS-120B model.
It replaces HuggingFace's PretrainedConfig with BatchGen's BaseModelConfig.

GPT-OSS-120B is a 117B parameter MoE model with:
- 36 layers, hidden_size=2880
- GQA: 64 attention heads, 8 KV heads (8:1 ratio)
- 128 experts, Top-4 routing
- MXFP4 quantization (~4.25 bits/param)
- Alternating sliding (128) / full attention (handled in model.py)
- YaRN RoPE with theta=150000, factor=32
- Custom SwiGLU with alpha=1.702 and limit=7.0
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List

from batchgen.config.model_config import BaseModelConfig
from batchgen.config.model_registry import register_config


@register_config("gpt_oss")
@dataclass
class GptOss120BConfig(BaseModelConfig):
    """GPT-OSS-120B configuration.

    Attention: GQA (64 Q heads, 8 KV heads = 8:1 ratio)
    Context: Sliding window size = 128 (interleaved pattern handled in model.py)

    Note: GPT-OSS uses interleaved attention where even layers use sliding
    window (128) and odd layers use full attention. This architecture pattern
    is implemented in model.py, not configured here.
    """

    # ==================== Identity ====================
    model_type: str = "gpt_oss"
    architectures: List[str] = field(default_factory=lambda: ["GptOssForCausalLM"])

    # ==================== Core Architecture ====================
    vocab_size: int = 201088
    hidden_size: int = 2880
    intermediate_size: int = 2880
    num_hidden_layers: int = 36

    # ==================== Attention Type: GQA with 8:1 ratio ====================
    attention_type: str = "gqa"
    num_attention_heads: int = 64
    num_key_value_heads: int = 8  # 64/8 = 8 Q heads per KV head
    head_dim: int = 64
    attention_bias: bool = True
    attention_dropout: float = 0.0

    # ==================== Context Handling ====================
    # Window size for sliding attention layers
    # Which layers use sliding vs full is determined by model.py (interleaved pattern)
    sliding_window_size: int = 128

    # ==================== Attention Sink (learned per-head bias) ====================
    use_attention_sink: bool = True

    # ==================== MoE ====================
    num_local_experts: int = 128
    num_experts_per_tok: int = 4

    # ==================== RoPE with YaRN scaling ====================
    max_position_embeddings: int = 131072
    rope_theta: float = 150000.0
    initial_context_length: int = 4096
    rope_scaling_factor: float = 32.0
    rope_ntk_alpha: float = 1.0
    rope_ntk_beta: float = 32.0
    rope_scaling: Dict[str, Any] = field(default_factory=lambda: {
        "rope_type": "yarn",
        "factor": 32.0,
        "original_max_position_embeddings": 4096,
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "truncate": False,
    })

    # ==================== Activation ====================
    hidden_act: str = "silu"
    swiglu_limit: float = 7.0
    rms_norm_eps: float = 1e-5

    # ==================== Quantization ====================
    quantization: str = "mxfp4"
    quantization_config: Dict[str, Any] = field(default_factory=lambda: {
        "quant_method": "mxfp4",
        "modules_to_not_convert": [
            "model.layers.*.self_attn",
            "model.layers.*.mlp.router",
            "model.embed_tokens",
            "lm_head",
        ],
    })

    # ==================== Tokenizer ====================
    pad_token_id: int = 199999
    eos_token_id: int = 200002
    tie_word_embeddings: bool = False

    # ==================== GPT-OSS specific ====================
    # Alias for compatibility with existing code
    @property
    def experts_per_token(self) -> int:
        """Alias for num_experts_per_tok."""
        return self.num_experts_per_tok

    @property
    def sliding_window(self) -> int:
        """Alias for sliding_window_size."""
        return self.sliding_window_size

    @property
    def num_kv_heads(self) -> int:
        """Alias for num_key_value_heads."""
        return self.num_key_value_heads
