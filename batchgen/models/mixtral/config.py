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

"""Mixtral model configuration for BatchGen.

This module provides the config class for Mixtral models.
It replaces HuggingFace's PretrainedConfig with BatchGen's BaseModelConfig.

Mixtral-8x7B is a 47B parameter MoE model with:
- 32 layers, hidden_size=4096
- GQA: 32 attention heads, 8 KV heads
- 8 experts, Top-2 routing
- Full attention with optional sliding window (4096)
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

from batchgen.config.model_config import BaseModelConfig
from batchgen.config.model_registry import register_config


@register_config("mixtral")
@dataclass
class MixtralConfig(BaseModelConfig):
    """Mixtral-8x7B configuration.

    Attention: GQA (32 Q heads, 8 KV heads = 4:1 ratio)
    Context: Sliding window attention (4096) - all layers
    """

    # ==================== Identity ====================
    model_type: str = "mixtral"
    architectures: List[str] = field(default_factory=lambda: ["MixtralForCausalLM"])

    # ==================== Core Architecture ====================
    vocab_size: int = 32000
    hidden_size: int = 4096
    intermediate_size: int = 14336
    num_hidden_layers: int = 32

    # ==================== Attention Type: GQA with 4:1 ratio ====================
    attention_type: str = "gqa"
    num_attention_heads: int = 32
    num_key_value_heads: int = 8  # 32/8 = 4 Q heads per KV head
    head_dim: int = 128
    attention_bias: bool = False
    attention_dropout: float = 0.0

    # ==================== Context Handling ====================
    # Mixtral uses sliding window attention for all layers
    sliding_window_size: int = 4096

    # ==================== MoE ====================
    num_local_experts: int = 8
    num_experts_per_tok: int = 2

    # Router/Gating
    router_aux_loss_coef: float = 0.02

    # ==================== Position Encoding ====================
    max_position_embeddings: int = 32768
    rope_theta: float = 1000000.0
    rope_scaling: Optional[Dict[str, Any]] = None

    # ==================== Normalization & Activation ====================
    rms_norm_eps: float = 1e-5
    hidden_act: str = "silu"

    # ==================== Quantization ====================
    quantization: str = "none"

    # ==================== Tokenizer ====================
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: Optional[int] = None
    tie_word_embeddings: bool = False

    # ==================== Aliases for compatibility ====================
    @property
    def num_kv_heads(self) -> int:
        """Alias for num_key_value_heads."""
        return self.num_key_value_heads

    @property
    def sliding_window(self) -> int:
        """Alias for sliding_window_size."""
        return self.sliding_window_size
