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

"""Qwen3 model configuration for BatchGen.

Qwen3 is a dense transformer with:
- 36 layers, hidden_size=4096
- GQA: 32 attention heads, 8 KV heads (4:1 ratio)
- head_dim=128
- Standard RoPE with theta=1,000,000
- SiLU activation (standard SwiGLU)
- No quantization (BF16 weights)
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List

from batchgen.config.model_config import BaseModelConfig
from batchgen.config.model_registry import register_config


@register_config("qwen3")
@dataclass
class Qwen3Config(BaseModelConfig):
    """Qwen3 configuration for Qwen3Guard-Gen-8B and similar models.

    Attention: GQA (32 Q heads, 8 KV heads = 4:1 ratio)
    Dense model (no MoE).
    """

    # ==================== Identity ====================
    model_type: str = "qwen3"
    architectures: List[str] = field(default_factory=lambda: ["Qwen3ForCausalLM"])

    # ==================== Core Architecture ====================
    vocab_size: int = 151936
    hidden_size: int = 4096
    intermediate_size: int = 12288
    num_hidden_layers: int = 36

    # ==================== Attention Type: GQA with 4:1 ratio ====================
    attention_type: str = "gqa"
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    attention_bias: bool = False
    attention_dropout: float = 0.0

    # ==================== Context Handling ====================
    max_position_embeddings: int = 32768

    # ==================== RoPE ====================
    rope_theta: float = 1000000.0

    # ==================== Activation ====================
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-6

    # ==================== Quantization ====================
    quantization: str = "none"

    # ==================== Tokenizer ====================
    bos_token_id: int = 151643
    eos_token_id: int = 151645
    pad_token_id: int = 151643
    tie_word_embeddings: bool = False

    @property
    def num_kv_heads(self) -> int:
        return self.num_key_value_heads
