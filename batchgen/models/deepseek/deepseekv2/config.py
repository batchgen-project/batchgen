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

"""DeepSeek-V2 model configuration for BatchGen.

This module provides the config class for DeepSeek-V2 model.
It replaces HuggingFace's PretrainedConfig with BatchGen's BaseModelConfig.

DeepSeek-V2 is a 236B parameter MoE model with:
- 30 layers, hidden_size=4096
- MLA (Multi-head Latent Attention) with low-rank KV compression
- Variable number of routed experts (typically 64) + shared experts
- Full attention (no sliding window)
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

from batchgen.config.model_config import BaseModelConfig
from batchgen.config.model_registry import register_config


@register_config("deepseek_v2")
@dataclass
class DeepSeekV2Config(BaseModelConfig):
    """DeepSeek-V2 configuration.

    Attention: MLA (Multi-head Latent Attention with low-rank KV compression)
    Context: Full attention (no sliding window)
    """

    # ==================== Identity ====================
    model_type: str = "deepseek_v2"
    architectures: List[str] = field(default_factory=lambda: ["DeepseekV2ForCausalLM"])

    # ==================== Core Architecture ====================
    vocab_size: int = 102400
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 30

    # ==================== Attention Type: MLA with low-rank compression ====================
    attention_type: str = "mla"
    num_attention_heads: int = 32
    num_key_value_heads: int = 32  # Not used in MLA (uses lora_rank instead)
    head_dim: int = 192  # Computed: qk_nope_head_dim + qk_rope_head_dim
    attention_bias: bool = False
    attention_dropout: float = 0.0

    # MLA-specific: Low-rank KV compression
    kv_lora_rank: int = 512        # KV compression dimension
    q_lora_rank: int = 1536        # Q compression dimension
    qk_nope_head_dim: int = 128    # Non-RoPE portion of Q/K
    qk_rope_head_dim: int = 64     # RoPE portion of Q/K
    v_head_dim: int = 128          # Value head dimension
    compressed_kv_dim: int = 576   # Total compressed KV dim for cache

    # ==================== Context Handling: Full attention (no sliding window) ====================
    sliding_window_size: Optional[int] = None

    # ==================== MoE Configuration ====================
    # Note: V2 has configurable experts, defaults are placeholders
    num_local_experts: int = 1
    n_routed_experts: Optional[int] = None
    n_shared_experts: Optional[int] = None
    num_experts_per_tok: Optional[int] = None
    first_k_dense_replace: int = 0
    moe_intermediate_size: int = 1407
    moe_layer_freq: int = 1

    # Router/Gating
    topk_method: str = "gready"  # Note: original typo preserved for compatibility
    n_group: Optional[int] = None
    topk_group: Optional[int] = None
    routed_scaling_factor: float = 1.0
    norm_topk_prob: bool = False
    scoring_func: str = "softmax"

    # Auxiliary loss
    aux_loss_alpha: float = 0.001

    # ==================== Position Encoding ====================
    max_position_embeddings: int = 2048
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict[str, Any]] = None

    # ==================== Normalization & Activation ====================
    rms_norm_eps: float = 1e-6
    hidden_act: str = "silu"

    # ==================== Quantization ====================
    quantization: str = "none"

    # ==================== Tokenizer ====================
    bos_token_id: int = 100000
    eos_token_id: int = 100001
    pad_token_id: Optional[int] = None
    tie_word_embeddings: bool = False

    # ==================== DeepSeek-specific ====================
    ep_size: int = 1
    seq_aux: bool = True

    # ==================== Aliases for compatibility ====================
    @property
    def num_kv_heads(self) -> int:
        """Alias for num_key_value_heads."""
        return self.num_key_value_heads
