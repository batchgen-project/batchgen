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

"""MiniMax-M2.5 model configuration for BatchGen.

MiniMax-M2.5 is a 230B parameter MoE model (~10B active) with:
- 62 layers, hidden_size=3072
- GQA: 48 Q heads, 8 KV heads, head_dim=128
- QK Norm: per-layer RMSNorm on Q/K projections
- Partial RoPE: rotary_dim=64 (50%), theta=5M, no scaling
- 256 routed experts, Top-8 sigmoid routing with correction bias, no shared experts
- FP8 e4m3fn quantization, block_size [128, 128]
- All 62 layers are MoE (no dense layers)
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

from batchgen.config.model_config import BaseModelConfig
from batchgen.config.model_registry import register_config


@register_config("minimax_m25")
@dataclass
class MiniMaxM25Config(BaseModelConfig):
    """MiniMax-M2.5 configuration.

    Attention: GQA (48 Q heads, 8 KV heads, head_dim=128)
    Context: Full attention (no sliding window)
    """

    # ==================== Identity ====================
    model_type: str = "minimax_m25"
    architectures: List[str] = field(default_factory=lambda: ["MiniMaxM2ForCausalLM"])

    # ==================== Core Architecture ====================
    vocab_size: int = 200064
    hidden_size: int = 3072
    intermediate_size: int = 1536  # per-expert FFN intermediate
    num_hidden_layers: int = 62

    # ==================== Attention: GQA ====================
    attention_type: str = "gqa"
    num_attention_heads: int = 48
    num_key_value_heads: int = 8
    head_dim: int = 128
    attention_bias: bool = False
    attention_dropout: float = 0.0

    # QK Norm
    use_qk_norm: bool = True

    # ==================== Context Handling ====================
    sliding_window_size: Optional[int] = None

    # ==================== MoE Configuration ====================
    num_local_experts: int = 256
    n_routed_experts: int = 256
    n_shared_experts: int = 0  # No shared experts
    num_experts_per_tok: int = 8
    first_k_dense_replace: int = 0  # All layers are MoE
    moe_intermediate_size: int = 1536
    moe_layer_freq: int = 1

    # Router/Gating
    scoring_func: str = "sigmoid"
    use_routing_bias: bool = True  # e_score_correction_bias
    n_group: int = 1
    topk_group: int = 1
    norm_topk_prob: bool = True

    # ==================== Position Encoding: Partial RoPE ====================
    max_position_embeddings: int = 196608
    rope_theta: float = 5000000.0
    rotary_dim: int = 64  # Only 64 of 128 head dims are rotated
    rope_scaling: Dict[str, Any] = field(default_factory=lambda: {})  # No scaling

    # ==================== Normalization & Activation ====================
    rms_norm_eps: float = 1e-6
    hidden_act: str = "silu"

    # ==================== Quantization: FP8 ====================
    quantization: str = "fp8"
    quantization_config: Dict[str, Any] = field(default_factory=lambda: {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "fmt": "float8_e4m3fn",
        "weight_block_size": [128, 128],
        "modules_to_not_convert": ["gate", "e_score_correction_bias", "unembedding"],
    })

    # ==================== Tokenizer ====================
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None
    tie_word_embeddings: bool = False

    # ==================== MTP (deferred) ====================
    num_mtp_modules: int = 3

    # ==================== Aliases ====================
    @property
    def num_kv_heads(self) -> int:
        return self.num_key_value_heads
