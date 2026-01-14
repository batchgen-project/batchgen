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

"""Configuration class for GPT-OSS-120B model."""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from typing import List, Optional

logger = logging.get_logger(__name__)


class GptOssConfig(PretrainedConfig):
    """Configuration class for GPT-OSS-120B model.

    GPT-OSS-120B is a 117B parameter MoE model with:
    - 36 layers, hidden_size=2880
    - GQA: 64 attention heads, 8 KV heads
    - 128 experts, Top-4 routing
    - MXFP4 quantization (~4.25 bits/param)
    - Alternating sliding (128) / full attention
    - YaRN RoPE with theta=150000, factor=32

    Args:
        vocab_size: Size of the vocabulary (default: 201088)
        hidden_size: Dimension of hidden representations (default: 2880)
        intermediate_size: Dimension of expert FFN (default: 2880, SwiGLU)
        num_hidden_layers: Number of transformer layers (default: 36)
        num_attention_heads: Number of attention heads (default: 64)
        num_key_value_heads: Number of KV heads for GQA (default: 8)
        head_dim: Dimension per attention head (default: 64)
        num_local_experts: Total number of experts (default: 128)
        num_experts_per_tok: Experts activated per token (default: 4)
        sliding_window: Sliding window size for alternating attention (default: 128)
        layer_types: List of attention types per layer (default: alternating sliding/full)
        rope_theta: Base frequency for RoPE (default: 150000)
        rope_scaling: YaRN scaling configuration (default: factor=32, beta_fast=32, beta_slow=1)
        swiglu_limit: Clamping value for SwiGLU activation (default: 7.0)
        rms_norm_eps: Epsilon for RMSNorm (default: 1e-5)
        attention_bias: Whether to use bias in attention projections (default: True)
        attention_dropout: Dropout rate for attention (default: 0.0)
        max_position_embeddings: Maximum sequence length (default: 131072)
        quantization_config: MXFP4 quantization configuration
    """

    model_type = "gpt_oss"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 201088,
        hidden_size: int = 2880,
        intermediate_size: int = 2880,
        num_hidden_layers: int = 36,
        num_attention_heads: int = 64,
        num_key_value_heads: int = 8,
        head_dim: int = 64,
        num_local_experts: int = 128,
        num_experts_per_tok: int = 4,
        experts_per_token: int = 4,  # Alias for compatibility
        sliding_window: int = 128,
        layer_types: Optional[List[str]] = None,
        hidden_act: str = "silu",
        rope_theta: float = 150000.0,
        rope_scaling: Optional[dict] = None,
        swiglu_limit: float = 7.0,
        rms_norm_eps: float = 1e-5,
        attention_bias: bool = True,
        attention_dropout: float = 0.0,
        max_position_embeddings: int = 131072,
        initial_context_length: int = 4096,
        initializer_range: float = 0.02,
        use_cache: bool = True,
        pad_token_id: int = 199999,
        eos_token_id: int = 200002,
        tie_word_embeddings: bool = False,
        output_router_logits: bool = False,
        router_aux_loss_coef: float = 0.9,
        quantization_config: Optional[dict] = None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.num_local_experts = num_local_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.experts_per_token = experts_per_token
        self.sliding_window = sliding_window
        self.hidden_act = hidden_act
        self.swiglu_limit = swiglu_limit
        self.rms_norm_eps = rms_norm_eps
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.max_position_embeddings = max_position_embeddings
        self.initial_context_length = initial_context_length
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef

        # BatchGen compatibility: GPT-OSS has all MoE layers (no dense-only layers)
        # first_k_dense_replace = 1 means layer 0+ are MoE (no dense prefix)
        self.first_k_dense_replace = 1

        # Layer types: alternating sliding/full attention
        if layer_types is None:
            self.layer_types = ["sliding_attention", "full_attention"] * (num_hidden_layers // 2)
        else:
            self.layer_types = layer_types

        # RoPE scaling configuration (YaRN)
        if rope_scaling is None:
            self.rope_scaling = {
                "rope_type": "yarn",
                "factor": 32.0,
                "original_max_position_embeddings": 4096,
                "beta_fast": 32.0,
                "beta_slow": 1.0,
                "truncate": False,
            }
        else:
            self.rope_scaling = rope_scaling
        self.rope_theta = rope_theta

        # Quantization config (MXFP4)
        if quantization_config is None:
            self.quantization_config = {
                "quant_method": "mxfp4",
                "modules_to_not_convert": [
                    "model.layers.*.self_attn",
                    "model.layers.*.mlp.router",
                    "model.embed_tokens",
                    "lm_head",
                ],
            }
        else:
            self.quantization_config = quantization_config

        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def num_kv_heads(self) -> int:
        """Alias for num_key_value_heads."""
        return self.num_key_value_heads

    def is_sliding_attention(self, layer_idx: int) -> bool:
        """Check if a layer uses sliding window attention."""
        if layer_idx < len(self.layer_types):
            return self.layer_types[layer_idx] == "sliding_attention"
        # Default to alternating pattern
        return layer_idx % 2 == 0
