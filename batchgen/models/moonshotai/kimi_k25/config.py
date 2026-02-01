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

"""Kimi K2.5 model configuration for BatchGen.

This module provides the config class for Kimi K2.5 model.

Kimi K2.5 is a 1.04T parameter MoE model (DeepSeek-V3 variant) with:
- 61 layers, hidden_size=7168
- MLA (Multi-head Latent Attention) — same as DeepSeek-V3
- 384 routed experts + 1 shared expert, Top-8 routing (no expert grouping)
- INT4 W4A16 quantization (weight-only, group_size=32, symmetric)
- BF16 attention (no FP8 quantization on attention)
- Full attention (no sliding window)
- First 3 layers are dense (no MoE)

Key deltas from DeepSeek-V3:
- 384 routed experts (vs 256), n_group=1 (vs 8)
- INT4 quantization (vs FP8) — needs W4A16 dequant kernel
- RoPE theta=50000 (vs 10000)
- BF16 attention (vs FP8)
- Same HF architecture class: DeepseekV3ForCausalLM
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

from batchgen.config.model_config import BaseModelConfig
from batchgen.config.model_registry import register_config


@register_config("kimi_k25")
@dataclass
class KimiK25Config(BaseModelConfig):
    """Kimi K2.5 configuration.

    Attention: MLA (Multi-head Latent Attention with low-rank KV compression)
    Context: Full attention (no sliding window)
    Architecture: DeepseekV3ForCausalLM (reuses DeepSeek-V3 model code)
    """

    # ==================== Identity ====================
    model_type: str = "kimi_k25"
    architectures: List[str] = field(default_factory=lambda: ["DeepseekV3ForCausalLM"])

    # ==================== Core Architecture ====================
    vocab_size: int = 129280
    hidden_size: int = 7168
    intermediate_size: int = 18432
    num_hidden_layers: int = 61

    # ==================== Attention Type: MLA with low-rank compression ====================
    attention_type: str = "mla"
    num_attention_heads: int = 64
    num_key_value_heads: int = 64  # Not used in MLA (uses lora_rank instead)
    head_dim: int = 192  # qk_nope_head_dim + qk_rope_head_dim
    attention_bias: bool = False
    attention_dropout: float = 0.0

    # MLA-specific: Low-rank KV compression (same as DeepSeek-V3)
    kv_lora_rank: int = 512
    q_lora_rank: int = 1536
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    compressed_kv_dim: int = 576  # kv_lora_rank + qk_rope_head_dim

    # ==================== Context Handling: Full attention ====================
    sliding_window_size: Optional[int] = None

    # ==================== MoE Configuration ====================
    num_local_experts: int = 384       # vs 256 in DeepSeek-V3
    n_routed_experts: int = 384        # vs 256
    n_shared_experts: int = 1
    num_experts_per_tok: int = 8
    first_k_dense_replace: int = 1     # First layer (layer 0) is dense, rest are MoE
    moe_intermediate_size: int = 2048
    moe_layer_freq: int = 1

    # Router/Gating — no expert grouping (n_group=1)
    topk_method: str = "noaux_tc"
    n_group: int = 1                   # vs 8 in DeepSeek-V3 (no grouping)
    topk_group: int = 1               # vs 4 in DeepSeek-V3
    routed_scaling_factor: float = 2.5
    norm_topk_prob: bool = True
    scoring_func: str = "sigmoid"

    # Auxiliary loss
    aux_loss_alpha: float = 0.001

    # ==================== Position Encoding ====================
    max_position_embeddings: int = 163840
    rope_theta: float = 50000.0        # vs 10000 in DeepSeek-V3
    rope_scaling: Dict[str, Any] = field(default_factory=lambda: {
        "type": "yarn",
        "factor": 40,
        "original_max_position_embeddings": 4096,
        "beta_fast": 32,
        "beta_slow": 1,
        "mscale": 1.0,
        "mscale_all_dim": 1.0,
    })

    # ==================== Normalization & Activation ====================
    rms_norm_eps: float = 1e-6
    hidden_act: str = "silu"

    # ==================== Quantization: INT4 W4A16 (weight-only) ====================
    quantization: str = "int4"
    quantization_config: Dict[str, Any] = field(default_factory=lambda: {
        "quant_method": "compressed-tensors",
        "weights": {
            "num_bits": 4,
            "group_size": 32,
            "symmetric": True,
        },
        # W4A16: activations NOT quantized (stay BF16)
        "input_activations": None,
        "output_activations": None,
    })

    # ==================== Tokenizer ====================
    bos_token_id: int = 0
    eos_token_id: int = 1
    pad_token_id: Optional[int] = None
    tie_word_embeddings: bool = False

    # ==================== K2.5-specific ====================
    num_nextn_predict_layers: int = 1
    ep_size: int = 1
    seq_aux: bool = True

    # ==================== Aliases for compatibility ====================
    @property
    def num_kv_heads(self) -> int:
        """Alias for num_key_value_heads."""
        return self.num_key_value_heads
