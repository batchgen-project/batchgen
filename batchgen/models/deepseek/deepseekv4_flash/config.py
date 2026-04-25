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

"""DeepSeek-V4-Flash model configuration for BatchGen."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from batchgen.config.model_config import BaseModelConfig
from batchgen.config.model_registry import register_config


@register_config("deepseek_v4")
@dataclass
class DeepSeekV4FlashConfig(BaseModelConfig):
    """DeepSeek-V4 Flash/Pro configuration entrypoint."""

    model_type: str = "deepseek_v4"
    architectures: List[str] = field(default_factory=lambda: ["DeepseekV4ForCausalLM"])

    vocab_size: int = 129280
    hidden_size: int = 4096
    intermediate_size: int = 18432
    num_hidden_layers: int = 43

    attention_type: str = "mla"
    num_attention_heads: int = 64
    num_key_value_heads: int = 1
    head_dim: int = 512
    attention_bias: bool = False
    attention_dropout: float = 0.0

    q_lora_rank: int = 1024
    qk_rope_head_dim: int = 64
    compressed_kv_dim: Optional[int] = None

    sliding_window_size: Optional[int] = 128

    num_local_experts: int = 256
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    num_experts_per_tok: int = 6
    first_k_dense_replace: int = 0
    moe_intermediate_size: int = 2048
    moe_layer_freq: int = 1

    topk_method: str = "noaux_tc"
    routed_scaling_factor: float = 1.5
    norm_topk_prob: bool = True
    scoring_func: str = "sqrtsoftplus"

    max_position_embeddings: int = 1048576
    rope_theta: float = 10000.0
    rope_scaling: Dict[str, Any] = field(default_factory=lambda: {
        "type": "yarn",
        "factor": 16,
        "original_max_position_embeddings": 65536,
        "beta_fast": 32,
        "beta_slow": 1,
    })

    rms_norm_eps: float = 1e-6
    hidden_act: str = "silu"
    swiglu_limit: float = 10.0

    quantization: str = "fp8"
    quantization_config: Dict[str, Any] = field(default_factory=lambda: {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "scale_fmt": "ue8m0",
        "weight_block_size": [128, 128],
    })

    bos_token_id: int = 0
    eos_token_id: int = 1
    pad_token_id: int = 1
    tie_word_embeddings: bool = False
