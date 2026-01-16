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

"""Model configuration classes for BatchGen.

This module provides a unified, extensible model configuration system that:
1. Reduces dependency on HuggingFace AutoConfig
2. Provides clean, consistent config interface across all models
3. Supports easy extension for future models with distinct features

The config flows through the engine:
    config.json -> load_config() -> ModelConfig
        -> Initializer, CoreEngine, PSManager, BatchGenWorker, Wrappers
"""

from dataclasses import dataclass, field, fields, asdict
from typing import Dict, List, Optional, Any
from enum import Enum
from pathlib import Path
import json


class AttentionType(str, Enum):
    """KV projection mechanism (how heads share K/V).

    This is orthogonal to context handling (sliding window vs full attention).
    The model.py determines which layers use sliding vs full attention.
    """
    MHA = "mha"   # Multi-Head Attention: all heads independent (1:1 Q:KV ratio)
    GQA = "gqa"   # Grouped Query Attention: KV heads shared across groups (N:1 ratio)
    MLA = "mla"   # Multi-head Latent Attention: low-rank KV compression via LoRA


class QuantizationType(str, Enum):
    """Weight quantization format."""
    NONE = "none"
    FP8 = "fp8"
    MXFP4 = "mxfp4"
    INT8 = "int8"


@dataclass
class BaseModelConfig:
    """Base model configuration for all BatchGen models.

    This config is instantiated from JSON and flows through:
    - Initializer -> creates config from local dir
    - ParallelStrategyManager -> uses config to build model
    - CoreEngine -> uses config for weight management
    - Wrappers -> use config for layer-specific behavior

    Attributes are organized by category:
    - Identity: model_type, architectures
    - Core Architecture: vocab_size, hidden_size, etc.
    - Attention Type (KV Mechanism): attention_type, num_attention_heads, etc.
    - MLA-specific: kv_lora_rank, q_lora_rank, etc.
    - Context Handling: sliding_window_size
    - Attention Sink: use_attention_sink
    - MoE Configuration: expert counts, routing, etc.
    - Position Encoding: RoPE parameters
    - Normalization & Activation: rms_norm_eps, hidden_act, etc.
    - Quantization: quantization type and config
    - Tokenizer: special token IDs
    """

    # ==================== Identity ====================
    model_type: str = ""
    architectures: List[str] = field(default_factory=list)
    _name_or_path: str = ""  # Original model path/name for compatibility

    # ==================== Core Architecture ====================
    vocab_size: int = 32000
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32

    # ==================== Attention Type (KV Mechanism) ====================
    attention_type: str = "gqa"  # mha, gqa, mla
    num_attention_heads: int = 32
    num_key_value_heads: int = 8  # For GQA: < num_attention_heads
    head_dim: int = 128
    attention_bias: bool = False
    attention_dropout: float = 0.0

    # MLA-specific parameters (DeepSeek)
    kv_lora_rank: Optional[int] = None       # Low-rank KV compression dim
    q_lora_rank: Optional[int] = None        # Low-rank Q compression dim
    qk_nope_head_dim: Optional[int] = None   # Non-RoPE head dim for Q/K
    qk_rope_head_dim: Optional[int] = None   # RoPE head dim for Q/K
    v_head_dim: Optional[int] = None         # Value head dim
    compressed_kv_dim: Optional[int] = None  # Total compressed KV dim

    # ==================== Context Handling (Orthogonal) ====================
    # Sliding window size (can combine with any attention type)
    # Which layers use sliding vs full attention is determined by model.py
    sliding_window_size: Optional[int] = None  # None = full attention for all layers

    # ==================== Attention Sink (GPT-OSS) ====================
    use_attention_sink: bool = False
    # Note: sink values are learned parameters stored in model, not config

    # ==================== MoE Configuration ====================
    # Expert counts
    num_local_experts: int = 1
    num_experts_per_tok: int = 1
    n_shared_experts: int = 0
    n_routed_experts: Optional[int] = None

    # MoE layer pattern
    moe_layer_freq: int = 1  # MoE every N layers
    first_k_dense_replace: int = 0  # First K layers are dense (no MoE)
    moe_intermediate_size: Optional[int] = None

    # Router/Gating
    topk_method: str = "softmax"  # softmax, greedy, noaux_tc
    n_group: Optional[int] = None
    topk_group: Optional[int] = None
    routed_scaling_factor: float = 1.0
    norm_topk_prob: bool = False
    scoring_func: str = "softmax"  # softmax, sigmoid

    # Auxiliary loss
    aux_loss_alpha: float = 0.001
    router_aux_loss_coef: float = 0.0

    # ==================== Position Encoding ====================
    max_position_embeddings: int = 131072
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict[str, Any]] = None

    # YaRN-specific (GPT-OSS)
    initial_context_length: int = 4096
    rope_scaling_factor: float = 1.0
    rope_ntk_alpha: float = 1.0
    rope_ntk_beta: float = 32.0

    # ==================== Normalization & Activation ====================
    rms_norm_eps: float = 1e-6
    hidden_act: str = "silu"
    swiglu_limit: Optional[float] = None  # GPT-OSS: 7.0

    # ==================== Quantization ====================
    quantization: str = "none"  # none, fp8, mxfp4
    quantization_config: Optional[Dict[str, Any]] = None

    # ==================== Tokenizer ====================
    pad_token_id: Optional[int] = None
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    tie_word_embeddings: bool = False

    # ==================== Runtime State (set dynamically) ====================
    _attn_implementation: str = "eager"  # "eager" or "flash_attention_2"
    phase: Optional[str] = None  # "prefill" or "decode" (set during inference)

    # ==================== Methods ====================
    @classmethod
    def from_json(cls, path: str) -> "BaseModelConfig":
        """Load config from JSON file.

        Args:
            path: Path to config.json file

        Returns:
            Config instance with values from JSON
        """
        with open(path, 'r') as f:
            data = json.load(f)
        # Filter to only known fields to avoid TypeError
        known_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)

    @classmethod
    def from_dir(cls, model_dir: str) -> "BaseModelConfig":
        """Load config from model directory.

        Args:
            model_dir: Path to model directory containing config.json

        Returns:
            Config instance
        """
        config_path = Path(model_dir) / "config.json"
        return cls.from_json(str(config_path))

    def is_moe(self) -> bool:
        """Check if this is an MoE model."""
        return self.num_local_experts > 1 or (
            self.n_routed_experts is not None and self.n_routed_experts > 1
        )

    def is_dense_layer(self, layer_idx: int) -> bool:
        """Check if a layer is dense (no MoE).

        Args:
            layer_idx: Layer index (0-based)

        Returns:
            True if layer is dense (no expert routing)
        """
        return layer_idx < self.first_k_dense_replace

    def is_mla(self) -> bool:
        """Check if model uses Multi-head Latent Attention."""
        return self.attention_type == "mla"

    def is_gqa(self) -> bool:
        """Check if model uses Grouped Query Attention."""
        return (
            self.attention_type == "gqa"
            and self.num_key_value_heads < self.num_attention_heads
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in asdict(self).items() if v is not None}

    def __str__(self) -> str:
        """String representation with key attributes."""
        return (
            f"{self.__class__.__name__}(\n"
            f"  model_type={self.model_type},\n"
            f"  hidden_size={self.hidden_size},\n"
            f"  num_hidden_layers={self.num_hidden_layers},\n"
            f"  attention_type={self.attention_type},\n"
            f"  num_attention_heads={self.num_attention_heads},\n"
            f"  num_key_value_heads={self.num_key_value_heads},\n"
            f"  num_local_experts={self.num_local_experts},\n"
            f"  sliding_window_size={self.sliding_window_size},\n"
            f"  quantization={self.quantization}\n"
            f")"
        )
