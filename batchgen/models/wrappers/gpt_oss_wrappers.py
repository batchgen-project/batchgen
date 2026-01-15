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

"""GPT-OSS-specific wrappers for BatchGen execution.

Provides wrappers for GPT-OSS-120B model with MXFP4 quantization:
- GptOssExpertWrapper: Expert wrapper with MXFP4 dequantization
- GptOssAttnWrapper: Attention wrapper (BF16, no quantization)

Optimized for single H20 GPU deployment (world_size == 1).
"""

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn

from .expert import ExpertWrapperBase
from .attention import AttnWrapperBase


class GptOssExpertWrapper(ExpertWrapperBase):
    """Expert wrapper with MXFP4 dequantization for GPT-OSS-120B.

    For world_size == 1 (single H20 GPU):
    - All 128 experts are local
    - No expert parallelism needed
    - Weights loaded from core_engine and dequantized on-the-fly

    MXFP4 format:
    - 32 FP4 values per scale
    - Packed as 2 values per uint8 byte
    - Scale stored as uint8, exponent = scale - 127

    Attributes:
        dequant_fn: MXFP4 dequantization function
    """

    def __init__(
        self,
        module: nn.Module,
        layer_idx: int,
        expert_idx: int,
        core_engine,
        engine_config,
        model_config,
    ):
        """Initialize GPT-OSS expert wrapper.

        Args:
            module: Expert FFN module (SwiGLU)
            layer_idx: Layer index in the model (0-35)
            expert_idx: Expert index (0-127)
            core_engine: BatchGen core engine
            engine_config: Engine configuration
            model_config: Model configuration
        """
        # GPT-OSS always loads weights (all experts local, no caching)
        super().__init__(
            module, layer_idx, expert_idx, core_engine, engine_config, model_config,
            get_weights=True
        )

        # Store dimensions for weight slicing
        self.intermediate_size = model_config.intermediate_size
        self.hidden_size = model_config.hidden_size
        self.num_experts = model_config.num_local_experts

        # Import MXFP4 dequantization
        try:
            from batchgen.quantization.mxfp4 import mxfp4_dequantize
            self.dequant_fn = mxfp4_dequantize
        except ImportError:
            logging.warning("MXFP4 dequantization not available, using identity")
            self.dequant_fn = lambda packed, scales, dtype=torch.bfloat16: packed

        # Store weights for passing to ExpertMLP.forward()
        # ExpertMLP expects weights_dict as argument, not as module parameters
        self._current_weights = None

    def _build_module_key(self) -> str:
        """Build module key for weight loading.

        GPT-OSS uses SHARED weights per layer, so all experts in a layer
        share the same module_key. Weight slicing is done in dequantize_weights().

        Returns:
            Per-layer module key: 'routed_expert_{layer_idx}'
        """
        return f"routed_expert_{self.layer_idx}"

    def dequantize_weights(
        self, weights_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Convert loaded weights to format expected by ExpertMLP.forward().

        ExpertMLP expects weights_dict with these keys:
        - 'mlp1.packed': [intermediate*2, hidden//2] uint8
        - 'mlp1.scales': [intermediate*2, hidden//32] uint8
        - 'mlp1.bias': [intermediate*2] BF16
        - 'mlp2.packed': [hidden, intermediate//2] uint8
        - 'mlp2.scales': [hidden, intermediate//32] uint8
        - 'mlp2.bias': [hidden] BF16

        The weights are kept in MXFP4 format - ExpertMLP handles dequantization
        internally via fused_mxfp4_gemm.

        Args:
            weights_dict: Dict with weights from core_engine

        Returns:
            Dict formatted for ExpertMLP.forward()
        """
        result = {}

        # Map weight names to ExpertMLP expected format
        # Core engine may provide weights with different naming conventions
        key_mapping = {
            'mlp1.packed': ['mlp1.packed', 'mlp1_weight.blocks', 'gate_up.packed'],
            'mlp1.scales': ['mlp1.scales', 'mlp1_weight.scales', 'gate_up.scales'],
            'mlp1.bias': ['mlp1.bias', 'mlp1_bias', 'gate_up.bias'],
            'mlp2.packed': ['mlp2.packed', 'mlp2_weight.blocks', 'down.packed'],
            'mlp2.scales': ['mlp2.scales', 'mlp2_weight.scales', 'down.scales'],
            'mlp2.bias': ['mlp2.bias', 'mlp2_bias', 'down.bias'],
        }

        for target_key, source_keys in key_mapping.items():
            for src_key in source_keys:
                if src_key in weights_dict:
                    result[target_key] = weights_dict[src_key]
                    break

        # Store for use in _forward_impl
        self._current_weights = result
        return result

    def _forward_impl(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward through ExpertMLP with MXFP4 weights.

        ExpertMLP.forward() expects (x, weights_dict) where weights_dict
        contains MXFP4 packed weights and scales.

        Args:
            hidden_states: Input tensor [num_tokens, hidden_size]

        Returns:
            Output tensor [num_tokens, hidden_size]
        """
        if self._current_weights is None:
            raise RuntimeError(
                f"Expert {self.expert_idx} layer {self.layer_idx}: "
                "weights not loaded before forward"
            )
        return self.module(hidden_states, self._current_weights)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass with MXFP4 weights.

        Flow:
        1. Load MXFP4 packed weights from core engine
        2. Format weights for ExpertMLP (stores in self._current_weights)
        3. Micro-batch forward through ExpertMLP
        4. Cleanup

        Args:
            hidden_states: Input tensor [num_tokens, hidden_size]

        Returns:
            Output tensor [num_tokens, hidden_size]
        """
        rank = self.get_rank_safe()
        logging.debug(
            f"[Rank {rank} Layer {self.layer_idx} Expert {self.expert_idx}] "
            f"GPT-OSS expert forward. Phase: {self.phase}"
        )

        # Load MXFP4 weights from core engine
        weights = self.load_weights(self.module_key)

        # Format weights for ExpertMLP (stores in self._current_weights)
        self.dequantize_weights(weights)

        # Micro-batch forward (uses self._current_weights)
        result = self.micro_batch_forward(hidden_states, "expert")

        # Cleanup
        torch.cuda.current_stream(
            self.engine_config.Basic_Config.device_torch
        ).synchronize()
        self.free_weights(self.module_key)
        self._current_weights = None  # Clear weights reference

        logging.debug(
            f"[Rank {rank} Layer {self.layer_idx} Expert {self.expert_idx}] "
            f"GPT-OSS expert forward complete. Phase: {self.phase}"
        )

        return result


class GptOssAttnWrapper(AttnWrapperBase):
    """Attention wrapper for GPT-OSS-120B.

    GPT-OSS attention is in BF16 (not quantized), so this wrapper
    handles:
    - Weight loading from core engine
    - GQA attention (64 query heads, 8 KV heads)
    - Alternating sliding (128) / full attention per layer

    Attributes:
        is_sliding: Whether this layer uses sliding window attention
        sliding_window: Window size for sliding attention (128)
    """

    def __init__(
        self,
        module: nn.Module,
        layer_idx: int,
        core_engine,
        engine_config,
        model_config,
    ):
        """Initialize GPT-OSS attention wrapper.

        Args:
            module: Attention module (GQA)
            layer_idx: Layer index (0-35)
            core_engine: BatchGen core engine
            engine_config: Engine configuration
            model_config: Model configuration
        """
        # GPT-OSS attention is BF16, no dequantization needed
        super().__init__(
            module, layer_idx, core_engine, engine_config, model_config,
            get_weights=True, weight_dequant_scale=None
        )

        # Determine if this layer uses sliding window
        # GPT-OSS uses alternating: even layers = sliding, odd = full
        self.is_sliding = (layer_idx % 2 == 0)
        self.sliding_window = 128 if self.is_sliding else None

    def dequantize_weights(
        self, weights_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """No dequantization needed - attention weights are BF16."""
        return weights_dict

    def _forward_prefill(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        """Prefill forward with GQA and optional sliding window.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            **kwargs: attention_mask, position_ids, past_key_value, etc.

        Returns:
            Output tensor
        """
        # Use module's forward directly
        # The module handles GQA and sliding window internally
        return self.module(hidden_states, **kwargs)

    def _forward_decode(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        """Decode forward with GQA.

        Args:
            hidden_states: Input tensor [batch, 1, hidden_size]
            **kwargs: attention_mask, position_ids, past_key_value, etc.

        Returns:
            Output tensor
        """
        return self.module(hidden_states, **kwargs)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Forward pass with weight loading.

        Args:
            *args: Positional arguments
            **kwargs: hidden_states, attention_mask, position_ids, etc.

        Returns:
            Attention output and optional cache
        """
        rank = self.get_rank_safe()
        logging.debug(
            f"[Rank {rank} Layer {self.layer_idx}] "
            f"GPT-OSS attn forward. Phase: {self.phase}, "
            f"sliding={self.is_sliding}"
        )

        # Load weights
        if self.get_weights:
            weights = self.load_weights(self.module_key)
            self.apply_weights(weights)

        # Route to phase handler
        # Extract hidden_states to avoid passing it twice (positionally and in kwargs)
        hidden_states = kwargs.pop("hidden_states", None)
        if self.phase == "prefill":
            result = self._forward_prefill(hidden_states, **kwargs)
        else:
            result = self._forward_decode(hidden_states, **kwargs)

        logging.debug(
            f"[Rank {rank} Layer {self.layer_idx}] "
            f"GPT-OSS attn forward complete. Phase: {self.phase}"
        )

        return result
