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

"""DeepSeek-specific wrappers for BatchGen execution.

Provides wrappers for DeepSeek-R1 and DeepSeek-V3 models with FP8 quantization:
- DeepSeekExpertWrapper: Expert wrapper with FP8 dequantization
- DeepSeekAttnWrapper: Attention wrapper with FP8 dequantization
"""

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn

from .expert import ExpertWrapperBase
from .attention import AttnWrapperBase


def deepseek_v3_dequantization(
    x: torch.Tensor, scale_inv: torch.Tensor, block_size: int = 128
) -> torch.Tensor:
    """Dequantize FP8 weights using block-wise scaling.

    Args:
        x: FP8 quantized weights
        scale_inv: Inverse scale factors
        block_size: Block size for dequantization

    Returns:
        Dequantized BF16 weights
    """
    # Import the actual dequantization function
    try:
        from batchgen.models.Wrapper import deepseek_v3_dequantization as _dequant
        return _dequant(x, scale_inv, block_size)
    except ImportError:
        # Fallback: simple multiplication
        return x.to(torch.bfloat16) * scale_inv


class DeepSeekExpertWrapper(ExpertWrapperBase):
    """Expert wrapper with FP8 dequantization for DeepSeek-R1/V3.

    Handles:
    - FP8 block-wise dequantization with scale factors
    - FP8 weight caching for local experts (no loading needed)
    - deepgemm kernel for forward computation

    Attributes:
        weight_dequant_scale: Dict of scale factors for dequantization
        fp8_gate: Cached FP8 gate_proj weights (when not loading)
        fp8_up: Cached FP8 up_proj weights
        fp8_down: Cached FP8 down_proj weights
    """

    def __init__(
        self,
        module: nn.Module,
        layer_idx: int,
        expert_idx: int,
        core_engine,
        engine_config,
        model_config,
        persistent: bool = False,
        weight_dequant_scale: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """Initialize DeepSeek expert wrapper.

        Args:
            module: Expert FFN module to wrap
            layer_idx: Layer index in the model
            expert_idx: Expert index (-1 for shared expert)
            core_engine: BatchGen core engine
            engine_config: Engine configuration
            model_config: Model configuration
            persistent: Whether weights are pre-loaded on GPU.
                        True = pre-loaded, no buffer fetch needed.
                        False = load from buffer each forward.
            weight_dequant_scale: Dict of FP8 scale factors
        """
        super().__init__(
            module, layer_idx, expert_idx, core_engine, engine_config, model_config,
            persistent
        )
        self.weight_dequant_scale = weight_dequant_scale or {}

        # FP8 weight caching for local experts
        self.fp8_gate = None
        self.fp8_up = None
        self.fp8_down = None

    def dequantize_weights(
        self, weights_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Dequantize FP8 weights using scale factors.

        Args:
            weights_dict: Dict mapping parameter names to FP8 weights

        Returns:
            Dict mapping parameter names to BF16 weights
        """
        result = {}
        for name, weight in weights_dict.items():
            scale_key = f"{name}_scale_inv"
            if scale_key in self.weight_dequant_scale:
                result[name] = deepseek_v3_dequantization(
                    weight, self.weight_dequant_scale[scale_key]
                )
            else:
                result[name] = weight
        return result

    def _register_fp8_weights(self):
        """Cache FP8 weights for local experts.

        Called when persistent=True - weights stay in GPU memory.
        """
        self.fp8_gate = self.module.gate_proj.weight.data
        self.fp8_up = self.module.up_proj.weight.data
        self.fp8_down = self.module.down_proj.weight.data

    def _unregister_fp8_weights(self):
        """Clear cached FP8 weights."""
        logging.debug(
            f"Clearing expert weights. Layer idx {self.layer_idx}, "
            f"expert idx {self.expert_idx}"
        )
        self.fp8_gate = None
        self.fp8_up = None
        self.fp8_down = None

    def _forward_impl(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass using deepgemm kernel.

        Args:
            hidden_states: Input tensor [num_tokens, hidden_size]

        Returns:
            Output tensor [num_tokens, hidden_size]
        """
        # Use deepgemm forward for both prefill and decode
        return self.module.deepgemm_forward(hidden_states, self.weight_dequant_scale)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass with FP8 handling.

        If persistent=True: Use cached FP8 weights directly (pre-loaded on GPU)
        If persistent=False: Load from core engine, dequantize, forward, cleanup

        Args:
            hidden_states: Input tensor [num_tokens, hidden_size]

        Returns:
            Output tensor [num_tokens, hidden_size]
        """
        rank = self.get_rank_safe()
        logging.debug(
            f"[Layer {self.layer_idx} Expert {self.expert_idx}] "
            f"Forward pass. Phase: {self.phase}"
        )

        if not self.persistent:
            # Load weights from core engine (non-persistent experts)
            weights = self.load_weights(self.module_key)

            # Apply weights to module (dequantization happens in deepgemm_forward)
            for name, param in self.module.named_parameters():
                param.data = weights[name]

        else:
            # Use cached FP8 weights (persistent experts)
            self.module.gate_proj.weight.data = self.fp8_gate
            self.module.up_proj.weight.data = self.fp8_up
            self.module.down_proj.weight.data = self.fp8_down

        # Micro-batch forward
        result = self.micro_batch_forward(hidden_states, "expert")

        # Cleanup for non-persistent experts
        if not self.persistent:
            torch.cuda.current_stream(
                self.engine_config.Basic_Config.device_torch
            ).synchronize()
            self.free_weights(self.module_key)
            self.clear_weights()

        logging.debug(
            f"[Rank {rank} Layer {self.layer_idx} Expert {self.expert_idx}] "
            f"Finish forward pass. Phase: {self.phase}"
        )

        return result


class DeepSeekAttnWrapper(AttnWrapperBase):
    """Attention wrapper with FP8 dequantization for DeepSeek-R1/V3.

    This is a placeholder - the full implementation requires porting
    the extensive prefill/decode logic from the original Wrapper.py.
    For now, it inherits the base behavior.
    """

    def __init__(
        self,
        module: nn.Module,
        layer_idx: int,
        core_engine,
        engine_config,
        model_config,
        persistent: bool = True,
        weight_dequant_scale: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """Initialize DeepSeek attention wrapper."""
        super().__init__(
            module, layer_idx, core_engine, engine_config, model_config,
            persistent, weight_dequant_scale
        )

        # FP8 weight caching
        self.fp8_q_proj = None
        self.fp8_k_proj = None
        self.fp8_v_proj = None
        self.fp8_o_proj = None

    def _register_fp8_weights(self):
        """Cache FP8 attention weights."""
        self.fp8_q_proj = self.module.q_proj.weight.data
        self.fp8_k_proj = self.module.k_proj.weight.data
        self.fp8_v_proj = self.module.v_proj.weight.data
        self.fp8_o_proj = self.module.o_proj.weight.data

    def _unregister_fp8_weights(self):
        """Clear cached FP8 attention weights."""
        self.fp8_q_proj = None
        self.fp8_k_proj = None
        self.fp8_v_proj = None
        self.fp8_o_proj = None

    def dequantize_weights(
        self, weights_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Dequantize FP8 attention weights."""
        result = {}
        for name, weight in weights_dict.items():
            scale_key = f"{name}_scale_inv"
            if scale_key in self.weight_dequant_scale:
                result[name] = deepseek_v3_dequantization(
                    weight, self.weight_dequant_scale[scale_key]
                )
            else:
                result[name] = weight
        return result

    def _forward_prefill(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        """Prefill forward - delegates to original module.

        Note: Full implementation requires porting from Wrapper.py.
        """
        # For now, use module's forward directly
        # The full implementation is in the original Attn_Wrapper
        return self.module(hidden_states, **kwargs)

    def _forward_decode(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        """Decode forward - delegates to original module.

        Note: Full implementation requires porting from Wrapper.py.
        """
        return self.module(hidden_states, **kwargs)
