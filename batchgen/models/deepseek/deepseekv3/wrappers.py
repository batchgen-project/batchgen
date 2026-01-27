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

from batchgen.models.wrappers import ExpertWrapperBase, AttnWrapperBase
from batchgen.quantization.fp8e4m3 import deepseek_v3_dequantization


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
    """Attention wrapper with FP8 DeepGEMM for DeepSeek-R1/V3.

    DeepSeek uses FP8 DeepGEMM with activation quantization:
    - Weights stay in FP8 format throughout (NO dequantization)
    - Activations are quantized to FP8 via act_quant() in w8a16_gemm
    - Uses deep_gemm.fp8_gemm_nt() for hardware-accelerated FP8 GEMM

    The base class AttnWrapperBase.forward() would call dequantize_weights(),
    which is WRONG for DeepSeek. This class overrides forward() to skip
    dequantization and use the FP8 DeepGEMM path directly.
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
        """Initialize DeepSeek attention wrapper.

        Args:
            module: Attention module (MLA) to wrap
            layer_idx: Layer index in the model
            core_engine: BatchGen core engine
            engine_config: Engine configuration
            model_config: Model configuration
            persistent: Whether weights are pre-loaded on GPU.
                        True = pre-loaded, no buffer fetch needed (default).
                        False = load from buffer each forward.
            weight_dequant_scale: Dict of FP8 inverse scale factors.
                                  These are passed to w8a16_gemm, NOT used for dequantization.
        """
        super().__init__(
            module, layer_idx, core_engine, engine_config, model_config,
            persistent, weight_dequant_scale
        )

        # FP8 weight caching for persistent mode
        self.fp8_weights_cached = {}

    def _register_fp8_weights(self):
        """Cache FP8 attention weights for persistent mode."""
        for name, param in self.module.named_parameters():
            if 'weight' in name:
                self.fp8_weights_cached[name] = param.data

    def _unregister_fp8_weights(self):
        """Clear cached FP8 attention weights."""
        self.fp8_weights_cached = {}

    def dequantize_weights(
        self, weights_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Override: DO NOT dequantize - return FP8 weights as-is.

        For DeepSeek, weights stay in FP8. The w8a16_gemm function
        handles activation quantization instead.

        Args:
            weights_dict: Dict mapping parameter names to FP8 weights

        Returns:
            Same dict unchanged (FP8 weights)
        """
        # Return weights unchanged - they stay in FP8 for DeepGEMM
        return weights_dict

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Forward pass using FP8 DeepGEMM (NO weight dequantization).

        Override base class to skip dequantization. For DeepSeek:
        - Weights stay in FP8 format
        - prefill_attn_w8a16/decode methods use w8a16_gemm
        - w8a16_gemm quantizes ACTIVATIONS to FP8, not weights

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments (hidden_states, attention_mask, etc.)

        Returns:
            Output tensor
        """
        rank = self.get_rank_safe()
        logging.debug(
            f"[Rank {rank} Layer {self.layer_idx}] "
            f"DeepSeek Attn forward. Phase: {self.phase}"
        )

        # Load FP8 weights if not persistent (NO dequantization!)
        if not self.persistent:
            weights_dict = self.load_weights(self.module_key)
            # Assign FP8 weights directly - NO dequantization for DeepSeek
            for name, param in self.module.named_parameters():
                if name in weights_dict:
                    param.data = weights_dict[name]

        # Route to appropriate phase handler
        hidden_states = kwargs.pop("hidden_states", None)
        if self.phase == "prefill":
            result = self._forward_prefill(hidden_states, **kwargs)
        else:
            result = self._forward_decode(hidden_states, **kwargs)

        # Release buffer for non-persistent attention
        if not self.persistent:
            torch.cuda.current_stream(
                self.engine_config.Basic_Config.device_torch
            ).synchronize()
            self.free_weights(self.module_key)
            self.clear_weights()

        logging.debug(
            f"[Rank {rank} Layer {self.layer_idx}] "
            f"DeepSeek Attn forward complete. Phase: {self.phase}"
        )

        return result

    def _forward_prefill(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        """Prefill forward using FP8 DeepGEMM.

        Calls prefill_attn_w8a16 which uses w8a16_gemm for each projection.
        w8a16_gemm quantizes activations to FP8, keeps weights in FP8,
        and calls deep_gemm.fp8_gemm_nt().

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            **kwargs: Additional args (attention_mask, position_ids, etc.)

        Returns:
            Attention output tensor
        """
        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        # Call prefill_attn_w8a16 which uses FP8 DeepGEMM
        # This method is dynamically attached by Parallel_Strategy_Manager
        output, offload_kv = self.module.prefill_attn_w8a16(
            hidden_states,
            attention_mask.to(hidden_states.device) if attention_mask is not None else None,
            position_ids.to(hidden_states.device) if position_ids is not None else None,
            self.weight_dequant_scale  # Inverse scales for w8a16_gemm
        )

        return output

    def _forward_decode(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        """Decode forward using FP8 DeepGEMM.

        TODO: Implement decode path using decoding_attn_mode_* methods.
        For now, delegates to module's forward (which should use decode methods).

        Args:
            hidden_states: Input tensor [batch, 1, hidden_size]
            **kwargs: Additional args

        Returns:
            Attention output tensor
        """
        # TODO: Implement proper decode path with decoding_attn_mode_3_* methods
        # For now, use module's forward directly
        return self.module(hidden_states, **kwargs)
