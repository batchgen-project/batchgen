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

"""Base expert wrapper for BatchGen MoE execution.

Provides common functionality for expert module wrappers:
- Module key generation for weight loading
- Dequantization hooks (override in subclass)
- Standard forward flow with weight load/apply/cleanup
"""

import logging
from typing import ClassVar, Dict, Optional

import torch
import torch.nn as nn

from .base import BaseModuleWrapper


class ExpertWrapperBase(BaseModuleWrapper):
    """Base wrapper for expert modules in MoE layers.

    Handles:
    - Module key generation (routed_expert_{layer}_{expert} or shared_expert_{layer})
    - Weight loading from core engine
    - Dequantization hooks (subclass implements model-specific logic)
    - Forward pass with micro-batching

    Subclasses should implement:
    - dequantize_weights(): Model-specific dequantization (FP8, MXFP4, etc.)
    - _forward_impl(): Actual forward computation

    Attributes:
        expert_idx: Expert index within the layer (-1 for shared expert)
        module_key: Key for weight loading from core engine
        persistent: Whether weights are pre-loaded on GPU (no buffer fetch needed)
    """

    # Execution phase (shared across all instances)
    phase: ClassVar[str] = "prefill"

    def __init__(
        self,
        module: nn.Module,
        layer_idx: int,
        expert_idx: int,
        core_engine,
        engine_config,
        model_config,
        persistent: bool = False,
    ):
        """Initialize expert wrapper.

        Args:
            module: Expert FFN module to wrap
            layer_idx: Layer index in the model
            expert_idx: Expert index within the layer (-1 for shared expert)
            core_engine: BatchGen core engine for weight management
            engine_config: Engine configuration
            model_config: Model configuration
            persistent: Whether weights are pre-loaded on GPU.
                        True = pre-loaded, no buffer fetch needed.
                        False = load from buffer each forward.
        """
        super().__init__(module, layer_idx, core_engine, engine_config, model_config)
        self.expert_idx = expert_idx
        self.persistent = persistent
        self.module_key = self._build_module_key()

    def _build_module_key(self) -> str:
        """Build module key for weight loading.

        Returns:
            str: Key like "routed_expert_0_5" or "shared_expert_0"
        """
        if self.expert_idx == -1:
            return f"shared_expert_{self.layer_idx}"
        return f"routed_expert_{self.layer_idx}_{self.expert_idx}"

    def dequantize_weights(
        self, weights_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Dequantize loaded weights.

        Override in subclass for model-specific dequantization logic
        (e.g., FP8 for DeepSeek, MXFP4 for GPT-OSS).

        Args:
            weights_dict: Dict mapping parameter names to quantized weights

        Returns:
            Dict mapping parameter names to dequantized weights
        """
        # Default: no dequantization needed
        return weights_dict

    def _forward_impl(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass through expert module.

        Override in subclass if needed for custom forward logic.

        Args:
            hidden_states: Input tensor [num_tokens, hidden_size]

        Returns:
            Output tensor [num_tokens, hidden_size]
        """
        return self.module(hidden_states)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass with weight loading and cleanup.

        Standard flow:
        1. Load weights from core engine (if get_weights=True)
        2. Dequantize weights
        3. Apply weights to module
        4. Run micro-batched forward
        5. Cleanup weights buffer

        Args:
            hidden_states: Input tensor [num_tokens, hidden_size]

        Returns:
            Output tensor [num_tokens, hidden_size]
        """
        rank = self.get_rank_safe()
        logging.debug(
            f"[Rank {rank} Layer {self.layer_idx} Expert {self.expert_idx}] "
            f"Forward pass. Phase: {self.phase}"
        )

        if not self.persistent:
            # Load weights from core engine (non-persistent experts)
            weights = self.load_weights(self.module_key)

            # Dequantize (subclass may override)
            dequant_weights = self.dequantize_weights(weights)

            # Apply to module
            self.apply_weights(dequant_weights)

        # Micro-batch forward
        result = self.micro_batch_forward(hidden_states, "expert")

        if not self.persistent:
            torch.cuda.current_stream().synchronize()
            self.free_weights(self.module_key)
            self.clear_weights()

        logging.debug(
            f"[Rank {rank} Layer {self.layer_idx} Expert {self.expert_idx}] "
            f"Finish forward pass. Phase: {self.phase}"
        )

        return result
