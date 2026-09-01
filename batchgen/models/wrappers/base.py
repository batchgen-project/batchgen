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

"""Base module wrapper for BatchGen execution.

Provides common functionality for all module wrappers:
- Safe distributed operations (handles world_size == 1)
- Weight loading from core engine
- Micro-batching support
"""

import logging
import math
from typing import ClassVar, Dict, Optional

import torch
import torch.nn as nn
import torch.distributed as dist


class BaseModuleWrapper(nn.Module):
    """Base wrapper for BatchGen module execution.

    Provides common functionality for attention and expert wrappers:
    - Safe rank/world_size queries (handles world_size == 1 without dist init)
    - Weight loading from core engine
    - Weight cleanup after forward pass
    - Micro-batching to avoid OOM

    Class Attributes:
        phase: Current execution phase ("prefill" or "decode")

    Instance Attributes:
        module: The wrapped PyTorch module
        layer_idx: Layer index in the model
        core_engine: BatchGen core engine for weight management
        engine_config: Engine configuration
        model_config: Model configuration
    """

    # Class-level phase tracking (shared across all wrapper instances)
    phase: ClassVar[str] = "prefill"

    # Cached rank/world_size to avoid repeated dist calls
    _cached_rank: ClassVar[Optional[int]] = None
    _cached_world_size: ClassVar[Optional[int]] = None

    def __init__(
        self,
        module: nn.Module,
        layer_idx: int,
        core_engine,
        engine_config,
        model_config,
    ):
        """Initialize the wrapper.

        Args:
            module: The PyTorch module to wrap
            layer_idx: Layer index in the model
            core_engine: BatchGen core engine for weight management
            engine_config: Engine configuration
            model_config: Model configuration
        """
        super().__init__()
        self.module = module
        self.layer_idx = layer_idx
        self.core_engine = core_engine
        self.engine_config = engine_config
        self.model_config = model_config

    @classmethod
    def get_rank_safe(cls) -> int:
        """Get rank with proper guard for world_size == 1.

        Returns 0 if distributed is not initialized, otherwise returns
        the actual rank. Caches the result for performance.

        Returns:
            int: Current process rank (0 if not distributed)
        """
        if cls._cached_rank is None:
            if not dist.is_initialized():
                cls._cached_rank = 0
            else:
                cls._cached_rank = dist.get_rank()
        return cls._cached_rank

    @classmethod
    def get_world_size_safe(cls) -> int:
        """Get world size with proper guard for world_size == 1.

        Returns 1 if distributed is not initialized, otherwise returns
        the actual world size. Caches the result for performance.

        Returns:
            int: World size (1 if not distributed)
        """
        if cls._cached_world_size is None:
            if not dist.is_initialized():
                cls._cached_world_size = 1
            else:
                cls._cached_world_size = dist.get_world_size()
        return cls._cached_world_size

    @classmethod
    def is_distributed(cls) -> bool:
        """Check if running in distributed mode.

        Returns:
            bool: True if distributed is initialized and world_size > 1
        """
        return dist.is_initialized() and cls.get_world_size_safe() > 1

    @classmethod
    def clear_cache(cls):
        """Clear cached rank/world_size values.

        Call this if distributed setup changes during runtime.
        """
        cls._cached_rank = None
        cls._cached_world_size = None

    def load_weights(self, module_key: str) -> Dict[str, torch.Tensor]:
        """Load weights from core engine.

        Args:
            module_key: Key identifying the module weights to load

        Returns:
            Dict mapping parameter names to weight tensors
        """
        return self.core_engine.get_weights(module_key, self.phase)

    def _sync_device_before_release(self) -> None:
        """Drain the current compute stream before releasing weight storage."""
        torch.cuda.current_stream().synchronize()

    def free_weights(self, module_key: str):
        """Free weights buffer after use.

        Args:
            module_key: Key identifying the module weights to free
        """
        self._sync_device_before_release()
        self.core_engine.free_weights_buffer(module_key)

    def apply_weights(self, weights_dict: Dict[str, torch.Tensor]):
        """Apply loaded weights to module parameters.

        Args:
            weights_dict: Dict mapping parameter names to weight tensors
        """
        applied = set()
        for name, param in self.module.named_parameters():
            if name in weights_dict:
                param.data = weights_dict[name]
                applied.add(name)
        # Track which parameter names we populated so clear_weights only wipes
        # buffer-backed ones. Skeleton-loaded params (q_a/kv_a_layernorm,
        # indexer.*) are never in weights_dict; wiping them would leave the
        # module with empty(0) params and fail the next forward.
        self._applied_param_keys = applied

    def clear_weights(self):
        """Clear only the buffer-loaded module parameters populated by the
        most recent apply_weights call; preserve skeleton-loaded params.
        """
        self._sync_device_before_release()
        self.clear_weight_bindings()

    def clear_weight_bindings(self):
        """Drop parameter views without synchronizing their backing storage.

        The caller must first transfer buffer ownership to an asynchronous
        release primitive that keeps the storage alive until the current CUDA
        stream reaches its recorded completion event.
        """
        applied = getattr(self, "_applied_param_keys", None)
        for name, param in self.module.named_parameters():
            if applied is not None and name not in applied:
                continue
            param.data = torch.empty(0, device=param.data.device)
        self._applied_param_keys = None

    def get_batch_size(self, batch_size_key: str) -> int:
        """Get micro-batch size from config.

        Args:
            batch_size_key: Key prefix for batch size config
                           (e.g., "expert" or "attn")

        Returns:
            int: Batch size for current phase
        """
        if self.phase == "prefill":
            attr_name = f"{batch_size_key}_prefill_batch_size_upper_bound"
        else:
            attr_name = f"{batch_size_key}_decoding_batch_size_upper_bound"

        return getattr(self.engine_config.Module_Batching_Config, attr_name)

    def micro_batch_forward(
        self,
        hidden_states: torch.Tensor,
        batch_size_key: str,
    ) -> torch.Tensor:
        """Forward pass with micro-batching to avoid OOM.

        Splits input into smaller batches if needed based on config.

        Args:
            hidden_states: Input tensor [num_tokens, hidden_size]
            batch_size_key: Key prefix for batch size config

        Returns:
            Output tensor with same shape as input
        """
        batch_size = self.get_batch_size(batch_size_key)
        num_tokens = hidden_states.size(0)

        # If input fits in one batch, just run forward
        if num_tokens <= batch_size:
            return self._forward_impl(hidden_states)

        # Split into micro-batches
        result = torch.empty_like(hidden_states)
        num_batches = math.ceil(num_tokens / batch_size)

        for i in range(num_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, num_tokens)
            result[start:end] = self._forward_impl(hidden_states[start:end])

        return result

    def _forward_impl(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Subclass implements actual forward logic.

        Args:
            hidden_states: Input tensor for one micro-batch

        Returns:
            Output tensor

        Raises:
            NotImplementedError: Subclasses must implement this
        """
        raise NotImplementedError("Subclasses must implement _forward_impl")

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Forward pass through the wrapped module.

        Subclasses should override this to implement model-specific
        weight loading, dequantization, and forward logic.

        Raises:
            NotImplementedError: Subclasses must implement this
        """
        raise NotImplementedError("Subclasses must implement forward")
