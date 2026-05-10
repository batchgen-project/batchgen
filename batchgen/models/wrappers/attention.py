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

"""Base attention wrapper for BatchGen execution.

Provides common functionality for attention module wrappers:
- Class-level state for batch info, attention mask, position IDs
- KV cache management
- Different handling for prefill vs decode phases
"""

import logging
from typing import Any, ClassVar, Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from .base import BaseModuleWrapper


class AttnWrapperBase(BaseModuleWrapper):
    """Base wrapper for attention modules.

    Handles:
    - Class-level batch state (cur_batch, attention_mask, position_ids)
    - Module key generation for weight loading
    - Weight dequantization hooks
    - Prefill and decode phase routing

    Class Attributes:
        phase: Current execution phase ("prefill" or "decode")
        attn_mode: Attention computation mode
        cur_batch: Current batch sequence IDs
        attention_mask: Attention mask tensor
        position_ids: Position IDs tensor
        kv_quantization_factor: KV cache quantization factors
        kv_append_callback: Callback for KV cache append
        async_kv_load_active: Flag for async KV load
        async_kv_load_task: Async KV load task object

    Subclasses should implement:
    - dequantize_weights(): Model-specific weight dequantization
    - _forward_prefill(): Prefill phase forward
    - _forward_decode(): Decode phase forward
    """

    # Class-level state (shared across all instances)
    attn_mode: ClassVar[int] = 0
    cur_batch: ClassVar[Optional[List[int]]] = None
    attention_mask: ClassVar[Optional[torch.Tensor]] = None
    position_ids: ClassVar[Optional[torch.Tensor]] = None
    kv_quantization_factor: ClassVar[Optional[List]] = None
    kv_append_callback: ClassVar[Optional[callable]] = None
    kv_append_callback_aux: ClassVar[Optional[callable]] = None
    batchgen_debug: ClassVar[Optional[Dict[str, Any]]] = None
    async_kv_load_active: ClassVar[bool] = False
    async_kv_load_task: ClassVar[Optional[object]] = None

    # Pending prefill-offload tasks. async_offload_layer_kv_to_host returns
    # KVAsyncTask futures; if Python discards them (fire-and-forget), the
    # CPU thread that queues cudaMemcpyAsync may not have run yet by the
    # time decode starts reading the host KV. Capture each task here and
    # .wait() on it before releasing the pinned source tensor references.
    pending_prefill_offload_tasks: ClassVar[list] = []
    # Tensor references kept alive for the duration of the async offload.
    # Mirrors the decode side's `_pending_kv_append_tensors`. The C++ async
    # lambda captures the tensor by value, but the underlying STORAGE may be
    # released back to PyTorch's caching allocator if no Python reference is
    # held — and with `expandable_segments:True` plus multi-seq packed prefill
    # the allocator may then hand the same physical pages to a later layer's
    # K/V tensor while the d2h memcpy is still in flight. Holding the source
    # tensors here pins the storage until the next-layer or end-of-prefill
    # retire point confirms the memcpy has completed.
    pending_prefill_offload_tensors: ClassVar[list] = []
    pending_prefill_offload_layer_idx: ClassVar[Optional[int]] = None

    @classmethod
    def _prefill_offload_sync_device(cls, device: Optional[torch.device]) -> None:
        if device is None or not torch.cuda.is_available():
            return
        sync_device = device if isinstance(device, torch.device) else torch.device(device)
        if sync_device.type == "cuda":
            torch.cuda.synchronize(sync_device)

    @classmethod
    def retire_pending_prefill_offloads(
        cls,
        *,
        device: Optional[torch.device] = None,
        reason: str = "",
    ) -> int:
        pending = cls.pending_prefill_offload_tasks
        pinned = cls.pending_prefill_offload_tensors
        if not pending and not pinned:
            cls.pending_prefill_offload_layer_idx = None
            return 0

        num_tasks = len(pending)
        for task in pending:
            task.wait()
        pending.clear()
        cls._prefill_offload_sync_device(device)
        pinned.clear()

        layer_idx = cls.pending_prefill_offload_layer_idx
        cls.pending_prefill_offload_layer_idx = None
        if num_tasks:
            suffix = f" ({reason})" if reason else ""
            logging.debug(
                f"[PREFILL_SYNC] retired {num_tasks} async KV offload tasks"
                f" from layer {layer_idx}{suffix}"
            )
        return num_tasks

    @classmethod
    def retire_pending_prefill_offloads_before_layer(
        cls,
        layer_idx: int,
        *,
        device: Optional[torch.device] = None,
    ) -> int:
        pending_layer = cls.pending_prefill_offload_layer_idx
        if pending_layer is None or pending_layer == layer_idx:
            return 0
        return cls.retire_pending_prefill_offloads(
            device=device,
            reason=f"before layer {layer_idx}",
        )

    @classmethod
    def pin_prefill_offload_tensor(cls, tensor: torch.Tensor, layer_idx: int) -> None:
        cls.pending_prefill_offload_layer_idx = layer_idx
        cls.pending_prefill_offload_tensors.append(tensor)

    @classmethod
    def track_prefill_offload_task(cls, task: object, layer_idx: int) -> None:
        cls.pending_prefill_offload_layer_idx = layer_idx
        if task is not None:
            cls.pending_prefill_offload_tasks.append(task)

    def prefix_cache_metadata(self):
        """Return validated prepack metadata for prefix-cache helpers."""
        from batchgen.attention.forward_metadata_context import (
            get_current_forward_batch_metadata,
        )
        from .prefix_cache import PrefixCachePrepackMetadata

        forward_metadata = get_current_forward_batch_metadata()
        if forward_metadata is not None:
            return PrefixCachePrepackMetadata.from_forward_metadata(forward_metadata)
        return PrefixCachePrepackMetadata.from_wrapper_cls(type(self))

    def host_prefix_reader(self):
        """Return a host prefix-cache page reader for this layer."""
        from .prefix_cache import HostPrefixPageReader

        return HostPrefixPageReader(
            core_engine=self.core_engine,
            engine_config=self.engine_config,
            layer_idx=self.layer_idx,
        )

    def prefix_attention_kv_builder(self):
        """Return a prefix-cache KV builder for this layer."""
        from .prefix_cache import PrefixAttentionKvBuilder

        return PrefixAttentionKvBuilder(self.host_prefix_reader())

    def offload_prepacked_gqa_kv(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        metadata=None,
        track_tasks: bool = False,
        sequence_callback=None,
    ) -> None:
        """Offload prepacked GQA KV with optional prefix-cache offsets."""
        from .prefix_cache import PrefixAwarePrefillOffloader

        metadata = metadata or self.prefix_cache_metadata()
        tracker = self.track_prefill_offload_task if track_tasks else None
        tensor_pinner = self.pin_prefill_offload_tensor if track_tasks else None
        offloader = PrefixAwarePrefillOffloader(
            worker_view=getattr(self.core_engine, "host_paged_kv_worker_view", None),
            layer_idx=self.layer_idx,
            metadata=metadata,
            track_task=tracker,
            pin_tensor=tensor_pinner,
        )
        offloader.offload_gqa(
            key=key,
            value=value,
            sequence_callback=sequence_callback,
        )

    def offload_prepacked_mla_kv(
        self,
        key: torch.Tensor,
        *,
        metadata=None,
        track_tasks: bool = False,
    ) -> None:
        """Offload prepacked MLA primary KV with optional prefix-cache offsets."""
        from .prefix_cache import PrefixAwarePrefillOffloader

        metadata = metadata or self.prefix_cache_metadata()
        tracker = self.track_prefill_offload_task if track_tasks else None
        tensor_pinner = self.pin_prefill_offload_tensor if track_tasks else None
        offloader = PrefixAwarePrefillOffloader(
            worker_view=getattr(self.core_engine, "host_paged_kv_worker_view", None),
            layer_idx=self.layer_idx,
            metadata=metadata,
            track_task=tracker,
            pin_tensor=tensor_pinner,
        )
        offloader.offload_mla(key=key)

    # Prepack mode state
    prepack_mode: ClassVar[bool] = False
    prepack_cu_seqlens: ClassVar[Optional[torch.Tensor]] = None
    prepack_max_seqlen: ClassVar[Optional[int]] = None
    prepack_num_sequences: ClassVar[Optional[int]] = None
    prepack_seq_lengths: ClassVar[Optional[List[int]]] = None
    prepack_prefix_reuse_mode: ClassVar[bool] = False
    prepack_prefix_shared_tokens: ClassVar[Optional[List[int]]] = None
    prepack_full_seq_lengths: ClassVar[Optional[List[int]]] = None
    prepack_full_hit_mode: ClassVar[bool] = False

    # KV cache state
    past_key_states: ClassVar[Optional[List[torch.Tensor]]] = None
    past_value_states: ClassVar[Optional[List[torch.Tensor]]] = None
    scale: ClassVar[Optional[List[torch.Tensor]]] = None
    cache_seqlens: ClassVar[Optional[torch.Tensor]] = None
    max_seqlen: ClassVar[Optional[int]] = None
    # DSA per-step dispatch hint: count of rows with cache_seqlen <= index_topk.
    # Set once per decode step by the worker so per-layer _forward_decode_dsa
    # branches on it without doing a D2H .sum().item() 78 times per step.
    _dsa_short_count: ClassVar[Optional[int]] = None
    # Whole-model CUDA graph can pad local rows to a global NCCL bucket. These
    # graph-owned overrides let GLM-5 DSA use explicit slot sentinels for padded
    # rows instead of deriving slot count from cur_batch.
    glm5_decode_primary_slot_indices: ClassVar[Optional[torch.Tensor]] = None
    glm5_decode_aux_slot_indices: ClassVar[Optional[torch.Tensor]] = None
    glm5_dsa_flashmla_graph_metadata: ClassVar[Optional[Dict[str, Any]]] = None
    gpu_paged_kv_manager: ClassVar[Optional[object]] = None
    host_paged_kv_worker_view: ClassVar[Optional[object]] = None
    prefill_prefix_materialization: ClassVar[Optional[object]] = None
    # DSA auxiliary caches (indexer KV for DeepSeek Sparse Attention)
    gpu_paged_kv_manager_aux: ClassVar[Optional[object]] = None
    host_paged_kv_worker_view_aux: ClassVar[Optional[object]] = None

    # Execution phase
    phase: ClassVar[str] = "prefill"

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
        """Initialize attention wrapper.

        Args:
            module: Attention module to wrap
            layer_idx: Layer index in the model
            core_engine: BatchGen core engine
            engine_config: Engine configuration
            model_config: Model configuration
            persistent: Whether weights are pre-loaded on GPU.
                        True = pre-loaded, no buffer fetch needed (default for attention).
                        False = load from buffer each forward.
            weight_dequant_scale: Dict of weight dequantization scales
        """
        super().__init__(module, layer_idx, core_engine, engine_config, model_config)
        self.persistent = persistent
        self.weight_dequant_scale = weight_dequant_scale or {}
        self.module_key = f"attn_{layer_idx}"

    @classmethod
    def _to_global_sequence_id(cls, local_sequence_id: int) -> int:
        """Convert local sequence ID to global ID.

        Global ID = (rank << 32) | local_id

        Args:
            local_sequence_id: Local sequence ID

        Returns:
            Global sequence ID
        """
        rank = cls.get_rank_safe()
        return (rank << 32) | (int(local_sequence_id) & 0xFFFFFFFF)

    @classmethod
    def _build_global_sequence_ids(cls, sequence_ids: Sequence[int]) -> List[int]:
        """Convert list of local sequence IDs to global IDs.

        Args:
            sequence_ids: List of local sequence IDs

        Returns:
            List of global sequence IDs
        """
        return [cls._to_global_sequence_id(seq_id) for seq_id in sequence_ids]

    def dequantize_weights(
        self, weights_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Dequantize loaded weights.

        Override in subclass for model-specific dequantization.

        Args:
            weights_dict: Dict mapping parameter names to quantized weights

        Returns:
            Dict mapping parameter names to dequantized weights
        """
        return weights_dict

    def _forward_prefill(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass for prefill phase.

        Override in subclass for model-specific prefill logic.

        Args:
            hidden_states: Input tensor
            **kwargs: Additional arguments (attention_mask, position_ids, etc.)

        Returns:
            Output tensor
        """
        raise NotImplementedError("Subclass must implement _forward_prefill")

    def _forward_decode(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass for decode phase.

        Override in subclass for model-specific decode logic.

        Args:
            hidden_states: Input tensor
            **kwargs: Additional arguments

        Returns:
            Output tensor
        """
        raise NotImplementedError("Subclass must implement _forward_decode")

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Forward pass with phase routing.

        Routes to _forward_prefill or _forward_decode based on current phase.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments (hidden_states, attention_mask, etc.)

        Returns:
            Output tensor and optional attention weights/cache
        """
        rank = self.get_rank_safe()
        logging.debug(
            f"[Rank {rank} Layer {self.layer_idx}] "
            f"Attn forward. Phase: {self.phase}"
        )

        # Load weights if not persistent (non-persistent attention modules)
        if not self.persistent:
            weights = self.load_weights(self.module_key)
            dequant_weights = self.dequantize_weights(weights)
            self.apply_weights(dequant_weights)

        # Route to appropriate phase handler
        # Extract hidden_states to avoid passing it twice (positionally and in kwargs)
        hidden_states = kwargs.pop("hidden_states", None)
        if self.phase == "prefill":
            result = self._forward_prefill(hidden_states, **kwargs)
        else:
            result = self._forward_decode(hidden_states, **kwargs)

        # Release buffer for non-persistent attention so the H2D worker can
        # load the next layer's weights.
        if not self.persistent:
            torch.cuda.current_stream().synchronize()
            self.free_weights(self.module_key)
            self.clear_weights()

        logging.debug(
            f"[Rank {rank} Layer {self.layer_idx}] "
            f"Attn forward complete. Phase: {self.phase}"
        )

        return result

    # FP8 weight caching methods (for models that use it)
    def _register_fp8_weights(self):
        """Cache FP8 weights for local attention (no loading needed).

        Override in subclass if FP8 caching is needed.
        """
        pass

    def _unregister_fp8_weights(self):
        """Clear cached FP8 weights.

        Override in subclass if FP8 caching is needed.
        """
        pass
