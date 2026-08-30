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
import time
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn as nn

from .base import BaseModuleWrapper


class AttnWrapperBase(BaseModuleWrapper):
    """Base wrapper for attention modules.

    The class-level attributes below form the **per-step contract** between the
    worker and the model: the worker (batchgen_worker.py) writes them right
    before calling model.forward, and the wrapper reads them inside
    _forward_prefill / _forward_decode. They are GLOBAL MUTABLE STATE, valid
    only for the current micro-batch/step. Full semantics + which worker line
    writes each field are documented in:
        batchgen-context/architecture/PSM_WORKER_CONTRACT.md  (§2)

    Handles:
    - Class-level batch state (cur_batch, attention_mask, position_ids)
    - Module key generation for weight loading
    - Weight dequantization hooks
    - Prefill and decode phase routing

    Class Attributes (grouped by when they are valid):
      Always (routing):
        phase           -- "prefill" | "decode"; selects the forward branch.
        cur_batch       -- list[int] global_seq_ids in this step's order; used
                           to index paged KV pages / model state pools.
        position_ids    -- token positions (flat in prepack; (bsz,1) in decode).
      Prefill, standard padded path:
        attention_mask  -- (bsz, seq_len) 1/0 padding mask. NOTE: the model's
                           _run_attn passes None for linear/KDA layers; those
                           read lengths here or from prepack_cu_seqlens.
      Prefill, prepack (varlen) path [default, --enable-prepack]:
        prepack_mode        -- bool; gate model code on this.
        prepack_cu_seqlens  -- (num_seq+1,) int32 cumulative token offsets.
        prepack_max_seqlen  -- int; longest sequence.
        prepack_num_sequences -- int.
        prepack_seq_lengths -- list[int] per-sequence lengths.
        host_paged_kv_worker_view (on core_engine) -- target for
                           async_offload_layer_kv_to_host(...) KV offload.
        pending_prefill_offload_tasks/_tensors -- async-offload bookkeeping
                           (worker waits on these; do not clear from model code).
      Decode path:
        attention_mask  -- None (unused in decode).
        cache_seqlens   -- (bsz,) int32 current context length per sequence.
        max_seqlen      -- int max context length in batch.
        gpu_paged_kv_manager (+ _aux) -- paged-KV read/write.
        kv_append_callback  (+ _aux)  -- fn(layer_idx, k, v) to append new KV.
      KV / misc:
        attn_mode, scale, past_key_states, past_value_states,
        kv_quantization_factor, batchgen_debug,
        async_kv_load_active / async_kv_load_task.

    Distributed: model MoE-EP collectives must use the worker's
    PyNcclCommunicator (bound onto modules by the PSM), NOT torch.distributed
    (deadlocks vs the engine's C++ NCCL). See PSM_WORKER_CONTRACT.md §3.

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
    # Phase C: glm5_dispatch_trace_* moved to GLM5AttnWrapper subclass
    # (batchgen/models/glm/glm5/wrappers.py) per audit §A finding #8 — these
    # are GLM-5-specific debug instrumentation and don't belong on the
    # generic base wrapper.
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
    # Optional runtime qualification ledger. GLM-5 enables it immediately
    # before prefill and consumes it after the final layer's offload retires.
    prefill_offload_retirements: ClassVar[Optional[list]] = None

    @classmethod
    def record_glm5_dispatch(
        cls,
        *,
        kind: str,
        path: str,
        layer_idx: int,
        bsz: int,
        reason: str,
    ) -> None:
        # Phase C: the underlying ClassVars (glm5_dispatch_trace_*) live on
        # GLM5AttnWrapper, not AttnWrapperBase. Read through it directly so
        # callers can keep calling AttnWrapperBase.record_glm5_dispatch(...).
        # Lazy-import the subclass to avoid a circular import at module load.
        from batchgen.models.glm.glm5.wrappers import GLM5AttnWrapper as _G
        if not _G.glm5_dispatch_trace_enabled:
            return
        counts = _G.glm5_dispatch_counts
        counter_key = f"{kind}_{path}"
        counts[counter_key] = counts.get(counter_key, 0) + 1

        trace_id = _G.glm5_dispatch_trace_id or "unknown"
        seen_key = (trace_id, kind, path, int(bsz))
        if seen_key in _G.glm5_dispatch_seen:
            return
        _G.glm5_dispatch_seen.add(seen_key)

        context = _G.glm5_dispatch_trace_context or {}
        logging.warning(
            "[GLM5_DISPATCH_TRACE] rank=%s trace=%s kind=%s path=%s "
            "layer=%s bsz=%s batch_ids=%s global_ids=%s debug_dsa=%s "
            "debug_moe=%s debug_moe_router=%s reason=%s count=%s",
            context.get("rank", "?"),
            trace_id,
            kind,
            path,
            layer_idx,
            bsz,
            context.get("batch_ids", "-"),
            context.get("global_ids", "-"),
            context.get("glm5_dsa_mode", "-"),
            context.get("glm5_moe_mode", "-"),
            context.get("glm5_moe_router_mode", "-"),
            reason,
            counts[counter_key],
        )

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
        wait_t0 = time.perf_counter()
        for task in pending:
            task.wait()
        wait_ms = (time.perf_counter() - wait_t0) * 1000.0
        try:
            from batchgen.timing import get_prefill_timer
            timer = get_prefill_timer()
            if timer is not None and timer.enabled:
                timer.record(
                    "host:kv_offload_retire_wait",
                    cls.pending_prefill_offload_layer_idx
                    if cls.pending_prefill_offload_layer_idx is not None else -1,
                    wait_ms,
                )
        except ImportError:
            pass
        pending.clear()
        # KVAsyncTask completes only after the dedicated D2H stream records and
        # synchronizes its completion event.  Waiting every task therefore
        # proves both producer ordering and copy completion; a device-wide
        # synchronization here only drains unrelated work.
        pinned.clear()

        layer_idx = cls.pending_prefill_offload_layer_idx
        cls.pending_prefill_offload_layer_idx = None
        if num_tasks and cls.prefill_offload_retirements is not None:
            cls.prefill_offload_retirements.append({
                "layer_idx": layer_idx,
                "tasks": num_tasks,
            })
        if num_tasks:
            suffix = f" ({reason})" if reason else ""
            logging.debug(
                f"[PREFILL_SYNC] retired {num_tasks} async KV offload tasks"
                f" from layer {layer_idx}{suffix}"
            )
        return num_tasks

    @classmethod
    def start_prefill_offload_retirement_audit(cls) -> None:
        if cls.prefill_offload_retirements is not None:
            raise RuntimeError("Prefill KV offload retirement audit is already active")
        if (
            cls.pending_prefill_offload_tasks
            or cls.pending_prefill_offload_tensors
            or cls.pending_prefill_offload_layer_idx is not None
        ):
            raise RuntimeError(
                "Cannot start prefill KV offload retirement audit with pending work"
            )
        cls.prefill_offload_retirements = []

    @classmethod
    def finish_prefill_offload_retirement_audit(cls) -> list:
        retirements = cls.prefill_offload_retirements
        cls.prefill_offload_retirements = None
        if retirements is None:
            raise RuntimeError("Prefill KV offload retirement audit was not active")
        return list(retirements)

    @classmethod
    def abort_prefill_offload_retirement_audit(
        cls,
        *,
        device: Optional[torch.device] = None,
    ) -> None:
        """Drain issued D2H work and reset all prefill-offload bookkeeping."""
        first_error = None
        try:
            for task in list(cls.pending_prefill_offload_tasks):
                try:
                    task.wait()
                except BaseException as error:
                    if first_error is None:
                        first_error = error
            try:
                cls._prefill_offload_sync_device(device)
            except BaseException as error:
                if first_error is None:
                    first_error = error
        finally:
            cls.pending_prefill_offload_tasks.clear()
            cls.pending_prefill_offload_tensors.clear()
            cls.pending_prefill_offload_layer_idx = None
            cls.prefill_offload_retirements = None
        if first_error is not None:
            raise RuntimeError(
                "Failed to drain GLM-5 prefill KV offload work during abort"
            ) from first_error

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

    # Prepack mode state
    prepack_mode: ClassVar[bool] = False
    prepack_cu_seqlens: ClassVar[Optional[torch.Tensor]] = None
    prepack_max_seqlen: ClassVar[Optional[int]] = None
    prepack_num_sequences: ClassVar[Optional[int]] = None
    prepack_seq_lengths: ClassVar[Optional[List[int]]] = None

    # KV cache state
    past_key_states: ClassVar[Optional[List[torch.Tensor]]] = None
    past_value_states: ClassVar[Optional[List[torch.Tensor]]] = None
    scale: ClassVar[Optional[List[torch.Tensor]]] = None
    cache_seqlens: ClassVar[Optional[torch.Tensor]] = None
    max_seqlen: ClassVar[Optional[int]] = None
    # Phase C: _dsa_short_count and glm5_decode_*_slot_indices /
    # glm5_dsa_graph_forward_state / glm5_dsa_flashmla_graph_metadata
    # moved to GLM5AttnWrapper subclass (batchgen/models/glm/glm5/wrappers.py)
    # per audit §A finding #8.
    gpu_paged_kv_manager: ClassVar[Optional[object]] = None
    host_paged_kv_worker_view: ClassVar[Optional[object]] = None
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
        prefill_timer = None
        if self.phase == "prefill":
            try:
                from batchgen.timing import get_prefill_timer
                prefill_timer = get_prefill_timer()
            except ImportError:
                pass

        if not self.persistent:
            if prefill_timer is not None:
                with prefill_timer.host_timed(
                    "attn_weight_acquire_bind", self.layer_idx
                ):
                    weights = self.load_weights(self.module_key)
                    dequant_weights = self.dequantize_weights(weights)
                    self.apply_weights(dequant_weights)
            else:
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
            async_release = (
                self.phase == "prefill"
                and hasattr(self.core_engine, "free_weights_buffer_async")
            )
            if prefill_timer is not None:
                with prefill_timer.host_timed(
                    "attn_weight_release", self.layer_idx
                ):
                    if async_release:
                        self.core_engine.free_weights_buffer_async(self.module_key)
                        self.clear_weight_bindings()
                    else:
                        torch.cuda.current_stream().synchronize()
                        self.free_weights(self.module_key)
                        self.clear_weights()
            else:
                if async_release:
                    self.core_engine.free_weights_buffer_async(self.module_key)
                    self.clear_weight_bindings()
                else:
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
