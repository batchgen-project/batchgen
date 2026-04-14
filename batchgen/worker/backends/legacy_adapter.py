"""Production adapter that wraps a legacy `BatchGenWorker` instance.

Satisfies :class:`batchgen.worker.protocols.LegacyInfraBackend`. The
`batchgen/worker/` handlers talk to this adapter instead of reaching
directly into the `BatchGenWorker` instance. That way, the adapter
surface becomes the explicit contract between the new orchestrator
package and the legacy worker — any method not listed on the adapter
is off-limits.

This is Phase-F1 of the "Eliminate Legacy Callbacks" plan: adapter
exists, orchestrator takes it as a new constructor arg, but no delegate
has been removed yet. Subsequent phases (F2–F10) port control-flow
methods from `batchgen_worker.py` into the `batchgen/worker/` package,
each calling `self._adapter.<method>` for the infrastructure bits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

    from batchgen.batchgen_worker import BatchGenWorker


class LegacyWorkerBackend:
    """Wraps a `BatchGenWorker` instance, exposing only the infrastructure
    surface defined by :class:`LegacyInfraBackend`.

    Usage (production):

        worker = BatchGenWorker(...)  # fully initialized
        adapter = LegacyWorkerBackend(worker)
        orchestrator = WorkerOrchestrator(..., legacy_infra=adapter)
    """

    def __init__(self, worker: "BatchGenWorker") -> None:
        self._w = worker
        # F5 setup flags. PyNccl init runs exactly once ever.
        # `_decode_setup_done` tracks "is the decode model + GPU KV
        # ready right now?" — it is cleared by
        # `prefill_flush_and_reconfigure` because that helper frees
        # the decode model and destroys the GPU KV cache.
        # Phase 2.7: symmetric `_prefill_setup_done` tracks "is the
        # prefill-configured model loaded right now?" — cleared by
        # `decode_setup_once` (which overwrites the parallel_manager
        # config with decode weights) and set by
        # `prefill_flush_and_reconfigure` (the expensive transition).
        self._pynccl_initialized: bool = False
        self._decode_setup_done: bool = False
        self._prefill_setup_done: bool = False

    # --- engine config passthrough (Phase 2.5 capacity-aware prepare_batch) ---
    @property
    def engine_config(self) -> Any:
        return getattr(self._w, "engine_config", None)

    # --- rank / topology ---
    @property
    def rank(self) -> int:
        return self._w.rank

    @property
    def local_rank(self) -> int:
        return self._w.local_rank

    @property
    def world_size(self) -> int:
        return self._w.world_size

    # --- model lifecycle ---
    def configure_prefill_model(self) -> Any:
        return self._w.parallel_manager.configure_prefill()

    def configure_decode_model(self, max_num_seq: int, comm: Any) -> Any:
        return self._w.parallel_manager.configure_decoding(
            padding_bsz=max_num_seq, comm=comm
        )

    def deep_free_model_memory(self) -> None:
        self._w.deep_free_model_memory()

    def init_nvshmem(self) -> None:
        self._w.init_nvshmem()

    def set_phase(self, phase: str) -> None:
        self._w.set_phase(phase)

    def destroy_gpu_paged_kv_cache(self) -> None:
        self._w._destroy_gpu_paged_kv_cache()

    # --- KV cache primitives ---
    def release_gpu_kv_pages(self, local_indices: list[int]) -> None:
        self._w._release_gpu_kv_pages(local_indices)

    def release_host_kv_pages_for_batch(self, uuids: list[str]) -> None:
        self._w._release_host_kv_pages_for_batch(uuids)

    def extend_gpu_kv_allocation(self, uuids: list[str]) -> bool:
        return self._w._extend_gpu_kv_allocation(uuids)

    def allocate_gpu_kv_two_page_buffer(
        self, local_indices: list[int], load_from_host: bool
    ) -> bool:
        return self._w._allocate_gpu_kv_two_page_buffer(
            local_indices, load_from_host=load_from_host
        )

    def flush_deferred_kv_to_host(self) -> None:
        self._w._flush_deferred_kv_to_host()

    def wait_pending_kv_append_tasks(self) -> int:
        return self._w._wait_pending_kv_append_tasks()

    def rebuild_page_table_for_batch(
        self, batch: list[int], gpu_manager: Any
    ) -> None:
        self._w._rebuild_page_table_for_batch(batch, gpu_manager)

    def finalize_async_load_minimal(self, *args: Any, **kwargs: Any) -> Any:
        return self._w._finalize_async_load_minimal(*args, **kwargs)

    def check_host_kv_watermark_trigger(self) -> bool:
        return self._w._check_host_kv_watermark_trigger()

    def get_effective_chunk_size(self) -> int:
        return self._w._get_effective_chunk_size()

    def put_sequences_on_hold(self, uuids: list[str]) -> None:
        self._w._put_sequences_on_hold(uuids)

    # --- boundary Stage 1 passthroughs ---
    def set_num_tokens_per_rank(self, n: int) -> None:
        self._w.parallel_manager.set_num_tokens_per_rank(n)

    def set_rank_token_counts(self, counts: "torch.Tensor") -> None:
        self._w.parallel_manager.set_rank_token_counts(counts)

    def host_paged_kv_worker_view(self) -> Any:
        return getattr(self._w.core_engine, "host_paged_kv_worker_view", None)

    def report_chunk_sizer_completion(self, decoded_length: int) -> None:
        sizer = getattr(self._w, "adaptive_chunk_sizer", None)
        if sizer is not None:
            sizer.report_completion(decoded_length)

    # --- decode Stage 2 passthroughs ---
    def bind_decode_context(
        self,
        *,
        batch: list[int],
        past_key_states: Any,
        past_value_states: Any,
        scale_dict: dict | None,
    ) -> tuple[Any, Any]:
        return self._w._decode_bind_attn_wrapper(
            batch, past_key_states, past_value_states, scale_dict,
        )

    def forward_decode_step(
        self,
        *,
        batch: list[int],
        new_tokens: "torch.Tensor",
        gpu_manager: Any,
        page_table_verified: bool,
        local_iteration: int,
    ) -> "torch.Tensor":
        return self._w._decode_forward_step(
            batch, new_tokens, gpu_manager,
            page_table_verified, local_iteration,
        )

    # --- index / UUID mapping ---
    def local_indices_to_global_seq_ids(self, batch: list[int]) -> list[int]:
        return self._w._local_indices_to_global_seq_ids(batch)

    def get_local_indices_for_uuids(self, uuids: list[str]) -> list[int]:
        return self._w._get_local_indices_for_uuids(uuids)

    def uuid_to_local_map(self) -> dict[str, int]:
        return self._w._uuid_to_local_map

    def local_to_uuid_map(self) -> dict[int, str]:
        return self._w._local_to_uuid_map

    def sequences_with_gpu_kv(self) -> set[str]:
        return self._w._sequences_with_gpu_kv

    # --- sampling / IO ---
    def select_tokens(self, logits: "torch.Tensor") -> "torch.Tensor":
        return self._w._select_tokens(logits)

    def should_stop_at_eos(self, token_id: int) -> bool:
        return self._w._should_stop_at_eos(token_id)

    def rebuild_input_tokens(self, batch: list[int]) -> "torch.Tensor":
        return self._w._rebuild_input_tokens(batch)

    def decode_tokens_to_string(self, tokens: "torch.Tensor") -> str:
        return self._w._decode_tokens_to_string(tokens)

    def report_completion(self, uuid: str, gathered_text: str | None) -> None:
        self._w._report_completion(uuid, gathered_text=gathered_text)

    def gather_completed_tokens(self, uuids: list[str]) -> dict[str, str]:
        return self._w._gather_completed_tokens(uuids)

    def submit_completed_to_incremental_writer(self, uuids: list[str]) -> None:
        self._w._submit_completed_to_incremental_writer(uuids)

    # --- admission / tokenization ---
    def poll_admission_queue_nowait(self) -> Any:
        queue = self._w._admission_queue
        if queue is None:
            import queue as _queue
            raise _queue.Empty
        return queue.get_nowait()

    def admit_sequences_from_message(self, msg: dict) -> list[str]:
        return self._w._admit_sequences_from_message(msg)

    def tokenize_admitted_sequences(self, uuids: list[str]) -> None:
        self._w._tokenize_admitted_sequences(uuids)

    def assign_admitted_sequences_to_ranks(self, uuids: list[str]) -> None:
        self._w._assign_admitted_sequences_to_ranks(uuids)

    def build_local_query_book_for_admitted(self, uuids: list[str]) -> None:
        self._w._build_local_query_book_for_admitted(uuids)

    def update_max_input_length(self, new_len: int) -> None:
        import logging
        if new_len > self._w.max_input_length:
            self._w.max_input_length = new_len
            if self._w.rank == 0:
                logging.info(f"[ADMIT] Updated max_input_length to {new_len}")
            self._w._update_config_after_tokenization()

    # --- sequence-batch helpers ---
    def is_sequence_completed(self, seq: Any) -> bool:
        return self._w._is_sequence_completed(seq)

    def update_batch_status(self, uuids: list[str], status: Any) -> None:
        self._w._update_batch_status(uuids, status)

    # --- lifecycle infrastructure ---
    def feed_watchdog(self) -> None:
        self._w.feed_watchdog()

    def enable_decode_watchdog(self) -> None:
        self._w.enable_decode_watchdog()

    def disable_decode_watchdog(self) -> None:
        self._w.disable_decode_watchdog()

    def feed_decode_watchdog(self) -> None:
        self._w.feed_decode_watchdog()

    # --- prefill forward (F4) ---
    def _uuids_to_local_indices(self, uuids: list[str]) -> list[int]:
        out: list[int] = []
        for uuid in uuids:
            local_idx = self._w._uuid_to_local_map.get(uuid)
            if local_idx is not None:
                out.append(local_idx)
        return out

    def prefill_forward(self, uuids: list[str]) -> Any:
        batch = self._uuids_to_local_indices(uuids)
        if not batch:
            return None
        return self._w.prefill(batch)

    def prefill_forward_prepacked(self, uuids: list[str]) -> Any:
        batch = self._uuids_to_local_indices(uuids)
        if not batch:
            return None
        return self._w.prefill_prepacked(batch)

    def enable_prepack(self) -> bool:
        return (
            bool(getattr(self._w, "enable_prepack", False))
            and hasattr(self._w, "prefill_prepacked")
        )

    # --- prefill sizing (Phase 2.7) ---
    def effective_chunk_size(self) -> int:
        return self._w._get_effective_chunk_size()

    def prefill_setup_done(self) -> bool:
        return self._prefill_setup_done

    # --- prefill config (F3) ---
    def prefill_flush_and_reconfigure(self) -> None:
        self._w._prefill_flush_and_reconfigure()
        # F5 invalidation: the decode model + GPU KV cache were freed
        # and destroyed by _prefill_flush_and_reconfigure, so the next
        # decode phase must re-run decode_setup_once.
        self._decode_setup_done = False
        # Phase 2.7: mark the prefill config live so subsequent prefill
        # rounds inside the same phase skip the expensive re-run.
        self._prefill_setup_done = True

    def prefill_prepare_reentry(self, uuids: list[str]) -> None:
        self._w._prefill_prepare_reentry(uuids)

    def prefill_allocate_host_kv(self, uuids: list[str]) -> None:
        self._w._prefill_allocate_host_kv(uuids)

    # --- decode setup + continuous (F5/F6) ---
    def decode_setup_once(self, max_num_seq: int) -> None:
        # PyNccl init is truly one-time — _forward_ep relies on
        # self.comm.change_state(...) which is None without this init.
        if not self._pynccl_initialized:
            self._w._generate_ensure_comms()
            self._pynccl_initialized = True

        # Decode model + GPU KV are reset by each prefill round via
        # prefill_flush_and_reconfigure; re-establish them when stale.
        if not self._decode_setup_done:
            self._w._load_decode_model(max_num_seq, getattr(self._w, "comm", None))
            self._w._init_gpu_kv_with_actual_size()
            self._decode_setup_done = True
            # Phase 2.7: loading the decode model overwrote the
            # parallel_manager's prefill config, so the next prefill
            # phase must re-run ensure_prefill_setup.
            self._prefill_setup_done = False

    def decode_config_for_batch(self, uuids: list[str]) -> None:
        # Repair CTX lengths (all-ranks safe local op)
        repair_fn = getattr(self._w, "_decode_config_repair_ctx_lengths", None)
        if repair_fn is not None:
            repair_fn(uuids)
        # Allocate GPU KV for owner rank's batch
        local_batch = self._uuids_to_local_indices(uuids)
        alloc_fn = getattr(self._w, "_decode_config_allocate_gpu_kv", None)
        if alloc_fn is not None and local_batch:
            alloc_fn(local_batch)

    def decoding_continuous(self, uuids: list[str]) -> None:
        local_batch = self._uuids_to_local_indices(uuids)
        if not local_batch:
            return
        # Build initial new_tokens tensor from decoded_tokens buffers.
        rebuild = getattr(self._w, "_rebuild_input_tokens", None)
        if rebuild is None:
            import torch as _torch
            new_tokens = _torch.zeros(
                (len(local_batch), 1),
                dtype=_torch.int64,
                device=self._w.torch_device,
            )
        else:
            new_tokens = rebuild(local_batch)
        self._w.decoding_continuous(new_tokens, list(uuids), list(local_batch))

    # --- distributed init ---
    def ensure_comms(self) -> None:
        self._w._generate_ensure_comms()

    def init_gpu_kv_with_actual_size(self) -> None:
        self._w._init_gpu_kv_with_actual_size()


__all__ = ["LegacyWorkerBackend"]
