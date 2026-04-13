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

    # --- distributed init ---
    def ensure_comms(self) -> None:
        self._w._generate_ensure_comms()

    def init_gpu_kv_with_actual_size(self) -> None:
        self._w._init_gpu_kv_with_actual_size()


__all__ = ["LegacyWorkerBackend"]
