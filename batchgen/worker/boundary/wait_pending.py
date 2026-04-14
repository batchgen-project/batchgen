"""Phase 0 of the boundary cycle — wait for pending async KV ops.

Ports legacy ``_boundary_wait_pending`` (batchgen_worker.py:6637-6724)
into the native ``batchgen/worker/boundary/`` package. Drops the
``BATCHGEN_CB_DEBUG`` desync-repair branch: the AdmissionCoordinator
makes rank-consistent uuids an invariant of the orchestrator lifecycle
(see docs/phase_2.8_stage1_design.md §3.1 — "the desync repair is not
an invariant of the native path"), so the native wait_pending has no
need for the extra ``all_gather_object(local_decode_set)`` collective.

Function shape matches legacy so the Stage 1i BoundaryHandler can
swap the callback 1:1:

    (decode_uuids, batch) = wait_pending(
        state, adapter, collectives,
        decode_uuids, batch, gpu_manager,
        pending_async_load_task,
        pending_load_uuids,
        pending_load_local,
        pending_load_global,
    )

Collective usage: one ``barrier()`` iff there is a pending async load
to wait on (matches legacy ``dist.barrier()`` at 6700). No collective
otherwise.
"""

from __future__ import annotations

from typing import Any

from batchgen.worker.protocols import UUID, CollectiveBackend, LegacyInfraBackend
from batchgen.worker.state import WorkerState


def wait_pending(
    state: WorkerState,
    adapter: LegacyInfraBackend,
    collectives: CollectiveBackend,
    *,
    decode_uuids: list[UUID],
    batch: list[int],
    gpu_manager: Any,
    pending_async_load_task: Any,
    pending_load_uuids: list[UUID],
    pending_load_local: list[int],
    pending_load_global: list[int],
) -> tuple[list[UUID], list[int]]:
    """Drain pending KV appends and finalize any prior-cycle async load.

    Steps (preserving legacy order):
      1. ``adapter.wait_pending_kv_append_tasks()`` — drain deferred KV
         writes from the previous decode iteration.
      2. If there are pending async-loaded sequences from the previous
         boundary, wait on the async handle + cross-rank
         ``collectives.barrier()``, then call
         ``adapter.finalize_async_load_minimal(...)`` to fold the
         sequences into the decode set.
      3. Rebuild the GPU page table for the updated batch via
         ``adapter.rebuild_page_table_for_batch`` and fix any
         post-finalize slot-to-seq ordering mismatch.

    Returns:
        ``(decode_uuids, batch)`` — possibly extended by the async
        load's sequences.
    """
    adapter.wait_pending_kv_append_tasks()

    if not pending_load_uuids:
        return decode_uuids, batch

    # There is an in-flight async host→GPU load started by the prior
    # boundary cycle; wait for it to land before we can touch its
    # sequences.
    if pending_async_load_task is not None:
        pending_async_load_task.wait()
        # torch.cuda.synchronize in legacy — we do it through the
        # adapter when wired in prod, but CPU tests can skip.
        _maybe_cuda_sync(state)

    # All ranks synchronize before folding the newly-loaded sequences
    # into the decode set (matches legacy line 6700).
    collectives.barrier()

    decode_uuids, batch = adapter.finalize_async_load_minimal(
        pending_async_load_task,
        pending_load_uuids,
        pending_load_local,
        pending_load_global,
        decode_uuids,
        batch,
        gpu_manager,
    )

    # Rebuild page table to include newly loaded sequences + verify.
    if batch and gpu_manager is not None and getattr(gpu_manager, "is_initialized", False):
        adapter.rebuild_page_table_for_batch(batch, gpu_manager)
        _verify_and_fix_slot_order(adapter, gpu_manager, batch)

    return decode_uuids, batch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _maybe_cuda_sync(state: WorkerState) -> None:
    """Match legacy's ``torch.cuda.synchronize(self.torch_device)``.

    No-op on CPU devices (every CPU-only unit test runs with
    ``torch.device("cpu")``), which keeps the test fakes simple.
    """
    import torch

    device = state.torch_device
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _verify_and_fix_slot_order(
    adapter: LegacyInfraBackend, gpu_manager: Any, batch: list[int]
) -> None:
    """Legacy lines 6717-6722. If the paged-KV slot order drifts from
    the current batch's global-ids after the finalize, re-rebuild the
    page table to restore parity. Handles a rarely-seen race between
    the async load's slot assignment and subsequent boundary work.
    """
    page_table_mgr = getattr(gpu_manager, "_gpu_page_table_manager", None)
    if page_table_mgr is None:
        return
    slot_order = getattr(page_table_mgr, "slot_to_seq_id", None)
    if slot_order is None:
        return
    post_finalize_slot_order = list(slot_order)
    post_finalize_batch_global_ids = adapter.local_indices_to_global_seq_ids(batch)
    if post_finalize_slot_order != post_finalize_batch_global_ids:
        gpu_manager.rebuild_page_table(post_finalize_batch_global_ids)


__all__ = ["wait_pending"]
