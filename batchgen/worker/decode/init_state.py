"""Phase 2.8.2b — port of ``_decode_init_state`` (batchgen_worker.py:7896-7922).

Initialises a :class:`DecodeState` from the current decode cohort +
batch, and fixes up the rank-local ``sequences_with_gpu_kv`` set so
every batch member is tracked. Legacy stashed async state on
``BatchGenWorker`` via ``self._pending_kv_append_tasks`` etc.; the
native path keeps that on the adapter (Phase 2.7 flush_deferred_kv
already owns it) so ``init_state`` is now a pure DecodeState builder.
"""

from __future__ import annotations

from batchgen.worker.decode.state import DecodeState
from batchgen.worker.protocols import UUID, LegacyInfraBackend
from batchgen.worker.state import WorkerState


def init_decode_state(
    state: WorkerState,
    adapter: LegacyInfraBackend,
    *,
    decode_uuids: list[UUID],
    batch: list[int],
) -> DecodeState:
    """Build a fresh :class:`DecodeState` for a ``run_continuous`` call.

    Steps, preserving legacy intent:

      1. Walk ``batch`` and register every owning uuid in
         ``adapter.sequences_with_gpu_kv()`` (legacy 7906-7909). The
         set is authoritative for "which uuids currently hold GPU
         pages"; any uuid in the local batch but missing from the
         set is a prior allocator bug we paper over here so the
         decode loop doesn't blow up mid-iteration.
      2. Initialise cumulative counters to zero. Legacy piggybacks
         on ``hasattr`` guards on ``BatchGenWorker``; the native path
         moves them onto :class:`DecodeState` so they are per-run
         and per-scheduler, not per-process.

    Returns a :class:`DecodeState` with ``local_iteration=0``,
    ``last_boundary=0``, and ``global_batch_size`` set to the current
    ``state.global_batch`` length (used by helpers that log / cap by
    batch progress).
    """
    # Reset deferred-KV-append task lists the legacy worker mutates
    # from ``_flush_deferred_kv_to_host``. Legacy
    # ``_decode_init_state`` initialised them inline (batchgen_worker
    # .py:7902); without this, the first flush raises
    # ``AttributeError: _pending_kv_append_tasks``.
    adapter.reset_pending_kv_append_tasks()

    uuid_to_local = adapter.uuid_to_local_map()
    local_to_uuid = adapter.local_to_uuid_map()
    sequences_with_gpu_kv = adapter.sequences_with_gpu_kv()

    # Reverse-lookup any uuids present in ``batch`` that the tracking
    # set missed. Two lookup paths match legacy 7907 (``_local_to_uuid_map``).
    for local_idx in batch:
        uuid = local_to_uuid.get(local_idx)
        if uuid is not None and uuid not in sequences_with_gpu_kv:
            sequences_with_gpu_kv.add(uuid)

    # Register listed decode_uuids too, in case the cohort includes
    # rank-owned uuids whose local_idx isn't in ``batch`` yet (can
    # happen when the caller passes a superset, e.g. queued async
    # loads integrating next boundary).
    for uuid in decode_uuids:
        if uuid in uuid_to_local and uuid not in sequences_with_gpu_kv:
            sequences_with_gpu_kv.add(uuid)

    return DecodeState(
        decode_uuids=list(decode_uuids),
        batch=list(batch),
        local_iteration=0,
        last_boundary=0,
        global_batch_size=len(state.global_batch.sequences),
    )


__all__ = ["init_decode_state"]
