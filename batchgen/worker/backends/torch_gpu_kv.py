"""TorchGpuKvBackend — wraps ``GpuPagedKVCacheManager`` from main.

Main's production GPU KV manager exposes batched token-count APIs:

    allocate_pages_for_sequences(global_ids: list[int], tokens: list[int])
    extend_pages_for_sequence(global_id: int, new_total_tokens: int)
    release_sequence_pages(global_ids: list[int])
    rebuild_page_table(global_ids: list[int])
    get_stats().num_free_pages

The adapter translates the per-uuid Protocol to these batched /
global_idx-indexed calls by looking up ``seq.global_idx`` on the
shared :class:`WorkerState`.

**append_kv is a no-op**: main handles per-layer KV append inside the
model forward pass (via layer callbacks), not as a free-standing
manager call. The orchestrator's ``KVCacheManager.append_async`` +
``flush_deferred`` path is a test-friendly abstraction; in production
the real KV write happens during ``ModelExecutorBackend.forward_decode``.
The Protocol method stays for API symmetry.
"""

from __future__ import annotations

from typing import Any

from batchgen.sequence import SequenceEntry
from batchgen.worker.state import WorkerState


class TorchGpuKvBackend:
    """Production adapter for :class:`GpuKvBackend`.

    Constructed with the worker's :class:`GpuPagedKVCacheManager` and a
    reference to the shared :class:`WorkerState` (for uuid→global_idx
    resolution).
    """

    def __init__(self, manager: Any, state: WorkerState) -> None:
        self._m = manager
        self._state = state

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _gid(self, uuid: str) -> int:
        seq = self._state.global_batch.get_sequence(uuid)
        if seq is None:
            raise KeyError(f"TorchGpuKvBackend: uuid {uuid!r} not in global_batch")
        return seq.global_idx

    # ------------------------------------------------------------------
    # Protocol methods
    # ------------------------------------------------------------------

    def allocate_pages(self, uuid: str, n: int) -> list[int]:
        gid = self._gid(uuid)
        tokens = n * SequenceEntry.PAGE_SIZE
        self._m.allocate_pages_for_sequences([gid], [tokens])
        # Return value is not consumed by the handlers (they track
        # allocation via seq.gpu_pages_allocated). Empty list keeps the
        # Protocol shape consistent with the fake.
        return []

    def release_pages(self, uuid: str) -> None:
        self._m.release_sequence_pages([self._gid(uuid)])

    def extend_pages(self, uuid: str, n: int) -> list[int]:
        gid = self._gid(uuid)
        seq = self._state.global_batch.get_sequence(uuid)
        if seq is None:
            return []
        new_total_pages = seq.gpu_pages_allocated + n
        new_total_tokens = new_total_pages * SequenceEntry.PAGE_SIZE
        self._m.extend_pages_for_sequence(gid, new_total_tokens)
        return []

    def append_kv(self, uuid: str, layer: int, kv: Any) -> None:
        """No-op in production — KV append happens inside the model
        forward pass, not via a free-standing manager call. Retained
        for Protocol symmetry so the orchestrator's abstract
        ``append_async`` + ``flush_deferred`` path still type-checks."""
        return

    def free_pages(self) -> int:
        return self._m.get_stats().num_free_pages

    def rebuild_page_table(self, uuids: list[str]) -> None:
        gids = [self._gid(u) for u in uuids]
        self._m.rebuild_page_table(gids)


__all__ = ["TorchGpuKvBackend"]
