"""TorchHostKvBackend — wraps ``HostPagedKVWorkerView`` from main.

Main's host KV worker view uses tuples for the batched allocation
API (different from the GPU side, which uses parallel lists):

    allocate_pages_for_sequences([(global_idx, tokens), ...])
    release_sequence_pages([global_idx])
    # No generic free_pages() at the view level; stats live on the
    # backing global HostPagedKVCacheManager.

``load_to_gpu_async`` is a no-op in production — the real host→GPU
transfer runs inside the model forward path via pre-registered
page-table lookups. The Protocol method is kept so the orchestrator
can emit ``AsyncLoadHostToGpu`` decisions with a stable shape; the
adapter just ignores them until M9's real-hardware session wires
the actual async handle machinery.
"""

from __future__ import annotations

from typing import Any

from batchgen.sequence import SequenceEntry
from batchgen.worker.state import WorkerState


class TorchHostKvBackend:
    """Production adapter for :class:`HostKvBackend`."""

    def __init__(
        self,
        worker_view: Any,
        state: WorkerState,
        *,
        total_pages: int,
    ) -> None:
        self._view = worker_view
        self._state = state
        self._total_pages = total_pages

    def _gid(self, uuid: str) -> int:
        seq = self._state.global_batch.get_sequence(uuid)
        if seq is None:
            raise KeyError(f"TorchHostKvBackend: uuid {uuid!r} not in global_batch")
        return seq.global_idx

    def allocate_pages(self, uuid: str, n: int) -> list[int]:
        gid = self._gid(uuid)
        tokens = n * SequenceEntry.PAGE_SIZE
        self._view.allocate_pages_for_sequences([(gid, tokens)])
        return []

    def release_pages(self, uuid: str) -> None:
        self._view.release_sequence_pages([self._gid(uuid)])

    def load_to_gpu_async(self, uuid: str, page_ids: list[int]) -> Any:
        """**Stub**: real async host→GPU transfer is DecodeScheduler +
        model-path work. Until that lands in a hardware session, this
        method is a no-op that returns a placeholder handle. The
        orchestrator's :class:`BoundaryExecutor._apply_async_load`
        already raises ``NotImplementedError`` for non-empty
        AsyncLoadHostToGpu decisions, so this code path is never hit
        by the planner we ship today."""
        return None

    def free_pages(self) -> int:
        """Free-page count is computed against the total. Main reads
        this off the global HostPagedKVCacheManager via an internal
        `allocated_pages` counter; if the worker view does not expose
        one the orchestrator's watermark checks fall back to
        ``total - allocated`` estimated from sequence metadata. For
        M9 we forward to the view's ``num_free_pages`` when it exists
        and otherwise return the static total (safe upper bound — the
        planner's watermark check fires only when below threshold)."""
        get_stats = getattr(self._view, "get_stats", None)
        if get_stats is not None:
            try:
                return int(get_stats().num_free_pages)
            except AttributeError:
                pass
        num_free = getattr(self._view, "num_free_pages", None)
        if num_free is not None:
            return int(num_free)
        return self._total_pages


__all__ = ["TorchHostKvBackend"]
