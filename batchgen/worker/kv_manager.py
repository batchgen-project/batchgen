"""KVCacheManager — GPU + host KV lifecycle primitives.

Wraps `GpuKvBackend` and `HostKvBackend` Protocols to expose the small set
of operations every handler needs. Owns deferred-append bookkeeping but
does NOT own eviction or migration (that's `HostKVRebalancer` in M3), and
does NOT call `status_transition()` on any sequence (that's
`BoundaryHandler` / `CompletionHandler`).

Public surface:
  - allocate_two_page_buffer(uuid): initial reservation of
    ``initial + extension`` pages for a newly-entering decode sequence.
    Mirrors main's ``_allocate_gpu_kv_two_page_buffer`` pattern.
  - extend_allocation(uuid, n): grow `uuid`'s GPU page allocation by `n`.
  - release_pages(uuids): release GPU pages for every uuid in the input.
    Multi-uuid signature matches the boundary executor's batch behavior.
  - append_async(uuid, layer, kv): queue a deferred KV-append op. Does
    not touch the GPU backend until `flush_deferred` runs.
  - flush_deferred(): apply every queued append via `GpuKvBackend.append_kv`
    and clear the queue.
  - wait_pending(): in the real impl this waits on the CUDA event; in
    the fake tests it's a no-op hook that tests assert was called.
  - get_host_free_pages(): delegates to `HostKvBackend.free_pages`.
  - get_gpu_free_pages(): delegates to `GpuKvBackend.free_pages`.
  - check_prefill_watermark_trigger(has_pending): True when host-KV
    free percentage is strictly ABOVE ``prefill_watermark_pct`` AND the
    caller passes ``has_pending=True`` (queued or evicted sequences
    exist). Matches main at lines 2345/2351: the watermark is the
    "underutilized" threshold — crossing it upward means the host has
    enough slack to switch from decode to prefill.
  - check_eviction_watermark_trigger(): True when host-KV free
    percentage is strictly BELOW ``eviction_watermark_pct``. This is
    the OTHER end of the host-KV capacity spectrum (default 10% in
    main at line 6474): when host is nearly full, fire eviction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from batchgen.worker.protocols import (
    UUID,
    GpuKvBackend,
    HostKvBackend,
)
from batchgen.worker.state import WorkerState


@dataclass(frozen=True)
class DeferredAppend:
    """One queued kv-append op awaiting `flush_deferred`."""

    uuid: UUID
    layer: int
    kv: Any


class KVCacheManager:
    def __init__(
        self,
        state: WorkerState,
        gpu_kv: GpuKvBackend,
        host_kv: HostKvBackend,
        *,
        initial_gpu_page_buffer: int,
        extension_gpu_page_buffer: int,
        host_kv_total_pages: int,
        prefill_watermark_pct: int,
        eviction_watermark_pct: int = 10,
    ) -> None:
        if initial_gpu_page_buffer < 1:
            raise ValueError(
                f"initial_gpu_page_buffer must be >= 1, got {initial_gpu_page_buffer}"
            )
        if extension_gpu_page_buffer < 0:
            raise ValueError(
                f"extension_gpu_page_buffer must be >= 0, got {extension_gpu_page_buffer}"
            )
        if host_kv_total_pages < 1:
            raise ValueError(
                f"host_kv_total_pages must be >= 1, got {host_kv_total_pages}"
            )
        if not 0 <= prefill_watermark_pct <= 100:
            raise ValueError(
                f"prefill_watermark_pct must be in [0, 100], got {prefill_watermark_pct}"
            )
        if not 0 <= eviction_watermark_pct <= 100:
            raise ValueError(
                f"eviction_watermark_pct must be in [0, 100], got {eviction_watermark_pct}"
            )
        if eviction_watermark_pct >= prefill_watermark_pct:
            raise ValueError(
                f"eviction_watermark_pct ({eviction_watermark_pct}) must be "
                f"< prefill_watermark_pct ({prefill_watermark_pct})"
            )
        self._state = state
        self._gpu_kv = gpu_kv
        self._host_kv = host_kv
        self._initial = initial_gpu_page_buffer
        self._extension = extension_gpu_page_buffer
        self._host_total = host_kv_total_pages
        self._prefill_watermark_pct = prefill_watermark_pct
        self._eviction_watermark_pct = eviction_watermark_pct
        self._deferred: list[DeferredAppend] = []
        self._pending_count: int = 0
        self._wait_pending_calls: int = 0

    # ------------------------------------------------------------------
    # Allocation
    # ------------------------------------------------------------------

    def allocate_two_page_buffer(self, uuid: UUID) -> list[int]:
        """Reserve the initial GPU page block for a sequence entering decode.

        Total pages = ``initial_gpu_page_buffer + extension_gpu_page_buffer``.
        Updates ``seq.gpu_pages_allocated`` and ``seq.had_initial_gpu_reservation``
        on the owning sequence (no-op if the sequence is missing).
        """
        total = self._initial + self._extension
        pages = self._gpu_kv.allocate_pages(uuid, total)
        seq = self._state.global_batch.get_sequence(uuid)
        if seq is not None:
            seq.gpu_pages_allocated = total
            seq.had_initial_gpu_reservation = True
        return pages

    def extend_allocation(self, uuid: UUID, n: int) -> list[int]:
        """Grow `uuid`'s GPU page allocation by `n` pages.

        Increments ``seq.gpu_pages_allocated`` by the granted page count.
        """
        if n <= 0:
            return []
        pages = self._gpu_kv.extend_pages(uuid, n)
        seq = self._state.global_batch.get_sequence(uuid)
        if seq is not None:
            seq.gpu_pages_allocated += len(pages)
        return pages

    def release_pages(self, uuids: list[UUID]) -> None:
        """Release GPU pages for every UUID in the input.

        Clears ``seq.gpu_pages_allocated`` and
        ``seq.had_initial_gpu_reservation`` on each sequence. Missing UUIDs
        are skipped silently.
        """
        for uuid in uuids:
            self._gpu_kv.release_pages(uuid)
            seq = self._state.global_batch.get_sequence(uuid)
            if seq is not None:
                seq.gpu_pages_allocated = 0
                seq.had_initial_gpu_reservation = False

    # ------------------------------------------------------------------
    # Deferred append queue
    # ------------------------------------------------------------------

    def append_async(self, uuid: UUID, layer: int, kv: Any) -> None:
        """Queue a kv-append op for `flush_deferred` to apply later."""
        self._deferred.append(DeferredAppend(uuid=uuid, layer=layer, kv=kv))
        self._pending_count += 1

    def flush_deferred(self) -> int:
        """Apply every queued append via the GPU backend, return op count.

        Called at every page boundary (plan Decision #10). Safe to call
        when the queue is empty — returns 0.
        """
        count = len(self._deferred)
        for op in self._deferred:
            self._gpu_kv.append_kv(op.uuid, op.layer, op.kv)
        self._deferred.clear()
        return count

    def wait_pending(self) -> None:
        """Hook: wait for all in-flight async ops to complete.

        In production this waits on the CUDA event chain. In tests it
        increments a counter so the test can assert it was called at the
        right point in the pipeline.
        """
        self._wait_pending_calls += 1
        self._pending_count = 0

    # ------------------------------------------------------------------
    # Free-page introspection + watermark trigger
    # ------------------------------------------------------------------

    def get_host_free_pages(self) -> int:
        return self._host_kv.free_pages()

    def get_gpu_free_pages(self) -> int:
        return self._gpu_kv.free_pages()

    def check_prefill_watermark_trigger(self, has_pending: bool) -> bool:
        """True when host free % is strictly above `prefill_watermark_pct`
        AND `has_pending` is True.

        Matches main's ``_check_host_kv_watermark_trigger`` at lines
        2345/2351: the watermark is the "underutilized" threshold —
        when host free rises above it AND there is queueing/evicted
        work pending, the scheduler switches from decode back to
        prefill. Strict ``>`` (not ``>=``) mirrors main.

        ``has_pending`` is passed in rather than derived from state so
        the planner stays a pure function over its inputs.
        """
        if not has_pending:
            return False
        free = self._host_kv.free_pages()
        free_pct = (free * 100) // self._host_total
        return free_pct > self._prefill_watermark_pct

    def check_eviction_watermark_trigger(self) -> bool:
        """True when host free % is strictly below `eviction_watermark_pct`.

        The OTHER end of the host-KV capacity spectrum (plan Decision
        #3 / main line 6474): host is nearly full → evict. Strict
        ``<`` mirrors main.
        """
        free = self._host_kv.free_pages()
        free_pct = (free * 100) // self._host_total
        return free_pct < self._eviction_watermark_pct

    # ------------------------------------------------------------------
    # Introspection for tests
    # ------------------------------------------------------------------

    @property
    def deferred_count(self) -> int:
        return len(self._deferred)

    @property
    def wait_pending_call_count(self) -> int:
        return self._wait_pending_calls


__all__ = ["DeferredAppend", "KVCacheManager"]
