"""PrefillScheduler — candidate selection and prefill forward-pass invocation.

``prepare_batch`` is the interesting method in M3:

  - Candidates: every sequence in ``QUEUEING`` (fresh) or ``EVICTED``
    (re-entering) status on ``state.global_batch``.
  - Priority: EVICTED comes before QUEUEING. Within EVICTED, sort by
    ``decoded_length`` **descending** so sequences that have made the
    most progress are reinstated first — preserving decode work that
    would otherwise be thrown away. Within QUEUEING, sort by
    ``global_idx`` so every rank computes the same order without a
    collective.
  - Host-KV capacity: selecting stops as soon as the next candidate
    would push the cumulative page budget past ``get_host_free_pages``.
    Page budget per sequence uses ``SequenceEntry.PAGE_SIZE`` (64) and
    the standard ``ceil((prompt + max_decode) / PAGE_SIZE)`` formula.

``config_for_batch`` / ``run`` / ``run_prepacked`` are intentionally thin
in M3: they delegate to :class:`ModelExecutorBackend` and record the
uuid list on a ``last_configured`` attribute so tests can pin down
call order. The full attention-mask + GPU-KV page-table setup lands
when DecodeScheduler in M5 wires the end-to-end path.
"""

from __future__ import annotations

from typing import Any

from batchgen.sequence import SequenceEntry, SequenceStatus
from batchgen.worker.kv_manager import KVCacheManager
from batchgen.worker.protocols import (
    UUID,
    CollectiveBackend,
    LegacyInfraBackend,
    ModelExecutorBackend,
)
from batchgen.worker.state import WorkerState


class PrefillScheduler:
    def __init__(
        self,
        state: WorkerState,
        kv: KVCacheManager,
        model: ModelExecutorBackend,
        *,
        legacy_infra: LegacyInfraBackend | None = None,
        collectives: CollectiveBackend | None = None,
    ) -> None:
        """
        legacy_infra: Phase-F3 production adapter. When set,
            :meth:`config_for_batch` runs the native three-phase
            prefill configuration via the adapter
            (``prefill_flush_and_reconfigure`` →
            ``prefill_prepare_reentry`` →
            ``prefill_allocate_host_kv``) followed by a
            ``collectives.barrier()`` to keep ranks in lockstep
            across the MoE layer swap before any rank enters the
            forward pass. CPU tests pass ``legacy_infra=None``.
        collectives: used only to issue the post-config barrier when
            ``legacy_infra`` is set. Optional; when absent the
            barrier is skipped (unit-test path).
        """
        self._state = state
        self._kv = kv
        self._model = model
        self._legacy = legacy_infra
        self._collectives = collectives
        self.last_configured: list[UUID] = []

    # ------------------------------------------------------------------
    # Candidate selection
    # ------------------------------------------------------------------

    def prepare_batch(self) -> list[UUID]:
        """Pick a capacity-bounded prefill batch from QUEUEING + EVICTED.

        Returns the selected UUIDs in execution order. Sequences that fit
        within the remaining host-KV page budget are appended in priority
        order; the first over-budget candidate terminates accumulation.

        Phase 2.7: the per-sequence page count is **byte-identical** to
        what ``_prefill_allocate_host_kv`` will reserve — computed via
        :meth:`SequenceEntry.get_host_pages_for_initial_chunk` with the
        ``chunk_size`` reported by the adapter. Previously the scheduler
        over-reserved ``ceil((prompt + max_decode) / PAGE_SIZE)`` which,
        at L4 scale (max_decode ≈ 256k), admitted ~1 sequence per round
        instead of the ~40+ that actually fit.
        """
        host_free = self._kv.get_host_free_pages()
        chunk_size = self._effective_chunk_size()

        # Resolve SequenceEntry objects for each status, skipping ghosts.
        evicted = self._seqs_in(SequenceStatus.EVICTED)
        queueing = self._seqs_in(SequenceStatus.QUEUEING)

        evicted.sort(key=lambda s: (-s.decoded_length, s.uuid))
        queueing.sort(key=lambda s: (s.global_idx, s.uuid))

        selected: list[UUID] = []
        used = 0
        for seq in evicted + queueing:
            need = self._pages_for(seq, chunk_size)
            if used + need > host_free:
                break
            selected.append(seq.uuid)
            used += need
        return selected

    def _seqs_in(self, status: SequenceStatus) -> list[SequenceEntry]:
        out: list[SequenceEntry] = []
        for uuid in self._state.global_batch.get_sequences_by_status(status):
            seq = self._state.global_batch.get_sequence(uuid)
            if seq is not None:
                out.append(seq)
        return out

    def _effective_chunk_size(self) -> int:
        """Return the adapter-reported chunk size, or a CPU-test fallback.

        The CPU-test fallback (``SequenceEntry.PAGE_SIZE``) is chosen so
        ``max(prompt + chunk, gpu_initial_tokens)`` in
        :meth:`SequenceEntry.get_host_pages_for_initial_chunk` reliably
        collapses to the ``gpu_initial_tokens`` branch, preserving the
        existing CPU-test expectation that a sequence's host-page budget
        is dominated by ``INITIAL_GPU_PAGE_BUFFER`` on empty state.
        """
        if self._legacy is not None:
            return int(self._legacy.effective_chunk_size())
        return SequenceEntry.PAGE_SIZE

    @staticmethod
    def _pages_for(seq: SequenceEntry, chunk_size: int) -> int:
        """Host-KV page count that ``_prefill_allocate_host_kv`` will
        reserve for ``seq``.

        Delegates to :meth:`SequenceEntry.get_host_pages_for_initial_chunk`
        which mirrors the allocator formula verbatim:

            initial_tokens = max(prompt + chunk_size,
                                 (ceil((prompt+1)/PAGE_SIZE) + INITIAL_GPU_PAGE_BUFFER) * PAGE_SIZE)
            initial_tokens = min(initial_tokens, kv_token_budget)
            pages = ceil(initial_tokens / PAGE_SIZE)
        """
        return seq.get_host_pages_for_initial_chunk(chunk_size)

    # ------------------------------------------------------------------
    # Forward-pass invocation (M3 thin — real GPU-KV wiring lands in M5)
    # ------------------------------------------------------------------

    def config_for_batch(self, uuids: list[UUID]) -> None:
        """Prepare GPU state for prefilling ``uuids``.

        Native (F3) path — when a :class:`LegacyInfraBackend` adapter is
        wired, this runs the three-phase configuration that was previously
        invoked by ``prefill_config_delegate``:

          1. ``prefill_flush_and_reconfigure`` — flush pending KV, free
             decode-model memory, destroy GPU KV cache, reconfigure model
             for prefill.
          2. ``prefill_prepare_reentry(uuids)`` — rebuild re-entry state
             for EVICTED uuids (scalar fields on all ranks + buffer/query
             rebuild on owner rank).
          3. ``prefill_allocate_host_kv(uuids)`` — register rank-owned
             uuids in local maps and allocate host KV pages for them.

        Afterwards a ``collectives.barrier()`` is issued so every rank
        completes the MoE layer swap before any rank enters the prefill
        forward pass — without this, fast ranks trigger all-to-all while
        slower ranks are still mid-swap and the forward pass fails with
        ``rope_cos`` shape mismatches.

        Unit-test path (``legacy_infra is None``) remains thin: we just
        record the uuid list for call-order assertions.
        """
        self.last_configured = list(uuids)
        if self._legacy is None:
            return

        self._legacy.prefill_flush_and_reconfigure()
        self._legacy.prefill_prepare_reentry(list(uuids))
        self._legacy.prefill_allocate_host_kv(list(uuids))
        if self._collectives is not None:
            self._collectives.barrier()

    def run(self, uuids: list[UUID]) -> Any:
        """Run the prefill forward pass.

        F4 native path: when a :class:`LegacyInfraBackend` adapter is
        wired, call ``prefill_forward_prepacked`` if the worker supports
        the prepacked path (``enable_prepack``), otherwise
        ``prefill_forward``. This replaces the ``prefill_fn`` closure
        that previously lived in ``worker_reextract_entry.py``.

        Unit-test path: delegate to :class:`ModelExecutorBackend`.
        """
        if self._legacy is not None:
            if self._legacy.enable_prepack():
                return self._legacy.prefill_forward_prepacked(list(uuids))
            return self._legacy.prefill_forward(list(uuids))
        return self._model.forward_prefill(
            {"uuids": list(uuids), "prepacked": False}
        )

    def run_prepacked(self, uuids: list[UUID]) -> Any:
        """Run the prepacked prefill variant explicitly.

        Used by tests only — production path auto-selects prepacked via
        :meth:`run` when the adapter reports ``enable_prepack()``.
        """
        if self._legacy is not None:
            return self._legacy.prefill_forward_prepacked(list(uuids))
        return self._model.forward_prefill(
            {"uuids": list(uuids), "prepacked": True}
        )


__all__ = ["PrefillScheduler"]
