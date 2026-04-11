"""DecodeScheduler — the IN_DECODE path glue.

Single-file implementation for M5. The M8 split (plan Staged rollout)
will carve this into ``worker/decode/{loop, config, continuous, watermark,
__init__}.py`` without changing the public surface.

Public methods (plan "Other modules — one-line contracts"):

  - ``prepare_batch()``: select PREFILLED + ON_HOLD sequences, return
    them sorted by ``(global_idx, uuid)`` so every rank computes the
    same candidate set without a collective.
  - ``try_load_new(uuids)``: bring ON_HOLD sequences back to the GPU
    by allocating their two-page buffer. Stops when GPU free drops
    below the initial reservation threshold. Missing and non-ON_HOLD
    UUIDs are skipped.
  - ``load_model()``: lazy one-time model load hook. Idempotent —
    every orchestrator calls it on the first decode iteration and
    subsequent calls are no-ops.
  - ``config_for_batch(uuids)``: pre-forward CTX invariant fast-fail
    (plan Decision #6). Raises :class:`CtxInvariantViolation` the
    moment any seq's current_context_length drifts from
    original_prompt_length + decoded_length.
  - ``run_continuous(uuids)``: run one "decision interval" —
    ``decision_frequency_pages * PAGE_SIZE`` forward_decode iterations
    with per-seq state updates — then invoke
    :meth:`BoundaryHandler.run` on the same UUID list. Returns a
    :class:`DecodeStepResult` summary of what happened.

DecodeScheduler does **not** own: the outer ``generate()`` loop
(orchestrator), EOS detection or finish reasons (:class:`CompletionHandler`),
eviction or migration (:class:`HostKVRebalancer`), GPU page allocation
semantics (:class:`KVCacheManager`).
"""

from __future__ import annotations

from dataclasses import dataclass

from batchgen.sequence import SequenceEntry, SequenceStatus
from batchgen.worker.boundary import BoundaryHandler, BoundaryPlan
from batchgen.worker.exceptions import CtxInvariantViolation
from batchgen.worker.kv_manager import KVCacheManager
from batchgen.worker.protocols import UUID, ModelExecutorBackend
from batchgen.worker.state import WorkerState


@dataclass(frozen=True)
class DecodeStepResult:
    """Summary of one `run_continuous` invocation."""

    tokens_produced: int
    uuids_decoded: tuple[UUID, ...]
    boundary_plan: BoundaryPlan


class DecodeScheduler:
    def __init__(
        self,
        state: WorkerState,
        kv: KVCacheManager,
        model: ModelExecutorBackend,
        boundary: BoundaryHandler,
        *,
        decision_frequency_pages: int,
        initial_gpu_page_buffer: int,
    ) -> None:
        if decision_frequency_pages < 1:
            raise ValueError(
                f"decision_frequency_pages must be >= 1, got {decision_frequency_pages}"
            )
        if initial_gpu_page_buffer < 1:
            raise ValueError(
                f"initial_gpu_page_buffer must be >= 1, got {initial_gpu_page_buffer}"
            )
        self._state = state
        self._kv = kv
        self._model = model
        self._boundary = boundary
        self._decision_frequency_pages = decision_frequency_pages
        self._initial_gpu_page_buffer = initial_gpu_page_buffer
        self._model_loaded = False
        self.last_configured: list[UUID] = []

    # ------------------------------------------------------------------
    # Candidate selection
    # ------------------------------------------------------------------

    def prepare_batch(self) -> list[UUID]:
        """Return PREFILLED + ON_HOLD UUIDs sorted by ``(global_idx, uuid)``.

        Sorting without a collective is safe because every rank reads
        the same cross-rank-synced ``state.global_batch`` and the sort
        key is deterministic.
        """
        candidates: list[SequenceEntry] = []
        for status in (SequenceStatus.PREFILLED, SequenceStatus.ON_HOLD):
            for uuid in self._state.global_batch.get_sequences_by_status(status):
                seq = self._state.global_batch.get_sequence(uuid)
                if seq is not None:
                    candidates.append(seq)
        candidates.sort(key=lambda s: (s.global_idx, s.uuid))
        return [s.uuid for s in candidates]

    # ------------------------------------------------------------------
    # ON_HOLD → IN_DECODE reload
    # ------------------------------------------------------------------

    def try_load_new(self, uuids: list[UUID]) -> list[UUID]:
        """Reload ON_HOLD sequences to IN_DECODE if GPU has room.

        Each reload costs ``initial_gpu_page_buffer`` pages from
        :meth:`KVCacheManager.allocate_two_page_buffer`. The loop stops
        at the first UUID that doesn't fit, rather than continuing
        past it — matching main's "greedy until budget exhausted"
        behavior.

        Returns the list of UUIDs successfully loaded. Skips missing
        UUIDs and UUIDs in any status other than ON_HOLD.
        """
        loaded: list[UUID] = []
        total = self._initial_gpu_page_buffer
        for uuid in uuids:
            seq = self._state.global_batch.get_sequence(uuid)
            if seq is None or seq.status != SequenceStatus.ON_HOLD:
                continue
            if self._kv.get_gpu_free_pages() < total:
                break
            self._kv.allocate_two_page_buffer(uuid)
            self._state.global_batch.update_status(uuid, SequenceStatus.IN_DECODE)
            loaded.append(uuid)
        return loaded

    # ------------------------------------------------------------------
    # Lazy model load
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """Lazy one-time model-load hook.

        Called by the orchestrator on the first decode iteration. In
        production this reads weights into GPU memory; the handler
        records a boolean flag here so the fake test path can observe
        idempotence and the orchestrator can skip subsequent calls.
        """
        if self._model_loaded:
            return
        self._model_loaded = True

    @property
    def model_loaded(self) -> bool:
        return self._model_loaded

    # ------------------------------------------------------------------
    # CTX fast-fail pre-forward check
    # ------------------------------------------------------------------

    def config_for_batch(self, uuids: list[UUID]) -> None:
        """Configure for the decode batch. Fast-fails on CTX drift.

        Plan Decision #6: main has a silent log-and-repair path at
        lines 3248-3258 / 5763-5799 that papers over real bugs. The
        re-extracted worker raises :class:`CtxInvariantViolation` the
        moment any sequence's current_context_length does not equal
        ``original_prompt_length + decoded_length``, so the next
        trace-replay run against main surfaces every drift site.
        """
        for uuid in uuids:
            seq = self._state.global_batch.get_sequence(uuid)
            if seq is None:
                continue
            expected = seq.original_prompt_length + seq.decoded_length
            if seq.current_context_length != expected:
                raise CtxInvariantViolation(
                    uuid=uuid,
                    side="sender",
                    had=seq.current_context_length,
                    expected=expected,
                )
        self.last_configured = list(uuids)

    # ------------------------------------------------------------------
    # run_continuous — one decision interval + boundary
    # ------------------------------------------------------------------

    def run_continuous(self, uuids: list[UUID]) -> DecodeStepResult:
        """Run exactly one decision interval then invoke the boundary.

        The interval length is ``decision_frequency_pages * PAGE_SIZE``
        iterations (POIS Q3: same pace for every IN_DECODE seq in M5,
        per-seq pacing lands when speculative decoding is wired). Each
        iteration:

          1. Call :meth:`ModelExecutorBackend.forward_decode` with the
             current uuid batch.
          2. Increment ``decoded_length`` and ``current_context_length``
             for every sequence still in IN_DECODE. Ticks are applied
             in a single write per seq so the CTX invariant holds
             between iterations.

        After the interval:

          3. Call :meth:`BoundaryHandler.run` on the final uuid list.
             The returned plan is carried back in the
             :class:`DecodeStepResult`.

        Sequences that drop out of IN_DECODE mid-interval (e.g. via a
        completion check in a future slice) no longer receive ticks.
        """
        page_size = SequenceEntry.PAGE_SIZE
        iterations = self._decision_frequency_pages * page_size
        tokens_produced = 0

        for _ in range(iterations):
            self._model.forward_decode({"uuids": list(uuids)})
            for uuid in uuids:
                seq = self._state.global_batch.get_sequence(uuid)
                if seq is None or seq.status != SequenceStatus.IN_DECODE:
                    continue
                seq.decoded_length += 1
                seq.current_context_length += 1
            tokens_produced += 1

        plan = self._boundary.run(list(uuids))
        return DecodeStepResult(
            tokens_produced=tokens_produced,
            uuids_decoded=tuple(uuids),
            boundary_plan=plan,
        )


__all__ = ["DecodeStepResult", "DecodeScheduler"]
