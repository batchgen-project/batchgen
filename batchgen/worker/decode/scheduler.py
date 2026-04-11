"""DecodeScheduler — the IN_DECODE path orchestration class.

Public surface unchanged across the M8 sub-package split. The inner
forward-pass loop lives in :mod:`.continuous_loop` so this file stays
focused on the scheduler's public method shape.

See ``batchgen/worker/decode/__init__.py`` for the package docstring and
the full M5 behavioral notes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from batchgen.sequence import SequenceEntry, SequenceStatus
from batchgen.worker.boundary import BoundaryHandler, BoundaryPlan
from batchgen.worker.decode.continuous_loop import run_decode_interval
from batchgen.worker.exceptions import CtxInvariantViolation
from batchgen.worker.kv_manager import KVCacheManager
from batchgen.worker.protocols import UUID, ModelExecutorBackend
from batchgen.worker.state import WorkerState


@dataclass(frozen=True)
class DecodeStepResult:
    """Summary of one :meth:`DecodeScheduler.run_continuous` invocation."""

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
        decode_delegate: Callable[[list[UUID]], None] | None = None,
    ) -> None:
        """
        decode_delegate: optional production hook. When provided,
            :meth:`run_continuous` calls the delegate ONCE per
            invocation (passing the full uuid batch) instead of
            running the fake tick loop + :meth:`BoundaryHandler.run`
            cycle. The delegate is responsible for the entire decode
            cycle including per-token state updates, page boundaries,
            and completion handling — in the hybrid production swap
            this is a closure around ``BatchGenWorker.decoding_continuous``.
            Unit tests leave this as ``None`` and rely on the fake
            tick loop for deterministic per-token assertions.
        """
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
        self._decode_delegate = decode_delegate
        self._model_loaded = False
        self.last_configured: list[UUID] = []

    # ------------------------------------------------------------------
    # Candidate selection
    # ------------------------------------------------------------------

    def prepare_batch(self) -> list[UUID]:
        """Return PREFILLED + ON_HOLD UUIDs sorted by ``(global_idx, uuid)``."""
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
        """Reload ON_HOLD sequences to IN_DECODE if GPU has room."""
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
        """Idempotent lazy model-load hook."""
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
        """Pre-forward CTX invariant fast-fail (plan Decision #6)."""
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

        Two execution modes:

        **Test mode** (``decode_delegate is None``): run the fake tick
        loop via :func:`run_decode_interval` for
        ``decision_frequency_pages * PAGE_SIZE`` iterations, then
        invoke :meth:`BoundaryHandler.run`. Unit tests rely on this
        path for deterministic per-token assertions.

        **Production mode** (``decode_delegate`` is set): call the
        delegate once with the full uuid list. The delegate is
        expected to run the entire decode cycle (forward passes,
        boundary checks, completion handling) inside
        ``BatchGenWorker.decoding_continuous`` and mutate
        ``state.global_batch`` in place. The orchestrator's
        :class:`BoundaryHandler` is NOT invoked — production legacy
        ``decoding_continuous`` already handles boundaries
        internally.
        """
        if self._decode_delegate is not None:
            self._decode_delegate(list(uuids))
            # Production path: return a synthetic result so the
            # orchestrator's run_batch can observe completion via
            # state.global_batch. ``tokens_produced=-1`` marks the
            # delegated path for traces.
            return DecodeStepResult(
                tokens_produced=-1,
                uuids_decoded=tuple(uuids),
                boundary_plan=BoundaryPlan(),
            )

        tokens_produced = run_decode_interval(
            self._state,
            self._model,
            uuids,
            decision_frequency_pages=self._decision_frequency_pages,
        )
        plan = self._boundary.run(list(uuids))
        return DecodeStepResult(
            tokens_produced=tokens_produced,
            uuids_decoded=tuple(uuids),
            boundary_plan=plan,
        )


__all__ = ["DecodeStepResult", "DecodeScheduler"]
