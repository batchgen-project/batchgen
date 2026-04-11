"""WorkerOrchestrator — composes every handler into a runnable pipeline.

This is the M6 replacement scaffold for the monolithic ``BatchGenWorker``
on main. It:

  - Takes a :class:`WorkerState` plus every backend as explicit
    constructor arguments. Tests pass fakes; production will pass
    Torch-backed wrappers around the existing ``GpuKvManager`` /
    ``HostKvManager`` / ``torch.distributed`` process group.
  - Takes a :class:`WorkerConfig` with every ``BATCHGEN_*`` knob
    pre-resolved (plan Decision #3: handlers never touch os.environ).
  - Instantiates every handler from M1-M5 in the right wiring order.
  - Exposes a small public surface:
      - ``init(n_queries)`` — one-time state wiring (idempotent).
      - ``run_batch()`` — drive one full prefill+decode cycle on the
        current ``state.global_batch`` until nothing remains IN_DECODE.
      - ``generate_persistent()`` — pool-mode persistent loop with
        admission polling between ``run_batch`` calls.

**M6 scope note**: This orchestrator is additive — it lives alongside
main's ``batchgen_worker.py`` rather than replacing it. The production
swap happens when M7 trace replay + smoke tests on wechat_87 confirm
the new path is equivalent. Until then, ``batchgen/batchgen_worker.py``
on main is untouched and every integration test on the remote H20
containers still runs against the original worker.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from batchgen.sequence import SequenceStatus
from batchgen.worker.admission import AdmissionCoordinator, AdmissionQueueBackend
from batchgen.worker.batch_formation import BatchFormation
from batchgen.worker.boundary import (
    BoundaryExecutor,
    BoundaryGuards,
    BoundaryHandler,
    BoundaryPlanner,
    BoundarySynchronizer,
    PlannerConfig,
)
from batchgen.worker.completion import CompletionHandler
from batchgen.worker.config import WorkerConfig
from batchgen.worker.decode import DecodeScheduler
from batchgen.worker.host_rebalancer import HostKVRebalancer
from batchgen.worker.indexing import IndexManager
from batchgen.worker.kv_manager import KVCacheManager
from batchgen.worker.prefill import PrefillScheduler
from batchgen.worker.protocols import (
    UUID,
    ClockBackend,
    CollectiveBackend,
    GpuKvBackend,
    HostKvBackend,
    LifespanLoggerBackend,
    ModelExecutorBackend,
    ResponseSinkBackend,
    TokenizerBackend,
)
from batchgen.worker.state import WorkerState
from batchgen.worker.sync import SyncCoordinator


@dataclass
class BatchStats:
    """Summary returned by :meth:`WorkerOrchestrator.run_batch`."""

    prefill_rounds: int = 0
    decode_intervals: int = 0
    completed_uuids: list[UUID] = field(default_factory=list)


class WorkerOrchestrator:
    def __init__(
        self,
        state: WorkerState,
        config: WorkerConfig,
        *,
        collectives: CollectiveBackend,
        gpu_kv: GpuKvBackend,
        host_kv: HostKvBackend,
        tokenizer: TokenizerBackend,
        model: ModelExecutorBackend,
        lifespan: LifespanLoggerBackend,
        sink: ResponseSinkBackend,
        clock: ClockBackend,
        admission_queue: AdmissionQueueBackend | None = None,
        decode_delegate: Callable[[list[UUID]], None] | None = None,
        admission_delegate: Callable[[], Any] | None = None,
    ) -> None:
        """
        decode_delegate: production-only hook (see
            :class:`DecodeScheduler` docstring). When set, the decode
            phase delegates one full cycle per invocation to the
            provided closure, bypassing the fake tick loop. In the
            hybrid production swap this closure wraps
            ``BatchGenWorker.decoding_continuous``.
        admission_delegate: production-only hook. When set, the
            orchestrator's AdmissionCoordinator short-circuits its
            own polling pipeline and calls the delegate, which is
            expected to perform the entire admission cycle
            (legacy ``_poll_admissions`` including tokenization +
            query_book build). Required for the hybrid production
            path because legacy ``prefill`` /
            ``decoding_continuous`` consume the
            legacy-built ``query_book``.
        """
        self._state = state
        self._config = config
        self._collectives = collectives
        self._model = model

        # -- handlers -------------------------------------------------
        self._index = IndexManager(state)
        self._sync = SyncCoordinator(state, collectives)
        self._kv = KVCacheManager(
            state,
            gpu_kv,
            host_kv,
            initial_gpu_page_buffer=config.initial_gpu_page_buffer,
            extension_gpu_page_buffer=config.extension_gpu_page_buffer,
            host_kv_total_pages=config.host_kv_total_pages,
            prefill_watermark_pct=config.prefill_watermark_pct,
            eviction_watermark_pct=config.eviction_watermark_pct,
        )
        self._batch_formation = BatchFormation(
            state,
            tokenizer,
            collectives,
            self._index,
            model_context_length=config.model_context_length,
        )
        self._admission = AdmissionCoordinator(
            state,
            collectives,
            admission_queue=admission_queue,
            admission_delegate=admission_delegate,
        )
        self._completion = CompletionHandler(
            state,
            tokenizer,
            collectives,
            sink,
            model_context_length=config.model_context_length,
            ignore_eos=config.ignore_eos,
            rep_detection_enabled=config.rep_detection_enabled,
        )
        self._rebalancer = HostKVRebalancer(state, self._kv, self._sync)
        self._prefill = PrefillScheduler(state, self._kv, model)
        self._boundary = BoundaryHandler(
            state,
            BoundarySynchronizer(state, self._sync, collectives),
            BoundaryPlanner(
                PlannerConfig(
                    prefill_watermark_pct=config.prefill_watermark_pct,
                    decision_frequency_pages=config.decision_frequency_pages,
                    extension_gpu_page_buffer=config.extension_gpu_page_buffer,
                    host_total_pages=config.host_kv_total_pages,
                )
            ),
            BoundaryExecutor(state, self._kv, self._rebalancer),
            BoundaryGuards(state),
            self._kv,
        )
        self._decode = DecodeScheduler(
            state,
            self._kv,
            model,
            self._boundary,
            decision_frequency_pages=config.decision_frequency_pages,
            initial_gpu_page_buffer=config.initial_gpu_page_buffer,
            decode_delegate=decode_delegate,
        )
        self._decode_delegate = decode_delegate

        self._initialized = False

    # ------------------------------------------------------------------
    # Introspection for tests + future orchestrator consumers
    # ------------------------------------------------------------------

    @property
    def state(self) -> WorkerState:
        return self._state

    @property
    def config(self) -> WorkerConfig:
        return self._config

    @property
    def initialized(self) -> bool:
        return self._initialized

    # Handler accessors exposed for tests and future wiring. They are
    # read-only by convention.
    @property
    def index(self) -> IndexManager:
        return self._index

    @property
    def sync(self) -> SyncCoordinator:
        return self._sync

    @property
    def kv(self) -> KVCacheManager:
        return self._kv

    @property
    def batch_formation(self) -> BatchFormation:
        return self._batch_formation

    @property
    def admission(self) -> AdmissionCoordinator:
        return self._admission

    @property
    def completion(self) -> CompletionHandler:
        return self._completion

    @property
    def rebalancer(self) -> HostKVRebalancer:
        return self._rebalancer

    @property
    def prefill(self) -> PrefillScheduler:
        return self._prefill

    @property
    def boundary(self) -> BoundaryHandler:
        return self._boundary

    @property
    def decode(self) -> DecodeScheduler:
        return self._decode

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def init(self) -> None:
        """One-time core initialization. Idempotent."""
        if self._initialized:
            return
        # In production this would load the model weights, allocate the
        # query book, initialize KV managers to their actual size, etc.
        # Those are all orchestrator-owned concerns (not handler state).
        self._decode.load_model()
        self._initialized = True

    # ------------------------------------------------------------------
    # Core scheduling loop (M6 thin version)
    # ------------------------------------------------------------------

    def run_batch(self) -> BatchStats:
        """Run prefill + decode until no more work remains on this batch.

        Thin composition of the handler surface. Each iteration:

          1. Prefill phase: while QUEUEING or EVICTED sequences exist,
             call :meth:`PrefillScheduler.prepare_batch`, run the
             forward pass, and advance status through
             ``QUEUEING → IN_PREFILL → PREFILLED``.
          2. Decode phase: while PREFILLED or ON_HOLD sequences
             exist, call :meth:`DecodeScheduler.prepare_batch`,
             transition to IN_DECODE, run one decision interval via
             :meth:`DecodeScheduler.run_continuous` (which ends in
             the boundary handler), then call
             :meth:`CompletionHandler.check_and_handle` to finalize
             anything that hit its length / EOS / repetition bound
             during the interval.

        Returns :class:`BatchStats` summarizing the round counts and
        the UUIDs reported as completed.
        """
        self.init()
        stats = BatchStats()

        while self._has_unfinished_work():
            stats.prefill_rounds += self._run_prefill_phase()
            stats.decode_intervals += self._run_decode_phase(stats)

        return stats

    # ------------------------------------------------------------------
    # Internal phases
    # ------------------------------------------------------------------

    def _run_prefill_phase(self) -> int:
        """Run prefill rounds until `prepare_batch` returns empty.

        Returns the number of rounds executed.
        """
        rounds = 0
        while True:
            uuids = self._prefill.prepare_batch()
            if not uuids:
                return rounds

            self._prefill.config_for_batch(uuids)
            self._prefill.run(uuids)

            for uuid in uuids:
                seq = self._state.global_batch.get_sequence(uuid)
                if seq is None:
                    continue
                if seq.status == SequenceStatus.QUEUEING:
                    self._state.global_batch.update_status(
                        uuid, SequenceStatus.IN_PREFILL
                    )
                if seq.status == SequenceStatus.EVICTED:
                    self._state.global_batch.update_status(
                        uuid, SequenceStatus.IN_PREFILL
                    )
                self._state.global_batch.update_status(
                    uuid, SequenceStatus.PREFILLED
                )

            rounds += 1

    def _run_decode_phase(self, stats: BatchStats) -> int:
        """Run decode intervals until PREFILLED/IN_DECODE/ON_HOLD are
        drained or a boundary plan returns the batch to prefill."""
        intervals = 0
        while True:
            uuids = self._decode.prepare_batch()
            if not uuids:
                return intervals

            self._decode.try_load_new(uuids)
            for uuid in uuids:
                seq = self._state.global_batch.get_sequence(uuid)
                if seq is None:
                    continue
                if seq.status == SequenceStatus.PREFILLED:
                    self._state.global_batch.update_status(
                        uuid, SequenceStatus.IN_DECODE
                    )

            # After transitions, re-filter to only IN_DECODE uuids.
            in_decode = [
                uuid
                for uuid in uuids
                if (
                    self._state.global_batch.get_sequence(uuid) is not None
                    and self._state.global_batch.get_sequence(uuid).status  # type: ignore[union-attr]
                    == SequenceStatus.IN_DECODE
                )
            ]
            if not in_decode:
                return intervals

            self._decode.config_for_batch(in_decode)
            self._decode.run_continuous(in_decode)

            completed = self._completion.check_and_handle(in_decode)
            stats.completed_uuids.extend(sorted(completed))
            intervals += 1

            # If the decode batch is now empty or all got OnHold'd,
            # return to the outer loop so the prefill phase can run
            # again (and the outer while on has_unfinished_work drives
            # the overall termination).
            if not self._state.global_batch.get_sequences_by_status(
                SequenceStatus.IN_DECODE
            ) and not self._state.global_batch.get_sequences_by_status(
                SequenceStatus.PREFILLED
            ):
                return intervals

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _has_unfinished_work(self) -> bool:
        """True if any non-terminal status has at least one sequence."""
        gb = self._state.global_batch
        for status in (
            SequenceStatus.QUEUEING,
            SequenceStatus.IN_PREFILL,
            SequenceStatus.PREFILLED,
            SequenceStatus.IN_DECODE,
            SequenceStatus.ON_HOLD,
            SequenceStatus.EVICTED,
        ):
            if gb.get_sequences_by_status(status):
                return True
        return False

    # ------------------------------------------------------------------
    # Pool mode
    # ------------------------------------------------------------------

    def generate_persistent(self, *, max_iterations: int | None = None) -> int:
        """Pool-mode persistent loop.

        Each iteration:
          1. Poll the admission queue (rank 0) and broadcast
             (all ranks) via :class:`AdmissionCoordinator`.
          2. If admissions landed, tokenize / assign ranks / build
             query book via :class:`BatchFormation`.
          3. Call :meth:`run_batch` to drive the newly-admitted work
             plus any existing live sequences.

        Termination:
          - ``max_iterations`` (test path): return after that many
            polls regardless of work state. Unit tests rely on this
            to keep runs bounded.
          - ``max_iterations=None`` (production path): block
            forever — the worker process only exits when the server
            sends SIGTERM or the ``_shutdown_requested`` flag is set
            on the legacy worker (main's semantic). Never return
            early on an empty queue — doing so triggers worker
            cleanup that races with live NCCL state.
        """
        import time as _time

        self.init()
        iteration = 0
        while True:
            admitted = self._admission.poll_and_broadcast()
            if admitted:
                self._batch_formation.tokenize(admitted)
                self._batch_formation.assign_ranks(admitted)
                self._batch_formation.build_query_book(admitted)

            if self._has_unfinished_work():
                self.run_batch()

            iteration += 1
            if max_iterations is not None and iteration >= max_iterations:
                return iteration

            if max_iterations is None:
                # Production path: keep polling. When nothing happened
                # this iteration, yield the CPU briefly so we do not
                # busy-spin on rank 0's queue.get_nowait().
                if not admitted and not self._has_unfinished_work():
                    _time.sleep(0.001)
            else:
                # Test path: if max_iterations is set (finite), still
                # honor the legacy "empty then done" termination so
                # bounded runs finish promptly.
                if not admitted and not self._has_unfinished_work():
                    return iteration


__all__ = ["BatchStats", "WorkerOrchestrator"]
