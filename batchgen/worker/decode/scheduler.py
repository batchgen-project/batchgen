"""DecodeScheduler — the IN_DECODE path orchestration class.

Public surface unchanged across the M8 sub-package split. The inner
forward-pass loop lives in :mod:`.continuous_loop` so this file stays
focused on the scheduler's public method shape.

See ``batchgen/worker/decode/__init__.py`` for the package docstring and
the full M5 behavioral notes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from batchgen.sequence import SequenceEntry, SequenceStatus
from batchgen.worker.boundary import BoundaryHandler, BoundaryPlan
from batchgen.worker.decode.bind import bind_decode_context
from batchgen.worker.decode.cleanup import decode_cleanup
from batchgen.worker.decode.continuous_loop import run_decode_interval
from batchgen.worker.decode.forward_step import forward_decode_step
from batchgen.worker.decode.handle_boundary import handle_boundary
from batchgen.worker.decode.init_state import init_decode_state
from batchgen.worker.decode.moe_sync import initial_moe_sync
from batchgen.worker.decode.update_sequences import update_sequences
from batchgen.worker.exceptions import CtxInvariantViolation
from batchgen.worker.kv_manager import KVCacheManager
from batchgen.worker.protocols import (
    UUID,
    CollectiveBackend,
    LegacyInfraBackend,
    ModelExecutorBackend,
)
from batchgen.worker.state import WorkerState


def _use_native_decode() -> bool:
    """Phase 2.8.2 shadow switch (read fresh on every call).

    Default ``BATCHGEN_NATIVE_DECODE=0`` keeps the production path on
    legacy ``decoding_continuous``. ``=1`` routes
    ``run_continuous`` through the Stage 2 native helpers. Checking
    the env var per call lets us flip it between benchmark rounds in
    the same server process (``os.environ[...] = "1"`` from a
    notebook / debugger) without restarting.
    """
    return os.environ.get("BATCHGEN_NATIVE_DECODE", "0") == "1"


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
        legacy_infra: LegacyInfraBackend | None = None,
        collectives: CollectiveBackend | None = None,
    ) -> None:
        """
        legacy_infra: Phase-F6 production adapter. When set,
            :meth:`run_continuous` delegates the full decode cycle
            (forward, sampling, page boundary, completion detection)
            to ``legacy_infra.decoding_continuous``, bypassing the
            fake tick loop + :meth:`BoundaryHandler.run`. Unit tests
            leave this ``None`` for deterministic per-token assertions.
        collectives: used to issue a barrier after the one-time
            ``decode_setup_once`` so all ranks complete the MoE decode
            model swap + GPU KV init before any rank enters a forward.
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
        self._legacy = legacy_infra
        self._collectives = collectives
        self._model_loaded = False
        self.last_configured: list[UUID] = []

    # ------------------------------------------------------------------
    # Candidate selection
    # ------------------------------------------------------------------

    def prepare_batch(self) -> list[UUID]:
        """Return PREFILLED + ON_HOLD UUIDs sorted by ``(global_idx, uuid)``.

        Phase 2.5: selection is now GPU-capacity-aware, matching the
        legacy `_prepare_decode_batch_two_page_buffer`:

          1. Sort candidates by (global_idx, uuid) for cross-rank
             determinism.
          2. For each candidate, sum ``get_gpu_pages_for_two_page_buffer()``.
          3. Stop as soon as the cumulative requirement would exceed
             ``self._kv.get_gpu_free_pages()``.
          4. Respect the legacy per-rank cap
             ``MoE_decoding_micro_batch_size`` when available on the
             LegacyInfraBackend (skipped in CPU unit tests where no
             adapter is wired).

        Previously the orchestrator returned every candidate and let
        ``_decode_config_allocate_gpu_kv`` raise `GpuKvExhaustion` on
        over-admit; the legacy path avoided that crash by capping here.
        """
        candidates: list[SequenceEntry] = []
        for status in (SequenceStatus.PREFILLED, SequenceStatus.ON_HOLD):
            for uuid in self._state.global_batch.get_sequences_by_status(status):
                seq = self._state.global_batch.get_sequence(uuid)
                if seq is not None:
                    candidates.append(seq)
        candidates.sort(key=lambda s: (s.global_idx, s.uuid))

        # Unit-test path: no adapter → the CPU harness doesn't track
        # `get_gpu_pages_for_two_page_buffer` semantics accurately, so
        # keep the pre-2.5 behavior of returning every candidate.
        if self._legacy is None:
            return [s.uuid for s in candidates]

        # Production path: capacity-aware selection.
        free_pages = self._kv.get_gpu_free_pages()
        max_per_rank = self._max_decode_seqs_per_rank()
        rank_counts: list[int] = [0] * max(self._state.world_size, 1)
        total_pages = 0
        selected: list[UUID] = []
        for seq in candidates:
            rank = getattr(seq, "assigned_rank", None)
            if rank is not None and max_per_rank is not None:
                if rank_counts[rank] >= max_per_rank:
                    continue
            pages = seq.get_gpu_pages_for_two_page_buffer()
            if total_pages + pages > free_pages:
                break
            selected.append(seq.uuid)
            total_pages += pages
            if rank is not None and max_per_rank is not None:
                rank_counts[rank] += 1
        return selected

    def _max_decode_seqs_per_rank(self) -> int | None:
        """Return the per-rank decode-batch cap.

        Prefers ``MoE_decoding_micro_batch_size`` but falls back to
        ``attn_decoding_micro_batch_size`` when the former is 0 (the
        config convention for "MoE path unused / default to attention
        cap"). A final 0 (or missing config) is treated as **no cap**
        so we don't lock out all admissions — matching legacy
        behavior observed at L4 where MoE_decoding=0 but
        attn_decoding=64 and decode still ran.
        """
        engine_config = getattr(self._legacy, "engine_config", None) if self._legacy else None
        if engine_config is None:
            w = getattr(self._legacy, "_w", None)
            engine_config = getattr(w, "engine_config", None) if w else None
        if engine_config is None:
            return None
        mod_batching = getattr(engine_config, "Module_Batching_Config", None)
        if mod_batching is None:
            return None
        moe = getattr(mod_batching, "MoE_decoding_micro_batch_size", 0) or 0
        if moe > 0:
            return moe
        attn = getattr(mod_batching, "attn_decoding_micro_batch_size", 0) or 0
        if attn > 0:
            return attn
        return None  # treat as uncapped

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

    def ensure_decode_setup(self) -> None:
        """Run the one-time decode setup (idempotent: returns
        immediately if already done).

        F5 native path: ``decode_setup_once`` re-establishes the
        decode model + GPU KV cache after every prefill round
        (the adapter clears its `_decode_setup_done` flag inside
        ``prefill_flush_and_reconfigure``). Followed by a
        ``collectives.barrier()`` so all ranks finish the MoE swap
        before any rank touches the GPU KV.

        Must be called BEFORE :meth:`try_load_new` because that
        method allocates GPU KV pages for ON_HOLD sequences and
        crashes if the GPU KV manager is None / uninitialized.
        """
        if self._legacy is None:
            return
        max_num_seq = max(len(self._state.global_batch), 1)
        self._legacy.decode_setup_once(max_num_seq)
        if self._collectives is not None:
            self._collectives.barrier()

    def config_for_batch(self, uuids: list[UUID]) -> None:
        """Pre-forward CTX invariant fast-fail (plan Decision #6).

        F5 native path: per-batch ``decode_config_for_batch`` (repair
        CTX + allocate GPU KV) via the adapter. The one-time setup
        (model + KV manager init) lives in :meth:`ensure_decode_setup`
        and must run earlier in the decode phase.
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

        if self._legacy is not None:
            self._legacy.decode_config_for_batch(list(uuids))

    # ------------------------------------------------------------------
    # run_continuous — one decision interval + boundary
    # ------------------------------------------------------------------

    def run_continuous(self, uuids: list[UUID]) -> DecodeStepResult:
        """Run exactly one decision interval then invoke the boundary.

        Two execution modes:

        **Test mode** (``legacy_infra is None``): run the fake tick
        loop via :func:`run_decode_interval` for
        ``decision_frequency_pages * PAGE_SIZE`` iterations, then
        invoke :meth:`BoundaryHandler.run`. Unit tests rely on this
        path for deterministic per-token assertions.

        **Production mode** (``legacy_infra`` is set): delegate the
        full cycle to ``legacy_infra.decoding_continuous(uuids)``
        which handles forward passes, boundary checks, completion
        handling, and mutates ``state.global_batch`` in place. The
        orchestrator's :class:`BoundaryHandler` is NOT invoked —
        production legacy ``decoding_continuous`` already handles
        boundaries internally.
        """
        if self._legacy is not None:
            # Phase 2.8.2i shadow switch: when BATCHGEN_NATIVE_DECODE=1
            # AND the model is on attention mode 3 (the only path the
            # native helpers have been written for), route through the
            # Stage 2 native loop. Any other case falls back to legacy
            # ``decoding_continuous`` — the pre-refactor baseline.
            if _use_native_decode() and self._can_run_native():
                return self._run_continuous_native(list(uuids))
            # F6 baseline: legacy `decoding_continuous` via the adapter.
            self._legacy.decoding_continuous(list(uuids))
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

    # ------------------------------------------------------------------
    # Phase 2.8.2i — native decode loop (shadow-gated)
    # ------------------------------------------------------------------

    def _can_run_native(self) -> bool:
        """Guard: the native loop only handles attention mode 3.

        Legacy ``decoding_continuous`` line 8526-8528 falls back to
        ``_decoding_legacy_modes`` when
        ``engine_config.Basic_Config.attn_mode != 3``; that path has
        ~400 LOC of model-specific branching that Stage 2 does not
        port. Return False so the scheduler keeps using legacy for
        those models while the native path shadows on attn_mode=3
        (DeepSeek / most MoE models).
        """
        engine_config = getattr(self._legacy, "engine_config", None)
        if engine_config is None:
            w = getattr(self._legacy, "_w", None)
            engine_config = getattr(w, "engine_config", None) if w else None
        if engine_config is None:
            # Tests / CPU harness: no engine_config wired → bail out.
            return False
        basic = getattr(engine_config, "Basic_Config", None)
        if basic is None:
            return False
        return int(getattr(basic, "attn_mode", 0)) == 3

    def _run_continuous_native(
        self, uuids: list[UUID],
    ) -> "DecodeStepResult":
        """Native replacement for ``legacy.decoding_continuous``.

        Ports ``decoding_continuous`` (batchgen_worker.py:8495-8601)
        using the Stage 2 helpers. Only runs when
        ``BATCHGEN_NATIVE_DECODE=1`` AND the model is on attn_mode 3
        — the shadow switch guard is in :meth:`run_continuous`.

        Flow:

          1. Translate uuids → local batch via adapter.
          2. Build initial ``new_tokens`` via
             ``adapter.rebuild_input_tokens(batch)``.
          3. ``bind_decode_context`` — class-level wrapper singletons.
          4. ``init_decode_state`` — fresh DecodeState with counters.
          5. ``initial_moe_sync`` — per-rank MoE buffer sizing.
          6. ``adapter.enable_decode_watchdog()``.
          7. Main loop:
              a. Increment iteration counters.
              b. Feed watchdog + decode-watchdog.
              c. Every ``decision_interval_tokens`` tokens, call
                 ``handle_boundary``; break / continue per outcome.
              d. ``forward_decode_step`` → new tokens.
              e. ``adapter.flush_deferred_kv_to_host``.
              f. ``update_sequences`` — EOS / rep / buffer write.
          8. ``decode_cleanup`` in ``finally`` so teardown runs
             even if the loop raises.

        Returns a :class:`DecodeStepResult` with the iteration count
        as ``tokens_produced`` (approximate — matches legacy's
        informational stat) and the final decode_uuids.
        """
        assert self._legacy is not None
        assert self._collectives is not None, (
            "DecodeScheduler._run_continuous_native requires the "
            "CollectiveBackend to be wired (needed for initial_moe_sync)."
        )
        adapter = self._legacy

        # Translate + build initial new_tokens.
        batch = adapter.get_local_indices_for_uuids(list(uuids))
        new_tokens = adapter.rebuild_input_tokens(batch) if batch else None

        gpu_manager, _worker_view = bind_decode_context(
            adapter,
            batch=batch,
            past_key_states=None,
            past_value_states=None,
            scale_dict=None,
        )

        decode_state = init_decode_state(
            self._state, adapter,
            decode_uuids=list(uuids), batch=batch,
        )
        decode_state.new_tokens = new_tokens
        decode_state.page_table_verified = True

        initial_moe_sync(
            self._state, adapter, self._collectives, batch=batch,
        )

        adapter.enable_decode_watchdog()

        decision_interval_tokens = (
            self._decision_frequency_pages * SequenceEntry.PAGE_SIZE
        )
        tokens_produced = 0
        last_plan: BoundaryPlan = BoundaryPlan()

        import logging as _native_log
        _native_log.info(
            f"[NATIVE_DECODE] rank={self._state.rank} entry: "
            f"decode_uuids={len(decode_state.decode_uuids)} "
            f"batch={len(batch)} "
            f"decision_interval_tokens={decision_interval_tokens}"
        )

        try:
            while decode_state.decode_uuids:
                decode_state.local_iteration += 1
                decode_state.cumulative_iterations += 1

                if decode_state.local_iteration % 64 == 0:
                    _native_log.info(
                        f"[NATIVE_DECODE] rank={self._state.rank} "
                        f"iter={decode_state.local_iteration} "
                        f"batch_size={len(decode_state.batch)} "
                        f"uuids={len(decode_state.decode_uuids)}"
                    )

                adapter.feed_watchdog()
                adapter.feed_decode_watchdog()

                if (
                    decode_state.local_iteration - decode_state.last_boundary
                    >= decision_interval_tokens
                ):
                    decode_state.last_boundary = decode_state.local_iteration
                    outcome = handle_boundary(
                        self._state, adapter, self._boundary,
                        decode_state, gpu_manager=gpu_manager,
                    )
                    last_plan = outcome.result.plan
                    _native_log.info(
                        f"[NATIVE_DECODE] rank={self._state.rank} "
                        f"boundary at iter={decode_state.local_iteration}: "
                        f"should_break={outcome.should_break} "
                        f"should_continue={outcome.should_continue} "
                        f"post_decode_uuids={len(decode_state.decode_uuids)} "
                        f"watermark_break={outcome.result.plan.watermark_break} "
                        f"wm_triggered={outcome.result.watermark_triggered}"
                    )
                    if outcome.should_break:
                        break
                    if outcome.should_continue:
                        continue

                # NOTE: DO NOT skip the forward step when batch is empty.
                # MoE models (GPT-OSS, DeepSeek) run an all-to-all + per-
                # iteration all_gather_into_tensor inside
                # ``_decode_forward_step`` that every rank must hit in
                # lockstep — if this rank bails, peers deadlock waiting
                # on the collective. Legacy documents the same rule at
                # batchgen_worker.py:8252-8254. Pass the empty batch in;
                # legacy's forward handles it correctly.
                new_tokens_out = forward_decode_step(
                    adapter,
                    batch=decode_state.batch,
                    new_tokens=decode_state.new_tokens,
                    gpu_manager=gpu_manager,
                    page_table_verified=decode_state.page_table_verified,
                    local_iteration=decode_state.local_iteration,
                )
                decode_state.new_tokens = new_tokens_out

                adapter.flush_deferred_kv_to_host()

                new_tokens_cpu = new_tokens_out.cpu()
                update_sequences(
                    self._state, adapter,
                    batch=decode_state.batch,
                    new_tokens_cpu=new_tokens_cpu,
                    local_iteration=decode_state.local_iteration,
                )
                tokens_produced += 1
        finally:
            decode_cleanup(adapter, self._boundary)

        _native_log.info(
            f"[NATIVE_DECODE] rank={self._state.rank} exit: "
            f"local_iteration={decode_state.local_iteration} "
            f"tokens_produced={tokens_produced} "
            f"final_decode_uuids={len(decode_state.decode_uuids)}"
        )

        return DecodeStepResult(
            tokens_produced=tokens_produced,
            uuids_decoded=tuple(decode_state.decode_uuids),
            boundary_plan=last_plan,
        )


__all__ = ["DecodeStepResult", "DecodeScheduler"]
