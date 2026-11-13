"""Decoupled GPU decode loop behind a model-state binder."""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Protocol

import torch

from batchgen import lifespan
from batchgen.eventlog.counters import STEP_COUNTERS_INTERVAL_S, StepCounters
from batchgen.lifespan import SeqEvent
from batchgen.sequence import SequenceEntry, SequenceStatus
from batchgen.worker.deferred_readback import (
    DeferredTokenReadbackError,
    TokenReadbackRing,
    deferred_decode_step_marks,
)
from batchgen.worker.sampling_residency import DecodeSamplingResidency
from batchgen.worker.step_timing import (
    FORWARD_EVENT_RING_SLOTS,
    ForwardEventRing,
    accumulate_decode_step_split,
    format_decode_step_split,
    new_decode_step_split,
)

# ``_wait_decode_tokens``' two outcomes (single owner; the worker imports
# them).  COMPLETED is exactly what ``synchronize()`` guarantees: the step's
# device work is done and its readback buffers hold this step's tokens.
# ABANDONED means the cohort is being torn down and the event may never fire,
# so none of the step's outputs may be read.
_DECODE_WAIT_COMPLETED = "completed"
_DECODE_WAIT_ABANDONED = "abandoned"
DECODE_DEFERRED_TOKEN_READBACK = True
# Env-evaluated debug flags, evaluated once here (single owner; the worker
# imports them).
# Debug logging level for continuous batching page boundary.
BATCHGEN_CB_DEBUG = os.environ.get("BATCHGEN_CB_LOG", "").upper() == "DEBUG"
# Optional decode-time invariant check: assert cache_seqlens <= gpu_pages_allocated * PAGE_SIZE.
BATCHGEN_DECODE_ASSERT = os.environ.get("BATCHGEN_DECODE_ASSERT", "0") == "1"
# Multi-batch diagnostic logging for investigating metadata corruption.
BATCHGEN_MULTI_BATCH_DIAG = os.environ.get("BATCHGEN_MULTI_BATCH_DIAG", "0") == "1"


class DecodeModelStateBinder(Protocol):
    def _bind_decode_entry_model_state(
        self, gpu_manager, worker_view, scale_dict, past_key_states, past_value_states, batch
    ) -> None: ...

    def _rebind_decode_replay_model_metadata(
        self, batch_sequences, cache_seqlens, max_ctx, batch, new_tokens, page_table_verified, gpu_manager
    ): ...

    def _install_decode_kv_append_callbacks(self, batch, batch_sequences) -> None: ...

    def _read_decode_step_model_metadata(
        self, global_decode_sequences, batch, probe_adapter, probe_round_request, probe_mismatch_logged
    ) -> bool: ...

    def _read_decode_max_seqlen(self): ...
    def _read_decode_cur_batch(self): ...
    def _read_decode_cache_seqlens(self): ...
    def _read_decode_position_ids(self): ...
    def _read_decode_attention_mask(self): ...
    def _read_decode_kv_append_callbacks(self): ...
    def _teardown_decode_model_state(self) -> None: ...


class DecoupledGpuDecode:
    def run(
        self,
        worker: Any,
        binder: DecodeModelStateBinder,
        decode_uuids,
        batch,
        new_tokens,
        past_key_states=None,
        past_value_states=None,
        scale_dict=None,
    ):
        # The moved loop body is kept VERBATIM from batchgen_worker.py:
        # several suites pin its source text (``self._graph.cuda_graph_manager
        # .replay``, the [COMPLETED_ROW_SKIP] block, statement order), and the
        # S4b move proof diffs the body against the rb4 baseline. Rebinding
        # ``self`` to the worker preserves every pinned spelling; the engine
        # instance itself is stateless on purpose (phase transitions rebind
        # model/KV handles between rounds, so nothing may be cached here).
        self = worker
        # Setup
        gpu_manager = self.gpu_paged_kv_cache_manager
        if gpu_manager is None:
            gpu_manager = getattr(self.core_engine, "gpu_paged_kv_manager", None)
        
        worker_view = getattr(self.core_engine, "host_paged_kv_worker_view", None)
        
        binder._bind_decode_entry_model_state(
            gpu_manager,
            worker_view,
            scale_dict,
            past_key_states,
            past_value_states,
            batch,
        )

        # Async state
        self._pending_kv_append_tasks = []
        self._pending_kv_append_tensors = []
        
        pending_async_task = None
        pending_load_uuids = []
        pending_load_local = []
        pending_load_global = []
        
        # Validation
        for local_idx in batch:
            uuid = self._local_to_uuid_map.get(local_idx)
            if uuid and uuid not in self._sequences_with_gpu_kv:
                self._sequences_with_gpu_kv.add(uuid)
        
        # Use cumulative counters that persist across prefill/decode switches
        # Initialize instance vars if not present (shouldn't happen, but safety)
        if not hasattr(worker, '_cumulative_decode_iterations'):
            self._cumulative_decode_iterations = 0
        if not hasattr(worker, '_cumulative_decode_boundaries'):
            self._cumulative_decode_boundaries = 0
        if not hasattr(worker, '_cumulative_boundary_ms'):
            self._cumulative_boundary_ms = 0.0
        if not hasattr(worker, '_cumulative_forward_ms'):
            self._cumulative_forward_ms = 0.0

        # Local iteration counter (for boundary interval tracking within this decode round)
        local_iteration = 0
        last_boundary = 0
        global_batch_size = len(self.global_batch)
        # Re-entry can be admitted with exactly enough host capacity for the
        # current context.  Force one planner pass before the first forward so
        # the next token never relies on an emergency append allocation.
        force_host_boundary = self._host_capacity_due_all_ranks(batch)

        # ========== INITIAL MOE BUFFER SYNC ==========
        # Sync buffer size BEFORE first forward pass to prevent overflow.
        # The boundary sync (in _page_boundary_fast) only happens after DECISION_INTERVAL
        # iterations, but the first forward pass runs immediately. Without this sync,
        # if one rank has more tokens than the initial estimate (ceil(total/world_size)),
        # we get buffer overflow.
        max_batch_size = self._sync_decode_moe_rank_counts(batch, reason="decode_entry")

        # OPTIMIZATION: Track if page table was verified since last batch change
        # Avoids redundant page table checks between boundaries
        _page_table_verified_this_batch = True  # Start True after entry check

        # P0: Pre-allocate pinned memory and one reusable completion event for the
        # only mandatory steady-state GPU→CPU dependency: the sampled token IDs.
        _new_tokens_pinned = torch.empty(max(max_batch_size, 1), 1, dtype=torch.long, pin_memory=True)
        _new_tokens_ready = torch.cuda.Event()
        # Deferred readback (DECODE_DEFERRED_TOKEN_READBACK): two alternating
        # pinned slots so the in-flight step's copy never targets the buffer
        # the host is about to read, plus the one launched-but-unread step.
        # Read once per round so the loop never mixes the two step shapes.
        _deferred_readback = DECODE_DEFERRED_TOKEN_READBACK
        _readback_ring = None
        if _deferred_readback:
            _readback_ring = TokenReadbackRing(
                max(max_batch_size, 1),
                lambda n: torch.empty(n, 1, dtype=torch.long, pin_memory=True),
                torch.cuda.Event,
            )
        _pending_step = None
        _bad_rows_drain_due = False
        self._decode_tokens_unread = False

        # Device-resident per-request sampling parameters. Rebuilding them each
        # step meant three `torch.tensor(..., device=cuda)` calls, i.e. three
        # pageable-source H2D copies that block the host until the whole-model
        # graph replay already queued on this stream retires. Resident, the step
        # path does one integer compare and three slices instead.
        #
        # Local to this decode round for the same reason as the accumulators
        # above: a hot reload rebinds methods on the live worker, so a device
        # buffer kept on `worker` could outlive a slot-layout change. A new round
        # starts a new residency.
        _model_device = next(self.model.parameters()).device
        if _model_device != self.torch_device:
            raise RuntimeError(
                f"Rank {self.rank}: decode device mismatch — model parameters are "
                f"on {_model_device} but worker torch_device is {self.torch_device}; "
                "the resident sampling tensors would be allocated on the wrong device"
            )
        _sampling_residency = DecodeSamplingResidency(
            max(max_batch_size, 1), self.torch_device
        )

        # Heartbeat state for the rate-limited [DECODE] progress line below
        _hb_last_time = time.perf_counter()
        _hb_tokens = 0
        # Per-worker step counters behind the rate-limited ``step.counters``
        # event. Plain ints/floats that persist across decode rounds, so a
        # window spanning several short rounds still reports.
        _step_counters = getattr(worker, "_step_counters", None)
        if _step_counters is None:
            _step_counters = StepCounters(logging.getLogger(__name__), _hb_last_time)
            self._step_counters = _step_counters
        _step_counters.begin_round(_hb_last_time)
        # Host-KV health line: same cadence, one per node (see _log_host_kv_health)
        _hkv_last_time = time.perf_counter()

        # Per-step CPU wall split, averaged into the rank-0 heartbeat. Local to
        # this decode round: a hot reload rebinds methods on the live worker,
        # so an accumulator kept on ``worker`` could outlive a slot-layout change.
        _step_split = new_decode_step_split()
        # Device-side time of the forward+sample segment from reusable timing
        # events, so the heartbeat reports real GPU forward time next to the
        # CPU split. The ring never makes the host wait (see
        # batchgen.worker.step_timing). Its record points sit in this loop
        # body, outside any graph capture: every capture site is a scoped
        # ``torch.cuda.graph`` context that ends before its callee returns
        # here, and this loop only replays graphs (a missing whole-model
        # bucket falls back to eager below instead of capturing).
        _fwd_ring = ForwardEventRing(
            FORWARD_EVENT_RING_SLOTS,
            lambda: torch.cuda.Event(enable_timing=True),
        )
        # Criticality probe of the resident graph, if this round's configure
        # installed one. Each replayed step's ring pair is tagged with the
        # probe's level, so the harvested device times feed it with no wait of
        # their own. A job admitted mid-round with a different request is
        # reported once and ignored: the graph is fixed until the next
        # configure.
        _probe = getattr(worker, "_criticality_probe", None)
        _probe_adapter = self._graph.cuda_graph_adapter
        _probe_round_request = getattr(worker, "_criticality_probe_request", None)
        _probe_mismatch_logged = False
        if _probe is not None:
            _probe.begin_round()

        # Main decode loop — enable decode watchdog for monitoring
        self.enable_decode_watchdog()
        while decode_uuids:
            # --- Decode-step teardown boundary (top of step) ---
            # The second of this loop's two flag tests.  The one after the token
            # readback below catches a rank the abort just freed; this one
            # catches the ordinary case where teardown began between steps, so
            # the step never issues any work at all.  See the post-readback
            # check for why neither may break out of this loop, and for why the
            # in-flight local work is quiesced on the way out.
            if self._cohort_shutdown_requested():
                self._cohort_shutdown_observed = True
                self._quiesce_decode_for_cohort_shutdown(pending_async_task)
                return decode_uuids, batch

            local_iteration += 1
            self._cumulative_decode_iterations += 1

            # Feed watchdogs to prevent timeout during long decoding
            self.feed_watchdog()
            self.feed_decode_watchdog()

            # Step counters: field bumps and one time compare per step; the
            # event payload is built only when its interval has elapsed.
            _step_now_s = time.perf_counter()
            _step_rows = len(decode_uuids)
            _step_counters.add_step(_step_now_s, _step_rows)
            _step_counters.maybe_emit(_step_now_s, STEP_COUNTERS_INTERVAL_S)

            # Rate-limited decode heartbeat (rank 0, ~every 30 s) so the log
            # monitor sees liveness during long decode phases
            _hb_tokens += _step_rows
            if self.rank == 0 and _step_now_s - _hb_last_time >= 30.0:
                _hb_elapsed = time.perf_counter() - _hb_last_time
                _hb_finished = len(self.global_batch.get_sequences_by_status(SequenceStatus.COMPLETED))
                _split_txt = ""
                if _step_split[0] > 0:
                    _split_txt = " " + format_decode_step_split(_step_split)
                    _step_split = new_decode_step_split()
                _split_txt += _fwd_ring.take_summary()
                logging.info(
                    f"[DECODE] step={self._cumulative_decode_iterations} "
                    f"active={len(decode_uuids)} finished={_hb_finished} "
                    f"tok/s={_hb_tokens / _hb_elapsed:.2f}{_split_txt}"
                )
                # The sampling fallback count is now accumulated on device so
                # the step path never reads it back. Drain it here, where the
                # sampled-token readback has already caught the device up, so an
                # invalid-probability row still surfaces instead of going silent.
                # With a deferred step in flight the device has NOT caught up
                # (that read would stall a whole forward), so the drain moves to
                # the next pipeline drain at the page boundary.
                if _pending_step is not None:
                    _bad_rows_drain_due = True
                else:
                    self._warn_sampling_bad_rows(_sampling_residency)
                _hb_last_time = time.perf_counter()
                _hb_tokens = 0

            # Host KV is shared per node, so local rank 0 of every node reports
            # its own regions; the audit walks the region under its locks, which
            # is why it rides this ~30 s cadence and never runs per step.
            if self.local_rank == 0 and time.perf_counter() - _hkv_last_time >= 30.0:
                self._log_host_kv_health()
                _hkv_last_time = time.perf_counter()

            # Page boundary check - use DECISION_INTERVAL (configurable via BATCHGEN_DECISION_FREQUENCY_PAGES)
            if force_host_boundary or local_iteration - last_boundary >= self.DECISION_INTERVAL:
                # [DEFERRED_READBACK_BOUNDARY_DRAIN] Every boundary decision
                # (completion, eviction, loads, watermark, loop exit) reads stop
                # state, so the deferred step is fully consumed first -- on
                # every rank, at the same step, which keeps the boundary in DP
                # lockstep. Nothing is enqueued behind it, so the drain may end
                # in the usual device-wide sync.
                if _pending_step is not None:
                    _finish = self._finish_deferred_decode_step(
                        _pending_step, _readback_ring, overlapped=False
                    )
                    _pending_step = None
                    if _finish != _DECODE_WAIT_COMPLETED:
                        self._cohort_shutdown_observed = True
                        self._quiesce_decode_for_cohort_shutdown(pending_async_task)
                        return decode_uuids, batch
                    if _bad_rows_drain_due:
                        _bad_rows_drain_due = False
                        self._warn_sampling_bad_rows(_sampling_residency)
                last_boundary = local_iteration
                force_host_boundary = False

                (decode_uuids, batch,
                 pending_async_task, pending_load_uuids,
                 pending_load_local, pending_load_global,
                 timing, watermark_triggered) = self._page_boundary_fast(
                    decode_uuids, batch, gpu_manager,
                    pending_async_task, pending_load_uuids,
                    pending_load_local, pending_load_global
                )
                # `batch` has been rebound to a new ordered row list; anything
                # keyed to the previous rows (the resident sampling tensors) is
                # now stale.
                self._bump_decode_row_epoch()

                self._cumulative_boundary_ms += timing.total_ms
                self._cumulative_decode_boundaries += 1

                # Batch may have changed - need to verify page table
                _page_table_verified_this_batch = False

                # Post-boundary: verify page table matches batch and fix if needed
                if batch and gpu_manager and gpu_manager.is_initialized and gpu_manager._gpu_page_table_manager:
                    post_boundary_slot_order = list(gpu_manager._gpu_page_table_manager.slot_to_seq_id) if gpu_manager._gpu_page_table_manager.slot_to_seq_id else []
                    post_boundary_batch_global_ids = self._local_indices_to_global_seq_ids(batch)

                    if post_boundary_slot_order != post_boundary_batch_global_ids:
                        # Fix: Rebuild page table to match batch
                        gpu_manager.rebuild_page_table(post_boundary_batch_global_ids)

                # Page table is now verified for this batch
                _page_table_verified_this_batch = True

                # Check if watermark triggered - interrupt decode for prefill
                if watermark_triggered:
                    # CRITICAL FIX: Wait for pending KV append tasks BEFORE going ON_HOLD!
                    # Without this, KV data may not be fully written to host when sequences
                    # are later resumed, causing KV corruption and gibberish output.
                    num_waited = self._wait_pending_kv_append_tasks(sync_distributed_errors=True)
                    if num_waited > 0:
                        logging.info(
                            f"[WATERMARK-KV-SYNC] Rank {self.rank}: Waited for {num_waited} pending KV append tasks "
                            f"before putting sequences ON_HOLD"
                        )

                    # An in-flight boundary load must be integrated before the
                    # hold: its rows are not in decode_uuids yet, but they
                    # already hold GPU pages with registered shared-page keys.
                    # Dropping them here strands those pages and the row's
                    # next load trips the writable-page sharing guard
                    # (cohort-fatal SCHED_INVARIANT).
                    if pending_load_uuids:
                        decode_uuids, batch = self._integrate_pending_async_load(
                            pending_async_task, pending_load_uuids,
                            pending_load_local, pending_load_global,
                            decode_uuids, batch, gpu_manager,
                        )
                        pending_async_task = None
                        pending_load_uuids = []
                        pending_load_local = []
                        pending_load_global = []

                    logging.info(
                        f"[WATERMARK] Rank {self.rank}: Decode interrupted - putting {len(decode_uuids)} "
                        f"sequences ON_HOLD, will trigger prefill"
                    )
                    # Put all remaining sequences ON_HOLD
                    self._put_sequences_on_hold(decode_uuids)
                    # Exit decode loop - will return to generate() which will trigger prefill
                    break

                # Poll for new admissions at each page boundary.
                # New batches may have been submitted during decode — drain them
                # and break for prefill if QUEUEING sequences arrive.
                if self._admission_queue is not None:
                    admitted = self._poll_admissions()
                    if admitted and self.rank == 0:
                        logging.info(
                            f"[DECODE] Mid-decode admission at iter {self._cumulative_decode_iterations}, "
                            f"total in batch: {len(self.global_batch)}"
                        )
                    has_q = self.global_batch.has_queueing()
                    if BATCHGEN_MULTI_BATCH_DIAG and self.rank == 0 and has_q:
                        num_q = len(self.global_batch.get_sequences_by_status(SequenceStatus.QUEUEING))
                        logging.info(
                            f"[MULTI_DIAG] has_queueing={has_q} num_q={num_q} "
                            f"watermark={watermark_triggered} admitted={admitted}"
                        )
                    if has_q and watermark_triggered:
                        if self.rank == 0:
                            logging.info(f"[DECODE] Breaking for new batch prefill (watermark triggered)")
                        break

                # Detailed logging at every boundary (only rank 0)
                if self.rank == 0:
                    # Get status counts
                    # - in_decode: sequences currently in decode batch (IN_DECODE status)
                    # - onhold: sequences paused with host KV (ON_HOLD status)  
                    # - prefilled: sequences prefilled but not yet decoding (PREFILLED status)
                    # - host_kv_total: total sequences with host KV = prefilled + onhold + in_decode
                    num_in_decode = timing.total_active
                    num_onhold = len(self.global_batch.get_sequences_by_status(SequenceStatus.ON_HOLD))
                    num_prefilled = timing.total_prefilled
                    num_completed_total = timing.total_completed_cumulative
                    num_host_kv_total = num_prefilled + num_onhold + num_in_decode
                    
                    # Get page stats if available
                    page_info = ""
                    if hasattr(self, '_host_kv_page_stats') and self._host_kv_page_stats:
                        ps = self._host_kv_page_stats
                        page_info = f" | Host KV: {ps['used']}/{ps['total']} pages ({ps['free_percent']}% free)"

                    if BATCHGEN_CB_DEBUG:
                        # Detailed timing log when debug is enabled
                        logging.info(
                            f"[Decode Interval {self._cumulative_decode_boundaries}] "
                            f"iter={self._cumulative_decode_iterations}, "
                            f"total={timing.total_ms:.1f}ms | "
                            f"wait_kv={timing.wait_kv_append_ms:.1f}({timing.num_kv_append_tasks}), "
                            f"wait_async={timing.wait_async_load_ms:.1f}, "
                            f"finalize={timing.finalize_load_ms:.1f}, "
                            f"sync_uuids={timing.sync_decode_uuids_ms:.1f}, "
                            f"gather={timing.gather_ms:.1f}, "
                            f"proc={timing.process_ms:.1f}, "
                            f"ext={timing.extension_ms:.1f}, "
                            f"load_sel={timing.load_select_ms:.1f}, "
                            f"load_alloc={timing.load_alloc_ms:.1f}, "
                            f"load_launch={timing.load_launch_ms:.1f}, "
                            f"rebuild={timing.rebuild_ms:.1f}, "
                            f"moe_buf={timing.moe_buffer_update_ms:.1f}, "
                            f"barrier={timing.barrier_ms:.1f}ms | "
                            f"STATUS: in_decode={num_in_decode}, onhold={num_onhold}, prefilled={num_prefilled}, "
                            f"host_kv_total={num_host_kv_total}, completed={num_completed_total}/{global_batch_size}, "
                            f"Δ completed={timing.num_completed}, loaded={timing.num_loaded}, onhold={timing.num_onhold}"
                            f"{page_info}"
                        )
                    else:
                        # Minimal log without timing details
                        logging.info(
                            f"[Decode {self._cumulative_decode_boundaries}] iter={self._cumulative_decode_iterations} | "
                            f"STATUS: in_decode={num_in_decode}, onhold={num_onhold}, prefilled={num_prefilled}, "
                            f"host_kv_total={num_host_kv_total}, completed={num_completed_total}/{global_batch_size}, "
                            f"Δ completed={timing.num_completed}, loaded={timing.num_loaded}, onhold={timing.num_onhold}"
                            f"{page_info}"
                        )
                
                if not decode_uuids:
                    # Check for pending loads
                    if pending_load_uuids:
                        decode_uuids, batch = self._integrate_pending_async_load(
                            pending_async_task, pending_load_uuids,
                            pending_load_local, pending_load_global,
                            decode_uuids, batch, gpu_manager,
                        )
                        self._rebuild_page_table_for_batch(batch, gpu_manager)
                        self._sync_decode_moe_rank_counts(
                            batch,
                            reason="post_pending_load_finalize",
                        )
                        
                        if batch:
                            new_tokens = self._rebuild_input_tokens(batch)
                        
                        pending_async_task = None
                        pending_load_uuids = []
                        pending_load_local = []
                        pending_load_global = []
                        
                        if decode_uuids:
                            force_host_boundary = self._host_capacity_due_all_ranks(batch)
                            continue
                    break

                # Newly loaded ON_HOLD/PREFILLED rows may have a reservation that
                # ends exactly at their current context.  Run the planner once more
                # before forward; the check is collective but boundary-rate, not
                # token-rate.
                force_host_boundary = self._host_capacity_due_all_ranks(batch)
                if force_host_boundary:
                    continue
                
                new_tokens = self._rebuild_input_tokens(batch)
                # DEBUG: Log tokens rebuild after boundary
                if new_tokens.shape[0] != len(batch):
                    logging.error(
                        f"Rank {self.rank}: POST-BOUNDARY new_tokens mismatch! "
                        f"batch_size={len(batch)}, new_tokens.shape={new_tokens.shape}"
                    )
            
            # Forward pass
            forward_start = time.perf_counter()

            # Pre-compute batch_sequences for use in both forward setup and update loop
            batch_sequences = [self.global_batch.get_sequence(self._local_to_uuid_map[idx]) for idx in batch] if batch else []
            global_decode_sequences = self._debug_sequences_for_decode_uuids(decode_uuids)
            _probe_mismatch_logged = binder._read_decode_step_model_metadata(
                global_decode_sequences,
                batch,
                _probe_adapter,
                _probe_round_request,
                _probe_mismatch_logged,
            )

            # Phase C: MoE-only graph mode retired; no MoE-specific warmup needed.

            # Invariant check: cache_seqlens must not exceed allocated pages.
            # Violations cause FlashAttention to read -1 sentinel → CUDA illegal access.
            if BATCHGEN_DECODE_ASSERT and batch:
                for seq in batch_sequences:
                    max_tokens = seq.gpu_pages_allocated * SequenceEntry.PAGE_SIZE
                    if seq.current_context_length > max_tokens:
                        logging.error(
                            f"DECODE_ASSERT FAIL rank={self.rank}: {seq.uuid[:8]} gid={seq.global_idx} "
                            f"ctx={seq.current_context_length} > max_tokens={max_tokens} "
                            f"(pages={seq.gpu_pages_allocated}, PAGE_SIZE={SequenceEntry.PAGE_SIZE}, "
                            f"prompt={seq.prompt_length}, orig_prompt={seq.original_prompt_length}, "
                            f"decoded={seq.decoded_length}, baseline={seq.reentry_decoded_baseline}, "
                            f"status={seq.status})"
                        )
                        raise RuntimeError(
                            f"cache_seqlens overrun: ctx={seq.current_context_length} > "
                            f"pages={seq.gpu_pages_allocated}×{SequenceEntry.PAGE_SIZE}="
                            f"{max_tokens} for {seq.uuid[:8]}"
                        )

            _split_t0 = time.perf_counter()
            # Start marker of this step's device forward+sample time. Takes a
            # free ring slot or skips timing this step; never waits.
            _fwd_slot = _fwd_ring.begin(torch.cuda.current_stream(self.torch_device))
            # Probe level of this step; stays None unless the graph replays.
            _probe_tag = None
            with torch.inference_mode():
                if batch:
                    # Collect context lengths with invariant validation
                    # ALWAYS: current_context_length == original_prompt_length + decoded_length
                    cache_seqlens = []
                    for seq in batch_sequences:
                        ctx_len = seq.current_context_length
                        expected = seq.original_prompt_length + seq.decoded_length
                        if ctx_len != expected:
                            logging.error(
                                f"Rank {self.rank}: CTX MISMATCH {seq.uuid[:8]} gid={seq.global_idx}: "
                                f"ctx={ctx_len} expected={expected} (orig_prompt={seq.original_prompt_length}, "
                                f"prompt={seq.prompt_length}, decoded={seq.decoded_length})"
                            )
                            seq.log_event(SeqEvent.CTX_MISMATCH, self.rank,
                                f"ctx={ctx_len}, expected={expected}, prompt={seq.prompt_length}")
                            lifespan.dump_lifespan(seq.uuid, seq.global_idx, seq._lifespan_log, "CTX_MISMATCH")
                            seq.current_context_length = expected
                            ctx_len = expected
                        cache_seqlens.append(ctx_len)

                    max_ctx = max(cache_seqlens)

                    # DIAG: Log cache_seqlens at first iteration of each decode group
                    if BATCHGEN_MULTI_BATCH_DIAG and self.rank == 0 and local_iteration <= 1:
                        fresh = [(s.uuid[:8], s.decoded_length, ctx) for s, ctx in zip(batch_sequences, cache_seqlens) if s.decoded_length <= 1]
                        resumed = [(s.uuid[:8], s.decoded_length, ctx, s.gpu_pages_allocated) for s, ctx in zip(batch_sequences, cache_seqlens) if s.decoded_length > 1]
                        logging.info(
                            f"[MULTI_DIAG] decode_group={self._decode_group_idx} iter={local_iteration}: "
                            f"batch={len(batch)}, fresh={len(fresh)}, resumed={len(resumed)}, "
                            f"max_ctx={max_ctx}"
                        )
                        for uid, dl, ctx in fresh[:5]:
                            logging.info(f"[MULTI_DIAG]   FRESH: {uid} decoded={dl} cache_seqlen={ctx}")
                        for uid, dl, ctx, pg in resumed[:5]:
                            logging.info(f"[MULTI_DIAG]   RESUMED: {uid} decoded={dl} cache_seqlen={ctx} gpu_pages={pg}")

                new_tokens, _page_table_verified_this_batch = binder._rebind_decode_replay_model_metadata(
                    batch_sequences,
                    cache_seqlens,
                    max_ctx,
                    batch,
                    new_tokens,
                    _page_table_verified_this_batch,
                    gpu_manager,
                )
                
                # NOTE: Do NOT skip forward pass even with empty batch!
                # MoE models have all-to-all collective operations that ALL ranks must participate in.
                # Skipping would cause deadlock as other ranks wait for this rank.

                # MoE buffer sync: only needed at decision boundaries (batch size changes).
                # Between boundaries, batch size is constant — skip the all_reduce + .item()
                # CPU-GPU sync that drains the GPU pipeline every step.
                # The sync is done in _page_boundary_fast and at initial setup (line ~7099).
                # Phase C: layer-graph mode retired; only whole-model needs the
                # globally-synced rank-count reuse.
                if (
                    self._graph.whole_model_graph
                    or self._glm5_whole_model_graph_requested_for_current_batch()
                ):
                    # Whole-model graph needs globally synced counts for NCCL bucket
                    # matching, but the count vector only changes at decode-entry,
                    # page-boundary, and async-load-finalize sync points. Reusing it
                    # avoids a per-token NCCL all_gather + D2H .item() sync.
                    _all_rank_counts = getattr(worker, "_current_decode_rank_token_counts", None)
                    _cached_local_bsz = int(getattr(worker, "_current_decode_local_batch_size", -1))
                    _max_bs = int(getattr(worker, "_current_decode_max_rank_batch_size", 0) or 0)
                    if _all_rank_counts is None or _max_bs <= 0 or _cached_local_bsz != len(batch):
                        _max_bs = self._sync_decode_moe_rank_counts(
                            batch,
                            reason="decode_step_batch_change",
                        )
                        _all_rank_counts = getattr(worker, "_current_decode_rank_token_counts", None)
                    _max_bs = max(int(_max_bs), 1)
                else:
                    # Per-layer graph or eager: no NCCL in graph, use local batch size
                    _max_bs = max(len(batch), 1)
                    _all_rank_counts = None

                binder._install_decode_kv_append_callbacks(batch, batch_sequences)
                # Phase C: layer-graph mode retired; only the whole-model graph
                # may need re-warmup here.
                if self._glm5_whole_model_graph_current_bucket_missing():
                    logging.info(
                        f"Rank {self.rank}: GLM-5 whole-model CUDA graph was not captured "
                        "during decode configuration; using eager decode instead of "
                        "capturing in the decode loop"
                    )
                    self._graph.glm5_whole_model_graph_capture_attempted_for_batch = True

                self._log_glm5_graph_path_for_forward(
                    local_bsz=len(batch),
                    max_rank_bsz=int(getattr(worker, "_current_decode_max_rank_batch_size", 0) or 0),
                    rank_counts=getattr(worker, "_current_decode_rank_token_counts", None),
                    gpu_manager=gpu_manager,
                    decode_iter=self._cumulative_decode_iterations,
                )
                # Phase C: DSA-only graph metadata prep retired (DSA graph mode
                # is no longer reachable). The whole-model graph's FA3 sparse
                # decode needs no per-forward attention scheduler metadata.

                _nsys_forward_idx = self._nsys_decode_profile_begin_forward(
                    local_iteration=local_iteration,
                    local_bsz=len(batch),
                    max_rank_bsz=int(getattr(worker, "_current_decode_max_rank_batch_size", 0) or 0),
                )

                # Forward
                # Phase C: layer-graph mode retired. The whole-model graph
                # composes per-layer captures internally; no separate
                # layer-graph dispatch is needed.

                _glm5_whole_graph_active = bool(
                    self._graph.glm5_whole_model_graph
                    and self._graph.cuda_graph_manager is not None
                )
                _glm5_whole_graph_over_bucket = False
                if _glm5_whole_graph_active:
                    try:
                        _glm5_whole_bucket = self._graph.whole_model_bucketing.get_padded_size(_max_bs)
                    except ValueError:
                        _glm5_whole_graph_over_bucket = True
                        _glm5_whole_graph_active = False
                    else:
                        # An all_short capture never runs indexer scoring, so a
                        # step whose longest context exceeds index_topk must NOT
                        # replay it: top-k selection would be silently skipped.
                        # Leaving the graph here either falls back to eager
                        # (correct top-k) or, under a required graph, raises
                        # below with reason=all_short_exceeded.
                        _glm5_all_short_exceeded = bool(
                            self._graph.glm5_whole_model_all_short
                            and int(self._graph.glm5_whole_model_index_topk or 0)
                            and binder._read_decode_max_seqlen()
                            > int(self._graph.glm5_whole_model_index_topk)
                        )
                        _glm5_whole_graph_active = (
                            not _glm5_all_short_exceeded
                            and _glm5_whole_bucket not in self._graph.glm5_whole_model_graph_failed_buckets
                            and self._graph.cuda_graph_manager.has_bucket_for_all_segments(_max_bs)
                            and binder._read_decode_max_seqlen() <= int(getattr(self._graph.whole_model_segment, "max_seqlen", 0))
                            and self._glm5_whole_model_graph_capture_signature(_glm5_whole_bucket) == self._graph.glm5_whole_model_graph_signature
                        )
                elif self._glm5_whole_model_graph_requested_for_current_batch():
                    try:
                        configured_max_bucket = int(self.args.cuda_graph_max_bucket_size)
                    except AttributeError:
                        pass
                    else:
                        _glm5_whole_graph_over_bucket = _max_bs > configured_max_bucket
                if (
                    self._glm5_whole_model_graph_requested_for_current_batch()
                    and not _glm5_whole_graph_active
                    and not _glm5_whole_graph_over_bucket
                ):
                    _, _required_bucket, _required_reason = self._glm5_whole_graph_path_state(_max_bs)
                    if _required_reason in self._WHOLE_GRAPH_EAGER_SERVABLE_REASONS:
                        # A step longer than the captured span runs eager --
                        # correct, just slower. Every rank evaluates this from
                        # the globally-synced `_max_bs` and its own identical
                        # span, so all ranks take this branch together and the
                        # graph's baked NCCL ops stay matched.
                        self._warn_decode_graph_span_exceeded(
                            binder._read_decode_max_seqlen(),
                            int(getattr(self._graph.whole_model_segment, "max_seqlen", 0) or 0),
                        )
                    else:
                        raise RuntimeError(
                            "GLM-5 whole-model CUDA graph was required but no replayable "
                            "whole-model graph is available for this decode step "
                            f"(bucket={_required_bucket}, reason={_required_reason})"
                        )
                _use_graph = (
                    self._graph.whole_model_graph
                    and self._graph.cuda_graph_manager is not None
                    and _max_bs <= self._graph.whole_model_bucketing._max_bucket
                    and (
                        not self._graph.glm5_whole_model_graph
                        or _glm5_whole_graph_active
                    )
                )
                if _use_graph:
                    _glm5_whole_compare = bool(
                        self._graph.glm5_whole_model_graph
                        and self._glm5_whole_model_graph_compare_requested_for_current_batch()
                    )
                    _glm5_whole_timing = bool(
                        self._graph.glm5_whole_model_graph
                        and self._glm5_whole_model_graph_timing_requested_for_current_batch()
                    )
                    _glm5_whole_timing_items = {}
                    _glm5_skip_graph_kv_offload = False
                    # Whole-model CUDA graph replay.
                    # CRITICAL: Use _max_bs (globally-synced max batch size) for bucket
                    # computation, NOT local len(batch). The graph has NCCL all_reduce
                    # baked inside — all ranks MUST replay the same bucket's graph,
                    # otherwise mismatched NCCL ops cause deadlock.
                    batch_size = len(batch)
                    bucket = self._graph.whole_model_bucketing.get_padded_size(_max_bs)
                    # An adapter, when present, routes the GLM-5 whole-model replay
                    # (the Phase-B dual gate is retired: it was always on).
                    # K2.5 always routes whole-model replay through the adapter
                    # (no legacy non-adapter K2.5 whole-model path exists).
                    _k25_whole_active = self._graph.k25_whole_model_graph
                    _adapter_dual_active = (
                        self._graph.cuda_graph_adapter is not None
                        and (
                            _k25_whole_active
                            or self._graph.glm5_whole_model_graph
                        )
                    )
                    _wm_seg_name = self._decode_whole_model_segment_name()
                    _adapter_decision = None
                    _adapter_batch_state = None
                    if _adapter_dual_active:
                        _adapter_batch_state = self._graph.make_batch_state(
                            local_bsz=batch_size,
                            max_rank_bsz=_max_bs,
                            rank_token_counts=_all_rank_counts,
                            cache_seqlens=binder._read_decode_cache_seqlens(),
                            position_ids=binder._read_decode_position_ids(),
                            max_seqlen=binder._read_decode_max_seqlen(),
                            cur_batch_sequence_ids=binder._read_decode_cur_batch(),
                            gpu_kv_manager=gpu_manager,
                            decode_iter=0,
                            input_ids=new_tokens,
                            device=self.torch_device,
                        )
                        _adapter_decision = self._graph.cuda_graph_adapter.eligibility(_adapter_batch_state)
                        if _adapter_decision.mode.value != "whole_model":
                            logging.info(
                                "Phase B: adapter eligibility=%s/%s; using legacy path for parity",
                                _adapter_decision.mode.value, _adapter_decision.reason,
                            )
                            _adapter_dual_active = False
                    if _adapter_dual_active:
                        # ADAPTER PATH (Phase B dual gate)
                        replay_inputs = self._graph.cuda_graph_adapter.prepare_replay_inputs(
                            decision=_adapter_decision,
                            batch_state=_adapter_batch_state,
                            segment_name=_wm_seg_name,
                        )
                        # Delay-table update for this step's level: an async copy
                        # on this stream, ahead of the replay that reads it.
                        if _probe is not None:
                            _probe_tag = _probe.before_replay(local_iteration)
                        if _glm5_whole_timing:
                            torch.cuda.synchronize(self.torch_device)
                            _glm5_replay_start = time.perf_counter()
                        graph_out = self._graph.cuda_graph_manager.replay(
                            _wm_seg_name, _max_bs, **replay_inputs,
                        )
                        if _glm5_whole_timing:
                            torch.cuda.synchronize(self.torch_device)
                            _glm5_whole_timing_items["replay_ms"] = (
                                time.perf_counter() - _glm5_replay_start
                            ) * 1000.0
                        # K2.5 whole-model eager-vs-graph compare via the standard
                        # model-agnostic facility (observability-only; does NOT change
                        # token selection). GLM-5 keeps its own bespoke compare block
                        # below; this path serves any adapter that has no such block.
                        if _k25_whole_active:
                            _dbg = self._graph.cuda_graph_adapter.debug_options(_adapter_batch_state)
                            if _dbg.compare_against_eager:
                                # graph_out has padded bucket rows; the eager reference
                                # produces local_bsz (=batch_size) rows. Slice to align.
                                # Graph probe keys are `probe_layer_<NNN>_hidden`; the
                                # eager reference uses `hidden_states_layer_<i>` — remap
                                # so the diff aligns by key.
                                _graph_cmp = {}
                                for _k, _v in graph_out.items():
                                    _vs = _v[:batch_size]
                                    if _k.startswith("probe_layer_") and _k.endswith("_hidden"):
                                        _idx = int(_k[len("probe_layer_"):-len("_hidden")])
                                        _graph_cmp[f"hidden_states_layer_{_idx}"] = _vs
                                    else:
                                        _graph_cmp[_k] = _vs
                                _report = self._graph.compare_decode_outputs(
                                    adapter=self._graph.cuda_graph_adapter,
                                    decision=_adapter_decision,
                                    batch_state=_adapter_batch_state,
                                    segment_name=_wm_seg_name,
                                    captured_inputs=replay_inputs,
                                    graph_outputs=_graph_cmp,
                                    probe_layers=_dbg.probe_layers,
                                    atol=_dbg.compare_atol,
                                    rtol=_dbg.compare_rtol,
                                    fail_on_mismatch=_dbg.fail_on_mismatch,
                                )
                                _log = logging.info if _report.passed else logging.error
                                _log(
                                    "[K25_WHOLE_GRAPH_COMPARE] rank=%s bucket=%s batch=%s "
                                    "status=%s max_abs=%.6g max_rel=%.6g mismatched=%s "
                                    "probes=%s",
                                    self.rank, _adapter_decision.bucket, batch_size,
                                    "OK" if _report.passed else "MISMATCH",
                                    _report.max_abs, _report.max_rel,
                                    _report.mismatched_keys, _report.probe_results,
                                )
                    elif self._graph.glm5_whole_model_graph:
                        primary_manager = getattr(gpu_manager, "primary", gpu_manager)
                        aux_manager = getattr(
                            gpu_manager,
                            "auxiliary",
                            getattr(self.core_engine, "gpu_paged_kv_manager_aux", None),
                        )
                        if aux_manager is None:
                            raise RuntimeError("GLM-5 whole-model graph replay requires auxiliary GPU KV manager")
                        graph_inputs = self._prepare_glm5_layer_graph_inputs(
                            local_bsz=batch_size,
                            bucket=bucket,
                            gpu_manager=gpu_manager,
                            graph_max_seqlen_override=int(getattr(self._graph.whole_model_segment, "max_seqlen", 0) or 0),
                        )
                        # Delay-table update for this step's level: an async copy
                        # on this stream, ahead of the replay that reads it.
                        if _probe is not None:
                            _probe_tag = _probe.before_replay(local_iteration)
                        if _glm5_whole_timing:
                            torch.cuda.synchronize(self.torch_device)
                            _glm5_replay_start = time.perf_counter()
                        graph_out = self._graph.cuda_graph_manager.replay(
                            "glm5_whole_model", _max_bs,
                            input_ids=new_tokens[:batch_size],
                            cache_seqlens=graph_inputs["cache_seqlens"],
                            position_ids=graph_inputs["position_ids"],
                            primary_slot_indices=graph_inputs["primary_slot_indices"],
                            aux_slot_indices=graph_inputs["aux_slot_indices"],
                            rank_token_counts=_all_rank_counts,
                            num_valid_tokens=graph_inputs["num_valid_tokens"],
                        )
                        if _glm5_whole_timing:
                            torch.cuda.synchronize(self.torch_device)
                            _glm5_whole_timing_items["replay_ms"] = (
                                time.perf_counter() - _glm5_replay_start
                            ) * 1000.0
                    else:
                        # A rank with no decode rows (it may never have built an
                        # active page table) passes zero-row inputs and replays
                        # the same bucket as an all-padding batch, so the in-graph
                        # MoE collectives still line up across ranks.
                        _pt_mgr = gpu_manager._gpu_page_table_manager
                        pt_slice, slot_slice = self._whole_model_replay_page_inputs(
                            active_page_table=_pt_mgr.gpu_table,
                            slot_index_tensor=_pt_mgr._slot_index_tensor,
                            batch_size=batch_size,
                            max_pages_per_seq=self._graph.whole_model_segment.max_pages_per_seq,
                            device=self.torch_device,
                        )
                        graph_out = self._graph.cuda_graph_manager.replay(
                            "whole_model", bucket,
                            input_ids=new_tokens,
                            cache_seqlens=binder._read_decode_cache_seqlens()[:batch_size],
                            page_table=pt_slice,
                            slot_indices=slot_slice,
                        )

                    logits = graph_out["logits"][:batch_size]
                    graph_hidden_states = graph_out.get("hidden_states")
                    if graph_hidden_states is not None:
                        graph_hidden_states = graph_hidden_states[:batch_size]
                    if _glm5_whole_compare:
                        graph_probe_hidden_states = {
                            key: value[:batch_size]
                            for key, value in graph_out.items()
                            if key.startswith("probe_layer_")
                        }
                        graph_tokens_for_compare = torch.argmax(logits, dim=-1, keepdim=True)
                        if _glm5_whole_timing:
                            torch.cuda.synchronize(self.torch_device)
                            _glm5_eager_start = time.perf_counter()
                        with self._glm5_force_segmented_graph_eager():
                            if getattr(self._graph.whole_model_segment, "compare_probe_layers", ()):
                                eager_probe_outputs = self._graph.whole_model_segment.run_model_with_probes(
                                    input_ids=new_tokens,
                                    attention_mask=binder._read_decode_attention_mask(),
                                    position_ids=binder._read_decode_position_ids(),
                                    use_layer_segments=False,
                                )
                                eager_hidden_states = eager_probe_outputs["hidden_states"]
                                eager_logits = eager_probe_outputs["logits"]
                                eager_probe_hidden_states = {
                                    key: value
                                    for key, value in eager_probe_outputs.items()
                                    if key.startswith("probe_layer_")
                                }
                            else:
                                eager_model_outputs = self.model.model(
                                    input_ids=new_tokens,
                                    attention_mask=binder._read_decode_attention_mask(),
                                    position_ids=binder._read_decode_position_ids(),
                                    use_cache=False,
                                )
                                eager_hidden_states = eager_model_outputs[0][:, -1, :]
                                eager_logits = self.model.lm_head(eager_model_outputs[0])[:, -1, :]
                                eager_probe_hidden_states = {}
                        if _glm5_whole_timing:
                            torch.cuda.synchronize(self.torch_device)
                            _glm5_whole_timing_items["eager_ms"] = (
                                time.perf_counter() - _glm5_eager_start
                            ) * 1000.0
                        eager_tokens_for_compare = torch.argmax(eager_logits, dim=-1, keepdim=True)
                        new_tokens_out = self._select_tokens(eager_logits, batch_sequences)
                        compare = self._compare_glm5_whole_model_graph_logits(
                            eager_logits=eager_logits,
                            graph_logits=logits,
                            eager_hidden_states=eager_hidden_states,
                            graph_hidden_states=graph_hidden_states,
                            eager_probe_hidden_states=eager_probe_hidden_states,
                            graph_probe_hidden_states=graph_probe_hidden_states,
                            eager_tokens=eager_tokens_for_compare,
                            graph_tokens=graph_tokens_for_compare,
                        )
                        _log = logging.info if compare["ok"] else logging.error
                        _log(
                            "[GLM5_WHOLE_GRAPH_COMPARE] rank=%s bucket=%s batch=%s status=%s "
                            "max_abs=%.6g mean_abs=%.6g hidden_max_abs=%.6g "
                            "hidden_mean_abs=%.6g probe_first_mismatch=%s "
                            "probe_max_abs=%.6g probe_mean_abs=%.6g "
                            "argmax_mismatch=%s token_mismatch=%s",
                            self.rank,
                            bucket,
                            batch_size,
                            "OK" if compare["ok"] else "MISMATCH",
                            compare["max_abs"],
                            compare["mean_abs"],
                            compare["hidden_max_abs"],
                            compare["hidden_mean_abs"],
                            compare["probe_first_mismatch"],
                            compare["probe_max_abs"],
                            compare["probe_mean_abs"],
                            compare["argmax_mismatch"],
                            compare["token_mismatch"],
                        )
                        if not compare["ok"] and self._glm5_whole_model_graph_compare_fail_on_mismatch():
                            raise RuntimeError(f"GLM-5 whole-model CUDA graph compare mismatch: {compare}")
                        _glm5_skip_graph_kv_offload = True
                    else:
                        new_tokens_out = self._select_tokens(
                            logits,
                            batch_sequences,
                            residency=_sampling_residency,
                            batch_rows=batch,
                        )

                    if not _glm5_skip_graph_kv_offload and _adapter_dual_active:
                        # Phase B: adapter owns post-graph KV staging (audit §A finding #6:
                        # contiguous-only clone path; no per-layer fallback branch).
                        if _glm5_whole_timing:
                            _glm5_offload_start = time.perf_counter()
                        self._graph.cuda_graph_adapter.stage_post_graph_kv(
                            decision=_adapter_decision,
                            batch_state=_adapter_batch_state,
                            graph_outputs=graph_out,
                        )
                        if _glm5_whole_timing:
                            _glm5_whole_timing_items["offload_callback_ms"] = (
                                time.perf_counter() - _glm5_offload_start
                            ) * 1000.0
                    elif not _glm5_skip_graph_kv_offload:
                        if _glm5_whole_timing:
                            _glm5_offload_start = time.perf_counter()
                        # Fire KV host offload callbacks for all layers.
                        # KV buffers are static-address tensors written inside the graph.
                        # Stage primary and aux as two contiguous clones before async
                        # D2H; cloning per layer adds 156 small GPU copies per decode
                        # token on GLM-5 and dominates the whole-graph replay overhead.
                        kv_cb, aux_cb = binder._read_decode_kv_append_callbacks()
                        wm_seg = self._graph.whole_model_segment
                        if (
                            batch_size > 0
                            and kv_cb is not None
                            and wm_seg is not None
                            and wm_seg._kv_buffers is not None
                        ):
                            primary_stage = None
                            primary_key_buffer = getattr(wm_seg, "_kv_key_buffer", None)
                            if primary_key_buffer is not None:
                                primary_stage = primary_key_buffer[:, :batch_size].clone()
                            for layer_idx in range(wm_seg.num_layers):
                                kv_buf = wm_seg._kv_buffers[layer_idx]
                                # K2.5 MLA has no separate V cache — pass None for v_tensor
                                v_buf = kv_buf.get("value")
                                v_clone = v_buf[:batch_size].clone() if v_buf is not None and v_buf.numel() > 0 and not getattr(wm_seg, '_no_v_cache', False) else None
                                k_tensor = (
                                    primary_stage[layer_idx]
                                    if primary_stage is not None
                                    else kv_buf["key"][:batch_size].clone()
                                )
                                kv_cb(
                                    layer_idx,
                                    k_tensor,
                                    v_clone,
                                )
                        aux_buffers = getattr(wm_seg, "_aux_kv_buffers", None) if wm_seg is not None else None
                        if batch_size > 0 and aux_cb is not None and aux_buffers is not None:
                            aux_stage = None
                            aux_key_buffer = getattr(wm_seg, "_aux_kv_key_buffer", None)
                            if aux_key_buffer is not None:
                                aux_stage = aux_key_buffer[:, :batch_size].clone()
                            for layer_idx in range(wm_seg.num_layers):
                                aux_cb(
                                    layer_idx,
                                    aux_stage[layer_idx]
                                    if aux_stage is not None
                                    else aux_buffers[layer_idx]["key"][:batch_size].clone(),
                                    None,
                                )
                        if _glm5_whole_timing:
                            _glm5_whole_timing_items["offload_callback_ms"] = (
                                time.perf_counter() - _glm5_offload_start
                            ) * 1000.0
                    if _glm5_whole_timing:
                        logging.info(
                            "[GLM5_WHOLE_GRAPH_TIMING] rank=%s bucket=%s batch=%s replay_ms=%.3f "
                            "eager_ms=%.3f offload_callback_ms=%.3f compare=%s",
                            self.rank,
                            bucket,
                            batch_size,
                            _glm5_whole_timing_items.get("replay_ms", -1.0),
                            _glm5_whole_timing_items.get("eager_ms", -1.0),
                            _glm5_whole_timing_items.get("offload_callback_ms", -1.0),
                            _glm5_whole_compare,
                        )
                else:
                    # Per-layer graph or eager forward
                    # CRITICAL: Pass position_ids to model to ensure correct RoPE positioning during decode.
                    # Without this, the model generates position_ids = [[0]] for all decode steps,
                    # causing RoPE to be applied at position 0 instead of the actual token position.
                    # Graph replays above stream nothing; an eager step is one
                    # weight pass and carries the next step's while decode runs.
                    self._weight_passes.begin(carry=True)
                    outputs = self._glm5_decode_model_forward(new_tokens)
                    new_tokens_out = self._select_tokens(outputs.logits[:, -1, :], batch_sequences)
                    self._weight_passes.end()
                self._nsys_decode_profile_end_forward(_nsys_forward_idx)

            new_tokens = new_tokens_out
            _split_t1 = time.perf_counter()
            # End marker, queued on the same stream ahead of the token copy below.
            _fwd_ring.end(_fwd_slot, torch.cuda.current_stream(self.torch_device), tag=_probe_tag)

            if _deferred_readback:
                # [DEFERRED_READBACK] This step is enqueued; only now wait for
                # the PREVIOUS step's tokens, so its host tail (KV drain,
                # bookkeeping, next setup) runs while this forward runs.
                bs = new_tokens.shape[0]
                _rb_slot = _readback_ring.launch(
                    new_tokens, bs, torch.cuda.current_stream(self.torch_device)
                )
                _split_t2 = time.perf_counter()
                _rb_marks = []
                if _pending_step is not None:
                    _finish = self._finish_deferred_decode_step(
                        _pending_step, _readback_ring, overlapped=True, marks=_rb_marks
                    )
                    _pending_step = None
                    if _finish != _DECODE_WAIT_COMPLETED:
                        self._cohort_shutdown_observed = True
                        self._quiesce_decode_for_cohort_shutdown(pending_async_task)
                        return decode_uuids, batch
                else:
                    _rb_marks = [_split_t2, _split_t2, _split_t2]
                # One step's host-KV appends in flight at a time: the previous
                # step's were drained just above.
                self._flush_deferred_kv_to_host()
                _split_kv = time.perf_counter()
                _pending_step = self._begin_deferred_decode_step(
                    batch, batch_sequences, _rb_slot, bs, local_iteration
                )
                _split_end = time.perf_counter()
                self._cumulative_forward_ms += (_split_end - forward_start) * 1000
                accumulate_decode_step_split(
                    _step_split,
                    deferred_decode_step_marks(
                        forward_start, _split_t0, _split_t1, _split_t2,
                        _rb_marks[0], _rb_marks[1], _rb_marks[2],
                        _split_kv, _split_end,
                    ),
                )
                _harvested = _fwd_ring.harvest()
                if _probe is not None:
                    for _tag, _elapsed_ms in _harvested:
                        _probe.record(_tag, _elapsed_ms)
                from batchgen.timing import get_decode_timer
                _dt = get_decode_timer()
                if _dt and _dt.enabled:
                    _dt.log_summary()
                    _dt.reset()
                continue

            # P1: Non-blocking GPU→CPU token transfer via pinned memory. Record the
            # exact token-readback boundary before launching host-KV copies on their
            # independent D2H stream.
            bs = new_tokens.shape[0]
            if bs > _new_tokens_pinned.shape[0]:
                _new_tokens_pinned = torch.empty(bs, 1, dtype=torch.long, pin_memory=True)
            _new_tokens_pinned[:bs].copy_(new_tokens[:bs], non_blocking=True)
            _new_tokens_ready.record(torch.cuda.current_stream(self.torch_device))

            # Host-KV offload orders its own stream with a device-side event, so
            # launching it needs no host wait. The host first waits for the
            # sampled-token readback (needed for exact EOS and output bookkeeping),
            # then drains the append tasks below.
            self._flush_deferred_kv_to_host()
            _split_t2 = time.perf_counter()
            # --- The measured production wedge point, made pollable ---
            # This was a blocking `_new_tokens_ready.synchronize()`.  A rank
            # whose peer died mid-decode blocks there below Python and runs
            # nothing again: the teardown boundary below is unreachable, and
            # aborting the dead communicator does not free it.
            # _wait_decode_tokens keeps
            # that exact call, byte for byte, whenever the shutdown probe is
            # unarmed, and otherwise polls the event so the flag can be seen.
            _decode_wait = self._wait_decode_tokens(_new_tokens_ready)
            _split_t3 = time.perf_counter()

            # --- Decode-step teardown boundary (the abandoned token wait) ---
            # The wait gave up on an event that may never fire, so NOTHING this
            # step produced may be read: the sampled tokens are not in the
            # pinned buffer, and the step's host-KV appends are parked behind
            # the same dead device work.  That is why this exit sits BEFORE the
            # per-step drain below rather than after it -- that drain is
            # unbounded and ends in a device sync, i.e. the same unwakeable
            # state one level down.  The D1 invariant it exists for is kept
            # instead by _quiesce_decode_for_cohort_shutdown, which runs the
            # same local drain under a deadline and self-exits rather than
            # unwind with those unlocked page-table readers still live.
            #
            # As with the two flag checks, the exit RETURNS (a break would run
            # the post-loop drain, which syncs errors across ranks on the
            # aborted communicator) and latches, so the supervisor names this
            # ordered-via-decode-boundary.
            if _decode_wait == _DECODE_WAIT_ABANDONED:
                self._cohort_shutdown_observed = True
                self._quiesce_decode_for_cohort_shutdown(pending_async_task)
                return decode_uuids, batch

            # [KV_APPEND_DRAIN] Complete this step's host appends before the
            # next step. Each task is a std::async thread that reads the worker
            # view's page table on its own thread at execution time, with no
            # lock. Letting up to 256 of them stay in flight across steps meant
            # they could overlap the next admission wave's register/allocate on
            # the main thread; a torn read then lands one token's KV in a page
            # that now belongs to a freshly prefilled sequence. Measured on the
            # 512x4096 K3 contract: every victim was in a wave after the first
            # and collapsed at its 2nd or ~28th token, while the first wave (no
            # page-table mutation in flight) was clean. The drain sits after the
            # token readback only so its wait (which ends in a full device sync)
            # gets its own split slot instead of being charged to the copy
            # launch; nothing between the launch above and here touches a page
            # table. defer_errors=True keeps it a purely local wait (a guarded
            # collective would deadlock when only some ranks have tasks); the
            # deferred errors surface at the next boundary PHASE 0 or loop-exit
            # drain, both of which sync errors across ranks.
            if self._pending_kv_append_tasks:
                self._wait_pending_kv_append_tasks(defer_errors=True)

            # --- Decode-step teardown boundary (the measured wedge point) ---
            # The sync a few lines up is where a rank whose peer died mid-decode
            # is stuck, far below Python, and where the cohort abort releases it.
            # The release is SILENT: the sync returns normally and nothing raises
            # afterwards -- in graph mode nothing ever will -- so this flag test
            # is the only thing that can retire the freed rank.  It sits as close
            # to the sync as the step's own invariants allow, because the window
            # is finite: the rank resumes stepping and then re-wedges
            # permanently, with no second warning.
            #
            # It sits AFTER the drain above, not before it, and that ordering is
            # a correctness requirement rather than a preference.  The appends
            # this step launched are std::async threads holding unlocked
            # pointers into the worker view's page table, and the ordered
            # teardown this exit leads to releases that mapping.  Returning with
            # them still in flight is a use-after-free during teardown, so the
            # step always finishes draining what it started.
            #
            # The exit is a RETURN, not a break.  Breaking would fall into this
            # loop's post-loop cleanup, which drains with
            # sync_distributed_errors=True and so issues a fresh
            # dist.all_gather_object on the communicator that was just aborted
            # -- which the dead peer will never join.
            if self._cohort_shutdown_requested():
                self._cohort_shutdown_observed = True
                self._quiesce_decode_for_cohort_shutdown(pending_async_task)
                return decode_uuids, batch

            _split_t4 = time.perf_counter()
            # One host read of this step's sampled tokens for the whole
            # microbatch, replacing two `.item()` calls per row below.
            new_token_ids = _new_tokens_pinned[:bs].reshape(-1).tolist()
            if len(new_token_ids) < len(batch_sequences):
                raise RuntimeError(
                    f"decode sampled {len(new_token_ids)} tokens for "
                    f"{len(batch_sequences)} rows"
                )

            # [COMPLETED_ROW_SKIP] A row that is already retired must not
            # receive a trajectory token and must not be advanced below;
            # losing this skip writes a token into a retired row's book.
            # The predicate is still evaluated exactly once per row, and it
            # reads only that row's own fields, so deciding every row before
            # the appends cannot change any row's answer.
            _row_active = [
                not self._is_sequence_completed(seq) for seq in batch_sequences
            ]
            # One batched QueryBook append for the step, over the live rows
            # only, instead of one pybind call per row.
            self._append_trajectory_tokens(
                [seq for seq, live in zip(batch_sequences, _row_active) if live],
                [tok for tok, live in zip(new_token_ids, _row_active) if live],
            )

            # Update sequences (reuse batch_sequences from forward pass setup)
            for i, (local_idx, seq) in enumerate(zip(batch, batch_sequences)):
                if not _row_active[i]:
                    continue

                seq.decoded_length += 1
                seq.current_context_length = (
                    seq.original_prompt_length + seq.decoded_length
                )

                # Already on the host from this step's single tolist()
                token_id = new_token_ids[i]
                self._apply_decode_row_token(seq, token_id, i, local_iteration)

            _split_t5 = time.perf_counter()
            self._cumulative_forward_ms += (_split_t5 - forward_start) * 1000
            # Per-step CPU wall split (setup, fwd_launch, kv_launch, readback,
            # kv_drain, bookkeeping), without adding any device sync of its own.
            accumulate_decode_step_split(
                _step_split,
                (forward_start, _split_t0, _split_t1, _split_t2, _split_t3, _split_t4, _split_t5),
            )
            # Non-blocking harvest of finished forward timing pairs; elapsed
            # time is read only from pairs whose two events already completed.
            _harvested = _fwd_ring.harvest()
            if _probe is not None:
                for _tag, _elapsed_ms in _harvested:
                    _probe.record(_tag, _elapsed_ms)

            # Decode timing ablation (BATCHGEN_DECODE_TIMING=1)
            from batchgen.timing import get_decode_timer
            _dt = get_decode_timer()
            if _dt and _dt.enabled:
                _dt.log_summary()
                _dt.reset()

        # Cleanup
        if _pending_step is not None:
            # Every loop exit leaves through the page boundary, which drains
            # the deferred step; a step still pending here means an exit path
            # skipped that drain and its tokens would be lost.
            raise DeferredTokenReadbackError(
                f"Rank {self.rank}: decode loop exited with the sampled tokens of "
                f"local step {_pending_step.local_iteration} still unread"
            )
        self._wait_pending_kv_append_tasks(sync_distributed_errors=True)
        if pending_async_task is not None:
            pending_async_task.wait()
            torch.cuda.synchronize(self.torch_device)
        _step_now_s = time.perf_counter()
        _step_counters.end_round(_step_now_s)
        _step_counters.maybe_emit(_step_now_s, STEP_COUNTERS_INTERVAL_S)
        # Decode intervals shorter than the 30 s heartbeat (128-token probes,
        # small admission waves) would otherwise never report their step split.
        if self.rank == 0 and _step_split[0] > 0:
            logging.info(
                f"[DECODE] interval end: steps={int(_step_split[0])} "
                f"{format_decode_step_split(_step_split)}{_fwd_ring.take_summary()}"
            )
        if _probe is not None:
            # The loop exits after the last step's readback, so its pair has
            # completed; this harvest still never waits.
            for _tag, _elapsed_ms in _fwd_ring.harvest():
                _probe.record(_tag, _elapsed_ms)
            if self.rank == 0:
                logging.info(_probe.report())

        binder._teardown_decode_model_state()
        # Summary (uses cumulative counters for accurate cross-round totals)
        # Only show when BATCHGEN_CB_LOG=DEBUG
        if self.rank == 0 and self._cumulative_decode_boundaries > 0 and BATCHGEN_CB_DEBUG:
            avg_forward = self._cumulative_forward_ms / self._cumulative_decode_iterations if self._cumulative_decode_iterations > 0 else 0
            avg_round = self._cumulative_boundary_ms / self._cumulative_decode_boundaries
            logging.debug(
                f"\n{'='*50}\n"
                f"DECODE SUMMARY (Rank 0)\n"
                f"{'='*50}\n"
                f"Total Iterations: {self._cumulative_decode_iterations}, Total Rounds: {self._cumulative_decode_boundaries}\n"
                f"Avg forward: {avg_forward:.2f}ms\n"
                f"Avg round overhead: {avg_round:.2f}ms\n"
                f"Round overhead/token: {avg_round / self.DECISION_INTERVAL:.3f}ms\n"
                f"{'='*50}"
            )

        self.disable_decode_watchdog()
        return decode_uuids, batch

