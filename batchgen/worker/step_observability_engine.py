"""Step observability behind a worker seam (S7, StepObservability).

``StepObservabilityEngine`` owns what a worker rank logs and measures about
its own steps: the per-batch and per-phase log lines (batch statistics, HBM and
parameter residency, the phase memory ledger, the captured-span warning), the
Nsight Systems capture windows around prefill and decode forwards, the
phase-switch clock, and the prefix-cache metrics.

The engine never receives the worker (phase D2). It is a small real class
that owns the observability state the worker used to carry — the Nsight
decode-capture counters and the phase-switch clock — and every other input
arrives as a frozen ``DecodePhaseContext`` or as an explicit value the
worker's same-name delegator resolves. State the worker side still owns is
reached through the passed state object's own API (the prefix drain latch
through ``PrefixReuseState.mark_drain_metric_emitted``, the span-warn clock
through the ``GraphRuntime`` state object), never through the worker.

The engine-owned state is why this stays a class rather than the D1
module-function shape: a decode capture window and the phase clock persist
across calls, and their single owner is this engine instance (one per worker,
built in the worker's ``__init__`` and respawned with the worker pool exactly
as the former worker attributes were). The context stays the shared
``DecodePhaseContext`` from ``admission/context.py`` — unlike D1's
transport-local install inputs, every worker-owned object these methods read
is already a declared field there, so a module-local context would only
duplicate that declaration.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional, Set

import numpy as np
import torch
import torch.distributed as dist

from batchgen.config.graph_span_policy import GRAPH_CONTEXT_STATICS_BUDGET_BYTES
from batchgen.worker.admission.context import DecodePhaseContext


# Seconds between two "step exceeds the captured span" warnings. The condition
# holds for every step of a long sequence, so it is logged at a rate a human
# reads rather than once per step.
_GRAPH_SPAN_WARN_INTERVAL_S = 60.0


# Optional Nsight Systems capture window for decode-forward profiling.
# Start nsys with: --capture-range=cudaProfilerApi --capture-range-end=stop.
BATCHGEN_NSYS_DECODE_PROFILE = os.environ.get("BATCHGEN_NSYS_DECODE_PROFILE", "0") == "1"
BATCHGEN_NSYS_DECODE_PROFILE_FORWARD_LIMIT = int(
    os.environ.get("BATCHGEN_NSYS_DECODE_PROFILE_FORWARD_LIMIT", "3")
)
if BATCHGEN_NSYS_DECODE_PROFILE_FORWARD_LIMIT <= 0:
    raise ValueError("BATCHGEN_NSYS_DECODE_PROFILE_FORWARD_LIMIT must be positive")
BATCHGEN_NSYS_DECODE_PROFILE_EXIT = os.environ.get("BATCHGEN_NSYS_DECODE_PROFILE_EXIT", "1") == "1"

def _parse_nsys_controller_ranks() -> Set[int]:
    raw = os.environ.get("BATCHGEN_NSYS_DECODE_PROFILE_CONTROLLER_RANKS", "0")
    return {int(part.strip()) for part in raw.split(",") if part.strip()}

BATCHGEN_NSYS_DECODE_PROFILE_CONTROLLER_RANKS = _parse_nsys_controller_ranks()
if BATCHGEN_NSYS_DECODE_PROFILE and not BATCHGEN_NSYS_DECODE_PROFILE_CONTROLLER_RANKS:
    raise ValueError("BATCHGEN_NSYS_DECODE_PROFILE_CONTROLLER_RANKS must not be empty")


class StepObservabilityEngine:
    """Step-observability operations; the engine owns its capture/clock state."""

    def __init__(self) -> None:
        # Nsight decode-forward capture window (formerly worker attributes;
        # since D2 this engine is their single owner and writer).
        self._nsys_decode_profile_forward_count = 0
        self._nsys_decode_profile_started = False
        self._nsys_decode_profile_stopped = False
        # Phase-switch clock: [phase, running since (None while idle), busy
        # seconds so far]; None between _log_phase_switch and _mark_phase_end.
        self._phase_end: Optional[list] = None

    # ---- Logging -----------------------------------------------------------

    def _warn_sampling_bad_rows(self, residency) -> None:
        """Drain the device-side sampling fallback counter and warn on it.

        Reads a device counter, so callers must be where the device has caught
        up with the sampling that wrote it (see drain_bad_rows).
        """
        _bad_rows = residency.drain_bad_rows()
        if _bad_rows:
            logging.warning(
                f"[DECODE] invalid probabilities after filtering for "
                f"{_bad_rows} row(s) since the last heartbeat; those rows "
                "fell back to greedy"
            )

    def _log_batch_statistics(self, ctx: DecodePhaseContext) -> None:
        """
        Log statistics about the completed batch including:
        - Global batch size
        - Prompt lengths: min, max, mean, median, P95, P99
        - Decoded token lengths: min, max, mean, median, P95, P99

        Only called from rank 0.
        """
        if ctx.sched.pool is None:
            return

        # Gather all sequences
        prompt_lengths = []
        decoded_lengths = []
        for seq in ctx.sched.pool:
            prompt_lengths.append(seq.prompt_length)
            decoded_lengths.append(seq.decoded_length)

        if not prompt_lengths:
            logging.info("[BATCH STATS] No sequences in batch.")
            return

        # Convert to numpy for statistics
        prompt_arr = np.array(prompt_lengths)
        decoded_arr = np.array(decoded_lengths)

        # Compute statistics
        def compute_stats(arr: np.ndarray) -> dict:
            return {
                'min': int(np.min(arr)),
                'max': int(np.max(arr)),
                'mean': float(np.mean(arr)),
                'median': float(np.median(arr)),
                'p95': float(np.percentile(arr, 95)),
                'p99': float(np.percentile(arr, 99)),
            }

        prompt_stats = compute_stats(prompt_arr)
        decoded_stats = compute_stats(decoded_arr)
        batch_size = len(prompt_lengths)

        # Log formatted output
        logging.info(
            f"\n{'='*60}\n"
            f"BATCH STATISTICS\n"
            f"{'='*60}\n"
            f"  Global Batch Size: {batch_size}\n"
            f"\n"
            f"  Prompt Lengths:\n"
            f"    Min: {prompt_stats['min']:,}  Max: {prompt_stats['max']:,}\n"
            f"    Mean: {prompt_stats['mean']:,.1f}  Median: {prompt_stats['median']:,.1f}\n"
            f"    P95: {prompt_stats['p95']:,.1f}  P99: {prompt_stats['p99']:,.1f}\n"
            f"\n"
            f"  Decoded Token Lengths:\n"
            f"    Min: {decoded_stats['min']:,}  Max: {decoded_stats['max']:,}\n"
            f"    Mean: {decoded_stats['mean']:,.1f}  Median: {decoded_stats['median']:,.1f}\n"
            f"    P95: {decoded_stats['p95']:,.1f}  P99: {decoded_stats['p99']:,.1f}\n"
            f"{'='*60}"
        )

    def _log_prefill_entry_hbm(self, ctx: DecodePhaseContext, when: str) -> None:
        if not torch.cuda.is_available():
            return
        free_mem, _ = torch.cuda.mem_get_info(ctx.facts.local_rank)
        allocated = torch.cuda.memory_allocated(ctx.facts.local_rank) / 1e9
        reserved = torch.cuda.memory_reserved(ctx.facts.local_rank) / 1e9
        logging.info(
            f"[HBM] Rank {ctx.facts.rank} {when} configure_prefill: "
            f"free={free_mem/1e9:.2f}GB alloc={allocated:.2f}GB rsv={reserved:.2f}GB"
        )

    def _log_decode_parameter_residency(
        self, ctx: DecodePhaseContext, *, decode_model
    ) -> None:
        """One-line HBM breakdown of the decode model's resident parameters and
        buffers by top-level module kind (rank 0, once per decode config), so
        the non-expert residency — what competes with KV pool and KDA slots
        for concurrency — is visible next to the resident expert shard line."""
        model = decode_model
        if model is None or not hasattr(model, "named_parameters"):
            return
        try:
            by_kind = {}
            seen = set()
            for name, t in list(model.named_parameters()) + list(model.named_buffers()):
                if not t.is_cuda or t.data_ptr() in seen:
                    continue
                seen.add(t.data_ptr())
                parts = name.split(".")
                kind = parts[0]
                for i, part in enumerate(parts):
                    if part == "layers" and i + 2 < len(parts):
                        kind = parts[i + 2]
                        break
                    if part in ("embed_tokens", "lm_head", "norm"):
                        kind = part
                        break
                by_kind[kind] = by_kind.get(kind, 0) + t.numel() * t.element_size()
            total = sum(by_kind.values())
            items = sorted(by_kind.items(), key=lambda kv: -kv[1])
            logging.info(
                "[HBM] decode parameters/buffers resident on rank %s: %.2f GiB total | %s",
                ctx.facts.rank, total / (1024 ** 3),
                ", ".join(f"{k}={v / (1024 ** 3):.2f}" for k, v in items[:12]),
            )
        except Exception as e:  # informational only
            logging.debug("parameter residency breakdown failed: %r", e)

    def _log_phase_memory_ledger(
        self,
        ctx: DecodePhaseContext,
        label: str,
        *,
        decode_model,
        phase_mem_switch_index: int,
    ) -> None:
        """One line per phase-switch stage: who holds device memory right now.

        The rebuild path tears a phase instance down and stages the next one, so
        "every cycle gives back what it took" is an invariant with no owner: a
        slice retained per switch only shows up as an OOM several cycles later,
        at whichever allocation happens to be the largest (the KV pool remap's
        ``cuMemCreate``). Torch's own counters cannot see that slice — the KV
        pool is CUDA-VMM, the captured graphs own private pools, and the weight
        buffers are C++ — so the driver's free/total is reported beside them,
        and the holders that survive a model are named explicitly.

        Cheap: four driver queries and attribute reads, once per switch stage,
        on rank 0 only. No synchronize (the callers that need one do it) and no
        ``.item()``.
        """
        if ctx.facts.rank != 0 or not torch.cuda.is_available():
            return
        free_bytes, total_bytes = torch.cuda.mem_get_info(ctx.facts.local_rank)
        gib = 1024 ** 3
        pm = ctx.parallel_manager
        kda_slots, kda_live = ctx.sched.trajectories.kda_stats()
        gpu_kv = ctx.gpu_kv
        logging.info(
            "[PHASE_MEM] rank=%d stage=%s switch=%d driver_used=%.2f "
            "driver_total=%.2f torch_alloc=%.2f torch_reserved=%.2f "
            "model=%d pm_model=%d resident_ep=%d sp8_buffer=%d decode_graph=%d "
            "kv_pool_mapped=%s kda_slots=%d kda_live=%d",
            ctx.facts.rank,
            label,
            phase_mem_switch_index,
            (total_bytes - free_bytes) / gib,
            total_bytes / gib,
            torch.cuda.memory_allocated(ctx.facts.local_rank) / gib,
            torch.cuda.memory_reserved(ctx.facts.local_rank) / gib,
            int(decode_model is not None),
            int(getattr(pm, "model", None) is not None),
            int(bool(getattr(pm, "_resident_ep_built", False))),
            int(getattr(pm, "_streamed_sp8_buffer", None) is not None),
            int(getattr(pm, "_decode_graph", None) is not None),
            getattr(gpu_kv, "pool_is_mapped", "na"),
            kda_slots,
            kda_live,
        )

    def _warn_decode_graph_span_exceeded(
        self,
        ctx: DecodePhaseContext,
        step_max_seqlen: int,
        span: int,
        *,
        graph_state,
    ) -> None:
        """Rate-limited WARN naming this step's max_seqlen and the captured span."""
        now = time.monotonic()
        last = graph_state.decode_graph_span_warn_monotonic
        if last is not None and now - last < _GRAPH_SPAN_WARN_INTERVAL_S:
            return
        graph_state.decode_graph_span_warn_monotonic = now
        logging.warning(
            "Rank %s: the whole-model decode graph covers %d context tokens but "
            "this step's max_seqlen is %d; running the step in eager decode. The "
            "span is the deployment context bound trimmed by the graph's "
            "context-proportional statics budget (%d bytes).",
            ctx.facts.rank,
            int(span),
            int(step_max_seqlen),
            GRAPH_CONTEXT_STATICS_BUDGET_BYTES,
        )

    # ---- Nsight Systems capture windows ------------------------------------

    def _nsys_prefill_profile_begin(
        self, ctx: DecodePhaseContext,
        *,
        local_bsz: int,
        global_bsz: int,
        profile_enabled: bool,
    ) -> bool:
        """Start a batch-debug-gated nsys prefill capture window.

        ``profile_enabled`` is the worker-resolved value of the
        ``glm5_prefill_nsys_profile`` server debug flag.
        """
        if not profile_enabled:
            return False

        torch.cuda.synchronize(ctx.facts.device)
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        if ctx.facts.rank == 0:
            logging.info(
                "[NSYS_PREFILL_PROFILE] starting cuda profiler capture "
                "global_bsz=%s",
                global_bsz,
            )
            torch.cuda.cudart().cudaProfilerStart()
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        torch.cuda.nvtx.range_push(
            f"BatchGen_prefill_rank_{ctx.facts.rank}_local_bsz_{local_bsz}"
            f"_global_bsz_{global_bsz}"
        )
        return True

    def _nsys_prefill_profile_end(self, ctx: DecodePhaseContext, active: bool) -> None:
        """Finish one batch-debug-gated nsys prefill capture window."""
        if not active:
            return
        torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize(ctx.facts.device)
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        if ctx.facts.rank == 0:
            logging.info("[NSYS_PREFILL_PROFILE] stopping cuda profiler capture")
            torch.cuda.cudart().cudaProfilerStop()
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    def _nsys_decode_profile_begin_forward(
        self,
        *,
        rank: int,
        device,
        local_iteration: int,
        local_bsz: int,
        max_rank_bsz: int,
    ) -> Optional[int]:
        """Start an env-gated nsys decode-forward capture window.

        The capture counters live on this engine; ``rank`` and ``device`` are
        the worker identity values its delegator resolves.
        """
        if not BATCHGEN_NSYS_DECODE_PROFILE:
            return None
        self._nsys_decode_profile_forward_count += 1
        forward_idx = self._nsys_decode_profile_forward_count
        if forward_idx > BATCHGEN_NSYS_DECODE_PROFILE_FORWARD_LIMIT:
            return None

        if (
            rank in BATCHGEN_NSYS_DECODE_PROFILE_CONTROLLER_RANKS
            and not self._nsys_decode_profile_started
        ):
            torch.cuda.synchronize(device)
            logging.info(
                "[NSYS_DECODE_PROFILE] rank=%s starting cuda profiler capture "
                "limit=%s controller_ranks=%s",
                rank,
                BATCHGEN_NSYS_DECODE_PROFILE_FORWARD_LIMIT,
                sorted(BATCHGEN_NSYS_DECODE_PROFILE_CONTROLLER_RANKS),
            )
            torch.cuda.cudart().cudaProfilerStart()
            self._nsys_decode_profile_started = True

        range_name = (
            f"BatchGen_decode_forward_{forward_idx}"
            f"_rank_{rank}_local_bsz_{local_bsz}_max_rank_bsz_{max_rank_bsz}"
            f"_iter_{local_iteration}"
        )
        torch.cuda.nvtx.range_push(range_name)
        return forward_idx

    def _nsys_decode_profile_end_forward(
        self,
        forward_idx: Optional[int],
        *,
        rank: int,
        device,
    ) -> None:
        """End one env-gated nsys decode-forward range and optionally exit."""
        if forward_idx is None:
            return
        torch.cuda.nvtx.range_pop()
        if forward_idx < BATCHGEN_NSYS_DECODE_PROFILE_FORWARD_LIMIT:
            return

        torch.cuda.synchronize(device)
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        if (
            rank in BATCHGEN_NSYS_DECODE_PROFILE_CONTROLLER_RANKS
            and not self._nsys_decode_profile_stopped
        ):
            logging.info(
                "[NSYS_DECODE_PROFILE] rank=%s stopping cuda profiler capture "
                "after %s decode forwards",
                rank,
                forward_idx,
            )
            torch.cuda.cudart().cudaProfilerStop()
            self._nsys_decode_profile_stopped = True
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        if BATCHGEN_NSYS_DECODE_PROFILE_EXIT:
            # Local: only this diagnostic exit needs it.
            from batchgen.eventlog.process import exit_process

            logging.info(
                "[NSYS_DECODE_PROFILE] rank=%s exiting after %s profiled decode forwards",
                rank,
                forward_idx,
            )
            exit_process(0, "nsys profile exit")

    # ---- Phase clock -------------------------------------------------------

    def _mark_phase_end(self, phase: str) -> None:
        # [phase, running since (None while idle), busy seconds so far]
        self._phase_end = [phase, time.perf_counter(), 0.0]

    def _pause_phase_switch_clock(self) -> None:
        """Stop counting while the pool is idle and waits for new admissions."""
        end = self._phase_end
        if end is not None and end[1] is not None:
            end[2] += time.perf_counter() - end[1]
            end[1] = None

    def _resume_phase_switch_clock(self) -> None:
        end = self._phase_end
        if end is not None and end[1] is None:
            end[1] = time.perf_counter()

    def _log_phase_switch(
        self, phase: str, *, rank: int, persistent_phase: bool
    ) -> None:
        """Per-rank busy time from the previous phase's compute end to this phase's start."""
        previous = self._phase_end
        self._phase_end = None
        if previous is None or previous[0] == phase:
            return
        seconds = previous[2]
        if previous[1] is not None:
            seconds += time.perf_counter() - previous[1]
        logging.info(
            "[PHASE_SWITCH] rank=%d direction=%s_to_%s seconds=%.3f persistent=%s",
            rank,
            previous[0],
            phase,
            seconds,
            persistent_phase,
        )

    # ---- Prefix-cache metrics ----------------------------------------------

    def _emit_prefix_cache_metric(
        self, ctx: DecodePhaseContext, phase: str, **fields
    ) -> None:
        if not ctx.facts.prefix_cache_debug_stats:
            return
        from batchgen.prefix_reuse.metrics import emit_prefix_cache_metric

        emit_prefix_cache_metric(
            phase=phase,
            rank=ctx.facts.rank,
            **fields,
        )

    def _maybe_emit_prefix_drain_metric(self, ctx: DecodePhaseContext) -> None:
        """Emit one prefix_drain row when the live sequence set drains to empty.

        Called from _report_completion, which every completion path runs after
        the sequence reached COMPLETED and its Host pages and prefix
        attachments were released, so the row records active_attachments at the
        drain moment. Live means any status other than COMPLETED: a QUEUEING,
        IN_PREFILL, PREFILLED, IN_DECODE, ON_HOLD or EVICTED sequence can still
        hold or regain an attachment. Edge-triggered: one row per drain,
        re-armed by _admit_sequences_from_message.
        """
        if not (ctx.facts.enable_prefix_cache and ctx.facts.prefix_cache_debug_stats):
            return
        if ctx.sched.prefix.prefix_drain_metric_emitted:
            return
        if not ctx.sched.pool.all_completed():
            return
        ctx.sched.prefix.mark_drain_metric_emitted()
        self._emit_prefix_cache_metric(
            ctx,
            "prefix_drain",
            **self._prefix_host_resource_fields(ctx),
            # M3 Slice 2: live attachments must be zero at the drain, beside
            # the coordinator's own active_attachments. Absent on a run that
            # never turned the trajectory flag on.
            **ctx.sched.trajectories.restore_fields(),
        )

    def _prefix_host_resource_fields(
        self, ctx: DecodePhaseContext
    ) -> dict[str, int]:
        from batchgen.prefix_reuse.metrics import host_resource_fields

        return host_resource_fields(
            coordinator=ctx.prefix_coordinator,
            worker_view=ctx.core_engine.host_paged_kv_worker_view,
        )
