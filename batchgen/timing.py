"""Precise per-invocation timing for BatchGen decode/prefill.

Env flags:
  BATCHGEN_DECODE_TIMING=1            enable decode timer
  BATCHGEN_PREFILL_TIMING=1           enable prefill timer
  BATCHGEN_DECODE_TIMING_CSV=<path>   write one row per invocation
  BATCHGEN_DECODE_TIMING_INTERVAL=N   flush/log every N steps (default 50)
  BATCHGEN_DECODE_TIMING_RANKS=0,1    ranks that emit output (default "0")

Design (vs prior `torch.cuda.synchronize()` + `time.perf_counter()`):

  - `dt.timed(name, layer_idx)` queues a pair of `torch.cuda.Event(
    enable_timing=True)` markers on the current stream. No synchronize()
    on the hot path. Events resolve lazily in `step_done()`.
  - One event sync per decode step (on the step's last end-event), not
    one per timed block × layer.
  - Per-invocation rows live in `_records`: every (step, layer, op, call,
    elapsed_ms) is kept so downstream analysis can compute mean, p50,
    p90, p99 without losing per-call detail.

Public API preserved for drop-in compat with existing call sites
(`dt.timed("q_proj", li)` across `batchgen/models/glm/glm5/...`):
  - `get_decode_timer()`, `init_decode_timer(model_name, categories)`
  - `get_prefill_timer()`, `init_prefill_timer(model_name, categories)`
  - `TimingStats.timed(name, layer_idx)` context manager
  - `.enable() / .disable() / .reset() / .add_category() / .record()`
  - `.log_summary()` — now computes percentile breakdown from records
  - `.step_done()` — new; decode loop calls it once per step

Behavior when disabled (default): `timed()` is a pure `yield`, no event
allocation, no dict touch. Zero host overhead on the hot path.
"""
from __future__ import annotations

import csv
import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch


@dataclass
class TimedEvent:
    """One invocation of a timed block.

    `start_event` / `end_event` are `torch.cuda.Event(enable_timing=True)`
    markers queued on the recording stream. `elapsed_ms` is populated by
    `TimingStats.step_done()` via `start.elapsed_time(end)` after the
    events have completed on the GPU.
    """
    step_idx: int
    layer_idx: int
    op_name: str
    call_idx: int
    start_event: torch.cuda.Event = field(repr=False)
    end_event: torch.cuda.Event = field(repr=False)
    elapsed_ms: float = -1.0  # filled on resolve


def _parse_rank_filter(raw: str) -> Optional[set]:
    """Parse comma-separated rank list from env. Empty / missing -> {0}."""
    raw = (raw or "").strip()
    if not raw:
        return {0}
    try:
        return {int(tok) for tok in raw.split(",") if tok.strip()}
    except ValueError:
        return {0}


def _current_rank() -> int:
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            return dist.get_rank()
    except Exception:
        pass
    return 0


class TimingStats:
    """Per-invocation decode/prefill timing with CUDA-event precision.

    Legacy call sites continue to work — `dt.timed(name, layer_idx)` is
    the same context manager. What changes is the underlying measurement
    (CUDA events, lazy resolve) and the addition of `step_done()` that
    the decode loop must call at every step boundary.
    """

    # Cap the event pool to avoid unbounded growth. At ~28 distinct ops
    # × 78 layers × 2 events each = 4368 events worst case per step; bump
    # a bit for safety. Allocations beyond the cap still work (fresh
    # Event()); they just don't recycle.
    _EVENT_POOL_MAX = 8192

    def __init__(self, model_name: str, phase: str, categories: List[str]):
        self.model_name = model_name
        self.phase = phase  # "decode" or "prefill"
        self._categories = list(categories)
        self.enabled: bool = False

        # Per-invocation storage.
        self._pending: List[TimedEvent] = []   # recorded this step, not yet resolved
        self._records: List[TimedEvent] = []   # resolved (all elapsed_ms filled)
        self._event_pool: List[torch.cuda.Event] = []

        self._step_idx: int = 0
        self._call_counter: int = 0

        # Rank filter for output.
        self._emit_ranks = _parse_rank_filter(
            os.environ.get("BATCHGEN_DECODE_TIMING_RANKS", "")
        )

        # CSV sink (lazy-opened on first flush).
        self._csv_path: Optional[str] = os.environ.get("BATCHGEN_DECODE_TIMING_CSV") or None
        self._csv_writer = None
        self._csv_file = None
        self._csv_rows_written = 0

        # Periodic flush cadence.
        try:
            self._interval = int(os.environ.get("BATCHGEN_DECODE_TIMING_INTERVAL", "50"))
        except ValueError:
            self._interval = 50

    # -- Lifecycle ---------------------------------------------------------

    def reset(self):
        """Clear records + pending. Preserves monotonic step counter.

        DOES NOT reset `_step_idx`: batchgen_worker.py calls
        `log_summary() + reset()` per decode step (batchgen_worker.py:9206),
        so zeroing step_idx here would stamp every step's events with
        step=0, making per-step analysis impossible. step_idx is monotonic
        from process start; the CSV carries it through so downstream
        pivots can aggregate or filter arbitrarily.
        """
        # Recycle pending events (they may never resolve but freeing them
        # into the pool is still better than leaking).
        for ev in self._pending:
            self._recycle_event(ev.start_event)
            self._recycle_event(ev.end_event)
        self._pending.clear()
        self._records.clear()
        self._call_counter = 0
        # Critical: CSV flush tracks `_csv_rows_written` as an index into
        # `_records`. If we clear records without resetting this cursor,
        # `_records[_csv_rows_written:]` is empty forever and subsequent
        # decode groups' records never reach the CSV. log_summary() runs
        # _flush_csv() before reset(), so any unflushed rows from the
        # cleared group already made it out.
        self._csv_rows_written = 0

    def enable(self):
        self.enabled = True
        self.reset()

    def disable(self):
        self.enabled = False
        self._close_csv()

    def add_category(self, name: str):
        """Register a category name (for log_summary display ordering).

        With per-invocation storage, any `op_name` is accepted at record
        time; the categories list just controls summary display.
        """
        if name not in self._categories:
            self._categories.append(name)

    # -- Event pool --------------------------------------------------------

    def _get_event(self) -> torch.cuda.Event:
        if self._event_pool:
            return self._event_pool.pop()
        return torch.cuda.Event(enable_timing=True)

    def _recycle_event(self, ev: torch.cuda.Event) -> None:
        if len(self._event_pool) < self._EVENT_POOL_MAX:
            self._event_pool.append(ev)

    # -- Recording API -----------------------------------------------------

    @contextmanager
    def host_timed(self, segment_name: str, layer_idx: int = -1):
        """Context manager: wall-clock a HOST-side segment (perf_counter).

        Rows land in the same records/CSV as GPU events, distinguished by
        the ``host:`` op-name prefix and layer_idx=-1, so the per-step
        budget-closure analysis (step_wall == GPU critical path + serial
        host + unattributed) reads one stream. Zero cost when disabled.
        """
        if not self.enabled:
            yield
            return
        import time as _time
        _t0 = _time.perf_counter()
        try:
            yield
        finally:
            self.record(
                f"host:{segment_name}", layer_idx,
                (_time.perf_counter() - _t0) * 1000.0,
            )

    @contextmanager
    def timed(self, op_name: str, layer_idx: int = 0):
        """Context manager: queue a pair of CUDA events around the block.

        No `torch.cuda.synchronize()` — this is the whole point. The
        events resolve when the step boundary calls `step_done()`.
        Disabled timer is a pure yield with no event work.
        """
        if not self.enabled:
            yield
            return

        # Skip recording while a CUDA graph is capturing — events recorded
        # inside a capture region get baked into the graph, which would
        # double-count at replay and inflate the record list with
        # un-resolvable capture-time events.
        if torch.cuda.is_current_stream_capturing():
            yield
            return

        start = self._get_event()
        end = self._get_event()
        stream = torch.cuda.current_stream()
        start.record(stream)
        try:
            yield
        finally:
            end.record(stream)
            self._pending.append(TimedEvent(
                step_idx=self._step_idx,
                layer_idx=layer_idx,
                op_name=op_name,
                call_idx=self._call_counter,
                start_event=start,
                end_event=end,
            ))
            self._call_counter += 1

    def record(self, category: str, layer_idx: int, time_ms: float):
        """Back-compat: accept a pre-measured time in ms.

        Legacy callers that computed their own wall-clock time can still
        push a scalar. The resulting `TimedEvent` has dummy events and a
        pre-filled elapsed_ms (resolved=True).
        """
        if not self.enabled:
            return
        self._records.append(TimedEvent(
            step_idx=self._step_idx,
            layer_idx=layer_idx,
            op_name=category,
            call_idx=self._call_counter,
            start_event=None,  # type: ignore[arg-type]
            end_event=None,    # type: ignore[arg-type]
            elapsed_ms=float(time_ms),
        ))
        self._call_counter += 1

    # -- Step boundary -----------------------------------------------------

    def step_done(self):
        """Drain `_pending`, resolve elapsed_ms, advance step counter.

        One `event.synchronize()` on the last end-event is enough to
        guarantee all prior events on that stream have completed. Works
        even when captured graphs are interleaved with eager ops (we just
        skip recording during capture, so `_pending` only holds eager
        events from this step).
        """
        if not self.enabled:
            return
        if self._pending:
            # Sync on the final end-event — serializes host until the
            # stream has finished this step. Much cheaper than the prior
            # per-block cuda.synchronize().
            self._pending[-1].end_event.synchronize()
            for ev in self._pending:
                ev.elapsed_ms = ev.start_event.elapsed_time(ev.end_event)
                self._recycle_event(ev.start_event)
                self._recycle_event(ev.end_event)
                ev.start_event = None  # type: ignore[assignment]
                ev.end_event = None    # type: ignore[assignment]
            self._records.extend(self._pending)
            self._pending.clear()

        self._step_idx += 1
        self._call_counter = 0

        # Periodic CSV flush + optional summary log.
        if self._interval > 0 and self._step_idx % self._interval == 0:
            self._flush_csv()

    # -- CSV sink ----------------------------------------------------------

    def _should_emit(self) -> bool:
        if _current_rank() not in self._emit_ranks:
            return False
        return True

    def _ensure_csv(self):
        if self._csv_writer is not None:
            return
        if self._csv_path is None or not self._should_emit():
            return
        # Append mode — multiple runs append to the same file.
        self._csv_file = open(self._csv_path, "a", buffering=1, newline="")
        self._csv_writer = csv.writer(self._csv_file)
        # Only write header if file is empty.
        if self._csv_file.tell() == 0:
            self._csv_writer.writerow([
                "step", "rank", "layer", "op", "call", "elapsed_ms",
            ])

    def _flush_csv(self):
        if self._csv_path is None or not self._should_emit():
            return
        self._ensure_csv()
        if self._csv_writer is None:
            return
        rank = _current_rank()
        # Drain records written since last flush.
        new_rows = self._records[self._csv_rows_written:]
        for ev in new_rows:
            if ev.elapsed_ms < 0:
                continue
            self._csv_writer.writerow([
                ev.step_idx, rank, ev.layer_idx, ev.op_name,
                ev.call_idx, f"{ev.elapsed_ms:.6f}",
            ])
        self._csv_rows_written = len(self._records)

    def _close_csv(self):
        if self._csv_file is not None:
            try:
                self._flush_csv()
            finally:
                self._csv_file.close()
                self._csv_file = None
                self._csv_writer = None

    # -- Summary aggregation -----------------------------------------------

    def _aggregate_by_op(self) -> Dict[str, Dict[str, float]]:
        """Compute per-op stats: count, mean, p50, p90, p99, total, %."""
        by_op: Dict[str, List[float]] = {}
        for ev in self._records:
            if ev.elapsed_ms < 0:
                continue
            by_op.setdefault(ev.op_name, []).append(ev.elapsed_ms)
        if not by_op:
            return {}
        total_all = sum(sum(v) for v in by_op.values())
        stats: Dict[str, Dict[str, float]] = {}
        for op, values in by_op.items():
            values.sort()
            n = len(values)
            total = sum(values)
            mean = total / n
            stats[op] = {
                "count": n,
                "mean_ms": mean,
                "p50_ms": values[n // 2],
                "p90_ms": values[min(n - 1, int(n * 0.90))],
                "p99_ms": values[min(n - 1, int(n * 0.99))],
                "total_ms": total,
                "pct": (total / total_all * 100.0) if total_all > 0 else 0.0,
            }
        return stats

    # -- Legacy views (derived from _records) ------------------------------

    @property
    def _totals(self) -> Dict[str, float]:
        out: Dict[str, float] = {c: 0.0 for c in self._categories}
        for ev in self._records:
            if ev.elapsed_ms < 0:
                continue
            out[ev.op_name] = out.get(ev.op_name, 0.0) + ev.elapsed_ms
        return out

    @property
    def _per_layer(self) -> Dict[str, Dict[int, float]]:
        out: Dict[str, Dict[int, float]] = {}
        for ev in self._records:
            if ev.elapsed_ms < 0:
                continue
            per = out.setdefault(ev.op_name, {})
            per[ev.layer_idx] = per.get(ev.layer_idx, 0.0) + ev.elapsed_ms
        return out

    @property
    def _counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for ev in self._records:
            if ev.elapsed_ms < 0:
                continue
            out[ev.op_name] = out.get(ev.op_name, 0) + 1
        return out

    def log_summary(self):
        """Emit a human-readable breakdown.

        Rows sorted by total_ms descending so the bottleneck is first.
        Includes count, mean, p50, p90, p99, total, % of timed work.
        """
        if not self.enabled or not self._should_emit():
            return
        # Flush any pending rows to CSV even if we're below interval.
        self._flush_csv()

        stats = self._aggregate_by_op()
        if not stats:
            return

        header = f"{self.model_name} {self.phase.capitalize()} Timing (steps={self._step_idx})"
        logging.info("=" * 90)
        logging.info(header)
        logging.info("=" * 90)
        cols = f"{'op':<22s} {'count':>7s} {'mean(us)':>10s} {'p50(us)':>10s} {'p90(us)':>10s} {'p99(us)':>10s} {'total(ms)':>11s} {'pct':>6s}"
        logging.info(cols)
        logging.info("-" * 90)
        for op, s in sorted(stats.items(), key=lambda kv: -kv[1]["total_ms"]):
            logging.info(
                f"{op:<22s} {int(s['count']):>7d} "
                f"{s['mean_ms']*1000:>10.1f} {s['p50_ms']*1000:>10.1f} "
                f"{s['p90_ms']*1000:>10.1f} {s['p99_ms']*1000:>10.1f} "
                f"{s['total_ms']:>11.2f} {s['pct']:>5.1f}%"
            )
        logging.info("=" * 90)


# ---------------------------------------------------------------------------
# Global singletons (one per phase per process)
# ---------------------------------------------------------------------------

_decode_timer: Optional[TimingStats] = None
_prefill_timer: Optional[TimingStats] = None


def get_decode_timer() -> Optional[TimingStats]:
    return _decode_timer


def init_decode_timer(model_name: str, categories: List[str]) -> TimingStats:
    global _decode_timer
    _decode_timer = TimingStats(model_name, "decode", categories)
    if os.environ.get("BATCHGEN_DECODE_TIMING", "0") == "1":
        _decode_timer.enable()
        logging.info(
            f"DecodeTimingStats enabled for {model_name} "
            f"(csv={_decode_timer._csv_path}, interval={_decode_timer._interval}, "
            f"ranks={sorted(_decode_timer._emit_ranks)})"
        )
    return _decode_timer


def get_prefill_timer() -> Optional[TimingStats]:
    return _prefill_timer


def init_prefill_timer(model_name: str, categories: List[str]) -> TimingStats:
    global _prefill_timer
    _prefill_timer = TimingStats(model_name, "prefill", categories)
    if os.environ.get("BATCHGEN_PREFILL_TIMING", "0") == "1":
        _prefill_timer.enable()
        logging.info(f"PrefillTimingStats enabled for {model_name}")
    return _prefill_timer
