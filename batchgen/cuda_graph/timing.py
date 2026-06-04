"""Formal timing facility for decode CUDA-graph development.

Emits per-step `StepTiming` rows to a JSONL file when
`BATCHGEN_DECODE_GRAPH_TIMING=1`. NO-OP otherwise (zero allocations on the
production path) — see contract §E guarantee #3.

Sections instrumented (key names in `StepTiming`):

* ``capture``        — graph capture work (only on capture step; 0.0 otherwise).
* ``replay``         — graph replay or eager forward.
* ``kv_offload``     — post-replay KV staging via ``stage_post_graph_kv``.
* ``compare_eager``  — eager re-run when ``BATCHGEN_DECODE_GRAPH_COMPARE=1``.

All sections use CUDA events for GPU-accurate timing; the recorder defers
``cudaEventElapsedTime`` synchronization to ``flush()`` so the hot path takes
zero blocking syncs.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from batchgen.cuda_graph.adapter import GraphMode

logger = logging.getLogger(__name__)


_TIMED_SECTIONS = ("capture", "replay", "kv_offload", "compare_eager")


@dataclass
class StepTiming:
    """Per-step decode timing — sampled only when timing is enabled."""
    decode_iter: int
    mode: str
    bucket: Optional[int]
    capture_ms: float = 0.0
    replay_ms: float = 0.0
    kv_offload_ms: float = 0.0
    compare_eager_ms: float = 0.0
    compare_max_abs: Optional[float] = None
    compare_max_rel: Optional[float] = None
    wall_t_start: float = field(default_factory=time.time)

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self))


class _PendingSection:
    """A start/end CUDA event pair whose elapsed_time is read at flush()."""

    __slots__ = ("key", "start", "end")

    def __init__(self, key: str, start: torch.cuda.Event, end: torch.cuda.Event):
        self.key = key
        self.start = start
        self.end = end


class StepTimingRecorder:
    """Single-instance recorder owned by the worker.

    Append-only; flushed to ``{log_dir}/decode_graph_timing.jsonl`` when
    enabled. When disabled, `begin()` returns a no-op context that performs
    zero CUDA-event creation, zero allocations, and zero JSON serialization.
    """

    ENV_VAR = "BATCHGEN_DECODE_GRAPH_TIMING"

    def __init__(self, log_dir: Optional[str] = None):
        self._enabled = os.environ.get(self.ENV_VAR, "0") == "1"
        self._log_dir = log_dir
        self._path = None
        if self._enabled and log_dir:
            os.makedirs(log_dir, exist_ok=True)
            self._path = os.path.join(log_dir, "decode_graph_timing.jsonl")
        self._lock = threading.Lock()
        self._pending: List[tuple] = []  # (StepTiming, [_PendingSection,...])

    @property
    def enabled(self) -> bool:
        return self._enabled

    @contextlib.contextmanager
    def begin(self, *, decode_iter: int, mode: "GraphMode", bucket: Optional[int]) -> Iterator["StepTimingContext"]:
        """Open a timing window for one decode step.

        Yields a `StepTimingContext` that exposes `time_section(key)` for
        each instrumented section. The window is closed (and the row queued
        for flush) on context exit.
        """
        if not self._enabled:
            yield _NULL_CONTEXT
            return
        ts = StepTiming(decode_iter=decode_iter, mode=str(mode.value if hasattr(mode, "value") else mode), bucket=bucket)
        sections: List[_PendingSection] = []
        try:
            yield StepTimingContext(timing=ts, sections=sections)
        finally:
            self._pending.append((ts, sections))

    def flush(self) -> None:
        """Resolve all pending CUDA-event timings and write JSONL rows.

        Synchronizes the current stream once at flush boundary; never inside
        the timing window. Safe to call between decode steps or on shutdown.
        NO-OP when disabled or when there are no pending rows.
        """
        if not self._enabled or not self._pending:
            return
        torch.cuda.current_stream().synchronize()
        rows: List[StepTiming] = []
        for ts, sections in self._pending:
            for sec in sections:
                ms = float(sec.start.elapsed_time(sec.end))
                if sec.key == "capture":
                    ts.capture_ms = ms
                elif sec.key == "replay":
                    ts.replay_ms = ms
                elif sec.key == "kv_offload":
                    ts.kv_offload_ms = ms
                elif sec.key == "compare_eager":
                    ts.compare_eager_ms = ms
                else:
                    logger.warning(
                        "StepTimingRecorder: unknown section key %r (expected one of %r)",
                        sec.key, _TIMED_SECTIONS,
                    )
            rows.append(ts)
        self._pending.clear()
        if self._path is None:
            return
        with self._lock, open(self._path, "a") as fh:
            for r in rows:
                fh.write(r.to_json() + "\n")


@dataclass
class StepTimingContext:
    """Per-step handle exposing `time_section(key)`."""
    timing: StepTiming
    sections: List[_PendingSection]

    @contextlib.contextmanager
    def time_section(self, key: str) -> Iterator[None]:
        if key not in _TIMED_SECTIONS:
            raise ValueError(
                f"time_section: key={key!r} not in {_TIMED_SECTIONS}. "
                f"Add it to _TIMED_SECTIONS first if you need a new section."
            )
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            end.record()
            self.sections.append(_PendingSection(key=key, start=start, end=end))


class _NullContext:
    """Returned by `begin()` when timing is disabled — every operation is a no-op."""

    @contextlib.contextmanager
    def time_section(self, key: str) -> Iterator[None]:
        yield

    timing: StepTiming = None  # type: ignore[assignment]


_NULL_CONTEXT = _NullContext()


__all__ = [
    "StepTiming",
    "StepTimingRecorder",
    "StepTimingContext",
]
