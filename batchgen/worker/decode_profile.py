"""One-shot decode profile capture, armed through the server debug state.

``decode_profile_steps = N`` asks the ranks listed in ``decode_profile_ranks``
(default rank 0) to wrap N decode steps in a profiler and log the per-kernel
table; ``decode_profile_trace = "<dir>"`` also writes the Chrome trace there.

The capture window belongs to the decode steps that carry the request. It
opens after ``DECODE_PROFILE_WARMUP_STEPS`` requesting steps and ends at the
first of: N captured steps, a step that no longer carries the request, or the
end of the decode round. Every end goes through ``_close``, so no profiler is
left running into decode steps that did not ask for it.

One capture per worker generation. Every rank follows the same window (the
request is read from the global decode batch, so all ranks see it at the same
step); only the listed ranks open a profiler. ``profiled`` is the per-rank
record that this worker generation ran a capture. Decode steps have been
observed to stay slower for the rest of a generation after a capture although
the profiler is stopped, so step timings of the generation are not admissible
for benchmarking from the capture on. Rank 0 reports the marker in its
``[DECODE]`` lines and in the batch group summary.

The module is torch-free: the caller injects the profiler factory and the
device synchronisation, so the window logic is testable on CPU.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Optional

# Requesting decode steps that run before the window opens.
DECODE_PROFILE_WARMUP_STEPS = 3
# Suffix of the rank-0 ``[DECODE]`` lines once the generation ran a capture.
DECODE_PROFILED_SUFFIX = " profiled=1"


def open_torch_decode_profiler() -> Any:
    """Start and return a CPU + CUDA ``torch.profiler`` session."""
    from torch.profiler import ProfilerActivity, profile

    profiler = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
    profiler.__enter__()
    return profiler


def _requested_steps(batchgen_debug: Optional[dict]) -> int:
    try:
        return int((batchgen_debug or {}).get("decode_profile_steps", 0) or 0)
    except (TypeError, ValueError):
        return 0


class DecodeProfileCapture:
    """The capture window of one rank for one worker generation."""

    def __init__(
        self,
        rank: int,
        open_profiler: Callable[[], Any],
        synchronize: Callable[[], None],
    ) -> None:
        self._rank = rank
        self._open_profiler = open_profiler
        self._synchronize = synchronize
        self._requesting_steps = 0
        self._window_open = False
        self._done = False
        self._profiler: Any = None
        self._target_steps = 0
        self._captured_steps = 0
        self._rows = 0
        self._trace_dir: Optional[str] = None
        self.profiled = False

    def step(self, batchgen_debug: Optional[dict], batch_rows: int) -> None:
        """Advance the window; called once per decode step, before its forward."""
        if self._done:
            return
        steps = _requested_steps(batchgen_debug)
        if self._window_open:
            # The previous step's forward ran inside the window.
            self._captured_steps += 1
            if steps <= 0:
                self._close("the request left the decode batch")
            elif self._captured_steps >= self._target_steps:
                self._close(None)
            return
        if steps <= 0:
            return
        self._requesting_steps += 1
        if self._requesting_steps > DECODE_PROFILE_WARMUP_STEPS:
            self._open(batchgen_debug or {}, steps, batch_rows)

    def log_suffix(self) -> str:
        """Marker for the rank-0 ``[DECODE]`` lines of a profiled generation."""
        return DECODE_PROFILED_SUFFIX if self.profiled else ""

    def end_round(self) -> None:
        """Decode round over: a window that is still open ends here."""
        if self._window_open:
            self._captured_steps += 1
            self._close("the decode round ended")

    def _open(self, batchgen_debug: dict, steps: int, batch_rows: int) -> None:
        self._window_open = True
        self.profiled = True
        self._target_steps = steps
        self._captured_steps = 0
        self._rows = batch_rows
        ranks = batchgen_debug.get("decode_profile_ranks") or [0]
        if self._rank not in ranks:
            return
        self._trace_dir = batchgen_debug.get("decode_profile_trace")
        try:
            self._profiler = self._open_profiler()
        except Exception as exc:  # noqa: BLE001 - diagnostics only
            logging.warning("[K3_DECODE_PROFILE] profiler start failed: %s", exc)
            return
        logging.info(
            "[K3_DECODE_PROFILE] capturing %d decode steps at %d rows", steps, batch_rows
        )

    def _close(self, early_reason: Optional[str]) -> None:
        profiler, self._profiler = self._profiler, None
        self._window_open = False
        self._done = True
        if profiler is None:
            return
        try:
            self._synchronize()
        finally:
            stopped = self._stop(profiler)
        logging.warning(
            "[K3_DECODE_PROFILE] rank %d ran a profiler capture in this worker "
            "generation; decode step timings of this generation are not admissible "
            "for benchmarking from this capture on. Restart the worker pool before "
            "timing decode again.",
            self._rank,
        )
        if stopped:
            self._report(profiler, early_reason)

    @staticmethod
    def _stop(profiler: Any) -> bool:
        try:
            profiler.__exit__(None, None, None)
        except Exception as exc:  # noqa: BLE001 - diagnostics only
            logging.warning("[K3_DECODE_PROFILE] profiler stop failed: %s", exc)
            return False
        return True

    def _report(self, profiler: Any, early_reason: Optional[str]) -> None:
        steps = self._captured_steps
        note = ""
        if early_reason is not None:
            note = f" [ended early: {early_reason}; {self._target_steps} requested]"
        try:
            table = profiler.key_averages().table(sort_by="cuda_time_total", row_limit=40)
            logging.info(
                "[K3_DECODE_PROFILE] %d steps at %d rows (per-step = table / %d)%s\n%s",
                steps, self._rows, steps, note, table,
            )
            # The table alone loses the kernel order within a step; the trace
            # keeps it.
            if self._trace_dir:
                os.makedirs(self._trace_dir, exist_ok=True)
                path = os.path.join(
                    self._trace_dir,
                    f"k3_decode_profile_rows{self._rows}_rank{self._rank}.json",
                )
                profiler.export_chrome_trace(path)
                logging.info("[K3_DECODE_PROFILE] trace written to %s", path)
        except Exception as exc:  # noqa: BLE001 - diagnostics only
            logging.warning("[K3_DECODE_PROFILE] report failed: %s", exc)
