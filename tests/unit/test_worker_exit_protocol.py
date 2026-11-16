"""The worker exit protocol, as defined once in ``batchgen.eventlog.process``.

The supervisor (``server_worker_main_loop``'s shutdown watchdog) and the
worker's own quiesce self-exit end a process with ``os._exit``, which runs no
handler and flushes nothing. Both use the same status, the same path marker and
the same pre-exit flush, which this module pins: the values, and the flush
order (event log, then the root logging handlers, then stdout and stderr).

CPU only; imports no torch.
"""

from __future__ import annotations

import logging
import sys

from batchgen import eventlog
from batchgen.eventlog import process


def test_the_exit_status_and_markers_keep_their_values():
    assert process.WORKER_SELF_EXIT_CODE == 86
    assert process.EXIT_PATH_SELF_EXIT == "self-exit"
    assert process.EXIT_PATH_SENTINEL == "ordered-via-sentinel"
    assert process.EXIT_PATH_FLAG == "ordered-via-flag"
    assert process.EXIT_PATH_ABORT == "ordered-via-abort"
    assert process.EXIT_PATH_DECODE_BOUNDARY == "ordered-via-decode-boundary"


class _Recorder:
    def __init__(self, calls, name, fail=False):
        self.calls, self.name, self.fail = calls, name, fail

    def flush(self, *args):
        self.calls.append(self.name)
        if self.fail:
            raise RuntimeError(f"{self.name} is wedged")


class _Handler(logging.Handler):
    def __init__(self, calls, name, fail=False):
        super().__init__()
        self._recorder = _Recorder(calls, name, fail)

    def emit(self, record):
        pass

    def flush(self):
        self._recorder.flush()


def _wire(monkeypatch, calls, *, fail=False):
    monkeypatch.setattr(eventlog, "flush", _Recorder(calls, "eventlog", fail).flush)
    root = logging.getLogger()
    monkeypatch.setattr(
        root, "handlers", [_Handler(calls, "handler-1", fail), _Handler(calls, "handler-2", fail)]
    )
    monkeypatch.setattr(sys, "stdout", _Recorder(calls, "stdout", fail))
    monkeypatch.setattr(sys, "stderr", _Recorder(calls, "stderr", fail))


def test_the_pre_exit_flush_runs_eventlog_then_root_handlers_then_streams(monkeypatch):
    calls = []
    _wire(monkeypatch, calls)
    process.flush_logs_before_exit()
    assert calls == ["eventlog", "handler-1", "handler-2", "stdout", "stderr"]


def test_every_flush_step_is_guarded_and_the_rest_still_run(monkeypatch):
    """It runs on the way out of a failing teardown; raising there would
    replace a bounded self-exit with an unhandled error."""
    calls = []
    _wire(monkeypatch, calls, fail=True)
    process.flush_logs_before_exit()
    assert calls == ["eventlog", "handler-1", "handler-2", "stdout", "stderr"]
