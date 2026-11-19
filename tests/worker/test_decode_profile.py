"""Unit tests for `batchgen.worker.decode_profile`.

Pure CPU: the capture takes an injected profiler factory and device
synchronisation, so these tests drive arm -> capture -> end with a fake
profiler and assert that every end of the window stops it.
"""

from __future__ import annotations

import logging

import pytest

from batchgen.worker.decode_profile import (
    DECODE_PROFILE_WARMUP_STEPS,
    DecodeProfileCapture,
)


class _FakeTable:
    def table(self, **kwargs):
        return "fake-table"


class _FakeProfiler:
    def __init__(self, log, fail_stop=False, fail_table=False):
        self._log = log
        self._fail_stop = fail_stop
        self._fail_table = fail_table
        self.running = True
        self.exits = 0

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._log.append("stop")
        self.exits += 1
        if self._fail_stop:
            raise RuntimeError("stop failed")
        self.running = False

    def key_averages(self):
        if self._fail_table:
            raise RuntimeError("no table")
        return _FakeTable()

    def export_chrome_trace(self, path):
        self._log.append(("trace", path))


class _Rig:
    def __init__(self, rank=0, **profiler_kwargs):
        self.log = []
        self.profilers = []
        self._profiler_kwargs = profiler_kwargs
        self.capture = DecodeProfileCapture(rank, self._open, self._sync)

    def _open(self):
        self.log.append("start")
        profiler = _FakeProfiler(self.log, **self._profiler_kwargs)
        self.profilers.append(profiler)
        return profiler

    def _sync(self):
        self.log.append("sync")

    def steps(self, n, debug, rows=8):
        for _ in range(n):
            self.capture.step(debug, rows)

    def assert_pre_capture_state(self):
        capture = self.capture
        assert capture._profiler is None
        assert capture._window_open is False
        assert all(not p.running for p in self.profilers)


def _debug(steps=5, **extra):
    return {"k3_decode_profile_steps": steps, **extra}


def test_full_window_starts_after_warmup_and_stops_after_n_steps(caplog):
    rig = _Rig()
    rig.steps(DECODE_PROFILE_WARMUP_STEPS, _debug())
    assert rig.log == [] and rig.capture.profiled is False

    rig.capture.step(_debug(), 8)
    assert rig.log == ["start"] and rig.capture.profiled is True

    rig.steps(4, _debug())
    assert rig.log == ["start"]
    with caplog.at_level(logging.INFO):
        rig.capture.step(_debug(), 8)
    assert rig.log == ["start", "sync", "stop"]
    rig.assert_pre_capture_state()
    assert "[K3_DECODE_PROFILE] 5 steps at 8 rows (per-step = table / 5)\n" in caplog.text
    assert "ended early" not in caplog.text
    inadmissible = [r for r in caplog.records if "not admissible" in r.getMessage()]
    assert len(inadmissible) == 1 and inadmissible[0].levelno == logging.WARNING


def test_round_end_stops_a_window_that_never_filled(caplog):
    rig = _Rig()
    rig.steps(DECODE_PROFILE_WARMUP_STEPS + 2, _debug(steps=40))
    assert rig.log == ["start"]
    with caplog.at_level(logging.INFO):
        rig.capture.end_round()
    assert rig.log == ["start", "sync", "stop"]
    rig.assert_pre_capture_state()
    assert "[K3_DECODE_PROFILE] 2 steps at 8 rows (per-step = table / 2)" in caplog.text
    assert "ended early: the decode round ended; 40 requested" in caplog.text


@pytest.mark.parametrize("later_debug", [None, {}, {"k3_decode_profile_steps": 0}])
def test_a_step_without_the_request_stops_the_window(later_debug):
    rig = _Rig()
    rig.steps(DECODE_PROFILE_WARMUP_STEPS + 1, _debug(steps=40))
    rig.capture.step(later_debug, 8)
    assert rig.log == ["start", "sync", "stop"]
    rig.assert_pre_capture_state()


def test_capture_is_one_per_generation():
    rig = _Rig()
    rig.steps(DECODE_PROFILE_WARMUP_STEPS + 1 + 5, _debug())
    rig.steps(50, _debug())
    rig.capture.end_round()
    assert rig.log == ["start", "sync", "stop"]
    assert rig.capture.profiled is True


def test_unlisted_rank_follows_the_window_without_a_profiler(caplog):
    rig = _Rig(rank=1)
    debug = _debug(k3_decode_profile_ranks=[0, 8])
    with caplog.at_level(logging.INFO):
        rig.steps(DECODE_PROFILE_WARMUP_STEPS + 1 + 5, debug)
        rig.capture.end_round()
    assert rig.log == [] and rig.profilers == []
    assert caplog.text == ""
    rig.assert_pre_capture_state()
    # The generation was profiled on other ranks; rank-0 style reporting
    # must still be able to say so.
    assert rig.capture.profiled is True


def test_listed_non_zero_rank_profiles():
    rig = _Rig(rank=8)
    rig.steps(DECODE_PROFILE_WARMUP_STEPS + 1 + 5, _debug(k3_decode_profile_ranks=[0, 8]))
    assert rig.log == ["start", "sync", "stop"]


def test_log_suffix_marks_only_a_profiled_generation():
    rig = _Rig(rank=1)
    assert rig.capture.log_suffix() == ""
    rig.steps(DECODE_PROFILE_WARMUP_STEPS + 1, _debug())
    assert rig.capture.log_suffix() == " profiled=1"


def test_no_request_leaves_the_rank_untouched():
    rig = _Rig()
    rig.steps(20, None)
    rig.steps(20, {"k3_decode_profile_steps": "not-a-number"})
    rig.capture.end_round()
    assert rig.log == [] and rig.capture.profiled is False
    assert rig.capture._done is False


def test_trace_is_written_when_requested(tmp_path):
    rig = _Rig()
    trace_dir = tmp_path / "traces"
    rig.steps(DECODE_PROFILE_WARMUP_STEPS + 1 + 5, _debug(k3_decode_profile_trace=str(trace_dir)))
    assert rig.log[-1] == ("trace", str(trace_dir / "k3_decode_profile_rows8_rank0.json"))
    assert trace_dir.is_dir()


def test_failed_stop_and_failed_report_do_not_raise_and_do_not_retry():
    for kwargs in ({"fail_stop": True}, {"fail_table": True}):
        rig = _Rig(**kwargs)
        rig.steps(DECODE_PROFILE_WARMUP_STEPS + 1 + 5, _debug())
        rig.steps(10, _debug())
        rig.capture.end_round()
        assert rig.log == ["start", "sync", "stop"]
        assert rig.capture._profiler is None and rig.capture._window_open is False


def test_failed_sync_still_stops_the_profiler():
    rig = _Rig()

    def _broken_sync():
        raise RuntimeError("device error")

    rig.capture._synchronize = _broken_sync
    rig.steps(DECODE_PROFILE_WARMUP_STEPS + 1 + 4, _debug())
    with pytest.raises(RuntimeError, match="device error"):
        rig.capture.step(_debug(), 8)
    assert rig.log == ["start", "stop"]
    rig.assert_pre_capture_state()


def test_failed_start_does_not_raise():
    def _broken_open():
        raise RuntimeError("profiler unavailable")

    capture = DecodeProfileCapture(0, _broken_open, lambda: None)
    for _ in range(DECODE_PROFILE_WARMUP_STEPS + 1 + 5):
        capture.step(_debug(), 8)
    assert capture._profiler is None and capture._window_open is False
