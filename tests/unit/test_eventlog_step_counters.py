"""``batchgen.eventlog.counters.StepCounters``: interval gating, delta/reset
semantics, payload shape, and the per-step fast path.

CPU only; no CUDA, no torch.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eventlog_testutil as tu  # noqa: E402

ev = tu.import_eventlog()
StepCounters = ev.counters.StepCounters
EVENT = ev.counters.STEP_COUNTERS_EVENT

PAYLOAD_KEYS = [
    "steps", "row_steps", "batch_min", "batch_max", "decode_wall_s",
    "window_s", "total_steps",
]


@pytest.fixture(autouse=True)
def _isolated():
    with tu.isolated_logging(ev):
        yield


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def captured():
    logger = logging.getLogger("batchgen.test.step_counters")
    handler = _Capture()
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    try:
        yield logger, handler.records
    finally:
        logger.removeHandler(handler)
        logger.propagate = True
        logger.setLevel(logging.NOTSET)


def _data(record: logging.LogRecord) -> dict:
    assert getattr(record, ev.schema.ATTR_EVENT) == EVENT
    return getattr(record, ev.schema.ATTR_DATA)


def test_event_name_is_valid_and_interval_matches_the_level_policy():
    assert ev.schema.validate_event_name(EVENT) == "step.counters"
    # INFO periodic summaries: at most one record per 30 s per rank.
    assert ev.counters.STEP_COUNTERS_INTERVAL_S >= 30.0


def test_no_record_before_the_interval_elapses(captured):
    logger, records = captured
    counters = StepCounters(logger, 100.0)
    counters.begin_round(100.0)
    for i in range(1, 300):
        now = 100.0 + i * 0.01  # 2.99 s of steps
        counters.add_step(now, 8)
        assert counters.maybe_emit(now, 10.0) is False
    assert records == []
    assert counters.steps == 299
    assert counters.row_steps == 299 * 8


def test_one_record_carries_the_deltas_then_resets(captured):
    logger, records = captured
    counters = StepCounters(logger, 100.0)
    counters.begin_round(100.5)
    counters.add_step(101.0, 4)
    counters.add_step(102.0, 6)
    counters.add_step(104.0, 2)
    assert counters.maybe_emit(104.0, 10.0) is False
    counters.add_step(110.0, 5)
    assert counters.maybe_emit(110.0, 10.0) is True

    assert len(records) == 1
    record = records[0]
    assert record.levelno == logging.INFO
    assert record.getMessage() == EVENT
    data = _data(record)
    assert list(data) == PAYLOAD_KEYS
    assert data == {
        "steps": 4,
        "row_steps": 17,
        "batch_min": 2,
        "batch_max": 6,
        "decode_wall_s": 9.5,   # 100.5 -> 110.0, round start to last step
        "window_s": 10.0,       # construction -> emit
        "total_steps": 4,
    }
    # Deltas reset; the window restarts at the emit time.
    assert (counters.steps, counters.row_steps, counters.batch_max) == (0, 0, 0)
    assert counters.decode_wall_s == 0.0
    assert counters.total_steps == 4
    assert counters.maybe_emit(119.9, 10.0) is False


def test_total_steps_accumulates_across_records(captured):
    logger, records = captured
    counters = StepCounters(logger, 0.0)
    counters.begin_round(0.0)
    for window in range(3):
        for i in range(1, 6):
            now = window * 10.0 + i * 2.0
            counters.add_step(now, 1)
            counters.maybe_emit(now, 10.0)
    assert [_data(r)["total_steps"] for r in records] == [5, 10, 15]
    assert [_data(r)["steps"] for r in records] == [5, 5, 5]


def test_time_between_rounds_is_not_decode_time(captured):
    logger, records = captured
    counters = StepCounters(logger, 0.0)
    counters.begin_round(1.0)
    counters.add_step(1.0, 3)
    counters.add_step(2.0, 3)
    counters.end_round(3.0)            # round 1: 2 s of decode
    counters.begin_round(20.0)         # 17 s of prefill in between
    counters.add_step(20.0, 7)
    counters.end_round(21.0)           # round 2: 1 s of decode
    assert counters.maybe_emit(21.0, 10.0) is True
    data = _data(records[0])
    assert data["decode_wall_s"] == pytest.approx(3.0)
    assert data["window_s"] == pytest.approx(21.0)
    assert (data["steps"], data["batch_min"], data["batch_max"]) == (3, 3, 7)


def test_a_window_without_steps_emits_nothing(captured):
    logger, records = captured
    counters = StepCounters(logger, 0.0)
    counters.begin_round(50.0)
    counters.end_round(50.0)
    assert counters.maybe_emit(50.0, 10.0) is False
    assert records == []
    # The window keeps running; the next step reports at once.
    counters.begin_round(60.0)
    counters.add_step(60.0, 2)
    assert counters.maybe_emit(60.0, 10.0) is True
    assert _data(records[0])["window_s"] == 60.0


def test_below_the_interval_no_payload_is_built(captured, monkeypatch):
    """The per-step path never reaches the payload or the logger."""
    logger, records = captured

    def _fail(*args, **kwargs):
        raise AssertionError("payload built below the interval")

    monkeypatch.setattr(StepCounters, "_emit", _fail)
    monkeypatch.setattr(logger, "isEnabledFor", _fail)
    counters = StepCounters(logger, 0.0)
    counters.begin_round(0.0)
    for i in range(10_000):
        now = i * 1e-4
        counters.add_step(now, 16)
        assert counters.maybe_emit(now, 10.0) is False
    assert records == []


def test_counters_have_fixed_slots_and_no_instance_dict(captured):
    logger, _ = captured
    counters = StepCounters(logger, 0.0)
    assert not hasattr(counters, "__dict__")
    with pytest.raises(AttributeError):
        counters.extra = 1  # type: ignore[attr-defined]


def test_disabled_level_skips_the_record_but_still_resets(captured):
    logger, records = captured
    logger.setLevel(logging.WARNING)
    counters = StepCounters(logger, 0.0)
    counters.begin_round(0.0)
    counters.add_step(10.0, 4)
    assert counters.maybe_emit(10.0, 10.0) is True
    assert records == []
    assert counters.steps == 0


def test_record_round_trips_through_the_jsonl_file(tmp_path):
    ev.configure("rank", node=0, rank=3, local_rank=3, run_id=tu.RUN_ID,
                 run_started_at=tu.RUN_STARTED_AT, log_dir=tmp_path,
                 console=False)
    counters = StepCounters(ev.get_logger("batchgen.batchgen_worker"), 0.0)
    counters.begin_round(0.0)
    counters.add_step(5.0, 12)
    counters.add_step(30.0, 10)
    assert counters.maybe_emit(30.0, 30.0) is True
    assert ev.flush(10.0)
    recs = [r for r in tu.read_jsonl(ev.jsonl_path()) if r["event"] == EVENT]
    assert len(recs) == 1
    rec = recs[0]
    assert (rec["level"], rec["proc"], rec["rank"], rec["logger"]) == (
        "INFO", "rank", 3, "batchgen.batchgen_worker")
    assert rec["data"] == {
        "steps": 2, "row_steps": 22, "batch_min": 10, "batch_max": 12,
        "decode_wall_s": 30.0, "window_s": 30.0, "total_steps": 2,
    }
