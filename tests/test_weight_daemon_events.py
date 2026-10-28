"""The weight daemon's events reach the server log as named weight_daemon.* events.

The watcher is driven by a fake daemon whose poll_events follows the binding:
dicts holding only the fields an event sets, dense sequence numbers, and a
dropped count computed from the oldest retained event like the C++ queue.
"""

import importlib.util
import itertools
import json
import logging
import re
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "batchgen/server/weight_daemon_events.py"


def _load_module():
    """Import the watcher alone (batchgen/__init__ pulls kernels)."""
    spec = importlib.util.spec_from_file_location(
        "_batchgen_weight_daemon_events", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


wde = _load_module()

# The structured event-log name rule (batchgen.eventlog.schema): two to four
# lowercase dotted segments, and prefixes the event log keeps for itself.
EVENT_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*){1,3}$")
RESERVED_EVENT_PREFIXES = ("log.", "eventlog.")

# The specification, written out rather than read back from the module.
EXPECTED = {
    "config": ("weight_daemon.config", logging.INFO),
    "network_ready": ("weight_daemon.network_ready", logging.INFO),
    "failed": ("weight_daemon.failed", logging.ERROR),
    "cohort_draining": ("weight_daemon.cohort.draining", logging.INFO),
    "cohort_admitting": ("weight_daemon.cohort.admitting", logging.INFO),
    "cohort_faulted": ("weight_daemon.cohort.faulted", logging.WARNING),
    "worker_admitted": ("weight_daemon.cohort.worker_admitted", logging.INFO),
    "hello_rejected": ("weight_daemon.hello.rejected", logging.WARNING),
    "acquire_stalled": ("weight_daemon.acquire.stalled", logging.WARNING),
    "stopped": ("weight_daemon.stopped", logging.INFO),
}

_STALL_FIELDS = {
    "module_key": "layer.12.experts",
    "fetching": 0,
    "ready": 4,
    "ready_unacquired": 4,
    "age_s": 30.5,
    "progressed": False,
    "oldest_unacquired": "layer.11.experts",
    "oldest_unacquired_worker": 3,
}


class _FakeDaemon:
    def __init__(self, *, first_seq=1, fail_blocking_polls=False):
        self._cond = threading.Condition()
        self._seq = itertools.count(first_seq)
        self._events = []
        self.fail_blocking_polls = fail_blocking_polls
        self.polls = []

    def push(self, kind, **fields):
        with self._cond:
            event = {"seq": next(self._seq), "wall_time_s": 1.0e9, "kind": kind}
            event.update(fields)
            self._events.append(event)
            self._cond.notify_all()
            return event["seq"]

    def _last_seq(self):
        return self._events[-1]["seq"] if self._events else 0

    def poll_events(self, after_seq, timeout_seconds):
        self.polls.append((after_seq, timeout_seconds))
        if self.fail_blocking_polls and timeout_seconds > 0:
            raise RuntimeError("poll failed")
        with self._cond:
            if timeout_seconds > 0:
                self._cond.wait_for(
                    lambda: self._last_seq() > after_seq, timeout=timeout_seconds
                )
            oldest = self._events[0]["seq"] if self._events else after_seq + 1
            dropped = max(0, oldest - after_seq - 1)
            events = [dict(e) for e in self._events if e["seq"] > after_seq]
        return events, dropped


class _Records(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.records = []

    def emit(self, record):
        self.records.append(record)


@pytest.fixture
def captured(request):
    event_logger = logging.getLogger(f"test.weight_daemon_events.{request.node.name}")
    event_logger.setLevel(logging.DEBUG)
    event_logger.propagate = False
    handler = _Records()
    event_logger.addHandler(handler)
    yield event_logger, handler.records
    event_logger.removeHandler(handler)


def _push_every_kind(daemon):
    daemon.push("config", cohort=1, detail=json.dumps({"transport": "rdma", "workers": 8}))
    daemon.push("network_ready")
    daemon.push("failed", message="ucp_get_nbx completion failed")
    daemon.push("cohort_draining", cohort=1)
    daemon.push(
        "cohort_admitting", cohort=2,
        detail=json.dumps({"from_cohort": 1, "to_cohort": 2, "released_leases": 4}),
    )
    daemon.push("cohort_faulted", cohort=2, worker=5, message="worker 5 disconnected")
    daemon.push("worker_admitted", cohort=2, worker=0)
    daemon.push("hello_rejected", message="stale cohort 1")
    daemon.push("acquire_stalled", cohort=2, worker=1, message="still waiting",
                **_STALL_FIELDS)
    daemon.push("stopped", cohort=2)


def _by_event(records):
    return {record.bg_event: record for record in records}


def test_every_kind_maps_to_its_named_event_and_level(captured):
    event_logger, records = captured
    daemon = _FakeDaemon()
    _push_every_kind(daemon)

    wde.WeightDaemonWatcher(daemon, event_logger).stop()

    assert [(r.bg_event, r.levelno) for r in records] == list(EXPECTED.values())
    for record in records:
        assert isinstance(record.bg_data, dict)
        assert record.getMessage().startswith(record.bg_event)
        assert "kind" not in record.bg_data


def test_every_event_name_satisfies_the_event_log_rule():
    names = [name for name, _ in wde.EVENT_NAMES.values()]
    names += [wde.EVENTS_DROPPED, wde.UNKNOWN_EVENT]
    assert len(set(names)) == len(names)
    for name in names:
        assert EVENT_NAME_RE.match(name), name
        assert not name.startswith(RESERVED_EVENT_PREFIXES), name
        assert name.startswith("weight_daemon.")
    assert {kind: value for kind, value in wde.EVENT_NAMES.items()} == EXPECTED


def test_record_carries_the_event_attributes_and_rendered_body(captured):
    event_logger, records = captured
    daemon = _FakeDaemon()
    daemon.push("cohort_faulted", cohort=2, worker=5, message="worker 5 disconnected")

    wde.WeightDaemonWatcher(daemon, event_logger).stop()

    (record,) = records
    assert record.bg_event == "weight_daemon.cohort.faulted"
    assert record.bg_data == {
        "seq": 1, "wall_time_s": 1.0e9, "cohort": 2, "worker": 5,
        "message": "worker 5 disconnected",
    }
    assert record.getMessage() == (
        'weight_daemon.cohort.faulted seq=1 wall_time_s=1000000000.0 cohort=2 '
        'worker=5 message="worker 5 disconnected"'
    )


def test_swap_report_and_config_detail_are_parsed_into_data(captured):
    event_logger, records = captured
    daemon = _FakeDaemon()
    _push_every_kind(daemon)

    wde.WeightDaemonWatcher(daemon, event_logger).stop()

    events = _by_event(records)
    admitting = events["weight_daemon.cohort.admitting"].bg_data
    assert admitting["from_cohort"] == 1
    assert admitting["to_cohort"] == 2
    assert admitting["released_leases"] == 4
    assert "detail" not in admitting
    config = events["weight_daemon.config"].bg_data
    assert config["transport"] == "rdma"
    assert config["workers"] == 8


def test_detail_that_is_not_a_json_object_is_kept_verbatim():
    name, level, data = wde.event_fields(
        {"seq": 3, "kind": "cohort_admitting", "detail": "not json"}
    )
    assert name == "weight_daemon.cohort.admitting"
    assert level == logging.INFO
    assert data == {"seq": 3, "detail": "not json"}


def test_event_fields_win_over_detail_keys():
    _, _, data = wde.event_fields(
        {"seq": 3, "kind": "cohort_admitting", "cohort": 2,
         "detail": json.dumps({"cohort": 99, "to_cohort": 2})}
    )
    assert data["cohort"] == 2
    assert data["to_cohort"] == 2


def test_unknown_kind_is_logged_not_lost():
    name, level, data = wde.event_fields({"seq": 1, "kind": "something_new"})
    assert name == wde.UNKNOWN_EVENT
    assert level == logging.WARNING
    assert data["kind"] == "something_new"


def test_stall_carries_the_ring_fields(captured):
    event_logger, records = captured
    daemon = _FakeDaemon()
    daemon.push("acquire_stalled", cohort=2, worker=1, message="still waiting",
                **_STALL_FIELDS)

    wde.WeightDaemonWatcher(daemon, event_logger).stop()

    (record,) = records
    assert record.levelno == logging.WARNING
    assert record.bg_event == "weight_daemon.acquire.stalled"
    for key, value in _STALL_FIELDS.items():
        assert record.bg_data[key] == value
    assert record.bg_data["worker"] == 1


def test_a_dropped_gap_is_reported_as_a_warning_before_the_events(captured):
    event_logger, records = captured
    # Events 1..4 were overflowed out of the daemon's queue.
    daemon = _FakeDaemon(first_seq=5)
    daemon.push("network_ready")

    wde.WeightDaemonWatcher(daemon, event_logger).stop()

    dropped, ready = records
    assert dropped.bg_event == "weight_daemon.events.dropped"
    assert dropped.levelno == logging.WARNING
    assert dropped.bg_data == {"dropped": 4, "after_seq": 0}
    assert ready.bg_event == "weight_daemon.network_ready"


def test_the_stopped_event_ends_the_thread(captured):
    event_logger, records = captured
    daemon = _FakeDaemon()
    daemon.push("config", cohort=1)
    watcher = wde.WeightDaemonWatcher(daemon, event_logger)
    watcher.start()
    thread = watcher._thread

    daemon.push("network_ready")
    daemon.push("stopped", cohort=1)
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert [r.bg_event for r in records] == [
        "weight_daemon.config", "weight_daemon.network_ready", "weight_daemon.stopped",
    ]
    # Polls block in the daemon rather than spinning.
    assert all(timeout == wde.DAEMON_POLL_TIMEOUT_S for _, timeout in daemon.polls)

    # stop() after the thread ended drains nothing twice.
    watcher.stop()
    assert len(records) == 3
    assert daemon.polls[-1] == (3, 0.0)


def test_stop_joins_then_drains_once_without_waiting(captured):
    event_logger, records = captured
    daemon = _FakeDaemon()
    daemon.push("config", cohort=1)
    watcher = wde.WeightDaemonWatcher(daemon, event_logger)
    watcher.start()
    thread = watcher._thread

    watcher.stop()

    assert not thread.is_alive()
    # Logged exactly once, by the thread or by the final drain.
    assert [r.bg_event for r in records] == ["weight_daemon.config"]
    assert daemon.polls[-1][1] == 0.0


def test_stop_drains_what_a_failed_thread_left(captured):
    event_logger, records = captured
    daemon = _FakeDaemon(fail_blocking_polls=True)
    watcher = wde.WeightDaemonWatcher(daemon, event_logger)
    watcher.start()
    watcher._thread.join(timeout=5)
    assert records[-1].levelno == logging.ERROR
    assert records[-1].getMessage() == "Weight daemon event watcher failed"

    # The daemon's stop pushes its last event before the watcher is stopped.
    daemon.push("failed", message="transport failure")
    daemon.push("stopped", cohort=1)
    watcher.stop()

    events = [getattr(r, "bg_event", None) for r in records]
    assert events[-2:] == ["weight_daemon.failed", "weight_daemon.stopped"]
    assert records[-2].levelno == logging.ERROR


def test_stop_never_raises_when_the_final_drain_fails(captured):
    event_logger, records = captured

    class Broken:
        def poll_events(self, after_seq, timeout_seconds):
            raise RuntimeError("daemon gone")

    wde.WeightDaemonWatcher(Broken(), event_logger).stop()
    assert records[-1].getMessage() == "Final drain of weight daemon events failed"
