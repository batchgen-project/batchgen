"""A SIGKILL'd worker is attributed to the kernel OOM killer, not to NCCL.

Pure CPU. The kernel ring buffer is never the running machine's: every test
feeds the parser lines captured from the 2026-11-03 incident, or a file
standing in for ``/dev/kmsg``.
"""

from __future__ import annotations

import logging
import signal
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.test_worker_pool_restart import (  # noqa: E402
    _load_worker_manager_class,
    _manager,
)

WorkerManager = _load_worker_manager_class()

# Captured from the 2026-11-03 incident (bug_log.md): the global OOM killer
# took one worker whose shmem-rss is the shared weights + host-KV memfds every
# rank maps. /dev/kmsg prefixes each record with "prio,seq,timestamp,flags;".
OOM_RECORDS = [
    "6,2201,431905271,-;ps invoked oom-killer: "
    "gfp_mask=0x140cca(GFP_HIGHUSER_MOVABLE|__GFP_COMP), order=0, "
    "oom_score_adj=0\n",
    "3,2244,431905993,-;Out of memory: Killed process 4242 (pt_main_thread) "
    "total-vm:1673723116kB, anon-rss:19923416kB, file-rss:2048kB, "
    "shmem-rss:1573723116kB, UID:0 pgtables:3104768kB oom_score_adj:999\n",
]


def _exit_reason_manager(processes, *, kmsg_records=None, record_reads=None):
    """A manager stripped to what _collect_worker_exit_reason reads."""

    manager = object.__new__(WorkerManager)
    manager.args = SimpleNamespace(node_rank=0)
    manager.worker_process = SimpleNamespace(processes=processes, error_files=())

    def read_ring_buffer(path=None):
        if record_reads is not None:
            record_reads.append(path)
        return list(kmsg_records or ())

    manager._read_kernel_ring_buffer = read_ring_buffer
    return manager


def _worker(pid, exitcode):
    return SimpleNamespace(pid=pid, exitcode=exitcode)


# --------------------------------------------------------------------------- #
#  The kmsg parser                                                            #
# --------------------------------------------------------------------------- #

def test_parser_finds_the_victim_pid_in_the_incident_records():
    killed = WorkerManager._parse_oom_kill_records(OOM_RECORDS)
    assert set(killed) == {4242}
    assert "shmem-rss:1573723116kB" in killed[4242]
    # The kmsg prefix is not part of the message.
    assert killed[4242].startswith("Out of memory: Killed process 4242")


def test_parser_reads_a_cgroup_kill_and_an_unprefixed_dmesg_line():
    killed = WorkerManager._parse_oom_kill_records(
        [
            "3,99,1,-;Memory cgroup out of memory: Killed process 77 (python3) "
            "total-vm:1kB, anon-rss:1kB\n",
            "Out of memory: Killed process 88 (python3) total-vm:1kB\n",
        ]
    )
    assert set(killed) == {77, 88}


def test_parser_ignores_everything_that_is_not_an_oom_kill():
    assert WorkerManager._parse_oom_kill_records(
        [
            "6,1,1,-;NET: Registered PF_VSOCK protocol family\n",
            "4,2,2,-;nvidia-uvm: Loaded the UVM driver\n",
            "3,3,3,-;traps: python3[4242] general protection fault\n",
        ]
    ) == {}


def test_parser_handles_multi_line_records_and_keeps_the_newest_kill():
    killed = WorkerManager._parse_oom_kill_records(
        [
            "3,1,1,-;Out of memory: Killed process 55 (python3) anon-rss:1kB\n"
            " SUBSYSTEM=acpi\n DEVICE=+acpi:PNP0C0C:00\n",
            "3,2,2,-;Out of memory: Killed process 55 (python3) anon-rss:2kB\n",
        ]
    )
    assert killed[55].endswith("anon-rss:2kB")


# --------------------------------------------------------------------------- #
#  Reading the ring buffer                                                    #
# --------------------------------------------------------------------------- #

def test_ring_buffer_read_is_silent_when_the_buffer_is_hidden(tmp_path):
    # An unprivileged container cannot open /dev/kmsg. Attribution is a
    # diagnostic and may not change the failure path it describes.
    assert WorkerManager._read_kernel_ring_buffer(
        str(tmp_path / "there-is-no-kmsg-here")
    ) == []


def test_ring_buffer_read_returns_the_records_it_can_read(tmp_path):
    captured = tmp_path / "kmsg"
    captured.write_text("".join(OOM_RECORDS))
    records = WorkerManager._read_kernel_ring_buffer(str(captured))
    killed = WorkerManager._parse_oom_kill_records(records)
    assert 4242 in killed


# --------------------------------------------------------------------------- #
#  Attribution on reap                                                        #
# --------------------------------------------------------------------------- #

def test_sigkilled_worker_matching_a_kmsg_kill_is_named_as_root_cause(caplog):
    manager = _exit_reason_manager(
        [_worker(4241, None), _worker(4242, -signal.SIGKILL)],
        kmsg_records=OOM_RECORDS,
    )
    with caplog.at_level(logging.ERROR):
        reason = manager._collect_worker_exit_reason()

    root_cause = manager._host_oom_root_cause
    assert root_cause is not None
    for figure in ("rank-idx=1", "node=0", "pid=4242",
                   "killed by the kernel OOM killer",
                   "[MEM-PLAN host]"):
        assert figure in root_cause
    # Exactly one ERROR names it, and it carries the kernel's own line.
    attributions = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.ERROR and "ROOT CAUSE" in record.getMessage()
    ]
    assert len(attributions) == 1
    assert "shmem-rss:1573723116kB" in attributions[0]
    # The root cause leads the reason, ahead of the per-rank exit details.
    assert reason.startswith(root_cause)
    assert "idx=1 pid=4242 exitcode=-9" in reason


def test_attribution_is_recorded_once_per_cohort(caplog):
    reads = []
    manager = _exit_reason_manager(
        [_worker(4242, -signal.SIGKILL)],
        kmsg_records=OOM_RECORDS,
        record_reads=reads,
    )
    with caplog.at_level(logging.ERROR):
        first = manager._collect_worker_exit_reason()
        second = manager._collect_worker_exit_reason()
    # The monitor hands the same dead cohort over on every poll; the kernel is
    # asked once and the line is logged once.
    assert first == second
    assert len(reads) == 1
    assert sum(
        "ROOT CAUSE" in record.getMessage() for record in caplog.records
    ) == 1


def test_no_matching_kill_leaves_behavior_unchanged():
    manager = _exit_reason_manager(
        [_worker(4242, -signal.SIGKILL)],
        kmsg_records=[
            "3,1,1,-;Out of memory: Killed process 777 (sshd) anon-rss:1kB\n"
        ],
    )
    reason = manager._collect_worker_exit_reason()
    assert manager._host_oom_root_cause is None
    assert reason == "Detected worker process exit: idx=0 pid=4242 exitcode=-9"


def test_the_match_is_on_pid_not_on_the_exit_status():
    """The manager interprets no exit status, so neither does the lookup.

    ``test_the_manager_interprets_no_exit_status_specially`` pins that
    contract. The kernel's record is keyed by pid, which is the only thing
    that decides here.
    """

    reads = []
    manager = _exit_reason_manager(
        [_worker(4242, 1)], kmsg_records=OOM_RECORDS, record_reads=reads
    )
    reason = manager._collect_worker_exit_reason()
    assert len(reads) == 1
    assert "pid=4242" in (manager._host_oom_root_cause or "")
    assert "idx=0 pid=4242 exitcode=1" in reason


def test_a_newly_reaped_rank_is_looked_up_too():
    # The OOM'd rank can take minutes to become reapable (the kernel is still
    # unpinning its registered pages), so it may well appear in a later poll
    # than the peer whose collective reported the failure first.
    processes = [_worker(4241, 1), _worker(4242, None)]
    reads = []
    manager = _exit_reason_manager(
        processes, kmsg_records=OOM_RECORDS, record_reads=reads
    )
    assert manager._collect_worker_exit_reason() is not None
    assert manager._host_oom_root_cause is None
    processes[1] = _worker(4242, -signal.SIGKILL)
    manager.worker_process = SimpleNamespace(processes=processes, error_files=())
    reason = manager._collect_worker_exit_reason()
    assert len(reads) == 2
    assert "rank-idx=1" in (manager._host_oom_root_cause or "")
    assert reason.startswith(manager._host_oom_root_cause)


def test_a_live_cohort_asks_nothing_and_reports_nothing():
    reads = []
    manager = _exit_reason_manager(
        [_worker(4242, None)], kmsg_records=OOM_RECORDS, record_reads=reads
    )
    assert manager._collect_worker_exit_reason() is None
    assert reads == []


# --------------------------------------------------------------------------- #
#  The cascade references the root cause                                      #
# --------------------------------------------------------------------------- #

def test_cascade_reason_is_marked_secondary_to_the_root_cause():
    manager = object.__new__(WorkerManager)
    manager._host_oom_root_cause = None
    nccl = "Detected worker process exit: idx=3 NCCL communicator was aborted"
    # Nothing known yet: a failure reason is untouched.
    assert manager._as_secondary_to_host_oom(nccl) == nccl
    manager._host_oom_root_cause = "worker rank-idx=1 ... OOM killer"
    marked = manager._as_secondary_to_host_oom(nccl)
    assert marked.startswith("secondary to host-OOM root cause [")
    assert manager._host_oom_root_cause in marked
    assert nccl in marked
    # The root cause is not reported as secondary to itself, and marking is
    # idempotent: the monitor hands the same reason over on every poll.
    assert manager._as_secondary_to_host_oom(marked) == marked
    assert (
        manager._as_secondary_to_host_oom(manager._host_oom_root_cause)
        == manager._host_oom_root_cause
    )


def test_workers_down_names_the_root_cause_not_the_first_symptom(
    monkeypatch, caplog
):
    manager = _manager(monkeypatch)

    class ExitState:
        reason = None

        def set_failure(self, reason, exc):
            self.reason = reason
            return True

    exit_state = ExitState()
    manager._worker_exit_state = exit_state
    reconciles = []
    manager._reconcile = lambda reason, fail_in_flight: reconciles.append(reason)
    manager._host_oom_root_cause = (
        "worker rank-idx=1 (node=0 pid=4242) was killed by the kernel OOM "
        "killer -- host memory exhausted; see the [MEM-PLAN host] block at "
        "startup"
    )
    with caplog.at_level(logging.ERROR, logger="uvicorn.error"):
        manager._handle_worker_failure(
            "DistBackendError: NCCL communicator was aborted", None
        )

    cohort_failures = [
        record.getMessage()
        for record in caplog.records
        if "entering workers_down" in record.getMessage()
    ]
    assert len(cohort_failures) == 1
    # Every consumer of the reason carries the cause, not the symptom: this
    # node's log, the state the restart path reads back, and the reconcile.
    for text in (cohort_failures[0], exit_state.reason, reconciles[0]):
        assert "secondary to host-OOM root cause" in text
        assert "pid=4242" in text
        assert "NCCL communicator was aborted" in text


def test_a_fresh_cohort_starts_without_the_previous_root_cause(monkeypatch):
    manager = _manager(monkeypatch)
    manager._host_oom_root_cause = "worker rank-idx=1 ... OOM killer"
    manager.restart_workers()
    assert manager._host_oom_root_cause is None
