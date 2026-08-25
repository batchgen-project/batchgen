import runpy
from pathlib import Path


_MODULE_PATH = (
    Path(__file__).parents[1] / "batchgen" / "server" / "worker_readiness.py"
)
_signal_local_worker_manager_ready = runpy.run_path(str(_MODULE_PATH))[
    "_signal_local_worker_manager_ready"
]


class _ReadyEvent:
    def __init__(self) -> None:
        self.set_calls = 0

    def set(self) -> None:
        self.set_calls += 1


def test_each_node_local_rank_zero_signals_its_worker_manager() -> None:
    for global_rank in (0, 8):
        event = _ReadyEvent()

        signaled = _signal_local_worker_manager_ready(
            event,
            local_rank=0,
            global_rank=global_rank,
        )

        assert signaled
        assert event.set_calls == 1


def test_nonzero_local_rank_does_not_signal_worker_manager() -> None:
    event = _ReadyEvent()

    signaled = _signal_local_worker_manager_ready(
        event,
        local_rank=1,
        global_rank=9,
    )

    assert not signaled
    assert event.set_calls == 0


def test_missing_ready_event_is_ignored() -> None:
    assert not _signal_local_worker_manager_ready(
        None,
        local_rank=0,
        global_rank=8,
    )
