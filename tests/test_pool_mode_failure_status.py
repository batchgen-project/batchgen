import asyncio
import importlib.util
import sys
import types
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]


def _load_real(name, monkeypatch):
    path = _ROOT.joinpath(*name.split(".")).with_suffix(".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def _stub(name, monkeypatch, **attrs):
    module = types.ModuleType(name)
    module.__path__ = []
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


@pytest.fixture
def scheduler_env(monkeypatch):
    # Stub the package parents so batchgen.server.__init__ (which pulls the whole
    # CUDA core_engine + model registry) never runs; load only the pure-Python
    # scheduler deps for real.
    _stub("batchgen", monkeypatch)
    _stub("batchgen.server", monkeypatch)
    io_struct = _load_real("batchgen.server.io_struct", monkeypatch)
    _load_real("batchgen.server.intake_pool", monkeypatch)
    _stub(
        "batchgen.server.worker_manager",
        monkeypatch,
        WorkerManager=type("WorkerManager", (), {}),
    )
    _stub(
        "batchgen.server.scheduling_pool",
        monkeypatch,
        SchedulingPool=type("SchedulingPool", (), {}),
    )
    _stub(
        "batchgen.server.server_args",
        monkeypatch,
        ServerArgs=type("ServerArgs", (), {}),
    )
    _stub(
        "batchgen.server.storage",
        monkeypatch,
        StorageManager=type("StorageManager", (), {}),
    )
    batch_scheduler = _load_real("batchgen.server.batch_scheduler", monkeypatch)
    return batch_scheduler.BatchScheduler, io_struct.BatchStatus


class _FakeStorage:
    def __init__(self):
        self.calls = []

    def update_batch_status(self, batch_id, status, **updates):
        self.calls.append((batch_id, status, updates))


class _RejectingIntakePool:
    max_capacity = 4

    def size(self):
        return 4

    def submit_batch(self, batch_id, entries, priority):
        return False


class _EmptySchedulingPool:
    def register_batch(self, **kwargs):
        return None

    def get_batch_tracker(self, batch_id):
        return None


def _bare_scheduler(BatchScheduler, storage):
    sched = BatchScheduler.__new__(BatchScheduler)
    sched.storage = storage
    sched._scheduling_pool = _EmptySchedulingPool()
    return sched


def test_pool_capacity_rejection_marks_batch_failed(scheduler_env):
    BatchScheduler, BatchStatus = scheduler_env
    # Given a pool-mode scheduler whose intake pool is at capacity
    storage = _FakeStorage()
    sched = _bare_scheduler(BatchScheduler, storage)
    sched._intake_pool = _RejectingIntakePool()
    req = types.SimpleNamespace(custom_id="r0")
    batch = types.SimpleNamespace(batchgen_debug={})

    # When a batch is submitted and the intake pool rejects it
    asyncio.run(
        sched._process_batch_pool_mode(
            batch_id="b0",
            batch=batch,
            requests=[req],
            prompts=["hi"],
            per_request_max_tokens=[8],
            sampling_params=[{}],
            incremental_kwargs={},
        )
    )

    # Then it is marked FAILED via update_batch_status (regression guard for the missing update_batch)
    assert len(storage.calls) == 1
    batch_id, status, updates = storage.calls[0]
    assert batch_id == "b0"
    assert status is BatchStatus.FAILED
    assert "capacity_exceeded" in updates["error"]


def test_pool_batch_timeout_marks_batch_failed(scheduler_env):
    BatchScheduler, BatchStatus = scheduler_env
    # Given a pool-mode scheduler whose batch deadline is already past and no tracker exists
    storage = _FakeStorage()
    sched = _bare_scheduler(BatchScheduler, storage)
    sched._batch_timeout = -1

    # When the finalize task waits for a batch that never completes
    asyncio.run(sched._wait_and_finalize_batch("b1", requests=[], prompts=[]))

    # Then it is marked FAILED via update_batch_status with a batch_failed error
    assert len(storage.calls) == 1
    batch_id, status, updates = storage.calls[0]
    assert batch_id == "b1"
    assert status is BatchStatus.FAILED
    assert "batch_failed" in updates["error"]
