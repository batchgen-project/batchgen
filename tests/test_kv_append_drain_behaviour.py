"""Executable behaviour of the KV-append drain's ready fast path.

The real ``_wait_pending_kv_append_tasks`` and ``_pending_kv_appends_done`` are
lifted from the worker source (``batchgen_worker`` cannot be imported on a
CPU-only machine) and driven with fake tasks and a fake ``torch``, so the
device sync is observable as a call count rather than a GPU effect.

The contract under test:

* all futures already ready AND the caller opted in -> join (no-op) and SKIP
  the device sync;
* anything else -> today's behaviour exactly: join all, then sync;
* the default (no opt-in) is today's behaviour even when everything is ready,
  because those callers are about to free pages or unwind;
* the error and cross-rank paths are untouched by the flag.
"""
import types

import pytest

from tests.worker_method_source import load_worker_methods


class FakeTask:
    def __init__(self, ready=True, raises=None):
        self._ready = ready
        self._raises = raises
        self.waited = 0

    def done(self):
        return self._ready

    def wait(self):
        self.waited += 1
        if self._raises is not None:
            raise self._raises


class FakeCuda:
    def __init__(self):
        self.syncs = []

    def synchronize(self, device):
        self.syncs.append(device)


class FakeTorch:
    def __init__(self):
        self.cuda = FakeCuda()


class FakeDist:
    def __init__(self, peer_payloads=()):
        self.gathers = 0
        self._peer = list(peer_payloads)

    def all_gather_object(self, out, payload):
        self.gathers += 1
        out[0] = payload
        for i, value in enumerate(self._peer, start=1):
            if i < len(out):
                out[i] = value


def _drain(tasks, *, world_size=1, dist_obj=None, tensors=None):
    """Return (bound_drain, stub, fake_torch, fake_dist)."""
    fake_torch = FakeTorch()
    fake_dist = dist_obj if dist_obj is not None else FakeDist()
    loaded = load_worker_methods(
        ["_wait_pending_kv_append_tasks", "_pending_kv_appends_done"],
        {
            "torch": fake_torch,
            "dist": fake_dist,
            "logging": __import__("logging"),
        },
    )
    stub = types.SimpleNamespace(
        rank=0,
        world_size=world_size,
        torch_device="cuda:0",
        _pending_kv_append_tasks=list(tasks),
    )
    if tensors is not None:
        stub._pending_kv_append_tensors = tensors
    stub._pending_kv_appends_done = types.MethodType(
        loaded["_pending_kv_appends_done"], stub
    )
    drain = types.MethodType(loaded["_wait_pending_kv_append_tasks"], stub)
    return drain, stub, fake_torch, fake_dist


# ------------------------------------------------------------------ fast path

def test_all_ready_and_opted_in_skips_the_device_sync():
    tasks = [FakeTask(ready=True), FakeTask(ready=True)]
    drain, stub, torch_, _ = _drain(tasks)
    assert drain(defer_errors=True, ready_fast_path=True) == 2
    assert torch_.cuda.syncs == [], "the device sync must be skipped"
    # The joins still run: they are what guards the host page-table race, and
    # on ready futures they return immediately.
    assert [t.waited for t in tasks] == [1, 1]
    assert stub._pending_kv_append_tasks == []
    assert stub._kv_drain_ready_fast == 1
    assert getattr(stub, "_kv_drain_ready_slow", 0) == 0


def test_one_task_in_flight_falls_back_to_the_device_sync():
    tasks = [FakeTask(ready=True), FakeTask(ready=False), FakeTask(ready=True)]
    drain, stub, torch_, _ = _drain(tasks)
    assert drain(defer_errors=True, ready_fast_path=True) == 3
    assert torch_.cuda.syncs == ["cuda:0"], "the fallback must sync the device"
    assert [t.waited for t in tasks] == [1, 1, 1]
    assert getattr(stub, "_kv_drain_ready_fast", 0) == 0
    assert stub._kv_drain_ready_slow == 1


def test_a_none_entry_is_skipped_by_the_probe_as_by_the_join():
    tasks = [None, FakeTask(ready=True), None]
    drain, _, torch_, _ = _drain(tasks)
    assert drain(defer_errors=True, ready_fast_path=True) == 3
    assert torch_.cuda.syncs == []


def test_an_empty_task_list_never_syncs_and_is_not_counted():
    drain, stub, torch_, _ = _drain([])
    assert drain(defer_errors=True, ready_fast_path=True) == 0
    assert torch_.cuda.syncs == []
    # No work means neither path ran; counting it would dilute the heartbeat.
    assert not hasattr(stub, "_kv_drain_ready_fast")


# ------------------------------------------------------------------ slow path

def test_the_default_keeps_the_device_sync_even_when_everything_is_ready():
    tasks = [FakeTask(ready=True)]
    drain, stub, torch_, _ = _drain(tasks)
    assert drain(defer_errors=True) == 1
    assert torch_.cuda.syncs == ["cuda:0"], (
        "a caller that did not opt in is about to free pages or unwind")
    # An un-opted drain is not one of the two counted paths.
    assert not hasattr(stub, "_kv_drain_ready_fast")


@pytest.mark.parametrize("ready", [True, False])
def test_a_join_error_suppresses_the_sync_on_both_paths(ready):
    boom = FakeTask(ready=ready, raises=RuntimeError("append failed"))
    tasks = [FakeTask(ready=ready), boom]
    drain, stub, torch_, _ = _drain(tasks)
    drain(defer_errors=True, ready_fast_path=True)
    # Unchanged from today: a failed join skips the sync and defers the error.
    assert torch_.cuda.syncs == []
    assert stub._deferred_kv_append_wait_errors == ["RuntimeError: append failed"]


def test_deferred_errors_surface_at_the_next_non_deferring_drain():
    drain, stub, torch_, _ = _drain([FakeTask(ready=True)])
    stub._deferred_kv_append_wait_errors = ["RuntimeError: earlier"]
    with pytest.raises(RuntimeError, match="earlier"):
        drain(ready_fast_path=True)


def test_the_cross_rank_error_sync_is_untouched_by_the_flag():
    fake_dist = FakeDist()
    drain, _, torch_, dist_ = _drain(
        [FakeTask(ready=True)], world_size=2, dist_obj=fake_dist
    )
    assert drain(sync_distributed_errors=True, ready_fast_path=True) == 1
    assert dist_.gathers == 1, "the collective must still run"
    assert torch_.cuda.syncs == []


def test_a_peer_error_still_raises_on_the_fast_path():
    fake_dist = FakeDist(peer_payloads=[{"rank": 1, "errors": ["boom"]}])
    drain, _, _, _ = _drain(
        [FakeTask(ready=True)], world_size=2, dist_obj=fake_dist
    )
    with pytest.raises(RuntimeError, match="at least one rank"):
        drain(sync_distributed_errors=True, ready_fast_path=True)


def test_tensor_references_are_released_on_both_paths():
    for ready in (True, False):
        tensors = ["k", "v"]
        drain, stub, _, _ = _drain(
            [FakeTask(ready=ready)], tensors=tensors
        )
        drain(defer_errors=True, ready_fast_path=True)
        assert stub._pending_kv_append_tensors == [], (
            "the source tensors must be released once the copies are done")


def test_a_worker_without_the_task_list_is_unaffected():
    loaded = load_worker_methods(
        ["_wait_pending_kv_append_tasks"],
        {"torch": FakeTorch(), "dist": FakeDist(), "logging": __import__("logging")},
    )
    stub = types.SimpleNamespace(rank=0, world_size=1, torch_device="cuda:0")
    drain = types.MethodType(loaded["_wait_pending_kv_append_tasks"], stub)
    assert drain(defer_errors=True, ready_fast_path=True) == 0
