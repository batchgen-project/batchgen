"""Unit tests for decode/cleanup.py (Phase 2.8.2g)."""

from __future__ import annotations

from batchgen.worker.decode.cleanup import decode_cleanup
from tests.unit.worker.fakes import FakeLegacyBackend


class _StubBoundaryHandler:
    def __init__(self) -> None:
        self._pending_async_task = None
        self._pending_load_uuids: list[str] = []
        self._pending_load_local: list[int] = []
        self._pending_load_global: list[int] = []


class _FakeTask:
    def __init__(self) -> None:
        self.waited = False

    def wait(self) -> None:
        self.waited = True


class TestDecodeCleanup:
    def test_always_drains_and_unbinds(self) -> None:
        legacy = FakeLegacyBackend()
        boundary = _StubBoundaryHandler()

        decode_cleanup(legacy, boundary)

        names = [c[0] for c in legacy.calls]
        assert "wait_pending_kv_append_tasks" in names
        assert "unbind_decode_context" in names
        assert "disable_decode_watchdog" in names
        # No pending stash → no wait_async_load_task
        assert "wait_async_load_task" not in names

    def test_pending_async_task_waited_and_cleared(self) -> None:
        legacy = FakeLegacyBackend()
        boundary = _StubBoundaryHandler()
        task = _FakeTask()
        boundary._pending_async_task = task
        boundary._pending_load_uuids = ["load"]
        boundary._pending_load_local = [7]
        boundary._pending_load_global = [42]

        decode_cleanup(legacy, boundary)

        assert any(c[0] == "wait_async_load_task" for c in legacy.calls)
        # Stash cleared
        assert boundary._pending_async_task is None
        assert boundary._pending_load_uuids == []
        assert boundary._pending_load_local == []
        assert boundary._pending_load_global == []

    def test_ordering_drain_then_wait_then_unbind(self) -> None:
        """Order matters: draining deferred KV writes first guarantees
        host-KV consistency before the async handle goes out of scope.
        Unbind happens last so the bind state is visible during the
        drain (if any trailing iteration touches it)."""
        legacy = FakeLegacyBackend()
        boundary = _StubBoundaryHandler()
        boundary._pending_async_task = _FakeTask()

        decode_cleanup(legacy, boundary)

        names = [c[0] for c in legacy.calls]
        drain_idx = names.index("wait_pending_kv_append_tasks")
        wait_idx = names.index("wait_async_load_task")
        unbind_idx = names.index("unbind_decode_context")
        disable_idx = names.index("disable_decode_watchdog")
        assert drain_idx < wait_idx < unbind_idx < disable_idx
