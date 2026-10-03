from __future__ import annotations

from types import SimpleNamespace

import pytest

# Importing the BatchGen KV package loads the optional Triton GPU backends.
# Keep this unit test collectable in the lightweight dev environment; the
# project runtime environment exercises it with Triton and the native host-KV
# bindings available.
pytest.importorskip("triton")

from batchgen.kv_cache.dual_host_kv_coordinator import DualHostKVCoordinator


class _View:
    def __init__(self, *, free_pages: int, error: Exception | None = None) -> None:
        self._free_pages = free_pages
        self._error = error
        self.grow_calls = []

    def get_stats(self):
        return SimpleNamespace(num_free_pages=self._free_pages)

    def grow_pages_for_sequences(self, requests) -> None:
        self.grow_calls.append(list(requests))
        if self._error is not None:
            raise self._error
        self._free_pages -= sum(int(pages) for _, pages in requests)


def test_dual_host_growth_poisoned_after_auxiliary_rejects() -> None:
    primary = _View(free_pages=8)
    auxiliary = _View(free_pages=8, error=RuntimeError("aux allocator failed"))
    coordinator = DualHostKVCoordinator(primary, auxiliary)

    with pytest.raises(RuntimeError, match="partially applied"):
        coordinator.grow_pages_for_sequences([(17, 2)])

    # There is no safe page-level rollback in the native worker-view API.  The
    # coordinator must fail closed rather than permit another operation against
    # divergent shared-memory views.
    assert primary.grow_calls == [[(17, 2)]]
    assert auxiliary.grow_calls == [[(17, 2)]]
    assert primary.get_stats().num_free_pages == 6
    assert auxiliary.get_stats().num_free_pages == 8
    with pytest.raises(RuntimeError, match="poisoned"):
        coordinator.grow_pages_for_sequences([(17, 1)])


def test_dual_host_growth_preflight_rejects_without_mutating_either_view() -> None:
    primary = _View(free_pages=1)
    auxiliary = _View(free_pages=4)
    coordinator = DualHostKVCoordinator(primary, auxiliary)

    with pytest.raises(RuntimeError, match="insufficient mirrored"):
        coordinator.grow_pages_for_sequences([(17, 2)])

    assert primary.grow_calls == []
    assert auxiliary.grow_calls == []
