from __future__ import annotations

import importlib.util
import sys
from types import ModuleType, SimpleNamespace
from pathlib import Path

import pytest


def _load_coordinator_type():
    """Load the pure coordinator wrapper without optional GPU backends."""
    module_name = "batchgen.kv_cache.dual_host_kv_coordinator_test_module"
    source = (
        Path(__file__).resolve().parents[1]
        / "batchgen"
        / "kv_cache"
        / "dual_host_kv_coordinator.py"
    )
    saved = {
        name: sys.modules.get(name)
        for name in (
            "batchgen.kv_cache",
            "batchgen.models",
            "batchgen.models.engine_loader",
        )
    }
    fake_kv = ModuleType("batchgen.kv_cache")
    fake_kv.__path__ = []
    fake_models = ModuleType("batchgen.models")
    fake_models.__path__ = []
    fake_loader = ModuleType("batchgen.models.engine_loader")
    fake_loader.core_engine = SimpleNamespace()
    sys.modules["batchgen.kv_cache"] = fake_kv
    sys.modules["batchgen.models"] = fake_models
    sys.modules["batchgen.models.engine_loader"] = fake_loader
    try:
        spec = importlib.util.spec_from_file_location(module_name, source)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module.DualHostKVCoordinator
    finally:
        sys.modules.pop(module_name, None)
        for name, value in saved.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


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
    DualHostKVCoordinator = _load_coordinator_type()
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
    DualHostKVCoordinator = _load_coordinator_type()
    primary = _View(free_pages=1)
    auxiliary = _View(free_pages=4)
    coordinator = DualHostKVCoordinator(primary, auxiliary)

    with pytest.raises(RuntimeError, match="insufficient mirrored"):
        coordinator.grow_pages_for_sequences([(17, 2)])

    assert primary.grow_calls == []
    assert auxiliary.grow_calls == []
