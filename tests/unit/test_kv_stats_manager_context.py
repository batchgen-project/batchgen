"""CPU checks for the context-built KV stats manager.

The stats backend used to be a class nested inside the worker-shaped
``_make_kv_cache_manager`` that closed over the whole worker. It is now a
module-level class constructed from the frozen context's two KV handles, and
the manager is built fresh on every call. These tests pin both halves:

- the backend holds exactly the two references it is constructed with, and
  neither the backend class nor ``_make_kv_cache_manager`` mentions a worker;
- ``_get_host_kv_free_pages`` still returns what the pre-move code returned
  for every stats-tier flag combination, reading the same host view at the
  same moments.

The worker module cannot be imported here (triton), so everything is driven
from the real source through the loaders in ``tests.worker_method_source``.
"""

from __future__ import annotations

import ast
import types
from typing import Optional

from batchgen.worker.kv_manager import KVCacheManager, KVStats

from tests.worker_method_source import (
    KV_CAPACITY_SOURCE,
    load_worker_shaped_methods,
)


def _engine_tree() -> ast.Module:
    return ast.parse(
        KV_CAPACITY_SOURCE.read_text(), filename=str(KV_CAPACITY_SOURCE)
    )


def _module_level_class(name: str) -> ast.ClassDef:
    for node in _engine_tree().body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"{KV_CAPACITY_SOURCE} has no module-level class {name}")


def _engine_method(name: str) -> ast.FunctionDef:
    for node in _engine_tree().body:
        if isinstance(node, ast.ClassDef) and node.name == "KVCapacityEngine":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == name:
                    return item
    raise AssertionError(f"KVCapacityEngine has no method {name}")


def _identifiers(node: ast.AST) -> set:
    names = set()
    for item in ast.walk(node):
        if isinstance(item, ast.Name):
            names.add(item.id)
        elif isinstance(item, ast.arg):
            names.add(item.arg)
        elif isinstance(item, ast.Attribute):
            names.add(item.attr)
        elif isinstance(item, ast.keyword) and item.arg is not None:
            names.add(item.arg)
    return names


def _compile_backend_class():
    node = _module_level_class("_TorchKVStatsBackend")
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    scope = {"KVStats": KVStats, "Optional": Optional}
    exec(compile(module, str(KV_CAPACITY_SOURCE), "exec"), scope)  # noqa: S102
    return scope["_TorchKVStatsBackend"]


class _FakeHostView:
    """Counts its reads so the moments of each read stay observable."""

    def __init__(self, free: int, used: int, total: int) -> None:
        self._stats = types.SimpleNamespace(
            num_free_pages=free, num_used_pages=used, num_total_pages=total
        )
        self.calls = 0

    def get_stats(self):
        self.calls += 1
        return self._stats


def test_backend_holds_exactly_the_two_constructed_references():
    backend_class = _compile_backend_class()
    host_view = object()
    gpu_kv = object()
    backend = backend_class(host_view, gpu_kv)
    assert vars(backend) == {"_host_view": host_view, "_gpu_kv": gpu_kv}


def test_backend_reads_the_given_handles_and_tolerates_unbound_gpu():
    backend_class = _compile_backend_class()
    host_view = _FakeHostView(free=11, used=5, total=16)
    backend = backend_class(host_view, None)
    assert backend.get_host_stats() == KVStats(
        num_free_pages=11, num_used_pages=5, num_total_pages=16
    )
    assert backend.get_gpu_stats() is None
    assert host_view.calls == 1


def test_backend_and_manager_factory_never_mention_a_worker():
    assert "worker" not in _identifiers(_module_level_class("_TorchKVStatsBackend"))
    assert "worker" not in _identifiers(_engine_method("_make_kv_cache_manager"))


# --------------------------------------------------------------------------- #
#  _get_host_kv_free_pages behaviour across the stats-tier flags
# --------------------------------------------------------------------------- #


def _stats_worker(*, native: bool, compare: bool, host_view) -> types.SimpleNamespace:
    """A stand-in carrying exactly what the free-pages read touches."""
    worker = types.SimpleNamespace(
        _kv_stats_native=native,
        _kv_stats_compare=compare,
        host_paged_kv_worker_view=host_view,
        gpu_paged_kv_cache_manager=None,
    )
    methods = load_worker_shaped_methods(
        [
            "_get_host_kv_free_pages",
            "_legacy_get_host_kv_free_pages",
            "_make_kv_cache_manager",
        ],
        {
            "KVCacheManager": KVCacheManager,
            "_TorchKVStatsBackend": _compile_backend_class(),
            "getattr": getattr,
        },
    )
    for name, function in methods.items():
        setattr(worker, name, types.MethodType(function, worker))
    return worker


def test_free_pages_legacy_path_matches_the_old_return():
    """compare=0, native=0: one legacy read of the host view."""
    view = _FakeHostView(free=42, used=58, total=100)
    worker = _stats_worker(native=False, compare=False, host_view=view)
    assert worker._get_host_kv_free_pages() == 42
    assert view.calls == 1


def test_free_pages_native_path_matches_the_old_return():
    """compare=0, native=1: one read through the fresh manager."""
    view = _FakeHostView(free=7, used=3, total=10)
    worker = _stats_worker(native=True, compare=False, host_view=view)
    assert worker._get_host_kv_free_pages() == 7
    assert view.calls == 1


def test_free_pages_compare_path_runs_both_and_returns_legacy():
    """compare=1, native=0: both paths read the view; legacy is returned."""
    view = _FakeHostView(free=9, used=1, total=10)
    worker = _stats_worker(native=False, compare=True, host_view=view)
    assert worker._get_host_kv_free_pages() == 9
    assert view.calls == 2
