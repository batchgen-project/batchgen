"""One decode-phase context per decode wave (AS3.3).

The decode admission context is built exactly ONCE per decode phase, at
decode-phase entry in ``generate`` (after the wave's GPU KV pool init), held
for the whole phase, and dropped on every path out. ``_decode_phase_context``
returns the held context inside a phase and a fresh frozen snapshot outside
one — outside a phase every context field is setup-stable, so a per-call
build is correct there.

Covered here:

* executable: a simulated decode phase builds the context exactly once, every
  in-phase read returns THE held object, and a field reassigned on the worker
  mid-phase does NOT leak into the held context (no silent rebuild);
* executable: the exit bracket drops the context, after which an out-of-phase
  read builds fresh; a nested entry raises by name;
* executable: ``_bind_gpu_paged_kv_manager`` — the one post-setup
  reassignment point of a context field — refuses to rebind while a
  decode-phase context is held;
* source-level: ``generate`` enters the bracket once per decode wave, after
  ``_init_gpu_kv_with_actual_size``, and the whole remainder of the wave sits
  in a ``try`` whose ``finally`` is exactly the exit bracket (so completion
  break, cohort-shutdown break, watermark break, loop-back and exceptions all
  drop it);
* source-level: the brackets have no other call site, the builder is reached
  only through the brackets/accessor, and ``decoding_continuous`` refuses to
  run with no held context.
"""

from __future__ import annotations

import ast
import pathlib
import sys
from types import ModuleType, SimpleNamespace

import pytest

from batchgen.worker.admission.context import (
    DecodePhaseContext,
    RankFacts,
    SchedulingState,
)
from batchgen.worker.gpu_residency import GpuResidencySet
from tests.kv_capacity_source import load_kv_capacity
from tests.worker_method_source import WORKER_CLASS, WORKER_SOURCE, load_worker_methods

_METHODS = load_worker_methods(
    [
        "_build_decode_phase_context",
        "_decode_phase_context",
        "_enter_decode_phase_context",
        "_exit_decode_phase_context",
    ],
    {
        "DecodePhaseContext": DecodePhaseContext,
        "RankFacts": RankFacts,
        "SchedulingState": SchedulingState,
    },
)


class _Worker:
    rank = 2
    world_size = 4
    local_rank = 0
    local_world_size = 2
    rank_ctx = SimpleNamespace()
    torch_device = "cpu"
    PAGE_SIZE = 64
    DECISION_INTERVAL = 128
    enable_prefix_cache = True
    enable_host_kv_eviction = True
    host_kv_chunk_size = 256
    prefix_cache_debug_stats = False
    _kv_stats_native = True
    _kv_stats_compare = False
    shared_prefix_admission = True

    def __init__(self):
        self.global_batch = object()
        self.seq_table = object()
        self._traj = object()
        self._prefix = object()
        self.gpu_paged_kv_cache_manager = object()
        self.host_paged_kv_worker_view = object()
        self.core_engine = object()
        self._sequences_with_gpu_kv = GpuResidencySet()
        self.prefix_cache_coordinator = object()
        self.prefix_cache_runtime_config = object()
        self.adaptive_chunk_sizer = None
        self.parallel_manager = SimpleNamespace(attn_tp_size=2)
        self._held_decode_phase_context = None
        self.build_count = 0
        # The lifted methods are plain functions; instance attributes make
        # `self._build_decode_phase_context()` resolve to the real compiled
        # builder (with a call counter) exactly as on the worker.
        self._build_decode_phase_context = self._counted_build
        self._decode_attn_tp_size = lambda: self.parallel_manager.attn_tp_size

    def _counted_build(self):
        self.build_count += 1
        return _METHODS["_build_decode_phase_context"](self)

    def enter(self):
        return _METHODS["_enter_decode_phase_context"](self)

    def exit(self):
        return _METHODS["_exit_decode_phase_context"](self)

    def context(self):
        return _METHODS["_decode_phase_context"](self)


def test_a_decode_phase_builds_the_context_exactly_once():
    worker = _Worker()
    worker.enter()
    held = worker._held_decode_phase_context
    assert isinstance(held, DecodePhaseContext)
    reads = [worker.context() for _ in range(5)]
    assert all(ctx is held for ctx in reads)
    assert worker.build_count == 1


def test_a_mid_phase_field_reassignment_does_not_fork_the_phase():
    """Inside the phase the held context wins; no silent rebuild per call."""
    worker = _Worker()
    original_manager = worker.gpu_paged_kv_cache_manager
    worker.enter()
    worker.gpu_paged_kv_cache_manager = object()
    ctx = worker.context()
    assert ctx.gpu_kv is original_manager
    assert worker.build_count == 1


def test_exit_drops_the_context_and_out_of_phase_reads_build_fresh():
    worker = _Worker()
    worker.enter()
    held = worker.context()
    worker.exit()
    assert worker._held_decode_phase_context is None
    # Out of a decode phase each call snapshots fresh worker state.
    fresh = worker.context()
    assert fresh is not held
    assert worker.build_count == 2
    # Exit is the drop on EVERY path out, so a second drop is a no-op.
    worker.exit()
    assert worker._held_decode_phase_context is None


def test_a_nested_decode_phase_entry_fails_by_name():
    worker = _Worker()
    worker.enter()
    with pytest.raises(RuntimeError, match="held decode"):
        worker.enter()


def test_the_gpu_kv_rebind_point_refuses_a_held_decode_context(monkeypatch):
    """The one post-setup reassignment of a context field is pinned shut."""
    bind = load_worker_methods(["_bind_gpu_paged_kv_manager"], {})[
        "_bind_gpu_paged_kv_manager"
    ]
    bound = []
    ns = load_kv_capacity()
    module = ModuleType("batchgen.worker.kv_capacity_engine")
    module.GPUKVBindContext = ns["GPUKVBindContext"]
    monkeypatch.setitem(sys.modules, module.__name__, module)

    def bind_manager(ctx, manager):
        assert isinstance(ctx, ns["GPUKVBindContext"])
        bound.append(manager)
        return ns["GPUKVBindOutcome"](manager, True)

    engine = SimpleNamespace(
        _bind_gpu_paged_kv_manager=bind_manager,
    )
    manager = object()
    worker = SimpleNamespace(
        rank=0,
        core_engine=object(),
        _kv_capacity_engine=engine,
        _held_decode_phase_context=object(),
    )
    with pytest.raises(RuntimeError, match="decode-phase context is held"):
        bind(worker, manager)
    assert bound == []

    worker._held_decode_phase_context = None
    bind(worker, manager)
    assert bound == [manager]
    assert worker.gpu_paged_kv_cache_manager is manager
    assert worker._gpu_kv_pool_created is True


_TREE = ast.parse(pathlib.Path(WORKER_SOURCE).read_text())
_CLASS = next(
    node
    for node in _TREE.body
    if isinstance(node, ast.ClassDef) and node.name == WORKER_CLASS
)


def _worker_method(name):
    return next(
        node
        for node in _CLASS.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    )


def _method_calls(root, name):
    return [
        node
        for node in ast.walk(root)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == name
    ]


def test_generate_brackets_the_decode_wave_with_enter_and_a_finally_exit():
    generate = _worker_method("generate")
    decode_loop = next(
        node
        for node in ast.walk(generate)
        if isinstance(node, ast.While)
        and "has_prefilled" in ast.unparse(node.test)
    )
    body = decode_loop.body
    enters = [
        index
        for index, stmt in enumerate(body)
        if isinstance(stmt, ast.Expr)
        and ast.unparse(stmt.value) == "self._enter_decode_phase_context()"
    ]
    assert len(enters) == 1, "exactly one decode-phase entry per wave"
    index = enters[0]

    # The entry runs after the wave's pool init: the GPU KV bind is the one
    # post-setup reassignment of a context field, so it precedes the build.
    before = ast.unparse(ast.Module(body=body[:index], type_ignores=[]))
    assert "self._init_gpu_kv_with_actual_size()" in before
    assert "self._decode_phase_context()" not in before

    # The entry and the try are the LAST two statements of the wave: every
    # statement that follows the entry is inside the try, and the finally is
    # exactly the exit bracket — so completion break, cohort-shutdown break,
    # watermark break, loop-back and exceptions all drop the held context.
    assert index == len(body) - 2
    guarded = body[index + 1]
    assert isinstance(guarded, ast.Try)
    assert guarded.handlers == [] and guarded.orelse == []
    assert [ast.unparse(stmt) for stmt in guarded.finalbody] == [
        "self._exit_decode_phase_context()"
    ]
    assert _method_calls(guarded, "decoding_continuous"), (
        "the continuous decode loop runs inside the held span"
    )
    assert _method_calls(guarded, "_prepare_decode_batch"), (
        "decode batch selection runs inside the held span"
    )


def test_the_brackets_and_the_builder_have_no_other_call_site():
    enters = _method_calls(_TREE, "_enter_decode_phase_context")
    exits = _method_calls(_TREE, "_exit_decode_phase_context")
    assert len(enters) == 1 and len(exits) == 1

    builders = _method_calls(_TREE, "_build_decode_phase_context")
    owners = set()
    for node in _CLASS.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for call in _method_calls(node, "_build_decode_phase_context"):
                owners.add(node.name)
    assert len(builders) == 2
    assert owners == {"_enter_decode_phase_context", "_decode_phase_context"}


def test_decoding_continuous_refuses_to_run_with_no_held_context():
    method = _worker_method("decoding_continuous")
    guards = [
        node
        for node in ast.walk(method)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "self._held_decode_phase_context is None"
    ]
    assert len(guards) == 1
    assert any(isinstance(stmt, ast.Raise) for stmt in guards[0].body)
