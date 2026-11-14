# ---------------------------------------------------------------------------- #
#  BatchGen                                                                     #
#  Copyright (c) 2025-2026 BatchGen Team                                        #
#  Licensed under the Apache License, Version 2.0                               #
# ---------------------------------------------------------------------------- #
"""Decode-exit disposal of an in-flight boundary GPU-KV load (CPU, no GPU).

A boundary that launches an async GPU-KV load allocates GPU pages (with
registered shared-page keys) for rows that are NOT yet in ``decode_uuids``;
only the finalize step moves them in. A decode-loop exit that drops its
pending-load state therefore strands those pages: no later path frees pages
for a row outside ``decode_uuids``, and the row's next load trips
``_assert_private_writable_page_keys`` ("only a committed prefix page may be
shared"), which the collective turns into a cohort-fatal SCHED_INVARIANT.

Two checks, each with a non-vacuity control:

  * ``_integrate_pending_async_load`` runs the canonical disposal protocol in
    order (task.wait -> cuda.synchronize -> dist.barrier -> finalize) and
    returns the finalize result; a None task skips the wait but never the
    barrier or the finalize.
  * In ``decoding_continuous``, every exit of the watermark branch integrates
    the pending load BEFORE ``_put_sequences_on_hold``, so the hold frees the
    loaded rows' pages like any other resident row's.

Run: python -m pytest tests/test_decode_exit_pending_load_integration.py -x -q
"""

from __future__ import annotations

import ast
import copy
import types
from pathlib import Path
from typing import List, Optional, Tuple

from tests.worker_method_source import delegated_method_names, worker_shaped_method_node

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "batchgen" / "batchgen_worker.py"
ENGINE = ROOT / "batchgen" / "worker" / "decode_engine.py"


def _isolated_worker_method(function_name, globals_=None):
    """Compile one worker method without importing the GPU/JIT-heavy module."""
    tree = ast.parse(WORKER.read_text())
    worker = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "BatchGenWorker"
    )
    # S10c: a delegated method compiles from its engine body (worker shape).
    method = worker_shaped_method_node(function_name) if function_name in delegated_method_names() else copy.deepcopy(next(
        node
        for node in worker.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    ))
    module = ast.Module(
        body=[ast.ClassDef(
            name="Isolated",
            bases=[],
            keywords=[],
            body=[method],
            decorator_list=[],
        )],
        type_ignores=[],
    )
    namespace = {
        "List": List,
        "Optional": Optional,
        "Tuple": Tuple,
        **(globals_ or {}),
    }
    exec(compile(ast.fix_missing_locations(module), str(WORKER), "exec"), namespace)
    return getattr(namespace["Isolated"], function_name)


def _make_worker_and_calls():
    calls: List[str] = []

    class _Task:
        def wait(self):
            calls.append("task.wait")

    def _finalize(task, uuids, local, global_ids, decode_uuids, batch, mgr):
        calls.append("finalize")
        assert uuids == ["u1"]
        assert local == [3]
        assert global_ids == [17]
        return decode_uuids + uuids, batch + local

    worker = types.SimpleNamespace(
        torch_device="cpu",
        _finalize_async_load_minimal=_finalize,
    )
    globals_ = {
        "torch": types.SimpleNamespace(
            cuda=types.SimpleNamespace(
                synchronize=lambda device: calls.append("cuda.synchronize")
            )
        ),
        "dist": types.SimpleNamespace(
            barrier=lambda: calls.append("dist.barrier")
        ),
    }
    return worker, calls, _Task, globals_


def test_integrate_pending_async_load_runs_protocol_in_order():
    worker, calls, _Task, globals_ = _make_worker_and_calls()
    method = _isolated_worker_method("_integrate_pending_async_load", globals_)

    decode_uuids, batch = method(
        worker, _Task(), ["u1"], [3], [17], ["u0"], [1], object()
    )

    assert calls == ["task.wait", "cuda.synchronize", "dist.barrier", "finalize"]
    assert decode_uuids == ["u0", "u1"]
    assert batch == [1, 3]


def test_integrate_pending_async_load_none_task_still_barriers_and_finalizes():
    worker, calls, _Task, globals_ = _make_worker_and_calls()
    method = _isolated_worker_method("_integrate_pending_async_load", globals_)

    decode_uuids, batch = method(
        worker, None, ["u1"], [3], [17], [], [], object()
    )

    assert calls == ["dist.barrier", "finalize"]
    assert decode_uuids == ["u1"]
    assert batch == [3]


def _watermark_branch_bodies():
    """Every ``if watermark_triggered:`` body inside decoding_continuous."""
    # S4b: body moved to decode_engine.DecoupledGpuDecode.run; same statements, same order.
    tree = ast.parse(ENGINE.read_text())
    worker = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "DecoupledGpuDecode"
    )
    loop = next(
        node
        for node in worker.body
        if isinstance(node, ast.FunctionDef) and node.name == "run"
    )
    bodies = []
    for node in ast.walk(loop):
        if (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "watermark_triggered"
        ):
            bodies.append(node.body)
    return bodies


def _method_call_order(body):
    order = []
    for node in body:
        for sub in ast.walk(node):
            if (
                isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Attribute)
                and isinstance(sub.func.value, ast.Name)
                and sub.func.value.id == "self"
            ):
                order.append(sub.func.attr)
    return order


def test_watermark_exit_integrates_pending_load_before_hold():
    bodies = _watermark_branch_bodies()
    # Non-vacuity: the watermark hold branch must exist.
    hold_bodies = [
        body
        for body in bodies
        if "_put_sequences_on_hold" in _method_call_order(body)
    ]
    assert hold_bodies, "watermark ON_HOLD branch not found in decoding_continuous"
    for body in hold_bodies:
        order = _method_call_order(body)
        assert "_integrate_pending_async_load" in order, (
            "watermark exit must dispose the in-flight async load; dropping it "
            "strands GPU pages whose shared-page keys later fail the "
            "writable-page guard"
        )
        assert order.index("_integrate_pending_async_load") < order.index(
            "_put_sequences_on_hold"
        ), "pending rows must join decode_uuids before the hold frees pages"
