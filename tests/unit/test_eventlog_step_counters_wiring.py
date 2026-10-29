"""Wiring of ``StepCounters`` into ``BatchGenWorker.decoding_continuous``.

batchgen_worker imports the whole engine, so the decode loop is pinned
structurally against the module's own AST, the technique of
``test_lmhead_fp32_debug_flag.py``. The hot-path contract: per step only
counter bumps and one ``maybe_emit`` time compare; ``log_event`` is never
called in the loop; no device sync or ``.item()`` is added for the counters.
"""

from __future__ import annotations

import ast
from pathlib import Path

WORKER = Path(__file__).resolve().parents[2] / "batchgen" / "batchgen_worker.py"
TREE = ast.parse(WORKER.read_text())
COUNTERS = "_step_counters"


def _decoding_continuous() -> ast.FunctionDef:
    worker = next(
        node for node in TREE.body
        if isinstance(node, ast.ClassDef) and node.name == "BatchGenWorker"
    )
    return next(
        node for node in worker.body
        if isinstance(node, ast.FunctionDef) and node.name == "decoding_continuous"
    )


FN = _decoding_continuous()
LOOP = next(
    node for node in FN.body
    if isinstance(node, ast.While) and ast.unparse(node.test) == "decode_uuids"
)


def _call_src(stmt: ast.stmt) -> str | None:
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
        return ast.unparse(stmt.value)
    return None


def _counter_calls(nodes) -> list[str]:
    return [
        ast.unparse(node)
        for root in nodes
        for node in ast.walk(root)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == COUNTERS
    ]


def test_the_counters_import_is_module_level():
    imports = [
        node for node in TREE.body
        if isinstance(node, ast.ImportFrom)
        and node.module == "batchgen.eventlog.counters"
    ]
    assert len(imports) == 1
    assert sorted(a.name for a in imports[0].names) == [
        "STEP_COUNTERS_INTERVAL_S", "StepCounters"]


def test_one_instance_per_worker_started_once_per_round_before_the_loop():
    loop_index = FN.body.index(LOOP)
    before = FN.body[:loop_index]
    ctor = [
        node for root in before for node in ast.walk(root)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "StepCounters"
    ]
    assert len(ctor) == 1
    # Kept on the worker so a window spans decode rounds.
    stores = [
        node for root in before for node in ast.walk(root)
        if isinstance(node, ast.Attribute) and node.attr == COUNTERS
        and isinstance(node.ctx, ast.Store)
    ]
    assert len(stores) == 1
    assert _counter_calls(before) == ["_step_counters.begin_round(_hb_last_time)"]
    # Nowhere else in the worker constructs one.
    all_ctor = [
        node for node in ast.walk(TREE)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "StepCounters"
    ]
    assert len(all_ctor) == 1


def test_each_step_bumps_then_maybe_emits_at_the_top_level_of_the_loop():
    body = LOOP.body
    calls = [_call_src(stmt) for stmt in body]
    add = calls.index("_step_counters.add_step(_step_now_s, _step_rows)")
    emit = calls.index(
        "_step_counters.maybe_emit(_step_now_s, STEP_COUNTERS_INTERVAL_S)")
    assert add + 1 == emit
    # Unconditional and exactly once per step: direct children of the loop
    # body, and no other counter call anywhere in the loop.
    assert _counter_calls(body) == [calls[add], calls[emit]]
    # The step clock is read once, before the bump, and the heartbeat
    # reuses it instead of reading the clock again.
    now_assigns = [
        i for i, stmt in enumerate(body)
        if isinstance(stmt, ast.Assign)
        and ast.unparse(stmt) == "_step_now_s = time.perf_counter()"
    ]
    assert len(now_assigns) == 1 and now_assigns[0] < add
    heartbeat = next(
        stmt for stmt in body
        if isinstance(stmt, ast.If) and "_hb_last_time >= 30.0" in ast.unparse(stmt.test)
    )
    assert ast.unparse(heartbeat.test) == (
        "self.rank == 0 and _step_now_s - _hb_last_time >= 30.0")


def test_the_round_is_closed_after_the_loop():
    loop_index = FN.body.index(LOOP)
    after = FN.body[loop_index + 1:]
    assert _counter_calls(after) == [
        "_step_counters.end_round(_step_now_s)",
        "_step_counters.maybe_emit(_step_now_s, STEP_COUNTERS_INTERVAL_S)",
    ]


def test_no_eventlog_log_event_in_the_decode_loop():
    """The event is built only inside ``StepCounters.maybe_emit``.

    ``seq.log_event`` is the per-sequence lifespan trace
    (``batchgen.lifespan``), a different API that predates the event log.
    """
    names = {
        ast.unparse(node.func) for node in ast.walk(LOOP)
        if isinstance(node, ast.Call)
    }
    offenders = [
        name for name in names
        if (name == "log_event" or name.endswith(".log_event"))
        and name != "seq.log_event"
    ]
    assert offenders == [], offenders


def test_the_counter_statements_carry_no_sync_or_item():
    stmts = [
        stmt for stmt in ast.walk(FN)
        if isinstance(stmt, (ast.Expr, ast.Assign))
        and (COUNTERS in ast.unparse(stmt) or "_step_now_s" in ast.unparse(stmt)
             or "_step_rows" in ast.unparse(stmt))
    ]
    assert stmts
    for stmt in stmts:
        text = ast.unparse(stmt)
        for forbidden in (".item(", "synchronize", "tolist(", ".cpu("):
            assert forbidden not in text, text
