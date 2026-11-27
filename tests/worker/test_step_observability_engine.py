"""D2: the step-observability engine never receives the worker.

The engine is a small real class that owns the observability state the worker
used to carry (the Nsight decode-capture counters and the phase-switch clock);
every other input arrives as a frozen ``DecodePhaseContext`` or as an explicit
value the worker's same-name delegator resolves. These tests make that
permanent: the engine file may not name the worker or rebind ``self``, its
``self`` may touch only the engine's own declared state and methods, and each
worker delegator builds the context / resolves the declared values before the
engine call.
"""

import ast
import logging
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ENGINE = ROOT / "batchgen" / "worker" / "step_observability_engine.py"
WORKER = ROOT / "batchgen" / "batchgen_worker.py"

# The engine-owned state (formerly worker attributes; single writer is the
# engine since D2) and nothing else.
ENGINE_STATE_FIELDS = {
    "_nsys_decode_profile_forward_count",
    "_nsys_decode_profile_started",
    "_nsys_decode_profile_stopped",
    "_phase_end",
}


def _engine_class() -> ast.ClassDef:
    tree = ast.parse(ENGINE.read_text())
    return next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "StepObservabilityEngine"
    )


def _engine_methods() -> dict:
    return {
        node.name: node for node in _engine_class().body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


# --------------------------------------------------------------------------- #
#  D2 permanence: the engine reads contexts and values, never a worker
# --------------------------------------------------------------------------- #


def test_engine_file_never_names_self_or_worker():
    """D2 permanence: no worker parameter and no ``self = worker`` rebinding.

    The boundary checker's census counts worker attributes only under the
    rebinding convention, so the moment either spelling comes back the census
    rises repo-wide; this local guard fails the suite in the same commit.
    ``self`` itself stays legal — it is the engine instance — so the companion
    guard below confines what it may reach.
    """
    tree = ast.parse(ENGINE.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.arg):
            assert node.arg != "worker", ast.dump(node)
        if isinstance(node, ast.Name):
            assert node.id != "worker", node.lineno
        if isinstance(node, ast.Assign):
            for target in node.targets:
                assert not (
                    isinstance(target, ast.Name) and target.id == "self"
                ), f"self rebinding at line {node.lineno}"


def test_engine_self_touches_only_declared_engine_state():
    """Every ``self.<attr>`` in the engine file is engine state or an engine
    method: a worker attribute smuggled back through ``self`` cannot parse."""
    methods = _engine_methods()
    allowed = ENGINE_STATE_FIELDS | set(methods)
    for node in ast.walk(_engine_class()):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
        ):
            assert node.attr in allowed, (node.attr, node.lineno)


def test_the_engine_owns_exactly_the_observability_state():
    """``__init__`` declares the four fields the worker used to carry — the
    nsys capture counters and the phase clock — and nothing more."""
    init = _engine_methods()["__init__"]
    assigned = {
        node.attr
        for stmt in ast.walk(init)
        if isinstance(stmt, (ast.Assign, ast.AnnAssign))
        for node in (stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target])
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    }
    assert assigned == ENGINE_STATE_FIELDS
    # The worker no longer initializes (or owns) the migrated counters.
    worker_src = WORKER.read_text()
    for field in (
        "_nsys_decode_profile_forward_count",
        "_nsys_decode_profile_started",
        "_nsys_decode_profile_stopped",
        "_phase_end",
    ):
        assert f"self.{field}" not in worker_src, field


# --------------------------------------------------------------------------- #
#  The worker delegators build the context and resolve the declared values
# --------------------------------------------------------------------------- #


def _delegator(name: str) -> str:
    tree = ast.parse(WORKER.read_text())
    cls = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "BatchGenWorker"
    )
    method = next(
        node for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    return ast.unparse(method)


CTX_DELEGATORS = (
    "_log_batch_statistics",
    "_log_prefill_entry_hbm",
    "_log_decode_parameter_residency",
    "_log_phase_memory_ledger",
    "_warn_decode_graph_span_exceeded",
    "_nsys_prefill_profile_begin",
    "_nsys_prefill_profile_end",
    "_emit_prefix_cache_metric",
    "_prefix_host_resource_fields",
    "_maybe_emit_prefix_drain_metric",
)

VALUE_DELEGATORS = (
    "_warn_sampling_bad_rows",
    "_nsys_decode_profile_begin_forward",
    "_nsys_decode_profile_end_forward",
    "_mark_phase_end",
    "_pause_phase_switch_clock",
    "_resume_phase_switch_clock",
    "_log_phase_switch",
)


@pytest.mark.parametrize("name", CTX_DELEGATORS)
def test_ctx_delegators_pass_the_frozen_context(name):
    body = _delegator(name)
    assert "self._decode_phase_context()" in body, name


@pytest.mark.parametrize("name", CTX_DELEGATORS + VALUE_DELEGATORS)
def test_no_delegator_hands_the_engine_the_worker(name):
    """No engine call in any delegator carries bare ``self`` as an argument."""
    tree = ast.parse(WORKER.read_text())
    cls = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "BatchGenWorker"
    )
    method = next(
        node for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    for call in ast.walk(method):
        if not isinstance(call, ast.Call):
            continue
        if not (
            isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "engine"
        ):
            continue
        for arg in list(call.args) + [kw.value for kw in call.keywords]:
            assert not (
                isinstance(arg, ast.Name) and arg.id == "self"
            ), f"{name} passes the worker to the engine"


def test_delegators_resolve_the_declared_values():
    """The worker-computed inputs travel as resolved values (D1 convention:
    the context carries state objects only, never a bound worker method)."""
    ledger = _delegator("_log_phase_memory_ledger")
    assert "decode_model=getattr(self, 'model', None)" in ledger
    assert (
        "phase_mem_switch_index=getattr(self, '_phase_mem_switch_index', 0)"
        in ledger
    )
    residency = _delegator("_log_decode_parameter_residency")
    assert "decode_model=getattr(self, 'model', None)" in residency
    span = _delegator("_warn_decode_graph_span_exceeded")
    assert "graph_state=self._graph" in span
    switch = _delegator("_log_phase_switch")
    assert "rank=self.rank" in switch
    assert "persistent_phase=self._persistent_phase_enabled()" in switch
    begin = _delegator("_nsys_prefill_profile_begin")
    assert "profile_enabled=self._debug_flag_enabled" in begin
    assert "debug.get('glm5_prefill_nsys_profile')" in begin
    for name in ("_nsys_decode_profile_begin_forward", "_nsys_decode_profile_end_forward"):
        body = _delegator(name)
        assert "rank=self.rank" in body, name
        assert "device=self.torch_device" in body, name


def test_drain_latch_goes_through_the_prefix_owner_call():
    """The P1 owner call stays: the drain latch is written only through
    ``PrefixReuseState.mark_drain_metric_emitted`` on the passed state."""
    methods = _engine_methods()
    drain = ast.unparse(methods["_maybe_emit_prefix_drain_metric"])
    assert "ctx.sched.prefix.mark_drain_metric_emitted()" in drain
    assert "ctx.sched.prefix.prefix_drain_metric_emitted" in drain
    assert "ctx.sched.pool.all_completed()" in drain


# --------------------------------------------------------------------------- #
#  Behavioral: the engine-owned phase clock
# --------------------------------------------------------------------------- #


def _engine():
    from batchgen.worker.step_observability_engine import StepObservabilityEngine

    return StepObservabilityEngine()


def test_phase_clock_logs_the_switch_once(caplog):
    engine = _engine()
    engine._mark_phase_end("prefill")
    with caplog.at_level(logging.INFO):
        engine._log_phase_switch("decode", rank=3, persistent_phase=False)
    lines = [r.getMessage() for r in caplog.records if "[PHASE_SWITCH]" in r.getMessage()]
    assert len(lines) == 1
    assert "direction=prefill_to_decode" in lines[0]
    assert "rank=3" in lines[0]
    # The switch consumed the clock: a second switch without a new mark is
    # silent, exactly as the worker-attribute clock behaved.
    caplog.clear()
    with caplog.at_level(logging.INFO):
        engine._log_phase_switch("prefill", rank=3, persistent_phase=False)
    assert not [r for r in caplog.records if "[PHASE_SWITCH]" in r.getMessage()]


def test_phase_clock_same_phase_is_silent(caplog):
    engine = _engine()
    engine._mark_phase_end("decode")
    with caplog.at_level(logging.INFO):
        engine._log_phase_switch("decode", rank=0, persistent_phase=True)
    assert not [r for r in caplog.records if "[PHASE_SWITCH]" in r.getMessage()]


def test_phase_clock_pause_excludes_idle_time(caplog):
    engine = _engine()
    engine._mark_phase_end("prefill")
    engine._pause_phase_switch_clock()
    paused_busy = engine._phase_end[2]
    assert engine._phase_end[1] is None
    time.sleep(0.05)  # idle while paused: must not count
    engine._resume_phase_switch_clock()
    assert engine._phase_end[1] is not None
    with caplog.at_level(logging.INFO):
        engine._log_phase_switch("decode", rank=0, persistent_phase=False)
    (line,) = [
        r.getMessage() for r in caplog.records if "[PHASE_SWITCH]" in r.getMessage()
    ]
    seconds = float(line.split("seconds=")[1].split(" ")[0])
    assert seconds < 0.05 + paused_busy + 0.04


def test_pause_and_resume_are_safe_with_no_marked_phase():
    engine = _engine()
    engine._pause_phase_switch_clock()
    engine._resume_phase_switch_clock()
    assert engine._phase_end is None
