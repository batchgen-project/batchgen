"""Only a RE-activation must find the captured decode graph ladder.

The ladder is captured once per process, at the startup decode configure,
which happens AFTER the first ``activate_decoding`` has built the instance.
So the first activation legitimately has no ladder, while every activation
after a SUSPEND must still find the one captured at startup (2026-11-16: a
guard that checked both killed ranks 4-7 of a four-node deployment during
the first decode load).
"""

from __future__ import annotations

import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

PSM = (Path(__file__).resolve().parents[2] / "batchgen" / "models" / "moonshotai"
       / "kimi_linear" / "Parallel_Strategy_Manager.py")


def _isolated_activate():
    """``activate_decoding`` lifted out of the class, with its calls stubbed."""
    tree = ast.parse(PSM.read_text())
    source_class = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "KimiLinearParallelStrategyManager"
    )
    method = next(
        copy.deepcopy(node) for node in source_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "activate_decoding"
    )
    module = ast.Module(body=[method], type_ignores=[])
    namespace = {"time": SimpleNamespace(perf_counter=lambda: 0.0), "RuntimeError": RuntimeError}
    exec(compile(ast.fix_missing_locations(module), str(PSM), "exec"), namespace)
    return namespace["activate_decoding"]


def _psm(*, decode_instance, ladder_captured, persist=True):
    calls = []
    return SimpleNamespace(
        global_rank=4,
        decode_instance=decode_instance,
        decode_graphs_persist_across_suspend=persist,
        _decode_graph=SimpleNamespace(ladder_captured=ladder_captured),
        _require_built_model=lambda phase: calls.append(("require", phase)),
        _enter_decoding=lambda: {"resident_ep": 1.1},
        _log_phase_activation=lambda *a: calls.append("logged"),
        _publish_decode_instance=lambda: "published",
    ), calls


def test_the_first_activation_needs_no_ladder():
    # Before the startup capture: the instance is being built for the first
    # time, so no ladder exists yet and the activation must proceed.
    psm, calls = _psm(decode_instance=None, ladder_captured=False)
    assert _isolated_activate()(psm) == "published"
    assert "logged" in calls


def test_a_reactivation_without_the_ladder_fails_loud():
    psm, _ = _psm(decode_instance=("model", {}), ladder_captured=False)
    with pytest.raises(RuntimeError, match="without its captured graph ladder"):
        _isolated_activate()(psm)


def test_a_reactivation_with_the_ladder_proceeds():
    psm, calls = _psm(decode_instance=("model", {}), ladder_captured=True)
    assert _isolated_activate()(psm) == "published"
    assert "logged" in calls


def test_a_rebuild_lifecycle_is_not_checked():
    # A deployment whose graphs do not persist rebuilds them per configure.
    psm, calls = _psm(decode_instance=("model", {}), ladder_captured=False, persist=False)
    assert _isolated_activate()(psm) == "published"
    assert "logged" in calls
