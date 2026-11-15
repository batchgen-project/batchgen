"""Kimi-K3 phase_exit: RETIRE of the decode instance releases the resident
EP shard.

The resident MXFP4 EP shard is a manager-owned buffer, not a tensor of the
model instance, so dropping the instance references on RETIRE never frees it
(~84 GiB/rank on world16 K3). SUSPEND already releases it; a RETIRE that
leaves it mapped OOMs the following prefill build on 2x8 H200, the only
deployment whose machine-profile row declines instance reuse (positive
``resident_ep_prefill_token_cap``).

The REAL ``phase_exit`` (and the properties it reads) is AST-extracted from
the production manager and exercised against stub state, matching the other
phase-hook suites.
"""

import ast
import copy
from pathlib import Path

import pytest

from batchgen.phase_lifecycle import ExitMode, PhaseKey


ROOT = Path(__file__).resolve().parents[2]
PSM = (
    ROOT / "batchgen" / "models" / "moonshotai" / "kimi_linear"
    / "Parallel_Strategy_Manager.py"
)

_EXTRACTED = {"phase_exit", "holds_instance", "decode_experts_resident"}

PREFILL = PhaseKey("prefill", "streamed_sp8")
DECODE = PhaseKey("decode", None)


def _isolated_psm_class():
    tree = ast.parse(PSM.read_text())
    source_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name.endswith("ParallelStrategyManager")
    )
    methods = [
        copy.deepcopy(node)
        for node in source_class.body
        if isinstance(node, ast.FunctionDef) and node.name in _EXTRACTED
    ]
    assert {node.name for node in methods} == _EXTRACTED
    module = ast.Module(
        body=[
            ast.ClassDef(
                name="Psm", bases=[], keywords=[], body=methods,
                decorator_list=[],
            )
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace = {
        "ExitMode": ExitMode,
        "PhaseKey": PhaseKey,
        "RuntimeError": RuntimeError,
        "ValueError": ValueError,
        "property": property,
        "bool": bool,
        "getattr": getattr,
    }
    exec(compile(module, str(PSM), "exec"), namespace)
    # decode_experts_resident keeps its production @property decorator.
    return namespace["Psm"]


def _psm(model):
    psm = _isolated_psm_class()()
    psm.model = model
    psm.weight_copy_task = {"stage": "decode"}
    psm.prefill_instance = None
    psm.decode_instance = (model, {}) if model is not None else None
    psm._resident_ep_built = model is not None
    psm._active_phase = "decode"
    psm._release_calls = []
    psm.release_decode_routed_experts = (
        lambda: psm._release_calls.append(psm.model)
    )
    return psm


def test_decode_retire_releases_resident_shard_while_model_alive():
    model = object()
    psm = _psm(model)
    psm.phase_exit(DECODE, ExitMode.RETIRE)
    # Released exactly once, and BEFORE the model reference was dropped:
    # the per-layer release walk needs the live model.
    assert psm._release_calls == [model]
    assert psm.model is None
    assert psm.decode_instance is None
    assert psm.weight_copy_task == {}


def test_decode_retire_without_model_skips_release():
    psm = _psm(None)
    psm.phase_exit(DECODE, ExitMode.RETIRE)
    assert psm._release_calls == []
    assert psm.model is None


def test_prefill_retire_never_touches_decode_shard():
    model = object()
    psm = _psm(model)
    psm.prefill_instance = (model, {})
    psm.phase_exit(PREFILL, ExitMode.RETIRE)
    assert psm._release_calls == []
    assert psm.prefill_instance is None
    # The decode instance (and its shard) is untouched by a prefill retire.
    assert psm.decode_instance == (model, {})


def test_decode_suspend_still_releases_resident_shard():
    model = object()
    psm = _psm(model)
    psm.phase_exit(DECODE, ExitMode.SUSPEND)
    assert psm._release_calls == [model]
    # SUSPEND keeps the instance for reuse.
    assert psm.decode_instance == (model, {})


def test_decode_suspend_without_resident_shard_refuses():
    psm = _psm(object())
    psm._resident_ep_built = False
    with pytest.raises(RuntimeError):
        psm.phase_exit(DECODE, ExitMode.SUSPEND)
