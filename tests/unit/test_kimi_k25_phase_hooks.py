"""AST-isolated delegation tests for the Kimi-K2.5 phase hooks."""

import ast
from pathlib import Path

import pytest

from batchgen.phase_lifecycle import (
    AdmitContext,
    ExitMode,
    PhaseEnterContext,
    PhaseKey,
    verify_phase_contract,
)


ROOT = Path(__file__).resolve().parents[2]
PSM = ROOT / "batchgen" / "models" / "moonshotai" / "kimi_k25" / "Parallel_Strategy_Manager.py"


def _isolated_manager_class():
    tree = ast.parse(PSM.read_text())
    source_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "KimiK25ParallelStrategyManager"
    )
    hooks = {"phase_enter", "admit", "phase_exit", "holds_instance"}
    methods = [
        node
        for node in source_class.body
        if isinstance(node, ast.FunctionDef) and node.name in hooks
    ]
    assert {node.name for node in methods} == hooks
    isolated = ast.Module(
        body=[
            ast.ClassDef(
                name="Isolated",
                bases=[],
                keywords=[],
                body=methods,
                decorator_list=[],
            )
        ],
        type_ignores=[],
    )
    namespace = {
        "AdmitContext": AdmitContext,
        "ExitMode": ExitMode,
        "PhaseEnterContext": PhaseEnterContext,
        "PhaseKey": PhaseKey,
    }
    exec(compile(ast.fix_missing_locations(isolated), str(PSM), "exec"), namespace)
    return namespace["Isolated"]


class _Recorder(_isolated_manager_class()):
    def __init__(self):
        self.calls = []
        self.prefill_result = (object(), {"prefill": True})
        self.decode_result = (object(), {"decode": True})

    def configure_prefill(self):
        self.calls.append(("configure_prefill",))
        return self.prefill_result

    def configure_decoding(self, padding_bsz=None, comm=None):
        self.calls.append(("configure_decoding", padding_bsz, comm))
        return self.decode_result


PREFILL = PhaseKey("prefill", None)
DECODE = PhaseKey("decode", None)


def test_isolated_hooks_satisfy_contract():
    manager = _Recorder()
    assert verify_phase_contract(manager) == []
    assert manager.calls == []


def test_holds_instance_is_false_for_both_phases():
    manager = _Recorder()
    assert manager.holds_instance("prefill") is False
    assert manager.holds_instance("decode") is False


def test_prefill_enter_delegates_and_preserves_identity():
    manager = _Recorder()
    result = manager.phase_enter(PREFILL, PhaseEnterContext())
    assert result is manager.prefill_result
    assert manager.calls == [("configure_prefill",)]


def test_decode_enter_forwards_context_and_preserves_identity():
    manager = _Recorder()
    comm = object()
    result = manager.phase_enter(
        DECODE, PhaseEnterContext(comm=comm, max_num_seq=96)
    )
    assert result is manager.decode_result
    assert manager.calls == [("configure_decoding", 96, comm)]


@pytest.mark.parametrize(
    "ctx,field",
    [
        (PhaseEnterContext(max_num_seq=96), "comm"),
        (PhaseEnterContext(comm=object()), "max_num_seq"),
    ],
)
def test_decode_enter_rejects_missing_context_fields(ctx, field):
    manager = _Recorder()
    with pytest.raises(ValueError, match=rf"PhaseEnterContext.{field}$"):
        manager.phase_enter(DECODE, ctx)
    assert manager.calls == []


def test_admit_and_exit_are_documented_no_ops():
    manager = _Recorder()
    for key in (PREFILL, DECODE):
        manager.admit(key, AdmitContext(uuids=["u"], local_indices=[0]))
        manager.phase_exit(key, ExitMode.SUSPEND)
        manager.phase_exit(key, ExitMode.RETIRE)
    assert manager.calls == []


def test_bad_phase_and_mode_raise():
    manager = _Recorder()
    bad_key = type("BadKey", (), {"phase": "verify"})()
    with pytest.raises(ValueError, match="unsupported phase 'verify'"):
        manager.phase_enter(bad_key, PhaseEnterContext())
    with pytest.raises(ValueError, match="unsupported phase 'verify'"):
        manager.admit(bad_key, AdmitContext(uuids=[]))
    with pytest.raises(ValueError, match="unsupported phase 'verify'"):
        manager.phase_exit(bad_key, ExitMode.SUSPEND)
    with pytest.raises(ValueError, match="unsupported mode"):
        manager.phase_exit(PREFILL, "suspend")
    assert manager.calls == []
