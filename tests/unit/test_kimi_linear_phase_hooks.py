"""CPU coverage for the Kimi-Linear phase-lifecycle hooks' delegation."""

import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from batchgen import phase_lifecycle
from batchgen.phase_lifecycle import (
    AdmitContext,
    ExitMode,
    PhaseEnterContext,
    PhaseKey,
    verify_phase_contract,
)


ROOT = Path(__file__).resolve().parents[2]
PSM = (
    ROOT
    / "batchgen"
    / "models"
    / "moonshotai"
    / "kimi_linear"
    / "Parallel_Strategy_Manager.py"
)

_EXTRACTED = {
    "phase_enter",
    "admit",
    "phase_exit",
    "holds_instance",
    "decode_experts_resident",
}
_STUBBED = {
    "configure_prefill",
    "configure_decoding",
    "activate_prefill",
    "activate_decoding",
    "release_decode_routed_experts",
}

PREFILL = PhaseKey("prefill", "streamed")
DECODE = PhaseKey("decode", None)


def _isolated_psm():
    tree = ast.parse(PSM.read_text())
    source_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "KimiLinearParallelStrategyManager"
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
                name="Psm",
                bases=[],
                keywords=[],
                body=methods,
                decorator_list=[],
            )
        ],
        type_ignores=[],
    )

    calls = []
    namespace = {
        "AdmitContext": AdmitContext,
        "ExitMode": ExitMode,
        "PhaseEnterContext": PhaseEnterContext,
        "PhaseKey": PhaseKey,
    }
    exec(compile(ast.fix_missing_locations(module), str(PSM), "exec"), namespace)
    cls = namespace["Psm"]

    results = {}
    for name in _STUBBED:
        results[name] = (object(), {"delegate": name})

        def _stub(self, *args, _name=name, **kwargs):
            calls.append((_name, args, kwargs))
            return results[_name]

        setattr(cls, name, _stub)

    psm = cls()
    psm.prefill_instance = None
    psm.decode_instance = None
    psm._resident_ep_built = False
    psm._active_phase = None
    return psm, calls, results


def test_isolated_hooks_satisfy_the_contract():
    psm, calls, _ = _isolated_psm()
    assert verify_phase_contract(psm) == []
    assert calls == []


def test_production_manager_source_imports_the_contract_types():
    tree = ast.parse(PSM.read_text())
    imported = {
        alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module == phase_lifecycle.__name__
        for alias in node.names
    }
    assert {
        "AdmitContext", "ExitMode", "PhaseEnterContext", "PhaseKey"
    } <= imported


def test_prefill_enter_builds_when_no_instance_is_kept():
    psm, calls, results = _isolated_psm()
    result = psm.phase_enter(PREFILL, PhaseEnterContext())
    assert result is results["configure_prefill"]
    assert calls == [("configure_prefill", (), {})]


@pytest.mark.parametrize(
    "phase,attr", [("prefill", "prefill_instance"), ("decode", "decode_instance")]
)
def test_holds_instance_reports_the_retained_phase_instance(phase, attr):
    psm, _, _ = _isolated_psm()
    assert psm.holds_instance(phase) is False
    setattr(psm, attr, (object(), {}))
    assert psm.holds_instance(phase) is True


def test_prefill_enter_resumes_the_kept_instance():
    psm, calls, results = _isolated_psm()
    psm.prefill_instance = (object(), {})
    result = psm.phase_enter(PREFILL, PhaseEnterContext())
    assert result is results["activate_prefill"]
    assert calls == [("activate_prefill", (), {})]


def test_decode_enter_builds_with_the_context_forwarded():
    psm, calls, results = _isolated_psm()
    comm = object()
    result = psm.phase_enter(DECODE, PhaseEnterContext(comm=comm, max_num_seq=96))
    assert result is results["configure_decoding"]
    assert calls == [("configure_decoding", (), {"padding_bsz": 96, "comm": comm})]


def test_decode_enter_resumes_the_kept_instance():
    psm, calls, results = _isolated_psm()
    psm.decode_instance = (object(), {})
    result = psm.phase_enter(
        DECODE, PhaseEnterContext(comm=object(), max_num_seq=96)
    )
    assert result is results["activate_decoding"]
    assert calls == [("activate_decoding", (), {})]


@pytest.mark.parametrize("kept", [False, True])
@pytest.mark.parametrize(
    "ctx,field",
    [
        (PhaseEnterContext(comm=None, max_num_seq=96), "comm"),
        (PhaseEnterContext(comm=object(), max_num_seq=None), "max_num_seq"),
    ],
)
def test_decode_enter_requires_both_context_fields(ctx, field, kept):
    psm, calls, _ = _isolated_psm()
    if kept:
        psm.decode_instance = (object(), {})
    with pytest.raises(ValueError, match=rf"PhaseEnterContext\.{field}$"):
        psm.phase_enter(DECODE, ctx)
    assert calls == []


@pytest.mark.parametrize("key", [PREFILL, DECODE], ids=["prefill", "decode"])
def test_admit_is_a_documented_noop_for_both_phases(key):
    psm, calls, _ = _isolated_psm()
    assert psm.admit(key, AdmitContext(uuids=["a"], local_indices=[3])) is None
    assert calls == []


def test_decode_suspend_releases_the_routed_experts():
    psm, calls, _ = _isolated_psm()
    psm._resident_ep_built = True
    psm._active_phase = "decode"
    assert psm.phase_exit(DECODE, ExitMode.SUSPEND) is None
    assert calls == [("release_decode_routed_experts", (), {})]


@pytest.mark.parametrize(
    "built,active", [(False, "decode"), (True, "prefill")], ids=["unbuilt", "inactive"]
)
def test_decode_suspend_without_a_resident_instance_fails_loud(built, active):
    psm, calls, _ = _isolated_psm()
    psm._resident_ep_built = built
    psm._active_phase = active
    with pytest.raises(RuntimeError, match="must RETIRE"):
        psm.phase_exit(DECODE, ExitMode.SUSPEND)
    assert calls == []


def test_prefill_suspend_makes_no_delegate_call():
    psm, calls, _ = _isolated_psm()
    assert psm.phase_exit(PREFILL, ExitMode.SUSPEND) is None
    assert calls == []


@pytest.mark.parametrize("key", [PREFILL, DECODE], ids=["prefill", "decode"])
def test_retire_makes_no_delegate_call(key):
    psm, calls, _ = _isolated_psm()
    psm._resident_ep_built = True
    psm._active_phase = "decode"
    assert psm.phase_exit(key, ExitMode.RETIRE) is None
    assert calls == []


def test_unsupported_mode_is_rejected():
    psm, calls, _ = _isolated_psm()
    with pytest.raises(ValueError, match="unsupported mode"):
        psm.phase_exit(DECODE, "suspend")
    assert calls == []


@pytest.mark.parametrize(
    "hook,arg",
    [
        ("phase_enter", PhaseEnterContext()),
        ("admit", AdmitContext(uuids=[])),
        ("phase_exit", ExitMode.RETIRE),
    ],
)
def test_unsupported_phase_is_rejected(hook, arg):
    psm, calls, _ = _isolated_psm()
    key = SimpleNamespace(phase="verify")
    with pytest.raises(ValueError, match="unsupported phase 'verify'"):
        getattr(psm, hook)(key, arg)
    assert calls == []
