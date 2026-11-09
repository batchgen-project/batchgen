"""CPU coverage for the GLM-5 phase-lifecycle hooks' delegation."""

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
PSM = ROOT / "batchgen" / "models" / "glm" / "glm5" / "Parallel_Strategy_Manager.py"

_EXTRACTED = {
    "phase_enter",
    "admit",
    "phase_exit",
    "holds_instance",
    "_prepare_prefill_activation",
    "decode_experts_resident",
}
_STUBBED = (
    "configure_prefill",
    "configure_decoding",
    "activate_prefill",
    "activate_decoding",
    "release_decode_routed_experts",
)

PREFILL = PhaseKey("prefill", "streamed")
DECODE = PhaseKey("decode", None)


def _isolated_psm():
    tree = ast.parse(PSM.read_text())
    source_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "GLM5ParallelStrategyManager"
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

    class _Grouped:
        @classmethod
        def reset_prefill_grouped_state(cls):
            calls.append("reset")

        @classmethod
        def init_prefill_grouped_buffers(cls, config, device):
            calls.append(("init", config, device))

    namespace = {
        "Glm5MoE": _Grouped,
        "_synchronize_prefill_preloads": lambda: calls.append("sync"),
        "AdmitContext": AdmitContext,
        "ExitMode": ExitMode,
        "PhaseEnterContext": PhaseEnterContext,
        "PhaseKey": PhaseKey,
    }
    exec(compile(ast.fix_missing_locations(module), str(PSM), "exec"), namespace)
    cls = namespace["Psm"]

    def _stub(name, result):
        def method(self, *args, **kwargs):
            calls.append((name, args, kwargs))
            return result

        return method

    results = {}
    for name in _STUBBED:
        results[name] = (object(), {"routed_expert": [name]})
        setattr(cls, name, _stub(name, results[name]))
    cls._check_late_deepgemm_compiles = lambda self, where: calls.append(
        ("check", where)
    )

    psm = cls()
    psm.rank = 0
    psm.prefill_instance = None
    psm.decode_instance = None
    psm._expert_arena = None
    psm.loaded_model_config = SimpleNamespace(phase="prefill")
    psm.engine_config = SimpleNamespace(
        Basic_Config=SimpleNamespace(device_torch="cpu")
    )
    psm.model = object()
    psm.weight_copy_task = {"attn": ["attn_0"]}
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
    assert calls[0][2]["comm"] is comm


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


def test_prefill_admit_runs_the_reactivate_body_on_the_live_instance():
    psm, calls, _ = _isolated_psm()
    live_model, live_task = psm.model, psm.weight_copy_task
    config = psm.loaded_model_config

    assert psm.admit(PREFILL, AdmitContext(uuids=["a", "b"])) is None

    assert psm.model is live_model
    assert psm.weight_copy_task is live_task
    assert config.phase == "prefill"
    assert calls == [("check", "admit"), "reset", ("init", config, "cpu"), "sync"]

def test_decode_admit_makes_no_delegate_call():
    psm, calls, _ = _isolated_psm()
    assert psm.admit(DECODE, AdmitContext(uuids=["a"], local_indices=[3])) is None
    assert calls == []


def test_decode_suspend_releases_the_routed_experts():
    psm, calls, _ = _isolated_psm()
    psm._expert_arena = SimpleNamespace(is_mapped=True)
    assert psm.phase_exit(DECODE, ExitMode.SUSPEND) is None
    assert calls == [("release_decode_routed_experts", (), {})]


@pytest.mark.parametrize(
    "arena", [None, SimpleNamespace(is_mapped=False)], ids=["no_arena", "unmapped"]
)
def test_decode_suspend_without_a_resident_instance_fails_loud(arena):
    psm, calls, _ = _isolated_psm()
    psm._expert_arena = arena
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
    psm._expert_arena = SimpleNamespace(is_mapped=True)
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
