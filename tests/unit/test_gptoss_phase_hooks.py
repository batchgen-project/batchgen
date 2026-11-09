"""AST-isolated delegation tests for the GPT-OSS phase-lifecycle hooks.

The GPT-OSS manager keeps no suspended phase instance (resident experts,
every transition rebuilds), so ``phase_enter`` always builds, ``admit`` and
``phase_exit`` are documented no-ops, and the only delegates are
``configure_prefill`` / ``configure_decoding``. Imports neither torch nor
the production manager.
"""

import ast
from pathlib import Path

import pytest

from batchgen.phase_lifecycle import (
    AdmitContext,
    ExitMode,
    PhaseEnterContext,
    PhaseKey,
)

ROOT = Path(__file__).resolve().parents[2]
PSM = (
    ROOT / "batchgen" / "models" / "openai" / "gpt_oss_120b"
    / "Parallel_Strategy_Manager.py"
)

HOOKS = ("phase_enter", "admit", "phase_exit")


def _isolated_manager_class():
    tree = ast.parse(PSM.read_text())
    source_class = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "GptOssParallelStrategyManager"
    )
    methods = [
        node for node in source_class.body
        if isinstance(node, ast.FunctionDef) and node.name in HOOKS
    ]
    assert len(methods) == len(HOOKS)
    cls = ast.ClassDef(
        name="Isolated", bases=[], keywords=[], body=methods,
        decorator_list=[],
    )
    module = ast.Module(body=[cls], type_ignores=[])
    ast.fix_missing_locations(module)
    ns = {
        "PhaseKey": PhaseKey, "PhaseEnterContext": PhaseEnterContext,
        "AdmitContext": AdmitContext, "ExitMode": ExitMode,
    }
    exec(compile(module, str(PSM), "exec"), ns)
    return ns["Isolated"]


class _Recorder(_isolated_manager_class()):
    def __init__(self):
        self.calls = []
        self._prefill_result = (object(), {"p": ["t"]})
        self._decode_result = (object(), {"d": ["t"]})

    def configure_prefill(self):
        self.calls.append(("configure_prefill",))
        return self._prefill_result

    def configure_decoding(self, padding_bsz=None, comm=None):
        self.calls.append(("configure_decoding", padding_bsz, comm))
        return self._decode_result


def test_prefill_enter_always_builds():
    m = _Recorder()
    key = PhaseKey("prefill", None)
    out = m.phase_enter(key, PhaseEnterContext())
    assert out is m._prefill_result
    assert m.calls == [("configure_prefill",)]


def test_decode_enter_forwards_ctx_and_requires_fields():
    m = _Recorder()
    key = PhaseKey("decode", None)
    comm = object()
    out = m.phase_enter(key, PhaseEnterContext(comm=comm, max_num_seq=96))
    assert out is m._decode_result
    assert m.calls == [("configure_decoding", 96, comm)]
    with pytest.raises(ValueError, match="comm"):
        m.phase_enter(key, PhaseEnterContext(max_num_seq=96))
    with pytest.raises(ValueError, match="max_num_seq"):
        m.phase_enter(key, PhaseEnterContext(comm=comm))


def test_admit_and_exit_are_no_ops_with_no_delegate_calls():
    m = _Recorder()
    for phase in ("prefill", "decode"):
        key = PhaseKey(phase, None)
        m.admit(key, AdmitContext(uuids=["u"]))
        m.phase_exit(key, ExitMode.SUSPEND)
        m.phase_exit(key, ExitMode.RETIRE)
    assert m.calls == []


def test_bad_phase_and_mode_raise():
    m = _Recorder()

    class _FakeKey:
        phase = "embed"
        prefill_moe_mode = None
        profile_overrides = ()

    with pytest.raises(ValueError):
        m.phase_enter(_FakeKey(), PhaseEnterContext())
    with pytest.raises(ValueError):
        m.admit(_FakeKey(), AdmitContext(uuids=[]))
    with pytest.raises(ValueError):
        m.phase_exit(_FakeKey(), ExitMode.SUSPEND)
    with pytest.raises(ValueError):
        m.phase_exit(PhaseKey("prefill", None), "not-a-mode")
