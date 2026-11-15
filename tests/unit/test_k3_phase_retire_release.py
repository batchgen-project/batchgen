"""Kimi-K3 decode exit releases the resident EP shard whatever happened to
the model reference.

The resident MXFP4 EP shard is a manager-owned buffer. The worker's
end-of-interval deep free drops ``self.model``
before the next ``phase_exit(decode, RETIRE)``, while the captured decode
graph still keeps that model, and the shard on it, alive. The release must
therefore walk the manager's own record of resident MoE blocks, and a model
rebuild that finds the record non-empty must fail loudly instead of
allocating beside an unreachable shard. Every machine-profile row with a
positive ``resident_ep_prefill_token_cap`` declines instance reuse and takes
this RETIRE path.

The REAL methods are AST-extracted from the production manager and exercised
against stub state, matching the other phase-hook suites.
"""

import ast
import copy
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from batchgen.phase_lifecycle import ExitMode, PhaseKey


ROOT = Path(__file__).resolve().parents[2]
PSM = (
    ROOT / "batchgen" / "models" / "moonshotai" / "kimi_linear"
    / "Parallel_Strategy_Manager.py"
)

PREFILL = PhaseKey("prefill", "streamed_sp8")
DECODE = PhaseKey("decode", None)


def _isolated_class(names, namespace):
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
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {node.name for node in methods} == set(names)
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
    exec(compile(module, str(PSM), "exec"), namespace)
    # Extracted methods keep their production decorators (e.g. @property).
    return namespace["Psm"]


# --- phase_exit ------------------------------------------------------------

def _exit_psm(model):
    cls = _isolated_class(
        {"phase_exit", "holds_instance", "decode_experts_resident"},
        {
            "ExitMode": ExitMode,
            "PhaseKey": PhaseKey,
            "RuntimeError": RuntimeError,
            "ValueError": ValueError,
            "property": property,
            "bool": bool,
            "getattr": getattr,
        },
    )
    psm = cls()
    psm.model = model
    psm.weight_copy_task = {"stage": "decode"}
    psm.prefill_instance = None
    psm.decode_instance = (model, {}) if model is not None else None
    psm._resident_ep_built = True
    psm._active_phase = "decode"
    psm._release_calls = []
    psm.release_decode_routed_experts = (
        lambda: psm._release_calls.append("release")
    )
    return psm


def test_decode_retire_releases_resident_shard():
    psm = _exit_psm(object())
    psm.phase_exit(DECODE, ExitMode.RETIRE)
    assert psm._release_calls == ["release"]
    assert psm.model is None
    assert psm.decode_instance is None
    assert psm.weight_copy_task == {}


def test_decode_retire_after_worker_dropped_model_still_releases():
    # End-of-interval deep free dropped self.model before this exit; the
    # shard is still mapped and must go back here.
    psm = _exit_psm(None)
    psm.phase_exit(DECODE, ExitMode.RETIRE)
    assert psm._release_calls == ["release"]
    assert psm.model is None


def test_prefill_retire_never_touches_decode_shard():
    model = object()
    psm = _exit_psm(model)
    psm.prefill_instance = (model, {})
    psm.phase_exit(PREFILL, ExitMode.RETIRE)
    assert psm._release_calls == []
    assert psm.prefill_instance is None
    assert psm.decode_instance == (model, {})


def test_decode_suspend_still_releases_resident_shard():
    model = object()
    psm = _exit_psm(model)
    psm.phase_exit(DECODE, ExitMode.SUSPEND)
    assert psm._release_calls == ["release"]
    assert psm.decode_instance == (model, {})


def test_decode_suspend_without_resident_shard_refuses():
    psm = _exit_psm(object())
    psm._resident_ep_built = False
    with pytest.raises(RuntimeError):
        psm.phase_exit(DECODE, ExitMode.SUSPEND)


# --- release and rebuild bookkeeping ---------------------------------------

class _Shard:
    def __init__(self, nbytes):
        self._nbytes = nbytes

    def nbytes(self):
        return self._nbytes


@pytest.fixture
def owner_psm(monkeypatch):
    events = []
    fake_module = types.ModuleType("batchgen.moe.fused_moe_mxfp4_resident")
    fake_module.ResidentEPMXFP4MoELayer = SimpleNamespace(
        release_prefill_output=lambda: events.append("prefill_output")
    )
    monkeypatch.setitem(
        sys.modules, "batchgen.moe.fused_moe_mxfp4_resident", fake_module
    )
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(empty_cache=lambda: events.append("empty_cache"))
    )
    cls = _isolated_class(
        {"_release_resident_ep_decode", "_reset_model_attached_state"},
        {
            "torch": fake_torch,
            "logging": SimpleNamespace(info=lambda *a, **k: None),
            "RuntimeError": RuntimeError,
            "getattr": getattr,
            "hasattr": hasattr,
            "int": int,
            "len": len,
        },
    )
    psm = cls()
    psm.rank = 0
    psm.model = None
    psm._resident_ep_built = True
    psm._streamed_sp8_buffer = None
    psm.prefill_instance = None
    psm.decode_instance = None
    psm._active_phase = "decode"
    psm.release_decode_graph = lambda: events.append("decode_graph")
    psm._resident_ep_moes = [
        SimpleNamespace(_resident_ep_moe=SimpleNamespace(shard=_Shard(3 << 30)))
        for _ in range(2)
    ]
    return psm, events


def test_release_walks_the_manager_record_without_a_model(owner_psm):
    psm, events = owner_psm
    blocks = list(psm._resident_ep_moes)
    assert psm.model is None
    released = psm._release_resident_ep_decode()
    assert released == 2 * (3 << 30)
    assert all(block._resident_ep_moe is None for block in blocks)
    assert psm._resident_ep_moes == []
    assert psm._resident_ep_built is False
    # Graphs bake the shard's addresses: they go before the shard does.
    assert events.index("decode_graph") < events.index("empty_cache")


def test_release_is_idempotent(owner_psm):
    psm, _ = owner_psm
    psm._release_resident_ep_decode()
    assert psm._release_resident_ep_decode() == 0
    assert psm._resident_ep_moes == []


def test_rebuild_with_unreleased_shard_fails_loud(owner_psm):
    psm, _ = owner_psm
    with pytest.raises(RuntimeError, match="still attached to 2 MoE blocks"):
        psm._reset_model_attached_state()


def test_rebuild_after_release_resets_bookkeeping(owner_psm):
    psm, _ = owner_psm
    psm._release_resident_ep_decode()
    psm._reset_model_attached_state()
    assert psm._resident_ep_built is False
    assert psm.decode_instance is None
    assert psm._active_phase is None
