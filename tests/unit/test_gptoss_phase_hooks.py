"""Phase-lifecycle hook contract of the GPT-OSS strategy manager.

GPT-OSS follows the reusable-instance lifecycle: a kept decode instance is
resumed by ``activate_decoding``, decode SUSPEND releases its routed experts,
RETIRE drops it, and the prefill instance is never kept (it replicates every
routed expert). The production manager is used with its build and
activation delegates recorded, so the dispatch itself is what is tested.
"""

from types import SimpleNamespace

import pytest

from batchgen.models.openai.gpt_oss_120b.Parallel_Strategy_Manager import (
    GptOssParallelStrategyManager,
)
from batchgen.phase_lifecycle import (
    AdmitContext,
    ExitMode,
    PhaseEnterContext,
    PhaseKey,
    verify_phase_contract,
)


class _Arena:
    def __init__(self, mapped):
        self.is_mapped = mapped


class _Recorder(GptOssParallelStrategyManager):
    def __init__(self, *, kept=False, mapped=False):
        self.rank = 0
        self.calls = []
        self.comm = object()
        self.padding_bsz = 96
        self.model = object()
        self.weight_copy_task = {"attn": []}
        self._prefill_result = (object(), {"p": ["t"]})
        self._decode_result = (object(), {"d": ["t"]})
        self.decode_instance = self._decode_result if kept else None
        self._expert_arena = _Arena(mapped) if (kept or mapped) else None

    def configure_prefill(self):
        self.calls.append(("configure_prefill",))
        return self._prefill_result

    def configure_decoding(self, padding_bsz=None, comm=None):
        self.calls.append(("configure_decoding", padding_bsz, comm))
        return self._decode_result

    def activate_decoding(self):
        self.calls.append(("activate_decoding",))
        return self.decode_instance

    def release_decode_routed_experts(self):
        self.calls.append(("release",))
        self._expert_arena.is_mapped = False


def test_production_manager_satisfies_the_phase_contract():
    manager = object.__new__(GptOssParallelStrategyManager)
    assert verify_phase_contract(manager) == []


def test_prefill_enter_always_builds_and_is_never_held():
    m = _Recorder(kept=True, mapped=False)
    out = m.phase_enter(PhaseKey("prefill", None), PhaseEnterContext())
    assert out is m._prefill_result
    assert m.calls == [("configure_prefill",)]
    assert m.holds_instance("prefill") is False


@pytest.mark.parametrize("kept", [False, True])
def test_holds_instance_reports_the_kept_decode_instance(kept):
    assert _Recorder(kept=kept).holds_instance("decode") is kept


def test_decode_enter_builds_with_the_context_forwarded():
    m = _Recorder()
    comm = object()
    out = m.phase_enter(
        PhaseKey("decode", None), PhaseEnterContext(comm=comm, max_num_seq=96)
    )
    assert out is m._decode_result
    assert m.calls == [("configure_decoding", 96, comm)]


def test_decode_enter_resumes_the_kept_instance():
    m = _Recorder(kept=True)
    out = m.phase_enter(
        PhaseKey("decode", None),
        PhaseEnterContext(comm=m.comm, max_num_seq=m.padding_bsz),
    )
    assert out is m.decode_instance
    assert m.calls == [("activate_decoding",)]


@pytest.mark.parametrize("field", ["comm", "width"])
def test_decode_resume_refuses_a_context_the_instance_was_not_built_for(field):
    m = _Recorder(kept=True)
    ctx = PhaseEnterContext(
        comm=object() if field == "comm" else m.comm,
        max_num_seq=m.padding_bsz + (1 if field == "width" else 0),
    )
    with pytest.raises(RuntimeError, match="retired and rebuilt"):
        m.phase_enter(PhaseKey("decode", None), ctx)
    assert m.calls == []


@pytest.mark.parametrize("kept", [False, True])
def test_decode_enter_requires_both_context_fields(kept):
    m = _Recorder(kept=kept)
    key = PhaseKey("decode", None)
    with pytest.raises(ValueError, match="comm"):
        m.phase_enter(key, PhaseEnterContext(max_num_seq=96))
    with pytest.raises(ValueError, match="max_num_seq"):
        m.phase_enter(key, PhaseEnterContext(comm=m.comm))
    assert m.calls == []


def test_decode_suspend_releases_the_routed_experts_and_keeps_the_instance():
    m = _Recorder(kept=True, mapped=True)
    m.phase_exit(PhaseKey("decode", None), ExitMode.SUSPEND)
    assert m.calls == [("release",)]
    assert m.holds_instance("decode") is True
    assert m.decode_experts_resident is False


@pytest.mark.parametrize("kept,mapped", [(False, False), (True, False), (False, True)])
def test_decode_suspend_without_a_resident_kept_instance_fails_loud(kept, mapped):
    m = _Recorder(kept=kept, mapped=mapped)
    with pytest.raises(RuntimeError, match="must RETIRE"):
        m.phase_exit(PhaseKey("decode", None), ExitMode.SUSPEND)
    assert m.calls == []


def test_decode_retire_drops_the_instance_and_unmaps_the_arena():
    m = _Recorder(kept=True, mapped=True)
    m.phase_exit(PhaseKey("decode", None), ExitMode.RETIRE)
    assert m.calls == [("release",)]
    assert m.holds_instance("decode") is False
    assert m.model is None and m.weight_copy_task == {}


def test_retire_of_an_unmapped_or_unkept_phase_releases_nothing():
    m = _Recorder(kept=True, mapped=False)
    m.phase_exit(PhaseKey("decode", None), ExitMode.RETIRE)
    assert m.calls == [] and m.decode_instance is None
    m = _Recorder(kept=True, mapped=True)
    m.phase_exit(PhaseKey("prefill", None), ExitMode.RETIRE)
    # Retiring prefill leaves the kept decode instance and its experts alone.
    assert m.calls == [] and m.holds_instance("decode") is True
    assert m.model is None


def test_prefill_suspend_and_admit_make_no_delegate_call():
    m = _Recorder(kept=True, mapped=True)
    m.phase_exit(PhaseKey("prefill", None), ExitMode.SUSPEND)
    for phase in ("prefill", "decode"):
        m.admit(PhaseKey(phase, None), AdmitContext(uuids=["u"]))
    assert m.calls == []


def test_persistent_phase_instances_flag_is_refused():
    m = _Recorder()
    assert m.persistent_phase_instances is False
    m.persistent_phase_instances = False
    with pytest.raises(RuntimeError, match="persistent-phase-instances"):
        m.persistent_phase_instances = True


def test_bad_phase_and_mode_raise():
    m = _Recorder()
    bad_key = SimpleNamespace(
        phase="embed", prefill_moe_mode=None, profile_overrides=()
    )
    with pytest.raises(ValueError):
        m.phase_enter(bad_key, PhaseEnterContext())
    with pytest.raises(ValueError):
        m.admit(bad_key, AdmitContext(uuids=[]))
    with pytest.raises(ValueError):
        m.phase_exit(bad_key, ExitMode.SUSPEND)
    with pytest.raises(ValueError):
        m.holds_instance("embed")
    with pytest.raises(ValueError):
        m.phase_exit(PhaseKey("prefill", None), "not-a-mode")
