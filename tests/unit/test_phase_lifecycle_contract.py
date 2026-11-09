"""CPU coverage for the phase-lifecycle manager contract."""

import dataclasses
import functools

import pytest

from batchgen.phase_lifecycle import (
    AdmitContext,
    ExitMode,
    PhaseEnterContext,
    PhaseKey,
    verify_phase_contract,
)


class _Conforming:
    def phase_enter(self, key, ctx):
        return None, {}

    def admit(self, key, ctx):
        return None

    def phase_exit(self, key, mode):
        return None

    def holds_instance(self, phase: str) -> bool:
        return False


def _manager_with(**overrides):
    body = {
        "phase_enter": _Conforming.phase_enter,
        "admit": _Conforming.admit,
        "phase_exit": _Conforming.phase_exit,
        "holds_instance": _Conforming.holds_instance,
    }
    for name, member in overrides.items():
        if member is None:
            body.pop(name)
        else:
            body[name] = member
    return type("Manager", (), body)()


def test_phase_key_is_a_frozen_hashable_compatibility_key():
    key = PhaseKey("prefill", "streamed", (("k3_prefill_profile", False),))
    assert [f.name for f in dataclasses.fields(PhaseKey)] == [
        "phase",
        "prefill_moe_mode",
        "profile_overrides",
    ]
    assert key == PhaseKey("prefill", "streamed", (("k3_prefill_profile", False),))
    assert key != PhaseKey("prefill", "streamed", (("k3_prefill_profile", True),))
    assert key != PhaseKey("prefill", "resident_ep", (("k3_prefill_profile", False),))
    assert hash(key) == hash(PhaseKey("prefill", "streamed", (("k3_prefill_profile", False),)))
    assert PhaseKey("decode", None).profile_overrides == ()
    with pytest.raises(dataclasses.FrozenInstanceError):
        key.phase = "decode"


def test_phase_key_rejects_unknown_phase_and_malformed_overrides():
    with pytest.raises(ValueError, match="PhaseKey.phase"):
        PhaseKey("verify", None)
    with pytest.raises(ValueError, match="profile_overrides"):
        PhaseKey("prefill", None, [("k3_prefill_profile", True)])
    with pytest.raises(ValueError, match="profile_overrides"):
        PhaseKey("prefill", None, (("k3_prefill_profile",),))


def test_exit_mode_has_exactly_suspend_and_retire():
    assert {m.name: m.value for m in ExitMode} == {
        "SUSPEND": "suspend",
        "RETIRE": "retire",
    }


def test_context_shapes():
    assert [(f.name, f.default) for f in dataclasses.fields(PhaseEnterContext)] == [
        ("comm", None),
        ("max_num_seq", None),
    ]
    admit_fields = dataclasses.fields(AdmitContext)
    assert [f.name for f in admit_fields] == ["uuids", "local_indices"]
    assert admit_fields[0].default is dataclasses.MISSING
    assert admit_fields[1].default is None
    ctx = AdmitContext(uuids=["a", "b"], local_indices=[0, 1])
    assert ctx.uuids == ["a", "b"] and ctx.local_indices == [0, 1]


def test_conforming_manager_has_no_errors():
    assert verify_phase_contract(_Conforming()) == []


def test_inherited_hooks_conform():
    class _Derived(_Conforming):
        pass

    assert verify_phase_contract(_Derived()) == []


@pytest.mark.parametrize(
    "name", ["phase_enter", "admit", "phase_exit", "holds_instance"]
)
def test_missing_hook_is_reported(name):
    errors = verify_phase_contract(_manager_with(**{name: None}))
    assert errors == [f"Manager.{name}: required phase hook is missing"]


def test_every_missing_hook_is_reported_separately():
    errors = verify_phase_contract(object())
    assert errors == [
        "object.phase_enter: required phase hook is missing",
        "object.admit: required phase hook is missing",
        "object.phase_exit: required phase hook is missing",
        "object.holds_instance: required phase hook is missing",
    ]


def test_holds_instance_requires_typed_exact_signature():
    def holds_instance(self, phase):
        return False

    errors = verify_phase_contract(_manager_with(holds_instance=holds_instance))
    assert errors == [
        "Manager.holds_instance: parameter 'phase' must be annotated str",
        "Manager.holds_instance: return must be annotated bool",
    ]


def test_var_positional_is_rejected():
    def phase_enter(self, key, ctx, *args):
        return None, {}

    errors = verify_phase_contract(_manager_with(phase_enter=phase_enter))
    assert errors == ["Manager.phase_enter: *args is not allowed"]


def test_var_keyword_is_rejected():
    def admit(self, key, ctx, **kwargs):
        return None

    errors = verify_phase_contract(_manager_with(admit=admit))
    assert errors == ["Manager.admit: **kwargs is not allowed"]


def test_catch_all_signature_reports_each_violation():
    def phase_exit(self, *args, **kwargs):
        return None

    errors = verify_phase_contract(_manager_with(phase_exit=phase_exit))
    assert errors == [
        "Manager.phase_exit: *args is not allowed",
        "Manager.phase_exit: **kwargs is not allowed",
        "Manager.phase_exit: parameters (self) do not match the contract "
        "(self, key, mode)",
    ]


def test_wrong_parameter_names_are_rejected():
    def phase_exit(self, phase, mode):
        return None

    errors = verify_phase_contract(_manager_with(phase_exit=phase_exit))
    assert errors == [
        "Manager.phase_exit: parameters (self, phase, mode) do not match the "
        "contract (self, key, mode)"
    ]


def test_extra_required_parameter_is_rejected():
    def phase_enter(self, key, ctx, comm):
        return None, {}

    errors = verify_phase_contract(_manager_with(phase_enter=phase_enter))
    assert errors == [
        "Manager.phase_enter: parameters (self, key, ctx, comm) do not match "
        "the contract (self, key, ctx)"
    ]


def test_extra_optional_parameter_is_rejected():
    def admit(self, key, ctx, extra=None):
        return None

    errors = verify_phase_contract(_manager_with(admit=admit))
    assert errors == [
        "Manager.admit: parameters (self, key, ctx, extra) do not match the "
        "contract (self, key, ctx)"
    ]


def test_keyword_only_and_defaulted_parameters_are_rejected():
    def phase_enter(self, key, *, ctx=None):
        return None, {}

    errors = verify_phase_contract(_manager_with(phase_enter=phase_enter))
    assert errors == [
        "Manager.phase_enter: parameter 'ctx' must be positional-or-keyword, "
        "got keyword-only",
        "Manager.phase_enter: parameter 'ctx' must not have a default",
    ]


def test_static_and_class_methods_are_rejected():
    errors = verify_phase_contract(
        _manager_with(
            admit=staticmethod(lambda key, ctx: None),
            phase_exit=classmethod(lambda cls, key, mode: None),
        )
    )
    assert errors == [
        "Manager.admit: must be a plain instance method, got staticmethod",
        "Manager.phase_exit: must be a plain instance method, got classmethod",
    ]


def test_catch_all_wrapper_is_judged_by_its_real_call_surface():
    def _passthrough(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    errors = verify_phase_contract(
        _manager_with(admit=_passthrough(_Conforming.admit))
    )
    assert "Manager.admit: *args is not allowed" in errors
    assert "Manager.admit: **kwargs is not allowed" in errors


def test_instance_attribute_does_not_satisfy_the_contract():
    manager = _manager_with(phase_exit=None)
    manager.phase_exit = lambda key, mode: None
    assert verify_phase_contract(manager) == [
        "Manager.phase_exit: required phase hook is missing"
    ]
