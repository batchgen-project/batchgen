"""CPU coverage for the consecutive-prefill fast-path condition."""

from types import SimpleNamespace

from batchgen.worker.prefill_fastpath import is_consecutive_prefill


class _Manager:
    def __init__(self, applicable=True):
        self.applicable = applicable

    def reactivate_prefill_in_place(self):
        return None

    def reactivate_prefill_in_place_applicable(self, *, prefill_moe_mode):
        return self.applicable


def _condition_kwargs(manager=None):
    return dict(
        persistent=False,
        phase="prefill",
        has_model=True,
        reuse_startup_prefill=False,
        manager=manager or _Manager(),
        prefill_moe_mode="streamed",
    )


def test_consecutive_prefill_condition_is_true_when_all_gates_hold():
    assert is_consecutive_prefill(**_condition_kwargs()) is True


def test_consecutive_prefill_condition_rejects_each_false_gate():
    cases = (
        {"phase": "decode"},
        {"persistent": True},
        {"reuse_startup_prefill": True},
        {"has_model": False},
        {"manager": _Manager(applicable=False)},
    )
    for override in cases:
        kwargs = _condition_kwargs()
        kwargs.update(override)
        assert is_consecutive_prefill(**kwargs) is False


def test_consecutive_prefill_condition_rejects_missing_manager_capabilities():
    with_reactivate_only = SimpleNamespace(reactivate_prefill_in_place=lambda: None)
    with_applicable_only = SimpleNamespace(
        reactivate_prefill_in_place_applicable=lambda **kwargs: True
    )
    for manager in (with_reactivate_only, with_applicable_only, SimpleNamespace()):
        assert is_consecutive_prefill(
            **_condition_kwargs(manager=manager)
        ) is False
