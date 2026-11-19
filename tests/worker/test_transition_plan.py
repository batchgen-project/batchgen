"""``SequenceBatch.apply(plan)`` is the one place a sequence status changes.

Covers the plan types, ``SequenceBatch.plan_transitions`` (the skip rules a
"move these uuids to X" decision applies) and ``SequenceBatch.apply`` (ordered
application, status-index maintenance, rejections returned rather than
raised), plus a source-level guard that no other module mutates status.
"""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import pytest

from batchgen.sequence import SequenceBatch, SequenceEntry, SequenceStatus
from batchgen.worker.admission import plan as plan_module
from batchgen.worker.admission.plan import (
    RejectedTransition,
    StatusEdge,
    Transition,
    TransitionPlan,
    apply_or_raise,
)

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "batchgen"
SEQUENCE = PACKAGE / "sequence.py"


def _batch(**statuses):
    batch = SequenceBatch()
    for idx, (uuid, status) in enumerate(statuses.items()):
        seq = SequenceEntry(uuid, global_idx=idx, prompt_length=4, max_decode_length=8)
        seq.status = status
        batch.add_sequence(seq)
    return batch


def _index(batch, status):
    return set(batch.get_sequences_by_status(status))


# --------------------------------------------------------------------------- #
#  plan types                                                                  #
# --------------------------------------------------------------------------- #


def test_plan_types_are_reexported_from_the_sequence_module():
    import batchgen.sequence as sequence

    assert plan_module.Transition is sequence.Transition
    assert plan_module.TransitionPlan is sequence.TransitionPlan
    assert plan_module.RejectedTransition is sequence.RejectedTransition


def test_status_edge_members_are_exactly_the_six_decisions():
    assert {edge.value for edge in StatusEdge} == {
        "intake",
        "prefill",
        "decode_entry",
        "boundary",
        "capacity_hold",
        "completion",
    }


def test_transition_plan_is_frozen_and_empty_plan_does_not_advance():
    empty = TransitionPlan(StatusEdge.PREFILL, ())
    assert empty.advances is False
    assert empty.payload is None
    with pytest.raises(dataclasses.FrozenInstanceError):
        empty.edge = StatusEdge.COMPLETION  # type: ignore[misc]

    one = TransitionPlan(StatusEdge.PREFILL, (Transition("a", SequenceStatus.IN_PREFILL),))
    assert one.advances is True
    with pytest.raises(dataclasses.FrozenInstanceError):
        one.transitions = ()  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        one.transitions[0].to = SequenceStatus.PREFILLED  # type: ignore[misc]


def test_transition_plan_rejects_a_non_tuple_transition_sequence():
    with pytest.raises(TypeError, match="must be a tuple"):
        TransitionPlan(StatusEdge.PREFILL, [Transition("a", SequenceStatus.IN_PREFILL)])


def test_the_admission_package_does_not_import_the_worker():
    for path in (PACKAGE / "worker" / "admission").glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert node.module != "batchgen.batchgen_worker", path
            elif isinstance(node, ast.Import):
                assert all(a.name != "batchgen.batchgen_worker" for a in node.names), path


# --------------------------------------------------------------------------- #
#  plan_transitions                                                            #
# --------------------------------------------------------------------------- #


def test_plan_transitions_drops_missing_in_target_and_completed_and_keeps_order():
    batch = _batch(
        d=SequenceStatus.IN_DECODE,
        a=SequenceStatus.PREFILLED,
        c=SequenceStatus.COMPLETED,
        h=SequenceStatus.ON_HOLD,
        b=SequenceStatus.IN_DECODE,
    )
    transitions, missing = batch.plan_transitions(
        ["h", "ghost", "a", "b", "c", "d", "other"], SequenceStatus.IN_DECODE
    )
    assert transitions == (
        Transition("h", SequenceStatus.IN_DECODE),
        Transition("a", SequenceStatus.IN_DECODE),
    )
    assert missing == ("ghost", "other")
    # Planning is pure: nothing moved.
    assert batch.get_sequence("h").status == SequenceStatus.ON_HOLD
    assert batch.get_sequence("a").status == SequenceStatus.PREFILLED


def test_plan_transitions_plans_a_repeated_uuid_once():
    batch = _batch(a=SequenceStatus.QUEUEING)
    transitions, missing = batch.plan_transitions(["a", "a"], SequenceStatus.IN_PREFILL)
    assert transitions == (Transition("a", SequenceStatus.IN_PREFILL),)
    assert missing == ()


def test_plan_transitions_keeps_an_invalid_transition_for_apply_to_reject():
    batch = _batch(q=SequenceStatus.QUEUEING)
    transitions, _ = batch.plan_transitions(["q"], SequenceStatus.IN_DECODE)
    assert transitions == (Transition("q", SequenceStatus.IN_DECODE),)


# --------------------------------------------------------------------------- #
#  apply                                                                       #
# --------------------------------------------------------------------------- #


def test_apply_moves_in_order_and_maintains_the_status_index():
    batch = _batch(a=SequenceStatus.QUEUEING, b=SequenceStatus.QUEUEING)
    plan = TransitionPlan(
        StatusEdge.PREFILL,
        (
            Transition("b", SequenceStatus.IN_PREFILL),
            Transition("a", SequenceStatus.IN_PREFILL),
            Transition("b", SequenceStatus.PREFILLED),
        ),
    )
    assert batch.apply(plan) == ()
    assert batch.get_sequence("a").status == SequenceStatus.IN_PREFILL
    assert batch.get_sequence("b").status == SequenceStatus.PREFILLED
    assert _index(batch, SequenceStatus.QUEUEING) == set()
    assert _index(batch, SequenceStatus.IN_PREFILL) == {"a"}
    assert _index(batch, SequenceStatus.PREFILLED) == {"b"}


def test_apply_returns_rejections_and_still_applies_later_transitions():
    batch = _batch(q=SequenceStatus.QUEUEING, d=SequenceStatus.IN_DECODE)
    bad = Transition("q", SequenceStatus.IN_DECODE)
    good = Transition("d", SequenceStatus.ON_HOLD)
    rejected = batch.apply(TransitionPlan(StatusEdge.BOUNDARY, (bad, good)))

    assert len(rejected) == 1
    assert isinstance(rejected[0], RejectedTransition)
    assert rejected[0].transition == bad
    assert rejected[0].reason == (
        "Invalid status transition from QUEUEING to IN_DECODE for sequence q"
    )
    # The rejected sequence and its index entry are untouched.
    assert batch.get_sequence("q").status == SequenceStatus.QUEUEING
    assert _index(batch, SequenceStatus.QUEUEING) == {"q"}
    assert _index(batch, SequenceStatus.IN_DECODE) == set()
    # The transition after it was applied.
    assert batch.get_sequence("d").status == SequenceStatus.ON_HOLD
    assert _index(batch, SequenceStatus.ON_HOLD) == {"d"}


def test_apply_raises_key_error_for_an_unknown_uuid():
    batch = _batch(a=SequenceStatus.QUEUEING)
    plan = TransitionPlan(
        StatusEdge.PREFILL,
        (
            Transition("a", SequenceStatus.IN_PREFILL),
            Transition("ghost", SequenceStatus.IN_PREFILL),
        ),
    )
    with pytest.raises(KeyError, match="ghost"):
        batch.apply(plan)
    # Transitions before the unknown uuid were applied, as a sequence of
    # single-uuid updates would have.
    assert batch.get_sequence("a").status == SequenceStatus.IN_PREFILL


def test_apply_of_an_empty_plan_changes_nothing():
    batch = _batch(a=SequenceStatus.QUEUEING)
    assert batch.apply(TransitionPlan(StatusEdge.INTAKE, ())) == ()
    assert _index(batch, SequenceStatus.QUEUEING) == {"a"}


def test_apply_or_raise_applies_every_transition_then_raises_the_first_rejection():
    batch = _batch(
        q=SequenceStatus.QUEUEING,
        d=SequenceStatus.IN_DECODE,
        p=SequenceStatus.PREFILLED,
    )
    plan = TransitionPlan(
        StatusEdge.BOUNDARY,
        (
            Transition("q", SequenceStatus.EVICTED),
            Transition("d", SequenceStatus.EVICTED),
            Transition("p", SequenceStatus.EVICTED),
        ),
    )
    with pytest.raises(
        ValueError,
        match="^Invalid status transition from QUEUEING to EVICTED for sequence q$",
    ):
        apply_or_raise(batch, plan)
    assert batch.get_sequence("d").status == SequenceStatus.EVICTED


# --------------------------------------------------------------------------- #
#  source guard: one mutation point                                            #
# --------------------------------------------------------------------------- #


def _calls_to(path: Path, attr: str):
    tree = ast.parse(path.read_text(), filename=str(path))
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == attr
    ]


def test_status_is_mutated_only_inside_the_sequence_module():
    offenders = []
    for path in sorted(PACKAGE.rglob("*.py")):
        if path == SEQUENCE:
            continue
        for attr in ("update_status", "_update_status", "status_transition"):
            for line in _calls_to(path, attr):
                offenders.append(f"{path.relative_to(ROOT)}:{line} .{attr}(")
    assert offenders == []


def test_sequence_module_calls_status_transition_only_from_apply():
    tree = ast.parse(SEQUENCE.read_text())
    batch = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SequenceBatch"
    )
    callers = {
        method.name
        for method in batch.body
        if isinstance(method, ast.FunctionDef)
        and any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "status_transition"
            for node in ast.walk(method)
        )
    }
    assert callers == {"apply"}
    method_names = {m.name for m in batch.body if isinstance(m, ast.FunctionDef)}
    assert "update_status" not in method_names
    assert "_update_status" not in method_names
