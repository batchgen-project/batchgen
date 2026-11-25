"""Unit tests for ``GpuResidencySet``, the GPU residency set's single owner.

The owner replaced a bare worker ``set``; these tests pin the exact
semantics the call sites relied on (idempotent add, discard-without-raise,
clear, membership / iteration / ``len`` / set equality) and that the
inherited read-only ``Set`` interface cannot mutate the storage.
"""
import pytest

from batchgen.worker.gpu_residency import GpuResidencySet


def test_mark_resident_and_membership():
    resident = GpuResidencySet()
    assert "u" not in resident
    resident.mark_resident("u")
    assert "u" in resident
    # Idempotent, exactly like set.add at the allocation call sites.
    resident.mark_resident("u")
    assert sorted(resident) == ["u"]
    assert len(resident) == 1


def test_drop_is_discard_not_remove():
    resident = GpuResidencySet()
    resident.mark_resident("u")
    resident.drop("u")
    assert "u" not in resident
    # Dropping an absent uuid must not raise (release paths drop
    # defensively after intersecting with the set).
    resident.drop("u")
    resident.drop("never-added")
    assert len(resident) == 0


def test_drop_all_empties_in_place():
    resident = GpuResidencySet()
    for uuid in ("a", "b", "c"):
        resident.mark_resident(uuid)
    resident.drop_all()
    assert len(resident) == 0
    assert resident == set()
    # Still usable after the pool-destroy clear.
    resident.mark_resident("d")
    assert set(resident) == {"d"}


def test_set_equality_both_directions():
    resident = GpuResidencySet()
    resident.mark_resident("u")
    # Tests and call sites compare against plain set literals.
    assert resident == {"u"}
    assert {"u"} == resident
    assert resident != {"u", "v"}
    assert GpuResidencySet() == set()


def test_iteration_and_comprehension_filters():
    resident = GpuResidencySet()
    for uuid in ("a", "b"):
        resident.mark_resident(uuid)
    completed = ["a", "c"]
    # The release paths' intersection idiom.
    assert [u for u in completed if u in resident] == ["a"]
    assert [u for u in completed if u not in resident] == ["c"]
    assert sorted(resident) == ["a", "b"]


def test_read_interface_cannot_mutate():
    resident = GpuResidencySet()
    resident.mark_resident("u")
    # The raw-set mutators do not exist on the owner.
    for raw in ("add", "discard", "clear", "update", "remove", "pop"):
        assert not hasattr(resident, raw)
    # Set algebra returns PLAIN sets and leaves the owner untouched.
    union = resident | {"v"}
    assert type(union) is set
    assert union == {"u", "v"}
    assert resident == {"u"}
    difference = resident - {"u"}
    assert type(difference) is set
    assert resident == {"u"}
    # The storage is slot-private: no __dict__ to poke new state into.
    with pytest.raises(AttributeError):
        resident.some_new_attribute = 1
