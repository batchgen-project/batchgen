"""`SequenceBatch` keeps a maintained ``global_idx -> entry`` index.

Decode resolves host-KV append metadata by ``global_idx`` on every step. That
resolution used to rebuild a dict over the whole batch per step; it now reads a
third maintained index, alongside ``_status_index`` and ``_rank_index``.
``global_idx`` is assigned once in ``SequenceEntry.__init__``, so the only
events that can change the mapping are the membership changes themselves, and
these tests pin the index against the dict comprehension it replaced —
including gid reuse after a removal and the uuid-collision path.
"""
import random

from batchgen.sequence import SequenceBatch, SequenceEntry, SequenceStatus


def _entry(uuid, global_idx, prompt_length=4, max_decode_length=8):
    return SequenceEntry(
        uuid=uuid,
        global_idx=global_idx,
        prompt_length=prompt_length,
        max_decode_length=max_decode_length,
    )


def _dictcomp(batch):
    """The per-step rebuild the maintained index replaced."""
    return {seq.global_idx: seq for seq in batch}


def _assert_index_matches_rebuild(batch):
    assert batch._gid_index == _dictcomp(batch)
    for global_idx, seq in batch._gid_index.items():
        assert batch.get_sequence_by_global_idx(global_idx) is seq
        assert batch.get_sequence(seq.uuid) is seq


def test_add_sequence_indexes_the_gid():
    batch = SequenceBatch()
    first = _entry("a", 0)
    second = _entry("b", 7)
    batch.add_sequence(first)
    batch.add_sequence(second)

    assert batch.get_sequence_by_global_idx(0) is first
    assert batch.get_sequence_by_global_idx(7) is second
    assert batch.get_sequence_by_global_idx(3) is None
    _assert_index_matches_rebuild(batch)


def test_remove_sequence_drops_the_gid():
    batch = SequenceBatch()
    seq = _entry("a", 5)
    batch.add_sequence(seq)

    assert batch.remove_sequence("a") is seq
    assert batch.get_sequence_by_global_idx(5) is None
    _assert_index_matches_rebuild(batch)


def test_remove_of_an_absent_uuid_leaves_the_index_alone():
    batch = SequenceBatch()
    seq = _entry("a", 5)
    batch.add_sequence(seq)

    assert batch.remove_sequence("missing") is None
    assert batch.get_sequence_by_global_idx(5) is seq
    _assert_index_matches_rebuild(batch)


def test_a_reused_gid_resolves_to_the_new_sequence():
    # Admission reuses a gid only after its previous owner was removed; the
    # fresh sequence must own the slot.
    batch = SequenceBatch()
    old = _entry("old", 2)
    batch.add_sequence(old)
    batch.remove_sequence("old")
    new = _entry("new", 2)
    batch.add_sequence(new)

    assert batch.get_sequence_by_global_idx(2) is new
    _assert_index_matches_rebuild(batch)


def test_uuid_collision_replaces_the_gid_mapping():
    # Pool mode reuses request ids across batches. The replaced entry's gid
    # must not outlive it, and the survivor must own its own gid.
    batch = SequenceBatch()
    first = _entry("same", 1)
    batch.add_sequence(first)
    second = _entry("same", 4)
    batch.add_sequence(second)

    assert batch.get_sequence_by_global_idx(1) is None
    assert batch.get_sequence_by_global_idx(4) is second
    assert len(batch) == 1
    _assert_index_matches_rebuild(batch)


def test_uuid_collision_keeping_the_same_gid_resolves_to_the_new_entry():
    batch = SequenceBatch()
    batch.add_sequence(_entry("same", 1))
    second = _entry("same", 1)
    batch.add_sequence(second)

    assert batch.get_sequence_by_global_idx(1) is second
    _assert_index_matches_rebuild(batch)


def test_removing_a_replaced_entry_does_not_evict_the_live_gid():
    # remove_sequence must delete a gid only while it still maps to the entry
    # being removed.
    batch = SequenceBatch()
    first = _entry("same", 1)
    batch.add_sequence(first)
    second = _entry("same", 1)
    batch.add_sequence(second)

    removed = batch.remove_sequence("same")
    assert removed is second
    assert batch.get_sequence_by_global_idx(1) is None
    _assert_index_matches_rebuild(batch)


def test_status_and_rank_updates_do_not_disturb_the_gid_index():
    batch = SequenceBatch()
    seq = _entry("a", 9)
    batch.add_sequence(seq)
    batch.assign_rank("a", 3)
    batch.update_status("a", SequenceStatus.IN_PREFILL)

    assert batch.get_sequence_by_global_idx(9) is seq
    _assert_index_matches_rebuild(batch)


def test_create_view_carries_the_gid_index():
    batch = SequenceBatch()
    first = _entry("a", 0)
    second = _entry("b", 1)
    batch.add_sequence(first)
    batch.add_sequence(second)

    view = batch.create_view(["b"])

    assert view.get_sequence_by_global_idx(1) is second
    assert view.get_sequence_by_global_idx(0) is None
    _assert_index_matches_rebuild(view)


def test_index_matches_the_rebuild_under_randomized_membership_churn():
    # The index is only correct if every membership path maintains it; drive
    # add/remove/replace in a fixed-seed random order and compare against the
    # dict comprehension after every operation. gids are unique among live
    # sequences (admission takes max(live gid) + 1) and are reused only after
    # their owner was removed, so the generator reproduces exactly that.
    rng = random.Random(20261017)
    batch = SequenceBatch()
    live = {}
    retired = []
    next_gid = 0
    for _ in range(600):
        if rng.random() < 0.5 or not live:
            uuid = f"req-{rng.randrange(12)}"
            if retired and rng.random() < 0.3:
                gid = retired.pop(rng.randrange(len(retired)))
            else:
                gid = next_gid
                next_gid += 1
            replaced = live.pop(uuid, None)
            if replaced is not None:
                retired.append(replaced)
            batch.add_sequence(_entry(uuid, gid))
            live[uuid] = gid
        else:
            uuid = rng.choice(sorted(live))
            batch.remove_sequence(uuid)
            retired.append(live.pop(uuid))
        _assert_index_matches_rebuild(batch)
    assert set(batch.sequences) == set(live)
    assert len(batch) == len(live)


def test_two_live_sequences_sharing_a_gid_resolve_to_the_later_one():
    # Admission never produces this (a gid is reused only after its owner was
    # removed), but the index must still have one defined answer rather than
    # an accidental one: the later add owns the slot, and removing it clears
    # the slot instead of silently resurrecting the earlier entry.
    batch = SequenceBatch()
    first = _entry("first", 6)
    second = _entry("second", 6)
    batch.add_sequence(first)
    batch.add_sequence(second)

    assert batch.get_sequence_by_global_idx(6) is second

    batch.remove_sequence("second")
    assert batch.get_sequence_by_global_idx(6) is None
    assert batch.get_sequence("first") is first
