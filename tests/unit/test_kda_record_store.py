"""Contract tests for the per-sequence KDA record store (方案甲).

Pure CPU. Only the two injected boundaries are faked -- the Host-page arena
(backed by real bytes, so round-trips are byte-accurate) and the GPU slot
reader/writer -- so the unit split/merge math, the page arithmetic
(``FrontierCharge``), the lifecycle ordering and every named refusal under
test are the real objects.

What these tests defend (the approved seam's four anti-properties):
* NO alignment requirement: a record written at an arbitrary ("unaligned")
  true token count is stored and readable -- where the frontier committer's
  ``commit_tokens`` would return 0 and skip;
* NO dedup: identity is the global sequence id, one record per sequence;
* NO gate: the store exists without any prefix-cache flag (nothing here
  touches one);
* NO capacity skip: the pages are reserved at admission; the only capacity
  outcome is the NAMED admission refusal, and a later write/restore failure
  is a hard error.
"""

import pytest
import torch

from batchgen.prefix_reuse.kda_record_map import (
    KDARecordMapFull,
    KDARecordMapGeometry,
    KDARecordMapLayoutMismatch,
    KDARecordSharedMap,
)
from batchgen.prefix_reuse.kda_record_store import (
    KDARecordMissing,
    KDARecordReservationUnavailable,
    KDARecordStateInvalid,
    KDARecordStore,
    record_page_count,
)
from batchgen.prefix_reuse.trajectory_commit import (
    SnapshotArenaUnavailable,
    SnapshotLayout,
    split_chunks_into_units,
)
from batchgen.prefix_reuse.trajectory_frontier import (
    FrontierCharge,
    KDASnapshotProfile,
)
from batchgen.worker import kda_lifecycle

HOST_PAGE_BYTES = 128
UNIT_COUNT = 4


def _full_profile() -> KDASnapshotProfile:
    """The WHOLE model's state geometry (the G==1 shard)."""

    return KDASnapshotProfile(
        num_kda_layers=2,
        num_heads=8,
        head_dim=4,
        conv_dim=32,
        conv_width=3,
        recurrent_element_bytes=4,
        conv_element_bytes=2,
    )


def _shard_profile(group_size: int) -> KDASnapshotProfile:
    from dataclasses import replace

    full = _full_profile()
    return replace(
        full,
        num_heads=full.num_heads // group_size,
        conv_dim=full.conv_dim // group_size,
    )


def _full_chunks(seed: int = 0) -> tuple:
    torch.manual_seed(seed)
    sizes = SnapshotLayout(profile=_full_profile()).chunk_bytes
    return tuple(
        torch.randint(0, 256, (size,), dtype=torch.uint8) for size in sizes
    )


def _shard_chunks(full_chunks, group_size: int, group_rank: int) -> tuple:
    """Rank ``group_rank``'s head shard of the full state, as flat uint8."""

    shards = split_chunks_into_units(
        full_chunks, profile=_full_profile(), unit_count=group_size
    )
    return shards[group_rank]


def _chunk_bytes(chunks) -> bytes:
    return b"".join(
        bytes(chunk.reshape(-1).view(torch.uint8).numpy().tobytes())
        for chunk in chunks
    )


class _PageArena:
    """Byte-backed Host-page arena shared by every rank's store.

    Implements the allocator and reader boundaries with the REAL byte
    semantics: a unit window is a whole number of pages, written from offset
    0 in chunk order, read back the same way (one KV group, like the Kimi
    deployment).
    """

    def __init__(self, *, free_pages: int = 10_000):
        self.pages: dict[int, bytearray] = {}
        self.free_pages = int(free_pages)
        self._next_page = 100
        self._next_sid = -1
        self.retained: list[int] = []
        self.released: list[int] = []
        self.rolled_back: list[int] = []

    # -- allocator boundary -------------------------------------------------

    def allocate(self, *, page_count: int, unit_count: int = 1):
        from batchgen.prefix_reuse.trajectory_commit import SnapshotArena

        if int(page_count) > self.free_pages:
            raise SnapshotArenaUnavailable(
                f"Host KV pool could not supply {page_count} page(s): "
                f"{self.free_pages} free"
            )
        self.free_pages -= int(page_count)
        page_ids = []
        for _ in range(int(page_count)):
            self.pages[self._next_page] = bytearray(HOST_PAGE_BYTES)
            page_ids.append(self._next_page)
            self._next_page += 1
        sid = self._next_sid
        self._next_sid -= 1
        return SnapshotArena(
            storage_handle=0,
            pseudo_sid=sid,
            page_ids_by_group={0: tuple(page_ids)},
            pages_per_group=int(page_count),
            capacity_bytes=int(page_count) * HOST_PAGE_BYTES,
            payload_bytes=0,
            unit_count=int(unit_count),
        )

    def write_unit(self, arena, *, unit_index: int, chunks) -> int:
        window = arena.unit_window(int(unit_index))[0]
        data = _chunk_bytes(chunks)
        capacity = arena.pages_per_unit * HOST_PAGE_BYTES
        assert len(data) <= capacity, "unit payload exceeds its window"
        for index, page in enumerate(window):
            segment = data[index * HOST_PAGE_BYTES:(index + 1) * HOST_PAGE_BYTES]
            if not segment:
                break
            self.pages[page][: len(segment)] = segment
        return len(data)

    def retain(self, arena):
        self.retained.append(int(arena.pseudo_sid))
        return arena

    def release(self, arena):
        self.released.append(int(arena.pseudo_sid))
        self.free_pages += int(arena.pages_per_group)
        return {0: int(arena.pages_per_group)}

    def rollback(self, arena):
        self.rolled_back.append(int(arena.pseudo_sid))
        self.free_pages += int(arena.pages_per_group)

    # -- reader boundary ----------------------------------------------------

    def read(self, arena, *, layout, unit_index: int = 0):
        window = arena.unit_window(int(unit_index))[0]
        blob = b"".join(bytes(self.pages[page]) for page in window)
        return tuple(
            torch.frombuffer(
                bytearray(blob[offset: offset + size]), dtype=torch.uint8
            )
            for offset, size in zip(layout.chunk_offsets, layout.chunk_bytes)
        )


class _SnapshotReader:
    """The D2H boundary: hands back this rank's live shard chunks."""

    def __init__(self, chunks):
        self._chunks = chunks
        self.reads: list[int] = []

    def read_slot(self, sequence_id: int):
        self.reads.append(int(sequence_id))
        return self._chunks


class _StateWriter:
    """The H2D / slot boundary: captures what a restore binds."""

    def __init__(self):
        self.slots: dict[int, int] = {}
        self.written: dict[int, bytes] = {}
        self._next_slot = 0
        self.released: list[int] = []

    def has_slot(self, sequence_id: int) -> bool:
        return int(sequence_id) in self.slots

    def allocate_slot(self, sequence_id: int) -> int:
        slot = self.slots.setdefault(int(sequence_id), self._next_slot)
        if slot == self._next_slot:
            self._next_slot += 1
        return slot

    def release_slot(self, sequence_id: int) -> None:
        self.released.append(int(sequence_id))
        self.slots.pop(int(sequence_id), None)

    def write_slot(self, sequence_id: int, chunks):
        data = _chunk_bytes(chunks)
        self.written[int(sequence_id)] = data
        return self.slots[int(sequence_id)], len(data)


def _record_pages() -> int:
    """The model constant every map geometry and store must agree on."""

    return record_page_count(
        rank_profile=_full_profile(),
        group_size=1,
        unit_count=UNIT_COUNT,
        host_page_bytes_all_layers=HOST_PAGE_BYTES,
    )


def _map_geometry(capacity: int = 32) -> KDARecordMapGeometry:
    return KDARecordMapGeometry(
        capacity=capacity,
        group_ids=(0,),
        pages_per_group=_record_pages(),
        unit_count=UNIT_COUNT,
        host_page_bytes_all_layers=HOST_PAGE_BYTES,
    )


def _region(capacity: int = 32) -> bytearray:
    """A fresh 'shared region' for the durable map (what outlives stores)."""

    return bytearray(KDARecordSharedMap.required_bytes(_map_geometry(capacity)))


def _shared_map(region=None, *, capacity: int = 32) -> KDARecordSharedMap:
    region = _region(capacity) if region is None else region
    return KDARecordSharedMap(region, geometry=_map_geometry(capacity))


def _store(
    *,
    arena: _PageArena,
    group_size: int,
    group_rank: int,
    full_chunks=None,
    state_writer=None,
    shared_map=None,
    rank=None,
) -> KDARecordStore:
    chunks = (
        _shard_chunks(full_chunks, group_size, group_rank)
        if full_chunks is not None
        else ()
    )
    return KDARecordStore(
        rank=group_rank if rank is None else rank,
        group_size=group_size,
        group_rank=group_rank,
        unit_count=UNIT_COUNT,
        rank_profile=_shard_profile(group_size),
        host_page_bytes_all_layers=HOST_PAGE_BYTES,
        arena_allocator=arena,
        arena_reader=arena,
        snapshot_reader=_SnapshotReader(chunks),
        state_writer=state_writer or _StateWriter(),
        shared_map=shared_map if shared_map is not None else _shared_map(),
    )


# ---------------------------------------------------------------------------
# Page arithmetic
# ---------------------------------------------------------------------------


def test_record_pages_is_the_frontier_charge_constant():
    arena = _PageArena()
    store = _store(arena=arena, group_size=2, group_rank=0)
    unit_bytes = SnapshotLayout(
        profile=store.unit_layout.profile
    ).total_bytes
    expected = UNIT_COUNT * (-(-unit_bytes // HOST_PAGE_BYTES))
    assert store.record_pages == expected
    assert (
        store.record_pages
        == FrontierCharge(
            page_count=0,
            host_page_bytes_all_layers=HOST_PAGE_BYTES,
            snapshot_bytes=unit_bytes,
            unit_count=UNIT_COUNT,
        ).arena_host_pages
    )


def test_reserve_takes_exactly_record_pages_and_rollback_returns_them():
    arena = _PageArena(free_pages=1000)
    store = _store(arena=arena, group_size=1, group_rank=0)
    before = arena.free_pages
    store.reserve(7)
    assert arena.free_pages == before - store.record_pages
    assert store.has_record(7) and not store.is_written(7)
    rolled = store.rollback_reserved([7, 8])
    assert rolled == (7,)
    assert arena.free_pages == before
    assert not store.has_record(7)


def test_admission_reservation_is_refused_by_name_when_pool_cannot_hold_it():
    arena = _PageArena(free_pages=0)
    store = _store(arena=arena, group_size=1, group_rank=0)
    with pytest.raises(KDARecordReservationUnavailable) as excinfo:
        store.reserve(42)
    assert "sequence 42" in str(excinfo.value)
    assert str(store.record_pages) in str(excinfo.value)
    assert not store.has_record(42)


def test_double_reservation_is_refused_by_name():
    arena = _PageArena()
    store = _store(arena=arena, group_size=1, group_rank=0)
    store.reserve(7)
    with pytest.raises(KDARecordStateInvalid):
        store.reserve(7)


# ---------------------------------------------------------------------------
# Prefill end: write at an unaligned token count, slot freed
# ---------------------------------------------------------------------------


UNALIGNED_TOKENS = 997  # deliberately not a multiple of anything page-like


def test_prefill_end_write_frees_the_slot_and_record_is_readable():
    """The lifecycle step: write at a TRUE (unaligned) token count, then the
    GPU slot is freed, and the record restores byte-identically."""

    full = _full_chunks(seed=1)
    arena = _PageArena()
    store = _store(
        arena=arena, group_size=1, group_rank=0, full_chunks=full
    )
    store.reserve(11)

    freed: list[list[int]] = []
    written = kda_lifecycle.offload_prefill_states(
        store,
        lambda ids: freed.append(list(ids)),
        [(11, UNALIGNED_TOKENS)],
        rank=0,
    )
    assert written == (11,)
    assert freed == [[11]]  # the GPU slot release ran, after the write
    assert store.is_written(11)
    assert store.record_for(11).token_count == UNALIGNED_TOKENS

    store.retain_written()
    writer = store._state_writer
    slot, nbytes = store.restore_rank_units(11, group_size=1, group_rank=0)
    assert writer.written[11] == _chunk_bytes(full)
    assert nbytes == len(_chunk_bytes(full))


def test_write_without_reservation_raises_by_name():
    full = _full_chunks()
    arena = _PageArena()
    store = _store(arena=arena, group_size=1, group_rank=0, full_chunks=full)
    with pytest.raises(KDARecordMissing):
        store.write_rank_units(5, token_count=UNALIGNED_TOKENS)


def test_second_write_for_the_same_record_raises_by_name():
    full = _full_chunks()
    arena = _PageArena()
    store = _store(arena=arena, group_size=1, group_rank=0, full_chunks=full)
    store.reserve(5)
    store.write_rank_units(5, token_count=UNALIGNED_TOKENS)
    with pytest.raises(KDARecordStateInvalid):
        store.write_rank_units(5, token_count=UNALIGNED_TOKENS + 1)


# ---------------------------------------------------------------------------
# Cross-group round trip: write under G=2, restore under G=4 and G=1
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("read_group_size", [1, 2, 4])
def test_restore_into_a_different_group_size_is_byte_identical(
    read_group_size,
):
    write_group = 2
    full = _full_chunks(seed=2)
    arena = _PageArena()

    # The owner (write group rank 0) reserves; rank 1 adopts the mirror.
    writers = []
    owner = _store(
        arena=arena,
        group_size=write_group,
        group_rank=0,
        full_chunks=full,
    )
    owner.reserve(21)
    writers.append(owner)
    follower = _store(
        arena=arena,
        group_size=write_group,
        group_rank=1,
        full_chunks=full,
    )
    follower.adopt(owner.mirror_payload([21]))
    writers.append(follower)

    # Every write rank writes its own unit span at the real prefill end.
    for writer_store in writers:
        writer_store.write_rank_units(21, token_count=UNALIGNED_TOKENS)
    owner.retain_written()

    # Every rank of the READ layout restores its own shard, byte-identical
    # to the directly sliced head shard of the full state.
    for read_rank in range(read_group_size):
        state_writer = _StateWriter()
        reader_store = _store(
            arena=arena,
            group_size=read_group_size,
            group_rank=read_rank,
            state_writer=state_writer,
        )
        reader_store.adopt(owner.mirror_payload([21]))
        reader_store.restore_rank_units(
            21, group_size=read_group_size, group_rank=read_rank
        )
        expected = _chunk_bytes(
            _shard_chunks(full, read_group_size, read_rank)
        )
        assert state_writer.written[21] == expected, (
            f"read rank {read_rank}/{read_group_size} restored different "
            "bytes than its head shard of the written state"
        )


def test_restore_refuses_a_group_size_that_does_not_divide_the_units():
    from batchgen.prefix_reuse.trajectory_commit import SnapshotArenaCorrupt

    full = _full_chunks()
    arena = _PageArena()
    store = _store(arena=arena, group_size=1, group_rank=0, full_chunks=full)
    store.reserve(3)
    store.write_rank_units(3, token_count=UNALIGNED_TOKENS)
    with pytest.raises(SnapshotArenaCorrupt):
        store.restore_rank_units(3, group_size=3, group_rank=0)


# ---------------------------------------------------------------------------
# Restore refusals
# ---------------------------------------------------------------------------


def test_restore_without_a_record_is_state_loss_and_raises_by_name():
    arena = _PageArena()
    store = _store(arena=arena, group_size=1, group_rank=0)
    with pytest.raises(KDARecordMissing):
        store.restore_rank_units(99, group_size=1, group_rank=0)


def test_restore_of_an_unwritten_record_raises_by_name():
    arena = _PageArena()
    store = _store(arena=arena, group_size=1, group_rank=0)
    store.reserve(4)
    with pytest.raises(KDARecordStateInvalid):
        store.restore_rank_units(4, group_size=1, group_rank=0)


def test_restore_over_a_live_slot_raises_and_lifecycle_skips_it():
    full = _full_chunks()
    arena = _PageArena()
    state_writer = _StateWriter()
    store = _store(
        arena=arena,
        group_size=1,
        group_rank=0,
        full_chunks=full,
        state_writer=state_writer,
    )
    store.reserve(6)
    store.write_rank_units(6, token_count=UNALIGNED_TOKENS)
    store.retain_written()
    state_writer.allocate_slot(6)  # an ON_HOLD re-entry's live slot
    with pytest.raises(KDARecordStateInvalid):
        store.restore_rank_units(6, group_size=1, group_rank=0)
    # The lifecycle helper leaves live-slot rows alone instead of raising.
    restored = kda_lifecycle.restore_decode_states(
        store, [6], group_size=1, group_rank=0, rank=0
    )
    assert restored == ()
    assert state_writer.written == {}


# ---------------------------------------------------------------------------
# Release: eviction and completion, owner and followers
# ---------------------------------------------------------------------------


def _written_store(sequence_id: int = 31, *, arena=None):
    full = _full_chunks(seed=3)
    arena = arena or _PageArena()
    store = _store(arena=arena, group_size=1, group_rank=0, full_chunks=full)
    store.reserve(sequence_id)
    store.write_rank_units(sequence_id, token_count=UNALIGNED_TOKENS)
    store.retain_written()
    return store, arena


def test_eviction_releases_record_pages_together_with_kv():
    """The release seam the eviction formula's benefit term relies on: the
    victim's record pages return to the pool, exactly ``record_pages``."""

    store, arena = _written_store(31)
    before = arena.free_pages
    pages = kda_lifecycle.release_records(
        store, [31], rank=0, where="host growth eviction"
    )
    assert pages == store.record_pages
    assert arena.free_pages == before + store.record_pages
    assert not store.has_record(31)
    # The formula's per-victim benefit term is this same constant.
    assert pages == store.record_pages


def test_completion_releases_the_record_and_double_release_raises():
    store, arena = _written_store(32)
    pages = kda_lifecycle.release_records(
        store, [32], rank=0, where="completion host release"
    )
    assert pages == store.record_pages
    with pytest.raises(KDARecordMissing):
        kda_lifecycle.release_records(
            store, [32], rank=0, where="completion host release"
        )


def test_follower_drops_its_mirror_and_never_touches_pages():
    arena = _PageArena()
    owner = _store(arena=arena, group_size=2, group_rank=0)
    owner.reserve(33)
    follower = _store(arena=arena, group_size=2, group_rank=1)
    follower.adopt(owner.mirror_payload([33]))
    before = arena.free_pages
    pages = kda_lifecycle.release_records(
        follower, [], [33], rank=1, where="completion mirror release"
    )
    assert pages == 0
    assert arena.free_pages == before  # pages are the owner's
    assert not follower.has_record(33)
    with pytest.raises(KDARecordStateInvalid):
        follower.adopt(owner.mirror_payload([33]))  # still owner-held
        follower.release(33)


def test_release_of_a_mirrored_record_is_refused_by_name():
    arena = _PageArena()
    owner = _store(arena=arena, group_size=2, group_rank=0)
    owner.reserve(34)
    follower = _store(arena=arena, group_size=2, group_rank=1)
    follower.adopt(owner.mirror_payload([34]))
    with pytest.raises(KDARecordStateInvalid):
        follower.release(34)


def test_release_before_retain_rolls_the_allocation_back():
    arena = _PageArena()
    store = _store(arena=arena, group_size=1, group_rank=0)
    store.reserve(35)
    before = arena.free_pages
    pages = store.release(35)
    assert pages == store.record_pages
    assert arena.rolled_back and not arena.released
    assert arena.free_pages == before + store.record_pages


# ---------------------------------------------------------------------------
# Durable map: the sequence-id -> pages map survives a pool respawn (方案甲
# orphan fix). "The shared region" is the bytearray the map objects share;
# a respawn is a NEW map/store object over the same bytes.
# ---------------------------------------------------------------------------


def test_map_entry_tracks_the_owner_lifecycle_and_clears_on_release():
    region = _region()
    smap = _shared_map(region)
    arena = _PageArena()
    full = _full_chunks(seed=4)
    store = _store(
        arena=arena, group_size=1, group_rank=0, full_chunks=full,
        shared_map=smap,
    )
    store.reserve(51)
    entry = smap.get(51)
    assert entry is not None and not entry.written and not entry.retained
    assert dict(entry.page_ids_by_group) == dict(
        store.record_for(51).arena.page_ids_by_group
    )
    store.write_rank_units(51, token_count=UNALIGNED_TOKENS)
    entry = smap.get(51)
    assert entry.written and entry.token_count == UNALIGNED_TOKENS
    assert not entry.retained
    store.retain_written()
    assert smap.get(51).retained
    store.release(51)
    assert smap.get(51) is None and len(smap) == 0


def test_admission_rollback_clears_the_map_entry():
    smap = _shared_map()
    arena = _PageArena()
    store = _store(
        arena=arena, group_size=1, group_rank=0, shared_map=smap
    )
    store.reserve(52)
    assert smap.get(52) is not None
    store.rollback_reserved([52])
    assert smap.get(52) is None and len(smap) == 0


def test_followers_never_write_the_map():
    region = _region()
    arena = _PageArena()
    full = _full_chunks(seed=6)
    owner = _store(
        arena=arena, group_size=2, group_rank=0, full_chunks=full,
        shared_map=_shared_map(region),
    )
    owner.reserve(53)
    follower = _store(
        arena=arena, group_size=2, group_rank=1, full_chunks=full,
        shared_map=KDARecordSharedMap(region, geometry=_map_geometry()),
        rank=1,
    )
    follower.adopt(owner.mirror_payload([53]))
    follower.write_rank_units(53, token_count=UNALIGNED_TOKENS)
    # The entry mirrors the OWNER's lifecycle alone: the follower's unit
    # write did not stamp it.
    entry = owner.shared_map.get(53)
    assert entry.owner_rank == 0 and not entry.written
    owner.write_rank_units(53, token_count=UNALIGNED_TOKENS)
    assert owner.shared_map.get(53).written


def test_current_cohort_entries_are_not_rehydrated_by_sibling_stores():
    """A store built while its node siblings already wrote entries must not
    install them: those mirrors arrive through ``adopt`` exactly as before."""

    region = _region()
    arena = _PageArena()
    owner = _store(
        arena=arena, group_size=2, group_rank=0,
        shared_map=_shared_map(region),
    )
    owner.reserve(54)
    follower = _store(
        arena=arena, group_size=2, group_rank=1,
        shared_map=KDARecordSharedMap(region, geometry=_map_geometry()),
        rank=1,
    )
    assert not follower.has_record(54)
    follower.adopt(owner.mirror_payload([54]))  # must not raise
    assert follower.record_for(54).mirrored


def test_pool_respawn_releases_dead_records_and_keeps_survivors():
    """The orphan fix end to end: a simulated pool respawn (new objects over
    the same shared region) releases the records of DEAD sequences exactly
    and keeps the SURVIVOR's record readable, with exact page accounting."""

    region = _region()
    arena = _PageArena(free_pages=1000)
    full = _full_chunks(seed=5)
    store = _store(
        arena=arena, group_size=1, group_rank=0, full_chunks=full,
        shared_map=_shared_map(region),
    )
    before = arena.free_pages
    pages = store.record_pages
    for sid in (41, 42):
        store.reserve(sid)
        store.write_rank_units(sid, token_count=UNALIGNED_TOKENS)
    store.retain_written()
    store.reserve(43)  # reserved, never written: still under its pseudo row
    assert arena.free_pages == before - 3 * pages
    unretained_arena = store.record_for(43).arena

    # The pool dies. The server's cohort reconcile (release_all_sequences)
    # returns every page still chain-owned by a pseudo row -- 43's -- and
    # cannot name the prefix-resident ones (41's and 42's).
    arena.rollback(unretained_arena)
    assert arena.free_pages == before - 2 * pages

    # Recovery sweep on the new cohort's node leader: 42 is still alive
    # server-side, 41 and 43 are not.
    recovery_map = KDARecordSharedMap(region, geometry=_map_geometry())
    released = kda_lifecycle.release_restart_orphans(
        recovery_map, arena, [42], rank=0
    )
    assert released == pages  # 41's resident pages, exactly once
    assert arena.free_pages == before - pages  # only 42's record remains
    assert sorted(recovery_map.entries()) == [42]

    # The new cohort's store over the same region sees the survivor...
    writer = _StateWriter()
    store2 = _store(
        arena=arena, group_size=1, group_rank=0, state_writer=writer,
        shared_map=KDARecordSharedMap(region, geometry=_map_geometry()),
    )
    assert store2.has_record(42) and store2.is_written(42)
    assert not store2.has_record(41) and not store2.has_record(43)
    assert not store2.record_for(42).mirrored
    assert store2.record_for(42).token_count == UNALIGNED_TOKENS

    # ...and its record restores byte-identically.
    store2.restore_rank_units(42, group_size=1, group_rank=0)
    assert writer.written[42] == _chunk_bytes(full)

    # Release accounts exactly, and a double release still raises by name.
    assert store2.release(42) == pages
    assert arena.free_pages == before
    with pytest.raises(KDARecordMissing):
        store2.release(42)
    assert len(recovery_map) == 0


def test_respawn_on_another_rank_rehydrates_the_survivor_as_a_mirror():
    region = _region()
    arena = _PageArena()
    full = _full_chunks(seed=7)
    store = _store(
        arena=arena, group_size=1, group_rank=0, full_chunks=full,
        shared_map=_shared_map(region),
    )
    store.reserve(45)
    store.write_rank_units(45, token_count=UNALIGNED_TOKENS)
    store.retain_written()
    kda_lifecycle.release_restart_orphans(
        KDARecordSharedMap(region, geometry=_map_geometry()),
        arena,
        [45],
        rank=0,
    )
    writer = _StateWriter()
    other_rank = _store(
        arena=arena, group_size=1, group_rank=0, state_writer=writer,
        shared_map=KDARecordSharedMap(region, geometry=_map_geometry()),
        rank=1,
    )
    record = other_rank.record_for(45)
    assert record.mirrored and record.written
    other_rank.restore_rank_units(45, group_size=1, group_rank=0)
    assert writer.written[45] == _chunk_bytes(full)
    with pytest.raises(KDARecordStateInvalid):
        other_rank.release(45)  # pages stay the owner's, exactly as before


def test_reserve_hands_the_pages_back_when_the_map_is_full():
    arena = _PageArena()
    store = _store(
        arena=arena, group_size=1, group_rank=0,
        shared_map=_shared_map(capacity=1),
    )
    store.reserve(61)
    before = arena.free_pages
    with pytest.raises(KDARecordMapFull):
        store.reserve(62)
    assert arena.free_pages == before  # the second arena went straight back
    assert not store.has_record(62)
    assert store.has_record(61)


def test_map_layout_mismatch_is_refused_by_name():
    from batchgen.prefix_reuse.kda_record_map import KDARecordMapCorrupt

    region = _region(capacity=32)
    _shared_map(region)  # initializes the header at capacity 32
    with pytest.raises(KDARecordMapLayoutMismatch):
        KDARecordSharedMap(region, geometry=_map_geometry(capacity=16))
    with pytest.raises(KDARecordMapCorrupt):
        KDARecordSharedMap.from_existing(bytearray(1024))


# ---------------------------------------------------------------------------
# A model without recurrent state: every lifecycle step is structurally empty
# ---------------------------------------------------------------------------


def test_stateless_model_has_no_store_and_every_step_is_empty():
    assert kda_lifecycle.offload_prefill_states(
        None, lambda ids: (_ for _ in ()).throw(AssertionError), [(1, 2)],
        rank=0,
    ) == ()
    assert kda_lifecycle.restore_decode_states(
        None, [1], group_size=1, group_rank=0, rank=0
    ) == ()
    assert kda_lifecycle.release_records(
        None, [1], rank=0, where="completion host release"
    ) == 0
    # Nothing persisted, nothing to recover: the startup sweep is the same
    # structural absence (a stateless model never sizes the map region).
    assert kda_lifecycle.release_restart_orphans(
        None, _PageArena(), (), rank=0
    ) == 0
