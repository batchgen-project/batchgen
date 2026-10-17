"""Contract tests for the trajectory-frontier commit transaction (M3 1b).

Pure Python, no GPU. Only the two injected boundaries are faked -- the D2H slot
copy and the Host-KV arena allocator -- so the registry, the record, the
snapshot profile and the charge math under test are the real objects.

What these tests are defending:
* the single batch-level guard, including the named refusal of a mistyped flag;
* the arena page count coming from ``FrontierCharge``, not a second formula;
* replace-on-advance ordering (new referenced before old released) and the
  rollback when the transaction fails between the copy and the swap;
* eviction releasing the SNAPSHOT pages plus the registry entry, and never the
  prefix node's own pages;
* the fail-safe skip when the Host pool cannot supply arena pages.
"""

import inspect
import logging
from dataclasses import replace
from types import SimpleNamespace

import pytest

from batchgen.prefix_reuse.commit import release_evicted_prefix_pages
from batchgen.prefix_reuse.trajectory_commit import (
    DEBUG_FLAG,
    SNAPSHOT_CHUNK_NAMES,
    HostKVSnapshotArenaAllocator,
    SnapshotArena,
    SnapshotArenaCorrupt,
    SnapshotArenaUnavailable,
    SnapshotLayout,
    TrajectoryCommitFlagInvalid,
    TrajectoryFrontierCommitter,
    TrajectorySnapshotShapeMismatch,
    TrajectoryStateUnavailable,
    parse_trajectory_commit_flag,
    profile_from_kda_state_config,
    trajectory_commit_enabled,
)
from batchgen.prefix_reuse.trajectory_frontier import (
    FrontierCharge,
    FrontierRecordInvalid,
    FrontierReleaseFailed,
    KDASnapshotProfile,
    session_key_for_baseline,
)


PAGE_TOKENS = 4
# One Host page across all layers; the real value is
# PrefixCacheRuntimeConfig.host_page_bytes_all_layers.
HOST_PAGE_BYTES = 2048
NAMESPACE = (11, 22, 33, 44)


def _profile() -> KDASnapshotProfile:
    """A small but structurally real profile: 3 arena pages at 2 KiB/page."""

    return KDASnapshotProfile(
        num_kda_layers=2,
        num_heads_per_rank=2,
        head_dim=16,
        conv_dim_per_rank=32,
        conv_width=4,
        recurrent_element_bytes=4,
        conv_element_bytes=2,
    )


class _Chunk:
    """A copied slot slice, reduced to the one thing the commit checks."""

    def __init__(self, nbytes: int):
        self._nbytes = int(nbytes)

    def numel(self) -> int:
        return self._nbytes


class _SnapshotReader:
    """The D2H boundary: hands back chunks in the fixed profile order."""

    def __init__(self, profile, *, fail=None, chunk_bytes=None):
        self.profile = profile
        self.calls = []
        self._fail = fail
        self._chunk_bytes = chunk_bytes

    def read_slot(self, sequence_id):
        self.calls.append(int(sequence_id))
        if self._fail is not None:
            raise self._fail
        sizes = self._chunk_bytes or SnapshotLayout(
            profile=self.profile
        ).chunk_bytes
        return tuple(_Chunk(size) for size in sizes)


class _ArenaAllocator:
    """The allocator boundary: hands out page ids, records every release."""

    def __init__(self, *, first_page=900, fail_times=0, pages_per_group=None):
        self.stores = []
        self.releases = []
        # The G>1 phases, recorded separately so a test can assert that a
        # follower never allocated and that an arena was retained exactly once.
        self.allocations = []
        self.shard_writes = []
        self.retains = []
        self.rollbacks = []
        self._next_page = int(first_page)
        self._next_sid = -1
        self._fail_times = int(fail_times)
        self._groups = pages_per_group or (0,)
        self.shard_write_error = None

    @staticmethod
    def identity(arena):
        """Arena identity that survives the caller stamping its handle."""

        return (
            int(arena.pseudo_sid),
            tuple(
                (int(group_id), tuple(int(page) for page in pages))
                for group_id, pages in sorted(arena.page_ids_by_group.items())
            ),
        )

    def store(self, *, page_count, chunks):
        if self._fail_times > 0:
            self._fail_times -= 1
            raise SnapshotArenaUnavailable("pool is out of Host pages")
        pages_by_group = {}
        for group_id in self._groups:
            pages = tuple(
                self._next_page + offset for offset in range(int(page_count))
            )
            self._next_page += int(page_count)
            pages_by_group[int(group_id)] = pages
        sid = self._next_sid
        self._next_sid -= 1
        arena = SnapshotArena(
            storage_handle=0,
            pseudo_sid=sid,
            page_ids_by_group=pages_by_group,
            pages_per_group=int(page_count),
            capacity_bytes=int(page_count) * HOST_PAGE_BYTES,
            payload_bytes=sum(int(chunk.numel()) for chunk in chunks),
        )
        self.stores.append(self.identity(arena))
        return arena

    # -- the G>1 phases ----------------------------------------------------

    def allocate(self, *, page_count, shard_count=1):
        if self._fail_times > 0:
            self._fail_times -= 1
            raise SnapshotArenaUnavailable("pool is out of Host pages")
        pages_by_group = {}
        for group_id in self._groups:
            pages = tuple(
                self._next_page + offset for offset in range(int(page_count))
            )
            self._next_page += int(page_count)
            pages_by_group[int(group_id)] = pages
        sid = self._next_sid
        self._next_sid -= 1
        arena = SnapshotArena(
            storage_handle=0,
            pseudo_sid=sid,
            page_ids_by_group=pages_by_group,
            pages_per_group=int(page_count),
            capacity_bytes=int(page_count) * HOST_PAGE_BYTES,
            payload_bytes=0,
            shard_count=int(shard_count),
        )
        self.allocations.append(self.identity(arena))
        return arena

    def write_shard(self, arena, *, shard_index, chunks):
        if self.shard_write_error is not None:
            raise self.shard_write_error
        window = arena.shard_window(int(shard_index))
        payload = sum(int(chunk.numel()) for chunk in chunks)
        self.shard_writes.append(
            (
                int(arena.pseudo_sid),
                int(shard_index),
                tuple(
                    (int(group_id), tuple(pages))
                    for group_id, pages in sorted(window.items())
                ),
                payload,
            )
        )
        return payload

    def retain(self, arena):
        self.retains.append(self.identity(arena))
        return arena

    def rollback(self, arena):
        self.rollbacks.append(int(arena.pseudo_sid))

    def release(self, arena):
        self.releases.append(self.identity(arena))
        return {
            int(group_id): len(pages)
            for group_id, pages in arena.page_ids_by_group.items()
        }

    # -- assertions helpers ------------------------------------------------

    def released_page_ids(self):
        ids = set()
        for _pseudo_sid, groups in self.releases:
            for _group_id, pages in groups:
                ids.update(int(page) for page in pages)
        return ids


def _committer(**kwargs):
    profile = kwargs.pop("profile", None) or _profile()
    reader = kwargs.pop("reader", None) or _SnapshotReader(profile)
    allocator = kwargs.pop("allocator", None) or _ArenaAllocator()
    committer = TrajectoryFrontierCommitter(
        rank=0,
        page_tokens=PAGE_TOKENS,
        host_page_bytes_all_layers=HOST_PAGE_BYTES,
        arena_allocator=allocator,
        snapshot_reader=reader,
        **kwargs,
    )
    return committer, allocator, reader


def _commit(
    committer,
    *,
    tokens,
    pages,
    sequence_id=7,
    group_id=0,
    prompt_token_count=None,
):
    """Commit a frontier. ``prompt_token_count`` defaults to the ALIGNED case
    (the prefill ran exactly the committed boundary), which is the only case
    Slice 1c stores a snapshot for."""

    return committer.commit(
        namespace_digest=NAMESPACE,
        frontier_token_ids=tokens,
        page_ids_by_group=pages,
        owning_group_id=group_id,
        sequence_id=sequence_id,
        prompt_token_count=(
            len(tokens) if prompt_token_count is None else prompt_token_count
        ),
    )


# ---------------------------------------------------------------------------
# The guard
# ---------------------------------------------------------------------------


def test_absent_flag_is_off_and_touches_nothing():
    assert trajectory_commit_enabled(None) is False
    assert trajectory_commit_enabled({}) is False
    assert trajectory_commit_enabled({"other_flag": "on"}) is False


def test_flag_accepts_only_on_and_off():
    assert parse_trajectory_commit_flag("on") is True
    assert parse_trajectory_commit_flag(" ON ") is True
    assert parse_trajectory_commit_flag("off") is False
    assert parse_trajectory_commit_flag("Off") is False
    assert parse_trajectory_commit_flag(None) is False
    assert trajectory_commit_enabled({DEBUG_FLAG: "on"}) is True
    assert trajectory_commit_enabled({DEBUG_FLAG: "off"}) is False


@pytest.mark.parametrize(
    "value", [True, False, 1, 0, "1", "0", "true", "yes", "enabled", "", "no"]
)
def test_mistyped_flag_is_refused_by_name(value):
    with pytest.raises(TrajectoryCommitFlagInvalid) as excinfo:
        parse_trajectory_commit_flag(value)
    assert DEBUG_FLAG in str(excinfo.value)
    assert '"on"' in str(excinfo.value)


def test_mistyped_flag_is_refused_through_the_guard():
    with pytest.raises(TrajectoryCommitFlagInvalid):
        trajectory_commit_enabled({DEBUG_FLAG: "1"})


def test_flag_off_leaves_the_registry_empty():
    committer, allocator, reader = _committer()
    enabled = trajectory_commit_enabled({DEBUG_FLAG: "off"})
    if enabled:  # pragma: no cover - the guard under test says it is off
        _commit(committer, tokens=range(8), pages={0: [1, 2]})
    assert enabled is False
    assert len(committer.registry) == 0
    assert committer.reverse_index() == {}
    assert allocator.stores == []
    assert reader.calls == []


def test_commit_cannot_touch_host_owned_pages():
    """R1: ``host_owned_pages`` is never moved by a snapshot.

    Structural, not incidental: the commit step is handed tokens, page ids, a
    group id and a sequence id -- never a sequence object -- so the release
    promise is not reachable from here and the 11222 retention deduction stays
    the only move at that boundary.
    """

    parameters = inspect.signature(TrajectoryFrontierCommitter.commit).parameters
    assert set(parameters) == {
        "self",
        "namespace_digest",
        "frontier_token_ids",
        "page_ids_by_group",
        "owning_group_id",
        "sequence_id",
        # Slice 1c: a token COUNT, not a sequence -- the boundary identity the
        # snapshot has to satisfy, with no path to the release promise.
        "prompt_token_count",
    }


# ---------------------------------------------------------------------------
# Arena geometry
# ---------------------------------------------------------------------------


def test_layout_chunk_sizes_and_offsets_follow_the_profile():
    profile = _profile()
    layout = SnapshotLayout(profile=profile)
    per_pool = profile.conv_bytes // profile.num_conv_pools
    assert layout.chunk_bytes == (profile.recurrent_bytes,) + (per_pool,) * 3
    assert len(SNAPSHOT_CHUNK_NAMES) == len(layout.chunk_bytes)
    assert layout.chunk_offsets == (
        0,
        profile.recurrent_bytes,
        profile.recurrent_bytes + per_pool,
        profile.recurrent_bytes + 2 * per_pool,
    )
    assert layout.total_bytes == profile.total_bytes
    assert sum(layout.chunk_bytes) == profile.total_bytes


def test_arena_page_count_comes_from_the_frontier_charge():
    layout = SnapshotLayout(profile=_profile())
    for prefix_pages in (1, 2, 17):
        expected = FrontierCharge(
            page_count=prefix_pages,
            host_page_bytes_all_layers=HOST_PAGE_BYTES,
            snapshot_bytes=layout.total_bytes,
        ).snapshot_host_pages
        assert (
            layout.arena_page_count(
                host_page_bytes_all_layers=HOST_PAGE_BYTES,
                prefix_page_count=prefix_pages,
            )
            == expected
        )


def test_arena_page_count_rounds_up_and_ignores_prefix_pages():
    layout = SnapshotLayout(profile=_profile())
    counts = {
        layout.arena_page_count(
            host_page_bytes_all_layers=HOST_PAGE_BYTES,
            prefix_page_count=prefix_pages,
        )
        for prefix_pages in (1, 5, 40)
    }
    assert counts == {-(-layout.total_bytes // HOST_PAGE_BYTES)}
    assert layout.total_bytes > 2 * HOST_PAGE_BYTES  # a multi-page snapshot


def test_arena_requires_a_negative_pseudo_sid():
    with pytest.raises(SnapshotArenaCorrupt) as excinfo:
        SnapshotArena(
            storage_handle=1,
            pseudo_sid=5,
            page_ids_by_group={0: (1,)},
            pages_per_group=1,
            capacity_bytes=HOST_PAGE_BYTES,
            payload_bytes=8,
        )
    assert "NEGATIVE" in str(excinfo.value)


def test_arena_refuses_a_payload_larger_than_its_pages():
    with pytest.raises(SnapshotArenaCorrupt):
        SnapshotArena(
            storage_handle=1,
            pseudo_sid=-5,
            page_ids_by_group={0: (1,)},
            pages_per_group=1,
            capacity_bytes=HOST_PAGE_BYTES,
            payload_bytes=HOST_PAGE_BYTES + 1,
        )


def test_profile_from_state_pool_config_matches_the_live_tensors():
    config = SimpleNamespace(
        num_kda_layers=2,
        num_heads=2,
        head_dim=16,
        conv_width=4,
        recurrent_dtype="torch.float32",
        conv_dtype="torch.bfloat16",
        resolved_conv_dim=lambda: 32,
    )
    assert profile_from_kda_state_config(config) == _profile()


def test_profile_from_state_pool_config_refuses_missing_geometry():
    with pytest.raises(TrajectoryStateUnavailable):
        profile_from_kda_state_config(SimpleNamespace(num_kda_layers=2))


# ---------------------------------------------------------------------------
# The commit transaction
# ---------------------------------------------------------------------------


def test_first_commit_begins_the_session_and_records_its_arena():
    committer, allocator, reader = _committer()
    outcome = _commit(committer, tokens=range(8), pages={0: [1, 2]})

    assert outcome is not None
    assert outcome.advanced is False
    assert outcome.previous_release is None
    assert outcome.session_key == session_key_for_baseline(
        namespace_digest=NAMESPACE, baseline_token_ids=list(range(8))
    )
    assert len(committer.registry) == 1
    committer.registry.check_invariants()
    assert reader.calls == [7]

    arena = committer.arena_for(outcome.record.snapshot.storage_handle)
    assert arena is outcome.arena
    assert arena.pages_per_group == 3  # ceil(5248 / 2048)
    assert arena.payload_bytes == committer.layout.total_bytes
    assert set(arena.page_ids_by_group[0]).isdisjoint({1, 2})
    assert allocator.releases == []


def test_commit_records_every_prefix_page_in_the_reverse_index():
    committer, _allocator, _reader = _committer()
    outcome = _commit(committer, tokens=range(8), pages={0: [1, 2]})
    assert committer.reverse_index() == {
        (0, 1): frozenset({outcome.session_key}),
        (0, 2): frozenset({outcome.session_key}),
    }


def test_commit_at_an_already_committed_boundary_is_a_no_op():
    committer, allocator, reader = _committer()
    first = _commit(committer, tokens=range(8), pages={0: [1, 2]})
    again = _commit(committer, tokens=range(8), pages={0: [1, 2]})

    assert again is None
    assert len(allocator.stores) == 1
    assert allocator.releases == []
    assert reader.calls == [7]
    assert committer.registry.require(first.session_key) is first.record


def test_advance_references_the_new_record_before_releasing_the_old():
    committer, allocator, _reader = _committer()
    first = _commit(committer, tokens=range(8), pages={0: [1, 2]})
    old_arena = first.arena
    second = _commit(
        committer, tokens=range(12), pages={0: [1, 2, 3]}, sequence_id=9
    )

    assert second is not None and second.advanced is True
    assert second.session_key == first.session_key
    assert second.record is not first.record
    # The old arena is released, and only AFTER the new one exists.
    assert allocator.stores == [
        _ArenaAllocator.identity(old_arena),
        _ArenaAllocator.identity(second.arena),
    ]
    # The old arena is released exactly once, and its store came FIRST: the
    # new snapshot existed before the old one was freed.
    assert allocator.releases == [_ArenaAllocator.identity(old_arena)]
    assert committer.arena_for(old_arena.storage_handle) is None
    assert committer.arena_for(second.arena.storage_handle) is second.arena
    assert second.previous_release is not None
    assert (
        second.previous_release.released_snapshot_handle
        == old_arena.storage_handle
    )
    assert len(committer.registry) == 1
    committer.registry.check_invariants()


def test_advance_rebuilds_the_reverse_index_for_the_session():
    committer, _allocator, _reader = _committer()
    first = _commit(committer, tokens=range(8), pages={0: [1, 2]})
    _commit(committer, tokens=range(12), pages={0: [5, 6, 7]})
    assert committer.reverse_index() == {
        (0, 5): frozenset({first.session_key}),
        (0, 6): frozenset({first.session_key}),
        (0, 7): frozenset({first.session_key}),
    }


def test_failure_between_the_copy_and_the_swap_leaves_the_old_frontier():
    committer, allocator, _reader = _committer()
    first = _commit(committer, tokens=range(8), pages={0: [1, 2]})
    old_arena = first.arena

    # A new frontier whose page count does not match its token count: the
    # record refuses to build, which is exactly the window between the D2H
    # copy (done) and the registry swap (not yet attempted).
    with pytest.raises(FrontierRecordInvalid):
        _commit(committer, tokens=range(12), pages={0: [1, 2]})

    # Old record and old pages survive untouched.
    assert committer.registry.require(first.session_key) is first.record
    assert first.record.is_released is False
    assert first.record.refcount == 1
    assert committer.arena_for(old_arena.storage_handle) is old_arena
    assert _ArenaAllocator.identity(old_arena) not in allocator.releases
    # The NEW arena is the only thing rolled back.
    assert len(allocator.stores) == 2
    assert allocator.releases == [allocator.stores[1]]
    assert allocator.stores[1] != allocator.stores[0]
    committer.registry.check_invariants()
    assert committer.reverse_index() == {
        (0, 1): frozenset({first.session_key}),
        (0, 2): frozenset({first.session_key}),
    }


def test_a_failing_d2h_copy_allocates_no_arena_at_all():
    reader = _SnapshotReader(_profile(), fail=RuntimeError("device fault"))
    committer, allocator, _reader = _committer(reader=reader)
    with pytest.raises(RuntimeError, match="device fault"):
        _commit(committer, tokens=range(8), pages={0: [1, 2]})
    assert allocator.stores == []
    assert allocator.releases == []
    assert len(committer.registry) == 0


def test_short_snapshot_chunks_are_refused_by_name():
    layout = SnapshotLayout(profile=_profile())
    sizes = list(layout.chunk_bytes)
    sizes[2] -= 1
    reader = _SnapshotReader(_profile(), chunk_bytes=tuple(sizes))
    committer, allocator, _reader = _committer(reader=reader)
    with pytest.raises(TrajectorySnapshotShapeMismatch) as excinfo:
        _commit(committer, tokens=range(8), pages={0: [1, 2]})
    assert "conv_k" in str(excinfo.value)
    assert allocator.stores == []
    assert len(committer.registry) == 0


def test_wrong_chunk_count_is_refused_by_name():
    reader = _SnapshotReader(_profile(), chunk_bytes=(16, 16))
    committer, _allocator, _reader = _committer(reader=reader)
    with pytest.raises(TrajectorySnapshotShapeMismatch) as excinfo:
        _commit(committer, tokens=range(8), pages={0: [1, 2]})
    assert "4 chunks" in str(excinfo.value)


# ---------------------------------------------------------------------------
# The allocation-failure fail-safe
# ---------------------------------------------------------------------------


def test_pool_pressure_skips_the_commit_with_a_named_log(caplog):
    allocator = _ArenaAllocator(fail_times=1)
    committer, _allocator, reader = _committer(allocator=allocator)
    with caplog.at_level(logging.WARNING):
        outcome = _commit(committer, tokens=range(8), pages={0: [1, 2]})

    assert outcome is None
    assert len(committer.registry) == 0
    assert committer.reverse_index() == {}
    assert allocator.releases == []
    assert reader.calls == [7]  # the copy happened; only the arena failed
    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "TRAJECTORY_FRONTIER" in message
        and "SKIPPED frontier commit" in message
        and "degrades to recompute" in message
        for message in messages
    ), messages


def test_pool_pressure_does_not_disturb_an_existing_frontier(caplog):
    allocator = _ArenaAllocator()
    committer, _allocator, _reader = _committer(allocator=allocator)
    first = _commit(committer, tokens=range(8), pages={0: [1, 2]})
    allocator._fail_times = 1
    with caplog.at_level(logging.WARNING):
        assert _commit(committer, tokens=range(12), pages={0: [1, 2, 3]}) is None

    assert committer.registry.require(first.session_key) is first.record
    assert committer.arena_for(first.arena.storage_handle) is first.arena
    assert allocator.releases == []
    committer.registry.check_invariants()


def test_a_later_commit_succeeds_after_pool_pressure_clears():
    allocator = _ArenaAllocator(fail_times=1)
    committer, _allocator, _reader = _committer(allocator=allocator)
    assert _commit(committer, tokens=range(8), pages={0: [1, 2]}) is None
    outcome = _commit(committer, tokens=range(8), pages={0: [1, 2]})
    assert outcome is not None
    assert len(committer.registry) == 1


# ---------------------------------------------------------------------------
# Eviction
# ---------------------------------------------------------------------------


def test_eviction_frees_the_snapshot_pages_and_the_entry_only():
    committer, allocator, _reader = _committer()
    outcome = _commit(committer, tokens=range(8), pages={0: [1, 2]})
    prefix_pages = {1, 2}

    released = committer.on_prefix_pages_released({0: [1]})

    assert len(released) == 1
    assert released[0].session_key == outcome.session_key
    assert (
        released[0].released_snapshot_handle == outcome.arena.storage_handle
    )
    # Registry entry gone, arena gone, reverse index gone.
    assert len(committer.registry) == 0
    assert committer.registry.get(outcome.session_key) is None
    assert committer.arena_for(outcome.arena.storage_handle) is None
    assert committer.reverse_index() == {}
    # The ONLY pages handed to the allocator are the arena's. The prefix pages
    # belong to the prefix node and were already released by the caller.
    assert allocator.releases == [_ArenaAllocator.identity(outcome.arena)]
    assert allocator.released_page_ids() == set(
        outcome.arena.page_ids_by_group[0]
    )
    assert allocator.released_page_ids().isdisjoint(prefix_pages)


def test_eviction_is_all_or_nothing_for_a_multi_page_frontier():
    committer, allocator, _reader = _committer()
    outcome = _commit(committer, tokens=range(12), pages={0: [4, 5, 6]})
    # ONE evicted page retires the whole record: a partial frontier is a miss.
    committer.on_prefix_pages_released({0: [6]})
    assert len(committer.registry) == 0
    assert allocator.releases == [_ArenaAllocator.identity(outcome.arena)]


def test_eviction_of_a_shared_page_retires_every_frontier_on_it():
    committer, allocator, _reader = _committer()
    first = _commit(committer, tokens=range(8), pages={0: [1, 2]})
    second = _commit(
        committer,
        tokens=[0, 1, 2, 3, 90, 91, 92, 93],
        pages={0: [1, 9]},
        sequence_id=8,
    )
    assert second is not None and second.session_key != first.session_key
    assert committer.reverse_index()[(0, 1)] == frozenset(
        {first.session_key, second.session_key}
    )

    released = committer.on_prefix_pages_released({0: [1]})

    assert {outcome.session_key for outcome in released} == {
        first.session_key,
        second.session_key,
    }
    assert len(committer.registry) == 0
    assert sorted(allocator.releases) == sorted(
        [
            _ArenaAllocator.identity(first.arena),
            _ArenaAllocator.identity(second.arena),
        ]
    )


def test_evicting_pages_no_frontier_owns_changes_nothing():
    committer, allocator, _reader = _committer()
    outcome = _commit(committer, tokens=range(8), pages={0: [1, 2]})
    assert committer.on_prefix_pages_released({0: [77, 78]}) == ()
    assert len(committer.registry) == 1
    assert allocator.releases == []
    assert committer.arena_for(outcome.arena.storage_handle) is outcome.arena


def test_a_record_whose_arena_vanished_is_refused_by_name():
    committer, _allocator, _reader = _committer()
    outcome = _commit(committer, tokens=range(8), pages={0: [1, 2]})
    committer._arenas_by_handle.pop(outcome.arena.storage_handle)
    with pytest.raises(FrontierReleaseFailed) as excinfo:
        committer.on_prefix_pages_released({0: [1]})
    assert "has no arena" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, SnapshotArenaCorrupt)
    # The release stays retryable: the record is handed back, not stranded.
    assert excinfo.value.record is outcome.record


# ---------------------------------------------------------------------------
# The prefix-eviction seam in commit.py
# ---------------------------------------------------------------------------


class _EvictionWorkerView:
    def __init__(self):
        self.released = []

    def release_resident_pages(self, page_ids):
        self.released.append(list(page_ids))


def _eviction_result(pages_by_group):
    return SimpleNamespace(
        evicted_group_pages=[
            SimpleNamespace(
                group_id=group_id,
                pages=[SimpleNamespace(page_id=page) for page in pages],
            )
            for group_id, pages in pages_by_group.items()
        ]
    )


def test_release_evicted_prefix_pages_is_unchanged_without_a_hook():
    view = _EvictionWorkerView()
    released = release_evicted_prefix_pages(
        eviction_result=_eviction_result({0: [3, 4]}),
        worker_views_by_group={0: view},
    )
    assert released == {0: 2}
    assert view.released == [[3, 4]]


def test_release_evicted_prefix_pages_passes_every_group_to_the_hook():
    views = {0: _EvictionWorkerView(), 1: _EvictionWorkerView()}
    seen = []
    released = release_evicted_prefix_pages(
        eviction_result=_eviction_result({0: [3, 4], 1: [8]}),
        worker_views_by_group=views,
        frontier_hook=seen.append,
    )
    assert released == {0: 2, 1: 1}
    assert seen == [{0: [3, 4], 1: [8]}]
    assert views[0].released == [[3, 4]]
    assert views[1].released == [[8]]


def test_the_hook_retires_the_frontier_standing_on_evicted_prefix_pages():
    committer, allocator, _reader = _committer()
    outcome = _commit(committer, tokens=range(8), pages={0: [1, 2]})
    view = _EvictionWorkerView()

    release_evicted_prefix_pages(
        eviction_result=_eviction_result({0: [1, 2]}),
        worker_views_by_group={0: view},
        frontier_hook=committer.on_prefix_pages_released,
    )

    # The prefix pages were released exactly once, by the prefix path.
    assert view.released == [[1, 2]]
    # The snapshot arena was released exactly once, by the frontier path.
    assert allocator.releases == [_ArenaAllocator.identity(outcome.arena)]
    assert len(committer.registry) == 0


# ---------------------------------------------------------------------------
# The real Host-KV arena allocator's refusals (no torch needed for these)
# ---------------------------------------------------------------------------


def test_host_arena_allocator_validates_its_geometry():
    with pytest.raises(SnapshotArenaCorrupt):
        HostKVSnapshotArenaAllocator(
            host_views_provider=dict,
            page_tokens=0,
            host_page_bytes_all_layers=HOST_PAGE_BYTES,
            next_pseudo_sid=lambda: -1,
        )
    with pytest.raises(SnapshotArenaCorrupt):
        HostKVSnapshotArenaAllocator(
            host_views_provider=dict,
            page_tokens=PAGE_TOKENS,
            host_page_bytes_all_layers=0,
            next_pseudo_sid=lambda: -1,
        )


def test_host_arena_allocator_refuses_a_zero_page_arena():
    allocator = HostKVSnapshotArenaAllocator(
        host_views_provider=dict,
        page_tokens=PAGE_TOKENS,
        host_page_bytes_all_layers=HOST_PAGE_BYTES,
        next_pseudo_sid=lambda: -1,
    )
    with pytest.raises(SnapshotArenaCorrupt):
        allocator.store(page_count=0, chunks=())


def test_host_arena_allocator_refuses_an_empty_view_set():
    allocator = HostKVSnapshotArenaAllocator(
        host_views_provider=dict,
        page_tokens=PAGE_TOKENS,
        host_page_bytes_all_layers=HOST_PAGE_BYTES,
        next_pseudo_sid=lambda: -1,
    )
    with pytest.raises(SnapshotArenaCorrupt) as excinfo:
        allocator.store(page_count=1, chunks=())
    assert "no Host KV worker views" in str(excinfo.value)


def test_host_arena_allocator_reports_pool_pressure_as_unavailable():
    class _View:
        def register_sequences(self, sids):
            return None

        def allocate_pages_for_sequences(self, requests):
            raise RuntimeError(
                "Insufficient free pages available for prefill batch"
            )

        def release_sequence_pages(self, sids):
            return None

        def unregister_sequences(self, sids):
            return None

    allocator = HostKVSnapshotArenaAllocator(
        host_views_provider=lambda: {0: _View()},
        page_tokens=PAGE_TOKENS,
        host_page_bytes_all_layers=HOST_PAGE_BYTES,
        next_pseudo_sid=lambda: -11,
    )
    with pytest.raises(SnapshotArenaUnavailable) as excinfo:
        allocator.store(page_count=3, chunks=())
    assert "Insufficient free pages" in str(excinfo.value)


def test_host_arena_allocator_uses_a_negative_pseudo_sid_and_retains():
    """The pool-segment lane, end to end, with the view faked.

    register -> allocate -> write -> retain -> unregister, and the pages leave
    through ``release_resident_pages`` because retain made them resident.
    """

    class _View:
        def __init__(self):
            self.calls = []

        def register_sequences(self, sids):
            self.calls.append(("register", list(sids)))

        def allocate_pages_for_sequences(self, requests):
            self.calls.append(("allocate", list(requests)))

        def build_page_table(self, sids):
            return [[21, 22]]

        def set_sequence_pages(self, sid, page_ids):
            # Rank-local page-table write: no allocator state moves.
            self.calls.append(("set_row", int(sid), list(page_ids)))

        def read_sequence_kv_to_cpu(self, sid):
            return (_FakeTensor(2 * HOST_PAGE_BYTES), None)

        def write_sequence_kv_from_cpu(self, sid, k, v):
            self.calls.append(("write", int(sid)))

        def retain_sequence_pages(self, sid, page_ids):
            self.calls.append(("retain", int(sid), list(page_ids)))
            return list(page_ids)

        def unregister_sequences(self, sids):
            self.calls.append(("unregister", list(sids)))

        def release_resident_pages(self, page_ids):
            self.calls.append(("release_resident", list(page_ids)))

    view = _View()
    allocator = HostKVSnapshotArenaAllocator(
        host_views_provider=lambda: {0: view},
        page_tokens=PAGE_TOKENS,
        host_page_bytes_all_layers=HOST_PAGE_BYTES,
        next_pseudo_sid=lambda: -13,
        torch_module=_FakeTorch(),
    )
    arena = allocator.store(page_count=2, chunks=(_FakeTensor(64),))

    assert arena.pseudo_sid == -13
    assert arena.page_ids_by_group == {0: (21, 22)}
    assert ("retain", -13, [21, 22]) in view.calls
    assert view.calls.index(("retain", -13, [21, 22])) > view.calls.index(
        ("write", -13)
    )
    assert view.calls[-1] == ("unregister", [-13])

    allocator.release(arena)
    assert view.calls[-1] == ("release_resident", [21, 22])


class _FakeTorch:
    uint8 = "uint8"

    def __init__(self):
        self.empty_calls = []

    def empty(self, shape, dtype=None):
        self.empty_calls.append((tuple(shape), dtype))
        nbytes = 1
        for dim in shape:
            nbytes *= int(dim)
        return _FakeTensor(nbytes)


class _FakeTensor:
    """The slice of tensor behavior ``_write_blob`` actually uses."""

    dtype = "uint8"

    def __init__(self, nbytes, offset=0):
        self._nbytes = int(nbytes)
        self._offset = int(offset)

    @property
    def shape(self):
        return (self._nbytes,)

    def numel(self):
        return self._nbytes

    def reshape(self, _shape):
        return self

    def view(self, _dtype):
        return self

    def zero_(self):
        return self

    def __getitem__(self, key):
        if isinstance(key, slice):
            start = key.start or 0
            stop = key.stop if key.stop is not None else self._nbytes
            return _FakeTensor(max(0, stop - start), self._offset + start)
        raise TypeError(key)

    def __setitem__(self, key, value):
        return None


def test_a_stale_reverse_index_entry_is_refused_by_name():
    """After a failed release the record is retryable, not silently dropped."""

    committer, _allocator, _reader = _committer()
    outcome = _commit(committer, tokens=range(8), pages={0: [1, 2]})
    committer._arenas_by_handle.pop(outcome.arena.storage_handle)
    with pytest.raises(FrontierReleaseFailed):
        committer.on_prefix_pages_released({0: [1]})
    # The index still names the session, and a second pass says so by name
    # rather than quietly retiring nothing.
    assert committer.reverse_index()[(0, 1)] == frozenset(
        {outcome.session_key}
    )
    with pytest.raises(SnapshotArenaCorrupt) as excinfo:
        committer.on_prefix_pages_released({0: [2]})
    assert "no longer holds" in str(excinfo.value)


def test_host_arena_allocator_does_not_downgrade_a_real_bug():
    """Only page exhaustion is recoverable; anything else must propagate."""

    class _View:
        def register_sequences(self, sids):
            return None

        def allocate_pages_for_sequences(self, requests):
            raise AttributeError("worker view has no page table")

        def release_sequence_pages(self, sids):
            return None

        def unregister_sequences(self, sids):
            return None

    allocator = HostKVSnapshotArenaAllocator(
        host_views_provider=lambda: {0: _View()},
        page_tokens=PAGE_TOKENS,
        host_page_bytes_all_layers=HOST_PAGE_BYTES,
        next_pseudo_sid=lambda: -11,
    )
    with pytest.raises(AttributeError, match="no page table"):
        allocator.store(page_count=3, chunks=())


def test_host_arena_allocator_learns_the_staging_shape_once():
    """The arena is read back once per geometry, not on every commit."""

    class _View:
        def __init__(self):
            self.read_backs = 0

        def register_sequences(self, sids):
            return None

        def allocate_pages_for_sequences(self, requests):
            return None

        def build_page_table(self, sids):
            return [[21, 22]]

        def set_sequence_pages(self, sid, page_ids):
            return None

        def read_sequence_kv_to_cpu(self, sid):
            self.read_backs += 1
            return (_FakeTensor(2 * HOST_PAGE_BYTES), None)

        def write_sequence_kv_from_cpu(self, sid, k, v):
            return None

        def retain_sequence_pages(self, sid, page_ids):
            return list(page_ids)

        def unregister_sequences(self, sids):
            return None

    view = _View()
    sids = iter([-21, -22, -23])
    torch_module = _FakeTorch()
    allocator = HostKVSnapshotArenaAllocator(
        host_views_provider=lambda: {0: view},
        page_tokens=PAGE_TOKENS,
        host_page_bytes_all_layers=HOST_PAGE_BYTES,
        next_pseudo_sid=lambda: next(sids),
        torch_module=torch_module,
    )
    for _ in range(3):
        allocator.store(page_count=2, chunks=(_FakeTensor(64),))

    assert view.read_backs == 1
    assert torch_module.empty_calls == [((2 * HOST_PAGE_BYTES,), "uint8")] * 2


# ---------------------------------------------------------------------------
# Slice 1c: the boundary identity -- a snapshot describes EXACTLY its boundary
# ---------------------------------------------------------------------------


def test_an_aligned_prompt_commits_a_record():
    """The prefill ran exactly the committed boundary: the slot IS that state."""

    committer, allocator, reader = _committer()
    outcome = _commit(
        committer, tokens=range(8), pages={0: [1, 2]}, prompt_token_count=8
    )

    assert outcome is not None
    assert outcome.installed is True
    assert outcome.record.frontier_boundary_tokens == 8
    assert len(committer.registry) == 1
    assert committer.skips()["boundary_misaligned"] == 0
    assert reader.calls == [7]
    assert len(allocator.stores) == 1


def test_a_misaligned_prompt_is_skipped_by_name(caplog):
    """The defect Slice 1c closes.

    The D2H reads the LIVE slot, which holds the state after all 11 tokens the
    prefill ran, while the record would be stamped with the page-aligned 8. A
    restore would then resume the suffix at token 8 on top of state that
    already contains tokens 8..10. No record is stored, and the skip says so.
    """

    committer, allocator, reader = _committer()
    with caplog.at_level(logging.WARNING):
        outcome = _commit(
            committer, tokens=range(8), pages={0: [1, 2]}, prompt_token_count=11
        )

    assert outcome is None
    assert len(committer.registry) == 0
    assert committer.reverse_index() == {}
    assert committer.skips()["boundary_misaligned"] == 1
    message = "\n".join(record.getMessage() for record in caplog.records)
    assert "SKIPPED frontier commit" in message
    assert "boundary" in message and "re-apply 3 token(s)" in message
    # The gate sits BEFORE the D2H and before the allocator: a misaligned
    # boundary costs neither a copy nor a Host page.
    assert reader.calls == []
    assert allocator.stores == []
    assert allocator.allocations == []


def test_the_boundary_identity_is_exact_not_a_tolerance():
    committer, _allocator, _reader = _committer()
    assert committer.boundary_is_aligned(prompt_token_count=8, commit_tokens=8)
    assert not committer.boundary_is_aligned(
        prompt_token_count=9, commit_tokens=8
    )
    # A boundary BEYOND the prompt is just as wrong as one behind it.
    assert not committer.boundary_is_aligned(
        prompt_token_count=7, commit_tokens=8
    )


def test_a_misaligned_prompt_leaves_an_existing_frontier_intact():
    committer, allocator, _reader = _committer()
    first = _commit(
        committer, tokens=range(8), pages={0: [1, 2]}, prompt_token_count=8
    )
    assert first is not None

    second = _commit(
        committer,
        tokens=range(12),
        pages={0: [1, 2, 3]},
        prompt_token_count=15,
    )

    assert second is None
    assert len(committer.registry) == 1
    record = committer.registry.require(first.session_key)
    assert record is first.record
    assert record.frontier_boundary_tokens == 8
    assert allocator.releases == []
    assert committer.skips()["boundary_misaligned"] == 1


# ---------------------------------------------------------------------------
# Slice 1c: the G-wide arena and its page-disjoint shard windows
# ---------------------------------------------------------------------------


def test_the_arena_is_g_times_one_shards_pages():
    layout = SnapshotLayout(profile=_profile())
    one = layout.shard_page_count(
        host_page_bytes_all_layers=HOST_PAGE_BYTES, prefix_page_count=2
    )
    for shards in (1, 2, 4, 8):
        assert layout.arena_page_count(
            host_page_bytes_all_layers=HOST_PAGE_BYTES,
            prefix_page_count=2,
            shard_count=shards,
        ) == shards * one
    # Each shard rounds up on its own: summing bytes first and rounding once
    # would hand two ranks a shared page.
    assert one * 4 >= -(-(4 * layout.total_bytes) // HOST_PAGE_BYTES)


def test_the_shard_windows_are_page_disjoint_and_cover_the_arena():
    arena = SnapshotArena(
        storage_handle=1,
        pseudo_sid=-5,
        page_ids_by_group={0: (10, 11, 12, 13), 1: (20, 21, 22, 23)},
        pages_per_group=4,
        capacity_bytes=4 * HOST_PAGE_BYTES,
        payload_bytes=0,
        shard_count=4,
    )
    assert arena.pages_per_shard == 1
    windows = [arena.shard_window(index) for index in range(4)]
    assert windows[0] == {0: (10,), 1: (20,)}
    assert windows[3] == {0: (13,), 1: (23,)}
    seen = set()
    for window in windows:
        for pages in window.values():
            assert not seen & set(pages)
            seen |= set(pages)
    assert seen == {10, 11, 12, 13, 20, 21, 22, 23}


def test_a_two_shard_window_is_a_contiguous_page_run():
    arena = SnapshotArena(
        storage_handle=1,
        pseudo_sid=-5,
        page_ids_by_group={0: (10, 11, 12, 13)},
        pages_per_group=4,
        capacity_bytes=4 * HOST_PAGE_BYTES,
        payload_bytes=0,
        shard_count=2,
    )
    assert arena.shard_window(0) == {0: (10, 11)}
    assert arena.shard_window(1) == {0: (12, 13)}


def test_an_out_of_range_shard_index_is_refused_by_name():
    arena = SnapshotArena(
        storage_handle=1,
        pseudo_sid=-5,
        page_ids_by_group={0: (10, 11)},
        pages_per_group=2,
        capacity_bytes=2 * HOST_PAGE_BYTES,
        payload_bytes=0,
        shard_count=2,
    )
    for index in (-1, 2, 99):
        with pytest.raises(SnapshotArenaCorrupt) as excinfo:
            arena.shard_window(index)
        assert "outside the arena's 2 shard window(s)" in str(excinfo.value)


def test_an_arena_that_does_not_divide_into_shards_is_refused():
    with pytest.raises(SnapshotArenaCorrupt) as excinfo:
        SnapshotArena(
            storage_handle=1,
            pseudo_sid=-5,
            page_ids_by_group={0: (10, 11, 12)},
            pages_per_group=3,
            capacity_bytes=3 * HOST_PAGE_BYTES,
            payload_bytes=0,
            shard_count=2,
        )
    assert "share a Host page with its neighbour" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Slice 1c: the G>1 transaction -- reserve, per-rank shard write, finalize
# ---------------------------------------------------------------------------


def _group(shard_count=4, *, allocator=None, profile=None):
    """One leader plus G-1 followers over ONE shared arena allocator.

    The shared allocator stands in for the node-shared Host KV region: every
    rank of a serve group maps the same pages, which is the whole reason a
    follower can write its own window without allocating anything.
    """

    profile = profile or _profile()
    allocator = allocator or _ArenaAllocator()
    ranks = []
    for shard_index in range(int(shard_count)):
        ranks.append(
            TrajectoryFrontierCommitter(
                rank=shard_index,
                page_tokens=PAGE_TOKENS,
                host_page_bytes_all_layers=HOST_PAGE_BYTES,
                arena_allocator=allocator,
                snapshot_reader=_SnapshotReader(profile),
                shard_count=int(shard_count),
                shard_index=shard_index,
                owns_registry=(shard_index == 0),
            )
        )
    return ranks, allocator


def _group_commit(ranks, allocator, *, tokens=tuple(range(8)), sequence_id=7):
    """The real pass order: reserve in config, every rank writes, finalize."""

    leader = ranks[0]
    arena = leader.reserve_arena(sequence_id=sequence_id, prefix_page_count=2)
    assert arena is not None
    # The arena page ids ride the host-KV mirror; every follower adopts them.
    for follower in ranks[1:]:
        follower.adopt_arena(sequence_id=sequence_id, arena=arena)
    outcomes = [
        leader.commit(
            namespace_digest=NAMESPACE,
            frontier_token_ids=tokens,
            page_ids_by_group={0: [1, 2]},
            owning_group_id=0,
            sequence_id=sequence_id,
            prompt_token_count=len(tokens),
        )
    ]
    for follower in ranks[1:]:
        outcomes.append(
            follower.write_shard(
                sequence_id=sequence_id,
                frontier_token_ids=tokens,
                prompt_token_count=len(tokens),
            )
        )
    return arena, outcomes


def test_the_leader_allocates_one_g_wide_arena():
    ranks, allocator = _group(shard_count=4)
    leader = ranks[0]
    one_shard = leader.layout.shard_page_count(
        host_page_bytes_all_layers=HOST_PAGE_BYTES, prefix_page_count=2
    )
    arena = leader.reserve_arena(sequence_id=7, prefix_page_count=2)

    assert arena is not None
    assert arena.shard_count == 4
    assert arena.pages_per_group == 4 * one_shard
    assert arena.pages_per_shard == one_shard
    assert len(allocator.allocations) == 1
    # Allocated, not yet retained: the pages left the free pool (the R1
    # invariant) but the pseudo row still owns them so retain can find them.
    assert allocator.retains == []
    assert leader.reserved_sequence_ids() == (7,)


def test_a_g_shard_commit_leaves_the_record_pending_until_finalize():
    ranks, allocator = _group(shard_count=4)
    leader = ranks[0]
    arena, outcomes = _group_commit(ranks, allocator)

    assert outcomes[0] is not None
    assert outcomes[0].installed is False
    assert outcomes[0].record is None
    # Nothing is installed and nothing is retained before the barrier.
    assert len(leader.registry) == 0
    assert allocator.retains == []
    assert leader.pending_sequence_ids() == (7,)

    finalized = leader.finalize()

    assert len(finalized.installed) == 1
    assert finalized.failed_sequence_ids == ()
    assert finalized.rolled_back_sequence_ids == ()
    assert len(leader.registry) == 1
    assert len(allocator.retains) == 1
    record = finalized.installed[0].record
    assert record.shard_count == 4
    assert record.frontier_boundary_tokens == 8


def test_every_rank_writes_only_its_own_window():
    ranks, allocator = _group(shard_count=4)
    arena, _outcomes = _group_commit(ranks, allocator)

    assert len(allocator.shard_writes) == 4
    indices = [write[1] for write in allocator.shard_writes]
    assert sorted(indices) == [0, 1, 2, 3]
    written_pages = set()
    for _sid, shard_index, groups, _payload in allocator.shard_writes:
        expected = arena.shard_window(shard_index)
        assert dict(groups) == {
            int(group_id): tuple(pages)
            for group_id, pages in expected.items()
        }
        for _group_id, pages in groups:
            assert not written_pages & set(pages)
            written_pages |= set(pages)
    # Together the four windows cover the arena exactly once.
    assert written_pages == set(arena.page_ids_by_group[0])


def test_only_the_leader_allocates_and_retains():
    ranks, allocator = _group(shard_count=4)
    _arena, _outcomes = _group_commit(ranks, allocator)
    ranks[0].finalize()
    for follower in ranks[1:]:
        follower.finalize()

    # One allocation and one retain for the whole group, both the leader's.
    assert len(allocator.allocations) == 1
    assert len(allocator.retains) == 1
    assert allocator.rollbacks == []
    # Followers keep no registry at all, so no record of theirs can go stale
    # when the leader later evicts the arena.
    for follower in ranks[1:]:
        assert len(follower.registry) == 0


def test_a_follower_may_not_allocate_and_a_leader_may_not_adopt():
    ranks, allocator = _group(shard_count=2)
    leader, follower = ranks
    with pytest.raises(SnapshotArenaCorrupt) as excinfo:
        follower.reserve_arena(sequence_id=7, prefix_page_count=2)
    assert "does not own its group's frontier registry" in str(excinfo.value)

    arena = leader.reserve_arena(sequence_id=7, prefix_page_count=2)
    with pytest.raises(SnapshotArenaCorrupt) as excinfo:
        leader.adopt_arena(sequence_id=7, arena=arena)
    assert "allocates its arenas rather than adopting one" in str(excinfo.value)


def test_a_follower_cannot_install_a_record_and_a_leader_cannot_write_only():
    ranks, allocator = _group(shard_count=2)
    leader, follower = ranks
    arena = leader.reserve_arena(sequence_id=7, prefix_page_count=2)
    follower.adopt_arena(sequence_id=7, arena=arena)

    with pytest.raises(SnapshotArenaCorrupt) as excinfo:
        follower.commit(
            namespace_digest=NAMESPACE,
            frontier_token_ids=range(8),
            page_ids_by_group={0: [1, 2]},
            owning_group_id=0,
            sequence_id=7,
            prompt_token_count=8,
        )
    assert "installs nothing" in str(excinfo.value)

    with pytest.raises(SnapshotArenaCorrupt) as excinfo:
        leader.write_shard(
            sequence_id=7, frontier_token_ids=range(8), prompt_token_count=8
        )
    assert "not write_shard" in str(excinfo.value)


def test_a_shard_write_failure_installs_no_record_and_gives_the_arena_back():
    """One rank's write fails -> no record, arena rolled back, old one intact."""

    ranks, allocator = _group(shard_count=4)
    leader = ranks[0]
    # An already committed frontier that must survive the failed pass.
    first_allocator = allocator
    older = leader.commit(
        namespace_digest=NAMESPACE,
        frontier_token_ids=range(4),
        page_ids_by_group={0: [1]},
        owning_group_id=0,
        sequence_id=7,
        prompt_token_count=4,
    )
    assert older is None  # G>1 needs a reserved arena; none was reserved
    assert leader.skips()["arena_missing"] == 1

    arena = leader.reserve_arena(sequence_id=7, prefix_page_count=2)
    for follower in ranks[1:]:
        follower.adopt_arena(sequence_id=7, arena=arena)
    allocator.shard_write_error = RuntimeError("shard write exploded")
    with pytest.raises(RuntimeError, match="shard write exploded"):
        leader.commit(
            namespace_digest=NAMESPACE,
            frontier_token_ids=range(8),
            page_ids_by_group={0: [1, 2]},
            owning_group_id=0,
            sequence_id=7,
            prompt_token_count=8,
        )

    # Nothing pending, nothing installed, nothing retained.
    assert leader.pending_sequence_ids() == ()
    assert len(leader.registry) == 0
    assert allocator.retains == []
    # The arena was consumed out of the reservation by the failed commit, so
    # the committer can no longer give it back: that is why the WORKER keeps
    # the arena it published and rolls it back on any raise. Asserted here so
    # the contract stays visible.
    assert leader.reserved_sequence_ids() == ()


def test_finalize_gives_back_an_arena_no_commit_consumed():
    """A reserved arena whose commit never happened must not keep its pages.

    The misaligned skip, the coordinator no longer holding the prefix, an
    already-committed boundary: every one of them leaves a pre-allocated arena
    behind, and finalize is where those pages go home.
    """

    ranks, allocator = _group(shard_count=4)
    leader = ranks[0]
    arena = leader.reserve_arena(sequence_id=7, prefix_page_count=2)
    assert arena is not None
    # No commit for it this pass at all.
    finalized = leader.finalize()

    assert finalized.installed == ()
    assert finalized.rolled_back_sequence_ids == (7,)
    assert allocator.rollbacks == [arena.pseudo_sid]
    assert allocator.retains == []
    assert len(leader.registry) == 0
    assert leader.reserved_sequence_ids() == ()


def test_a_misaligned_boundary_at_g_gt_1_rolls_its_arena_back():
    ranks, allocator = _group(shard_count=4)
    leader = ranks[0]
    arena = leader.reserve_arena(sequence_id=7, prefix_page_count=2)
    outcome = leader.commit(
        namespace_digest=NAMESPACE,
        frontier_token_ids=range(8),
        page_ids_by_group={0: [1, 2]},
        owning_group_id=0,
        sequence_id=7,
        prompt_token_count=11,
    )

    assert outcome is None
    assert leader.skips()["boundary_misaligned"] == 1
    finalized = leader.finalize()
    assert finalized.rolled_back_sequence_ids == (7,)
    assert allocator.rollbacks == [arena.pseudo_sid]
    assert len(leader.registry) == 0


def test_a_failed_finalize_leaves_the_previous_record_installed():
    ranks, allocator = _group(shard_count=2)
    leader = ranks[0]
    arena_one, _ = _group_commit(ranks, allocator, tokens=tuple(range(8)))
    first = leader.finalize().installed[0]
    assert len(leader.registry) == 1

    # A second pass whose retain explodes.
    arena_two, _ = _group_commit(ranks, allocator, tokens=tuple(range(12)))

    def _boom(arena):
        raise RuntimeError("retain exploded")

    allocator.retain = _boom
    finalized = leader.finalize()

    assert finalized.installed == ()
    assert finalized.failed_sequence_ids == (7,)
    # The old frontier is still installed, still referenced, still unreleased.
    assert len(leader.registry) == 1
    record = leader.registry.require(first.session_key)
    assert record is first.record
    assert record.frontier_boundary_tokens == 8
    assert record.is_released is False
    assert allocator.releases == []


def test_the_charge_counts_every_shard_of_the_snapshot():
    ranks, allocator = _group(shard_count=4)
    leader = ranks[0]
    _group_commit(ranks, allocator)
    record = leader.finalize().installed[0].record

    charge = record.charge()
    one_shard = charge.snapshot_host_pages
    assert charge.shard_count == 4
    assert charge.arena_host_pages == 4 * one_shard
    assert charge.owned_host_pages == charge.page_count + 4 * one_shard
    assert charge.snapshot_total_bytes == 4 * charge.snapshot_bytes
    # The arena the leader actually allocated matches the charge exactly: the
    # planner cannot double-promise bytes the snapshot is holding.
    assert leader.plan_arena_pages(prefix_page_count=charge.page_count) == (
        charge.arena_host_pages
    )


def test_a_committer_refuses_an_impossible_shard_identity():
    for bad in ({"shard_count": 0}, {"shard_index": 4, "shard_count": 4}):
        with pytest.raises(SnapshotArenaCorrupt):
            _committer(**bad)
    with pytest.raises(SnapshotArenaCorrupt) as excinfo:
        _committer(shard_count=1, owns_registry=False)
    assert "must own its registry" in str(excinfo.value)


def test_a_reservation_can_be_given_back_outside_finalize():
    """A pass that dies before the barrier must not keep its arena's pages.

    The teardown seam: without this the pages stay out of the free pool with no
    record naming them, AND the next pass's reservation for that sequence is
    refused as a double reservation.
    """

    ranks, allocator = _group(shard_count=4)
    leader = ranks[0]
    arena = leader.reserve_arena(sequence_id=7, prefix_page_count=2)

    assert leader.rollback_reservation(sequence_id=7) is True
    assert allocator.rollbacks == [arena.pseudo_sid]
    assert leader.reserved_sequence_ids() == ()
    # Idempotent: a second call has nothing to give back and says so.
    assert leader.rollback_reservation(sequence_id=7) is False
    assert allocator.rollbacks == [arena.pseudo_sid]
    # And the sequence can be reserved again.
    assert leader.reserve_arena(sequence_id=7, prefix_page_count=2) is not None


def test_a_follower_giving_back_an_adopted_arena_frees_nothing():
    """The leader owns the pages; a follower may only drop its own view."""

    ranks, allocator = _group(shard_count=2)
    leader, follower = ranks
    arena = leader.reserve_arena(sequence_id=7, prefix_page_count=2)
    follower.adopt_arena(sequence_id=7, arena=arena)

    assert follower.rollback_reservation(sequence_id=7) is True
    assert allocator.rollbacks == []
    assert follower.reserved_sequence_ids() == ()
    # The leader's reservation is untouched.
    assert leader.reserved_sequence_ids() == (7,)


def test_a_double_reservation_is_refused_by_name():
    ranks, _allocator = _group(shard_count=2)
    leader = ranks[0]
    leader.reserve_arena(sequence_id=7, prefix_page_count=2)
    with pytest.raises(SnapshotArenaCorrupt) as excinfo:
        leader.reserve_arena(sequence_id=7, prefix_page_count=2)
    assert "would leak the first one's Host pages" in str(excinfo.value)


def test_pool_pressure_at_reservation_time_is_named_and_reserves_nothing():
    allocator = _ArenaAllocator(fail_times=1)
    ranks, allocator = _group(shard_count=4, allocator=allocator)
    leader = ranks[0]
    assert leader.reserve_arena(sequence_id=7, prefix_page_count=2) is None
    assert leader.skips()["arena_unavailable"] == 1
    assert leader.reserved_sequence_ids() == ()
    assert allocator.rollbacks == []
    # Finalize has nothing to install and nothing to give back.
    finalized = leader.finalize()
    assert finalized.installed == ()
    assert finalized.rolled_back_sequence_ids == ()


def test_a_follower_adopting_a_foreign_shard_count_is_refused():
    ranks, _allocator = _group(shard_count=4)
    leader, follower = ranks[0], ranks[1]
    arena = leader.reserve_arena(sequence_id=7, prefix_page_count=2)
    two_shard = replace(arena, shard_count=2)
    with pytest.raises(SnapshotArenaCorrupt) as excinfo:
        follower.adopt_arena(sequence_id=7, arena=two_shard)
    assert "2 shard window(s) but this rank serves a 4-rank group" in str(
        excinfo.value
    )
