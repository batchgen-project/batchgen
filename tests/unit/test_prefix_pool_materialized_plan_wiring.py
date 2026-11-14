"""M3 Slice 3: how the worker turns a frontier hit into a plan input.

The planner itself is covered in ``test_prefix_pool_materialized_segment``;
here the subject is the worker seam around it:

* every rank of a serve group derives the SAME hit set, from replicated data
  (the mirrored session index) and never from the registry only its leader
  holds;
* the leader's live records VETO a stale mirrored set, and that veto rides the
  budget reduce the planner already does, so the whole group falls back
  together;
* a hit the pool cannot materialize is skipped BY NAME, identically on every
  rank;
* a hit's prefix state ends up with ``compute_cached == frontier boundary``
  (the identity the KDA restore refuses to run without), its frontier pages
  are bound as the load source and its prefix-cache attachment -- what keeps
  those pages resident -- is NOT released for the wave;
* a frontier too large for the pool falls back to a recompute by name.

The worker methods are the REAL ones, compiled out of ``batchgen_worker.py``
onto a fake that holds only what they read, which is the pattern
``test_k3_prefix_pool_group`` established for the pool planner; its stubs are
reused here rather than copied.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.worker.worker_fakes import (
    attach_identity,
    attach_prefix_state,
    attach_trajectory_state,
)
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_k3_prefix_pool_group import (  # noqa: E402
    GEOMETRY,
    K3_LAYOUT,
    K3_PAGE_BYTES,
    PAGE_TOKENS,
    FakeDist,
    RecordingCoordinator,
    _bind,
    _fake_torch,
    _stub_heavy_modules,
    _worker_methods,
)

NAMESPACE = (11, 22)
# One continuing trajectory (a 3-block frontier plus a suffix) beside one
# ordinary prompt long enough to share nothing.
FRONTIER = list(range(500, 500 + 3 * PAGE_TOKENS))
HIT_PROMPT = FRONTIER + [901, 902, 903]
# Two ordinary prompts sharing two blocks, so the wave has a computed segment
# as well: a plan that drops its materialization is still a pooled plan.
PLAIN_SHARED = list(range(700, 700 + 2 * PAGE_TOKENS))
PLAIN_PROMPT = PLAIN_SHARED + [1, 2, 3]
PLAIN_PROMPT_2 = PLAIN_SHARED + [4, 5, 6]
UUIDS = ["hit", "plain", "plain2"]
GLOBAL_IDS = [100, 101, 102]
PROMPTS = [HIT_PROMPT, PLAIN_PROMPT, PLAIN_PROMPT_2]
BOUNDARY = 3 * PAGE_TOKENS
FRONTIER_PAGES = {0: [71, 72, 73]}

METHODS = (
    "_plan_prefill_pool_wave",
    "_materialized_prefix_hits",
    "_unmaterializable_frontier",
    "_trajectory_commit_requested",
    "_owns_host_kv",
    "_check_pool_plan_identical",
    "_prefill_pool_layout",
    "_prefill_pool_psm_layout",
    "_prefill_pool_page_dims",
    "_prefill_pool_group",
    "_prefill_pool_free_state_slots",
    "_replace_prefix_lookup_hits",
    "_settle_trajectory_restore_plan",
    "_restore_trajectory_frontiers",
    "_restore_mirrored_trajectory_shards",
    "_get_trajectory_restorer",
    "_trajectory_restore_payload",
    "_trajectory_group_geometry",
    "_decode_attn_tp_size",
    "_prefill_pool_page_bytes",
    "_pool_pad_poison_enabled",
    "_new_pool_kernel_dump_recorder",
    "_active_batchgen_debug_for_sequences",
    "_debug_flag_enabled",
)


# --------------------------------------------------------------------------- #
#  Fakes
# --------------------------------------------------------------------------- #


class _Record:
    def __init__(self, pages=None):
        self.page_ids_by_group = FRONTIER_PAGES if pages is None else pages


class _Attachment:
    """What ``TrajectoryRestorer.lookup`` handed the worker for a hit."""

    def __init__(
        self, *, boundary=BOUNDARY, group_id=0, pages=None, sequence_id=GLOBAL_IDS[0]
    ):
        self.boundary_tokens = boundary
        self.lcp_tokens = boundary
        self.owning_group_id = group_id
        self.record = _Record(pages)
        self.session_key = "s1"
        self.sequence_id = sequence_id
        self.storage_handle = 5
        self.consumed = False


class _RecordingRestorer:
    """The restorer surface the plan and the restore seam call.

    Consuming an attachment twice is the leak/double-release the plan-time
    decline must never commit, so it fails the test here.
    """

    def __init__(self):
        self.declines = []
        self.restores = []

    def _consume(self, attachment):
        assert not attachment.consumed, "an attachment was consumed twice"
        attachment.consumed = True

    def decline(self, attachment, *, reason, detail=""):
        self._consume(attachment)
        self.declines.append((attachment.sequence_id, reason, detail))

    def restore(self, attachment, *, serving_group_id, compute_cached_tokens):
        self._consume(attachment)
        self.restores.append((attachment.sequence_id, compute_cached_tokens))


class _Index:
    """The mirrored session->group index, as the folded map behaves."""

    def __init__(self, entries):
        # prompt prefix tuple -> entry
        self._entries = entries
        self.resolved = 0

    def resolve(self, *, namespace_digest, prompt_token_ids):
        self.resolved += 1
        assert tuple(namespace_digest) == NAMESPACE
        for boundary, entry in self._entries:
            if tuple(prompt_token_ids[:boundary]) == tuple(FRONTIER[:boundary]):
                return entry
        return None


def _entry(*, boundary=BOUNDARY, group_id=0):
    return SimpleNamespace(
        session_key="s1",
        owning_group_id=group_id,
        boundary_tokens=boundary,
        frontier_index_key="k",
    )


class _Lookup:
    """A prefix-cache lookup result carrying a live attachment handle."""

    def __init__(self, handle):
        self.attachment_handle = handle


def _state(handle, attached):
    from batchgen.prefix_reuse.prefill import PrefixCacheSequenceState

    return PrefixCacheSequenceState(
        lookup_result=_Lookup(handle),
        attached_tokens=attached,
        compute_cached_tokens=attached,
    )


def _seqs(*, debug="on", group=0, prompts=PROMPTS):
    return {
        uuid: SimpleNamespace(
            uuid=uuid,
            global_idx=global_id,
            assigned_rank=0,
            decode_dp_group=group,
            prompt_length=len(prompt),
            input_ids=torch.tensor([prompt], dtype=torch.long),
            batchgen_debug=(
                None if debug is None else {"prefix_trajectory_commit": debug}
            ),
        )
        for uuid, global_id, prompt in zip(UUIDS, GLOBAL_IDS, prompts)
    }


def _worker(
    *,
    free_pages=40,
    fake_dist=None,
    attn_tp_size=2,
    rank=0,
    attachments=None,
    index=None,
    states=None,
    seqs=None,
    owns_host_kv=True,
    token_cap=64,
    restorer=None,
):
    methods = _worker_methods(
        METHODS,
        {
            "torch": _fake_torch(free_pages * K3_PAGE_BYTES),
            "dist": fake_dist or FakeDist(peer_budget=None),
            "DualHostKVCoordinator": type("DualHostKVCoordinator", (), {}),
        },
    )
    parallel_manager = SimpleNamespace(
        attn_tp_size=attn_tp_size,
        attn_tp_group="group" if attn_tp_size > 1 else None,
        prefill_pool_layout=lambda: dict(K3_LAYOUT),
        supports_group_pool_prefill=lambda: True,
    )
    sequences = _seqs() if seqs is None else seqs
    worker = SimpleNamespace(
        torch_device="cpu",
        huggingface_ckpt_name="moonshotai/Kimi-K3",
        parallel_manager=parallel_manager,
        engine_config=SimpleNamespace(Module_Batching_Config=SimpleNamespace(
            prefill_micro_batch_token_cap=token_cap,
            prefill_prefix_pool_workspace_bytes=0,
        )),
        prefix_cache_runtime_config=SimpleNamespace(
            group_specs=(SimpleNamespace(raw_page_tokens=PAGE_TOKENS),),
            namespace_digest=NAMESPACE,
        ),
        host_paged_kv_worker_view=object(),
        global_batch=SimpleNamespace(get_sequence=sequences.get),
        prefix_cache_coordinator=RecordingCoordinator(),
        _prefill_pool_base=(0, 0, 0),
        _prefill_pool_page_budget=free_pages,
        _batchgen_debug=None,
        _any_sequence_batchgen_debug=True,
    )
    attach_identity(
        worker,
        rank=rank, world_size=max(1, attn_tp_size), local_world_size=8,
        local_rank=0, decode_group_size=max(1, attn_tp_size),
        bound=UUIDS,
    )
    attach_prefix_state(worker, prefix_sequence_states=dict(states or {}))
    attach_trajectory_state(
        worker,
        trajectory_attachments=dict(attachments or {}),
        trajectory_session_groups=index,
        trajectory_restorer=_RecordingRestorer() if restorer is None else restorer,
        # The leader's handle->arena table, read by the mirrored payload.
        trajectory_frontier_committer=SimpleNamespace(
            arena_for=lambda handle: SimpleNamespace(
                shard_count=attn_tp_size,
                pseudo_sid=-3,
                pages_per_group=1,
                capacity_bytes=64,
                page_ids_by_group={0: (41,)},
            ),
        ),
        trajectory_mirrored_restores={},
    )
    # Bound-method references build_restorer takes (never called when the
    # store already holds a restorer, as these fakes always do).
    worker._host_views_by_group = lambda: {}
    worker._next_pool_segment_sid = lambda: -1
    # ``_owns_host_kv`` is the group-leader predicate; a follower answers no.
    worker._owns_host_kv = lambda seq: bool(owns_host_kv)
    # The prompt reader the worker shares with the trajectory book.
    worker._trajectory_tokens = (
        lambda seq, length=None, **_: seq.input_ids[0, :length]
    )
    return _bind(worker, methods), sequences


def _hit_states():
    """The prefix lookup that preceded the plan, hit attached past the frontier."""
    return {
        GLOBAL_IDS[0]: _state(handle=77, attached=BOUNDARY),
        GLOBAL_IDS[1]: _state(handle=0, attached=0),
        GLOBAL_IDS[2]: _state(handle=0, attached=0),
    }


def _plan(worker, states=None):
    """Plan with the prefix states the admission lookup produced."""

    return worker._plan_prefill_pool_wave(
        UUIDS, dict(worker._prefix.prefix_sequence_states if states is None else states)
    )


# --------------------------------------------------------------------------- #
#  1. Flag off
# --------------------------------------------------------------------------- #


def test_the_flag_being_off_derives_no_hit_and_votes_yes(monkeypatch):
    _stub_heavy_modules(monkeypatch, free_state_slots=32)
    index = _Index([(BOUNDARY, _entry())])
    worker, _ = _worker(
        seqs=_seqs(debug="off"),
        attachments={GLOBAL_IDS[0]: _Attachment()},
        index=index,
        states=_hit_states(),
    )
    hits, confirmed, pages, _ = worker._materialized_prefix_hits(
        [worker.global_batch.get_sequence(uuid) for uuid in UUIDS],
        page_tokens=PAGE_TOKENS,
        group="group",
    )
    assert (hits, confirmed, pages) == ([], True, {})
    assert index.resolved == 0  # the index is not even consulted


def test_a_flag_off_wave_plans_exactly_as_before(monkeypatch):
    _stub_heavy_modules(monkeypatch, free_state_slots=32)
    fake_dist = FakeDist(peer_budget=[40, 32])
    worker, _ = _worker(seqs=_seqs(debug="off"), fake_dist=fake_dist)
    wave = _plan(worker)
    assert wave is None or wave.materialized == frozenset()
    assert fake_dist.reduced == [[40, 32, 1]]


# --------------------------------------------------------------------------- #
#  2. Where the hit set comes from
# --------------------------------------------------------------------------- #


def test_at_g_gt_1_the_mirrored_index_decides_and_the_leader_confirms(monkeypatch):
    _stub_heavy_modules(monkeypatch, free_state_slots=32)
    index = _Index([(BOUNDARY, _entry())])
    worker, seqs = _worker(
        attachments={GLOBAL_IDS[0]: _Attachment()},
        index=index,
        states=_hit_states(),
    )
    hits, confirmed, pages, _ = worker._materialized_prefix_hits(
        [seqs[uuid] for uuid in UUIDS], page_tokens=PAGE_TOKENS, group="group"
    )
    assert [(hit.index, hit.tokens) for hit in hits] == [(0, BOUNDARY)]
    assert confirmed is True
    assert pages == {0: FRONTIER_PAGES}
    assert index.resolved == len(UUIDS)


def test_a_follower_derives_the_same_hit_set_with_no_record_of_its_own(monkeypatch):
    _stub_heavy_modules(monkeypatch, free_state_slots=32)
    worker, seqs = _worker(
        rank=1,
        attachments={},  # a follower holds no frontier record
        index=_Index([(BOUNDARY, _entry())]),
        states={},
        owns_host_kv=False,
    )
    hits, confirmed, pages, _ = worker._materialized_prefix_hits(
        [seqs[uuid] for uuid in UUIDS], page_tokens=PAGE_TOKENS, group="group"
    )
    assert [(hit.index, hit.tokens) for hit in hits] == [(0, BOUNDARY)]
    # No opinion, no pages: the leader mirrors them inside this same pass.
    assert confirmed is True
    assert pages == {}


def test_at_g_1_the_ranks_own_attachment_is_the_hit_set(monkeypatch):
    _stub_heavy_modules(monkeypatch, free_state_slots=32)
    index = _Index([(BOUNDARY, _entry())])
    worker, seqs = _worker(
        attn_tp_size=1,
        attachments={GLOBAL_IDS[0]: _Attachment()},
        index=index,
        states=_hit_states(),
    )
    hits, confirmed, pages, _ = worker._materialized_prefix_hits(
        [seqs[uuid] for uuid in UUIDS], page_tokens=PAGE_TOKENS, group=None
    )
    assert [(hit.index, hit.tokens) for hit in hits] == [(0, BOUNDARY)]
    assert (confirmed, pages) == (True, {0: FRONTIER_PAGES})
    assert index.resolved == 0  # no group, so no mirrored index is needed


# --------------------------------------------------------------------------- #
#  3. The veto
# --------------------------------------------------------------------------- #


def test_a_mirrored_hit_the_leader_cannot_confirm_votes_no(monkeypatch):
    _stub_heavy_modules(monkeypatch, free_state_slots=32)
    worker, seqs = _worker(
        attachments={},  # the leader's record is gone since it published
        index=_Index([(BOUNDARY, _entry())]),
        states=_hit_states(),
    )
    hits, confirmed, pages, _ = worker._materialized_prefix_hits(
        [seqs[uuid] for uuid in UUIDS], page_tokens=PAGE_TOKENS, group="group"
    )
    assert len(hits) == 1  # the SET stays the group's; only the vote differs
    assert confirmed is False
    assert pages == {}


def test_an_advanced_frontier_votes_no(monkeypatch):
    _stub_heavy_modules(monkeypatch, free_state_slots=32)
    worker, seqs = _worker(
        attachments={GLOBAL_IDS[0]: _Attachment(boundary=BOUNDARY + PAGE_TOKENS)},
        index=_Index([(BOUNDARY, _entry())]),
        states=_hit_states(),
    )
    _, confirmed, pages, _ = worker._materialized_prefix_hits(
        [seqs[uuid] for uuid in UUIDS], page_tokens=PAGE_TOKENS, group="group"
    )
    assert confirmed is False and pages == {}


def test_a_record_naming_no_resident_page_votes_no(monkeypatch):
    _stub_heavy_modules(monkeypatch, free_state_slots=32)
    worker, seqs = _worker(
        attachments={GLOBAL_IDS[0]: _Attachment(pages={})},
        index=_Index([(BOUNDARY, _entry())]),
        states=_hit_states(),
    )
    _, confirmed, _, _ = worker._materialized_prefix_hits(
        [seqs[uuid] for uuid in UUIDS], page_tokens=PAGE_TOKENS, group="group"
    )
    assert confirmed is False


def test_an_attachment_short_of_the_frontier_votes_no(monkeypatch):
    """The resident pages are kept resident by THAT attachment."""

    _stub_heavy_modules(monkeypatch, free_state_slots=32)
    states = _hit_states()
    states[GLOBAL_IDS[0]] = _state(handle=77, attached=BOUNDARY - PAGE_TOKENS)
    worker, seqs = _worker(
        attachments={GLOBAL_IDS[0]: _Attachment()},
        index=_Index([(BOUNDARY, _entry())]),
        states=states,
    )
    _, confirmed, pages, _ = worker._materialized_prefix_hits(
        [seqs[uuid] for uuid in UUIDS], page_tokens=PAGE_TOKENS, group="group"
    )
    assert confirmed is False and pages == {}


def test_the_vote_rides_the_budget_reduce_and_the_group_min_decides(monkeypatch):
    _stub_heavy_modules(monkeypatch, free_state_slots=32)
    # This rank confirms; a peer does not.
    fake_dist = FakeDist(peer_budget=[40, 32, 0])
    worker, _ = _worker(
        fake_dist=fake_dist,
        attachments={GLOBAL_IDS[0]: _Attachment()},
        index=_Index([(BOUNDARY, _entry())]),
        states=_hit_states(),
    )
    wave = _plan(worker)
    assert fake_dist.reduced == [[40, 32, 1]], "this rank votes yes"
    assert wave is not None
    assert wave.materialized == frozenset(), "the peer's no takes it back"
    assert wave.plan.materialized_tokens == 0
    # And this rank's live attachment is declined by name at plan time.
    assert worker._traj.trajectory_restorer.declines == [(
        GLOBAL_IDS[0], "not_planned",
        "at plan time: the serve group could not confirm the mirrored "
        "frontier set",
    )]
    assert worker._traj.trajectory_attachments == {}


def test_the_leaders_no_reaches_the_reduce(monkeypatch):
    _stub_heavy_modules(monkeypatch, free_state_slots=32)
    fake_dist = FakeDist(peer_budget=[40, 32, 1])
    worker, _ = _worker(
        fake_dist=fake_dist,
        attachments={},  # stale mirror
        index=_Index([(BOUNDARY, _entry())]),
        states=_hit_states(),
    )
    wave = _plan(worker)
    assert fake_dist.reduced == [[40, 32, 0]]
    assert wave is None or wave.materialized == frozenset()


# --------------------------------------------------------------------------- #
#  4. Hits the pool cannot materialize
# --------------------------------------------------------------------------- #


def test_a_boundary_off_the_page_grid_is_skipped_by_name(monkeypatch, caplog):
    _stub_heavy_modules(monkeypatch, free_state_slots=32)
    worker, seqs = _worker(
        attachments={GLOBAL_IDS[0]: _Attachment(boundary=BOUNDARY - 1)},
        index=_Index([(BOUNDARY - 1, _entry(boundary=BOUNDARY - 1))]),
        states=_hit_states(),
    )
    with caplog.at_level("INFO"):
        hits, confirmed, _, drops = worker._materialized_prefix_hits(
            [seqs[uuid] for uuid in UUIDS], page_tokens=PAGE_TOKENS, group="group"
        )
    assert hits == [] and confirmed is True
    assert "boundary_off_the_page_grid" in caplog.text
    assert drops[0] == "the pool cannot materialize it (boundary_off_the_page_grid)"


def test_a_frontier_with_no_private_token_left_is_skipped_by_name(monkeypatch, caplog):
    _stub_heavy_modules(monkeypatch, free_state_slots=32)
    # The prompt IS the frontier: nothing would be left to produce a logit.
    prompts = [FRONTIER, PLAIN_PROMPT, PLAIN_PROMPT_2]
    seqs = _seqs(prompts=prompts)
    worker, _ = _worker(
        seqs=seqs,
        attachments={GLOBAL_IDS[0]: _Attachment()},
        index=_Index([(BOUNDARY, _entry())]),
        states=_hit_states(),
    )
    with caplog.at_level("INFO"):
        hits, confirmed, _, drops = worker._materialized_prefix_hits(
            [seqs[uuid] for uuid in UUIDS], page_tokens=PAGE_TOKENS, group="group"
        )
    assert hits == [] and confirmed is True
    assert "no_private_token_left" in caplog.text
    assert drops[0] == "the pool cannot materialize it (no_private_token_left)"


def test_a_hit_serving_off_its_owning_group_is_skipped_by_name(monkeypatch, caplog):
    _stub_heavy_modules(monkeypatch, free_state_slots=32)
    seqs = _seqs(group=1)  # serving on group 1, snapshot sharded for group 0
    worker, _ = _worker(
        seqs=seqs,
        attachments={GLOBAL_IDS[0]: _Attachment(group_id=0)},
        index=_Index([(BOUNDARY, _entry(group_id=0))]),
        states=_hit_states(),
    )
    with caplog.at_level("INFO"):
        hits, confirmed, _, drops = worker._materialized_prefix_hits(
            [seqs[uuid] for uuid in UUIDS], page_tokens=PAGE_TOKENS, group="group"
        )
    assert hits == [] and confirmed is True
    assert "serving_group_is_not_the_owning_group" in caplog.text
    assert drops[0] == (
        "the pool cannot materialize it (serving_group_is_not_the_owning_group)"
    )


@pytest.mark.parametrize(
    "boundary,reason",
    [
        (0, "boundary_off_the_page_grid"),
        (PAGE_TOKENS + 1, "boundary_off_the_page_grid"),
        (4 * PAGE_TOKENS, "no_private_token_left"),  # past a 15-token prompt
    ],
)
def test_the_reason_predicate_names_each_refusal(monkeypatch, boundary, reason):
    _stub_heavy_modules(monkeypatch, free_state_slots=32)
    worker, seqs = _worker()
    assert worker._unmaterializable_frontier(
        seqs[UUIDS[0]], boundary=boundary, owning_group_id=0,
        page_tokens=PAGE_TOKENS,
    ) == reason
    assert worker._unmaterializable_frontier(
        seqs[UUIDS[0]], boundary=BOUNDARY, owning_group_id=0,
        page_tokens=PAGE_TOKENS,
    ) is None


# --------------------------------------------------------------------------- #
#  5. The planned wave
# --------------------------------------------------------------------------- #


def _materialized_wave(monkeypatch, **kwargs):
    _stub_heavy_modules(monkeypatch, free_state_slots=32)
    worker, seqs = _worker(
        fake_dist=FakeDist(peer_budget=[40, 32, 1]),
        attachments={GLOBAL_IDS[0]: _Attachment()},
        index=_Index([(BOUNDARY, _entry())]),
        states=_hit_states(),
        **kwargs,
    )
    states = dict(worker._prefix.prefix_sequence_states)
    wave = worker._plan_prefill_pool_wave(UUIDS, states)
    return worker, seqs, wave, states


def test_the_hit_is_planned_as_a_materialized_segment(monkeypatch):
    _, _, wave, _ = _materialized_wave(monkeypatch)
    assert wave is not None
    assert wave.materialized == frozenset({0})
    seg = wave.plan.segments[0]
    assert (seg.token_start, seg.token_end, seg.members) == (0, BOUNDARY, (0,))
    assert wave.plan.materialized_tokens == BOUNDARY


def test_the_frontiers_resident_pages_are_bound_as_the_load_source(monkeypatch):
    _, _, wave, _ = _materialized_wave(monkeypatch)
    assert wave.segment_host_pages[0] == {0: [71, 72, 73]}


def test_the_hits_prefix_state_starts_at_the_frontier_boundary(monkeypatch):
    """The identity the KDA restore refuses to run without."""

    worker, _, wave, states = _materialized_wave(monkeypatch)
    state = worker._prefix.prefix_sequence_states[GLOBAL_IDS[0]]
    assert state.compute_cached_tokens == BOUNDARY
    assert state.attached_tokens == BOUNDARY
    assert states[GLOBAL_IDS[0]].compute_cached_tokens == BOUNDARY
    assert wave.chain_tokens(0) == BOUNDARY
    # The plan's restore decision: this row, at exactly its boundary.
    assert state.trajectory_restore_boundary == BOUNDARY
    assert worker._prefix.prefix_sequence_states[GLOBAL_IDS[1]].trajectory_restore_boundary == 0
    # A planned attachment is NOT declined; it stays for the restore seam.
    assert worker._traj.trajectory_restorer.declines == []
    assert set(worker._traj.trajectory_attachments) == {GLOBAL_IDS[0]}


def test_the_attachment_holding_the_frontier_resident_is_kept_for_the_wave(monkeypatch):
    worker, _, _, _ = _materialized_wave(monkeypatch)
    assert worker.prefix_cache_coordinator.released == [], (
        "releasing it would let the pages the load reads be evicted mid-wave"
    )
    state = worker._prefix.prefix_sequence_states[GLOBAL_IDS[0]]
    # Carried where the one release lane every sequence leaves through
    # frees it exactly once.
    assert state.commit_attachment_handles == (77,)


def test_a_non_hit_still_releases_its_lookup(monkeypatch):
    _stub_heavy_modules(monkeypatch, free_state_slots=32)
    states = _hit_states()
    states[GLOBAL_IDS[1]] = _state(handle=88, attached=PAGE_TOKENS)
    worker, _ = _worker(
        fake_dist=FakeDist(peer_budget=[40, 32, 1]),
        attachments={GLOBAL_IDS[0]: _Attachment()},
        index=_Index([(BOUNDARY, _entry())]),
        states=states,
    )
    worker._plan_prefill_pool_wave(UUIDS, dict(states))
    assert worker.prefix_cache_coordinator.released == [88]
    # migrated: worker._prefix.prefix_sequence_states -> worker._prefix.prefix_sequence_states, same owner object.
    assert worker._prefix.prefix_sequence_states[GLOBAL_IDS[1]].commit_attachment_handles == ()


def test_a_loaded_segment_asks_for_no_recurrent_state_slot(monkeypatch):
    """Its tokens run no row, so they hold no slot; only rows do."""

    # Exactly one slot per prompt plus one per COMPUTED segment. If the
    # loaded segment asked for a slot too, this wave would decline.
    _stub_heavy_modules(monkeypatch, free_state_slots=len(PROMPTS) + 1)
    worker, _ = _worker(
        fake_dist=FakeDist(peer_budget=[40, len(PROMPTS) + 1, 1]),
        attachments={GLOBAL_IDS[0]: _Attachment()},
        index=_Index([(BOUNDARY, _entry())]),
        states=_hit_states(),
    )
    # Three prompts, one loaded segment and one computed one: four slots.
    wave = worker._plan_prefill_pool_wave(
        UUIDS, dict(worker._prefix.prefix_sequence_states)
    )
    assert wave is not None and wave.materialized == frozenset({0})


def test_a_dropped_hit_releases_its_attachment_exactly_once(monkeypatch):
    """Dropped at plan time, so nothing reads those pages: the hold goes."""

    _stub_heavy_modules(monkeypatch, free_state_slots=32)
    worker, _ = _worker(
        fake_dist=FakeDist(peer_budget=[40, 32, 0]),  # a peer vetoes
        attachments={GLOBAL_IDS[0]: _Attachment()},
        index=_Index([(BOUNDARY, _entry())]),
        states=_hit_states(),
    )
    wave = _plan(worker)
    assert wave is not None and wave.materialized == frozenset()
    assert worker.prefix_cache_coordinator.released == [77]
    state = worker._prefix.prefix_sequence_states[GLOBAL_IDS[0]]
    assert state.commit_attachment_handles == (), "released, not also retained"
    # The sequence recomputes from token 0 and carries no restore: its
    # trajectory attachment was declined when the wave was planned, so it
    # never reaches the restore seam.
    assert state.compute_cached_tokens == 0
    assert state.trajectory_restore_boundary == 0
    assert [reason for _, reason, _ in worker._traj.trajectory_restorer.declines] == [
        "not_planned"
    ]
    assert worker._traj.trajectory_attachments == {}


def test_a_frontier_too_large_for_the_pool_recomputes_by_name(monkeypatch, caplog):
    _stub_heavy_modules(monkeypatch, free_state_slots=32)
    long_frontier = list(range(900, 900 + 8 * PAGE_TOKENS))
    prompts = [long_frontier + [1, 2, 3], PLAIN_PROMPT, PLAIN_PROMPT_2]
    seqs = _seqs(prompts=prompts)
    index = _Index([])

    class _AlwaysHit:
        resolved = 0

        def resolve(self, *, namespace_digest, prompt_token_ids):
            self.resolved += 1
            if tuple(prompt_token_ids[:3]) == tuple(long_frontier[:3]):
                return _entry(boundary=8 * PAGE_TOKENS)
            return None

    del index
    worker, _ = _worker(
        seqs=seqs,
        # Room for the wave's scratch and its computed segment, but not for
        # the eight pages the frontier would take.
        free_pages=17,
        token_cap=32,
        fake_dist=FakeDist(peer_budget=[17, 32, 1]),
        attachments={
            GLOBAL_IDS[0]: _Attachment(boundary=8 * PAGE_TOKENS, pages={
                0: list(range(60, 68))
            })
        },
        index=_AlwaysHit(),
        states={
            GLOBAL_IDS[0]: _state(handle=77, attached=8 * PAGE_TOKENS),
            GLOBAL_IDS[1]: _state(handle=0, attached=0),
            GLOBAL_IDS[2]: _state(handle=0, attached=0),
        },
    )
    with caplog.at_level("WARNING"):
        wave = worker._plan_prefill_pool_wave(
            UUIDS, dict(worker._prefix.prefix_sequence_states)
        )
    assert "declined by the planner" in caplog.text
    assert "pool_materialization_does_not_fit" in caplog.text
    # The row's restore is declined by name at plan time, once.
    assert worker._traj.trajectory_restorer.declines == [(
        GLOBAL_IDS[0], "not_planned",
        "at plan time: the wave planner declined it "
        "(pool_materialization_does_not_fit)",
    )]
    assert worker._traj.trajectory_attachments == {}
    if wave is not None:
        assert wave.materialized == frozenset()
        # The sequence still completes: it is computed in full.
        assert wave.chain_tokens(0) == 0


# --------------------------------------------------------------------------- #
#  6. The plan IS the restore decision
# --------------------------------------------------------------------------- #
#
# A row whose frontier the plan materializes carries
# ``trajectory_restore_boundary`` and starts its compute there; every other
# live attachment is declined at PLAN time with ``not_planned`` and the drop
# reason, exactly once, and never reaches the restore seam. Before this, a
# dropped row kept its attachment, started at its pool chain (e.g. 64) and
# the restore declined it with ``suffix_offset_mismatch`` against its
# boundary (e.g. 5120).

# One block more than FRONTIER, shared with the hit row, for arbitration.
LONGER_SHARE = FRONTIER + [5, 5, 5, 5] + [6]


def _drop_case(name, monkeypatch):
    """A worker whose next plan drops the hit row's restore on path ``name``."""

    slots = 2 if name == "state_slots_short" else 32
    _stub_heavy_modules(monkeypatch, free_state_slots=slots)
    kwargs = dict(
        fake_dist=FakeDist(peer_budget=[40, slots, 1]),
        attachments={GLOBAL_IDS[0]: _Attachment()},
        index=_Index([(BOUNDARY, _entry())]),
        states=_hit_states(),
    )
    if name == "group_veto":
        kwargs["fake_dist"] = FakeDist(peer_budget=[40, 32, 0])
    elif name == "pool_smaller_than_scratch":
        kwargs.update(free_pages=2, fake_dist=FakeDist(peer_budget=[2, 32, 1]))
    elif name == "no_private_token_left":
        kwargs["seqs"] = _seqs(prompts=[FRONTIER, PLAIN_PROMPT, PLAIN_PROMPT_2])
    elif name == "index_does_not_place_it":
        kwargs["index"] = _Index([])
    elif name == "pool_chain_longer_than_frontier":
        # G == 1, so the hit set is the rank's own attachments; the hit row's
        # frontier is two blocks but it shares three with another prompt.
        kwargs.update(
            attn_tp_size=1,
            attachments={GLOBAL_IDS[0]: _Attachment(boundary=2 * PAGE_TOKENS)},
            seqs=_seqs(prompts=[HIT_PROMPT, LONGER_SHARE, PLAIN_PROMPT_2]),
            index=None,
        )
    worker, _ = _worker(**kwargs)
    if name == "budget_unmeasured":
        worker._prefill_pool_page_budget = None
    elif name == "same_chunk_gate":
        worker.parallel_manager.supports_group_pool_prefill = lambda: False
    return worker


DROP_CASES = {
    "group_veto": "the serve group could not confirm the mirrored frontier set",
    "budget_unmeasured": "the prefill-phase pool budget is not yet measured",
    "pool_smaller_than_scratch": "the wave is not pooled: pool 2 pages <= scratch",
    "state_slots_short": "the wave is not pooled:",
    "no_private_token_left": "the pool cannot materialize it (no_private_token_left)",
    "index_does_not_place_it": "the mirrored session index does not place this session",
    "same_chunk_gate": "this model's pool does not read a segment resident in its own chunk",
    "pool_chain_longer_than_frontier": (
        "the wave planner declined it (in_wave_pool_chain_longer_than_frontier)"
    ),
}


@pytest.mark.parametrize("name", sorted(DROP_CASES))
def test_every_drop_path_ends_in_one_named_plan_time_decline(monkeypatch, name):
    worker = _drop_case(name, monkeypatch)
    attachment = worker._traj.trajectory_attachments[GLOBAL_IDS[0]]
    worker._plan_prefill_pool_wave(UUIDS, dict(worker._prefix.prefix_sequence_states))

    restorer = worker._traj.trajectory_restorer
    assert len(restorer.declines) == 1, restorer.declines
    sequence_id, reason, detail = restorer.declines[0]
    assert (sequence_id, reason) == (GLOBAL_IDS[0], "not_planned")
    assert detail.startswith("at plan time: " + DROP_CASES[name]), detail
    # Released exactly once (the recording restorer refuses a second
    # consumption) and gone from the live map ...
    assert attachment.consumed is True
    assert worker._traj.trajectory_attachments == {}
    # ... so the restore seam has nothing to apply for the row.
    worker._restore_trajectory_frontiers([0, 1, 2])
    assert restorer.restores == []


def test_the_planned_row_is_restored_at_exactly_its_boundary(monkeypatch):
    worker, _, wave, _ = _materialized_wave(monkeypatch)
    assert wave is not None and wave.materialized == frozenset({0})
    worker._restore_trajectory_frontiers([0, 1, 2])
    restorer = worker._traj.trajectory_restorer
    assert restorer.restores == [(GLOBAL_IDS[0], BOUNDARY)]
    assert restorer.declines == []
    assert worker._traj.trajectory_attachments == {}


def test_an_unplanned_attachment_at_the_restore_seam_is_a_broken_invariant(monkeypatch):
    """Not restored and not tolerated: declined once, then raised."""

    _stub_heavy_modules(monkeypatch, free_state_slots=32)
    worker, _ = _worker(
        attachments={GLOBAL_IDS[0]: _Attachment()},
        states=_hit_states(),  # a lookup state, never planned
    )
    attachment = worker._traj.trajectory_attachments[GLOBAL_IDS[0]]
    with pytest.raises(RuntimeError, match="never planned"):
        worker._restore_trajectory_frontiers([0])
    restorer = worker._traj.trajectory_restorer
    assert restorer.restores == []
    assert [reason for _, reason, _ in restorer.declines] == ["not_planned"]
    assert attachment.consumed is True


def test_the_mirrored_payload_carries_planned_rows_only(monkeypatch):
    worker, _, wave, _ = _materialized_wave(monkeypatch)
    assert wave is not None
    # A second live attachment the plan never took (a row outside this
    # wave's materialization, e.g. one left from an earlier pass).
    worker._traj.trajectory_attachments[GLOBAL_IDS[1]] = _Attachment(
        sequence_id=GLOBAL_IDS[1]
    )
    payload = worker._trajectory_restore_payload()
    assert sorted(payload) == [GLOBAL_IDS[0]]
    entry = payload[GLOBAL_IDS[0]]
    assert entry["boundary_tokens"] == BOUNDARY
    assert entry["lcp_tokens"] == BOUNDARY  # for the follower's identity check


def test_the_plan_keeps_the_hit_whose_frontier_covers_its_pool_chain(monkeypatch):
    """Arbitration's other side: a frontier at least as long as the chain wins."""

    _stub_heavy_modules(monkeypatch, free_state_slots=32)
    worker, _ = _worker(
        attn_tp_size=1,
        attachments={GLOBAL_IDS[0]: _Attachment()},  # three-block frontier
        seqs=_seqs(prompts=[HIT_PROMPT, LONGER_SHARE, PLAIN_PROMPT_2]),
        states=_hit_states(),
    )
    wave = _plan(worker)
    assert wave is not None and wave.materialized == frozenset({0})
    assert wave.chain_tokens(0) == BOUNDARY
    state = worker._prefix.prefix_sequence_states[GLOBAL_IDS[0]]
    assert state.trajectory_restore_boundary == BOUNDARY
    assert worker._traj.trajectory_restorer.declines == []


# --------------------------------------------------------------------------- #
#  7. A planned row never reaches the forward without its restored state
# --------------------------------------------------------------------------- #
#
# The plan built the wave around the row's start at the boundary, so the seam
# fails loud whenever it cannot hand that row its state: a refusal inside the
# restore propagates, and a planned row with nothing to restore it from (no
# live attachment on the registry rank, no mirrored decision on a follower)
# is raised by name before any forward is launched.


def test_a_planned_rows_restore_failure_propagates_out_of_the_seam(monkeypatch):
    from batchgen.prefix_reuse.trajectory_restore import (
        DECLINE_SLOTS_EXHAUSTED,
        TrajectoryPlannedRestoreFailed,
    )

    worker, _, wave, _ = _materialized_wave(monkeypatch)
    assert wave is not None and wave.materialized == frozenset({0})
    restorer = worker._traj.trajectory_restorer

    def _exhausted(attachment, *, serving_group_id, compute_cached_tokens):
        restorer._consume(attachment)
        raise TrajectoryPlannedRestoreFailed(
            reason=DECLINE_SLOTS_EXHAUSTED,
            detail="no free KDA state slot",
            rank=0,
            sequence_id=attachment.sequence_id,
            session_key=attachment.session_key,
            boundary_tokens=attachment.boundary_tokens,
        )

    restorer.restore = _exhausted
    with pytest.raises(TrajectoryPlannedRestoreFailed) as excinfo:
        worker._restore_trajectory_frontiers([0, 1, 2])
    assert excinfo.value.reason == DECLINE_SLOTS_EXHAUSTED
    assert excinfo.value.sequence_id == GLOBAL_IDS[0]
    assert worker._traj.trajectory_attachments == {}


def test_a_planned_row_whose_attachment_is_gone_is_raised_by_name(monkeypatch):
    from batchgen.prefix_reuse.trajectory_restore import (
        FAIL_NO_RESTORE_SOURCE,
        TrajectoryPlannedRestoreFailed,
    )

    worker, _, wave, _ = _materialized_wave(monkeypatch)
    assert wave is not None and wave.materialized == frozenset({0})
    worker._traj.trajectory_attachments.pop(GLOBAL_IDS[0])
    with pytest.raises(TrajectoryPlannedRestoreFailed) as excinfo:
        worker._restore_trajectory_frontiers([0, 1, 2])
    assert excinfo.value.reason == FAIL_NO_RESTORE_SOURCE
    assert excinfo.value.sequence_id == GLOBAL_IDS[0]
    assert excinfo.value.boundary_tokens == BOUNDARY
    assert worker._traj.trajectory_restorer.restores == []


def test_a_follower_planned_row_without_a_mirrored_decision_is_raised(monkeypatch):
    """The leader's payload skipping a planned row must not go unnoticed."""

    from batchgen.prefix_reuse.prefill import PrefixCacheSequenceState
    from batchgen.prefix_reuse.trajectory_restore import (
        FAIL_NO_RESTORE_SOURCE,
        TrajectoryPlannedRestoreFailed,
    )

    _stub_heavy_modules(monkeypatch, free_state_slots=32)
    # A follower: no attachment of its own, the leader's plan in its state.
    worker, _ = _worker(
        rank=1,
        owns_host_kv=False,
        states={
            GLOBAL_IDS[0]: PrefixCacheSequenceState(
                lookup_result=None,
                attached_tokens=BOUNDARY,
                compute_cached_tokens=BOUNDARY,
                trajectory_restore_boundary=BOUNDARY,
            ),
        },
    )
    assert worker._traj.trajectory_attachments == {}
    assert worker._traj.trajectory_mirrored_restores == {}
    with pytest.raises(TrajectoryPlannedRestoreFailed) as excinfo:
        worker._restore_trajectory_frontiers([0, 1, 2])
    assert excinfo.value.reason == FAIL_NO_RESTORE_SOURCE
    assert excinfo.value.rank == 1
    assert excinfo.value.sequence_id == GLOBAL_IDS[0]


def test_unplanned_rows_with_no_source_pass_the_seam(monkeypatch):
    """The plan-time decline path: nothing planned, nothing to restore."""

    _stub_heavy_modules(monkeypatch, free_state_slots=32)
    worker, _ = _worker(states=_hit_states())
    worker._restore_trajectory_frontiers([0, 1, 2])
    assert worker._traj.trajectory_restorer.restores == []
    assert worker._traj.trajectory_restorer.declines == []
