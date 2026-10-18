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
    # Every pool site takes its geometry from here (the packed-FP8 primary
    # cache is unpacked back to the bf16 activation latent for the pool).
    "_prefill_pool_geometry",
    "_prefill_pool_page_dims",
    "_prefill_pool_group",
    "_prefill_pool_free_state_slots",
    "_replace_prefix_lookup_hits",
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

    def __init__(self, *, boundary=BOUNDARY, group_id=0, pages=None):
        self.boundary_tokens = boundary
        self.owning_group_id = group_id
        self.record = _Record(pages)
        self.session_key = "s1"


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
        rank=rank,
        local_rank=0,
        world_size=max(1, attn_tp_size),
        local_world_size=8,
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
        _prefix_sequence_states=dict(states or {}),
        _prefill_pool_base=(0, 0, 0),
        _prefill_pool_page_budget=free_pages,
        _batchgen_debug=None,
        _any_sequence_batchgen_debug=True,
        _trajectory_attachments=dict(attachments or {}),
        _trajectory_session_groups=index,
    )
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
        UUIDS, dict(worker._prefix_sequence_states if states is None else states)
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
    hits, confirmed, pages = worker._materialized_prefix_hits(
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
    hits, confirmed, pages = worker._materialized_prefix_hits(
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
    hits, confirmed, pages = worker._materialized_prefix_hits(
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
    hits, confirmed, pages = worker._materialized_prefix_hits(
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
    hits, confirmed, pages = worker._materialized_prefix_hits(
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
    _, confirmed, pages = worker._materialized_prefix_hits(
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
    _, confirmed, _ = worker._materialized_prefix_hits(
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
    _, confirmed, pages = worker._materialized_prefix_hits(
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
        hits, confirmed, _ = worker._materialized_prefix_hits(
            [seqs[uuid] for uuid in UUIDS], page_tokens=PAGE_TOKENS, group="group"
        )
    assert hits == [] and confirmed is True
    assert "boundary_off_the_page_grid" in caplog.text


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
        hits, confirmed, _ = worker._materialized_prefix_hits(
            [seqs[uuid] for uuid in UUIDS], page_tokens=PAGE_TOKENS, group="group"
        )
    assert hits == [] and confirmed is True
    assert "no_private_token_left" in caplog.text


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
        hits, confirmed, _ = worker._materialized_prefix_hits(
            [seqs[uuid] for uuid in UUIDS], page_tokens=PAGE_TOKENS, group="group"
        )
    assert hits == [] and confirmed is True
    assert "serving_group_is_not_the_owning_group" in caplog.text


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
    states = dict(worker._prefix_sequence_states)
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
    state = worker._prefix_sequence_states[GLOBAL_IDS[0]]
    assert state.compute_cached_tokens == BOUNDARY
    assert state.attached_tokens == BOUNDARY
    assert states[GLOBAL_IDS[0]].compute_cached_tokens == BOUNDARY
    assert wave.chain_tokens(0) == BOUNDARY


def test_the_attachment_holding_the_frontier_resident_is_kept_for_the_wave(monkeypatch):
    worker, _, _, _ = _materialized_wave(monkeypatch)
    assert worker.prefix_cache_coordinator.released == [], (
        "releasing it would let the pages the load reads be evicted mid-wave"
    )
    state = worker._prefix_sequence_states[GLOBAL_IDS[0]]
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
    assert worker._prefix_sequence_states[GLOBAL_IDS[1]].commit_attachment_handles == ()


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
        UUIDS, dict(worker._prefix_sequence_states)
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
    state = worker._prefix_sequence_states[GLOBAL_IDS[0]]
    assert state.commit_attachment_handles == (), "released, not also retained"
    # And the sequence recomputes from token 0, which is what makes the KDA
    # restore decline on its own suffix-offset identity.
    assert state.compute_cached_tokens == 0


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
            UUIDS, dict(worker._prefix_sequence_states)
        )
    assert "WITHOUT its 1 materialized prefix" in caplog.text
    if wave is not None:
        assert wave.materialized == frozenset()
        # The sequence still completes: it is computed in full.
        assert wave.chain_tokens(0) == 0
