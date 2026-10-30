"""GLM-5.3 session hold, end to end on CPU: hold -> resume -> pool load -> close.

A pool-only model (DSA) has no suffix-prefill path but the in-wave prefill
pool, so a resumed turn's held span becomes a LOADED pool segment:

* the resume is planned on whole pages (accepted granularity): turn 2's
  cached_tokens == (turn1_kv // 64) * 64; the partial last held page is
  freed and its tokens are recomputed by the turn's tail;
* ``_plan_session_pool_wave`` turns each resumed turn into a materialized
  segment over its held pages in BOTH Host regions (strict page-id
  equality), computes no shared segment, and the turn's tail starts at the
  boundary, writing into the turn's own first private page;
* a wave the pool cannot take declines its resumes: each keeps its kept
  pages attached (so it reserves exactly what footprint admission charged)
  and is recomputed from token 0 into them; the decline is settled on every
  rank at commit;
* the loaded segment lands byte-exactly in the pool (both regions);
* close frees every held page in both regions;
* GPT-OSS (one region) keeps its partial-page resume and never plans a pool
  wave.

Worker methods are compiled from the real source onto the fakes of
``tests/test_session_kv_hold.py``.
"""

from __future__ import annotations

import dataclasses
import logging
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "unit"))

from batchgen.models.wrappers.attention import AttnWrapperBase  # noqa: E402
from batchgen.prefix_reuse.executor import PoolChunkExecutor  # noqa: E402
from batchgen.prefix_reuse.pool_materialize import (  # noqa: E402
    DualRegionPageMismatch,
)
from batchgen.prefix_reuse.wave_plan import (  # noqa: E402
    MaterializedPrefix,
    plan_wave_prefix_sharing,
)
from batchgen.sequence import (  # noqa: E402
    INITIAL_GPU_PAGE_BUFFER,
    SequenceEntry,
    SequenceStatus,
)
from batchgen.worker import kv_admission as kv_admission_lib  # noqa: E402
from batchgen.worker.prefill import compute_prefill_host_reservation  # noqa: E402
from batchgen.worker import session_hold as session_hold_lib  # noqa: E402
from tests import test_session_kv_hold as hold_tests  # noqa: E402
from tests.worker_method_source import load_worker_methods  # noqa: E402

import test_glm_pool_dual_region_materialize as load_tests  # noqa: E402

PAGE = hold_tests.PAGE

_POOL_METHODS = (
    "_plan_session_pool_wave",
    "_apply_session_pool_declines",
    "_session_pool_resume_enabled",
    "_require_session_hold_pool_resume",
    "_finish_prefill_pool_wave",
    "_materialized_segment_loader",
    "_dual_region_segment_loader",
    "_materialized_dual_region_layers",
    "_load_dual_region_segments",
    "_load_materialized_segments",
    "_prefill_pool_page_tokens",
    "_kv_budget_pages",
    "_kv_footprint_candidates",
    "_prefill_host_reservation_plan",
)


class _GatherDist(hold_tests._Dist):
    """``all_gather_object`` returns what the peer ranks declined."""

    peer_declined: List[List[str]] = []
    gathers = 0

    @classmethod
    def all_gather_object(cls, out, mine):
        cls.gathers += 1
        parts = [list(mine)] + [list(part) for part in cls.peer_declined]
        assert len(parts) == len(out)
        out[:] = parts


_EXTRA = load_worker_methods(_POOL_METHODS, {
    "replace": dataclasses.replace,
    "session_hold_lib": session_hold_lib,
    "kv_admission_lib": kv_admission_lib,
    "compute_prefill_host_reservation": compute_prefill_host_reservation,
    "INITIAL_GPU_PAGE_BUFFER": INITIAL_GPU_PAGE_BUFFER,
    "SequenceEntry": SequenceEntry,
    "SequenceStatus": SequenceStatus,
    "logging": logging,
    "torch": torch,
    "time": time,
    "dist": _GatherDist,
    "List": List, "Optional": Optional, "Sequence": Sequence,
    "Dict": Dict, "Tuple": Tuple,
})


class _PoolWorker(hold_tests._DualWorker):
    """A GLM-5.3 hold rank: two mirrored Host regions, pool resume on."""

    world_size = 1

    def __init__(self, rank=0, *, mirrored=True, budget=64):
        super().__init__(rank=rank)
        if mirrored:
            # DualHostKVCoordinator mirrors every allocation: same page ids.
            self.aux = hold_tests._HostView(free_pages=range(100, 200))
        self._session_hold_pool_resume = True
        self._session_pool_declined = []
        self._prefill_pool_wave = None
        self._prefill_pool_page_budget = budget
        self._kv_resume_keep_cache = {}
        self.local_world_size = 8
        self.prefix_cache_runtime_config = None
        self.huggingface_ckpt_name = "zai-org/GLM-5.3-FP8"
        self.engine_config = SimpleNamespace(Module_Batching_Config=SimpleNamespace(
            prefill_micro_batch_token_cap=256,
            prefill_prefix_pool_workspace_bytes=24 * 2**30,
        ))

    def __getattr__(self, name):
        if name in _EXTRA:
            return _EXTRA[name].__get__(self)
        return super().__getattr__(name)

    _kv_resume_keep_cache: dict

    def _get_effective_chunk_size(self):
        return 8192

    def _owns_local_sequence(self, seq):
        return seq is not None and int(seq.assigned_rank) == self.rank

    def _prefill_pool_layout(self, geometry):
        return False, tuple(range(load_tests.LAYERS)), load_tests.INDEXER


@pytest.fixture(autouse=True)
def _geometry(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "batchgen.kv_cache.host_kv_mananger_config",
        load_tests._geometry(),
    )
    _GatherDist.peer_declined = []
    _GatherDist.gathers = 0
    yield
    AttnWrapperBase.prefill_pool = None
    AttnWrapperBase.prefill_pool_batch = None


PROMPT = list(range(1, 101))
DECODED = [500 + i for i in range(30)]
TURN1_KV = len(PROMPT) + len(DECODED) - 1  # 129: the last sample has no KV
TURN2 = PROMPT + DECODED + [900 + i for i in range(40)]  # extends turn 1


def _held(worker):
    seq = hold_tests._finish_dual_turn(
        worker, "turn1", 7, PROMPT, DECODED, own_pages=5
    )
    hold_tests._dual_hold(worker, seq)
    return worker._session_hold.get("s1")


def _resume(worker, *, fresh=True):
    worker.add_seq("turn2", 8, TURN2, rank=worker.rank)
    uuids = ["turn2"]
    if fresh:
        worker.add_seq("fresh", 9, [7] * 50, rank=worker.rank, session="s2")
        uuids.append("fresh")
    worker._plan_session_resumes(uuids)
    worker._plan_session_pool_wave(uuids)
    return uuids


# --------------------------------------------------------------------------- #
#  1. Whole-page resume (accepted granularity)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "lcp, kv, prompt, cached, keep, release",
    [
        (129, 129, 300, 128, 2, 1),   # turn1_kv 129 -> 128, partial page freed
        (128, 129, 300, 128, 2, 1),
        (200, 200, 200, 192, 3, 1),   # prompt-1 = 199 -> 192
        (127, 200, 300, 64, 1, 3),
    ],
)
def test_a_pool_resume_is_planned_on_whole_pages(lcp, kv, prompt, cached, keep, release):
    plan = session_hold_lib.plan_resume(
        session_id="s", lcp_tokens=lcp, kv_tokens=kv, prompt_tokens=prompt,
        page_tokens=PAGE, whole_pages=True,
    )
    assert (plan.cached_tokens, plan.keep_pages, plan.release_pages) == (
        cached, keep, release
    )
    assert plan.cached_tokens == plan.keep_pages * PAGE


def test_a_pool_resume_below_one_whole_page_drops():
    plan = session_hold_lib.plan_resume(
        session_id="s", lcp_tokens=100, kv_tokens=129, prompt_tokens=300,
        page_tokens=128, whole_pages=True,
    )
    assert (plan.cached_tokens, plan.keep_pages) == (0, 0)
    plan = session_hold_lib.plan_resume(
        session_id="s", lcp_tokens=63, kv_tokens=129, prompt_tokens=300,
        page_tokens=PAGE, whole_pages=True,
    )
    assert plan.drop_reason == session_hold_lib.DROP_LCP_BELOW_PAGE


# --------------------------------------------------------------------------- #
#  2. The resumed turn becomes a loaded pool segment
# --------------------------------------------------------------------------- #


def test_turn2_loads_the_held_span_as_one_segment_in_both_regions():
    worker = _PoolWorker()
    record = _held(worker)
    assert record.pages_by_group() == {0: (100, 101, 102), 1: (100, 101, 102)}
    _resume(worker)
    state = worker._prefix_sequence_states[8]
    expected = (TURN1_KV // PAGE) * PAGE
    assert state.compute_cached_tokens == expected == 128
    assert state.attached_tokens == expected
    assert state.shared_page_ids_for_group(0) == (100, 101)
    assert state.shared_page_ids_for_group(1) == (100, 101)
    wave = worker._prefill_pool_wave
    assert wave is not None and wave.uuids == ["turn2", "fresh"]
    assert len(wave.materialized) == 1 and not wave.computed_segments()
    (segment_id,) = wave.materialized
    assert wave.plan.segments[segment_id].tokens == expected
    assert wave.segment_host_pages[segment_id] == {0: [100, 101], 1: [100, 101]}
    assert wave.plan.computed_tokens == (len(TURN2) - expected) + 50
    assert wave.chain_tokens(0) == expected and wave.chain_tokens(1) == 0


def test_the_tail_writes_from_the_boundary_into_its_own_page():
    worker = _PoolWorker()
    _held(worker)
    _resume(worker)
    wave = worker._prefill_pool_wave
    wave.pool = _pool64()
    wave.executor = PoolChunkExecutor(wave.plan, wave.prompts, wave.pool)
    rows = wave.executor.build(wave.plan.chunks[0])
    ids, starts = wave.row_host_targets(rows)
    targets = {
        (ids[row], starts[row]) for row, item in enumerate(rows.items)
        if item.kind == "tail"
    }
    assert (8, 128) in targets and (9, 0) in targets


def test_commit_keeps_whole_pages_and_frees_the_partial_one_in_both_regions():
    worker = _PoolWorker()
    _held(worker)
    _resume(worker)
    worker._commit_session_resumes()
    assert worker.host.resident() == [100, 101]
    assert worker.aux.resident() == [100, 101]
    assert worker._session_attached_pages == {8: {0: (100, 101), 1: (100, 101)}}
    plan = worker._session_resume_plans["turn2"]
    assert plan.cached_tokens == (TURN1_KV // PAGE) * PAGE
    snap = worker._session_hold.snapshot()
    assert snap["resumes"] == 1 and snap["reload_tokens"] == 128
    assert snap["held_sessions"] == 0
    assert _GatherDist.gathers == 0  # world_size 1: nothing to settle


def test_close_after_the_resumed_turn_leaves_no_held_page():
    worker = _PoolWorker()
    _held(worker)
    _resume(worker)
    worker._commit_session_resumes()
    worker._release_session_prefix_states([8])
    assert worker.host.resident() == [] and worker.aux.resident() == []
    assert worker._session_hold.snapshot()["held_kv_pages"] == 0


def _pool64():
    from batchgen.prefix_reuse.pool import PrefillPrefixPool

    pool = PrefillPrefixPool(
        num_layers=load_tests.LAYERS, num_pages=16, page_tokens=PAGE,
        num_kv_heads=1, head_dim=load_tests.LATENT, dtype=torch.float32,
        device="cpu", with_v=False,
        indexer_layers=tuple(range(load_tests.LAYERS)),
        indexer_dim=load_tests.INDEXER,
    )
    pool.k.zero_()
    pool.indexer.zero_()
    return pool


def _host64(dim, seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(
        (load_tests.LAYERS, 110, PAGE, dim), generator=generator,
        dtype=torch.float32,
    )


def test_the_loaded_segment_lands_in_the_pool_byte_exactly():
    worker = _PoolWorker()
    _held(worker)
    _resume(worker)
    wave = worker._prefill_pool_wave
    latent_host = _host64(load_tests.LATENT, 1)
    indexer_host = _host64(load_tests.INDEXER, 2)
    views = {
        0: load_tests._CopyingHostView(latent_host),
        1: load_tests._CopyingHostView(indexer_host),
    }
    pool = _pool64()
    wave.pool = pool
    wave.executor = PoolChunkExecutor(wave.plan, wave.prompts, pool)
    worker.model = SimpleNamespace(model=SimpleNamespace(layers=[
        SimpleNamespace(self_attn=load_tests._GLMLayer(i, has_indexer=True))
        for i in range(load_tests.LAYERS)
    ]))
    worker._host_views_by_group = lambda: views
    loader = worker._materialized_segment_loader(wave)
    rows = wave.executor.build(wave.plan.chunks[0])
    AttnWrapperBase.prefill_pool = pool
    AttnWrapperBase.prefill_pool_batch = rows
    assert worker._load_materialized_segments(wave, rows, loader) == 128
    assert views[0].loads == [(100, 101)] and views[1].loads == [(100, 101)]
    stage = rows.materialized[0]
    held = [100, 101]
    for layer in range(load_tests.LAYERS):
        pooled = pool.k[layer].view(-1, load_tests.LATENT).index_select(
            0, stage.slots
        )
        assert torch.equal(pooled, _host_span(latent_host, layer, held))
        pooled = pool.indexer[pool.indexer_row[layer]].view(
            -1, load_tests.INDEXER
        ).index_select(0, stage.slots)
        assert torch.equal(pooled, _host_span(indexer_host, layer, held))


def _host_span(host, layer, pages):
    return torch.cat([host[layer, page] for page in pages], dim=0)


def test_a_hold_wave_finishes_without_publishing():
    worker = _PoolWorker()
    _held(worker)
    _resume(worker)
    worker._finish_prefill_pool_wave()
    assert worker._prefill_pool_wave is None


# --------------------------------------------------------------------------- #
#  3. Fail loud: regions that disagree
# --------------------------------------------------------------------------- #


def test_held_regions_with_different_page_ids_are_refused():
    worker = _PoolWorker(mirrored=False)  # aux allocator starts at 300
    _held(worker)
    worker.add_seq("turn2", 8, TURN2)
    worker._plan_session_resumes(["turn2"])
    with pytest.raises(DualRegionPageMismatch, match="disagree"):
        worker._plan_session_pool_wave(["turn2"])
    assert worker._prefill_pool_wave is None


# --------------------------------------------------------------------------- #
#  4. The pool declines: the turn prefills in full, on every rank alike
# --------------------------------------------------------------------------- #


def test_a_declined_resume_is_recomputed_into_its_kept_pages():
    worker = _PoolWorker(budget=None)  # budget not measured yet
    _held(worker)
    _resume(worker, fresh=False)
    assert worker._prefill_pool_wave is None
    state = worker._prefix_sequence_states[8]
    # Nothing cached (the ordinary prefill recomputes from token 0 and never
    # reaches the full-K/V materialization) but the kept pages stay attached.
    assert state.compute_cached_tokens == 0
    assert state.attached_tokens == 2 * PAGE
    assert state.shared_page_ids_for_group(1) == (100, 101)
    assert worker._session_pool_declined == ["turn2"]
    worker._commit_session_resumes()
    plan = worker._session_resume_plans["turn2"]
    assert plan.drop_reason == session_hold_lib.DROP_POOL_DECLINED
    assert (plan.cached_tokens, plan.keep_pages, plan.release_pages) == (0, 2, 1)
    # The partial page is freed; the kept ones are the turn's, freed with it.
    assert worker.host.resident() == [100, 101] == worker.aux.resident()
    assert worker._session_hold.snapshot()["drops"] == {"pool_declined": 1}
    assert worker._session_hold.snapshot()["held_sessions"] == 0
    assert "turn2" not in worker._session_resumed_uuids
    worker._release_session_prefix_states([8])
    assert worker.host.resident() == [] and worker.aux.resident() == []


def test_a_declined_resume_reserves_exactly_what_admission_charged():
    """3 held pages, 2 kept: admission charges budget - 2; so must the
    reservation after the pool declines, or the wave is rejected on every
    pass while admission keeps saying it fits."""
    worker = _PoolWorker(budget=None)
    _held(worker)
    assert worker._session_hold.get("s1").pages == 3
    worker.add_seq("turn2", 8, TURN2)
    candidates, _, keep = worker._kv_footprint_candidates(["turn2"])
    (candidate,) = candidates
    assert keep == {"turn2": 2}
    assert candidate.host_pages == candidate.budget_pages - 2
    worker._plan_session_resumes(["turn2"])
    worker._plan_session_pool_wave(["turn2"])
    assert worker._session_pool_declined == ["turn2"]
    reservations, owner_requests = worker._prefill_host_reservation_plan(["turn2"])
    reserved_pages, _ = reservations["turn2"]
    ((uuid, private_tokens),) = owner_requests
    assert uuid == "turn2"
    # The same discount admission applied: exactly the kept pages.
    assert reserved_pages - private_tokens // PAGE == keep["turn2"]


def test_a_scratch_that_does_not_fit_declines():
    worker = _PoolWorker(budget=2)
    _held(worker)
    _resume(worker, fresh=False)
    assert worker._prefill_pool_wave is None
    assert worker._session_pool_declined == ["turn2"]


def test_a_peer_ranks_decline_is_applied_on_this_rank():
    worker = _PoolWorker(rank=1)
    worker.world_size = 2
    _held(worker)  # rank 1: same record, no pages
    worker.add_seq("turn2", 8, TURN2, rank=0)  # served by rank 0
    worker._plan_session_resumes(["turn2"])
    worker._plan_session_pool_wave(["turn2"])
    assert worker._prefill_pool_wave is None  # not this rank's turn
    _GatherDist.peer_declined = [["turn2"]]
    worker._commit_session_resumes()
    assert _GatherDist.gathers == 1
    plan = worker._session_resume_plans["turn2"]
    assert plan.drop_reason == session_hold_lib.DROP_POOL_DECLINED
    assert worker._session_hold.snapshot()["drops"] == {"pool_declined": 1}


def test_a_wave_without_a_resume_does_not_gather():
    worker = _PoolWorker()
    worker.world_size = 2
    _held(worker)
    worker.add_seq("turn2", 8, PROMPT[:40] + [7] * 100)  # LCP below a page
    worker._plan_session_resumes(["turn2"])
    worker._plan_session_pool_wave(["turn2"])
    worker._commit_session_resumes()
    assert _GatherDist.gathers == 0
    assert worker._prefill_pool_wave is None


# --------------------------------------------------------------------------- #
#  5. Init gate and GPT-OSS
# --------------------------------------------------------------------------- #


def test_a_pool_resume_hold_without_a_pool_workspace_is_refused_at_init():
    worker = _PoolWorker()
    worker.engine_config.Module_Batching_Config.prefill_prefix_pool_workspace_bytes = 0
    worker.parallel_manager = SimpleNamespace(attn_tp_size=1)
    with pytest.raises(session_hold_lib.SessionHoldUnsupported, match="workspace"):
        worker._require_session_hold_pool_resume()


def test_a_pool_resume_hold_with_a_workspace_passes_init():
    worker = _PoolWorker()
    worker.parallel_manager = SimpleNamespace(attn_tp_size=1)
    worker._require_session_hold_pool_resume()


def test_gpt_oss_keeps_its_partial_page_resume_and_plans_no_pool_wave():
    worker, prompt, decoded = hold_tests._held_worker()
    assert not worker._session_pool_resume_enabled()
    worker.add_seq("turn2", 8, prompt + decoded + [1] * 5)
    worker._plan_session_resumes(["turn2"])
    state = worker._prefix_sequence_states[8]
    assert state.compute_cached_tokens == TURN1_KV  # 129, not page-floored
    assert state.attached_tokens == 3 * PAGE
    worker._commit_session_resumes()
    assert worker._session_resume_plans["turn2"].cached_tokens == TURN1_KV


def test_the_hold_planner_computes_no_shared_segment():
    shared = list(range(100, 100 + 4 * PAGE))
    prompts = [shared + [1, 2], shared + [3, 4], list(range(900, 900 + 3 * PAGE)) + [5]]
    hit = [MaterializedPrefix(index=2, tokens=3 * PAGE)]
    kwargs = dict(block_tokens=PAGE, pool_pages=64, chunk_tokens=1024,
                  same_chunk_ancestors=True, materialized=hit)
    sharing = plan_wave_prefix_sharing(prompts, **kwargs)
    assert any(not seg.materialized for seg in sharing.segments)
    hold = plan_wave_prefix_sharing(prompts, share_computed=False, **kwargs)
    assert [seg.materialized for seg in hold.segments] == [True]
    assert hold.computed_tokens == 2 * (4 * PAGE + 2) + 1
