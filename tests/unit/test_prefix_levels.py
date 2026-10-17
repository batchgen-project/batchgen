"""Levels of one prepacked chunk, and the segment mirror a follower installs.

CPU only; the modules come from their files because importing the ``batchgen``
package pulls in the native kernels.
"""

import importlib.util
import pickle
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch


def _load(name, *relative):
    path = Path(__file__).parents[2].joinpath(*relative)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_PLAN = _load("test_levels_plan_mod", "batchgen", "prefix_reuse", "wave_plan.py")
_POOL = _load("test_levels_pool_mod", "batchgen", "prefix_reuse", "pool.py")
_EXEC = _load("test_levels_exec_mod", "batchgen", "prefix_reuse", "executor.py")
_LEVELS = _load("test_levels_mod", "batchgen", "prefix_reuse", "levels.py")


def _load_wave_runtime():
    """``wave_runtime`` imports the package for its annotations only; serve
    those names from stubs and leave ``sys.modules`` as it was found."""

    names = (
        "batchgen",
        "batchgen.prefix_reuse",
        "batchgen.prefix_reuse.executor",
        "batchgen.prefix_reuse.pool",
        "batchgen.prefix_reuse.wave_plan",
    )
    saved = {name: sys.modules.get(name) for name in names}
    try:
        for name in names:
            stub = ModuleType(name)
            stub.__path__ = []
            sys.modules[name] = stub
        sys.modules["batchgen.prefix_reuse.executor"].ChunkBatch = object
        sys.modules["batchgen.prefix_reuse.executor"].PoolChunkExecutor = object
        sys.modules["batchgen.prefix_reuse.pool"].PrefillPrefixPool = object
        sys.modules["batchgen.prefix_reuse.wave_plan"].WavePrefixPlan = object
        return _load(
            "test_levels_runtime_mod", "batchgen", "prefix_reuse", "wave_runtime.py"
        )
    finally:
        for name, module in saved.items():
            if module is None:
                del sys.modules[name]
            else:
                sys.modules[name] = module


_RUNTIME = _load_wave_runtime()

B = 4  # block tokens = page tokens


def blk(n):
    return [n * 10 + k for k in range(B)]


# One root block, then three two-way branches. The deepest block is shared by
# nobody, so it stays in the prompt's tail and every tail reads three pooled
# segments: 7 pooled segments and 8 tails, four levels of rows.
DEPTH3 = [
    blk(1) + blk(2 + a) + blk(4 + 2 * a + b) + blk(8 + 4 * a + 2 * b + c) + [900 + c]
    for a in (0, 1)
    for b in (0, 1)
    for c in (0, 1)
]
# One pooled segment of two blocks, then a private block per prompt.
DEPTH1 = [blk(1) + blk(2) + blk(20 + i) + [910 + i] for i in range(3)]


def _ordered(prompts, same_chunk=True):
    """Plan a wave, then order each chunk by level; both must validate."""

    plan = _PLAN.plan_wave_prefix_sharing(
        prompts, block_tokens=B, pool_pages=1000, chunk_tokens=1000,
        same_chunk_ancestors=same_chunk,
    )
    _PLAN.validate_wave_prefix_plan(plan, prompts)
    ordered = _LEVELS.order_chunk_items_by_level(plan)
    _PLAN.validate_wave_prefix_plan(ordered, prompts)
    return plan, ordered


def _batches(plan, prompts):
    """Every chunk of a plan as a ChunkBatch, the pool sized as the worker does."""

    scratch = _POOL.scratch_pages_bound(
        plan.chunk_tokens, max(len(p) for p in prompts), len(prompts), B
    )
    pool = _POOL.PrefillPrefixPool(
        num_layers=1, num_pages=plan.pool_pages + scratch, page_tokens=B,
        num_kv_heads=1, head_dim=2, dtype=torch.float32, device="cpu",
    )
    executor = _EXEC.PoolChunkExecutor(plan, prompts, pool)
    batches = []
    for chunk in plan.chunks:
        batches.append(executor.build(chunk))
        executor.finish(chunk)
    return batches


def _sids(plan):
    """Negative pseudo sequence ids, as the worker hands out per segment."""

    return {seg.segment_id: -(1 + seg.segment_id) for seg in plan.segments}


def test_level_order_keeps_every_plan_invariant():
    plan, ordered = _ordered(DEPTH3)

    assert len(ordered.chunks) == 1
    # DFS order interleaves the levels; ordering by level does not.
    assert [(i.kind, i.ref) for i in plan.chunks[0].items] != [
        (i.kind, i.ref) for i in ordered.chunks[0].items
    ]
    assert [len(i.prefix_segments) for i in plan.chunks[0].items] == [
        0, 1, 2, 3, 3, 2, 3, 3, 1, 2, 3, 3, 2, 3, 3
    ]
    assert [len(i.prefix_segments) for i in ordered.chunks[0].items] == [
        0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3
    ]
    assert ordered.computed_tokens == plan.computed_tokens
    assert [c.release_after for c in ordered.chunks] == [
        c.release_after for c in plan.chunks
    ]


def test_chunk_levels_bound_each_level_to_its_rows():
    _, ordered = _ordered(DEPTH3)
    batch, = _batches(ordered, DEPTH3)

    levels = _LEVELS.chunk_levels(batch)

    assert [(l.depth, l.row_start, l.row_end) for l in levels] == [
        (0, 0, 1), (1, 1, 3), (2, 3, 7), (3, 7, 15)
    ]
    assert [(l.token_start, l.token_end) for l in levels] == [
        (0, 4), (4, 12), (12, 28), (28, 68)
    ]
    assert levels[-1].token_end == int(batch.input_ids.numel())
    for level in levels:
        assert level.bounds.dtype == batch.cu_seqlens_q.dtype
        assert level.bounds.numel() == level.row_end - level.row_start + 1
        assert torch.equal(
            level.bounds,
            batch.cu_seqlens_q[level.row_start:level.row_end + 1]
            - level.token_start,
        )
    dtype = batch.cu_seqlens_q.dtype
    # Four 4-token segments, then eight 5-token tails, rebased to their level.
    assert torch.equal(levels[2].bounds, torch.tensor([0, 4, 8, 12, 16], dtype=dtype))
    assert torch.equal(
        levels[3].bounds,
        torch.tensor([0, 5, 10, 15, 20, 25, 30, 35, 40], dtype=dtype),
    )


def test_chunk_levels_refuse_rows_that_are_not_grouped_by_level():
    plan, _ = _ordered(DEPTH3)
    batch, = _batches(plan, DEPTH3)

    with pytest.raises(ValueError, match="not grouped by level"):
        _LEVELS.chunk_levels(batch)


def test_fork_pairs_resolve_a_parent_in_this_chunk_and_in_an_earlier_one():
    _, one_chunk = _ordered(DEPTH1)
    _, split = _ordered(DEPTH1, same_chunk=False)
    sids = _sids(one_chunk)
    assert len(one_chunk.chunks) == 1 and len(split.chunks) == 2

    # The parent segment is row 0 of the same chunk.
    batch, = _batches(one_chunk, DEPTH1)
    assert _LEVELS.fork_pairs(batch, sids) == ((1, -1), (2, -1), (3, -1))

    # The parent segment ran in the previous chunk; its pseudo id is the same.
    segment_chunk, tail_chunk = _batches(split, DEPTH1)
    assert _LEVELS.fork_pairs(segment_chunk, sids) == ()
    assert _LEVELS.fork_pairs(tail_chunk, sids) == ((0, -1), (1, -1), (2, -1))
    assert [
        (l.depth, l.row_start, l.row_end, l.token_start, l.token_end)
        for l in _LEVELS.chunk_levels(tail_chunk)
    ] == [(1, 0, 3, 0, 15)]


def test_fork_pairs_follow_a_depth_three_tree_to_the_nearest_ancestor():
    _, ordered = _ordered(DEPTH3)
    batch, = _batches(ordered, DEPTH3)
    sids = _sids(ordered)

    pairs = _LEVELS.fork_pairs(batch, sids)

    # Every row but the root segment forks, from its nearest pooled ancestor.
    assert [row for row, _ in pairs] == list(range(1, 15))
    for row, sid in pairs:
        assert sid == sids[ordered.chunks[0].items[row].prefix_segments[-1]]
        assert sid < 0
    # Two level-1 segments fork from the root, four level-2 segments from
    # those two, and the eight tails from those four.
    assert {sid for row, sid in pairs if row < 3} == {sids[0]}
    assert len({sid for row, sid in pairs if 3 <= row < 7}) == 2
    assert len({sid for row, sid in pairs if row >= 7}) == 4


def test_fork_pairs_fail_when_a_parent_segment_has_no_pseudo_id():
    _, ordered = _ordered(DEPTH1)
    batch, = _batches(ordered, DEPTH1)

    with pytest.raises(RuntimeError, match="no pseudo"):
        _LEVELS.fork_pairs(batch, {})


class FakeView:
    """Mirrors HostKVPageTable::RegisterOrUpdate: a full overwrite, no backend."""

    def __init__(self):
        self.table = {}

    def set_sequence_pages(self, sequence_id, page_ids):
        self.table[sequence_id] = list(page_ids)


def _wave(plan, prompts):
    pool = _POOL.PrefillPrefixPool(
        num_layers=1, num_pages=1, page_tokens=B, num_kv_heads=1, head_dim=2,
        dtype=torch.float32, device="cpu",
    )
    return _RUNTIME.PoolWave(
        plan=plan,
        prompts=prompts,
        uuids=[f"u{index}" for index in range(len(prompts))],
        global_ids=list(range(len(prompts))),
        page_tokens=pool.page_tokens,
        total_pages=pool.num_pages,
        pool=pool,
        executor=None,
    )


def test_segment_mirror_payload_round_trips_to_a_follower():
    _, ordered = _ordered(DEPTH3)
    leader, follower = _wave(ordered, DEPTH3), _wave(ordered, DEPTH3)
    for seg in ordered.segments:  # one 4-token page per segment, two KV groups
        leader.segment_sids[seg.segment_id] = -(1 + seg.segment_id)
        leader.segment_host_pages[seg.segment_id] = {
            0: [100 + seg.segment_id], 1: [200 + seg.segment_id],
        }

    payload = _RUNTIME.segment_mirror_payload(leader)
    assert payload == pickle.loads(pickle.dumps(payload))

    views = {0: FakeView(), 1: FakeView()}
    _RUNTIME.install_mirrored_segments(follower, payload, views)

    assert follower.segment_sids == leader.segment_sids
    assert follower.segment_host_pages == leader.segment_host_pages
    for group_id, base in ((0, 100), (1, 200)):
        assert views[group_id].table == {
            -(1 + seg.segment_id): [base + seg.segment_id]
            for seg in ordered.segments
        }
    # A member's chain is the leader's pages, ancestors first, per group.
    assert follower.chain_host_pages(0) == leader.chain_host_pages(0)


def test_mirroring_refuses_a_payload_that_misses_a_segment_or_its_pages():
    _, ordered = _ordered(DEPTH1)
    leader = _wave(ordered, DEPTH1)
    with pytest.raises(RuntimeError, match="no Host pages to mirror"):
        _RUNTIME.segment_mirror_payload(leader)

    seg = ordered.segments[0]  # two blocks, so two Host pages
    leader.segment_sids[seg.segment_id] = -1
    leader.segment_host_pages[seg.segment_id] = {0: [7, 8]}
    payload = _RUNTIME.segment_mirror_payload(leader)

    follower = _wave(ordered, DEPTH1)
    with pytest.raises(RuntimeError, match="missing from the mirror payload"):
        _RUNTIME.install_mirrored_segments(
            follower, {"segment_sids": {}, "segment_host_pages": {}},
            {0: FakeView()},
        )
    with pytest.raises(RuntimeError, match="Host pages in group 1"):
        _RUNTIME.install_mirrored_segments(
            follower, payload, {0: FakeView(), 1: FakeView()}
        )
    assert follower.segment_sids == {}

    _RUNTIME.install_mirrored_segments(follower, payload, {0: FakeView()})
    assert follower.segment_host_pages == {0: {0: [7, 8]}}


def test_pool_v_head_dim_defaults_to_the_k_head_dim():
    pool = _POOL.PrefillPrefixPool(
        num_layers=2, num_pages=3, page_tokens=B, num_kv_heads=2, head_dim=8,
        dtype=torch.float32, device="cpu",
    )

    assert pool.v.shape == pool.k.shape
    assert pool.page_bytes == 2 * 2 * B * 2 * 8 * 4


def test_pool_holds_a_wider_k_than_v():
    pool = _POOL.PrefillPrefixPool(
        num_layers=24, num_pages=3, page_tokens=B, num_kv_heads=2,
        head_dim=192, v_head_dim=128, dtype=torch.float32, device="cpu",
    )

    assert pool.k.shape[-1] == 192 and pool.v.shape[-1] == 128
    assert pool.page_bytes == 24 * B * 2 * (192 + 128) * 4
    assert pool.slot_list([1], 0, 2) == [B, B + 1]


def test_pool_without_v_charges_only_the_latent():
    pool = _POOL.PrefillPrefixPool(
        num_layers=2, num_pages=3, page_tokens=B, num_kv_heads=1, head_dim=576,
        dtype=torch.float32, device="cpu", with_v=False,
    )

    assert pool.v is None
    assert pool.page_bytes == 2 * B * 576 * 4
