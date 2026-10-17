"""M3 Slice 3: the materialized pool segment, option A.

A trajectory re-entry hit's prefix is LOADED into the in-wave prefix pool
instead of computed. These tests pin the contract that makes that safe:

* the plan carries a materialized segment with a computed segment's geometry,
  so a reader's features are unchanged;
* the executor allocates its pages from the same allocator, in the same order,
  and hands the loader exactly the slots a computed segment would have written;
* a recurrent layer does not fork state from a segment nothing ran, and refuses
  by name to run a reader whose own state was never restored;
* the Host side allocates, mirrors and publishes only what this wave computed;
* the model's expansion is this layer's own ``kv_b_proj``, writing what
  computing the same latent would have written.

Real planner, real pool, real executor, real level cut and a real projection
module; only the Host-KV view and the H2D task are fakes, at the exact
boundary where a CPU test cannot follow.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from batchgen.prefix_reuse.executor import (  # noqa: E402
    MaterializedStage,
    PoolChunkExecutor,
)
from batchgen.prefix_reuse.levels import (  # noqa: E402
    fork_pairs,
    state_levels,
)
from batchgen.prefix_reuse.pool import PrefillPrefixPool  # noqa: E402
from batchgen.prefix_reuse.pool_materialize import (  # noqa: E402
    MaterializedSegmentLoader,
)
from batchgen.prefix_reuse.wave_plan import (  # noqa: E402
    MaterializedPrefix,
    PlanChunk,
    PlanItem,
    PlanSegment,
    PoolMaterializationDoesNotFit,
    WavePrefixPlan,
    group_materialized_prefixes,
    plan_wave_prefix_sharing,
    prefill_plan_work,
    validate_wave_prefix_plan,
)
from batchgen.prefix_reuse.wave_runtime import (  # noqa: E402
    PoolWave,
    allocate_segment_host_pages,
    install_mirrored_segments,
    segment_mirror_payload,
)

B = 4  # block (= Host page) tokens, small so the geometry is readable
CHUNK = 256


# --------------------------------------------------------------------------- #
#  Wave shapes
# --------------------------------------------------------------------------- #


def _frontier(uid, blocks):
    """A committed frontier: ``blocks`` blocks of one trajectory's tokens."""
    return [uid * 1000 + i for i in range(blocks * B)]


def _hit_prompt(uid, blocks, suffix):
    return _frontier(uid, blocks) + [uid * 1000 + 900 + k for k in range(suffix)]


def _plan(prompts, *, hits=(), pool_pages=64, chunk_tokens=CHUNK):
    return plan_wave_prefix_sharing(
        prompts,
        block_tokens=B,
        pool_pages=pool_pages,
        chunk_tokens=chunk_tokens,
        same_chunk_ancestors=True,
        materialized=hits,
    )


# --------------------------------------------------------------------------- #
#  1. The plan
# --------------------------------------------------------------------------- #


def test_a_hit_becomes_one_materialized_segment_over_its_frontier():
    prompt = _hit_prompt(1, 3, 5)
    other = _frontier(2, 2) + [7, 7, 7]
    prompts = [prompt, other]
    plan = _plan(prompts, hits=[MaterializedPrefix(index=0, tokens=3 * B)])
    validate_wave_prefix_plan(plan, prompts)

    loaded = [seg for seg in plan.segments if seg.materialized]
    assert len(loaded) == 1
    seg = loaded[0]
    assert (seg.segment_id, seg.token_start, seg.token_end) == (0, 0, 3 * B)
    assert seg.members == (0,)
    assert seg.parent_id is None
    assert plan.materialized_segments == (0,)
    # The hit's own tail picks up exactly where the frontier ends.
    tail = next(
        item
        for chunk in plan.chunks
        for item in chunk.items
        if item.kind == "tail" and item.ref == 0
    )
    assert tail.token_start == 3 * B
    assert tail.token_end == len(prompt)
    assert tail.prefix_segments == (0,)


def test_the_loaded_prefix_is_neither_computed_nor_counted_as_computed():
    prompt = _hit_prompt(1, 3, 5)
    prompts = [prompt, _frontier(2, 2) + [7, 7, 7]]
    plan = _plan(prompts, hits=[MaterializedPrefix(index=0, tokens=3 * B)])

    computed_tokens = sum(
        item.tokens
        for chunk in plan.chunks
        for item in chunk.items
        if item.kind != "materialize"
    )
    assert plan.computed_tokens == computed_tokens
    assert plan.materialized_tokens == 3 * B
    # Nothing recomputes the frontier: no row covers [0, boundary) of prompt 0.
    assert not [
        item
        for chunk in plan.chunks
        for item in chunk.items
        if item.kind == "tail" and item.ref == 0 and item.token_start == 0
    ]
    # And the load is not attention work.
    work = prefill_plan_work(plan)
    assert work.item_rows == sum(
        1
        for chunk in plan.chunks
        for item in chunk.items
        if item.kind != "materialize"
    )


def test_a_hit_takes_no_part_in_the_in_wave_tree():
    """Its tokens must not be computed once for the pool and loaded once."""

    frontier = _frontier(1, 3)
    # A second prompt shares the whole frontier: without the skip the tree
    # would pool those blocks and compute them beside the load.
    prompts = [frontier + [1, 2, 3], frontier + [4, 5, 6]]
    plan = _plan(prompts, hits=[MaterializedPrefix(index=0, tokens=3 * B)])
    validate_wave_prefix_plan(plan, prompts)

    assert [seg.materialized for seg in plan.segments] == [True]
    assert plan.segments[0].members == (0,)
    # Prompt 1 is untouched by the materialization: it computes its own prompt.
    tail = next(
        item
        for chunk in plan.chunks
        for item in chunk.items
        if item.kind == "tail" and item.ref == 1
    )
    assert (tail.token_start, tail.prefix_segments) == (0, ())


def test_two_prompts_on_the_same_frontier_read_one_segment():
    frontier = _frontier(1, 3)
    prompts = [frontier + [1, 2], frontier + [3, 4, 5]]
    plan = _plan(
        prompts,
        hits=[
            MaterializedPrefix(index=0, tokens=3 * B),
            MaterializedPrefix(index=1, tokens=3 * B),
        ],
    )
    validate_wave_prefix_plan(plan, prompts)
    assert len(plan.segments) == 1
    assert plan.segments[0].members == (0, 1)
    assert plan.materialized_tokens == 3 * B  # loaded once, read twice


def test_different_frontiers_get_one_segment_each():
    prompts = [_hit_prompt(1, 2, 3), _hit_prompt(2, 3, 2)]
    plan = _plan(
        prompts,
        hits=[
            MaterializedPrefix(index=0, tokens=2 * B),
            MaterializedPrefix(index=1, tokens=3 * B),
        ],
    )
    validate_wave_prefix_plan(plan, prompts)
    assert [(seg.segment_id, seg.token_end, seg.members) for seg in plan.segments] == [
        (0, 2 * B, (0,)),
        (1, 3 * B, (1,)),
    ]


def test_grouping_is_ordered_by_first_member_so_every_rank_numbers_alike():
    frontier = _frontier(9, 2)
    prompts = [_hit_prompt(1, 3, 2), frontier + [1], frontier + [2]]
    groups = group_materialized_prefixes(
        prompts,
        [
            MaterializedPrefix(index=2, tokens=2 * B),
            MaterializedPrefix(index=0, tokens=3 * B),
            MaterializedPrefix(index=1, tokens=2 * B),
        ],
        block_tokens=B,
    )
    # Declaration order decides, and members keep their declaration order.
    assert groups == ((2 * B, (2, 1)), (3 * B, (0,)))


def test_the_plan_of_a_hit_is_a_pure_function_of_its_inputs():
    prompts = [_hit_prompt(1, 3, 5), _frontier(2, 2) + [7, 7, 7]]
    hits = [MaterializedPrefix(index=0, tokens=3 * B)]
    assert _plan(prompts, hits=hits) == _plan(prompts, hits=hits)


# --------------------------------------------------------------------------- #
#  2. Features a reader sees: load against compute
# --------------------------------------------------------------------------- #


def _pool(num_pages, *, layers=2, heads=2, k_dim=6, v_dim=4):
    return PrefillPrefixPool(
        num_layers=layers,
        num_pages=num_pages,
        page_tokens=B,
        num_kv_heads=heads,
        head_dim=k_dim,
        v_head_dim=v_dim,
        dtype=torch.float32,
        device="cpu",
    )


def _run(plan, prompts, pool):
    """Build every chunk of a plan; returns the ChunkBatches in order."""
    executor = PoolChunkExecutor(plan, prompts, pool)
    built = []
    for chunk in plan.chunks:
        built.append(executor.build(chunk))
        executor.finish(chunk)
    executor.close()
    return built


def test_the_reader_of_a_loaded_segment_sees_a_computed_segments_features():
    """Same geometry, same page table, same cache length -- different writer.

    The comparison wave shares the identical frontier between two prompts, so
    its first prompt reads a COMPUTED segment of exactly the geometry the
    materialized one has.
    """

    frontier = _frontier(1, 3)
    suffix = [1, 2, 3, 4, 5]
    hit_prompts = [frontier + suffix]
    hit_plan = _plan(hit_prompts, hits=[MaterializedPrefix(index=0, tokens=3 * B)])

    shared_prompts = [frontier + suffix, frontier + [9, 9, 9, 9, 9]]
    shared_plan = _plan(shared_prompts)

    hit_rows = _run(hit_plan, hit_prompts, _pool(16))
    shared_rows = _run(shared_plan, shared_prompts, _pool(16))

    def tail_row(batches, wave_index):
        for batch in batches:
            for row, item in enumerate(batch.items):
                if item.kind == "tail" and item.ref == wave_index:
                    return batch, row
        raise AssertionError("tail not built")

    hit_batch, hit_row = tail_row(hit_rows, 0)
    shared_batch, shared_row = tail_row(shared_rows, 0)

    assert hit_batch.items[hit_row].token_start == shared_batch.items[shared_row].token_start
    assert int(hit_batch.cache_seqlens[hit_row]) == int(
        shared_batch.cache_seqlens[shared_row]
    )
    assert hit_batch.page_table[hit_row].tolist() == (
        shared_batch.page_table[shared_row].tolist()
    )
    assert int(hit_batch.cu_seqlens_q[hit_row + 1] - hit_batch.cu_seqlens_q[hit_row]) == (
        int(shared_batch.cu_seqlens_q[shared_row + 1]
            - shared_batch.cu_seqlens_q[shared_row])
    )
    start = int(hit_batch.cu_seqlens_q[hit_row])
    end = int(hit_batch.cu_seqlens_q[hit_row + 1])
    other = int(shared_batch.cu_seqlens_q[shared_row])
    assert hit_batch.input_ids[start:end].tolist() == suffix
    assert hit_batch.position_ids[start:end].tolist() == (
        shared_batch.position_ids[other:other + (end - start)].tolist()
    )


def test_the_loader_is_handed_the_slots_a_computed_segment_would_have_written():
    frontier = _frontier(1, 3)
    hit_prompts = [frontier + [1, 2, 3]]
    hit_plan = _plan(hit_prompts, hits=[MaterializedPrefix(index=0, tokens=3 * B)])
    shared_prompts = [frontier + [1, 2, 3], frontier + [4, 5, 6]]
    shared_plan = _plan(shared_prompts)

    hit_batches = _run(hit_plan, hit_prompts, _pool(16))
    shared_batches = _run(shared_plan, shared_prompts, _pool(16))

    stage = next(
        stage for batch in hit_batches for stage in batch.materialized
    )
    computed = next(
        (batch, row)
        for batch in shared_batches
        for row, item in enumerate(batch.items)
        if item.kind == "segment"
    )
    batch, row = computed
    start = int(batch.cu_seqlens_q[row])
    end = int(batch.cu_seqlens_q[row + 1])
    assert stage.tokens == end - start
    assert stage.slots.tolist() == batch.slots[start:end].tolist()
    assert stage.segment_id == 0
    assert stage.pages == tuple(batch.page_table[row].tolist()[:len(stage.pages)])


def test_a_loaded_segment_is_not_a_row_of_the_forward():
    prompts = [_hit_prompt(1, 3, 5)]
    plan = _plan(prompts, hits=[MaterializedPrefix(index=0, tokens=3 * B)])
    batches = _run(plan, prompts, _pool(16))
    for batch in batches:
        assert all(item.kind != "materialize" for item in batch.items)
        assert len(batch.row_bounds) == len(batch.items) + 1
        assert int(batch.cu_seqlens_q[-1]) == int(batch.input_ids.numel())
        assert batch.max_seqlen_q == max(item.tokens for item in batch.items)
    assert sum(len(batch.materialized) for batch in batches) == 1


def test_the_pool_gives_the_loaded_pages_back_with_every_other_page():
    prompts = [_hit_prompt(1, 3, 5), _frontier(2, 2) + [1, 2, 3]]
    plan = _plan(prompts, hits=[MaterializedPrefix(index=0, tokens=3 * B)])
    pool = _pool(16)
    _run(plan, prompts, pool)
    assert pool.free_pages == pool.num_pages


def test_flag_off_plans_and_chunks_are_byte_identical():
    prompts = [_frontier(1, 3) + [1, 2, 3], _frontier(1, 3) + [4, 5, 6]]
    with_arg = _plan(prompts, hits=())
    without_arg = plan_wave_prefix_sharing(
        prompts, block_tokens=B, pool_pages=64, chunk_tokens=CHUNK,
        same_chunk_ancestors=True,
    )
    assert with_arg == without_arg
    assert with_arg.materialized_tokens == 0
    assert with_arg.materialized_segments == ()
    pool_a, pool_b = _pool(16), _pool(16)
    a = _run(with_arg, prompts, pool_a)
    b = _run(without_arg, prompts, pool_b)
    assert len(a) == len(b)
    for left, right in zip(a, b):
        assert left.items == right.items
        assert left.materialized == () == right.materialized
        assert torch.equal(left.slots, right.slots)
        assert torch.equal(left.page_table, right.page_table)
        assert torch.equal(left.input_ids, right.input_ids)


# --------------------------------------------------------------------------- #
#  3. Named refusals
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("tokens", [0, -B, B + 1, 2 * B - 1])
def test_a_boundary_off_the_block_grid_is_refused(tokens):
    prompts = [_hit_prompt(1, 4, 5)]
    with pytest.raises(ValueError, match="multiple of the"):
        _plan(prompts, hits=[MaterializedPrefix(index=0, tokens=tokens)])


def test_a_frontier_that_swallows_the_last_prompt_token_is_refused():
    prompts = [_frontier(1, 3)]  # prompt == frontier, nothing left to compute
    with pytest.raises(ValueError, match="must stay"):
        _plan(prompts, hits=[MaterializedPrefix(index=0, tokens=3 * B)])


def test_two_frontiers_for_one_prompt_are_refused():
    prompts = [_hit_prompt(1, 3, 5)]
    with pytest.raises(ValueError, match="two materialized prefixes"):
        _plan(
            prompts,
            hits=[
                MaterializedPrefix(index=0, tokens=B),
                MaterializedPrefix(index=0, tokens=2 * B),
            ],
        )


def test_a_prompt_outside_the_wave_is_refused():
    prompts = [_hit_prompt(1, 3, 5)]
    with pytest.raises(ValueError, match="not one of this wave"):
        _plan(prompts, hits=[MaterializedPrefix(index=3, tokens=B)])


def test_a_pool_that_cannot_read_its_own_chunk_is_refused():
    prompts = [_hit_prompt(1, 3, 5)]
    with pytest.raises(ValueError, match="same_chunk_ancestors"):
        plan_wave_prefix_sharing(
            prompts, block_tokens=B, pool_pages=64, chunk_tokens=CHUNK,
            materialized=[MaterializedPrefix(index=0, tokens=3 * B)],
        )


def test_a_frontier_too_large_for_the_pool_falls_back_by_name():
    prompts = [_hit_prompt(1, 8, 5)]
    with pytest.raises(PoolMaterializationDoesNotFit, match="do not schedule"):
        _plan(prompts, hits=[MaterializedPrefix(index=0, tokens=8 * B)], pool_pages=4)


def test_the_same_wave_plans_without_materialization_after_the_refusal():
    """The fallback: the sequence still completes, as a full recompute."""

    prompts = [_hit_prompt(1, 8, 5)]
    with pytest.raises(PoolMaterializationDoesNotFit):
        _plan(prompts, hits=[MaterializedPrefix(index=0, tokens=8 * B)], pool_pages=4)
    plan = _plan(prompts, hits=(), pool_pages=4)
    validate_wave_prefix_plan(plan, prompts)
    assert plan.materialized_segments == ()
    assert plan.computed_tokens == len(prompts[0])
    tail = plan.chunks[0].items[0]
    assert (tail.kind, tail.token_start, tail.prefix_segments) == ("tail", 0, ())


# --------------------------------------------------------------------------- #
#  4. What the validator refuses
# --------------------------------------------------------------------------- #


def _hand_plan(segments, chunks, prompts, **kwargs):
    return WavePrefixPlan(
        block_tokens=B,
        pool_pages=64,
        chunk_tokens=CHUNK,
        threshold=2,
        segments=segments,
        chunks=chunks,
        prompt_tokens=sum(len(prompt) for prompt in prompts),
        computed_tokens=kwargs.pop("computed_tokens"),
        peak_pool_pages=kwargs.pop("peak_pool_pages"),
        same_chunk_ancestors=True,
        materialized_tokens=kwargs.pop("materialized_tokens"),
    )


def test_the_validator_refuses_a_loaded_segment_under_a_parent():
    prompts = [_hit_prompt(1, 2, 3)]
    segments = (
        PlanSegment(0, None, 0, B, (0,), materialized=False),
        PlanSegment(1, 0, B, 2 * B, (0,), materialized=True),
    )
    chunks = (
        PlanChunk(
            items=(
                PlanItem("segment", 0, (), 0, B),
                PlanItem("materialize", 1, (0,), B, 2 * B),
                PlanItem("tail", 0, (0, 1), 2 * B, len(prompts[0])),
            ),
            pool_pages_in_use=2,
            release_after=(0, 1),
        ),
    )
    plan = _hand_plan(
        segments, chunks, prompts,
        computed_tokens=B + 3, peak_pool_pages=2, materialized_tokens=B,
    )
    with pytest.raises(AssertionError):
        validate_wave_prefix_plan(plan, prompts)


def test_the_validator_refuses_a_chunk_of_loads_with_no_forward():
    prompts = [_hit_prompt(1, 2, 3)]
    segments = (PlanSegment(0, None, 0, 2 * B, (0,), materialized=True),)
    chunks = (
        PlanChunk(
            items=(PlanItem("materialize", 0, (), 0, 2 * B),),
            pool_pages_in_use=2,
            release_after=(),
        ),
        PlanChunk(
            items=(PlanItem("tail", 0, (0,), 2 * B, len(prompts[0])),),
            pool_pages_in_use=0,
            release_after=(0,),
        ),
    )
    plan = _hand_plan(
        segments, chunks, prompts,
        computed_tokens=3, peak_pool_pages=2, materialized_tokens=2 * B,
    )
    with pytest.raises(AssertionError, match="forward its loads ride"):
        validate_wave_prefix_plan(plan, prompts)


def test_the_validator_refuses_a_mislabelled_item():
    prompts = [_hit_prompt(1, 2, 3)]
    segments = (PlanSegment(0, None, 0, 2 * B, (0,), materialized=True),)
    chunks = (
        PlanChunk(
            items=(
                PlanItem("segment", 0, (), 0, 2 * B),  # claims it computed it
                PlanItem("tail", 0, (0,), 2 * B, len(prompts[0])),
            ),
            pool_pages_in_use=2,
            release_after=(0,),
        ),
    )
    plan = _hand_plan(
        segments, chunks, prompts,
        computed_tokens=2 * B + 3, peak_pool_pages=2, materialized_tokens=2 * B,
    )
    with pytest.raises(AssertionError):
        validate_wave_prefix_plan(plan, prompts)


def test_the_executor_refuses_a_load_only_chunk():
    prompts = [_hit_prompt(1, 2, 3)]
    segments = (PlanSegment(0, None, 0, 2 * B, (0,), materialized=True),)
    chunk = PlanChunk(
        items=(PlanItem("materialize", 0, (), 0, 2 * B),),
        pool_pages_in_use=2,
        release_after=(),
    )
    plan = _hand_plan(
        segments, (chunk,), prompts,
        computed_tokens=0, peak_pool_pages=2, materialized_tokens=2 * B,
    )
    executor = PoolChunkExecutor(plan, prompts, _pool(16))
    with pytest.raises(RuntimeError, match="no row to run them with"):
        executor.build(chunk)


# --------------------------------------------------------------------------- #
#  5. Recurrent state: no fork from a segment nothing ran
# --------------------------------------------------------------------------- #


def _state_inputs(batch, *, restored_rows=()):
    row_ids = [
        -100 - item.ref if item.kind == "segment" else 1000 + item.ref
        for item in batch.items
    ]
    slots = torch.arange(len(batch.items), dtype=torch.long)
    has_initial = torch.tensor(
        [row in set(restored_rows) for row in range(len(batch.items))],
        dtype=torch.bool,
    )
    return row_ids, slots, has_initial


def test_a_loaded_parent_contributes_no_fork():
    prompts = [_hit_prompt(1, 3, 5)]
    plan = _plan(prompts, hits=[MaterializedPrefix(index=0, tokens=3 * B)])
    batch = _run(plan, prompts, _pool(16))[0]
    assert fork_pairs(batch, {}, plan.materialized_segments) == ()
    # Without being told, the same cut cannot resolve the parent at all.
    with pytest.raises(RuntimeError, match="no pseudo sequence id"):
        fork_pairs(batch, {})


def test_the_reader_of_a_loaded_segment_must_already_hold_its_state():
    prompts = [_hit_prompt(1, 3, 5)]
    plan = _plan(prompts, hits=[MaterializedPrefix(index=0, tokens=3 * B)])
    batch = _run(plan, prompts, _pool(16))[0]
    row_ids, slots, has_initial = _state_inputs(batch)  # nothing restored
    with pytest.raises(RuntimeError, match="holds no recurrent state"):
        state_levels(
            batch,
            segment_sids={},
            row_ids=row_ids,
            slots=slots,
            has_initial_state=has_initial,
            slot_of=lambda row_id: 0,
            materialized=plan.materialized_segments,
        )


def test_a_restored_reader_runs_with_its_own_state_and_no_fork():
    prompts = [_hit_prompt(1, 3, 5)]
    plan = _plan(prompts, hits=[MaterializedPrefix(index=0, tokens=3 * B)])
    batch = _run(plan, prompts, _pool(16))[0]
    hit_row = next(
        row for row, item in enumerate(batch.items)
        if item.kind == "tail" and item.ref == 0
    )
    row_ids, slots, has_initial = _state_inputs(batch, restored_rows=(hit_row,))
    levels = state_levels(
        batch,
        segment_sids={},
        row_ids=row_ids,
        slots=slots,
        has_initial_state=has_initial,
        slot_of=lambda row_id: 0,
        materialized=plan.materialized_segments,
    )
    assert all(level.fork_parents is None for level in levels)
    assert all(level.fork_children is None for level in levels)
    resumed = torch.cat([level.has_initial_state for level in levels])
    assert bool(resumed[hit_row])


def test_a_computed_parent_still_forks():
    prompts = [_frontier(1, 3) + [1, 2, 3], _frontier(1, 3) + [4, 5, 6]]
    plan = _plan(prompts)
    batch = _run(plan, prompts, _pool(16))[0]
    sids = {seg.segment_id: -50 - seg.segment_id for seg in plan.segments}
    pairs = fork_pairs(batch, sids, plan.materialized_segments)
    assert pairs  # the computed segment is forked into its tails
    assert {sid for _, sid in pairs} == {-50}


# --------------------------------------------------------------------------- #
#  6. Host side: allocate, mirror, publish only what this wave computed
# --------------------------------------------------------------------------- #


class _RecordingView:
    """A Host-KV worker view that records what a wave asks of it."""

    def __init__(self, group_id, first_page=100):
        self.group_id = group_id
        self.registered: list[int] = []
        self.allocated: list[tuple[int, int]] = []
        self.set_pages: list[tuple[int, list[int]]] = []
        self._next = first_page
        self._tables: dict[int, list[int]] = {}

    def register_sequences(self, sids):
        self.registered.extend(int(sid) for sid in sids)

    def allocate_pages_for_sequences(self, requests):
        for sid, tokens in requests:
            pages = int(tokens) // B
            self._tables[int(sid)] = list(range(self._next, self._next + pages))
            self._next += pages
            self.allocated.append((int(sid), int(tokens)))

    def build_page_table(self, sids):
        return [self._tables[int(sid)] for sid in sids]

    def set_sequence_pages(self, sid, pages):
        self.set_pages.append((int(sid), [int(page) for page in pages]))


def _wave(prompts, plan):
    return PoolWave(
        plan=plan,
        prompts=[list(prompt) for prompt in prompts],
        uuids=[f"u{i}" for i in range(len(prompts))],
        global_ids=[10 + i for i in range(len(prompts))],
        page_tokens=B,
        total_pages=64,
    )


def _hit_wave():
    """One hit beside a shared pair, so a wave carries both segment kinds."""

    shared = _frontier(2, 2)
    prompts = [
        _hit_prompt(1, 3, 5),
        shared + [1, 2, 3],
        shared + [4, 5, 6],
    ]
    plan = _plan(prompts, hits=[MaterializedPrefix(index=0, tokens=3 * B)])
    assert [seg.materialized for seg in plan.segments] == [True, False]
    return prompts, plan, _wave(prompts, plan)


def test_the_wave_allocates_no_host_page_for_a_loaded_segment():
    prompts, plan, wave = _hit_wave()
    wave.bind_materialized_host_pages(0, {0: [7, 8, 9]})
    view = _RecordingView(0)
    sids = iter([-11, -12, -13])
    total = allocate_segment_host_pages(wave, {0: view}, lambda: next(sids))

    computed = [seg for seg in plan.segments if not seg.materialized]
    assert total == sum(seg.tokens // B for seg in computed)
    assert sorted(wave.segment_sids) == [seg.segment_id for seg in computed]
    assert len(view.registered) == len(computed)
    assert wave.segment_host_pages[0] == {0: [7, 8, 9]}  # untouched


def test_an_unbound_loaded_segment_stops_the_allocation_by_name():
    _, _, wave = _hit_wave()
    with pytest.raises(RuntimeError, match="have no Host pages bound"):
        allocate_segment_host_pages(wave, {0: _RecordingView(0)}, lambda: -11)


def test_binding_checks_the_page_count_against_the_geometry():
    _, _, wave = _hit_wave()
    with pytest.raises(RuntimeError, match="Host page"):
        wave.bind_materialized_host_pages(0, {0: [7, 8]})
    with pytest.raises(RuntimeError, match="no Host pages"):
        wave.bind_materialized_host_pages(0, {})


def test_a_computed_segment_cannot_be_bound_as_a_load_source():
    prompts, plan, wave = _hit_wave()
    computed = next(seg for seg in plan.segments if not seg.materialized)
    with pytest.raises(RuntimeError, match="not materialized in this wave"):
        wave.bind_materialized_host_pages(computed.segment_id, {0: [1, 2]})


def test_the_mirror_carries_the_loaded_pages_as_a_read_source():
    prompts, plan, wave = _hit_wave()
    wave.bind_materialized_host_pages(0, {0: [7, 8, 9]})
    view = _RecordingView(0)
    sids = iter([-11, -12, -13])
    allocate_segment_host_pages(wave, {0: view}, lambda: next(sids))
    payload = segment_mirror_payload(wave)

    assert payload["materialized_host_pages"] == {0: {0: [7, 8, 9]}}
    assert 0 not in payload["segment_host_pages"]
    assert 0 not in payload["segment_sids"]

    follower_view = _RecordingView(0)
    follower = _wave(prompts, plan)
    install_mirrored_segments(follower, payload, {0: follower_view})
    assert follower.segment_host_pages[0] == {0: [7, 8, 9]}
    # A follower registers only the segments a row of it writes.
    assert [sid for sid, _ in follower_view.set_pages] == sorted(
        payload["segment_sids"].values()
    )


def test_a_mirror_without_the_loaded_pages_is_refused():
    prompts, plan, wave = _hit_wave()
    wave.bind_materialized_host_pages(0, {0: [7, 8, 9]})
    sids = iter([-11, -12, -13])
    allocate_segment_host_pages(wave, {0: _RecordingView(0)}, lambda: next(sids))
    payload = segment_mirror_payload(wave)
    payload["materialized_host_pages"] = {}
    with pytest.raises(RuntimeError, match="missing from the mirror payload"):
        install_mirrored_segments(_wave(prompts, plan), payload, {0: _RecordingView(0)})


def test_a_loaded_segment_is_not_published_again():
    from batchgen.prefix_reuse.wave_runtime import commit_pool_segments

    prompts, plan, wave = _hit_wave()
    wave.bind_materialized_host_pages(0, {0: [7, 8, 9]})
    sids = iter([-11, -12, -13])
    allocate_segment_host_pages(wave, {0: _RecordingView(0)}, lambda: next(sids))

    committed: list[int] = []

    class _Coordinator:
        def lookup_and_attach(self, digest, tokens):
            return SimpleNamespace(
                common_cached_tokens=len(tokens), attachment_handle=5
            )

        def release_attachment(self, handle):
            pass

    import batchgen.prefix_reuse.wave_runtime as runtime

    class _Request:
        def __init__(self, tokens):
            self.commit_tokens = tokens

    def fake_build(*, namespace_digest, token_ids, publish_boundary_tokens,
                   pages_by_group):
        committed.append(len(token_ids))
        return _Request(len(token_ids))

    def fake_commit(*, request, coordinator, worker_views_by_group, max_scan_nodes):
        return SimpleNamespace(
            commit_result=SimpleNamespace(
                active_attachment_handle=3, inserted_nodes=1
            )
        )

    import batchgen.prefix_reuse.commit as commit_mod
    import batchgen.prefix_reuse.eviction as eviction_mod

    old_build = commit_mod.build_prefix_commit_request
    old_retain = commit_mod.retain_inserted_prefix_pages
    old_commit = eviction_mod.commit_prefix_pages_with_capacity_retry
    commit_mod.build_prefix_commit_request = fake_build
    commit_mod.retain_inserted_prefix_pages = lambda **kwargs: None
    eviction_mod.commit_prefix_pages_with_capacity_retry = fake_commit
    try:
        commit_pool_segments(
            wave,
            coordinator=_Coordinator(),
            worker_views_by_group={0: _RecordingView(0)},
            namespace_digest=(1, 2),
            publish_boundary_tokens=B,
            max_scan_nodes=16,
        )
    finally:
        commit_mod.build_prefix_commit_request = old_build
        commit_mod.retain_inserted_prefix_pages = old_retain
        eviction_mod.commit_prefix_pages_with_capacity_retry = old_commit
        del runtime

    computed = [seg for seg in plan.segments if not seg.materialized]
    assert committed == [seg.token_end for seg in computed]


# --------------------------------------------------------------------------- #
#  7. The H2D loader
# --------------------------------------------------------------------------- #


class _FakeHostView:
    """Captures each load and reports it complete only once waited on."""

    def __init__(self):
        self.calls: list[dict] = []
        self.waits = 0

    def async_load_prefix_pages_to_device(
        self, *, host_page_ids, active_page_counts, k_device_ptrs, v_device_ptrs
    ):
        self.calls.append({
            "host_page_ids": host_page_ids.clone(),
            "active_page_counts": active_page_counts.clone(),
            "k_device_ptrs": k_device_ptrs.clone(),
            "v_device_ptrs": v_device_ptrs,
            "waited_at_yield": None,
        })
        view = self

        class _Task:
            def wait(self_inner):
                view.waits += 1

        return _Task()


def _loader(view, *, layers=3, latent=8, slab_bytes=None):
    element = torch.empty((), dtype=torch.float32).element_size()
    page_bytes = B * latent * element
    if slab_bytes is None:
        slab_bytes = page_bytes * layers * 2  # two pages per slab
    return MaterializedSegmentLoader(
        worker_view=view,
        num_layers=layers,
        page_tokens=B,
        latent_dim=latent,
        dtype=torch.float32,
        device="cpu",
        max_slab_bytes=slab_bytes,
    )


def test_the_load_is_one_task_per_slab_over_every_layer():
    view = _FakeHostView()
    loader = _loader(view)
    assert loader.slab_pages == 2
    slabs = list(loader.slabs(host_page_ids=[11, 12, 13, 14, 15], tokens=5 * B))

    assert [slab.token_start for slab in slabs] == [0, 2 * B, 4 * B]
    assert [slab.tokens for slab in slabs] == [2 * B, 2 * B, B]
    assert [call["host_page_ids"].tolist() for call in view.calls] == [
        [[11, 12]], [[13, 14]], [[15]],
    ]
    assert [int(call["active_page_counts"][0]) for call in view.calls] == [2, 2, 1]
    assert view.waits == 3  # every slab is waited on before it is handed over
    for call in view.calls:
        assert call["v_device_ptrs"] is None
        assert call["k_device_ptrs"].dtype == torch.int64
        assert call["k_device_ptrs"].device.type == "cpu"
        assert call["k_device_ptrs"].shape[:2] == (3, 1)


def test_every_destination_pointer_addresses_its_own_staging_row():
    view = _FakeHostView()
    loader = _loader(view, layers=3, latent=8)
    slabs = list(loader.slabs(host_page_ids=[11, 12], tokens=2 * B))
    staging = slabs[0].latent
    element = staging.element_size()
    page_bytes = B * 8 * element
    base = staging.data_ptr()
    pointers = view.calls[0]["k_device_ptrs"]
    for layer in range(3):
        for page in range(2):
            # The destination of (layer, page) is where this slab's view says
            # those tokens live, so an expansion reads what the copy wrote.
            expected = staging[layer, page * B, :].data_ptr()
            assert int(pointers[layer, 0, page]) == expected
            assert int(pointers[layer, 0, page]) == (
                base + layer * 2 * page_bytes + page * page_bytes
            )


def test_the_slab_view_is_the_tokens_of_that_slab():
    view = _FakeHostView()
    loader = _loader(view, layers=2, latent=4)
    slabs = list(loader.slabs(host_page_ids=[1, 2, 3], tokens=3 * B))
    assert [tuple(slab.latent.shape) for slab in slabs] == [
        (2, 2 * B, 4), (2, B, 4),
    ]


def test_a_page_count_that_contradicts_the_token_count_is_refused():
    loader = _loader(_FakeHostView())
    with pytest.raises(ValueError, match="Host page"):
        list(loader.slabs(host_page_ids=[1, 2], tokens=3 * B))
    with pytest.raises(ValueError, match="page"):
        list(loader.slabs(host_page_ids=[1, 2], tokens=2 * B + 1))


def test_a_staging_budget_below_one_page_is_refused():
    with pytest.raises(ValueError, match="holds no page"):
        _loader(_FakeHostView(), slab_bytes=8)


# --------------------------------------------------------------------------- #
#  8. The model's expansion
# --------------------------------------------------------------------------- #


class _MLAModule(torch.nn.Module):
    """The projection geometry ``mla_prefill_nope_prepacked`` expands with."""

    def __init__(self, *, heads=2, nope=6, rope=2, v_dim=4, lora=8):
        super().__init__()
        self.num_heads = heads
        self.qk_nope_head_dim = nope
        self.qk_rope_head_dim = rope
        self.v_head_dim = v_dim
        self.kv_lora_rank = lora
        self.q_head_dim = nope + rope
        self.kv_b_proj = torch.nn.Linear(
            lora, heads * (nope + v_dim), bias=False, dtype=torch.float32
        )
        self.calls = 0

    def forward(self, *args, **kwargs):  # pragma: no cover - never called
        raise AssertionError("the module is used through its projections only")


def _expansion_pool(module, pages):
    return PrefillPrefixPool(
        num_layers=2,
        num_pages=pages,
        page_tokens=B,
        num_kv_heads=module.num_heads,
        head_dim=module.q_head_dim,
        v_head_dim=module.v_head_dim,
        dtype=torch.float32,
        device="cpu",
    )


def _expand_like_the_forward(module, latent):
    """What ``mla_prefill_nope_prepacked`` writes for the same latent."""

    total = latent.shape[0]
    normed_kv, k_pe = torch.split(
        latent, [module.kv_lora_rank, module.qk_rope_head_dim], dim=-1
    )
    kv = module.kv_b_proj(normed_kv)
    kv = kv.view(total, module.num_heads, module.qk_nope_head_dim + module.v_head_dim)
    k_nope, value_states = torch.split(
        kv, [module.qk_nope_head_dim, module.v_head_dim], dim=-1
    )
    key_states = kv.new_empty(total, module.num_heads, module.q_head_dim)
    key_states[:, :, : module.qk_nope_head_dim] = k_nope
    key_states[:, :, module.qk_nope_head_dim :] = k_pe.view(
        total, 1, module.qk_rope_head_dim
    )
    return key_states.contiguous(), value_states.contiguous()


def test_the_expansion_writes_what_computing_the_same_latent_writes():
    from batchgen.models.moonshotai.kimi_linear.serving_modules import (
        mla_materialize_pool_segment,
    )

    torch.manual_seed(0)
    module = _MLAModule()
    pool = _expansion_pool(module, 8)
    latent = torch.randn(
        2 * B, module.kv_lora_rank + module.qk_rope_head_dim, dtype=torch.float32
    )
    pages = pool.alloc(2)
    slots = pool.slots(pages, 0, 2 * B)
    # The pool is uninitialised memory, which can hold NaN -- zero it so
    # "untouched" is a comparison and not a NaN lottery.
    pool.k.zero_()
    pool.v.zero_()

    written = mla_materialize_pool_segment(
        module, latent, pool=pool, pool_layer=1, slots=slots
    )
    assert written == 2 * B

    key, value = _expand_like_the_forward(module, latent)
    k_cache = pool.k[1].view(-1, *pool.k.shape[-2:])
    v_cache = pool.v[1].view(-1, *pool.v.shape[-2:])
    assert torch.equal(k_cache.index_select(0, slots), key)
    assert torch.equal(v_cache.index_select(0, slots), value)
    # Only the addressed layer is touched.
    assert torch.count_nonzero(pool.k[0]) == 0
    assert torch.count_nonzero(pool.v[0]) == 0


def test_the_expansion_goes_through_the_layers_own_projection_object():
    from batchgen.models.moonshotai.kimi_linear.serving_modules import (
        mla_materialize_pool_segment,
    )

    module = _MLAModule()
    pool = _expansion_pool(module, 4)
    seen: list[object] = []
    real = module.kv_b_proj

    class _Probe(torch.nn.Module):
        def forward(self_inner, x):
            seen.append(real)
            return real(x)

    module.kv_b_proj = _Probe()
    latent = torch.randn(B, module.kv_lora_rank + module.qk_rope_head_dim)
    pages = pool.alloc(1)
    mla_materialize_pool_segment(
        module, latent, pool=pool, pool_layer=0,
        slots=pool.slots(pages, 0, B),
    )
    assert seen == [real]  # the module's own weights, not a re-implementation


def test_a_latent_of_the_wrong_width_is_refused():
    from batchgen.models.moonshotai.kimi_linear.serving_modules import (
        mla_materialize_pool_segment,
    )

    module = _MLAModule()
    pool = _expansion_pool(module, 4)
    pages = pool.alloc(1)
    with pytest.raises(ValueError, match="wide"):
        mla_materialize_pool_segment(
            module, torch.randn(B, 3), pool=pool, pool_layer=0,
            slots=pool.slots(pages, 0, B),
        )


def test_a_slot_count_that_does_not_match_the_tokens_is_refused():
    from batchgen.models.moonshotai.kimi_linear.serving_modules import (
        mla_materialize_pool_segment,
    )

    module = _MLAModule()
    pool = _expansion_pool(module, 4)
    pages = pool.alloc(1)
    latent = torch.randn(B, module.kv_lora_rank + module.qk_rope_head_dim)
    with pytest.raises(ValueError, match="pool slot"):
        mla_materialize_pool_segment(
            module, latent, pool=pool, pool_layer=0,
            slots=pool.slots(pages, 0, B - 1),
        )


def test_the_loaded_slabs_of_one_segment_fill_it_exactly_once():
    """Loader slabs and expansion together cover a segment's slots, in order."""

    from batchgen.models.moonshotai.kimi_linear.serving_modules import (
        mla_materialize_pool_segment,
    )

    torch.manual_seed(1)
    module = _MLAModule()
    pool = _expansion_pool(module, 8)
    pages = pool.alloc(3)
    slots = pool.slots(pages, 0, 3 * B)

    view = _FakeHostView()
    loader = MaterializedSegmentLoader(
        worker_view=view,
        num_layers=2,
        page_tokens=B,
        latent_dim=module.kv_lora_rank + module.qk_rope_head_dim,
        dtype=torch.float32,
        device="cpu",
        max_slab_bytes=(
            B * (module.kv_lora_rank + module.qk_rope_head_dim) * 4 * 2 * 2
        ),
    )
    covered: list[int] = []
    for slab in loader.slabs(host_page_ids=[1, 2, 3], tokens=3 * B):
        window = slots[slab.token_start:slab.token_start + slab.tokens]
        mla_materialize_pool_segment(
            module, slab.latent[0], pool=pool, pool_layer=0, slots=window
        )
        covered.extend(window.tolist())
    assert covered == slots.tolist()
