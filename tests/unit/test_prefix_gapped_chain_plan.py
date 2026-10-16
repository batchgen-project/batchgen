"""Oracle check of a gapped nested-chain family through the prefill pool.

A chain corpus is one nested family: member ``k + 1`` is member ``k`` plus one
page-aligned block of new tokens. When the admission balancer splits such a
family across two serve groups, each group pools a chain WITH GAPS - a
member's nearest ancestor may live in the other group - and consecutive blocks
then carry equal prompt counts, so the planner compresses them into segments
whose stride is several pages instead of one.

Every assertion here compares against an oracle derived from the synthetic
construction alone, never against the planner's own output:

* the token at absolute position ``p`` of every member is
  ``ORACLE_TOKEN[p]``, because each member is a prefix of the full chain;
* ``oracle_segments`` recomputes the block runs of equal prompt count from the
  prompt lengths, independently of ``_build_tree``.

Covered splits: the odd half and the even half of a 64-member family (both
gapped, 2-page strides), a contiguous 32-prefix and the full 64 (both gap-free
controls, 1-page strides).
"""

from __future__ import annotations

import pytest
import torch

from batchgen.prefix_reuse.executor import PoolChunkExecutor
from batchgen.prefix_reuse.pool import PrefillPrefixPool, scratch_pages_bound
from batchgen.prefix_reuse.wave_plan import (
    plan_wave_prefix_sharing,
    validate_wave_prefix_plan,
)
from batchgen.prefix_reuse.wave_runtime import PoolWave, allocate_segment_host_pages

PAGE = 64  # block_tokens == page_tokens
FAMILY = 64  # members, so the deepest prompt is 4096 tokens
POOL_PAGES = 512  # generous: every shared segment must be poolable
CHUNK_TOKENS = 8192


def block_of(index: int) -> list[int]:
    """The ``index``-th 64-token block; token values are distinct per block."""

    base = 50_000 + index * 1_000
    return [base + offset for offset in range(PAGE)]


# Absolute position -> token id. Member k is ORACLE_TOKEN[:(k + 1) * PAGE].
ORACLE_TOKEN = [token for index in range(FAMILY) for token in block_of(index)]


def member_prompt(member: int) -> list[int]:
    return ORACLE_TOKEN[: (member + 1) * PAGE]


def oracle_segments(members: list[int]) -> list[tuple[int, int, tuple[int, ...]]]:
    """Block runs of equal prompt count, recomputed from the lengths alone.

    The last block of a prompt is private, so member ``k`` passes through
    blocks ``[0, (len - 1) // PAGE)``. Counts fall as shorter members drop
    out; a run of blocks with the same member set is one segment.
    """

    through = {
        member: (len(member_prompt(member)) - 1) // PAGE for member in members
    }
    runs: list[list] = []
    block = 0
    while True:
        here = tuple(sorted(m for m in members if through[m] > block))
        if not here:
            break
        if runs and runs[-1][2] == here:
            runs[-1][1] = block + 1
        else:
            runs.append([block, block + 1, here])
        block += 1
    return [(start, end, here) for start, end, here in runs]


def oracle_pooled(members: list[int]) -> list[tuple[int, int, tuple[int, ...]]]:
    """The segments a generous pool shares: every run of two or more members."""

    runs = oracle_segments(members)
    pooled = [run for run in runs if len(run[2]) >= 2]
    # Counts never increase down the chain, so the pooled set is a prefix of
    # the chain and therefore closed under ancestors.
    assert pooled == runs[: len(pooled)]
    return pooled


class FakeWorkerView:
    """Host KV page allocator: hands out one distinct page per PAGE tokens."""

    def __init__(self, first_page: int) -> None:
        self.next_page = first_page
        self.tables: dict[int, list[int]] = {}

    def register_sequences(self, ids) -> None:
        for sid in ids:
            assert sid not in self.tables, f"sequence {sid} registered twice"
            self.tables[sid] = []

    def allocate_pages_for_sequences(self, requests) -> None:
        for sid, tokens in requests:
            for _ in range(-(-tokens // PAGE)):
                self.tables[sid].append(self.next_page)
                self.next_page += 1

    def build_page_table(self, ids):
        return [list(self.tables[sid]) for sid in ids]


SPLITS = {
    # Gapped: the L2 balancer sends every other member to the other group.
    "odd-32-gapped": list(range(1, FAMILY, 2)),
    "even-32-gapped": list(range(0, FAMILY, 2)),
    # Gap-free controls.
    "prefix-32-contiguous": list(range(32)),
    "full-64-single-group": list(range(FAMILY)),
}


def plan_for(members: list[int]):
    prompts = [member_prompt(member) for member in members]
    plan = plan_wave_prefix_sharing(
        prompts,
        block_tokens=PAGE,
        pool_pages=POOL_PAGES,
        chunk_tokens=CHUNK_TOKENS,
    )
    return prompts, plan


def host_wave(members: list[int], prompts, plan) -> PoolWave:
    wave = PoolWave(
        plan=plan,
        prompts=[list(prompt) for prompt in prompts],
        uuids=[f"m{member:02d}" for member in members],
        global_ids=list(range(len(members))),
        page_tokens=PAGE,
        total_pages=POOL_PAGES,
    )
    sids = iter(range(-1, -1 - len(plan.segments), -1))
    allocate_segment_host_pages(wave, {0: FakeWorkerView(10_000)}, lambda: next(sids))
    return wave


@pytest.mark.parametrize("split", sorted(SPLITS))
def test_plan_matches_the_oracle_segment_runs(split):
    """Segment ids, bounds and member sets equal the independently derived runs."""

    members = SPLITS[split]
    prompts, plan = plan_for(members)
    validate_wave_prefix_plan(plan, prompts)
    # A generous pool must share every shared segment; otherwise the oracle
    # below (which assumes threshold 2) does not describe this plan.
    assert plan.threshold == 2, f"{split}: pool budget forced threshold up"

    pooled = oracle_pooled(members)
    wave_index = {member: index for index, member in enumerate(members)}
    assert len(plan.segments) == len(pooled), (
        f"{split}: planner made {len(plan.segments)} segments, "
        f"oracle has {len(pooled)}"
    )
    for position, (segment, (start, end, here)) in enumerate(
        zip(plan.segments, pooled)
    ):
        context = f"{split} segment {position}"
        assert segment.segment_id == position, context
        # One linear chain: each segment's parent is the segment before it.
        assert segment.parent_id == (None if position == 0 else position - 1), context
        assert (segment.token_start, segment.token_end) == (
            start * PAGE,
            end * PAGE,
        ), (
            f"{context}: planner [{segment.token_start}, {segment.token_end}) "
            f"!= oracle [{start * PAGE}, {end * PAGE})"
        )
        assert segment.members == tuple(wave_index[m] for m in here), context
        # The segment's tokens are the oracle tokens of that absolute range.
        for member in here:
            prompt = prompts[wave_index[member]]
            assert (
                prompt[segment.token_start:segment.token_end]
                == ORACLE_TOKEN[segment.token_start:segment.token_end]
            ), f"{context}: member {member} disagrees with the oracle tokens"


@pytest.mark.parametrize("split", sorted(SPLITS))
def test_chain_tokens_and_host_starts_meet_the_member_tails(split):
    """chain_tokens, segment page maps and tail offsets agree with the oracle."""

    members = SPLITS[split]
    prompts, plan = plan_for(members)
    validate_wave_prefix_plan(plan, prompts)
    wave = host_wave(members, prompts, plan)

    pooled = oracle_pooled(members)
    pooled_blocks = pooled[-1][1] if pooled else 0
    tails = {
        item.ref: item
        for chunk in plan.chunks
        for item in chunk.items
        if item.kind == "tail"
    }

    # Ascending member order, so the first failure is the shallowest member.
    for index, member in enumerate(members):
        context = f"{split} member {member} (wave index {index})"
        prompt_len = len(prompts[index])
        through = (prompt_len - 1) // PAGE

        # (a) chain_tokens is the shared prefix the pool actually holds for it.
        oracle_chain = PAGE * min(through, pooled_blocks)
        assert wave.chain_tokens(index) == oracle_chain, (
            f"{context}: chain_tokens {wave.chain_tokens(index)} "
            f"!= oracle {oracle_chain}"
        )

        # (b) the chain tiles [0, chain_tokens) exactly once, and every
        # segment's token range carries the oracle tokens.
        cursor = 0
        for segment_id in wave.chains[index]:
            segment = plan.segments[segment_id]
            assert segment.token_start == cursor, (
                f"{context}: segment {segment_id} starts at "
                f"{segment.token_start}, expected {cursor}"
            )
            assert index in segment.members, context
            assert (
                prompts[index][segment.token_start:segment.token_end]
                == ORACLE_TOKEN[segment.token_start:segment.token_end]
            ), f"{context}: segment {segment_id} tokens are not the oracle's"
            cursor = segment.token_end
        assert cursor == oracle_chain, (
            f"{context}: chain covers {cursor} tokens, oracle {oracle_chain}"
        )

        # (c) the tail starts exactly where the chain ends: no gap, no overlap.
        tail = tails[index]
        assert tail.token_start == oracle_chain, (
            f"{context}: tail starts at {tail.token_start}, "
            f"chain ends at {oracle_chain}"
        )
        assert tail.token_end == prompt_len, context
        assert (
            prompts[index][tail.token_start:tail.token_end]
            == ORACLE_TOKEN[tail.token_start:tail.token_end]
        ), f"{context}: tail tokens are not the oracle's"

        # (d) Host attach offset: the chain's Host pages must be exactly the
        # pages the tail's destination start skips over.
        chain_pages = wave.chain_host_pages(index)
        if oracle_chain:
            assert len(chain_pages[0]) == oracle_chain // PAGE, (
                f"{context}: {len(chain_pages[0])} chain Host pages for "
                f"{oracle_chain} tokens of {PAGE}"
            )
        else:
            assert chain_pages == {}, context

    # segment_prefix_pages includes the segment itself, so it is what a commit
    # of tokens[:token_end] needs: one page per page of that prefix.
    for segment in plan.segments:
        pages = wave.segment_prefix_pages(segment.segment_id)[0]
        assert len(pages) == segment.token_end // PAGE, (
            f"{split} segment {segment.segment_id}: {len(pages)} prefix Host "
            f"pages for token_end {segment.token_end}"
        )
        assert len(set(pages)) == len(pages), "a Host page is attached twice"


@pytest.mark.parametrize("split", sorted(SPLITS))
def test_executor_rows_cover_every_oracle_token_exactly_once(split):
    """Replay the chunk layout and re-derive each row's tokens from the oracle."""

    members = SPLITS[split]
    prompts, plan = plan_for(members)
    validate_wave_prefix_plan(plan, prompts)
    wave = host_wave(members, prompts, plan)

    scratch = scratch_pages_bound(
        CHUNK_TOKENS, max(len(prompt) for prompt in prompts), len(prompts), PAGE
    )
    num_pages = plan.pool_pages + scratch
    pool = PrefillPrefixPool(
        num_layers=1,
        num_pages=num_pages,
        page_tokens=PAGE,
        num_kv_heads=1,
        head_dim=2,
        dtype=torch.float32,
        device="cpu",
    )
    executor = PoolChunkExecutor(plan, prompts, pool)

    # Fake paged KV: one token id and one absolute position per slot.
    cached = [-1] * (num_pages * PAGE)
    cached_pos = [-1] * (num_pages * PAGE)
    # (wave index, absolute position) -> the item that computed it.
    producer: dict[tuple[int, int], tuple[str, int]] = {}
    tails_seen: list[int] = []
    computed = 0

    for position, chunk in enumerate(plan.chunks):
        batch = executor.build(chunk)
        # A layer writes the whole chunk's KV before attending over it.
        for slot, token, pos in zip(
            batch.slots.tolist(),
            batch.input_ids.tolist(),
            batch.position_ids.tolist(),
        ):
            cached[slot] = token
            cached_pos[slot] = pos
        computed += int(batch.input_ids.numel())

        host_ids, host_starts = wave.row_host_targets(batch)
        bounds = batch.row_bounds
        for row, item in enumerate(batch.items):
            context = f"{split} chunk {position} row {row} {item.kind} {item.ref}"

            # cache_seqlens is the oracle context length of this row.
            if item.kind == "tail":
                oracle_end = len(prompts[item.ref])
            else:
                oracle_end = plan.segments[item.ref].token_end
            assert int(batch.cache_seqlens[row]) == oracle_end, (
                f"{context}: cache_seqlens {int(batch.cache_seqlens[row])} "
                f"!= oracle context length {oracle_end}"
            )
            assert item.token_end == oracle_end, context

            # Walk the row's page table over [0, cache_seqlens) and compare
            # every key slot against the oracle token and position.
            pages = batch.page_table[row].tolist()
            length = int(batch.cache_seqlens[row])
            slots = [
                pages[offset // PAGE] * PAGE + offset % PAGE
                for offset in range(length)
            ]
            got = [cached[slot] for slot in slots]
            if got != ORACLE_TOKEN[:length]:
                first = next(
                    i for i in range(length) if got[i] != ORACLE_TOKEN[i]
                )
                pytest.fail(
                    f"{context}: key {first} (page {first // PAGE}, offset "
                    f"{first % PAGE}) holds token {got[first]}, oracle "
                    f"{ORACLE_TOKEN[first]}; item tokens "
                    f"[{item.token_start}, {item.token_end}), chain "
                    f"{item.prefix_segments}"
                )
            assert [cached_pos[slot] for slot in slots] == list(range(length)), (
                f"{context}: position ids are not 0..{length - 1}"
            )

            # Query rows carry exactly [token_start, token_end) of the oracle.
            start, end = bounds[row], bounds[row + 1]
            assert end - start == item.tokens, context
            assert (
                batch.input_ids[start:end].tolist()
                == ORACLE_TOKEN[item.token_start:item.token_end]
            ), f"{context}: query tokens are not the oracle's"

            # Host destination: segments write at 0 under their pseudo id,
            # a tail writes at its own offset under its global id.
            if item.kind == "segment":
                assert host_ids[row] == wave.segment_sids[item.ref], context
                assert host_starts[row] == 0, context
                reached = plan.segments[item.ref].members
            else:
                assert host_ids[row] == wave.global_ids[item.ref], context
                assert host_starts[row] == wave.chain_tokens(item.ref), (
                    f"{context}: Host start {host_starts[row]} != chain_tokens "
                    f"{wave.chain_tokens(item.ref)}"
                )
                reached = (item.ref,)

            # Attribute every token this row produced to its readers.
            for index in reached:
                for absolute in range(item.token_start, item.token_end):
                    key = (index, absolute)
                    assert key not in producer, (
                        f"{context}: wave index {index} position {absolute} "
                        f"already computed by {producer[key]}"
                    )
                    producer[key] = (item.kind, item.ref)

        tails_seen.extend(batch.tail_wave_indices)
        executor.finish(chunk)
    executor.close()

    assert sorted(tails_seen) == list(range(len(prompts)))
    assert computed == plan.computed_tokens

    # Full coverage: every token of every member computed-or-attached once.
    expected = {
        (index, absolute)
        for index, prompt in enumerate(prompts)
        for absolute in range(len(prompt))
    }
    assert set(producer) == expected, (
        f"{split}: {len(expected - set(producer))} member tokens are never "
        f"computed, {len(set(producer) - expected)} are out of range"
    )


def test_gapped_split_really_has_multi_page_segment_strides():
    """Guard the premise: the gapped halves differ structurally from the controls.

    Without this the oracle tests above could pass on a family that never
    exercises a segment longer than one page.
    """

    strides = {}
    for split, members in SPLITS.items():
        _, plan = plan_for(members)
        strides[split] = sorted({segment.tokens // PAGE for segment in plan.segments})
    assert strides["odd-32-gapped"] == [1, 2]
    assert strides["even-32-gapped"] == [2]
    assert strides["prefix-32-contiguous"] == [1]
    assert strides["full-64-single-group"] == [1]
