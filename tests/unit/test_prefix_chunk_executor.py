import importlib.util
import random
import sys
from pathlib import Path

import torch


def _load(name, *relative):
    path = Path(__file__).parents[2].joinpath(*relative)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_PLAN = _load("test_chunk_exec_plan_mod", "batchgen", "prefix_reuse", "wave_plan.py")
_POOL = _load("test_chunk_exec_pool_mod", "batchgen", "prefix_reuse", "pool.py")
_EXEC = _load("test_chunk_exec_mod", "batchgen", "prefix_reuse", "executor.py")

plan_wave_prefix_sharing = _PLAN.plan_wave_prefix_sharing
validate_wave_prefix_plan = _PLAN.validate_wave_prefix_plan
PrefillPrefixPool = _POOL.PrefillPrefixPool
scratch_pages_bound = _POOL.scratch_pages_bound
PoolChunkExecutor = _EXEC.PoolChunkExecutor

B = 4  # block tokens = page tokens


def blk(n):
    return [n * 10 + k for k in range(B)]


A = blk(1) + blk(2)
A1, A2 = blk(3), blk(4)
BB = blk(5) + blk(6)
TREE = [
    A + A1 + blk(70) + [900],
    A + A1 + blk(71) + [901],
    A + A2 + blk(72) + [902],
    A + A2 + blk(73) + [903],
    BB + blk(74) + [904],
    BB + blk(75) + [905],
    blk(76) + blk(77) + [906],
]


def item_prompt(plan, prompts, item):
    """Any prompt that item's tokens belong to; members agree on a segment."""

    if item.kind == "tail":
        return prompts[item.ref]
    return prompts[plan.segments[item.ref].members[0]]


def replay(prompts, pool_pages=1000, chunk_tokens=1000):
    """Run every chunk against a fake paged cache and check what it reads."""

    plan = plan_wave_prefix_sharing(
        prompts, block_tokens=B, pool_pages=pool_pages, chunk_tokens=chunk_tokens
    )
    validate_wave_prefix_plan(plan, prompts)
    scratch = scratch_pages_bound(
        chunk_tokens, max(len(p) for p in prompts), len(prompts), B
    )
    num_pages = plan.pool_pages + scratch
    pool = PrefillPrefixPool(
        num_layers=1,
        num_pages=num_pages,
        page_tokens=B,
        num_kv_heads=1,
        head_dim=2,
        dtype=torch.float32,
        device="cpu",
    )
    executor = PoolChunkExecutor(plan, prompts, pool)
    tok = torch.full((num_pages * B,), -1, dtype=torch.int64)
    pos = torch.full((num_pages * B,), -1, dtype=torch.int64)

    computed = 0
    peak = 0
    tails_seen = []
    for chunk in plan.chunks:
        batch = executor.build(chunk)
        tok[batch.slots] = batch.input_ids
        pos[batch.slots] = batch.position_ids
        peak = max(peak, pool.num_pages - pool.free_pages)
        computed += int(batch.input_ids.numel())

        assert len(batch.items) == len(chunk.items)
        assert batch.cu_seqlens_q.tolist()[-1] == batch.input_ids.numel()
        cached, positions = tok.tolist(), pos.tolist()
        for row, item in enumerate(batch.items):
            prompt = item_prompt(plan, prompts, item)
            pages = batch.page_table[row].tolist()
            length = int(batch.cache_seqlens[row])
            assert length == item.token_end
            slots = [pages[t // B] * B + t % B for t in range(length)]
            assert [cached[s] for s in slots] == list(prompt[:length])
            assert [positions[s] for s in slots] == list(range(length))

        for row, index in zip(batch.tail_rows.tolist(), batch.tail_wave_indices):
            assert int(batch.input_ids[row]) == prompts[index][-1]
        tails_seen.extend(batch.tail_wave_indices)
        executor.finish(chunk)
    executor.close()

    assert sorted(tails_seen) == list(range(len(prompts)))
    assert computed == plan.computed_tokens
    assert peak <= plan.peak_pool_pages + scratch
    return plan


def test_tree_replays_every_prompt_through_the_pool():
    plan = replay(TREE)
    assert plan.saved_tokens == 40
    assert len(plan.chunks) == 3


def test_tight_pool_replays_with_deferred_segments():
    plan = replay(TREE, pool_pages=3)
    assert plan.peak_pool_pages <= 3
    assert plan.saved_tokens == 40


def test_small_chunks_replay():
    replay(TREE, chunk_tokens=10)


def test_no_sharing_replays_as_plain_tails():
    plan = replay([blk(1) + [1], blk(2) + [2], [7]])
    assert plan.segments == ()


def test_random_waves_replay():
    rng = random.Random(0)
    for _ in range(100):
        vocab = [blk(n) for n in range(rng.randint(2, 6))]
        prompts = []
        for _ in range(rng.randint(1, 12)):
            blocks = rng.randint(0, 5)
            prompts.append(
                [t for _ in range(blocks) for t in rng.choice(vocab)]
                + [rng.randint(0, 3) for _ in range(rng.randint(1, 5))]
            )
        replay(
            prompts,
            pool_pages=rng.randint(0, 12),
            chunk_tokens=rng.randint(1, 40),
        )
