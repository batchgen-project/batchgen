"""GLM-5.2 prefix-pool prefill addressing: every query's causal keys are its prompt.

Drives the real wave planner and chunk executor, writes each chunk's token ids
into a fake pool (as a layer writes the chunk's KV before attending), and
checks that a query at absolute position p reads exactly prompt[:p + 1]
through key_slots[causal_start:causal_end].
"""

import importlib.util
import random
import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("triton")
from batchgen.models.glm.glm5 import sparse_prefill  # noqa: E402


def _load(name, *relative):
    path = Path(__file__).parents[2].joinpath(*relative)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_PLAN = _load("test_glm_pool_plan_mod", "batchgen", "prefix_reuse", "wave_plan.py")
_POOL = _load("test_glm_pool_pool_mod", "batchgen", "prefix_reuse", "pool.py")
_EXEC = _load("test_glm_pool_exec_mod", "batchgen", "prefix_reuse", "executor.py")

B = 4


def blk(n):
    return [n * 10 + k for k in range(B)]


def check_wave(prompts, chunk_tokens=1000, same_chunk=True):
    plan = _PLAN.plan_wave_prefix_sharing(
        prompts, block_tokens=B, pool_pages=1000, chunk_tokens=chunk_tokens,
        same_chunk_ancestors=same_chunk,
    )
    _PLAN.validate_wave_prefix_plan(plan, prompts)
    scratch = _POOL.scratch_pages_bound(chunk_tokens, max(map(len, prompts)), len(prompts), B)
    pool = _POOL.PrefillPrefixPool(
        num_layers=1, num_pages=plan.pool_pages + scratch, page_tokens=B,
        num_kv_heads=1, head_dim=1, dtype=torch.float32, device="cpu",
        with_v=False,
    )
    executor = _EXEC.PoolChunkExecutor(plan, prompts, pool)
    tokens = torch.full((pool.num_pages * B,), -1, dtype=torch.int64)
    for chunk in plan.chunks:
        batch = executor.build(chunk)
        tokens[batch.slots] = batch.input_ids
        ctx = sparse_prefill.build_glm52_pool_prefill(
            page_table=batch.page_table,
            cache_seqlens=batch.cache_seqlens,
            cu_seqlens_q=batch.cu_seqlens_q,
            position_ids=batch.position_ids,
            slots=batch.slots,
            page_tokens=B,
            max_cache_seqlen=batch.max_cache_seqlen,
        )
        assert torch.equal(ctx.slots, batch.slots)
        assert int((ctx.causal_ends - ctx.causal_starts).max()) <= ctx.max_cache_seqlen
        keys = tokens[ctx.key_slots].tolist()
        starts, ends = ctx.causal_starts.tolist(), ctx.causal_ends.tolist()
        positions = batch.position_ids.tolist()
        for row, item in enumerate(batch.items):
            prompt = (
                prompts[item.ref] if item.kind == "tail"
                else prompts[plan.segments[item.ref].members[0]]
            )
            for q in range(batch.row_bounds[row], batch.row_bounds[row + 1]):
                p = positions[q]
                assert ends[q] - starts[q] == p + 1
                assert keys[starts[q]:ends[q]] == list(prompt[:p + 1])
        executor.finish(chunk)
    executor.close()
    return plan


TREE = [
    blk(1) + blk(2) + blk(3) + blk(70) + [900],
    blk(1) + blk(2) + blk(3) + blk(71) + [901],
    blk(1) + blk(2) + blk(4) + blk(72) + [902],
    blk(1) + blk(2) + blk(4) + blk(73) + [903],
    blk(5) + blk(6) + blk(74) + [904],
    blk(5) + blk(6) + blk(75) + [905],
    blk(76) + blk(77) + [906],
]


def test_tree_in_one_chunk_reads_every_prompt_prefix():
    plan = check_wave(TREE)
    assert len(plan.chunks) == 1


def test_multi_chunk_and_small_chunks_read_every_prompt_prefix():
    check_wave(TREE, same_chunk=False)
    check_wave(TREE, chunk_tokens=10)


def test_random_waves_read_every_prompt_prefix():
    rng = random.Random(1)
    for _ in range(60):
        vocab = [blk(n) for n in range(rng.randint(2, 6))]
        prompts = [
            [t for _ in range(rng.randint(0, 5)) for t in rng.choice(vocab)]
            + [rng.randint(0, 3) for _ in range(rng.randint(1, 5))]
            for _ in range(rng.randint(1, 10))
        ]
        check_wave(prompts, chunk_tokens=rng.randint(1, 40), same_chunk=rng.random() < 0.5)


def test_topk_offsets_map_to_pool_slots_and_keep_padding():
    key_slots = torch.tensor([40, 41, 42, 7, 8], dtype=torch.int64)
    topk = torch.tensor([[0, 2, -1], [3, 4, 1]], dtype=torch.int32)
    sparse_prefill._map_topk_to_pool_slots_(topk, key_slots, rows_per_step=1)
    assert topk.tolist() == [[40, 42, -1], [7, 8, 41]]
