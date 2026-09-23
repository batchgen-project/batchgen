import dataclasses
import importlib.util
import random
import sys
from pathlib import Path

import pytest


_MODULE_PATH = (
    Path(__file__).parents[2] / "batchgen" / "prefix_reuse" / "wave_plan.py"
)
_SPEC = importlib.util.spec_from_file_location("test_prefix_wave_plan_mod", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

plan_wave_prefix_sharing = _MODULE.plan_wave_prefix_sharing
validate_wave_prefix_plan = _MODULE.validate_wave_prefix_plan

B = 4  # block tokens, small for readable trees


def blk(n):
    return [n * 10 + k for k in range(B)]


def plan(prompts, pool_pages=1000, chunk_tokens=1000):
    result = plan_wave_prefix_sharing(
        prompts, block_tokens=B, pool_pages=pool_pages, chunk_tokens=chunk_tokens
    )
    validate_wave_prefix_plan(result, prompts)
    return result


def chunk_view(result):
    names = []
    for chunk in result.chunks:
        names.append([
            f"s{item.ref}" if item.kind == "segment" else f"p{item.ref}"
            for item in chunk.items
        ])
    return names


# A -> {A1, A2}, B, and one prompt sharing nothing:
#   p0,p1: A + A1 + own   p2,p3: A + A2 + own   p4,p5: B + own   p6: own
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


def test_mixed_tree_pools_each_shared_segment_once():
    result = plan(TREE)
    spans = [(s.token_start, s.token_end, s.members) for s in result.segments]
    # DFS ids: A=0, A1=1, A2=2, B=3
    assert spans == [
        (0, 8, (0, 1, 2, 3)),
        (8, 12, (0, 1)),
        (8, 12, (2, 3)),
        (0, 8, (4, 5)),
    ]
    assert result.threshold == 2
    assert chunk_view(result) == [
        ["s0", "s3", "p6"],
        ["s1", "s2", "p4", "p5"],
        ["p0", "p1", "p2", "p3"],
    ]
    assert [c.release_after for c in result.chunks] == [(), (3,), (0, 1, 2)]
    assert result.prompt_tokens == 103
    # saved = (4-1)*8 + (2-1)*4 + (2-1)*4 + (2-1)*8
    assert result.saved_tokens == 40
    assert result.computed_tokens == 63
    assert result.peak_pool_pages == 6


def test_tight_pool_defers_segments_but_keeps_all_sharing():
    result = plan(TREE, pool_pages=3)
    assert result.threshold == 2
    assert result.saved_tokens == 40
    assert result.peak_pool_pages <= 3
    assert len(result.chunks) > 3


def test_pool_too_small_for_children_raises_threshold():
    result = plan(TREE, pool_pages=2)
    # Only A (4 prompts) is pooled; A1, A2, B are recomputed per prompt.
    assert result.threshold == 4
    assert [(s.token_start, s.token_end) for s in result.segments] == [(0, 8)]
    assert result.saved_tokens == 3 * 8


def test_zero_pool_means_no_sharing():
    result = plan(TREE, pool_pages=0)
    assert result.threshold is None
    assert result.segments == ()
    assert result.saved_tokens == 0
    assert chunk_view(result) == [[f"p{i}" for i in [0, 1, 2, 3, 4, 5, 6]]]


def test_chunk_cap_splits_but_preserves_dependencies():
    result = plan(TREE, chunk_tokens=10)
    assert all(
        sum(item.tokens for item in c.items) <= 10 or len(c.items) == 1
        for c in result.chunks
    )
    assert result.saved_tokens == 40


def test_identical_prompts_keep_last_block_private():
    prompt = blk(1) + blk(2) + blk(3)  # 12 tokens: 2 shareable blocks
    result = plan([prompt, prompt, prompt])
    assert [(s.token_start, s.token_end) for s in result.segments] == [(0, 8)]
    tails = [i for c in result.chunks for i in c.items if i.kind == "tail"]
    assert {(t.token_start, t.token_end) for t in tails} == {(8, 12)}
    assert result.saved_tokens == 2 * 8


def test_prompt_ending_inside_shared_region_splits_segment():
    x, y, z = blk(1), blk(2), blk(3)
    prompts = [x + y + [1], x + y + z + [2], x + y + z + [3]]
    result = plan(prompts)
    assert [(s.token_start, s.token_end, s.members) for s in result.segments] == [
        (0, 8, (0, 1, 2)),
        (8, 12, (1, 2)),
    ]
    assert chunk_view(result) == [["s0"], ["p0", "s1"], ["p1", "p2"]]


def test_no_sharing_is_one_tail_per_prompt():
    prompts = [blk(1) + [1], blk(2) + [2], [7]]
    result = plan(prompts)
    assert result.segments == ()
    assert result.computed_tokens == result.prompt_tokens == 11


def test_random_waves_satisfy_invariants():
    rng = random.Random(0)
    for trial in range(200):
        vocab = [blk(n) for n in range(rng.randint(2, 6))]
        prompts = []
        for _ in range(rng.randint(1, 12)):
            blocks = rng.randint(0, 5)
            prompts.append(
                [t for _ in range(blocks) for t in rng.choice(vocab)]
                + [rng.randint(0, 3) for _ in range(rng.randint(1, 5))]
            )
        plan(
            prompts,
            pool_pages=rng.randint(0, 12),
            chunk_tokens=rng.randint(1, 40),
        )


def test_validator_rejects_item_before_its_segment():
    result = plan(TREE)
    first, second, third = result.chunks
    swapped = dataclasses.replace(result, chunks=(second, first, third))
    with pytest.raises(AssertionError):
        validate_wave_prefix_plan(swapped, TREE)


def test_validator_rejects_missing_tail():
    result = plan(TREE)
    last = result.chunks[-1]
    trimmed = dataclasses.replace(last, items=last.items[:-1])
    broken = dataclasses.replace(result, chunks=result.chunks[:-1] + (trimmed,))
    with pytest.raises(AssertionError):
        validate_wave_prefix_plan(broken, TREE)


def test_rejects_empty_prompt():
    with pytest.raises(ValueError):
        plan_wave_prefix_sharing([[]], block_tokens=B, pool_pages=1, chunk_tokens=1)
