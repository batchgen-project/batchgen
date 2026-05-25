import pytest
import torch

from batchgen.prefill.prefix_reuse import (
    build_prefix_reuse_prefill_plan,
    split_prefix_reuse_plan_for_micro_batch,
)


def test_build_prefix_reuse_prefill_plan_mixed_hit_and_miss():
    input_ids = [
        torch.tensor([[10, 11, 12, 13, 14, 15]]),
        torch.tensor([[20, 21, 22, 23]]),
        torch.tensor([[30, 31, 32, 33, 34]]),
    ]
    plan = build_prefix_reuse_prefill_plan(
        local_indices=[0, 1, 2],
        sequence_ids=[100, 101, 102],
        input_ids=input_ids,
        prompt_lengths=[6, 4, 5],
        prefix_shared_tokens=[4, 0, 5],
    )

    assert [item.raw_prefix_shared_tokens for item in plan.sequences] == [
        4,
        0,
        5,
    ]
    assert [item.suffix_length for item in plan.sequences] == [2, 4, 1]
    assert [item.suffix_start_pos for item in plan.sequences] == [4, 0, 4]
    assert [tensor.tolist() for tensor in plan.suffix_input_ids] == [
        [14, 15],
        [20, 21, 22, 23],
        [34],
    ]
    assert [tensor.tolist() for tensor in plan.suffix_position_ids] == [
        [4, 5],
        [0, 1, 2, 3],
        [4],
    ]
    assert plan.cache_seqlens.tolist() == [4, 0, 4]
    assert plan.total_prompt_tokens == 15
    assert plan.total_suffix_tokens == 7
    assert plan.saved_prefill_tokens == 8


def test_split_prefix_reuse_prefill_plan_recomputes_stats():
    plan = build_prefix_reuse_prefill_plan(
        local_indices=[0, 1, 2],
        sequence_ids=[100, 101, 102],
        input_ids=[
            torch.arange(0, 6),
            torch.arange(10, 14),
            torch.arange(20, 25),
        ],
        prompt_lengths=[6, 4, 5],
        prefix_shared_tokens=[4, 0, 2],
    )

    micro = split_prefix_reuse_plan_for_micro_batch(plan, 1, 3)

    assert [item.sequence_id for item in micro.sequences] == [101, 102]
    assert [tensor.tolist() for tensor in micro.suffix_input_ids] == [
        [10, 11, 12, 13],
        [22, 23, 24],
    ]
    assert micro.cache_seqlens.tolist() == [0, 2]
    assert micro.total_prompt_tokens == 9
    assert micro.total_suffix_tokens == 7
    assert micro.saved_prefill_tokens == 2


def test_build_prefix_reuse_prefill_plan_accepts_clamped_full_hit():
    plan = build_prefix_reuse_prefill_plan(
        local_indices=[0],
        sequence_ids=[100],
        input_ids=[torch.arange(0, 4)],
        prompt_lengths=[4],
        prefix_shared_tokens=[4],
    )

    assert plan.sequences[0].is_full_hit is True
    assert plan.sequences[0].raw_prefix_shared_tokens == 4
    assert plan.sequences[0].prefix_shared_tokens == 3
    assert plan.sequences[0].suffix_start_pos == 3
    assert plan.sequences[0].suffix_length == 1
    assert plan.suffix_input_ids[0].tolist() == [3]


def test_build_prefix_reuse_prefill_plan_one_token_full_hit_has_no_saved_tokens():
    plan = build_prefix_reuse_prefill_plan(
        local_indices=[0],
        sequence_ids=[100],
        input_ids=[torch.tensor([42])],
        prompt_lengths=[1],
        prefix_shared_tokens=[1],
    )

    assert plan.sequences[0].is_full_hit is True
    assert plan.sequences[0].raw_prefix_shared_tokens == 1
    assert plan.sequences[0].prefix_shared_tokens == 0
    assert plan.sequences[0].suffix_start_pos == 0
    assert plan.sequences[0].suffix_length == 1
    assert plan.cache_seqlens.tolist() == [0]
    assert plan.total_prompt_tokens == 1
    assert plan.total_suffix_tokens == 1
    assert plan.saved_prefill_tokens == 0
    assert plan.suffix_input_ids[0].tolist() == [42]
    assert plan.suffix_position_ids[0].tolist() == [0]


def test_build_prefix_reuse_prefill_plan_validates_lengths():
    with pytest.raises(ValueError, match="exceeds prompt_length"):
        build_prefix_reuse_prefill_plan(
            local_indices=[0],
            sequence_ids=[100],
            input_ids=[torch.arange(0, 4)],
            prompt_lengths=[4],
            prefix_shared_tokens=[5],
        )
