"""GPT-OSS whole-model graph replay inputs, including a rank with no rows."""

import pytest
import torch

from batchgen.cuda_graph.graph_manager import BatchSizeBucketing
from batchgen.models.openai.gpt_oss_120b.whole_model_graph_inputs import (
    WholeModelReplayInputError,
    whole_model_replay_page_inputs,
)

CPU = torch.device("cpu")


def test_empty_rank_without_page_table_gets_zero_row_inputs():
    # A rank that never held a sequence has no active page table. It must
    # still replay the bucket every other rank replays, as all padding.
    page_table, slots = whole_model_replay_page_inputs(
        active_page_table=None,
        slot_index_tensor=None,
        batch_size=0,
        max_pages_per_seq=6,
        device=CPU,
    )
    assert tuple(page_table.shape) == (0, 6)
    assert page_table.dtype == torch.int32
    assert tuple(slots.shape) == (0,)
    assert slots.dtype == torch.int32


def test_empty_rank_with_cleared_page_table_gets_zero_row_inputs():
    cleared = torch.full((0, 3), -1, dtype=torch.int32)
    page_table, slots = whole_model_replay_page_inputs(
        active_page_table=cleared,
        slot_index_tensor=torch.empty((0,), dtype=torch.int32),
        batch_size=0,
        max_pages_per_seq=6,
        device=CPU,
    )
    assert tuple(page_table.shape) == (0, 6)
    assert tuple(slots.shape) == (0,)


def test_rows_without_page_table_fail_loud():
    with pytest.raises(WholeModelReplayInputError, match="no active GPU page table"):
        whole_model_replay_page_inputs(
            active_page_table=None,
            slot_index_tensor=None,
            batch_size=1,
            max_pages_per_seq=6,
            device=CPU,
        )


def test_rows_beyond_page_table_fail_loud():
    with pytest.raises(WholeModelReplayInputError, match="has 1 row"):
        whole_model_replay_page_inputs(
            active_page_table=torch.zeros((1, 4), dtype=torch.int32),
            slot_index_tensor=None,
            batch_size=2,
            max_pages_per_seq=6,
            device=CPU,
        )


def test_rows_are_padded_to_captured_width():
    active = torch.tensor([[3, 4], [5, -1], [9, 9]], dtype=torch.int32)
    slot_view = torch.tensor([1, 0, 2], dtype=torch.int32)
    page_table, slots = whole_model_replay_page_inputs(
        active_page_table=active,
        slot_index_tensor=slot_view,
        batch_size=2,
        max_pages_per_seq=4,
        device=CPU,
    )
    torch.testing.assert_close(
        page_table, torch.tensor([[3, 4, 0, 0], [5, -1, 0, 0]], dtype=torch.int32)
    )
    torch.testing.assert_close(slots, torch.tensor([1, 0], dtype=torch.int32))


def test_rows_are_truncated_to_captured_width_and_slots_default_to_order():
    active = torch.arange(10, dtype=torch.int32).view(2, 5)
    page_table, slots = whole_model_replay_page_inputs(
        active_page_table=active,
        slot_index_tensor=None,
        batch_size=2,
        max_pages_per_seq=3,
        device=CPU,
    )
    torch.testing.assert_close(page_table, active[:, :3])
    torch.testing.assert_close(slots, torch.tensor([0, 1], dtype=torch.int32))


def test_every_rank_pads_to_the_global_bucket():
    # The worker picks the bucket from the global max row count, so an empty
    # rank and a one-row rank replay the same captured graph.
    bucketing = BatchSizeBucketing([1, 2, 3, 4, 5, 7, 10, 14])
    global_max_rows = 1
    bucket = bucketing.get_padded_size(global_max_rows)
    for local_rows in (0, 1):
        page_table, slots = whole_model_replay_page_inputs(
            active_page_table=None if local_rows == 0 else torch.zeros((1, 2), dtype=torch.int32),
            slot_index_tensor=None,
            batch_size=local_rows,
            max_pages_per_seq=2,
            device=CPU,
        )
        assert page_table.shape[0] == local_rows <= bucket
        assert slots.shape[0] == local_rows


def test_negative_rows_rejected():
    with pytest.raises(ValueError, match="batch_size"):
        whole_model_replay_page_inputs(
            active_page_table=None,
            slot_index_tensor=None,
            batch_size=-1,
            max_pages_per_seq=2,
            device=CPU,
        )
