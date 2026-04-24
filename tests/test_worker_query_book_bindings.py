from types import SimpleNamespace

import torch

from batchgen.query_book import (
    QueryBookEntry,
    bind_local_sequence_to_query_book,
    release_local_query_slot,
)
from batchgen.sequence import SequenceEntry


def _make_sequence(uuid: str, global_idx: int, token_offset: int) -> SequenceEntry:
    seq = SequenceEntry(
        uuid,
        global_idx=global_idx,
        prompt_length=4,
        max_decode_length=8,
        text=f"text-{uuid}",
    )
    seq.input_ids = torch.tensor(
        [[token_offset + 1, token_offset + 2, token_offset + 3, token_offset + 4]],
        dtype=torch.long,
    )
    seq.decoded_tokens = torch.full((1, 8), token_offset, dtype=torch.int64)
    return seq


def test_release_local_query_slot_clears_query_book_for_freed_local_slot():
    seq = _make_sequence("seq-a", global_idx=7, token_offset=10)

    state = SimpleNamespace(
        uuid_to_local_map={seq.uuid: 4},
        local_to_uuid_map={4: seq.uuid},
        free_local_indices=set(),
        query_book={
            4: QueryBookEntry(
                text=seq.text,
                encoded={"input_ids": seq.input_ids},
                decoded_tokens=seq.decoded_tokens,
                kv_token_budget=seq.kv_token_budget,
            )
        },
    )

    local_idx = release_local_query_slot(
        seq.uuid,
        uuid_to_local_map=state.uuid_to_local_map,
        local_to_uuid_map=state.local_to_uuid_map,
        query_book=state.query_book,
        free_local_indices=state.free_local_indices,
    )

    assert local_idx == 4
    assert state.uuid_to_local_map == {}
    assert state.local_to_uuid_map == {}
    assert state.query_book == {}
    assert state.free_local_indices == {4}


def test_bind_local_sequence_to_query_book_refreshes_reused_slot():
    seq = _make_sequence("seq-b", global_idx=11, token_offset=20)

    stale_input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    stale_decoded_tokens = torch.full((1, 8), -1, dtype=torch.int64)

    state = SimpleNamespace(
        uuid_to_local_map={},
        local_to_uuid_map={},
        free_local_indices={4},
        next_local_idx=5,
        query_book={
            4: QueryBookEntry(
                text="stale",
                encoded={"input_ids": stale_input_ids},
                decoded_tokens=stale_decoded_tokens,
                kv_token_budget=99,
            )
        },
    )

    local_idx, next_local_idx = bind_local_sequence_to_query_book(
        seq.uuid,
        seq,
        query_book=state.query_book,
        local_to_uuid_map=state.local_to_uuid_map,
        uuid_to_local_map=state.uuid_to_local_map,
        free_local_indices=state.free_local_indices,
        next_local_idx=state.next_local_idx,
    )

    assert local_idx == 4
    assert next_local_idx == 5
    assert state.uuid_to_local_map == {seq.uuid: 4}
    assert state.local_to_uuid_map == {4: seq.uuid}
    assert state.free_local_indices == set()
    assert state.query_book[4].text == seq.text
    assert state.query_book[4].kv_token_budget == seq.kv_token_budget
    assert state.query_book[4].encoded["input_ids"].data_ptr() == seq.input_ids.data_ptr()
    assert state.query_book[4].decoded_tokens.data_ptr() == seq.decoded_tokens.data_ptr()
