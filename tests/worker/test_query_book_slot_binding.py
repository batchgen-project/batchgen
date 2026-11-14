"""The query-book half of slot binding, over a ``SequenceTable``.

``bind_query_book_slot`` / ``release_query_book_slot`` own the query-book
entry; the slot binding itself is the table's. The pinned raw-container
helpers (``bind_local_sequence_to_query_book`` /
``release_local_query_slot``, covered by
``tests/test_worker_query_book_bindings.py``) run the same logic through
``SequenceTable.adopting``, so the two forms must agree.
"""

import torch

import pytest

from batchgen.query_book import (
    bind_local_sequence_to_query_book,
    bind_query_book_slot,
    release_local_query_slot,
    release_query_book_slot,
)
from batchgen.sequence import SequenceEntry
from batchgen.worker.sequence_table import SequenceTable


def _sequence(uuid, token_offset=10):
    seq = SequenceEntry(
        uuid, global_idx=7, prompt_length=4, max_decode_length=8,
    )
    seq.input_ids = torch.tensor(
        [[token_offset + 1, token_offset + 2]], dtype=torch.long
    )
    seq.decoded_tokens = torch.full((1, 8), token_offset, dtype=torch.int64)
    return seq


def test_bind_writes_the_table_and_the_query_book_entry():
    table, query_book = SequenceTable(), {}
    seq = _sequence("a")

    local_idx = bind_query_book_slot(
        "a", seq, query_book=query_book, table=table
    )

    assert local_idx == 0
    assert dict(table.uuid_to_local) == {"a": 0}
    assert query_book[0].kv_token_budget == seq.kv_token_budget
    assert query_book[0].encoded["input_ids"].data_ptr() == seq.input_ids.data_ptr()


def test_bind_refuses_a_missing_sequence_by_name():
    table, query_book = SequenceTable(), {}
    with pytest.raises(KeyError, match="not found in global_batch"):
        bind_query_book_slot("gone", None, query_book=query_book, table=table)
    assert len(table) == 0 and query_book == {}


def test_bind_refreshes_a_reused_slot():
    table, query_book = SequenceTable(), {}
    bind_query_book_slot("a", _sequence("a"), query_book=query_book, table=table)
    release_query_book_slot("a", query_book=query_book, table=table)

    fresh = _sequence("b", token_offset=20)
    local_idx = bind_query_book_slot(
        "b", fresh, query_book=query_book, table=table
    )

    assert local_idx == 0, "the freed slot is reused"
    assert query_book[0].encoded["input_ids"].data_ptr() == fresh.input_ids.data_ptr()
    assert query_book[0].decoded_tokens.data_ptr() == fresh.decoded_tokens.data_ptr()


def test_release_drops_the_entry_and_frees_the_slot():
    table, query_book = SequenceTable(), {}
    bind_query_book_slot("a", _sequence("a"), query_book=query_book, table=table)

    assert release_query_book_slot("a", query_book=query_book, table=table) == 0
    assert dict(table.uuid_to_local) == {} and dict(table.local_to_uuid) == {}
    assert query_book == {}
    assert set(table.free_local_indices) == {0}


def test_release_of_an_unbound_uuid_touches_nothing():
    table, query_book = SequenceTable(), {"keep": object()}
    assert release_query_book_slot("gone", query_book=query_book, table=table) is None
    assert set(query_book) == {"keep"}
    assert set(table.free_local_indices) == set()


def test_the_pinned_raw_container_helpers_agree_with_the_table_form():
    """One rule, two signatures: the pinned helpers must land the same state."""
    raw_forward, raw_reverse, raw_free, raw_book = {}, {}, set(), {}
    local_idx, next_idx = bind_local_sequence_to_query_book(
        "a", _sequence("a"),
        query_book=raw_book,
        local_to_uuid_map=raw_reverse,
        uuid_to_local_map=raw_forward,
        free_local_indices=raw_free,
        next_local_idx=0,
    )

    table, book = SequenceTable(), {}
    table_idx = bind_query_book_slot(
        "a", _sequence("a"), query_book=book, table=table
    )

    assert (local_idx, next_idx) == (table_idx, table.next_local_idx) == (0, 1)
    assert raw_forward == dict(table.uuid_to_local)
    assert raw_reverse == dict(table.local_to_uuid)
    assert raw_free == set(table.free_local_indices)

    assert release_local_query_slot(
        "a", uuid_to_local_map=raw_forward, local_to_uuid_map=raw_reverse,
        query_book=raw_book, free_local_indices=raw_free,
    ) == release_query_book_slot("a", query_book=book, table=table) == 0
    assert raw_forward == dict(table.uuid_to_local) == {}
    assert raw_free == set(table.free_local_indices) == {0}
    assert raw_book == book == {}


def test_adopting_mutates_the_callers_own_containers():
    forward, reverse, free = {}, {}, set()
    table = SequenceTable.adopting(
        local_to_uuid=reverse, uuid_to_local=forward,
        free_local_indices=free, next_local_idx=3,
    )
    table.bind("a")
    assert forward == {"a": 3} and reverse == {3: "a"}
    assert table.next_local_idx == 4
    table.unbind("a")
    assert forward == {} and free == {3}
