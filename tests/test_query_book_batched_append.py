"""`QueryBook.append_tokens` must stay token-for-token equal to per-row appends.

The decode step appends one sampled token per live row. That used to be one
`append_token` call per row, each doing a scalar tensor store plus a record and
turn rebuild. `append_tokens` now validates every row first and then issues a
single vectorized store over the flat page storage, so these tests pin the
observable result: stored tokens, `QueryBookSlot` lengths, the turn ledger, the
cached tail, and the capacity rejection (which must reject before writing
anything).
"""
import pytest
import torch

from batchgen.query_book import QueryBook, QueryBookCapacityError


def _book(page_tokens=4, pages=8):
    return QueryBook(
        capacity_bytes=pages * page_tokens * QueryBook.TOKEN_BYTES,
        page_tokens=page_tokens,
    )


def _bind_with_prompt(book, sequence_id, prompt, max_tokens):
    slot = book.bind(sequence_id, max_tokens=max_tokens)
    book.write_prompt(slot, torch.tensor(prompt, dtype=torch.int32))
    return slot


def _state(book, slot):
    record = book.metadata(slot)
    return (
        book.tokens(slot).tolist(),
        (record.prompt_length, record.token_length, record.decoded_length),
        book.turns(slot),
        book._tail_pages[slot],
        book._tail_offsets[slot],
    )


def test_batched_append_matches_per_row_appends():
    # Two books, same binds, same tokens: one appends per row, the other in a
    # single batched call. Every observable must agree.
    per_row = _book()
    batched = _book()
    prompts = {"a": [11, 12], "b": [21], "c": [31, 32, 33]}
    slots = {}
    for sequence_id, prompt in prompts.items():
        slots[sequence_id] = (
            _bind_with_prompt(per_row, sequence_id, prompt, 8),
            _bind_with_prompt(batched, sequence_id, prompt, 8),
        )

    steps = [[101, 201, 301], [102, 202, 302], [103, 203, 303]]
    for tokens in steps:
        for (sequence_id, _), token in zip(prompts.items(), tokens):
            per_row.append_token(slots[sequence_id][0], token)
        batched.append_tokens(
            [slots[sequence_id][1] for sequence_id in prompts],
            tokens,
        )

    for sequence_id in prompts:
        row_slot, batch_slot = slots[sequence_id]
        assert _state(per_row, row_slot) == _state(batched, batch_slot)
    assert torch.equal(per_row.storage, batched.storage)


def test_batched_append_crosses_a_page_boundary_like_per_row_appends():
    # page_tokens=4 with a 3-token prompt puts the second append on the next
    # page, which is exactly where a flat-index store could go wrong.
    per_row = _book(page_tokens=4, pages=8)
    batched = _book(page_tokens=4, pages=8)
    row_slot = _bind_with_prompt(per_row, "only", [1, 2, 3], 8)
    batch_slot = _bind_with_prompt(batched, "only", [1, 2, 3], 8)

    for token in (7, 8, 9, 10, 11):
        per_row.append_token(row_slot, token)
        batched.append_tokens([batch_slot], [token])

    assert _state(per_row, row_slot) == _state(batched, batch_slot)
    assert batched.tokens(batch_slot).tolist() == [1, 2, 3, 7, 8, 9, 10, 11]


def test_batched_append_writes_only_the_named_rows():
    book = _book()
    first = _bind_with_prompt(book, "first", [1], 4)
    second = _bind_with_prompt(book, "second", [2], 4)
    third = _bind_with_prompt(book, "third", [3], 4)

    book.append_tokens([first, third], [41, 43])

    assert book.tokens(first).tolist() == [1, 41]
    assert book.tokens(third).tolist() == [3, 43]
    # The skipped row keeps its own lengths and turn ledger untouched.
    assert book.tokens(second).tolist() == [2]
    assert book.metadata(second).decoded_length == 0
    assert book.turns(second)[-1].generated_length == 0


def test_batched_append_of_an_empty_batch_is_a_no_op():
    book = _book()
    slot = _bind_with_prompt(book, "only", [1, 2], 4)
    before = book.storage.clone()

    book.append_tokens([], [])

    assert _state(book, slot)[1] == (2, 2, 0)
    assert torch.equal(book.storage, before)


def test_batched_append_rejects_duplicate_slots():
    book = _book()
    slot = _bind_with_prompt(book, "only", [1], 4)
    with pytest.raises(ValueError):
        book.append_tokens([slot, slot], [5, 6])
    assert book.tokens(slot).tolist() == [1]


def test_batched_append_rejects_length_mismatch():
    book = _book()
    slot = _bind_with_prompt(book, "only", [1], 4)
    with pytest.raises(ValueError):
        book.append_tokens([slot], [5, 6])
    assert book.tokens(slot).tolist() == [1]


def test_batched_append_rejects_an_inactive_slot_before_writing():
    book = _book()
    live = _bind_with_prompt(book, "live", [1], 4)
    dead = _bind_with_prompt(book, "dead", [2], 4)
    book.release(dead)
    before = book.storage.clone()

    with pytest.raises(QueryBookCapacityError):
        book.append_tokens([live, dead], [5, 6])

    # Validation precedes the single store, so the live row must not have
    # advanced either.
    assert book.tokens(live).tolist() == [1]
    assert torch.equal(book.storage, before)


def test_batched_append_rejects_an_overfull_row_without_advancing_any_row():
    book = _book()
    full = _bind_with_prompt(book, "full", [1, 2, 3, 4], 4)
    room = _bind_with_prompt(book, "room", [5], 4)
    before = book.storage.clone()

    with pytest.raises(QueryBookCapacityError):
        book.append_tokens([room, full], [6, 7])

    assert book.tokens(room).tolist() == [5]
    assert book.tokens(full).tolist() == [1, 2, 3, 4]
    assert torch.equal(book.storage, before)


def test_batched_append_tracks_generated_spans_for_reentry_turns():
    # A re-entry turn opens a new generated span; the batched append must
    # extend the last turn exactly as the per-row path did.
    per_row = _book(page_tokens=8, pages=8)
    batched = _book(page_tokens=8, pages=8)
    row_slot = _bind_with_prompt(per_row, "only", [1, 2], 12)
    batch_slot = _bind_with_prompt(batched, "only", [1, 2], 12)

    per_row.append_token(row_slot, 3)
    batched.append_tokens([batch_slot], [3])
    per_row.append_prompt(row_slot, torch.tensor([4, 5], dtype=torch.int32))
    batched.append_prompt(batch_slot, torch.tensor([4, 5], dtype=torch.int32))
    per_row.append_token(row_slot, 6)
    batched.append_tokens([batch_slot], [6])

    assert _state(per_row, row_slot) == _state(batched, batch_slot)
    assert per_row.generated_tokens(row_slot).tolist() == (
        batched.generated_tokens(batch_slot).tolist()
    )
