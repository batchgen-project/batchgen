import pytest
import torch

from batchgen.query_book import QueryBook, QueryBookCapacityError, QueryBookTurn


def test_query_book_allocates_fixed_page_store():
    book = QueryBook(capacity_bytes=4 * 4 * 4, page_tokens=4)

    assert tuple(book.storage.shape) == (4, 4)
    assert book.storage.dtype == torch.int32
    assert book.memory_bytes == 4 * 4 * 4
    assert book.free_page_count == 4

    first = book.bind("first", max_tokens=5)
    second = book.bind("second", max_tokens=4)
    assert first == 0
    assert second == 1
    assert book.active_count == 2
    assert book.free_page_count == 1


def test_query_book_rejects_non_integral_reservation_sizes():
    with pytest.raises(ValueError):
        QueryBook(capacity_bytes=16.0, page_tokens=4)

    book = QueryBook(capacity_bytes=4 * 4 * 4, page_tokens=4)
    with pytest.raises(ValueError):
        book.bind("fractional", max_tokens=0.5)
    assert book.active_count == 0
    assert book.free_page_count == 4


def test_query_book_reserves_variable_full_lengths_without_growth():
    book = QueryBook(capacity_bytes=4 * 4 * 4, page_tokens=4)
    assert book.can_reserve(8)
    first = book.bind("first", max_tokens=5)
    second = book.bind("second", max_tokens=8)

    assert book.metadata(first).max_tokens == 5
    assert book.metadata(second).max_tokens == 8
    assert book.free_page_count == 0
    with pytest.raises(QueryBookCapacityError):
        book.bind("third", max_tokens=1)


def test_query_book_admission_checks_pages_before_binding():
    book = QueryBook(capacity_bytes=3 * 4 * 4, page_tokens=4)
    assert book.can_reserve(9)
    assert not book.can_reserve(13)
    with pytest.raises(QueryBookCapacityError):
        book.bind("too-large", max_tokens=13)
    assert book.active_count == 0
    assert book.free_page_count == 3


def test_query_book_batch_admission_is_aggregate_and_atomic():
    book = QueryBook(capacity_bytes=3 * 4 * 4, page_tokens=4)

    assert book.can_reserve_batch([5, 4])
    assert not book.can_reserve_batch([5, 5])
    with pytest.raises(QueryBookCapacityError):
        book.bind_batch(["q-0", "q-1"], [5, 5])

    assert book.active_count == 0
    assert book.free_page_count == 3
    with pytest.raises(ValueError):
        book.bind_batch(["q-0", ""], [4, 4])
    assert book.active_count == 0
    assert book.free_page_count == 3
    slots = book.bind_batch(["q-0", "q-1"], [5, 4])
    assert slots == [0, 1]
    assert book.free_page_count == 0


def test_query_book_prompt_and_decode_share_reserved_pages():
    book = QueryBook(capacity_bytes=3 * 4 * 4, page_tokens=4)
    slot = book.bind("request-1", max_tokens=10)
    book.write_prompt(slot, torch.tensor([10, 11, 12, 13, 14], dtype=torch.int64))

    assert torch.equal(book.tokens(slot), torch.tensor([10, 11, 12, 13, 14], dtype=torch.int32))
    assert book.metadata(slot).prompt_length == 5
    assert book.metadata(slot).token_length == 5
    assert book.metadata(slot).decoded_length == 0

    assert book.append_token(slot, 21) == 5
    book.append_tokens([slot], [22])

    assert torch.equal(
        book.tokens(slot), torch.tensor([10, 11, 12, 13, 14, 21, 22], dtype=torch.int32)
    )
    assert book.metadata(slot).token_length == 7
    assert book.metadata(slot).decoded_length == 2


def test_query_book_batch_updates_different_page_positions():
    book = QueryBook(capacity_bytes=6 * 4 * 4, page_tokens=4)
    slots = [
        book.bind("request-0", max_tokens=8),
        book.bind("request-1", max_tokens=8),
        book.bind("request-2", max_tokens=8),
    ]
    book.write_prompts(
        slots,
        [
            torch.tensor([1, 2, 3], dtype=torch.int32),
            torch.tensor([4, 5, 6, 7, 8], dtype=torch.int32),
            torch.tensor([9], dtype=torch.int32),
        ],
    )

    book.append_tokens(slots, [101, 102, 103])
    book.append_tokens(slots, [111, 112, 113])

    assert torch.equal(book.tokens(slots[0]), torch.tensor([1, 2, 3, 101, 111], dtype=torch.int32))
    assert torch.equal(
        book.tokens(slots[1]), torch.tensor([4, 5, 6, 7, 8, 102, 112], dtype=torch.int32)
    )
    assert torch.equal(book.tokens(slots[2]), torch.tensor([9, 103, 113], dtype=torch.int32))


def test_query_book_copy_to_int64_and_reuse_output():
    book = QueryBook(capacity_bytes=2 * 4 * 4, page_tokens=4)
    slot = book.bind("request-1", max_tokens=5)
    book.write_prompt(slot, torch.tensor([7, 8, 9], dtype=torch.int32))
    book.append_token(slot, 10)

    copied = book.copy_to(slot, device="cpu", dtype=torch.int64)
    assert copied.dtype == torch.int64
    assert torch.equal(copied, torch.tensor([7, 8, 9, 10], dtype=torch.int64))

    output = torch.empty(8, dtype=torch.int64)
    reused = book.copy_to(slot, device="cpu", dtype=torch.int64, out=output)
    assert reused.data_ptr() == output.data_ptr()
    assert torch.equal(reused, copied)

    with pytest.raises(ValueError):
        book.copy_span_to(slot, torch.empty((2, 2), dtype=torch.int32).t())
    with pytest.raises(ValueError):
        book.copy_to(
            slot,
            device="cpu",
            dtype=torch.int64,
            out=torch.empty((2, 2), dtype=torch.int64).t(),
        )


def test_query_book_batch_updates_validate_before_mutation():
    book = QueryBook(capacity_bytes=4 * 4 * 4, page_tokens=4)
    slots = book.bind_batch(["q-0", "q-1"], [4, 4])
    book.write_prompts(slots, [torch.arange(4), torch.arange(1)])

    with pytest.raises(QueryBookCapacityError):
        book.append_tokens(slots, [10, 11])
    assert torch.equal(book.tokens(slots[0]), torch.arange(4, dtype=torch.int32))
    assert torch.equal(book.tokens(slots[1]), torch.arange(1, dtype=torch.int32))

    with pytest.raises(QueryBookCapacityError):
        book.write_prompts(slots, [torch.arange(5), torch.tensor([3])])
    assert torch.equal(book.tokens(slots[0]), torch.arange(4, dtype=torch.int32))
    assert torch.equal(book.tokens(slots[1]), torch.arange(1, dtype=torch.int32))


def test_query_book_release_returns_all_reserved_pages():
    book = QueryBook(capacity_bytes=3 * 4 * 4, page_tokens=4)
    slot = book.bind("request-1", max_tokens=9)
    assert book.free_page_count == 0
    book.release(slot)
    assert book.active_count == 0
    assert book.free_page_count == 3
    reused = book.bind("request-2", max_tokens=9)
    assert reused == slot


def test_query_book_lifecycle_and_length_guards():
    book = QueryBook(capacity_bytes=2 * 4 * 4, page_tokens=4)
    slot = book.bind("request-1", max_tokens=2)
    with pytest.raises(ValueError):
        book.bind("request-1", max_tokens=2)

    book.write_prompt(slot, torch.tensor([1, 2], dtype=torch.int32))
    with pytest.raises(QueryBookCapacityError):
        book.append_token(slot, 3)

    book.release(slot)
    with pytest.raises(QueryBookCapacityError):
        book.tokens(slot)


def test_query_book_writes_after_inference_mode_construction():
    with torch.inference_mode():
        book = QueryBook(capacity_bytes=2 * 4 * 4, page_tokens=4)
    slot = book.bind("request-1", max_tokens=4)
    book.write_prompt(slot, torch.tensor([1, 2]))
    book.append_token(slot, 3)
    assert torch.equal(book.tokens(slot), torch.tensor([1, 2, 3], dtype=torch.int32))


def test_query_book_batch_bind_is_atomic_when_free_pages_are_fragmented():
    book = QueryBook(capacity_bytes=8 * 4 * 4, page_tokens=4)
    slots = book.bind_batch(["a", "b", "c"], [8, 8, 8])
    book.release(slots[1])
    # Four pages are free in aggregate, but no contiguous three-page extent
    # exists because the first/last reservations still fence the hole.
    with pytest.raises(QueryBookCapacityError):
        book.bind_batch(["d", "e"], [12, 4])
    assert book.active_count == 2
    assert book.free_page_count == 4
    assert book.slot_for("a") == slots[0]
    assert book.slot_for("c") == slots[2]


def test_query_book_restore_preserves_trajectory_and_lengths():
    book = QueryBook(capacity_bytes=4 * 4 * 4, page_tokens=4)
    slot = book.bind("request-1", max_tokens=12)
    trajectory = torch.arange(9, dtype=torch.int32)
    book.restore(slot, trajectory, prompt_length=7, decoded_length=2)

    assert torch.equal(book.tokens(slot), trajectory)
    metadata = book.metadata(slot)
    assert metadata.prompt_length == 7
    assert metadata.token_length == 9
    assert metadata.decoded_length == 2
    book.append_token(slot, 9)
    assert torch.equal(book.tokens(slot), torch.arange(10, dtype=torch.int32))


def test_query_book_multi_turn_appends_prompt_and_tracks_generated_spans():
    book = QueryBook(capacity_bytes=4 * 4 * 4, page_tokens=4)
    slot = book.bind("conversation", max_tokens=16)
    book.write_prompt(slot, torch.tensor([1, 2]))
    book.append_token(slot, 10)
    book.append_prompt(slot, torch.tensor([20, 21]))
    book.append_token(slot, 30)

    assert torch.equal(
        book.tokens(slot), torch.tensor([1, 2, 10, 20, 21, 30], dtype=torch.int32)
    )
    assert torch.equal(
        book.generated_tokens(slot), torch.tensor([10, 30], dtype=torch.int32)
    )
    turns = book.turns(slot)
    assert [(t.prompt_start, t.prompt_length, t.generated_start, t.generated_length) for t in turns] == [
        (0, 2, 2, 1),
        (3, 2, 5, 1),
    ]
    assert book.metadata(slot).prompt_length == 5
    assert book.metadata(slot).decoded_length == 2


def test_query_book_reentry_turn_replays_prefix_without_double_counting():
    book = QueryBook(capacity_bytes=4 * 4 * 4, page_tokens=4)
    slot = book.bind("reentry", max_tokens=16)
    book.write_prompt(slot, torch.tensor([1, 2]))
    book.append_tokens([slot], [10])
    book.append_tokens([slot], [11])

    first = book.begin_reentry_turn(slot, prompt_length=4)
    second = book.begin_reentry_turn(slot, prompt_length=4)
    assert first == second
    book.append_token(slot, 12)

    assert book.metadata(slot).prompt_length == 4
    assert book.metadata(slot).decoded_length == 3
    assert torch.equal(
        book.generated_tokens(slot), torch.tensor([10, 11, 12], dtype=torch.int32)
    )
    assert book.turns(slot)[-1].generated_start == 4
    assert book.turns(slot)[-1].generated_length == 1


def test_query_book_restore_round_trips_turn_ledger():
    source = QueryBook(capacity_bytes=4 * 4 * 4, page_tokens=4)
    source_slot = source.bind("conversation", max_tokens=16)
    source.write_prompt(source_slot, torch.tensor([1, 2]))
    source.append_token(source_slot, 10)
    source.append_prompt(source_slot, torch.tensor([20]))
    source.append_token(source_slot, 30)

    destination = QueryBook(capacity_bytes=4 * 4 * 4, page_tokens=4)
    destination_slot = destination.bind("conversation", max_tokens=16)
    destination.restore(
        destination_slot,
        source.tokens(source_slot),
        prompt_length=source.metadata(source_slot).prompt_length,
        decoded_length=source.metadata(source_slot).decoded_length,
        turns=QueryBook.turns_from_payload(source.turns_payload(source_slot)),
    )

    assert torch.equal(destination.tokens(destination_slot), source.tokens(source_slot))
    assert destination.turns(destination_slot) == source.turns(source_slot)
    assert torch.equal(
        destination.generated_tokens(destination_slot),
        torch.tensor([10, 30], dtype=torch.int32),
    )


def test_query_book_restore_rejects_inconsistent_turn_metadata():
    book = QueryBook(capacity_bytes=2 * 4 * 4, page_tokens=4)
    slot = book.bind("bad", max_tokens=8)
    with pytest.raises(ValueError, match="does not match decoded_length"):
        book.restore(
            slot,
            torch.tensor([1, 2, 3], dtype=torch.int32),
            prompt_length=2,
            decoded_length=2,
            turns=[QueryBookTurn(0, 0, 2, 2, 1)],
        )
    assert book.metadata(slot).token_length == 0
