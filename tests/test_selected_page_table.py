"""GPU gates for the selected-page-table transform.

The kernel maps logical DSA-selected positions to physical token ids over the
resident paged KV (short rows: dense prefix; long rows: indexer top-k), so
FlashAttention-3 can read the cache through a page-size-1 table without a
selected-KV copy. Padded graph rows and invalid selections must come out as
length 0 / id -1.
"""

import pytest
import torch

pytest.importorskip("triton")

_CUDA = torch.cuda.is_available()


def _reference(page_table, cache_seqlens, topk, page_size, slots, num_valid):
    """Pure-python mirror of the kernel contract."""
    batch, index_topk = topk.shape
    ids = torch.full((batch, index_topk), -1, dtype=torch.int32)
    lengths = torch.zeros(batch, dtype=torch.int32)
    max_pages = page_table.shape[1]
    for row in range(batch):
        if num_valid is not None and row >= int(num_valid.item()):
            continue
        slot = int(slots[row]) if slots is not None else row
        if slot < 0:
            continue
        seqlen = int(cache_seqlens[row])
        selected = min(seqlen, index_topk)
        lengths[row] = selected
        for pos in range(index_topk):
            logical = int(topk[row, pos]) if seqlen > index_topk else pos
            if not (0 <= logical < seqlen and pos < selected):
                continue
            page = logical // page_size
            if page >= max_pages:
                continue
            physical_page = int(page_table[slot, page])
            if physical_page < 0:
                continue
            ids[row, pos] = physical_page * page_size + (logical - page * page_size)
    return ids, lengths


@pytest.mark.skipif(not _CUDA, reason="CUDA required")
def test_dense_long_slots_and_padding_rows():
    from batchgen_kernels.attention.dsa.selected_page_table import (
        transform_selected_positions_out,
    )

    page_table = torch.tensor([[2, 3], [4, 5]], device="cuda", dtype=torch.int32)
    cache_seqlens = torch.tensor([2, 6, 0, 0], device="cuda", dtype=torch.int32)
    topk = torch.tensor(
        [[3, 2, 1, 0], [5, 0, 4, 1], [0, 0, 0, 0], [0, 0, 0, 0]],
        device="cuda",
        dtype=torch.int32,
    )
    slots = torch.tensor([0, 1, 999, 999], device="cuda", dtype=torch.int32)
    physical = torch.empty(4, 4, device="cuda", dtype=torch.int32)
    lengths = torch.empty(4, device="cuda", dtype=torch.int32)
    num_valid = torch.tensor([2], device="cuda", dtype=torch.int32)

    transform_selected_positions_out(
        page_table,
        cache_seqlens,
        topk,
        physical,
        lengths,
        page_size=4,
        primary_slot_indices=slots,
        num_valid_tokens=num_valid,
    )
    torch.cuda.synchronize()

    # Row 0 is short (seqlen 2 <= topk 4): a dense prefix, positions past the
    # length stay -1. Row 1 is long: top-k order is preserved. Rows 2-3 are
    # graph padding: length 0, every id -1.
    assert torch.equal(lengths, torch.tensor([2, 4, 0, 0], device="cuda", dtype=torch.int32))
    assert torch.equal(
        physical,
        torch.tensor(
            [[8, 9, -1, -1], [21, 16, 20, 17], [-1] * 4, [-1] * 4],
            device="cuda",
            dtype=torch.int32,
        ),
    )


@pytest.mark.skipif(not _CUDA, reason="CUDA required")
def test_invalid_selections_and_page_boundaries():
    from batchgen_kernels.attention.dsa.selected_page_table import (
        transform_selected_positions_out,
    )

    # One long row; top-k holds the indexer's -1 fill, an out-of-range
    # position, and both edges of a page.
    page_table = torch.tensor([[7, 9, -1]], device="cuda", dtype=torch.int32)
    cache_seqlens = torch.tensor([8], device="cuda", dtype=torch.int32)
    topk = torch.tensor([[-1, 8, 3, 4]], device="cuda", dtype=torch.int32)
    physical = torch.empty(1, 4, device="cuda", dtype=torch.int32)
    lengths = torch.empty(1, device="cuda", dtype=torch.int32)

    transform_selected_positions_out(
        page_table,
        cache_seqlens,
        topk,
        physical,
        lengths,
        page_size=4,
    )
    torch.cuda.synchronize()

    # seqlen 8 > topk 4 -> long row; -1 and position 8 (>= seqlen) are
    # dropped; 3 is the last token of page 0 (7*4+3), 4 the first of page 1.
    assert torch.equal(lengths, torch.tensor([4], device="cuda", dtype=torch.int32))
    assert torch.equal(
        physical,
        torch.tensor([[-1, -1, 31, 36]], device="cuda", dtype=torch.int32),
    )


@pytest.mark.skipif(not _CUDA, reason="CUDA required")
def test_matches_reference_on_random_mixed_batch():
    from batchgen_kernels.attention.dsa.selected_page_table import (
        transform_selected_positions_out,
    )

    torch.manual_seed(7)
    batch, index_topk, page_size, max_pages, slots_total = 64, 128, 64, 48, 96
    page_table = torch.randperm(slots_total * max_pages, dtype=torch.int32)[
        : slots_total * max_pages
    ].reshape(slots_total, max_pages).to("cuda")
    # Short, crossing and long rows around index_topk.
    cache_seqlens = torch.randint(1, index_topk * 3, (batch,), dtype=torch.int32).to("cuda")
    topk = torch.randint(-1, index_topk * 3, (batch, index_topk), dtype=torch.int32).to("cuda")
    slots = torch.randperm(slots_total, dtype=torch.int32)[:batch].to("cuda")
    slots[5] = -1  # an unassigned row
    num_valid = torch.tensor([batch - 3], device="cuda", dtype=torch.int32)
    physical = torch.empty(batch, index_topk, device="cuda", dtype=torch.int32)
    lengths = torch.empty(batch, device="cuda", dtype=torch.int32)

    transform_selected_positions_out(
        page_table,
        cache_seqlens,
        topk,
        physical,
        lengths,
        page_size=page_size,
        primary_slot_indices=slots,
        num_valid_tokens=num_valid,
    )
    torch.cuda.synchronize()

    expected_ids, expected_lengths = _reference(
        page_table.cpu(), cache_seqlens.cpu(), topk.cpu(), page_size, slots.cpu(), num_valid.cpu()
    )
    assert torch.equal(lengths.cpu(), expected_lengths)
    assert torch.equal(physical.cpu(), expected_ids)


@pytest.mark.skipif(not _CUDA, reason="CUDA required")
def test_row_identity_slots_without_slot_indices():
    from batchgen_kernels.attention.dsa.selected_page_table import (
        transform_selected_positions_out,
    )

    page_table = torch.tensor([[3], [6]], device="cuda", dtype=torch.int32)
    cache_seqlens = torch.tensor([2, 3], device="cuda", dtype=torch.int32)
    topk = torch.zeros(2, 4, device="cuda", dtype=torch.int32)
    physical = torch.empty(2, 4, device="cuda", dtype=torch.int32)
    lengths = torch.empty(2, device="cuda", dtype=torch.int32)

    transform_selected_positions_out(
        page_table, cache_seqlens, topk, physical, lengths, page_size=4
    )
    torch.cuda.synchronize()

    assert torch.equal(lengths, torch.tensor([2, 3], device="cuda", dtype=torch.int32))
    assert torch.equal(
        physical,
        torch.tensor([[12, 13, -1, -1], [24, 25, 26, -1]], device="cuda", dtype=torch.int32),
    )


def test_wrapper_validation_fails_loud():
    from batchgen_kernels.attention.dsa.selected_page_table import (
        transform_selected_positions_out,
    )

    page_table = torch.zeros(2, 2, dtype=torch.int32)
    seqlens = torch.zeros(2, dtype=torch.int32)
    topk = torch.zeros(2, 4, dtype=torch.int32)
    ids = torch.zeros(2, 4, dtype=torch.int32)
    lengths = torch.zeros(2, dtype=torch.int32)

    with pytest.raises(ValueError, match="page_size"):
        transform_selected_positions_out(page_table, seqlens, topk, ids, lengths, page_size=0)
    with pytest.raises(ValueError, match="cache_seqlens"):
        transform_selected_positions_out(
            page_table, torch.zeros(3, dtype=torch.int32), topk, ids, lengths, page_size=4
        )
    with pytest.raises(TypeError, match="physical_token_ids"):
        transform_selected_positions_out(
            page_table, seqlens, topk, ids.to(torch.int64), lengths, page_size=4
        )
    with pytest.raises(ValueError, match="primary_slot_indices"):
        transform_selected_positions_out(
            page_table, seqlens, topk, ids, lengths,
            page_size=4, primary_slot_indices=torch.zeros(5, dtype=torch.int32),
        )
    with pytest.raises(ValueError, match="num_valid_tokens"):
        transform_selected_positions_out(
            page_table, seqlens, topk, ids, lengths,
            page_size=4, num_valid_tokens=torch.zeros(2, dtype=torch.int32),
        )
