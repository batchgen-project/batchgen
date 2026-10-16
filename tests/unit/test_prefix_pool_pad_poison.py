"""``batchgen_debug.prefix_pool_pad_poison``: what a row's pad columns hold.

A chunk's page table is one rectangle, so every row shorter than the widest is
padded. The pad value is 0 - a REAL pool page, normally the one holding the
start of the root segment. A kernel that honours ``cache_seqlens`` never reads
those columns; one that derives its page count from the table width instead
reads that page as if it were the row's own context, and the widest-padded rows
are the shortest members.

The probe pads each row with ITS OWN last valid page instead, which is equally
unread under a correct kernel. Output parity between off and on exonerates the
padding; a difference proves the padding is read.

Pinned here: off is byte-identical to the 0-padded table it always built, on
pads with the row's own last page, nothing else about the batch moves, and the
pad columns lie outside every row's addressed range either way (so a correct
kernel cannot see them). Single-page rows, which have the widest padding, are
covered explicitly.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


def _load(name, *relative):
    path = Path(__file__).resolve().parents[2].joinpath(*relative)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_PLAN = _load("test_pad_poison_plan_mod", "batchgen", "prefix_reuse", "wave_plan.py")
_POOL = _load("test_pad_poison_pool_mod", "batchgen", "prefix_reuse", "pool.py")
_EXEC = _load("test_pad_poison_exec_mod", "batchgen", "prefix_reuse", "executor.py")

B = 4  # block tokens == page tokens


def blk(index: int) -> list[int]:
    return [index * 10 + offset for offset in range(B)]


# A nested chain of five members (so ancestor counts, and therefore row widths,
# differ by row) plus one prompt that shares nothing and fits in a single page:
# the widest padding in the chunk.
PROMPTS = [
    [token for index in range(depth + 1) for token in blk(index)]
    for depth in range(1, 6)
] + [[777]]


def _build(pad_poison: bool):
    """One chunk of the wave, built with the flag off or on."""

    plan = _PLAN.plan_wave_prefix_sharing(
        PROMPTS, block_tokens=B, pool_pages=64, chunk_tokens=4096,
        same_chunk_ancestors=True,
    )
    _PLAN.validate_wave_prefix_plan(plan, PROMPTS)
    scratch = _POOL.scratch_pages_bound(
        plan.chunk_tokens, max(len(p) for p in PROMPTS), len(PROMPTS), B
    )
    pool = _POOL.PrefillPrefixPool(
        num_layers=1, num_pages=plan.pool_pages + scratch, page_tokens=B,
        num_kv_heads=1, head_dim=2, dtype=torch.float32, device="cpu",
    )
    executor = _EXEC.PoolChunkExecutor(
        plan, PROMPTS, pool, pad_poison=pad_poison
    )
    chunk, = plan.chunks  # same_chunk_ancestors packs the whole tree
    batch = executor.build(chunk)
    return plan, batch


def _row_pages(batch, row: int) -> int:
    """How many page columns the row's own context actually addresses."""

    length = int(batch.cache_seqlens[row])
    return -(-length // B)


def test_the_chunk_really_has_rows_of_different_widths():
    """Guard the premise: without padding there is nothing to poison."""

    _, batch = _build(pad_poison=False)
    width = batch.page_table.shape[1]
    used = [_row_pages(batch, row) for row in range(len(batch.items))]

    assert width > 1
    assert min(used) == 1, "no single-page row, the widest-padded case"
    assert max(used) == width
    assert sum(1 for pages in used if pages < width) >= 3


def test_flag_off_pads_with_zero_exactly_as_before():
    """Byte-compare against the 0-padded table the executor always built."""

    _, batch = _build(pad_poison=False)
    table = batch.page_table.tolist()
    width = batch.page_table.shape[1]

    for row in range(len(batch.items)):
        used = _row_pages(batch, row)
        assert table[row][used:] == [0] * (width - used), (
            f"row {row}: pad columns are {table[row][used:]}, expected zeros"
        )
    assert batch.page_table.dtype == torch.int32


def test_flag_on_pads_with_the_row_s_own_last_page():
    _, batch = _build(pad_poison=True)
    table = batch.page_table.tolist()
    width = batch.page_table.shape[1]

    for row in range(len(batch.items)):
        used = _row_pages(batch, row)
        last = table[row][used - 1]
        assert table[row][used:] == [last] * (width - used), (
            f"row {row}: pad columns are {table[row][used:]}, expected "
            f"{width - used} copies of its own last page {last}"
        )
        # The whole point: a poisoned pad is never page 0's content unless the
        # row's own last page happens to be page 0.
        if last != 0 and used < width:
            assert 0 not in table[row][used:]


def test_the_flag_moves_nothing_but_the_pad_columns():
    """Same rows, same tokens, same slots, same page ids inside the context."""

    plan_off, off = _build(pad_poison=False)
    plan_on, on = _build(pad_poison=True)

    assert plan_off.chunks == plan_on.chunks
    assert off.items == on.items
    assert off.row_bounds == on.row_bounds
    assert off.tail_wave_indices == on.tail_wave_indices
    for field in ("input_ids", "position_ids", "cu_seqlens_q", "cache_seqlens",
                  "slots", "tail_rows"):
        assert torch.equal(getattr(off, field), getattr(on, field)), field
    assert (off.max_seqlen_q, off.max_cache_seqlen) == (
        on.max_seqlen_q, on.max_cache_seqlen
    )
    assert off.page_table.shape == on.page_table.shape

    # Identical inside every row's addressed range; the flag only rewrites the
    # columns past it.
    for row in range(len(off.items)):
        used = _row_pages(off, row)
        assert off.page_table[row][:used].tolist() == \
            on.page_table[row][:used].tolist(), row


def test_pad_columns_are_outside_every_row_s_addressed_range():
    """A kernel honouring cache_seqlens cannot reach a pad column either way.

    This is what makes the probe decisive: if the two runs differ on the GPU,
    the kernel is reading past ``cache_seqlens``.
    """

    for pad_poison in (False, True):
        _, batch = _build(pad_poison)
        width = batch.page_table.shape[1]
        for row, item in enumerate(batch.items):
            length = int(batch.cache_seqlens[row])
            used = _row_pages(batch, row)
            assert length == item.token_end
            assert used * B >= length
            assert (used - 1) * B < length, (
                f"row {row}: page {used - 1} holds no context token"
            )
            assert used <= width


def test_off_is_the_default():
    """Nothing is poisoned unless a batch asks: the production path is off."""

    plan = _PLAN.plan_wave_prefix_sharing(
        PROMPTS, block_tokens=B, pool_pages=64, chunk_tokens=4096,
    )
    pool = _POOL.PrefillPrefixPool(
        num_layers=1, num_pages=plan.pool_pages + 32, page_tokens=B,
        num_kv_heads=1, head_dim=2, dtype=torch.float32, device="cpu",
    )

    assert _EXEC.PoolChunkExecutor(plan, PROMPTS, pool).pad_poison is False
    assert _EXEC.PAD_POISON_DEBUG_FLAG == "prefix_pool_pad_poison"


def test_the_worker_reads_the_flag_through_the_batch_debug_seam():
    """The plumbing is batch level, like prefix_pool_offload_check."""

    source = Path(__file__).resolve().parents[2].joinpath(
        "batchgen", "batchgen_worker.py"
    ).read_text()

    assert "def _pool_pad_poison_enabled(self, wave) -> bool:" in source
    assert "pad_poison=self._pool_pad_poison_enabled(wave)," in source
    # Same seam the offload check uses: the wave's own sequences.
    seam = source.split("def _pool_pad_poison_enabled")[1].split("def ")[0]
    assert "_active_batchgen_debug_for_sequences" in seam
    assert "_debug_flag_enabled" in seam
    assert "PAD_POISON_DEBUG_FLAG" in seam
    assert "return False" in seam
