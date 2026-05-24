"""Compressed attention metadata initialisation for DSA V4 indexer.

Computes per-request metadata for two compression levels (4x and 128x) used by
the V4 sparse attention path.  For each request the kernel derives compressed
output locations, aligned positions, and clamped sequence lengths from the raw
(uncompressed) metadata.  Optionally it also builds a page-index lookup table
for the 128x-compressed view so that paged KV reads can be translated into flat
compressed offsets.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _init_compressed_attn_metadata_kernel(
    seq_lens_ptr,
    positions_ptr,
    raw_out_loc_ptr,
    page_table_ptr,
    # ---- compress-4 outputs ----
    c4_out_loc_ptr,
    c4_positions_ptr,
    c4_seq_lens_raw_ptr,
    c4_seq_lens_clamp1_ptr,
    # ---- compress-128 outputs ----
    c128_out_loc_ptr,
    c128_positions_ptr,
    c128_seq_lens_clamp1_ptr,
    c128_page_indices_ptr,
    # ---- scalars / constexprs ----
    batch_size,
    max_pages,
    page_size: tl.constexpr,
    c128_max_seq_len: tl.constexpr,
    c128_page_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    COMPUTE_PAGE_INDICES: tl.constexpr,
):
    """Per-request metadata for 4x and 128x compressed attention views.

    Grid: ``(batch_size,)`` — one programme per request.
    """
    batch_id = tl.program_id(0)
    if batch_id >= batch_size:
        return

    seq_len = tl.load(seq_lens_ptr + batch_id)
    position = tl.load(positions_ptr + batch_id)
    raw_out_loc = tl.load(raw_out_loc_ptr + batch_id)

    c4_should_compress = (seq_len % 4) == 0
    c4_out_loc = tl.where(c4_should_compress, raw_out_loc // 4, 0)
    c4_positions = position & (~3)
    c4_seq_lens_raw = seq_len // 4
    c4_seq_lens_clamp1 = tl.maximum(c4_seq_lens_raw, 1)

    tl.store(c4_out_loc_ptr + batch_id, c4_out_loc)
    tl.store(c4_positions_ptr + batch_id, c4_positions)
    tl.store(c4_seq_lens_raw_ptr + batch_id, c4_seq_lens_raw)
    tl.store(c4_seq_lens_clamp1_ptr + batch_id, c4_seq_lens_clamp1)

    c128_should_compress = (seq_len % 128) == 0
    c128_out_loc = tl.where(c128_should_compress, raw_out_loc // 128, 0)
    c128_positions = position & (~127)
    c128_seq_lens_raw = seq_len // 128
    c128_seq_lens_clamp1 = tl.maximum(c128_seq_lens_raw, 1)

    tl.store(c128_out_loc_ptr + batch_id, c128_out_loc)
    tl.store(c128_positions_ptr + batch_id, c128_positions)
    tl.store(c128_seq_lens_clamp1_ptr + batch_id, c128_seq_lens_clamp1)

    if COMPUTE_PAGE_INDICES:
        page_indices_base = batch_id * c128_max_seq_len
        for block_start in range(0, c128_max_seq_len, BLOCK_SIZE):
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < c128_max_seq_len

            page_idx = offsets // c128_page_size
            offset_in_page = offsets % c128_page_size

            page_mask = mask & (page_idx < max_pages)
            page_table_vals = tl.load(
                page_table_ptr + batch_id * max_pages + page_idx,
                mask=page_mask,
                other=0,
            )

            compressed_page_indices = (
                page_table_vals * c128_page_size + offset_in_page
            )

            valid_mask = offsets < c128_seq_lens_raw
            compressed_page_indices = tl.where(
                valid_mask, compressed_page_indices, -1
            )

            tl.store(
                c128_page_indices_ptr + page_indices_base + offsets,
                compressed_page_indices,
                mask=mask,
            )


def init_compressed_attention_metadata(
    seq_lens: torch.Tensor,
    positions: torch.Tensor,
    raw_out_loc: torch.Tensor,
    page_table: Optional[torch.Tensor] = None,
    page_size: int = 0,
    compute_page_indices: bool = True,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
]:
    """Initialise compressed attention metadata for the V4 indexer.

    For each request in the batch, computes output locations, positions, and
    sequence lengths at two compression levels (4x and 128x).  When
    *compute_page_indices* is ``True`` an additional ``[batch_size,
    c128_max_seq_len]`` int32 tensor of flattened page indices is produced for
    the 128x view.

    Args:
        seq_lens: ``[batch_size]`` int32 — per-request sequence lengths.
        positions: ``[batch_size]`` int32 — per-request current positions.
        raw_out_loc: ``[batch_size]`` int32 — uncompressed output locations.
        page_table: ``[batch_size, max_pages]`` int32 — paged KV page table
            (required when *compute_page_indices* is ``True``).
        page_size: Physical page size in tokens (must be >0 when computing
            page indices).
        compute_page_indices: Whether to derive the 128x page-index map.

    Returns:
        Tuple of eight tensors (the last one is ``None`` when
        *compute_page_indices* is ``False``):

        * ``c4_out_loc``           — ``[bs]`` int32
        * ``c4_positions``         — ``[bs]`` int32
        * ``c4_seq_lens_raw``      — ``[bs]`` int32
        * ``c4_seq_lens_clamp1``   — ``[bs]`` int32
        * ``c128_out_loc``         — ``[bs]`` int32
        * ``c128_positions``       — ``[bs]`` int32
        * ``c128_seq_lens_clamp1`` — ``[bs]`` int32
        * ``c128_page_indices``    — ``[bs, c128_max_seq_len]`` int32 or None
    """
    batch_size = seq_lens.shape[0]
    device = seq_lens.device

    c4_out_loc = torch.empty(batch_size, dtype=torch.int32, device=device)
    c4_positions = torch.empty(batch_size, dtype=torch.int32, device=device)
    c4_seq_lens_raw = torch.empty(batch_size, dtype=torch.int32, device=device)
    c4_seq_lens_clamp1 = torch.empty(
        batch_size, dtype=torch.int32, device=device
    )

    c128_out_loc = torch.empty(batch_size, dtype=torch.int32, device=device)
    c128_positions = torch.empty(batch_size, dtype=torch.int32, device=device)
    c128_seq_lens_clamp1 = torch.empty(
        batch_size, dtype=torch.int32, device=device
    )

    if compute_page_indices:
        assert (
            page_table is not None
        ), "page_table is required when compute_page_indices=True"
        assert (
            page_size > 0
        ), "page_size must be >0 when compute_page_indices=True"
        max_pages = page_table.shape[1]
        c128_page_size = page_size // 128
        c128_max_seq_len = c128_page_size * max_pages
        c128_page_indices = torch.empty(
            batch_size, c128_max_seq_len, dtype=torch.int32, device=device
        )
        block_size = triton.next_power_of_2(max(c128_page_size, 64))
    else:
        max_pages = 0
        c128_page_size = 1
        c128_max_seq_len = 0
        c128_page_indices = None
        block_size = 64
        if page_table is None:
            page_table = torch.empty(0, dtype=torch.int32, device=device)

    grid = (batch_size,)
    _init_compressed_attn_metadata_kernel[grid](
        seq_lens,
        positions,
        raw_out_loc,
        page_table,
        c4_out_loc,
        c4_positions,
        c4_seq_lens_raw,
        c4_seq_lens_clamp1,
        c128_out_loc,
        c128_positions,
        c128_seq_lens_clamp1,
        (
            c128_page_indices
            if c128_page_indices is not None
            else torch.empty(0, dtype=torch.int32, device=device)
        ),
        batch_size,
        max_pages,
        page_size if page_size > 0 else 128,
        c128_max_seq_len,
        c128_page_size,
        block_size,
        compute_page_indices,
    )

    return (
        c4_out_loc,
        c4_positions,
        c4_seq_lens_raw,
        c4_seq_lens_clamp1,
        c128_out_loc,
        c128_positions,
        c128_seq_lens_clamp1,
        c128_page_indices,
    )
