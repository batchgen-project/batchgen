"""Kernel helpers for synthetic FlashMLA selected-KV block tables."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fill_sequential_block_table_kernel(
    out_ptr,
    n_elements: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    tl.store(out_ptr + offs, offs.to(tl.int32), mask=mask)


def make_selected_block_table(
    batch_size: int,
    pages_per_row: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Create ``[B, pages_per_row]`` sequential block table with Triton."""

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if pages_per_row <= 0:
        raise ValueError(f"pages_per_row must be positive, got {pages_per_row}")

    out = torch.empty(
        batch_size,
        pages_per_row,
        dtype=torch.int32,
        device=device,
    )
    n_elements = batch_size * pages_per_row
    block = 256
    grid = (triton.cdiv(n_elements, block),)
    _fill_sequential_block_table_kernel[grid](
        out,
        n_elements=n_elements,
        BLOCK=block,
    )
    return out
