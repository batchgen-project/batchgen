from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


def _reference_compressed_metadata(
    seq_lens: torch.Tensor,
    positions: torch.Tensor,
    raw_out_loc: torch.Tensor,
    page_table: torch.Tensor | None = None,
    page_size: int = 0,
    compute_page_indices: bool = True,
):
    batch_size = seq_lens.shape[0]

    c4_should_compress = (seq_lens % 4) == 0
    c4_out_loc = torch.where(
        c4_should_compress, raw_out_loc // 4, torch.zeros_like(raw_out_loc)
    )
    c4_positions = positions & (~3)
    c4_seq_lens_raw = seq_lens // 4
    c4_seq_lens_clamp1 = torch.clamp(c4_seq_lens_raw, min=1)

    c128_should_compress = (seq_lens % 128) == 0
    c128_out_loc = torch.where(
        c128_should_compress, raw_out_loc // 128, torch.zeros_like(raw_out_loc)
    )
    c128_positions = positions & (~127)
    c128_seq_lens_raw = seq_lens // 128
    c128_seq_lens_clamp1 = torch.clamp(c128_seq_lens_raw, min=1)

    c128_page_indices = None
    if compute_page_indices and page_table is not None and page_size > 0:
        max_pages = page_table.shape[1]
        c128_page_size = page_size // 128
        c128_max_seq_len = c128_page_size * max_pages

        c128_page_indices = torch.full(
            (batch_size, c128_max_seq_len),
            -1,
            dtype=torch.int32,
            device=seq_lens.device,
        )
        for b in range(batch_size):
            for off in range(c128_max_seq_len):
                page_idx = off // c128_page_size
                offset_in_page = off % c128_page_size
                if page_idx < max_pages:
                    pt_val = page_table[b, page_idx].item()
                    val = pt_val * c128_page_size + offset_in_page
                    if off < c128_seq_lens_raw[b].item():
                        c128_page_indices[b, off] = val
                    else:
                        c128_page_indices[b, off] = -1

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


def _run_equivalence(seq_len_value: int, page_size: int = 256) -> None:
    from batchgen.attention.dsa.v4_indexer_metadata import (
        init_compressed_attention_metadata,
    )

    batch_size = 4
    device = "cuda"

    seq_lens = torch.full(
        (batch_size,), seq_len_value, dtype=torch.int32, device=device
    )
    positions = (seq_lens - 1).to(torch.int32)
    raw_out_loc = torch.arange(
        0,
        batch_size * seq_len_value,
        seq_len_value,
        dtype=torch.int32,
        device=device,
    )

    max_pages = (seq_len_value + page_size - 1) // page_size
    max_pages = max(max_pages, 1)
    page_table = (
        torch.arange(max_pages, dtype=torch.int32, device=device)
        .unsqueeze(0)
        .expand(batch_size, -1)
        .contiguous()
    )

    actual = init_compressed_attention_metadata(
        seq_lens,
        positions,
        raw_out_loc,
        page_table=page_table,
        page_size=page_size,
        compute_page_indices=True,
    )
    expected = _reference_compressed_metadata(
        seq_lens,
        positions,
        raw_out_loc,
        page_table=page_table,
        page_size=page_size,
        compute_page_indices=True,
    )

    names = [
        "c4_out_loc",
        "c4_positions",
        "c4_seq_lens_raw",
        "c4_seq_lens_clamp1",
        "c128_out_loc",
        "c128_positions",
        "c128_seq_lens_clamp1",
        "c128_page_indices",
    ]
    for name, act, exp in zip(names, actual, expected):
        if act is None and exp is None:
            continue
        assert act is not None and exp is not None, f"{name}: one is None"
        torch.testing.assert_close(act, exp, msg=lambda m: f"{name}: {m}")


def test_seq_len_1():
    _run_equivalence(1)


def test_seq_len_128():
    _run_equivalence(128)


def test_seq_len_1024():
    _run_equivalence(1024)


def test_seq_len_8192():
    _run_equivalence(8192)


def test_no_page_indices():
    from batchgen.attention.dsa.v4_indexer_metadata import (
        init_compressed_attention_metadata,
    )

    batch_size = 2
    device = "cuda"
    seq_len_value = 128

    seq_lens = torch.full(
        (batch_size,), seq_len_value, dtype=torch.int32, device=device
    )
    positions = (seq_lens - 1).to(torch.int32)
    raw_out_loc = torch.arange(
        0,
        batch_size * seq_len_value,
        seq_len_value,
        dtype=torch.int32,
        device=device,
    )

    result = init_compressed_attention_metadata(
        seq_lens,
        positions,
        raw_out_loc,
        compute_page_indices=False,
    )
    expected = _reference_compressed_metadata(
        seq_lens,
        positions,
        raw_out_loc,
        compute_page_indices=False,
    )

    for i in range(7):
        torch.testing.assert_close(result[i], expected[i])
    assert result[7] is None
    assert expected[7] is None
