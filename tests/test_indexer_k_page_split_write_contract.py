"""Contract of the fused page-split FP8 indexer-K writer.

The kernel in ``batchgen_kernels/src/attention/fused_indexer_k_write_fp8.cu``
does three things in one launch: quantize the step's bf16 indexer keys to
e4m3, write them into the page's K region plus the page's fp32 scale region,
and emit the ``(k_fp8, scales)`` pair the host KV region's append path takes.

This file owns the ORACLE for all three, built only from torch and from the
already-validated quantizer/page reference, plus:

* CPU tests of the launch planner (``plan_page_split_write``), which is the
  kernel's precondition set expressed in Python;
* CPU tests that the oracle places bytes at the hand-computed offsets of the
  layout ``deep_gemm.fp8_paged_mqa_logits`` reads;
* CPU tests of the row gating (``-1`` slot sentinel, ``num_valid_tokens``,
  unallocated page) the kernel must reproduce;
* a CUDA-gated test that runs the real kernel against the oracle.

``batchgen_kernels.attention.dsa.indexer_k_quant`` is torch-only, but its
PACKAGE imports Triton-backed kernels on purpose, so on a host without Triton
(any non-GPU dev machine) the module is loaded directly from its file. The
CUDA test imports the kernel wrapper normally and therefore only runs where
the whole package imports.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest
import torch

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_QUANT_PATH = (
    _REPO_ROOT / "batchgen_kernels" / "attention" / "dsa" / "indexer_k_quant.py"
)


def _load_quant_module():
    name = "_indexer_k_quant_contract"
    spec = importlib.util.spec_from_file_location(name, _QUANT_PATH)
    module = importlib.util.module_from_spec(spec)
    # Registered before exec because @dataclass resolves its own module.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_quant = _load_quant_module()

SCALE_BYTES_PER_TOKEN = _quant.SCALE_BYTES_PER_TOKEN
DATA_DIM = 128
PAGE_SIZE = 64
NUM_PAGES = 6
PAGE_BYTES = PAGE_SIZE * (DATA_DIM + SCALE_BYTES_PER_TOKEN)


def _empty_cache() -> torch.Tensor:
    return torch.zeros(
        NUM_PAGES, PAGE_SIZE, 1, DATA_DIM + SCALE_BYTES_PER_TOKEN,
        dtype=torch.float8_e4m3fn,
    )


def reference_page_split_write(
    *,
    cache_pages: torch.Tensor,
    k_src: torch.Tensor,
    page_table: torch.Tensor,
    slot_indices: torch.Tensor,
    token_indices: torch.Tensor,
    page_size: int,
    num_valid_tokens: int | None = None,
):
    """Oracle: the pages and the host sources one fused launch must produce.

    Mirrors the kernel exactly: the host sources are quantized for EVERY row,
    while a row reaches the cache only when it is inside ``num_valid_tokens``,
    its slot is not the ``-1`` sentinel, and its page-table entry is a real
    page.
    """

    num_pages, page_bytes = int(cache_pages.shape[0]), int(cache_pages.shape[1])
    data_dim = int(k_src.shape[-1])
    k_flat = k_src.reshape(-1, data_dim)
    k_fp8, scales = _quant.quantize_indexer_k_fp8(k_flat)

    absolute_slots = []
    rows = []
    for row in range(k_flat.shape[0]):
        if num_valid_tokens is not None and row >= int(num_valid_tokens):
            continue
        slot = int(slot_indices[row])
        if slot < 0:
            continue
        token_index = int(token_indices[row])
        page_slot = token_index // page_size
        if page_slot >= int(page_table.shape[1]):
            continue
        page = int(page_table[slot, page_slot])
        if page < 0 or page >= num_pages:
            continue
        absolute_slots.append(page * page_size + token_index % page_size)
        rows.append(row)

    expected = cache_pages.clone()
    if rows:
        row_idx = torch.tensor(rows, dtype=torch.int64)
        _quant.page_split_write_reference(
            expected,
            torch.tensor(absolute_slots, dtype=torch.int64),
            k_fp8[row_idx],
            scales[row_idx],
            page_size,
        )
    assert expected.shape == (num_pages, page_bytes)
    return expected, k_fp8, scales


# ---------------------------------------------------------------------------
# Launch planner: the kernel's precondition set
# ---------------------------------------------------------------------------


def test_plan_resolves_the_page_split_geometry():
    plan = _quant.plan_page_split_write(
        _empty_cache(),
        torch.zeros(5, 1, 1, DATA_DIM, dtype=torch.bfloat16),
        PAGE_SIZE,
    )
    assert plan.num_tokens == 5
    assert plan.num_pages == NUM_PAGES
    assert plan.page_size == PAGE_SIZE
    assert plan.page_bytes == PAGE_BYTES == 8448
    assert plan.data_dim == DATA_DIM
    assert plan.scale_region_offset == PAGE_SIZE * DATA_DIM == 8192


def test_plan_accepts_a_flat_source():
    plan = _quant.plan_page_split_write(
        _empty_cache(), torch.zeros(3, DATA_DIM, dtype=torch.bfloat16), PAGE_SIZE
    )
    assert plan.num_tokens == 3


def test_plan_rejects_a_page_size_that_is_not_the_cache_s():
    with pytest.raises(ValueError, match="page_size_tokens 32 does not match"):
        _quant.plan_page_split_write(
            _empty_cache(),
            torch.zeros(1, DATA_DIM, dtype=torch.bfloat16),
            32,
        )


def test_plan_rejects_a_non_bf16_source():
    with pytest.raises(ValueError, match="quantizes from bfloat16"):
        _quant.plan_page_split_write(
            _empty_cache(),
            torch.zeros(1, DATA_DIM, dtype=torch.float32),
            PAGE_SIZE,
        )


def test_plan_rejects_a_source_whose_trailing_dim_is_the_byte_carrier():
    # 132 is the per-token BYTE count, not the element count: handing it over
    # would quantize the scale bytes as if they were keys.
    with pytest.raises(ValueError, match="does not match the cache's 128"):
        _quant.plan_page_split_write(
            _empty_cache(),
            torch.zeros(1, DATA_DIM + SCALE_BYTES_PER_TOKEN, dtype=torch.bfloat16),
            PAGE_SIZE,
        )


def test_plan_rejects_a_data_dim_the_warp_layout_cannot_serve():
    cache = torch.zeros(2, PAGE_SIZE, 1, 48 + SCALE_BYTES_PER_TOKEN, dtype=torch.uint8)
    with pytest.raises(ValueError, match="multiple of 32"):
        _quant.plan_page_split_write(
            cache, torch.zeros(1, 48, dtype=torch.bfloat16), PAGE_SIZE
        )


def test_plan_rejects_a_two_byte_cache():
    cache = torch.zeros(2, PAGE_SIZE, 1, DATA_DIM, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="1-byte element type"):
        _quant.plan_page_split_write(
            cache, torch.zeros(1, DATA_DIM, dtype=torch.bfloat16), PAGE_SIZE
        )


def test_page_bytes_view_is_the_same_storage():
    cache = _empty_cache()
    pages = _quant.page_split_cache_pages(cache)
    assert pages.shape == (NUM_PAGES, PAGE_BYTES)
    assert pages.dtype is torch.uint8
    assert pages.data_ptr() == cache.data_ptr()


# ---------------------------------------------------------------------------
# Format discriminator: one shared decode path serves both aux layouts
# ---------------------------------------------------------------------------


def test_discriminator_reads_page_split_off_the_cache():
    assert _quant.cache_is_page_split(_empty_cache(), DATA_DIM) is True
    assert _quant.cache_data_dim(_empty_cache()) == DATA_DIM


def test_discriminator_reads_aos_off_the_cache():
    aos = torch.zeros(2, PAGE_SIZE, 1, DATA_DIM, dtype=torch.bfloat16)
    assert _quant.cache_is_page_split(aos, DATA_DIM) is False
    assert _quant.cache_data_dim(aos) == DATA_DIM


def test_discriminator_rejects_a_one_byte_cache_without_scale_bytes():
    sized_as_aos = torch.zeros(2, PAGE_SIZE, 1, DATA_DIM, dtype=torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="must be page-split"):
        _quant.cache_is_page_split(sized_as_aos, DATA_DIM)


def test_discriminator_rejects_a_mis_sized_aos_cache():
    wrong = torch.zeros(2, PAGE_SIZE, 1, DATA_DIM + 8, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="must carry 128 elements"):
        _quant.cache_is_page_split(wrong, DATA_DIM)


# ---------------------------------------------------------------------------
# Oracle placement and row gating
# ---------------------------------------------------------------------------


def _page_table(rows: list[list[int]]) -> torch.Tensor:
    return torch.tensor(rows, dtype=torch.int32)


def test_oracle_places_keys_and_scale_at_the_hand_computed_offsets():
    torch.manual_seed(0)
    cache = _quant.page_split_cache_pages(_empty_cache())
    k_src = (torch.randn(1, 1, 1, DATA_DIM) * 0.1).to(torch.bfloat16)
    # slot 1 -> logical page 1 -> physical page 4; token 70 -> offset 6.
    page_table = _page_table([[2, 3], [0, 4]])
    expected, k_fp8, scales = reference_page_split_write(
        cache_pages=cache,
        k_src=k_src,
        page_table=page_table,
        slot_indices=torch.tensor([1], dtype=torch.int32),
        token_indices=torch.tensor([70], dtype=torch.int32),
        page_size=PAGE_SIZE,
    )
    page, offset = 4, 6
    k_bytes = expected[page, offset * DATA_DIM : (offset + 1) * DATA_DIM]
    assert torch.equal(k_bytes, k_fp8[0].view(torch.uint8))
    scale_at = PAGE_SIZE * DATA_DIM + offset * SCALE_BYTES_PER_TOKEN
    written_scale = expected[page, scale_at : scale_at + SCALE_BYTES_PER_TOKEN]
    assert torch.equal(written_scale, scales[:1].contiguous().view(torch.uint8))
    # Nothing else in the cache moved.
    untouched = expected.clone()
    untouched[page, offset * DATA_DIM : (offset + 1) * DATA_DIM] = 0
    untouched[page, scale_at : scale_at + SCALE_BYTES_PER_TOKEN] = 0
    assert int(untouched.sum()) == 0


@pytest.mark.parametrize(
    "slots, num_valid, written_rows",
    [
        ([0, 1], None, [0, 1]),
        ([0, -1], None, [0]),          # -1 sentinel: CUDA graph bucket padding
        ([0, 1], 1, [0]),              # rows beyond num_valid_tokens
        ([-1, -1], None, []),
    ],
)
def test_oracle_row_gating(slots, num_valid, written_rows):
    cache = _quant.page_split_cache_pages(_empty_cache())
    k_src = (torch.randn(2, DATA_DIM) * 0.1).to(torch.bfloat16)
    page_table = _page_table([[1], [2]])
    expected, k_fp8, scales = reference_page_split_write(
        cache_pages=cache,
        k_src=k_src,
        page_table=page_table,
        slot_indices=torch.tensor(slots, dtype=torch.int32),
        token_indices=torch.tensor([0, 0], dtype=torch.int32),
        page_size=PAGE_SIZE,
        num_valid_tokens=num_valid,
    )
    pages_written = {int(page_table[int(s), 0]) for i, s in enumerate(slots) if i in written_rows}
    nonzero_pages = {int(p) for p in expected.sum(dim=1).nonzero().flatten()}
    assert nonzero_pages == pages_written
    # The host sources are produced for EVERY row regardless of gating: the
    # append decides which rows reach the host, not this launch.
    assert k_fp8.shape == (2, DATA_DIM)
    assert scales.shape == (2,)


def test_oracle_drops_an_unallocated_page_instead_of_writing_through_it():
    cache = _quant.page_split_cache_pages(_empty_cache())
    k_src = (torch.randn(1, DATA_DIM) * 0.1).to(torch.bfloat16)
    expected, _, _ = reference_page_split_write(
        cache_pages=cache,
        k_src=k_src,
        page_table=_page_table([[-1]]),
        slot_indices=torch.tensor([0], dtype=torch.int32),
        token_indices=torch.tensor([3], dtype=torch.int32),
        page_size=PAGE_SIZE,
    )
    assert int(expected.sum()) == 0


# ---------------------------------------------------------------------------
# The real kernel (node only)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_fused_kernel_matches_the_oracle_byte_for_byte():
    from batchgen_kernels.attention.dsa.indexer.indexer_k_write_fp8 import (
        fused_indexer_k_write_fp8,
    )

    torch.manual_seed(0)
    device = torch.device("cuda")
    num_tokens = 5
    k_src = (torch.randn(num_tokens, 1, 1, DATA_DIM) * 0.1).to(torch.bfloat16)
    page_table_rows = [[0, 1], [2, 3], [4, 5], [0, 1], [2, 3]]
    slots = [0, 1, -1, 2, 3]
    token_indices = [0, 65, 7, 70, 1]
    num_valid = 4

    cache = _empty_cache()
    expected, expected_k_fp8, expected_scales = reference_page_split_write(
        cache_pages=_quant.page_split_cache_pages(cache),
        k_src=k_src,
        page_table=_page_table(page_table_rows),
        slot_indices=torch.tensor(slots, dtype=torch.int32),
        token_indices=torch.tensor(token_indices, dtype=torch.int32),
        page_size=PAGE_SIZE,
        num_valid_tokens=num_valid,
    )

    cache_cuda = _empty_cache().to(device)
    k_fp8, scales = fused_indexer_k_write_fp8(
        aux_cache=cache_cuda,
        k_src=k_src.to(device),
        page_table=_page_table(page_table_rows).to(device),
        slot_indices=torch.tensor(slots, dtype=torch.int32, device=device),
        token_indices=torch.tensor(token_indices, dtype=torch.int32, device=device),
        page_size_tokens=PAGE_SIZE,
        num_valid_tokens=torch.tensor([num_valid], dtype=torch.int32, device=device),
    )
    torch.cuda.synchronize()

    written = _quant.page_split_cache_pages(cache_cuda).cpu()
    assert torch.equal(written, expected)
    assert torch.equal(
        k_fp8.view(num_tokens, DATA_DIM).view(torch.uint8).cpu(),
        expected_k_fp8.view(torch.uint8),
    )
    assert torch.equal(scales.cpu(), expected_scales)
