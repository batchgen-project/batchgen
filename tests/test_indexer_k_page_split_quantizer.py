"""The page-split FP8 indexer-K quantizer is byte-exact, and so is its page.

Pure-CPU contract tests for the bytes GLM-5.2's aux cache stores:

* ``quantize_indexer_k_fp8`` must reproduce the validated reference
  (per-token amax, ``clamp(1e-4) / 448``, reciprocal MULTIPLY, e4m3 cast)
  byte for byte — including zeros, subnormals, an amax under the floor and
  saturation past +-448.
* ``page_split_write_reference`` must place token ``o`` at
  ``[o * data_dim, (o + 1) * data_dim)`` of the page's K region and its fp32
  scale at ``page_size * data_dim + o * 4``: the layout
  ``deep_gemm.fp8_paged_mqa_logits`` reads and the C++ host copy paths write.
"""

import pytest
import torch

# The quantizer itself is torch-only, but importing it goes through
# `batchgen_kernels.attention.dsa.__init__`, which imports Triton-backed
# kernels on purpose (a DSA import failure must not be swallowed).
pytest.importorskip("triton")

from batchgen_kernels.attention.dsa.indexer_k_quant import (  # noqa: E402
    AMAX_FLOOR,
    FP8_MAX,
    SCALE_BYTES_PER_TOKEN,
    page_split_geometry,
    page_split_write_reference,
    quantize_indexer_k_fp8,
)

DATA_DIM = 128
PAGE_SIZE = 64
PAGE_BYTES = PAGE_SIZE * (DATA_DIM + SCALE_BYTES_PER_TOKEN)


def _reference_quantize(k: torch.Tensor):
    """The validated reference (batchgen_kernel_dev/dsa/split_write_fp8.py)."""

    scale = k.abs().float().amax(dim=-1, keepdim=True).clamp(AMAX_FLOOR) / FP8_MAX
    k_fp8 = (k * (1.0 / scale)).to(torch.float8_e4m3fn)
    return k_fp8, scale.squeeze(-1)


def _as_bytes(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.contiguous().view(torch.uint8).reshape(-1)


def test_page_identity_is_exact():
    assert PAGE_BYTES == 8448
    # 132 B/token against 256 B/token for the bf16 AoS cache it replaces.
    assert PAGE_BYTES // PAGE_SIZE == DATA_DIM + SCALE_BYTES_PER_TOKEN == 132
    assert PAGE_SIZE * DATA_DIM == 8192  # K region
    assert PAGE_SIZE * SCALE_BYTES_PER_TOKEN == 256  # scale region
    assert PAGE_SIZE * DATA_DIM + PAGE_SIZE * SCALE_BYTES_PER_TOKEN == PAGE_BYTES


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_quantizer_matches_reference_byte_for_byte(dtype):
    torch.manual_seed(0)
    k = (torch.randn(257, DATA_DIM) * 0.1).to(dtype)
    k_fp8, scales = quantize_indexer_k_fp8(k)
    ref_fp8, ref_scales = _reference_quantize(k)

    assert k_fp8.dtype is torch.float8_e4m3fn
    assert scales.dtype is torch.float32
    assert scales.shape == (257,)
    assert torch.equal(_as_bytes(k_fp8), _as_bytes(ref_fp8))
    assert torch.equal(_as_bytes(scales), _as_bytes(ref_scales))


def test_quantizer_edge_cases():
    rows = torch.zeros(5, DATA_DIM, dtype=torch.bfloat16)
    # 0: all zeros -> amax floor decides the scale.
    # 1: tiny values well under the floor.
    rows[1].fill_(1e-8)
    # 2: subnormal-ish bf16 magnitudes.
    rows[2].fill_(torch.finfo(torch.bfloat16).tiny)
    # 3/4: saturation at both signs, far past the e4m3 range.
    rows[3].fill_(1e4)
    rows[4].fill_(-1e4)

    k_fp8, scales = quantize_indexer_k_fp8(rows)
    ref_fp8, ref_scales = _reference_quantize(rows)
    assert torch.equal(_as_bytes(k_fp8), _as_bytes(ref_fp8))
    assert torch.equal(_as_bytes(scales), _as_bytes(ref_scales))

    # The floor pins the scale of a zero/tiny row, so no row ever gets a zero
    # scale that would dequantize every token to NaN.
    assert torch.all(scales > 0)
    assert float(scales[0]) == pytest.approx(AMAX_FLOOR / FP8_MAX)
    assert float(scales[1]) == pytest.approx(AMAX_FLOOR / FP8_MAX)
    # A saturating row keeps its finite amax scale and finite bytes.
    assert torch.isfinite(k_fp8[3].float()).all()
    assert torch.isfinite(k_fp8[4].float()).all()


def test_page_split_write_places_both_regions():
    torch.manual_seed(1)
    num_pages = 3
    buf = torch.zeros(num_pages, PAGE_BYTES, dtype=torch.uint8)
    tokens = torch.tensor([0, 1, 63, 64, 130], dtype=torch.int32)
    k = (torch.randn(tokens.numel(), DATA_DIM) * 0.1).to(torch.bfloat16)
    k_fp8, scales = quantize_indexer_k_fp8(k)

    page_split_write_reference(buf, tokens, k_fp8, scales, page_size=PAGE_SIZE)

    k_bytes = k_fp8.view(torch.uint8)
    scale_bytes = scales.view(torch.uint8).view(-1, SCALE_BYTES_PER_TOKEN)
    for row, slot in enumerate(tokens.tolist()):
        page, offset = divmod(slot, PAGE_SIZE)
        k_start = offset * DATA_DIM
        assert torch.equal(
            buf[page, k_start : k_start + DATA_DIM], k_bytes[row]
        )
        s_start = PAGE_SIZE * DATA_DIM + offset * SCALE_BYTES_PER_TOKEN
        assert torch.equal(
            buf[page, s_start : s_start + SCALE_BYTES_PER_TOKEN],
            scale_bytes[row],
        )
    # Nothing else was touched: exactly the written tokens' bytes are non-zero
    # candidates, every other byte stays zero.
    written = torch.zeros_like(buf, dtype=torch.bool)
    for slot in tokens.tolist():
        page, offset = divmod(slot, PAGE_SIZE)
        written[page, offset * DATA_DIM : (offset + 1) * DATA_DIM] = True
        s_start = PAGE_SIZE * DATA_DIM + offset * SCALE_BYTES_PER_TOKEN
        written[page, s_start : s_start + SCALE_BYTES_PER_TOKEN] = True
    assert not buf[~written].any()


def test_dequantized_page_round_trips_within_fp8_resolution():
    torch.manual_seed(2)
    k = (torch.randn(PAGE_SIZE, DATA_DIM) * 0.1).to(torch.bfloat16)
    k_fp8, scales = quantize_indexer_k_fp8(k)
    buf = torch.zeros(1, PAGE_BYTES, dtype=torch.uint8)
    slots = torch.arange(PAGE_SIZE, dtype=torch.int32)
    page_split_write_reference(buf, slots, k_fp8, scales, page_size=PAGE_SIZE)

    # Read the page back exactly as a reader must: K region then scale region.
    k_region = buf[0, : PAGE_SIZE * DATA_DIM].view(PAGE_SIZE, DATA_DIM)
    read_k = k_region.view(torch.float8_e4m3fn).float()
    read_scales = (
        buf[0, PAGE_SIZE * DATA_DIM :].contiguous().view(torch.float32)
    )
    assert torch.equal(read_scales, scales)
    dequantized = read_k * read_scales[:, None]

    reference = k_fp8.float() * scales[:, None]
    assert torch.equal(dequantized, reference)
    # e4m3 keeps 3 mantissa bits, so judge the end-to-end error with the
    # similarity metric used for FP8, not an element-wise tolerance.
    x, y = dequantized.double(), k.double()
    diff = 1 - (2 * (x * y).sum() / (x * x + y * y).sum()).item()
    assert diff < 1e-3


def test_page_split_geometry_rejects_wrong_shapes():
    good = torch.zeros(2, PAGE_SIZE, 1, 132, dtype=torch.uint8)
    num_pages, page_bytes, data_dim, scale_offset = page_split_geometry(good)
    assert (num_pages, page_bytes, data_dim, scale_offset) == (
        2,
        PAGE_BYTES,
        DATA_DIM,
        PAGE_SIZE * DATA_DIM,
    )

    with pytest.raises(ValueError):  # 2-byte elements are not a byte carrier
        page_split_geometry(torch.zeros(2, PAGE_SIZE, 1, 132, dtype=torch.bfloat16))
    with pytest.raises(ValueError):  # wrong rank
        page_split_geometry(torch.zeros(2, PAGE_SIZE, 132, dtype=torch.uint8))
    with pytest.raises(ValueError):  # multi-head K
        page_split_geometry(torch.zeros(2, PAGE_SIZE, 2, 132, dtype=torch.uint8))
    with pytest.raises(ValueError):  # no data bytes left
        page_split_geometry(torch.zeros(2, PAGE_SIZE, 1, 4, dtype=torch.uint8))
    with pytest.raises(ValueError):  # non-contiguous
        page_split_geometry(
            torch.zeros(2, PAGE_SIZE, 1, 264, dtype=torch.uint8)[..., ::2]
        )
