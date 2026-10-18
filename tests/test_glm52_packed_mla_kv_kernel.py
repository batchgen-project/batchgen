"""The Triton packed-FP8 MLA quantizer, pinned byte-for-byte to the reference.

Node-only: needs Triton and a GPU. The reference it is compared against is
torch-only and fully exercised on the dev machine by
``test_glm52_packed_mla_kv_record.py``, so what this file adds is exactly one
claim -- the kernel and the reference emit the SAME BYTES -- plus the
degenerate and boundary inputs where a kernel is most likely to diverge from
its reference (zero blocks, saturation, odd token counts, non-contiguous
sources).
"""

import pytest
import torch

pytest.importorskip("triton")

from batchgen_kernels.attention.mla_kv_packed import (  # noqa: E402
    SOURCE_DIM,
    TOKEN_BYTES,
    dequantize_mla_kv_packed,
    quantize_mla_kv_packed_reference,
)
from batchgen_kernels.triton.mla_kv_packed import (  # noqa: E402
    quantize_mla_kv_packed,
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="packed MLA quantizer needs a GPU"
)


def _kv(tokens, scale=0.1, seed=0):
    torch.manual_seed(seed)
    return (torch.randn(tokens, SOURCE_DIM, device="cuda") * scale).to(
        torch.bfloat16
    )


@requires_cuda
@pytest.mark.parametrize("tokens", [1, 2, 7, 31, 32, 63, 64, 65, 128, 257, 1024])
def test_kernel_is_byte_identical_to_the_reference(tokens):
    kv = _kv(tokens, seed=tokens)
    got = quantize_mla_kv_packed(kv)
    want = quantize_mla_kv_packed_reference(kv.cpu()).to("cuda")
    assert got.shape == (tokens, TOKEN_BYTES)
    assert got.dtype == torch.uint8
    assert torch.equal(got, want), (got != want).sum().item()


@requires_cuda
def test_kernel_writes_into_a_caller_supplied_destination():
    """The production call shape: quantize straight into the staged buffer."""

    kv = _kv(96, seed=5)
    out = torch.full((96, TOKEN_BYTES), 0xAB, dtype=torch.uint8, device="cuda")
    returned = quantize_mla_kv_packed(kv, out=out)
    assert returned.data_ptr() == out.data_ptr()
    assert torch.equal(out, quantize_mla_kv_packed_reference(kv.cpu()).to("cuda"))


@requires_cuda
def test_kernel_defines_the_zero_block_the_same_way_as_the_reference():
    """The one deliberate divergence from SGLang must match on BOTH sides."""

    kv = torch.zeros(8, SOURCE_DIM, dtype=torch.bfloat16, device="cuda")
    packed = quantize_mla_kv_packed(kv)
    assert bool((packed == 0).all())
    assert torch.equal(packed, quantize_mla_kv_packed_reference(kv.cpu()).to("cuda"))
    assert torch.isfinite(dequantize_mla_kv_packed(packed).float()).all()


@requires_cuda
def test_kernel_handles_a_zero_block_beside_live_blocks():
    kv = _kv(16, seed=9)
    kv[:, 128:256] = 0.0
    kv[3, :] = 0.0
    packed = quantize_mla_kv_packed(kv)
    assert torch.equal(packed, quantize_mla_kv_packed_reference(kv.cpu()).to("cuda"))
    assert torch.isfinite(dequantize_mla_kv_packed(packed).float()).all()


@requires_cuda
def test_kernel_saturates_instead_of_overflowing():
    kv = torch.zeros(2, SOURCE_DIM, dtype=torch.bfloat16, device="cuda")
    kv[0, :128] = 57344.0
    kv[1, :128] = -57344.0
    packed = quantize_mla_kv_packed(kv)
    assert torch.equal(packed, quantize_mla_kv_packed_reference(kv.cpu()).to("cuda"))
    assert torch.isfinite(dequantize_mla_kv_packed(packed).float()).all()


@requires_cuda
def test_kernel_reads_a_strided_source_row():
    """Decode hands the kernel a row slice of a wider activation buffer."""

    wide = (
        torch.randn(48, SOURCE_DIM + 64, device="cuda") * 0.1
    ).to(torch.bfloat16)
    kv = wide[:, :SOURCE_DIM]
    assert kv.stride(1) == 1 and not kv.is_contiguous()
    got = quantize_mla_kv_packed(kv)
    assert torch.equal(
        got, quantize_mla_kv_packed_reference(kv.cpu().contiguous()).to("cuda")
    )


@requires_cuda
def test_kernel_round_trip_through_flashmla_dequant_semantics():
    kv = _kv(512, seed=13)
    got = dequantize_mla_kv_packed(quantize_mla_kv_packed(kv))
    assert torch.equal(got[:, 512:], kv[:, 512:])
    ref = kv.float()[:, :512]
    blocks = ref.view(-1, 4, 128)
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    err = (got.float()[:, :512] - ref).abs().view(-1, 4, 128)
    assert bool((err <= amax * 2.0**-4).all())


@requires_cuda
def test_kernel_rejects_the_same_operands_the_reference_rejects():
    with pytest.raises(ValueError, match="must be bfloat16"):
        quantize_mla_kv_packed(
            torch.empty(4, SOURCE_DIM, dtype=torch.float32, device="cuda")
        )
    with pytest.raises(ValueError, match="byte carrier"):
        quantize_mla_kv_packed(
            torch.empty(4, SOURCE_DIM, dtype=torch.bfloat16, device="cuda"),
            out=torch.empty(4, TOKEN_BYTES, dtype=torch.bfloat16, device="cuda"),
        )


@requires_cuda
def test_kernel_zero_token_batch_launches_nothing():
    kv = torch.empty(0, SOURCE_DIM, dtype=torch.bfloat16, device="cuda")
    out = quantize_mla_kv_packed(kv)
    assert out.shape == (0, TOKEN_BYTES)
