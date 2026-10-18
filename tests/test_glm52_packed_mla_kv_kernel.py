"""The Triton packed-FP8 MLA quantizer, pinned byte-for-byte to the reference.

Node-only: needs Triton and a GPU. The reference it is compared against is
torch-only and fully exercised on the dev machine by
``test_glm52_packed_mla_kv_record.py``, so what this file adds is exactly one
claim -- the kernel and the reference emit the SAME BYTES -- plus the
degenerate and boundary inputs where a kernel is most likely to diverge from
its reference (zero blocks, saturation, odd token counts, non-contiguous
sources).

Byte-exactness is a SAME-DEVICE contract, so every assertion below runs the
reference on the kernel's own device. A CPU reference is NOT byte-identical to
a GPU one on this input class: ``scale = fp32_amax / 448`` rounds differently
per device, and a 1-ULP difference in the scale flips payload bytes wherever a
rescaled element lands on an e4m3 rounding boundary. Measured on the node
(``node_diag_item2aII.log``): kernel vs GPU reference is 0 differing bytes on
all 11 token counts, while CPU vs GPU reference differs in 11 bytes at 1 token
and 2495 at 1024 -- the payload/scale split (460, 2035) at 1024 shows the
scale is what moves. That cross-device delta stays VISIBLE here as an
informational measurement (``test_cross_device_reference_delta_is_informational``)
instead of as a false failure, and it does not touch the dev-side record tests,
which compare the CPU reference against CPU-side expectations only.
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


def _reference_same_device(kv):
    """The torch reference, evaluated on the KERNEL's device.

    The reference imports nothing but torch and is device-agnostic, so this is
    the same code the dev machine runs -- just not on a different device than
    the thing it is gating. See the module docstring for why the device has to
    match for a BYTE comparison.
    """

    return quantize_mla_kv_packed_reference(kv)


@requires_cuda
@pytest.mark.parametrize("tokens", [1, 2, 7, 31, 32, 63, 64, 65, 128, 257, 1024])
def test_kernel_is_byte_identical_to_the_reference(tokens):
    kv = _kv(tokens, seed=tokens)
    got = quantize_mla_kv_packed(kv)
    want = _reference_same_device(kv)
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
    assert torch.equal(out, _reference_same_device(kv))


@requires_cuda
def test_kernel_defines_the_zero_block_the_same_way_as_the_reference():
    """The one deliberate divergence from SGLang must match on BOTH sides."""

    kv = torch.zeros(8, SOURCE_DIM, dtype=torch.bfloat16, device="cuda")
    packed = quantize_mla_kv_packed(kv)
    assert bool((packed == 0).all())
    assert torch.equal(packed, _reference_same_device(kv))
    assert torch.isfinite(dequantize_mla_kv_packed(packed).float()).all()


@requires_cuda
def test_kernel_handles_a_zero_block_beside_live_blocks():
    kv = _kv(16, seed=9)
    kv[:, 128:256] = 0.0
    kv[3, :] = 0.0
    packed = quantize_mla_kv_packed(kv)
    assert torch.equal(packed, _reference_same_device(kv))
    assert torch.isfinite(dequantize_mla_kv_packed(packed).float()).all()


@requires_cuda
def test_kernel_saturates_instead_of_overflowing():
    kv = torch.zeros(2, SOURCE_DIM, dtype=torch.bfloat16, device="cuda")
    kv[0, :128] = 57344.0
    kv[1, :128] = -57344.0
    packed = quantize_mla_kv_packed(kv)
    assert torch.equal(packed, _reference_same_device(kv))
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
    # The reference takes the strided row too (it only requires stride(-1)==1),
    # so both sides read the SAME source view as well as the same device.
    assert torch.equal(got, _reference_same_device(kv))


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


@requires_cuda
def test_cross_device_reference_delta_is_informational():
    """How far the CPU reference is from the GPU one -- MEASURED, not gated.

    The gate is the same-device comparison (asserted here as well as in every
    test above). The cross-device delta is printed instead of asserted because
    it is a property of fp32 ``amax / 448`` rounding per device, not of this
    kernel: the kernel is byte-exact against the reference it actually runs
    beside, and a CPU-vs-GPU byte count is a measurement of the host-vs-device
    division, which is exactly the thing that must stay visible rather than
    either silently gating or silently disappearing.

    Run with ``-rA`` (or ``-s``) to see the table in the node log.
    """

    rows = []
    for tokens in (1, 32, 1024):
        kv = _kv(tokens, seed=tokens)
        kernel = quantize_mla_kv_packed(kv)
        gpu_ref = _reference_same_device(kv)
        cpu_ref = quantize_mla_kv_packed_reference(kv.cpu()).to(kv.device)
        # The gate, restated on the inputs being measured.
        assert torch.equal(kernel, gpu_ref)
        rows.append(
            (
                tokens,
                int((kernel != cpu_ref).sum().item()),
                int((cpu_ref != gpu_ref).sum().item()),
            )
        )

    print("\nCROSS_DEVICE_REFERENCE_DELTA (informational, NOT a gate)")
    print("tokens | kernel-vs-GPUref | kernel-vs-CPUref | CPUref-vs-GPUref")
    for tokens, kernel_vs_cpu, cpu_vs_gpu in rows:
        print(
            f"{tokens:6d} | {0:16d} | {kernel_vs_cpu:16d} | {cpu_vs_gpu:16d}"
        )
    print(
        "Differing BYTES out of tokens*656. A nonzero cross-device count is "
        "expected: fp32 amax/448 rounds per device and a 1-ULP scale shift "
        "flips payload bytes at e4m3 rounding boundaries. The byte contract "
        "is SAME-DEVICE."
    )
