"""The single producer of a primary MLA KV cache record, dispatched on format.

`batchgen.attention.mla.primary_kv_record` is the one place a primary record is
produced, for the captured decode graph, the eager decode selector and the
prefill host offloads alike. It imports cleanly without Triton -- the quantizer
is pulled lazily -- so its dispatch, its bf16 passthrough and its rejections
are measured here rather than only on the node.

The packed branch needs the Triton kernel and is covered on the node by
`test_glm52_packed_mla_kv_kernel.py`; the record bytes it produces are covered
on the dev machine by `test_glm52_packed_mla_kv_record.py` against the
torch-only reference.
"""

import pytest
import torch

from batchgen.attention.mla.primary_kv_record import (
    PACKED_SOURCE_DIM,
    PACKED_TOKEN_BYTES,
    dequantize_mla_kv_packed,
    prepare_primary_mla_kv,
    primary_cache_is_packed_fp8,
    primary_record_dim,
    primary_record_dtype,
)


SOURCE_DIM = 576


def _bf16_cache(pages=4, page=64):
    return torch.empty(pages, page, 1, SOURCE_DIM, dtype=torch.bfloat16)


def _packed_cache(pages=4, page=64):
    return torch.zeros(pages, page, 1, PACKED_TOKEN_BYTES, dtype=torch.uint8)


def test_constants_agree_with_the_kernel_contract():
    assert PACKED_SOURCE_DIM == SOURCE_DIM == 576
    assert PACKED_TOKEN_BYTES == 656


def test_record_shape_and_dtype_come_from_the_cache():
    """Not from the model's kv_lora_rank + qk_rope_head_dim.

    Every buffer and TensorSpec on the write path derives from these two, so
    that the GPU store source, the staged host tensor and the host region's
    own geometry cannot each pick a different answer.
    """

    bf16 = _bf16_cache()
    packed = _packed_cache()
    assert primary_record_dim(bf16) == 576
    assert primary_record_dtype(bf16) is torch.bfloat16
    assert primary_record_dim(packed) == 656
    assert primary_record_dtype(packed) is torch.uint8


def test_discriminator_distinguishes_the_two_live_formats():
    assert primary_cache_is_packed_fp8(_bf16_cache(), SOURCE_DIM) is False
    assert primary_cache_is_packed_fp8(_packed_cache(), SOURCE_DIM) is True


@pytest.mark.parametrize(
    "cache",
    [
        torch.empty(2, 64, 1, 640, dtype=torch.uint8),
        torch.empty(2, 64, 1, 576, dtype=torch.uint8),
        torch.empty(2, 64, 1, 512, dtype=torch.bfloat16),
        torch.empty(2, 64, 1, 656, dtype=torch.bfloat16),
    ],
)
def test_discriminator_names_a_missized_cache(cache):
    with pytest.raises(ValueError):
        primary_cache_is_packed_fp8(cache, SOURCE_DIM)


def test_bf16_cache_record_is_the_latent_byte_for_byte():
    """The pre-migration behaviour must be preserved exactly.

    GLM-5/5.1/5.3 keep this path, so the record it produces has to be the
    latent itself -- no requantization, no reordering, no dtype change.
    """

    torch.manual_seed(0)
    latent = (torch.randn(9, SOURCE_DIM) * 0.1).to(torch.bfloat16)
    out = torch.empty(9, 1, 1, SOURCE_DIM, dtype=torch.bfloat16)

    returned = prepare_primary_mla_kv(
        primary_blocked_k=_bf16_cache(),
        compressed_kv=latent,
        out=out,
        source_dim=SOURCE_DIM,
    )

    assert returned.data_ptr() == out.data_ptr()
    assert torch.equal(out.view(9, SOURCE_DIM), latent)


def test_bf16_record_accepts_a_four_dimensional_latent_view():
    """The decode writers hand over `[B, 1, 1, 576]`-shaped views."""

    latent = torch.zeros(5, 1, 1, SOURCE_DIM, dtype=torch.bfloat16)
    latent.view(5, SOURCE_DIM)[:, 0] = 1.0
    out = torch.empty(5, 1, 1, SOURCE_DIM, dtype=torch.bfloat16)
    prepare_primary_mla_kv(
        primary_blocked_k=_bf16_cache(),
        compressed_kv=latent,
        out=out,
        source_dim=SOURCE_DIM,
    )
    assert torch.equal(out, latent)


def test_a_missized_cache_is_refused_before_anything_is_written():
    """The rejection must happen at the dispatch, not inside a kernel."""

    out = torch.full((3, 1, 1, SOURCE_DIM), 7, dtype=torch.bfloat16)
    before = out.clone()
    with pytest.raises(ValueError):
        prepare_primary_mla_kv(
            primary_blocked_k=torch.empty(2, 64, 1, 640, dtype=torch.uint8),
            compressed_kv=torch.zeros(3, SOURCE_DIM, dtype=torch.bfloat16),
            out=out,
            source_dim=SOURCE_DIM,
        )
    assert torch.equal(out, before)


def test_packed_dispatch_reaches_the_quantizer():
    """On a packed cache the dispatch must go to the Triton quantizer.

    Skipped where Triton is absent (the dev machine), which is exactly the
    point of the lazy import: a bf16 profile stays fully usable and fully
    testable there, and only the packed path needs the kernel.
    """

    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("the packed quantizer is a GPU kernel")

    torch.manual_seed(0)
    latent = (torch.randn(11, SOURCE_DIM, device="cuda") * 0.1).to(torch.bfloat16)
    out = torch.empty(11, 1, 1, PACKED_TOKEN_BYTES, dtype=torch.uint8, device="cuda")
    prepare_primary_mla_kv(
        primary_blocked_k=_packed_cache().to("cuda"),
        compressed_kv=latent,
        out=out,
        source_dim=SOURCE_DIM,
    )
    got = dequantize_mla_kv_packed(out.view(11, PACKED_TOKEN_BYTES))
    # RoPE is stored verbatim, so it must come back exactly.
    assert torch.equal(got[:, 512:], latent[:, 512:])


def test_dequant_helper_is_exported_for_the_debug_readers():
    """Debug/verify paths need a value-level view of the packed bytes."""

    page = torch.zeros(64, PACKED_TOKEN_BYTES, dtype=torch.uint8)
    got = dequantize_mla_kv_packed(page)
    assert got.shape == (64, SOURCE_DIM)
    assert got.dtype is torch.bfloat16
    assert bool((got == 0).all())
