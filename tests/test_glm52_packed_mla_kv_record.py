"""The packed FP8 MLA record contract, measured without Triton or a GPU.

The 656-byte record is the byte-exactness contract between the GLM-5.2 decode
and prefill writers, every staging buffer, the host shm pages and FlashMLA's
sparse FP8 kernel. The reference implementation of that contract is torch-only
precisely so this file can hold it to the contract on the dev machine; the
Triton kernel is pinned against this same reference in
``test_glm52_packed_mla_kv_kernel.py`` (node, needs Triton).
"""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


def _load_packed():
    """Load the torch-only packed-record module by path.

    ``batchgen_kernels.attention`` pulls a precompiled decode extension and
    ``batchgen_kernels.triton`` needs Triton, neither of which exists on a
    CPU-only machine. Loading the module by path is what keeps the contract
    dev-measured rather than quietly skipped -- the mis-report corrected in
    the aux slice's part-2 report.
    """

    path = ROOT / "batchgen_kernels" / "attention" / "mla_kv_packed.py"
    name = "_glm52_packed_mla_kv_for_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PACKED = _load_packed()


# --- the record FlashMLA hard-checks -------------------------------------


def test_record_geometry_matches_the_flashmla_contract():
    # csrc/pybind.cpp:249-256 -- bytes_per_token = 512 + 64*2 + (512/128)*4,
    # page_block_size == 64 (:236), whole block contiguous.
    assert PACKED.NOPE_DIM == 512
    assert PACKED.ROPE_DIM == 64
    assert PACKED.SOURCE_DIM == 576
    assert PACKED.QUANT_TILE == 128
    assert PACKED.NUM_SCALES == 4
    assert PACKED.SCALE_BYTES == 16
    assert PACKED.ROPE_BYTES == 128
    assert PACKED.TOKEN_BYTES == 512 + 64 * 2 + (512 // 128) * 4 == 656
    assert PACKED.PAGE_SIZE == 64
    assert PACKED.FP8_MAX == torch.finfo(torch.float8_e4m3fn).max

    geom = PACKED.packed_record_geometry()
    assert (geom.nope_start, geom.nope_end) == (0, 512)
    assert (geom.scale_start, geom.scale_end) == (512, 528)
    assert (geom.rope_start, geom.rope_end) == (528, 656)


# --- cache format discriminator -------------------------------------------


def test_cache_discriminator_accepts_both_live_formats():
    packed = torch.empty(5, 64, 1, 656, dtype=torch.uint8)
    bf16 = torch.empty(5, 64, 1, 576, dtype=torch.bfloat16)
    assert PACKED.cache_is_packed_fp8(packed, 576) is True
    assert PACKED.cache_is_packed_fp8(bf16, 576) is False


@pytest.mark.parametrize(
    "cache,source_dim",
    [
        # A 1-byte cache that is not 656 wide is a mis-sized packed cache,
        # not a different format to fall back on.
        (torch.empty(2, 64, 1, 576, dtype=torch.uint8), 576),
        (torch.empty(2, 64, 1, 660, dtype=torch.uint8), 576),
        (torch.empty(2, 64, 1, 528, dtype=torch.float8_e4m3fn), 576),
        # The record only encodes a 576-element latent.
        (torch.empty(2, 64, 1, 656, dtype=torch.uint8), 512),
        # A 2-byte cache must carry exactly source_dim elements.
        (torch.empty(2, 64, 1, 656, dtype=torch.bfloat16), 576),
    ],
)
def test_cache_discriminator_names_a_missized_cache(cache, source_dim):
    with pytest.raises(ValueError):
        PACKED.cache_is_packed_fp8(cache, source_dim)


# --- quantize / dequantize ------------------------------------------------


def _random_kv(tokens, scale=0.1, seed=0):
    torch.manual_seed(seed)
    return (torch.randn(tokens, 576) * scale).to(torch.bfloat16)


def test_rope_half_is_stored_verbatim():
    """RoPE is NOT quantized (README:117-121), so its bytes must be identical."""

    kv = _random_kv(33)
    packed = PACKED.quantize_mla_kv_packed_reference(kv)
    stored = packed[:, 528:656].contiguous().view(torch.bfloat16)
    assert torch.equal(stored, kv[:, 512:])
    assert torch.equal(
        PACKED.dequantize_mla_kv_packed(packed)[:, 512:], kv[:, 512:]
    )


def test_scales_are_amax_over_448_per_128_block():
    kv = _random_kv(16, seed=3)
    packed = PACKED.quantize_mla_kv_packed_reference(kv)
    scales = packed[:, 512:528].contiguous().view(torch.float32)
    for tile in range(4):
        lo, hi = tile * 128, (tile + 1) * 128
        amax = kv[:, lo:hi].float().abs().amax(dim=-1)
        torch.testing.assert_close(scales[:, tile], amax / 448.0)


def test_roundtrip_error_is_within_e4m3_mantissa():
    """The storage precision this migration trades for 656 B/token.

    e4m3 has 3 mantissa bits, so a value in its NORMAL range (|x| >= 2**-6
    times the block scale) round-trips to within 2**-4 relative. Elements far
    below their block's amax land in e4m3's subnormals, where RELATIVE error
    is unbounded -- one element in ~131k here reaches 22% -- but their
    ABSOLUTE error is bounded by the subnormal spacing, which is why the
    universal invariant is stated per block against that block's amax:

        |err| <= amax_of_block * 2**-4

    (normal range: |err| <= |x| * 2**-4 <= amax * 2**-4; subnormal range:
    |err| <= scale * 2**-10 = amax * 2**-10 / 448, far smaller.)

    Asserting both halves means a numerics regression shows up as a unit-test
    failure rather than only as a parity-gate failure on the node.
    """

    kv = _random_kv(257, seed=7)
    got = PACKED.dequantize_mla_kv_packed(
        PACKED.quantize_mla_kv_packed_reference(kv)
    )
    nope_ref = kv.float()[:, :512]
    nope_got = got.float()[:, :512]
    err = (nope_got - nope_ref).abs()

    blocks_ref = nope_ref.view(-1, 4, 128)
    amax = blocks_ref.abs().amax(dim=-1, keepdim=True)
    assert bool((err.view(-1, 4, 128) <= amax * 2.0**-4).all())

    # In the normal range the relative bound holds elementwise.
    scale = amax / PACKED.FP8_MAX
    normal = blocks_ref.abs() >= scale * 2.0**-6
    assert bool(normal.any())
    rel_blocks = err.view(-1, 4, 128) / blocks_ref.abs().clamp_min(1e-30)
    assert rel_blocks[normal].max().item() <= 2.0**-4

    # And the bulk is far better than the worst case (design: RMS 2-3%).
    rel = err / nope_ref.abs().clamp_min(1e-6)
    assert rel.pow(2).mean().sqrt().item() < 0.05


def test_block_amax_element_reconstructs_exactly():
    """The amax element maps to +-448, the one exactly representable end."""

    kv = torch.zeros(1, 576, dtype=torch.bfloat16)
    kv[0, 0] = 2.0
    kv[0, 1] = -2.0
    kv[0, 5] = 0.5
    got = PACKED.dequantize_mla_kv_packed(
        PACKED.quantize_mla_kv_packed_reference(kv)
    )
    assert got[0, 0].item() == 2.0
    assert got[0, 1].item() == -2.0


def test_values_beyond_the_block_scale_saturate_rather_than_overflow():
    """No input can overflow e4m3: the scale is derived from the block amax."""

    kv = torch.zeros(2, 576, dtype=torch.bfloat16)
    kv[0, :128] = 57344.0  # bf16 max normal-ish magnitude
    kv[1, :128] = -57344.0
    packed = PACKED.quantize_mla_kv_packed_reference(kv)
    nope = packed[:, :512].contiguous().view(torch.float8_e4m3fn).float()
    assert nope[0, :128].max().item() == 448.0
    assert nope[1, :128].min().item() == -448.0
    assert torch.isfinite(nope).all()
    got = PACKED.dequantize_mla_kv_packed(packed).float()
    assert torch.isfinite(got).all()


# --- the zero block: the one deliberate divergence from SGLang -------------


def test_an_all_zero_token_packs_to_all_zero_bytes():
    """The degenerate input, defined instead of left to NaN.

    SGLang's two implementations disagree here (torch ref stores an e4m3 NaN,
    the Triton kernel stores a clamped +-448); both reconstruct garbage from a
    0.0f scale. Defining it keeps a NaN out of the KV cache AND makes a
    quantized zero token byte-identical to a never-written one.
    """

    kv = torch.zeros(4, 576, dtype=torch.bfloat16)
    packed = PACKED.quantize_mla_kv_packed_reference(kv)
    assert bool((packed == 0).all())
    got = PACKED.dequantize_mla_kv_packed(packed)
    assert torch.equal(got, kv)
    assert torch.isfinite(got.float()).all()


def test_a_zero_block_beside_a_live_block_is_independent():
    """Per-128-block scales: one dead block must not disturb its neighbours."""

    kv = _random_kv(8, seed=11)
    kv[:, 128:256] = 0.0
    packed = PACKED.quantize_mla_kv_packed_reference(kv)
    scales = packed[:, 512:528].contiguous().view(torch.float32)
    assert bool((scales[:, 1] == 0.0).all())
    assert bool((scales[:, 0] > 0.0).all())
    got = PACKED.dequantize_mla_kv_packed(packed)
    assert bool((got[:, 128:256] == 0).all())
    assert torch.isfinite(got.float()).all()


def test_a_freshly_allocated_page_dequantizes_to_zero():
    """The allocation invariant the packed pool relies on.

    ``gpu_paged_kv_manager`` allocates the cache with ``torch.zeros``; an
    untouched token is therefore scale bits 0.0f with 0x00 data bytes, which
    must read back as 0 and not as NaN, because a short row's unselected slots
    and a padding row's -1 indices both land on untouched bytes.
    """

    page = torch.zeros(64, 656, dtype=torch.uint8)
    got = PACKED.dequantize_mla_kv_packed(page)
    assert got.shape == (64, 576)
    assert bool((got == 0).all())
    assert torch.isfinite(got.float()).all()


def test_amax_below_the_bf16_subnormal_floor_still_round_trips():
    """A tiny-but-nonzero block keeps a real scale, not a floored one.

    The aux quantizer floors amax at 1e-4 because DeepGEMM's logits consume
    the scale directly; the packed record has no such consumer -- FlashMLA
    multiplies by whatever fp32 scale is stored -- so flooring here would only
    destroy small values. This pins that there is no floor.
    """

    kv = torch.zeros(1, 576, dtype=torch.bfloat16)
    kv[0, :128] = 1.0e-6
    packed = PACKED.quantize_mla_kv_packed_reference(kv)
    scales = packed[:, 512:528].contiguous().view(torch.float32)
    assert scales[0, 0].item() > 0.0
    assert scales[0, 0].item() < 1.0e-4
    got = PACKED.dequantize_mla_kv_packed(packed)
    torch.testing.assert_close(
        got[0, :128].float(),
        kv[0, :128].float(),
        rtol=2.0**-4,
        atol=0.0,
    )


# --- fidelity of the port to SGLang's quantizer ---------------------------


def _sglang_ref_quantize(kv):
    """Verbatim transcription of SGLang's `_quantize_k_cache_ref`.

    Source: references/sglang/python/sglang/srt/layers/attention/nsa/
    quant_k_cache.py (`_quantize_k_cache_ref`), itself copied from FlashMLA's
    tests/quant.py. Transcribed rather than imported so this test does not
    depend on a vendored external tree being present.

    Note the two numerical choices that differ from the SHIPPED
    `_quantize_k_cache_fast` kernel, and therefore from our port: the amax
    reduction stays in bf16 (the input dtype), and the quantization is a true
    division rather than a multiply by the reciprocal.
    """

    dv, tile = 512, 128
    num_tiles = dv // tile
    out = torch.empty(
        (kv.shape[0], dv + num_tiles * 4 + 2 * (576 - dv)),
        dtype=torch.float8_e4m3fn,
    )
    nope = out[..., :dv]
    scale = out[..., dv : dv + num_tiles * 4].view(torch.float32)
    rope = out[..., dv + num_tiles * 4 :].view(torch.bfloat16)
    rope[:] = kv[..., dv:]
    for tile_idx in range(num_tiles):
        lo, hi = tile_idx * tile, (tile_idx + 1) * tile
        inv = torch.abs(kv[..., lo:hi]).max(dim=-1).values / 448.0
        scale[:, tile_idx] = inv
        inv = inv.unsqueeze(-1)
        nope[..., lo:hi] = (kv[..., lo:hi].float() / inv.float()).to(
            torch.float8_e4m3fn
        )
    return out.view(torch.uint8)


def _port_with_sglang_ref_numerics(kv):
    """Our packing, with the SGLang REF's two numerical choices put back."""

    out = torch.zeros(kv.shape[0], 656, dtype=torch.uint8)
    scales = out[:, 512:528].view(torch.float32)
    for tile_idx in range(4):
        lo, hi = tile_idx * 128, (tile_idx + 1) * 128
        inv = torch.abs(kv[:, lo:hi]).max(dim=-1).values / 448.0  # bf16
        scales[:, tile_idx] = inv
        quantized = (kv[:, lo:hi].float() / inv.float().unsqueeze(-1)).to(
            torch.float8_e4m3fn
        )
        out[:, lo:hi] = quantized.view(torch.uint8)
    out[:, 528:656] = kv[:, 512:].contiguous().view(torch.uint8)
    return out


def test_port_reproduces_the_sglang_reference_byte_for_byte_when_its_numerics_are_restored():
    """Localizes the port's divergence to exactly the two named choices.

    If this passes while the next test shows a difference, then our layout,
    region order, scale definition and RoPE handling are identical to the
    oracle's and the ONLY differences are the deliberate ones.
    """

    kv = _random_kv(64, seed=21)
    assert torch.equal(_port_with_sglang_ref_numerics(kv), _sglang_ref_quantize(kv))


def test_the_only_divergence_from_the_sglang_reference_is_the_scale_reduction():
    """Our fp32 reduction is never worse, and the layout is untouched."""

    kv = _random_kv(64, seed=23)
    ours = PACKED.quantize_mla_kv_packed_reference(kv)
    theirs = _sglang_ref_quantize(kv)

    # RoPE bytes and the record's region boundaries are identical.
    assert torch.equal(ours[:, 528:656], theirs[:, 528:656])

    ours_s = ours[:, 512:528].contiguous().view(torch.float32)
    theirs_s = theirs[:, 512:528].contiguous().view(torch.float32)
    # Their scale is ours rounded to bf16.
    torch.testing.assert_close(
        theirs_s, ours_s.to(torch.bfloat16).float(), rtol=0.0, atol=0.0
    )

    # And ours reconstructs at least as accurately.
    ref = kv.float()[:, :512]
    err_ours = (
        PACKED.dequantize_mla_kv_packed(ours).float()[:, :512] - ref
    ).abs()
    deq_theirs = PACKED.dequantize_mla_kv_packed(theirs).float()[:, :512]
    err_theirs = (deq_theirs - ref).abs()
    assert err_ours.mean().item() <= err_theirs.mean().item()


# --- quantize-once / byte-copy invariant ---------------------------------


def test_every_hop_after_the_quantizer_is_a_byte_copy():
    """D2H -> shm page -> H2D must be byte-identical to the written row.

    The GPU store, the staging buffer, the host append and the H2D restore are
    all byte moves of one 656-byte record, so the same bytes must survive a
    page-granular round trip. Modelled here with host buffers; the node
    battery runs the same assertion across the real D2H/H2D path.
    """

    kv = _random_kv(64, seed=31)
    produced = PACKED.quantize_mla_kv_packed_reference(kv)

    # GPU paged store: token-major into a [page_size, 656] page.
    gpu_page = torch.zeros(64, 656, dtype=torch.uint8)
    gpu_page.copy_(produced)
    # Host staging buffer handed to the append (same bytes, no requantize).
    staged = produced.clone()
    assert torch.equal(gpu_page, staged)
    # D2H into one shm page, then H2D back into a fresh GPU page.
    shm_page = torch.zeros(64 * 656, dtype=torch.uint8)
    shm_page.copy_(staged.reshape(-1))
    restored = torch.zeros(64, 656, dtype=torch.uint8)
    restored.reshape(-1).copy_(shm_page)

    assert torch.equal(restored, produced)
    # Byte identity implies value identity, which is the point: the second
    # quantization that would break it does not exist.
    assert torch.equal(
        PACKED.dequantize_mla_kv_packed(restored),
        PACKED.dequantize_mla_kv_packed(produced),
    )


def test_invalid_rows_select_to_a_zero_record():
    """The writer's row mask, on the actual uint8 record.

    The decode writer selects an invalid row to the zero record with
    ``torch.where(mask, packed, zeros_row, out=staged)`` -- a broadcast
    integer select with an ``out=``, which is not something every torch op
    accepts, and which is also the one place an invalid row's bytes could
    escape into a host page. Pinned here rather than discovered on the node.
    """

    packed = torch.randint(1, 255, (5, 1, 1, 656), dtype=torch.uint8)
    zeros_row = torch.zeros(1, 1, 1, 656, dtype=torch.uint8)
    staged = torch.full((5, 1, 1, 656), 0xAB, dtype=torch.uint8)
    valid = torch.tensor([True, False, True, False, True]).view(5, 1, 1, 1)

    torch.where(valid, packed, zeros_row, out=staged)

    assert staged.dtype == torch.uint8
    for row in (0, 2, 4):
        assert torch.equal(staged[row], packed[row])
    for row in (1, 3):
        assert bool((staged[row] == 0).all())
    # A zero record is a VALID zero token, not a hole: it dequantizes to 0,
    # identically to a never-written slot of a freshly allocated page.
    assert bool((PACKED.dequantize_mla_kv_packed(staged[1]) == 0).all())


def test_quantizing_twice_is_idempotent_on_the_reconstructed_values():
    """Re-quantizing a dequantized record must not drift further.

    Not a path we take -- it is the SGLang branch we rejected -- but if a
    future change re-introduces a second quantization point, this is what it
    must still satisfy, and the assertion documents why one point is cheaper.
    """

    kv = _random_kv(32, seed=37)
    once = PACKED.quantize_mla_kv_packed_reference(kv)
    twice = PACKED.quantize_mla_kv_packed_reference(
        PACKED.dequantize_mla_kv_packed(once)
    )
    assert torch.equal(once, twice)


# --- operand validation ---------------------------------------------------


@pytest.mark.parametrize(
    "kv,out,match",
    [
        (
            torch.empty(4, 1, 576, dtype=torch.bfloat16),
            torch.empty(4, 656, dtype=torch.uint8),
            "flattened",
        ),
        (
            torch.empty(4, 512, dtype=torch.bfloat16),
            torch.empty(4, 656, dtype=torch.uint8),
            "576 source elements",
        ),
        (
            torch.empty(4, 576, dtype=torch.float32),
            torch.empty(4, 656, dtype=torch.uint8),
            "must be bfloat16",
        ),
        (
            torch.empty(4, 576, dtype=torch.bfloat16),
            torch.empty(4, 576, dtype=torch.uint8),
            r"writes \[4, 656\]",
        ),
        (
            torch.empty(4, 576, dtype=torch.bfloat16),
            torch.empty(4, 656, dtype=torch.bfloat16),
            "byte carrier",
        ),
    ],
)
def test_operand_validation_names_the_violation(kv, out, match):
    with pytest.raises(ValueError, match=match):
        PACKED.validate_packed_quantize_operands(kv, out)


def test_non_contiguous_destination_is_rejected():
    big = torch.empty(4, 1312, dtype=torch.uint8)
    with pytest.raises(ValueError, match="contiguous destination"):
        PACKED.validate_packed_quantize_operands(
            torch.empty(4, 576, dtype=torch.bfloat16), big[:, :656]
        )


def test_zero_token_batch_is_a_no_op():
    kv = torch.empty(0, 576, dtype=torch.bfloat16)
    out = PACKED.quantize_mla_kv_packed_reference(kv)
    assert out.shape == (0, 656)
