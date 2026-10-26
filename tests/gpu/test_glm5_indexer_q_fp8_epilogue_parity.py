"""GPU byte-parity for the indexer-Q fp8 cast folded into the RoPE+Hadamard epilogue.

The fold replaces two whole-tensor ATen launches -- ``clamp(-448, 448)`` then a
bf16 -> ``float8_e4m3fn`` ``copy_`` -- with a few instructions in the kernel's
epilogue. "Byte-identical" is the requirement, not "close": the indexer feeds a
top-k over the whole context, so a single flipped e4m3 byte can swap two logits
and change which 2048 tokens attention reads.

The reference here IS the removed code. Calling the kernel without a fp8 buffer
gives exactly the bf16 tensor the old path produced, and
``saturate_q_to_e4m3_range(...).to(torch.float8_e4m3fn)`` on it is exactly the
two launches that used to follow. So the parity claim is self-contained: no
checkout of the pre-fold commit is needed to prove it, and that also means this
file keeps proving it after the old code is gone.

The file also carries the rotation-arm parity for the same kernel. GLM-5.2's
page-split FP8 route follows upstream V3.2 and drops the Hadamard incoherence
rotation on BOTH q and k (``indexer_applies_hadamard``), which is the
``apply_hadamard=False`` instantiation; the bf16 AoS indexer profiles keep the
rotation. Both arms are proven here against an INDEPENDENT fp32 reference --
interleaved RoPE over the leading 64 dims, then the normalized Sylvester
Hadamard or nothing -- so neither arm can quietly become the other.

Three things are checked for the fp8 fold:

1. the bf16 output is unchanged when the fp8 epilogue is switched on -- the
   fold must not perturb the path it rides along with;
2. the fp8 bytes equal the ATen pair's bytes, over the production bucket
   shapes and over adversarial magnitudes;
3. the downstream scorer agrees -- same logits, same top-k -- whether it casts
   q itself or trusts the epilogue.

Adversarial values are reached by construction rather than by luck. With the
cos/sin caches set to 1 and 0 the kernel's RoPE is the identity, and a row that
is the constant ``v`` turns into ``(sqrt(128) * v, 0, ..., 0)``: one knob that
places the output at any magnitude, including both sides of 448, bf16's maximum
and infinity. NaN and -NaN are fed directly.

Run on an SM90 node:
    pytest tests/gpu/test_glm5_indexer_q_fp8_epilogue_parity.py -v
"""

from __future__ import annotations

import math

import pytest
import torch


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="indexer-Q fp8 epilogue parity needs an SM90 CUDA device",
)

_HEAD_DIM = 128
_NUM_INDEX_HEADS = 32
# Decode bucket sizes the GLM-5.2 graph captures.
_BUCKETS = (8, 32, 64, 128)
_HADAMARD_SCALE = _HEAD_DIM ** -0.5
_E4M3_448 = 0x7E  # 0 1111 110 -- e4m3's largest finite magnitude
_E4M3_NAN = 0x7F


def _kernels():
    from batchgen_kernels.attention.dsa.indexer import fused_rope_hadamard_out
    from batchgen_kernels.attention.dsa.paged_mqa_logits import (
        saturate_q_to_e4m3_range,
    )

    return fused_rope_hadamard_out, saturate_q_to_e4m3_range


def _identity_rope_tables(max_pos: int, device: torch.device):
    """cos = 1, sin = 0 -- the kernel's RoPE stage becomes the identity."""
    cos = torch.ones(max_pos, 64, dtype=torch.float32, device=device)
    sin = torch.zeros(max_pos, 64, dtype=torch.float32, device=device)
    return cos, sin


def _random_rope_tables(max_pos: int, device: torch.device, seed: int = 0):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    angles = torch.rand(max_pos, 64, generator=gen, dtype=torch.float32) * (2 * math.pi)
    return angles.cos().to(device), angles.sin().to(device)


def _run(x, cos, sin, positions, *, with_fp8, apply_hadamard=True):
    """Return (bf16 out, fp8 out or None) for one kernel launch."""
    fused_rope_hadamard_out, _ = _kernels()
    out = torch.empty_like(x)
    out_fp8 = (
        torch.empty(x.shape, dtype=torch.float8_e4m3fn, device=x.device)
        if with_fp8
        else None
    )
    fused_rope_hadamard_out(
        x, cos, sin, positions, out, _HADAMARD_SCALE, out_fp8=out_fp8,
        apply_hadamard=apply_hadamard,
    )
    return out, out_fp8


def _sylvester_hadamard_128(device: torch.device) -> torch.Tensor:
    """``H[e, f] = (-1) ** popcount(e & f)``, built here and not imported, so
    the reference stays independent of the kernel's own helpers."""
    idx = torch.arange(128, dtype=torch.int64, device=device)
    bits = idx[:, None] & idx[None, :]
    parity = torch.zeros_like(bits)
    for shift in range(7):
        parity = parity ^ ((bits >> shift) & 1)
    return torch.where(
        parity.bool(),
        torch.full_like(parity, -1, dtype=torch.float32),
        torch.ones_like(parity, dtype=torch.float32),
    )


def _reference_rope_then_maybe_rotate(x, cos, sin, positions, *, apply_hadamard):
    """Independent fp32 chain: interleaved RoPE on dims 0..63, then H/sqrt(128).

    Returns bf16, the dtype the kernel stores. The 1/sqrt(128) factor is part
    of the normalized Hadamard, so the no-rotation arm applies no scale at all.
    """
    rows = x.reshape(-1, _HEAD_DIM).float()
    pos = positions.reshape(-1).long()
    cos_rows = cos.index_select(0, pos)[:, : _HEAD_DIM // 4].float()
    sin_rows = sin.index_select(0, pos)[:, : _HEAD_DIM // 4].float()

    out = rows.clone()
    even = rows[:, :64:2]
    odd = rows[:, 1:64:2]
    out[:, :64:2] = even * cos_rows - odd * sin_rows
    out[:, 1:64:2] = odd * cos_rows + even * sin_rows

    if apply_hadamard:
        out = (out @ _sylvester_hadamard_128(x.device)) * _HADAMARD_SCALE
    return out.to(torch.bfloat16).reshape(x.shape)


def _aten_reference_bytes(out_bf16):
    """The two launches the epilogue replaced, as raw bytes."""
    _, saturate_q_to_e4m3_range = _kernels()
    ref = saturate_q_to_e4m3_range(out_bf16).to(torch.float8_e4m3fn)
    return ref.view(torch.uint8)


def _assert_fp8_matches_aten(out_bf16, out_fp8, label: str):
    got = out_fp8.view(torch.uint8)
    want = _aten_reference_bytes(out_bf16)
    mismatch = (got != want).nonzero()
    assert mismatch.numel() == 0, (
        f"{label}: {mismatch.shape[0]} of {got.numel()} e4m3 bytes differ from "
        f"the clamp+copy_ reference; first at {mismatch[0].tolist()} "
        f"(kernel {int(got.flatten()[0])} vs aten {int(want.flatten()[0])})"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Production bucket shapes, realistic magnitudes
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("bucket", _BUCKETS)
def test_bf16_output_is_untouched_by_the_fp8_epilogue(bucket):
    device = torch.device("cuda")
    rows = bucket * _NUM_INDEX_HEADS
    gen = torch.Generator(device="cpu").manual_seed(1234 + bucket)
    x = (torch.randn(rows, _HEAD_DIM, generator=gen) * 0.1).to(
        device=device, dtype=torch.bfloat16
    )
    cos, sin = _random_rope_tables(4096, device, seed=bucket)
    positions = torch.randint(
        0, 4096, (rows,), generator=gen, dtype=torch.int64
    ).to(device)

    base, _ = _run(x, cos, sin, positions, with_fp8=False)
    folded, out_fp8 = _run(x, cos, sin, positions, with_fp8=True)

    assert torch.equal(base.view(torch.int16), folded.view(torch.int16)), (
        f"bucket {bucket}: the bf16 RoPE+Hadamard output changed when the fp8 "
        "epilogue was enabled"
    )
    _assert_fp8_matches_aten(folded, out_fp8, f"bucket {bucket}")


@pytest.mark.parametrize("bucket", _BUCKETS)
@pytest.mark.parametrize("magnitude", [0.01, 0.1, 1.0, 10.0, 100.0])
def test_fp8_bytes_match_aten_across_magnitudes(bucket, magnitude):
    device = torch.device("cuda")
    rows = bucket * _NUM_INDEX_HEADS
    gen = torch.Generator(device="cpu").manual_seed(99 + bucket)
    x = (torch.randn(rows, _HEAD_DIM, generator=gen) * magnitude).to(
        device=device, dtype=torch.bfloat16
    )
    cos, sin = _random_rope_tables(4096, device, seed=7)
    positions = torch.randint(
        0, 4096, (rows,), generator=gen, dtype=torch.int64
    ).to(device)

    out, out_fp8 = _run(x, cos, sin, positions, with_fp8=True)
    _assert_fp8_matches_aten(out, out_fp8, f"bucket {bucket} x{magnitude}")


# ──────────────────────────────────────────────────────────────────────────────
# Adversarial values, reached by construction
# ──────────────────────────────────────────────────────────────────────────────

def _constant_row_cases():
    """(label, v) pairs whose output lands at a chosen magnitude.

    With identity RoPE a constant row ``v`` becomes ``sqrt(128) * v`` in
    column 0 and zero elsewhere, so ``v = target / sqrt(128)`` aims the output.
    """
    s = math.sqrt(_HEAD_DIM)
    return [
        ("just below 448", 447.0 / s),
        ("exactly 448", 448.0 / s),
        ("just above 448", 449.0 / s),
        ("far above 448", 4480.0 / s),
        ("bf16 near-max", 3.0e38 / s),
        ("huge -> inf", 3.0e38),
        ("negative far above 448", -4480.0 / s),
        ("zero", 0.0),
        ("negative zero", -0.0),
        # Denormal floats are orders of magnitude below e4m3's smallest
        # magnitude (2^-9), so both paths must land on signed zero -- the one
        # place --use_fast_math's flush-to-zero could have differed.
        ("tiny subnormal-ish", 1e-40),
        ("negative tiny subnormal-ish", -1e-40),
    ]


@pytest.mark.parametrize("label,value", _constant_row_cases())
def test_adversarial_magnitudes_match_aten(label, value):
    device = torch.device("cuda")
    rows = 64 * _NUM_INDEX_HEADS
    x = torch.full(
        (rows, _HEAD_DIM), value, dtype=torch.bfloat16, device=device
    )
    cos, sin = _identity_rope_tables(16, device)
    positions = torch.zeros(rows, dtype=torch.int64, device=device)

    out, out_fp8 = _run(x, cos, sin, positions, with_fp8=True)
    _assert_fp8_matches_aten(out, out_fp8, label)


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_out_of_range_saturates_to_448_and_never_to_nan(sign):
    """torch's bare cast NaNs above range; the clamp is what prevents that."""
    device = torch.device("cuda")
    rows = 64 * _NUM_INDEX_HEADS
    x = torch.full(
        (rows, _HEAD_DIM),
        sign * 4480.0 / math.sqrt(_HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
    )
    cos, sin = _identity_rope_tables(16, device)
    positions = torch.zeros(rows, dtype=torch.int64, device=device)

    out, out_fp8 = _run(x, cos, sin, positions, with_fp8=True)
    col0 = out_fp8.view(torch.uint8)[:, 0]
    want = _E4M3_448 | (0x80 if sign < 0 else 0x00)
    assert torch.all(col0 == want), (
        f"out-of-range column 0 should be {hex(want)} (+/-448), got "
        f"{sorted(set(col0.tolist()))}; bf16 column 0 was {out[0, 0].item()}"
    )
    _assert_fp8_matches_aten(out, out_fp8, f"saturation sign {sign}")


@pytest.mark.parametrize("nan_sign", ["+", "-"])
def test_nan_passes_through_with_its_sign(nan_sign):
    """ATen's clamp leaves NaN alone and torch's cast keeps the sign bit."""
    device = torch.device("cuda")
    rows = 64 * _NUM_INDEX_HEADS
    nan = float("nan") if nan_sign == "+" else -float("nan")
    x = torch.full((rows, _HEAD_DIM), nan, dtype=torch.bfloat16, device=device)
    cos, sin = _identity_rope_tables(16, device)
    positions = torch.zeros(rows, dtype=torch.int64, device=device)

    out, out_fp8 = _run(x, cos, sin, positions, with_fp8=True)
    assert torch.isnan(out).all(), "NaN input must give a NaN bf16 output"
    bytes_ = out_fp8.view(torch.uint8)
    assert torch.all((bytes_ & 0x7F) == _E4M3_NAN), (
        f"NaN must encode as e4m3 NaN, got {sorted(set(bytes_.flatten().tolist()))}"
    )
    _assert_fp8_matches_aten(out, out_fp8, f"{nan_sign}NaN")


def test_inf_is_clamped_not_turned_into_nan():
    device = torch.device("cuda")
    rows = 64 * _NUM_INDEX_HEADS
    x = torch.full(
        (rows, _HEAD_DIM),
        float("inf"),
        dtype=torch.bfloat16,
        device=device,
    )
    cos, sin = _identity_rope_tables(16, device)
    positions = torch.zeros(rows, dtype=torch.int64, device=device)

    out, out_fp8 = _run(x, cos, sin, positions, with_fp8=True)
    _assert_fp8_matches_aten(out, out_fp8, "+inf")
    finite = out_fp8.view(torch.uint8)[torch.isinf(out)]
    if finite.numel():
        assert torch.all((finite & 0x7F) == _E4M3_448), (
            "an infinite bf16 value must clamp to 448, not cast to NaN"
        )


# ──────────────────────────────────────────────────────────────────────────────
# The rotation arms: upstream V3.2 drops the Hadamard on both q and k
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("apply_hadamard", (False, True))
@pytest.mark.parametrize("bucket", _BUCKETS)
def test_each_rotation_arm_matches_the_independent_reference(bucket, apply_hadamard):
    """``apply_hadamard=False`` is the GLM-5.2 production arm; True is the bf16
    AoS indexer's. Both instantiations must compute what they claim."""
    device = torch.device("cuda")
    rows = bucket * _NUM_INDEX_HEADS
    gen = torch.Generator(device="cpu").manual_seed(4242 + bucket)
    x = (
        torch.randn(rows, _HEAD_DIM, generator=gen, dtype=torch.float32)
        / math.sqrt(_HEAD_DIM)
    ).to(device=device, dtype=torch.bfloat16)
    cos, sin = _random_rope_tables(1024, device, seed=7 + bucket)
    positions = torch.randint(
        0, 1024, (rows,), generator=gen, dtype=torch.int64
    ).to(device)

    out, _ = _run(x, cos, sin, positions, with_fp8=False,
                  apply_hadamard=apply_hadamard)
    want = _reference_rope_then_maybe_rotate(
        x, cos, sin, positions, apply_hadamard=apply_hadamard
    )
    # bf16 has 8 mantissa bits; the kernel sums the 7 butterfly stages in a
    # different fp32 order than the matmul, so this is a value check, not bytes.
    torch.testing.assert_close(
        out.float(), want.float(), atol=4e-2, rtol=4e-2
    )


@pytest.mark.parametrize("bucket", _BUCKETS)
def test_the_two_rotation_arms_are_actually_different(bucket):
    """Guards against the switch degenerating into a no-op: a build where both
    arms agree would pass every other test here and still be wrong."""
    device = torch.device("cuda")
    rows = bucket * _NUM_INDEX_HEADS
    gen = torch.Generator(device="cpu").manual_seed(99 + bucket)
    x = (
        torch.randn(rows, _HEAD_DIM, generator=gen, dtype=torch.float32)
        / math.sqrt(_HEAD_DIM)
    ).to(device=device, dtype=torch.bfloat16)
    cos, sin = _random_rope_tables(1024, device, seed=5)
    positions = torch.zeros(rows, dtype=torch.int64, device=device)

    rotated, _ = _run(x, cos, sin, positions, with_fp8=False, apply_hadamard=True)
    plain, _ = _run(x, cos, sin, positions, with_fp8=False, apply_hadamard=False)
    assert not torch.allclose(rotated.float(), plain.float(), atol=1e-3), (
        "the rotation-off instantiation returns the rotated values"
    )


@pytest.mark.parametrize("bucket", _BUCKETS)
def test_the_fp8_epilogue_still_matches_the_aten_pair_without_rotation(bucket):
    """The fold and the rotation switch are independent: the e4m3 bytes of the
    no-rotation arm must still be the clamp+copy_ bytes of its own bf16 out."""
    device = torch.device("cuda")
    rows = bucket * _NUM_INDEX_HEADS
    gen = torch.Generator(device="cpu").manual_seed(777 + bucket)
    x = (
        torch.randn(rows, _HEAD_DIM, generator=gen, dtype=torch.float32)
        / math.sqrt(_HEAD_DIM)
    ).to(device=device, dtype=torch.bfloat16)
    cos, sin = _random_rope_tables(1024, device, seed=11)
    positions = torch.randint(
        0, 1024, (rows,), generator=gen, dtype=torch.int64
    ).to(device)

    out, out_fp8 = _run(
        x, cos, sin, positions, with_fp8=True, apply_hadamard=False
    )
    _assert_fp8_matches_aten(out, out_fp8, f"no-rotation bucket {bucket}")


def test_the_q_wrapper_passes_the_rotation_through():
    """``rope_hadamard_q_out`` is what the captured segments call."""
    from batchgen_kernels.attention.dsa.fused_indexer_score import (
        rope_hadamard_q_out,
    )

    device = torch.device("cuda")
    batch = 8
    gen = torch.Generator(device="cpu").manual_seed(2024)
    q = torch.randn(
        batch, _NUM_INDEX_HEADS, _HEAD_DIM, generator=gen, dtype=torch.bfloat16
    ).to(device)
    cos, sin = _random_rope_tables(1024, device, seed=3)
    positions = torch.randint(
        0, 1024, (batch * _NUM_INDEX_HEADS,), generator=gen, dtype=torch.int64
    ).to(device)

    plain = torch.empty_like(q)
    rope_hadamard_q_out(q, cos, sin, positions, plain, apply_hadamard=False)
    want = _reference_rope_then_maybe_rotate(
        q, cos, sin, positions, apply_hadamard=False
    )
    torch.testing.assert_close(
        plain.float(), want.float(), atol=4e-2, rtol=4e-2
    )


# ──────────────────────────────────────────────────────────────────────────────
# The buffer contract
# ──────────────────────────────────────────────────────────────────────────────

def test_fp8_buffer_is_optional_and_rejected_when_wrong():
    from batchgen_kernels.attention.dsa.fused_indexer_score import (
        rope_hadamard_q_out,
    )

    device = torch.device("cuda")
    batch = 8
    q = torch.randn(
        batch, _NUM_INDEX_HEADS, _HEAD_DIM, dtype=torch.bfloat16, device=device
    )
    out = torch.empty_like(q)
    cos, sin = _random_rope_tables(1024, device)
    positions = torch.zeros(
        batch * _NUM_INDEX_HEADS, dtype=torch.int64, device=device
    )

    # Omitted buffer: the old call shape still works.
    rope_hadamard_q_out(q, cos, sin, positions, out)

    with pytest.raises(TypeError, match="float8_e4m3fn"):
        rope_hadamard_q_out(
            q, cos, sin, positions, out, out_fp8=torch.empty_like(q)
        )

    with pytest.raises(ValueError, match="elements"):
        rope_hadamard_q_out(
            q,
            cos,
            sin,
            positions,
            out,
            out_fp8=torch.empty(
                batch, _HEAD_DIM, dtype=torch.float8_e4m3fn, device=device
            ),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Downstream: the scorer must not care who cast q
# ──────────────────────────────────────────────────────────────────────────────

_PAGE_SIZE = 64
_SCALE_BYTES_PER_TOKEN = 4


def _page_split_aux_cache(batch: int, length: int, device: torch.device, seed: int):
    """A real page-split FP8 aux cache plus its page table, built from bf16 keys.

    Same construction the ReLU-oracle node harness uses: e4m3 values in the
    first ``page_size * head_dim`` bytes of each page, the per-token fp32
    scales in the 4-byte tail.
    """
    from batchgen_kernels.attention.dsa.indexer_k_quant import (
        quantize_indexer_k_fp8,
    )

    gen = torch.Generator(device="cpu").manual_seed(seed)
    pages_per_row = length // _PAGE_SIZE
    num_pages = batch * pages_per_row

    keys = torch.randn(
        num_pages, _PAGE_SIZE, _HEAD_DIM, generator=gen, dtype=torch.float32
    ).to(device=device, dtype=torch.bfloat16)
    k_fp8, scales = quantize_indexer_k_fp8(keys)

    cache = torch.zeros(
        num_pages,
        _PAGE_SIZE,
        1,
        _HEAD_DIM + _SCALE_BYTES_PER_TOKEN,
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    pages = cache.view(torch.uint8).reshape(num_pages, -1)
    pages[:, : _PAGE_SIZE * _HEAD_DIM] = k_fp8.reshape(num_pages, -1).view(torch.uint8)
    pages[:, _PAGE_SIZE * _HEAD_DIM :] = (
        scales.reshape(num_pages, _PAGE_SIZE).contiguous().view(torch.uint8)
    )

    page_table = torch.arange(
        num_pages, dtype=torch.int32, device=device
    ).reshape(batch, pages_per_row)
    return cache, page_table


@pytest.mark.parametrize("bucket", _BUCKETS)
@pytest.mark.parametrize("length", [4096, 8192])
def test_scorer_logits_and_topk_are_identical_either_way(bucket, length):
    """Same q, two routes into ``fp8_paged_mqa_logits``: identical selections.

    This is the end of the chain the fold touches, so it is the claim that
    actually matters -- the selected page ids must not move. One call casts q
    itself (the pre-fold behaviour), the other is handed the bytes the kernel
    epilogue writes.
    """
    pytest.importorskip("deep_gemm")
    from batchgen_kernels.attention.dsa.paged_mqa_scorer import (
        paged_mqa_score_and_topk_out,
        saturate_q_to_e4m3_range,
    )

    device = torch.device("cuda")
    heads, dim = _NUM_INDEX_HEADS, _HEAD_DIM
    topk = 2048
    cache, page_table = _page_split_aux_cache(bucket, length, device, 2026 + bucket)
    pages_per_row = page_table.shape[1]

    gen = torch.Generator(device="cpu").manual_seed(31337 + bucket + length)
    q = (
        torch.randn(bucket, heads, dim, generator=gen, dtype=torch.float32)
        / math.sqrt(dim)
    ).to(device=device, dtype=torch.bfloat16)
    head_gates = torch.randn(
        bucket, heads, generator=gen, dtype=torch.float32
    ).to(device)
    cache_seqlens = torch.full(
        (bucket,), length, dtype=torch.int32, device=device
    )
    # Identity slot indirection: batch row b reads page-table row b.
    slot_indices = torch.arange(bucket, dtype=torch.int32, device=device)

    def score(*, prequantized: bool) -> tuple[torch.Tensor, torch.Tensor]:
        q_fp8 = torch.empty(
            bucket, 1, heads, dim, dtype=torch.float8_e4m3fn, device=device
        )
        if prequantized:
            # Exactly what the kernel epilogue leaves behind.
            q_fp8.view(bucket, heads, dim).copy_(
                saturate_q_to_e4m3_range(q).to(torch.float8_e4m3fn)
            )
        top_k_indices = torch.full(
            (bucket, topk), -1, dtype=torch.int32, device=device
        )
        paged_mqa_score_and_topk_out(
            q=q,
            aux_blocked_k=cache,
            aux_page_table=page_table,
            aux_slot_indices=slot_indices,
            head_gates=head_gates,
            cache_seqlens=cache_seqlens,
            q_fp8=q_fp8,
            block_table=torch.empty(
                bucket, pages_per_row, dtype=torch.int32, device=device
            ),
            context_lens=torch.empty(
                bucket, 1, dtype=torch.int32, device=device
            ),
            top_k_indices=top_k_indices,
            topk=topk,
            max_seqlen=length,
            index_head_dim=dim,
            page_size=_PAGE_SIZE,
            q_prequantized=prequantized,
        )
        return q_fp8, top_k_indices

    q_fp8_self, picked_self = score(prequantized=False)
    q_fp8_folded, picked_folded = score(prequantized=True)

    assert torch.equal(
        q_fp8_self.view(torch.uint8), q_fp8_folded.view(torch.uint8)
    ), "the two q quantizations differ, so the scorer comparison is moot"
    # The production top-k's OUTPUT ORDER is nondeterministic (sub-project
    # glm52-dsa-topk-order-nondeterminism): running ONE route twice, with
    # byte-identical q_fp8, already disagrees positionally on ~92% of entries
    # at L=4096 (measured: same-route 15194/16384, cross-route 15024/16384 --
    # cross-route is no worse than same-route), while the sorted rows are equal
    # in every case. So compare the SELECTED SET, which is the claim this test
    # is actually making ("the selected page ids must not move"); a positional
    # torch.equal here pins the kernel's tie-break order, not the fold.
    assert torch.equal(
        picked_self.sort(dim=-1).values, picked_folded.sort(dim=-1).values
    ), (
        f"bucket {bucket} L={length}: the selected SET moved -- "
        f"{int((picked_self.sort(dim=-1).values != picked_folded.sort(dim=-1).values).sum())}"
        f" of {picked_self.numel()} sorted entries differ"
    )
    # Every selected id must be a real in-range page slot, so a set comparison
    # cannot pass on two equally-corrupt selections.
    for name, picked in (("self", picked_self), ("folded", picked_folded)):
        assert int(picked.min()) >= 0, f"{name}: unfilled top-k slot"
        assert int(picked.max()) < length, (
            f"{name}: selected index {int(picked.max())} is outside L={length}"
        )
