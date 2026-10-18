"""The sparse FP8 reader contract: its CPU-measurable part, then the real call.

Most of this file needs no GPU, because two of the kernel's preconditions are
pure tensor/arithmetic logic and are exactly where a silent wrong answer would
come from, so they are measured on the dev machine:

* the index-tensor width must be a whole number of the kernel's 64-index
  blocks, because the kernel sweeps whole blocks and `FLASH_ASSERT`s the
  divisibility;
* every row must carry at least one valid index, because a row of only -1
  leaves the softmax denominator at zero and the epilogue divides by it,
  returning NaN for that row.

The second of those is a claim ABOUT THE KERNEL, so asserting it only against
our own guard leaves the hazard itself unmeasured. The `@requires_cuda` block
at the bottom therefore calls the REAL
``flash_mla_with_kvcache(indices=..., is_fp8_kvcache=True)`` on a real packed
cache built by our own quantizer, with a batch that contains a fully-empty
padding row: it gates that the guarded call is NaN-free, probes (without
asserting) what the unguarded call does, and pins the guard's idempotence on
the real path rather than only on a tensor.
"""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


def _load_reader():
    """Load the sparse-FP8 reader module by path.

    `batchgen.attention.mla.__init__` pulls GPU-only backends and
    `batchgen_kernels.triton` needs Triton, so the package cannot be imported
    on a CPU-only machine. The module itself imports `flash_mla` lazily
    (inside `_flash_mla_ops`) precisely so its pure logic stays testable here.
    """

    name = "_glm52_packed_sparse_reader_for_test"
    if name in sys.modules:
        return sys.modules[name]
    path = (
        ROOT / "batchgen" / "attention" / "mla" / "packed_fp8_sparse_decode.py"
    )
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


READER = _load_reader()


def test_topk_block_size_matches_the_pin():
    # csrc/sm90/decode/sparse_fp8/components/config.h:15
    assert READER.TOPK_BLOCK_SIZE == 64


@pytest.mark.parametrize(
    "max_kv_tokens,index_topk,expected",
    [
        # The selecting capture uses the whole budget.
        (2048, 2048, 2048),
        (1_000_000, 2048, 2048),
        # An all-short capture rounds its own KV bound UP to a whole block.
        (1, 2048, 64),
        (64, 2048, 64),
        (65, 2048, 128),
        (500, 2048, 512),
        (512, 2048, 512),
        (2047, 2048, 2048),
    ],
)
def test_indices_width_is_whole_kernel_blocks(max_kv_tokens, index_topk, expected):
    assert READER.sparse_indices_width(max_kv_tokens, index_topk) == expected


def test_indices_width_never_exceeds_the_selection_budget():
    """Rounding up must not walk past topk and break the divisibility assert."""

    for topk in (64, 128, 1024, 2048):
        for bound in (1, topk - 1, topk, topk + 1, topk * 3):
            width = READER.sparse_indices_width(bound, topk)
            assert width <= topk
            assert width % READER.TOPK_BLOCK_SIZE == 0
            assert width >= min(bound, topk)


@pytest.mark.parametrize("index_topk", [0, -64, 100, 2000, 63])
def test_indices_width_rejects_a_budget_the_kernel_would_assert_on(index_topk):
    with pytest.raises(ValueError):
        READER.sparse_indices_width(1024, index_topk)


def test_indices_width_rejects_a_nonpositive_bound():
    with pytest.raises(ValueError):
        READER.sparse_indices_width(0, 2048)


# --- the empty-row guard -------------------------------------------------


def test_guard_gives_a_padding_row_one_valid_index():
    """A padding row is all -1 and would otherwise come back NaN."""

    indices = torch.full((4, 8), -1, dtype=torch.int32)
    # Rows 0 and 2 are live; 1 and 3 are padding.
    indices[0] = torch.tensor([5, 9, 13, -1, -1, -1, -1, -1], dtype=torch.int32)
    indices[2] = torch.tensor([0, 1, -1, -1, -1, -1, -1, -1], dtype=torch.int32)
    zeros = torch.zeros(4, dtype=torch.int32)

    READER.guard_sparse_indices_against_empty_rows(indices, zeros)

    # Every row now has a valid first index...
    assert bool((indices[:, 0] >= 0).all())
    # ...and no live row was disturbed.
    assert indices[0].tolist() == [5, 9, 13, -1, -1, -1, -1, -1]
    assert indices[2].tolist() == [0, 1, -1, -1, -1, -1, -1, -1]
    # The guarded rows attend to physical token 0 and keep every other slot
    # masked, so they consume one token's work and produce a finite result
    # that the segment's row mask then zeroes.
    assert indices[1].tolist() == [0] + [-1] * 7
    assert indices[3].tolist() == [0] + [-1] * 7


def test_guard_is_idempotent():
    indices = torch.full((3, 4), -1, dtype=torch.int32)
    zeros = torch.zeros(3, dtype=torch.int32)
    READER.guard_sparse_indices_against_empty_rows(indices, zeros)
    once = indices.clone()
    READER.guard_sparse_indices_against_empty_rows(indices, zeros)
    assert torch.equal(indices, once)


def test_guard_leaves_a_fully_populated_row_untouched():
    indices = torch.arange(2 * 64, dtype=torch.int32).view(2, 64)
    before = indices.clone()
    READER.guard_sparse_indices_against_empty_rows(
        indices, torch.zeros(2, dtype=torch.int32)
    )
    assert torch.equal(indices, before)


def _metadata(batch=4, topk=128, heads=8):
    return READER.SparseFp8MlaMetadata(
        tile_scheduler_metadata=torch.zeros(1, 8, dtype=torch.int32),
        num_splits=torch.zeros(batch + 1, dtype=torch.int32),
        batch_size=batch,
        topk=topk,
        num_heads_q=heads,
    )


def _reader_operands(batch=4, topk=128, heads=8):
    return dict(
        query=torch.zeros(batch, 1, heads, 576, dtype=torch.bfloat16),
        packed_blocked_k=torch.zeros(2, 64, 1, 656, dtype=torch.uint8),
        indices=torch.zeros(batch, 1, topk, dtype=torch.int32),
        block_table=torch.zeros(batch, 1, dtype=torch.int32),
        cache_seqlens=torch.zeros(batch, dtype=torch.int32),
        metadata=_metadata(batch, topk, heads),
        softmax_scale=0.1,
        head_dim_v=512,
    )


@pytest.mark.parametrize(
    "override,match",
    [
        # The metadata fixes how many index blocks the kernel sweeps.
        ({"indices": torch.zeros(4, 1, 64, dtype=torch.int32)}, "indices width"),
        # num_splits is sized batch+1 and the tile schedule is per-row.
        (
            {"query": torch.zeros(3, 1, 8, 576, dtype=torch.bfloat16)},
            "query batch",
        ),
        # The head count sets the work split.
        (
            {"query": torch.zeros(4, 1, 16, 576, dtype=torch.bfloat16)},
            "heads but the metadata",
        ),
        ({"indices": torch.zeros(5, 1, 128, dtype=torch.int32)}, "indices batch"),
    ],
)
def test_reader_rejects_operands_that_disagree_with_its_metadata(override, match):
    """These disagreements are NOT caught downstream.

    The kernel reads the tile schedule the metadata baked and the indices this
    call passes; if they disagree it attends to the wrong window or the wrong
    number of rows and returns a plausible tensor. So the agreement is checked
    here, before the kernel is even resolved -- which is also what makes these
    guards measurable on a machine with no GPU.
    """

    operands = _reader_operands()
    operands.update(override)
    with pytest.raises(ValueError, match=match):
        READER.run_sparse_fp8_mla_decode(**operands)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"topk": 100}, "topk %"),
        ({"topk": 63}, "topk %"),
        ({"num_heads_k": 2}, "num_heads_k == 1"),
        ({"batch_size": 0}, "batch_size must be positive"),
        ({"num_heads_q": 0}, "positive multiple"),
        ({"num_heads_q": 7, "num_heads_k": 2}, "num_heads_k == 1"),
    ],
)
def test_metadata_builder_rejects_shapes_the_pin_asserts_on(kwargs, match):
    base = dict(
        batch_size=4,
        topk=2048,
        num_heads_q=8,
        num_heads_k=1,
        num_q_tokens_per_q_seq=1,
        device=torch.device("cpu"),
    )
    base.update(kwargs)
    with pytest.raises(ValueError, match=match):
        READER.prepare_sparse_fp8_mla_metadata(**base)


class _KernelSeamReached(Exception):
    """Raised by the stubbed import seam, so reaching it is observable."""


def test_reader_validates_before_it_needs_the_kernel(monkeypatch):
    """Validation fires BEFORE the kernel seam is touched; then the seam.

    Pinning this keeps the guards above from silently becoming unreachable: if
    the kernel were resolved first, every one of them would be dead on any
    machine without FlashMLA installed.

    The claim is about ORDER, so the kernel seam is stubbed rather than assumed
    absent. Asserting ``ImportError`` instead made the test say "FlashMLA is
    not installed here", which is not the claim and is false on the node: there
    the real call was reached and raised ``RuntimeError: q must be on CUDA``,
    i.e. the ordering this test exists to pin was in fact HOLDING while the
    test reported a failure. Stubbing ``_flash_mla_ops`` -- the module's one
    lazy import point -- makes the measurement identical with or without
    FlashMLA, on CPU or on a GPU.
    """

    seam_calls = []

    def _stub_flash_mla_ops():
        seam_calls.append(True)
        raise _KernelSeamReached

    monkeypatch.setattr(READER, "_flash_mla_ops", _stub_flash_mla_ops)

    # An inconsistent operand is rejected by OUR layer, and the seam is never
    # reached -- which is what makes every guard above measurable here.
    bad = _reader_operands()
    bad["indices"] = torch.zeros(4, 1, 64, dtype=torch.int32)
    with pytest.raises(ValueError, match="indices width"):
        READER.run_sparse_fp8_mla_decode(**bad)
    assert seam_calls == []

    # Consistent operands get past validation and only then need flash_mla.
    with pytest.raises(_KernelSeamReached):
        READER.run_sparse_fp8_mla_decode(**_reader_operands())
    assert seam_calls == [True]


def test_metadata_builder_validates_before_it_needs_the_kernel(monkeypatch):
    """The same ordering claim for the metadata builder's own guards."""

    seam_calls = []

    def _stub_flash_mla_ops():
        seam_calls.append(True)
        raise _KernelSeamReached

    monkeypatch.setattr(READER, "_flash_mla_ops", _stub_flash_mla_ops)

    base = dict(
        batch_size=4,
        topk=2048,
        num_heads_q=8,
        num_heads_k=1,
        num_q_tokens_per_q_seq=1,
        device=torch.device("cpu"),
    )
    with pytest.raises(ValueError, match="topk %"):
        READER.prepare_sparse_fp8_mla_metadata(**{**base, "topk": 100})
    assert seam_calls == []

    with pytest.raises(_KernelSeamReached):
        READER.prepare_sparse_fp8_mla_metadata(**base)
    assert seam_calls == [True]


def test_guard_rejects_the_wrong_dtype_or_rank():
    with pytest.raises(TypeError):
        READER.guard_sparse_indices_against_empty_rows(
            torch.full((2, 4), -1, dtype=torch.int64),
            torch.zeros(2, dtype=torch.int32),
        )
    with pytest.raises(ValueError):
        READER.guard_sparse_indices_against_empty_rows(
            torch.full((2, 1, 4), -1, dtype=torch.int32),
            torch.zeros(2, dtype=torch.int32),
        )
    with pytest.raises(ValueError):
        READER.guard_sparse_indices_against_empty_rows(
            torch.full((2, 4), -1, dtype=torch.int32),
            torch.zeros(3, dtype=torch.int32),
        )


# --- the real kernel call (node-only) ------------------------------------
#
# Everything above measures OUR layer. What was never measured is the claim
# the layer exists for: that the pinned FlashMLA sparse FP8 kernel, called on a
# cache our quantizer wrote, returns a finite result for a batch containing a
# fully-empty padding row once the guard has run. These tests close that gap.

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="the real sparse FP8 FlashMLA call needs a GPU",
)

# Shapes the pin hard-requires or that mirror the production call:
#   page_block_size == 64 and the whole 656-byte record contiguous
#   (csrc/pybind.cpp:236,249-256); head_dim_k == 576, head_dim_v == 512;
#   num_heads_k == 1 for FP8; topk % 64 == 0.
# num_heads_q == 64 is one BLOCK_M of the kernel's q tiling
# (sparse_fp8/components/config.h:14) and is GLM-5.2's 128 heads at TP=2.
_REAL_PAGES = 4
_REAL_PAGE_SIZE = 64
_REAL_BATCH = 4
_REAL_TOPK = 64
_REAL_HEADS = 64
_REAL_HEAD_DIM = 576
_REAL_HEAD_DIM_V = 512
_REAL_SOFTMAX_SCALE = 0.1
# Row 3 is the padding row: zero selected tokens, i.e. indices all -1, exactly
# what `transform_selected_positions_out` writes for a slot-less bucket row.
_REAL_SELECTED_LENGTHS = (37, 1, 64, 0)


def _packed_cache(device):
    """A real packed-FP8 cache, written by OUR quantizer.

    Returns ``(blocked_k, dequantized)``: the ``[pages, 64, 1, 656]`` uint8
    cache exactly as the pool allocates it, and the bf16 values FlashMLA will
    read back out of it, for the reference attention below.
    """

    from batchgen_kernels.attention.mla_kv_packed import (
        TOKEN_BYTES,
        dequantize_mla_kv_packed,
    )
    from batchgen_kernels.triton.mla_kv_packed import quantize_mla_kv_packed

    torch.manual_seed(20261018)
    total_tokens = _REAL_PAGES * _REAL_PAGE_SIZE
    kv = (
        torch.randn(total_tokens, _REAL_HEAD_DIM, device=device) * 0.1
    ).to(torch.bfloat16)
    packed = torch.zeros(
        total_tokens, TOKEN_BYTES, dtype=torch.uint8, device=device
    )
    quantize_mla_kv_packed(kv, out=packed)
    blocked_k = packed.view(_REAL_PAGES, _REAL_PAGE_SIZE, 1, TOKEN_BYTES)
    return blocked_k, dequantize_mla_kv_packed(packed)


def _real_indices(device):
    """``[batch, topk]`` int32, physical ids, with one all -1 padding row."""

    indices = torch.full(
        (_REAL_BATCH, _REAL_TOPK), -1, dtype=torch.int32, device=device
    )
    for row, length in enumerate(_REAL_SELECTED_LENGTHS):
        if length:
            indices[row, :length] = (
                torch.arange(length, dtype=torch.int32, device=device)
                + row * _REAL_PAGE_SIZE
            )
    return indices


def _real_call_operands(device):
    """Everything `run_sparse_fp8_mla_decode` needs for a real call."""

    pytest.importorskip("flash_mla")
    blocked_k, dequantized = _packed_cache(device)
    query = (
        torch.randn(
            _REAL_BATCH, 1, _REAL_HEADS, _REAL_HEAD_DIM, device=device
        )
        .clamp_(-1.0, 1.0)
        .to(torch.bfloat16)
    )
    metadata = READER.prepare_sparse_fp8_mla_metadata(
        batch_size=_REAL_BATCH,
        topk=_REAL_TOPK,
        num_heads_q=_REAL_HEADS,
        num_heads_k=1,
        num_q_tokens_per_q_seq=1,
        device=device,
    )
    return dict(
        blocked_k=blocked_k,
        dequantized=dequantized,
        query=query,
        metadata=metadata,
        indices=_real_indices(device),
        cache_seqlens=torch.tensor(
            _REAL_SELECTED_LENGTHS, dtype=torch.int32, device=device
        ),
        block_table=torch.arange(
            _REAL_PAGES, dtype=torch.int32, device=device
        )
        .view(1, _REAL_PAGES)
        .repeat(_REAL_BATCH, 1)
        .contiguous(),
    )


def _run_real(ops, indices):
    return READER.run_sparse_fp8_mla_decode(
        query=ops["query"],
        packed_blocked_k=ops["blocked_k"],
        indices=indices.unsqueeze(1),
        block_table=ops["block_table"],
        cache_seqlens=ops["cache_seqlens"],
        metadata=ops["metadata"],
        softmax_scale=_REAL_SOFTMAX_SCALE,
        head_dim_v=_REAL_HEAD_DIM_V,
    )


def _reference_attention(ops, indices):
    """fp32 sparse MLA over the DEQUANTIZED packed cache, per row.

    Reads the same bytes the kernel reads, so FP8 quantization error cancels
    and what is left is accumulation order -- which is why the bound below is
    a sanity bound, not a numerics gate.
    """

    query = ops["query"].float()
    kv = ops["dequantized"].float()
    out = torch.zeros(
        _REAL_BATCH,
        1,
        _REAL_HEADS,
        _REAL_HEAD_DIM_V,
        dtype=torch.float32,
        device=query.device,
    )
    for row in range(_REAL_BATCH):
        valid = indices[row][indices[row] >= 0].long()
        if valid.numel() == 0:
            continue
        k = kv[valid]
        scores = (query[row, 0] @ k.transpose(0, 1)) * _REAL_SOFTMAX_SCALE
        probs = torch.softmax(scores, dim=-1)
        out[row, 0] = probs @ k[:, :_REAL_HEAD_DIM_V]
    return out


@requires_cuda
def test_real_sparse_fp8_call_with_a_padding_row_is_finite():
    """(a) The guarded real call: finite everywhere, including the empty row.

    This is the gate the whole guard exists for. The padding row's own output
    is meaningless -- it attends to physical token 0 and the segment's row mask
    discards it -- but it must not be NaN, because a NaN here is what would
    propagate into the step's reduction.
    """

    device = torch.device("cuda")
    ops = _real_call_operands(device)
    indices = ops["indices"]
    READER.guard_sparse_indices_against_empty_rows(
        indices, torch.zeros(_REAL_BATCH, dtype=torch.int32, device=device)
    )
    out = _run_real(ops, indices)

    assert out.shape == (_REAL_BATCH, 1, _REAL_HEADS, _REAL_HEAD_DIM_V)
    assert out.dtype == torch.bfloat16
    # NO NaN anywhere -- padding row included.
    assert int(torch.isnan(out.float()).sum().item()) == 0
    assert bool(torch.isfinite(out.float()).all())
    # And each live row is finite in its own right, so a finite total cannot
    # come from a row that was silently all zeros.
    for row, length in enumerate(_REAL_SELECTED_LENGTHS):
        if length:
            assert bool(torch.isfinite(out[row].float()).all())
            assert bool((out[row].float().abs() > 0).any())


@requires_cuda
def test_real_sparse_fp8_call_matches_a_dequantized_reference():
    """The same call, checked against fp32 attention over the same bytes.

    Shape and finiteness do not catch "attended to the wrong window", which is
    the failure mode the metadata guards above describe. This does.
    """

    device = torch.device("cuda")
    ops = _real_call_operands(device)
    indices = ops["indices"]
    READER.guard_sparse_indices_against_empty_rows(
        indices, torch.zeros(_REAL_BATCH, dtype=torch.int32, device=device)
    )
    got = _run_real(ops, indices).float()
    want = _reference_attention(ops, indices)

    err = (got - want).abs().max().item()
    print(f"\nREAL_SPARSE_FP8_MAX_ABS_ERR_VS_DEQUANT_REFERENCE {err:.6g}")
    # Sanity bound: both sides read identical FP8 bytes, so only accumulation
    # order and the bf16 output cast separate them.
    assert err <= 5e-2, err


@requires_cuda
def test_unguarded_padding_row_hazard_probe():
    """(b) RECORD, DO NOT ASSERT: the same call with the guard NOT applied.

    `packed_fp8_sparse_decode` note 3 reads the pin as returning NaN for a row
    of only -1 (``rO / rL`` with ``rL == 0``,
    sparse_fp8/components/epilogue.h:34). The pin's own test suite also runs an
    all-indices-invalid case against a reference that zeroes such rows, so what
    the shipped kernel actually emits is an OPEN question about upstream, not a
    property of our code. Asserting either answer would pin a guess, so this
    prints the observation and the node log carries it.

    Run with ``-rA`` (or ``-s``) to see the observation in the log.
    """

    device = torch.device("cuda")
    ops = _real_call_operands(device)
    indices = ops["indices"]
    # Deliberately NOT guarded.
    assert bool((indices[3] < 0).all())
    out = _run_real(ops, indices).float()

    nan_per_row = torch.isnan(out).flatten(1).any(dim=1).tolist()
    live_rows = [
        row for row, length in enumerate(_REAL_SELECTED_LENGTHS) if length
    ]
    print("\nUNGUARDED_EMPTY_ROW_HAZARD_PROBE (observation, NOT a gate)")
    print(f"  selected_lengths     {list(_REAL_SELECTED_LENGTHS)}")
    print(f"  nan_per_row          {nan_per_row}")
    print(f"  nan_elements_total   {int(torch.isnan(out).sum().item())}")
    print(
        "  padding_row_is_nan   "
        f"{nan_per_row[3]}  (row 3 has zero selected tokens)"
    )
    print(
        "  live_rows_are_nan    "
        f"{[nan_per_row[row] for row in live_rows]}"
    )
    print(
        "  reading: NaN in the padding row confirms note 3's hazard and the "
        "guard's necessity; no NaN means the shipped kernel already defines "
        "the empty row, and the guard is belt-and-braces. Either way the "
        "guarded path above is the gate."
    )

    # The only assertion here is that the call RAN, so the probe cannot pass
    # vacuously on a machine where the kernel refused the operands.
    assert out.shape == (_REAL_BATCH, 1, _REAL_HEADS, _REAL_HEAD_DIM_V)


@requires_cuda
def test_guard_is_idempotent_on_the_real_call_path():
    """(c) Guarding twice changes neither the indices nor the kernel's output.

    The captured graph re-runs the guard every step over the same static
    buffer, so "already guarded" is the steady state, not the exception.
    """

    device = torch.device("cuda")
    ops = _real_call_operands(device)
    indices = ops["indices"]
    zeros = torch.zeros(_REAL_BATCH, dtype=torch.int32, device=device)

    READER.guard_sparse_indices_against_empty_rows(indices, zeros)
    after_once = indices.clone()
    out_once = _run_real(ops, indices)

    READER.guard_sparse_indices_against_empty_rows(indices, zeros)
    assert torch.equal(indices, after_once)
    out_twice = _run_real(ops, indices)

    # Same operands, same fixed tile schedule -> bitwise identical replay.
    assert torch.equal(out_once, out_twice)
