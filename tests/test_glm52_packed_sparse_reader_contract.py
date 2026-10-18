"""The parts of the sparse FP8 reader contract that need no GPU.

The kernel call itself is node-only, but two of its preconditions are pure
tensor/arithmetic logic and are exactly where a silent wrong answer would come
from, so they are measured here:

* the index-tensor width must be a whole number of the kernel's 64-index
  blocks, because the kernel sweeps whole blocks and `FLASH_ASSERT`s the
  divisibility;
* every row must carry at least one valid index, because a row of only -1
  leaves the softmax denominator at zero and the epilogue divides by it,
  returning NaN for that row.
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


def test_reader_validates_before_it_needs_the_kernel():
    """Consistent operands get past validation and only then need flash_mla.

    Pinning this keeps the guards above from silently becoming unreachable: if
    the kernel were resolved first, every one of them would be dead on any
    machine without FlashMLA installed.
    """

    with pytest.raises((ImportError, ModuleNotFoundError)):
        READER.run_sparse_fp8_mla_decode(**_reader_operands())


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
