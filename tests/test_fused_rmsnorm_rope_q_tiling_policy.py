# ---------------------------------------------------------------------------- #
#  BatchGen — GLM-5                                                            #
#  Copyright (c) 2025-2026 BatchGen Team                                       #
#  Licensed under the Apache License, Version 2.0                              #
# ---------------------------------------------------------------------------- #
"""CPU contract for the fused RMSNorm+RoPE q-side launch tiling.

``select_q_rope_tiling`` decides how the decode-time q RoPE work is spread
over the launch grid.  It is a pure function of the launch geometry, so the
whole partitioning contract is testable without a GPU:

  A  The head tiles cover ``[0, num_heads)`` exactly once -- no head computed
     twice (the kernel writes q_pe in place, so a double-covered head would
     be rotated twice) and none left unrotated.
  B  Both tile axes are powers of two (``tl.arange`` requires it) and wide
     enough to stay ordinary tiles rather than one-or-two-lane special cases,
     and the per-program tile stays inside the register budget.
  C  The launch grows as wide as the CTA target allows and no wider, and the
     knob is pinned for the shapes decode actually uses.
  D  Unsupported or nonsensical geometry fails loud with a named error
     instead of silently falling back to a different partitioning.

The module is loaded from its file with a stubbed ``triton`` so this runs on
the dev box.  Numerical equivalence against the pre-change kernel is a GPU
gate: ``tests/gpu/test_fused_rmsnorm_rope_q_native.py``.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import types

import pytest

_MODULE_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "batchgen_kernels"
    / "triton"
    / "fused_rmsnorm_rope.py"
)


def _load_module_with_stubbed_triton() -> types.ModuleType:
    """Import the kernel module for its pure-Python policy helpers only.

    ``triton`` is not installed on a CPU-only box, and the module evaluates
    ``tl.constexpr`` in its kernel signatures at import time, so the stub has
    to carry that one attribute.  Nothing here executes a kernel body.
    """
    triton_stub = types.ModuleType("triton")
    triton_stub.jit = lambda fn: fn
    language_stub = types.ModuleType("triton.language")
    language_stub.constexpr = object
    triton_stub.language = language_stub

    saved = {
        name: sys.modules.get(name) for name in ("triton", "triton.language")
    }
    sys.modules["triton"] = triton_stub
    sys.modules["triton.language"] = language_stub
    try:
        spec = importlib.util.spec_from_file_location(
            "_fused_rmsnorm_rope_under_test", _MODULE_PATH
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


@pytest.fixture(scope="module")
def kernel_module() -> types.ModuleType:
    return _load_module_with_stubbed_triton()


# GLM-5.2 decode geometry: 64 q heads, qk_rope_head_dim 64 (half_dim 32).
GLM52_HEADS = 64
GLM52_HALF_DIM = 32

# Buckets the graphed decode path captures, plus the extremes.
BUCKETS = [1, 2, 4, 8, 16, 32, 64, 128, 192, 256, 384, 512]


def _is_pow2(value: int) -> bool:
    return value >= 1 and value & (value - 1) == 0


@pytest.mark.parametrize("bsz", BUCKETS)
@pytest.mark.parametrize(
    "num_heads",
    [1, 2, 3, 7, 8, 16, 32, GLM52_HEADS, 96, 128],
)
def test_head_tiles_cover_every_head_exactly_once(kernel_module, bsz, num_heads):
    """A: the tiles partition the head range -- contract A."""
    block_h, _, num_tiles = kernel_module.select_q_rope_tiling(
        bsz, num_heads, GLM52_HALF_DIM
    )

    covered: list[int] = []
    for tile in range(num_tiles):
        covered.extend(
            head
            for head in range(tile * block_h, (tile + 1) * block_h)
            if head < num_heads
        )

    assert covered == list(range(num_heads))
    # No tile is pure padding: that would be a wasted CTA per row.
    assert (num_tiles - 1) * block_h < num_heads


@pytest.mark.parametrize("bsz", BUCKETS)
@pytest.mark.parametrize("num_heads", [1, 3, 7, 8, GLM52_HEADS, 128])
@pytest.mark.parametrize("half_dim", [1, 2, 8, 32, 64, 128, 256])
def test_tile_shape_is_launchable(kernel_module, bsz, num_heads, half_dim):
    """B: power-of-two axes inside the register budget -- contract B."""
    block_h, block_pairs, _ = kernel_module.select_q_rope_tiling(
        bsz, num_heads, half_dim
    )

    assert _is_pow2(block_h)
    assert _is_pow2(block_pairs)
    assert block_pairs >= half_dim
    assert block_pairs >= kernel_module.Q_ROPE_MIN_PAIRS
    assert block_h * block_pairs <= kernel_module.Q_ROPE_MAX_TILE_ELEMS

    # The head axis never degenerates below its floor, whatever the head
    # count: a tile narrower than the head count is fine (there will be more
    # of them) and a tile wider than it is fine too (the surplus lanes are
    # masked).  Only the register budget may push it under the floor, and
    # then it is as wide as that budget allows.
    assert (
        block_h >= kernel_module.Q_ROPE_MIN_BLOCK_H
        or block_h * 2 * block_pairs > kernel_module.Q_ROPE_MAX_TILE_ELEMS
    )


@pytest.mark.parametrize("bsz", BUCKETS)
def test_launch_is_as_wide_as_the_target_allows(kernel_module, bsz):
    """C: the tiling is maximal -- no CTA width left on the table.

    Widening the tile once more must be blocked by something: the CTA target,
    the register budget, or the head count.  Otherwise the policy should have
    widened it and moved more bytes per program.  A launch already at the
    floor has nothing to prove.
    """
    target = kernel_module.Q_ROPE_TARGET_CTAS
    block_h, block_pairs, num_tiles = kernel_module.select_q_rope_tiling(
        bsz, GLM52_HEADS, GLM52_HALF_DIM
    )
    ctas = bsz * (1 + num_tiles)

    assert ctas >= target or block_h <= kernel_module.Q_ROPE_MIN_BLOCK_H

    widening_fits_the_budget = (
        block_h * 2 * block_pairs <= kernel_module.Q_ROPE_MAX_TILE_ELEMS
    )
    if (
        block_h * 2 <= GLM52_HEADS
        and block_h >= kernel_module.Q_ROPE_MIN_BLOCK_H
        and widening_fits_the_budget
    ):
        wider_tiles = -(-GLM52_HEADS // (block_h * 2))
        assert bsz * (1 + wider_tiles) < target


def test_pinned_tiling_for_the_decode_buckets(kernel_module):
    """C: pin the knob for the geometries the decode path actually launches.

    GLM-5 attention is data-parallel, so a rank's bucket width is the whole
    row count it was handed.  Before this change every bucket ran on ``bsz``
    CTAs with ``num_heads`` serial memory round trips inside each one; these
    are the replacement launch widths, pinned so a later retune is a visible
    diff rather than a silent one.
    """
    expected = {
        # bsz: (BLOCK_H, BLOCK_PAIRS, head tiles, CTAs)
        1: (16, 32, 4, 1 * 5),
        8: (16, 32, 4, 8 * 5),
        16: (16, 32, 4, 16 * 5),
        64: (16, 32, 4, 64 * 5),
        128: (16, 32, 4, 128 * 5),
        256: (32, 32, 2, 256 * 3),
        512: (32, 32, 2, 512 * 3),
    }
    for bsz, (want_h, want_pairs, want_tiles, want_ctas) in expected.items():
        block_h, block_pairs, num_tiles = kernel_module.select_q_rope_tiling(
            bsz, GLM52_HEADS, GLM52_HALF_DIM
        )
        got = (block_h, block_pairs, num_tiles, bsz * (1 + num_tiles))
        assert got == (want_h, want_pairs, want_tiles, want_ctas), (
            f"bsz={bsz}: got {got}, want "
            f"{(want_h, want_pairs, want_tiles, want_ctas)}"
        )


def test_every_bucket_beats_the_previous_single_cta_per_row_launch(kernel_module):
    """C: the whole point of the change, stated as a contract.

    The pre-change launch was ``grid = (bsz,)``.  For every bucket the new
    launch must be strictly wider, because that serial head loop is what the
    extra CTAs replace.
    """
    for bsz in BUCKETS:
        _, _, num_tiles = kernel_module.select_q_rope_tiling(
            bsz, GLM52_HEADS, GLM52_HALF_DIM
        )
        assert bsz * (1 + num_tiles) > bsz


@pytest.mark.parametrize("half_dim", [257, 512, 4096])
def test_oversized_rope_half_dim_fails_loud(kernel_module, half_dim):
    """D: unsupported head dim raises, never silently retiles."""
    with pytest.raises(ValueError, match="rope half-dim of at most"):
        kernel_module.select_q_rope_tiling(64, GLM52_HEADS, half_dim)


@pytest.mark.parametrize(
    ("bsz", "num_heads", "half_dim", "match"),
    [
        (0, 64, 32, "bsz must be positive"),
        (-1, 64, 32, "bsz must be positive"),
        (64, 0, 32, "num_heads must be positive"),
        (64, -4, 32, "num_heads must be positive"),
        (64, 64, 0, "half_dim must be positive"),
        (64, 64, -2, "half_dim must be positive"),
    ],
)
def test_degenerate_geometry_fails_loud(
    kernel_module, bsz, num_heads, half_dim, match
):
    """D: no zero-sized or negative launch is silently accepted."""
    with pytest.raises(ValueError, match=match):
        kernel_module.select_q_rope_tiling(bsz, num_heads, half_dim)


def test_tiling_is_a_pure_function_of_the_geometry(kernel_module):
    """The policy must be deterministic: same geometry, same tiling."""
    first = kernel_module.select_q_rope_tiling(64, GLM52_HEADS, GLM52_HALF_DIM)
    for _ in range(4):
        assert (
            kernel_module.select_q_rope_tiling(64, GLM52_HEADS, GLM52_HALF_DIM)
            == first
        )
