"""CPU tests for the DeepGEMM grouped-prefill helpers (no deep_gemm import)."""

import pytest
import torch

from batchgen.moe.grouped_fp8_deepgemm_moe import (
    build_grouped_layout,
    dense_group_view,
)


def _arena_members(groups=4, shape=(8, 16), pad_elems=0, dtype=torch.int8):
    step = int(torch.tensor(shape).prod()) + pad_elems
    arena = torch.zeros(groups * step, dtype=dtype)
    members = [
        arena[g * step: g * step + int(torch.tensor(shape).prod())].view(shape)
        for g in range(groups)
    ]
    return arena, members


def test_dense_group_view_over_exact_arena():
    arena, members = _arena_members()
    for g, m in enumerate(members):
        m.fill_(g)
    dense = dense_group_view(members)
    assert dense.shape == (4, 8, 16)
    for g in range(4):
        assert int(dense[g, 0, 0]) == g
    # A write through the view lands in the arena (no copy).
    dense[2].fill_(42)
    assert int(members[2][0, 0]) == 42


def test_dense_group_view_rejects_padded_stride():
    _, members = _arena_members(pad_elems=8)
    with pytest.raises(ValueError, match="uniform dense stride"):
        dense_group_view(members)


def test_dense_group_view_rejects_foreign_allocation():
    _, members = _arena_members()
    stranger = torch.zeros(8, 16, dtype=torch.int8)
    with pytest.raises(ValueError, match="different allocation"):
        dense_group_view([members[0], stranger])


def test_grouped_layout_marks_live_rows_and_padding():
    seqlens = torch.tensor([3, 0, 5])
    cu_seqlens = torch.tensor([0, 128, 128, 256])
    grouped = build_grouped_layout(seqlens, cu_seqlens, 384, alignment=128)
    assert grouped.dtype == torch.int32
    assert grouped[:3].tolist() == [0, 0, 0]
    assert int(grouped[3]) == -1 and int(grouped[127]) == -1
    # Group 1 is empty: its (zero-width) range stays padding.
    assert grouped[128:133].tolist() == [2, 2, 2, 2, 2]
    assert int(grouped[133]) == -1
    assert (grouped[256:] == -1).all()


def test_grouped_layout_rejects_misaligned_group_start():
    seqlens = torch.tensor([3, 5])
    cu_seqlens = torch.tensor([0, 64, 192])
    with pytest.raises(ValueError, match="aligned"):
        build_grouped_layout(seqlens, cu_seqlens, 192, alignment=128)


def test_grouped_layout_rejects_overflow():
    seqlens = torch.tensor([3])
    cu_seqlens = torch.tensor([0, 256])
    with pytest.raises(ValueError, match="exceeds"):
        build_grouped_layout(seqlens, cu_seqlens, 128, alignment=128)


def test_grouped_layout_applies_expert_to_group_permutation():
    # A staggered ring delivers each member its own experts first, so the
    # dense stack is a rotation of expert-id order; rows must map to the
    # expert's SLOT position, not its id.
    seqlens = torch.tensor([3, 2, 1])
    cu_seqlens = torch.tensor([0, 128, 256, 384])
    group_ids = torch.tensor([2, 0, 1], dtype=torch.int32)
    grouped = build_grouped_layout(
        seqlens, cu_seqlens, 384, alignment=128, group_ids=group_ids
    )
    assert grouped[:3].tolist() == [2, 2, 2]
    assert grouped[128:130].tolist() == [0, 0]
    assert int(grouped[256]) == 1
    assert int(grouped[3]) == -1 and int(grouped[257]) == -1


def test_grouped_layout_rejects_short_group_ids():
    seqlens = torch.tensor([3, 2])
    cu_seqlens = torch.tensor([0, 128, 256])
    with pytest.raises(ValueError, match="group_ids"):
        build_grouped_layout(
            seqlens, cu_seqlens, 256, alignment=128,
            group_ids=torch.tensor([0], dtype=torch.int32),
        )
