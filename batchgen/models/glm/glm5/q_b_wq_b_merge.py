# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  Copyright (c) 2025-2026 BatchGen Team                                            #
# ---------------------------------------------------------------------------- #

"""Load-time concatenation of same-input FP8 projections into one GEMM operand.

A 128x128 block-scaled FP8 GEMM reads its weight scale for output row ``n``
from scale row ``n // 128``. Concatenating two such weights along N therefore
keeps every member's numerics only if each member's rows start on a 128-row
boundary: then member ``i``'s scale rows land exactly at
``row_offset_i // 128`` in the concatenated table and the scale-to-row map is
the identity per member. A member whose row count is not a multiple of 128 may
only be LAST -- in any other position the following member's rows would be
scaled by the wrong block, silently. ``MergedGemmAlignmentError`` is that
precondition, raised by name instead of producing wrong numerics.

Torch-only on purpose: the wrapper calls this at load time and the CPU tests
exercise the real function.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch

#: The FP8 weight block size every member's scale table is laid out in
#: (``quantization_config.weight_block_size == [128, 128]``).
FP8_WEIGHT_BLOCK = 128


class MergedGemmAlignmentError(ValueError):
    """A non-last member of a merged block-scaled GEMM is not 128-row aligned."""


class MergedGemmOperandError(ValueError):
    """Members of a merged block-scaled GEMM disagree on K, dtype, device, or
    their scale table does not match their weight."""


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def concat_block_scaled_rows(
    members: Sequence[Tuple[str, torch.Tensor, torch.Tensor]],
    *,
    block: int = FP8_WEIGHT_BLOCK,
) -> Tuple[torch.Tensor, torch.Tensor, List[int], List[int]]:
    """Concatenate ``(name, weight [N_i, K], scale [ceil(N_i/b), ceil(K/b)])``
    members along N.

    Returns ``(weight [sum N_i, K], scale [sum ceil(N_i/b), ceil(K/b)],
    row_offsets, scale_row_offsets)``; both outputs are fresh contiguous
    tensors, so the caller decides which sources to re-point at their slices.
    """
    if len(members) < 2:
        raise MergedGemmOperandError(
            f"a merged GEMM needs at least two members, got {len(members)}"
        )
    first_name, first_w, _ = members[0]
    if first_w.dim() != 2:
        raise MergedGemmOperandError(
            f"{first_name}: weight must be 2-D [N, K], got {tuple(first_w.shape)}"
        )
    k = int(first_w.shape[1])
    k_blocks = _ceil_div(k, block)
    row_offsets: List[int] = []
    scale_row_offsets: List[int] = []
    rows = 0
    scale_rows = 0
    for index, (name, weight, scale) in enumerate(members):
        if weight.dim() != 2 or int(weight.shape[1]) != k:
            raise MergedGemmOperandError(
                f"{name}: weight must be [N, {k}] to share {first_name}'s K, "
                f"got {tuple(weight.shape)}"
            )
        if weight.dtype != first_w.dtype or weight.device != first_w.device:
            raise MergedGemmOperandError(
                f"{name}: weight {weight.dtype}/{weight.device} differs from "
                f"{first_name}'s {first_w.dtype}/{first_w.device}"
            )
        n = int(weight.shape[0])
        expected_scale = (_ceil_div(n, block), k_blocks)
        if (
            scale is None
            or scale.dim() != 2
            or tuple(int(d) for d in scale.shape) != expected_scale
        ):
            raise MergedGemmOperandError(
                f"{name}: block scale must be {expected_scale} for a "
                f"{tuple(weight.shape)} weight in {block}x{block} blocks, got "
                f"{None if scale is None else tuple(scale.shape)}"
            )
        if scale.dtype != torch.float32 or scale.device != first_w.device:
            raise MergedGemmOperandError(
                f"{name}: block scale must be float32 on {first_w.device}, got "
                f"{scale.dtype} on {scale.device}"
            )
        is_last = index == len(members) - 1
        if not is_last and n % block != 0:
            raise MergedGemmAlignmentError(
                f"{name}: {n} rows is not a multiple of {block}, so it can only "
                f"be the LAST member of a merged block-scaled GEMM; in position "
                f"{index} of {len(members)} the next member's rows would be "
                f"scaled by the wrong weight block"
            )
        row_offsets.append(rows)
        scale_row_offsets.append(scale_rows)
        rows += n
        scale_rows += expected_scale[0]
    if scale_rows != _ceil_div(rows, block):
        # Unreachable when every non-last member is aligned; kept as the
        # defining identity of a gap-free scale table.
        raise MergedGemmAlignmentError(
            f"merged scale table has {scale_rows} rows for {rows} weight rows, "
            f"expected {_ceil_div(rows, block)}"
        )
    weight = torch.cat([w for _, w, _ in members], dim=0).contiguous()
    scale = torch.cat([s for _, _, s in members], dim=0).contiguous()
    return weight, scale, row_offsets, scale_row_offsets


@dataclass(frozen=True)
class QbWqbMergedWeights:
    """``[q_b | wq_b]`` as one DeepGEMM NT operand pair."""

    weight: torch.Tensor   # [q_b_rows + wq_b_rows, q_lora_rank] e4m3
    scale: torch.Tensor    # [ceil(rows / 128), q_lora_rank / 128] fp32
    q_b_rows: int
    q_b_scale_rows: int

    @property
    def wq_b_rows(self) -> int:
        return int(self.weight.shape[0]) - self.q_b_rows


def build_q_b_wq_b_merged(
    q_b_weight: torch.Tensor,
    q_b_scale: torch.Tensor,
    wq_b_weight: torch.Tensor,
    wq_b_scale: torch.Tensor,
) -> QbWqbMergedWeights:
    """Build the merged ``[q_b | wq_b]`` operand, q_b first.

    Order: both members are 128-row multiples on GLM-5.x (16384 and 4096), so
    either order is legal; q_b first keeps the q_b slice at row 0 of the
    output, which is the slice the MLA spine reads.
    """
    weight, scale, row_offsets, scale_row_offsets = concat_block_scaled_rows(
        (
            ("q_b_proj", q_b_weight, q_b_scale),
            ("indexer.wq_b", wq_b_weight, wq_b_scale),
        )
    )
    return QbWqbMergedWeights(
        weight=weight,
        scale=scale,
        q_b_rows=row_offsets[1],
        q_b_scale_rows=scale_row_offsets[1],
    )
