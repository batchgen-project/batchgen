# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                          #
#                                                                               #
#  licensed under the apache license, version 2.0 (the "license");              #
# ---------------------------------------------------------------------------- #

"""Prefill->decode KDA state reshard (Kimi-Linear M2b, CORE).

Prefill runs streamed DP-32: a sequence's FULL 96-head KDA state (recurrent +
short-conv) is produced on its single ``assigned_rank``. Decode runs
DP-(world/G) x TP-G: the G ranks of the owning ``decode_dp_group`` each hold a
HEAD SHARD of that state (96/G heads per rank, e.g. 12 for G=8), matching the
head-parallel weight sharding the PSM already applies (M2a
``_head_shard_kda_tensor``). The transition therefore

  1. head-slices the full state into G shards along the head axis, and
  2. moves shard g to rank ``group_base + g`` (the "rank->group move"),

over the existing Gloo transport (``KVMigrationHelper``). This module holds the
PURE, GPU-free slice/gather/plan arithmetic so head-independence (P0.6) is
bit-exact and unit-testable; the collective execution is the worker's.

State tensor layouts (per sequence, stacked over its L KDA layers), matching
``KDAStateGPUManager`` per-layer views and the M2a parity test:
  * recurrent : ``[..., num_heads, head_dim, head_dim]`` — head axis is the
    third-from-last (``ndim - 3``).
  * conv (q/k/v): ``[..., num_heads * head_dim, W-1]`` — the conv_dim axis
    (``ndim - 2``) is sliced by contiguous per-head blocks of ``head_dim``.
"""

from __future__ import annotations

from typing import List, Tuple

import torch


def _check_group(num_heads: int, group_size: int, group_rank: int) -> int:
    if group_size <= 0:
        raise ValueError(f"group_size must be > 0, got {group_size}")
    if num_heads % group_size != 0:
        raise ValueError(
            f"num_heads {num_heads} not divisible by group_size {group_size}"
        )
    if not (0 <= group_rank < group_size):
        raise ValueError(
            f"group_rank {group_rank} out of range [0,{group_size})"
        )
    return num_heads // group_size


def head_shard_recurrent(
    recurrent: torch.Tensor, group_size: int, group_rank: int, num_heads: int
) -> torch.Tensor:
    """Slice the recurrent state to this rank's head block ``[g*Hl:(g+1)*Hl]``.

    Head axis is ``recurrent.ndim - 3`` (the two trailing axes are the
    head_dim x head_dim state matrix). Returns a view (contiguous-friendly).
    """
    Hl = _check_group(num_heads, group_size, group_rank)
    axis = recurrent.ndim - 3
    if recurrent.shape[axis] != num_heads:
        raise ValueError(
            f"recurrent head axis {axis} has size {recurrent.shape[axis]}, "
            f"expected num_heads {num_heads}"
        )
    return recurrent.narrow(axis, group_rank * Hl, Hl)


def gather_recurrent(shards: List[torch.Tensor]) -> torch.Tensor:
    """Inverse of ``head_shard_recurrent``: concat G shards along the head axis."""
    if not shards:
        raise ValueError("gather_recurrent needs at least one shard")
    return torch.cat(shards, dim=shards[0].ndim - 3)


def head_shard_conv(
    conv: torch.Tensor,
    group_size: int,
    group_rank: int,
    num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Slice a conv state (q/k/v) to this rank's head block on the conv_dim axis.

    conv_dim = num_heads * head_dim; the block is ``[g*Hl*head_dim : ...]`` on
    axis ``conv.ndim - 2`` (last axis is the W-1 conv history).
    """
    Hl = _check_group(num_heads, group_size, group_rank)
    axis = conv.ndim - 2
    if conv.shape[axis] != num_heads * head_dim:
        raise ValueError(
            f"conv dim axis {axis} has size {conv.shape[axis]}, expected "
            f"num_heads*head_dim {num_heads * head_dim}"
        )
    return conv.narrow(axis, group_rank * Hl * head_dim, Hl * head_dim)


def gather_conv(shards: List[torch.Tensor]) -> torch.Tensor:
    """Inverse of ``head_shard_conv``: concat G shards along the conv_dim axis."""
    if not shards:
        raise ValueError("gather_conv needs at least one shard")
    return torch.cat(shards, dim=shards[0].ndim - 2)


def build_reshard_move_plan(
    prefill_rank: int, decode_dp_group: int, group_size: int
) -> List[Tuple[int, int, int]]:
    """Where each head shard must land.

    The full state sits on ``prefill_rank``; decode group ``decode_dp_group``
    owns ranks ``[g*G, (g+1)*G)`` and rank ``group_base + i`` needs head shard
    ``i``. Returns ``[(src_rank, dst_rank, shard_index)]`` for i in range(G).
    A ``(src == dst)`` entry is a local slice (no transport); the rest are Gloo
    rank->group moves.
    """
    if group_size <= 0:
        raise ValueError(f"group_size must be > 0, got {group_size}")
    group_base = decode_dp_group * group_size
    return [(prefill_rank, group_base + i, i) for i in range(group_size)]
