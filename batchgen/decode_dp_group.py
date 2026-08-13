# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                          #
#                                                                               #
#  licensed under the apache license, version 2.0 (the "license");              #
# ---------------------------------------------------------------------------- #

"""Serve-group assignment for DP-(world/G) x TP-G resident serving (Kimi-Linear,
Option 1, CORE).

Option 1 (unified resident TP): the world splits into ``num_dp = world_size // G``
serve-groups of G ranks, and a sequence binds to ALL G ranks of its group at
PREFILL and stays there for decode. The G ranks of a group hold the SAME
sequences (replicated attention, head-sharded KDA state) so both phases run in
TP-G lockstep and the o_proj all_reduce couples matching tokens — there is NO
prefill->decode reshard (superseding the earlier M2b "streamed pure-DP prefill
then reshard to the decode group" model). ``seq.decode_dp_group`` (0..num_dp-1)
names the group; every rank keys membership on

    seq.decode_dp_group == global_rank // G

which is exactly ``rank_in_decode_group`` below. G==1 collapses num_dp to
world_size and the group id to the rank, so nothing here changes the validated
pure-DP path (the worker never calls the assignment for G==1).

This module is deliberately pure (no torch, no worker) so the assignment /
membership / MoE-padding arithmetic is unit-testable off the GPU.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence


def num_decode_dp_groups(world_size: int, group_size: int) -> int:
    """Number of decode DP groups = world_size // G (G == attn_tp_size)."""
    if group_size <= 0:
        raise ValueError(f"group_size must be > 0, got {group_size}")
    if world_size % group_size != 0:
        raise ValueError(
            f"world_size {world_size} not divisible by group_size {group_size}"
        )
    return world_size // group_size


def rank_in_decode_group(
    decode_dp_group: Optional[int], global_rank: int, group_size: int
) -> bool:
    """True iff ``global_rank`` belongs to the group that owns the sequence.

    The G ranks [g*G, (g+1)*G) all map to group id g via ``global_rank // G``,
    so this predicate resolves the SAME sequence set on every rank of a group —
    the replicated-attention invariant TP-KDA decode needs.
    """
    if decode_dp_group is None:
        return False
    if group_size <= 0:
        raise ValueError(f"group_size must be > 0, got {group_size}")
    return int(decode_dp_group) == global_rank // group_size


def assign_decode_dp_groups(
    lengths: Sequence[int],
    num_dp: int,
    prior_load: Optional[Sequence[float]] = None,
    use_l2: bool = True,
) -> List[int]:
    """Balance sequences across ``num_dp`` decode groups (FFD + least-load argmin).

    Mirrors the prefill rank balancer (``_assign_admitted_sequences_to_ranks``):
    longest-first (First-Fit-Decreasing) placement onto the currently
    least-loaded group, where load is ``sum(L**2)`` (attention is O(L^2)) by
    default or ``count`` when ``use_l2=False``.

    Args:
        lengths: prompt length per sequence to assign (input order preserved).
        num_dp: number of decode groups.
        prior_load: optional length-``num_dp`` pre-existing load per group
            (sequences already in decode), so incremental admissions stay
            balanced against the standing decode batch.
        use_l2: L^2 load (default) vs. least-count.

    Returns:
        list of group ids (0..num_dp-1), one per input sequence, in input order.
    """
    if num_dp <= 0:
        raise ValueError(f"num_dp must be > 0, got {num_dp}")
    if prior_load is not None and len(prior_load) != num_dp:
        raise ValueError(
            f"prior_load length {len(prior_load)} != num_dp {num_dp}"
        )
    load = [float(x) for x in prior_load] if prior_load is not None \
        else [0.0] * num_dp

    # FFD: place the longest sequences first (input order is the tiebreak, so
    # the assignment is deterministic across ranks that see the same input).
    order = sorted(range(len(lengths)), key=lambda i: (-(lengths[i] or 0), i))
    groups = [0] * len(lengths)
    for i in order:
        L = lengths[i] or 0
        g = min(range(num_dp), key=lambda r: (load[r], r))
        groups[i] = g
        load[g] += float(L) * float(L) if use_l2 else 1.0
    return groups


def moe_ntp_from_group_max(max_group_batch: int, group_size: int) -> int:
    """Per-rank resident-EP MoE row count after the intra-group row scatter.

    The worker's 32-way count all_gather sees each group's batch B_grp on all G
    of its ranks (they hold identical sequences), so the global max is
    ``max_B_grp``. Decode scatters those B_grp rows evenly across the group's G
    ranks before the DP-32 resident MoE, so each rank owns at most
    ``ceil(max_B_grp / G)`` distinct rows — the padded all_gather / all_reduce
    layout size ``ResidentEPMoELayer.num_tokens_per_rank``. G==1 returns
    ``max_group_batch`` unchanged (validated pure-DP path).
    """
    if group_size <= 0:
        raise ValueError(f"group_size must be > 0, got {group_size}")
    return int(math.ceil(int(max_group_batch) / group_size))
