# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                          #
#                                                                               #
#  licensed under the apache license, version 2.0 (the "license");              #
# ---------------------------------------------------------------------------- #

"""Intra-group MoE token scatter/gather for TP-G decode (Kimi-Linear M2b, model).

``ResidentEPMoELayer`` is a DP-32 contract: it all_gathers over ALL ``world_size``
ranks assuming every rank holds DISTINCT decode rows. But under DP-(world/G) x
TP-G the G ranks of a decode group hold the SAME B_grp rows (replicated
attention). To feed the resident MoE its DP contract WITHOUT touching it, decode

    scatter_rows(x, G, rank)  ->  resident.forward (UNCHANGED)  ->  all_gather_rows

so the group's B_grp rows are split into G distinct contiguous slices (one per
rank, DP-32 restored), routed, then reassembled to the full B_grp on every rank
(attention downstream needs all rows on all ranks).

``scatter_rows`` is a pure LOCAL slice — x is already replicated across the
group, so no collective is needed. Only ``all_gather_rows`` communicates. Both
key off ``balanced_row_split`` (deterministic, identical on every rank), so the
split a rank scatters and the split every rank re-gathers agree.

Empty-rank invariant: when B_grp < G the tail ranks get 0 rows; they still run
``resident.forward`` (its collectives run every step) and still contribute a
zero-padded slot to the intra-group gather — reassembly skips their empty span.
"""

from __future__ import annotations

from typing import List, Tuple

import torch


def balanced_row_split(num_rows: int, group_size: int) -> List[Tuple[int, int]]:
    """Contiguous balanced split of ``num_rows`` across G ranks.

    Rank g owns ``[start, end)``; the first ``num_rows % G`` ranks get one extra
    row so the max per-rank count is exactly ``ceil(num_rows / G)`` (== the MoE
    padding ``ntp``). Contiguity means concatenating the per-rank slices in rank
    order restores the original row order on gather.
    """
    if group_size <= 0:
        raise ValueError(f"group_size must be > 0, got {group_size}")
    base, rem = divmod(int(num_rows), group_size)
    offs = [0]
    for g in range(group_size):
        offs.append(offs[-1] + base + (1 if g < rem else 0))
    return [(offs[g], offs[g + 1]) for g in range(group_size)]


def scatter_rows(x: torch.Tensor, group_size: int, group_rank: int) -> torch.Tensor:
    """This rank's contiguous slice of the group-replicated rows ``x`` (n_g, H).

    Pure: no collective (x is identical on every rank of the group).
    """
    s, e = balanced_row_split(x.shape[0], group_size)[group_rank]
    return x[s:e]


def reassemble_rows(
    gathered: torch.Tensor, num_rows: int, group_size: int
) -> torch.Tensor:
    """Rebuild the (num_rows, H) group batch from a padded per-rank gather.

    ``gathered`` is (G, ntp, H) where ntp == ceil(num_rows / G); rank g's first
    ``end-start`` rows are its real output, the rest is zero pad. Pure.
    """
    splits = balanced_row_split(num_rows, group_size)
    H = gathered.shape[-1]
    out = gathered.new_empty((num_rows, H))
    for g, (s, e) in enumerate(splits):
        if e > s:
            out[s:e].copy_(gathered[g, : e - s])
    return out


def all_gather_rows(
    routed_local: torch.Tensor,
    num_rows: int,
    group_size: int,
    group_rank: int,
    group,
) -> torch.Tensor:
    """Reassemble the full (num_rows, H) group batch on every rank.

    Each rank pads its ``n_g`` routed rows to ntp == ceil(num_rows / G), all
    ranks all_gather, then ``reassemble_rows`` extracts + concatenates the real
    spans. ``group`` is the attn_tp torch process group.
    """
    import torch.distributed as dist

    splits = balanced_row_split(num_rows, group_size)
    ntp = max((e - s) for s, e in splits)  # == ceil(num_rows / G)
    H = routed_local.shape[-1]
    padded = routed_local.new_zeros((ntp, H))
    n = routed_local.shape[0]
    if n > 0:
        padded[:n].copy_(routed_local)
    gathered = routed_local.new_empty((group_size * ntp, H))
    dist.all_gather_into_tensor(gathered, padded, group=group)
    return reassemble_rows(gathered.view(group_size, ntp, H), num_rows, group_size)
