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
    n = routed_local.shape[0]
    if n == ntp:
        send = routed_local
    else:
        send = routed_local.new_zeros((ntp, H))
        if n > 0:
            send[:n].copy_(routed_local)
    gathered = routed_local.new_empty((group_size * ntp, H))
    dist.all_gather_into_tensor(gathered, send, group=group)
    if num_rows == group_size * ntp:
        # Rank-major concatenation is already the original contiguous row
        # order for an even split. Returning it directly avoids allocating and
        # copying a second full hidden tensor on exact64 K3 prefill.
        return gathered
    return reassemble_rows(gathered.view(group_size, ntp, H), num_rows, group_size)


def all_gather_rows_into(
    output: torch.Tensor,
    local_rows: torch.Tensor,
    num_rows: int,
    group_size: int,
    group_rank: int,
    group,
    chunk_rows: int = 2048,
) -> torch.Tensor:
    """Reassemble row-sharded output into caller-owned full-row storage.

    Unlike :func:`all_gather_rows`, this never holds a second full output.
    The local-row axis is gathered in bounded chunks and copied directly into
    ``output``.  This is used by K3's one dense prefill FFN: every TP rank runs
    the unchanged FFN on a disjoint row slice, then restores the replicated
    hidden state required by the following attention layer.
    """
    import torch.distributed as dist

    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    if output.shape[0] != num_rows:
        raise ValueError("output row count does not match num_rows")

    splits = balanced_row_split(num_rows, group_size)
    ntp = max((end - start) for start, end in splits)
    hidden = local_rows.shape[-1]
    if output.shape[-1] != hidden:
        raise ValueError("output hidden size does not match local rows")
    expected_local = splits[group_rank][1] - splits[group_rank][0]
    if local_rows.shape[0] != expected_local:
        raise ValueError("local row count does not match this TP rank")

    even = num_rows == group_size * ntp
    even_output = output.view(group_size, ntp, hidden) if even else None
    for local_start in range(0, ntp, chunk_rows):
        local_end = min(local_start + chunk_rows, ntp)
        count = local_end - local_start
        valid_end = min(local_end, expected_local)
        valid_count = max(0, valid_end - local_start)
        if valid_count == count:
            send = local_rows[local_start:local_end]
        else:
            send = local_rows.new_zeros((count, hidden))
            if valid_count:
                send[:valid_count].copy_(
                    local_rows[local_start:valid_end]
                )
        gathered = local_rows.new_empty((group_size * count, hidden))
        dist.all_gather_into_tensor(gathered, send, group=group)
        gathered = gathered.view(group_size, count, hidden)
        if even:
            even_output[:, local_start:local_end].copy_(gathered)
        else:
            for rank, (start, end) in enumerate(splits):
                rank_count = max(
                    0, min(local_end, end - start) - local_start
                )
                if rank_count:
                    output[
                        start + local_start : start + local_start + rank_count
                    ].copy_(gathered[rank, :rank_count])
        del gathered
        if valid_count != count:
            del send
    return output


def all_gather_rows_add_(
    output: torch.Tensor,
    routed_local: torch.Tensor,
    num_rows: int,
    group_size: int,
    group_rank: int,
    group,
    chunk_rows: int = 256,
) -> torch.Tensor:
    """Add TP-row-sharded results into an existing full-row output.

    This is the bounded-scratch counterpart of :func:`all_gather_rows`. The
    full destination already contains the shared-expert result; each
    rank-local row chunk is gathered and added in place, so no second
    ``(num_rows, H)`` tensor is allocated.
    """
    import torch.distributed as dist

    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    if output.shape[0] != num_rows:
        raise ValueError("output row count does not match num_rows")

    splits = balanced_row_split(num_rows, group_size)
    ntp = max((e - s) for s, e in splits)
    H = routed_local.shape[-1]
    if output.shape[-1] != H:
        raise ValueError("output hidden size does not match routed rows")
    local_rows = splits[group_rank][1] - splits[group_rank][0]
    if routed_local.shape[0] != local_rows:
        raise ValueError("routed_local row count does not match this TP rank")

    even = num_rows == group_size * ntp
    even_output = output.view(group_size, ntp, H) if even else None
    for local_start in range(0, ntp, chunk_rows):
        local_end = min(local_start + chunk_rows, ntp)
        count = local_end - local_start
        valid_end = min(local_end, local_rows)
        valid_count = max(0, valid_end - local_start)
        if valid_count == count:
            send = routed_local[local_start:local_end]
        else:
            send = routed_local.new_zeros((count, H))
            if valid_count:
                send[:valid_count].copy_(
                    routed_local[local_start:valid_end]
                )
        gathered = routed_local.new_empty((group_size * count, H))
        dist.all_gather_into_tensor(gathered, send, group=group)
        gathered = gathered.view(group_size, count, H)
        if even:
            even_output[:, local_start:local_end].add_(gathered)
        else:
            for rank, (start, end) in enumerate(splits):
                rank_count = max(
                    0, min(local_end, end - start) - local_start
                )
                if rank_count:
                    output[
                        start + local_start : start + local_start + rank_count
                    ].add_(gathered[rank, :rank_count])
        del gathered
        if valid_count != count:
            del send
    return output
