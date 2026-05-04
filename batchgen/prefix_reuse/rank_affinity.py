"""Prefix-aware rank assignment helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Set, Tuple

from batchgen.sequence import SequenceEntry


@dataclass(frozen=True)
class RankAssignmentResult:
    assignments: List[Tuple[str, int]]
    prefix_assigned_count: int
    prefix_assigned_by_rank: List[int]
    rank_load: Optional[List[float]] = None


def assign_admitted_ranks(
    *,
    uuids: Sequence[str],
    existing_sequences: Iterable[SequenceEntry],
    get_sequence: Callable[[str], Optional[SequenceEntry]],
    world_size: int,
    use_l2_balance: bool,
    prefix_rank_lookup: Optional[Callable[[SequenceEntry], Optional[int]]] = None,
) -> RankAssignmentResult:
    """Plan rank assignments without mutating SequenceBatch."""
    if use_l2_balance:
        return _assign_by_l2_load(
            uuids=uuids,
            existing_sequences=existing_sequences,
            get_sequence=get_sequence,
            world_size=world_size,
            prefix_rank_lookup=prefix_rank_lookup,
        )
    return _assign_by_count(
        uuids=uuids,
        existing_sequences=existing_sequences,
        get_sequence=get_sequence,
        world_size=world_size,
        prefix_rank_lookup=prefix_rank_lookup,
    )


def _assign_by_l2_load(
    *,
    uuids: Sequence[str],
    existing_sequences: Iterable[SequenceEntry],
    get_sequence: Callable[[str], Optional[SequenceEntry]],
    world_size: int,
    prefix_rank_lookup: Optional[Callable[[SequenceEntry], Optional[int]]],
) -> RankAssignmentResult:
    pending_uuids = set(uuids)
    rank_load = [0.0] * world_size
    for seq in existing_sequences:
        if seq.uuid in pending_uuids or seq.assigned_rank is None:
            continue
        prompt_len = getattr(seq, "prompt_length", 0) or 0
        rank_load[seq.assigned_rank] += float(prompt_len) * float(prompt_len)

    assignments: List[Tuple[str, int]] = []
    prefix_assigned, prefix_assigned_by_rank = _assign_prefix_hint_ranks(
        uuids=uuids,
        get_sequence=get_sequence,
        world_size=world_size,
        prefix_rank_lookup=prefix_rank_lookup,
        assignments=assignments,
        on_assigned=lambda seq, rank: _add_l2_load(rank_load, seq, rank),
    )

    remaining: List[Tuple[int, str]] = []
    for uuid in uuids:
        if uuid in prefix_assigned:
            continue
        seq = get_sequence(uuid)
        if seq is None:
            continue
        remaining.append((getattr(seq, "prompt_length", 0) or 0, uuid))
    remaining.sort(key=lambda item: -item[0])

    for prompt_len, uuid in remaining:
        min_rank = min(range(world_size), key=lambda rank: rank_load[rank])
        assignments.append((uuid, min_rank))
        rank_load[min_rank] += float(prompt_len) * float(prompt_len)

    return RankAssignmentResult(
        assignments=assignments,
        prefix_assigned_count=len(prefix_assigned),
        prefix_assigned_by_rank=prefix_assigned_by_rank,
        rank_load=rank_load,
    )


def _assign_by_count(
    *,
    uuids: Sequence[str],
    existing_sequences: Iterable[SequenceEntry],
    get_sequence: Callable[[str], Optional[SequenceEntry]],
    world_size: int,
    prefix_rank_lookup: Optional[Callable[[SequenceEntry], Optional[int]]],
) -> RankAssignmentResult:
    pending_uuids = set(uuids)
    rank_counts = [0] * world_size
    for seq in existing_sequences:
        if seq.uuid not in pending_uuids and seq.assigned_rank is not None:
            rank_counts[seq.assigned_rank] += 1

    assignments: List[Tuple[str, int]] = []
    prefix_assigned, prefix_assigned_by_rank = _assign_prefix_hint_ranks(
        uuids=uuids,
        get_sequence=get_sequence,
        world_size=world_size,
        prefix_rank_lookup=prefix_rank_lookup,
        assignments=assignments,
        on_assigned=lambda _seq, rank: _add_count(rank_counts, rank),
    )

    for uuid in uuids:
        if uuid in prefix_assigned:
            continue
        seq = get_sequence(uuid)
        if seq is None:
            continue
        min_rank = rank_counts.index(min(rank_counts))
        assignments.append((uuid, min_rank))
        rank_counts[min_rank] += 1

    return RankAssignmentResult(
        assignments=assignments,
        prefix_assigned_count=len(prefix_assigned),
        prefix_assigned_by_rank=prefix_assigned_by_rank,
    )


def _assign_prefix_hint_ranks(
    *,
    uuids: Sequence[str],
    get_sequence: Callable[[str], Optional[SequenceEntry]],
    world_size: int,
    prefix_rank_lookup: Optional[Callable[[SequenceEntry], Optional[int]]],
    assignments: List[Tuple[str, int]],
    on_assigned: Callable[[SequenceEntry, int], None],
) -> Tuple[Set[str], List[int]]:
    prefix_assigned: Set[str] = set()
    prefix_assigned_by_rank = [0] * world_size
    if prefix_rank_lookup is None:
        return prefix_assigned, prefix_assigned_by_rank

    for uuid in uuids:
        seq = get_sequence(uuid)
        if seq is None:
            continue
        cached_rank = prefix_rank_lookup(seq)
        if cached_rank is None or not (0 <= int(cached_rank) < world_size):
            continue
        rank = int(cached_rank)
        assignments.append((uuid, rank))
        on_assigned(seq, rank)
        prefix_assigned.add(uuid)
        prefix_assigned_by_rank[rank] += 1
    return prefix_assigned, prefix_assigned_by_rank


def _add_l2_load(rank_load: List[float], seq: SequenceEntry, rank: int) -> None:
    prompt_len = getattr(seq, "prompt_length", 0) or 0
    rank_load[rank] += float(prompt_len) * float(prompt_len)


def _add_count(rank_counts: List[int], rank: int) -> None:
    rank_counts[rank] += 1
