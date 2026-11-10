"""Pure metadata logic for the intra-node weight replica group.

The transport imports this module without torch.  A rank is grouped only with
same-node, same-fingerprint ranks when peer access is mutual in both
directions.  Daemon-backed and ``hierarchical_gdr`` sources deliberately stay
as groups of one until their source partition is unified with this seam.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from typing import Any


PeerAccessRow = Sequence[bool] | Mapping[int, bool]


@dataclass(frozen=True)
class RankRecord:
    """Published rank metadata used to form a replica group."""

    node_identity: str
    device_index: int
    peer_access_row: PeerAccessRow
    list_fingerprint: str
    weight_source_kind: str
    rank: int | None = None


_OWNER_EVERYWHERE = {
    "daemon",
    "daemon_backed",
    "distributed",
    "distributed_daemon",
    "hierarchical_gdr",
}


def _field(record: Any, name: str, *, aliases: tuple[str, ...] = ()) -> Any:
    if isinstance(record, Mapping):
        for key in (name, *aliases):
            if key in record:
                return record[key]
        raise ValueError(f"rank record lacks {name}")
    for key in (name, *aliases):
        if hasattr(record, key):
            return getattr(record, key)
    raise ValueError(f"rank record lacks {name}")


def _normalise_records(records: Sequence[RankRecord] | Mapping[int, Any]) -> list[RankRecord]:
    items = list(records.items()) if isinstance(records, Mapping) else list(enumerate(records))
    normalised: list[RankRecord] = []
    seen: set[int] = set()
    for position, raw in items:
        if isinstance(raw, RankRecord):
            rank = position if raw.rank is None else int(raw.rank)
            record = RankRecord(
                node_identity=raw.node_identity,
                device_index=int(raw.device_index),
                peer_access_row=raw.peer_access_row,
                list_fingerprint=raw.list_fingerprint,
                weight_source_kind=str(raw.weight_source_kind),
                rank=rank,
            )
        else:
            try:
                rank_value = _field(raw, "rank", aliases=("global_rank",))
            except ValueError:
                rank_value = None
            rank = position if rank_value is None else int(rank_value)
            record = RankRecord(
                node_identity=str(_field(raw, "node_identity", aliases=("node",))),
                device_index=int(_field(raw, "device_index", aliases=("device",))),
                peer_access_row=_field(
                    raw, "peer_access_row", aliases=("peer_access",)
                ),
                list_fingerprint=str(
                    _field(raw, "list_fingerprint", aliases=("fingerprint",))
                ),
                weight_source_kind=str(
                    _field(raw, "weight_source_kind", aliases=("source_kind",))
                ),
                rank=rank,
            )
        if rank in seen:
            raise ValueError(f"duplicate rank {rank} in replica metadata")
        seen.add(rank)
        normalised.append(record)
    if not normalised:
        raise ValueError("replica metadata must contain at least one rank")
    return normalised


def _source_owns_every_reader(source_kind: str) -> bool:
    normalised = source_kind.strip().lower().replace("-", "_")
    return normalised in _OWNER_EVERYWHERE


def _row_value(row: PeerAccessRow, target_rank: int, target_position: int) -> bool:
    if isinstance(row, Mapping):
        if target_rank in row:
            return bool(row[target_rank])
        if str(target_rank) in row:
            return bool(row[str(target_rank)])
        return False
    if target_position >= len(row):
        return False
    return bool(row[target_position])


def _has_peer_access(
    source: RankRecord, target: RankRecord, target_position: int
) -> bool:
    if source.rank == target.rank:
        return True
    row = source.peer_access_row
    if not isinstance(row, Mapping) and int(target.rank) < len(row):
        target_position = int(target.rank)
    return _row_value(row, int(target.rank), target_position)


def _peer_matrix(records: Sequence[RankRecord]) -> list[list[int]]:
    return [
        [
            int(_has_peer_access(source, target, target_position))
            for target_position, target in enumerate(records)
        ]
        for source in records
    ]


def _raise_partial_access(records: Sequence[RankRecord]) -> None:
    matrix = _peer_matrix(records)
    if all(matrix[row][column] and matrix[column][row]
           for row in range(len(records))
           for column in range(row + 1, len(records))):
        return
    ranks = [int(record.rank) for record in records]
    header = "      " + " ".join(f"{rank:>3}" for rank in ranks)
    rows = [header]
    rows.extend(
        f"{rank:>3}: " + " ".join(str(value) for value in row)
        for rank, row in zip(ranks, matrix)
    )
    raise ValueError(
        "partial peer access among same-fingerprint same-node ranks "
        f"{ranks}; matrix:\n" + "\n".join(rows)
    )


def compute_replica_groups(
    records: Sequence[RankRecord] | Mapping[int, Any],
) -> dict[int, tuple[int, ...]]:
    """Return each rank's ordered replica group.

    The returned tuples are ordered by global rank.  A partial peer-access
    matrix is an install error, even if a smaller clique could be formed.
    """

    normalised = _normalise_records(records)
    buckets: dict[tuple[str, str], list[RankRecord]] = {}
    groups: dict[int, tuple[int, ...]] = {}
    for record in normalised:
        rank = int(record.rank)
        if _source_owns_every_reader(record.weight_source_kind):
            groups[rank] = (rank,)
            continue
        buckets.setdefault(
            (record.node_identity, record.list_fingerprint), []
        ).append(record)

    for bucket_records in buckets.values():
        ordered = sorted(bucket_records, key=lambda record: int(record.rank))
        if len(ordered) > 1:
            _raise_partial_access(ordered)
        group = tuple(int(record.rank) for record in ordered)
        for rank in group:
            groups[rank] = group
    return groups


def ordered_replica_group(
    records: Sequence[RankRecord] | Mapping[int, Any], rank: int
) -> tuple[int, ...]:
    """Return the ordered group containing ``rank``."""

    groups = compute_replica_groups(records)
    try:
        return groups[int(rank)]
    except KeyError as error:
        raise ValueError(f"rank {rank} is absent from replica metadata") from error


replica_groups = compute_replica_groups

