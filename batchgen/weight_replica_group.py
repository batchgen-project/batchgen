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



# ---------------------------------------------------------------------------
# Frame derivation from the installed task's module names (spec v4, 4.3/4.7).
#
# The names follow the grammar the daemon and Weights_Storage already parse:
#   routed_expert_<layer>_<expert>, attn_<layer>, shared_expert_<layer>.
# Per type, consecutive entries group by layer; every type must yield the
# SAME ordered layer sequence with a constant per-type frame size. A task
# that does not fit forms single-member groups (every record HostPrivate):
# the same record state machine with trivial barriers, never a second path.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FramePlan:
    """Frame layout derived from one installed weight-copy task."""

    layer_sequence: tuple[int, ...]
    frame_size: dict[str, int]
    # module name -> (layer_position, expert index or None)
    position: dict[str, tuple[int, int | None]]


def _parse_module_name(module_type: str, name: str) -> tuple[int, int | None] | None:
    prefix = f"{module_type}_"
    if not name.startswith(prefix):
        return None
    tail = name[len(prefix):].split("_")
    try:
        numbers = [int(part) for part in tail]
    except ValueError:
        return None
    if len(numbers) == 1:
        return numbers[0], None
    if len(numbers) == 2:
        return numbers[0], numbers[1]
    return None


def derive_frame_plan(task: Mapping[str, Sequence[str]]) -> FramePlan | None:
    """Derive the frame layout, or None when the task does not fit the grammar."""
    layer_sequences: dict[str, list[int]] = {}
    frame_size: dict[str, int] = {}
    position: dict[str, tuple[int, int | None]] = {}
    for module_type, names in task.items():
        if not names:
            continue
        layers: list[int] = []
        counts: list[int] = []
        for name in names:
            parsed = _parse_module_name(module_type, name)
            if parsed is None:
                return None
            layer, expert = parsed
            if not layers or layer != layers[-1]:
                if layers and layer < layers[-1]:
                    return None  # layers out of order: not a frame schedule
                layers.append(layer)
                counts.append(0)
            counts[-1] += 1
        if len(set(counts)) != 1:
            return None  # ragged frames
        frame_size[module_type] = counts[0]
        layer_sequences[module_type] = layers
    if not frame_size:
        return None
    # A type may cover only part of the pass (GLM's first dense layers have
    # attention but no experts): the frame sequence is the ordered union, and
    # a frame simply has no record of an absent type.
    union = sorted({layer for seq in layer_sequences.values() for layer in seq})
    union_index = {layer: pos for pos, layer in enumerate(union)}
    for module_type, names in task.items():
        if not names:
            continue
        for name in names:
            layer, expert = _parse_module_name(module_type, name)  # type: ignore[misc]
            position[name] = (union_index[layer], expert)
    return FramePlan(
        layer_sequence=tuple(union),
        frame_size=dict(frame_size),
        position=position,
    )


def derive_owner_map(
    ordered_group: Sequence[int],
    task: Mapping[str, Sequence[str]],
    plan: FramePlan,
) -> dict[str, int]:
    """Owner rank per module: contiguous expert blocks for routed experts,
    layer round-robin for single-module types (spec v4, 4.1)."""
    group = list(ordered_group)
    owners: dict[str, int] = {}
    for module_type, names in task.items():
        if not names:
            continue
        experts = plan.frame_size[module_type]
        for name in names:
            layer_position, expert = plan.position[name]
            if expert is None:
                owners[name] = group[layer_position % len(group)]
            else:
                if experts % len(group) != 0:
                    raise ValueError(
                        f"{module_type} frame size {experts} is not divisible "
                        f"by the replica group size {len(group)}"
                    )
                owners[name] = group[expert // (experts // len(group))]
    return owners


def install_record_kinds(
    rank: int,
    ordered_group: Sequence[int],
    owners: Mapping[str, int],
) -> dict[str, str]:
    """Record kind per module for `rank`. Within a replica group every
    member's list is identical, so the readers are the whole group."""
    group = list(ordered_group)
    if rank not in group:
        raise ValueError(f"rank {rank} is not in the replica group {group}")
    kinds: dict[str, str] = {}
    for module, owner in owners.items():
        if owner == rank:
            kinds[module] = (
                "HostShared" if len(group) > 1 else "HostPrivate"
            )
        else:
            kinds[module] = "Replica"
    return kinds


def stagger_task_for_member(
    task: Mapping[str, Sequence[str]],
    plan: FramePlan,
    owners: Mapping[str, int],
    ordered_group: Sequence[int],
    rank: int,
) -> dict[str, list[str]]:
    """Reorder each layer's records for `rank`: its own block first, then
    the other owners' blocks starting at the next member round-robin.

    Record identity is canonical by name (``plan.position``), so the walk
    order is free within a layer. The canonical walk made every member
    traverse owners in the same order, so the group's seven readers
    converged on ONE owner at a time and shared that owner's egress
    (~1/7th each) while the other owners' links sat idle, and a member
    only published its own block after pulling every earlier owner's.
    Own-first publishes immediately; the stagger spreads the readers
    across all owners' egress at once.
    """
    group = list(ordered_group)
    me = group.index(rank)
    size = len(group)
    out: dict[str, list[str]] = {}
    for module_type, names in task.items():
        if not names:
            out[module_type] = list(names)
            continue
        layer_order: list[int] = []
        by_layer: dict[int, list[str]] = {}
        for name in names:
            layer_position, _ = plan.position[name]
            if layer_position not in by_layer:
                by_layer[layer_position] = []
                layer_order.append(layer_position)
            by_layer[layer_position].append(name)

        def walk_key(name: str) -> tuple[int, int]:
            _, expert = plan.position[name]
            owner_member = group.index(owners[name])
            return (owner_member - me) % size, expert if expert is not None else 0

        reordered: list[str] = []
        for layer_position in layer_order:
            bucket = by_layer[layer_position]
            if len(bucket) > 1:
                bucket = sorted(bucket, key=walk_key)
            reordered.extend(bucket)
        out[module_type] = reordered
    return out
