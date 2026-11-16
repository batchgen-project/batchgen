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


def install_cross_record_kinds(
    rank: int,
    group: CrossNodeGroup,
    intra_group: Sequence[int],
    intra_owners: Mapping[str, int],
) -> dict[str, str]:
    """Record kind per member of one hierarchical cross-node group."""
    if not isinstance(group, CrossNodeGroup):
        raise ValueError("group must be a CrossNodeGroup")
    if rank not in group.ranks:
        raise ValueError(
            f"rank {rank} is not in the cross-node group {list(group.ranks)}"
        )
    if rank != group.source_root:
        return {module: "CrossReplica" for module in intra_owners}
    return install_record_kinds(rank, intra_group, intra_owners)


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


# K3/K3.1-Pro streamed-sp8 task names follow the builder grammar in
# batchgen/models/moonshotai/kimi_linear/Parallel_Strategy_Manager.py:1049-1053,
# :1066-1077, and :1088-1098.  Non-source ranks skip routed experts at
# :1078-1087; the cross-node group and source-root construction is at
# :2293-2303 in that file.
_CROSS_NODE_MODULE_TYPES = frozenset(
    {"attn", "kda_attn", "shared_expert", "routed_expert"}
)


@dataclass(frozen=True)
class CrossNodeTopology:
    """Topology inputs for cross-node groups of equal local TP slots."""

    num_nodes: int
    local_world: int

    def __post_init__(self) -> None:
        for field_name, value in (
            ("num_nodes", self.num_nodes),
            ("local_world", self.local_world),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field_name} must be a positive integer")


@dataclass(frozen=True)
class CrossNodeGroup:
    """One local-slot group spanning every node, ordered by global rank."""

    local_slot: int
    ranks: tuple[int, ...]
    source_root: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.local_slot, bool)
            or not isinstance(self.local_slot, int)
            or self.local_slot < 0
        ):
            raise ValueError("local_slot must be a non-negative integer")
        if not isinstance(self.ranks, tuple) or not self.ranks:
            raise ValueError("ranks must be a non-empty tuple")
        if any(
            isinstance(rank, bool) or not isinstance(rank, int) or rank < 0
            for rank in self.ranks
        ):
            raise ValueError("ranks must contain non-negative integers")
        if tuple(sorted(self.ranks)) != self.ranks:
            raise ValueError("ranks must be ordered by global rank")
        if len(set(self.ranks)) != len(self.ranks):
            raise ValueError("ranks must be unique")
        if isinstance(self.source_root, bool) or not isinstance(self.source_root, int):
            raise ValueError("source_root must be an integer")
        if self.source_root not in self.ranks:
            raise ValueError("source_root must be a member of ranks")


def compute_cross_node_groups(
    topology: CrossNodeTopology,
) -> dict[int, CrossNodeGroup]:
    """Return the cross-node group containing each global rank.

    For local slot ``g``, the group is ``g + node * G`` for each node, and its
    source root is ``((g * num_nodes) // G) * G + g``.  The returned mapping
    contains one entry for every rank in the topology world.
    """
    if not isinstance(topology, CrossNodeTopology):
        raise ValueError("topology must be a CrossNodeTopology")

    num_nodes = topology.num_nodes
    local_world = topology.local_world
    world = num_nodes * local_world
    groups: dict[int, CrossNodeGroup] = {}
    for local_slot in range(local_world):
        ranks = tuple(local_slot + node * local_world for node in range(num_nodes))
        source_root = (
            (local_slot * num_nodes) // local_world
        ) * local_world + local_slot
        group = CrossNodeGroup(
            local_slot=local_slot,
            ranks=ranks,
            source_root=source_root,
        )
        if not 0 <= source_root < world:
            raise ValueError("computed source_root is outside the topology world")
        for rank in ranks:
            groups[rank] = group
    return groups


def _validate_cross_node_task(
    group_task: Mapping[str, Sequence[str]],
    plan: FramePlan | None,
) -> FramePlan:
    """Validate the strict K3 task grammar and return its matching plan."""
    if not isinstance(group_task, Mapping):
        raise ValueError("group_task must be a mapping of module types to names")
    unknown_types = set(group_task) - _CROSS_NODE_MODULE_TYPES
    if unknown_types:
        names = ", ".join(sorted(str(name) for name in unknown_types))
        raise ValueError(f"unknown cross-node module type(s): {names}")

    all_names: list[str] = []
    for module_type, names in group_task.items():
        if not isinstance(names, Sequence) or isinstance(names, (str, bytes)):
            raise ValueError(f"{module_type} task must be a sequence of names")
        for name in names:
            if not isinstance(name, str):
                raise ValueError(f"{module_type} task contains a non-string name")
            parsed = _parse_module_name(module_type, name)
            expected_expert = module_type == "routed_expert"
            if parsed is None or (parsed[1] is not None) != expected_expert:
                raise ValueError(
                    f"module name {name!r} does not fit the cross-node grammar"
                )
            layer, expert = parsed
            if layer < 0 or (expert is not None and expert < 0):
                raise ValueError(
                    f"module name {name!r} has a negative layer or expert index"
                )
            all_names.append(name)

    if len(set(all_names)) != len(all_names):
        raise ValueError("group_task contains duplicate module names")

    derived_plan = derive_frame_plan(group_task)
    if derived_plan is None:
        raise ValueError("group_task does not fit the frame grammar")
    if plan is None or plan != derived_plan:
        raise ValueError("plan does not match the supplied group_task")
    return derived_plan


def derive_cross_node_owner_map(
    group_task: Mapping[str, Sequence[str]],
    plan: FramePlan,
    group: CrossNodeGroup,
) -> dict[str, int]:
    """Return the group-level owner for every record in the source task.

    ``hierarchical_gdr`` has one source rank that streams the task and
    broadcasts it to the other group members.  This matches
    ``_acquires_from_host`` and ``cross_source`` in
    ``batchgen/moe/streamed_sp8_mxfp4.py:121-126``.  The returned map therefore
    assigns every record to ``group.source_root``.
    """
    if not isinstance(group, CrossNodeGroup):
        raise ValueError("group must be a CrossNodeGroup")
    _validate_cross_node_task(group_task, plan)
    return {
        name: group.source_root
        for names in group_task.values()
        for name in names
    }


def derive_intra_node_owner_map(
    expert_task: Mapping[str, Sequence[str]] | Sequence[str],
    num_experts: int,
    local_world: int,
) -> dict[str, int]:
    """Return local-slot owners for routed-expert records.

    The source task may contain only one local slot's expert names, so
    ``num_experts`` is explicit.  ``expert_task`` may be the routed-expert
    name sequence itself or a task mapping containing ``routed_expert``.
    """
    if (
        isinstance(num_experts, bool)
        or not isinstance(num_experts, int)
        or num_experts < 1
    ):
        raise ValueError("num_experts must be a positive integer")
    if (
        isinstance(local_world, bool)
        or not isinstance(local_world, int)
        or local_world < 1
    ):
        raise ValueError("local_world must be a positive integer")
    if isinstance(expert_task, Mapping):
        names = expert_task.get("routed_expert", ())
    else:
        names = expert_task
    if not isinstance(names, Sequence) or isinstance(names, (str, bytes)):
        raise ValueError("expert_task must contain routed-expert names")

    # Match the contiguous block rule in batchgen/moe/streamed_sp8_mxfp4.py:125-126:
    # experts_per_rank = num_experts // tp_size; expert_start =
    # tp_rank * experts_per_rank.
    experts_per_rank = num_experts // local_world
    if experts_per_rank < 1:
        raise ValueError("local_world must not exceed num_experts")
    owners: dict[str, int] = {}
    for name in names:
        if not isinstance(name, str):
            raise ValueError("expert_task contains a non-string name")
        parsed = _parse_module_name("routed_expert", name)
        if parsed is None or parsed[1] is None:
            raise ValueError(f"invalid routed-expert name: {name!r}")
        _, expert = parsed
        if not 0 <= expert < experts_per_rank * local_world:
            raise ValueError(
                f"expert index is outside the contiguous local range: {name!r}"
            )
        if name in owners:
            raise ValueError(f"duplicate routed-expert name: {name!r}")
        owners[name] = expert // experts_per_rank
    return owners
