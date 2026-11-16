"""CPU-only planning for producer-driven cross-node weight broadcasts.

The copy producer owns the CUDA/NCCL boundary.  This module only decides the
canonical frame order and records the externally completed ``CrossReplica``
landings, so the same sequencing can be tested without importing torch.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .weight_replica_group import (
    CrossNodeGroup,
    CrossNodeTopology,
    FramePlan,
    compute_cross_node_groups,
    derive_frame_plan,
    derive_intra_node_owner_map,
    derive_owner_map,
    install_cross_record_kinds,
)


@dataclass(frozen=True)
class BroadcastRecord:
    """One module record in canonical frame order."""

    module_type: str
    module: str
    frame: int
    record_index: int


@dataclass(frozen=True)
class BroadcastFrame:
    """All records that one collective launch carries."""

    frame: int
    records: tuple[BroadcastRecord, ...]


@dataclass(frozen=True)
class BroadcastPlan:
    """The ordered cross-node launches for one producer pass."""

    source_root: int
    rank: int
    frames: tuple[BroadcastFrame, ...]

    @property
    def is_source(self) -> bool:
        return self.rank == self.source_root


def _canonical_names(
    task: Mapping[str, Sequence[str]], plan: FramePlan
) -> dict[str, tuple[str, ...]]:
    """Return each module type in a stable layer/expert order."""
    canonical: dict[str, tuple[str, ...]] = {}
    for module_type, names in task.items():
        ordered = sorted(
            names,
            key=lambda name: (
                plan.position[name][0],
                -1 if plan.position[name][1] is None else plan.position[name][1],
                name,
            ),
        )
        canonical[module_type] = tuple(ordered)
    return canonical


def in_frame_positions(
    task: Mapping[str, Sequence[str]], plan: FramePlan
) -> dict[str, int]:
    """Each record's 0-based position among its type's records in its frame.

    Canonical by (layer, expert, name), so it is identical on every member
    and independent of a member's walk order. A rank whose experts are the
    contiguous 0..E-1 gets position == expert id; a streamed-SP8 slot, whose
    task holds a block of GLOBAL expert ids, gets 0..block-1 - which is what
    the per-frame record table (sized by the frame's record count) needs.
    """
    positions: dict[str, int] = {}
    for module_type, names in _canonical_names(task, plan).items():
        counter: dict[int, int] = {}
        for name in names:
            layer_position = plan.position[name][0]
            positions[name] = counter.get(layer_position, 0)
            counter[layer_position] = positions[name] + 1
    return positions


def plan_broadcast_frames(
    task: Mapping[str, Sequence[str]],
    plan: FramePlan,
    *,
    rank: int,
    source_root: int,
    pass_id: int = 1,
) -> BroadcastPlan:
    """Build frame-ordered records for either side of one broadcast.

    ``task`` is the source member's complete schedule.  Every member uses the
    same result, including non-source members whose local host schedule was
    empty before the cross-node plan was installed.
    """
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
        raise ValueError("rank must be a non-negative integer")
    if (
        isinstance(source_root, bool)
        or not isinstance(source_root, int)
        or source_root < 0
    ):
        raise ValueError("source_root must be a non-negative integer")
    if isinstance(pass_id, bool) or not isinstance(pass_id, int) or pass_id < 1:
        raise ValueError("pass_id must be a positive integer")
    if not plan.layer_sequence:
        raise ValueError("frame plan must contain at least one layer")

    canonical = _canonical_names(task, plan)
    type_offsets: dict[str, int] = {}
    offset = 0
    for module_type in sorted(plan.frame_size):
        type_offsets[module_type] = offset
        offset += plan.frame_size[module_type]

    records_by_frame: dict[int, list[BroadcastRecord]] = {
        frame: [] for frame in range(len(plan.layer_sequence))
    }
    positions = in_frame_positions(task, plan)
    for module_type in sorted(canonical):
        names = canonical[module_type]
        for name in names:
            layer_position, _expert = plan.position[name]
            record_index = type_offsets[module_type] + positions[name]
            records_by_frame[layer_position].append(
                BroadcastRecord(
                    module_type=module_type,
                    module=name,
                    frame=(pass_id - 1) * len(plan.layer_sequence)
                    + layer_position,
                    record_index=record_index,
                )
            )

    frames = tuple(
        BroadcastFrame(
            frame=(pass_id - 1) * len(plan.layer_sequence) + layer_position,
            records=tuple(
                sorted(
                    records,
                    key=lambda record: (
                        record.module_type,
                        record.record_index,
                        record.module,
                    ),
                )
            ),
        )
        for layer_position, records in records_by_frame.items()
        if records
    )
    return BroadcastPlan(
        source_root=source_root,
        rank=rank,
        frames=frames,
    )


class CrossReplicaLandingBook:
    """Pure bookkeeping for externally completed cross-node records."""

    def __init__(self) -> None:
        self._registered: dict[int, frozenset[int]] = {}
        self._landed: dict[int, set[int]] = {}
        self._pending: dict[int, list[int]] = {}

    def register_frame(self, frame: int, record_indices: Sequence[int]) -> None:
        if frame in self._registered:
            raise ValueError(f"frame {frame} is already registered")
        expected = frozenset(int(index) for index in record_indices)
        self._registered[frame] = expected
        landed = self._landed.setdefault(frame, set())
        for index in self._pending.pop(frame, []):
            self._mark(frame, index, landed)

    def record_landing(self, frame: int, record_index: int) -> None:
        frame = int(frame)
        record_index = int(record_index)
        if frame not in self._registered:
            self._pending.setdefault(frame, []).append(record_index)
            return
        self._mark(frame, record_index, self._landed.setdefault(frame, set()))

    def frame_landed(self, frame: int) -> bool:
        frame = int(frame)
        expected = self._registered.get(frame)
        if expected is None:
            return False
        return self._landed.get(frame, set()) == set(expected)

    def landed_records(self, frame: int) -> frozenset[int]:
        return frozenset(self._landed.get(int(frame), set()))

    def _mark(self, frame: int, record_index: int, landed: set[int]) -> None:
        expected = self._registered[frame]
        if record_index not in expected:
            raise ValueError(
                f"record {record_index} is not a CrossReplica record of frame "
                f"{frame}"
            )
        if record_index in landed:
            raise ValueError(f"record {record_index} of frame {frame} is terminal")
        landed.add(record_index)


@dataclass(frozen=True)
class CrossNodePlan:
    """One member's complete cross-node transport state for a pass."""

    group: CrossNodeGroup
    kinds: Mapping[str, str]
    broadcast: BroadcastPlan
    landing: "CrossReplicaLandingBook"


def build_cross_node_plan(
    task: Mapping[str, Sequence[str]],
    *,
    num_nodes: int,
    local_world: int,
    rank: int,
    num_experts: int | None = None,
    pass_id: int = 1,
) -> CrossNodePlan:
    """Derive one member's cross-node group, record kinds and frame order.

    ``task`` is the source member's complete schedule for this local slot.
    Intra-node owners are expressed as GLOBAL ranks of the source root's own
    node (the source root need not live on node 0: for local slot ``g`` it is
    ``((g*num_nodes)//local_world)*local_world + g``), which is what
    ``install_cross_record_kinds`` requires for the source-side kind rule.
    """
    topology = CrossNodeTopology(num_nodes=num_nodes, local_world=local_world)
    group = compute_cross_node_groups(topology)[rank]
    # Positions and record indices come from the FULL task (the engine
    # numbers frames and records from the same plan), but only routed
    # experts cross nodes: every other type is fetched by each node from its
    # own host source, so it gets no cross kind and no broadcast record.
    plan = derive_frame_plan(task)
    if plan is None:
        raise ValueError("task does not fit the layer grammar")
    routed = list(task.get("routed_expert", ()))
    if not routed:
        raise ValueError("task has no routed experts to replicate across nodes")
    if num_experts is None:
        # Under streamed SP8 a slot's task holds only its own expert
        # block (global indices g*E/G .. (g+1)*E/G - 1), so the per-layer
        # count of the task can never stand in for the model's expert
        # count: the owner rule divides by the WHOLE-model count.
        raise ValueError(
            "num_experts (the model's routed expert count) is required "
            "for a task with routed experts"
        )
    source_node = group.source_root // local_world
    intra_group = tuple(
        source_node * local_world + slot for slot in range(local_world)
    )
    owners_local = derive_intra_node_owner_map(routed, num_experts, local_world)
    intra_owners = {
        name: source_node * local_world + owner
        for name, owner in owners_local.items()
    }
    kinds = install_cross_record_kinds(rank, group, intra_group, intra_owners)
    broadcast = plan_broadcast_frames(
        {"routed_expert": routed},
        plan,
        rank=rank,
        source_root=group.source_root,
        pass_id=pass_id,
    )
    return CrossNodePlan(
        group=group,
        kinds=kinds,
        broadcast=broadcast,
        landing=CrossReplicaLandingBook(),
    )


class BroadcastDriver:
    """Sequence successive per-layer broadcasts onto the plan's frame order.

    The streamed layer path requests one broadcast per layer prefetch, in
    layer order, pass after pass.  The driver maps each request to the next
    canonical frame (extending frame ids past the first pass by whole-pass
    offsets) and applies the frame's landings on completion, so the engine's
    external-landing seam sees exactly the records the collective carried.
    """

    def __init__(
        self,
        plan: BroadcastPlan,
        landing: CrossReplicaLandingBook,
        *,
        layers_per_pass: int,
    ) -> None:
        if not plan.frames:
            raise ValueError("broadcast plan has no frames")
        if layers_per_pass < len(plan.frames):
            raise ValueError(
                "layers_per_pass cannot be smaller than the planned frames"
            )
        self._plan = plan
        self._landing = landing
        self._layers_per_pass = layers_per_pass
        self._cursor = 0

    @property
    def source_root(self) -> int:
        return self._plan.source_root

    def next_frame(self) -> BroadcastFrame:
        count = len(self._plan.frames)
        base = self._plan.frames[self._cursor % count]
        offset = (self._cursor // count) * self._layers_per_pass
        self._cursor += 1
        if not offset:
            return base
        return BroadcastFrame(
            frame=base.frame + offset,
            records=tuple(
                BroadcastRecord(
                    module_type=record.module_type,
                    module=record.module,
                    frame=record.frame + offset,
                    record_index=record.record_index,
                )
                for record in base.records
            ),
        )

    def complete(self, frame: BroadcastFrame) -> None:
        self._landing.register_frame(
            frame.frame, [record.record_index for record in frame.records]
        )
        for record in frame.records:
            self._landing.record_landing(frame.frame, record.record_index)


@dataclass(frozen=True)
class BroadcastSubmission:
    """A transport handle paired with the frame it covers."""

    frame: BroadcastFrame
    handle: Any


def submit_broadcast_frames(
    plan: BroadcastPlan,
    *,
    transport: Callable[[Sequence[BroadcastRecord], Any, Any], Any],
    group: Any,
    stream: Any,
) -> tuple[BroadcastSubmission, ...]:
    """Submit every frame in order through an injected transport callable."""
    submissions: list[BroadcastSubmission] = []
    for frame in plan.frames:
        handle = transport(frame.records, group, stream)
        submissions.append(BroadcastSubmission(frame=frame, handle=handle))
    return tuple(submissions)


def mark_completed_submission(
    submission: BroadcastSubmission, landing: CrossReplicaLandingBook
) -> None:
    """Apply one transport completion to the frame's CrossReplica records."""
    frame = submission.frame.frame
    landing.register_frame(frame, [record.record_index for record in submission.frame.records])
    for record in submission.frame.records:
        landing.record_landing(frame, record.record_index)

