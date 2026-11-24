# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  Copyright (c) 2025-2026 BatchGen Team                                            #
# ---------------------------------------------------------------------------- #

"""Model-declared parallel layout, read from the parallel-strategy manager.

The attention and MoE parallelism a deployment runs is decided inside each
model's parallel-strategy manager, so only the manager can say what it is.
This module is the worker's typed contract for that declaration: a manager
that implements ``parallel_declaration(decode_graph_state=None)`` returns a
``ParallelDeclaration``
describing each (phase, module) slot, computed from the launch configuration
at start-up (world size, EP configuration, planner output) — never hard-coded
where the layout depends on launch arguments.

The contract is deliberately optional, like ``recurrent_state_owner``: the
worker reads it with ``read_parallel_declaration`` below, and a manager that
has not declared is reported as ``undeclared`` in the ``[SERVER_CONFIG]``
line rather than guessed at. There is no default declaration: a base-supplied
guess would silently report something untrue for the models it is wrong for.

Vocabulary (closed; every registered model fits it):

Schemes
    ``dp``
        Replicated: every rank runs the whole module on its own rows.
    ``tp``
        Sharded over a group of ``group_size`` ranks (head- or dim-parallel);
        groups are replicated data-parallel across the world.
    ``ep``
        Experts partitioned over ``group_size`` ranks.
    ``none``
        The module does not exist in this model (a dense model's MoE slot).

Residency
    ``resident``
        The phase's weights for this module stay on the device.
    ``streamed``
        The weights stream from host memory as the phase uses them.
    ``partial``
        A resident shard plus a streamed remainder (e.g. an EP rank keeping
        only part of its expert shard resident).

A slot renders as ``scheme:residency`` with the group width appended to a
sharded scheme — ``dp:streamed``, ``tp8:resident``, ``ep8:partial`` — or as
the bare ``none``. ``undeclared`` is reserved for a manager without the
contract and is never a value a declaration may carry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

SCHEME_DP = "dp"
SCHEME_TP = "tp"
SCHEME_EP = "ep"
SCHEME_NONE = "none"
SCHEMES = (SCHEME_DP, SCHEME_TP, SCHEME_EP, SCHEME_NONE)

# The schemes whose rendered token carries the shard-group width.
_GROUPED_SCHEMES = (SCHEME_TP, SCHEME_EP)

RESIDENCY_RESIDENT = "resident"
RESIDENCY_STREAMED = "streamed"
RESIDENCY_PARTIAL = "partial"
RESIDENCIES = (RESIDENCY_RESIDENT, RESIDENCY_STREAMED, RESIDENCY_PARTIAL)

GRAPH_MODE_OFF = "off"
GRAPH_MODE_GRAPH = "graph"
GRAPH_MODE_COMPARE = "compare"
GRAPH_MODES = (GRAPH_MODE_OFF, GRAPH_MODE_GRAPH, GRAPH_MODE_COMPARE)

# What the [SERVER_CONFIG] line reports for every slot of a manager that has
# not declared. Never a legal value inside a ParallelDeclaration.
UNDECLARED = "undeclared"

# The four declaration slots, in the order the [SERVER_CONFIG] line prints.
DECLARATION_SLOTS = (
    "prefill_attention",
    "prefill_moe",
    "decode_attention",
    "decode_moe",
)


class ParallelDeclarationError(ValueError):
    """A parallel declaration outside this module's closed vocabulary."""


@dataclass(frozen=True)
class DecodeGraphDeclaration:
    """The model-owned decode CUDA-graph mode and bucket ladder.

    ``buckets`` is the installed or captured ladder owned by the declaring
    model.  A worker-owned graph owner is converted to this same record by
    ``read_parallel_declaration`` before it reaches a strategy manager.
    """

    mode: str = GRAPH_MODE_OFF
    buckets: Optional[Tuple[int, ...]] = None

    def __post_init__(self) -> None:
        if self.mode not in GRAPH_MODES:
            raise ParallelDeclarationError(
                f"decode graph mode {self.mode!r} is not one of {GRAPH_MODES}"
            )
        if self.buckets is None:
            if self.mode != GRAPH_MODE_OFF:
                raise ParallelDeclarationError(
                    "an enabled decode graph declaration needs buckets"
                )
            return
        if (
            not isinstance(self.buckets, tuple)
            or not self.buckets
            or any(
                not isinstance(bucket, int)
                or isinstance(bucket, bool)
                or bucket <= 0
                for bucket in self.buckets
            )
            or list(self.buckets) != sorted(set(self.buckets))
        ):
            raise ParallelDeclarationError(
                "decode graph buckets must be a strictly increasing tuple of "
                f"positive ints, got {self.buckets!r}"
            )
        if self.mode == GRAPH_MODE_OFF:
            raise ParallelDeclarationError(
                "an off decode graph declaration cannot carry buckets"
            )

    @classmethod
    def off(cls) -> "DecodeGraphDeclaration":
        return cls()


def decode_graph_from_owner(owner) -> DecodeGraphDeclaration:
    """Convert the worker graph owner into the manager-facing graph record."""
    if not getattr(owner, "whole_model_graph", False):
        return DecodeGraphDeclaration.off()
    manager = getattr(owner, "cuda_graph_manager", None)
    bucketing = getattr(owner, "whole_model_bucketing", None)
    if manager is None or bucketing is None:
        return DecodeGraphDeclaration.off()
    return DecodeGraphDeclaration(
        mode=GRAPH_MODE_GRAPH,
        buckets=tuple(int(bucket) for bucket in bucketing.bucket_sizes),
    )


@dataclass(frozen=True)
class ModuleParallelism:
    """One (phase, module) slot of a model's declared parallel layout."""

    scheme: str
    # None exactly when the module does not exist (scheme == "none").
    residency: Optional[str]
    # Width of one shard group; 1 for a replicated (dp) module. An ep module
    # declares the rank count its experts are partitioned over, a tp module
    # the ranks one head/dim shard group spans.
    group_size: int = 1

    def __post_init__(self) -> None:
        if self.scheme not in SCHEMES:
            raise ParallelDeclarationError(
                f"parallelism scheme {self.scheme!r} is not one of {SCHEMES}"
            )
        if not isinstance(self.group_size, int) or isinstance(self.group_size, bool):
            raise ParallelDeclarationError(
                f"group_size must be an int, got {self.group_size!r}"
            )
        if self.scheme == SCHEME_NONE:
            if self.residency is not None or self.group_size != 1:
                raise ParallelDeclarationError(
                    "a 'none' module carries no residency and group_size 1"
                )
            return
        if self.residency not in RESIDENCIES:
            raise ParallelDeclarationError(
                f"residency {self.residency!r} is not one of {RESIDENCIES}"
            )
        if self.scheme in _GROUPED_SCHEMES:
            if self.group_size < 1:
                raise ParallelDeclarationError(
                    f"{self.scheme} group_size must be >= 1, got {self.group_size}"
                )
        elif self.group_size != 1:
            raise ParallelDeclarationError(
                f"a {self.scheme} module has no shard group; group_size must be 1"
            )

    def render(self) -> str:
        """The slot's space-free ``[SERVER_CONFIG]`` token."""
        if self.scheme == SCHEME_NONE:
            return SCHEME_NONE
        scheme = self.scheme
        if scheme in _GROUPED_SCHEMES:
            scheme = f"{scheme}{self.group_size}"
        return f"{scheme}:{self.residency}"


@dataclass(frozen=True)
class ParallelDeclaration:
    """A model's declared layout and decode-graph state."""

    prefill_attention: ModuleParallelism
    prefill_moe: ModuleParallelism
    decode_attention: ModuleParallelism
    decode_moe: ModuleParallelism
    decode_graph: DecodeGraphDeclaration = DecodeGraphDeclaration()

    def __post_init__(self) -> None:
        for slot in DECLARATION_SLOTS:
            value = getattr(self, slot)
            if not isinstance(value, ModuleParallelism):
                raise ParallelDeclarationError(
                    f"declaration slot {slot} must be a ModuleParallelism, "
                    f"got {value!r}"
                )
        if (self.prefill_moe.scheme == SCHEME_NONE) != (
            self.decode_moe.scheme == SCHEME_NONE
        ):
            raise ParallelDeclarationError(
                "a model either has a MoE in both phases or in neither"
            )
        for slot in ("prefill_attention", "decode_attention"):
            if getattr(self, slot).scheme == SCHEME_NONE:
                raise ParallelDeclarationError(f"{slot} cannot be 'none'")
        if not isinstance(self.decode_graph, DecodeGraphDeclaration):
            raise ParallelDeclarationError(
                "declaration decode_graph must be a DecodeGraphDeclaration, "
                f"got {self.decode_graph!r}"
            )

    def rendered_slots(self) -> dict:
        """``{slot: token}`` in ``DECLARATION_SLOTS`` order."""
        return {slot: getattr(self, slot).render() for slot in DECLARATION_SLOTS}


def undeclared_slots() -> dict:
    """What the ``[SERVER_CONFIG]`` line reports without a declaration."""
    return {slot: UNDECLARED for slot in DECLARATION_SLOTS}


def read_parallel_declaration(
    parallel_manager, decode_graph_owner=None
) -> Optional[ParallelDeclaration]:
    """The manager's declaration, or None when it does not implement one.

    Read like the worker's other optional model declarations
    (``recurrent_state_owner``): absence is a legal, explicit state — the
    caller reports ``undeclared`` — but a present contract that returns the
    wrong type is a hard error, never coerced.  When supplied, the one graph
    owner argument is converted to ``DecodeGraphDeclaration`` and passed to
    every manager through the same method parameter.
    """
    declares = getattr(parallel_manager, "parallel_declaration", None)
    if declares is None:
        return None
    if decode_graph_owner is None:
        declaration = declares()
    else:
        declaration = declares(decode_graph_from_owner(decode_graph_owner))
    if not isinstance(declaration, ParallelDeclaration):
        raise ParallelDeclarationError(
            f"{type(parallel_manager).__name__}.parallel_declaration() must "
            f"return a ParallelDeclaration, got {type(declaration).__name__}"
        )
    return declaration


# ---------------------------------------------------------------------------
# Phase layout: the rank groups a phase's two modules form.
#
# A phase gives the attention and the MoE each their own scheme and group
# size. Sequences are owned by an attention group; no activation collective
# leaves a DP worker, whose size is the larger of the two group sizes. A
# group is a tuple of global ranks, never "the first G local ranks of a
# node", so that groups may later span machines.
# ---------------------------------------------------------------------------

PHASE_PREFILL = "prefill"
PHASE_DECODE = "decode"
LAYOUT_PHASES = (PHASE_PREFILL, PHASE_DECODE)


def _rank_groups(world_size: int, group_size: int) -> Tuple[Tuple[int, ...], ...]:
    return tuple(
        tuple(range(start, start + group_size))
        for start in range(0, world_size, group_size)
    )


@dataclass(frozen=True)
class PhaseLayout:
    """One phase's layout over a world of ranks."""

    phase: str
    attention: ModuleParallelism
    moe: ModuleParallelism
    world_size: int

    def __post_init__(self) -> None:
        if self.phase not in LAYOUT_PHASES:
            raise ParallelDeclarationError(
                f"phase {self.phase!r} is not one of {LAYOUT_PHASES}"
            )
        for name in ("attention", "moe"):
            if not isinstance(getattr(self, name), ModuleParallelism):
                raise ParallelDeclarationError(
                    f"{self.phase} {name} must be a ModuleParallelism"
                )
        if self.attention.scheme == SCHEME_NONE:
            raise ParallelDeclarationError(f"{self.phase} attention cannot be 'none'")
        if (
            not isinstance(self.world_size, int)
            or isinstance(self.world_size, bool)
            or self.world_size < 1
        ):
            raise ParallelDeclarationError(
                f"world_size must be a positive int, got {self.world_size!r}"
            )
        small, large = sorted((self.attention.group_size, self.moe.group_size))
        if large % small:
            raise ParallelDeclarationError(
                f"{self.phase} layout {self.render()}: the attention group size "
                f"{self.attention.group_size} and the MoE group size "
                f"{self.moe.group_size} must divide one another"
            )
        if self.world_size % large:
            raise ParallelDeclarationError(
                f"{self.phase} layout {self.render()}: the DP worker size {large} "
                f"does not divide the world size {self.world_size}"
            )

    @property
    def dp_worker_size(self) -> int:
        """Ranks one DP worker spans: the larger of the two group sizes."""
        return max(self.attention.group_size, self.moe.group_size)

    @property
    def num_dp_workers(self) -> int:
        return self.world_size // self.dp_worker_size

    @property
    def dp_workers(self) -> Tuple[Tuple[int, ...], ...]:
        return _rank_groups(self.world_size, self.dp_worker_size)

    @property
    def attention_groups(self) -> Tuple[Tuple[int, ...], ...]:
        return _rank_groups(self.world_size, self.attention.group_size)

    @property
    def moe_groups(self) -> Tuple[Tuple[int, ...], ...]:
        return _rank_groups(self.world_size, self.moe.group_size)

    def _check_rank(self, rank: int) -> None:
        if not isinstance(rank, int) or isinstance(rank, bool) or not (
            0 <= rank < self.world_size
        ):
            raise ParallelDeclarationError(
                f"rank {rank!r} is outside the world of {self.world_size} ranks"
            )

    def attention_group_of(self, rank: int) -> Tuple[int, ...]:
        """The attention group that owns the sequences served on ``rank``."""
        self._check_rank(rank)
        return self.attention_groups[rank // self.attention.group_size]

    def moe_group_of(self, rank: int) -> Tuple[int, ...]:
        self._check_rank(rank)
        return self.moe_groups[rank // self.moe.group_size]

    def dp_worker_of(self, rank: int) -> Tuple[int, ...]:
        """The ranks an activation collective issued on ``rank`` may reach."""
        self._check_rank(rank)
        return self.dp_workers[rank // self.dp_worker_size]

    def render(self) -> str:
        return f"attention={self.attention.render()} moe={self.moe.render()}"


@dataclass(frozen=True)
class ServerLayout:
    """The two phase layouts of a server; each phase owns its own (C1)."""

    prefill: PhaseLayout
    decode: PhaseLayout

    def __post_init__(self) -> None:
        for phase in LAYOUT_PHASES:
            layout = getattr(self, phase)
            if not isinstance(layout, PhaseLayout) or layout.phase != phase:
                raise ParallelDeclarationError(
                    f"server layout slot {phase} must be the {phase} PhaseLayout"
                )
        if self.prefill.world_size != self.decode.world_size:
            raise ParallelDeclarationError(
                "both phases run on the same world: "
                f"{self.prefill.world_size} != {self.decode.world_size}"
            )
        if (self.prefill.moe.scheme == SCHEME_NONE) != (
            self.decode.moe.scheme == SCHEME_NONE
        ):
            raise ParallelDeclarationError(
                "a model either has a MoE in both phases or in neither"
            )

    def phase(self, phase: str) -> PhaseLayout:
        if phase not in LAYOUT_PHASES:
            raise ParallelDeclarationError(
                f"phase {phase!r} is not one of {LAYOUT_PHASES}"
            )
        return getattr(self, phase)

    def declaration(
        self, decode_graph: DecodeGraphDeclaration = DecodeGraphDeclaration()
    ) -> ParallelDeclaration:
        """The four-slot report rendered from this layout, so that what is
        reported and what is executed come from one object."""
        return ParallelDeclaration(
            prefill_attention=self.prefill.attention,
            prefill_moe=self.prefill.moe,
            decode_attention=self.decode.attention,
            decode_moe=self.decode.moe,
            decode_graph=decode_graph,
        )


def server_layout_from_declaration(
    declaration: ParallelDeclaration, world_size: int
) -> ServerLayout:
    """The rank groups a manager's declared layout forms over ``world_size``."""
    if not isinstance(declaration, ParallelDeclaration):
        raise ParallelDeclarationError(
            f"expected a ParallelDeclaration, got {type(declaration).__name__}"
        )
    return ServerLayout(
        prefill=PhaseLayout(
            PHASE_PREFILL,
            declaration.prefill_attention,
            declaration.prefill_moe,
            world_size,
        ),
        decode=PhaseLayout(
            PHASE_DECODE,
            declaration.decode_attention,
            declaration.decode_moe,
            world_size,
        ),
    )


# ---------------------------------------------------------------------------
# Implemented layouts: what a model family has built, declared as data.
#
# A manager lists, per phase, the (attention, MoE) combinations it implements.
# The layout a server resolved is checked against that list at start-up; one
# outside it is refused by name and the message lists the implemented ones.
# The list is the model family's own statement, so "what is implemented" can
# be read (and rendered into one table) without reading the model code.
# ---------------------------------------------------------------------------

GROUP_ONE = "1"            # a replicated module: no shard group
GROUP_WORLD = "world"      # one group spanning every rank of the server
GROUP_DIVISOR = "g"        # any group size that divides the world size
GROUP_ATTENTION = "attn"   # the same group as the phase's attention
_GROUP_RULES = (GROUP_ONE, GROUP_WORLD, GROUP_DIVISOR, GROUP_ATTENTION)
# A group rule may also be a fixed size, written as an int greater than 1.


def _is_fixed_group(group) -> bool:
    return isinstance(group, int) and not isinstance(group, bool) and group > 1


class UnsupportedLayoutError(ParallelDeclarationError):
    """The resolved layout is not one the model family implements."""


@dataclass(frozen=True)
class ModuleSupport:
    """The shapes of one module that a layout entry admits."""

    scheme: str
    residencies: Tuple[str, ...] = ()
    group: object = GROUP_ONE

    def __post_init__(self) -> None:
        if self.scheme not in SCHEMES:
            raise ParallelDeclarationError(
                f"parallelism scheme {self.scheme!r} is not one of {SCHEMES}"
            )
        if self.group not in _GROUP_RULES and not _is_fixed_group(self.group):
            raise ParallelDeclarationError(
                f"group rule {self.group!r} is not one of {_GROUP_RULES} "
                "or a fixed size greater than 1"
            )
        if self.scheme == SCHEME_NONE:
            if self.residencies or self.group != GROUP_ONE:
                raise ParallelDeclarationError(
                    "a 'none' module admits no residency and no group"
                )
            return
        if not self.residencies or any(r not in RESIDENCIES for r in self.residencies):
            raise ParallelDeclarationError(
                f"residencies {self.residencies!r} must be a non-empty subset "
                f"of {RESIDENCIES}"
            )
        if (self.scheme in _GROUPED_SCHEMES) == (self.group == GROUP_ONE):
            raise ParallelDeclarationError(
                f"a {self.scheme} module {'needs' if self.scheme in _GROUPED_SCHEMES else 'has no'} "
                f"shard group; got group rule {self.group!r}"
            )

    def admits(
        self, module: ModuleParallelism, world_size: int, attention_group: int
    ) -> bool:
        if module.scheme != self.scheme:
            return False
        if self.scheme == SCHEME_NONE:
            return True
        if module.residency not in self.residencies:
            return False
        size = module.group_size
        if _is_fixed_group(self.group):
            return size == self.group
        if self.group == GROUP_ONE:
            return size == 1
        if self.group == GROUP_WORLD:
            return size == world_size
        if self.group == GROUP_ATTENTION:
            return size == attention_group
        return size >= 1 and world_size % size == 0

    def render(self) -> str:
        if self.scheme == SCHEME_NONE:
            return SCHEME_NONE
        scheme = self.scheme
        if scheme in _GROUPED_SCHEMES:
            scheme = f"{scheme}<{self.group}>"
        return f"{scheme}:{'|'.join(self.residencies)}"


@dataclass(frozen=True)
class LayoutSupport:
    """One implemented (attention, MoE) combination of a phase."""

    attention: ModuleSupport
    moe: ModuleSupport

    def __post_init__(self) -> None:
        for name in ("attention", "moe"):
            if not isinstance(getattr(self, name), ModuleSupport):
                raise ParallelDeclarationError(
                    f"layout support {name} must be a ModuleSupport"
                )
        if self.attention.group == GROUP_ATTENTION:
            raise ParallelDeclarationError(
                "the attention group cannot be defined by itself"
            )

    def admits(self, layout: PhaseLayout) -> bool:
        attention_group = layout.attention.group_size
        return self.attention.admits(
            layout.attention, layout.world_size, attention_group
        ) and self.moe.admits(layout.moe, layout.world_size, attention_group)

    def render(self) -> str:
        return f"attention={self.attention.render()} moe={self.moe.render()}"


def replicated_prefill_world_ep_decode(
    *,
    prefill_moe: Tuple[str, ...] = (RESIDENCY_STREAMED,),
    decode_attention: Tuple[str, ...] = (RESIDENCY_RESIDENT,),
    decode_moe: Tuple[str, ...] = (RESIDENCY_RESIDENT, RESIDENCY_PARTIAL),
) -> dict:
    """The one layout most families implement: every module replicated per
    rank in prefill with the attention weights streamed, and in decode a
    replicated attention with the routed experts partitioned over the world."""
    return {
        PHASE_PREFILL: (
            LayoutSupport(
                ModuleSupport(SCHEME_DP, (RESIDENCY_STREAMED,)),
                ModuleSupport(SCHEME_DP, prefill_moe),
            ),
        ),
        PHASE_DECODE: (
            LayoutSupport(
                ModuleSupport(SCHEME_DP, decode_attention),
                ModuleSupport(SCHEME_EP, decode_moe, GROUP_WORLD),
            ),
        ),
    }


def read_supported_layouts(parallel_manager) -> Optional[dict]:
    """The manager's ``SUPPORTED_LAYOUTS`` (``{phase: (LayoutSupport, ...)}``),
    or None when the family does not declare one. A present table of the wrong
    shape is a hard error."""
    table = getattr(parallel_manager, "SUPPORTED_LAYOUTS", None)
    if table is None:
        return None
    if not isinstance(table, dict) or set(table) != set(LAYOUT_PHASES):
        raise ParallelDeclarationError(
            f"{type(parallel_manager).__name__}.SUPPORTED_LAYOUTS must map "
            f"exactly {LAYOUT_PHASES}"
        )
    for phase, entries in table.items():
        if not isinstance(entries, tuple) or not entries or any(
            not isinstance(entry, LayoutSupport) for entry in entries
        ):
            raise ParallelDeclarationError(
                f"SUPPORTED_LAYOUTS[{phase!r}] must be a non-empty tuple of "
                "LayoutSupport"
            )
    return table


def check_layout_supported(model: str, supported: dict, layout: ServerLayout) -> None:
    """Refuse, by name, a layout the model family does not implement."""
    for phase in LAYOUT_PHASES:
        phase_layout = layout.phase(phase)
        entries = supported[phase]
        if any(entry.admits(phase_layout) for entry in entries):
            continue
        implemented = "; ".join(entry.render() for entry in entries)
        raise UnsupportedLayoutError(
            f"layout {phase} {phase_layout.render()} on {phase_layout.world_size} "
            f"ranks is not implemented for {model}; implemented {phase} "
            f"layouts: {implemented}"
        )


def resolve_server_layout(
    parallel_manager, world_size: int, decode_graph_owner=None
) -> Tuple[Optional[ServerLayout], Optional[ParallelDeclaration]]:
    """The layout the manager built, checked against what its family declares
    as implemented. ``(None, None)`` when the manager declares nothing."""
    declaration = read_parallel_declaration(parallel_manager, decode_graph_owner)
    if declaration is None:
        return None, None
    layout = server_layout_from_declaration(declaration, world_size)
    supported = read_supported_layouts(parallel_manager)
    if supported is not None:
        check_layout_supported(type(parallel_manager).__name__, supported, layout)
    return layout, declaration


REQUESTED_GROUP_FLAGS = {
    (PHASE_PREFILL, "attention"): "--prefill-attention-tp",
    (PHASE_PREFILL, "moe"): "--prefill-moe-ep",
    (PHASE_DECODE, "attention"): "--decode-attention-tp",
    (PHASE_DECODE, "moe"): "--decode-moe-ep",
}


def check_requested_group_sizes(
    model: str, layout: Optional[ServerLayout], requested: dict
) -> None:
    """Refuse, by name, a launch flag whose group size the model did not
    resolve to. ``requested`` maps ``(phase, "attention"|"moe")`` to the
    flag's value or None; a model that declares no layout runs every module
    replicated, which is group size 1."""
    for slot, flag in REQUESTED_GROUP_FLAGS.items():
        value = requested.get(slot)
        if value is None:
            continue
        phase, module = slot
        if layout is None:
            actual, rendered = 1, "an undeclared (replicated) layout"
        else:
            parallelism = getattr(layout.phase(phase), module)
            actual = parallelism.group_size
            rendered = f"{phase} {module}={parallelism.render()}"
        if int(value) != actual:
            raise UnsupportedLayoutError(
                f"{flag} {value} was requested, but {model} resolved "
                f"{rendered} (group size {actual}) on this deployment; the "
                "requested layout is not implemented for it"
            )


def planned_resident_experts(ep_config, world_size: int, num_routed_experts: int) -> tuple:
    """(offloading enabled, routed experts a rank keeps resident in decode).

    A deployment constant: derived from the world size and the EP
    configuration only, so a family's planner, its layout and its strategy
    manager all read this one answer.
    """
    experts_per_rank = int(num_routed_experts) // int(world_size)
    if world_size > 8:
        # Multi-node: every rank's shard is resident.
        return False, experts_per_rank
    if ep_config.enable_offloading:
        return True, int(experts_per_rank * (1 - ep_config.offloading_ratio))
    configured = ep_config.num_local_expert_per_layer
    if configured is None or configured == 0:
        return False, experts_per_rank
    return False, configured


def replicated_prefill_world_ep_decode_layout(
    world_size: int,
    *,
    decode_moe_fully_resident: bool,
    prefill_moe_residency: str = "streamed",
    decode_residency: str = "resident",
):
    """The layout most families run, as an object resolved before the model
    manager is built: every module replicated per rank in prefill with the
    attention weights streamed; in decode a replicated attention and the
    routed experts partitioned over the world. ``decode_residency`` is the
    residency of the decode attention and of a fully resident expert shard;
    a rank that keeps only part of its shard resident is ``partial``."""
    world_size = int(world_size)
    return ServerLayout(
        prefill=PhaseLayout(
            PHASE_PREFILL,
            ModuleParallelism(SCHEME_DP, RESIDENCY_STREAMED),
            ModuleParallelism(SCHEME_DP, prefill_moe_residency),
            world_size,
        ),
        decode=PhaseLayout(
            PHASE_DECODE,
            ModuleParallelism(SCHEME_DP, decode_residency),
            ModuleParallelism(
                SCHEME_EP,
                decode_residency if decode_moe_fully_resident else RESIDENCY_PARTIAL,
                group_size=world_size,
            ),
            world_size,
        ),
    )


def check_planned_server_layout(manager_class, planned, world_size: int) -> None:
    """Check the layout a family's planner resolved, before its manager is
    built: it must be a ``ServerLayout`` over this world and one the family
    declares as implemented. Nothing of the model exists yet when this
    refuses."""
    name = getattr(manager_class, "__name__", type(manager_class).__name__)
    if not isinstance(planned, ServerLayout):
        raise ParallelDeclarationError(
            f"the planned layout of {name} must be a ServerLayout, got "
            f"{type(planned).__name__}"
        )
    if planned.decode.world_size != int(world_size):
        raise ParallelDeclarationError(
            f"the planned layout of {name} is over {planned.decode.world_size} "
            f"ranks but the server runs {int(world_size)}"
        )
    supported = read_supported_layouts(manager_class)
    if supported is not None:
        check_layout_supported(name, supported, planned)


def check_built_layout_is_planned(model: str, planned: ServerLayout, built) -> None:
    """Refuse a manager that was built for another layout than the planned
    one, naming both."""
    if built == planned:
        return

    def render(layout) -> str:
        if layout is None:
            return "no layout"
        return (
            f"prefill {layout.prefill.render()}; decode {layout.decode.render()}"
        )

    raise ParallelDeclarationError(
        f"{model} was built for another layout than the planned one: planned "
        f"{render(planned)}; built {render(built)}"
    )


def render_supported_layouts(tables: dict) -> str:
    """One Markdown table of every family's implemented layouts."""
    lines = ["| family | phase | attention | MoE |", "|---|---|---|---|"]
    for family in sorted(tables):
        for phase in LAYOUT_PHASES:
            for entry in tables[family][phase]:
                lines.append(
                    f"| {family} | {phase} | {entry.attention.render()} | "
                    f"{entry.moe.render()} |"
                )
    return "\n".join(lines)
