# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  Copyright (c) 2025-2026 BatchGen Team                                            #
# ---------------------------------------------------------------------------- #

"""Model-declared parallel layout, read from the parallel-strategy manager.

The attention and MoE parallelism a deployment runs is decided inside each
model's parallel-strategy manager, so only the manager can say what it is.
This module is the worker's typed contract for that declaration: a manager
that implements ``parallel_declaration()`` returns a ``ParallelDeclaration``
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
from typing import Optional

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
    """A model's declared layout for every (phase, module) slot."""

    prefill_attention: ModuleParallelism
    prefill_moe: ModuleParallelism
    decode_attention: ModuleParallelism
    decode_moe: ModuleParallelism

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

    def rendered_slots(self) -> dict:
        """``{slot: token}`` in ``DECLARATION_SLOTS`` order."""
        return {slot: getattr(self, slot).render() for slot in DECLARATION_SLOTS}


def undeclared_slots() -> dict:
    """What the ``[SERVER_CONFIG]`` line reports without a declaration."""
    return {slot: UNDECLARED for slot in DECLARATION_SLOTS}


def read_parallel_declaration(parallel_manager) -> Optional[ParallelDeclaration]:
    """The manager's declaration, or None when it does not implement one.

    Read like the worker's other optional model declarations
    (``recurrent_state_owner``): absence is a legal, explicit state — the
    caller reports ``undeclared`` — but a present contract that returns the
    wrong type is a hard error, never coerced.
    """
    declares = getattr(parallel_manager, "parallel_declaration", None)
    if declares is None:
        return None
    declaration = declares()
    if not isinstance(declaration, ParallelDeclaration):
        raise ParallelDeclarationError(
            f"{type(parallel_manager).__name__}.parallel_declaration() must "
            f"return a ParallelDeclaration, got {type(declaration).__name__}"
        )
    return declaration
