"""Required manager contract of the phase-lifecycle refactor.

Every parallel strategy manager implements exactly four hooks:

    phase_enter(self, key: PhaseKey, ctx: PhaseEnterContext)
        -> tuple[object, dict]          # (model, weight_copy_task)
    admit(self, key: PhaseKey, ctx: AdmitContext) -> None
    phase_exit(self, key: PhaseKey, mode: ExitMode) -> None
    holds_instance(self, phase: str) -> bool

``phase_enter`` runs only on a phase transition and returns the
``(model, weight_copy_task)`` pair the worker installs; whether the manager
builds a fresh instance or resumes a suspended one is its private strategy.
``admit`` runs once per admission wave inside a live phase. ``phase_exit``
leaves the live phase in one of the two ``ExitMode`` s. ``holds_instance``
reports whether the manager retains a resumable instance for the named phase.

The hooks are required, not optional capabilities: ``verify_phase_contract``
reports every missing or drifted hook so a nonconforming manager fails at
startup, never at its first phase switch, and there is no ``hasattr``-style
downgrade path.

Manager hooks own MANAGER state only: phase instances, routed experts,
grouped prefill buffers, copy-task identity. Worker-owned mechanics -- H2D
worker stop and ring release, GPU KV destroy and remap,
``deep_free_model_memory``, CUDA-graph handling -- stay worker methods and
are orchestrated by the worker in a fixed order around these hooks.

The worker calls these hooks at phase transitions and admission waves. The
manager owns phase-instance state while the worker owns the surrounding
device mechanics.
"""

from __future__ import annotations

import enum
import inspect
from dataclasses import dataclass

PHASES = ("prefill", "decode")


@dataclass(frozen=True)
class PhaseKey:
    """Compatibility key of a phase instance.

    Two keys that compare equal name the same instance: re-entering with an
    equal key never rebuilds, while a key that differs from the live one
    forces ``phase_exit(RETIRE)`` followed by ``phase_enter``.

    ``profile_overrides`` is the normalized projection of the batch-level
    debug/profile selections that change which instance is valid, as sorted
    ``(name, value)`` pairs. The only selection that feeds phase selection
    today is ``(("k3_prefill_profile", bool),)``.
    """

    phase: str
    prefill_moe_mode: str | None
    profile_overrides: tuple[tuple[str, object], ...] = ()

    def __post_init__(self):
        if self.phase not in PHASES:
            raise ValueError(
                f"PhaseKey.phase must be one of {PHASES}, got {self.phase!r}"
            )
        if not isinstance(self.profile_overrides, tuple) or not all(
            isinstance(item, tuple)
            and len(item) == 2
            and isinstance(item[0], str)
            for item in self.profile_overrides
        ):
            raise ValueError(
                "PhaseKey.profile_overrides must be a tuple of (str, value) "
                f"pairs, got {self.profile_overrides!r}"
            )


class ExitMode(enum.Enum):
    """How ``phase_exit`` leaves the live phase.

    SUSPEND: stop the phase's consumers and release its transient memory,
    but keep every fixed-address structure -- KV pool virtual addresses,
    captured CUDA graphs, pinned host expert copies -- so the next
    ``phase_enter`` of the same key resumes without rebuilding. This is
    today's implicit decode suspend: the reused-instance switch keeps the
    decode graphs, KV pools and their addresses
    (the worker's decode suspend orchestration),
    the persistent switch keeps the pool buffers and releases only the
    routed experts (batchgen_worker.py:12075-12081,
    and the prefill H2D ring is
    released before decode builds (batchgen_worker.py:11603-11608).

    RETIRE: full deep free; nothing of the instance survives. This is
    today's rebuild path: ``deep_free_model_memory()`` before a prefill or
    decode rebuild (batchgen_worker.py:11291-11292, :11633) and the GPU KV
    destroy (batchgen_worker.py:11295-11305). Used on a key mismatch,
    cohort end, memory pressure and shutdown.
    """

    SUSPEND = "suspend"
    RETIRE = "retire"


@dataclass(frozen=True)
class PhaseEnterContext:
    """Worker-supplied inputs of ``phase_enter``.

    Both fields are decode-only; prefill ignores them. A decode entry
    requires both whichever internal strategy the manager runs, because the
    worker cannot know whether the manager resumes or builds.

    comm: the NCCL communicator the decode MoE all-gathers/reduces over.
    max_num_seq: the per-rank decode batch the decode buffers are sized for,
        already resolved by the worker (the admissible maximum for a reused
        instance, the batch estimate otherwise).
    """

    comm: object | None = None
    max_num_seq: int | None = None


@dataclass(frozen=True)
class AdmitContext:
    """One admission wave.

    uuids: the sequences admitted by this wave.
    local_indices: decode only -- each admitted sequence's local decode row.
    """

    uuids: list[str]
    local_indices: list[int] | None = None


_HOOK_PARAMETERS = {
    "phase_enter": ("self", "key", "ctx"),
    "admit": ("self", "key", "ctx"),
    "phase_exit": ("self", "key", "mode"),
    "holds_instance": ("self", "phase"),
}


def _signature_errors(owner, name, member, expected):
    where = f"{owner}.{name}"
    if not inspect.isfunction(member):
        return [
            f"{where}: must be a plain instance method, "
            f"got {type(member).__name__}"
        ]
    errors = []
    params = list(
        inspect.signature(member, follow_wrapped=False).parameters.values()
    )
    for param in params:
        if param.kind is inspect.Parameter.VAR_POSITIONAL:
            errors.append(f"{where}: *{param.name} is not allowed")
        elif param.kind is inspect.Parameter.VAR_KEYWORD:
            errors.append(f"{where}: **{param.name} is not allowed")
    named = [
        param
        for param in params
        if param.kind
        not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]
    names = tuple(param.name for param in named)
    if names != expected:
        errors.append(
            f"{where}: parameters ({', '.join(names)}) do not match the "
            f"contract ({', '.join(expected)})"
        )
        return errors
    for param in named:
        if param.kind is not inspect.Parameter.POSITIONAL_OR_KEYWORD:
            errors.append(
                f"{where}: parameter {param.name!r} must be positional-or-"
                f"keyword, got {param.kind.description}"
            )
        if param.default is not inspect.Parameter.empty:
            errors.append(
                f"{where}: parameter {param.name!r} must not have a default"
            )
    return errors


def verify_phase_contract(manager) -> list[str]:
    """Return one readable error per contract violation of ``manager``.

    Inspects ``type(manager)`` for exactly ``phase_enter``, ``admit``,
    ``phase_exit`` and ``holds_instance`` with the contract's parameter names
    and kinds. An empty
    list means the manager conforms.
    """
    cls = type(manager)
    owner = cls.__qualname__
    errors = []
    for name, expected in _HOOK_PARAMETERS.items():
        try:
            member = inspect.getattr_static(cls, name)
        except AttributeError:
            errors.append(f"{owner}.{name}: required phase hook is missing")
            continue
        errors.extend(_signature_errors(owner, name, member, expected))
        if name == "holds_instance" and inspect.isfunction(member):
            signature = inspect.signature(member, follow_wrapped=False)
            params = list(signature.parameters.values())
            if len(params) == 2 and params[1].name == "phase":
                if params[1].annotation not in (str, "str"):
                    errors.append(
                        f"{owner}.holds_instance: parameter 'phase' must be "
                        "annotated str"
                    )
            if signature.return_annotation not in (bool, "bool"):
                errors.append(
                    f"{owner}.holds_instance: return must be annotated bool"
                )
    return errors
