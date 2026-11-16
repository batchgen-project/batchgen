# ---------------------------------------------------------------------------- #
#  BatchGen                                                                     #
#  Copyright (c) 2025-2026 BatchGen Team                                        #
#                                                                               #
#  licensed under the apache license, version 2.0 (the "license");              #
# ---------------------------------------------------------------------------- #

"""The JIT kernel coverage registry: one source of truth binding kernel
backends, their call sites, and the operator slots that select them.

This module is the FRAMEWORK only (dataclasses, registration, checks) and
names no model: every backend inventory and operator slot is registered by a
model-layer data module (``batchgen/models/<family>/jit_coverage.py``) through
the ``batchgen.models.jit_coverage`` seam, keeping model knowledge below the
kernel layer. Importable without torch/triton/CUDA.

Two layers:

* **Backend inventories** — for every JIT backend, the scan roots, the
  wrapper callees, the non-kernel attributes, and the pinned call-site table
  ``(path, enclosing scope, callee) -> (count, reaches)``. A repo-wide CI
  test scans the roots (direct attribute calls AND ``getattr(<backend>,
  "<kernel>")`` resolutions) and requires the table to match exactly, so a
  newly merged call site cannot escape prewarm coverage silently. ``reaches``
  names the prewarm site(s) a call compiles through, ``"wrapper"`` for a
  helper whose own call sites are listed, or ``"unreachable: ..."``.

* **Operator slots** — a replaceable operator slot a model family declares:
  its validated backends (each marked ``("jit", <inventory backend>)`` or
  ``("aot",)``) and the default. ``check_slot_backend`` is the ONLY
  validation path for a selector's resolved value: an unknown slot, an
  unvalidated backend, or a JIT backend without a registered inventory is
  refused with a named error — never a silent fallback. Families without
  registered slots are, truthfully, not checked.
"""

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

__all__ = [
    "BackendInventory",
    "JitCoverageError",
    "OperatorSlot",
    "backend_inventories",
    "backend_inventory",
    "check_family_coverage",
    "check_model_type_coverage",
    "check_slot_backend",
    "default_backend",
    "family_for_model_type",
    "register_backend_inventory",
    "register_model_type_family",
    "register_slot",
    "slots_for",
]


class JitCoverageError(RuntimeError):
    """A selector resolved to a backend the registry does not cover."""


@dataclass(frozen=True)
class BackendInventory:
    """One JIT backend's static call-site contract."""

    backend: str
    paths: Tuple[str, ...]
    callees: Tuple[str, ...]
    non_kernels: Tuple[str, ...]
    call_sites: Mapping[Tuple[str, str, str], Tuple[int, Any]]


@dataclass(frozen=True)
class OperatorSlot:
    """One replaceable operator slot of a model family.

    ``backends`` maps each VALIDATED backend to its coverage:
    ``("jit", <inventory backend>)`` requires that backend inventory to be
    registered; ``("aot",)`` compiles nothing at runtime and needs none.
    """

    family: str
    slot: str
    backends: Mapping[str, Tuple[str, ...]]
    default: str

    def __post_init__(self):
        if self.default not in self.backends:
            raise ValueError(
                f"slot {self.family}.{self.slot}: default backend "
                f"{self.default!r} is not among its validated backends "
                f"{sorted(self.backends)}"
            )


_INVENTORIES: Dict[str, BackendInventory] = {}
_SLOTS: Dict[Tuple[str, str], OperatorSlot] = {}
_MODEL_TYPE_FAMILIES: Dict[str, str] = {}


def register_backend_inventory(inventory: BackendInventory) -> BackendInventory:
    """Register a backend's call-site contract (idempotent re-registration
    of an identical contract is a no-op; a conflicting one is refused)."""
    existing = _INVENTORIES.get(inventory.backend)
    if existing is not None and existing != inventory:
        raise JitCoverageError(
            f"backend {inventory.backend!r} is already registered with a "
            "different inventory"
        )
    _INVENTORIES[inventory.backend] = inventory
    return inventory


def register_slot(slot: OperatorSlot) -> OperatorSlot:
    """Register one operator slot (same idempotency rule as inventories)."""
    key = (slot.family, slot.slot)
    existing = _SLOTS.get(key)
    if existing is not None and existing != slot:
        raise JitCoverageError(
            f"slot {slot.family}.{slot.slot} is already registered with a "
            "different declaration"
        )
    _SLOTS[key] = slot
    return slot


def register_model_type_family(prefix: str, family: str) -> None:
    """Map runtime model types starting with ``prefix`` to a registry family
    (registered by the family's coverage module, like its slots)."""
    existing = _MODEL_TYPE_FAMILIES.get(prefix)
    if existing is not None and existing != family:
        raise JitCoverageError(
            f"model-type prefix {prefix!r} is already mapped to family "
            f"{existing!r}"
        )
    _MODEL_TYPE_FAMILIES[prefix] = family


def family_for_model_type(model_type: str):
    """The registry family of a model type, or None (unregistered model
    types declare no slots and are, truthfully, not coverage-checked)."""
    for prefix, family in _MODEL_TYPE_FAMILIES.items():
        if model_type.startswith(prefix):
            return family
    return None


def check_model_type_coverage(model_type: str) -> Tuple[str, ...]:
    """``check_family_coverage`` of the model type's registered family."""
    return check_family_coverage(family_for_model_type(model_type))


def backend_inventories() -> Tuple[BackendInventory, ...]:
    """Every registered backend inventory (for the repo-wide scanner)."""
    return tuple(_INVENTORIES.values())


def backend_inventory(backend: str) -> BackendInventory:
    """One backend's inventory; raises ``JitCoverageError`` when unknown."""
    try:
        return _INVENTORIES[backend]
    except KeyError:
        raise JitCoverageError(
            f"JIT backend {backend!r} has no registered call-site inventory; "
            f"registered: {sorted(_INVENTORIES)}"
        ) from None


def slots_for(family: str) -> Tuple[OperatorSlot, ...]:
    """The operator slots a model family declares (may be empty)."""
    return tuple(s for (f, _), s in _SLOTS.items() if f == family)


def default_backend(family: str, slot: str) -> str:
    """The slot's default backend, with its coverage validated (an undeclared
    slot or a default whose JIT inventory is missing raises)."""
    declared = _SLOTS.get((family, slot))
    if declared is None:
        raise JitCoverageError(
            f"model family {family!r} declares no operator slot {slot!r}; "
            f"declared: {sorted(s for f, s in _SLOTS if f == family)}"
        )
    check_slot_backend(family, slot, declared.default)
    return declared.default


def check_slot_backend(family: str, slot: str, backend: str) -> OperatorSlot:
    """Validate a selector's resolved backend against the registry.

    Raises ``JitCoverageError`` for an undeclared slot, a backend outside the
    slot's validated set, or a JIT backend whose inventory is missing. This
    is the single validation path for every selector — a caller must not
    keep its own allowlist beside it.
    """
    declared = _SLOTS.get((family, slot))
    if declared is None:
        raise JitCoverageError(
            f"model family {family!r} declares no operator slot {slot!r}; "
            f"declared: {sorted(s for f, s in _SLOTS if f == family)}"
        )
    coverage = declared.backends.get(backend)
    if coverage is None:
        raise JitCoverageError(
            f"backend {backend!r} is not validated for {family}.{slot}; "
            f"validated: {sorted(declared.backends)}"
        )
    if coverage[0] == "jit":
        backend_inventory(coverage[1])
    return declared


def check_family_coverage(family) -> Tuple[str, ...]:
    """Validate every operator slot a family declares.

    Returns ``"family.slot=default"`` strings for the preflight manifest;
    ``None`` or an unknown family declares no slots and yields ``()``,
    truthfully unchecked. Raises ``JitCoverageError`` when a declared slot's
    default backend is unvalidated or lacks its JIT inventory.
    """
    if family is None:
        return ()
    checked = []
    for declared in slots_for(family):
        default_backend(declared.family, declared.slot)
        checked.append(f"{declared.family}.{declared.slot}={declared.default}")
    return tuple(checked)
