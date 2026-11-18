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

  One backend may be called from several model families' sources (the DSA
  indexer's DeepGEMM kernels serve GLM-5 and Kimi-K3.1). Each family then
  registers its own inventory PART for the backend: it names its ``family``,
  its scan roots (disjoint from every other part's, so one call site has one
  owner) and its own table, and the scanner holds each part's roots to its
  own table. ``backend_inventories`` lists the parts; ``backend_inventory``
  is the backend's merged view.

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
    """One JIT backend's static call-site contract (or one family's part of
    it, see the module docstring)."""

    backend: str
    paths: Tuple[str, ...]
    callees: Tuple[str, ...]
    non_kernels: Tuple[str, ...]
    call_sites: Mapping[Tuple[str, str, str], Tuple[int, Any]]
    #: The model family owning this part. Required once a backend has more
    #: than one part; a single unnamed part is the backend's whole contract.
    family: str = ""


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


# backend -> family -> part
_INVENTORIES: Dict[str, Dict[str, BackendInventory]] = {}
_SLOTS: Dict[Tuple[str, str], OperatorSlot] = {}
_MODEL_TYPE_FAMILIES: Dict[str, str] = {}


def _paths_overlap(a: str, b: str) -> bool:
    a, b = a.rstrip("/"), b.rstrip("/")
    return a == b or a.startswith(b + "/") or b.startswith(a + "/")


def register_backend_inventory(inventory: BackendInventory) -> BackendInventory:
    """Register a backend's call-site contract, or one family's part of it.

    Idempotent re-registration of an identical part is a no-op. Refused: a
    different part for an already registered ``(backend, family)``; a second
    part when either part names no family; a part whose scan roots overlap
    another part's (a call site would have two owners).
    """
    parts = _INVENTORIES.setdefault(inventory.backend, {})
    existing = parts.get(inventory.family)
    if existing is not None:
        if existing != inventory:
            raise JitCoverageError(
                f"backend {inventory.backend!r} is already registered with a "
                f"different inventory for family {inventory.family!r}"
            )
        return existing
    if parts and (not inventory.family or "" in parts):
        raise JitCoverageError(
            f"backend {inventory.backend!r} is already registered with a "
            "different inventory; a backend with several inventory parts "
            "needs every part to name its family"
        )
    for other in parts.values():
        clashes = sorted(
            (mine, theirs) for mine in inventory.paths for theirs in other.paths
            if _paths_overlap(mine, theirs)
        )
        if clashes:
            raise JitCoverageError(
                f"backend {inventory.backend!r}: the {inventory.family!r} "
                f"inventory's scan roots overlap the {other.family!r} part's "
                f"{clashes}; every call site needs exactly one owner"
            )
    parts[inventory.family] = inventory
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
    """Every registered inventory part (for the repo-wide scanner, which
    holds each part's scan roots to that part's table)."""
    return tuple(part for parts in _INVENTORIES.values()
                 for part in parts.values())


def backend_inventory(backend: str) -> BackendInventory:
    """One backend's inventory, merged over its family parts (scan roots,
    callees and non-kernels in registration order, every call site); raises
    ``JitCoverageError`` when unknown."""
    parts = tuple(_INVENTORIES.get(backend, {}).values())
    if not parts:
        raise JitCoverageError(
            f"JIT backend {backend!r} has no registered call-site inventory; "
            f"registered: {sorted(b for b, p in _INVENTORIES.items() if p)}"
        )
    if len(parts) == 1:
        return parts[0]

    def merged(name):
        return tuple(dict.fromkeys(v for part in parts
                                   for v in getattr(part, name)))

    call_sites: Dict[Tuple[str, str, str], Tuple[int, Any]] = {}
    for part in parts:
        call_sites.update(part.call_sites)
    return BackendInventory(
        backend=backend, paths=merged("paths"), callees=merged("callees"),
        non_kernels=merged("non_kernels"), call_sites=call_sites,
    )


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
