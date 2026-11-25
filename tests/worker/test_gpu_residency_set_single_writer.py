"""Source guard: the GPU residency set has ONE owner that mutates it.

``batchgen.worker.gpu_residency.GpuResidencySet`` exclusively owns the set of
sequence uuids that currently hold GPU KV pages.  The worker holds it as
``_sequences_with_gpu_kv`` and the admission boundary reaches the same object
through the context field ``gpu_resident``; every other module writes it only
through the owner's named methods (``mark_resident`` / ``drop`` /
``drop_all``).  This guard proves it at the source level:

* no raw container mutation of either spelling exists anywhere in the package
  (``KNOWN_OFFENDERS`` is empty and may never grow);
* the owner is created exactly once, in the worker's ``__init__`` — the only
  sanctioned attribute store is ``self._sequences_with_gpu_kv =
  GpuResidencySet()`` in ``batchgen/batchgen_worker.py``;
* no module other than the owner's reaches the private storage
  (``_gpu_resident_uuids``) by attribute or bare name.

Covered patterns: attribute rebinding/``del``/augmented assignment of
``._sequences_with_gpu_kv`` or ``.gpu_resident`` on any receiver (the single
sanctioned owner construction excepted), item writes, container-mutator calls
(``add``, ``discard``, ``clear``, ``update``, ...) on either spelling, the
same mutations on a bare local/parameter name with either spelling, and
mutator calls/item writes through a one-level local alias assigned in the
same module (``resident = ctx.gpu_resident; resident.add(...)`` — no such
alias exists in the tree today).  NOT caught: aliases created by passing the
set into a function whose parameter has another name, aliases of an alias,
aliases rebound between uses, ``getattr``/``setattr`` with dynamic names, and
in-place union through a plain-name augmented assignment on an alias created
outside the module.  The owner's named write methods are not in the mutator
list on purpose: calling them is the sanctioned write path, and
``GpuResidencySet`` itself carries no ``add``/``discard``/``clear``, so a raw
mutation spelling that this guard misses still fails at runtime.
"""
import ast
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "batchgen"
OWNER_MODULE = PACKAGE / "worker" / "gpu_residency.py"
OWNER_CLASS = "GpuResidencySet"
OWNER_STORAGE = "_gpu_resident_uuids"
TARGET_NAMES = {"_sequences_with_gpu_kv", "gpu_resident"}
MUTATORS = {
    "add", "append", "appendleft", "clear", "difference_update", "discard",
    "extend", "extendleft", "insert", "intersection_update", "pop", "popitem",
    "popleft", "remove", "reverse", "setdefault", "sort",
    "symmetric_difference_update", "update",
}


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


@cache
def _package_trees():
    return [(path, _parse(path)) for path in sorted(PACKAGE.rglob("*.py"))]


def _target_of(expr: ast.AST, aliases: dict):
    """The guarded spelling ``expr`` denotes, or ``None``."""
    if isinstance(expr, ast.Attribute) and expr.attr in TARGET_NAMES:
        return expr.attr
    if isinstance(expr, ast.Name):
        if expr.id in TARGET_NAMES:
            return expr.id
        return aliases.get(expr.id)
    return None


def _module_aliases(tree: ast.Module) -> dict:
    """Local names assigned from a guarded spelling anywhere in the module."""
    aliases = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            target = _target_of(node.value, {})
            if target is not None:
                aliases[node.targets[0].id] = target
    return aliases


def _is_owner_construction(value) -> bool:
    return (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id == OWNER_CLASS
    )


def _sanctioned_creation_targets(tree: ast.Module) -> set:
    """``id()`` of each attribute node stored to by an owner construction.

    The ONLY store of a guarded spelling that is not an offence is binding a
    freshly constructed ``GpuResidencySet()`` to it; where (and how often)
    that may happen is pinned separately by
    ``test_the_worker_creates_the_owner_exactly_once``.
    """
    sanctioned = set()
    for node in ast.walk(tree):
        targets = []
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        else:
            continue
        if not _is_owner_construction(value):
            continue
        for target in targets:
            if isinstance(target, ast.Attribute) and target.attr in TARGET_NAMES:
                sanctioned.add(id(target))
    return sanctioned


def _offences(tree: ast.Module):
    aliases = _module_aliases(tree)
    sanctioned = _sanctioned_creation_targets(tree)
    found = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and node.attr in TARGET_NAMES
            and isinstance(node.ctx, (ast.Store, ast.Del))
            and id(node) not in sanctioned
        ):
            found.append((node.lineno, f"rebinding .{node.attr}"))
        elif isinstance(node, ast.Subscript) and isinstance(
            node.ctx, (ast.Store, ast.Del)
        ):
            target = _target_of(node.value, aliases)
            if target is not None:
                found.append((node.lineno, f"item write on .{target}"))
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr in MUTATORS:
                target = _target_of(func.value, aliases)
                if target is not None:
                    found.append((node.lineno, f".{target}.{func.attr}("))
    return found


# Raw mutation sites that bypass the owner.  The list ended empty when the
# owner component landed; it may NEVER grow — a new entry means a new module
# mutates the residency set without going through ``GpuResidencySet``.
KNOWN_OFFENDERS = []


def test_the_gpu_residency_set_is_mutated_only_through_the_owner():
    offenders = sorted(
        {
            f"{path.relative_to(ROOT)}: {what}"
            for path, tree in _package_trees()
            for _line, what in _offences(tree)
        }
    )
    assert offenders == KNOWN_OFFENDERS


def test_the_worker_creates_the_owner_exactly_once():
    """The sanctioned-creation exemption is a single site in the worker."""
    creations = [
        (path, node.lineno)
        for path, tree in _package_trees()
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (
            node.targets if isinstance(node, ast.Assign) else [node.target]
        )
        if isinstance(target, ast.Attribute)
        and target.attr in TARGET_NAMES
        and _is_owner_construction(
            node.value if node.value is not None else None
        )
    ]
    assert [path.relative_to(ROOT) for path, _line in creations] == [
        Path("batchgen/batchgen_worker.py")
    ]


def test_no_module_but_the_owner_reaches_the_private_storage():
    """``_gpu_resident_uuids`` (read OR write) appears only in the owner module."""
    reachers = sorted(
        {
            f"{path.relative_to(ROOT)}:{node.lineno}"
            for path, tree in _package_trees()
            if path != OWNER_MODULE
            for node in ast.walk(tree)
            if (isinstance(node, ast.Attribute) and node.attr == OWNER_STORAGE)
            or (isinstance(node, ast.Name) and node.id == OWNER_STORAGE)
            or (isinstance(node, ast.Constant) and node.value == OWNER_STORAGE)
        }
    )
    assert reachers == []


def test_the_guard_sees_each_kind_of_write(tmp_path):
    """The guard is not vacuous: every kind of mutation it names is detected."""
    sample = tmp_path / "sample.py"
    sample.write_text(
        "def f(self, ctx, gpu_resident):\n"
        "    self._sequences_with_gpu_kv = set()\n"
        "    self._sequences_with_gpu_kv.add('u')\n"
        "    ctx.gpu_resident.discard('u')\n"
        "    ctx.gpu_resident |= {'u'}\n"
        "    gpu_resident.clear()\n"
        "    resident = ctx.gpu_resident\n"
        "    resident.add('u')\n"
        "    unrelated = ctx.other\n"
        "    unrelated.add('u')\n"
        "    self._sequences_with_gpu_kv = GpuResidencySet()\n"
        "    self._sequences_with_gpu_kv = GpuResidencySet({'u'})\n"
        "    return 'u' in ctx.gpu_resident\n"
    )
    lines = sorted(line for line, _ in _offences(_parse(sample)))
    # Line 5 is an augmented in-place union (target has Store context); line 8
    # mutates through the local alias of line 7; line 10 mutates a name that
    # does not alias the set and is not flagged.  Lines 11-12 are owner
    # constructions: sanctioned by _offences, pinned to one worker site by
    # test_the_worker_creates_the_owner_exactly_once.
    assert lines == [2, 3, 4, 5, 6, 8]
