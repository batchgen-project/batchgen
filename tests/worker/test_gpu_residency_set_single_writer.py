"""Source guard: the GPU residency set is mutated only where pinned.

``_sequences_with_gpu_kv`` is a bare ``set`` on the worker that records which
sequences currently hold GPU KV pages.  The admission boundary reaches the
same set through the context field ``gpu_resident``.  The acceptance target
is a single owner component; today there is none, so EVERY mutation site in
the package (including the worker's own initialisation rebinding) is pinned
in ``KNOWN_OFFENDERS`` as a list that can only shrink: the test fails when a
NEW mutation site appears, and it fails when a listed offender no longer
exists (stale entry), so every removal is deliberate.

Covered patterns: attribute rebinding/``del``/augmented assignment of
``._sequences_with_gpu_kv`` or ``.gpu_resident`` on any receiver, item
writes, container-mutator calls (``add``, ``discard``, ``clear``, ``update``,
...) on either spelling, the same mutations on a bare local/parameter name
with either spelling, and mutator calls/item writes through a one-level local
alias assigned in the same module (``resident = ctx.gpu_resident;
resident.add(...)`` — no such alias exists in the tree today).  NOT caught:
aliases created by passing the set into a function whose parameter has
another name, aliases of an alias, aliases rebound between uses,
``getattr``/``setattr`` with dynamic names, and in-place union through a
plain-name augmented assignment on an alias created outside the module.
"""
import ast
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "batchgen"
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


def _offences(tree: ast.Module):
    aliases = _module_aliases(tree)
    found = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and node.attr in TARGET_NAMES
            and isinstance(node.ctx, (ast.Store, ast.Del))
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


# The exact mutation sites that exist today (one entry per file and kind of
# mutation, no line numbers so the list does not churn).  Entries may only be
# removed (by moving the mutation behind an owner component), never added.
KNOWN_OFFENDERS = [
    "batchgen/batchgen_worker.py: ._sequences_with_gpu_kv.add(",
    "batchgen/batchgen_worker.py: ._sequences_with_gpu_kv.discard(",
    "batchgen/batchgen_worker.py: rebinding ._sequences_with_gpu_kv",
    "batchgen/worker/admission/boundary_executor.py: .gpu_resident.discard(",
    "batchgen/worker/decode_engine.py: ._sequences_with_gpu_kv.add(",
    "batchgen/worker/kv_capacity_engine.py: ._sequences_with_gpu_kv.add(",
    "batchgen/worker/kv_capacity_engine.py: ._sequences_with_gpu_kv.clear(",
    "batchgen/worker/kv_capacity_engine.py: ._sequences_with_gpu_kv.discard(",
    "batchgen/worker/kv_capacity_engine.py: .gpu_resident.add(",
    "batchgen/worker/kv_capacity_engine.py: .gpu_resident.discard(",
]


def test_the_gpu_residency_set_is_mutated_only_where_pinned():
    offenders = sorted(
        {
            f"{path.relative_to(ROOT)}: {what}"
            for path, tree in _package_trees()
            for _line, what in _offences(tree)
        }
    )
    assert offenders == KNOWN_OFFENDERS


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
        "    return 'u' in ctx.gpu_resident\n"
    )
    lines = sorted(line for line, _ in _offences(_parse(sample)))
    # Line 5 is an augmented in-place union (target has Store context); line 8
    # mutates through the local alias of line 7; line 10 mutates a name that
    # does not alias the set and is not flagged.
    assert lines == [2, 3, 4, 5, 6, 8]
