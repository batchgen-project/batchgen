"""Source guard: the trajectory store is the only writer of trajectory state.

``TrajectoryStore`` (``batchgen/worker/trajectory_store.py``) owns the
worker's trajectory book, frontier lifecycle objects and their lazy state;
everything else must go through its methods.  This test fails when any module
other than the owner touches the store's private storage, or mutates one of
its public fields (rebinding, ``del``, an item write, or a container mutator
call).  Reads of public fields and calls of owner methods are allowed.

The field names are derived from the owner class's AST (every attribute the
class assigns on ``self``), so new fields are guarded automatically.  AST
carries no types, so a field name that some other definition in the package
also uses (a ``self`` assignment in another class, a method, a class-level or
module-level name) cannot be attributed to the store by spelling alone.  Such
ambiguous names are checked only when the receiver expression is recognisably
the store: a bare name or an attribute chain ending in one of the holder
spellings the code base uses (``self._traj``; ``trajectory_store`` is the
conventional spelling for any future holder).  Unambiguous names are checked
on every receiver.  Not caught: aliasing through a local variable whose name
is not a holder spelling, and mutation via ``getattr``/``setattr`` strings.

The tree currently contains known outside writes: the worker and the prefill
engine still initialize and rebind the store's fields directly.  They are
pinned in ``KNOWN_OFFENDERS`` (one entry per file and kind of write, no line
numbers so the list does not churn); the test asserts exact equality, so the
list can shrink deliberately but never grow silently.
"""
import ast
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "batchgen"
OWNER = PACKAGE / "worker" / "trajectory_store.py"
OWNER_CLASSES = {"TrajectoryStore"}
HOLDER_NAMES = {"_traj", "trajectory_store"}
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


def _self_assigned_names(classdef: ast.ClassDef) -> set:
    """Attribute names the class assigns on ``self``."""
    names = set()
    for sub in ast.walk(classdef):
        if isinstance(sub, ast.Assign):
            targets = sub.targets
        elif isinstance(sub, (ast.AnnAssign, ast.AugAssign)):
            targets = [sub.target]
        else:
            continue
        for target in targets:
            for node in ast.walk(target):
                if (
                    isinstance(node, ast.Attribute)
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "self"
                ):
                    names.add(node.attr)
    return names


@cache
def _owner_fields():
    """``(private, public)`` field-name sets of the owner class(es)."""
    private, public = set(), set()
    for node in ast.walk(_parse(OWNER)):
        if isinstance(node, ast.ClassDef) and node.name in OWNER_CLASSES:
            for name in _self_assigned_names(node):
                (private if name.startswith("_") else public).add(name)
    return private, public


@cache
def _ambiguous_names():
    """Owner field names that some non-owner definition in the package also
    uses, so an access cannot be attributed to the owner by spelling alone."""
    private, public = _owner_fields()
    fields = private | public
    foreign = set()
    for path, tree in _package_trees():
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            if path == OWNER and node.name in OWNER_CLASSES:
                continue
            foreign |= _self_assigned_names(node)
            for stmt in node.body:
                if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    foreign.add(stmt.name)
                elif isinstance(stmt, ast.AnnAssign) and isinstance(
                    stmt.target, ast.Name
                ):
                    foreign.add(stmt.target.id)
                elif isinstance(stmt, ast.Assign):
                    foreign.update(
                        t.id for t in stmt.targets if isinstance(t, ast.Name)
                    )
        for stmt in tree.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                foreign.add(stmt.name)
            elif isinstance(stmt, ast.AnnAssign) and isinstance(
                stmt.target, ast.Name
            ):
                foreign.add(stmt.target.id)
            elif isinstance(stmt, ast.Assign):
                foreign.update(
                    t.id for t in stmt.targets if isinstance(t, ast.Name)
                )
    return fields & foreign


def _is_recognised_owner(expr: ast.AST) -> bool:
    if isinstance(expr, ast.Name):
        return expr.id in HOLDER_NAMES
    if isinstance(expr, ast.Attribute):
        return expr.attr in HOLDER_NAMES
    return False


def _checked(receiver: ast.AST, name: str, ambiguous: set) -> bool:
    return name not in ambiguous or _is_recognised_owner(receiver)


def _offences(tree: ast.Module, private: set, public: set, ambiguous: set):
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            name = node.attr
            if name in private:
                if _checked(node.value, name, ambiguous):
                    found.append((node.lineno, f"private storage .{name}"))
            elif name in public and isinstance(node.ctx, (ast.Store, ast.Del)):
                if _checked(node.value, name, ambiguous):
                    found.append((node.lineno, f"rebinding .{name}"))
        elif isinstance(node, ast.Subscript) and isinstance(
            node.ctx, (ast.Store, ast.Del)
        ):
            value = node.value
            if (
                isinstance(value, ast.Attribute)
                and value.attr in public
                and _checked(value.value, value.attr, ambiguous)
            ):
                found.append((node.lineno, f"item write on .{value.attr}"))
        elif isinstance(node, ast.Call):
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr in MUTATORS
                and isinstance(func.value, ast.Attribute)
                and func.value.attr in public
                and _checked(func.value.value, func.value.attr, ambiguous)
            ):
                found.append(
                    (node.lineno, f".{func.value.attr}.{func.attr}(")
                )
    return found


# The exact outside writes that exist today.  Entries may only be removed
# (by moving the write behind a TrajectoryStore method), never added.
KNOWN_OFFENDERS = [
    "batchgen/batchgen_worker.py: .trajectory_attachments.pop(",
    "batchgen/batchgen_worker.py: item write on .trajectory_attachments",
    "batchgen/batchgen_worker.py: rebinding .trajectory_arena_payload",
    "batchgen/batchgen_worker.py: rebinding .trajectory_attachments",
    "batchgen/batchgen_worker.py: rebinding .trajectory_book",
    "batchgen/batchgen_worker.py: rebinding .trajectory_frontier_committer",
    "batchgen/batchgen_worker.py: rebinding .trajectory_memfd",
    "batchgen/batchgen_worker.py: rebinding .trajectory_mirrored_restores",
    "batchgen/batchgen_worker.py: rebinding .trajectory_pool_capacity_signature",
    "batchgen/batchgen_worker.py: rebinding .trajectory_restorer",
    "batchgen/batchgen_worker.py: rebinding .trajectory_session_groups",
    "batchgen/worker/prefill_engine.py: item write on .trajectory_attachments",
    "batchgen/worker/prefill_engine.py: rebinding .trajectory_arena_payload",
    "batchgen/worker/prefill_engine.py: rebinding .trajectory_mirrored_restores",
]


def test_only_the_trajectory_store_writes_its_state():
    private, public = _owner_fields()
    ambiguous = _ambiguous_names()
    offenders = sorted(
        {
            f"{path.relative_to(ROOT)}: {what}"
            for path, tree in _package_trees()
            if path != OWNER
            for _line, what in _offences(tree, private, public, ambiguous)
        }
    )
    assert offenders == KNOWN_OFFENDERS


def test_derived_fields_describe_the_owner():
    """The guard is not vacuous: the derived sets match a manual read."""
    private, public = _owner_fields()
    assert len(private) >= 1
    assert "_recurrent_state_owner" in private
    assert len(public) >= 9
    assert {
        "trajectory_book",
        "trajectory_attachments",
        "trajectory_frontier_committer",
    } <= public
    ambiguous = _ambiguous_names()
    # The worker defines a ``_recurrent_state_owner`` method of its own, so
    # that spelling is receiver-restricted; the trajectory_* fields are
    # unique in the package and stay checked on every receiver.
    assert "_recurrent_state_owner" in ambiguous
    assert "trajectory_book" not in ambiguous
    assert "trajectory_attachments" not in ambiguous


def test_the_guard_sees_each_kind_of_write(tmp_path):
    """The guard is not vacuous: every kind of write it names is detected."""
    private, public = _owner_fields()
    ambiguous = _ambiguous_names()
    sample = tmp_path / "sample.py"
    sample.write_text(
        "def f(worker, ctx):\n"
        "    worker._traj._recurrent_state_owner = None\n"
        "    ctx.trajectory_book = None\n"
        "    worker._traj.trajectory_attachments[3] = object()\n"
        "    del ctx.trajectory_restorer\n"
        "    worker._traj.trajectory_arena_payload.pop(3)\n"
        "    return worker._traj.trajectory_book\n"
    )
    lines = sorted(
        line
        for line, _ in _offences(_parse(sample), private, public, ambiguous)
    )
    assert lines == [2, 3, 4, 5, 6]
