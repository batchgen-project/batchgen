"""Source guard: the prefix-reuse state is written only where pinned.

``PrefixReuseState`` (``batchgen/worker/prefix_reuse_state.py``) owns the
prefix sequence states, the pool pseudo-sid namespace/refcounts, the digest
cache and the credit latches; everything else must go through its methods.
The worker holds it as ``self._prefix`` and hands it to the admission
contexts as ``prefix`` (``ctx.prefix``, ``ctx.sched.prefix``).  This test
pins the exact set of outside writes as a list that could only shrink; the
migration is COMPLETE and the list is empty, so it fails when any outside
write appears: every write must go through an owner method.

The field names are derived from the owner class's AST (every attribute it
assigns on ``self``), so new fields are guarded automatically.  AST carries
no types, so a field name that some other definition in the package also uses
(``rank`` and ``world_size`` are assigned on ``self`` by dozens of classes)
cannot be attributed to the state by spelling alone; such ambiguous names are
checked only when the receiver is recognisably the state: a bare name or an
attribute chain ending in ``_prefix``/``prefix``, or a local name assigned
from such a spelling in the same module.  Unambiguous names (the containers
and latches) are checked on every receiver.

Covered patterns: attribute rebinding/``del``/augmented assignment of a field
(``self._prefix.shared_prefix_credit_suspended = False``,
``ctx.prefix.shared_prefix_credit_applied = ...``), item writes on a field
(``self._prefix.prefix_sequence_states[gid] = state``), container-mutator
calls on a field (``....active_prefix_materializations.append(...)``), and
the same through a one-level local alias of the state object
(``prefix = self._prefix; prefix.pool_member_sids[gid] = chain`` — no such
alias exists in the tree today).  NOT caught: an alias of a FIELD rather
than of the state (``states = ctx.prefix.prefix_sequence_states;
states[gid] = ...``), aliases through a function parameter with another
name, ``getattr``/``setattr`` with dynamic names, and writes to an ambiguous
name through an unrecognised receiver spelling.
"""
import ast
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "batchgen"
OWNER = PACKAGE / "worker" / "prefix_reuse_state.py"
OWNER_CLASSES = {"PrefixReuseState"}
HOLDER_NAMES = {"_prefix", "prefix"}
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
    """Field names of the owner class (all public today)."""
    fields = set()
    for node in ast.walk(_parse(OWNER)):
        if isinstance(node, ast.ClassDef) and node.name in OWNER_CLASSES:
            fields |= _self_assigned_names(node)
    return fields


@cache
def _ambiguous_names():
    """Owner field names that some non-owner definition in the package also
    uses, so an access cannot be attributed to the state by spelling alone."""
    fields = _owner_fields()
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


def _holder_aliases(tree: ast.Module) -> set:
    """Local names assigned from a holder spelling anywhere in the module."""
    aliases = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Attribute)
            and node.value.attr in HOLDER_NAMES
        ):
            aliases.add(node.targets[0].id)
    return aliases


def _is_recognised_owner(expr: ast.AST, aliases: set) -> bool:
    if isinstance(expr, ast.Name):
        return expr.id in HOLDER_NAMES or expr.id in aliases
    if isinstance(expr, ast.Attribute):
        return expr.attr in HOLDER_NAMES
    return False


def _checked(receiver, name: str, ambiguous: set, aliases: set) -> bool:
    return name not in ambiguous or _is_recognised_owner(receiver, aliases)


def _offences(tree: ast.Module, fields: set, ambiguous: set):
    aliases = _holder_aliases(tree)
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            if node.attr in fields and isinstance(node.ctx, (ast.Store, ast.Del)):
                if _checked(node.value, node.attr, ambiguous, aliases):
                    found.append((node.lineno, f"rebinding .{node.attr}"))
        elif isinstance(node, ast.Subscript) and isinstance(
            node.ctx, (ast.Store, ast.Del)
        ):
            value = node.value
            if (
                isinstance(value, ast.Attribute)
                and value.attr in fields
                and _checked(value.value, value.attr, ambiguous, aliases)
            ):
                found.append((node.lineno, f"item write on .{value.attr}"))
        elif isinstance(node, ast.Call):
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr in MUTATORS
                and isinstance(func.value, ast.Attribute)
                and func.value.attr in fields
                and _checked(func.value.value, func.value.attr, ambiguous, aliases)
            ):
                found.append(
                    (node.lineno, f".{func.value.attr}.{func.attr}(")
                )
    return found


# Every outside write has been moved behind a PrefixReuseState method
# (entries were one per file and kind of write while the migration ran).
# The list stays empty: a new entry may never be added.
KNOWN_OFFENDERS = []


def test_only_the_prefix_reuse_state_writes_its_fields():
    fields = _owner_fields()
    ambiguous = _ambiguous_names()
    offenders = sorted(
        {
            f"{path.relative_to(ROOT)}: {what}"
            for path, tree in _package_trees()
            if path != OWNER
            for _line, what in _offences(tree, fields, ambiguous)
        }
    )
    assert offenders == KNOWN_OFFENDERS


def test_derived_fields_describe_the_owner():
    """The guard is not vacuous: the derived sets match a manual read."""
    fields = _owner_fields()
    assert {
        "prefix_sequence_states",
        "active_prefix_materializations",
        "pool_sid_counter",
        "pool_sid_members",
        "pool_member_sids",
        "shared_prefix_credit_suspended",
        "shared_prefix_credit_applied",
        "prefix_page_digest_cache",
        "prefix_drain_metric_emitted",
        "rank",
        "world_size",
    } <= fields
    ambiguous = _ambiguous_names()
    # ``rank``/``world_size`` are assigned on ``self`` by dozens of other
    # classes, so those spellings are receiver-restricted; the containers and
    # latches are unique in the package and stay checked on every receiver.
    assert {"rank", "world_size"} <= ambiguous
    assert "prefix_sequence_states" not in ambiguous
    assert "shared_prefix_credit_applied" not in ambiguous


def test_the_guard_sees_each_kind_of_write(tmp_path):
    """The guard is not vacuous: every kind of write it names is detected."""
    fields = _owner_fields()
    ambiguous = _ambiguous_names()
    sample = tmp_path / "sample.py"
    sample.write_text(
        "def f(self, ctx, other):\n"
        "    self._prefix.shared_prefix_credit_suspended = False\n"
        "    ctx.prefix.shared_prefix_credit_applied = True\n"
        "    ctx.sched.prefix.prefix_sequence_states[3] = object()\n"
        "    self._prefix.active_prefix_materializations.append(1)\n"
        "    del ctx.prefix.prefix_page_digest_cache\n"
        "    prefix_state = self._prefix\n"
        "    prefix_state.rank = 1\n"
        "    other.rank = 1\n"
        "    return ctx.prefix.prefix_sequence_states\n"
    )
    lines = sorted(
        line for line, _ in _offences(_parse(sample), fields, ambiguous)
    )
    # Line 8 writes the ambiguous ``rank`` through the local alias of line 7,
    # which is recognised; line 9 writes ``rank`` on an unrecognised receiver
    # and is not flagged.
    assert lines == [2, 3, 4, 5, 6, 8]
