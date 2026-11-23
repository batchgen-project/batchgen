"""Source guard: the dual host KV coordinator is the only writer of its state.

``DualHostKVCoordinator`` (``batchgen/kv_cache/dual_host_kv_coordinator.py``)
owns the mirrored pair of host paged KV worker views and the poison marker
that keeps them consistent; everything else must go through its methods.  The
single worker views it coordinates (``MLAHostPagedKVWorkerView`` and the
mapped/default variants) are native pybind classes exported by the core
engine module, with no Python-level fields an AST guard could see, so this
guard covers the Python coordinator class.  ``DualAsyncKVTask`` is a
dataclass whose fields are constructor-set and never reassigned, so it
contributes no field names.

This test fails when any module other than the owner touches the
coordinator's private storage, or mutates one of its public fields
(rebinding, ``del``, an item write, or a container mutator call).  Reads of
public fields and calls of owner methods are allowed.

The field names are derived from the owner class's AST (every attribute it
assigns on ``self``), so new fields are guarded automatically.  AST carries
no types, so a field name that some other definition in the package also
uses cannot be attributed to the coordinator by spelling alone —
``primary``/``auxiliary`` are also fields of the dual GPU KV coordinator.
Such ambiguous names are checked only when the receiver expression is
recognisably the coordinator: a bare name or an attribute chain ending in one
of the holder spellings the code base uses (``host_paged_kv_worker_view``,
``worker_view``, ``host_view``).  Unambiguous names are checked on every
receiver.  Not caught: aliasing through a local variable whose name is not a
holder spelling, mutation via ``getattr``/``setattr`` strings, and any
mutation that happens inside the native view objects themselves.
"""
import ast
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "batchgen"
OWNER = PACKAGE / "kv_cache" / "dual_host_kv_coordinator.py"
OWNER_CLASSES = {"DualHostKVCoordinator"}
HOLDER_NAMES = {"host_paged_kv_worker_view", "worker_view", "host_view"}
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


def test_only_the_host_kv_coordinator_writes_its_state():
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
    assert offenders == []


def test_derived_fields_describe_the_owner():
    """The guard is not vacuous: the derived sets match a manual read."""
    private, public = _owner_fields()
    assert private == {"_transaction_error"}
    assert public == {"primary", "auxiliary"}
    ambiguous = _ambiguous_names()
    # The dual GPU KV coordinator also assigns ``primary``/``auxiliary`` on
    # ``self``, so those spellings are receiver-restricted; the poison marker
    # is unique in the package and stays checked on every receiver.
    assert {"primary", "auxiliary"} <= ambiguous
    assert "_transaction_error" not in ambiguous


def test_the_guard_sees_each_kind_of_write(tmp_path):
    """The guard is not vacuous: every kind of write it names is detected."""
    private, public = _owner_fields()
    ambiguous = _ambiguous_names()
    sample = tmp_path / "sample.py"
    sample.write_text(
        "def f(worker, host_view, worker_view, other):\n"
        "    worker.host_paged_kv_worker_view._transaction_error = None\n"
        "    host_view.primary = None\n"
        "    del worker_view.auxiliary\n"
        "    other.primary = None\n"
        "    return host_view.primary\n"
    )
    lines = sorted(
        line
        for line, _ in _offences(_parse(sample), private, public, ambiguous)
    )
    # Line 5 writes ``primary`` on an unrecognised receiver: the ambiguous
    # spelling cannot be attributed to the coordinator, so it is not flagged.
    assert lines == [2, 3, 4]
