"""Source guard: the sequence table is the only writer of the uuid/local maps.

The worker and the engines read the maps through read-only views
(``seq_table.uuid_to_local`` and friends, or the worker's same-named
properties). This test fails when any module other than
``batchgen/worker/sequence_table.py`` reaches the table's private storage or
mutates one of the views' names.
"""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "batchgen"
OWNER = PACKAGE / "worker" / "sequence_table.py"

PRIVATE_STORAGE = {
    "_local_to_uuid", "_uuid_to_local",
    "_local_to_uuid_view", "_uuid_to_local_view", "_free_view",
}
VIEW_NAMES = {
    "local_to_uuid", "uuid_to_local", "free_local_indices",
    "_local_to_uuid_map", "_uuid_to_local_map", "_free_local_indices",
}
MUTATORS = {
    "pop", "popitem", "update", "clear", "setdefault",
    "add", "discard", "remove", "difference_update", "intersection_update",
}


def _is_view(node: ast.AST) -> bool:
    return isinstance(node, ast.Attribute) and node.attr in VIEW_NAMES


def _offences(path: Path):
    tree = ast.parse(path.read_text(), filename=str(path))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in PRIVATE_STORAGE:
            found.append((node.lineno, f"private storage .{node.attr}"))
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.ctx, (ast.Store, ast.Del))
            and _is_view(node.value)
        ):
            found.append((node.lineno, f"item write on .{node.value.attr}"))
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.ctx, (ast.Store, ast.Del))
            and node.attr in VIEW_NAMES
        ):
            found.append((node.lineno, f"rebinding .{node.attr}"))
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in MUTATORS
            and _is_view(node.func.value)
        ):
            found.append((node.lineno, f".{node.func.value.attr}.{node.func.attr}("))
    return found


def test_only_the_sequence_table_writes_the_uuid_local_maps():
    offenders = [
        f"{path.relative_to(ROOT)}:{line} {what}"
        for path in sorted(PACKAGE.rglob("*.py"))
        if path != OWNER
        for line, what in _offences(path)
    ]
    assert offenders == []


def test_the_guard_sees_each_kind_of_write(tmp_path):
    """The guard is not vacuous: every kind of write it names is detected."""
    sample = tmp_path / "sample.py"
    sample.write_text(
        "def f(worker, table):\n"
        "    table._uuid_to_local['a'] = 1\n"
        "    worker._uuid_to_local_map['a'] = 1\n"
        "    del worker._local_to_uuid_map[0]\n"
        "    worker._free_local_indices = set()\n"
        "    table.free_local_indices.add(3)\n"
        "    table.uuid_to_local.pop('a')\n"
        "    return worker._uuid_to_local_map.get('a')\n"
    )
    assert sorted(line for line, _ in _offences(sample)) == [2, 3, 4, 5, 6, 7]
