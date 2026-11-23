"""Every debug key the code reads is declared, and every declared key is read.

The registry (``batchgen/server_debug_state.py`` plus each model family's
``server_debug_keys.py``) is the only place a key is defined. This file checks the
source tree against it, so a key cannot be added by reading it, a retired key
cannot linger in a reader, and a declared key cannot be dead.
"""
import ast
from pathlib import Path

import pytest

from batchgen.config.model_registry import DEBUG_KEY_REGISTRY

ROOT = Path(__file__).resolve().parents[2]
SCANNED = ("batchgen", "batchgen_kernels")
DECLARATIONS = {
    ROOT / "batchgen" / "server_debug_state.py",
    ROOT / "batchgen" / "models" / "glm" / "glm5" / "server_debug_keys.py",
    ROOT / "batchgen" / "models" / "moonshotai" / "kimi_linear" / "server_debug_keys.py",
}


def _trees():
    for top in SCANNED:
        for path in sorted((ROOT / top).rglob("*.py")):
            if path in DECLARATIONS:
                continue
            yield path, ast.parse(path.read_text(), filename=str(path))


TREES = list(_trees())


def _string_constants():
    found = {}
    for path, tree in TREES:
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                found.setdefault(node.value, set()).add(path.relative_to(ROOT).as_posix())
    return found


STRINGS = _string_constants()


def test_the_declaration_files_exist_and_are_the_registry():
    assert all(path.is_file() for path in DECLARATIONS)
    assert len(DEBUG_KEY_REGISTRY.keys) == 35


@pytest.mark.parametrize("name", sorted(DEBUG_KEY_REGISTRY.keys))
def test_every_declared_key_has_a_reader(name):
    assert STRINGS.get(name), f"{name} is declared but no code names it: remove it or read it"


@pytest.mark.parametrize("name", sorted(DEBUG_KEY_REGISTRY.retired))
def test_no_code_names_a_retired_key(name):
    assert not STRINGS.get(name), (
        f"retired key {name} is still named in {sorted(STRINGS[name])}"
    )


def _debug_get_literals():
    """``<something debug>.get("<literal>")`` reads, with where they are."""
    reads = []
    for path, tree in TREES:
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                continue
            receiver = ast.unparse(node.func.value).lower()
            if "debug" not in receiver or "stats" in receiver:
                continue
            reads.append((
                node.args[0].value,
                f"{path.relative_to(ROOT).as_posix()}:{node.lineno}",
            ))
    return reads


def test_every_key_read_from_a_debug_mapping_is_declared():
    reads = _debug_get_literals()
    assert len(reads) >= 10, "the scan found too few reads to be looking at the right code"
    undeclared = sorted(
        (name, where) for name, where in reads if name not in DEBUG_KEY_REGISTRY.keys
    )
    assert not undeclared, f"keys read but not declared: {undeclared}"


def test_the_request_field_is_named_only_where_it_is_refused():
    # ``batchgen_debug`` as a REQUEST field: the schema and the three admission
    # refusals. (The attention wrappers' class attribute of the same name is
    # the bound server state, not a request field.)
    named = sorted(
        path.relative_to(ROOT).as_posix()
        for path, tree in TREES
        if any(
            isinstance(node, ast.Attribute)
            and node.attr == "batchgen_debug"
            and ast.unparse(node.value) in ("body", "batch")
            for node in ast.walk(tree)
        )
    )
    assert named == [
        "batchgen/server/batch_scheduler.py",
        "batchgen/server/http_server.py",
        "batchgen/server/io_struct.py",
        "batchgen/server/turn_api.py",
    ]
