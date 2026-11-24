"""Source guard: the query book is mutated directly only where pinned.

The worker's ``query_book`` dict (local slot -> ``QueryBookEntry``) is owned
by ``batchgen/query_book.py``: its binding/release helpers
(``bind_query_book_slot``, ``release_query_book_slot``,
``bind_local_sequence_to_query_book``, ``release_local_query_slot``,
``make_query_book_entry``) and the ``QueryBook`` classes are the intended
write path, so every mutation inside that module is an owner write.  This
test pins TODAY's exact set of direct mutations outside the owner module as a
list that can only shrink: it fails when a NEW outside mutation appears, and
it fails when a listed offender no longer exists (stale entry), so every
removal is deliberate.

Covered patterns, on any receiver whose final spelling is ``query_book``
(``self.query_book``, ``req.query_book``, a bare local/parameter named
``query_book`` — the dict is passed into engine helpers under that name):
subscript assignment and ``del``, container-mutator calls (``pop``,
``update``, ``clear``, ``setdefault``, ...), and attribute rebinding/``del``
of ``.query_book``.  Rebinding a BARE name is a local rebind, not a mutation
of the worker's dict, and is not flagged.  NOT caught: mutation through an
alias with another spelling (``book = self.query_book; book[i] = e`` — no
such alias exists in the tree today), ``getattr``/``setattr`` with dynamic
names, and mutation after the dict is passed to a parameter with another
name.  ``_pending_migrated_query_book`` is a different dict and is out of
scope here.
"""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "batchgen"
OWNER = PACKAGE / "query_book.py"
TARGET = "query_book"
MUTATORS = {
    "clear", "pop", "popitem", "setdefault", "update",
}


def _is_target(expr: ast.AST) -> bool:
    return (
        isinstance(expr, ast.Attribute) and expr.attr == TARGET
    ) or (
        isinstance(expr, ast.Name) and expr.id == TARGET
    )


def _offences(path: Path):
    tree = ast.parse(path.read_text(), filename=str(path))
    found = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and node.attr == TARGET
            and isinstance(node.ctx, (ast.Store, ast.Del))
        ):
            found.append((node.lineno, f"rebinding .{TARGET}"))
        elif isinstance(node, ast.Subscript) and isinstance(
            node.ctx, (ast.Store, ast.Del)
        ):
            if _is_target(node.value):
                found.append((node.lineno, f"item write on {TARGET}"))
        elif isinstance(node, ast.Call):
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr in MUTATORS
                and _is_target(func.value)
            ):
                found.append((node.lineno, f"{TARGET}.{func.attr}("))
    return found


# The exact outside mutations that exist today (one entry per file and kind
# of mutation, no line numbers so the list does not churn).  Entries may only
# be removed (by moving the mutation behind the query_book helpers), never
# added.
KNOWN_OFFENDERS = [
    "batchgen/batchgen_worker.py: rebinding .query_book",
    "batchgen/worker/kv_capacity_engine.py: item write on query_book",
    "batchgen/worker/kv_capacity_engine.py: query_book.pop(",
    "batchgen/worker/prefill_engine.py: item write on query_book",
    "batchgen/worker/worker_lifecycle_engine.py: rebinding .query_book",
]


def test_the_query_book_is_mutated_only_where_pinned():
    offenders = sorted(
        {
            f"{path.relative_to(ROOT)}: {what}"
            for path in sorted(PACKAGE.rglob("*.py"))
            if path != OWNER
            for _line, what in _offences(path)
        }
    )
    assert offenders == KNOWN_OFFENDERS


def test_the_guard_sees_each_kind_of_write(tmp_path):
    """The guard is not vacuous: every kind of mutation it names is detected."""
    sample = tmp_path / "sample.py"
    sample.write_text(
        "def f(self, query_book, entry):\n"
        "    self.query_book = {}\n"
        "    self.query_book[0] = entry\n"
        "    del query_book[1]\n"
        "    query_book.pop(2, None)\n"
        "    self.query_book.update({})\n"
        "    query_book = {}\n"
        "    self._pending_migrated_query_book[3] = entry\n"
        "    return query_book.get(0)\n"
    )
    lines = sorted(line for line, _ in _offences(sample))
    # Line 7 rebinds a bare local name (no state mutation) and line 8 touches
    # a different dict; neither is flagged.
    assert lines == [2, 3, 4, 5, 6]
