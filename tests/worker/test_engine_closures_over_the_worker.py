"""Source guard: closures in an engine file that capture the worker handed to it.

The third and last route by which the worker reaches another object, and the one
criterion P3 names without testing.  Its two siblings beside this file cover the
worker handing out a bound method of itself, and a closure defined inside the
worker file.  This one covers the mirror image: an engine operation **receives**
the worker, defines a function inside itself, and that function refers to the
received worker -- so the worker travels wherever the function goes.

A capture counts when, in a file other than ``batchgen/batchgen_worker.py``, a
function whose own parameters include ``worker`` (or which opens by rebinding
``self = worker``, the spelling the pre-seam engines use so a moved body reads
unchanged) contains a nested function or lambda that refers to that name without
taking it as its own parameter.

Each entry carries where the closure goes, for the same reason the sibling guard
does: a closure handed to a sort key and one handed to a constructor are
different facts, and a list that recorded only "a capture exists here" would not
notice the change.

**This list is expected to empty out, not to sit still.** Every entry exists
because its enclosing operation still receives the worker. When that operation
is converted to take declared inputs instead, ``self`` becomes the engine, the
capture disappears, and the entry must be removed -- which the exactness half
below forces. So this guard doubles as a progress measure for the decoupling
work, read from the source rather than from a counter kept outside the
repository.

NOT caught, stated so the claim is not read as wider than it is: a closure over
a *bound method* taken from the received worker (``f = worker.method`` then a
closure over ``f``), a closure built in a third file from a worker handed on
twice, and ``functools.partial`` over the received worker. None exists today.

Measured on ``8ef48633``: 11 captures across 3 engine files.
"""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "batchgen"
WORKER_FILE = PACKAGE / "batchgen_worker.py"

# (file, enclosing operation, closure, destination) -- exact, only shrinks.
ALLOWED = {
    ("decode_engine.py", "run", "<lambda>", "local"),
    ("prefill_engine.py", "_new_pool_epilogue_tracer", "<lambda>", "run_probe"),
    ("prefill_engine.py", "_new_pool_kernel_dump_recorder", "<lambda>", "run_probe"),
    ("prefill_engine.py", "_new_pool_offload_check_recorder", "<lambda>", "run_probe"),
    ("prefill_engine.py", "_prepare_prefill_batch", "<lambda>", "evicted_uuids.sort"),
    ("prefill_engine.py", "_prepare_prefill_batch", "<lambda>", "queueing_uuids.sort"),
    ("prefill_engine.py", "_record_unsharded_epilogue_gather", "<lambda>", "run_probe"),
    ("prefill_engine.py", "prefill_prepacked", "<lambda>", "run_probe"),
    ("prefill_engine.py", "prefill_prepacked", "_memprof_probe", "_engine._prefill_model_pass"),
    ("replica_transport_engine.py", "_install_replica_transport", "_decline", "local"),
    ("replica_transport_engine.py", "_install_replica_transport", "_mark", "local"),
}


def _worker_names(fn) -> set:
    names = {
        arg.arg
        for arg in fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs
        if arg.arg in ("worker", "_worker")
    }
    if "worker" in {
        arg.arg for arg in fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs
    }:
        for statement in fn.body[:3]:
            if (
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
                and statement.targets[0].id == "self"
                and isinstance(statement.value, ast.Name)
                and statement.value.id == "worker"
            ):
                names.add("self")
    return names


def _destination(parents, outer, inner, name) -> str:
    cursor = inner
    while cursor is not None and cursor is not outer:
        parent = parents.get(cursor)
        if isinstance(parent, ast.Return):
            return "returned"
        if isinstance(parent, ast.Call) and cursor is not parent.func:
            return ast.unparse(parent.func)
        if isinstance(parent, ast.keyword):
            grand = parents.get(parent)
            if isinstance(grand, ast.Call):
                return ast.unparse(grand.func)
        cursor = parent
    if name != "<lambda>":
        for node in ast.walk(outer):
            if isinstance(node, ast.Call):
                for arg in list(node.args) + [kw.value for kw in node.keywords]:
                    if isinstance(arg, ast.Name) and arg.id == name:
                        return ast.unparse(node.func)
    return "local"


def _captures():
    found = set()
    for path in sorted(PACKAGE.rglob("*.py")):
        if path == WORKER_FILE:
            continue
        try:
            tree = ast.parse(path.read_text(), filename=str(path))
        except SyntaxError:  # pragma: no cover - a parse failure is its own bug
            continue
        parents = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                parents[child] = node
        for outer in [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]:
            holders = _worker_names(outer)
            if not holders:
                continue
            for inner in ast.walk(outer):
                if inner is outer:
                    continue
                if not isinstance(
                    inner, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
                ):
                    continue
                own = {
                    arg.arg
                    for arg in inner.args.posonlyargs
                    + inner.args.args
                    + inner.args.kwonlyargs
                }
                if not any(
                    holder not in own
                    and any(
                        isinstance(x, ast.Name) and x.id == holder
                        for x in ast.walk(inner)
                    )
                    for holder in holders
                ):
                    continue
                name = getattr(inner, "name", "<lambda>")
                found.add(
                    (
                        path.name,
                        outer.name,
                        name,
                        _destination(parents, outer, inner, name),
                    )
                )
    return found


def test_no_unlisted_engine_closure_captures_the_worker():
    new = sorted(_captures() - ALLOWED)
    assert not new, "\n  ".join(
        [
            "an engine closure captures the worker it was handed and is not on "
            "the only-shrinks list (or an existing one changed destination); "
            "give the closure the values it needs, or convert the enclosing "
            "operation to declared inputs:"
        ]
        + [f"{f}: {outer} -> {name} ({dest})" for f, outer, name, dest in new]
    )


def test_the_list_is_not_padded_with_captures_that_are_already_gone():
    """The exactness half, which is also the progress measure.

    An entry disappears when its enclosing operation stops receiving the
    worker. Leaving the entry behind would hide both a regression and the
    progress, so a stale one fails here and has to be deleted.
    """
    stale = sorted(ALLOWED - _captures())
    assert not stale, "\n  ".join(
        [
            "these listed captures are gone, or no longer go where the list "
            "says -- delete the entry, which is what converting the enclosing "
            "operation is supposed to achieve:"
        ]
        + [f"{f}: {outer} -> {name} ({dest})" for f, outer, name, dest in stale]
    )


def test_every_capture_lives_in_an_operation_that_still_receives_the_worker():
    """Guards the guard: the premise that makes the list finite and shrinking.

    If an entry survived in an operation that no longer takes the worker, the
    scan would be matching something else -- a local named ``self`` on an
    engine, say -- and the list would stop meaning what it says.
    """
    wrong = []
    for file_name, outer_name, closure, _dest in sorted(ALLOWED):
        matches = [
            path for path in PACKAGE.rglob(file_name) if path != WORKER_FILE
        ]
        assert matches, f"{file_name} is gone; delete its entries"
        tree = ast.parse(matches[0].read_text(), filename=str(matches[0]))
        operations = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == outer_name
        ]
        if not operations:
            wrong.append(f"{file_name}: {outer_name} no longer exists")
            continue
        if not any(_worker_names(op) for op in operations):
            wrong.append(
                f"{file_name}: {outer_name} no longer receives the worker, so "
                f"its entry for {closure} must go"
            )
    assert not wrong, "\n  ".join(
        ["the list's premise no longer holds:"] + wrong
    )
