"""Load individual ``BatchGenWorker`` methods without importing the worker.

``batchgen.batchgen_worker`` pulls in triton, UCX and the CUDA kernel
extensions at import time, so it cannot be imported on a CPU-only machine.
Several worker methods are nonetheless pure host logic that must be unit
tested, and copying them into a test would mean the test keeps passing after
the real method changes.

This loads the method **from the real source file** instead: the module is
parsed with ``ast``, the requested ``FunctionDef`` nodes are lifted out of the
class body, and each is compiled on its own into a namespace the caller
supplies. The result is a plain function whose first parameter is ``self``, so
a test drives it with any stub object that carries the attributes that method
actually touches.

The test therefore fails if the real method's logic changes, which is the whole
point; it does not re-implement anything.
"""

from __future__ import annotations

import ast
import copy
import pathlib
from typing import Any, Dict, Iterable

WORKER_SOURCE = (
    pathlib.Path(__file__).resolve().parents[1] / "batchgen" / "batchgen_worker.py"
)
WORKER_CLASS = "BatchGenWorker"
PREFILL_ENGINE_SOURCE = (
    pathlib.Path(__file__).resolve().parents[1]
    / "batchgen"
    / "worker"
    / "prefill_engine.py"
)
PREFILL_ENGINE_CLASS = "PrefillEngine"
RANK_SYNC_SOURCE = (
    pathlib.Path(__file__).resolve().parents[1]
    / "batchgen"
    / "worker"
    / "rank_sync_engine.py"
)
RANK_SYNC_CLASS = "RankSyncEngine"
STEP_OBSERVABILITY_SOURCE = (
    pathlib.Path(__file__).resolve().parents[1]
    / "batchgen"
    / "worker"
    / "step_observability_engine.py"
)
STEP_OBSERVABILITY_CLASS = "StepObservabilityEngine"
# The worker-seam engines (S8, S7) whose methods the worker reaches through
# same-name canonical delegators.
DELEGATED_ENGINES = (
    (RANK_SYNC_SOURCE, RANK_SYNC_CLASS),
    (STEP_OBSERVABILITY_SOURCE, STEP_OBSERVABILITY_CLASS),
)


def load_worker_methods(
    names: Iterable[str],
    namespace: Dict[str, Any],
) -> Dict[str, Any]:
    """Compile the named ``BatchGenWorker`` methods as standalone functions.

    Args:
        names: method names to lift out of the class body.
        namespace: globals each compiled method sees. Anything the method
            references at module scope (``torch``, ``logging``, module-level
            flags) must be in here, which also makes those dependencies
            explicit at the call site instead of implicit.

    Returns:
        ``{name: function}``. Each function takes ``self`` first.
    """
    wanted = set(names)
    tree = ast.parse(WORKER_SOURCE.read_text(), filename=str(WORKER_SOURCE))
    classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == WORKER_CLASS
    ]
    if len(classes) != 1:
        raise AssertionError(
            f"expected exactly one class {WORKER_CLASS} in {WORKER_SOURCE}, "
            f"found {len(classes)}"
        )

    loaded: Dict[str, Any] = {}
    for node in classes[0].body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name not in wanted:
            continue
        if node.name in loaded:
            raise AssertionError(
                f"{WORKER_CLASS}.{node.name} is defined more than once; "
                "the test would be pinning an arbitrary one of them"
            )
        module = ast.Module(body=[node], type_ignores=[])
        ast.fix_missing_locations(module)
        scope: Dict[str, Any] = dict(namespace)
        exec(compile(module, str(WORKER_SOURCE), "exec"), scope)  # noqa: S102
        loaded[node.name] = scope[node.name]

    missing = wanted - set(loaded)
    if missing:
        raise AssertionError(
            f"{WORKER_CLASS} has no method(s) {sorted(missing)} in {WORKER_SOURCE}"
        )
    return loaded


def load_worker_functions(
    names: Iterable[str],
    namespace: Dict[str, Any],
) -> Dict[str, Any]:
    """Compile the named module-level functions of the worker source.

    The module-level counterpart of :func:`load_worker_methods`, with the same
    ``namespace`` contract.
    """
    wanted = set(names)
    tree = ast.parse(WORKER_SOURCE.read_text(), filename=str(WORKER_SOURCE))
    loaded: Dict[str, Any] = {}
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or node.name not in wanted:
            continue
        module = ast.Module(body=[node], type_ignores=[])
        ast.fix_missing_locations(module)
        scope: Dict[str, Any] = dict(namespace)
        exec(compile(module, str(WORKER_SOURCE), "exec"), scope)  # noqa: S102
        loaded[node.name] = scope[node.name]

    missing = wanted - set(loaded)
    if missing:
        raise AssertionError(
            f"{WORKER_SOURCE} has no module-level function(s) {sorted(missing)}"
        )
    return loaded


def load_prefill_engine_methods(
    names: Iterable[str],
    namespace: Dict[str, Any],
) -> Dict[str, Any]:
    """Compile S5a ``PrefillEngine`` methods from their real source.

    The engine methods have the explicit S4b-style seam ``(self, worker, ...)``.
    This helper adapts them back to the worker-shaped call used by CPU fakes:
    the engine instance is stateless, so ``None`` is sufficient for its
    pre-rebinding ``self`` and the fake is passed as ``worker``.
    """
    wanted = set(names)
    tree = ast.parse(
        PREFILL_ENGINE_SOURCE.read_text(), filename=str(PREFILL_ENGINE_SOURCE)
    )
    classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == PREFILL_ENGINE_CLASS
    ]
    if len(classes) != 1:
        raise AssertionError(
            f"expected exactly one class {PREFILL_ENGINE_CLASS} in "
            f"{PREFILL_ENGINE_SOURCE}, found {len(classes)}"
        )

    loaded: Dict[str, Any] = {}
    for node in classes[0].body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name not in wanted:
            continue
        module = ast.Module(body=[node], type_ignores=[])
        ast.fix_missing_locations(module)
        scope: Dict[str, Any] = dict(namespace)
        exec(compile(module, str(PREFILL_ENGINE_SOURCE), "exec"), scope)  # noqa: S102
        engine_method = scope[node.name]

        def worker_method(worker, *args, _engine_method=engine_method, **kwargs):
            return _engine_method(None, worker, *args, **kwargs)

        loaded[node.name] = worker_method

    missing = wanted - set(loaded)
    if missing:
        raise AssertionError(
            f"{PREFILL_ENGINE_CLASS} has no method(s) {sorted(missing)} in "
            f"{PREFILL_ENGINE_SOURCE}"
        )
    return loaded


def engine_method_as_worker_method(node: ast.FunctionDef) -> ast.FunctionDef:
    """An engine method ``(self, worker, ...)`` back in its worker shape.

    The S5/S8 engines take the worker as an explicit collaborator and rebind
    ``self = worker`` as the first statement after the docstring, so the moved
    body is statement for statement the old worker method. Dropping exactly
    that parameter and that one statement gives the original method's AST,
    which is what a structural pin written against the worker method reads.
    """
    method = copy.deepcopy(node)
    args = method.args.args
    if len(args) < 2 or args[0].arg != "self" or args[1].arg != "worker":
        raise AssertionError(f"{node.name} is not an engine method (self, worker, ...)")
    del args[1]
    rebinds = [
        index for index, stmt in enumerate(method.body)
        if isinstance(stmt, ast.Assign)
        and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Name)
        and stmt.targets[0].id == "self"
        and isinstance(stmt.value, ast.Name)
        and stmt.value.id == "worker"
    ]
    if len(rebinds) != 1:
        raise AssertionError(f"{node.name}: expected one `self = worker`, found {len(rebinds)}")
    del method.body[rebinds[0]]
    return method


def _class_methods(path: pathlib.Path, class_name: str) -> Dict[str, ast.FunctionDef]:
    tree = ast.parse(path.read_text(), filename=str(path))
    cls = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return {
        node.name: node for node in cls.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def rank_sync_method_names() -> set:
    """The worker methods whose bodies live in ``RankSyncEngine`` (S8)."""
    return set(_class_methods(RANK_SYNC_SOURCE, RANK_SYNC_CLASS))


def step_observability_method_names() -> set:
    """The worker methods whose bodies live in ``StepObservabilityEngine`` (S7)."""
    return set(_class_methods(STEP_OBSERVABILITY_SOURCE, STEP_OBSERVABILITY_CLASS))


def worker_shaped_method_node(name: str) -> ast.FunctionDef:
    """The body a worker method runs, in the worker's own shape.

    For a method S8/S7 moved into ``RankSyncEngine`` / ``StepObservabilityEngine``
    the worker keeps a canonical delegator; the statements that run are the
    engine's, so this returns the engine method converted back by
    :func:`engine_method_as_worker_method`. Every other name resolves to the
    worker class method itself.
    """
    for source, class_name in DELEGATED_ENGINES:
        engine = _class_methods(source, class_name)
        if name in engine:
            return engine_method_as_worker_method(engine[name])
    worker = _class_methods(WORKER_SOURCE, WORKER_CLASS)
    if name not in worker:
        raise AssertionError(f"{WORKER_CLASS} has no method {name}")
    return copy.deepcopy(worker[name])
