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
import pathlib
from typing import Any, Dict, Iterable

WORKER_SOURCE = (
    pathlib.Path(__file__).resolve().parents[1] / "batchgen" / "batchgen_worker.py"
)
WORKER_CLASS = "BatchGenWorker"


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
