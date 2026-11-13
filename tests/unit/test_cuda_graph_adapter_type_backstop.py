"""The worker rejects a non-ModelCudaGraphAdapter adapter once, at fetch time.

Every model's graph behavior must come through the ``ModelCudaGraphAdapter``
ABC, and the fetch goes through
``getattr(initializer, "get_cuda_graph_adapter")``. ``getattr`` is
structural: an initializer could return any object at all, and a duck-typed
near-miss would survive every call site that touches the adapter and only
blow up mid-decode on the first hook it does not implement. So the type is
asserted once, on the fetch.

S2 moved the fetch, the handle and this backstop out of the worker into
``GraphRuntime.adopt_adapter`` (``batchgen/worker/graph_runtime.py``), which
the worker calls once from ``_initialize_core_components``. The checks below
follow it there unchanged; the startup-only check additionally pins the
worker's single call site.

Three things are checked here:

* that the check still sits in the fetch's ``is not None`` branch, raising a
  RuntimeError that names the model and the ABC -- asserted against the
  AST, because neither module can be imported on a machine without the
  native engine extensions;
* that the ABC actually rejects a duck type at runtime, which is what makes
  the isinstance check worth having. ``batchgen.cuda_graph.adapter`` is
  import-light enough for that (torch only), so it is imported for real;
* that every adapter the tree already ships would pass it.
"""

from __future__ import annotations

import ast
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
WORKER = _ROOT / "batchgen" / "batchgen_worker.py"
GRAPH_RUNTIME = _ROOT / "batchgen" / "worker" / "graph_runtime.py"
WORKER_TREE = ast.parse(WORKER.read_text())
TREE = ast.parse(GRAPH_RUNTIME.read_text())

FETCH = "get_cuda_graph_adapter"
ABC_NAME = "ModelCudaGraphAdapter"
ADOPT = "adopt_adapter"


def _method(name: str, tree: ast.AST = None) -> ast.FunctionDef:
    return next(
        node
        for node in ast.walk(tree if tree is not None else TREE)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _guard_branch() -> ast.If:
    """The ``if self.cuda_graph_adapter is not None:`` branch of the fetch."""
    branches = [
        node
        for node in ast.walk(_method(ADOPT))
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and ast.unparse(node.test) == "self.cuda_graph_adapter is not None"
    ]
    assert len(branches) == 1, [ast.unparse(b.test) for b in branches]
    return branches[0]


def test_the_adapter_is_fetched_exactly_once():
    """One fetch site means one place the type can be asserted."""

    fetches = [
        node
        for node in ast.walk(TREE)
        if isinstance(node, ast.Constant) and node.value == FETCH
    ]
    assert len(fetches) == 1, len(fetches)

    # And the worker no longer fetches at all: it delegates, so there is no
    # second site that could skip the check.
    worker_fetches = [
        node
        for node in ast.walk(WORKER_TREE)
        if isinstance(node, ast.Constant) and node.value == FETCH
    ]
    assert worker_fetches == [], len(worker_fetches)


def test_the_fetched_adapter_is_isinstance_checked_against_the_abc():
    checks = [
        node
        for node in ast.walk(_guard_branch())
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "isinstance"
        and ast.unparse(node) == f"isinstance(self.cuda_graph_adapter, {ABC_NAME})"
    ]
    assert len(checks) == 1, [ast.unparse(c) for c in checks]


def test_a_failed_check_raises_runtimeerror_naming_the_model_and_the_abc():
    raises = [
        node
        for node in ast.walk(_guard_branch())
        if isinstance(node, ast.Raise)
        and isinstance(node.exc, ast.Call)
        and isinstance(node.exc.func, ast.Name)
        and node.exc.func.id == "RuntimeError"
    ]
    assert len(raises) == 1, [ast.unparse(r) for r in raises]

    message = ast.unparse(raises[0])
    assert "model_name" in message, message
    assert "cuda_graph" in message, message  # the import path to subclass
    assert ABC_NAME in message, message

    # The raise is the negative arm of the isinstance check, not a sibling
    # statement that could fire on a legitimate adapter.
    negative = [
        node
        for node in ast.walk(_guard_branch())
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.UnaryOp)
        and isinstance(node.test.op, ast.Not)
        and ast.unparse(node.test)
        == f"not isinstance(self.cuda_graph_adapter, {ABC_NAME})"
    ]
    assert len(negative) == 1, [ast.unparse(n.test) for n in negative]
    assert any(
        isinstance(stmt, ast.Raise) for stmt in ast.walk(negative[0])
    ), ast.unparse(negative[0])


def test_the_check_runs_at_startup_and_never_per_step():
    """``_initialize_core_components`` is startup; the decode loop is not."""

    enclosing = [
        node.name
        for node in ast.walk(TREE)
        if isinstance(node, ast.FunctionDef)
        and any(
            isinstance(inner, ast.Call)
            and isinstance(inner.func, ast.Name)
            and inner.func.id == "isinstance"
            and ast.unparse(inner)
            == f"isinstance(self.cuda_graph_adapter, {ABC_NAME})"
            for inner in ast.walk(node)
        )
    ]
    assert enclosing == [ADOPT], enclosing

    # The worker reaches it from exactly one method, and that method is
    # startup -- so the check cannot move onto a per-step path.
    callers = [
        node.name
        for node in ast.walk(WORKER_TREE)
        if isinstance(node, ast.FunctionDef)
        and any(
            isinstance(inner, ast.Call)
            and isinstance(inner.func, ast.Attribute)
            and ast.unparse(inner.func) == f"self._graph.{ADOPT}"
            for inner in ast.walk(node)
        )
    ]
    assert callers == ["_initialize_core_components"], callers


def test_the_abc_rejects_a_duck_typed_adapter():
    """Why the isinstance check is load-bearing rather than decorative."""

    from batchgen.cuda_graph.adapter import GraphMode, ModelCudaGraphAdapter

    class LooksRight:
        """Every name the worker calls on a live adapter, none of the ABC."""

        def advertised_modes(self):
            return [GraphMode.EAGER]

        def is_supported(self, *args, **kwargs):
            return True

    assert not isinstance(LooksRight(), ModelCudaGraphAdapter)

    # A real subclass passes. The stubs are generated from the ABC's own
    # abstract set so this stays true as the method list grows -- that list
    # itself is frozen by tests/contracts/test_pin_cuda_graph.py.
    abstract = ModelCudaGraphAdapter.__abstractmethods__
    assert abstract, "the ABC has no abstract methods: a duck type would pass"
    real = type(
        "RealAdapter",
        (ModelCudaGraphAdapter,),
        {name: (lambda self, *a, **k: None) for name in abstract},
    )
    assert isinstance(real(), ModelCudaGraphAdapter)


def test_every_shipped_initializer_returns_a_real_subclass():
    """The backstop must not reject the adapters the tree already ships.

    Each ``get_cuda_graph_adapter`` implementation under ``batchgen/models/``
    is read for the class it constructs and returns; that class's own
    definition must name the ABC as a base. AST again, because importing a
    model package needs the native extensions.
    """

    models = _ROOT / "batchgen" / "models"
    trees = {
        path: ast.parse(path.read_text())
        for path in sorted(models.rglob("*.py"))
    }

    # class name -> True when its definition subclasses the ABC
    subclasses = {
        node.name: any(
            isinstance(base, ast.Name) and base.id == ABC_NAME
            for base in node.bases
        )
        for tree in trees.values()
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
    }

    returned = []
    for path, tree in trees.items():
        for node in ast.walk(tree):
            if not (isinstance(node, ast.FunctionDef) and node.name == FETCH):
                continue
            for inner in ast.walk(node):
                if (
                    isinstance(inner, ast.Return)
                    and isinstance(inner.value, ast.Call)
                    and isinstance(inner.value.func, ast.Name)
                ):
                    returned.append((path.name, inner.value.func.id))

    assert returned, "no model initializer implements %s" % FETCH
    bad = [
        "%s returns %s" % (where, cls)
        for where, cls in returned
        if not subclasses.get(cls)
    ]
    assert not bad, (
        "these shipped adapters would be rejected by the worker's isinstance "
        "backstop: %r" % bad
    )
