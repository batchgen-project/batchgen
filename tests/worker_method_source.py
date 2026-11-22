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
from typing import Any, Dict, Iterable, Optional

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
WORKER_LIFECYCLE_SOURCE = (
    pathlib.Path(__file__).resolve().parents[1]
    / "batchgen"
    / "worker"
    / "worker_lifecycle_engine.py"
)
WORKER_LIFECYCLE_CLASS = "WorkerLifecycleEngine"
KV_CAPACITY_SOURCE = (
    pathlib.Path(__file__).resolve().parents[1]
    / "batchgen"
    / "worker"
    / "kv_capacity_engine.py"
)
KV_CAPACITY_CLASS = "KVCapacityEngine"
GRAPH_RUNTIME_SOURCE = (
    pathlib.Path(__file__).resolve().parents[1]
    / "batchgen"
    / "worker"
    / "graph_runtime.py"
)
GRAPH_RUNTIME_CLASS = "GraphRuntime"
# The worker-seam engines (S8, S7, S9, S10a, repatriation) whose methods the
# worker reaches through same-name canonical delegators. Only an engine method
# in the delegated shape ``(self, worker, ...)`` is a delegator; engines that
# also carry their own state API (GraphRuntime) keep those methods out of the
# delegation scan via that shape filter.
DELEGATED_ENGINES = (
    (RANK_SYNC_SOURCE, RANK_SYNC_CLASS),
    (STEP_OBSERVABILITY_SOURCE, STEP_OBSERVABILITY_CLASS),
    (WORKER_LIFECYCLE_SOURCE, WORKER_LIFECYCLE_CLASS),
    (KV_CAPACITY_SOURCE, KV_CAPACITY_CLASS),
    (PREFILL_ENGINE_SOURCE, PREFILL_ENGINE_CLASS),
    (GRAPH_RUNTIME_SOURCE, GRAPH_RUNTIME_CLASS),
)


def _is_delegator_shape(node: ast.FunctionDef) -> bool:
    args = node.args.args
    return (
        len(args) >= 2
        and args[0].arg == "self"
        and args[1].arg in {"worker", "ctx"}
    )


def _context_attr_worker_expr(node: ast.AST) -> Optional[ast.AST]:
    """Map a converted engine context path back to the worker spelling."""
    text = ast.unparse(node)
    paths = {
        "ctx.facts.rank": "self.rank",
        "ctx.facts.world_size": "self.world_size",
        "ctx.facts.local_rank": "self.local_rank",
        "ctx.facts.local_world_size": "self.local_world_size",
        "ctx.facts.rank_ctx": "self.rank_ctx",
        "ctx.facts.device": "self.torch_device",
        "ctx.facts.page_size": "self.PAGE_SIZE",
        "ctx.facts.decision_interval": "self.DECISION_INTERVAL",
        "ctx.facts.enable_prefix_cache": "self.enable_prefix_cache",
        "ctx.facts.enable_host_kv_eviction": "self.enable_host_kv_eviction",
        "ctx.facts.host_kv_chunk_size": "self.host_kv_chunk_size",
        "ctx.sched.pool": "self.global_batch",
        "ctx.sched.seq_table.uuid_to_local": "self._uuid_to_local_map",
        "ctx.sched.seq_table.local_to_uuid": "self._local_to_uuid_map",
        "ctx.sched.trajectories": "self._traj",
        "ctx.gpu_kv": "self.gpu_paged_kv_cache_manager",
        "ctx.host_view": "getattr(self, 'host_paged_kv_worker_view', None)",
        "ctx.core_engine": "self.core_engine",
        "ctx.gpu_resident": "self._sequences_with_gpu_kv",
        "ctx.prefix_coordinator": "self.prefix_cache_coordinator",
        "ctx.prefix_runtime": "self.prefix_cache_runtime_config",
        "ctx.chunk_sizer": "self.adaptive_chunk_sizer",
        "ctx.facts.prefix_cache_debug_stats": "self.prefix_cache_debug_stats",
        "ctx.parallel_manager": "self.parallel_manager",
    }
    if text == "ctx.facts.attn_tp_size":
        return ast.parse("self._decode_attn_tp_size()", mode="eval").body
    replacement = paths.get(text)
    if replacement is None:
        return None
    return ast.parse(replacement, mode="eval").body


class _ContextToWorker(ast.NodeTransformer):
    def visit_Attribute(self, node):
        node = self.generic_visit(node)
        return _context_attr_worker_expr(node) or node

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            replacements = {
                "max_rank_bsz": "getattr(self, '_decode_padding_bsz', 0) or 0",
                "pending_tasks": "self._pending_kv_append_tasks",
                "pending_tensors": "getattr(self, '_pending_kv_append_tensors', None)",
                "deferred_errors": "getattr(self, '_deferred_kv_append_wait_errors', [])",
                "decode_local_count_tensor": "self._decode_local_count_tensor",
                "decode_all_rank_counts": "self._decode_all_rank_counts",
                "previous_max": "getattr(self, '_current_decode_max_rank_batch_size', 0) or 0",
            }
            replacement = replacements.get(node.id)
            if replacement is not None:
                return ast.parse(replacement, mode="eval").body
        return node

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "IndexManager"
            and len(node.args) == 2
            and isinstance(node.args[0], ast.Call)
            and isinstance(node.args[0].func, ast.Name)
            and node.args[0].func.id == "IndexLookupRequest"
        ):
            # The engine builds the lookup request from the context; the worker
            # spelling of the same lookup is its index helper of that name.
            return ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="self", ctx=ast.Load()),
                    attr="_" + node.func.attr,
                    ctx=ast.Load(),
                ),
                args=[node.args[1]],
                keywords=[],
            )
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "self"
            and node.args
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == "ctx"
        ):
            node.args.pop(0)
        return node


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
    """Convert an engine method ``(self, worker/ctx, ...)`` to worker shape.

    Older seam methods take the worker as an explicit collaborator and rebind
    ``self = worker``. AS3.1 methods take a frozen context, so their context
    paths are mapped back to the worker spelling for source-pinned tests.
    """
    method = copy.deepcopy(node)
    args = method.args.args
    if len(args) < 2 or args[0].arg != "self" or args[1].arg not in {"worker", "ctx"}:
        raise AssertionError(f"{node.name} is not an engine method (self, worker/ctx, ...)")
    if args[1].arg == "ctx":
        del args[1]
        if node.name == "_make_boundary_decision_request":
            del args[-1]
        method.body = _ContextToWorker().visit(method).body
        worker_method = _class_methods(WORKER_SOURCE, WORKER_CLASS).get(node.name)
        if worker_method is not None:
            method.args = copy.deepcopy(worker_method.args)
        return method
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

    For a method moved into one of the worker seam engines, the worker keeps a
    canonical delegator. The statements that run are the engine's, so this
    returns the engine method converted back by
    :func:`engine_method_as_worker_method`. Every other name resolves to the
    worker class method itself.
    """
    for source, class_name in DELEGATED_ENGINES:
        engine = _class_methods(source, class_name)
        if name in engine and _is_delegator_shape(engine[name]):
            return engine_method_as_worker_method(engine[name])
    worker = _class_methods(WORKER_SOURCE, WORKER_CLASS)
    if name not in worker:
        raise AssertionError(f"{WORKER_CLASS} has no method {name}")
    return copy.deepcopy(worker[name])


def delegated_method_source(name: str) -> str:
    """The raw source text of a delegated method's engine body.

    Text pins that used to slice the worker file keep their assertions
    byte-identical by reading the engine's own source segment (located by
    AST, so no indentation or delimiter assumptions). The segment is the
    engine method as written, including its seam parameter.
    """
    for source, class_name in DELEGATED_ENGINES:
        engine = _class_methods(source, class_name)
        if name in engine and _is_delegator_shape(engine[name]):
            return ast.get_source_segment(source.read_text(), engine[name])
    raise KeyError(f"{name} is not a delegated engine method")


def delegated_method_names() -> set:
    """Every worker method whose body lives in a delegated engine (S8/S7/S9/S10a)."""
    names: set = set()
    for source, class_name in DELEGATED_ENGINES:
        names |= {
            name for name, node in _class_methods(source, class_name).items()
            if _is_delegator_shape(node)
        }
    return names


def load_worker_shaped_methods(
    names: Iterable[str],
    namespace: Dict[str, Any],
) -> Dict[str, Any]:
    """:func:`load_worker_methods` for the bodies that actually run.

    A delegated method compiles from its engine body in the worker's shape,
    so the caller's ``namespace`` (stub ``torch``, ``sys``, ...) reaches the
    moved statements exactly as it reached them on the worker; every other
    name compiles from the worker class as before.
    """
    loaded: Dict[str, Any] = {}
    for name in names:
        node = worker_shaped_method_node(name)
        module = ast.Module(body=[node], type_ignores=[])
        ast.fix_missing_locations(module)
        scope: Dict[str, Any] = dict(namespace)
        exec(compile(module, str(WORKER_SOURCE), "exec"), scope)  # noqa: S102
        loaded[name] = scope[name]
    return loaded
