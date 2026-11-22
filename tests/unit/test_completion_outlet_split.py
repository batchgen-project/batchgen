"""Completion split: duties of `_report_completion` stay in their layer.

The split keeps the original statement order but routes each duty to its
owner: the worker reads the fallback text and sends the result (the
`_completion_outlet_read` / `_completion_outlet_send` pair), the KV capacity
engine frees the trajectory slot and the local slot/index, and the two
worker-side statements that talk to the response queue or observability
(`_publish_trajectory_pool_capacity`, `_maybe_emit_prefix_drain_metric`)
remain worker calls at their original positions.

Both checks are source-level (AST), derived from the real files, so they fail
when the real code regresses rather than pinning a copy.
"""

import ast
import pathlib

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_WORKER = _ROOT / "batchgen" / "batchgen_worker.py"
_ENGINE = _ROOT / "batchgen" / "worker" / "kv_capacity_engine.py"

# The engine methods this split introduced.
_ENGINE_METHODS = (
    "_release_completed_trajectory_slot",
    "_release_completed_slot_index",
)


def _class_method(path: pathlib.Path, class_name: str, name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(), filename=str(path))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return next(
        node
        for node in cls.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    )


def test_engine_release_methods_never_touch_the_output_path():
    """The output path (queue, tokenizer) must not leak into the engine.

    The engine owns pure resource release; reading tokens, decoding text and
    putting the result belong to the worker outlet pair. Checked over the
    engine methods this split introduced: no reference to `_response_queue`,
    `tokenizer`, `_decode_tokens_to_string`, and no `.put(...)` call.
    """
    banned_attrs = {"_response_queue", "tokenizer", "_decode_tokens_to_string"}
    for name in _ENGINE_METHODS:
        method = _class_method(_ENGINE, "KVCapacityEngine", name)
        for node in ast.walk(method):
            if isinstance(node, ast.Attribute):
                assert node.attr not in banned_attrs, (name, node.attr)
            if isinstance(node, ast.Name):
                assert node.id not in banned_attrs, (name, node.id)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert node.func.attr != "put", name


def test_report_completion_runs_its_duties_in_the_original_order():
    """read -> trajectory slot release -> capacity publish -> slot/index
    release -> prefix drain metric -> send, derived from the real source."""
    method = _class_method(_WORKER, "BatchGenWorker", "_report_completion")
    calls = [
        node.func.attr
        for node in sorted(
            (
                node
                for node in ast.walk(method)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "self"
            ),
            key=lambda node: (node.lineno, node.col_offset),
        )
    ]
    assert calls == [
        "_completion_outlet_read",
        "_release_completed_trajectory_slot",
        "_publish_trajectory_pool_capacity",
        "_release_completed_slot_index",
        "_maybe_emit_prefix_drain_metric",
        "_completion_outlet_send",
    ]
    # The unknown-uuid early return sits between the read and the releases,
    # so a missing row releases nothing and sends nothing.
    body = method.body[1:] if isinstance(method.body[0], ast.Expr) else method.body
    guard_index = next(
        index
        for index, statement in enumerate(body)
        if isinstance(statement, ast.If)
        and any(isinstance(node, ast.Return) for node in ast.walk(statement))
    )
    read_index = next(
        index
        for index, statement in enumerate(body)
        if any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_completion_outlet_read"
            for node in ast.walk(statement)
        )
    )
    release_index = next(
        index
        for index, statement in enumerate(body)
        if any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_release_completed_trajectory_slot"
            for node in ast.walk(statement)
        )
    )
    assert read_index < guard_index < release_index
