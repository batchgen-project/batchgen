"""Regression for cross-rank Host-KV release-before-growth ordering."""

import ast
from pathlib import Path


WORKER = Path(__file__).resolve().parents[1] / "batchgen" / "batchgen_worker.py"


def _page_boundary_method():
    tree = ast.parse(WORKER.read_text())
    return next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_page_boundary_fast"
    )


def _attribute_calls(method, name):
    return [
        node for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == name
    ]


def test_shared_host_releases_are_synchronized_before_growth():
    method = _page_boundary_method()
    sync_guards = [
        node for node in ast.walk(method)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Call)
        and isinstance(node.test.func, ast.Name)
        and node.test.func.id == "requires_host_kv_release_barrier"
        and any(
            isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "dist"
            and call.func.attr == "barrier"
            for call in ast.walk(node)
            if isinstance(call, ast.Call)
        )
    ]
    assert len(sync_guards) == 1

    sync_line = sync_guards[0].lineno
    completed_releases = _attribute_calls(
        method, "_release_host_kv_pages_for_batch"
    )
    evicted_releases = _attribute_calls(method, "release_sequence_pages")
    growth_calls = _attribute_calls(method, "grow_pages_for_sequences")

    assert completed_releases and evicted_releases and growth_calls
    assert max(call.lineno for call in completed_releases) < sync_line
    assert max(call.lineno for call in evicted_releases) < sync_line
    assert sync_line < min(call.lineno for call in growth_calls)
