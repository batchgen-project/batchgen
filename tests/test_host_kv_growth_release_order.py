"""Regression for cross-rank Host-KV release-before-growth ordering.

The host KV pool is one shared-memory region per node, but in the page
boundary each rank releases only its own completed and evicted rows, while
rank 0's growth plan already counts all of those pages as free.  Without a
barrier between the release phases and growth, a fast rank grows against the
pre-release free count and ``AcquirePages`` raises ``Insufficient free
pages`` (observed on a long-decode GPT-OSS batch: 3 free + 451 pending
release, a rank asked for 16 and found 3).  This pins the order
release -> collective barrier -> growth inside ``_page_boundary_fast``, with
the barrier guarded by a predicate built only from the broadcast decisions so
every rank takes the same branch.
"""

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


def _contains_dist_barrier(node):
    return any(
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "dist"
        and call.func.attr == "barrier"
        for call in ast.walk(node)
    )


def test_shared_host_releases_are_synchronized_before_growth():
    method = _page_boundary_method()
    sync_guards = [
        node for node in ast.walk(method)
        if isinstance(node, ast.If)
        and _contains_dist_barrier(node)
        and {
            attr.attr
            for attr in ast.walk(node.test)
            if isinstance(attr, ast.Attribute)
        } >= {"completed_uuids", "host_evicted_uuids"}
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
