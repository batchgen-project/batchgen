"""A decode step on a rank with no rows must not read names bound only for rows.

``DecoupledGpuDecode.run`` runs on every rank, including a DP rank whose
decode batch is empty (its MoE collectives must still execute). Since the
#717 engine extraction the step passes the context-length metadata to the
binder on both paths, so every name the step reads after an ``if batch:``
block must be bound on the empty path too (2026-11-16: GLM-5.3 with fewer
requests than DP ranks died with UnboundLocalError on the empty ranks).
"""

from __future__ import annotations

import ast
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[2] / "batchgen" / "worker" / "decode_engine.py"


def _stored(nodes) -> set[str]:
    return {n.id for stmt in nodes for n in ast.walk(stmt)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}


def _run_function() -> ast.FunctionDef:
    tree = ast.parse(SOURCE.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "DecoupledGpuDecode":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "run":
                    return item
    raise AssertionError("DecoupledGpuDecode.run not found")


def test_names_bound_only_for_rows_are_not_read_after_the_branch():
    run = _run_function()
    problems = []
    for parent in ast.walk(run):
        body = getattr(parent, "body", None)
        if not isinstance(body, list):
            continue
        for index, stmt in enumerate(body):
            if not (isinstance(stmt, ast.If) and isinstance(stmt.test, ast.Name) and stmt.test.id == "batch"):
                continue
            only_rows = _stored(stmt.body) - _stored(stmt.orelse) - _stored(body[:index])
            for later in body[index + 1:]:
                for node in ast.walk(later):
                    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in only_rows:
                        problems.append(f"{node.id} read at line {node.lineno}")
    assert not problems, "names bound only when the batch has rows are read on every path: " + ", ".join(problems)


def test_empty_path_binds_the_context_metadata():
    run = _run_function()
    stores = _stored(run.body)
    assert {"cache_seqlens", "max_ctx"} <= stores
