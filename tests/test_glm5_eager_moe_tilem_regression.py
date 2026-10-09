"""Regression: the eager 3D MoE decode path must not reference undefined names.

PR #443 left `_glm5_moe_tilem_avg_override() or ...` in
`Glm5MoE._fp8_blockwise_gemm_3d` while only importing
`gemm_tilem_avg_effective` (which applies the override itself). The eager
decode path never runs under CUDA graphs, so the NameError stayed latent
until a graph-off server hit the first MoE decode step.
"""

import ast
import pathlib

_MODEL_SRC = pathlib.Path(__file__).resolve().parents[1] / "batchgen/models/glm/glm5/model.py"


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"{name} not found")


def test_eager_gemm_3d_references_only_defined_tilem_helpers():
    source = _MODEL_SRC.read_text()
    assert "_glm5_moe_tilem_avg_override" not in source
    tree = ast.parse(source)
    fn = _function(tree, "_fp8_blockwise_gemm_3d")
    calls = {
        node.func.id
        for node in ast.walk(fn)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "_glm5_gemm_tilem_avg" in calls
    # Every bare-name call in the function resolves at module scope.
    module_names = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    } | {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    builtins_ok = {"max", "min", "int", "getattr", "len", "range", "isinstance", "print"}
    local_defs = {
        n.name for n in ast.walk(fn)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for name in calls:
        assert (
            name in module_names or name in builtins_ok or name in local_defs
        ), f"call to {name} does not resolve at module scope"
