"""Regression: the eager 3D MoE decode path must not reference undefined names.

PR #443 rewrote the eager TileM-average line against the CALLER's scope:
`_glm5_moe_tilem_avg_override` was never defined anywhere, and `num_global` /
`topk` exist in `_forward_decode_3d` but not in `_fp8_blockwise_gemm_3d`.
Graph-mode serving never runs this path, so both NameErrors stayed latent
until a graph-off server hit its first MoE decode step. This test resolves
EVERY loaded name in the function — calls and arguments alike — against the
function's own scope, its parameters, module scope and builtins.
"""

import ast
import builtins
import pathlib

_MODEL_SRC = pathlib.Path(__file__).resolve().parents[1] / "batchgen/models/glm/glm5/model.py"


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"{name} not found")


def _module_scope(tree: ast.Module) -> set[str]:
    names = {
        alias.asname or alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def _local_scope(fn: ast.FunctionDef) -> set[str]:
    args = fn.args
    names = {a.arg for a in args.args + args.kwonlyargs + args.posonlyargs}
    if args.vararg:
        names.add(args.vararg.arg)
    if args.kwarg:
        names.add(args.kwarg.arg)
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.For, ast.comprehension)):
            target = node.target
            for t in ast.walk(target):
                if isinstance(t, ast.Name):
                    names.add(t.id)
        elif isinstance(node, ast.withitem) and node.optional_vars is not None:
            for t in ast.walk(node.optional_vars):
                if isinstance(t, ast.Name):
                    names.add(t.id)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
    return names


def test_eager_gemm_3d_resolves_every_loaded_name():
    source = _MODEL_SRC.read_text()
    assert "_glm5_moe_tilem_avg_override" not in source
    tree = ast.parse(source)
    fn = _function(tree, "_fp8_blockwise_gemm_3d")
    resolvable = _module_scope(tree) | _local_scope(fn) | set(dir(builtins))
    unresolved = sorted({
        node.id
        for node in ast.walk(fn)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        and node.id not in resolvable
    })
    assert not unresolved, f"unresolved names in _fp8_blockwise_gemm_3d: {unresolved}"
    # And the caller threads the two values the TileM formula needs.
    caller = _function(tree, "_forward_decode_3d")
    call = next(
        node for node in ast.walk(caller)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_fp8_blockwise_gemm_3d"
    )
    assert {kw.arg for kw in call.keywords} >= {"num_global", "topk"}
