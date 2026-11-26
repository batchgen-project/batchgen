"""GLM-5 model fail-fast contracts (silent-fallback audit 2026-11-26).

glm5/model.py imports triton at module scope, so a CPU-only machine checks
the import contract on the AST and exercises the extracted hadamard helpers
directly.
"""
import ast
import logging
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "batchgen" / "models" / "glm" / "glm5" / "model.py"


def _tree():
    return ast.parse(MODEL.read_text())


# ----------------------------------------------- C6: unconditional MoE imports

def test_moe_decode_kernel_imports_are_unconditional():
    """The 3D FP8 MoE decode modules import at module scope outside any
    try/except: a broken checkout fails at import, decode and prefill alike."""
    tree = _tree()
    guarded = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            for sub in ast.walk(node):
                if isinstance(sub, ast.ImportFrom) and sub.module:
                    guarded.add(sub.module)
    unconditional = {
        node.module for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module
    }
    for module in ("batchgen.moe.grouped_fp8_blockwise_moe",
                   "batchgen.moe.dispatch_scatter_3d"):
        assert module in unconditional, f"{module} must import at top level"
        assert module not in guarded, f"{module} must not be try/except-ed"


def test_the_availability_flags_are_gone():
    source = MODEL.read_text()
    assert "_GLM5_HAS_FP8_BLOCKWISE" not in source
    assert "_GLM5_HAS_DISPATCH_3D" not in source


# ------------------------------------- item 8: hadamard matmul fallback warns

def _hadamard_namespace():
    """Exec the hadamard-related module-level defs without importing
    glm5.model (whose module scope needs triton)."""
    wanted = {
        "_get_hadamard_matrix", "_hadamard_transform",
        "_warn_hadamard_matmul_fallback_once",
    }
    tree = _tree()
    def _targets(node):
        if isinstance(node, ast.Assign):
            return node.targets
        if isinstance(node, ast.AnnAssign):
            return [node.target]
        return []

    body = [
        node for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in wanted)
        or any(
            isinstance(t, ast.Name) and t.id in (
                "_hadamard_matrix_cache", "_warned_hadamard_matmul_fallback",
            ) for t in _targets(node)
        )
    ]
    assert len(body) == 5, "hadamard defs moved; update this extraction"
    from typing import Dict, Tuple
    namespace = {
        "torch": torch, "logging": logging, "Dict": Dict, "Tuple": Tuple,
        "_hadamard_cuda_fn": None,
        "_HADAMARD_IMPORT_ERROR": ImportError("JIT build failed: no nvcc"),
    }
    module = ast.Module(body=body, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(MODEL), "exec"),
         namespace)
    return namespace


def test_hadamard_matmul_fallback_warns_once_with_reason(caplog):
    ns = _hadamard_namespace()
    x = torch.randn(4, 8)
    with caplog.at_level(logging.WARNING):
        first = ns["_hadamard_transform"](x)
        second = ns["_hadamard_transform"](x)
    warnings = [
        r for r in caplog.records
        if "hadamard_transform kernel unavailable" in r.getMessage()
    ]
    assert len(warnings) == 1
    assert "no nvcc" in warnings[0].getMessage()
    # The matmul form stays numerically the Hadamard transform.
    H = ns["_get_hadamard_matrix"](8, x.device, x.dtype)
    assert torch.allclose(first, x @ H)
    assert torch.equal(first, second)


def test_hadamard_import_guard_is_narrow_and_keeps_the_reason():
    source = MODEL.read_text()
    assert "except (ImportError, Exception)" not in source
    tree = _tree()
    for node in tree.body:
        if not isinstance(node, ast.Try):
            continue
        imports = [
            sub for sub in ast.walk(node)
            if isinstance(sub, ast.ImportFrom)
            and sub.module == "batchgen.other_kernels.hadamard_transform"
            and any(alias.name == "hadamard_transform"
                    for alias in sub.names)
        ]
        if not imports:
            continue
        handler = node.handlers[0]
        assert isinstance(handler.type, ast.Tuple)
        names = {ast.unparse(e) for e in handler.type.elts}
        assert names == {"ImportError", "OSError", "RuntimeError"}
        assert handler.name is not None, "the caught reason must be kept"
        return
    raise AssertionError("hadamard_transform import guard not found")
