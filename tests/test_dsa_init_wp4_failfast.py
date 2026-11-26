"""The WP4 fused-indexer-score import in batchgen_kernels dsa/__init__ is a
top-level fail-fast import, not dead code inside another block's except suite.

Regression for the 2026-11-26 audit finding: the WP4 import block was indented
into the WP2 ``except ...: pass`` suite (comments do not terminate a suite), so
it only executed when WP2 had failed, while ``__all__`` advertised the seven
names unconditionally. These are AST checks so they run without CUDA/triton.
"""
import ast
from pathlib import Path

INIT = (Path(__file__).resolve().parents[1]
        / "batchgen_kernels" / "attention" / "dsa" / "__init__.py")

WP4_NAMES = {
    "FP8WqbWeightsCUDA", "cuda_wq_b_proj_out", "fused_score_pipeline",
    "fused_score_and_topk", "fused_score_and_topk_out",
    "fused_paged_score_and_topk_out", "rope_hadamard_q_out",
}


def _module():
    return ast.parse(INIT.read_text())


def test_wp4_import_is_top_level_and_unguarded():
    found = None
    for node in _module().body:  # module body only: top-level statements
        if (isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.endswith("fused_indexer_score")):
            found = node
    assert found is not None, (
        "the fused_indexer_score import is not a top-level statement "
        "(it is nested inside another block, i.e. dead or conditional)")
    assert {a.name for a in found.names} == WP4_NAMES


def test_wp4_import_not_wrapped_in_try():
    for node in ast.walk(_module()):
        if isinstance(node, ast.Try):
            for inner in ast.walk(node):
                if (isinstance(inner, ast.ImportFrom)
                        and inner.module
                        and inner.module.endswith("fused_indexer_score")):
                    raise AssertionError(
                        "WP4 fused_indexer_score import must fail fast, "
                        "not be wrapped in try/except")
