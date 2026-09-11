"""Prevent unrelated Torch JIT extensions from sharing one build directory."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LEGACY_HADAMARD = (
    ROOT / "batchgen" / "other_kernels" / "hadamard_transform" / "__init__.py"
)
KERNELS_INDEXER = (
    ROOT
    / "batchgen_kernels"
    / "attention"
    / "dsa"
    / "indexer"
    / "__init__.py"
)


def _load_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    names = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "load":
            continue
        for keyword in node.keywords:
            if keyword.arg != "name":
                continue
            if not isinstance(keyword.value, ast.Constant):
                raise AssertionError(f"{path}: load(name=...) must be literal")
            names.add(keyword.value.value)
    return names


def test_hadamard_jit_extension_names_are_disjoint():
    legacy_names = _load_names(LEGACY_HADAMARD)
    kernels_names = _load_names(KERNELS_INDEXER)

    assert legacy_names == {
        "batchgen_glm5_fast_hadamard_transform_cuda",
        "batchgen_glm5_fused_rope_hadamard_cuda",
    }
    assert kernels_names == {
        "fast_hadamard_transform_cuda",
        "fused_rope_hadamard_cuda",
    }
    assert legacy_names.isdisjoint(kernels_names)
