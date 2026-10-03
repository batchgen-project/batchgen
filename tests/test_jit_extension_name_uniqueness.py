"""Keep legacy JIT loaders separate from packaged DSA extensions."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LEGACY_HADAMARD = (
    ROOT / "batchgen" / "other_kernels" / "hadamard_transform" / "__init__.py"
)
LEGACY_CSRC = ROOT / "batchgen" / "other_kernels" / "hadamard_transform" / "csrc"
KERNELS_INDEXER = (
    ROOT
    / "batchgen_kernels"
    / "attention"
    / "dsa"
    / "indexer"
    / "__init__.py"
)
KERNEL_SETUP = ROOT / "batchgen_kernels" / "setup.py"
INDEXER_CSRC = KERNELS_INDEXER.parent / "csrc"


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


def _source_text(path: Path) -> str:
    return path.read_text()


def test_hadamard_jit_extension_names_are_disjoint():
    legacy_names = _load_names(LEGACY_HADAMARD)
    kernels_names = _load_names(KERNELS_INDEXER)

    assert legacy_names == set()
    assert kernels_names == set()
    assert legacy_names.isdisjoint(kernels_names)

    legacy_source = _source_text(LEGACY_HADAMARD)
    assert "torch.utils.cpp_extension" not in legacy_source
    assert "fused_rope_hadamard_out" in legacy_source
    # The compatibility package must not retain a second kernel source tree;
    # the packaged DSA AOT sources are the single source of truth.
    assert not LEGACY_CSRC.exists()

    source = _source_text(KERNELS_INDEXER)
    assert "torch.utils.cpp_extension" not in source
    assert "batchgen_dsa_fast_hadamard_transform_cuda" in source
    assert "batchgen_dsa_fused_rope_hadamard_cuda" in source
    assert "allow_dev_jit=False" in source


def test_dsa_aot_sources_avoid_broad_cuda_context_header():
    for name in (
        "hadamard_binding.cpp",
        "fused_rope_hadamard_binding.cpp",
        "fused_rope_hadamard.cu",
    ):
        source = (INDEXER_CSRC / name).read_text()
        assert "ATen/cuda/CUDAContext.h" not in source

    setup = KERNEL_SETUP.read_text()
    assert "batchgen_dsa_fast_hadamard_transform_cuda" in setup
    assert "batchgen_dsa_fused_rope_hadamard_cuda" in setup
    assert "_cuda_dependency_include_paths" in setup
    assert "attention/dsa/indexer/csrc/hadamard_binding.cpp" in setup
    assert "attention/dsa/indexer/csrc/fused_rope_hadamard_binding.cpp" in setup
