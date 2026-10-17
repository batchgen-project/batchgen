"""Every AOT CUDA extension is declared once and mirrored once.

``setup.py`` builds the extensions at install time and ``_jit_registry.py``
mirrors them for dev-mode JIT; an extension present in only one of the two
either cannot be built or cannot be JIT-compiled, and the failure surfaces
much later, on a node, as an ImportError inside decode.

The two files are read with ``ast`` instead of being imported: ``setup.py``
imports torch and resolves CUDA_HOME at module scope, which no CPU-only test
host can do.
"""

from __future__ import annotations

import ast
import pathlib

_KERNELS_ROOT = (
    pathlib.Path(__file__).resolve().parents[1] / "batchgen_kernels"
)
_SETUP = _KERNELS_ROOT / "setup.py"
_REGISTRY = _KERNELS_ROOT / "_jit_registry.py"

NEW_INDEXER_WRITER = "batchgen_kernels.attention._C_fused_indexer_k_write_fp8"


def _string_list(node: ast.AST) -> list[str]:
    if not isinstance(node, ast.List):
        return []
    return [
        element.value
        for element in node.elts
        if isinstance(element, ast.Constant) and isinstance(element.value, str)
    ]


def _setup_extensions() -> dict[str, list[str]]:
    """name -> sources, for every CUDAExtension(...) call in setup.py."""

    tree = ast.parse(_SETUP.read_text())
    extensions: dict[str, list[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Name) and func.id == "CUDAExtension"):
            continue
        keywords = {kw.arg: kw.value for kw in node.keywords}
        name_node = keywords.get("name")
        assert isinstance(name_node, ast.Constant), "CUDAExtension needs a literal name"
        extensions[name_node.value] = _string_list(keywords.get("sources"))
    return extensions


def _registry_entries() -> dict[str, list[str]]:
    """name -> sources, for every entry of _jit_registry.get_registry()."""

    tree = ast.parse(_REGISTRY.read_text())
    entries: dict[str, list[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values):
            if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
                continue
            if not key.value.startswith("batchgen_kernels."):
                continue
            sources: list[str] = []
            if isinstance(value, ast.Dict):
                for inner_key, inner_value in zip(value.keys, value.values):
                    if (
                        isinstance(inner_key, ast.Constant)
                        and inner_key.value == "sources"
                    ):
                        sources = _string_list(inner_value)
            entries[key.value] = sources
    return entries


def test_setup_and_jit_registry_declare_the_same_extensions():
    setup_extensions = _setup_extensions()
    registry = _registry_entries()
    assert setup_extensions, "no CUDAExtension found in setup.py"
    assert set(setup_extensions) == set(registry), (
        "setup.py ext_modules and _jit_registry must name the same extensions; "
        f"only in setup.py={sorted(set(setup_extensions) - set(registry))}, "
        f"only in registry={sorted(set(registry) - set(setup_extensions))}"
    )


def test_every_declared_source_exists():
    for name, sources in _setup_extensions().items():
        assert sources, f"{name} declares no sources"
        for source in sources:
            assert (_KERNELS_ROOT / source).is_file(), f"{name}: missing {source}"


def test_registry_sources_match_setup_sources():
    setup_extensions = _setup_extensions()
    for name, sources in _registry_entries().items():
        assert sources == setup_extensions[name], (
            f"{name}: _jit_registry sources {sources} do not match setup.py "
            f"{setup_extensions[name]}"
        )


def test_fused_indexer_k_write_fp8_is_registered_aot():
    setup_extensions = _setup_extensions()
    assert NEW_INDEXER_WRITER in setup_extensions
    assert setup_extensions[NEW_INDEXER_WRITER] == [
        "src/attention/fused_indexer_k_write_fp8.cu"
    ]
    assert NEW_INDEXER_WRITER in _registry_entries()
