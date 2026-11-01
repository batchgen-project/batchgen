"""No runtime native compilation in the serving packages.

A ``torch.utils.cpp_extension.load``/``load_inline`` call compiles C++/CUDA
inside a worker: minutes of nvcc on a cold node, a shared build directory that
concurrent workers contend for, and a failure mode that only appears on the
node. Native kernels are built ahead of time by ``batchgen_kernels/setup.py``
and imported with ``batchgen_kernels.load_extension``.

The scan is static (``ast``), so it runs on any host without CUDA.
"""

from __future__ import annotations

import ast
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]

SCANNED_PACKAGES = ("batchgen", "batchgen_kernels")

_JIT_FUNCTIONS = frozenset({"load", "load_inline"})
_CPP_EXTENSION = "torch.utils.cpp_extension"

# Paths (relative to the repository root) allowed to reach the JIT builders.
ALLOWED = frozenset(
    {
        # The dev-mode fallback (BATCHGEN_KERNELS_DEV=1) behind load_extension.
        "batchgen_kernels/__init__.py",
        # Build-time only.
        "batchgen_kernels/setup.py",
        # Still compiles its MoE kernels with load_inline; moved to AOT by
        # kernel PR B of the runtime-JIT elimination. Remove this entry there.
        "batchgen/moe/fp8_wgmma_pipeline.py",
    }
)
# Vendored trees, build tooling and in-tree build output (stale copies of
# sources), never imported by a serving process.
ALLOWED_PREFIXES = ("op_builder/", "batchgen_kernels/3rd/", "batchgen_kernels/build/")


def _is_allowed(relative: str) -> bool:
    return relative in ALLOWED or relative.startswith(ALLOWED_PREFIXES)


def _python_files():
    for package in SCANNED_PACKAGES:
        for path in sorted((ROOT / package).rglob("*.py")):
            relative = path.relative_to(ROOT).as_posix()
            if not _is_allowed(relative):
                yield relative, path


def _jit_uses(tree: ast.Module) -> list[str]:
    """Imports of, and calls to, the cpp_extension JIT builders."""

    # local name -> builder name, for ``from torch.utils.cpp_extension import``
    imported: dict[str, str] = {}
    # local names bound to the cpp_extension module itself
    module_aliases: set[str] = set()
    uses: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == _CPP_EXTENSION:
            for alias in node.names:
                if alias.name in _JIT_FUNCTIONS:
                    imported[alias.asname or alias.name] = alias.name
                    uses.append(f"line {node.lineno}: imports {alias.name}")
        elif isinstance(node, ast.ImportFrom) and node.module == "torch.utils":
            for alias in node.names:
                if alias.name == "cpp_extension":
                    module_aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == _CPP_EXTENSION and alias.asname:
                    module_aliases.add(alias.asname)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            if func.id in imported or func.id == "load_inline":
                uses.append(f"line {node.lineno}: calls {func.id}(...)")
        elif isinstance(func, ast.Attribute) and func.attr in _JIT_FUNCTIONS:
            owner = ast.unparse(func.value)
            if (
                owner in module_aliases
                or owner == _CPP_EXTENSION
                or owner.endswith(".cpp_extension")
                or func.attr == "load_inline"
            ):
                uses.append(f"line {node.lineno}: calls {ast.unparse(func)}(...)")
    return uses


def test_scan_covers_the_serving_packages():
    files = [relative for relative, _ in _python_files()]
    assert "batchgen_kernels/attention/dsa/indexer/__init__.py" in files
    assert "batchgen_kernels/attention/dsa/fast_topk_cuda.py" in files
    assert "batchgen_kernels/attention/dsa/fused_indexer_kv_proj_cuda.py" in files
    assert "batchgen/other_kernels/hadamard_transform/__init__.py" in files


def test_detector_flags_every_jit_spelling():
    source = "\n".join(
        [
            "from torch.utils.cpp_extension import load as jit_load, load_inline",
            "import torch.utils.cpp_extension as cpp_ext",
            "import torch",
            "jit_load(name='a', sources=[])",
            "load_inline(name='b', cpp_sources='')",
            "cpp_ext.load(name='c', sources=[])",
            "torch.utils.cpp_extension.load_inline(name='d', cpp_sources='')",
        ]
    )
    uses = _jit_uses(ast.parse(source))
    calls = [use for use in uses if "calls" in use]
    assert len(calls) == 4, uses


def test_no_runtime_jit_outside_the_allowlist():
    offenders = []
    for relative, path in _python_files():
        uses = _jit_uses(ast.parse(path.read_text(), filename=str(path)))
        offenders += [f"{relative} {use}" for use in uses]
    assert not offenders, (
        "runtime native compilation found; build the extension in "
        "batchgen_kernels/setup.py and import it with load_extension:\n  "
        + "\n  ".join(offenders)
    )


def test_allowlisted_files_still_exist():
    for relative in ALLOWED:
        assert (ROOT / relative).is_file(), (
            f"{relative} is allow-listed but gone; drop it from ALLOWED"
        )
