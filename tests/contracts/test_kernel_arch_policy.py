"""Arch-packaging pins for the kernel build trees (PR_MERGE_POLICY.md §1.8).

Per-arch release packaging policy, per-tree subset (after vLLM / flashinfer
practice): kernel build flags carry SASS-only targets. A ``code=compute_XX``
embed trades wheel size and a silent minutes-long driver JIT for coverage the
arch list must provide explicitly, and PTX forward-compat is unavailable for
``a`` targets anyway — so a missing arch must fail loud at runtime, never JIT
silently. ``sm_103a`` is reserved for kernels using sm_103-exclusive
instructions and must be registered here by file; nothing else may compile it.

The whole-tree counterpart of the per-PR added-line check in
``.github/workflows/scripts/check-pr-hygiene.sh``: that gate stops new
violations entering through a diff, this one pins the tree state (release
builds run the test suite, not the diff gate). Nothing here imports
batchgen_kernels — the registry module imports torch at module scope — so the
scan walks string constants in each file's AST.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# Kernel build trees whose Python files declare nvcc flag lists.
BUILD_TREES = ("batchgen_kernels", "op_builder")

# Vendored third-party sources are not ours to police.
EXCLUDED_PARTS = {"3rd"}

# Files allowed to compile an sm_103-exclusive (103a) variant, by repo-relative
# path. Registering a file here is the §1.8 act that unlocks the per-PR gate's
# `noqa: hygiene` waiver for it. Empty: no shipped kernel needs f2fp-class
# instructions today.
SM103A_ALLOWLIST: frozenset[str] = frozenset()


def _python_files():
    for tree in BUILD_TREES:
        root = REPO / tree
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.py")):
            if EXCLUDED_PARTS.intersection(path.relative_to(REPO).parts):
                continue
            yield path


def _string_constants(path: Path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node.lineno, node.value


def _hits(predicate):
    found = []
    for path in _python_files():
        rel = str(path.relative_to(REPO))
        for lineno, value in _string_constants(path):
            if predicate(rel, value):
                found.append(f"{rel}:{lineno}: {value!r}")
    return found


def test_no_ptx_embeds_in_kernel_build_flags():
    assert not _hits(lambda rel, s: "code=compute_" in s)


def test_no_arch_shorthand_in_kernel_build_flags():
    # The `-arch=` shorthand lets torch cpp_extension append its own arch
    # flags, which both build files document as emitting a plain compute image
    # that ptxas rejects for wgmma.* — always the explicit two-item -gencode.
    assert not _hits(lambda rel, s: s.startswith("-arch="))


def test_sm_103a_only_for_registered_exclusive_kernels():
    assert not _hits(
        lambda rel, s: (
            ("sm_103a" in s or "compute_103a" in s)
            and rel not in SM103A_ALLOWLIST
        )
    )


def _raise_messages(path: Path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Raise):
            for const in ast.walk(node):
                if isinstance(const, ast.Constant) and isinstance(const.value, str):
                    yield const.value


def test_arch_dispatchers_fail_loud_on_unknown_targets():
    # Missing/unknown arch = loud fail, never a silent fallback: both arch
    # dispatchers raise by name on an unsupported selection.
    setup_msgs = " ".join(_raise_messages(REPO / "batchgen_kernels" / "setup.py"))
    assert "Unsupported BUILD_ARCH" in setup_msgs
    jit_msgs = " ".join(
        _raise_messages(REPO / "batchgen_kernels" / "_jit_registry.py")
    )
    assert "Unsupported CUDA compute capability" in jit_msgs


def test_op_builder_refuses_ptx_suffixed_arch_entries():
    # The DeepSpeed-derived op builder once honored `+PTX` suffixes in
    # TORCH_CUDA_ARCH_LIST (and auto-appended one in jit_mode); it now refuses
    # them by name instead of emitting the embed.
    msgs = " ".join(_raise_messages(REPO / "op_builder" / "builder.py"))
    assert "requests a PTX embed" in msgs
