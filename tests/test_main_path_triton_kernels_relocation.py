"""Structural gate for the main-path Triton kernels moved into ``batchgen_kernels``.

Four FP8 grouped-MoE Triton modules used to live under ``batchgen/``.  Their
canonical home is now ``batchgen_kernels/triton/``, where the Triton
version-management, registry and test conventions are kept.  The old module
paths survive as thin re-export shims so no call site changed.

This file pins the three properties that make that relocation safe:

1. **No canonical definition is left behind.**  The old ``batchgen/`` modules
   must contain no ``@triton.jit`` / ``@triton.autotune`` / ``@triton.heuristics``
   definition at all -- otherwise a second, silently diverging copy of a kernel
   would exist.
2. **The shim re-exports exactly the canonical module's public surface**, and
   nothing else, so ``from <old path> import X`` keeps resolving.
3. **The relocated kernel sources are byte-identical to the pre-move sources.**
   The move was mechanical; a changed kernel body would mean changed compiled
   behaviour slipped in under cover of a file move.

Everything here is CPU-only and import-free by design: the relocated modules
call ``triton.runtime.driver.active.get_current_target()`` at import time (the
unguarded ``HAS_TENSOR_DESC`` probe, pre-existing), so they can only be imported
on a machine with a live CUDA driver.  The object-identity check that does need
a real import is therefore guarded and skips on CPU-only machines.
"""

from __future__ import annotations

import ast
import hashlib
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Commit the relocation was cut from.  The pre-move blobs are read from it when
#: the object database still has it (it is the branch's fork point).
PRE_MOVE_COMMIT = "c7cbaa70"

TRITON_DECORATORS = frozenset(
    {"triton.jit", "triton.autotune", "triton.heuristics"}
)

#: ``canonical module`` -> relocation record.
#:
#: ``shim``        old import path, now a re-export shim
#: ``file_sha256`` SHA-256 of the whole pre-move file
#: ``kernels``     SHA-256 of each pre-move Triton definition's source segment,
#:                 keyed by kernel name (``name#2`` for a redefinition -- see
#:                 ``fused_fp8_moe_persistent_kernel``, which the pre-move file
#:                 already defined twice)
#: ``extra_exports`` public non-function top-level names the shim re-exports
#: ``private_not_reexported`` module-private names deliberately left behind
RELOCATIONS: dict[str, dict] = {
    "batchgen_kernels/triton/w8a8_grouped_gemm_stage_1.py": {
        "shim": "batchgen/gemm/w8a8_grouped_gemm_stage_1.py",
        "file_sha256": "7b461180aac57ccce5c975f0fd3550a0cb7160ebd7dbf5f1b80c745bea4cdc0a",
        "kernels": {
            "fused_fp8_moe_parallel_experts_kernel_optimized": "71932d8c45c5226daf67baf54705e8b2238463a9e217aaa41e1768a951411fdb",
            "fused_fp8_moe_baseline_optimized_v2": "cb1265bd689a89708b9d68ff216ed1fac9a5bcdafccdb564f7e8fe0845ba3bd8",
            "fused_fp8_moe_parallel_experts_kernel_tma": "7f91e4206f6d3ba9685eb529a5d68a569e6ee425b2f7dd3ff256f164d401687e",
            "fused_fp8_moe_persistent_descriptor_kernel_v2": "15b58976041a8db57dce2bafb790203b6e8b9f8f2c806868c567320bc76c24d7",
            "fused_fp8_moe_parallel_experts_kernel_no_activation": "a1f51def0b5a87757954c6458d1f9c586b3cc16c9f347448d54d1b1a1ed75fb3",
            "fused_fp8_moe_pipelined_kernel": "216722d9524fb29d7cc7a33335b750508d4330c676c40fd77033ef1b24094f55",
            "fused_fp8_moe_bf16_epilogue_kernel": "152fd0c869278d3fa0027b73d4dacb30e9564c16621ebfc76b43d385c9135aff",
            "fused_fp8_moe_persistent_kernel": "81ed4113e7f00e8f1e77af37adb781627f9ed55c19ab045d8eee021256a146f3",
            "fused_fp8_moe_persistent_kernel#2": "cde29e5ffc4298169ca678399a1257ba6c556dfd2b7ede6e8cf958da0c2eddab",
        },
        "extra_exports": {"HAS_TENSOR_DESC"},
        "private_not_reexported": {"_setup_allocator_once", "_allocator_set"},
    },
    "batchgen_kernels/triton/w8a8_grouped_gemm_stage_2.py": {
        "shim": "batchgen/gemm/w8a8_grouped_gemm_stage_2.py",
        "file_sha256": "0c5f1770f69b42045345dbfa7565ad761b21491ebc4db22911c9a8e29c307488",
        "kernels": {
            "fused_dequant_grouped_gemm_fp8_fp8_kernel_optimized": "20cf64e7e5b76e13d725894c2ba7e2e2c04f8b8e0212e6ba3bcf0a1a73e36858",
            "fused_dequant_grouped_gemm_fp8_fp8_fp32_kernel": "feeccbd8ebca7729020ba68ccaa38bdaee5b9ddf48115c3f32deea7f10fc2138",
        },
        "extra_exports": set(),
        "private_not_reexported": set(),
    },
    "batchgen_kernels/triton/fused_grouped_dequant_gemm.py": {
        "shim": "batchgen/moe/fused_grouped_dequant_gemm.py",
        "file_sha256": "1a4bc529eef8eaa84593f7e6a5ba284575f7bc021458c1b9550e71798431d04d",
        "kernels": {
            "fused_dequant_grouped_gemm_bf16_fp8_kernel": "33cb47916decc11dfc274eb0ceac25ecfc9c1b55327f0348632e850649681ba0",
            "fused_dequant_grouped_gemm_bf16_fp8_kernel_v2": "e200cc4507892023d8b51fa5db49c5312d14a8842cbdfd0303746058f37af808",
            "fused_dequant_grouped_gemm_bf16_fp8_kernel_fp32": "1c7d2352d851567b9fa65b0fade5ad0405d74ab270b9ae8090dffa968d88afad",
            "fused_dequant_grouped_gemm_fp8_fp8_kernel": "9ab2fc2e185771d78bb5c8e82af8268c586134a0f1269caf5a2014c0c502cf12",
            "fused_dequant_grouped_gemm_fp8_tma_kernel": "c237603b71a0030f5893f2726a03347453337e7b7c550f71456981b33c83a1ae",
        },
        "extra_exports": {"HAS_TENSOR_DESC"},
        "private_not_reexported": set(),
    },
    "batchgen_kernels/triton/fused_dequant_moe.py": {
        "shim": "batchgen/moe/fused_dequant_moe.py",
        "file_sha256": "12c24e07f591996bb73821d07ca1b667a836aa2a30a52a475143556baa8f55f0",
        "kernels": {
            "fused_dequant_weighted_moe_stage_1_kernel": "29ef1387f3ac00615fc8369e0c361cd23b3800f44c6a7ef4d0a240e926904ef6",
            "fused_fp8_moe_stage_1_kernel": "39090f77cc6da9871543f6bd02aef7a334c1dc0408e0dabe52532ab95d9c8f64",
            "fused_fp8_moe_stage_1_kernel_v2": "ff80b86f965bb7f24d50ae8ca7bfeee9d73d625f179e8fd00db2a6a411188103",
        },
        "extra_exports": set(),
        "private_not_reexported": set(),
    },
}

CANONICAL_MODULES = sorted(RELOCATIONS)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _decorator_name(node: ast.expr) -> str:
    """Dotted name of a decorator expression (``@triton.jit`` -> ``triton.jit``)."""
    while isinstance(node, ast.Call):
        node = node.func
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _parse(rel_path: str) -> tuple[ast.Module, str]:
    path = REPO_ROOT / rel_path
    assert path.is_file(), f"missing file: {rel_path}"
    source = path.read_text()
    return ast.parse(source), source


def _triton_definitions(tree: ast.Module, source: str) -> dict[str, str]:
    """Triton-decorated defs in *tree*, name -> SHA-256 of its source segment.

    A name redefined in the same module is suffixed ``#2``, ``#3``, ... in
    definition order, matching how the expected hashes were generated.
    """
    found: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        names = {_decorator_name(d) for d in node.decorator_list}
        if not (names & TRITON_DECORATORS):
            continue
        segment = ast.get_source_segment(source, node)
        assert segment is not None, f"no source segment for {node.name}"
        key = node.name
        suffix = 2
        while key in found:
            key = f"{node.name}#{suffix}"
            suffix += 1
        found[key] = hashlib.sha256(segment.encode()).hexdigest()
    return found


def _public_functions(tree: ast.Module) -> set[str]:
    return {
        n.name
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not n.name.startswith("_")
    }


def _shim_import(tree: ast.Module) -> ast.ImportFrom:
    imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert len(imports) == 1, (
        f"a shim must contain exactly one import statement, found {len(imports)}"
    )
    node = imports[0]
    assert isinstance(node, ast.ImportFrom), "shim import must be a 'from ... import ...'"
    return node


def _module_dotted(rel_path: str) -> str:
    return rel_path[: -len(".py")].replace("/", ".")


def _pre_move_blob(rel_path: str) -> str | None:
    """Pre-move file content from the git odb, or ``None`` if unreachable."""
    try:
        done = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "show", f"{PRE_MOVE_COMMIT}:{rel_path}"],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout if done.returncode == 0 else None


# --------------------------------------------------------------------------
# 1. no canonical definition left behind
# --------------------------------------------------------------------------

@pytest.mark.parametrize("canonical", CANONICAL_MODULES)
def test_shim_has_no_triton_definitions(canonical: str) -> None:
    """The vacated ``batchgen/`` module must define no Triton kernel at all."""
    shim = RELOCATIONS[canonical]["shim"]
    tree, source = _parse(shim)
    leftover = _triton_definitions(tree, source)
    assert leftover == {}, (
        f"{shim} is a re-export shim but still defines Triton kernels "
        f"{sorted(leftover)}; the canonical definition belongs in {canonical}."
    )


@pytest.mark.parametrize("canonical", CANONICAL_MODULES)
def test_shim_is_only_a_docstring_and_one_reexport(canonical: str) -> None:
    """A shim is a docstring plus one ``from <canonical> import (...)`` -- nothing else."""
    shim = RELOCATIONS[canonical]["shim"]
    tree, _ = _parse(shim)

    assert ast.get_docstring(tree), f"{shim} must document that the kernels moved"

    body = [n for n in tree.body if not isinstance(n, ast.Expr)]
    assert len(body) == 1, (
        f"{shim} must contain exactly one statement besides its docstring, "
        f"found {len(body)}: {[type(n).__name__ for n in body]}"
    )

    node = _shim_import(tree)
    assert node.level == 0, f"{shim} must re-export via an absolute import"
    assert node.module == _module_dotted(canonical), (
        f"{shim} re-exports from {node.module}, expected {_module_dotted(canonical)}"
    )
    assert not any(a.name == "*" for a in node.names), (
        f"{shim} must name its re-exports explicitly, not use a star import"
    )
    assert all(a.asname is None for a in node.names), (
        f"{shim} must re-export under the original names"
    )


# --------------------------------------------------------------------------
# 2. the shim covers the canonical public surface exactly
# --------------------------------------------------------------------------

@pytest.mark.parametrize("canonical", CANONICAL_MODULES)
def test_shim_reexports_canonical_public_surface(canonical: str) -> None:
    """Every public function of the canonical module is re-exported, and no stranger."""
    record = RELOCATIONS[canonical]
    shim = record["shim"]

    canon_tree, _ = _parse(canonical)
    shim_tree, _ = _parse(shim)

    expected = _public_functions(canon_tree) | set(record["extra_exports"])
    actual = {a.name for a in _shim_import(shim_tree).names}

    assert actual == expected, (
        f"{shim} re-export list is out of sync with {canonical}.\n"
        f"  missing from shim: {sorted(expected - actual)}\n"
        f"  not in canonical : {sorted(actual - expected)}"
    )


@pytest.mark.parametrize("canonical", CANONICAL_MODULES)
def test_mutable_module_state_is_not_reexported(canonical: str) -> None:
    """Module-private one-shot state must not be aliased into the shim.

    Re-exporting a name the canonical module rebinds through ``global`` would
    freeze a stale snapshot in the shim namespace.
    """
    record = RELOCATIONS[canonical]
    if not record["private_not_reexported"]:
        pytest.skip(f"{canonical} has no private module state to guard")
    shim_tree, _ = _parse(record["shim"])
    actual = {a.name for a in _shim_import(shim_tree).names}
    leaked = actual & set(record["private_not_reexported"])
    assert not leaked, (
        f"{record['shim']} must not re-export module-private state {sorted(leaked)}"
    )


# --------------------------------------------------------------------------
# 3. the relocated sources are byte-identical to the pre-move sources
# --------------------------------------------------------------------------

@pytest.mark.parametrize("canonical", CANONICAL_MODULES)
def test_relocated_kernel_bodies_match_pre_move(canonical: str) -> None:
    """Each relocated Triton definition hashes to its pre-move source segment."""
    expected = RELOCATIONS[canonical]["kernels"]
    tree, source = _parse(canonical)
    actual = _triton_definitions(tree, source)

    assert set(actual) == set(expected), (
        f"{canonical} Triton definition set changed.\n"
        f"  missing: {sorted(set(expected) - set(actual))}\n"
        f"  added  : {sorted(set(actual) - set(expected))}"
    )
    changed = {k: (expected[k], actual[k]) for k in expected if expected[k] != actual[k]}
    assert not changed, (
        f"{canonical}: kernel source changed during what must be a pure relocation. "
        f"Changed kernels: {sorted(changed)}. If a kernel was intentionally edited, "
        f"that is a behaviour change and belongs in its own reviewed commit with the "
        f"expected hash updated deliberately."
    )


@pytest.mark.parametrize("canonical", CANONICAL_MODULES)
def test_relocated_file_is_byte_identical_to_pre_move_blob(canonical: str) -> None:
    """Whole-file check against the pre-move blob in the git object database."""
    record = RELOCATIONS[canonical]
    blob = _pre_move_blob(record["shim"])
    if blob is None:
        pytest.skip(
            f"{PRE_MOVE_COMMIT}:{record['shim']} is not reachable in this "
            f"object database (shallow clone or detached export)"
        )
    assert hashlib.sha256(blob.encode()).hexdigest() == record["file_sha256"], (
        f"the recorded pre-move hash for {record['shim']} does not match the blob "
        f"at {PRE_MOVE_COMMIT}; the recorded baseline is wrong"
    )
    actual = (REPO_ROOT / canonical).read_text()
    assert hashlib.sha256(actual.encode()).hexdigest() == record["file_sha256"], (
        f"{canonical} is not byte-identical to {PRE_MOVE_COMMIT}:{record['shim']}. "
        f"This gate asserts the move was mechanical; a deliberate kernel edit must "
        f"land in its own commit with this baseline updated."
    )


# --------------------------------------------------------------------------
# 4. live object identity (needs a real import, hence a CUDA driver)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("canonical", CANONICAL_MODULES)
def test_shim_objects_are_the_canonical_objects(canonical: str) -> None:
    """``shim.X is canonical.X`` for every re-exported name.

    Skipped without Triton + a CUDA driver: these modules evaluate
    ``triton.runtime.driver.active.get_current_target()`` at import time.
    """
    pytest.importorskip("triton", reason="relocated modules import triton at module scope")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("module-scope HAS_TENSOR_DESC probe needs a live CUDA driver")

    import importlib

    record = RELOCATIONS[canonical]
    canon_mod = importlib.import_module(_module_dotted(canonical))
    shim_mod = importlib.import_module(_module_dotted(record["shim"]))

    shim_tree, _ = _parse(record["shim"])
    names = sorted(a.name for a in _shim_import(shim_tree).names)
    assert names, "shim re-exports nothing"

    for name in names:
        assert hasattr(shim_mod, name), f"{record['shim']} lost re-export {name}"
        assert getattr(shim_mod, name) is getattr(canon_mod, name), (
            f"{record['shim']}.{name} is not the same object as {canonical}.{name}"
        )


# --------------------------------------------------------------------------
# 5. the relocation is complete: no caller reaches a vacated definition
# --------------------------------------------------------------------------

def _resolve_import_module(rel_path: str, node: ast.ImportFrom) -> str | None:
    """Absolute dotted module a ``from ... import`` names, relative imports included."""
    if node.level == 0:
        return node.module
    package = Path(rel_path).parent.as_posix().replace("/", ".")
    parts = package.split(".") if package else []
    keep = len(parts) - (node.level - 1)
    if keep < 0:
        return None
    base = parts[:keep]
    tail = node.module.split(".") if node.module else []
    return ".".join(base + tail)


def test_every_call_site_import_is_covered_by_its_shim() -> None:
    """No call site breaks: each name imported from an old path is re-exported.

    This is the assertion that the relocation is source-compatible.  It resolves
    relative imports too -- ``modeling_deepseek_v3.py`` reaches one of these
    modules as ``from ....moe.fused_grouped_dequant_gemm import ...``.
    """
    shim_exports: dict[str, set[str]] = {}
    for canonical, record in RELOCATIONS.items():
        shim_tree, _ = _parse(record["shim"])
        shim_exports[_module_dotted(record["shim"])] = {
            a.name for a in _shim_import(shim_tree).names
        }

    missing: list[str] = []
    checked = 0
    for root in ("batchgen", "tests"):
        for path in (REPO_ROOT / root).rglob("*.py"):
            rel = path.relative_to(REPO_ROOT).as_posix()
            if rel in {r["shim"] for r in RELOCATIONS.values()}:
                continue
            try:
                tree = ast.parse(path.read_text())
            except (SyntaxError, UnicodeDecodeError):
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom):
                    continue
                target = _resolve_import_module(rel, node)
                if target not in shim_exports:
                    continue
                checked += 1
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    if alias.name not in shim_exports[target]:
                        missing.append(
                            f"{rel}:{node.lineno} imports "
                            f"{alias.name!r} from {target}, which the shim does "
                            f"not re-export"
                        )

    assert checked, (
        "found no call site importing a relocated module -- the scan is broken, "
        "or every consumer was rewritten (then this gate should be retired)"
    )
    assert not missing, "call sites would break:\n  " + "\n  ".join(sorted(missing))


def test_no_module_imports_a_relocated_kernel_from_a_third_place() -> None:
    """Only the shim may import the canonical module's kernels for the old names.

    Guards against a future partial revert that re-creates a kernel under
    ``batchgen/`` while the shim still forwards to ``batchgen_kernels``.
    """
    duplicates: list[str] = []
    for canonical, record in RELOCATIONS.items():
        stem = Path(canonical).stem
        for path in (REPO_ROOT / "batchgen").rglob("*.py"):
            rel = path.relative_to(REPO_ROOT).as_posix()
            if rel == record["shim"]:
                continue
            if path.name != f"{stem}.py":
                continue
            duplicates.append(rel)
    assert not duplicates, (
        f"modules shadowing a relocated kernel module still exist under batchgen/: "
        f"{sorted(duplicates)}"
    )
