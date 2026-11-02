"""Every AOT CUDA extension is declared once and mirrored once.

``setup.py`` builds the extensions at install time and ``_jit_registry.py``
mirrors them for dev-mode JIT; an extension present in only one of the two
either cannot be built or cannot be JIT-compiled, and the failure surfaces
much later, on a node, as an ImportError inside decode.

``setup.py`` is read with ``ast`` instead of being imported: it imports torch
and resolves CUDA_HOME at module scope, which no CPU-only test host can do.
``_jit_registry.py`` is pure Python; it is read with ``ast`` for the
declaration checks and loaded by path where resolved flags are needed.

SM90a code must be selected with the explicit two-item
``-gencode arch=compute_90a,code=sm_90a``. nvcc's ``-arch=sm_90a`` shorthand
lets torch append its own arch flags (GPU detection, which crashes on a host
without a GPU) and emits a plain compute_90 image that ptxas rejects for
``wgmma.*``; a cold build of the DSA indexer projection failed that way.
"""

from __future__ import annotations

import ast
import importlib.util
import pathlib

_KERNELS_ROOT = (
    pathlib.Path(__file__).resolve().parents[1] / "batchgen_kernels"
)
_SETUP = _KERNELS_ROOT / "setup.py"
_REGISTRY = _KERNELS_ROOT / "_jit_registry.py"

NEW_INDEXER_WRITER = "batchgen_kernels.attention._C_fused_indexer_k_write_fp8"

SM90A_GENCODE = ["-gencode", "arch=compute_90a,code=sm_90a"]

# The DSA indexer kernels that used to be compiled at runtime.
DSA_AOT_EXTENSIONS = {
    "batchgen_kernels.attention._C_fast_hadamard_transform": [
        "src/attention/hadamard/hadamard_binding.cpp",
        "src/attention/hadamard/fast_hadamard_transform_cuda.cu",
    ],
    "batchgen_kernels.attention._C_fused_rope_hadamard": [
        "src/attention/hadamard/fused_rope_hadamard_binding.cpp",
        "src/attention/hadamard/fused_rope_hadamard.cu",
    ],
    "batchgen_kernels.attention._C_dsa_fast_topk": [
        "src/attention/dsa_fast_topk.cu",
    ],
    "batchgen_kernels.attention._C_indexer_kv_proj_wgmma": [
        "src/attention/indexer_kv_proj_wgmma.cu",
    ],
}
KV_PROJ = "batchgen_kernels.attention._C_indexer_kv_proj_wgmma"


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


# ──────────────────────────────────────────────────────────────────────────────
# Resolved nvcc flags
# ──────────────────────────────────────────────────────────────────────────────

def _setup_bindings(tree: ast.Module) -> dict[str, ast.AST]:
    """First module-level binding of each name, taking the first ``if`` arm.

    The first arm of every module-level ``if`` is the default configuration
    (``BUILD_ARCH=sm90a``), so this resolves the flag lists the default build
    compiles with.
    """

    bindings: dict[str, ast.AST] = {}

    def visit(statements):
        for statement in statements:
            if (
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
            ):
                bindings.setdefault(statement.targets[0].id, statement.value)
            elif isinstance(statement, ast.If):
                visit(statement.body)

    visit(tree.body)
    return bindings


def _flatten(node: ast.AST, bindings: dict[str, ast.AST]) -> list[str]:
    if isinstance(node, (ast.List, ast.Tuple)):
        flags: list[str] = []
        for element in node.elts:
            flags += _flatten(element, bindings)
        return flags
    if isinstance(node, ast.Starred):
        return _flatten(node.value, bindings)
    if isinstance(node, ast.Constant):
        return [str(node.value)]
    if isinstance(node, ast.Name):
        if node.id in bindings:
            return _flatten(bindings[node.id], bindings)
        return [f"<{node.id}>"]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _flatten(node.left, bindings) + _flatten(node.right, bindings)
    if isinstance(node, ast.Call):
        # e.g. os.getenv("NVCC_THREADS", "4"): one opaque argument
        return [f"<{ast.unparse(node)}>"]
    raise AssertionError(f"cannot resolve nvcc flag expression {ast.unparse(node)}")


def _extension_call_nvcc(call: ast.Call, bindings) -> list[str]:
    keywords = {kw.arg: kw.value for kw in call.keywords}
    args = keywords.get("extra_compile_args")
    assert isinstance(args, ast.Dict), ast.unparse(call)
    for key, value in zip(args.keys, args.values):
        if isinstance(key, ast.Constant) and key.value == "nvcc":
            return _flatten(value, bindings)
    raise AssertionError(f"no nvcc flags in {ast.unparse(call)}")


def _setup_nvcc_flags() -> tuple[dict[str, list[str]], set[str]]:
    """(name -> resolved nvcc flags, names in ``_sm90a_extensions``)."""

    tree = ast.parse(_SETUP.read_text())
    bindings = _setup_bindings(tree)
    flags: dict[str, list[str]] = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "CUDAExtension"
        ):
            name = {kw.arg: kw.value for kw in node.keywords}["name"].value
            flags[name] = _extension_call_nvcc(node, bindings)

    sm90a_list = bindings.get("_sm90a_extensions")
    assert isinstance(sm90a_list, ast.List), "setup.py has no _sm90a_extensions list"
    sm90a = {
        {kw.arg: kw.value for kw in call.keywords}["name"].value
        for call in sm90a_list.elts
        if isinstance(call, ast.Call)
    }
    return flags, sm90a


def _registry():
    spec = importlib.util.spec_from_file_location("_jit_registry_under_test", _REGISTRY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_registry()


def _gencode_targets(flags: list[str]) -> list[str]:
    return [
        flags[index + 1]
        for index, flag in enumerate(flags)
        if flag == "-gencode" and index + 1 < len(flags)
    ]


def test_dsa_indexer_extensions_are_registered_aot():
    setup_extensions = _setup_extensions()
    registry = _registry_entries()
    for name, sources in DSA_AOT_EXTENSIONS.items():
        assert setup_extensions.get(name) == sources, name
        assert registry.get(name) == sources, name
        for source in sources:
            assert (_KERNELS_ROOT / source).is_file(), f"{name}: missing {source}"


def test_hadamard_extensions_share_the_hadamard_include_dir():
    registry = _registry()
    for name in (
        "batchgen_kernels.attention._C_fast_hadamard_transform",
        "batchgen_kernels.attention._C_fused_rope_hadamard",
    ):
        assert registry[name]["include_dirs"] == ["src/attention/hadamard"], name
        assert (_KERNELS_ROOT / "src/attention/hadamard").is_dir()


def test_every_sm90a_extension_uses_explicit_gencode():
    flags, sm90a = _setup_nvcc_flags()
    assert KV_PROJ in sm90a
    for name in sm90a:
        targets = _gencode_targets(flags[name])
        assert targets == ["arch=compute_90a,code=sm_90a"], (
            f"{name}: expected exactly one explicit sm90a -gencode at the "
            f"default BUILD_ARCH, got {targets} from {flags[name]}"
        )


def test_no_extension_uses_the_arch_shorthand():
    flags, _ = _setup_nvcc_flags()
    for name, nvcc in flags.items():
        offenders = [flag for flag in nvcc if flag.startswith("-arch")]
        assert not offenders, f"setup.py {name}: {offenders}"
    for name, config in _registry().items():
        offenders = [
            flag for flag in config.get("nvcc_flags", []) if flag.startswith("-arch")
        ]
        assert not offenders, f"_jit_registry {name}: {offenders}"


def test_indexer_kv_proj_is_built_with_four_stages_for_sm90a_only():
    flags, _ = _setup_nvcc_flags()
    registry_flags = _registry()[KV_PROJ]["nvcc_flags"]
    for nvcc in (flags[KV_PROJ], registry_flags):
        assert _gencode_targets(nvcc) == ["arch=compute_90a,code=sm_90a"], nvcc
        stages = [flag for flag in nvcc if flag.startswith("-DNUM_STAGES")]
        assert stages == ["-DNUM_STAGES=4"], nvcc
