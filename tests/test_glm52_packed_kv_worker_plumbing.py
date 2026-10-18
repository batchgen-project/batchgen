"""CPU tests for the worker-side GLM-5.2 packed-KV seams.

Two things are CPU-testable and both are load-bearing:

1. ``BatchGenWorker._prefill_pool_geometry`` -- the prefill ACTIVATION pool
   must keep the bf16 576-element latent even though the PAGED primary cache
   is now uint8/656. Getting this wrong allocates uint8 rows under a bf16
   scatter (``batchgen/models/glm/glm5/sparse_prefill.py:797-799``).
2. the ``--disable-glm52-packed-kv`` -> ``set_glm52_packed_kv`` plumbing, whose
   whole correctness is an ORDERING property: the latch must be set before the
   process's first KV-profile resolution, in both the server and the worker
   process.

``batchgen`` cannot be imported on a CPU-only machine, so these tests use the
module-by-path pattern of ``tests/test_glm52_host_kv_profile.py`` for the config
module, and compile the ONE worker method under test straight out of
``batchgen/batchgen_worker.py`` via ``ast`` -- the real source, not a copy.
"""

import ast
import importlib.util
import sys
import types
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Optional, Sequence

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
WORKER_PATH = ROOT / "batchgen" / "batchgen_worker.py"
SERVER_ARGS_PATH = ROOT / "batchgen" / "server" / "server_args.py"
WORKER_MANAGER_PATH = ROOT / "batchgen" / "server" / "worker_manager.py"
HOST_KV_CONFIG_PATH = ROOT / "batchgen" / "kv_cache" / "host_kv_mananger_config.py"

# GLM-5.2 is the only model whose primary profile moves.
GLM52 = "zai-org/GLM-5.2-FP8"
# GLM-5.3 shares the bf16 `glm5_mla` primary profile and must be unaffected.
GLM53 = "zai-org/GLM-5.3-FP8"

PACKED_TOKEN_BYTES = 656
UNPACKED_SOURCE_DIM = 576


@dataclass
class _FakeGPUPagedKVConfig:
    """Same fields ``build_gpu_kv_config`` constructs, so ``replace`` works.

    A dataclass rather than the test-wide ``object`` stub of
    ``test_glm52_host_kv_profile.py``: ``_prefill_pool_geometry`` returns
    ``dataclasses.replace(geometry, ...)``, which needs a real dataclass.
    """

    num_layers: int
    num_pages: int
    page_size_tokens: int
    num_k_heads: int
    k_head_dim: int
    num_v_heads: Optional[int]
    v_head_dim: Optional[int]
    kv_dtype: torch.dtype
    logical_to_physical_layer: Optional[Sequence[int]] = None


def _install_host_kv_config(monkeypatch):
    """Load the real config module by path and publish it under its dotted name.

    The worker method under test does a function-local
    ``from batchgen.kv_cache.host_kv_mananger_config import ...``, so the
    module has to be reachable by that name; the parent packages are stubs
    because importing the real ``batchgen`` package needs CUDA extensions.
    """

    gpu_manager = types.ModuleType("batchgen.kv_cache.gpu_paged_kv_manager")
    gpu_manager.GPUPagedKVConfig = _FakeGPUPagedKVConfig
    engine_loader = types.ModuleType("batchgen.models.engine_loader")
    engine_loader.core_engine = types.SimpleNamespace()

    for name, module in (
        ("batchgen", types.ModuleType("batchgen")),
        ("batchgen.kv_cache", types.ModuleType("batchgen.kv_cache")),
        ("batchgen.models", types.ModuleType("batchgen.models")),
        ("batchgen.kv_cache.gpu_paged_kv_manager", gpu_manager),
        ("batchgen.models.engine_loader", engine_loader),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    spec = importlib.util.spec_from_file_location(
        "batchgen.kv_cache.host_kv_mananger_config", HOST_KV_CONFIG_PATH
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(
        sys.modules, "batchgen.kv_cache.host_kv_mananger_config", module
    )
    spec.loader.exec_module(module)
    return module


def _worker_tree():
    return ast.parse(WORKER_PATH.read_text())


def _class_def(tree, name):
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"class {name} not found")


def _method_def(tree, class_name, method_name):
    for node in _class_def(tree, class_name).body:
        if isinstance(node, ast.FunctionDef) and node.name == method_name:
            return node
    raise AssertionError(f"{class_name}.{method_name} not found")


def _compile_method(method_node):
    """Compile ONE method out of the real worker source and return the function."""

    module = ast.Module(body=[method_node], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"torch": torch, "replace": replace}
    exec(  # noqa: S102 - compiling a named function out of the repo's own source
        compile(module, str(WORKER_PATH), "exec"), namespace
    )
    return namespace[method_node.name]


def _prefill_pool_geometry_fn():
    return _compile_method(
        _method_def(_worker_tree(), "BatchGenWorker", "_prefill_pool_geometry")
    )


class _WorkerStub:
    def __init__(self, model_name: str):
        self.huggingface_ckpt_name = model_name


# --------------------------------------------------------------------------
# Task 1: the prefill ACTIVATION pool geometry
# --------------------------------------------------------------------------


def test_pool_geometry_unpacks_the_glm52_record_to_bf16_576(monkeypatch):
    config = _install_host_kv_config(monkeypatch)
    geometry_fn = _prefill_pool_geometry_fn()

    paged = config.build_gpu_kv_config(GLM52, [64])
    assert paged.kv_dtype is torch.uint8
    assert paged.k_head_dim == PACKED_TOKEN_BYTES

    pool = geometry_fn(_WorkerStub(GLM52), 64)
    # The pool holds the activation, which is what sparse_prefill writes.
    assert pool.kv_dtype is torch.bfloat16
    assert pool.k_head_dim == UNPACKED_SOURCE_DIM
    # Everything the pool shares with the paged cache is untouched: the layer
    # count is cross-checked against the model in `_prefill_pool_layout`, and
    # the page size is the same for both formats.
    assert pool.num_layers == paged.num_layers
    assert pool.page_size_tokens == paged.page_size_tokens
    assert pool.num_k_heads == paged.num_k_heads


def test_pool_geometry_page_bytes_halve_rather_than_following_the_record():
    """656 B/token paged vs 576 x 2 B/token pooled -- the numbers must differ."""

    assert PACKED_TOKEN_BYTES != UNPACKED_SOURCE_DIM * 2


def test_pool_geometry_is_identity_for_a_bf16_profile(monkeypatch):
    """GLM-5.3 (and every non-packed model) must see no substitution at all."""

    config = _install_host_kv_config(monkeypatch)
    geometry_fn = _prefill_pool_geometry_fn()

    paged = config.build_gpu_kv_config(GLM53, [64])
    pool = geometry_fn(_WorkerStub(GLM53), 64)
    assert pool == paged
    assert pool.kv_dtype is torch.bfloat16
    assert pool.k_head_dim == UNPACKED_SOURCE_DIM


def test_pool_geometry_is_identity_when_the_kill_switch_is_off(monkeypatch):
    """With packed KV disabled the pool path is byte-identical to pre-migration."""

    config = _install_host_kv_config(monkeypatch)
    config.set_glm52_packed_kv(False)
    geometry_fn = _prefill_pool_geometry_fn()

    paged = config.build_gpu_kv_config(GLM52, [64])
    pool = geometry_fn(_WorkerStub(GLM52), 64)
    assert pool == paged
    assert pool.kv_dtype is torch.bfloat16
    assert pool.k_head_dim == UNPACKED_SOURCE_DIM


def test_pool_geometry_names_a_profile_geometry_disagreement(monkeypatch):
    """A packed profile whose geometry is not the packed record must raise.

    No silent fallback: substituting blindly would hand the pool a width that
    nothing writes.
    """

    config = _install_host_kv_config(monkeypatch)
    geometry_fn = _prefill_pool_geometry_fn()

    real_build = config.build_gpu_kv_config

    def _drifted(model_name, sequence_tokens):
        geometry = real_build(model_name, sequence_tokens)
        return replace(geometry, kv_dtype=torch.float8_e4m3fn)

    monkeypatch.setattr(config, "build_gpu_kv_config", _drifted)
    with pytest.raises(RuntimeError, match="not the packed record"):
        geometry_fn(_WorkerStub(GLM52), 64)


# --------------------------------------------------------------------------
# Task 3: arg -> latch plumbing, including its ordering property
# --------------------------------------------------------------------------

FLAG = "disable_glm52_packed_kv"
CLI_FLAG = "--disable-glm52-packed-kv"


def _server_args_module():
    """Import the leaf module without running the HTTP server package init.

    Same shim as ``tests/test_server_args.py``.
    """

    package_name = "batchgen.server"
    previous = sys.modules.get(package_name)
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT / "batchgen" / "server")]
    sys.modules[package_name] = package
    try:
        import importlib

        return importlib.import_module("batchgen.server.server_args")
    finally:
        if previous is None:
            sys.modules.pop(package_name, None)
        else:
            sys.modules[package_name] = previous


def test_cli_round_trip_defaults_to_packed_and_the_flag_turns_it_off():
    """The real parser, not the source text: `--disable-...` must reach the field."""

    parser = _server_args_module()._build_parser()
    assert getattr(parser.parse_args(["--model", "m"]), FLAG) is False
    assert getattr(parser.parse_args(["--model", "m", CLI_FLAG]), FLAG) is True


def test_server_args_declares_the_switch_and_defaults_to_packed():
    source = SERVER_ARGS_PATH.read_text()
    tree = ast.parse(source)

    field_defaults = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "ServerArgs":
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(
                    stmt.target, ast.Name
                ):
                    field_defaults[stmt.target.id] = stmt.value
    assert FLAG in field_defaults, "ServerArgs is missing the switch"
    default = field_defaults[FLAG]
    assert isinstance(default, ast.Constant) and default.value is False, (
        "the default must be False, i.e. the PACKED record"
    )
    assert CLI_FLAG in source, "the argparse flag is missing"
    assert f"{FLAG}=parsed.{FLAG}" in source, (
        "parsed value is not carried into the ServerArgs construction"
    )


def test_worker_args_carry_the_switch_from_the_server():
    worker_source = WORKER_PATH.read_text()
    manager_source = WORKER_MANAGER_PATH.read_text()

    worker_fields = {}
    for node in _class_def(_worker_tree(), "BatchGenWorkerArgs").body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            worker_fields[node.target.id] = node.value
    assert FLAG in worker_fields, "BatchGenWorkerArgs is missing the switch"
    default = worker_fields[FLAG]
    assert isinstance(default, ast.Constant) and default.value is False

    assert f"{FLAG}=self.args.{FLAG}" in manager_source, (
        "WorkerManager does not pass the switch into BatchGenWorkerArgs"
    )
    assert f"args.{FLAG}" in worker_source


def _first_lineno(node, predicate):
    hits = [
        child.lineno
        for child in ast.walk(node)
        if hasattr(child, "lineno") and predicate(child)
    ]
    return min(hits) if hits else None


def _latch_lineno(node):
    return _first_lineno(
        node,
        lambda child: isinstance(child, ast.Call)
        and isinstance(child.func, ast.Name)
        and child.func.id.endswith("set_glm52_packed_kv"),
    )


# Names whose evaluation resolves a KV profile. `_resolve_profile` latches, so
# any of these running first would commit the default.
_RESOLVING_NAMES = (
    "_resolve_profile",
    "build_host_kv_config",
    "build_host_kv_worker_view",
    "build_gpu_kv_config",
    "build_gpu_kv_config_fixed_size",
    "from_budget",
    "build_host_configs",
    "create_managers",
)


def _first_resolution_lineno(node):
    def _is_resolution(child):
        if not isinstance(child, ast.Call):
            return False
        func = child.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
        return name in _RESOLVING_NAMES

    return _first_lineno(node, _is_resolution)


def test_worker_process_sets_the_latch_before_its_first_resolution():
    init = _method_def(_worker_tree(), "BatchGenWorker", "__init__")
    latch = _latch_lineno(init)
    assert latch is not None, "BatchGenWorker.__init__ never sets the latch"
    resolution = _first_resolution_lineno(init)
    assert resolution is not None, (
        "expected BatchGenWorker.__init__ to resolve a KV profile; if that "
        "moved, this ordering test must follow it"
    )
    assert latch < resolution, (
        f"latch at line {latch} runs AFTER the first profile resolution at "
        f"line {resolution}"
    )


def test_server_process_sets_the_latch_in_the_worker_manager_constructor():
    """`WorkerManager.__init__` dominates every resolution in that process.

    `allocate_host_kv_cache` and `_initialize_prefix_cache_owner` are the only
    resolving sites in `worker_manager.py`, and both are reached from
    `_start_impl`, i.e. strictly after the constructor.
    """

    tree = ast.parse(WORKER_MANAGER_PATH.read_text())
    init = _method_def(tree, "WorkerManager", "__init__")
    assert _latch_lineno(init) is not None, (
        "WorkerManager.__init__ never sets the latch"
    )
    assert _first_resolution_lineno(init) is None, (
        "WorkerManager.__init__ itself resolves a KV profile; the latch call "
        "must be re-checked against it"
    )
    for method in ("allocate_host_kv_cache", "_initialize_prefix_cache_owner"):
        assert _first_resolution_lineno(_method_def(tree, "WorkerManager", method))


def test_the_latch_rejects_a_post_resolution_change(monkeypatch):
    """A wrong call ORDER must raise, not quietly keep the committed format."""

    config = _install_host_kv_config(monkeypatch)
    assert config._resolve_profile(GLM52) is config._GLM5_2_MLA_PACKED_PROFILE
    with pytest.raises(RuntimeError, match="cannot change"):
        config.set_glm52_packed_kv(False)


def test_disable_true_selects_the_pre_migration_bf16_profile(monkeypatch):
    """The exact expression both processes evaluate: `not disable_...`."""

    config = _install_host_kv_config(monkeypatch)
    config.set_glm52_packed_kv(not True)  # disable_glm52_packed_kv=True
    assert config._resolve_profile(GLM52) is config._GLM5_MLA_PROFILE
    assert config._resolve_profile(GLM53) is config._GLM5_MLA_PROFILE


def test_disable_false_selects_the_packed_profile(monkeypatch):
    config = _install_host_kv_config(monkeypatch)
    config.set_glm52_packed_kv(not False)
    assert config._resolve_profile(GLM52) is config._GLM5_2_MLA_PACKED_PROFILE
    assert config._resolve_profile(GLM53) is config._GLM5_MLA_PROFILE


# --------------------------------------------------------------------------
# Task 2: the warmup's reachability and its synthetic-input invariants
# --------------------------------------------------------------------------


def test_sparse_fp8_warmup_is_reached_from_every_pre_forward_point():
    tree = _worker_tree()
    for method in (
        # before `_setup_cuda_graphs` enters a capture
        "_warmup_cuda_graphs",
        # the startup path of the --persistent-phase-instances lane
        "_build_persistent_phase_instances",
        # the decode-configure path, which covers the graphs-DISABLED lane
        "generate",
    ):
        node = _method_def(tree, "BatchGenWorker", method)
        calls = [
            child
            for child in ast.walk(node)
            if isinstance(child, ast.Call)
            and getattr(child.func, "attr", None)
            == "_warmup_packed_fp8_sparse_mla"
        ]
        assert calls, f"{method} does not warm the sparse-FP8 MLA kernel"

    warmup = _method_def(tree, "BatchGenWorker", "_warmup_cuda_graphs")
    warm_line = _first_lineno(
        warmup,
        lambda child: isinstance(child, ast.Call)
        and getattr(child.func, "attr", None) == "_warmup_packed_fp8_sparse_mla",
    )
    setup_line = _first_lineno(
        warmup,
        lambda child: isinstance(child, ast.Call)
        and getattr(child.func, "attr", None) == "_setup_cuda_graphs",
    )
    assert warm_line is not None and setup_line is not None
    assert warm_line < setup_line, (
        "the kernel must be warmed BEFORE _setup_cuda_graphs enters a capture"
    )


def test_sparse_fp8_warmup_never_passes_an_all_minus_one_indices_row():
    """A row of only -1 returns NaN (epilogue divides by a zero denominator)."""

    source = ast.get_source_segment(
        WORKER_PATH.read_text(),
        _method_def(
            _worker_tree(), "BatchGenWorker", "_warmup_packed_fp8_sparse_mla"
        ),
    )
    assert "indices[..., 0] = 0" in source, (
        "the warmup row must hold at least one valid index"
    )
    assert "is_current_stream_capturing" in source, (
        "the warmup must refuse to run inside a capture"
    )
