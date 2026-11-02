"""The FP8 WGMMA MoE pipeline imports its AOT extensions, once, and never JITs.

``batchgen_kernels.load_extension`` is replaced by a recorder, so these run on
any host: no CUDA and no built extension. The binding checks read the ``.cu``
sources and the pipeline module as text and ``ast``.
"""

from __future__ import annotations

import ast
import pathlib
import re
import types

import pytest

import batchgen_kernels
from batchgen.moe import dispatch_scatter_3d
from batchgen.moe import fp8_wgmma_pipeline as pipeline

ROOT = pathlib.Path(__file__).resolve().parents[1]
_PIPELINE = ROOT / "batchgen" / "moe" / "fp8_wgmma_pipeline.py"
_MOE_SRC = ROOT / "batchgen_kernels" / "src" / "moe"

GROUPED_GEMM = "batchgen_kernels.moe._C_fp8_wgmma_grouped_gemm"
QUANT_OPS = "batchgen_kernels.moe._C_fp8_wgmma_quant_ops"
DISPATCH = "batchgen_kernels.moe._C_dispatch_scatter_3d"

# WGMMAMoEBuffers attribute holding each module -> the source that binds it.
_MODULE_SOURCES = {
    "wgmma_mod": _MOE_SRC / "fp8_wgmma_grouped_gemm.cu",
    "fast_mod": _MOE_SRC / "fp8_wgmma_quant_ops.cu",
    "dr_mod": _MOE_SRC / "dispatch_scatter_3d.cu",
}


class _Recorder:
    """Stands in for ``load_extension``; hands out one fake module per name."""

    def __init__(self):
        self.loads: list[tuple[str, bool]] = []
        self.missing: set[str] = set()
        self.modules: dict[str, types.SimpleNamespace] = {}

    def __call__(self, module_name: str, allow_dev_jit: bool = True):
        self.loads.append((module_name, allow_dev_jit))
        if module_name in self.missing:
            raise ImportError(f"No module named {module_name!r}")
        return self.modules.setdefault(
            module_name, types.SimpleNamespace(name=module_name)
        )


@pytest.fixture
def recorder(monkeypatch):
    fake = _Recorder()
    monkeypatch.setattr(batchgen_kernels, "load_extension", fake)
    monkeypatch.setattr(pipeline, "_module_cache", {})
    monkeypatch.setattr(dispatch_scatter_3d, "_dispatch_reduce_module", None)
    return fake


def test_build_all_modules_loads_each_aot_extension_once(recorder):
    wgmma_mod, fast_mod, dr_mod = pipeline.build_all_modules()
    assert pipeline.build_all_modules() == (wgmma_mod, fast_mod, dr_mod)

    assert wgmma_mod is recorder.modules[GROUPED_GEMM]
    assert fast_mod is recorder.modules[QUANT_OPS]
    assert [name for name, _ in recorder.loads] == [GROUPED_GEMM, QUANT_OPS, DISPATCH]
    new_extension_loads = [load for load in recorder.loads if load[0] != DISPATCH]
    assert new_extension_loads == [(GROUPED_GEMM, False), (QUANT_OPS, False)]


def test_dispatch_module_comes_from_dispatch_scatter_3d(recorder):
    _, _, dr_mod = pipeline.build_all_modules()
    assert dr_mod is recorder.modules[DISPATCH]
    assert dispatch_scatter_3d.require_dispatch_scatter_3d_kernels() is dr_mod
    assert [name for name, _ in recorder.loads].count(DISPATCH) == 1


@pytest.mark.parametrize(
    "missing, cache_key", ((GROUPED_GEMM, "wgmma"), (QUANT_OPS, "fast"))
)
def test_a_missing_aot_extension_fails_and_is_not_cached(recorder, missing, cache_key):
    recorder.missing.add(missing)
    with pytest.raises(ImportError, match=missing.rsplit(".", 1)[-1]):
        pipeline.build_all_modules()
    assert cache_key not in pipeline._module_cache
    assert (missing, False) in recorder.loads

    recorder.missing.clear()
    wgmma_mod, fast_mod, _ = pipeline.build_all_modules()
    assert wgmma_mod is recorder.modules[GROUPED_GEMM]
    assert fast_mod is recorder.modules[QUANT_OPS]


def test_a_missing_dispatch_extension_fails_closed(recorder):
    recorder.missing.add(DISPATCH)
    with pytest.raises(RuntimeError, match="dispatch_scatter_3d"):
        pipeline.build_all_modules()


def test_pipeline_module_has_no_runtime_compilation():
    tree = ast.parse(_PIPELINE.read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    assert not any(name.startswith("torch.utils.cpp_extension") for name in imported)
    assert "os" not in imported

    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert "load_inline" not in names
    assert "print" not in names

    for constant in (
        "WGMMA_CUDA_SOURCE", "WGMMA_CPP_SOURCE",
        "FAST_CUDA_SOURCE", "FAST_CPP_SOURCE",
        "DISPATCH_REDUCE_CUDA_SOURCE", "DISPATCH_REDUCE_CPP_SOURCE",
    ):
        assert not hasattr(pipeline, constant), constant


def _bound_functions(source: pathlib.Path) -> set[str]:
    text = source.read_text()
    block = text[text.index("PYBIND11_MODULE("):]
    return set(re.findall(r'm\.def\(\s*"(\w+)"', block))


def _pipeline_calls() -> dict[str, set[str]]:
    """``self.<module attr>.<function>`` calls made by WGMMAMoEBuffers."""

    calls: dict[str, set[str]] = {attr: set() for attr in _MODULE_SOURCES}
    for node in ast.walk(ast.parse(_PIPELINE.read_text())):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        owner = node.func.value
        if (
            isinstance(owner, ast.Attribute)
            and isinstance(owner.value, ast.Name)
            and owner.value.id == "self"
            and owner.attr in calls
        ):
            calls[owner.attr].add(node.func.attr)
    return calls


def test_every_pipeline_call_is_bound_by_its_extension():
    calls = _pipeline_calls()
    assert calls["wgmma_mod"] == {
        "create_tma_descriptors_v8b", "create_tma_descriptors_v8c",
        "grouped_fp8_moe_gemm_v8b", "grouped_fp8_moe_gemm_v8c",
    }
    assert calls["fast_mod"] == {"act_quant_3d", "fused_silu_quant_3d"}
    assert calls["dr_mod"] == {"dispatch_scatter_3d", "reduce_weighted_scatter"}
    for attr, source in _MODULE_SOURCES.items():
        missing = calls[attr] - _bound_functions(source)
        assert not missing, f"{source.name} does not bind {sorted(missing)}"


def test_new_extensions_bind_exactly_the_former_load_inline_functions():
    assert _bound_functions(_MODULE_SOURCES["wgmma_mod"]) == {
        "create_tma_descriptors_v8b", "grouped_fp8_moe_gemm_v8b",
        "create_tma_descriptors_v8c", "grouped_fp8_moe_gemm_v8c",
    }
    assert _bound_functions(_MODULE_SOURCES["fast_mod"]) == {
        "act_quant_3d", "fused_silu_quant_3d",
    }
