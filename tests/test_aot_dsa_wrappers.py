"""The DSA indexer wrappers import their AOT extensions, once, and never JIT.

``batchgen_kernels.load_extension`` is replaced by a recorder, so these run on
any host: no CUDA, no built extension, no Triton. The real
``batchgen_kernels.attention.dsa`` package ``__init__`` imports Triton kernels,
so it is replaced by a bare package that keeps the real ``__path__``; the
wrapper modules themselves are the real files.
"""

from __future__ import annotations

import importlib
import pathlib
import sys
import types

import pytest
import torch

import batchgen_kernels

ROOT = pathlib.Path(__file__).resolve().parents[1]
_DSA_DIR = ROOT / "batchgen_kernels" / "attention" / "dsa"
_DSA = "batchgen_kernels.attention.dsa"

HADAMARD = "batchgen_kernels.attention._C_fast_hadamard_transform"
ROPE_HADAMARD = "batchgen_kernels.attention._C_fused_rope_hadamard"
FAST_TOPK = "batchgen_kernels.attention._C_dsa_fast_topk"
KV_PROJ = "batchgen_kernels.attention._C_indexer_kv_proj_wgmma"

_FRESH_MODULES = (
    f"{_DSA}.indexer",
    f"{_DSA}.indexer.indexer_k_write_fp8",
    f"{_DSA}.indexer.indexer_k_norm_rope_write_fp8",
    f"{_DSA}.indexer_k_quant",
    f"{_DSA}.fast_topk_cuda",
    f"{_DSA}.fused_indexer_kv_proj_cuda",
    "batchgen.other_kernels.hadamard_transform",
)


class _Recorder:
    """Stands in for ``load_extension``; hands out one fake module per name."""

    def __init__(self):
        self.loads: list[tuple[str, bool]] = []
        self.calls: list[tuple[str, str, tuple]] = []
        self.missing: set[str] = set()
        self.modules: dict[str, types.SimpleNamespace] = {}

    def _module(self, module_name: str) -> types.SimpleNamespace:
        if module_name not in self.modules:
            recorder = self

            class _Fake(types.SimpleNamespace):
                def __getattr__(self, attr):
                    if attr.startswith("_"):
                        raise AttributeError(attr)

                    def call(*args, **kwargs):
                        recorder.calls.append((module_name, attr, args))
                        return (module_name, attr)

                    return call

            self.modules[module_name] = _Fake()
        return self.modules[module_name]

    def __call__(self, module_name: str, allow_dev_jit: bool = True):
        self.loads.append((module_name, allow_dev_jit))
        if module_name in self.missing:
            raise ImportError(f"No module named {module_name!r}")
        return self._module(module_name)


@pytest.fixture
def recorder(monkeypatch):
    package = types.ModuleType(_DSA)
    package.__path__ = [str(_DSA_DIR)]
    package.__package__ = _DSA
    monkeypatch.setitem(sys.modules, _DSA, package)
    # Modules imported here hold the fake loader's modules; none may outlive
    # the test, and whatever was imported before must come back.
    saved = {name: sys.modules.pop(name) for name in _FRESH_MODULES if name in sys.modules}
    fake = _Recorder()
    monkeypatch.setattr(batchgen_kernels, "load_extension", fake)
    yield fake
    for name in _FRESH_MODULES:
        sys.modules.pop(name, None)
        parent_name, _, child = name.rpartition(".")
        parent = sys.modules.get(parent_name)
        if parent is None:
            continue
        if name in saved:
            setattr(parent, child, saved[name])
        elif getattr(parent, child, None) is not None:
            delattr(parent, child)
    sys.modules.update(saved)


def _import(name: str):
    return importlib.import_module(name)


def test_importing_the_indexer_package_loads_nothing(recorder):
    _import(f"{_DSA}.indexer")
    _import(f"{_DSA}.fast_topk_cuda")
    _import(f"{_DSA}.fused_indexer_kv_proj_cuda")
    assert recorder.loads == []


def test_hadamard_transform_loads_its_extension_once(recorder):
    indexer = _import(f"{_DSA}.indexer")
    x = torch.zeros(2, 128)
    assert indexer.hadamard_transform(x, scale=0.5) == (HADAMARD, "fast_hadamard_transform")
    indexer.hadamard_transform(x)
    assert recorder.loads == [(HADAMARD, False)]
    assert [call[:2] for call in recorder.calls] == [
        (HADAMARD, "fast_hadamard_transform"),
        (HADAMARD, "fast_hadamard_transform"),
    ]


def test_rope_hadamard_entry_points_share_one_load(recorder):
    indexer = _import(f"{_DSA}.indexer")
    x = torch.zeros(2, 128)
    cos = sin = torch.zeros(4, 64)
    positions = torch.zeros(2, dtype=torch.int64)
    out = torch.empty(2, 128)

    indexer.fused_rope_hadamard(x, cos, sin, positions)
    assert indexer.fused_rope_hadamard_out(x, cos, sin, positions, out) is out
    strided = torch.zeros(1, 2, 128)
    strided_out = torch.empty(1, 2, 128)
    assert (
        indexer.fused_rope_hadamard_strided_out(
            strided, cos, sin, positions, strided_out, 256
        )
        is strided_out
    )

    assert recorder.loads == [(ROPE_HADAMARD, False)]
    assert [call[1] for call in recorder.calls] == [
        "fused_rope_hadamard",
        "fused_rope_hadamard_out",
        "fused_rope_hadamard_strided_out",
    ]
    # Positional forwarding is unchanged, including the defaults.
    assert recorder.calls[0][2][4:] == (128 ** -0.5, True)
    assert recorder.calls[1][2][5:] == (128 ** -0.5, None, True)
    assert recorder.calls[2][2][5:] == (256, 128 ** -0.5, None, True)


def test_load_hadamard_extensions_loads_both(recorder):
    indexer = _import(f"{_DSA}.indexer")
    indexer.load_hadamard_extensions()
    indexer.load_hadamard_extensions()
    assert recorder.loads == [(HADAMARD, False), (ROPE_HADAMARD, False)]


def test_fast_topk_loads_its_extension_once(recorder):
    topk = _import(f"{_DSA}.fast_topk_cuda")
    first = topk._get_module()
    assert topk._get_module() is first
    assert recorder.loads == [(FAST_TOPK, False)]


def test_build_module_returns_the_memoised_aot_module(recorder):
    kv_proj = _import(f"{_DSA}.fused_indexer_kv_proj_cuda")
    first = kv_proj.build_module()
    assert kv_proj.build_module(4) is first
    assert kv_proj.build_module(num_stages=4) is first
    assert recorder.loads == [(KV_PROJ, False)]


@pytest.mark.parametrize("num_stages", (1, 3, 5, 8))
def test_build_module_rejects_other_pipeline_depths(recorder, num_stages):
    kv_proj = _import(f"{_DSA}.fused_indexer_kv_proj_cuda")
    with pytest.raises(ValueError, match="NUM_STAGES=4"):
        kv_proj.build_module(num_stages)
    assert recorder.loads == []


def test_other_kernels_reexports_the_same_functions(recorder):
    indexer = _import(f"{_DSA}.indexer")
    legacy = _import("batchgen.other_kernels.hadamard_transform")
    assert legacy.hadamard_transform is indexer.hadamard_transform
    assert legacy.fused_rope_hadamard is indexer.fused_rope_hadamard
    # Imported eagerly, so the model's import-time availability check holds.
    assert recorder.loads == [(HADAMARD, False), (ROPE_HADAMARD, False)]


@pytest.mark.parametrize("missing", (HADAMARD, ROPE_HADAMARD))
def test_other_kernels_import_fails_when_an_extension_is_missing(recorder, missing):
    recorder.missing.add(missing)
    with pytest.raises(ImportError, match=missing.rsplit(".", 1)[-1]):
        _import("batchgen.other_kernels.hadamard_transform")
