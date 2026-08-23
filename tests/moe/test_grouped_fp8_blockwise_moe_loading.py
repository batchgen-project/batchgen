import importlib.util
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


EXTENSION_NAME = "batchgen_kernels.moe._C_fp8_blockwise_gemm"
WRAPPER_PATH = (
    Path(__file__).resolve().parents[2]
    / "batchgen/moe/grouped_fp8_blockwise_moe.py"
)


@pytest.fixture
def fp8_moe():
    spec = importlib.util.spec_from_file_location(
        "_fp8_moe_loading_test", WRAPPER_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_kernel_module_loads_once_and_serves_both_symbols(fp8_moe, monkeypatch):
    grouped_gemm = object()
    fused_s1 = object()
    extension = SimpleNamespace(
        fp8_blockwise_grouped_gemm=grouped_gemm,
        fp8_blockwise_fused_s1=fused_s1,
    )
    calls = []

    def load_extension(module_name):
        calls.append(module_name)
        return extension

    monkeypatch.setitem(
        sys.modules,
        "batchgen_kernels",
        SimpleNamespace(load_extension=load_extension),
    )

    assert fp8_moe._get_kernel() is grouped_gemm
    assert fp8_moe._get_fused_s1_kernel() is fused_s1
    assert calls == [EXTENSION_NAME]


def test_missing_grouped_kernel_warns_once_and_fails_closed(
    fp8_moe, monkeypatch, caplog
):
    def load_extension(_module_name):
        raise ImportError("extension unavailable")

    monkeypatch.setitem(
        sys.modules,
        "batchgen_kernels",
        SimpleNamespace(load_extension=load_extension),
    )

    with caplog.at_level(logging.WARNING, logger="batchgen.moe.fp8_blockwise"):
        assert fp8_moe._get_kernel() is None
        with pytest.raises(RuntimeError, match="kernel not compiled"):
            fp8_moe.grouped_fp8_blockwise_gemm(
                None, None, None, None, None, None, 64
            )

    warnings = [
        record.getMessage()
        for record in caplog.records
        if "grouped GEMM kernel not available" in record.getMessage()
    ]
    assert len(warnings) == 1
    assert "Triton" not in warnings[0]


def test_jit_build_error_propagates(fp8_moe, monkeypatch):
    def load_extension(_module_name):
        raise RuntimeError("ninja: build stopped")

    monkeypatch.setitem(
        sys.modules,
        "batchgen_kernels",
        SimpleNamespace(load_extension=load_extension),
    )

    with pytest.raises(RuntimeError, match="ninja"):
        fp8_moe._get_kernel()


def test_missing_fused_symbol_preserves_unfused_allocating_fallback(
    fp8_moe, monkeypatch
):
    extension = SimpleNamespace(fp8_blockwise_grouped_gemm=object())
    monkeypatch.setitem(
        sys.modules,
        "batchgen_kernels",
        SimpleNamespace(load_extension=lambda _name: extension),
    )
    fallback_result = object()
    monkeypatch.setattr(
        fp8_moe,
        "grouped_fp8_blockwise_s1_silu",
        lambda *args, **kwargs: fallback_result,
    )

    assert (
        fp8_moe.grouped_fp8_blockwise_fused_s1(
            None, None, None, None, None, None, None, None, 64
        )
        is fallback_result
    )


def test_missing_fused_symbol_rejects_persistent_output(fp8_moe, monkeypatch):
    extension = SimpleNamespace(fp8_blockwise_grouped_gemm=object())
    monkeypatch.setitem(
        sys.modules,
        "batchgen_kernels",
        SimpleNamespace(load_extension=lambda _name: extension),
    )

    with pytest.raises(RuntimeError, match="persistent buffer"):
        fp8_moe.grouped_fp8_blockwise_fused_s1(
            None, None, None, None, None, None, None, None, 64, output=object()
        )
