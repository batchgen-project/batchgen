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
    ptrs_gemm = object()
    fused_s1_ptrs = object()
    extension = SimpleNamespace(
        fp8_blockwise_grouped_gemm=grouped_gemm,
        fp8_blockwise_fused_s1=fused_s1,
        fp8_blockwise_grouped_gemm_ptrs=ptrs_gemm,
        fp8_blockwise_fused_s1_ptrs=fused_s1_ptrs,
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
    assert fp8_moe._get_ptrs_kernel() is ptrs_gemm
    assert fp8_moe._get_fused_s1_ptrs_kernel() is fused_s1_ptrs
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


def test_missing_pointer_array_kernel_fails_closed(fp8_moe, monkeypatch):
    extension = SimpleNamespace(fp8_blockwise_grouped_gemm=object())
    monkeypatch.setitem(
        sys.modules,
        "batchgen_kernels",
        SimpleNamespace(load_extension=lambda _name: extension),
    )

    with pytest.raises(RuntimeError, match="pointer-array grouped GEMM"):
        fp8_moe.grouped_fp8_blockwise_gemm_ptrs(
            None, None, None, None, None, None, None, None, 64
        )


def test_pointer_array_wrapper_forwards_exact_abi(fp8_moe, monkeypatch):
    calls = []

    def ptrs_kernel(*args):
        calls.append(args)
        return "result"

    extension = SimpleNamespace(fp8_blockwise_grouped_gemm_ptrs=ptrs_kernel)
    monkeypatch.setitem(
        sys.modules,
        "batchgen_kernels",
        SimpleNamespace(load_extension=lambda _name: extension),
    )
    args = tuple(object() for _ in range(8))
    output = object()

    assert (
        fp8_moe.grouped_fp8_blockwise_gemm_ptrs(
            *args, 40, output=output
        )
        == "result"
    )
    assert calls == [(*args, 64, output)]


def test_pointer_array_fused_s1_forwards_exact_abi(fp8_moe, monkeypatch):
    calls = []

    def fused_s1_ptrs(*args):
        calls.append(args)
        return "s1"

    extension = SimpleNamespace(fp8_blockwise_fused_s1_ptrs=fused_s1_ptrs)
    monkeypatch.setitem(
        sys.modules,
        "batchgen_kernels",
        SimpleNamespace(load_extension=lambda _name: extension),
    )
    # x/x-scale + four prototype/pointer pairs + seqlens/cu_seqlens
    args = tuple(object() for _ in range(12))
    output = object()

    assert (
        fp8_moe.grouped_fp8_blockwise_fused_s1_ptrs(
            *args, 40, output=output
        )
        == "s1"
    )
    # Python API reorders x_scale behind routing metadata for the C++ ABI.
    expected = (
        args[0], args[2], args[3], args[4], args[5],
        args[10], args[11], args[1], args[6], args[7], args[8], args[9],
        64, output,
    )
    assert calls == [expected]


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
