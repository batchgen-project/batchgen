"""CPU-only tests for FP8 blockwise grouped GEMM extension loading."""

import inspect
import sys
import types
from unittest import mock

import pytest
import torch

import batchgen.moe.grouped_fp8_blockwise_moe as fp8_moe


@pytest.fixture
def loader(monkeypatch):
    """Install a fake batchgen_kernels with a mock load_extension; reset memo."""
    monkeypatch.setattr(fp8_moe, "_module", None)
    monkeypatch.setattr(fp8_moe, "_module_loaded", False)
    monkeypatch.setattr(fp8_moe, "_warned_gemm", False)
    monkeypatch.setattr(fp8_moe, "_warned_fused_s1", False)
    monkeypatch.setattr(fp8_moe, "_warned_ptrs", False)
    monkeypatch.setattr(fp8_moe, "_warned_fused_s1_ptrs", False)
    monkeypatch.setattr(fp8_moe, "logger", mock.MagicMock())
    fake_pkg = types.ModuleType("batchgen_kernels")
    fake_pkg.load_extension = mock.MagicMock()
    monkeypatch.setitem(sys.modules, "batchgen_kernels", fake_pkg)
    return fake_pkg.load_extension


def _s1_args():
    x = torch.ones(4, 8)
    return dict(
        x_fp8=x, x_scale=torch.ones(1, 4),
        gate_w3d=torch.full((2, 3, 8), 2.0), up_w3d=torch.full((2, 3, 8), 3.0),
        gate_ws3d=torch.ones(2, 1, 4), up_ws3d=torch.ones(2, 1, 4),
        seqlens=torch.full((2,), 2, dtype=torch.int32),
        cu_seqlens=torch.tensor([0, 2, 4], dtype=torch.int32),
        num_seq_per_group_avg=2,
    )


def _fake_gemm(x_fp8, weight_3d, *args):
    return torch.full((x_fp8.shape[0], weight_3d.shape[1]), weight_3d[0, 0, 0].item())


def test_source_has_no_direct_extension_imports():
    src = inspect.getsource(fp8_moe)
    assert "from batchgen_kernels.moe._C_fp8_blockwise_gemm import" not in src


def test_one_loader_call_for_all_symbols(loader):
    gemm, fused, ptrs, fused_ptrs = object(), object(), object(), object()
    loader.return_value = types.SimpleNamespace(
        fp8_blockwise_grouped_gemm=gemm,
        fp8_blockwise_fused_s1=fused,
        fp8_blockwise_grouped_gemm_ptrs=ptrs,
        fp8_blockwise_fused_s1_ptrs=fused_ptrs,
    )
    assert fp8_moe._get_kernel() is gemm
    assert fp8_moe._get_fused_s1_kernel() is fused
    assert fp8_moe._get_ptrs_kernel() is ptrs
    assert fp8_moe._get_fused_s1_ptrs_kernel() is fused_ptrs
    assert fp8_moe._get_kernel() is gemm
    loader.assert_called_once_with("batchgen_kernels.moe._C_fp8_blockwise_gemm")
    fp8_moe.logger.warning.assert_not_called()


def test_missing_extension_warns_once_and_gemm_fails_closed(loader):
    loader.side_effect = ImportError("no module")
    assert fp8_moe._get_kernel() is None
    assert fp8_moe._get_kernel() is None
    with pytest.raises(RuntimeError, match="not compiled"):
        fp8_moe.grouped_fp8_blockwise_s3(
            torch.ones(4, 3), torch.ones(1, 4), torch.ones(2, 8, 3),
            torch.ones(2, 1, 4), torch.ones(2, dtype=torch.int32),
            torch.tensor([0, 2, 4], dtype=torch.int32), 2,
        )
    loader.assert_called_once()
    assert fp8_moe.logger.warning.call_count == 1


def test_loader_runtime_error_propagates(loader):
    loader.side_effect = RuntimeError("JIT build failed")
    with pytest.raises(RuntimeError, match="JIT build failed"):
        fp8_moe._get_kernel()
    with pytest.raises(RuntimeError, match="JIT build failed"):
        fp8_moe._get_fused_s1_kernel()
    fp8_moe.logger.warning.assert_not_called()


def test_missing_fused_symbol_with_output_rejects(loader):
    gemm = mock.MagicMock(side_effect=_fake_gemm)
    loader.return_value = types.SimpleNamespace(fp8_blockwise_grouped_gemm=gemm)
    with pytest.raises(RuntimeError, match="output buffer was supplied"):
        fp8_moe.grouped_fp8_blockwise_fused_s1(
            **_s1_args(), output=torch.empty(4, 3),
        )
    gemm.assert_not_called()


def test_fused_symbol_forwards_output(loader):
    output = torch.empty(4, 3)
    fused = mock.MagicMock(return_value=output)
    loader.return_value = types.SimpleNamespace(fp8_blockwise_fused_s1=fused)
    assert fp8_moe.grouped_fp8_blockwise_fused_s1(
        **_s1_args(), output=output,
    ) is output
    assert fused.call_args.args[-1] is output


def test_missing_fused_symbol_without_output_uses_allocating_fallback(loader):
    gemm = mock.MagicMock(side_effect=_fake_gemm)
    loader.return_value = types.SimpleNamespace(fp8_blockwise_grouped_gemm=gemm)
    out = fp8_moe.grouped_fp8_blockwise_fused_s1(**_s1_args())
    gate, up = torch.full((4, 3), 2.0), torch.full((4, 3), 3.0)
    torch.testing.assert_close(out, torch.nn.functional.silu(gate) * up)
    assert gemm.call_count == 2
    assert all(c.args[-1] is None for c in gemm.call_args_list)  # tma_desc
    assert all(c.args[-2] is None for c in gemm.call_args_list)  # output
    loader.assert_called_once()
    assert fp8_moe.logger.warning.call_count == 1
