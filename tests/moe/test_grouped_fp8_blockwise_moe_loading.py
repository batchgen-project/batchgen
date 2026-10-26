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


# --------------------------------------------------------------------------
# Shared S1/S3 TMA plan (one preparation launch for both GEMMs of a layer)
# --------------------------------------------------------------------------
def test_tile_m_mirrors_the_compiled_dispatch():
    for avg, tile_m in ((1, 16), (16, 16), (17, 32), (32, 32), (33, 64),
                        (48, 64), (49, 64), (4096, 64)):
        assert fp8_moe.grouped_fp8_tile_m(avg) == tile_m, avg


def _plan_buffers():
    rows, k1, n1, k3, n3, e = 8, 16, 4, 4, 16, 2
    return dict(
        s1_x=torch.zeros(rows, k1), s1_output=torch.zeros(rows, n1),
        s3_x=torch.zeros(rows, k3), s3_output=torch.zeros(rows, n3),
        seqlens=torch.full((e,), 3, dtype=torch.int32),
        cu_seqlens=torch.tensor([0, 4, 8], dtype=torch.int32),
        s1_tma_desc=torch.zeros(2 * e, 128, dtype=torch.uint8),
        s3_tma_desc=torch.zeros(2 * e, 128, dtype=torch.uint8),
        tiles=torch.zeros(e, dtype=torch.int32),
        cu_tiles=torch.zeros(e + 1, dtype=torch.int32),
    )


def _prepare(loader, bufs, avg=20, **symbols):
    prepare = mock.MagicMock()
    loader.return_value = types.SimpleNamespace(
        fp8_blockwise_prepare_s1_s3_tma=prepare, **symbols)
    plan = fp8_moe.prepare_grouped_fp8_s1_s3_tma(
        bufs["s1_x"], bufs["s1_output"], bufs["s3_x"], bufs["s3_output"],
        bufs["seqlens"], bufs["cu_seqlens"], avg,
        s1_tma_desc=bufs["s1_tma_desc"], s3_tma_desc=bufs["s3_tma_desc"],
        tiles=bufs["tiles"], cu_tiles=bufs["cu_tiles"])
    return prepare, plan


def test_prepare_forwards_every_buffer_once(loader):
    bufs = _plan_buffers()
    prepare, plan = _prepare(loader, bufs, avg=20)
    prepare.assert_called_once()
    args = prepare.call_args.args
    assert args[0] is bufs["s1_x"] and args[1] is bufs["s1_output"]
    assert args[2] is bufs["s3_x"] and args[3] is bufs["s3_output"]
    assert args[4] is bufs["seqlens"] and args[5] is bufs["cu_seqlens"]
    assert args[6] == 20
    assert args[7] is bufs["s1_tma_desc"] and args[8] is bufs["s3_tma_desc"]
    assert args[9] is bufs["tiles"] and args[10] is bufs["cu_tiles"]
    assert plan.tile_m == 32
    assert plan.s1_tma_desc is bufs["s1_tma_desc"]
    assert plan.s3_tma_desc is bufs["s3_tma_desc"]


def test_prepare_without_the_symbol_fails_closed(loader):
    loader.return_value = types.SimpleNamespace()
    bufs = _plan_buffers()
    with pytest.raises(RuntimeError, match="fp8_blockwise_prepare_s1_s3_tma"):
        fp8_moe.prepare_grouped_fp8_s1_s3_tma(
            bufs["s1_x"], bufs["s1_output"], bufs["s3_x"], bufs["s3_output"],
            bufs["seqlens"], bufs["cu_seqlens"], 20,
            s1_tma_desc=bufs["s1_tma_desc"], s3_tma_desc=bufs["s3_tma_desc"],
            tiles=bufs["tiles"], cu_tiles=bufs["cu_tiles"])


def _s1_call(bufs, plan, **over):
    kw = dict(x_fp8=bufs["s1_x"], x_scale=torch.ones(1, 8),
              gate_w3d=torch.ones(2, 4, 16), up_w3d=torch.ones(2, 4, 16),
              gate_ws3d=torch.ones(2, 1, 4), up_ws3d=torch.ones(2, 1, 4),
              seqlens=bufs["seqlens"], cu_seqlens=bufs["cu_seqlens"],
              num_seq_per_group_avg=20, output=bufs["s1_output"],
              tma_plan=plan)
    kw.update(over)
    return fp8_moe.grouped_fp8_blockwise_fused_s1(**kw)


def _s3_call(bufs, plan, **over):
    kw = dict(x_fp8=bufs["s3_x"], x_scale=torch.ones(1, 8),
              down_w3d=torch.ones(2, 16, 4), down_ws3d=torch.ones(2, 1, 4),
              seqlens=bufs["seqlens"], cu_seqlens=bufs["cu_seqlens"],
              num_seq_per_group_avg=20, output=bufs["s3_output"],
              tma_plan=plan)
    kw.update(over)
    return fp8_moe.grouped_fp8_blockwise_s3(**kw)


def test_both_gemms_consume_the_plan_without_their_own_preparation(loader):
    bufs = _plan_buffers()
    fused = mock.MagicMock(return_value=bufs["s1_output"])
    gemm = mock.MagicMock(return_value=bufs["s3_output"])
    _, plan = _prepare(loader, bufs, fp8_blockwise_fused_s1=fused,
                       fp8_blockwise_grouped_gemm=gemm)
    assert _s1_call(bufs, plan) is bufs["s1_output"]
    assert _s3_call(bufs, plan) is bufs["s3_output"]
    s1 = fused.call_args.args
    assert s1[-5] is bufs["s1_output"]
    assert s1[-4] is bufs["s1_tma_desc"] and s1[-3] is bufs["tiles"]
    assert s1[-2] is bufs["cu_tiles"] and s1[-1] is False   # update_tma
    s3 = gemm.call_args.args
    assert s3[-5] is bufs["s3_output"]
    assert s3[-4] is bufs["s3_tma_desc"] and s3[-3] is bufs["tiles"]
    assert s3[-2] is bufs["cu_tiles"] and s3[-1] is False


@pytest.mark.parametrize("stage, over, match", [
    ("s1", "swap_x", r"\(x, output, K, N\)"),
    ("s1", "other_output", r"\(x, output, K, N\)"),
    ("s1", "no_output", "output buffer it was prepared for"),
    ("s1", "other_seqlens", "seqlens / cu_seqlens"),
    ("s1", "other_tile_m", "TileM 64"),
    ("s3", "swap_x", r"\(x, output, K, N\)"),
    ("s3", "other_seqlens", "seqlens / cu_seqlens"),
    ("s3", "other_tile_m", "TileM 16"),
])
def test_a_plan_for_other_buffers_counts_or_tilem_is_refused(
        loader, stage, over, match):
    """The descriptors encode the prepared buffers' addresses: consuming a
    plan with anything else would load/store through the wrong descriptors
    without any device-side error, so the wrapper refuses on host metadata."""
    bufs = _plan_buffers()
    fused, gemm = mock.MagicMock(), mock.MagicMock()
    _, plan = _prepare(loader, bufs, fp8_blockwise_fused_s1=fused,
                       fp8_blockwise_grouped_gemm=gemm)
    call = _s1_call if stage == "s1" else _s3_call
    own_x = "s1_x" if stage == "s1" else "s3_x"
    other_x = "s3_x" if stage == "s1" else "s1_x"
    kw = {
        "swap_x": dict(x_fp8=bufs[other_x]),
        "other_output": dict(output=bufs["s1_output"].clone()),
        "no_output": dict(output=None),
        "other_seqlens": dict(seqlens=bufs["seqlens"].clone()),
        "other_tile_m": dict(num_seq_per_group_avg=64 if stage == "s1" else 8),
    }[over]
    assert own_x not in kw or kw["x_fp8"] is not bufs[own_x]
    with pytest.raises(ValueError, match=match):
        call(bufs, plan, **kw)
    fused.assert_not_called()
    gemm.assert_not_called()


def test_s3_plan_and_explicit_descriptor_scratch_are_exclusive(loader):
    bufs = _plan_buffers()
    gemm = mock.MagicMock()
    _, plan = _prepare(loader, bufs, fp8_blockwise_grouped_gemm=gemm)
    with pytest.raises(ValueError, match="not both"):
        fp8_moe.grouped_fp8_blockwise_gemm(
            bufs["s3_x"], torch.ones(2, 16, 4), bufs["seqlens"],
            bufs["cu_seqlens"], torch.ones(1, 8), torch.ones(2, 1, 4), 20,
            output=bufs["s3_output"], tma_desc=bufs["s3_tma_desc"],
            tma_plan=plan)
    gemm.assert_not_called()


def test_s1_plan_without_the_fused_kernel_fails_closed(loader):
    bufs = _plan_buffers()
    gemm = mock.MagicMock()
    _, plan = _prepare(loader, bufs, fp8_blockwise_grouped_gemm=gemm)
    with pytest.raises(RuntimeError, match="tma_plan was supplied"):
        _s1_call(bufs, plan)
    gemm.assert_not_called()
