"""Pointer-array FP8 grouped GEMM correctness and graph-safety coverage."""

import importlib
import importlib.util
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


WRAPPER_PATH = (
    Path(__file__).resolve().parents[1]
    / "batchgen/moe/grouped_fp8_blockwise_moe.py"
)


@pytest.fixture
def wrapper():
    spec = importlib.util.spec_from_file_location("_fp8_ptr_wrapper_test", WRAPPER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_fake_extension(monkeypatch, **symbols):
    extension_name = "batchgen_kernels.moe._C_fp8_blockwise_gemm"
    monkeypatch.setitem(sys.modules, extension_name, SimpleNamespace(**symbols))


def test_pointer_wrapper_fails_loud_when_symbol_is_missing(wrapper, monkeypatch):
    _install_fake_extension(monkeypatch, fp8_blockwise_grouped_gemm=object())
    with pytest.raises(RuntimeError, match="pointer-array grouped GEMM"):
        wrapper.grouped_fp8_blockwise_gemm_ptrs(
            None, None, None, None, None, None, None, None, 64
        )


def test_require_pointer_kernels_checks_both_symbols(wrapper, monkeypatch):
    _install_fake_extension(
        monkeypatch,
        fp8_blockwise_grouped_gemm_ptrs=object(),
    )
    with pytest.raises(RuntimeError, match="fp8_blockwise_fused_s1_ptrs"):
        wrapper.require_grouped_fp8_blockwise_ptr_kernels()


def test_pointer_wrappers_forward_persistent_workspaces(wrapper, monkeypatch):
    calls = []

    def grouped(*args):
        calls.append(("grouped", args))
        return "grouped-result"

    def fused(*args):
        calls.append(("fused", args))
        return "fused-result"

    _install_fake_extension(
        monkeypatch,
        fp8_blockwise_grouped_gemm_ptrs=grouped,
        fp8_blockwise_fused_s1_ptrs=fused,
    )
    grouped_args = tuple(object() for _ in range(8))
    fused_args = tuple(object() for _ in range(12))
    output, tma_desc, tiles, cu_tiles = (object() for _ in range(4))

    assert wrapper.grouped_fp8_blockwise_gemm_ptrs(
        *grouped_args,
        40,
        output=output,
        tma_desc=tma_desc,
        tiles=tiles,
        cu_tiles=cu_tiles,
    ) == "grouped-result"
    assert wrapper.grouped_fp8_blockwise_fused_s1_ptrs(
        *fused_args,
        40,
        output=output,
        tma_desc=tma_desc,
        tiles=tiles,
        cu_tiles=cu_tiles,
    ) == "fused-result"

    assert calls[0] == (
        "grouped",
        (*grouped_args, 64, output, tma_desc, tiles, cu_tiles),
    )
    expected_fused = (
        fused_args[0],
        fused_args[2],
        fused_args[3],
        fused_args[4],
        fused_args[5],
        fused_args[10],
        fused_args[11],
        fused_args[1],
        fused_args[6],
        fused_args[7],
        fused_args[8],
        fused_args[9],
        64,
        output,
        tma_desc,
        tiles,
        cu_tiles,
    )
    assert calls[1] == ("fused", expected_fused)


def _require_extension():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    try:
        module = importlib.import_module(
            "batchgen_kernels.moe._C_fp8_blockwise_gemm"
        )
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"FP8 extension unavailable: {exc}")
    for symbol in (
        "fp8_blockwise_grouped_gemm",
        "fp8_blockwise_grouped_gemm_ptrs",
        "fp8_blockwise_fused_s1",
        "fp8_blockwise_fused_s1_ptrs",
    ):
        if not hasattr(module, symbol):
            pytest.skip(f"extension predates batchgen_kernels 0.4.3: {symbol}")
    return module


@pytest.fixture(scope="module")
def extension():
    return _require_extension()


def _layout(counts):
    cu = [0]
    for count in counts:
        cu.append(cu[-1] + ((count + 63) // 64) * 64)
    return cu


def _make_case(counts=(5, 0, 17), n=128, k=128, seed=1):
    device = torch.device("cuda")
    generator = torch.Generator().manual_seed(seed)
    cu = _layout(counts)
    m = max(cu[-1], 64)
    e = len(counts)
    k_blocks = k // 128
    n_blocks = n // 128
    k_pad4 = (k_blocks + 3) // 4 * 4

    x = (
        torch.randn(m, k, generator=generator)
        .mul_(0.1)
        .to(torch.float8_e4m3fn)
        .to(device)
    )
    x_scale = (
        torch.rand(k_blocks, m, generator=generator).mul_(0.1).add_(0.01).to(device)
    )
    weights = [
        torch.randn(n, k, generator=generator)
        .mul_(0.1)
        .to(torch.float8_e4m3fn)
        .to(device)
        for _ in range(e)
    ]
    scales = [
        torch.rand(n_blocks, k_pad4, generator=generator)
        .mul_(0.1)
        .add_(0.01)
        .to(device)
        for _ in range(e)
    ]
    stacked_w = torch.stack(weights)
    stacked_s = torch.stack(scales)
    weight_ptrs = torch.tensor(
        [tensor.data_ptr() for tensor in weights], dtype=torch.int64, device=device
    )
    scale_ptrs = torch.tensor(
        [tensor.data_ptr() for tensor in scales], dtype=torch.int64, device=device
    )
    seqlens = torch.tensor(counts, dtype=torch.int32, device=device)
    cu_seqlens = torch.tensor(cu, dtype=torch.int32, device=device)
    active_rows = [
        torch.arange(cu[i], cu[i] + count, device=device)
        for i, count in enumerate(counts)
        if count
    ]
    rows = (
        torch.cat(active_rows)
        if active_rows
        else torch.empty(0, dtype=torch.int64, device=device)
    )
    return {
        "x": x,
        "x_scale": x_scale,
        "weights": weights,
        "scales": scales,
        "stacked_w": stacked_w,
        "stacked_s": stacked_s,
        "weight_ptrs": weight_ptrs,
        "scale_ptrs": scale_ptrs,
        "seqlens": seqlens,
        "cu_seqlens": cu_seqlens,
        "rows": rows,
        "m": m,
        "n": n,
        "e": e,
    }


def _scratch(case, descriptors):
    device = case["x"].device
    return (
        torch.empty(case["e"] * descriptors, 128, dtype=torch.uint8, device=device),
        torch.empty(case["e"], dtype=torch.int32, device=device),
        torch.empty(case["e"] + 1, dtype=torch.int32, device=device),
    )


@pytest.mark.parametrize("tile_average", [16, 32, 64])
def test_pointer_gemm_matches_stacked_ragged(extension, tile_average):
    case = _make_case()
    expected = extension.fp8_blockwise_grouped_gemm(
        case["x"],
        case["stacked_w"],
        case["seqlens"],
        case["cu_seqlens"],
        case["x_scale"],
        case["stacked_s"],
        tile_average,
    )
    output = torch.full_like(expected, float("nan"))
    tma_desc, tiles, cu_tiles = _scratch(case, 4)
    actual = extension.fp8_blockwise_grouped_gemm_ptrs(
        case["x"],
        case["weights"][0],
        case["weight_ptrs"],
        case["seqlens"],
        case["cu_seqlens"],
        case["x_scale"],
        case["scales"][0],
        case["scale_ptrs"],
        tile_average,
        output,
        tma_desc,
        tiles,
        cu_tiles,
    )
    torch.cuda.synchronize()
    assert actual.data_ptr() == output.data_ptr()
    assert torch.equal(actual[case["rows"]], expected[case["rows"]])


def test_pointer_gemm_follows_slot_permutation(extension):
    case = _make_case(seed=2)
    permutation = [2, 0, 1]
    permuted_w = torch.stack([case["weights"][index] for index in permutation])
    permuted_s = torch.stack([case["scales"][index] for index in permutation])
    ptrs_w = torch.tensor(
        [case["weights"][index].data_ptr() for index in permutation],
        dtype=torch.int64,
        device="cuda",
    )
    ptrs_s = torch.tensor(
        [case["scales"][index].data_ptr() for index in permutation],
        dtype=torch.int64,
        device="cuda",
    )
    expected = extension.fp8_blockwise_grouped_gemm(
        case["x"], permuted_w, case["seqlens"], case["cu_seqlens"],
        case["x_scale"], permuted_s, 64,
    )
    actual = extension.fp8_blockwise_grouped_gemm_ptrs(
        case["x"], case["weights"][0], ptrs_w, case["seqlens"],
        case["cu_seqlens"], case["x_scale"], case["scales"][0], ptrs_s, 64,
    )
    torch.cuda.synchronize()
    assert torch.equal(actual[case["rows"]], expected[case["rows"]])


def test_pointer_fused_s1_matches_stacked_ragged(extension):
    gate = _make_case(seed=3)
    up = _make_case(seed=4)
    expected = extension.fp8_blockwise_fused_s1(
        gate["x"],
        gate["stacked_w"],
        up["stacked_w"],
        gate["seqlens"],
        gate["cu_seqlens"],
        gate["x_scale"],
        gate["stacked_s"],
        up["stacked_s"],
        64,
    )
    output = torch.empty_like(expected)
    tma_desc, tiles, cu_tiles = _scratch(gate, 6)
    actual = extension.fp8_blockwise_fused_s1_ptrs(
        gate["x"],
        gate["weights"][0],
        gate["weight_ptrs"],
        up["weights"][0],
        up["weight_ptrs"],
        gate["seqlens"],
        gate["cu_seqlens"],
        gate["x_scale"],
        gate["scales"][0],
        gate["scale_ptrs"],
        up["scales"][0],
        up["scale_ptrs"],
        64,
        output,
        tma_desc,
        tiles,
        cu_tiles,
    )
    torch.cuda.synchronize()
    assert actual.data_ptr() == output.data_ptr()
    assert torch.equal(actual[gate["rows"]], expected[gate["rows"]])


def test_pointer_gemm_all_empty_is_noop(extension):
    case = _make_case(counts=(0, 0), seed=5)
    output = torch.full(
        (case["m"], case["n"]), 7.0, dtype=torch.bfloat16, device="cuda"
    )
    extension.fp8_blockwise_grouped_gemm_ptrs(
        case["x"], case["weights"][0], case["weight_ptrs"],
        case["seqlens"], case["cu_seqlens"], case["x_scale"],
        case["scales"][0], case["scale_ptrs"], 64, output,
    )
    torch.cuda.synchronize()
    assert bool((output == 7).all())


def test_pointer_gemm_cuda_graph_refreshes_descriptors(extension):
    first = _make_case(counts=(3, 5), seed=6)
    second = _make_case(counts=(3, 5), seed=7)
    pointer_storage = first["weight_ptrs"].clone()
    scale_pointer_storage = first["scale_ptrs"].clone()
    output = torch.empty(
        first["m"], first["n"], dtype=torch.bfloat16, device="cuda"
    )
    tma_desc, tiles, cu_tiles = _scratch(first, 4)

    expected_first = extension.fp8_blockwise_grouped_gemm(
        first["x"], first["stacked_w"], first["seqlens"], first["cu_seqlens"],
        first["x_scale"], first["stacked_s"], 64,
    )
    expected_second = extension.fp8_blockwise_grouped_gemm(
        first["x"], second["stacked_w"], first["seqlens"], first["cu_seqlens"],
        first["x_scale"], second["stacked_s"], 64,
    )
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        extension.fp8_blockwise_grouped_gemm_ptrs(
            first["x"], first["weights"][0], pointer_storage,
            first["seqlens"], first["cu_seqlens"], first["x_scale"],
            first["scales"][0], scale_pointer_storage, 64, output,
            tma_desc, tiles, cu_tiles,
        )
    graph.replay()
    torch.cuda.synchronize()
    actual_first = output.clone()

    pointer_storage.copy_(second["weight_ptrs"])
    scale_pointer_storage.copy_(second["scale_ptrs"])
    graph.replay()
    torch.cuda.synchronize()
    actual_second = output.clone()

    rows = first["rows"]
    assert torch.equal(actual_first[rows], expected_first[rows])
    assert torch.equal(actual_second[rows], expected_second[rows])
    assert not torch.equal(actual_first[rows], actual_second[rows])


def test_pointer_tables_reject_wrong_device_or_dtype(extension):
    case = _make_case(seed=8)
    with pytest.raises(RuntimeError, match="weight_ptrs must be a CUDA tensor"):
        extension.fp8_blockwise_grouped_gemm_ptrs(
            case["x"], case["weights"][0], case["weight_ptrs"].cpu(),
            case["seqlens"], case["cu_seqlens"], case["x_scale"],
            case["scales"][0], case["scale_ptrs"], 64,
        )
    with pytest.raises(RuntimeError, match="weight_ptrs must be int64"):
        extension.fp8_blockwise_grouped_gemm_ptrs(
            case["x"], case["weights"][0], case["weight_ptrs"].to(torch.int32),
            case["seqlens"], case["cu_seqlens"], case["x_scale"],
            case["scales"][0], case["scale_ptrs"], 64,
        )


def test_misaligned_pointer_traps_in_subprocess():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    code = textwrap.dedent(
        """
        import importlib
        import torch

        ext = importlib.import_module(
            "batchgen_kernels.moe._C_fp8_blockwise_gemm"
        )
        if not hasattr(ext, "fp8_blockwise_grouped_gemm_ptrs"):
            raise SystemExit(77)
        device = "cuda"
        x = torch.zeros(64, 128, dtype=torch.float8_e4m3fn, device=device)
        weight = torch.zeros(128, 128, dtype=torch.float8_e4m3fn, device=device)
        scale = torch.ones(1, 4, dtype=torch.float32, device=device)
        weight_ptrs = torch.tensor(
            [weight.data_ptr() + 1], dtype=torch.int64, device=device
        )
        scale_ptrs = torch.tensor(
            [scale.data_ptr()], dtype=torch.int64, device=device
        )
        seqlens = torch.tensor([1], dtype=torch.int32, device=device)
        cu = torch.tensor([0, 64], dtype=torch.int32, device=device)
        x_scale = torch.ones(1, 64, dtype=torch.float32, device=device)
        ext.fp8_blockwise_grouped_gemm_ptrs(
            x, weight, weight_ptrs, seqlens, cu, x_scale, scale, scale_ptrs, 64
        )
        torch.cuda.synchronize()
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        text=True,
        capture_output=True,
        timeout=60,
    )
    if result.returncode == 77:
        pytest.skip("extension predates batchgen_kernels 0.4.3")
    assert result.returncode != 0
