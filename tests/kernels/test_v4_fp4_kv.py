"""Round-trip tests for FP4 KV cache quantization methods."""

import pytest
import torch

from batchgen.quantization.v4_fp4_kv_cache import (
    BlockFP4KVQuantizeUtil,
    NVFP4KVQuantizeUtil,
    get_fp4_kv_cache_quant_method,
    _is_sm90_supported,
)

CUDA_AVAILABLE = torch.cuda.is_available()
SKIP_NO_CUDA = pytest.mark.skipif(not CUDA_AVAILABLE, reason="No CUDA device")

B, M, N = 4, 8, 128  # batch, heads, head_dim (must be divisible by 16)


def _relative_error(
    original: torch.Tensor, reconstructed: torch.Tensor
) -> float:
    orig_f32 = original.float()
    recon_f32 = reconstructed.float()
    denom = orig_f32.abs().mean()
    if denom < 1e-8:
        return (orig_f32 - recon_f32).abs().mean().item()
    return ((orig_f32 - recon_f32).abs().mean() / denom).item()


@SKIP_NO_CUDA
def test_blockfp4_round_trip():
    torch.manual_seed(42)
    x = torch.randn(B, M, N, dtype=torch.bfloat16, device="cuda")

    packed, scales = BlockFP4KVQuantizeUtil.batched_quantize(x)

    assert packed.shape == (B, M, N // 2)
    assert packed.dtype == torch.uint8

    recon = BlockFP4KVQuantizeUtil.batched_dequantize(packed, scales)

    assert recon.shape == x.shape
    assert recon.dtype == torch.bfloat16

    err = _relative_error(x, recon)
    assert err < 0.1, f"BlockFP4 round-trip relative error {err:.4f} >= 0.1"


@SKIP_NO_CUDA
def test_blockfp4_round_trip_small_values():
    torch.manual_seed(7)
    x = torch.randn(B, M, N, dtype=torch.bfloat16, device="cuda") * 0.01

    packed, scales = BlockFP4KVQuantizeUtil.batched_quantize(x)
    recon = BlockFP4KVQuantizeUtil.batched_dequantize(packed, scales)

    err = _relative_error(x, recon)
    assert err < 0.5, f"BlockFP4 small-values relative error {err:.4f} >= 0.5"


@SKIP_NO_CUDA
@pytest.mark.skipif(
    not (CUDA_AVAILABLE and _is_sm90_supported()),
    reason="NVFP4 requires SM90+ GPU",
)
def test_nvfp4_round_trip():
    try:
        from flashinfer import fp4_quantize  # noqa: F401
    except ImportError:
        pytest.skip("flashinfer not installed")

    torch.manual_seed(42)
    x = torch.randn(B, M, N, dtype=torch.bfloat16, device="cuda")
    global_scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")

    fp4_data, block_scales, gs = NVFP4KVQuantizeUtil.quantize(x, global_scale)

    assert fp4_data.shape == (B, M, N // 2)

    recon = NVFP4KVQuantizeUtil.dequantize(fp4_data, block_scales, gs)

    assert recon.shape == x.shape

    err = _relative_error(x, recon)
    assert err < 0.1, f"NVFP4 round-trip relative error {err:.4f} >= 0.1"


@SKIP_NO_CUDA
def test_blockfp4_method_create_buffers():
    method = get_fp4_kv_cache_quant_method("blockfp4")
    buffers = method.create_buffers(
        size=32, head_num=M, head_dim=N, layer_num=2, device="cuda"
    )
    assert len(buffers["k_buffer"]) == 2
    assert buffers["k_buffer"][0].shape == (32, M, N // 2)
    assert buffers["store_dtype"] == torch.uint8
    assert method.needs_dequant_workspace()


def test_factory_registry():
    with pytest.raises(ValueError, match="Unknown fp4_kv_cache_recipe"):
        get_fp4_kv_cache_quant_method("nonexistent")


@SKIP_NO_CUDA
def test_nvfp4_method_create_buffers():
    method = get_fp4_kv_cache_quant_method("nvfp4", num_layers=4, device="cuda")
    assert method.needs_global_scale()
    assert method.name == "nvfp4"
    buffers = method.create_buffers(
        size=16, head_num=M, head_dim=N, layer_num=4, device="cuda"
    )
    assert len(buffers["k_buffer"]) == 4
    assert buffers["dq_k_buffer"].dtype == torch.float8_e4m3fn


@SKIP_NO_CUDA
def test_nvfp4_set_layer_scales():
    method = get_fp4_kv_cache_quant_method(
        "nvfp4", num_layers=4, device="cuda", sm_version=120
    )
    method.set_layer_scales(0, k_scale=2.0, v_scale=3.0)
    assert method.k_scales_gpu[0].item() == pytest.approx(2.0)
    assert method.v_scales_gpu[0].item() == pytest.approx(3.0)


@SKIP_NO_CUDA
def test_compute_cell_size():
    method = get_fp4_kv_cache_quant_method("blockfp4")
    cell = method.compute_cell_size(
        head_num=8, head_dim=128, num_layers=32, kv_size=1
    )
    assert cell > 0
    assert isinstance(cell, int)
