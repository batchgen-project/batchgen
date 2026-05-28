# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                          #
# ---------------------------------------------------------------------------- #

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


def _make_cos_sin_cache(
    max_pos: int, rope_dim: int, device: str = "cuda"
) -> torch.Tensor:
    half = rope_dim // 2
    inv_freq = 1.0 / (
        10000.0
        ** (
            torch.arange(0, rope_dim, 2, device=device, dtype=torch.float32)
            / rope_dim
        )
    )
    positions = torch.arange(max_pos, device=device, dtype=torch.float32)
    angles = torch.outer(positions, inv_freq)
    return torch.cat((angles.cos(), angles.sin()), dim=-1)


def _apply_rope_ref(
    x: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    rope_dim: int = 64,
) -> torch.Tensor:
    out = x.clone()
    half = rope_dim // 2
    rope = out[..., -rope_dim:].float().view(*out.shape[:-1], half, 2)
    cache = cos_sin_cache.index_select(0, positions)
    cos = cache[:, :half].unsqueeze(1)
    sin = cache[:, half:].unsqueeze(1)
    even = rope[..., 0]
    odd = rope[..., 1]
    rot_even = even * cos - odd * sin
    rot_odd = even * sin + odd * cos
    out[..., -rope_dim:] = (
        torch.stack((rot_even, rot_odd), dim=-1).flatten(-2).to(x.dtype)
    )
    return out


def _apply_inv_rope_ref(
    x: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    rope_dim: int = 64,
) -> torch.Tensor:
    out = x.clone()
    half = rope_dim // 2
    rope = out[..., -rope_dim:].float().view(*out.shape[:-1], half, 2)
    cache = cos_sin_cache.index_select(0, positions)
    cos = cache[:, :half].unsqueeze(1)
    sin = cache[:, half:].unsqueeze(1)
    even = rope[..., 0]
    odd = rope[..., 1]
    inv_even = even * cos + odd * sin
    inv_odd = odd * cos - even * sin
    out[..., -rope_dim:] = (
        torch.stack((inv_even, inv_odd), dim=-1).flatten(-2).to(x.dtype)
    )
    return out


def _group_output(x: torch.Tensor, groups: int) -> torch.Tensor:
    t, h, d = x.shape
    return (
        x.view(t, groups, h // groups, d)
        .reshape(t, groups, -1)
        .permute(1, 0, 2)
        .contiguous()
    )


def _quantize_block_ref(
    x: torch.Tensor, block: int = 128
) -> tuple[torch.Tensor, torch.Tensor]:
    x_fp32 = x.float()
    blocks = x_fp32.view(*x.shape[:-1], x.shape[-1] // block, block)
    absmax = blocks.abs().amax(dim=-1)
    scale = absmax / torch.finfo(torch.float8_e4m3fn).max
    safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    q = torch.clamp(blocks / safe_scale.unsqueeze(-1), -448.0, 448.0).to(
        torch.float8_e4m3fn
    )
    return q.view_as(x), scale


def _dequantize_block(
    x_fp8: torch.Tensor, scale: torch.Tensor, block: int = 128
) -> torch.Tensor:
    expanded = (
        scale.unsqueeze(-1).expand(*scale.shape, block).reshape(*x_fp8.shape)
    )
    return x_fp8.float() * expanded


def test_rope_inv_rope_identity():
    from batchgen_kernels.triton.v4_inv_rope_fp8 import apply_inverse_rope

    t, h, hd = 32, 64, 512
    torch.manual_seed(0)
    x = torch.randn(t, h, hd, device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(t, device="cuda", dtype=torch.int64)
    cache = _make_cos_sin_cache(512, 64)

    rotated = _apply_rope_ref(x, positions, cache)
    restored = apply_inverse_rope(rotated, positions, cache)

    assert torch.allclose(restored, x, atol=1e-2, rtol=1e-2)


def test_full_roundtrip():
    from batchgen_kernels.triton.v4_inv_rope_fp8 import fused_inv_rope_fp8_quant

    t, h, hd, groups = 32, 64, 512, 8
    torch.manual_seed(1)
    x = torch.randn(t, h, hd, device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(t, device="cuda", dtype=torch.int64)
    cache = _make_cos_sin_cache(512, 64)

    rotated = _apply_rope_ref(x, positions, cache)
    x_fp8, x_scale = fused_inv_rope_fp8_quant(rotated, positions, cache, groups)
    restored = _dequantize_block(x_fp8, x_scale)
    expected = _group_output(x, groups).float()

    assert torch.allclose(restored, expected, atol=0.1, rtol=0.1)


@pytest.mark.parametrize("groups", [8, 16])
def test_grouped_output(groups):
    from batchgen_kernels.triton.v4_inv_rope_fp8 import fused_inv_rope_fp8_quant
    from tests.kernels.conftest import _assert_fp8_close

    t, hd = 32, 512
    h = groups * 8
    torch.manual_seed(groups)
    x = torch.randn(t, h, hd, device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(t, device="cuda", dtype=torch.int64)
    cache = _make_cos_sin_cache(512, 64)

    x_fp8, x_scale = fused_inv_rope_fp8_quant(x, positions, cache, groups)
    expected = _group_output(_apply_inv_rope_ref(x, positions, cache), groups)
    restored = _dequantize_block(x_fp8, x_scale)

    assert x_fp8.shape == (groups, t, h // groups * hd)
    assert x_scale.shape == (groups, t, h // groups * hd // 128)
    _assert_fp8_close(restored, expected.float(), msg=f"groups={groups}")


def test_block_scaled_fp8():
    from batchgen_kernels.triton.v4_inv_rope_fp8 import fused_inv_rope_fp8_quant

    t, h, hd, groups = 32, 64, 512, 8
    torch.manual_seed(2)
    x = torch.randn(t, h, hd, device="cuda", dtype=torch.bfloat16)
    positions = torch.zeros(t, device="cuda", dtype=torch.int64)
    cache = _make_cos_sin_cache(1, 64)

    x_fp8, x_scale = fused_inv_rope_fp8_quant(x, positions, cache, groups)
    grouped = _group_output(x, groups).float().view(groups, t, -1, 128)
    expected_scale = grouped.abs().amax(dim=-1) / 448.0

    assert torch.allclose(x_scale, expected_scale, atol=1e-5, rtol=1e-5)
    assert x_fp8.shape[-1] == grouped.shape[2] * 128


def test_output_dtype():
    from batchgen_kernels.triton.v4_inv_rope_fp8 import fused_inv_rope_fp8_quant

    x = torch.randn(8, 64, 512, device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(8, device="cuda", dtype=torch.int64)
    cache = _make_cos_sin_cache(32, 64)

    x_fp8, x_scale = fused_inv_rope_fp8_quant(x, positions, cache, 8)

    assert x_fp8.dtype == torch.float8_e4m3fn
    assert x_scale.dtype == torch.float32


def test_all_zero():
    from batchgen_kernels.triton.v4_inv_rope_fp8 import fused_inv_rope_fp8_quant

    x = torch.zeros(32, 64, 512, device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(32, device="cuda", dtype=torch.int64)
    cache = _make_cos_sin_cache(64, 64)

    x_fp8, x_scale = fused_inv_rope_fp8_quant(x, positions, cache, 8)

    assert torch.count_nonzero(x_fp8.float()).item() == 0
    assert torch.count_nonzero(x_scale).item() == 0


def test_position_zero():
    from batchgen_kernels.triton.v4_inv_rope_fp8 import fused_inv_rope_fp8_quant
    from tests.kernels.conftest import _assert_fp8_close

    torch.manual_seed(3)
    x = torch.randn(1, 64, 512, device="cuda", dtype=torch.bfloat16)
    positions = torch.zeros(1, device="cuda", dtype=torch.int64)
    cache = _make_cos_sin_cache(1, 64)

    x_fp8, x_scale = fused_inv_rope_fp8_quant(x, positions, cache, 8)
    restored = _dequantize_block(x_fp8, x_scale)
    expected = _group_output(x, 8)

    _assert_fp8_close(restored, expected.float(), msg="position_zero")


@pytest.mark.parametrize("T", [1, 32, 128])
@pytest.mark.parametrize("H", [64, 128])
def test_benchmark(T, H):
    from batchgen_kernels.triton.v4_inv_rope_fp8 import fused_inv_rope_fp8_quant
    from tests.kernels.conftest import _bench

    groups = 8 if H == 64 else 16
    torch.manual_seed(T * 1000 + H)
    x = torch.randn(T, H, 512, device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(T, device="cuda", dtype=torch.int64)
    cache = _make_cos_sin_cache(max(T, 1) + 1, 64)

    def separate():
        inv = _apply_inv_rope_ref(x, positions, cache)
        return _quantize_block_ref(_group_output(inv, groups))

    fused_ms = _bench(fused_inv_rope_fp8_quant, x, positions, cache, groups)
    separate_ms = _bench(separate)
    print(
        f"\nK8 benchmark T={T} H={H} fused={fused_ms:.3f} ms separate={separate_ms:.3f} ms"
    )

    assert fused_ms > 0
    assert separate_ms > 0
