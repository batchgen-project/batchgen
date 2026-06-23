# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                          #
# ---------------------------------------------------------------------------- #

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)

# MXFP4 cvt.e2m1x2 PTX is rejected by ptxas on sm_90a; sm120+ only.
requires_mxfp4 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 12,
    reason="MXFP4 requires sm120+ (cvt.e2m1x2 unsupported on sm_90a)",
)


def _make_cos_sin_cache(
    max_pos: int, rope_dim: int = 64, device: str = "cuda"
) -> torch.Tensor:
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


def _rope_ref(
    index_q: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    rope_dim: int = 64,
) -> torch.Tensor:
    out = index_q.float().clone()
    half = rope_dim // 2
    rope = out[..., -rope_dim:].view(*out.shape[:-1], half, 2)
    cache = cos_sin_cache.index_select(0, positions)
    cos = cache[:, :half].unsqueeze(1)
    sin = cache[:, half:].unsqueeze(1)
    even = rope[..., 0]
    odd = rope[..., 1]
    rotated = torch.stack(
        (even * cos - odd * sin, odd * cos + even * sin), dim=-1
    ).flatten(-2)
    out[..., -rope_dim:] = rotated.to(torch.bfloat16).float()
    return out


def _fp8_scale_ref(x: torch.Tensor) -> torch.Tensor:
    amax = x.abs().amax(dim=-1)
    scale = torch.clamp_min(amax, 1e-4) / 448.0
    return torch.pow(2.0, torch.ceil(torch.log2(scale)))


def _fp8_quant_ref(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = _fp8_scale_ref(x)
    q = torch.clamp(x / scale.unsqueeze(-1), -448.0, 448.0).to(
        torch.float8_e4m3fn
    )
    return q, scale


def _dequant_fp8(x_fp8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x_fp8.float() * scale.unsqueeze(-1)


def _mxfp4_scales(scale_i32: torch.Tensor, blocks: int = 4) -> torch.Tensor:
    scale_u8 = (
        scale_i32.contiguous().view(torch.uint8).view(*scale_i32.shape, blocks)
    )
    return torch.pow(2.0, scale_u8.float() - 127.0)


@pytest.mark.parametrize("T", [1, 32, 128, 1024])
@pytest.mark.parametrize("H", [64, 128])
def test_fp8_rope_quant_vs_pytorch(T, H):
    from batchgen_kernels.triton.v4_fused_indexer_q import fused_indexer_q_fp8

    torch.manual_seed(T * 1000 + H)
    index_q = torch.randn(T, H, 128, device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(T, device="cuda", dtype=torch.int64)
    cache = _make_cos_sin_cache(T + 1)
    weights = torch.randn(T, H, device="cuda", dtype=torch.float32)

    out_fp8, weights_out = fused_indexer_q_fp8(
        index_q, cache, positions, weights
    )
    rotated = _rope_ref(index_q, positions, cache)
    ref_fp8, scale = _fp8_quant_ref(rotated)
    restored = _dequant_fp8(out_fp8, scale)
    ref_restored = _dequant_fp8(ref_fp8, scale)

    from tests.kernels.conftest import _assert_fp8_close

    _assert_fp8_close(out_fp8.float(), ref_fp8.float(), msg=f"fp8 T={T} H={H}")
    _assert_fp8_close(restored, ref_restored, msg=f"restored T={T} H={H}")
    _assert_fp8_close(weights_out, weights / scale, msg=f"weights T={T} H={H}")


def test_rope_on_last_64_dims_only():
    from batchgen_kernels.triton.v4_fused_indexer_q import fused_indexer_q_fp8

    torch.manual_seed(1)
    index_q = torch.randn(32, 64, 128, device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(32, device="cuda", dtype=torch.int64) + 7
    cache = _make_cos_sin_cache(64)
    weights = torch.ones(32, 64, device="cuda", dtype=torch.float32)

    out_fp8, _ = fused_indexer_q_fp8(index_q, cache, positions, weights)
    rotated = _rope_ref(index_q, positions, cache)
    ref_fp8, scale = _fp8_quant_ref(rotated)
    restored = _dequant_fp8(out_fp8, scale)
    ref_restored = _dequant_fp8(ref_fp8, scale)

    assert torch.allclose(
        restored[..., :64], ref_restored[..., :64], atol=1e-2, rtol=1e-2
    )
    assert not torch.allclose(
        ref_restored[..., 64:], index_q[..., 64:].float(), atol=1e-2, rtol=1e-2
    )


def test_weight_folding():
    from batchgen_kernels.triton.v4_fused_indexer_q import fused_indexer_q_fp8

    torch.manual_seed(2)
    index_q = torch.randn(32, 64, 128, device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(32, device="cuda", dtype=torch.int64)
    cache = _make_cos_sin_cache(64)
    weights = torch.randn(32, 64, device="cuda", dtype=torch.float32)
    softmax_scale = 0.125
    head_scale = 0.5

    _, weights_out = fused_indexer_q_fp8(
        index_q,
        cache,
        positions,
        weights,
        softmax_scale=softmax_scale,
        head_scale=head_scale,
    )
    scale = _fp8_scale_ref(_rope_ref(index_q, positions, cache))

    assert torch.allclose(
        weights_out,
        weights * softmax_scale * head_scale / scale,
        atol=1e-2,
        rtol=1e-2,
    )


@requires_mxfp4
def test_mxfp4_variant():
    from batchgen_kernels.common.v4_fp4_dequant import dequant_fp4_e2m1
    from batchgen_kernels.triton.v4_fused_indexer_q import fused_indexer_q_mxfp4

    torch.manual_seed(3)
    index_q = torch.randn(32, 64, 128, device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(32, device="cuda", dtype=torch.int64)
    cache = _make_cos_sin_cache(64)
    weights = torch.randn(32, 64, device="cuda", dtype=torch.float32)

    (packed, scale_i32), weights_out = fused_indexer_q_mxfp4(
        index_q, cache, positions, weights
    )
    scale = _mxfp4_scales(scale_i32)
    restored = dequant_fp4_e2m1(
        packed.view(-1, packed.shape[-1]),
        scale.view(-1, scale.shape[-1]),
        torch.float32,
    ).view(index_q.shape)

    assert packed.shape == (32, 64, 64)
    assert scale_i32.shape == (32, 64)
    assert restored.shape == index_q.shape
    assert torch.isfinite(restored).all()
    assert torch.allclose(weights_out, weights, atol=1e-2, rtol=1e-2)


def test_fp8_output_dtype():
    from batchgen_kernels.triton.v4_fused_indexer_q import fused_indexer_q_fp8

    index_q = torch.randn(8, 64, 128, device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(8, device="cuda", dtype=torch.int64)
    cache = _make_cos_sin_cache(16)
    weights = torch.ones(8, 64, device="cuda", dtype=torch.float32)

    out_fp8, weights_out = fused_indexer_q_fp8(
        index_q, cache, positions, weights
    )

    assert out_fp8.dtype == torch.float8_e4m3fn
    assert weights_out.dtype == torch.float32


def test_single_decode():
    from batchgen_kernels.triton.v4_fused_indexer_q import fused_indexer_q_fp8

    torch.manual_seed(4)
    index_q = torch.randn(1, 64, 128, device="cuda", dtype=torch.bfloat16)
    positions = torch.zeros(1, device="cuda", dtype=torch.int64)
    cache = _make_cos_sin_cache(1)
    weights = torch.randn(1, 64, device="cuda", dtype=torch.float32)

    out_fp8, weights_out = fused_indexer_q_fp8(
        index_q, cache, positions, weights
    )
    rotated = _rope_ref(index_q, positions, cache)
    ref_fp8, scale = _fp8_quant_ref(rotated)

    assert torch.allclose(
        _dequant_fp8(out_fp8, scale),
        _dequant_fp8(ref_fp8, scale),
        atol=1e-2,
        rtol=1e-2,
    )
    assert torch.allclose(weights_out, weights / scale, atol=1e-2, rtol=1e-2)


def test_all_zero_index_q():
    from batchgen_kernels.triton.v4_fused_indexer_q import fused_indexer_q_fp8

    index_q = torch.zeros(32, 64, 128, device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(32, device="cuda", dtype=torch.int64)
    cache = _make_cos_sin_cache(64)
    weights = torch.ones(32, 64, device="cuda", dtype=torch.float32)

    out_fp8, weights_out = fused_indexer_q_fp8(
        index_q, cache, positions, weights
    )
    min_scale = torch.pow(
        torch.tensor(2.0, device="cuda"),
        torch.ceil(torch.log2(torch.tensor(1e-4 / 448.0, device="cuda"))),
    )

    assert torch.count_nonzero(out_fp8.float()).item() == 0
    assert torch.allclose(
        weights_out,
        torch.ones_like(weights_out) / min_scale,
        atol=1e-2,
        rtol=1e-2,
    )


@pytest.mark.parametrize("T", [1, 128])
def test_flash_shape(T):
    from batchgen_kernels.triton.v4_fused_indexer_q import fused_indexer_q_fp8

    H = 64
    index_q = torch.randn(T, H, 128, device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(T, device="cuda", dtype=torch.int64)
    cache = _make_cos_sin_cache(T + 1)
    weights = torch.ones(T, H, device="cuda", dtype=torch.float32)

    out_fp8, weights_out = fused_indexer_q_fp8(
        index_q, cache, positions, weights
    )

    assert out_fp8.shape == (T, H, 128)
    assert weights_out.shape == (T, H)


@pytest.mark.parametrize("T", [1, 128])
def test_pro_shape(T):
    from batchgen_kernels.triton.v4_fused_indexer_q import fused_indexer_q_fp8

    H = 128
    index_q = torch.randn(T, H, 128, device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(T, device="cuda", dtype=torch.int64)
    cache = _make_cos_sin_cache(T + 1)
    weights = torch.ones(T, H, device="cuda", dtype=torch.float32)

    out_fp8, weights_out = fused_indexer_q_fp8(
        index_q, cache, positions, weights
    )

    assert out_fp8.shape == (T, H, 128)
    assert weights_out.shape == (T, H)


def test_empty_input():
    from batchgen_kernels.triton.v4_fused_indexer_q import fused_indexer_q_fp8

    index_q = torch.empty(0, 64, 128, device="cuda", dtype=torch.bfloat16)
    positions = torch.empty(0, device="cuda", dtype=torch.int64)
    cache = _make_cos_sin_cache(1)
    weights = torch.empty(0, 64, device="cuda", dtype=torch.float32)

    out_fp8, weights_out = fused_indexer_q_fp8(
        index_q, cache, positions, weights
    )

    assert out_fp8.shape == index_q.shape
    assert weights_out.shape == weights.shape


def test_dispatch_default_picks_by_capability(monkeypatch):
    import batchgen_kernels.triton.v4_fused_indexer_q as mod

    calls = {"fp8": 0, "mxfp4": 0}
    monkeypatch.setattr(
        mod,
        "fused_indexer_q_fp8",
        lambda *a, **k: calls.__setitem__("fp8", calls["fp8"] + 1),
    )
    monkeypatch.setattr(
        mod,
        "fused_indexer_q_mxfp4",
        lambda *a, **k: calls.__setitem__("mxfp4", calls["mxfp4"] + 1),
    )
    args = (None, None, None, None)

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (12, 0))
    mod.fused_indexer_q(*args)
    assert (calls["mxfp4"], calls["fp8"]) == (1, 0)

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (9, 0))
    mod.fused_indexer_q(*args)
    assert (calls["mxfp4"], calls["fp8"]) == (1, 1)

    mod.fused_indexer_q(*args, use_fp4=True)
    assert (calls["mxfp4"], calls["fp8"]) == (2, 1)


@pytest.mark.parametrize("T", [1, 128])
@pytest.mark.parametrize("H", [64, 128])
def test_benchmark(T, H):
    from batchgen_kernels.triton.v4_fused_indexer_q import fused_indexer_q_fp8
    from tests.kernels.conftest import _bench

    torch.manual_seed(T * 1000 + H + 9)
    index_q = torch.randn(T, H, 128, device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(T, device="cuda", dtype=torch.int64)
    cache = _make_cos_sin_cache(T + 1)
    weights = torch.randn(T, H, device="cuda", dtype=torch.float32)

    def separate():
        rotated = _rope_ref(index_q, positions, cache)
        scale = _fp8_scale_ref(rotated)
        q_fp8 = torch.clamp(rotated / scale.unsqueeze(-1), -448.0, 448.0).to(
            torch.float8_e4m3fn
        )
        return q_fp8, weights / scale

    fused_ms = _bench(fused_indexer_q_fp8, index_q, cache, positions, weights)
    separate_ms = _bench(separate)
    print(
        f"\nK14 benchmark T={T} H={H} fused={fused_ms:.3f} ms separate={separate_ms:.3f} ms"
    )

    assert fused_ms > 0
    assert separate_ms > 0
