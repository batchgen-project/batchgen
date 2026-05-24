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


def _rms_norm_ref(
    x: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    x_fp32 = x.float()
    x_fp32 = x_fp32 * torch.rsqrt(x_fp32.square().mean(-1, keepdim=True) + eps)
    return (x_fp32 * weight.float()).to(x.dtype)


def test_prefill_output_shape():
    from batchgen_kernels.attention.v4_compressor import DeepSeekV4Compressor

    torch.manual_seed(0)
    compressor = DeepSeekV4Compressor(512, 512, 64, 4, 1e-6).cuda()
    hidden_states = torch.randn(128, 512, device="cuda", dtype=torch.float32)
    positions = torch.arange(128, device="cuda", dtype=torch.int64)
    cache = _make_cos_sin_cache(128, 64)

    out = compressor.forward_prefill(hidden_states, positions, cache)

    assert out.shape == (32, 512)


def test_gated_pooling_softmax():
    from batchgen_kernels.attention.v4_compressor import DeepSeekV4Compressor

    torch.manual_seed(1)
    compressor = DeepSeekV4Compressor(16, 8, 4, 4, 1e-6).cuda()
    hidden_states = torch.randn(4, 16, device="cuda", dtype=torch.float32)
    gate = compressor._reshape_projected(compressor.wgate(hidden_states)).view(
        1, 4, 1, 8
    )
    weights = torch.softmax(gate.float().reshape(1, 4, 8), dim=1)

    torch.testing.assert_close(
        weights.sum(dim=1),
        torch.ones(1, 8, device="cuda", dtype=torch.float32),
    )


def test_ape_addition():
    from batchgen_kernels.attention.v4_compressor import DeepSeekV4Compressor

    compressor = DeepSeekV4Compressor(8, 8, 4, 4, 1e-6).cuda()
    with torch.no_grad():
        compressor.wkv.weight.zero_()
        compressor.wgate.weight.zero_()
        compressor.norm.weight.fill_(1.0)
        compressor.ape.copy_(
            torch.tensor(
                [
                    [1.0, 2.0, 3.0, 4.0, 0.5, 1.0, 1.5, 2.0],
                    [2.0, 3.0, 4.0, 5.0, 1.0, 1.5, 2.0, 2.5],
                    [3.0, 4.0, 5.0, 6.0, 1.5, 2.0, 2.5, 3.0],
                    [4.0, 5.0, 6.0, 7.0, 2.0, 2.5, 3.0, 3.5],
                ],
                device="cuda",
                dtype=torch.float32,
            )
        )
    hidden_states = torch.zeros(4, 8, device="cuda", dtype=torch.float32)
    positions = torch.arange(4, device="cuda", dtype=torch.int64)
    cache = _make_cos_sin_cache(4, 4)

    out = compressor.forward_prefill(hidden_states, positions, cache)
    expected_pre_norm = compressor.ape.mean(dim=0, keepdim=True)
    expected = _rms_norm_ref(expected_pre_norm, compressor.norm.weight, 1e-6)

    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)


def test_norm_after_compress():
    from batchgen_kernels.attention.v4_compressor import DeepSeekV4Compressor

    compressor = DeepSeekV4Compressor(8, 8, 4, 4, 1e-6).cuda()
    with torch.no_grad():
        compressor.wkv.weight.copy_(torch.eye(8, device="cuda"))
        compressor.wgate.weight.zero_()
        compressor.ape.zero_()
        compressor.norm.weight.copy_(
            torch.tensor(
                [1.0, 2.0, 3.0, 4.0, 1.5, 2.5, 3.5, 4.5], device="cuda"
            )
        )
    hidden_states = torch.ones(4, 8, device="cuda", dtype=torch.float32)
    positions = torch.arange(4, device="cuda", dtype=torch.int64)
    cache = _make_cos_sin_cache(4, 4)

    out = compressor.forward_prefill(hidden_states, positions, cache)
    expected = _rms_norm_ref(
        torch.ones(1, 8, device="cuda", dtype=torch.float32),
        compressor.norm.weight,
        1e-6,
    )

    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)


def test_decode_single_token():
    from batchgen_kernels.attention.v4_compressor import DeepSeekV4Compressor

    torch.manual_seed(2)
    compressor = DeepSeekV4Compressor(16, 8, 4, 4, 1e-6).cuda()
    kv_state = torch.zeros(4, 8, device="cuda", dtype=torch.float32)
    score_state = torch.zeros(4, 8, device="cuda", dtype=torch.float32)
    cache = _make_cos_sin_cache(8, 4)

    for pos in range(3):
        hidden = torch.randn(1, 16, device="cuda", dtype=torch.float32)
        out, kv_state, score_state = compressor.forward_decode(
            hidden,
            kv_state,
            score_state,
            torch.tensor([pos], device="cuda", dtype=torch.int64),
            cache,
        )
        assert out.shape == (0, 8)

    hidden = torch.randn(1, 16, device="cuda", dtype=torch.float32)
    out, kv_state, score_state = compressor.forward_decode(
        hidden,
        kv_state,
        score_state,
        torch.tensor([3], device="cuda", dtype=torch.int64),
        cache,
    )

    assert out.shape == (1, 8)
    assert torch.count_nonzero(kv_state).item() > 0
    assert torch.count_nonzero(score_state).item() > 0


def test_overlap_mode():
    from batchgen_kernels.attention.v4_compressor import DeepSeekV4Compressor

    torch.manual_seed(3)
    compressor = DeepSeekV4Compressor(
        512, 512, 64, 4, 1e-6, overlap=True
    ).cuda()
    hidden_states = torch.randn(128, 512, device="cuda", dtype=torch.float32)
    positions = torch.arange(128, device="cuda", dtype=torch.int64)
    cache = _make_cos_sin_cache(128, 64)

    out = compressor.forward_prefill(hidden_states, positions, cache)

    assert out.shape == (32, 512)


@pytest.mark.parametrize("T", [128, 1024])
def test_benchmark(T):
    from batchgen_kernels.attention.v4_compressor import DeepSeekV4Compressor
    from tests.kernels.conftest import _bench

    torch.manual_seed(T)
    compressor = DeepSeekV4Compressor(512, 512, 64, 4, 1e-6).cuda()
    hidden_states = torch.randn(T, 512, device="cuda", dtype=torch.float32)
    positions = torch.arange(T, device="cuda", dtype=torch.int64)
    cache = _make_cos_sin_cache(T, 64)

    ms = _bench(
        compressor.forward_prefill, hidden_states, positions, cache, iters=5
    )
    out = compressor.forward_prefill(hidden_states, positions, cache)

    assert out.shape == (T // 4, 512)
    assert torch.isfinite(torch.tensor(ms))
    assert ms >= 0.0
