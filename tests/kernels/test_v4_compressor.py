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


def _canonical_prefill_reference(
    compressor,
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
) -> torch.Tensor:
    ratio = compressor.compress_ratio
    coeff = compressor.coeff
    num_chunks = hidden_states.shape[0] // ratio
    tokens = num_chunks * ratio
    if tokens == 0:
        return hidden_states.new_empty(0, compressor.head_dim)
    hidden_states = hidden_states[:tokens].float()
    positions = positions[:tokens]
    kv = compressor.wkv(hidden_states).view(
        num_chunks, ratio * coeff, compressor.head_dim
    )
    gate = compressor.wgate(hidden_states).view(
        num_chunks, ratio * coeff, compressor.head_dim
    )
    ape = compressor.ape.view(ratio * coeff, compressor.head_dim)
    weights = torch.softmax(gate + ape.unsqueeze(0), dim=1)
    pooled = (kv * weights).sum(dim=1)
    pooled = _rms_norm_ref(pooled, compressor.norm.weight, compressor.norm.eps)
    chunk_positions = positions.view(num_chunks, ratio)[:, 0]
    return compressor._apply_rope(pooled, chunk_positions, cos_sin_cache)


def _canonical_decode_reference(
    compressor,
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
) -> tuple[list[tuple[int, torch.Tensor]], torch.Tensor, torch.Tensor]:
    kv_state = torch.zeros(
        compressor.compress_ratio,
        compressor.coeff * compressor.head_dim,
        device=hidden_states.device,
        dtype=torch.float32,
    )
    score_state = torch.zeros_like(kv_state)
    outputs: list[tuple[int, torch.Tensor]] = []
    for hidden_state, position in zip(
        hidden_states.float(), positions, strict=False
    ):
        slot = int(position.item()) % compressor.compress_ratio
        kv = compressor.wkv(hidden_state.unsqueeze(0)).squeeze(0)
        gate = compressor.wgate(hidden_state.unsqueeze(0)).squeeze(0)
        kv_state[slot].copy_(kv)
        score_state[slot].copy_(gate + compressor.ape[slot])
        if slot == compressor.compress_ratio - 1:
            pooled = (
                kv_state.float() * torch.softmax(score_state.float(), dim=0)
            ).sum(dim=0, keepdim=True)
            pooled = _rms_norm_ref(
                pooled, compressor.norm.weight, compressor.norm.eps
            )
            chunk_start = position.view(1).to(torch.int64) & (
                ~(compressor.compress_ratio - 1)
            )
            outputs.append(
                (
                    int(position.item()),
                    compressor._apply_rope(pooled, chunk_start, cos_sin_cache),
                )
            )
    return outputs, kv_state, score_state


def test_prefill_output_shape():
    from batchgen_kernels.attention.v4_compressor import DeepSeekV4Compressor

    torch.manual_seed(0)
    compressor = DeepSeekV4Compressor(512, 512, 64, 4, 1e-6).cuda()
    hidden_states = torch.randn(128, 512, device="cuda", dtype=torch.float32)
    positions = torch.arange(128, device="cuda", dtype=torch.int64)
    cache = _make_cos_sin_cache(128, 64)

    out = compressor.forward_prefill(hidden_states, positions, cache)

    assert out.shape == (32, 512)


def test_rotate_applies_hadamard_to_output():
    from batchgen_kernels.attention.v4_compressor import DeepSeekV4Compressor
    from batchgen_kernels.attention.dsa.fused_indexer_score import (
        get_hadamard_matrix,
    )

    torch.manual_seed(7)
    plain = DeepSeekV4Compressor(512, 512, 64, 4, 1e-6).cuda()
    rotated = DeepSeekV4Compressor(512, 512, 64, 4, 1e-6, rotate=True).cuda()
    rotated.load_state_dict(plain.state_dict())

    hidden_states = torch.randn(128, 512, device="cuda", dtype=torch.float32)
    positions = torch.arange(128, device="cuda", dtype=torch.int64)
    cache = _make_cos_sin_cache(128, 64)

    out_plain = plain.forward_prefill(hidden_states, positions, cache)
    out_rot = rotated.forward_prefill(hidden_states, positions, cache)

    H = get_hadamard_matrix(512, out_plain.device, torch.float32)
    expected = (out_plain.float() @ H).to(out_plain.dtype)
    assert torch.allclose(out_rot, expected, atol=1e-4, rtol=0)
    assert not torch.allclose(out_rot, out_plain, atol=1e-3)


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
    expected = torch.zeros_like(out)

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


def test_c128_prefill_matches_canonical_assets_math():
    from batchgen_kernels.attention.v4_compressor import DeepSeekV4Compressor

    torch.manual_seed(4)
    compressor = DeepSeekV4Compressor(16, 8, 4, 128, 1e-6).cuda()
    hidden_states = torch.randn(256, 16, device="cuda", dtype=torch.float32)
    positions = torch.arange(256, device="cuda", dtype=torch.int64)
    cache = _make_cos_sin_cache(256, 4)

    actual = compressor.forward_prefill(hidden_states, positions, cache)
    expected = _canonical_prefill_reference(
        compressor, hidden_states, positions, cache
    )

    torch.testing.assert_close(actual, expected, atol=5e-2, rtol=0)


def test_c128_decode_matches_canonical_assets_math_with_remainder():
    from batchgen_kernels.attention.v4_compressor import DeepSeekV4Compressor

    torch.manual_seed(5)
    compressor = DeepSeekV4Compressor(16, 8, 4, 128, 1e-6).cuda()
    hidden_states = torch.randn(300, 16, device="cuda", dtype=torch.float32)
    positions = torch.arange(300, device="cuda", dtype=torch.int64)
    cache = _make_cos_sin_cache(300, 4)
    kv_state = torch.zeros(128, 8, device="cuda", dtype=torch.float32)
    score_state = torch.zeros(128, 8, device="cuda", dtype=torch.float32)

    actual_outputs = []
    for start in range(hidden_states.shape[0]):
        out, kv_state, score_state = compressor.forward_decode(
            hidden_states[start : start + 1],
            kv_state,
            score_state,
            positions[start : start + 1],
            cache,
        )
        if out.numel():
            actual_outputs.append((start, out.clone()))

    expected_outputs, expected_kv_state, expected_score_state = (
        _canonical_decode_reference(compressor, hidden_states, positions, cache)
    )

    assert [idx for idx, _ in actual_outputs] == [127, 255]
    assert [idx for idx, _ in expected_outputs] == [127, 255]
    for (_, actual), (_, expected) in zip(
        actual_outputs, expected_outputs, strict=True
    ):
        torch.testing.assert_close(actual, expected, atol=5e-2, rtol=0)
    torch.testing.assert_close(kv_state, expected_kv_state, atol=1e-6, rtol=0)
    torch.testing.assert_close(
        score_state, expected_score_state, atol=1e-6, rtol=0
    )


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

    try:
        from tests.kernels.conftest import _bench
    except ModuleNotFoundError:
        pytest.skip("tests.kernels.conftest is unavailable in this environment")

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
