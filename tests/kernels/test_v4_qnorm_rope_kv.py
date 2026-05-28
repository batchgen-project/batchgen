# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                          #
# ---------------------------------------------------------------------------- #

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)

HEAD_DIM = 512
NOPE_DIM = 448
ROPE_DIM = 64
QUANT_BLOCK_SIZE = 64
SCALE_DIM = NOPE_DIM // QUANT_BLOCK_SIZE + 1
TOKEN_DATA_SIZE = NOPE_DIM + ROPE_DIM * 2
TOKEN_BYTES = TOKEN_DATA_SIZE + SCALE_DIM
FP8_MAX = float(torch.finfo(torch.float8_e4m3fn).max)


def _run_k2(*args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
    from batchgen_kernels.attention.v4_fused_qnorm_rope_kv import (
        fused_v4_qnorm_rope_kv_insert,
    )

    return fused_v4_qnorm_rope_kv_insert(*args, **kwargs)


def _make_cos_sin_cache(
    max_pos: int,
    rope_dim: int = ROPE_DIM,
    device: str = "cuda",
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
    cos = torch.repeat_interleave(angles.cos(), 2, dim=-1)
    sin = torch.repeat_interleave(angles.sin(), 2, dim=-1)
    return torch.stack((cos, sin), dim=-1)


def _make_cache(num_pages: int, device: str = "cuda") -> torch.Tensor:
    return torch.zeros(
        (num_pages, TOKEN_BYTES), device=device, dtype=torch.uint8
    )


def _q_rmsnorm_ref(x: torch.Tensor, eps: float) -> torch.Tensor:
    x_fp32 = x.float()
    return (
        x_fp32 * torch.rsqrt(x_fp32.square().mean(dim=-1, keepdim=True) + eps)
    ).to(x.dtype)


def _kv_rmsnorm_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    x_fp32 = x.float()
    return (
        x_fp32
        * torch.rsqrt(x_fp32.square().mean(dim=-1, keepdim=True) + eps)
        * weight.float()
    ).to(x.dtype)


def _apply_gptj_rope_ref(
    x: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
) -> torch.Tensor:
    if x.shape[0] == 0:
        return x.clone()
    out = x.clone()
    rope = out[:, -ROPE_DIM:].float().view(-1, ROPE_DIM // 2, 2)
    cache = cos_sin_cache.index_select(0, positions.long())
    cos = cache[:, 0::2, 0]
    sin = cache[:, 0::2, 1]
    even = rope[..., 0]
    odd = rope[..., 1]
    rot_even = even * cos - odd * sin
    rot_odd = even * sin + odd * cos
    out[:, -ROPE_DIM:] = (
        torch.stack((rot_even, rot_odd), dim=-1).flatten(-2).to(x.dtype)
    )
    return out


def _encode_scale_ref(absmax: torch.Tensor) -> torch.Tensor:
    absmax = absmax.float()
    nonzero = absmax > 0
    safe = torch.where(nonzero, absmax, torch.ones_like(absmax))
    exponent = torch.ceil(torch.log2(safe / FP8_MAX))
    encoded = torch.where(nonzero, exponent + 127.0, torch.zeros_like(exponent))
    return encoded.clamp_(0.0, 255.0).to(torch.uint8)


def _decode_scale_ref(encoded: torch.Tensor) -> torch.Tensor:
    return torch.where(
        encoded == 0,
        torch.zeros_like(encoded, dtype=torch.float32),
        torch.exp2(encoded.float() - 127.0),
    )


def _assemble_cache_rows_ref(kv_processed: torch.Tensor) -> torch.Tensor:
    rows = torch.zeros(
        (kv_processed.shape[0], TOKEN_BYTES),
        device=kv_processed.device,
        dtype=torch.uint8,
    )
    if kv_processed.shape[0] == 0:
        return rows
    nope = (
        kv_processed[:, :NOPE_DIM]
        .float()
        .view(-1, NOPE_DIM // QUANT_BLOCK_SIZE, QUANT_BLOCK_SIZE)
    )
    absmax = nope.abs().amax(dim=-1)
    encoded = _encode_scale_ref(absmax)
    scale = torch.where(
        encoded == 0,
        torch.ones_like(absmax),
        torch.exp2(encoded.float() - 127.0),
    )
    nope_fp8 = torch.clamp(nope / scale.unsqueeze(-1), -FP8_MAX, FP8_MAX).to(
        torch.float8_e4m3fn
    )
    rows[:, :NOPE_DIM] = nope_fp8.reshape(-1, NOPE_DIM).view(torch.uint8)
    rows[:, NOPE_DIM:TOKEN_DATA_SIZE] = (
        kv_processed[:, NOPE_DIM:].contiguous().view(torch.uint8)
    )
    rows[:, TOKEN_DATA_SIZE:TOKEN_BYTES] = torch.cat(
        (
            encoded,
            torch.zeros(
                kv_processed.shape[0],
                1,
                device=kv_processed.device,
                dtype=torch.uint8,
            ),
        ),
        dim=-1,
    )
    return rows


def _decode_cache_rows(
    rows: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    flat_rows = rows.reshape(-1, rows.shape[-1]).contiguous()
    nope_fp8 = flat_rows[:, :NOPE_DIM].contiguous().view(torch.float8_e4m3fn)
    encoded = flat_rows[:, TOKEN_DATA_SIZE:TOKEN_BYTES][
        :, : NOPE_DIM // QUANT_BLOCK_SIZE
    ]
    rope = (
        flat_rows[:, NOPE_DIM:TOKEN_DATA_SIZE]
        .contiguous()
        .view(torch.bfloat16)
        .reshape(-1, ROPE_DIM)
    )
    scale = _decode_scale_ref(encoded).unsqueeze(-1)
    nope = (
        nope_fp8.float().view(
            -1, NOPE_DIM // QUANT_BLOCK_SIZE, QUANT_BLOCK_SIZE
        )
        * scale
    ).reshape(-1, NOPE_DIM)
    return nope, rope, encoded


def _reference_full_pipeline(
    q: torch.Tensor,
    kv: torch.Tensor,
    kv_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_out = _apply_gptj_rope_ref(
        _q_rmsnorm_ref(q, eps), positions, cos_sin_cache
    )
    kv_out = _apply_gptj_rope_ref(
        _kv_rmsnorm_ref(kv, kv_weight, eps), positions, cos_sin_cache
    )
    return q_out, kv_out, _assemble_cache_rows_ref(kv_out)


@pytest.mark.parametrize("T", [1, 32, 128, 1024])
def test_q_rmsnorm_no_weight(T):
    eps = 1e-6
    torch.manual_seed(T)
    q = torch.randn(T, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(T, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    kv_weight = torch.randn(HEAD_DIM, device="cuda", dtype=torch.float32)
    positions = torch.zeros(T, device="cuda", dtype=torch.int64)
    cos_sin_cache = _make_cos_sin_cache(1)
    kv_cache = _make_cache(max(T, 1))
    block_table = torch.arange(T, device="cuda", dtype=torch.int64)

    q_out, _ = _run_k2(
        q, kv, kv_weight, cos_sin_cache, positions, kv_cache, block_table, eps
    )

    torch.testing.assert_close(q_out.float(), _q_rmsnorm_ref(q, eps).float())


@pytest.mark.parametrize("T", [1, 32, 128, 1024])
def test_kv_rmsnorm_with_weight(T):
    eps = 1e-6
    torch.manual_seed(T + 100)
    q = torch.randn(T, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(T, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    kv_weight = torch.randn(HEAD_DIM, device="cuda", dtype=torch.float32)
    positions = torch.zeros(T, device="cuda", dtype=torch.int64)
    cos_sin_cache = _make_cos_sin_cache(1)
    kv_cache = _make_cache(max(T, 1))
    block_table = torch.arange(T, device="cuda", dtype=torch.int64)

    _, kv_out = _run_k2(
        q, kv, kv_weight, cos_sin_cache, positions, kv_cache, block_table, eps
    )

    torch.testing.assert_close(
        kv_out.float(), _kv_rmsnorm_ref(kv, kv_weight, eps).float()
    )


def test_gptj_rope_last_64_dims():
    eps = 1e-6
    T = 32
    torch.manual_seed(2)
    q = torch.randn(T, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(T, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    kv_weight = torch.randn(HEAD_DIM, device="cuda", dtype=torch.float32)
    positions = torch.arange(T, device="cuda", dtype=torch.int64)
    cos_sin_cache = _make_cos_sin_cache(T + 1)
    kv_cache = _make_cache(T)
    block_table = torch.arange(T, device="cuda", dtype=torch.int64)

    q_out, kv_out = _run_k2(
        q, kv, kv_weight, cos_sin_cache, positions, kv_cache, block_table, eps
    )
    q_norm = _q_rmsnorm_ref(q, eps)
    kv_norm = _kv_rmsnorm_ref(kv, kv_weight, eps)
    q_expected = _apply_gptj_rope_ref(q_norm, positions, cos_sin_cache)
    kv_expected = _apply_gptj_rope_ref(kv_norm, positions, cos_sin_cache)

    torch.testing.assert_close(
        q_out[:, :NOPE_DIM].float(), q_norm[:, :NOPE_DIM].float()
    )
    torch.testing.assert_close(
        kv_out[:, :NOPE_DIM].float(), kv_norm[:, :NOPE_DIM].float()
    )
    torch.testing.assert_close(
        q_out[:, NOPE_DIM:].float(), q_expected[:, NOPE_DIM:].float()
    )
    torch.testing.assert_close(
        kv_out[:, NOPE_DIM:].float(), kv_expected[:, NOPE_DIM:].float()
    )


def test_rope_position_0_identity():
    eps = 1e-6
    T = 128
    torch.manual_seed(3)
    q = torch.randn(T, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(T, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    kv_weight = torch.randn(HEAD_DIM, device="cuda", dtype=torch.float32)
    positions = torch.zeros(T, device="cuda", dtype=torch.int64)
    cos_sin_cache = _make_cos_sin_cache(1)
    kv_cache = _make_cache(T)
    block_table = torch.arange(T, device="cuda", dtype=torch.int64)

    q_out, kv_out = _run_k2(
        q, kv, kv_weight, cos_sin_cache, positions, kv_cache, block_table, eps
    )

    torch.testing.assert_close(q_out.float(), _q_rmsnorm_ref(q, eps).float())
    torch.testing.assert_close(
        kv_out.float(), _kv_rmsnorm_ref(kv, kv_weight, eps).float()
    )


def test_nope_fp8_quant():
    eps = 1e-6
    T = 32
    torch.manual_seed(4)
    q = torch.randn(T, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(T, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    kv_weight = torch.randn(HEAD_DIM, device="cuda", dtype=torch.float32)
    positions = torch.arange(T, device="cuda", dtype=torch.int64)
    cos_sin_cache = _make_cos_sin_cache(T + 1)
    kv_cache = _make_cache(T)
    block_table = torch.arange(T, device="cuda", dtype=torch.int64)

    _, kv_out = _run_k2(
        q, kv, kv_weight, cos_sin_cache, positions, kv_cache, block_table, eps
    )
    nope, _, _ = _decode_cache_rows(kv_cache[block_table])

    torch.testing.assert_close(
        nope.float(), kv_out[:, :NOPE_DIM].float(), atol=0.05, rtol=0.05
    )


def test_rope_bf16_preserved():
    eps = 1e-6
    T = 32
    torch.manual_seed(5)
    q = torch.randn(T, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(T, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    kv_weight = torch.randn(HEAD_DIM, device="cuda", dtype=torch.float32)
    positions = torch.arange(T, device="cuda", dtype=torch.int64)
    cos_sin_cache = _make_cos_sin_cache(T + 1)
    kv_cache = _make_cache(T)
    block_table = torch.arange(T, device="cuda", dtype=torch.int64)

    _, kv_out = _run_k2(
        q, kv, kv_weight, cos_sin_cache, positions, kv_cache, block_table, eps
    )
    _, rope, _ = _decode_cache_rows(kv_cache[block_table])

    assert rope.dtype == torch.bfloat16
    torch.testing.assert_close(rope, kv_out[:, NOPE_DIM:])


def test_ue8m0_scale():
    eps = 1e-6
    q = torch.zeros(1, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    kv = torch.zeros(1, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    kv[0, ::QUANT_BLOCK_SIZE] = torch.tensor(
        [448.0, 449.0, 224.0, 896.0, 112.0, 56.0, 28.0, 0.0],
        device="cuda",
        dtype=torch.bfloat16,
    )
    kv_weight = torch.ones(HEAD_DIM, device="cuda", dtype=torch.float32)
    positions = torch.zeros(1, device="cuda", dtype=torch.int64)
    cos_sin_cache = _make_cos_sin_cache(1)
    kv_cache = _make_cache(1)
    block_table = torch.zeros(1, device="cuda", dtype=torch.int64)

    _, kv_out = _run_k2(
        q, kv, kv_weight, cos_sin_cache, positions, kv_cache, block_table, eps
    )
    expected = _encode_scale_ref(
        kv_out[:, :NOPE_DIM]
        .float()
        .view(1, NOPE_DIM // QUANT_BLOCK_SIZE, QUANT_BLOCK_SIZE)
        .abs()
        .amax(dim=-1)
    )
    scales = kv_cache[:, TOKEN_DATA_SIZE:TOKEN_BYTES]

    assert torch.equal(scales[:, :-1], expected)
    assert torch.equal(
        scales[:, -1], torch.zeros(1, device="cuda", dtype=torch.uint8)
    )


def test_cache_insert_placement():
    eps = 1e-6
    T = 4
    torch.manual_seed(6)
    q = torch.randn(T, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(T, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    kv_weight = torch.randn(HEAD_DIM, device="cuda", dtype=torch.float32)
    positions = torch.tensor([0, 3, 7, 11], device="cuda", dtype=torch.int64)
    cos_sin_cache = _make_cos_sin_cache(16)
    kv_cache = _make_cache(5)
    block_table = torch.tensor([3, 1, 4, 0], device="cuda", dtype=torch.int64)

    _, kv_out = _run_k2(
        q, kv, kv_weight, cos_sin_cache, positions, kv_cache, block_table, eps
    )
    expected_rows = _assemble_cache_rows_ref(kv_out)

    assert torch.equal(kv_cache[block_table], expected_rows)
    assert torch.count_nonzero(kv_cache[2]).item() == 0


def test_full_pipeline():
    eps = 1e-6
    T = 128
    torch.manual_seed(7)
    q = torch.randn(T, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(T, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    kv_weight = torch.randn(HEAD_DIM, device="cuda", dtype=torch.float32)
    positions = torch.arange(T, device="cuda", dtype=torch.int64)
    cos_sin_cache = _make_cos_sin_cache(T + 1)
    kv_cache = _make_cache(T)
    block_table = torch.arange(T - 1, -1, -1, device="cuda", dtype=torch.int64)

    q_out, kv_out = _run_k2(
        q, kv, kv_weight, cos_sin_cache, positions, kv_cache, block_table, eps
    )
    q_expected, kv_expected, cache_expected = _reference_full_pipeline(
        q, kv, kv_weight, cos_sin_cache, positions, eps
    )

    torch.testing.assert_close(q_out.float(), q_expected.float())
    torch.testing.assert_close(kv_out.float(), kv_expected.float())
    assert torch.equal(kv_cache[block_table], cache_expected)


def test_single_token_decode():
    eps = 1e-6
    torch.manual_seed(8)
    q = torch.randn(1, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(1, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    kv_weight = torch.randn(HEAD_DIM, device="cuda", dtype=torch.float32)
    positions = torch.tensor([17], device="cuda", dtype=torch.int64)
    cos_sin_cache = _make_cos_sin_cache(32)
    kv_cache = _make_cache(3)
    block_table = torch.tensor([2], device="cuda", dtype=torch.int64)

    q_out, kv_out = _run_k2(
        q, kv, kv_weight, cos_sin_cache, positions, kv_cache, block_table, eps
    )
    q_expected, kv_expected, cache_expected = _reference_full_pipeline(
        q, kv, kv_weight, cos_sin_cache, positions, eps
    )

    torch.testing.assert_close(q_out.float(), q_expected.float())
    torch.testing.assert_close(kv_out.float(), kv_expected.float())
    assert torch.equal(kv_cache[block_table], cache_expected)


def test_output_dtype():
    q = torch.randn(32, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(32, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    kv_weight = torch.randn(HEAD_DIM, device="cuda", dtype=torch.float32)
    positions = torch.arange(32, device="cuda", dtype=torch.int64)
    cos_sin_cache = _make_cos_sin_cache(33)
    kv_cache = _make_cache(32)
    block_table = torch.arange(32, device="cuda", dtype=torch.int64)

    q_out, kv_out = _run_k2(
        q, kv, kv_weight, cos_sin_cache, positions, kv_cache, block_table
    )

    assert q_out.dtype == torch.bfloat16
    assert kv_out.dtype == torch.bfloat16


def test_empty_input():
    q = torch.empty(0, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    kv = torch.empty(0, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    kv_weight = torch.randn(HEAD_DIM, device="cuda", dtype=torch.float32)
    positions = torch.empty(0, device="cuda", dtype=torch.int64)
    cos_sin_cache = _make_cos_sin_cache(1)
    kv_cache = _make_cache(2)
    kv_cache_before = kv_cache.clone()
    block_table = torch.empty(0, device="cuda", dtype=torch.int64)

    q_out, kv_out = _run_k2(
        q, kv, kv_weight, cos_sin_cache, positions, kv_cache, block_table
    )

    assert q_out.shape == (0, HEAD_DIM)
    assert kv_out.shape == (0, HEAD_DIM)
    assert torch.equal(kv_cache, kv_cache_before)


def test_flash_shape():
    T = 128
    H = 64
    q = torch.randn(T, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(T, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    kv_weight = torch.ones(HEAD_DIM, device="cuda", dtype=torch.float32)
    positions = torch.arange(T, device="cuda", dtype=torch.int64)
    cos_sin_cache = _make_cos_sin_cache(T + 1)
    kv_cache = _make_cache(T)
    block_table = torch.arange(T, device="cuda", dtype=torch.int64)

    q_out, kv_out = _run_k2(
        q, kv, kv_weight, cos_sin_cache, positions, kv_cache, block_table
    )

    assert H == 64
    assert q_out.shape == (T, HEAD_DIM)
    assert kv_out.shape == (T, HEAD_DIM)


def test_pro_shape():
    T = 128
    H = 128
    q = torch.randn(T, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(T, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    kv_weight = torch.ones(HEAD_DIM, device="cuda", dtype=torch.float32)
    positions = torch.arange(T, device="cuda", dtype=torch.int64)
    cos_sin_cache = _make_cos_sin_cache(T + 1)
    kv_cache = _make_cache(T)
    block_table = torch.arange(T, device="cuda", dtype=torch.int64)

    q_out, kv_out = _run_k2(
        q, kv, kv_weight, cos_sin_cache, positions, kv_cache, block_table
    )

    assert H == 128
    assert q_out.shape == (T, HEAD_DIM)
    assert kv_out.shape == (T, HEAD_DIM)


def test_large_position():
    eps = 1e-6
    torch.manual_seed(9)
    q = torch.randn(2, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(2, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    kv_weight = torch.randn(HEAD_DIM, device="cuda", dtype=torch.float32)
    positions = torch.tensor([99999, 100000], device="cuda", dtype=torch.int64)
    cos_sin_cache = _make_cos_sin_cache(100001)
    kv_cache = _make_cache(2)
    block_table = torch.arange(2, device="cuda", dtype=torch.int64)

    q_out, kv_out = _run_k2(
        q, kv, kv_weight, cos_sin_cache, positions, kv_cache, block_table, eps
    )
    q_expected, kv_expected, _ = _reference_full_pipeline(
        q, kv, kv_weight, cos_sin_cache, positions, eps
    )

    torch.testing.assert_close(q_out.float(), q_expected.float())
    torch.testing.assert_close(kv_out.float(), kv_expected.float())


def test_negative_values():
    eps = 1e-6
    torch.manual_seed(10)
    q = (
        -torch.rand(32, HEAD_DIM, device="cuda", dtype=torch.float32).to(
            torch.bfloat16
        )
        * 512
    )
    kv = (
        -torch.rand(32, HEAD_DIM, device="cuda", dtype=torch.float32).to(
            torch.bfloat16
        )
        * 1024
    )
    kv_weight = -torch.rand(HEAD_DIM, device="cuda", dtype=torch.float32)
    positions = torch.arange(32, device="cuda", dtype=torch.int64)
    cos_sin_cache = _make_cos_sin_cache(33)
    kv_cache = _make_cache(32)
    block_table = torch.arange(32, device="cuda", dtype=torch.int64)

    q_out, kv_out = _run_k2(
        q, kv, kv_weight, cos_sin_cache, positions, kv_cache, block_table, eps
    )
    q_expected, kv_expected, _ = _reference_full_pipeline(
        q, kv, kv_weight, cos_sin_cache, positions, eps
    )

    assert torch.isfinite(q_out.float()).all()
    assert torch.isfinite(kv_out.float()).all()
    torch.testing.assert_close(q_out.float(), q_expected.float())
    torch.testing.assert_close(kv_out.float(), kv_expected.float())


def test_benchmark():
    from tests.kernels.conftest import _bench

    eps = 1e-6
    T = 128
    torch.manual_seed(11)
    q = torch.randn(T, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(T, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    kv_weight = torch.randn(HEAD_DIM, device="cuda", dtype=torch.float32)
    positions = torch.arange(T, device="cuda", dtype=torch.int64)
    cos_sin_cache = _make_cos_sin_cache(T + 1)
    block_table = torch.arange(T, device="cuda", dtype=torch.int64)

    def fused() -> tuple[torch.Tensor, torch.Tensor]:
        kv_cache = _make_cache(T)
        return _run_k2(
            q,
            kv,
            kv_weight,
            cos_sin_cache,
            positions,
            kv_cache,
            block_table,
            eps,
        )

    def separate() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _reference_full_pipeline(
            q, kv, kv_weight, cos_sin_cache, positions, eps
        )

    fused_ms = _bench(fused)
    separate_ms = _bench(separate)
    print(
        f"\nK2 benchmark T={T} fused={fused_ms:.3f} ms separate={separate_ms:.3f} ms"
    )

    assert fused_ms > 0
    assert separate_ms > 0
