from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)

HEAD_DIM = 512


def _pool(num_tokens):
    from batchgen.kv_cache.deepseek_v4_single_kv_pool import (
        DeepSeekV4SingleKVPool,
    )

    pool = DeepSeekV4SingleKVPool(
        num_layers=1,
        num_pages=num_tokens + 8,
        page_size_tokens=128,
        device="cuda",
    )
    pool.initialize()
    return pool


def _loop_pack_reference(pool, kv_processed):
    from batchgen.kv_cache.deepseek_v4_single_kv_pool import (
        _MODEL1_NUM_TILES,
        _MODEL1_TILE_SIZE,
        NOPE_DIM,
        TOKEN_DATA_SIZE,
    )

    num_tokens = kv_processed.shape[0]
    packed = torch.zeros(
        (num_tokens, pool.bytes_per_token), dtype=torch.uint8, device="cuda"
    )
    packed[:, NOPE_DIM:TOKEN_DATA_SIZE] = (
        kv_processed[:, NOPE_DIM:]
        .contiguous()
        .view(torch.uint8)
        .reshape(num_tokens, -1)
    )
    for tile_idx in range(_MODEL1_NUM_TILES):
        start = tile_idx * _MODEL1_TILE_SIZE
        end = start + _MODEL1_TILE_SIZE
        cur = kv_processed[:, start:end].float()
        scale = torch.pow(
            2.0,
            torch.ceil(
                torch.log2(
                    torch.clamp_min(cur.abs().amax(dim=-1) / 448.0, 1e-4)
                )
            ),
        )
        packed[:, TOKEN_DATA_SIZE + tile_idx] = scale.to(
            torch.float8_e8m0fnu
        ).view(torch.uint8)
        packed[:, start:end] = (
            (cur / scale.unsqueeze(-1))
            .to(torch.float8_e4m3fn)
            .view(torch.uint8)
            .reshape(num_tokens, -1)
        )
    return packed


@pytest.mark.parametrize("num_tokens", [1, 2, 8, 64, 257])
def test_vectorized_pack_byte_exact(num_tokens):
    torch.manual_seed(num_tokens)
    pool = _pool(num_tokens)
    try:
        kv = torch.randn(
            num_tokens, HEAD_DIM, dtype=torch.bfloat16, device="cuda"
        )
        expected = _loop_pack_reference(pool, kv)
        actual = pool._pack_model1_rows(kv)
        assert torch.equal(actual, expected)
    finally:
        pool.destroy()


def test_vectorized_pack_extreme_values():
    pool = _pool(16)
    try:
        kv = torch.zeros(16, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        kv[0].fill_(0.0)
        kv[1].fill_(448.0)
        kv[2].fill_(-448.0)
        kv[3, 0] = 1e4
        kv[4, ::2] = 1e-5
        expected = _loop_pack_reference(pool, kv)
        actual = pool._pack_model1_rows(kv)
        assert torch.equal(actual, expected)
    finally:
        pool.destroy()
