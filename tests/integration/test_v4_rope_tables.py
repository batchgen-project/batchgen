from __future__ import annotations

import torch

from batchgen.attention.dsa.v4_flashmla_adapter import (
    build_v4_rope_cache,
    build_v4_rope_tables,
)


def test_rope_tables_match_canonical_no_yarn():
    max_pos = 256
    theta = 10000.0
    rope_dim = 64

    cos_table, sin_table = build_v4_rope_tables(
        max_pos=max_pos, theta=theta, rope_head_dim=rope_dim
    )

    freqs = 1.0 / (
        theta ** (torch.arange(0, rope_dim, 2, dtype=torch.float32) / rope_dim)
    )
    t = torch.arange(max_pos, dtype=torch.float32)
    angles = t[:, None] * freqs[None, :]
    exp_cos = torch.cos(angles).repeat(1, 2)
    exp_sin = torch.sin(angles).repeat(1, 2)

    assert cos_table.shape == (max_pos, rope_dim)
    assert torch.allclose(cos_table, exp_cos, atol=1e-5)
    assert torch.allclose(sin_table, exp_sin, atol=1e-5)


def test_rope_tables_consistent_with_complex_cache():
    max_pos = 128
    theta = 160000.0
    rope_dim = 64

    freqs_cis = build_v4_rope_cache(
        max_pos=max_pos,
        theta=theta,
        rope_head_dim=rope_dim,
        original_seq_len=65536,
        factor=16.0,
    )
    cos_table, sin_table = build_v4_rope_tables(
        max_pos=max_pos,
        theta=theta,
        rope_head_dim=rope_dim,
        original_seq_len=65536,
        factor=16.0,
    )

    half = rope_dim // 2
    assert torch.allclose(
        cos_table[:, :half], freqs_cis.real.float(), atol=1e-5
    )
    assert torch.allclose(
        sin_table[:, :half], freqs_cis.imag.float(), atol=1e-5
    )
