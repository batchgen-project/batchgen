from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

from batchgen.models.moonshotai.kimi_linear.serving_modules import (
    _kda_o_norm_eps,
    _kda_project,
    fuse_kda_decode_projections,
)


class _AttentionStub(nn.Module):
    def __init__(self, *, full_rank=True, dtype=torch.float32):
        super().__init__()
        self.use_full_rank_gate = full_rank
        self.num_heads = 3
        self.head_dim = 2
        hidden = 5
        projection = self.num_heads * self.head_dim
        self.q_proj = nn.Linear(hidden, projection, bias=False, dtype=dtype)
        self.k_proj = nn.Linear(hidden, projection, bias=False, dtype=dtype)
        self.v_proj = nn.Linear(hidden, projection, bias=False, dtype=dtype)
        self.g_proj = nn.Linear(hidden, projection, bias=False, dtype=dtype)
        self.b_proj = nn.Linear(hidden, self.num_heads, bias=False, dtype=dtype)
        self.f_a_proj = nn.Linear(hidden, self.head_dim, bias=False, dtype=dtype)
        self.f_b_proj = nn.Linear(self.head_dim, projection, bias=False, dtype=dtype)


def test_fused_decode_projection_matches_native_projection_values():
    attn = _AttentionStub()
    x = torch.randn(7, 5)
    expected = (
        F.linear(x, attn.q_proj.weight),
        F.linear(x, attn.k_proj.weight),
        F.linear(x, attn.v_proj.weight),
        F.linear(F.linear(x, attn.f_a_proj.weight), attn.f_b_proj.weight).view(
            7, 3, 2
        ),
        F.linear(x, attn.b_proj.weight),
        F.linear(x, attn.g_proj.weight).view(7, 3, 2),
    )

    assert fuse_kda_decode_projections(attn)
    got = _kda_project(attn, x, decode=True)
    for actual, reference in zip(got, expected):
        torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-6)

    # Each source projection remains usable by prefill, but all six weights
    # share one storage allocation with no duplicate HBM copy.
    base = attn._kda_decode_fused_weight.data_ptr()
    assert all(
        projection.weight.untyped_storage().data_ptr() == base
        for projection in (
            attn.q_proj,
            attn.k_proj,
            attn.v_proj,
            attn.g_proj,
            attn.b_proj,
            attn.f_a_proj,
        )
    )


def test_fused_decode_projection_is_idempotent():
    attn = _AttentionStub()
    assert fuse_kda_decode_projections(attn)
    first = attn._kda_decode_fused_weight
    assert fuse_kda_decode_projections(attn)
    assert attn._kda_decode_fused_weight is first


def test_low_rank_gate_is_not_fused():
    attn = _AttentionStub(full_rank=False)
    assert not fuse_kda_decode_projections(attn)
    assert not hasattr(attn, "_kda_decode_fused_weight")


def test_fused_kda_accepts_fla_output_norm_eps_alias():
    """FLA names FusedRMSNormGated's epsilon ``eps``, not ``variance_epsilon``."""
    attention = SimpleNamespace(
        o_norm=SimpleNamespace(eps=2.5e-6),
        config=SimpleNamespace(rms_norm_eps=1e-5),
    )

    assert _kda_o_norm_eps(attention) == 2.5e-6
