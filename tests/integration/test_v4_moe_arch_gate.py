from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

import batchgen.models.deepseek.deepseekv4_flash.model as v4_model

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


def test_owned_experts_dispatches_grouped_path(monkeypatch):
    hidden, inter, num_experts = 64, 128, 4
    moe = _minimal_moe(num_experts, hidden, inter)

    expected = torch.randn(8, hidden, device="cuda", dtype=torch.float32)
    grouped_mock = MagicMock(return_value=expected)
    monkeypatch.setattr(moe, "_run_owned_experts_grouped", grouped_mock)

    token_states = torch.randn(8, hidden, device="cuda") / 8
    topk_weights = torch.rand(8, 2, device="cuda")
    topk_indices = torch.randint(
        0, num_experts, (8, 2), device="cuda", dtype=torch.int64
    )

    out = moe._run_owned_experts(token_states, topk_weights, topk_indices)

    grouped_mock.assert_called_once_with(
        token_states, topk_weights, topk_indices
    )
    assert out is expected


def _minimal_moe(num_experts=4, hidden=64, inter=128):
    config = SimpleNamespace(
        hidden_size=hidden,
        moe_intermediate_size=inter,
        n_routed_experts=num_experts,
        n_activated_experts=2,
        swiglu_limit=10.0,
        pad_token_id=0,
    )
    moe = v4_model.DeepSeekV4FlashMoE(config, layer_idx=0)
    return moe.cuda()


def _fp4_weight(out_dim, in_dim):
    packed = torch.randint(
        0, 256, (out_dim, in_dim // 2), dtype=torch.uint8, device="cuda"
    )
    scale = torch.randint(
        120, 132, (out_dim, in_dim // 32), dtype=torch.uint8, device="cuda"
    )
    return packed, scale


def _stage_fp4_experts(moe, hidden, inter):
    for e in range(moe.total_experts):
        w1, s1 = _fp4_weight(inter, hidden)
        w3, s3 = _fp4_weight(inter, hidden)
        w2, s2 = _fp4_weight(hidden, inter)
        moe.experts[e].set_runtime_tensors(
            {
                "w1.weight": w1,
                "w1.scale": s1,
                "w3.weight": w3,
                "w3.scale": s3,
                "w2.weight": w2,
                "w2.scale": s2,
            }
        )


def test_owned_experts_grouped_path_returns_finite_output():
    hidden, inter, num_experts = 64, 128, 4
    moe = _minimal_moe(num_experts, hidden, inter)
    _stage_fp4_experts(moe, hidden, inter)

    tokens = 8
    token_states = torch.randn(tokens, hidden, device="cuda") / 8
    topk_weights = torch.rand(tokens, 2, device="cuda")
    topk_indices = torch.randint(
        0, num_experts, (tokens, 2), device="cuda", dtype=torch.int64
    )

    out = moe._run_owned_experts(token_states, topk_weights, topk_indices)

    assert out.shape == (tokens, hidden)
    assert torch.isfinite(out.float()).all()


def test_grouped_moe_runs_on_current_cuda_arch():
    hidden, inter, num_experts = 64, 128, 4
    moe = _minimal_moe(num_experts, hidden, inter)
    _stage_fp4_experts(moe, hidden, inter)

    tokens = 8
    token_states = torch.randn(tokens, hidden, device="cuda") / 8
    topk_weights = torch.rand(tokens, 2, device="cuda")
    topk_indices = torch.randint(
        0, num_experts, (tokens, 2), device="cuda", dtype=torch.int64
    )

    out = moe._run_owned_experts_grouped(
        token_states, topk_weights, topk_indices
    )
    assert out.shape == (tokens, hidden)
    assert torch.isfinite(out.float()).all()
