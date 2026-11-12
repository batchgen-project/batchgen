"""CUDA-free contracts for the GPT-OSS startup decode probe."""

import pytest
import torch

from batchgen.models.openai.gpt_oss_120b.Parallel_Strategy_Manager import (
    GptOssParallelStrategyManager,
    StartupDecodeProbeError,
    StartupDecodeProbePlan,
    plan_startup_decode_probe,
    uniform_expert_assignment,
)


def test_probe_shape_plan_uses_requested_width_and_one_past_token():
    plan = plan_startup_decode_probe(8, 16)
    assert plan == StartupDecodeProbePlan(batch_size=8, past_tokens=1)


def test_probe_shape_plan_refuses_width_above_configured_cap():
    with pytest.raises(StartupDecodeProbeError, match="exceeds configured decode cap"):
        plan_startup_decode_probe(17, 16)


def test_probe_shape_plan_refuses_non_integer_width_without_truncation():
    with pytest.raises(StartupDecodeProbeError, match="requires integer"):
        plan_startup_decode_probe(8.5, 16)


def test_uniform_expert_assignment_balances_topk_rows():
    assignment = uniform_expert_assignment(32, 8, 2)
    counts = torch.bincount(assignment.reshape(-1), minlength=8)
    assert assignment.shape == (32, 2)
    assert int(counts.max()) - int(counts.min()) <= 1
    assert torch.all((assignment >= 0) & (assignment < 8))


def test_uniform_expert_assignment_refuses_fractional_dimensions():
    with pytest.raises(StartupDecodeProbeError, match="requires integer dimensions"):
        uniform_expert_assignment(8.5, 8, 2)


def test_probe_refuses_without_cuda(monkeypatch):
    manager = object.__new__(GptOssParallelStrategyManager)
    manager.padding_bsz = 8
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(StartupDecodeProbeError, match="requires CUDA"):
        manager.startup_decode_probe_step(8)
