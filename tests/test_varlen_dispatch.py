"""Tests for batchgen.attention.varlen_dispatch (Blackwell MLA prefill shim).

The dispatch table is exercised with mocked capabilities; the SDPA fallback
is checked against a naive float32 reference on CPU. flashinfer and
flash_attn_interface are not installed in the CPU test venv, so their
branches are covered through injected fakes.
"""

from __future__ import annotations

import math
import sys
import types

import pytest
import torch

from batchgen.attention import varlen_dispatch


@pytest.fixture(autouse=True)
def _reset_backend_cache(monkeypatch):
    monkeypatch.setattr(varlen_dispatch, "_BACKEND_CACHE", None)
    monkeypatch.setattr(varlen_dispatch, "_BACKEND_CAPABILITY", None)
    monkeypatch.setattr(varlen_dispatch, "_SDPA_FALLBACK_WARNED", False)


def _naive_varlen(q, k, v, cu_q, cu_k, scale, causal):
    outs = []
    for qs, qe, ks, ke in zip(cu_q[:-1], cu_q[1:], cu_k[:-1], cu_k[1:]):
        qq, kk, vv = q[qs:qe].float(), k[ks:ke].float(), v[ks:ke].float()
        # [len, heads, dim] -> per-head attention
        scores = torch.einsum("qhd,khd->hqk", qq, kk) * scale
        if causal:
            q_len, k_len = qq.shape[0], kk.shape[0]
            mask = torch.ones(q_len, k_len, dtype=torch.bool).tril(
                diagonal=k_len - q_len
            )
            scores = scores.masked_fill(~mask, float("-inf"))
        probs = scores.softmax(dim=-1)
        outs.append(torch.einsum("hqk,khd->qhd", probs, vv))
    return torch.cat(outs, dim=0)


def test_sdpa_fallback_matches_naive_reference():
    torch.manual_seed(0)
    heads, dim = 4, 32
    lens = [5, 1, 9]
    total = sum(lens)
    q = torch.randn(total, heads, dim)
    k = torch.randn(total, heads, dim)
    v = torch.randn(total, heads, dim)
    cu = torch.tensor([0, 5, 6, 15], dtype=torch.int32)
    scale = 1.0 / math.sqrt(dim)

    out = varlen_dispatch._sdpa_varlen_prefill(
        q, k, v,
        cu_seqlens_q=cu, cu_seqlens_k=cu,
        max_seqlen_q=max(lens), max_seqlen_k=max(lens),
        softmax_scale=scale, causal=True,
    )
    ref = _naive_varlen(q, k, v, cu.tolist(), cu.tolist(), scale, True)
    torch.testing.assert_close(out.float(), ref, rtol=2e-4, atol=2e-4)


def test_dispatch_hopper_selects_fa3(monkeypatch):
    sentinel = object()
    fake = types.ModuleType("flash_attn_interface")
    fake.flash_attn_varlen_func = lambda *a, **kw: sentinel
    monkeypatch.setitem(sys.modules, "flash_attn_interface", fake)
    backend = varlen_dispatch._select_backend((9, 0))
    assert backend is fake.flash_attn_varlen_func


def test_dispatch_blackwell_selects_flashinfer():
    backend = varlen_dispatch._select_backend((10, 3))
    assert backend is varlen_dispatch._flashinfer_varlen_prefill


def test_dispatch_unknown_capability_falls_back_to_sdpa():
    backend = varlen_dispatch._select_backend((8, 0))
    assert backend is varlen_dispatch._sdpa_varlen_prefill


def test_blackwell_flashinfer_import_failure_degrades_to_sdpa(monkeypatch):
    monkeypatch.setattr(
        varlen_dispatch, "_detect_capability", lambda: (10, 3)
    )
    monkeypatch.setitem(sys.modules, "flashinfer", None)  # forces ImportError

    torch.manual_seed(1)
    q = torch.randn(4, 2, 16)
    cu = torch.tensor([0, 4], dtype=torch.int32)
    out = varlen_dispatch.varlen_attention_func(
        q, q, q,
        cu_seqlens_q=cu, cu_seqlens_k=cu,
        max_seqlen_q=4, max_seqlen_k=4,
        softmax_scale=0.25, causal=True,
    )
    assert out.shape == q.shape
    assert varlen_dispatch._BACKEND_CACHE is varlen_dispatch._sdpa_varlen_prefill


@pytest.mark.parametrize(
    "psm_path",
    [
        "batchgen/models/deepseek/deepseekv3/Parallel_Strategy_Manager.py",
        "batchgen/models/moonshotai/kimi_k25/Parallel_Strategy_Manager.py",
        "batchgen/models/glm/glm5/Parallel_Strategy_Manager.py",
    ],
)
def test_non_k3_psms_fail_loud_on_blackwell(psm_path):
    from pathlib import Path

    text = (Path(__file__).resolve().parents[1] / psm_path).read_text()
    assert 'gpu_arch == "blackwell"' in text
    assert "NotImplementedError" in text
