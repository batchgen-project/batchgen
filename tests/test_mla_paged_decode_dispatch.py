"""Dispatch contract of batchgen.attention.mla.paged_decode (CPU-only).

The kernels themselves are validated on GPU against an fp32 reference; this
covers capability selection and the flash_mla legacy-kwarg adaptation.
"""

from __future__ import annotations

import sys
import types

import pytest

from batchgen.attention.mla import paged_decode


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    monkeypatch.setattr(paged_decode, "_BACKEND", None)
    monkeypatch.setattr(paged_decode, "_FLASHMLA_DENSE_KWARGS", None)


@pytest.mark.parametrize(
    "capability, expected",
    [((9, 0), "flashmla"), ((10, 0), "flashinfer-trtllm-gen"), ((10, 3), "flashinfer-trtllm-gen")],
)
def test_backend_by_capability(monkeypatch, capability, expected):
    monkeypatch.setattr(paged_decode.torch.cuda, "get_device_capability", lambda *a: capability)
    assert paged_decode._backend() == expected


def test_unknown_capability_fails_loud(monkeypatch):
    monkeypatch.setattr(paged_decode.torch.cuda, "get_device_capability", lambda *a: (8, 0))
    with pytest.raises(RuntimeError, match="no dense paged MLA decode backend"):
        paged_decode._backend()


def test_flashmla_legacy_kwarg_only_when_supported(monkeypatch):
    module = types.ModuleType("flash_mla")

    def current(q, k_cache, block_table, cache_seqlens, head_dim_v, meta, splits,
                softmax_scale=None, causal=False, use_legacy=False):
        pass

    module.flash_mla_with_kvcache = current
    monkeypatch.setitem(sys.modules, "flash_mla", module)
    assert paged_decode.flashmla_dense_kwargs() == {"use_legacy": True}

    paged_decode._FLASHMLA_DENSE_KWARGS = None

    def original(q, k_cache, block_table, cache_seqlens, head_dim_v, meta, splits,
                 softmax_scale=None, causal=False):
        pass

    module.flash_mla_with_kvcache = original
    assert paged_decode.flashmla_dense_kwargs() == {}
