from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from batchgen.attention.prefix_aware_backend import (
    GqaPrefixAwareAttentionBackend,
    MlaProjectedPrefixAwareAttentionBackend,
)
from batchgen.models.wrappers.prefix_cache import PrefixCachePrepackMetadata


class _FakePrefixKvBuilder:
    def __init__(self):
        self.prefix_calls = []
        self.full_hit_calls = []

    def build_gqa_prefix_kv(self, **kwargs):
        self.prefix_calls.append(kwargs)
        key = torch.full((5, 1, 2), 2.0)
        value = torch.full((5, 1, 2), 3.0)
        return key, value, torch.tensor([0, 5], dtype=torch.int32), 5

    def build_gqa_full_hit_kv(self, **kwargs):
        self.full_hit_calls.append(kwargs)
        key = torch.full((4, 1, 2), 4.0)
        value = torch.full((4, 1, 2), 5.0)
        return key, value, torch.tensor([0, 4], dtype=torch.int32), 4

    def build_mla_prefix_kv(self, **kwargs):
        self.prefix_calls.append(kwargs)
        kv_dim = int(kwargs["kv_dim"])
        key = torch.full((5, 1, kv_dim), 6.0)
        return key, torch.tensor([0, 5], dtype=torch.int32), 5


class _FakeGpuKvManager:
    def __init__(self):
        self.prepare_calls = []
        self.append_calls = []

    def prepare_prefill_suffix_append(self, **kwargs):
        self.prepare_calls.append(kwargs)
        return object()

    def append_layer_prefill_suffix_tokens(self, **kwargs):
        self.append_calls.append(kwargs)


class _FakePagedGpuKvManager(_FakeGpuKvManager):
    def __init__(self, *, k_cache: torch.Tensor, v_cache: torch.Tensor | None = None):
        super().__init__()
        self.k_cache = k_cache
        self.v_cache = v_cache

    def prepare_prefill_suffix_append(self, **kwargs):
        self.prepare_calls.append(kwargs)
        return SimpleNamespace(
            cache_seqlens=torch.tensor([5], dtype=torch.int32),
            page_table=torch.tensor([[0, 1]], dtype=torch.int32),
        )

    def get_layer_kv_with_page_table(self, layer_idx):
        return self.k_cache, self.v_cache, torch.tensor([[0, 1]], dtype=torch.int32)


def _metadata(
    *,
    prefix_reuse: bool = False,
    full_hit: bool = False,
) -> PrefixCachePrepackMetadata:
    if full_hit:
        cu_seqlens = torch.tensor([0, 1], dtype=torch.int32)
        max_seqlen = 1
        seq_lengths = [1]
        prefix_tokens = [4]
        full_lengths = [4]
    else:
        cu_seqlens = torch.tensor([0, 2], dtype=torch.int32)
        max_seqlen = 2
        seq_lengths = [2]
        prefix_tokens = [3] if prefix_reuse else None
        full_lengths = [5] if prefix_reuse else None
    return PrefixCachePrepackMetadata(
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        num_sequences=1,
        seq_lengths=seq_lengths,
        global_sequence_ids=[100],
        prefix_reuse_mode=prefix_reuse,
        full_hit_mode=full_hit,
        prefix_shared_tokens=prefix_tokens,
        full_seq_lengths=full_lengths,
    )


def test_gqa_backend_no_prefix_uses_query_cu_seqlens_for_kv():
    recorded = {}

    def attention_fn(**kwargs):
        recorded.update(kwargs)
        return kwargs["q"] + 1, None

    builder = _FakePrefixKvBuilder()
    backend = GqaPrefixAwareAttentionBackend(
        prefix_kv_builder=builder,
        num_kv_heads=1,
        head_dim=2,
        attention_fn=attention_fn,
    )
    query = torch.zeros((2, 2, 2))
    key = torch.ones((2, 1, 2))
    value = key + 10

    output = backend.forward_prefill(
        query=query,
        key=key,
        value=value,
        metadata=_metadata(),
    )

    torch.testing.assert_close(output, query + 1)
    assert builder.prefix_calls == []
    assert recorded["k"] is key
    assert recorded["v"] is value
    assert recorded["cu_seqlens_q"].tolist() == [0, 2]
    assert recorded["cu_seqlens_k"].tolist() == [0, 2]
    assert recorded["max_seqlen_q"] == 2
    assert recorded["max_seqlen_k"] == 2


def test_gqa_backend_prefix_reuse_uses_prefix_kv_builder():
    recorded = {}

    def attention_fn(**kwargs):
        recorded.update(kwargs)
        return kwargs["q"], None

    builder = _FakePrefixKvBuilder()
    backend = GqaPrefixAwareAttentionBackend(
        prefix_kv_builder=builder,
        num_kv_heads=1,
        head_dim=2,
        attention_fn=attention_fn,
    )
    query = torch.zeros((2, 2, 2))
    key = torch.ones((2, 1, 2))
    value = key + 10

    backend.forward_prefill(
        query=query,
        key=key,
        value=value,
        metadata=_metadata(prefix_reuse=True),
    )

    assert len(builder.prefix_calls) == 1
    assert recorded["k"].shape == (5, 1, 2)
    assert recorded["v"].shape == (5, 1, 2)
    assert recorded["cu_seqlens_k"].tolist() == [0, 5]
    assert recorded["max_seqlen_k"] == 5


def test_gqa_backend_full_hit_uses_full_hit_kv_builder():
    recorded = {}

    def attention_fn(**kwargs):
        recorded.update(kwargs)
        return kwargs["q"], None

    builder = _FakePrefixKvBuilder()
    backend = GqaPrefixAwareAttentionBackend(
        prefix_kv_builder=builder,
        num_kv_heads=1,
        head_dim=2,
        attention_fn=attention_fn,
    )

    backend.forward_prefill(
        query=torch.zeros((1, 2, 2)),
        key=torch.ones((1, 1, 2)),
        value=torch.ones((1, 1, 2)),
        metadata=_metadata(full_hit=True),
    )

    assert len(builder.full_hit_calls) == 1
    assert recorded["k"].shape == (4, 1, 2)
    assert recorded["v"].shape == (4, 1, 2)
    assert recorded["cu_seqlens_q"].tolist() == [0, 1]
    assert recorded["cu_seqlens_k"].tolist() == [0, 4]
    assert recorded["max_seqlen_q"] == 1
    assert recorded["max_seqlen_k"] == 4


def test_gqa_backend_missing_value_raises():
    backend = GqaPrefixAwareAttentionBackend(
        prefix_kv_builder=_FakePrefixKvBuilder(),
        num_kv_heads=1,
        head_dim=2,
        attention_fn=lambda **kwargs: (kwargs["q"], None),
    )

    with pytest.raises(RuntimeError, match="value tensor"):
        backend.forward_prefill(
            query=torch.zeros((1, 1, 2)),
            key=torch.zeros((1, 1, 2)),
            value=None,
            metadata=_metadata(),
        )


def test_gqa_backend_missing_metadata_raises():
    backend = GqaPrefixAwareAttentionBackend(
        prefix_kv_builder=_FakePrefixKvBuilder(),
        num_kv_heads=1,
        head_dim=2,
        attention_fn=lambda **kwargs: (kwargs["q"], None),
    )

    with pytest.raises(TypeError, match="metadata"):
        backend.forward_prefill(
            query=torch.zeros((1, 1, 2)),
            key=torch.zeros((1, 1, 2)),
            value=torch.zeros((1, 1, 2)),
            metadata=object(),
        )


def test_gqa_backend_can_append_suffix_kv_to_gpu_manager():
    manager = _FakeGpuKvManager()
    kv_cache_metadata = type("KVCache", (), {"gpu_paged_kv_manager": manager})()
    backend = GqaPrefixAwareAttentionBackend(
        prefix_kv_builder=_FakePrefixKvBuilder(),
        num_kv_heads=1,
        head_dim=2,
        attention_fn=lambda **kwargs: (kwargs["q"], None),
        layer_idx=7,
        enable_gpu_suffix_append=True,
    )
    key = torch.ones((2, 1, 2))
    value = key + 10

    backend.forward_prefill(
        query=torch.zeros((2, 2, 2)),
        key=key,
        value=value,
        metadata=_metadata(prefix_reuse=True),
        kv_cache_metadata=kv_cache_metadata,
    )

    assert manager.prepare_calls == [
        {
            "sequence_ids": [100],
            "prefix_lens": [3],
            "suffix_lens": [2],
        }
    ]
    assert len(manager.append_calls) == 1
    append_call = manager.append_calls[0]
    assert append_call["k_tensor"] is key
    assert append_call["v_tensor"] is value
    assert append_call["layer_idx"] == 7


def test_gqa_backend_gpu_append_requires_manager():
    backend = GqaPrefixAwareAttentionBackend(
        prefix_kv_builder=_FakePrefixKvBuilder(),
        num_kv_heads=1,
        head_dim=2,
        attention_fn=lambda **kwargs: (kwargs["q"], None),
        layer_idx=0,
        enable_gpu_suffix_append=True,
    )

    with pytest.raises(RuntimeError, match="gpu_paged_kv_manager"):
        backend.forward_prefill(
            query=torch.zeros((2, 2, 2)),
            key=torch.ones((2, 1, 2)),
            value=torch.ones((2, 1, 2)),
            metadata=_metadata(prefix_reuse=True),
            kv_cache_metadata=object(),
        )


def test_gqa_backend_gpu_page_table_attention_uses_manager(monkeypatch):
    monkeypatch.setenv("BATCHGEN_PREFIX_REUSE_GPU_EXTEND_ATTENTION", "1")
    manager = _FakePagedGpuKvManager(
        k_cache=torch.zeros((2, 4, 1, 2)),
        v_cache=torch.zeros((2, 4, 1, 2)),
    )
    kv_cache_metadata = type("KVCache", (), {"gpu_paged_kv_manager": manager})()
    recorded = {}
    builder = _FakePrefixKvBuilder()

    def paged_attention_fn(**kwargs):
        recorded.update(kwargs)
        return kwargs["q"] + 3, None

    backend = GqaPrefixAwareAttentionBackend(
        prefix_kv_builder=builder,
        num_kv_heads=1,
        head_dim=2,
        paged_attention_fn=paged_attention_fn,
        layer_idx=2,
    )
    query = torch.zeros((2, 2, 2))
    key = torch.ones((2, 1, 2))
    value = key + 10

    output = backend.forward_prefill(
        query=query,
        key=key,
        value=value,
        metadata=_metadata(prefix_reuse=True),
        kv_cache_metadata=kv_cache_metadata,
    )

    torch.testing.assert_close(output, query + 3)
    assert builder.prefix_calls == []
    assert recorded["q"].shape == (1, 2, 2, 2)
    assert recorded["cache_seqlens"].tolist() == [5]
    assert recorded["block_table"].tolist() == [[0, 1]]
    assert len(manager.append_calls) == 1


def test_mla_backend_can_append_suffix_kv_to_gpu_manager():
    manager = _FakeGpuKvManager()
    kv_cache_metadata = type("KVCache", (), {"gpu_paged_kv_manager": manager})()
    recorded = {}

    def attention_fn(**kwargs):
        recorded.update(kwargs)
        return kwargs["query_states"] + 2

    backend = MlaProjectedPrefixAwareAttentionBackend(
        prefix_kv_builder=_FakePrefixKvBuilder(),
        page_size=4,
        kv_dim=3,
        num_heads=2,
        kv_lora_rank=1,
        softmax_scale=0.5,
        attention_fn=attention_fn,
        layer_idx=3,
        enable_gpu_suffix_append=True,
    )
    query = torch.zeros((1, 2, 2, 3))
    key = torch.ones((2, 3))

    output = backend.forward_prefill(
        query=query,
        key=key,
        value=None,
        metadata=_metadata(prefix_reuse=True),
        kv_cache_metadata=kv_cache_metadata,
    )

    torch.testing.assert_close(output, query + 2)
    assert recorded["blocked_k"].shape == (2, 4, 1, 3)
    assert manager.prepare_calls == [
        {
            "sequence_ids": [100],
            "prefix_lens": [3],
            "suffix_lens": [2],
        }
    ]
    assert len(manager.append_calls) == 1
    append_call = manager.append_calls[0]
    assert append_call["k_tensor"] is key
    assert append_call["v_tensor"] is None
    assert append_call["layer_idx"] == 3


def test_mla_backend_gpu_page_table_attention_uses_manager(monkeypatch):
    monkeypatch.setenv("BATCHGEN_PREFIX_REUSE_GPU_EXTEND_ATTENTION", "1")
    manager = _FakePagedGpuKvManager(
        k_cache=torch.zeros((2, 4, 1, 3)),
    )
    kv_cache_metadata = type("KVCache", (), {"gpu_paged_kv_manager": manager})()
    builder = _FakePrefixKvBuilder()
    recorded = {}

    def attention_fn(**kwargs):
        recorded.update(kwargs)
        return kwargs["query_states"] + 4

    backend = MlaProjectedPrefixAwareAttentionBackend(
        prefix_kv_builder=builder,
        page_size=4,
        kv_dim=3,
        num_heads=2,
        kv_lora_rank=1,
        softmax_scale=0.5,
        attention_fn=attention_fn,
        layer_idx=4,
    )
    query = torch.zeros((1, 2, 2, 3))
    key = torch.ones((2, 3))

    output = backend.forward_prefill(
        query=query,
        key=key,
        value=None,
        metadata=_metadata(prefix_reuse=True),
        kv_cache_metadata=kv_cache_metadata,
    )

    torch.testing.assert_close(output, query + 4)
    assert builder.prefix_calls == []
    assert recorded["blocked_k"].shape == (2, 4, 1, 3)
    assert recorded["cache_seqlens"].tolist() == [5]
    assert recorded["block_table"].tolist() == [[0, 1]]
    assert len(manager.append_calls) == 1
