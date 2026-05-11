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
        self.reader = SimpleNamespace(layer_idx=2)

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
        cu_seqlens_cpu=[int(value) for value in cu_seqlens.tolist()],
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


def test_gqa_backend_prefix_reuse_requires_gpu_materialization():
    builder = _FakePrefixKvBuilder()
    backend = GqaPrefixAwareAttentionBackend(
        prefix_kv_builder=builder,
        num_kv_heads=1,
        head_dim=2,
    )
    query = torch.zeros((2, 2, 2))
    key = torch.ones((2, 1, 2))
    value = key + 10

    with pytest.raises(RuntimeError, match="GPU paged materialization"):
        backend.forward_prefill(
            query=query,
            key=key,
            value=value,
            metadata=_metadata(prefix_reuse=True),
        )


def test_gqa_backend_full_hit_requires_gpu_materialization():
    builder = _FakePrefixKvBuilder()
    backend = GqaPrefixAwareAttentionBackend(
        prefix_kv_builder=builder,
        num_kv_heads=1,
        head_dim=2,
    )

    with pytest.raises(RuntimeError, match="GPU paged materialization"):
        backend.forward_prefill(
            query=torch.zeros((1, 2, 2)),
            key=torch.ones((1, 1, 2)),
            value=torch.ones((1, 1, 2)),
            metadata=_metadata(full_hit=True),
        )


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


def test_mla_backend_prefix_reuse_uses_host_replay_path():
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
    )
    query = torch.zeros((1, 2, 2, 3))
    key = torch.ones((2, 3))

    output = backend.forward_prefill(
        query=query,
        key=key,
        value=None,
        metadata=_metadata(prefix_reuse=True),
        kv_cache_metadata=object(),
    )

    torch.testing.assert_close(output, query + 2)
    assert recorded["blocked_k"].shape == (2, 4, 1, 3)
    assert recorded["cache_seqlens"].tolist() == [5]


class _FakeMlaMaterializedManager:
    def __init__(self):
        self.config = SimpleNamespace(has_v_cache=False)
        self.blocked_k = torch.zeros((3, 4, 1, 3))
        self.block_table = torch.tensor([[0, 1, 2]], dtype=torch.int32)
        self.append_calls = []

    def append_layer_prefill_suffix_tokens(self, **kwargs):
        self.append_calls.append(kwargs)

    def get_layer_kv_with_page_table(self, layer_idx):
        assert layer_idx == 2
        return self.blocked_k, None, self.block_table


class _FakeMlaMaterialization:
    def __init__(self):
        self.manager = _FakeMlaMaterializedManager()
        self.append_plan = SimpleNamespace(
            cache_seqlens=torch.tensor([5], dtype=torch.int32)
        )
        self.waited = False

    def wait_for_load(self):
        self.waited = True


def test_mla_backend_prefix_reuse_uses_gpu_materialization():
    recorded = {}

    def attention_fn(**kwargs):
        recorded.update(kwargs)
        return kwargs["query_states"] + 3

    builder = _FakePrefixKvBuilder()
    materialization = _FakeMlaMaterialization()
    backend = MlaProjectedPrefixAwareAttentionBackend(
        prefix_kv_builder=builder,
        page_size=4,
        kv_dim=3,
        num_heads=2,
        kv_lora_rank=1,
        softmax_scale=0.5,
        attention_fn=attention_fn,
    )
    query = torch.zeros((1, 2, 2, 3))
    key = torch.ones((2, 3))

    output = backend.forward_prefill(
        query=query,
        key=key,
        value=None,
        metadata=_metadata(prefix_reuse=True),
        kv_cache_metadata=SimpleNamespace(
            prefill_prefix_materialization=materialization
        ),
    )

    torch.testing.assert_close(output, query + 3)
    assert materialization.waited
    assert len(materialization.manager.append_calls) == 1
    append_call = materialization.manager.append_calls[0]
    assert append_call["k_tensor"] is key
    assert append_call["v_tensor"] is None
    assert append_call["layer_idx"] == 2
    assert builder.prefix_calls == []
    assert recorded["blocked_k"] is materialization.manager.blocked_k
    assert recorded["block_table"] is materialization.manager.block_table
    assert recorded["cache_seqlens"].tolist() == [5]
    assert recorded["query_len"] == 2
