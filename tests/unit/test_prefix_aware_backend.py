from __future__ import annotations

import pytest
import torch

from batchgen.attention.prefix_aware_backend import GqaPrefixAwareAttentionBackend
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
