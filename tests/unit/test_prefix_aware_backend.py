from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from batchgen.attention.forward_metadata import (
    ForwardBatchMetadata,
    KVCacheMetadata,
    PrefillAttentionMetadata,
)
from batchgen.attention.forward_metadata_context import bind_forward_batch_metadata
from batchgen.attention.prefix_aware_backend import (
    GqaPrefixAwareAttentionBackend,
    MlaProjectedPrefixAwareAttentionBackend,
)
from batchgen.models.wrappers.prefix_cache import PrefixCachePrepackMetadata
from batchgen.models.wrappers.prefix_gqa_replay import (
    GqaReplaySpec,
    run_prefix_gqa_prefill_attention,
)


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


class _FakeGqaReplayWrapper:
    def __init__(self, builder):
        self._builder = builder

    def prefix_attention_kv_builder(self):
        return self._builder


def test_gqa_replay_passes_bound_kv_cache_metadata(monkeypatch):
    recorded = {}

    def fake_forward_prefill(self, **kwargs):
        recorded.update(kwargs)
        return kwargs["query"]

    monkeypatch.setattr(
        GqaPrefixAwareAttentionBackend,
        "forward_prefill",
        fake_forward_prefill,
    )
    kv_cache = KVCacheMetadata(
        prefill_prefix_materialization=object(),
    )
    forward_metadata = ForwardBatchMetadata(
        phase="prefill",
        global_sequence_ids=[100],
        prefill=PrefillAttentionMetadata(
            cu_seqlens_q=torch.tensor([0, 2], dtype=torch.int32),
            cu_seqlens_k=torch.tensor([0, 5], dtype=torch.int32),
            max_seqlen_q=2,
            max_seqlen_k=5,
            q_seq_lens=[2],
            kv_seq_lens=[5],
            position_ids=torch.tensor([3, 4], dtype=torch.int64),
        ),
        kv_cache=kv_cache,
    )
    query = torch.zeros((2, 2, 2))
    key = torch.ones((2, 1, 2))
    value = key + 10

    with bind_forward_batch_metadata(forward_metadata):
        output = run_prefix_gqa_prefill_attention(
            wrapper=_FakeGqaReplayWrapper(_FakePrefixKvBuilder()),
            query=query,
            key=key,
            value=value,
            metadata=_metadata(prefix_reuse=True),
            spec=GqaReplaySpec(num_kv_heads=1, head_dim=2),
        )

    assert output is query
    assert recorded["kv_cache_metadata"] is kv_cache


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


def test_mla_backend_prefix_reuse_requires_gpu_materialization():
    backend = MlaProjectedPrefixAwareAttentionBackend(
        prefix_kv_builder=_FakePrefixKvBuilder(),
        page_size=4,
        kv_dim=3,
        num_heads=2,
        kv_lora_rank=1,
        softmax_scale=0.5,
    )
    query = torch.zeros((1, 2, 2, 3))
    key = torch.ones((2, 3))

    with pytest.raises(RuntimeError, match="GPU paged materialization"):
        backend.forward_prefill(
            query=query,
            key=key,
            value=None,
            metadata=_metadata(prefix_reuse=True),
            kv_cache_metadata=object(),
        )


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
            cache_seqlens=torch.tensor([5], dtype=torch.int32),
            slot_indices=torch.tensor([0], dtype=torch.int32),
        )
        self.waited_layers = []

    def wait_for_layer(self, layer_idx):
        self.waited_layers.append(int(layer_idx))


def test_mla_backend_prefix_reuse_uses_flashinfer_gpu_materialization(monkeypatch):
    recorded = {}

    from batchgen.attention.mla import flashinfer_paged_prefill

    def flashinfer_fn(**kwargs):
        recorded.update(kwargs)
        return torch.full((1, 2, 2, 1), 3.0)

    monkeypatch.setattr(
        flashinfer_paged_prefill,
        "run_flashinfer_mla_paged_prefill",
        flashinfer_fn,
    )

    builder = _FakePrefixKvBuilder()
    materialization = _FakeMlaMaterialization()
    backend = MlaProjectedPrefixAwareAttentionBackend(
        prefix_kv_builder=builder,
        page_size=4,
        kv_dim=3,
        num_heads=2,
        kv_lora_rank=1,
        softmax_scale=0.5,
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

    torch.testing.assert_close(output, torch.full((1, 2, 2, 1), 3.0))
    assert materialization.waited_layers == [2]
    assert len(materialization.manager.append_calls) == 1
    append_call = materialization.manager.append_calls[0]
    assert append_call["k_tensor"] is key
    assert append_call["v_tensor"] is None
    assert append_call["layer_idx"] == 2
    assert builder.prefix_calls == []
    assert recorded["compressed_kv_cache"] is materialization.manager.blocked_k
    assert recorded["page_table"] is materialization.manager.block_table
    assert recorded["cache_seqlens"].tolist() == [5]
    assert recorded["slot_indices"].tolist() == [0]


def test_mla_backend_full_hit_uses_flashinfer_gpu_materialization(monkeypatch):
    recorded = {}

    from batchgen.attention.mla import flashinfer_paged_prefill

    def flashinfer_fn(**kwargs):
        recorded.update(kwargs)
        return torch.full((1, 1, 2, 1), 4.0)

    monkeypatch.setattr(
        flashinfer_paged_prefill,
        "run_flashinfer_mla_paged_prefill",
        flashinfer_fn,
    )

    materialization = _FakeMlaMaterialization()
    backend = MlaProjectedPrefixAwareAttentionBackend(
        prefix_kv_builder=_FakePrefixKvBuilder(),
        page_size=4,
        kv_dim=3,
        num_heads=2,
        kv_lora_rank=1,
        softmax_scale=0.5,
    )

    output = backend.forward_prefill(
        query=torch.zeros((1, 1, 2, 3)),
        key=None,
        value=None,
        metadata=_metadata(full_hit=True),
        kv_cache_metadata=SimpleNamespace(
            prefill_prefix_materialization=materialization
        ),
    )

    torch.testing.assert_close(output, torch.full((1, 1, 2, 1), 4.0))
    assert materialization.waited_layers == [2]
    assert materialization.manager.append_calls == []
    assert recorded["compressed_kv_cache"] is materialization.manager.blocked_k
    assert recorded["page_table"] is materialization.manager.block_table
    assert recorded["cache_seqlens"].tolist() == [5]
    assert recorded["slot_indices"].tolist() == [0]
