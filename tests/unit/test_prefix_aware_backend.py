from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from batchgen.attention.forward_metadata import (
    ForwardBatchMetadata,
    KVCacheMetadata,
    PrefillAttentionMetadata,
)
from batchgen.attention.forward_metadata_context import (
    bind_forward_batch_metadata,
)
from batchgen.attention.prefix_aware_backend import (
    GqaPrefixAwareAttentionBackend,
)
from batchgen.models.wrappers.prefix_gqa_extend import (
    GqaExtendSpec,
    run_prefix_gqa_prefill_attention,
)
from batchgen.prefix_reuse.materialization import PrefixMaterializationBundle

_LAYER_IDX = 2


def _metadata(
    *,
    prefix_reuse: bool = False,
) -> ForwardBatchMetadata:
    cu_seqlens = torch.tensor([0, 2], dtype=torch.int32)
    max_seqlen = 2
    seq_lengths = [2]
    kv_seq_lengths = [5] if prefix_reuse else list(seq_lengths)
    return ForwardBatchMetadata(
        phase="prefill",
        global_sequence_ids=[100],
        prefill=PrefillAttentionMetadata(
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=torch.tensor(
                [0, kv_seq_lengths[0]],
                dtype=torch.int32,
            ),
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max(kv_seq_lengths),
            q_seq_lens=seq_lengths,
            kv_seq_lens=kv_seq_lengths,
            position_ids=torch.tensor([0, 1], dtype=torch.int64),
            append_seq_lens=seq_lengths,
        ),
    )


def _clamped_full_hit_metadata() -> ForwardBatchMetadata:
    return ForwardBatchMetadata(
        phase="prefill",
        global_sequence_ids=[100],
        prefill=PrefillAttentionMetadata(
            cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32),
            cu_seqlens_k=torch.tensor([0, 5], dtype=torch.int32),
            max_seqlen_q=1,
            max_seqlen_k=5,
            q_seq_lens=[1],
            kv_seq_lens=[5],
            position_ids=torch.tensor([4], dtype=torch.int64),
            append_seq_lens=[1],
        ),
    )


def test_gqa_backend_no_prefix_uses_query_cu_seqlens_for_kv():
    recorded = {}

    def attention_fn(**kwargs):
        recorded.update(kwargs)
        return kwargs["q"] + 1, None

    backend = GqaPrefixAwareAttentionBackend(
        layer_idx=_LAYER_IDX,
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
    assert recorded["k"] is key
    assert recorded["v"] is value
    assert recorded["cu_seqlens_q"].tolist() == [0, 2]
    assert recorded["cu_seqlens_k"].tolist() == [0, 2]
    assert recorded["max_seqlen_q"] == 2
    assert recorded["max_seqlen_k"] == 2


def test_gqa_backend_prefix_reuse_requires_gpu_materialization():
    backend = GqaPrefixAwareAttentionBackend(
        layer_idx=_LAYER_IDX,
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


class _FakeGqaMaterializedManager:
    def __init__(self):
        self.k_cache = torch.zeros((4, 4, 1, 2))
        self.v_cache = torch.ones((4, 4, 1, 2))
        self.append_calls = []
        self.page_table = torch.tensor(
            [
                [0, 1],
                [2, 3],
            ],
            dtype=torch.int32,
        )

    def get_layer_kv_with_page_table(self, layer_idx):
        assert layer_idx == _LAYER_IDX
        return self.k_cache, self.v_cache, self.page_table

    def append_layer_prefill_suffix_tokens(self, **kwargs):
        self.append_calls.append(kwargs)


class _FakeGqaMaterialization:
    def __init__(self):
        self.manager = _FakeGqaMaterializedManager()
        self.append_plan = SimpleNamespace(
            slot_values=torch.tensor([0], dtype=torch.int32),
            cache_seqlens=torch.tensor([5], dtype=torch.int32),
        )
        self.waited_layers = []

    def wait_for_layer(self, layer_idx):
        self.waited_layers.append(int(layer_idx))


def test_gqa_backend_clamped_full_hit_uses_extend_prefill(monkeypatch):
    recorded = {}

    import batchgen.attention.gqa as gqa

    def fake_extend(**kwargs):
        recorded.update(kwargs)
        return kwargs["q"] + 10, None

    monkeypatch.setattr(gqa, "gqa_extend_fa", fake_extend)

    materialization = _FakeGqaMaterialization()
    backend = GqaPrefixAwareAttentionBackend(
        layer_idx=_LAYER_IDX,
        num_kv_heads=1,
        head_dim=2,
    )
    query = torch.arange(4, dtype=torch.float32).reshape(1, 2, 2)
    key = torch.ones((1, 1, 2))
    value = key + 1

    output = backend.forward_prefill(
        query=query,
        key=key,
        value=value,
        metadata=_clamped_full_hit_metadata(),
        kv_cache_metadata=SimpleNamespace(
            prefill_prefix_materialization=PrefixMaterializationBundle(
                by_group_id={0: materialization}
            )
        ),
    )

    torch.testing.assert_close(output, query + 10)
    assert materialization.waited_layers == [_LAYER_IDX]
    assert materialization.manager.append_calls[0]["k_tensor"] is key
    assert materialization.manager.append_calls[0]["v_tensor"] is value
    assert recorded["q"] is query
    assert recorded["k_cache"] is materialization.manager.k_cache
    assert recorded["v_cache"] is materialization.manager.v_cache
    assert recorded["cache_seqlens"].tolist() == [5]


class _FakeGqaExtendWrapper:
    layer_idx = _LAYER_IDX


def test_gqa_extend_passes_bound_kv_cache_metadata(monkeypatch):
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
            wrapper=_FakeGqaExtendWrapper(),
            query=query,
            key=key,
            value=value,
            metadata=_metadata(prefix_reuse=True),
            spec=GqaExtendSpec(num_kv_heads=1, head_dim=2),
        )

    assert output is query
    assert recorded["kv_cache_metadata"] is kv_cache


def test_gqa_backend_missing_value_raises():
    backend = GqaPrefixAwareAttentionBackend(
        layer_idx=_LAYER_IDX,
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
        layer_idx=_LAYER_IDX,
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
