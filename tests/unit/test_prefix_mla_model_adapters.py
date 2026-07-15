from __future__ import annotations

import importlib
import sys
import types
from types import SimpleNamespace

import torch

from batchgen.attention.forward_metadata import (
    ForwardBatchMetadata,
    PrefillAttentionMetadata,
)


def _prefill_metadata() -> PrefillAttentionMetadata:
    return PrefillAttentionMetadata(
        cu_seqlens_q=torch.tensor([0, 2], dtype=torch.int32),
        cu_seqlens_k=torch.tensor([0, 5], dtype=torch.int32),
        max_seqlen_q=2,
        max_seqlen_k=5,
        q_seq_lens=[2],
        kv_seq_lens=[5],
        position_ids=torch.tensor([3, 4], dtype=torch.int64),
    )


def _wrapper():
    module = SimpleNamespace(
        kv_lora_rank=4,
        qk_rope_head_dim=2,
        num_heads=2,
        softmax_scale=0.5,
    )
    return SimpleNamespace(module=module)


def _prefix_mla_adapters(monkeypatch):
    kv_cache_stub = types.ModuleType("batchgen.kv_cache")
    kv_cache_stub.__path__ = []
    monkeypatch.setitem(sys.modules, "batchgen.kv_cache", kv_cache_stub)
    prefill_offload_stub = types.ModuleType("batchgen.kv_cache.prefill_offload")
    prefill_offload_stub.PrefillHostKVOffloader = object
    monkeypatch.setitem(
        sys.modules,
        "batchgen.kv_cache.prefill_offload",
        prefill_offload_stub,
    )
    return importlib.import_module(
        "batchgen.models.wrappers.prefix_mla_model_adapters"
    )


def test_mla_model_adapters_accept_explicit_prefill_metadata(monkeypatch):
    adapters = _prefix_mla_adapters(monkeypatch)
    prefill = _prefill_metadata()
    metadata = ForwardBatchMetadata(
        phase="prefill",
        global_sequence_ids=[100],
        prefill=prefill,
    )
    wrapper = _wrapper()

    contexts = [
        adapters.build_deepseek_prefix_backend_context(
            wrapper=wrapper, metadata=metadata
        ),
        adapters.build_glm5_prefix_backend_context(
            wrapper=wrapper, metadata=metadata
        ),
        adapters.build_kimi_prefix_backend_context(
            wrapper=wrapper, metadata=metadata
        ),
    ]

    for context in contexts:
        assert context.prefix_reuse_mode is True
        assert context.metadata.global_sequence_ids == [100]
        assert context.metadata.prefix_shared_tokens == [3]
        assert context.metadata.full_seq_lengths == [5]
        assert (
            context.rotary_seq_len(prefill.position_ids, fallback_seq_len=2)
            == 5
        )
