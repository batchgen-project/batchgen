from __future__ import annotations

from types import SimpleNamespace

import torch

from batchgen.attention.forward_metadata import (
    PrefillAttentionMetadata,
    PrefixReuseMetadata,
)
from batchgen.models.wrappers.prefix_mla_model_adapters import (
    build_deepseek_prefix_backend_context,
    build_glm5_prefix_backend_context,
    build_kimi_prefix_backend_context,
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
        prefix_reuse=PrefixReuseMetadata(
            prefix_lens=torch.tensor([3], dtype=torch.int32),
            suffix_lens=torch.tensor([2], dtype=torch.int32),
            full_seq_lens=torch.tensor([5], dtype=torch.int32),
            saved_tokens=3,
            is_full_hit=torch.tensor([False], dtype=torch.bool),
            global_sequence_ids=[100],
        ),
    )


def _wrapper():
    module = SimpleNamespace(
        kv_lora_rank=4,
        qk_rope_head_dim=2,
        num_heads=2,
        softmax_scale=0.5,
    )
    return SimpleNamespace(module=module)


def test_mla_model_adapters_accept_explicit_prefill_metadata():
    metadata = _prefill_metadata()
    wrapper = _wrapper()

    contexts = [
        build_deepseek_prefix_backend_context(wrapper=wrapper, metadata=metadata),
        build_glm5_prefix_backend_context(wrapper=wrapper, metadata=metadata),
        build_kimi_prefix_backend_context(wrapper=wrapper, metadata=metadata),
    ]

    for context in contexts:
        assert context.prefix_reuse_mode is True
        assert context.full_hit_mode is False
        assert context.metadata.global_sequence_ids == [100]
        assert context.metadata.prefix_shared_tokens == [3]
        assert context.metadata.full_seq_lengths == [5]
        assert context.rotary_seq_len(metadata.position_ids, fallback_seq_len=2) == 5
