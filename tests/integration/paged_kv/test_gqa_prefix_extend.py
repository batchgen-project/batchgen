import pytest
import torch

from batchgen.attention.gqa import gqa_extend_fa, gqa_prefill_fa
from batchgen.attention.gqa import fa_extend, fa_prefill
from batchgen.kv_cache.gpu_paged_kv_manager import (
    GPUPagedKVCacheManager,
    GPUPagedKVConfig,
)


pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
    pytest.mark.skipif(
        fa_extend._flash_with_kvcache is None
        or fa_prefill._flash_varlen_func is None,
        reason="FlashAttention paged and varlen kernels are required",
    ),
]


def _split_suffix_rows(
    tensor: torch.Tensor,
    *,
    full_lengths: list[int],
    suffix_lengths: list[int],
) -> torch.Tensor:
    rows = []
    start = 0
    for full, suffix in zip(full_lengths, suffix_lengths):
        rows.append(tensor[start + full - suffix : start + full])
        start += full
    return torch.cat(rows, dim=0)


def test_gqa_paged_prefix_extend_matches_full_prefill():
    torch.manual_seed(17)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    sequence_ids = [101, 202]
    prefix_lengths = [256, 128]
    suffix_lengths = [17, 31]
    full_lengths = [
        prefix + suffix
        for prefix, suffix in zip(prefix_lengths, suffix_lengths)
    ]
    q_heads = 4
    kv_heads = 2
    head_dim = 64
    dtype = torch.bfloat16

    q_full = torch.randn(
        sum(full_lengths), q_heads, head_dim, device=device, dtype=dtype
    )
    k_full = torch.randn(
        sum(full_lengths), kv_heads, head_dim, device=device, dtype=dtype
    )
    v_full = torch.randn_like(k_full)
    cu_full = torch.tensor(
        [0, full_lengths[0], sum(full_lengths)],
        device=device,
        dtype=torch.int32,
    )
    reference, _ = gqa_prefill_fa(
        q=q_full,
        k=k_full,
        v=v_full,
        cu_seqlens_q=cu_full,
        cu_seqlens_k=cu_full,
        max_seqlen_q=max(full_lengths),
        max_seqlen_k=max(full_lengths),
    )

    manager = GPUPagedKVCacheManager(
        config=GPUPagedKVConfig(
            num_layers=1,
            num_pages=sum((length + 255) // 256 for length in full_lengths),
            page_size_tokens=256,
            num_k_heads=kv_heads,
            k_head_dim=head_dim,
            num_v_heads=kv_heads,
            v_head_dim=head_dim,
            kv_dtype=dtype,
        ),
        device=device,
    )
    manager.initialize()
    try:
        manager.allocate_pages_for_sequences(sequence_ids, full_lengths)
        manager.rebuild_page_table(sequence_ids)
        full_write_plan = manager.prepare_prefill_suffix_append(
            sequence_ids=sequence_ids,
            prefix_lens=[0, 0],
            suffix_lens=full_lengths,
            rebuild_page_table=False,
        )
        manager.append_layer_prefill_suffix_tokens(
            k_tensor=k_full,
            v_tensor=v_full,
            append_plan=full_write_plan,
            layer_idx=0,
        )
        extend_plan = manager.prepare_prefill_suffix_append(
            sequence_ids=sequence_ids,
            prefix_lens=prefix_lengths,
            suffix_lens=suffix_lengths,
            rebuild_page_table=False,
        )
        k_cache, v_cache, page_table = manager.get_layer_kv_with_page_table(0)
        q_suffix = _split_suffix_rows(
            q_full,
            full_lengths=full_lengths,
            suffix_lengths=suffix_lengths,
        )
        cu_suffix = torch.tensor(
            [0, suffix_lengths[0], sum(suffix_lengths)],
            device=device,
            dtype=torch.int32,
        )
        actual, _ = gqa_extend_fa(
            q=q_suffix,
            k_cache=k_cache,
            v_cache=v_cache,
            cache_seqlens=extend_plan.cache_seqlens,
            page_table=page_table,
            cu_seqlens_q=cu_suffix,
            max_seqlen_q=max(suffix_lengths),
        )
        expected = _split_suffix_rows(
            reference,
            full_lengths=full_lengths,
            suffix_lengths=suffix_lengths,
        )
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    finally:
        manager.destroy()
