from __future__ import annotations

import pytest
import torch

from batchgen.attention.dsa import v4_flashmla_adapter as adapter
from batchgen.kv_cache.deepseek_v4_kv_coordinator import DeepSeekV4KVCoordinator

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


def _coordinator(seq_len, base_page_size):
    device = torch.device("cuda")
    coordinator = DeepSeekV4KVCoordinator(
        compress_ratios=[0, 4, 128],
        num_pages=max(256, seq_len + 16),
        device=device,
        base_page_size=base_page_size,
    )
    coordinator.initialize()
    return coordinator, device


@pytest.mark.parametrize(
    "seq_len,base_page_size",
    [
        (1, 256),
        (127, 256),
        (128, 256),
        (129, 256),
        (255, 256),
        (256, 256),
        (257, 256),
        (300, 256),
        (512, 256),
    ],
)
def test_fast_prefix_matches_slow(seq_len, base_page_size):
    coordinator, device = _coordinator(seq_len, base_page_size)
    sequence_ids = [31337]
    try:
        coordinator.allocate_pages_for_sequences(sequence_ids, [seq_len])
        coordinator.rebuild_page_table(sequence_ids)
        cache_seqlens = torch.tensor(
            [seq_len], dtype=torch.int32, device=device
        )

        slow_idx, slow_len = adapter._build_full_prefix_indices_slow(
            coordinator, sequence_ids, cache_seqlens
        )
        fast_idx, fast_len = adapter._build_full_prefix_indices_fast(
            coordinator, sequence_ids, cache_seqlens
        )

        assert fast_idx is not None
        assert torch.equal(fast_len, slow_len)
        n = int(slow_len[0].item())
        assert torch.equal(fast_idx[0, 0, :n], slow_idx[0, 0, :n])
        capacity = coordinator.swa.num_pages * coordinator.swa.page_size_tokens
        assert (fast_idx[0, 0, :n] >= 0).all()
        assert (fast_idx[0, 0, :n] < capacity).all()
        assert (fast_idx[0, 0, n:] == -1).all()
    finally:
        coordinator.destroy()


def test_fast_prefix_after_page_extension():
    coordinator, device = _coordinator(seq_len=600, base_page_size=256)
    sequence_ids = [99]
    try:
        coordinator.allocate_pages_for_sequences(sequence_ids, [10])
        for target in (10, 256, 257, 600):
            coordinator.allocate_pages_for_sequences(sequence_ids, [target])
            coordinator.rebuild_page_table(sequence_ids)
            cache_seqlens = torch.tensor(
                [target], dtype=torch.int32, device=device
            )
            slow_idx, slow_len = adapter._build_full_prefix_indices_slow(
                coordinator, sequence_ids, cache_seqlens
            )
            fast_idx, fast_len = adapter._build_full_prefix_indices_fast(
                coordinator, sequence_ids, cache_seqlens
            )
            assert fast_idx is not None, f"fast path unavailable at {target}"
            assert torch.equal(fast_len, slow_len)
            n = int(slow_len[0].item())
            assert torch.equal(fast_idx[0, 0, :n], slow_idx[0, 0, :n])
    finally:
        coordinator.destroy()


def test_fast_prefix_disabled_without_page_table():
    coordinator, device = _coordinator(seq_len=128, base_page_size=256)
    sequence_ids = [7]
    try:
        coordinator.allocate_pages_for_sequences(sequence_ids, [128])
        coordinator.swa._clear_page_table()
        cache_seqlens = torch.tensor([128], dtype=torch.int32, device=device)
        fast_idx, _ = adapter._build_full_prefix_indices_fast(
            coordinator, sequence_ids, cache_seqlens
        )
        assert fast_idx is None
    finally:
        coordinator.destroy()


def test_dispatcher_equivalence():
    coordinator, device = _coordinator(seq_len=300, base_page_size=256)
    sequence_ids = [5]
    try:
        coordinator.allocate_pages_for_sequences(sequence_ids, [300])
        coordinator.rebuild_page_table(sequence_ids)
        cache_seqlens = torch.tensor([300], dtype=torch.int32, device=device)

        slow_idx, slow_len = adapter._build_full_prefix_indices_slow(
            coordinator, sequence_ids, cache_seqlens
        )
        disp_idx, disp_len = adapter._build_full_prefix_indices(
            coordinator, sequence_ids, cache_seqlens
        )
        assert torch.equal(disp_len, slow_len)
        n = int(slow_len[0].item())
        assert torch.equal(disp_idx[0, 0, :n], slow_idx[0, 0, :n])
    finally:
        coordinator.destroy()
