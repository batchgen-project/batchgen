from __future__ import annotations

import pytest
import torch

from batchgen.attention.dsa import v4_flashmla_adapter as adapter
from batchgen.kv_cache.deepseek_v4_kv_coordinator import DeepSeekV4KVCoordinator

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)

_WINDOW = 128


def _coordinator(seq_len):
    device = torch.device("cuda")
    coordinator = DeepSeekV4KVCoordinator(
        compress_ratios=[0, 4, 128],
        num_pages=max(256, seq_len + 16),
        device=device,
        base_page_size=256,
    )
    coordinator.initialize()
    return coordinator, device


def _slow_swa(coordinator, sequence_ids, cache_seqlens):
    window = _WINDOW
    lengths = torch.minimum(
        cache_seqlens.to(dtype=torch.long),
        torch.full_like(cache_seqlens.to(dtype=torch.long), window),
    )
    padded_topk = (
        adapter._aligned_topk(int(lengths.max().item()))
        if lengths.numel()
        else 0
    )
    starts = (cache_seqlens.to(dtype=torch.long) - lengths).clamp_min(0)
    offsets = torch.arange(
        padded_topk, device=cache_seqlens.device, dtype=torch.long
    )
    logical = starts[:, None] + offsets[None, :]
    logical = torch.where(
        offsets[None, :] < lengths[:, None],
        logical,
        torch.full_like(logical, -1),
    )
    fallback = [
        row[row >= 0].to(dtype=torch.long, device=cache_seqlens.device)
        for row in logical
    ]
    return adapter._build_slot_indices_from_positions(
        coordinator.swa, sequence_ids, fallback, device=cache_seqlens.device
    )


@pytest.mark.parametrize(
    "seq_len",
    [1, 64, 127, 128, 129, 200, 255, 256, 257, 300, 512],
)
def test_fast_swa_matches_slow(seq_len):
    coordinator, device = _coordinator(seq_len)
    sequence_ids = [31337]
    try:
        coordinator.allocate_pages_for_sequences(sequence_ids, [seq_len])
        coordinator.rebuild_page_table(sequence_ids)
        cache_seqlens = torch.tensor(
            [seq_len], dtype=torch.int32, device=device
        )

        slow_idx, slow_len = _slow_swa(coordinator, sequence_ids, cache_seqlens)
        fast_idx, fast_len = adapter._build_swa_window_indices_fast(
            coordinator, sequence_ids, cache_seqlens, window=_WINDOW
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


def test_fast_swa_after_page_extension():
    coordinator, device = _coordinator(seq_len=600)
    sequence_ids = [99]
    try:
        coordinator.allocate_pages_for_sequences(sequence_ids, [10])
        for target in (10, 128, 256, 384, 600):
            coordinator.allocate_pages_for_sequences(sequence_ids, [target])
            coordinator.rebuild_page_table(sequence_ids)
            cache_seqlens = torch.tensor(
                [target], dtype=torch.int32, device=device
            )
            slow_idx, slow_len = _slow_swa(
                coordinator, sequence_ids, cache_seqlens
            )
            fast_idx, fast_len = adapter._build_swa_window_indices_fast(
                coordinator, sequence_ids, cache_seqlens, window=_WINDOW
            )
            assert fast_idx is not None, f"unavailable at {target}"
            assert torch.equal(fast_len, slow_len)
            n = int(slow_len[0].item())
            assert torch.equal(fast_idx[0, 0, :n], slow_idx[0, 0, :n])
    finally:
        coordinator.destroy()


def test_fast_swa_disabled_without_page_table():
    coordinator, device = _coordinator(seq_len=200)
    sequence_ids = [7]
    try:
        coordinator.allocate_pages_for_sequences(sequence_ids, [200])
        coordinator.swa._clear_page_table()
        cache_seqlens = torch.tensor([200], dtype=torch.int32, device=device)
        fast_idx, _ = adapter._build_swa_window_indices_fast(
            coordinator, sequence_ids, cache_seqlens, window=_WINDOW
        )
        assert fast_idx is None
    finally:
        coordinator.destroy()


def test_dispatcher_swa_equivalence():
    coordinator, device = _coordinator(seq_len=300)
    sequence_ids = [5]
    try:
        coordinator.allocate_pages_for_sequences(sequence_ids, [300])
        coordinator.rebuild_page_table(sequence_ids)
        cache_seqlens = torch.tensor([300], dtype=torch.int32, device=device)

        slow_idx, slow_len = _slow_swa(coordinator, sequence_ids, cache_seqlens)
        disp_idx, disp_len = adapter._build_swa_window_indices(
            coordinator, sequence_ids, cache_seqlens
        )
        assert torch.equal(disp_len, slow_len)
        n = int(slow_len[0].item())
        assert torch.equal(disp_idx[0, 0, :n], slow_idx[0, 0, :n])
    finally:
        coordinator.destroy()
