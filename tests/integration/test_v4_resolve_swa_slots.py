from __future__ import annotations

import pytest
import torch

from batchgen.attention.dsa import v4_flashmla_adapter as adapter
from batchgen.kv_cache.deepseek_v4_kv_coordinator import DeepSeekV4KVCoordinator

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


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


def _slow_resolve(coordinator, sequence_ids, positions):
    slots = [
        coordinator.swa.sequence_token_slots(seq_id, [int(position.item())])[0]
        for seq_id, position in zip(sequence_ids, positions)
    ]
    return torch.stack(slots).to(dtype=torch.int32, device=positions.device)


@pytest.mark.parametrize("position", [0, 1, 127, 128, 129, 255, 256, 300])
def test_fast_resolve_matches_slow_single(position):
    coordinator, device = _coordinator(position + 1)
    sequence_ids = [31337]
    try:
        coordinator.allocate_pages_for_sequences(sequence_ids, [position + 1])
        coordinator.rebuild_page_table(sequence_ids)
        positions = torch.tensor([position], dtype=torch.long, device=device)

        slow = _slow_resolve(coordinator, sequence_ids, positions)
        fast = adapter._resolve_swa_token_slots(
            coordinator, sequence_ids, positions
        )
        assert torch.equal(fast, slow)
        assert fast.dtype == torch.int32
    finally:
        coordinator.destroy()


def test_fast_resolve_multi_sequence():
    device = torch.device("cuda")
    coordinator = DeepSeekV4KVCoordinator(
        compress_ratios=[0, 4, 128],
        num_pages=512,
        device=device,
        base_page_size=256,
    )
    coordinator.initialize()
    sequence_ids = [11, 22, 33]
    seq_lens = [130, 256, 64]
    try:
        coordinator.allocate_pages_for_sequences(sequence_ids, seq_lens)
        coordinator.rebuild_page_table(sequence_ids)
        positions = torch.tensor(
            [s - 1 for s in seq_lens], dtype=torch.long, device=device
        )
        slow = _slow_resolve(coordinator, sequence_ids, positions)
        fast = adapter._resolve_swa_token_slots(
            coordinator, sequence_ids, positions
        )
        assert torch.equal(fast, slow)
    finally:
        coordinator.destroy()
