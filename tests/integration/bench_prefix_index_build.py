from __future__ import annotations

import torch

from batchgen.attention.dsa import v4_flashmla_adapter as adapter
from batchgen.kv_cache.deepseek_v4_kv_coordinator import DeepSeekV4KVCoordinator


def _bench(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def run(seq_len):
    device = torch.device("cuda")
    coordinator = DeepSeekV4KVCoordinator(
        compress_ratios=[0, 4, 128],
        num_pages=max(256, seq_len + 16),
        device=device,
        base_page_size=256,
    )
    coordinator.initialize()
    sequence_ids = [31337]
    try:
        coordinator.allocate_pages_for_sequences(sequence_ids, [seq_len])
        coordinator.rebuild_page_table(sequence_ids)
        cache_seqlens = torch.tensor(
            [seq_len], dtype=torch.int32, device=device
        )

        slow_ms = _bench(
            lambda: adapter._build_full_prefix_indices_slow(
                coordinator, sequence_ids, cache_seqlens
            )
        )
        fast_ms = _bench(
            lambda: adapter._build_full_prefix_indices_fast(
                coordinator, sequence_ids, cache_seqlens
            )
        )
        return slow_ms, fast_ms
    finally:
        coordinator.destroy()


def main():
    name = torch.cuda.get_device_name(0)
    print(f"device={name}")
    print(f"\n{'seq_len':>8} {'slow_ms':>9} {'fast_ms':>9} {'speedup':>9}")
    for seq_len in (128, 512, 2048, 8192):
        slow, fast = run(seq_len)
        print(f"{seq_len:>8} {slow:>9.4f} {fast:>9.4f} {slow / fast:>8.2f}x")


if __name__ == "__main__":
    main()
