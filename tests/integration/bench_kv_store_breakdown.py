from __future__ import annotations

import torch

from batchgen.attention.dsa import v4_flashmla_adapter as adapter
from batchgen.kv_cache.deepseek_v4_kv_coordinator import DeepSeekV4KVCoordinator

HEAD_DIM = 512


def _make_rope_cache(max_pos, rope_dim=64, base=10000.0):
    device = torch.device("cuda")
    inv = 1.0 / (
        base
        ** (
            torch.arange(0, rope_dim, 2, device=device, dtype=torch.float32)
            / rope_dim
        )
    )
    ang = torch.outer(
        torch.arange(max_pos, device=device, dtype=torch.float32), inv
    )
    return torch.cat((ang.cos(), ang.sin()), dim=-1)


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


def run(seq_len=200):
    device = torch.device("cuda")
    coordinator = DeepSeekV4KVCoordinator(
        compress_ratios=[0, 4, 128],
        num_pages=256,
        device=device,
        base_page_size=256,
    )
    coordinator.initialize()
    sequence_ids = [31337]
    try:
        coordinator.allocate_pages_for_sequences(sequence_ids, [seq_len])
        coordinator.rebuild_page_table(sequence_ids)
        positions = torch.tensor([seq_len - 1], dtype=torch.long, device=device)
        rope_cache = _make_rope_cache(seq_len + 4)
        kv = torch.randn(1, HEAD_DIM, dtype=torch.bfloat16, device=device)
        slots = torch.zeros(1, dtype=torch.int32, device=device)

        rope_ms = _bench(lambda: adapter._apply_rope(kv, positions, rope_cache))
        resolve_ms = _bench(
            lambda: adapter._resolve_swa_token_slots(
                coordinator, sequence_ids, positions
            )
        )
        store_ms = _bench(
            lambda: coordinator.swa.store_kv(
                layer_idx=0, token_slots=slots, kv_processed=kv
            )
        )
        return rope_ms, resolve_ms, store_ms
    finally:
        coordinator.destroy()


def main():
    print(f"device={torch.cuda.get_device_name(0)}")
    rope, resolve, store = run()
    print(f"\n{'subpiece':>26} {'ms':>9}")
    print(f"{'_apply_rope':>26} {rope:>9.4f}")
    print(f"{'_resolve_swa_token_slots':>26} {resolve:>9.4f}")
    print(f"{'store_kv (pack+scatter)':>26} {store:>9.4f}")
    print(f"{'SUM':>26} {rope + resolve + store:>9.4f}")


if __name__ == "__main__":
    main()
