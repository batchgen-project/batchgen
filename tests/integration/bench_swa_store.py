from __future__ import annotations

import torch

from batchgen.kv_cache.deepseek_v4_single_kv_pool import DeepSeekV4SingleKVPool

HEAD_DIM = 512


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


def run(B):
    device = torch.device("cuda")
    pool = DeepSeekV4SingleKVPool(
        num_layers=1, num_pages=B + 8, page_size_tokens=128, device="cuda"
    )
    pool.initialize()
    kv = torch.randn(B, HEAD_DIM, dtype=torch.bfloat16, device=device)
    slots = torch.arange(B, device=device, dtype=torch.int64)

    pack_ms = _bench(lambda: pool._pack_model1_rows(kv))
    store_ms = _bench(
        lambda: pool.store_kv(layer_idx=0, token_slots=slots, kv_processed=kv)
    )
    return pack_ms, store_ms


def main():
    print(f"device={torch.cuda.get_device_name(0)}")
    print(f"\n{'B':>5} {'pack_ms':>9} {'store_ms':>9} {'scatter_ms':>11}")
    for B in (1, 8, 64, 256):
        pack, store = run(B)
        print(f"{B:>5} {pack:>9.4f} {store:>9.4f} {store - pack:>11.4f}")


if __name__ == "__main__":
    main()
