from __future__ import annotations

import torch

from batchgen.attention.dsa.v4_mla_sm120_triton import (
    _run_triton_sparse_decode,
    flash_mla_sparse_decode_sm120,
)

_PAGE_SIZE = 64
_PAGE_BYTES = _PAGE_SIZE * 576 + _PAGE_SIZE * 8
_HEAD_DIM = 512


def _make_cache(num_pages):
    return torch.randint(
        0, 255, (num_pages, _PAGE_BYTES), dtype=torch.uint8, device="cuda"
    ).view(num_pages, _PAGE_BYTES)


def _bench(fn, warmup=20, iters=100):
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


def run(B, H, topk, num_pages):
    torch.manual_seed(0)
    q = torch.randn(B, 1, H, _HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    k_cache = _make_cache(num_pages)
    max_idx = num_pages * _PAGE_SIZE
    indices = torch.randint(
        0, max_idx, (B, topk), dtype=torch.int32, device="cuda"
    )
    topk_len = torch.full((B,), topk, dtype=torch.int32, device="cuda")
    softmax_scale = _HEAD_DIM**-0.5

    def fn():
        _run_triton_sparse_decode(q, k_cache, indices, topk_len, softmax_scale)

    return _bench(fn)


def main():
    cap = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    print(f"device={name} sm{cap[0]}{cap[1]} SMs={props.multi_processor_count}")
    print(
        f"\n{'B':>4} {'H':>4} {'topk':>6} {'grid':>8} {'ms':>9} "
        f"{'occ_programs':>13}"
    )
    H = 64
    for B in (1, 4, 16, 64):
        for topk in (512, 2048):
            ms = run(B, H, topk, num_pages=512)
            grid = B * H
            print(f"{B:>4} {H:>4} {topk:>6} {grid:>8} {ms:>9.4f} {grid:>13}")


if __name__ == "__main__":
    main()
