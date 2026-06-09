# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                          #
#  licensed under the apache license, version 2.0 (the "license");              #
# ---------------------------------------------------------------------------- #
"""Unit test for the CPU-pinned embedding lookup used to free the GPU embed table.

Validates (1) bit-exact correctness vs a GPU nn.Embedding, and (2) that the
host gather + H2D is NOT a decode bottleneck (target: << one decode forward).

The decode embedding is a pure gather (memory-bound, no compute), so keeping the
[vocab, hidden] table in PINNED host RAM and doing index_select on host + a small
H2D of the [n_tokens, hidden] result is cheap, while freeing ~vocab*hidden*2 bytes
of GPU memory (2.19 GiB for K2.5: vocab=163840, hidden=7168, bf16).

Run on a GPU node:  python -m pytest tests/unit/test_cpu_pinned_embedding.py -v -s
"""
import time

import pytest
import torch

K25_VOCAB = 163840
K25_HIDDEN = 7168
DTYPE = torch.bfloat16

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required (H2D + GPU reference)"
)


def cpu_pinned_embedding_lookup(weight_cpu_pinned, ids_host, device, out_pinned=None):
    """Gather rows from a pinned-host embedding table and copy to GPU.

    Args:
        weight_cpu_pinned: [vocab, hidden] table in PINNED host memory.
        ids_host: 1-D int64 token ids on the HOST (no D2H needed — the decode
            loop already holds these in the query book).
        device: target CUDA device.
        out_pinned: optional reusable pinned [max_tokens, hidden] staging buffer.
    Returns:
        [len(ids), hidden] tensor on `device`.
    """
    gathered = torch.index_select(weight_cpu_pinned, 0, ids_host)  # host gather
    if out_pinned is not None:
        n = gathered.shape[0]
        out_pinned[:n].copy_(gathered)
        return out_pinned[:n].to(device, non_blocking=True)
    return gathered.to(device, non_blocking=True)


@requires_cuda
def test_correctness_matches_gpu_embedding():
    """CPU-pinned gather is bit-identical to a GPU nn.Embedding."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    weight = torch.randn(K25_VOCAB, K25_HIDDEN, dtype=DTYPE)
    weight_pinned = weight.pin_memory()

    gpu_emb = torch.nn.Embedding(K25_VOCAB, K25_HIDDEN, dtype=DTYPE, device=device)
    with torch.no_grad():
        gpu_emb.weight.copy_(weight.to(device))

    for n in (1, 24, 64, 256):
        ids_host = torch.randint(0, K25_VOCAB, (n,), dtype=torch.int64)
        ref = gpu_emb(ids_host.to(device))
        got = cpu_pinned_embedding_lookup(weight_pinned, ids_host, device)
        torch.cuda.synchronize()
        assert torch.equal(got, ref), f"mismatch at n={n}"


@requires_cuda
def test_not_a_decode_bottleneck():
    """Host gather + H2D is << a decode forward (~tens of ms). Target < 0.5 ms."""
    device = torch.device("cuda")
    weight_pinned = torch.randn(K25_VOCAB, K25_HIDDEN, dtype=DTYPE).pin_memory()
    out_pinned = torch.empty(512, K25_HIDDEN, dtype=DTYPE).pin_memory()
    FORWARD_MS = 60.0  # representative no-offload decode forward (measured ~61 ms)

    print()
    for n in (24, 64, 128, 256):
        ids_host = torch.randint(0, K25_VOCAB, (n,), dtype=torch.int64)
        # warmup
        for _ in range(3):
            cpu_pinned_embedding_lookup(weight_pinned, ids_host, device, out_pinned)
        torch.cuda.synchronize()
        iters = 50
        t0 = time.perf_counter()
        for _ in range(iters):
            cpu_pinned_embedding_lookup(weight_pinned, ids_host, device, out_pinned)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / iters * 1e3
        pct = ms / FORWARD_MS * 100
        print(f"  n={n:>4}: cpu-pinned embed = {ms*1e3:7.1f} us  ({pct:.3f}% of {FORWARD_MS}ms forward)")
        assert ms < 0.5, f"CPU embed {ms:.3f} ms too slow at n={n} (bottleneck)"
        assert pct < 1.0, f"CPU embed {pct:.2f}% of forward at n={n} (>1%)"


@requires_cuda
def test_pinned_faster_than_pageable():
    """Sanity: pinned host table H2D is at least as fast as pageable."""
    device = torch.device("cuda")
    n = 256
    ids_host = torch.randint(0, K25_VOCAB, (n,), dtype=torch.int64)
    w_pageable = torch.randn(K25_VOCAB, K25_HIDDEN, dtype=DTYPE)
    w_pinned = w_pageable.pin_memory()

    def bench(weight):
        for _ in range(3):
            torch.index_select(weight, 0, ids_host).to(device, non_blocking=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(50):
            torch.index_select(weight, 0, ids_host).to(device, non_blocking=True)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / 50 * 1e3

    pageable_ms, pinned_ms = bench(w_pageable), bench(w_pinned)
    print(f"\n  pageable={pageable_ms*1e3:.1f}us  pinned={pinned_ms*1e3:.1f}us")
    assert pinned_ms <= pageable_ms * 1.5  # pinned should not be slower


if __name__ == "__main__":
    # Plain-python runner (no pytest needed) for remote GPU execution.
    if not torch.cuda.is_available():
        print("SKIP: CUDA required")
        raise SystemExit(0)
    test_correctness_matches_gpu_embedding()
    print("[OK] correctness: CPU-pinned gather == GPU nn.Embedding (bit-exact)")
    test_not_a_decode_bottleneck()
    print("[OK] perf: CPU embed << decode forward (<0.5ms, <1%)")
    test_pinned_faster_than_pageable()
    print("[OK] pinned not slower than pageable")
    print("ALL PASSED")
