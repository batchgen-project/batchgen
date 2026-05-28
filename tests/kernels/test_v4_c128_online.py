"""Tests for c128_online streaming HCA compress kernel.

Compares one-token-at-a-time CUDA online softmax against batch
softmax+weighted-sum reference (the compress step in v4_fused_compress_quant).
"""

from __future__ import annotations

import pytest
import torch

CUDA_AVAILABLE = torch.cuda.is_available()


def _ref_compress_chunk(kv: torch.Tensor, score: torch.Tensor) -> torch.Tensor:
    """Batch softmax + weighted sum — equivalent to the compress step in
    ``v4_fused_compress_quant`` at compress_ratio=128, without norm/rope/quant.

    Args:
        kv:    [C, D] float32
        score: [C, D] float32
    Returns:
        [D] float32  — compressed kv
    """
    weights = torch.softmax(score, dim=0)
    return (kv * weights).sum(dim=0)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA required")
@pytest.mark.parametrize("T_in", [128, 1024])
@pytest.mark.parametrize("head_dim", [128, 512])
def test_c128_online_vs_reference(T_in: int, head_dim: int):
    from batchgen_kernels.attention.c128_online import c128_online_compress

    COMPRESS_RATIO = 128
    num_chunks = T_in // COMPRESS_RATIO
    assert T_in % COMPRESS_RATIO == 0

    torch.manual_seed(42)
    device = "cuda"

    kv_all = torch.randn(T_in, head_dim, device=device, dtype=torch.float32)
    score_all = torch.randn(T_in, head_dim, device=device, dtype=torch.float32)

    ref_outputs: list[torch.Tensor] = []
    for c in range(num_chunks):
        s, e = c * COMPRESS_RATIO, (c + 1) * COMPRESS_RATIO
        ref_outputs.append(_ref_compress_chunk(kv_all[s:e], score_all[s:e]))

    num_slots = 1
    buffer = torch.zeros(
        num_slots, head_dim * 3, device=device, dtype=torch.float32
    )
    indices = torch.zeros(1, dtype=torch.int32, device=device)

    cuda_outputs: list[torch.Tensor] = []
    for t in range(T_in):
        inp = torch.cat([kv_all[t : t + 1], score_all[t : t + 1]], dim=1)
        out = c128_online_compress(buffer, inp, indices)
        if (t + 1) % COMPRESS_RATIO == 0:
            cuda_outputs.append(out.squeeze(0).clone())
            buffer.zero_()

    assert len(cuda_outputs) == len(ref_outputs)
    for i, (ref, cuda) in enumerate(zip(ref_outputs, cuda_outputs)):
        torch.testing.assert_close(
            cuda,
            ref,
            atol=0.05,
            rtol=1e-3,
            msg=f"Chunk {i} mismatch (T_in={T_in}, D={head_dim})",
        )


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA required")
def test_c128_online_multi_batch():
    """Multiple independent sequences compressed in parallel."""
    from batchgen_kernels.attention.c128_online import c128_online_compress

    head_dim = 128
    batch_size = 4
    COMPRESS_RATIO = 128
    device = "cuda"
    torch.manual_seed(123)

    kv_all = torch.randn(
        COMPRESS_RATIO, batch_size, head_dim, device=device, dtype=torch.float32
    )
    score_all = torch.randn(
        COMPRESS_RATIO, batch_size, head_dim, device=device, dtype=torch.float32
    )

    ref = torch.stack(
        [
            _ref_compress_chunk(kv_all[:, b, :], score_all[:, b, :])
            for b in range(batch_size)
        ]
    )

    buffer = torch.zeros(
        batch_size, head_dim * 3, device=device, dtype=torch.float32
    )
    indices = torch.arange(batch_size, dtype=torch.int32, device=device)

    for t in range(COMPRESS_RATIO):
        inp = torch.cat([kv_all[t], score_all[t]], dim=1)
        out = c128_online_compress(buffer, inp, indices)

    torch.testing.assert_close(out, ref, atol=0.05, rtol=1e-3)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA required")
def test_c128_online_empty_input():
    from batchgen_kernels.attention.c128_online import c128_online_compress

    buffer = torch.zeros(1, 128 * 3, device="cuda", dtype=torch.float32)
    inp = torch.empty(0, 128 * 2, device="cuda", dtype=torch.float32)
    indices = torch.empty(0, dtype=torch.int32, device="cuda")
    out = c128_online_compress(buffer, inp, indices)
    assert out.shape[0] == 0
