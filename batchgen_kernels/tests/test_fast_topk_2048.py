"""Regression tests for fast_topk_2048 (batchgen_kernels DSA top-k selector).

Primary regression: the eager DSA decode path calls ``fast_topk_2048(score, lengths)``
with **no** ``num_valid_tokens`` and a batch of B>1 rows. Before 0.3.3.post1 the kernel
validated the (ignored) ``num_valid_tokens`` slot unconditionally, so the wrapper's [B]
placeholder tripped ``num_valid_tokens must contain one element`` for B>1. This is
reachable in production whenever decode bsz > cuda-graph-max-bucket-size and rows cross
2048 context.

Runs on GPU (H20 conda ``batchgen`` env). The CUDA module JIT-compiles on first use.
"""

import pytest
import torch

from batchgen_kernels.attention.dsa.fast_topk_cuda import (
    fast_topk_2048,
    fast_topk_2048_out,
)

_TOPK = 2048

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="fast_topk_2048 requires CUDA"
)


def _check_row(score_row: torch.Tensor, length: int, row_indices: list[int]) -> None:
    """Assert one output row matches the kernel's documented semantics."""
    if length <= _TOPK:
        # dense_prefix_topk: [0, 1, ..., length-1, -1, -1, ...]
        assert row_indices == list(range(length)) + [-1] * (_TOPK - length)
        return
    # length > TOPK: an exact top-2048 selection over scores[:length].
    valid = [i for i in row_indices if i >= 0]
    assert len(valid) == _TOPK
    assert min(valid) >= 0 and max(valid) < length
    sel = torch.tensor(valid)
    scores = score_row[:length].cpu()
    mask = torch.ones(length, dtype=torch.bool)
    mask[sel] = False
    # Every selected score must be >= every non-selected score (top-k definition).
    assert scores[sel].min().item() >= scores[mask].max().item() - 1e-5


@requires_cuda
@pytest.mark.parametrize(
    "lengths",
    [
        [100, 3000, 2048, 4000],  # mixed short/long, B=4 — the crash repro (B>1 + long rows)
        [4096, 4096],             # all long, B=2
        [500, 700, 900],          # all short, B=3
        [4000],                   # B=1 long (previously the only passing long case)
    ],
)
def test_eager_no_num_valid_tokens(lengths):
    torch.manual_seed(0)
    device = "cuda"
    B = len(lengths)
    N = max(lengths)
    score = torch.randn(B, N, dtype=torch.float32, device=device)
    lens = torch.tensor(lengths, dtype=torch.int32, device=device)

    # Regression: must NOT raise for B>1 without num_valid_tokens.
    out = fast_topk_2048(score, lens)
    assert out.shape == (B, _TOPK)
    assert out.dtype == torch.int32

    out_cpu = out.cpu().tolist()
    for b, length in enumerate(lengths):
        _check_row(score[b], length, out_cpu[b])


@requires_cuda
def test_num_valid_tokens_padded_batch():
    """Graph-style padded batch: rows >= num_valid_tokens are emitted as all -1."""
    torch.manual_seed(1)
    device = "cuda"
    B, N = 4, 4096
    lengths = [3000, 3000, 3000, 3000]
    score = torch.randn(B, N, dtype=torch.float32, device=device)
    lens = torch.tensor(lengths, dtype=torch.int32, device=device)
    num_valid = torch.tensor([2], dtype=torch.int32, device=device)  # only first 2 rows live
    indices = torch.empty(B, _TOPK, dtype=torch.int32, device=device)

    fast_topk_2048_out(score, lens, indices, num_valid_tokens=num_valid)
    out = indices.cpu()
    assert (out[2] == -1).all() and (out[3] == -1).all()       # padding rows
    assert (out[0] >= 0).sum().item() == _TOPK                 # live rows: full top-2048
    assert (out[1] >= 0).sum().item() == _TOPK


@requires_cuda
def test_malformed_num_valid_tokens_raises():
    """A genuinely-supplied num_valid_tokens with numel != 1 must still be rejected."""
    device = "cuda"
    B, N = 2, 4096
    score = torch.randn(B, N, dtype=torch.float32, device=device)
    lens = torch.tensor([3000, 3000], dtype=torch.int32, device=device)
    indices = torch.empty(B, _TOPK, dtype=torch.int32, device=device)
    bad = torch.tensor([1, 1], dtype=torch.int32, device=device)  # numel 2
    with pytest.raises((ValueError, RuntimeError), match="one element"):
        fast_topk_2048_out(score, lens, indices, num_valid_tokens=bad)
