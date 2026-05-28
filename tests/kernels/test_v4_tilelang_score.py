# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                          #
# ---------------------------------------------------------------------------- #

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

try:
    import tilelang  # noqa: F401
except ImportError:
    pytest.skip("tilelang not installed", allow_module_level=True)

from batchgen_kernels.attention.dsa.tilelang_score import (
    FP8_,
    tilelang_fp8_paged_mqa_logits,
)


def _pytorch_reference(
    q_fp8: torch.Tensor,
    kvcache_fp8: torch.Tensor,
    weight: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    max_seq_len: int,
    block_size: int = 64,
    head_dim: int = 128,
) -> torch.Tensor:
    batch_size = q_fp8.shape[0]
    num_heads = q_fp8.shape[2]

    q = q_fp8.view(batch_size, num_heads, head_dim).float()
    flat = kvcache_fp8.view(-1, block_size * (head_dim + 4))
    k_raw = flat[..., : block_size * head_dim].contiguous().view(dtype=FP8_)
    k_all = k_raw.view(-1, block_size, head_dim).float()
    k_scale = (
        flat[..., block_size * head_dim :]
        .contiguous()
        .view(dtype=torch.float32)
    )

    logits = torch.zeros(batch_size, max_seq_len, device=q.device)

    for b in range(batch_size):
        sl = seq_lens[b].item()
        n_pages = (sl + block_size - 1) // block_size
        for p_idx in range(n_pages):
            page_id = page_table[b, p_idx].item()
            k_block = k_all[page_id]
            ks = k_scale[page_id]

            scores_bh = k_block @ q[b].T
            scores_bh = torch.relu(scores_bh) * weight[b].unsqueeze(0)
            scores_sum = scores_bh.sum(dim=1) * ks.squeeze(-1)
            start = p_idx * block_size
            end = min(start + block_size, max_seq_len)
            logits[b, start:end] = scores_sum[: end - start]

    return logits


def _make_inputs(
    batch_size: int = 4,
    num_heads: int = 32,
    head_dim: int = 128,
    max_seq_len: int = 512,
    block_size: int = 64,
):
    device = "cuda"
    torch.manual_seed(42)

    max_pages = max_seq_len // block_size
    num_blocks = batch_size * max_pages + 4

    q_f32 = torch.randn(batch_size, 1, num_heads, head_dim, device=device)
    q_fp8 = q_f32.to(FP8_)

    raw_bytes = head_dim + 4
    kvcache_fp8_flat = torch.zeros(
        num_blocks, block_size, 1, raw_bytes, device=device, dtype=torch.uint8
    )
    for blk in range(num_blocks):
        k_data = torch.randn(block_size, head_dim, device=device)
        k_fp8 = k_data.to(FP8_)
        k_bytes = k_fp8.view(torch.uint8)
        kvcache_fp8_flat[blk, :, 0, :head_dim] = k_bytes

        scale = torch.rand(block_size, 1, device=device) * 0.1 + 0.01
        scale_bytes = scale.view(torch.uint8)
        kvcache_fp8_flat[blk, :, 0, head_dim : head_dim + 4] = scale_bytes

    weight = (
        torch.randn(
            batch_size, num_heads, device=device, dtype=torch.float32
        ).abs()
        * 0.1
    )

    seq_lens = torch.full(
        (batch_size,), max_seq_len, device=device, dtype=torch.int32
    )

    page_table = torch.zeros(
        batch_size, max_pages, device=device, dtype=torch.int32
    )
    for b in range(batch_size):
        page_table[b] = torch.arange(max_pages, device=device) + b * max_pages

    return q_fp8, kvcache_fp8_flat, weight, seq_lens, page_table, max_seq_len


def test_tilelang_vs_pytorch_reference():
    B, H, D, S = 4, 32, 128, 512
    q_fp8, kvcache_fp8, weight, seq_lens, page_table, max_seq_len = (
        _make_inputs(batch_size=B, num_heads=H, head_dim=D, max_seq_len=S)
    )

    actual = tilelang_fp8_paged_mqa_logits(
        q_fp8=q_fp8,
        kvcache_fp8=kvcache_fp8,
        weight=weight,
        seq_lens=seq_lens,
        page_table=page_table,
        max_seq_len=max_seq_len,
        clean_logits=False,
    )

    expected = _pytorch_reference(
        q_fp8=q_fp8,
        kvcache_fp8=kvcache_fp8,
        weight=weight,
        seq_lens=seq_lens,
        page_table=page_table,
        max_seq_len=max_seq_len,
    )

    assert actual.shape == (B, S)
    assert expected.shape == (B, S)
    torch.testing.assert_close(actual, expected, atol=0.05, rtol=0.05)
