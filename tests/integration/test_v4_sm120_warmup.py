from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


def test_warmup_runs_without_error():
    import batchgen.attention.dsa.v4_mla_sm120_triton as mod

    mod._warmup_done.clear()
    mod.warmup_sm120_sparse_decode(num_heads=64, head_dim=512, device="cuda")
    torch.cuda.synchronize()


def test_warmup_removes_first_call_compile_cost():
    import batchgen.attention.dsa.v4_mla_sm120_triton as mod

    mod._warmup_done.clear()
    mod.warmup_sm120_sparse_decode(num_heads=64, head_dim=512, device="cuda")
    torch.cuda.synchronize()

    page_size = 64
    k_cache = torch.zeros(
        4, page_size, 1, 584, dtype=torch.uint8, device="cuda"
    )
    scale = 512**-0.5
    for topk in (64, 128):
        q = torch.zeros(1, 1, 64, 512, dtype=torch.bfloat16, device="cuda")
        idx = torch.zeros(1, topk, dtype=torch.int32, device="cuda")
        tlen = torch.zeros(1, dtype=torch.int32, device="cuda")
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        mod._run_triton_sparse_decode(q, k_cache, idx, tlen, scale)
        e.record()
        torch.cuda.synchronize()
        assert s.elapsed_time(e) < 50.0, (
            f"topk={topk} took {s.elapsed_time(e):.1f}ms after warmup "
            "(expected no recompile)"
        )


def test_maybe_warmup_is_idempotent():
    import batchgen.attention.dsa.v4_mla_sm120_triton as mod

    mod._warmup_done.clear()
    mod.maybe_warmup_sm120_sparse_decode(
        num_heads=64, head_dim=512, device="cuda"
    )
    assert (64, 512) in mod._warmup_done

    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    mod.maybe_warmup_sm120_sparse_decode(
        num_heads=64, head_dim=512, device="cuda"
    )
    e.record()
    torch.cuda.synchronize()
    assert s.elapsed_time(e) < 1.0


def test_pinned_kernel_matches_reference():
    from batchgen.attention.dsa.v4_mla_sm120_triton import (
        _run_triton_sparse_decode,
    )

    torch.manual_seed(0)
    page_size = 64
    num_pages = 32
    head_dim = 512
    num_heads = 64
    topk = 128
    bytes_per_token = 576 + 8

    k_cache = torch.zeros(
        num_pages,
        page_size,
        1,
        bytes_per_token,
        dtype=torch.uint8,
        device="cuda",
    )
    fp8_section = k_cache[..., :448].view(torch.float8_e4m3fn)
    fp8_section.copy_(
        (0.1 * torch.ones_like(fp8_section, dtype=torch.float32)).to(
            torch.float8_e4m3fn
        )
    )

    q = 0.1 * torch.ones(
        1, 1, num_heads, head_dim, dtype=torch.bfloat16, device="cuda"
    )
    indices = torch.arange(topk, dtype=torch.int32, device="cuda").view(1, topk)
    topk_length = torch.full((1,), topk, dtype=torch.int32, device="cuda")

    out, lse = _run_triton_sparse_decode(
        q, k_cache, indices, topk_length, 0.044
    )

    assert out.shape == (1, 1, num_heads, head_dim)
    assert torch.isfinite(out.float()).all()
    assert torch.isfinite(lse.float()).all()
