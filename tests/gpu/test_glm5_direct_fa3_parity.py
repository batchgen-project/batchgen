"""Numerical parity: direct-index FA3 sparse decode vs the gather+FlashMLA path.

Reference = today's production pipeline (fused selected-KV gather into a
padded BF16 buffer, then FlashMLA over the synthetic paged view). Candidate =
the transform kernel + FlashAttention-3 over a page-size-1 view of the same
resident paged KV (the whole-model graph's new selected branch), and, for the
all-short case, FA3 directly over the resident page-size-64 cache (the
all_short branch). Rows cover short, crossing-boundary and long contexts
around index_topk, in one mixed batch.
"""

import math

import pytest
import torch

pytest.importorskip("triton")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

H = 64            # GLM-5 attention heads
KV_LORA = 512
ROPE = 64
KV_DIM = KV_LORA + ROPE
PAGE = 64
TOPK = 256        # scaled-down index_topk; the kernels take it as a parameter
SCALE = 1.0 / math.sqrt(576)


def _metrics(candidate: torch.Tensor, reference: torch.Tensor):
    c = candidate.float().reshape(-1)
    r = reference.float().reshape(-1)
    calc_diff = ((c - r) ** 2).sum().item() / max((r ** 2).sum().item(), 1e-30)
    cosine = torch.nn.functional.cosine_similarity(c, r, dim=0).item()
    max_abs = (c - r).abs().max().item()
    return calc_diff, cosine, max_abs


def _make_case(seed=0):
    torch.manual_seed(seed)
    device = "cuda"
    # Short, at-boundary, crossing and long rows (relative to TOPK).
    cache_seqlens = torch.tensor(
        [37, TOPK - 1, TOPK, TOPK + 1, TOPK + 300, 2 * TOPK + 5, 129, TOPK + 77],
        dtype=torch.int32,
        device=device,
    )
    batch = cache_seqlens.shape[0]
    slots_total = batch + 4
    max_pages = (int(cache_seqlens.max()) + PAGE - 1) // PAGE + 1
    num_pages = slots_total * max_pages
    blocked_k = (torch.randn(num_pages, PAGE, 1, KV_DIM, device=device, dtype=torch.bfloat16) * 0.3)
    page_table = torch.randperm(num_pages, dtype=torch.int32, device=device)[
        : slots_total * max_pages
    ].reshape(slots_total, max_pages)
    slots = torch.randperm(slots_total, dtype=torch.int32, device=device)[:batch]
    topk_indices = torch.full((batch, TOPK), -1, dtype=torch.int32, device=device)
    for row in range(batch):
        n = int(cache_seqlens[row])
        if n > TOPK:
            sel = torch.randperm(n, device=device)[:TOPK].to(torch.int32)
            topk_indices[row] = sel
    absorbed_q = torch.randn(batch, H, KV_LORA, device=device, dtype=torch.bfloat16) * 0.2
    q_rope = torch.randn(batch, H, ROPE, device=device, dtype=torch.bfloat16) * 0.2
    return blocked_k, page_table, cache_seqlens, topk_indices, slots, absorbed_q, q_rope


def _reference_gather_flashmla(blocked_k, page_table, cache_seqlens, topk_indices, slots,
                               absorbed_q, q_rope):
    from batchgen.attention.dsa.sparse_decode_mla import (
        prepare_sparse_flash_mla_decode_inputs,
        run_prepared_sparse_flash_mla_decode,
    )
    from batchgen.attention.dsa.unified_selector import select_mla_kv_for_flashmla_bf16

    selected_kv, selected_lengths, _, _ = select_mla_kv_for_flashmla_bf16(
        blocked_k,
        page_table,
        cache_seqlens,
        topk_indices.to(torch.long),
        index_topk=TOPK,
        page_size=PAGE,
        return_indices=False,
        primary_slot_indices=slots,
    )
    batch = cache_seqlens.shape[0]
    query_states = torch.cat([absorbed_q, q_rope], dim=-1).view(batch, 1, H, KV_DIM)
    prepared = prepare_sparse_flash_mla_decode_inputs(
        query_states,
        selected_kv,
        selected_lengths,
        H,
        SCALE,
        head_dim_v=KV_LORA,
        page_size=PAGE,
    )
    out = run_prepared_sparse_flash_mla_decode(prepared)
    return out.reshape(batch, H, KV_LORA)


def _candidate_transform_fa3(blocked_k, page_table, cache_seqlens, topk_indices, slots,
                             absorbed_q, q_rope):
    from flash_attn_interface import flash_attn_with_kvcache
    from batchgen_kernels.attention.dsa.selected_page_table import (
        transform_selected_positions_out,
    )

    batch = cache_seqlens.shape[0]
    ids = torch.empty(batch, TOPK, dtype=torch.int32, device=blocked_k.device)
    lengths = torch.empty(batch, dtype=torch.int32, device=blocked_k.device)
    transform_selected_positions_out(
        page_table,
        cache_seqlens,
        topk_indices,
        ids,
        lengths,
        page_size=PAGE,
        primary_slot_indices=slots,
    )
    flat = blocked_k.view(-1, 1, 1, KV_DIM)
    out = flash_attn_with_kvcache(
        q=q_rope.unsqueeze(1),
        k_cache=flat[..., KV_LORA:],
        v_cache=flat[..., :KV_LORA],
        qv=absorbed_q.unsqueeze(1),
        page_table=ids,
        cache_seqlens=lengths,
        softmax_scale=SCALE,
        causal=True,
        num_splits=0,
        return_softmax_lse=False,
    )
    return out.reshape(batch, H, KV_LORA)


def test_selected_branch_matches_gather_flashmla():
    case = _make_case()
    reference = _reference_gather_flashmla(*case)
    candidate = _candidate_transform_fa3(*case)
    torch.cuda.synchronize()
    calc_diff, cosine, max_abs = _metrics(candidate, reference)
    print(f"selected-branch parity: calc_diff={calc_diff:.3e} cosine={cosine:.8f} max_abs={max_abs:.3e}")
    assert torch.isfinite(candidate.float()).all()
    assert calc_diff <= 1e-5, calc_diff
    assert cosine >= 0.99999, cosine
    assert max_abs <= 5e-4, max_abs


def test_all_short_branch_matches_gather_flashmla():
    blocked_k, page_table, cache_seqlens, topk_indices, slots, absorbed_q, q_rope = _make_case(1)
    from flash_attn_interface import flash_attn_with_kvcache

    short = cache_seqlens <= TOPK
    idx = short.nonzero().squeeze(1)
    cache_seqlens = cache_seqlens[idx].contiguous()
    slots = slots[idx].contiguous()
    topk_indices = topk_indices[idx].contiguous()
    absorbed_q = absorbed_q[idx].contiguous()
    q_rope = q_rope[idx].contiguous()
    assert cache_seqlens.numel() >= 3

    reference = _reference_gather_flashmla(
        blocked_k, page_table, cache_seqlens, topk_indices, slots, absorbed_q, q_rope
    )
    out = flash_attn_with_kvcache(
        q=q_rope.unsqueeze(1),
        k_cache=blocked_k[..., KV_LORA:],
        v_cache=blocked_k[..., :KV_LORA],
        qv=absorbed_q.unsqueeze(1),
        page_table=page_table,
        cache_batch_idx=slots,
        cache_seqlens=cache_seqlens,
        softmax_scale=SCALE,
        causal=True,
        num_splits=0,
        return_softmax_lse=False,
    )
    torch.cuda.synchronize()
    candidate = out.reshape(cache_seqlens.shape[0], H, KV_LORA)
    calc_diff, cosine, max_abs = _metrics(candidate, reference)
    print(f"all-short parity: calc_diff={calc_diff:.3e} cosine={cosine:.8f} max_abs={max_abs:.3e}")
    assert torch.isfinite(candidate.float()).all()
    assert calc_diff <= 1e-5, calc_diff
    assert cosine >= 0.99999, cosine
    assert max_abs <= 5e-4, max_abs


def test_zero_length_row_stays_finite_under_fa3():
    """A padded/unassigned row (selected length 0) must come out finite."""
    from flash_attn_interface import flash_attn_with_kvcache

    device = "cuda"
    blocked_k = torch.randn(8, PAGE, 1, KV_DIM, device=device, dtype=torch.bfloat16)
    flat = blocked_k.view(-1, 1, 1, KV_DIM)
    ids = torch.full((2, TOPK), -1, dtype=torch.int32, device=device)
    ids[0, :5] = torch.arange(5, dtype=torch.int32, device=device)
    lengths = torch.tensor([5, 0], dtype=torch.int32, device=device)
    q_rope = torch.randn(2, H, ROPE, device=device, dtype=torch.bfloat16)
    absorbed_q = torch.randn(2, H, KV_LORA, device=device, dtype=torch.bfloat16)
    out = flash_attn_with_kvcache(
        q=q_rope.unsqueeze(1),
        k_cache=flat[..., KV_LORA:],
        v_cache=flat[..., :KV_LORA],
        qv=absorbed_q.unsqueeze(1),
        page_table=ids,
        cache_seqlens=lengths,
        softmax_scale=SCALE,
        causal=True,
        num_splits=0,
        return_softmax_lse=False,
    )
    torch.cuda.synchronize()
    assert torch.isfinite(out.float()).all()
