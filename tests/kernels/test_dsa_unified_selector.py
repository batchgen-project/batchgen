"""Self-contained BF16 DSA selected-KV contract tests.

These tests validate the immediate GLM-5 DSA contract before introducing a
CUDA-graph-owned selected-KV buffer:

    selected_mla_kv [B, 2048, 1, 576]
      -> selected_mla_kv.view(B * 32, 64, 1, 576)
      -> FlashMLA dense decode with block_table [B, 32]
      -> selected_lengths [B] controls the valid prefix per row.
"""

from __future__ import annotations

import math

import pytest
import torch

from batchgen.attention.dsa.sparse_decode_mla import (
    prepare_sparse_flash_mla_decode_inputs,
    run_prepared_sparse_flash_mla_decode,
)
from batchgen.attention.dsa.sparse_gather import sparse_gather_from_paged_kv
from batchgen.attention.dsa.unified_selector import (
    make_flashmla_selected_block_table,
    select_mla_kv_for_flashmla_bf16,
    view_selected_mla_kv_as_flashmla_pages,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="DSA FlashMLA contract tests require CUDA",
)


INDEX_TOPK = 2048
PAGE_SIZE = 64
PAGES_PER_SELECTED_ROW = INDEX_TOPK // PAGE_SIZE
H_Q = 64
H_KV = 1
D_QK = 576
D_V = 512
FLASHMLA_ATOL = 8e-4
FLASHMLA_RTOL = 2.01 / 128


def _require_flash_mla():
    return pytest.importorskip("flash_mla")


def _make_query(batch_size: int, *, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    q = torch.randn(
        batch_size,
        1,
        H_Q,
        D_QK,
        device="cuda",
        dtype=torch.bfloat16,
    )
    return (q / 10).clamp(-1.0, 1.0)


def _make_selected_mla_kv(
    selected_lengths: torch.Tensor,
    *,
    seed: int,
) -> torch.Tensor:
    torch.manual_seed(seed)
    batch_size = int(selected_lengths.numel())
    selected = torch.randn(
        batch_size,
        INDEX_TOPK,
        H_KV,
        D_QK,
        device="cuda",
        dtype=torch.bfloat16,
    )
    selected = (selected / 10).clamp(-1.0, 1.0)
    for row in range(batch_size):
        valid = int(selected_lengths[row].item())
        if valid < INDEX_TOPK:
            selected[row, valid:] = 0
    return selected


def _selected_attention_reference(
    query_states: torch.Tensor,
    selected_mla_kv: torch.Tensor,
    selected_lengths: torch.Tensor,
    *,
    softmax_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch reference for FlashMLA dense decode over selected KV prefixes."""

    batch_size = query_states.shape[0]
    out = torch.empty(
        batch_size,
        1,
        H_Q,
        D_V,
        device=query_states.device,
        dtype=torch.float32,
    )
    lse = torch.empty(
        batch_size,
        H_Q,
        1,
        device=query_states.device,
        dtype=torch.float32,
    )

    for row in range(batch_size):
        valid = int(selected_lengths[row].item())
        assert 0 < valid <= INDEX_TOPK
        q = query_states[row, 0].float()  # [H_Q, D_QK]
        kv = selected_mla_kv[row, :valid, 0].float()  # [valid, D_QK]
        scores = torch.matmul(q, kv.transpose(0, 1)) * softmax_scale
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32)
        out[row, 0] = torch.matmul(probs, kv[:, :D_V])
        lse[row, :, 0] = torch.logsumexp(scores, dim=-1)

    return out.to(query_states.dtype), lse


def _run_flashmla_dense(
    query_states: torch.Tensor,
    selected_mla_kv: torch.Tensor,
    selected_lengths: torch.Tensor,
    *,
    softmax_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    flash_mla = _require_flash_mla()
    batch_size = query_states.shape[0]
    blocked_k = view_selected_mla_kv_as_flashmla_pages(
        selected_mla_kv,
        page_size=PAGE_SIZE,
    )
    block_table = make_flashmla_selected_block_table(
        batch_size,
        index_topk=INDEX_TOPK,
        page_size=PAGE_SIZE,
        device=query_states.device,
    )
    # BatchGen's production wrapper still targets the older FlashMLA metadata
    # signature. Newer FlashMLA accepts extra args via *args, so this remains
    # compatible with both versions.
    tile_scheduler_metadata, num_splits = flash_mla.get_mla_metadata(
        selected_lengths.to(device=query_states.device, dtype=torch.int32),
        H_Q,
        H_KV,
    )

    return flash_mla.flash_mla_with_kvcache(
        query_states,
        blocked_k,
        block_table,
        selected_lengths.to(device=query_states.device, dtype=torch.int32),
        D_V,
        tile_scheduler_metadata,
        num_splits,
        softmax_scale,
        False,
    )


def _assert_flashmla_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(
        actual,
        expected,
        atol=FLASHMLA_ATOL,
        rtol=FLASHMLA_RTOL,
    )


def _reference_select_mla_kv_for_flashmla_bf16(
    primary_blocked_k: torch.Tensor,
    primary_page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    long_topk_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Standalone PyTorch reference for the unified selector contract."""

    batch_size, index_topk = long_topk_indices.shape
    max_pages = primary_page_table.shape[1]
    num_k_heads = primary_blocked_k.shape[2]
    kv_dim = primary_blocked_k.shape[3]
    seqlens = cache_seqlens.to(device=primary_blocked_k.device, dtype=torch.long)
    dense_positions = torch.arange(
        index_topk,
        device=primary_blocked_k.device,
        dtype=torch.long,
    ).expand(batch_size, index_topk)
    is_long = seqlens > index_topk
    logical_indices = torch.where(
        is_long.unsqueeze(1),
        long_topk_indices.to(dtype=torch.long),
        dense_positions,
    )
    valid = (logical_indices >= 0) & (logical_indices < seqlens.unsqueeze(1))
    safe_logical = logical_indices.clamp_min(0)
    logical_page = torch.div(safe_logical, PAGE_SIZE, rounding_mode="floor")
    page_offset = safe_logical - logical_page * PAGE_SIZE
    page_in_table = logical_page < max_pages
    physical_page = torch.gather(
        primary_page_table.to(dtype=torch.long),
        1,
        logical_page.clamp(max=max_pages - 1),
    )
    valid = valid & page_in_table & (physical_page >= 0)
    flat_index = physical_page.clamp_min(0) * PAGE_SIZE + page_offset
    gathered = primary_blocked_k.reshape(-1, num_k_heads * kv_dim)[
        flat_index.reshape(-1)
    ].view(batch_size, index_topk, num_k_heads, kv_dim)
    selected = torch.where(
        valid.view(batch_size, index_topk, 1, 1),
        gathered,
        torch.zeros((), dtype=primary_blocked_k.dtype, device=primary_blocked_k.device),
    )
    selected_indices = torch.where(
        valid,
        logical_indices,
        torch.full_like(logical_indices, -1),
    ).to(dtype=long_topk_indices.dtype)
    selected_lengths = torch.minimum(
        cache_seqlens.to(dtype=torch.int32),
        torch.full((batch_size,), index_topk, dtype=torch.int32, device=cache_seqlens.device),
    )
    row_modes = is_long.to(dtype=torch.int32)
    return selected, selected_lengths, selected_indices, row_modes


def test_flashmla_dense_selected_kv_contract_mixed_lengths():
    torch.cuda.set_device(0)
    selected_lengths = torch.tensor(
        [37, INDEX_TOPK, 513],
        device="cuda",
        dtype=torch.int32,
    )
    query_states = _make_query(selected_lengths.numel(), seed=7)
    selected_mla_kv = _make_selected_mla_kv(selected_lengths, seed=11)
    softmax_scale = D_QK**-0.5

    attn_out, lse = _run_flashmla_dense(
        query_states,
        selected_mla_kv,
        selected_lengths,
        softmax_scale=softmax_scale,
    )
    ref_out, ref_lse = _selected_attention_reference(
        query_states,
        selected_mla_kv,
        selected_lengths,
        softmax_scale=softmax_scale,
    )

    assert attn_out.shape == (selected_lengths.numel(), 1, H_Q, D_V)
    assert lse.shape == (selected_lengths.numel(), H_Q, 1)
    _assert_flashmla_close(attn_out, ref_out)
    torch.testing.assert_close(lse, ref_lse, atol=1e-5, rtol=8.01 / 65536)


def test_prepared_sparse_flashmla_inputs_match_dense_contract():
    torch.cuda.set_device(0)
    selected_lengths = torch.tensor(
        [41, INDEX_TOPK],
        device="cuda",
        dtype=torch.int32,
    )
    query_states = _make_query(selected_lengths.numel(), seed=101)
    selected_mla_kv = _make_selected_mla_kv(selected_lengths, seed=103)
    softmax_scale = D_QK**-0.5

    prepared = prepare_sparse_flash_mla_decode_inputs(
        query_states,
        selected_mla_kv,
        selected_lengths,
        H_Q,
        softmax_scale,
        head_dim_v=D_V,
        page_size=PAGE_SIZE,
    )
    assert prepared.blocked_k.shape == (
        selected_lengths.numel() * PAGES_PER_SELECTED_ROW,
        PAGE_SIZE,
        H_KV,
        D_QK,
    )
    assert prepared.block_table.shape == (
        selected_lengths.numel(),
        PAGES_PER_SELECTED_ROW,
    )
    assert prepared.cache_seqlens.dtype == torch.int32

    attn_out = run_prepared_sparse_flash_mla_decode(prepared)
    ref_out, _ = _selected_attention_reference(
        query_states,
        selected_mla_kv,
        selected_lengths,
        softmax_scale=softmax_scale,
    )
    _assert_flashmla_close(attn_out, ref_out)


def test_flashmla_dense_selected_lengths_ignore_short_row_tail():
    torch.cuda.set_device(0)
    selected_lengths = torch.tensor(
        [41, INDEX_TOPK],
        device="cuda",
        dtype=torch.int32,
    )
    query_states = _make_query(selected_lengths.numel(), seed=13)
    selected_mla_kv = _make_selected_mla_kv(selected_lengths, seed=17)
    softmax_scale = D_QK**-0.5

    baseline_out, _ = _run_flashmla_dense(
        query_states,
        selected_mla_kv,
        selected_lengths,
        softmax_scale=softmax_scale,
    )

    tail_poisoned = selected_mla_kv.clone()
    tail_poisoned[0, int(selected_lengths[0].item()) :] = 8.0
    poisoned_out, _ = _run_flashmla_dense(
        query_states,
        tail_poisoned,
        selected_lengths,
        softmax_scale=softmax_scale,
    )

    _assert_flashmla_close(poisoned_out, baseline_out)

    wrong_lengths = torch.full_like(selected_lengths, INDEX_TOPK)
    wrong_out, _ = _run_flashmla_dense(
        query_states,
        tail_poisoned,
        wrong_lengths,
        softmax_scale=softmax_scale,
    )
    max_diff = (wrong_out[0].float() - baseline_out[0].float()).abs().max()
    assert max_diff > 1.0, (
        "control check failed: poisoning short-row tail should materially change "
        "the output if FlashMLA is asked to attend all 2048 selected slots"
    )


def _make_paged_primary_cache(
    *,
    batch_size: int,
    max_tokens: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create logical KV plus a non-identity paged physical cache."""

    torch.manual_seed(seed)
    pages_per_seq = math.ceil(max_tokens / PAGE_SIZE)
    total_pages = batch_size * pages_per_seq
    logical_kv = torch.randn(
        batch_size,
        max_tokens,
        H_KV,
        D_QK,
        device="cuda",
        dtype=torch.bfloat16,
    )
    logical_kv = (logical_kv / 10).clamp(-1.0, 1.0)

    page_table = torch.randperm(
        total_pages,
        device="cuda",
        dtype=torch.int32,
    ).view(batch_size, pages_per_seq)
    blocked_k = torch.empty(
        total_pages,
        PAGE_SIZE,
        H_KV,
        D_QK,
        device="cuda",
        dtype=torch.bfloat16,
    )

    for row in range(batch_size):
        for logical_page in range(pages_per_seq):
            physical_page = int(page_table[row, logical_page].item())
            start = logical_page * PAGE_SIZE
            end = min(start + PAGE_SIZE, max_tokens)
            blocked_k[physical_page].zero_()
            blocked_k[physical_page, : end - start] = logical_kv[row, start:end]

    return logical_kv, blocked_k, page_table


def _build_clamped_dense_token_indices(
    cache_seqlens: torch.Tensor,
    width: int,
) -> torch.Tensor:
    base = torch.arange(
        width,
        device=cache_seqlens.device,
        dtype=torch.long,
    ).unsqueeze(0).expand(cache_seqlens.numel(), -1)
    cap = (cache_seqlens.to(dtype=torch.long) - 1).clamp(min=0).unsqueeze(1)
    return torch.minimum(base, cap)


def _current_glm_hot_indices(
    cache_seqlens: torch.Tensor,
    max_seqlen: int,
    long_topk_indices: torch.Tensor,
) -> torch.Tensor:
    short_mask = cache_seqlens <= INDEX_TOPK
    if bool(short_mask.all().item()):
        return _build_clamped_dense_token_indices(cache_seqlens, max_seqlen)
    if not bool(short_mask.any().item()):
        return long_topk_indices

    out = torch.empty_like(long_topk_indices)
    out[short_mask] = _build_clamped_dense_token_indices(
        cache_seqlens[short_mask],
        INDEX_TOPK,
    )
    out[~short_mask] = long_topk_indices[~short_mask]
    return out


@pytest.mark.parametrize(
    "cache_values",
    [
        [512, 512],
        [INDEX_TOPK, INDEX_TOPK],
        [4096, 3072],
        [64, 2047, INDEX_TOPK, 4096],
    ],
)
def test_fused_selector_flashmla_matches_current_glm_hot_path(cache_values):
    torch.cuda.set_device(0)
    batch_size = len(cache_values)
    max_tokens = max(max(cache_values), INDEX_TOPK)
    cache_seqlens = torch.tensor(cache_values, device="cuda", dtype=torch.int32)
    _, primary_blocked_k, primary_page_table = _make_paged_primary_cache(
        batch_size=batch_size,
        max_tokens=max_tokens,
        seed=37 + batch_size,
    )

    long_topk_indices = torch.full(
        (batch_size, INDEX_TOPK),
        -1,
        device="cuda",
        dtype=torch.long,
    )
    for row, seqlen in enumerate(cache_values):
        if seqlen > INDEX_TOPK:
            long_topk_indices[row] = torch.randperm(seqlen, device="cuda")[:INDEX_TOPK]

    current_indices = _current_glm_hot_indices(
        cache_seqlens,
        max(cache_values),
        long_topk_indices,
    )
    current_selected = sparse_gather_from_paged_kv(
        primary_blocked_k,
        primary_page_table,
        current_indices,
        PAGE_SIZE,
    )
    current_lengths = torch.clamp(
        cache_seqlens,
        max=current_indices.shape[1],
    ).to(dtype=torch.int32)

    fused_selected, fused_lengths, _, fused_row_modes = select_mla_kv_for_flashmla_bf16(
        primary_blocked_k,
        primary_page_table,
        cache_seqlens,
        long_topk_indices,
        index_topk=INDEX_TOPK,
        page_size=PAGE_SIZE,
        return_indices=False,
    )

    assert fused_selected.shape == (batch_size, INDEX_TOPK, H_KV, D_QK)
    torch.testing.assert_close(fused_lengths, torch.clamp(cache_seqlens, max=INDEX_TOPK))
    torch.testing.assert_close(
        fused_row_modes,
        (cache_seqlens > INDEX_TOPK).to(dtype=torch.int32),
    )

    query_states = _make_query(batch_size, seed=41 + batch_size)
    softmax_scale = D_QK**-0.5
    current_out, _ = _run_flashmla_dense(
        query_states,
        current_selected,
        current_lengths,
        softmax_scale=softmax_scale,
    )
    fused_out, _ = _run_flashmla_dense(
        query_states,
        fused_selected,
        fused_lengths,
        softmax_scale=softmax_scale,
    )
    _assert_flashmla_close(fused_out, current_out)


@pytest.mark.parametrize(
    "cache_values",
    [
        [33, INDEX_TOPK],
        [4096, 3072],
        [17, INDEX_TOPK, 4096],
    ],
)
def test_unified_selector_matches_standalone_reference(cache_values):
    torch.cuda.set_device(0)
    batch_size = len(cache_values)
    max_tokens = max(max(cache_values), INDEX_TOPK)
    cache_seqlens = torch.tensor(cache_values, device="cuda", dtype=torch.int32)
    _, primary_blocked_k, primary_page_table = _make_paged_primary_cache(
        batch_size=batch_size,
        max_tokens=max_tokens,
        seed=19 + batch_size,
    )
    for row, seqlen in enumerate(cache_values):
        valid_pages = math.ceil(seqlen / PAGE_SIZE)
        primary_page_table[row, valid_pages:] = -1

    long_topk_indices = torch.full(
        (batch_size, INDEX_TOPK),
        -1,
        device="cuda",
        dtype=torch.int64,
    )
    for row, seqlen in enumerate(cache_values):
        if seqlen > INDEX_TOPK:
            perm = torch.randperm(seqlen, device="cuda")[:INDEX_TOPK]
            perm[:8] = torch.tensor(
                [0, 1, 63, 64, 65, seqlen - 3, seqlen - 2, seqlen - 1],
                device="cuda",
            )
            long_topk_indices[row] = perm

    actual = select_mla_kv_for_flashmla_bf16(
        primary_blocked_k,
        primary_page_table,
        cache_seqlens,
        long_topk_indices,
        index_topk=INDEX_TOPK,
        page_size=PAGE_SIZE,
    )
    expected = _reference_select_mla_kv_for_flashmla_bf16(
        primary_blocked_k,
        primary_page_table,
        cache_seqlens,
        long_topk_indices,
    )

    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor)


def test_reference_selector_output_feeds_flashmla_dense_contract():
    torch.cuda.set_device(0)
    batch_size = 3
    max_tokens = 4096
    cache_seqlens = torch.tensor(
        [33, INDEX_TOPK, max_tokens],
        device="cuda",
        dtype=torch.int32,
    )
    logical_kv, primary_blocked_k, primary_page_table = _make_paged_primary_cache(
        batch_size=batch_size,
        max_tokens=max_tokens,
        seed=23,
    )

    long_topk_indices = torch.zeros(
        batch_size,
        INDEX_TOPK,
        device="cuda",
        dtype=torch.int64,
    )
    long_topk_indices[2] = torch.randperm(max_tokens, device="cuda")[:INDEX_TOPK]

    selected, selected_lengths, selected_indices, row_modes = (
        select_mla_kv_for_flashmla_bf16(
            primary_blocked_k,
            primary_page_table,
            cache_seqlens,
            long_topk_indices,
            index_topk=INDEX_TOPK,
            page_size=PAGE_SIZE,
        )
    )

    assert selected.shape == (batch_size, INDEX_TOPK, H_KV, D_QK)
    assert selected_lengths.tolist() == [33, INDEX_TOPK, INDEX_TOPK]
    assert row_modes.tolist() == [0, 0, 1]
    torch.testing.assert_close(selected[0, :33], logical_kv[0, :33])
    torch.testing.assert_close(
        selected_indices[0, :33],
        torch.arange(33, device="cuda"),
    )
    assert torch.all(selected_indices[0, 33:] == -1)
    assert torch.count_nonzero(selected[0, 33:]) == 0
    torch.testing.assert_close(selected[1], logical_kv[1, :INDEX_TOPK])
    torch.testing.assert_close(selected[2], logical_kv[2, selected_indices[2]])

    query_states = _make_query(batch_size, seed=29)
    softmax_scale = D_QK**-0.5
    attn_out, _ = _run_flashmla_dense(
        query_states,
        selected,
        selected_lengths,
        softmax_scale=softmax_scale,
    )
    ref_out, _ = _selected_attention_reference(
        query_states,
        selected,
        selected_lengths,
        softmax_scale=softmax_scale,
    )
    _assert_flashmla_close(attn_out, ref_out)
