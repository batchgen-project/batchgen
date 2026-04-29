"""Eager BF16 DSA selected-KV contract tests.

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

from batchgen.attention.dsa.glm5_decode_selector import _select_glm5_dsa_indices
from batchgen.attention.dsa.sparse_decode_mla import (
    prepare_sparse_flash_mla_decode_inputs,
    run_prepared_sparse_flash_mla_decode,
)
from batchgen.attention.dsa.unified_selector import (
    make_flashmla_selected_block_table,
    select_mla_kv_for_flashmla_bf16,
    view_selected_mla_kv_as_flashmla_pages,
)
from batchgen.models.wrappers import AttnWrapperBase


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


class _FakeIndexer:
    index_topk = INDEX_TOPK


class _FakeModule:
    indexer = _FakeIndexer()


class _FakeWrapper:
    module = _FakeModule()
    layer_idx = 0


def test_glm5_dsa_index_selector_all_short_keeps_fixed_topk_shape():
    torch.cuda.set_device(0)
    cache_seqlens = torch.tensor(
        [33, INDEX_TOPK],
        device="cuda",
        dtype=torch.int32,
    )
    old_short_count = AttnWrapperBase._dsa_short_count
    AttnWrapperBase._dsa_short_count = None
    try:
        top_k_indices, branch_label, row_modes = _select_glm5_dsa_indices(
            _FakeWrapper(),
            hidden_states=torch.empty(2, 1, 1, device="cuda"),
            q_a_normed=torch.empty(2, 1, device="cuda"),
            cache_seqlens=cache_seqlens,
            max_seqlen=INDEX_TOPK,
            new_token_pos=cache_seqlens - 1,
            gpu_paged_kv_manager_aux=None,
            aux_slot_indices=torch.empty(2, dtype=torch.int32, device="cuda"),
        )
    finally:
        AttnWrapperBase._dsa_short_count = old_short_count

    assert branch_label == "dense-short-circuit"
    assert top_k_indices.shape == (2, INDEX_TOPK)
    assert row_modes.tolist() == [0, 0]
    torch.testing.assert_close(
        top_k_indices[0, :33],
        torch.arange(33, device="cuda"),
    )
    assert torch.all(top_k_indices[0, 33:] == 32)
    torch.testing.assert_close(
        top_k_indices[1],
        torch.arange(INDEX_TOPK, device="cuda"),
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
