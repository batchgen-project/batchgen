"""GLM-5 DSA CUDA graph segment tests."""

from __future__ import annotations

import pytest
import torch

from batchgen.cuda_graph import BatchSizeBucketing, CUDAGraphManager
from batchgen.models.glm.glm5.cuda_graph_segments import (
    Glm5DsaAttnSegment,
    make_glm5_dsa_graph_segment_name,
)
from batchgen_kernels.attention.dsa.fp8_absorb import FP8AbsorbWeights
from batchgen_kernels.attention.dsa.fused_indexer_kv_proj_cuda import build_module
from batchgen_kernels.attention.dsa.fused_indexer_score import FP8WqbWeightsCUDA


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="GLM-5 DSA graph segment tests require CUDA",
)


PAGE_SIZE = 64
INDEX_HEADS = 32
ATTN_HEADS = 64
INDEX_DIM = 128
Q_RANK = 2048
Q_NOPE = 192
KV_DIM = 576
KV_LORA = 512
ATTN_OUT = 256


def _require_flash_mla():
    return pytest.importorskip("flash_mla")


def _rope_tables(max_pos: int, rope_dim: int = 64):
    theta = 1000000.0
    freqs = 1.0 / (
        theta ** (torch.arange(0, rope_dim, 2, device="cuda").float() / rope_dim)
    )
    t = torch.arange(max_pos, device="cuda").float()
    angles = t[:, None] * freqs[None, :]
    return (
        torch.cos(angles).repeat(1, 2).contiguous(),
        torch.sin(angles).repeat(1, 2).contiguous(),
    )


def _make_primary_cache(batch_size: int, max_seqlen: int):
    pages_per_seq = (max_seqlen + PAGE_SIZE - 1) // PAGE_SIZE
    total_pages = batch_size * pages_per_seq
    blocked_k = (
        torch.randn(
            total_pages,
            PAGE_SIZE,
            1,
            KV_DIM,
            device="cuda",
            dtype=torch.bfloat16,
        )
        * 0.1
    ).contiguous()
    page_table = torch.arange(
        total_pages, device="cuda", dtype=torch.int32
    ).view(batch_size, pages_per_seq)
    return blocked_k, page_table


def _make_aux_cache(batch_size: int, max_seqlen: int):
    pages_per_seq = (max_seqlen + PAGE_SIZE - 1) // PAGE_SIZE
    total_pages = batch_size * pages_per_seq
    blocked_k = (
        torch.randn(
            total_pages,
            PAGE_SIZE,
            1,
            INDEX_DIM,
            device="cuda",
            dtype=torch.bfloat16,
        )
        * 0.1
    ).contiguous()
    page_table = torch.arange(
        total_pages, device="cuda", dtype=torch.int32
    ).view(batch_size, pages_per_seq)
    return blocked_k, page_table


def _make_inputs(batch_size: int, max_seqlen: int):
    positions = torch.arange(
        max_seqlen - batch_size, max_seqlen, device="cuda", dtype=torch.int64
    ).contiguous()
    return {
        "q_a": (
            torch.randn(batch_size, Q_RANK, device="cuda", dtype=torch.bfloat16) * 0.1
        ).contiguous(),
        "q_nope": (
            torch.randn(batch_size, ATTN_HEADS, Q_NOPE, device="cuda", dtype=torch.bfloat16)
            * 0.1
        ).contiguous(),
        "q_rope": (
            torch.randn(batch_size, ATTN_HEADS, 64, device="cuda", dtype=torch.bfloat16)
            * 0.1
        ).contiguous(),
        "head_gates": torch.randn(
            batch_size, INDEX_HEADS, device="cuda", dtype=torch.float32
        ).contiguous(),
        "cache_seqlens": torch.full(
            (batch_size,), max_seqlen, device="cuda", dtype=torch.int32
        ),
        "positions_expanded": positions[:, None]
        .expand(batch_size, INDEX_HEADS)
        .contiguous(),
    }


def test_glm5_dsa_segment_replay_matches_eager_forward():
    _require_flash_mla()
    torch.cuda.set_device(0)
    torch.manual_seed(20260430)

    batch_size = 2
    max_seqlen = 1024
    index_topk = 128
    module = build_module()
    primary_blocked_k, primary_page_table = _make_primary_cache(batch_size, max_seqlen)
    aux_blocked_k, aux_page_table = _make_aux_cache(batch_size, max_seqlen)
    cos, sin = _rope_tables(max_seqlen + 8)

    wq_b = (
        torch.randn(INDEX_HEADS * INDEX_DIM, Q_RANK, device="cuda", dtype=torch.bfloat16)
        * 0.01
    ).contiguous()
    q_absorb = (
        torch.randn(ATTN_HEADS, Q_NOPE, KV_LORA, device="cuda", dtype=torch.bfloat16)
        * 0.01
    ).contiguous()
    out_absorb = (
        torch.randn(ATTN_HEADS, ATTN_OUT, KV_LORA, device="cuda", dtype=torch.bfloat16)
        * 0.01
    ).contiguous()

    segment = Glm5DsaAttnSegment(
        primary_blocked_k=primary_blocked_k,
        aux_blocked_k=aux_blocked_k,
        wq_b_weights=FP8WqbWeightsCUDA(wq_b, module),
        absorb_weights=FP8AbsorbWeights(q_absorb, out_absorb),
        cuda_module=module,
        cos_table=cos,
        sin_table=sin,
        max_seqlen=max_seqlen,
        index_topk=index_topk,
        page_size=PAGE_SIZE,
        softmax_scale=KV_DIM**-0.5,
    )
    segment_name = make_glm5_dsa_graph_segment_name(0)
    manager = CUDAGraphManager(
        BatchSizeBucketing([1, batch_size]),
        device=torch.device("cuda"),
    )
    manager.register_segment(segment_name, segment)
    manager.warmup_and_capture_all()

    inputs = _make_inputs(batch_size, max_seqlen)
    inputs["primary_page_table"] = primary_page_table
    inputs["aux_page_table"] = aux_page_table
    expected = {
        key: value.clone()
        for key, value in segment.forward(**inputs).items()
    }

    actual = manager.replay(segment_name, batch_size, **inputs)
    torch.cuda.synchronize()

    torch.testing.assert_close(actual["attn_heads"], expected["attn_heads"], atol=0, rtol=0)
    torch.testing.assert_close(
        actual["selected_lengths"], expected["selected_lengths"], atol=0, rtol=0
    )
    torch.testing.assert_close(
        actual["top_k_indices"], expected["top_k_indices"], atol=0, rtol=0
    )
    torch.testing.assert_close(
        actual["selected_mla_kv"], expected["selected_mla_kv"], atol=0, rtol=0
    )

    small_inputs = _make_inputs(1, max_seqlen)
    small_inputs["primary_page_table"] = primary_page_table[:1]
    small_inputs["aux_page_table"] = aux_page_table[:1]
    small_expected = {
        key: value.clone()
        for key, value in segment.forward(**small_inputs).items()
    }
    small_actual = manager.replay(segment_name, 1, **small_inputs)
    torch.cuda.synchronize()

    torch.testing.assert_close(
        small_actual["attn_heads"], small_expected["attn_heads"], atol=0, rtol=0
    )
    torch.testing.assert_close(
        small_actual["top_k_indices"], small_expected["top_k_indices"], atol=0, rtol=0
    )
