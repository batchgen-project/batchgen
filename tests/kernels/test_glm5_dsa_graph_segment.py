"""GLM-5 DSA CUDA graph segment tests."""

from __future__ import annotations

import pytest
import torch

from batchgen.cuda_graph import BatchSizeBucketing, CUDAGraphManager
from batchgen.attention.dsa.sparse_decode_mla import (
    prepare_sparse_flash_mla_decode_tensor_metadata,
    prepare_sparse_flash_mla_decode_inputs,
    run_prepared_sparse_flash_mla_decode,
)
from batchgen.models.glm.glm5.cuda_graph_segments import (
    Glm5DsaAttnSegment,
    make_glm5_dsa_graph_segment_name,
)
from batchgen.models.glm.glm5.cuda_graph_policy import GLM5_POWER_OF_TWO_BUCKETS_32
from batchgen_kernels.attention.dsa.fp8_absorb import FP8AbsorbWeights, fp8_out_absorb_out
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


def _make_primary_cache(num_slots: int, max_seqlen: int):
    pages_per_seq = (max_seqlen + PAGE_SIZE - 1) // PAGE_SIZE
    total_pages = num_slots * pages_per_seq
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
    ).view(num_slots, pages_per_seq)
    return blocked_k, page_table


def _make_aux_cache(num_slots: int, max_seqlen: int):
    pages_per_seq = (max_seqlen + PAGE_SIZE - 1) // PAGE_SIZE
    total_pages = num_slots * pages_per_seq
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
    ).view(num_slots, pages_per_seq)
    return blocked_k, page_table


def _make_inputs(
    batch_size: int,
    max_seqlen: int,
    cache_seqlens: torch.Tensor | None = None,
):
    if cache_seqlens is None:
        cache_seqlens = torch.full(
            (batch_size,), max_seqlen, device="cuda", dtype=torch.int32
        )
    else:
        cache_seqlens = cache_seqlens.to(device="cuda", dtype=torch.int32).contiguous()
    positions = (cache_seqlens.to(torch.int64) - 1).clamp_min_(0).contiguous()
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
        "cache_seqlens": cache_seqlens,
        "positions_expanded": positions[:, None]
        .expand(batch_size, INDEX_HEADS)
        .contiguous(),
    }


def _flashmla_graph_metadata_inputs(
    cache_seqlens: torch.Tensor,
    *,
    batch_size: int,
    bucket_size: int,
    index_topk: int,
    num_heads: int = ATTN_HEADS,
    padding_selected_length: int | None = None,
) -> dict[str, torch.Tensor]:
    if padding_selected_length is None:
        padding_selected_length = index_topk
    selected_lengths = torch.empty(bucket_size, device="cuda", dtype=torch.int32)
    selected_lengths[:batch_size].copy_(
        torch.clamp(cache_seqlens[:batch_size].to(dtype=torch.int32), max=index_topk)
    )
    if batch_size < bucket_size:
        selected_lengths[batch_size:].fill_(padding_selected_length)
    tile_scheduler_metadata, num_splits = prepare_sparse_flash_mla_decode_tensor_metadata(
        selected_lengths,
        num_heads,
    )
    return {
        "flashmla_tile_scheduler_metadata": tile_scheduler_metadata,
        "flashmla_num_splits": num_splits,
    }


def _with_flashmla_graph_metadata(
    manager: CUDAGraphManager,
    inputs: dict[str, torch.Tensor],
    *,
    batch_size: int,
    index_topk: int,
    max_seqlen: int,
    metadata_rows: int | None = None,
) -> dict[str, torch.Tensor]:
    if metadata_rows is None:
        metadata_rows = manager.bucketing.get_padded_size(batch_size)
    return {
        **inputs,
        **_flashmla_graph_metadata_inputs(
            inputs["cache_seqlens"],
            batch_size=batch_size,
            bucket_size=metadata_rows,
            index_topk=index_topk,
            padding_selected_length=min(max_seqlen, index_topk),
        ),
    }


def test_glm5_dsa_segments_share_static_buffers_by_bucket():
    _require_flash_mla()
    torch.cuda.set_device(0)
    torch.manual_seed(20260502)

    batch_size = 2
    max_seqlen = 256
    index_topk = 128
    module = build_module()
    primary_blocked_k, primary_page_table = _make_primary_cache(4, max_seqlen)
    aux_blocked_k, aux_page_table = _make_aux_cache(4, max_seqlen)
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
    shared_buffers = {}

    segment_a = Glm5DsaAttnSegment(
        primary_blocked_k=primary_blocked_k,
        aux_blocked_k=aux_blocked_k,
        primary_page_table=primary_page_table,
        aux_page_table=aux_page_table,
        wq_b_weights=FP8WqbWeightsCUDA(wq_b, module),
        absorb_weights=FP8AbsorbWeights(q_absorb, out_absorb),
        cuda_module=module,
        cos_table=cos,
        sin_table=sin,
        max_seqlen=max_seqlen,
        index_topk=index_topk,
        page_size=PAGE_SIZE,
        softmax_scale=KV_DIM**-0.5,
        shared_buffers=shared_buffers,
    )
    segment_b = Glm5DsaAttnSegment(
        primary_blocked_k=primary_blocked_k,
        aux_blocked_k=aux_blocked_k,
        primary_page_table=primary_page_table,
        aux_page_table=aux_page_table,
        wq_b_weights=FP8WqbWeightsCUDA(wq_b, module),
        absorb_weights=FP8AbsorbWeights(q_absorb, out_absorb),
        cuda_module=module,
        cos_table=cos,
        sin_table=sin,
        max_seqlen=max_seqlen,
        index_topk=index_topk,
        page_size=PAGE_SIZE,
        softmax_scale=KV_DIM**-0.5,
        shared_buffers=shared_buffers,
    )

    segment_a.setup_static_buffers(batch_size)
    segment_b.setup_static_buffers(batch_size)

    assert len(shared_buffers) == 1
    assert segment_a._buffers[batch_size] is segment_b._buffers[batch_size]
    assert (
        segment_a._buffers[batch_size].selected_mla_kv.data_ptr()
        == segment_b._buffers[batch_size].selected_mla_kv.data_ptr()
    )
    assert (
        segment_a._attn_head_outputs[batch_size].data_ptr()
        != segment_b._attn_head_outputs[batch_size].data_ptr()
    )
    segment_a.release_static_buffers(batch_size)
    assert batch_size not in shared_buffers
    assert batch_size not in segment_a._attn_head_outputs


def test_glm5_dsa_segment_replay_matches_eager_forward():
    _require_flash_mla()
    torch.cuda.set_device(0)
    torch.manual_seed(20260430)

    batch_size = 2
    num_slots = 4
    max_seqlen = 1024
    index_topk = 128
    primary_slot_indices = torch.tensor([2, 0], device="cuda", dtype=torch.int32)
    aux_slot_indices = torch.tensor([3, 1], device="cuda", dtype=torch.int32)
    module = build_module()
    primary_blocked_k, primary_page_table = _make_primary_cache(num_slots, max_seqlen)
    aux_blocked_k, aux_page_table = _make_aux_cache(num_slots, max_seqlen)
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

    absorb = FP8AbsorbWeights(q_absorb, out_absorb)
    graph_segment = Glm5DsaAttnSegment(
        primary_blocked_k=primary_blocked_k,
        aux_blocked_k=aux_blocked_k,
        primary_page_table=primary_page_table,
        aux_page_table=aux_page_table,
        wq_b_weights=FP8WqbWeightsCUDA(wq_b, module),
        absorb_weights=absorb,
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
        BatchSizeBucketing(GLM5_POWER_OF_TWO_BUCKETS_32),
        device=torch.device("cuda"),
    )
    manager.register_segment(segment_name, segment)
    manager.warmup_and_capture_all()
    assert (
        manager.get_capture_stats()["graphs_per_segment"][segment_name]
        == GLM5_POWER_OF_TWO_BUCKETS_32
    )

    inputs = _make_inputs(batch_size, max_seqlen)
    inputs["primary_slot_indices"] = primary_slot_indices
    inputs["aux_slot_indices"] = aux_slot_indices
    expected = {
        key: value.clone()
        for key, value in segment.forward(
            **_with_flashmla_graph_metadata(
                manager,
                inputs,
                batch_size=batch_size,
                index_topk=index_topk,
                max_seqlen=max_seqlen,
                metadata_rows=batch_size,
            )
        ).items()
    }

    actual = manager.replay(
        segment_name,
        batch_size,
        **_with_flashmla_graph_metadata(
            manager,
            inputs,
            batch_size=batch_size,
            index_topk=index_topk,
            max_seqlen=max_seqlen,
        ),
    )
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
    small_inputs["primary_slot_indices"] = primary_slot_indices[:1]
    small_inputs["aux_slot_indices"] = aux_slot_indices[:1]
    small_expected = {
        key: value.clone()
        for key, value in segment.forward(
            **_with_flashmla_graph_metadata(
                manager,
                small_inputs,
                batch_size=1,
                index_topk=index_topk,
                max_seqlen=max_seqlen,
                metadata_rows=1,
            )
        ).items()
    }
    small_actual = manager.replay(
        segment_name,
        1,
        **_with_flashmla_graph_metadata(
            manager,
            small_inputs,
            batch_size=1,
            index_topk=index_topk,
            max_seqlen=max_seqlen,
        ),
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(
        small_actual["attn_heads"], small_expected["attn_heads"], atol=0, rtol=0
    )
    torch.testing.assert_close(
        small_actual["top_k_indices"], small_expected["top_k_indices"], atol=0, rtol=0
    )

    mid_batch_size = 17
    mid_inputs = _make_inputs(mid_batch_size, max_seqlen)
    mid_slots = (
        torch.arange(mid_batch_size, device="cuda", dtype=torch.int32) % num_slots
    )
    mid_inputs["primary_slot_indices"] = mid_slots
    mid_inputs["aux_slot_indices"] = mid_slots
    mid_expected = {
        key: value.clone()
        for key, value in segment.forward(
            **_with_flashmla_graph_metadata(
                manager,
                mid_inputs,
                batch_size=mid_batch_size,
                index_topk=index_topk,
                max_seqlen=max_seqlen,
                metadata_rows=mid_batch_size,
            )
        ).items()
    }
    mid_actual = manager.replay(
        segment_name,
        mid_batch_size,
        **_with_flashmla_graph_metadata(
            manager,
            mid_inputs,
            batch_size=mid_batch_size,
            index_topk=index_topk,
            max_seqlen=max_seqlen,
        ),
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(
        mid_actual["attn_heads"], mid_expected["attn_heads"], atol=0, rtol=0
    )
    torch.testing.assert_close(
        mid_actual["top_k_indices"], mid_expected["top_k_indices"], atol=0, rtol=0
    )


def test_glm5_dsa_segment_replay_supports_unified_short_and_mixed_rows():
    _require_flash_mla()
    torch.cuda.set_device(0)
    torch.manual_seed(20260430)

    batch_size = 3
    num_slots = 5
    max_seqlen = 256
    index_topk = 128
    primary_slot_indices = torch.tensor([4, 0, 2], device="cuda", dtype=torch.int32)
    aux_slot_indices = torch.tensor([3, 1, 4], device="cuda", dtype=torch.int32)
    cache_seqlens = torch.tensor([17, 128, 129], device="cuda", dtype=torch.int32)
    module = build_module()
    primary_blocked_k, primary_page_table = _make_primary_cache(num_slots, max_seqlen)
    aux_blocked_k, aux_page_table = _make_aux_cache(num_slots, max_seqlen)
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
    absorb = FP8AbsorbWeights(q_absorb, out_absorb)

    segment = Glm5DsaAttnSegment(
        primary_blocked_k=primary_blocked_k,
        aux_blocked_k=aux_blocked_k,
        primary_page_table=primary_page_table,
        aux_page_table=aux_page_table,
        wq_b_weights=FP8WqbWeightsCUDA(wq_b, module),
        absorb_weights=absorb,
        cuda_module=module,
        cos_table=cos,
        sin_table=sin,
        max_seqlen=max_seqlen,
        index_topk=index_topk,
        page_size=PAGE_SIZE,
        softmax_scale=KV_DIM**-0.5,
    )
    eager_segment = Glm5DsaAttnSegment(
        primary_blocked_k=primary_blocked_k,
        aux_blocked_k=aux_blocked_k,
        primary_page_table=primary_page_table,
        aux_page_table=aux_page_table,
        wq_b_weights=FP8WqbWeightsCUDA(wq_b, module),
        absorb_weights=absorb,
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
        BatchSizeBucketing([batch_size]),
        device=torch.device("cuda"),
    )
    manager.register_segment(segment_name, graph_segment)
    manager.warmup_and_capture_all()

    inputs = _make_inputs(batch_size, max_seqlen, cache_seqlens)
    inputs["primary_slot_indices"] = primary_slot_indices
    inputs["aux_slot_indices"] = aux_slot_indices
    expected = {
        key: value.clone()
        for key, value in segment.forward(
            **_with_flashmla_graph_metadata(
                manager,
                inputs,
                batch_size=batch_size,
                index_topk=index_topk,
                max_seqlen=max_seqlen,
                metadata_rows=batch_size,
            )
        ).items()
    }
    buffers = segment._buffers[batch_size]
    assert buffers.prepared_flashmla.cache_seqlens.data_ptr() == buffers.selected_lengths.data_ptr()
    assert expected["selected_lengths"].tolist() == [17, 128, 128]
    assert buffers.row_modes.tolist() == [0, 0, 1]

    fresh_prepared = prepare_sparse_flash_mla_decode_inputs(
        buffers.query_states.clone(),
        expected["selected_mla_kv"].clone(),
        expected["selected_lengths"].clone(),
        ATTN_HEADS,
        KV_DIM**-0.5,
        head_dim_v=KV_LORA,
        page_size=PAGE_SIZE,
    )
    fresh_attn = run_prepared_sparse_flash_mla_decode(fresh_prepared)
    fresh_heads = torch.empty_like(expected["attn_heads"])
    fp8_out_absorb_out(fresh_attn, absorb, fresh_heads)
    torch.testing.assert_close(
        expected["attn_heads"], fresh_heads, atol=5e-2, rtol=5e-2
    )

    actual = manager.replay(
        segment_name,
        batch_size,
        **_with_flashmla_graph_metadata(
            manager,
            inputs,
            batch_size=batch_size,
            index_topk=index_topk,
            max_seqlen=max_seqlen,
        ),
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(actual["selected_lengths"], expected["selected_lengths"], atol=0, rtol=0)
    torch.testing.assert_close(actual["top_k_indices"], expected["top_k_indices"], atol=0, rtol=0)
    torch.testing.assert_close(actual["selected_mla_kv"], expected["selected_mla_kv"], atol=0, rtol=0)
    torch.testing.assert_close(actual["attn_heads"], expected["attn_heads"], atol=0, rtol=0)


def test_glm5_dsa_segment_replay_uses_external_flashmla_metadata_for_production_short_rows():
    _require_flash_mla()
    torch.cuda.set_device(0)
    torch.manual_seed(20260505)

    batch_size = 3
    num_slots = 5
    max_seqlen = 4096
    index_topk = 2048
    primary_slot_indices = torch.tensor([4, 0, 2], device="cuda", dtype=torch.int32)
    aux_slot_indices = torch.tensor([3, 1, 4], device="cuda", dtype=torch.int32)
    module = build_module()
    primary_blocked_k, primary_page_table = _make_primary_cache(num_slots, max_seqlen)
    aux_blocked_k, aux_page_table = _make_aux_cache(num_slots, max_seqlen)
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
    absorb = FP8AbsorbWeights(q_absorb, out_absorb)

    segment = Glm5DsaAttnSegment(
        primary_blocked_k=primary_blocked_k,
        aux_blocked_k=aux_blocked_k,
        primary_page_table=primary_page_table,
        aux_page_table=aux_page_table,
        wq_b_weights=FP8WqbWeightsCUDA(wq_b, module),
        absorb_weights=absorb,
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
        BatchSizeBucketing([batch_size]),
        device=torch.device("cuda"),
    )
    manager.register_segment(segment_name, segment)
    manager.warmup_and_capture_all()

    for cache_values in ([970, 982, 2048], [1024, 1536, 2049]):
        cache_seqlens = torch.tensor(cache_values, device="cuda", dtype=torch.int32)
        inputs = _make_inputs(batch_size, max_seqlen, cache_seqlens)
        inputs["primary_slot_indices"] = primary_slot_indices
        inputs["aux_slot_indices"] = aux_slot_indices

        expected = {
            key: value.clone()
            for key, value in eager_segment.forward(
                **_with_flashmla_graph_metadata(
                    manager,
                    inputs,
                    batch_size=batch_size,
                    index_topk=index_topk,
                    max_seqlen=max_seqlen,
                    metadata_rows=batch_size,
                )
            ).items()
        }
        assert expected["selected_lengths"].tolist() == [
            min(value, index_topk) for value in cache_values
        ]

        actual = manager.replay(
            segment_name,
            batch_size,
            **_with_flashmla_graph_metadata(
                manager,
                inputs,
                batch_size=batch_size,
                index_topk=index_topk,
                max_seqlen=max_seqlen,
            ),
        )
        torch.cuda.synchronize()

        torch.testing.assert_close(
            actual["selected_lengths"], expected["selected_lengths"], atol=0, rtol=0
        )
        torch.testing.assert_close(
            actual["raw_attn_out"], expected["raw_attn_out"], atol=5e-2, rtol=5e-2
        )
        torch.testing.assert_close(
            actual["attn_heads"], expected["attn_heads"], atol=5e-2, rtol=5e-2
        )
