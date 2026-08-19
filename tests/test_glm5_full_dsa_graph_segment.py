import types

import pytest
import torch

from batchgen.cuda_graph.graph_manager import BatchSizeBucketing, CUDAGraphManager
from batchgen.models.glm.glm5 import cuda_graph_segments as segments
from batchgen.models.glm.glm5.cuda_graph_segments import (
    Glm5FullDsaAttnSegment,
    make_glm5_full_dsa_graph_segment_name,
)
from batchgen.models.glm.glm5.wrappers import GLM5AttnWrapper


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


class _Identity(torch.nn.Module):
    def forward(self, x):
        return x


def _fake_linear_weight(out_features, in_features, device):
    values = torch.arange(
        out_features * in_features,
        device=device,
        dtype=torch.float32,
    ).view(out_features, in_features)
    return (values.remainder(7).sub(3).mul_(0.03)).to(torch.bfloat16)


def _build_fake_wrapper(device):
    hidden_size = 8
    q_lora_rank = 4
    num_heads = 2
    qk_nope_head_dim = 2
    qk_rope_head_dim = 2
    q_head_dim = qk_nope_head_dim + qk_rope_head_dim
    kv_lora_rank = 2
    index_head_dim = 3
    index_n_heads = 2

    indexer = types.SimpleNamespace(
        index_topk=3,
        index_head_dim=index_head_dim,
        index_n_heads=index_n_heads,
        k_norm=_Identity().to(device),
        weights_proj=torch.nn.Linear(hidden_size, index_n_heads, bias=False).to(device),
    )
    with torch.no_grad():
        indexer.weights_proj.weight.copy_(
            torch.arange(
                index_n_heads * hidden_size,
                device=device,
                dtype=torch.float32,
            ).view(index_n_heads, hidden_size).mul_(0.01)
        )

    def _fused_rope_hadamard_or_fallback(k_normed, positions, max_seqlen):
        del positions, max_seqlen
        return k_normed + 0.125

    indexer._fused_rope_hadamard_or_fallback = _fused_rope_hadamard_or_fallback

    attn = types.SimpleNamespace(
        hidden_size=hidden_size,
        q_lora_rank=q_lora_rank,
        num_heads=num_heads,
        q_head_dim=q_head_dim,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        kv_lora_rank=kv_lora_rank,
        v_head_dim=kv_lora_rank,
        softmax_scale=0.25,
        indexer=indexer,
        q_a_layernorm=_Identity().to(device),
        kv_a_layernorm=types.SimpleNamespace(
            weight=torch.ones(kv_lora_rank, device=device, dtype=torch.bfloat16),
            eps=1e-5,
        ),
    )
    attn.q_a_proj = types.SimpleNamespace(
        weight=_fake_linear_weight(q_lora_rank, hidden_size, device)
    )
    attn.q_b_proj = types.SimpleNamespace(
        weight=_fake_linear_weight(num_heads * q_head_dim, q_lora_rank, device)
    )
    attn.kv_a_proj_with_mqa = types.SimpleNamespace(
        weight=_fake_linear_weight(kv_lora_rank + qk_rope_head_dim, hidden_size, device)
    )
    attn.o_proj = types.SimpleNamespace(
        weight=_fake_linear_weight(hidden_size, num_heads * kv_lora_rank, device)
    )

    weight_dequant_scale = {
        "q_a_proj.weight_scale_inv": torch.ones(1, device=device, dtype=torch.float32),
        "q_b_proj.weight_scale_inv": torch.ones(1, device=device, dtype=torch.float32),
        "kv_a_proj_with_mqa.weight_scale_inv": torch.ones(1, device=device, dtype=torch.float32),
        "o_proj.weight_scale_inv": torch.ones(1, device=device, dtype=torch.float32),
    }
    return types.SimpleNamespace(
        module=attn,
        layer_idx=0,
        weight_dequant_scale=weight_dequant_scale,
        _fp8_qkv_a_proj=torch.cat(
            (attn.q_a_proj.weight, attn.kv_a_proj_with_mqa.weight),
            dim=0,
        ).contiguous(),
        _fp8_qkv_a_scale=torch.ones(1, device=device, dtype=torch.float32),
        _indexer_cuda_weights=object(),
        _indexer_cuda_module=object(),
    )


def test_glm5_registers_fused_qkv_a_storage_views():
    device = torch.device("cuda")
    q_weight = torch.arange(
        12,
        device=device,
        dtype=torch.float32,
    ).view(3, 4).to(torch.bfloat16)
    kv_weight = torch.arange(
        8,
        device=device,
        dtype=torch.float32,
    ).view(2, 4).add_(100).to(torch.bfloat16)
    q_scale = torch.tensor([[1.0, 2.0]], device=device)
    kv_scale = torch.tensor([[3.0, 4.0]], device=device)
    wrapper = types.SimpleNamespace(
        layer_idx=0,
        module=types.SimpleNamespace(
            q_a_proj=types.SimpleNamespace(weight=types.SimpleNamespace(data=q_weight)),
            q_b_proj=types.SimpleNamespace(weight=types.SimpleNamespace(data=q_weight)),
            kv_a_proj_with_mqa=types.SimpleNamespace(
                weight=types.SimpleNamespace(data=kv_weight)
            ),
            kv_b_proj=types.SimpleNamespace(weight=types.SimpleNamespace(data=kv_weight)),
            o_proj=types.SimpleNamespace(weight=types.SimpleNamespace(data=q_weight)),
        ),
        weight_dequant_scale={
            "q_a_proj.weight_scale_inv": q_scale,
            "kv_a_proj_with_mqa.weight_scale_inv": kv_scale,
        },
    )

    GLM5AttnWrapper._register_fp8_weights(wrapper)

    assert torch.equal(
        wrapper._fp8_qkv_a_proj,
        torch.cat((q_weight, kv_weight), dim=0),
    )
    assert torch.equal(
        wrapper._fp8_qkv_a_scale,
        torch.cat((q_scale, kv_scale), dim=0),
    )
    assert (
        wrapper.module.q_a_proj.weight.data.untyped_storage().data_ptr()
        == wrapper._fp8_qkv_a_proj.untyped_storage().data_ptr()
    )
    assert (
        wrapper.module.kv_a_proj_with_mqa.weight.data.untyped_storage().data_ptr()
        == wrapper._fp8_qkv_a_proj.untyped_storage().data_ptr()
    )


def _patch_full_dsa_dependencies(monkeypatch, *, bucket_size, index_topk, kv_dim, v_dim, device):
    calls = {
        "head_gates": 0,
        "wq_b": 0,
        "rope_hadamard_q": 0,
        "score_topk": 0,
        "transform": 0,
        "select": 0,
        "flashmla": 0,
        "fa3": 0,
    }
    topk_template = torch.arange(index_topk, device=device, dtype=torch.int32).view(1, index_topk)
    topk_template = topk_template.expand(bucket_size, index_topk).contiguous()
    selected_template = topk_template.to(torch.bfloat16).view(bucket_size, index_topk, 1, 1)
    selected_template = selected_template.expand(bucket_size, index_topk, 1, kv_dim).contiguous()

    def fake_act_quant(x, num_valid_tokens=None, scale_tma_aligned=False):
        del num_valid_tokens, scale_tma_aligned
        return x.contiguous(), torch.ones(x.shape[0], 1, device=x.device, dtype=torch.float32)

    def fake_w8a8_deepgemm(a, a_scale, w, w_scale, c=None, disable_ue8m0_cast=True, recipe=None, out=None, num_valid_tokens=None, expected_m=None):
        del a_scale, w_scale, c, disable_ue8m0_cast, recipe, expected_m, num_valid_tokens
        result = a.float().matmul(w.float().t()).to(torch.bfloat16)
        if out is None:
            return result
        out.copy_(result)
        return out

    def fake_rmsnorm_rope(new_compressed_kv, q_rope, cos, sin, position_ids, weight, kv_lora_rank, qk_rope_head_dim, eps):
        del cos, sin, position_ids, weight, kv_lora_rank, qk_rope_head_dim, eps
        q_rope.add_(0.0625)
        return new_compressed_kv + 0.25

    def fake_make_scratch(batch, cols, cuda_module, device):
        del cuda_module
        return (
            torch.empty(batch, cols, dtype=torch.bfloat16, device=device),
            torch.empty(batch, 1, dtype=torch.float32, device=device),
            torch.empty(1, dtype=torch.uint8, device=device),
        )

    def fake_wk_proj(hidden_flat, weights, cuda_module, x_fp8, x_scale, tma_desc, out, num_valid_tokens=None):
        del weights, cuda_module, x_fp8, x_scale, tma_desc, num_valid_tokens
        out.copy_(hidden_flat[:, : out.shape[1]] + 0.5)
        return out

    def fake_wq_b_proj(q_a_normed, weights, cuda_module, x_fp8, x_scale, tma_desc, out, num_valid_tokens=None):
        del weights, cuda_module, x_fp8, x_scale, tma_desc, num_valid_tokens
        calls["wq_b"] += 1
        out.copy_(q_a_normed[:, :1].expand_as(out) + 0.25)
        return out

    def fake_head_gates(hidden, weight, out, *, scale, num_valid_tokens=None):
        del hidden, weight, scale, num_valid_tokens
        calls["head_gates"] += 1
        out.fill_(1)
        return out

    def fake_rope_hadamard_q(q_flat, cos, sin, positions, out):
        del cos, sin, positions
        calls["rope_hadamard_q"] += 1
        out.copy_(q_flat + 0.03125)
        return out

    def fake_score_topk(q_index, aux_blocked_k, aux_page_table, aux_slot_indices, head_gates, cache_seqlens, agg_scores, top_k_indices, *, topk, page_size, max_seqlen, num_valid_tokens=None):
        del q_index, aux_blocked_k, aux_page_table, aux_slot_indices, head_gates, cache_seqlens, agg_scores, page_size, max_seqlen, num_valid_tokens
        calls["score_topk"] += 1
        assert topk == index_topk
        top_k_indices.copy_(topk_template[: top_k_indices.shape[0]])
        return top_k_indices

    def fake_select(primary_blocked_k, primary_page_table, cache_seqlens, top_k_indices, page_size, selected_mla_kv, selected_lengths, selected_indices, row_modes, *, index_topk, return_indices, primary_slot_indices=None, num_valid_tokens=None):
        del primary_blocked_k, primary_page_table, top_k_indices, page_size, selected_indices, return_indices, num_valid_tokens
        calls["select"] += 1
        selected_mla_kv.copy_(selected_template[: selected_mla_kv.shape[0]])
        if primary_slot_indices is not None:
            selected_mla_kv.mul_((cache_seqlens > 0).to(torch.bfloat16).view(-1, 1, 1, 1))
        selected_lengths.fill_(index_topk)
        row_modes.zero_()
        return selected_mla_kv, selected_lengths, None, row_modes

    def fake_transform(primary_page_table, cache_seqlens, top_k_indices, physical_token_ids, selected_lengths, *, page_size, primary_slot_indices=None, num_valid_tokens=None):
        del primary_page_table, page_size, primary_slot_indices
        calls["transform"] += 1
        physical_token_ids.copy_(top_k_indices.to(torch.int32))
        selected_lengths.copy_(torch.clamp(cache_seqlens, max=index_topk))
        if num_valid_tokens is not None:
            valid = (
                torch.arange(
                    physical_token_ids.shape[0],
                    device=physical_token_ids.device,
                    dtype=torch.int32,
                )
                < num_valid_tokens
            )
            physical_token_ids.masked_fill_(~valid.view(-1, 1), -1)
            selected_lengths.masked_fill_(~valid, 0)
        return physical_token_ids, selected_lengths

    def fake_metadata(selected_lengths, num_heads):
        del selected_lengths, num_heads
        return (
            torch.arange(4, device=device, dtype=torch.int32).view(1, 4),
            torch.ones(1, device=device, dtype=torch.int32),
        )

    def fake_prepare(query_states, selected_mla_kv, selected_lengths, num_heads, softmax_scale, *, head_dim_v, page_size):
        del num_heads, softmax_scale, head_dim_v, page_size
        return types.SimpleNamespace(
            query_states=query_states,
            selected_mla_kv=selected_mla_kv,
            selected_lengths=selected_lengths,
        )

    def fake_run_prepared(prepared, *, tile_scheduler_metadata, num_splits):
        del tile_scheduler_metadata, num_splits
        calls["flashmla"] += 1
        selected = prepared.selected_mla_kv[:, :1, :, :v_dim]
        return prepared.query_states[..., :v_dim] + selected

    def fake_fa3(*, q, k_cache, v_cache, qv, page_table, cache_seqlens, **kwargs):
        del q, k_cache, v_cache, page_table, cache_seqlens, kwargs
        calls["fa3"] += 1
        return qv

    def fake_q_absorb(q_nope, weights, absorbed_q, num_valid_tokens=None):
        del weights, num_valid_tokens
        absorbed_q.copy_(q_nope[..., : absorbed_q.shape[-1]])
        return absorbed_q

    def fake_pack_query(absorbed_q, q_rope, query_states, num_valid_tokens=None):
        del num_valid_tokens
        query_states.zero_()
        query_states[:, :, :, : absorbed_q.shape[-1]].copy_(absorbed_q.unsqueeze(1))
        query_states[:, :, :, -q_rope.shape[-1] :].copy_(q_rope.unsqueeze(1))
        return query_states

    def fake_out_absorb(attn_out, weights, attn_heads, num_valid_tokens=None):
        del weights, num_valid_tokens
        attn_heads.copy_(attn_out)
        return attn_heads

    monkeypatch.setattr(segments, "act_quant", fake_act_quant)
    monkeypatch.setattr(segments, "w8a8_deepgemm", fake_w8a8_deepgemm)
    monkeypatch.setattr(segments, "_fused_rmsnorm_rope", fake_rmsnorm_rope)
    monkeypatch.setattr(segments, "make_fp8_activation_scratch", fake_make_scratch)
    monkeypatch.setattr(segments, "cuda_wk_proj_gemm_only_out", fake_wk_proj)
    monkeypatch.setattr(segments, "head_gates_out", fake_head_gates)
    monkeypatch.setattr(segments, "cuda_wq_b_proj_out", fake_wq_b_proj)
    monkeypatch.setattr(segments, "rope_hadamard_q_out", fake_rope_hadamard_q)
    monkeypatch.setattr(segments, "fused_paged_score_and_topk_with_slots_out", fake_score_topk)
    monkeypatch.setattr(segments, "select_mla_kv_for_flashmla_bf16_out", fake_select)
    monkeypatch.setattr(segments, "transform_selected_positions_out", fake_transform)
    monkeypatch.setattr(segments, "prepare_sparse_flash_mla_decode_tensor_metadata", fake_metadata)
    monkeypatch.setattr(segments, "prepare_sparse_flash_mla_decode_inputs", fake_prepare)
    monkeypatch.setattr(segments, "run_prepared_sparse_flash_mla_decode", fake_run_prepared)
    monkeypatch.setattr(segments, "_fa3_with_kvcache", fake_fa3)
    monkeypatch.setattr(segments, "fp8_q_absorb_out", fake_q_absorb)
    monkeypatch.setattr(segments, "pack_flashmla_query_out", fake_pack_query)
    monkeypatch.setattr(segments, "fp8_out_absorb_out", fake_out_absorb)
    return calls


def test_glm5_full_dsa_segment_graph_replay_matches_eager_and_writes_kv(monkeypatch):
    device = torch.device("cuda")
    torch.manual_seed(0)
    bucket_size = 4
    actual_bsz = 2
    page_size = 4
    max_seqlen = 8
    wrapper = _build_fake_wrapper(device)
    attn = wrapper.module
    kv_dim = attn.kv_lora_rank + attn.qk_rope_head_dim
    index_dim = attn.indexer.index_head_dim
    calls = _patch_full_dsa_dependencies(
        monkeypatch,
        bucket_size=bucket_size,
        index_topk=attn.indexer.index_topk,
        kv_dim=kv_dim,
        v_dim=attn.v_head_dim,
        device=device,
    )

    primary_cache = torch.zeros(4, page_size, 1, kv_dim, dtype=torch.bfloat16, device=device)
    aux_cache = torch.zeros(4, page_size, 1, index_dim, dtype=torch.bfloat16, device=device)
    primary_page_table = torch.tensor([[0, -1], [1, -1]], dtype=torch.int32, device=device)
    aux_page_table = torch.tensor([[2, -1], [3, -1]], dtype=torch.int32, device=device)
    cos = torch.ones(max_seqlen, attn.qk_rope_head_dim, dtype=torch.bfloat16, device=device)
    sin = torch.zeros_like(cos)
    shared_buffers = {}
    segment = Glm5FullDsaAttnSegment(
        wrapper=wrapper,
        primary_blocked_k=primary_cache,
        aux_blocked_k=aux_cache,
        primary_page_table=primary_page_table,
        aux_page_table=aux_page_table,
        wq_b_weights=object(),
        absorb_weights=object(),
        cuda_module=object(),
        cos_table=cos,
        sin_table=sin,
        max_seqlen=max_seqlen,
        index_topk=attn.indexer.index_topk,
        page_size=page_size,
        aux_page_size=page_size,
        shared_buffers=shared_buffers,
    )

    hidden = torch.randn(actual_bsz, 1, attn.hidden_size, dtype=torch.bfloat16, device=device)
    position_ids = torch.tensor([[1], [2]], dtype=torch.int64, device=device)
    cache_seqlens = torch.tensor([2, 3], dtype=torch.int32, device=device)
    primary_slots = torch.tensor([0, 1], dtype=torch.int32, device=device)
    aux_slots = torch.tensor([0, 1], dtype=torch.int32, device=device)
    metadata = torch.arange(4, dtype=torch.int32, device=device).view(1, 4)
    num_splits = torch.ones(1, dtype=torch.int32, device=device)
    num_valid_tokens = torch.tensor([actual_bsz], dtype=torch.int32, device=device)

    def run_eager():
        primary_cache.zero_()
        aux_cache.zero_()
        outputs = segment.forward(
            hidden_states=hidden,
            position_ids=position_ids,
            cache_seqlens=cache_seqlens,
            primary_slot_indices=primary_slots,
            aux_slot_indices=aux_slots,
            num_valid_tokens=num_valid_tokens,
            flashmla_tile_scheduler_metadata=metadata,
            flashmla_num_splits=num_splits,
        )
        torch.cuda.synchronize()
        return (
            {key: value.detach().clone() for key, value in outputs.items()},
            primary_cache.detach().clone(),
            aux_cache.detach().clone(),
        )

    eager_outputs, eager_primary_cache, eager_aux_cache = run_eager()
    assert calls["transform"] == 1
    assert calls["select"] == 0
    assert calls["flashmla"] == 0
    assert calls["fa3"] == 1

    manager = CUDAGraphManager(BatchSizeBucketing([bucket_size]), device=device)
    manager.WARMUP_ITERATIONS = 1
    name = make_glm5_full_dsa_graph_segment_name(0)
    manager.register_segment(name, segment)
    manager.warmup_and_capture_buckets([bucket_size])
    assert bucket_size in shared_buffers
    assert bucket_size in segment._outputs
    captured = manager._graphs[name][bucket_size]
    assert torch.equal(
        captured.static_inputs["num_valid_tokens"],
        torch.ones(1, dtype=torch.int32, device=device),
    )
    assert torch.equal(
        captured.static_inputs["cache_seqlens"],
        torch.tensor([1, 0, 0, 0], dtype=torch.int32, device=device),
    )
    assert torch.equal(
        captured.static_inputs["primary_slot_indices"],
        torch.tensor([0, -1, -1, -1], dtype=torch.int32, device=device),
    )

    primary_cache.zero_()
    aux_cache.zero_()
    graph_outputs = manager.replay(
        name,
        actual_bsz,
        hidden_states=hidden,
        position_ids=position_ids,
        cache_seqlens=cache_seqlens,
        primary_slot_indices=primary_slots,
        aux_slot_indices=aux_slots,
        flashmla_tile_scheduler_metadata=metadata,
        flashmla_num_splits=num_splits,
    )
    torch.cuda.synchronize()
    graph_primary_cache = primary_cache.detach().clone()
    graph_aux_cache = aux_cache.detach().clone()

    for key in ("attn_output", "primary_k_tensor", "indexer_k_tensor"):
        assert torch.equal(graph_outputs[key], eager_outputs[key]), key
    assert torch.equal(graph_primary_cache, eager_primary_cache)
    assert torch.equal(graph_aux_cache, eager_aux_cache)
    assert torch.count_nonzero(graph_primary_cache[2:]).item() == 0
    assert torch.count_nonzero(graph_aux_cache[:2]).item() == 0
    buffers = shared_buffers[bucket_size]
    static_outputs = segment._outputs[bucket_size]
    expected_safe_slots = torch.tensor([0, 1, 0, 0], dtype=torch.int32, device=device)
    expected_kv_slots = torch.tensor([0, 1, -1, -1], dtype=torch.int32, device=device)
    expected_safe_seqlens = torch.tensor([2, 3, 0, 0], dtype=torch.int32, device=device)
    assert torch.equal(buffers.safe_primary_slot_indices, expected_safe_slots)
    assert torch.equal(buffers.safe_aux_slot_indices, expected_safe_slots)
    assert torch.equal(buffers.kv_primary_slot_indices, expected_kv_slots)
    assert torch.equal(buffers.kv_aux_slot_indices, expected_kv_slots)
    assert torch.equal(buffers.safe_cache_seqlens, expected_safe_seqlens)
    assert torch.equal(captured.static_inputs["num_valid_tokens"], num_valid_tokens)
    assert torch.all(buffers.selected_token_ids[actual_bsz:] == -1)
    assert torch.count_nonzero(buffers.selected_lengths[actual_bsz:]).item() == 0
    assert torch.count_nonzero(buffers.attn_heads[actual_bsz:]).item() == 0
    assert torch.count_nonzero(static_outputs.attn_output[actual_bsz:]).item() == 0

    manager.drop_bucket(bucket_size)
    assert bucket_size not in shared_buffers
    assert bucket_size not in segment._outputs


def test_glm5_full_dsa_all_short_skips_query_score_but_writes_indexer_k(monkeypatch):
    device = torch.device("cuda")
    bucket_size = 2
    page_size = 4
    wrapper = _build_fake_wrapper(device)
    attn = wrapper.module
    kv_dim = attn.kv_lora_rank + attn.qk_rope_head_dim
    index_dim = attn.indexer.index_head_dim
    calls = _patch_full_dsa_dependencies(
        monkeypatch,
        bucket_size=bucket_size,
        index_topk=attn.indexer.index_topk,
        kv_dim=kv_dim,
        v_dim=attn.v_head_dim,
        device=device,
    )
    primary_cache = torch.zeros(2, page_size, 1, kv_dim, dtype=torch.bfloat16, device=device)
    aux_cache = torch.zeros(2, page_size, 1, index_dim, dtype=torch.bfloat16, device=device)
    page_table = torch.tensor([[0], [1]], dtype=torch.int32, device=device)
    cos = torch.ones(attn.indexer.index_topk, attn.qk_rope_head_dim, dtype=torch.bfloat16, device=device)
    sin = torch.zeros_like(cos)
    segment = Glm5FullDsaAttnSegment(
        wrapper=wrapper,
        primary_blocked_k=primary_cache,
        aux_blocked_k=aux_cache,
        primary_page_table=page_table,
        aux_page_table=page_table,
        wq_b_weights=object(),
        absorb_weights=object(),
        cuda_module=object(),
        cos_table=cos,
        sin_table=sin,
        max_seqlen=attn.indexer.index_topk,
        index_topk=attn.indexer.index_topk,
        page_size=page_size,
        aux_page_size=page_size,
        all_short=True,
    )

    outputs = segment.forward(
        hidden_states=torch.randn(bucket_size, 1, attn.hidden_size, dtype=torch.bfloat16, device=device),
        position_ids=torch.tensor([[1], [2]], dtype=torch.int64, device=device),
        cache_seqlens=torch.tensor([2, 3], dtype=torch.int32, device=device),
        primary_slot_indices=torch.tensor([0, 1], dtype=torch.int32, device=device),
        aux_slot_indices=torch.tensor([0, 1], dtype=torch.int32, device=device),
        num_valid_tokens=torch.tensor([bucket_size], dtype=torch.int32, device=device),
        flashmla_tile_scheduler_metadata=torch.arange(4, dtype=torch.int32, device=device).view(1, 4),
        flashmla_num_splits=torch.ones(1, dtype=torch.int32, device=device),
    )
    torch.cuda.synchronize()

    assert calls == {
        "head_gates": 0,
        "wq_b": 0,
        "rope_hadamard_q": 0,
        "score_topk": 0,
        "transform": 0,
        "select": 0,
        "flashmla": 0,
        "fa3": 1,
    }
    assert torch.count_nonzero(outputs["indexer_k_tensor"]).item() > 0
    assert torch.count_nonzero(aux_cache).item() > 0
