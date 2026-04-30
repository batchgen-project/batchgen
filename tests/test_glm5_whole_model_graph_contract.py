import torch
import pytest

from batchgen.models.glm.glm5.whole_model_cuda_graph_segments import (
    Glm5WholeModelSegment,
    compare_glm5_whole_model_graph_logits,
    make_glm5_whole_model_graph_segment_name,
)


class _FakeIndexer:
    index_head_dim = 128


class _FakeAttnModule:
    kv_lora_rank = 512
    qk_rope_head_dim = 64
    indexer = _FakeIndexer()


class _FakeSelfAttn:
    module = _FakeAttnModule()


class _FakeLayer:
    self_attn = _FakeSelfAttn()


class _FakeInnerModel:
    layers = [_FakeLayer(), _FakeLayer()]


class _FakeModel:
    model = _FakeInnerModel()


def _make_segment(**kwargs):
    defaults = dict(
        model=_FakeModel(),
        device=torch.device("cpu"),
        world_size=16,
        max_pages_per_seq=128,
        max_aux_pages_per_seq=64,
        vocab_size=151552,
        hidden_size=6144,
        max_bucket_size=2,
        max_seqlen=8192,
    )
    defaults.update(kwargs)
    return Glm5WholeModelSegment(**defaults)


def test_glm5_whole_model_segment_static_contract():
    segment = _make_segment()

    assert make_glm5_whole_model_graph_segment_name() == "glm5_whole_model"
    assert segment.primary_kv_dim == 576
    assert segment.aux_kv_dim == 128

    inputs = segment.get_static_input_specs(bucket_size=2)
    assert inputs["input_ids"].resolve_shape(2) == (2, 1)
    assert inputs["cache_seqlens"].resolve_shape(2) == (2,)
    assert inputs["position_ids"].resolve_shape(2) == (2, 1)
    assert inputs["primary_page_table"].resolve_shape(2) == (2, 128)
    assert inputs["aux_page_table"].resolve_shape(2) == (2, 64)
    assert inputs["primary_slot_indices"].fill_value == -1
    assert inputs["aux_slot_indices"].fill_value == -1
    assert inputs["rank_token_counts"].resolve_shape(2) == (16,)
    assert inputs["num_valid_tokens"].resolve_shape(2) == (1,)

    outputs = segment.get_static_output_specs(bucket_size=2)
    assert outputs["logits"].resolve_shape(2) == (2, 151552)


def test_glm5_whole_model_segment_allocates_primary_and_aux_offload_buffers():
    segment = _make_segment(max_bucket_size=4)

    segment.setup_static_buffers(bucket_size=2)

    assert segment._kv_buffers is not None
    assert segment._aux_kv_buffers is not None
    assert len(segment._kv_buffers) == 2
    assert len(segment._aux_kv_buffers) == 2
    assert segment._kv_buffers[0]["key"].shape == (4, 1, 1, 576)
    assert segment._aux_kv_buffers[0]["key"].shape == (4, 1, 1, 128)
    assert segment._no_v_cache


def test_glm5_whole_model_segment_rejects_hidden_state_boundary_until_hardened():
    with pytest.raises(NotImplementedError, match="input_ids -> embedding"):
        _make_segment(include_embedding=False)


def test_glm5_whole_model_segment_rejects_hidden_output_until_hardened():
    with pytest.raises(NotImplementedError, match="returns logits"):
        _make_segment(include_lm_head=False)


def test_glm5_whole_model_compare_reports_match_and_token_mismatch():
    eager = torch.tensor([[1.0, 3.0], [4.0, 2.0]], dtype=torch.float32)
    graph = eager + torch.tensor([[0.0, 0.001], [0.001, 0.0]], dtype=torch.float32)

    match = compare_glm5_whole_model_graph_logits(
        eager_logits=eager,
        graph_logits=graph,
        eager_tokens=torch.tensor([[1], [0]]),
        graph_tokens=torch.tensor([[1], [0]]),
        atol=1e-2,
        rtol=1e-2,
    )

    assert match["ok"]
    assert match["shape_match"]
    assert match["argmax_mismatch"] == 0
    assert match["token_mismatch"] == 0

    mismatch = compare_glm5_whole_model_graph_logits(
        eager_logits=eager,
        graph_logits=graph,
        eager_tokens=torch.tensor([[1], [0]]),
        graph_tokens=torch.tensor([[0], [0]]),
        atol=1e-2,
        rtol=1e-2,
    )

    assert not mismatch["ok"]
    assert mismatch["token_mismatch"] == 1


def test_glm5_whole_model_compare_reports_shape_mismatch():
    result = compare_glm5_whole_model_graph_logits(
        eager_logits=torch.zeros(2, 3),
        graph_logits=torch.zeros(2, 4),
    )

    assert not result["ok"]
    assert not result["shape_match"]
    assert result["eager_shape"] == (2, 3)
    assert result["graph_shape"] == (2, 4)
