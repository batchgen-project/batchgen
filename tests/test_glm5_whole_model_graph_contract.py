import torch
import pytest

from batchgen.models.glm.glm5.layer_cuda_graph_segments import (
    Glm5DecoderLayerGraphSegment,
    make_glm5_layer_graph_segment_name,
)
from batchgen.models.glm.glm5.whole_model_cuda_graph_segments import (
    Glm5WholeModelSegment,
    compare_glm5_whole_model_graph_logits,
    glm5_whole_model_layer_chunks,
    make_glm5_whole_model_graph_chunk_name,
    make_glm5_whole_model_graph_segment_name,
)


class _FakeIndexer:
    index_head_dim = 128
    index_topk = 2048


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
    assert segment.chunk_segments == []

    inputs = segment.get_static_input_specs(bucket_size=2)
    assert inputs["input_ids"].resolve_shape(2) == (2, 1)
    assert inputs["cache_seqlens"].resolve_shape(2) == (2,)
    assert inputs["position_ids"].resolve_shape(2) == (2, 1)
    assert "primary_page_table" not in inputs
    assert "aux_page_table" not in inputs
    assert inputs["primary_slot_indices"].fill_value == -1
    assert inputs["aux_slot_indices"].fill_value == -1
    assert inputs["rank_token_counts"].resolve_shape(2) == (16,)
    assert "num_valid_tokens" not in inputs

    outputs = segment.get_static_output_specs(bucket_size=2)
    assert outputs["hidden_states"].resolve_shape(2) == (2, 6144)
    assert outputs["logits"].resolve_shape(2) == (2, 151552)


def test_glm5_whole_model_layer_chunks_stay_below_production_moe_limit():
    chunks = glm5_whole_model_layer_chunks(78)

    assert len(chunks) == 20
    assert chunks[:2] == ((0, 4), (4, 8))
    assert chunks[-1] == (76, 78)
    assert all(layer_end - layer_start <= 4 for layer_start, layer_end in chunks)
    assert [layer_idx for start, end in chunks for layer_idx in range(start, end)] == list(
        range(78)
    )
    assert make_glm5_whole_model_graph_chunk_name(1, 4, 8) == (
        "glm5_whole_model_chunk_01_layers_04_08"
    )


def test_glm5_chunk_input_and_output_boundaries():
    class _FakeInnerModel21:
        layers = [_FakeLayer() for _ in range(21)]

    class _FakeModel21:
        model = _FakeInnerModel21()

    layer_segments = [_FakeDsaSegment() for _ in range(21)]
    segment = _make_segment(
        model=_FakeModel21(),
        layer_segments=layer_segments,
    )
    first, *_, last = segment.chunk_segments

    first_inputs = first.get_static_input_specs(bucket_size=2)
    last_inputs = last.get_static_input_specs(bucket_size=2)
    assert "input_ids" in first_inputs
    assert "hidden_states" not in first_inputs
    assert "input_ids" not in last_inputs
    assert last_inputs["hidden_states"].resolve_shape(2) == (2, 1, 6144)

    first_outputs = first.get_static_output_specs(bucket_size=2)
    last_outputs = last.get_static_output_specs(bucket_size=2)
    assert first_outputs["hidden_states"].resolve_shape(2) == (2, 1, 6144)
    assert "logits" not in first_outputs
    assert last_outputs["hidden_states"].resolve_shape(2) == (2, 6144)
    assert last_outputs["logits"].resolve_shape(2) == (2, 151552)


def test_glm5_whole_model_segment_probe_output_contract():
    segment = _make_segment(compare_probe_layers=(0, 1))

    outputs = segment.get_static_output_specs(bucket_size=2)

    assert outputs["probe_layer_000_hidden"].resolve_shape(2) == (2, 6144)
    assert outputs["probe_layer_001_hidden"].resolve_shape(2) == (2, 6144)


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


def test_glm5_whole_model_segment_accepts_padded_capture_inputs():
    segment = _make_segment(max_bucket_size=2)
    specs = segment.get_static_input_specs(bucket_size=2)
    static_inputs = {
        name: torch.full(
            spec.resolve_shape(2),
            spec.fill_value,
            dtype=spec.dtype,
            device="cpu",
        )
        for name, spec in specs.items()
    }
    segment.set_capture_inputs(
        input_ids=torch.tensor([[7], [0]], dtype=torch.int64),
        cache_seqlens=torch.tensor([128, 1], dtype=torch.int32),
        position_ids=torch.tensor([[127], [0]], dtype=torch.int64),
        primary_slot_indices=torch.tensor([3, -1], dtype=torch.int32),
        aux_slot_indices=torch.tensor([3, -1], dtype=torch.int32),
        rank_token_counts=torch.tensor([1] + [0] * 15, dtype=torch.int64),
    )

    segment.initialize_static_inputs(static_inputs, bucket_size=2)

    assert segment._capture_dsa_short_count == 1
    assert static_inputs["primary_slot_indices"].tolist() == [3, -1]
    assert static_inputs["cache_seqlens"].tolist() == [128, 1]


def test_glm5_whole_model_segment_materializes_bucket_from_real_rows():
    segment = _make_segment(max_bucket_size=4, max_seqlen=8192)
    specs = segment.get_static_input_specs(bucket_size=4)
    static_inputs = {
        name: torch.empty(spec.resolve_shape(4), dtype=spec.dtype, device="cpu")
        for name, spec in specs.items()
    }
    segment.set_capture_inputs(
        input_ids=torch.tensor([[7]], dtype=torch.int64),
        cache_seqlens=torch.tensor([128], dtype=torch.int32),
        position_ids=torch.tensor([[127]], dtype=torch.int64),
        primary_slot_indices=torch.tensor([3], dtype=torch.int32),
        aux_slot_indices=torch.tensor([4], dtype=torch.int32),
        rank_token_counts=torch.tensor([1, 2] + [0] * 14, dtype=torch.int64),
    )

    segment.initialize_static_inputs(static_inputs, bucket_size=4)

    assert static_inputs["input_ids"].tolist() == [[7], [0], [0], [0]]
    assert static_inputs["cache_seqlens"].tolist() == [128, 8192, 8192, 8192]
    assert static_inputs["position_ids"].tolist() == [[127], [0], [0], [0]]
    assert static_inputs["primary_slot_indices"].tolist() == [3, -1, -1, -1]
    assert static_inputs["aux_slot_indices"].tolist() == [4, -1, -1, -1]
    assert static_inputs["rank_token_counts"].tolist() == [1, 2] + [0] * 14
    assert segment._capture_dsa_short_count == 1


def test_glm5_whole_model_segment_rejects_capture_input_larger_than_bucket():
    segment = _make_segment(max_bucket_size=4)
    specs = segment.get_static_input_specs(bucket_size=2)
    static_inputs = {
        name: torch.empty(spec.resolve_shape(2), dtype=spec.dtype, device="cpu")
        for name, spec in specs.items()
    }
    segment.set_capture_inputs(
        input_ids=torch.tensor([[7], [8], [9]], dtype=torch.int64),
        cache_seqlens=torch.tensor([128, 129, 130], dtype=torch.int32),
        position_ids=torch.tensor([[127], [128], [129]], dtype=torch.int64),
        primary_slot_indices=torch.tensor([3, 4, 5], dtype=torch.int32),
        aux_slot_indices=torch.tensor([3, 4, 5], dtype=torch.int32),
        rank_token_counts=torch.tensor([3] + [0] * 15, dtype=torch.int64),
    )

    with pytest.raises(ValueError, match="batch dim 3 exceeds"):
        segment.initialize_static_inputs(static_inputs, bucket_size=2)


def test_glm5_whole_model_segment_uses_moe_bucket_resizer(monkeypatch):
    import types

    class _FakeGlm5MoE(torch.nn.Module):
        _rank_token_counts = None

    fake_model_module = types.ModuleType("batchgen.models.glm.glm5.model")
    fake_model_module.Glm5MoE = _FakeGlm5MoE
    monkeypatch.setitem(
        __import__("sys").modules,
        "batchgen.models.glm.glm5.model",
        fake_model_module,
    )

    mlp = _FakeGlm5MoE()
    resize_calls = []

    def _record_resize(bucket_size):
        resize_calls.append(bucket_size)
        mlp.num_tokens_per_rank = bucket_size

    mlp.set_num_tokens_per_rank = _record_resize

    class _FakeLayerWithMlp(_FakeLayer):
        def __init__(self, mlp):
            self.self_attn = _FakeSelfAttn()
            self.mlp = mlp

    class _FakeInnerModelWithMlp:
        def __init__(self, mlp):
            self.layers = [_FakeLayerWithMlp(mlp)]

    class _FakeModelWithMlp:
        def __init__(self, mlp):
            self.model = _FakeInnerModelWithMlp(mlp)

    segment = _make_segment(model=_FakeModelWithMlp(mlp), max_bucket_size=4)
    rank_counts = torch.tensor([3] + [0] * 15, dtype=torch.int64)

    segment._set_moe_bucket_state(3, rank_counts)

    assert resize_calls == [3]
    assert mlp.num_tokens_per_rank == 3
    assert _FakeGlm5MoE._rank_token_counts is rank_counts


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


class _IdentityModule(torch.nn.Module):
    def forward(self, x):
        return x


class _FakePostNorm:
    weight = torch.ones(4, dtype=torch.bfloat16)
    eps = 1e-5


class _FakeLayerForGraph:
    layer_idx = 7
    hidden_size = 4
    input_layernorm = _IdentityModule()
    post_attention_layernorm = _FakePostNorm()
    mlp = _IdentityModule()


class _FakeDsaSegment:
    max_seqlen = 16
    index_topk = 8
    primary_blocked_k = torch.empty(1, 1, 1, 6)
    aux_blocked_k = torch.empty(1, 1, 1, 3)

    def __init__(self):
        self.setup_calls = []
        self.init_calls = []
        self.release_calls = []

    def _flashmla_tensor_metadata_specs(self, bucket_size):
        return (bucket_size, 2), torch.int32, (1,), torch.int32

    def setup_static_buffers(self, bucket_size):
        self.setup_calls.append(bucket_size)

    def initialize_static_inputs(self, static_inputs, bucket_size):
        self.init_calls.append(bucket_size)
        static_inputs["num_valid_tokens"].fill_(1)

    def release_static_buffers(self, bucket_size):
        self.release_calls.append(bucket_size)


def test_glm5_layer_graph_segment_static_contract_and_delegation():
    dsa = _FakeDsaSegment()
    segment = Glm5DecoderLayerGraphSegment(
        layer=_FakeLayerForGraph(),
        dsa_segment=dsa,
        moe_segment=None,
        device=torch.device("cpu"),
        world_size=16,
    )

    assert make_glm5_layer_graph_segment_name(7) == "glm5_layer_7_full_layer"

    inputs = segment.get_static_input_specs(bucket_size=2)
    assert inputs["hidden_states"].resolve_shape(2) == (2, 1, 4)
    assert inputs["primary_slot_indices"].fill_value == -1
    assert inputs["aux_slot_indices"].fill_value == -1
    assert inputs["rank_token_counts"].resolve_shape(2) == (16,)
    assert inputs["flashmla_tile_scheduler_metadata"].resolve_shape(2) == (2, 2)

    outputs = segment.get_static_output_specs(bucket_size=2)
    assert outputs["hidden_states"].resolve_shape(2) == (2, 1, 4)
    assert outputs["primary_k_tensor"].resolve_shape(2) == (2, 1, 1, 6)
    assert outputs["indexer_k_tensor"].resolve_shape(2) == (2, 1, 1, 3)

    static_inputs = {
        name: torch.full(
            spec.resolve_shape(2),
            spec.fill_value,
            dtype=spec.dtype,
        )
        for name, spec in inputs.items()
    }
    segment.setup_static_buffers(bucket_size=2)
    segment.initialize_static_inputs(static_inputs, bucket_size=2)
    segment.release_static_buffers(bucket_size=2)

    assert dsa.setup_calls == [2]
    assert dsa.init_calls == [2]
    assert dsa.release_calls == [2]
    assert static_inputs["rank_token_counts"].tolist() == [1] * 16


def test_glm5_layer_graph_segment_empty_rank_capture_context():
    dsa = _FakeDsaSegment()
    rank_counts = torch.tensor([1, 1, 0, 0] + [0] * 12, dtype=torch.int64)
    segment = Glm5DecoderLayerGraphSegment(
        layer=_FakeLayerForGraph(),
        dsa_segment=dsa,
        moe_segment=None,
        device=torch.device("cpu"),
        world_size=16,
        capture_local_bsz=0,
        capture_rank_token_counts=rank_counts,
    )
    inputs = segment.get_static_input_specs(bucket_size=2)
    static_inputs = {
        name: torch.full(
            spec.resolve_shape(2),
            spec.fill_value,
            dtype=spec.dtype,
        )
        for name, spec in inputs.items()
    }

    segment.initialize_static_inputs(static_inputs, bucket_size=2)

    assert static_inputs["num_valid_tokens"].item() == 0
    assert static_inputs["rank_token_counts"].tolist() == rank_counts.tolist()
