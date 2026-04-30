import torch
import pytest

from batchgen.attention.dsa.sparse_gather import sparse_gather_from_paged_kv
from batchgen.models.glm.glm5.decode_utils import (
    build_flat_paged_gather_indices,
    build_batch_slot_indices,
    build_paged_gather_cache_key,
    build_clamped_dense_token_indices,
    clamp_token_indices_to_seqlens,
    reorder_block_table_to_batch_slots,
)
from batchgen.models.glm.glm5.cuda_graph_policy import (
    glm5_dsa_cuda_graph_requested_for_model,
    should_warmup_cuda_graphs_before_decode,
)
from batchgen.models.glm.glm5.wrappers import (
    GLM5AttnWrapper,
    _glm5_dsa_graph_compare_active,
    _glm5_dsa_graph_compare_layer_enabled,
    _glm5_dsa_cuda_graph_can_replay,
    _fail_if_glm5_dsa_cuda_graph_required_without_replay,
)
from batchgen.models.wrappers import AttnWrapperBase


def test_build_clamped_dense_token_indices_caps_each_row():
    cache_seqlens = torch.tensor([1, 65, 128], dtype=torch.int32)

    indices = build_clamped_dense_token_indices(
        cache_seqlens,
        max_seqlen=128,
        device=torch.device("cpu"),
    )

    assert indices.shape == (3, 128)
    assert indices[0, :6].tolist() == [0, 0, 0, 0, 0, 0]
    assert indices[1, 60:68].tolist() == [60, 61, 62, 63, 64, 64, 64, 64]
    assert indices[2, 124:128].tolist() == [124, 125, 126, 127]
    assert bool(
        (indices <= (cache_seqlens.to(torch.long) - 1).unsqueeze(-1)).all().item()
    )


def test_clamp_token_indices_to_seqlens_caps_topk_tail():
    indices = torch.tensor([[0, 1, 2, 9], [5, 8, 9, 10]], dtype=torch.long)
    cache_seqlens = torch.tensor([3, 9], dtype=torch.int32)

    clamped = clamp_token_indices_to_seqlens(indices, cache_seqlens)

    assert clamped.tolist() == [[0, 1, 2, 2], [5, 8, 8, 8]]


def test_clamped_dense_indices_prevent_stale_tail_reads():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    page_size = 64
    blocked_k = torch.zeros(4, page_size, 1, 1, dtype=torch.float32, device=device)

    blocked_k[0, :, 0, 0] = 1000 + torch.arange(page_size, device=device)
    blocked_k[1, 0, 0, 0] = 2000
    blocked_k[1, 1:, 0, 0] = 9000 + torch.arange(page_size - 1, device=device)
    blocked_k[2, :, 0, 0] = 3000 + torch.arange(page_size, device=device)
    blocked_k[3, :, 0, 0] = 4000 + torch.arange(page_size, device=device)

    block_table = torch.tensor([[0, 1, -1], [2, 3, -1]], dtype=torch.int64, device=device)
    cache_seqlens = torch.tensor([65, 128], dtype=torch.int32, device=device)

    clamped_indices = build_clamped_dense_token_indices(
        cache_seqlens,
        max_seqlen=128,
        device=device,
    )
    gathered = sparse_gather_from_paged_kv(
        blocked_k, block_table, clamped_indices, page_size
    ).squeeze(-1).squeeze(-1)

    assert gathered[0, 60:68].tolist() == [1060.0, 1061.0, 1062.0, 1063.0, 2000.0, 2000.0, 2000.0, 2000.0]
    assert not bool((gathered[0, 65:128] >= 9000).any().item())


def test_build_batch_slot_indices_uses_explicit_slot_mapping():
    slots = build_batch_slot_indices(
        current_batch=[105, 101, 109],
        seq_id_to_slot={101: 0, 105: 2, 109: 1},
        batch_size=3,
        device=torch.device("cpu"),
    )

    assert slots.tolist() == [2, 0, 1]


def test_reordered_block_table_prevents_cross_sequence_reads():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    page_size = 4
    blocked_k = torch.zeros(2, page_size, 1, 1, dtype=torch.float32, device=device)
    blocked_k[0, :, 0, 0] = torch.tensor([100.0, 101.0, 102.0, 103.0], device=device)
    blocked_k[1, :, 0, 0] = torch.tensor([200.0, 201.0, 202.0, 203.0], device=device)

    slot_order_block_table = torch.tensor([[1, -1], [0, -1]], dtype=torch.int64, device=device)
    top_k_indices = torch.tensor([[0, 1], [0, 1]], dtype=torch.long, device=device)

    wrong = sparse_gather_from_paged_kv(
        blocked_k, slot_order_block_table, top_k_indices, page_size
    ).squeeze(-1).squeeze(-1)

    reordered = reorder_block_table_to_batch_slots(
        slot_order_block_table,
        torch.tensor([1, 0], dtype=torch.int32, device=device),
    )
    fixed = sparse_gather_from_paged_kv(
        blocked_k, reordered, top_k_indices, page_size
    ).squeeze(-1).squeeze(-1)

    assert wrong.tolist() == [[200.0, 201.0], [100.0, 101.0]]
    assert fixed.tolist() == [[100.0, 101.0], [200.0, 201.0]]


def test_glm5_dsa_selector_preserves_dense_short_circuit():
    pytest.importorskip("flash_attn_interface")
    from batchgen.attention.dsa.glm5_decode_selector import _select_glm5_dsa_indices

    class FakeIndexer:
        index_topk = 8

        def __init__(self):
            self.calls = []

        def score_and_select_paged(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            raise AssertionError("short rows must not run indexer scoring")

    class FakeManager:
        config = type("Config", (), {"page_size_tokens": 4})()

        def get_layer_kv_with_page_table(self, layer_idx):
            raise AssertionError("short rows must not fetch aux page tables")

    indexer = FakeIndexer()
    wrapper = type(
        "Wrapper",
        (),
        {"module": type("Module", (), {"indexer": indexer})(), "layer_idx": 0},
    )()
    old_short_count = AttnWrapperBase._dsa_short_count
    AttnWrapperBase._dsa_short_count = None
    try:
        top_k, branch, row_modes = _select_glm5_dsa_indices(
            wrapper,
            hidden_states=torch.zeros(3, 1, 4),
            q_a_normed=torch.zeros(3, 4),
            cache_seqlens=torch.tensor([2, 4, 6], dtype=torch.int32),
            max_seqlen=128,
            new_token_pos=torch.tensor([1, 3, 5], dtype=torch.int64),
            gpu_paged_kv_manager_aux=FakeManager(),
            aux_slot_indices=torch.tensor([0, 1, 2], dtype=torch.int32),
        )
    finally:
        AttnWrapperBase._dsa_short_count = old_short_count

    assert branch == "dense-short-circuit"
    assert row_modes.tolist() == [0, 0, 0]
    assert top_k.tolist() == [
        [0, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 2, 3, 3, 3, 3, 3],
        [0, 1, 2, 3, 4, 5, 5, 5],
    ]
    assert indexer.calls == []


def test_glm5_dsa_selector_scores_only_long_rows_in_mixed_batch():
    pytest.importorskip("flash_attn_interface")
    from batchgen.attention.dsa.glm5_decode_selector import _select_glm5_dsa_indices

    class FakeIndexer:
        index_topk = 8

        def __init__(self):
            self.seen = None

        def score_and_select_paged(
            self,
            q_a,
            hidden_states,
            indexer_blocked_k,
            idx_block_table,
            cache_seqlens,
            manager,
            page_size,
            *,
            positions,
            max_seqlen,
        ):
            self.seen = {
                "q_a_shape": tuple(q_a.shape),
                "hidden_shape": tuple(hidden_states.shape),
                "block_table": idx_block_table.clone(),
                "cache_seqlens": cache_seqlens.clone(),
                "positions": positions.clone(),
                "page_size": page_size,
                "max_seqlen": max_seqlen,
            }
            return torch.tensor(
                [
                    [10, 11, 12, 13, 14, 15, 16, 17],
                    [20, 21, 22, 23, 24, 25, 26, 27],
                ],
                dtype=torch.long,
            )

    class FakeManager:
        config = type("Config", (), {"page_size_tokens": 4})()

        def get_layer_kv_with_page_table(self, layer_idx):
            blocked_k = torch.empty(1)
            block_table = torch.tensor(
                [
                    [100, 101, 102],
                    [200, 201, 202],
                    [300, 301, 302],
                    [400, 401, 402],
                ],
                dtype=torch.int32,
            )
            return blocked_k, None, block_table

    indexer = FakeIndexer()
    wrapper = type(
        "Wrapper",
        (),
        {"module": type("Module", (), {"indexer": indexer})(), "layer_idx": 0},
    )()
    old_short_count = AttnWrapperBase._dsa_short_count
    AttnWrapperBase._dsa_short_count = 2
    try:
        top_k, branch, row_modes = _select_glm5_dsa_indices(
            wrapper,
            hidden_states=torch.zeros(4, 1, 4),
            q_a_normed=torch.zeros(4, 4),
            cache_seqlens=torch.tensor([3, 10, 5, 12], dtype=torch.int32),
            max_seqlen=128,
            new_token_pos=torch.tensor([2, 9, 4, 11], dtype=torch.int64),
            gpu_paged_kv_manager_aux=FakeManager(),
            aux_slot_indices=torch.tensor([3, 1, 2, 0], dtype=torch.int32),
        )
    finally:
        AttnWrapperBase._dsa_short_count = old_short_count

    assert branch == "mixed"
    assert row_modes.tolist() == [0, 1, 0, 1]
    assert top_k.tolist() == [
        [0, 1, 2, 2, 2, 2, 2, 2],
        [10, 11, 12, 13, 14, 15, 16, 17],
        [0, 1, 2, 3, 4, 4, 4, 4],
        [20, 21, 22, 23, 24, 25, 26, 27],
    ]
    assert indexer.seen["q_a_shape"] == (2, 1, 4)
    assert indexer.seen["hidden_shape"] == (2, 1, 4)
    assert indexer.seen["cache_seqlens"].tolist() == [10, 12]
    assert indexer.seen["positions"].tolist() == [9, 11]
    assert indexer.seen["max_seqlen"] == 12
    assert indexer.seen["block_table"].tolist() == [
        [200, 201, 202],
        [100, 101, 102],
    ]


def test_paged_gather_cache_key_invalidates_in_place_page_table_rebuild():
    block_table = torch.tensor([[0, 1], [2, 3]], dtype=torch.int64)
    original_ptr = block_table.data_ptr()

    key_v1 = build_paged_gather_cache_key(
        block_table,
        max_seqlen=4,
        page_size=2,
        page_table_version=1,
    )
    flat_v1 = build_flat_paged_gather_indices(
        block_table,
        max_seqlen=4,
        page_size=2,
    )

    block_table[:, :] = torch.tensor([[2, 3], [0, 1]], dtype=torch.int64)
    rebuilt_ptr = block_table.data_ptr()
    flat_v2 = build_flat_paged_gather_indices(
        block_table,
        max_seqlen=4,
        page_size=2,
    )
    key_v2 = build_paged_gather_cache_key(
        block_table,
        max_seqlen=4,
        page_size=2,
        page_table_version=2,
    )

    assert rebuilt_ptr == original_ptr
    assert key_v1 != key_v2
    assert flat_v1.tolist() == [0, 1, 2, 3, 4, 5, 6, 7]
    assert flat_v2.tolist() == [4, 5, 6, 7, 0, 1, 2, 3]


def test_glm5_dsa_cuda_graph_required_fast_fails_without_replay(monkeypatch):
    monkeypatch.setenv("BATCHGEN_GLM5_DSA_CUDA_GRAPH", "1")

    with pytest.raises(RuntimeError, match="Refusing to silently fall back"):
        _fail_if_glm5_dsa_cuda_graph_required_without_replay()


def test_glm5_dsa_decode_routes_to_registered_graph_when_requested(monkeypatch):
    monkeypatch.setenv("BATCHGEN_GLM5_DSA_CUDA_GRAPH", "1")
    wrapper = object.__new__(GLM5AttnWrapper)
    wrapper.module = type(
        "Attn",
        (),
        {"hidden_size": 16, "indexer": type("Indexer", (), {"index_topk": 4})()},
    )()
    wrapper.layer_idx = 0
    wrapper._dsa_cuda_graph_max_seqlen = 4096
    wrapper._dsa_cuda_graph_segment_name = "glm5_layer_0_dsa_attn"
    wrapper._dsa_cuda_graph_manager = type(
        "GraphManager",
        (),
        {"has_graph": lambda self, name, batch_size: name == "glm5_layer_0_dsa_attn" and batch_size == 2},
    )()
    expected = torch.ones(2, 1, 16)

    def fake_graph_route(self, hidden_states, position_ids, cache_seqlens, max_seqlen, primary, aux):
        assert hidden_states.shape == (2, 1, 16)
        assert position_ids.dtype == torch.int64
        assert cache_seqlens.dtype == torch.int32
        assert max_seqlen == 4096
        assert primary == "primary"
        assert aux == "aux"
        return expected

    monkeypatch.setattr(GLM5AttnWrapper, "_forward_decode_dsa_graph", fake_graph_route)

    actual = wrapper._forward_decode_dsa(
        torch.zeros(2, 1, 16),
        torch.tensor([[7], [8]], dtype=torch.int64),
        torch.tensor([8, 9], dtype=torch.int32),
        4096,
        "primary",
        "aux",
    )

    assert actual is expected


def test_glm5_dsa_graph_compare_returns_eager_and_runs_side_channel(monkeypatch):
    wrapper = object.__new__(GLM5AttnWrapper)
    wrapper.module = type(
        "Attn",
        (),
        {"hidden_size": 16, "indexer": type("Indexer", (), {"index_topk": 4})()},
    )()
    wrapper.layer_idx = 0
    wrapper._dsa_cuda_graph_max_seqlen = 4096
    wrapper._dsa_cuda_graph_segment_name = "glm5_layer_0_dsa_attn"
    wrapper._dsa_cuda_graph_manager = type(
        "GraphManager",
        (),
        {"has_graph": lambda self, name, batch_size: name == "glm5_layer_0_dsa_attn" and batch_size == 2},
    )()
    expected = torch.ones(2, 1, 16)
    calls = {}

    def fake_eager(self, *args, return_debug=False, **kwargs):
        calls["return_debug"] = return_debug
        assert return_debug
        return expected, {"selector_inputs": None, "attn_heads": None}

    def fake_graph(self, *args, **kwargs):
        raise AssertionError("compare mode must not return graph output")

    def fake_compare(self, *args, eager_output, eager_debug, **kwargs):
        calls["compare"] = True
        assert eager_output is expected
        assert eager_debug["selector_inputs"] is None

    monkeypatch.setattr(GLM5AttnWrapper, "_forward_decode_dsa_eager", fake_eager)
    monkeypatch.setattr(GLM5AttnWrapper, "_forward_decode_dsa_graph", fake_graph)
    monkeypatch.setattr(GLM5AttnWrapper, "_forward_decode_dsa_graph_compare", fake_compare)

    old_debug = AttnWrapperBase.batchgen_debug
    AttnWrapperBase.batchgen_debug = {"glm5_dsa_graph_compare": True}
    try:
        actual = wrapper._forward_decode_dsa(
            torch.zeros(2, 1, 16),
            torch.tensor([[7], [8]], dtype=torch.int64),
            torch.tensor([8, 9], dtype=torch.int32),
            4096,
            "primary",
            "aux",
        )
    finally:
        AttnWrapperBase.batchgen_debug = old_debug

    assert actual is expected
    assert calls == {"return_debug": True, "compare": True}


def test_glm5_dsa_graph_compare_layer_filter(monkeypatch):
    old_debug = AttnWrapperBase.batchgen_debug
    AttnWrapperBase.batchgen_debug = {
        "glm5_dsa_graph_compare": True,
        "glm5_dsa_graph_compare_layers": [1, 3],
    }
    try:
        assert _glm5_dsa_graph_compare_active()
        assert not _glm5_dsa_graph_compare_layer_enabled(0)
        assert _glm5_dsa_graph_compare_layer_enabled(1)
        assert _glm5_dsa_graph_compare_layer_enabled(3)
    finally:
        AttnWrapperBase.batchgen_debug = old_debug


def test_glm5_dsa_cuda_graph_replay_gate_requires_all_long_rows():
    index_topk = 4

    assert _glm5_dsa_cuda_graph_can_replay(
        torch.tensor([5, 7], dtype=torch.int32),
        max_seqlen=8,
        index_topk=index_topk,
    )
    assert not _glm5_dsa_cuda_graph_can_replay(
        torch.tensor([4, 7], dtype=torch.int32),
        max_seqlen=8,
        index_topk=index_topk,
    )
    assert not _glm5_dsa_cuda_graph_can_replay(
        torch.tensor([5, 7], dtype=torch.int32),
        max_seqlen=4,
        index_topk=index_topk,
    )
    assert not _glm5_dsa_cuda_graph_can_replay(
        torch.tensor([5, 7], dtype=torch.int32),
        max_seqlen=8,
        index_topk=index_topk,
        captured_max_seqlen=6,
    )


def test_glm5_dsa_warmup_policy_allows_capture_with_queued_prefill():
    env = {"BATCHGEN_GLM5_DSA_CUDA_GRAPH": "1"}

    assert glm5_dsa_cuda_graph_requested_for_model("zai-org/GLM-5-FP8", environ=env)
    assert should_warmup_cuda_graphs_before_decode(
        graph_manager_is_initialized=False,
        global_batch_has_queueing=True,
        model_name="zai-org/GLM-5-FP8",
        environ=env,
    )
    assert not should_warmup_cuda_graphs_before_decode(
        graph_manager_is_initialized=True,
        global_batch_has_queueing=True,
        model_name="zai-org/GLM-5-FP8",
        environ=env,
    )
    assert not should_warmup_cuda_graphs_before_decode(
        graph_manager_is_initialized=False,
        global_batch_has_queueing=True,
        model_name="gpt-oss-120b",
        environ=env,
    )
    assert should_warmup_cuda_graphs_before_decode(
        graph_manager_is_initialized=False,
        global_batch_has_queueing=False,
        model_name="gpt-oss-120b",
        environ={},
    )


def test_glm5_dsa_graph_route_fast_fails_without_registered_segment(monkeypatch):
    monkeypatch.setenv("BATCHGEN_GLM5_DSA_CUDA_GRAPH", "1")
    wrapper = object.__new__(GLM5AttnWrapper)
    wrapper.layer_idx = 0
    wrapper._dsa_cuda_graph_manager = None
    wrapper._dsa_cuda_graph_segment_name = None

    with pytest.raises(RuntimeError, match="no registered DSA CUDA graph segment"):
        wrapper._forward_decode_dsa_graph(
            torch.zeros(1, 1, 16),
            torch.tensor([[1]], dtype=torch.int64),
            torch.tensor([2], dtype=torch.int32),
            128,
            object(),
            object(),
        )


def test_glm5_prefill_indexer_kv_uses_legacy_dynamic_max_seqlen(monkeypatch):
    wrapper = object.__new__(GLM5AttnWrapper)
    wrapper.layer_idx = 0
    wrapper.prepack_mode = True
    wrapper.position_ids = torch.tensor([[0, 1, 2]], dtype=torch.int64)
    wrapper.prepack_cu_seqlens = torch.tensor([0, 3], dtype=torch.int32)
    wrapper.prepack_max_seqlen = 4096
    wrapper.prepack_num_sequences = 1
    wrapper.weight_dequant_scale = None

    class FakeIndexer:
        def compute_indexer_kv(self, hidden_states, *, positions, max_seqlen=None):
            assert hidden_states.shape == (1, 3, 4)
            assert positions.tolist() == [[0, 1, 2]]
            assert max_seqlen is None
            return torch.zeros(1, 3, 1, 128)

    class FakeModule:
        indexer = FakeIndexer()

        def prefill_attn_w8a16_prepacked(
            self,
            hidden_states_2d,
            position_ids,
            prepack_cu_seqlens,
            prepack_max_seqlen,
            prepack_num_sequences,
            weight_dequant_scale,
        ):
            assert prepack_max_seqlen == 4096
            return torch.zeros_like(hidden_states_2d), torch.zeros(3, 1, 576)

    wrapper.module = FakeModule()
    monkeypatch.setattr(GLM5AttnWrapper, "_offload_prepacked_indexer_kv", lambda self, kv: None)
    monkeypatch.setattr(GLM5AttnWrapper, "_offload_prepacked_kv", lambda self, kv: None)

    attn_output, _, _ = wrapper._forward_prefill(torch.ones(1, 3, 4))

    assert attn_output.shape == (1, 3, 4)


def test_glm5_prefill_requires_indexer_and_prepack_mode():
    wrapper = object.__new__(GLM5AttnWrapper)
    wrapper.layer_idx = 0
    wrapper.prepack_mode = True
    wrapper.position_ids = torch.tensor([[0, 1]], dtype=torch.int64)
    wrapper.prepack_cu_seqlens = torch.tensor([0, 2], dtype=torch.int32)
    wrapper.prepack_max_seqlen = 2
    wrapper.prepack_num_sequences = 1
    wrapper.weight_dequant_scale = None

    class NoIndexerModule:
        def prefill_attn_w8a16_prepacked(
            self,
            hidden_states_2d,
            position_ids,
            prepack_cu_seqlens,
            prepack_max_seqlen,
            prepack_num_sequences,
            weight_dequant_scale,
        ):
            return torch.zeros_like(hidden_states_2d), torch.zeros(2, 1, 576)

    wrapper.module = NoIndexerModule()

    with pytest.raises(RuntimeError, match="requires indexer KV"):
        wrapper._forward_prefill(torch.ones(1, 2, 4))

    wrapper.prepack_mode = False
    with pytest.raises(RuntimeError, match="requires prepack_mode"):
        wrapper._forward_prefill(torch.ones(1, 2, 4))


def test_glm5_prefill_indexer_offload_requires_aux_host_view(monkeypatch):
    wrapper = object.__new__(GLM5AttnWrapper)
    monkeypatch.setattr(AttnWrapperBase, "host_paged_kv_worker_view_aux", None)

    with pytest.raises(RuntimeError, match="auxiliary host KV worker view is required"):
        wrapper._offload_prepacked_indexer_kv(torch.zeros(2, 1, 128))


def test_prefill_offload_lifetime_retires_previous_layer(monkeypatch):
    class DummyTask:
        def __init__(self):
            self.waited = False

        def wait(self):
            self.waited = True

    task = DummyTask()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(AttnWrapperBase, "pending_prefill_offload_tasks", [task])
    monkeypatch.setattr(AttnWrapperBase, "pending_prefill_offload_tensors", [torch.zeros(1)])
    monkeypatch.setattr(AttnWrapperBase, "pending_prefill_offload_layer_idx", 3)

    retired = AttnWrapperBase.retire_pending_prefill_offloads_before_layer(
        4,
        device=torch.device("cpu"),
    )

    assert retired == 1
    assert task.waited
    assert AttnWrapperBase.pending_prefill_offload_tasks == []
    assert AttnWrapperBase.pending_prefill_offload_tensors == []
    assert AttnWrapperBase.pending_prefill_offload_layer_idx is None


def test_prefill_offload_lifetime_keeps_current_layer_refs(monkeypatch):
    class DummyTask:
        def __init__(self):
            self.waited = False

        def wait(self):
            self.waited = True

    task = DummyTask()
    tensor = torch.zeros(1)
    monkeypatch.setattr(AttnWrapperBase, "pending_prefill_offload_tasks", [task])
    monkeypatch.setattr(AttnWrapperBase, "pending_prefill_offload_tensors", [tensor])
    monkeypatch.setattr(AttnWrapperBase, "pending_prefill_offload_layer_idx", 4)

    retired = AttnWrapperBase.retire_pending_prefill_offloads_before_layer(
        4,
        device=torch.device("cpu"),
    )

    assert retired == 0
    assert not task.waited
    assert AttnWrapperBase.pending_prefill_offload_tasks == [task]
    assert AttnWrapperBase.pending_prefill_offload_tensors == [tensor]
    assert AttnWrapperBase.pending_prefill_offload_layer_idx == 4
