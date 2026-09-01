import os
import sys
import types

import torch
import pytest

from batchgen.attention.dsa.sparse_gather import sparse_gather_from_paged_kv
from batchgen.cuda_graph import BatchSizeBucketing
from batchgen.models.glm.glm5.decode_utils import (
    build_flat_paged_gather_indices,
    build_batch_slot_indices,
    build_paged_gather_cache_key,
    build_clamped_dense_token_indices,
    clamp_token_indices_to_seqlens,
    reorder_block_table_to_batch_slots,
)
from batchgen.models.glm.glm5.cuda_graph_policy import (
    GLM5_POWER_OF_TWO_BUCKETS_32,
    glm5_any_cuda_graph_requested_for_model,
    glm5_cuda_graph_bucket_for_batch_size,
    glm5_dsa_cuda_graph_requested_for_model,
    glm5_effective_decode_attn_mode,
    glm5_moe_graph_bucket_capacity,
    glm5_moe_cuda_graph_requested_for_model,
    glm5_segmented_cuda_graph_requested_for_model,
    glm5_whole_model_cuda_graph_compare_requested_for_model,
    glm5_whole_model_cuda_graph_requested_for_model,
    should_warmup_cuda_graphs_before_decode,
)
from batchgen.models.glm.glm5.model import (
    Glm5MoE,
    _glm5_moe_3d_blockwise_supported,
    _glm5_moe_graph_compare_active,
    _glm5_moe_graph_compare_layer_enabled,
    _glm5_moe_router_mode,
)
import batchgen.models.glm.glm5.model as glm5_model
from batchgen.models.glm.glm5.wrappers import (
    GLM5AttnWrapper,
    _glm5_dsa_cuda_graph_required,
    _glm5_dsa_graph_compare_active,
    _glm5_dsa_graph_compare_layer_enabled,
    _glm5_dsa_cuda_graph_can_replay,
    _glm5_dsa_gpu_page_table_tensor,
    _glm5_dsa_page_table_signature,
    _fail_if_glm5_dsa_cuda_graph_required_without_replay,
)
from batchgen.models.wrappers import AttnWrapperBase
from batchgen.sequence import SequenceBatch, SequenceEntry


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


def test_glm5_moe_graph_compare_layer_selection(monkeypatch):
    monkeypatch.setattr(
        AttnWrapperBase,
        "batchgen_debug",
        {
            "glm5_moe_graph_compare": True,
            "glm5_moe_graph_compare_layers": "3,20,77",
        },
        raising=False,
    )
    monkeypatch.delenv("BATCHGEN_GLM5_MOE_GRAPH_COMPARE", raising=False)

    assert _glm5_moe_graph_compare_active()
    assert _glm5_moe_graph_compare_layer_enabled(3)
    assert _glm5_moe_graph_compare_layer_enabled(20)
    assert _glm5_moe_graph_compare_layer_enabled(77)
    assert not _glm5_moe_graph_compare_layer_enabled(4)


def test_glm5_moe_graph_compare_defaults_to_layer3(monkeypatch):
    monkeypatch.setattr(
        AttnWrapperBase,
        "batchgen_debug",
        {"glm5_moe_graph_compare": True},
        raising=False,
    )
    monkeypatch.delenv("BATCHGEN_GLM5_MOE_GRAPH_COMPARE_LAYERS", raising=False)

    assert _glm5_moe_graph_compare_layer_enabled(3)
    assert not _glm5_moe_graph_compare_layer_enabled(20)


def test_glm5_moe_router_mode_defaults_to_custom(monkeypatch):
    monkeypatch.setattr(AttnWrapperBase, "batchgen_debug", {}, raising=False)
    monkeypatch.delenv("BATCHGEN_GLM5_MOE_ROUTER_MODE", raising=False)

    assert _glm5_moe_router_mode() == "custom"


def test_glm5_moe_router_mode_batch_debug_overrides_env(monkeypatch):
    monkeypatch.setattr(
        AttnWrapperBase,
        "batchgen_debug",
        {"glm5_moe_router_mode": "cublas"},
        raising=False,
    )
    monkeypatch.setenv("BATCHGEN_GLM5_MOE_ROUTER_MODE", "custom")

    assert _glm5_moe_router_mode() == "cublas"


def test_glm5_moe_router_mode_rejects_unknown_values(monkeypatch):
    monkeypatch.setattr(
        AttnWrapperBase,
        "batchgen_debug",
        {"glm5_moe_router_mode": "not-a-router"},
        raising=False,
    )
    monkeypatch.setenv("BATCHGEN_GLM5_MOE_ROUTER_MODE", "cublas")

    assert _glm5_moe_router_mode() == "custom"


def test_glm5_moe_3d_blockwise_requires_all_persistent_experts():
    assert _glm5_moe_3d_blockwise_supported(
        experts_per_rank=16,
        num_persistent_local_experts=16,
        enable_ep_offloading=False,
    )
    assert not _glm5_moe_3d_blockwise_supported(
        experts_per_rank=32,
        num_persistent_local_experts=24,
        enable_ep_offloading=False,
    )
    assert not _glm5_moe_3d_blockwise_supported(
        experts_per_rank=32,
        num_persistent_local_experts=32,
        enable_ep_offloading=True,
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
    old_short_count = GLM5AttnWrapper._dsa_short_count
    GLM5AttnWrapper._dsa_short_count = None
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
        GLM5AttnWrapper._dsa_short_count = old_short_count

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
    old_short_count = GLM5AttnWrapper._dsa_short_count
    GLM5AttnWrapper._dsa_short_count = 2
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
        GLM5AttnWrapper._dsa_short_count = old_short_count

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

    class FakeBucketing:
        def get_padded_size(self, batch_size):
            assert batch_size == 2
            return 2

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
        {
            "bucketing": FakeBucketing(),
            "has_graph": lambda self, name, batch_size: name == "glm5_layer_0_dsa_attn" and batch_size == 2,
        },
    )()
    expected = torch.ones(2, 1, 16)
    monkeypatch.setattr(
        GLM5AttnWrapper,
        "glm5_dsa_graph_forward_state",
        {
            "path": "graph",
            "bucket": 2,
            "reason": "captured",
            "local_bsz": 2,
            "metadata_prepared": True,
        },
        raising=False,
    )
    monkeypatch.setattr(
        GLM5AttnWrapper,
        "glm5_dsa_flashmla_graph_metadata",
        {
            "bucket_size": 2,
            "tile_scheduler_metadata": torch.empty(1, dtype=torch.int32),
            "num_splits": torch.empty(1, dtype=torch.int32),
        },
        raising=False,
    )

    def fake_graph_route(self, hidden_states, position_ids, cache_seqlens, max_seqlen, primary, aux):
        assert hidden_states.shape == (2, 1, 16)
        assert position_ids.dtype == torch.int64
        assert cache_seqlens.dtype == torch.int32
        assert max_seqlen == 4096
        assert primary == "primary"
        assert aux == "aux"
        return expected

    monkeypatch.setattr(GLM5AttnWrapper, "_forward_decode_dsa_graph", fake_graph_route)
    monkeypatch.setattr(
        GLM5AttnWrapper,
        "_dsa_cuda_graph_page_tables_match",
        lambda self, primary, aux: True,
    )

    actual = wrapper._forward_decode_dsa(
        torch.zeros(2, 1, 16),
        torch.tensor([[7], [8]], dtype=torch.int64),
        torch.tensor([4, 9], dtype=torch.int32),
        4096,
        "primary",
        "aux",
    )

    assert actual is expected


def test_glm5_dsa_decode_does_not_replay_graph_without_forward_metadata(monkeypatch):
    monkeypatch.setenv("BATCHGEN_GLM5_DSA_CUDA_GRAPH", "1")

    class FakeBucketing:
        def get_padded_size(self, batch_size):
            assert batch_size == 2
            return 2

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
        {
            "bucketing": FakeBucketing(),
            "has_graph": lambda self, name, batch_size: name == "glm5_layer_0_dsa_attn" and batch_size == 2,
        },
    )()
    expected = torch.ones(2, 1, 16)
    calls = {}

    def fake_graph_route(self, *args, **kwargs):
        raise AssertionError("graph replay must not run without prepared metadata")

    def fake_eager(self, *args, **kwargs):
        calls["eager"] = True
        return expected

    monkeypatch.setattr(GLM5AttnWrapper, "_forward_decode_dsa_graph", fake_graph_route)
    monkeypatch.setattr(GLM5AttnWrapper, "_forward_decode_dsa_eager", fake_eager)
    monkeypatch.setattr(
        GLM5AttnWrapper,
        "_dsa_cuda_graph_page_tables_match",
        lambda self, primary, aux: True,
    )
    monkeypatch.setattr(
        GLM5AttnWrapper,
        "glm5_dsa_graph_forward_state",
        {
            "path": "eager",
            "bucket": 2,
            "reason": "primary_page_table_state_invalid",
            "local_bsz": 2,
            "metadata_prepared": False,
        },
        raising=False,
    )
    monkeypatch.setattr(
        GLM5AttnWrapper,
        "glm5_dsa_flashmla_graph_metadata",
        None,
        raising=False,
    )

    actual = wrapper._forward_decode_dsa(
        torch.zeros(2, 1, 16),
        torch.tensor([[7], [8]], dtype=torch.int64),
        torch.tensor([4, 9], dtype=torch.int32),
        4096,
        "primary",
        "aux",
    )

    assert actual is expected
    assert calls == {"eager": True}


def test_glm5_dsa_graph_compare_returns_eager_and_runs_side_channel(monkeypatch):
    class FakeBucketing:
        def get_padded_size(self, batch_size):
            assert batch_size == 2
            return 2

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
        {
            "bucketing": FakeBucketing(),
            "has_graph": lambda self, name, batch_size: name == "glm5_layer_0_dsa_attn" and batch_size == 2,
        },
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
    monkeypatch.setattr(
        GLM5AttnWrapper,
        "_dsa_cuda_graph_page_tables_match",
        lambda self, primary, aux: True,
    )

    old_debug = AttnWrapperBase.batchgen_debug
    AttnWrapperBase.batchgen_debug = {"glm5_dsa_graph_compare": True}
    GLM5AttnWrapper.glm5_dsa_graph_forward_state = {
        "path": "graph",
        "bucket": 2,
        "reason": "captured",
        "local_bsz": 2,
        "metadata_prepared": True,
    }
    GLM5AttnWrapper.glm5_dsa_flashmla_graph_metadata = {
        "bucket_size": 2,
        "tile_scheduler_metadata": torch.empty(1, dtype=torch.int32),
        "num_splits": torch.empty(1, dtype=torch.int32),
    }
    try:
        actual = wrapper._forward_decode_dsa(
            torch.zeros(2, 1, 16),
            torch.tensor([[7], [8]], dtype=torch.int64),
            torch.tensor([4, 9], dtype=torch.int32),
            4096,
            "primary",
            "aux",
        )
    finally:
        AttnWrapperBase.batchgen_debug = old_debug
        GLM5AttnWrapper.glm5_dsa_graph_forward_state = None
        GLM5AttnWrapper.glm5_dsa_flashmla_graph_metadata = None

    assert actual is expected
    assert calls == {"return_debug": True, "compare": True}


def test_glm5_compare_tensor_summary_exact_checks_bfloat_values():
    graph = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
    eager = torch.tensor([1.0078125, 2.0], dtype=torch.bfloat16)

    failed, summary = GLM5AttnWrapper._compare_tensor_summary(
        "primary_k_tensor",
        graph,
        eager,
        exact=True,
    )

    assert failed
    assert "mismatch=1/2" in summary


def test_glm5_dsa_graph_segment_inputs_expose_rotated_q_rope(monkeypatch):
    flash_attn_mod = types.ModuleType("flash_attn_interface")
    flash_attn_mod.flash_attn_varlen_func = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "flash_attn_interface", flash_attn_mod)
    fa3_mod = types.ModuleType("batchgen.attention.mla.fa3_backend")
    fa3_mod.act_quant = lambda x: (x, torch.ones(x.shape[0], 1, dtype=torch.float32, device=x.device))
    flashmla_backend_mod = types.ModuleType("batchgen.attention.mla.flashmla_backend")
    flashmla_backend_mod.deepseek_v3_dequantization = lambda weight, scale: weight
    rope_mod = types.ModuleType("batchgen.attention.mla.fused_rmsnorm_rope")
    rope_mod.fused_rmsnorm_rope_with_q_native = lambda *args, **kwargs: None
    gemm_mod = types.ModuleType("batchgen.gemm.w8a8_deepgemm")
    gemm_mod.w8a8_deepgemm = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "batchgen.attention.mla.fa3_backend", fa3_mod)
    monkeypatch.setitem(sys.modules, "batchgen.attention.mla.flashmla_backend", flashmla_backend_mod)
    monkeypatch.setitem(sys.modules, "batchgen.attention.mla.fused_rmsnorm_rope", rope_mod)
    monkeypatch.setitem(sys.modules, "batchgen.gemm.w8a8_deepgemm", gemm_mod)

    from batchgen.attention.dsa import glm5_decode_selector as selector

    batch_size = 2
    num_heads = 2
    qk_nope = 3
    qk_rope = 2
    q_head_dim = qk_nope + qk_rope
    kv_lora_rank = 4
    q_rank = 4
    index_heads = 2
    index_dim = 4

    def fake_act_quant(x):
        return x, torch.ones(x.shape[0], 1, dtype=torch.float32, device=x.device)

    def fake_w8a8(x, x_scale, weight, weight_scale):
        if weight == "q_a":
            return torch.arange(
                x.shape[0] * q_rank, dtype=torch.float32, device=x.device,
            ).view(x.shape[0], q_rank).to(torch.bfloat16)
        if weight == "q_b":
            return torch.arange(
                x.shape[0] * num_heads * q_head_dim,
                dtype=torch.float32,
                device=x.device,
            ).view(x.shape[0], num_heads * q_head_dim).to(torch.bfloat16)
        if weight == "kv_a":
            return torch.zeros(
                x.shape[0],
                kv_lora_rank + qk_rope,
                dtype=torch.bfloat16,
                device=x.device,
            )
        raise AssertionError(weight)

    def fake_fused_rmsnorm_rope(
        new_compressed_kv,
        q_pe,
        cos,
        sin,
        position_ids,
        weight,
        kv_lora,
        rope_dim,
        *,
        eps,
    ):
        q_pe.add_(100)
        return torch.zeros_like(new_compressed_kv)

    monkeypatch.setattr(selector, "act_quant", fake_act_quant)
    monkeypatch.setattr(selector, "w8a8_deepgemm", fake_w8a8)
    monkeypatch.setattr(selector, "_fused_rmsnorm_rope", fake_fused_rmsnorm_rope)

    kv_proj_mod = types.ModuleType("batchgen_kernels.attention.dsa.fused_indexer_kv_proj_cuda")
    kv_proj_mod.cuda_wk_proj_gemm_only = lambda hidden_flat, weights, module: torch.zeros(
        hidden_flat.shape[0],
        index_dim,
        dtype=torch.bfloat16,
        device=hidden_flat.device,
    )
    score_mod = types.ModuleType("batchgen_kernels.attention.dsa.fused_indexer_score")
    score_mod.compute_head_gates = lambda hidden_flat, weight, heads, dim: torch.ones(
        hidden_flat.shape[0],
        heads,
        dtype=torch.float32,
        device=hidden_flat.device,
    )
    monkeypatch.setitem(
        sys.modules,
        "batchgen_kernels.attention.dsa.fused_indexer_kv_proj_cuda",
        kv_proj_mod,
    )
    monkeypatch.setitem(
        sys.modules,
        "batchgen_kernels.attention.dsa.fused_indexer_score",
        score_mod,
    )

    class FakeLayerNorm:
        weight = torch.ones(kv_lora_rank)
        eps = 1e-6

        def __call__(self, x):
            return x

    class FakeIndexer:
        index_n_heads = index_heads
        index_head_dim = index_dim
        weights_proj = type("WeightsProj", (), {"weight": torch.ones(index_heads, index_dim)})()

        def k_norm(self, x):
            return x

        def _fused_rope_hadamard_or_fallback(self, x, positions, *, max_seqlen):
            return x

    class FakeAttn:
        def __init__(self):
            self.qk_nope_head_dim = qk_nope
            self.qk_rope_head_dim = qk_rope
            self.q_head_dim = q_head_dim
            self.num_heads = num_heads
            self.kv_lora_rank = kv_lora_rank
            self.q_a_proj = type("Proj", (), {"weight": "q_a"})()
            self.q_b_proj = type("Proj", (), {"weight": "q_b"})()
            self.kv_a_proj_with_mqa = type("Proj", (), {"weight": "kv_a"})()
            self.q_a_layernorm = FakeLayerNorm()
            self.kv_a_layernorm = FakeLayerNorm()
            self.indexer = FakeIndexer()

        def rotary_emb(self, q_pe, *, seq_len):
            return torch.ones(1), torch.zeros(1)

    class FakePageTableManager:
        seq_id_to_slot = {101: 1, 102: 0}

    class FakeManager:
        device = torch.device("cpu")
        _gpu_page_table_manager = FakePageTableManager()

        def update_layer_decode_new_token(self, *args, **kwargs):
            raise AssertionError("write_kv=False should not update caches")

    wrapper = type(
        "Wrapper",
        (),
        {
            "weight_dequant_scale": {
                "q_a_proj.weight_scale_inv": None,
                "q_b_proj.weight_scale_inv": None,
                "kv_a_proj_with_mqa.weight_scale_inv": None,
            },
            "module": FakeAttn(),
            "layer_idx": 0,
            "_indexer_cuda_weights": object(),
            "_indexer_cuda_module": object(),
        },
    )()
    hidden_states = torch.zeros(batch_size, 1, q_rank, dtype=torch.bfloat16)
    position_ids = torch.tensor([[7], [8]], dtype=torch.int64)
    cache_seqlens = torch.tensor([8, 9], dtype=torch.int32)

    old_batch = AttnWrapperBase.cur_batch
    AttnWrapperBase.cur_batch = [101, 102]
    try:
        graph_inputs = selector.build_glm5_dsa_graph_segment_inputs(
            wrapper,
            hidden_states,
            position_ids,
            cache_seqlens,
            max_seqlen=16,
            gpu_paged_kv_manager=FakeManager(),
            gpu_paged_kv_manager_aux=FakeManager(),
            write_kv=False,
        )
    finally:
        AttnWrapperBase.cur_batch = old_batch

    raw_q = fake_w8a8(
        torch.empty(batch_size, q_rank),
        None,
        "q_b",
        None,
    ).view(batch_size, 1, num_heads, q_head_dim).transpose(1, 2)
    expected_q_rope = raw_q[..., qk_nope:].squeeze(2).contiguous() + 100

    torch.testing.assert_close(graph_inputs.q_rope, expected_q_rope)
    torch.testing.assert_close(
        graph_inputs.q_nope,
        raw_q[..., :qk_nope].squeeze(2).contiguous(),
    )
    assert graph_inputs.primary_slot_indices.tolist() == [1, 0]
    assert graph_inputs.aux_slot_indices.tolist() == [1, 0]


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


def test_glm5_dsa_cuda_graph_replay_gate_allows_short_rows_with_fixed_selected_kv():
    index_topk = 4

    assert _glm5_dsa_cuda_graph_can_replay(
        torch.tensor([5, 7], dtype=torch.int32),
        max_seqlen=8,
        index_topk=index_topk,
    )
    assert _glm5_dsa_cuda_graph_can_replay(
        torch.tensor([4, 4], dtype=torch.int32),
        max_seqlen=4,
        index_topk=index_topk,
        captured_max_seqlen=8,
    )
    assert _glm5_dsa_cuda_graph_can_replay(
        torch.tensor([1, 4], dtype=torch.int32),
        max_seqlen=4,
        index_topk=index_topk,
        captured_max_seqlen=8,
    )
    assert _glm5_dsa_cuda_graph_can_replay(
        torch.tensor([2, 4, 7], dtype=torch.int32),
        max_seqlen=7,
        index_topk=index_topk,
        captured_max_seqlen=8,
    )
    assert not _glm5_dsa_cuda_graph_can_replay(
        torch.tensor([[1, 4]], dtype=torch.int32),
        max_seqlen=4,
        index_topk=index_topk,
        captured_max_seqlen=8,
    )
    assert not _glm5_dsa_cuda_graph_can_replay(
        torch.tensor([1, 4], dtype=torch.int32),
        max_seqlen=0,
        index_topk=index_topk,
    )
    assert not _glm5_dsa_cuda_graph_can_replay(
        torch.tensor([1, 4], dtype=torch.int32),
        max_seqlen=4,
        index_topk=0,
    )
    assert _glm5_dsa_cuda_graph_can_replay(
        torch.tensor([5, 7], dtype=torch.int32),
        max_seqlen=8,
        index_topk=index_topk,
        captured_max_seqlen=6,
    )






def test_glm5_whole_model_compare_policy_requests_warmup():
    env = {"BATCHGEN_GLM5_WHOLE_MODEL_GRAPH_COMPARE": "1"}
    model_name = "zai-org/GLM-5-FP8"

    assert glm5_whole_model_cuda_graph_compare_requested_for_model(
        model_name, environ=env
    )
    assert not glm5_whole_model_cuda_graph_compare_requested_for_model(
        "gpt-oss-120b", environ=env
    )
    assert not glm5_whole_model_cuda_graph_requested_for_model(model_name, environ=env)
    assert glm5_any_cuda_graph_requested_for_model(model_name, environ=env)
    assert should_warmup_cuda_graphs_before_decode(
        graph_manager_is_initialized=False,
        global_batch_has_queueing=True,
        model_name=model_name,
        environ=env,
    )


def test_glm5_effective_decode_attn_mode_uses_continuous_path():
    assert glm5_effective_decode_attn_mode("glm_moe_dsa", 1) == 3
    assert glm5_effective_decode_attn_mode("zai-org/GLM-5-FP8", 0) == 3
    assert glm5_effective_decode_attn_mode("deepseek_v3", 1) == 1




def test_glm5_enable_cuda_graph_defaults_to_whole_model_graph():
    env = {}

    assert glm5_whole_model_cuda_graph_requested_for_model(
        "zai-org/GLM-5-FP8",
        enable_cuda_graph=True,
        environ=env,
    )
    assert glm5_whole_model_cuda_graph_requested_for_model(
        "zai-org/GLM-5.1-FP8",
        enable_cuda_graph=True,
        environ=env,
    )
    assert not glm5_dsa_cuda_graph_requested_for_model(
        "zai-org/GLM-5-FP8",
        enable_cuda_graph=True,
        environ=env,
    )
    assert not glm5_moe_cuda_graph_requested_for_model(
        "zai-org/GLM-5.1-FP8",
        enable_cuda_graph=True,
        environ=env,
    )
    assert not glm5_segmented_cuda_graph_requested_for_model(
        "zai-org/GLM-5-FP8",
        enable_cuda_graph=True,
        environ=env,
    )
    assert glm5_any_cuda_graph_requested_for_model(
        "zai-org/GLM-5.1-FP8",
        enable_cuda_graph=True,
        environ=env,
    )
    assert not glm5_whole_model_cuda_graph_requested_for_model(
        "zai-org/GLM-5-FP8",
        enable_cuda_graph=False,
        environ=env,
    )
    assert not glm5_whole_model_cuda_graph_requested_for_model(
        "zai-org/GLM-5",
        enable_cuda_graph=True,
        environ=env,
    )
    assert not glm5_any_cuda_graph_requested_for_model(
        "gpt-oss-120b",
        enable_cuda_graph=True,
        environ=env,
    )




def test_server_enable_cuda_graph_flag_is_user_facing(tmp_path, monkeypatch):
    import batchgen.server.server_args as server_args_module

    monkeypatch.delenv("BATCHGEN_SEGMENTED_GRAPH", raising=False)
    monkeypatch.delenv("BATCHGEN_GLM5_DSA_CUDA_GRAPH", raising=False)
    monkeypatch.delenv("BATCHGEN_GLM5_MOE_CUDA_GRAPH", raising=False)
    monkeypatch.delenv("BATCHGEN_GLM5_WHOLE_MODEL_CUDA_GRAPH", raising=False)
    monkeypatch.setattr(server_args_module, "is_port_available", lambda port: True)

    args = server_args_module.prepare_server_args([
        "--model",
        "zai-org/GLM-5-FP8",
        "--listen-port",
        "11999",
        "--enable-cuda-graph",
        "--storage-path",
        str(tmp_path / "storage"),
    ])

    assert args.enable_cuda_graph
    assert not args.disable_cuda_graphs
    assert "BATCHGEN_SEGMENTED_GRAPH" not in os.environ
    assert "BATCHGEN_GLM5_DSA_CUDA_GRAPH" not in os.environ
    assert "BATCHGEN_GLM5_MOE_CUDA_GRAPH" not in os.environ
    assert "BATCHGEN_GLM5_WHOLE_MODEL_CUDA_GRAPH" not in os.environ






def test_glm5_whole_graph_decode_cap_matches_largest_bucket(monkeypatch):
    from batchgen.batchgen_worker import BatchGenWorker

    monkeypatch.delenv("BATCHGEN_MAX_DECODE_RANK_BSZ", raising=False)
    worker = object.__new__(BatchGenWorker)
    worker.model_name = "zai-org/GLM-5.2-FP8"
    worker.args = types.SimpleNamespace(
        enable_cuda_graph=True,
        disable_cuda_graphs=False,
        cuda_graph_max_bucket_size=256,
    )

    assert worker._decode_rank_batch_cap() == 256


def test_glm52_qprep_initializes_every_layer(monkeypatch):
    from batchgen.batchgen_worker import BatchGenWorker

    calls = []
    worker = object.__new__(BatchGenWorker)
    worker.rank = 0
    worker._batchgen_debug = {}
    worker.model_config = types.SimpleNamespace(model_type="glm_moe_dsa_5_2")
    worker.model = types.SimpleNamespace(
        model=types.SimpleNamespace(
            layers=[
                types.SimpleNamespace(
                    self_attn=types.SimpleNamespace(
                        _initialize_folded_q_b=lambda layer=i: calls.append(layer)
                    )
                )
                for i in range(3)
            ]
        )
    )
    monkeypatch.setattr(
        worker,
        "_glm5_whole_model_graph_requested_for_current_batch",
        lambda: True,
    )

    worker._initialize_glm52_folded_q_b_for_decode()

    assert calls == [0, 1, 2]


def test_glm52_qprep_skips_without_whole_graph(monkeypatch):
    from batchgen.batchgen_worker import BatchGenWorker

    calls = []
    worker = object.__new__(BatchGenWorker)
    worker._batchgen_debug = {}
    worker.model_config = types.SimpleNamespace(model_type="glm_moe_dsa_5_2")
    worker.model = types.SimpleNamespace(
        model=types.SimpleNamespace(
            layers=[
                types.SimpleNamespace(
                    self_attn=types.SimpleNamespace(
                        _initialize_folded_q_b=lambda: calls.append(True)
                    )
                )
            ]
        )
    )
    monkeypatch.setattr(
        worker,
        "_glm5_whole_model_graph_requested_for_current_batch",
        lambda: False,
    )

    worker._initialize_glm52_folded_q_b_for_decode()

    assert calls == []


def test_decode_cap_retains_default_without_glm5_whole_graph(monkeypatch):
    from batchgen.batchgen_worker import BatchGenWorker

    monkeypatch.delenv("BATCHGEN_MAX_DECODE_RANK_BSZ", raising=False)
    worker = object.__new__(BatchGenWorker)
    worker.model_name = "zai-org/GLM-5.2-FP8"
    worker.args = types.SimpleNamespace(
        enable_cuda_graph=False,
        disable_cuda_graphs=True,
        cuda_graph_max_bucket_size=256,
    )

    assert worker._decode_rank_batch_cap() == 128


def test_decode_cap_explicit_override_is_authoritative(monkeypatch):
    from batchgen.batchgen_worker import BatchGenWorker

    monkeypatch.setenv("BATCHGEN_MAX_DECODE_RANK_BSZ", "80")
    monkeypatch.delenv("BATCHGEN_MAX_RANK_BSZ", raising=False)
    worker = object.__new__(BatchGenWorker)
    worker.model_name = "zai-org/GLM-5.2-FP8"
    worker.args = types.SimpleNamespace(
        enable_cuda_graph=True,
        disable_cuda_graphs=False,
        cuda_graph_max_bucket_size=256,
    )

    assert worker._decode_rank_batch_cap() == 80


def test_decode_cap_legacy_override_is_authoritative(monkeypatch):
    from batchgen.batchgen_worker import BatchGenWorker

    monkeypatch.delenv("BATCHGEN_MAX_DECODE_RANK_BSZ", raising=False)
    monkeypatch.setenv("BATCHGEN_MAX_RANK_BSZ", "96")
    worker = object.__new__(BatchGenWorker)
    worker.model_name = "zai-org/GLM-5.2-FP8"
    worker.args = types.SimpleNamespace(
        enable_cuda_graph=True,
        disable_cuda_graphs=False,
        cuda_graph_max_bucket_size=256,
    )

    assert worker._decode_rank_batch_cap() == 96


def test_decode_cap_rejects_conflicting_overrides(monkeypatch):
    from batchgen.batchgen_worker import BatchGenWorker

    monkeypatch.setenv("BATCHGEN_MAX_DECODE_RANK_BSZ", "256")
    monkeypatch.setenv("BATCHGEN_MAX_RANK_BSZ", "128")
    worker = object.__new__(BatchGenWorker)

    with pytest.raises(RuntimeError, match="must match"):
        worker._decode_rank_batch_cap()




def test_glm5_dispatch_trace_records_requested_paths(monkeypatch):
    from batchgen.batchgen_worker import BatchGenWorker

    monkeypatch.delenv("BATCHGEN_GLM5_DISPATCH_TRACE", raising=False)
    monkeypatch.setattr(
        AttnWrapperBase,
        "batchgen_debug",
        {
            "glm5_dispatch_trace": True,
            "glm5_dsa_mode": "eager",
            "glm5_moe_mode": "graph",
        },
        raising=False,
    )
    monkeypatch.setattr(GLM5AttnWrapper, "glm5_dispatch_trace_enabled", False, raising=False)
    monkeypatch.setattr(GLM5AttnWrapper, "glm5_dispatch_trace_id", None, raising=False)
    monkeypatch.setattr(GLM5AttnWrapper, "glm5_dispatch_trace_context", None, raising=False)
    monkeypatch.setattr(GLM5AttnWrapper, "glm5_dispatch_counts", {}, raising=False)
    monkeypatch.setattr(GLM5AttnWrapper, "glm5_dispatch_seen", set(), raising=False)

    worker = object.__new__(BatchGenWorker)
    worker.rank = 0
    seqs = [
        types.SimpleNamespace(batch_id="batch-a", global_idx=7),
        types.SimpleNamespace(batch_id="batch-a", global_idx=3),
    ]

    worker._configure_glm5_dispatch_trace(seqs)
    assert GLM5AttnWrapper.glm5_dispatch_trace_enabled
    assert GLM5AttnWrapper.glm5_dispatch_trace_context == {
        "rank": 0,
        "batch_ids": "batch-a",
        "global_ids": "3,7",
        "bsz": 2,
        "glm5_dsa_mode": "eager",
        "glm5_moe_mode": "graph",
        "glm5_moe_router_mode": "-",
    }

    AttnWrapperBase.record_glm5_dispatch(
        kind="dsa",
        path="eager",
        layer_idx=0,
        bsz=2,
        reason="debug mode requested eager",
    )
    AttnWrapperBase.record_glm5_dispatch(
        kind="dsa",
        path="eager",
        layer_idx=1,
        bsz=2,
        reason="debug mode requested eager",
    )
    AttnWrapperBase.record_glm5_dispatch(
        kind="moe",
        path="graph",
        layer_idx=3,
        bsz=2,
        reason="graph replay",
    )

    assert GLM5AttnWrapper.glm5_dispatch_counts == {
        "dsa_eager": 2,
        "moe_graph": 1,
    }

    monkeypatch.setattr(AttnWrapperBase, "batchgen_debug", {}, raising=False)
    worker._configure_glm5_dispatch_trace(seqs)
    assert not GLM5AttnWrapper.glm5_dispatch_trace_enabled
    assert GLM5AttnWrapper.glm5_dispatch_counts == {}


def test_glm5_dsa_debug_mode_selects_actual_dispatch_branch(monkeypatch):
    monkeypatch.setenv("BATCHGEN_GLM5_DSA_FULL_CUDA_GRAPH", "1")
    monkeypatch.setattr(GLM5AttnWrapper, "glm5_dispatch_trace_enabled", True, raising=False)
    monkeypatch.setattr(GLM5AttnWrapper, "glm5_dispatch_trace_id", "unit-dsa", raising=False)
    monkeypatch.setattr(GLM5AttnWrapper, "glm5_dispatch_trace_context", {}, raising=False)
    monkeypatch.setattr(GLM5AttnWrapper, "glm5_dispatch_counts", {}, raising=False)
    monkeypatch.setattr(GLM5AttnWrapper, "glm5_dispatch_seen", set(), raising=False)

    wrapper = object.__new__(GLM5AttnWrapper)
    wrapper.layer_idx = 0
    wrapper.module = types.SimpleNamespace(
        hidden_size=4,
        indexer=types.SimpleNamespace(index_topk=3),
    )
    wrapper._dsa_cuda_graph_manager = types.SimpleNamespace(has_graph=lambda name, bsz: True)
    wrapper._dsa_cuda_graph_segment_name = "dsa"
    wrapper._dsa_cuda_graph_max_seqlen = 16
    monkeypatch.setattr(
        wrapper,
        "_dsa_cuda_graph_page_tables_match",
        lambda primary, aux: True,
    )
    monkeypatch.setattr(
        wrapper,
        "_dsa_cuda_graph_forward_state_allows_replay",
        lambda bsz: (True, "captured"),
    )
    monkeypatch.setattr(
        wrapper,
        "_forward_decode_dsa_graph",
        lambda *args, **kwargs: torch.full((1, 1, 4), 11.0),
    )
    monkeypatch.setattr(
        wrapper,
        "_forward_decode_dsa_eager",
        lambda *args, **kwargs: torch.full((1, 1, 4), 22.0),
    )
    hidden = torch.zeros(1, 1, 4)
    position_ids = torch.zeros(1, 1, dtype=torch.int64)
    cache_seqlens = torch.ones(1, dtype=torch.int32)

    monkeypatch.setattr(
        AttnWrapperBase,
        "batchgen_debug",
        {"glm5_dsa_mode": "graph"},
        raising=False,
    )
    graph_out = wrapper._forward_decode_dsa(
        hidden,
        position_ids,
        cache_seqlens,
        1,
        object(),
        object(),
    )
    assert graph_out[0, 0, 0].item() == 11.0
    assert GLM5AttnWrapper.glm5_dispatch_counts["dsa_graph"] == 1

    GLM5AttnWrapper.glm5_dispatch_counts = {}
    monkeypatch.setattr(
        AttnWrapperBase,
        "batchgen_debug",
        {"glm5_dsa_mode": "eager"},
        raising=False,
    )
    eager_out = wrapper._forward_decode_dsa(
        hidden,
        position_ids,
        cache_seqlens,
        1,
        object(),
        object(),
    )
    assert eager_out[0, 0, 0].item() == 22.0
    assert GLM5AttnWrapper.glm5_dispatch_counts["dsa_eager"] == 1


def test_glm5_moe_debug_mode_selects_actual_dispatch_branch(monkeypatch):
    monkeypatch.setenv("BATCHGEN_GLM5_MOE_CUDA_GRAPH", "1")
    monkeypatch.setattr(GLM5AttnWrapper, "glm5_dispatch_trace_enabled", True, raising=False)
    monkeypatch.setattr(GLM5AttnWrapper, "glm5_dispatch_trace_id", "unit-moe", raising=False)
    monkeypatch.setattr(GLM5AttnWrapper, "glm5_dispatch_trace_context", {}, raising=False)
    monkeypatch.setattr(GLM5AttnWrapper, "glm5_dispatch_counts", {}, raising=False)
    monkeypatch.setattr(GLM5AttnWrapper, "glm5_dispatch_seen", set(), raising=False)
    monkeypatch.setattr(glm5_model, "_GLM5_HAS_DISPATCH_3D", True)
    monkeypatch.setattr(Glm5MoE, "_3d_buf", object(), raising=False)

    moe = object.__new__(Glm5MoE)
    moe.layer_idx = 3
    moe.use_3d_moe = True
    moe._fp8_blockwise_ready = True
    moe._moe_cuda_graph_required = True
    moe.num_tokens_per_rank = [1]
    monkeypatch.setattr(moe, "_moe_cuda_graph_exceeds_max_bucket", lambda: False)
    monkeypatch.setattr(
        moe,
        "_forward_decode_3d_graph",
        lambda hidden_states: hidden_states + 11,
    )
    monkeypatch.setattr(
        moe,
        "_forward_decode_3d",
        lambda hidden_states: hidden_states + 22,
    )
    hidden = torch.zeros(1, 4)

    monkeypatch.setattr(
        AttnWrapperBase,
        "batchgen_debug",
        {"glm5_moe_mode": "graph"},
        raising=False,
    )
    graph_out = moe._forward_decode(hidden)
    assert graph_out[0, 0].item() == 11.0
    assert GLM5AttnWrapper.glm5_dispatch_counts["moe_graph"] == 1

    GLM5AttnWrapper.glm5_dispatch_counts = {}
    monkeypatch.setattr(
        AttnWrapperBase,
        "batchgen_debug",
        {"glm5_moe_mode": "eager"},
        raising=False,
    )
    eager_out = moe._forward_decode(hidden)
    assert eager_out[0, 0].item() == 22.0
    assert GLM5AttnWrapper.glm5_dispatch_counts["moe_eager"] == 1


def test_glm5_debug_modes_override_env_graph_requirements(monkeypatch):
    monkeypatch.setenv("BATCHGEN_GLM5_DSA_FULL_CUDA_GRAPH", "1")
    monkeypatch.setenv("BATCHGEN_GLM5_MOE_CUDA_GRAPH", "1")
    monkeypatch.setattr(
        AttnWrapperBase,
        "batchgen_debug",
        {"glm5_dsa_mode": "eager", "glm5_moe_mode": "eager"},
        raising=False,
    )

    assert not _glm5_dsa_cuda_graph_required()
    assert not glm5_model._glm5_moe_cuda_graph_required()
    assert not _glm5_dsa_graph_compare_active()
    assert not glm5_model._glm5_moe_graph_compare_active()

    monkeypatch.setattr(
        AttnWrapperBase,
        "batchgen_debug",
        {"glm5_dsa_mode": "graph", "glm5_moe_mode": "graph"},
        raising=False,
    )

    assert _glm5_dsa_cuda_graph_required()
    assert glm5_model._glm5_moe_cuda_graph_required()




def test_glm5_dsa_graph_enable_records_cli_required_state():
    wrapper = object.__new__(GLM5AttnWrapper)

    wrapper.enable_dsa_cuda_graph(
        manager=object(),
        segment_name="glm5_dsa_layer_0",
        max_seqlen=131072,
        graph_output_required=True,
    )

    assert wrapper._dsa_cuda_graph_required is True




def test_glm5_moe_graph_enable_records_cli_required_state():
    moe = object.__new__(Glm5MoE)

    moe.enable_moe_cuda_graph(
        manager=object(),
        segment_name="glm5_moe_layer_3",
        segment=object(),
        bucketing=object(),
        graph_output_required=True,
    )

    assert moe._moe_cuda_graph_required is True


def test_glm5_moe_graph_output_is_full_module_boundary(monkeypatch):
    class FakeBucketing:
        def get_padded_size(self, batch_size):
            assert batch_size == 4
            return 4

    class FakePool:
        def __init__(self):
            self.padded = torch.empty(4, 2)

        def get(self, bucket):
            assert bucket == 4
            return types.SimpleNamespace(padded=self.padded)

    class FakeManager:
        def __init__(self):
            self.replay_inputs = None

        def has_graph(self, segment_name, batch_size):
            return segment_name == "glm5_layer_3_moe" and batch_size == 4

        def replay(self, segment_name, bucket, **inputs):
            assert segment_name == "glm5_layer_3_moe"
            assert bucket == 4
            self.replay_inputs = inputs
            return {
                "moe_output": torch.tensor(
                    [
                        [10.0, 11.0],
                        [12.0, 13.0],
                        [99.0, 99.0],
                        [99.0, 99.0],
                    ]
                ),
                "routed_global_output": torch.full((8, 2), -999.0),
            }

    class FakeComm:
        all_reduce_calls = 0

        class _State:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb):
                return False

        def change_state(self, enable=True):
            return self._State()

        def all_reduce(self, *args, **kwargs):
            self.all_reduce_calls += 1

    manager = FakeManager()
    segment = types.SimpleNamespace(pool=FakePool())
    comm = FakeComm()

    moe = object.__new__(Glm5MoE)
    moe.layer_idx = 3
    moe.rank = 1
    moe.world_size = 2
    moe.device = torch.device("cpu")
    moe.num_tokens_per_rank = 4
    moe._moe_cuda_graph_manager = manager
    moe._moe_cuda_graph_segment_name = "glm5_layer_3_moe"
    moe._moe_cuda_graph_segment = segment
    moe._moe_cuda_graph_bucketing = FakeBucketing()
    moe.comm = comm

    def unexpected_shared_expert(_identity):
        raise AssertionError("shared expert must be inside the MoE graph")

    moe.shared_expert_forward = unexpected_shared_expert
    monkeypatch.setattr(Glm5MoE, "_rank_token_counts", None)

    hidden = torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]])
    out = moe._forward_decode_3d_graph(hidden)

    assert torch.equal(out, torch.tensor([[[10.0, 11.0]], [[12.0, 13.0]]]))
    assert comm.all_reduce_calls == 0
    assert torch.equal(segment.pool.padded[:2], hidden.view(2, 2))
    assert torch.equal(segment.pool.padded[2:], torch.zeros(2, 2))
    assert manager.replay_inputs["padded"] is segment.pool.padded
    assert manager.replay_inputs["rank_token_counts"].tolist() == [4, 4]


def test_glm5_graph_buckets_cover_production_local_batches():
    buckets = GLM5_POWER_OF_TWO_BUCKETS_32

    assert buckets == [64, 192, 256]
    assert [
        glm5_cuda_graph_bucket_for_batch_size(batch_size, buckets)
        for batch_size in [0, 1, 64, 65, 192, 193, 256, 257]
    ] == [None, 64, 64, 192, 192, 256, 256, None]


def test_glm5_moe_bucket_capacity_uses_h200_world_size():
    buckets = GLM5_POWER_OF_TWO_BUCKETS_32

    assert glm5_moe_graph_bucket_capacity(
        max_rank_batch_size=42,
        world_size=8,
        bucket_sizes=buckets,
    ) == (64, 512)
    assert glm5_moe_graph_bucket_capacity(
        max_rank_batch_size=65,
        world_size=8,
        bucket_sizes=buckets,
    ) == (192, 1536)
    assert glm5_moe_graph_bucket_capacity(
        max_rank_batch_size=193,
        world_size=8,
        bucket_sizes=buckets,
    ) == (256, 2048)
    assert glm5_moe_graph_bucket_capacity(
        max_rank_batch_size=257,
        world_size=8,
        bucket_sizes=buckets,
    ) is None


def test_glm5_moe_graph_capacity_validates_world_size():
    with pytest.raises(ValueError, match="world_size must be positive"):
        glm5_moe_graph_bucket_capacity(
            max_rank_batch_size=1,
            world_size=0,
        )


def test_glm5_moe_graph_over_bucket_routes_eager(monkeypatch):
    monkeypatch.setenv("BATCHGEN_GLM5_MOE_CUDA_GRAPH", "1")
    monkeypatch.setattr(glm5_model, "_GLM5_HAS_DISPATCH_3D", True)
    monkeypatch.setattr(Glm5MoE, "_3d_buf", object())

    moe = object.__new__(Glm5MoE)
    moe.layer_idx = 3
    moe.use_3d_moe = True
    moe._fp8_blockwise_ready = True
    moe.num_tokens_per_rank = 70
    moe._moe_cuda_graph_bucketing = BatchSizeBucketing([1, 2])
    moe._moe_cuda_graph_manager = object()
    moe._moe_cuda_graph_segment_name = "glm5_moe_layer_3"
    moe._moe_cuda_graph_segment = object()

    def eager(self, hidden_states):
        return hidden_states + 1

    def graph(self, hidden_states):
        raise AssertionError("graph path should not be used for over-bucket MoE")

    monkeypatch.setattr(Glm5MoE, "_forward_decode_3d", eager)
    monkeypatch.setattr(Glm5MoE, "_forward_decode_3d_graph", graph)

    hidden = torch.zeros(1, 1, 2)
    out = moe._forward_decode(hidden)

    assert torch.equal(out, hidden + 1)




def test_glm5_dsa_page_table_signature_prefers_stable_graph_storage():
    storage = torch.empty(4, 8, dtype=torch.int32)

    class FakeManager:
        def get_cuda_graph_page_table_storage(self):
            return storage

        def get_cuda_graph_page_table(self):
            raise RuntimeError("active graph table is invalid")

    assert _glm5_dsa_gpu_page_table_tensor(FakeManager()) is storage






def test_glm5_gpu_kv_config_uses_actual_prompt_for_graph_page_table():
    from batchgen.batchgen_worker import BatchGenWorker
    from batchgen.kv_cache.gpu_paged_kv_manager import (
        GPUPagedKVCacheManager,
        GPUPagedKVConfig,
    )

    worker = object.__new__(BatchGenWorker)
    worker.max_input_length = 65355
    worker.max_decoding_length = 4096
    worker.engine_config = None
    worker.args = types.SimpleNamespace(cuda_graph_max_bucket_size=64)
    config = GPUPagedKVConfig(
        num_layers=1,
        num_pages=4096,
        page_size_tokens=64,
        num_k_heads=1,
        k_head_dim=1,
        num_v_heads=0,
        v_head_dim=0,
        kv_dtype=torch.bfloat16,
    )

    updated = worker._with_cuda_graph_page_table_capacity(config)
    expected_pages = (65355 + 4096 + 63) // 64

    assert updated.cuda_graph_max_pages_per_sequence == expected_pages
    assert updated.cuda_graph_max_slots == 64
    manager = GPUPagedKVCacheManager(config=updated, device="cpu")
    assert (
        manager._gpu_page_table_manager.graph_max_pages_per_sequence
        == expected_pages
    )
    assert manager._gpu_page_table_manager.max_slots == 64


def test_glm5_gpu_kv_config_uses_model_max_for_graph_page_table():
    from batchgen.batchgen_worker import BatchGenWorker
    from batchgen.kv_cache.gpu_paged_kv_manager import GPUPagedKVConfig

    worker = object.__new__(BatchGenWorker)
    worker.max_input_length = 0
    worker.max_decoding_length = 0
    worker.engine_config = None
    worker.model_config = types.SimpleNamespace(max_position_embeddings=131072)
    worker.args = types.SimpleNamespace(cuda_graph_max_bucket_size=64)
    config = GPUPagedKVConfig(
        num_layers=1,
        num_pages=4096,
        page_size_tokens=64,
        num_k_heads=1,
        k_head_dim=1,
        num_v_heads=0,
        v_head_dim=0,
        kv_dtype=torch.bfloat16,
    )

    updated = worker._with_cuda_graph_page_table_capacity(config)

    assert updated.cuda_graph_max_pages_per_sequence == 2048
    assert updated.cuda_graph_max_slots == 64


def test_glm5_dsa_graph_score_capacity_uses_page_table_capacity(monkeypatch):
    from batchgen.batchgen_worker import BatchGenWorker

    primary_page_table = torch.empty(2, 320, dtype=torch.int32)
    aux_page_table = torch.empty(2, 512, dtype=torch.int32)
    monkeypatch.setenv("BATCHGEN_GLM5_DSA_CUDA_GRAPH_MAX_SEQLEN", "8192")

    assert BatchGenWorker._glm5_dsa_graph_score_capacity_tokens(
        primary_page_table,
        64,
        aux_page_table,
        64,
        model_max_position_embeddings=131072,
    ) == 20480
    assert BatchGenWorker._glm5_dsa_graph_score_capacity_tokens(
        primary_page_table,
        64,
        aux_page_table,
        64,
        model_max_position_embeddings=16889,
    ) == 16889


def test_glm5_dsa_graph_required_tokens_uses_active_sequence_budgets():
    from batchgen.batchgen_worker import BatchGenWorker

    worker = object.__new__(BatchGenWorker)
    worker.global_batch = SequenceBatch()
    for gid, budget in ((11, 551), (17, 4097), (23, 900)):
        seq = SequenceEntry(
            uuid=f"seq-{gid}",
            global_idx=gid,
            prompt_length=1,
            max_decode_length=budget - 1,
        )
        worker.global_batch.add_sequence(seq)
    worker.max_input_length = 8192
    worker.max_decoding_length = 512

    assert worker._glm5_dsa_graph_required_tokens([11, 23], page_size=64) == 4160
    assert worker._glm5_dsa_graph_required_tokens([17], page_size=64) == 4160


def test_glm5_dsa_graph_required_tokens_falls_back_to_worker_budget():
    from batchgen.batchgen_worker import BatchGenWorker

    worker = object.__new__(BatchGenWorker)
    worker.global_batch = SequenceBatch()
    worker.max_input_length = 39
    worker.max_decoding_length = 512

    assert worker._glm5_dsa_graph_required_tokens([], page_size=64) == 576






















def test_glm5_layer_graph_compare_eager_reference_forces_segmented_modes(
    monkeypatch,
):
    from batchgen.batchgen_worker import BatchGenWorker

    original_debug = {
        "glm5_layer_graph_compare": True,
        "glm5_dsa_mode": "graph",
        "glm5_moe_mode": "graph",
        "glm5_moe_router_mode": "custom",
    }
    monkeypatch.setattr(
        AttnWrapperBase,
        "batchgen_debug",
        original_debug,
        raising=False,
    )

    worker = object.__new__(BatchGenWorker)
    with worker._glm5_force_segmented_graph_eager():
        active_debug = AttnWrapperBase.batchgen_debug
        assert active_debug is not original_debug
        assert active_debug["glm5_layer_graph_compare"] is True
        assert active_debug["glm5_dsa_mode"] == "eager"
        assert active_debug["glm5_moe_mode"] == "eager"
        assert active_debug["glm5_moe_router_mode"] == "custom"
        assert original_debug["glm5_dsa_mode"] == "graph"
        assert original_debug["glm5_moe_mode"] == "graph"

    assert AttnWrapperBase.batchgen_debug is original_debug














def test_glm5_layer_signature_uses_stable_page_table_storage():
    from batchgen.batchgen_worker import BatchGenWorker

    class FakeManager:
        def __init__(self, storage):
            self.storage = storage

        def get_cuda_graph_page_table_storage(self):
            return self.storage

        def get_cuda_graph_page_table(self):
            raise RuntimeError("active graph table is invalid")

    primary = FakeManager(torch.empty(4, 8, dtype=torch.int32))
    aux = FakeManager(torch.empty(4, 9, dtype=torch.int32))
    worker = object.__new__(BatchGenWorker)
    worker.gpu_paged_kv_cache_manager = types.SimpleNamespace(
        primary=primary,
        auxiliary=aux,
    )
    worker.core_engine = types.SimpleNamespace()

    signature = worker._glm5_whole_model_graph_capture_signature()

    assert signature[0][0] == primary.storage.data_ptr()
    assert signature[0][1] == (4, 8)
    assert signature[1][0] == aux.storage.data_ptr()
    assert signature[1][1] == (4, 9)










def test_glm5_deep_free_resets_segmented_graph_capture_attempts(monkeypatch):
    from batchgen.batchgen_worker import BatchGenWorker

    class FakeModel:
        pass

    worker = object.__new__(BatchGenWorker)
    worker.model = FakeModel()
    worker.torch_device = torch.device("cpu")
    worker.parallel_manager = None
    worker._cuda_graph_manager = object()
    worker._glm5_moe_cuda_graph_manager = object()
    worker._glm5_layer_cuda_graph_manager = object()
    worker._glm5_dsa_graph_capture_attempted_for_batch = True
    worker._glm5_moe_graph_capture_attempted_for_batch = True
    worker._glm5_layer_graph_capture_attempted_for_batch = True
    worker._glm5_dsa_graph_page_table_change_after_capture_logged = True
    worker._whole_model_segment = object()
    worker._whole_model_bucketing = object()
    worker._glm5_whole_model_capture_input_ids = object()
    worker._glm5_moe_graph_failed_buckets = {64}
    worker._glm5_layer_graph_failed_buckets = {32}
    worker._glm5_layer_graph_signature = object()
    worker._glm5_layer_graph_max_seqlen = 8192
    worker._whole_model_graph = True
    worker._glm5_whole_model_graph = True
    worker._glm5_whole_model_graph_failed_buckets = {64}
    worker._glm5_whole_model_graph_signature = object()
    worker._glm5_whole_model_graph_unavailable_reason = "stale"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    worker.deep_free_model_memory()

    assert worker.model is None
    assert worker._cuda_graph_manager is None
    assert worker._glm5_moe_cuda_graph_manager is None
    assert worker._glm5_layer_cuda_graph_manager is None
    assert not worker._glm5_dsa_graph_capture_attempted_for_batch
    assert not worker._glm5_moe_graph_capture_attempted_for_batch
    assert not worker._glm5_layer_graph_capture_attempted_for_batch
    assert not worker._glm5_dsa_graph_page_table_change_after_capture_logged
    assert worker._glm5_layer_graph_failed_buckets == set()
    assert worker._glm5_layer_graph_signature is None
    assert worker._glm5_layer_graph_max_seqlen is None




def test_glm5_graph_path_log_flag_uses_batch_debug_and_env(monkeypatch):
    from batchgen.batchgen_worker import BatchGenWorker

    worker = object.__new__(BatchGenWorker)
    worker._batchgen_debug = {"glm5_graph_path_log": True}
    monkeypatch.delenv("BATCHGEN_GLM5_GRAPH_PATH_LOG", raising=False)

    assert worker._glm5_graph_path_log_requested_for_current_batch()

    worker._batchgen_debug = {}
    assert not worker._glm5_graph_path_log_requested_for_current_batch()

    monkeypatch.setenv("BATCHGEN_GLM5_GRAPH_PATH_LOG", "1")
    assert worker._glm5_graph_path_log_requested_for_current_batch()






def test_glm5_whole_model_warmup_policy_allows_capture_with_queued_prefill():
    assert should_warmup_cuda_graphs_before_decode(
        graph_manager_is_initialized=False,
        global_batch_has_queueing=True,
        model_name="zai-org/GLM-5.2-FP8",
        enable_cuda_graph=True,
        environ={},
    )
    assert not should_warmup_cuda_graphs_before_decode(
        graph_manager_is_initialized=True,
        global_batch_has_queueing=True,
        model_name="zai-org/GLM-5.2-FP8",
        enable_cuda_graph=True,
        environ={},
    )


def test_glm5_whole_model_segment_composes_decoder_layer_segments():
    from batchgen.models.glm.glm5.whole_model_cuda_graph_segments import (
        Glm5WholeModelSegment,
    )

    hidden_size = 4
    vocab_size = 5

    class FakeEmbedding:
        def __call__(self, input_ids):
            return input_ids.to(torch.bfloat16).unsqueeze(-1).repeat(1, 1, hidden_size)

    class FakeNorm:
        def __call__(self, hidden_states):
            return hidden_states

    class FakeLmHead:
        def __call__(self, hidden_states):
            logits = hidden_states.new_zeros(
                hidden_states.shape[0],
                hidden_states.shape[1],
                vocab_size,
            )
            logits[..., :hidden_size] = hidden_states
            return logits

    class FakeLayer:
        def __init__(self, layer_idx):
            self.layer_idx = layer_idx
            self.hidden_size = hidden_size
            self.self_attn = types.SimpleNamespace(
                module=types.SimpleNamespace(
                    kv_lora_rank=2,
                    qk_rope_head_dim=1,
                    indexer=types.SimpleNamespace(index_head_dim=2, index_topk=4),
                )
            )

        def __call__(self, *args, **kwargs):
            raise AssertionError("whole-model graph should use layer segments directly")

    class FakeLayerSegment:
        def __init__(self, value):
            self.value = value
            self.calls = 0
            self.setup_bucket = None
            self.release_bucket = None

        def setup_static_buffers(self, bucket_size):
            self.setup_bucket = bucket_size

        def release_static_buffers(self, bucket_size):
            self.release_bucket = bucket_size

        def forward(self, *, hidden_states, **kwargs):
            self.calls += 1
            rows = hidden_states.shape[0]
            return {
                "hidden_states": hidden_states + self.value,
                "primary_k_tensor": torch.full(
                    (rows, 1, 1, 3),
                    self.value,
                    dtype=torch.bfloat16,
                ),
                "indexer_k_tensor": torch.full(
                    (rows, 1, 1, 2),
                    self.value,
                    dtype=torch.bfloat16,
                ),
            }

    layers = [FakeLayer(0), FakeLayer(1)]
    model = types.SimpleNamespace(
        model=types.SimpleNamespace(
            layers=layers,
            embed_tokens=FakeEmbedding(),
            norm=FakeNorm(),
        ),
        lm_head=FakeLmHead(),
    )
    layer_segments = [FakeLayerSegment(1), FakeLayerSegment(2)]
    segment = Glm5WholeModelSegment(
        model=model,
        device=torch.device("cpu"),
        world_size=2,
        max_pages_per_seq=1,
        max_aux_pages_per_seq=1,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        max_bucket_size=2,
        max_seqlen=16,
        layer_segments=layer_segments,
    )

    specs = segment.get_static_input_specs(2)
    assert "num_valid_tokens" in specs
    assert "flashmla_tile_scheduler_metadata" not in specs
    assert "flashmla_num_splits" not in specs

    segment.setup_static_buffers(2)
    outputs = segment.forward(
        input_ids=torch.tensor([[1], [2]], dtype=torch.long),
        cache_seqlens=torch.tensor([4, 5], dtype=torch.int32),
        position_ids=torch.tensor([[3], [4]], dtype=torch.long),
        primary_slot_indices=torch.tensor([0, 1], dtype=torch.int32),
        aux_slot_indices=torch.tensor([0, 1], dtype=torch.int32),
        rank_token_counts=torch.tensor([2, 2], dtype=torch.int64),
        num_valid_tokens=torch.tensor([2], dtype=torch.int32),
    )

    assert [layer_segment.calls for layer_segment in layer_segments] == [1, 1]
    assert [layer_segment.setup_bucket for layer_segment in layer_segments] == [2, 2]
    assert outputs["logits"][0, :hidden_size].tolist() == [4.0, 4.0, 4.0, 4.0]
    assert outputs["logits"][1, :hidden_size].tolist() == [5.0, 5.0, 5.0, 5.0]
    assert segment.primary_kv_offload_buffers[1]["key"][0, 0, 0, 0].item() == 2.0
    assert segment.aux_kv_offload_buffers[1]["key"][0, 0, 0, 0].item() == 2.0
    segment.release_static_buffers(2)
    assert [layer_segment.release_bucket for layer_segment in layer_segments] == [2, 2]
    assert segment._kv_buffers is None
    assert segment._aux_kv_buffers is None
    assert segment._kv_key_buffer is None
    assert segment._aux_kv_key_buffer is None
    assert segment.primary_kv_offload_buffers is None
    assert segment.aux_kv_offload_buffers is None


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


def test_glm5_prefill_indexer_kv_passes_prepacked_max_seqlen(monkeypatch):
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
            assert max_seqlen == 4096
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


def test_glm5_prefill_offloads_packed_primary_and_indexer_once(monkeypatch):
    class DummyTask:
        pass

    class DummyView:
        def __init__(self):
            self.calls = []

        def async_offload_packed_layer_kv_to_host(self, **kwargs):
            self.calls.append(kwargs)
            return DummyTask()

    primary_view = DummyView()
    auxiliary_view = DummyView()
    wrapper = object.__new__(GLM5AttnWrapper)
    wrapper.layer_idx = 7
    wrapper.prepack_seq_lengths = [2, 1, 3]
    wrapper.prepack_num_sequences = 3
    wrapper.cur_batch = [101, 202, 303]
    wrapper.core_engine = types.SimpleNamespace(
        host_paged_kv_worker_view=primary_view
    )

    monkeypatch.setattr(
        AttnWrapperBase, "host_paged_kv_worker_view_aux", auxiliary_view
    )
    monkeypatch.setattr(AttnWrapperBase, "pending_prefill_offload_tasks", [])
    monkeypatch.setattr(AttnWrapperBase, "pending_prefill_offload_tensors", [])
    monkeypatch.setattr(
        AttnWrapperBase, "pending_prefill_offload_layer_idx", None
    )
    monkeypatch.setattr(
        torch.cuda,
        "Event",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Python producer event must not be created")
        ),
    )

    wrapper._offload_prepacked_indexer_kv(torch.zeros(6, 1, 128))
    wrapper._offload_prepacked_kv(torch.zeros(6, 576))

    assert len(auxiliary_view.calls) == 1
    assert len(primary_view.calls) == 1
    assert auxiliary_view.calls[0]["sequence_ids"] == [101, 202, 303]
    assert auxiliary_view.calls[0]["sequence_lengths"] == [2, 1, 3]
    assert auxiliary_view.calls[0]["k_tensor"].shape == (6, 1, 128)
    assert primary_view.calls[0]["sequence_ids"] == [101, 202, 303]
    assert primary_view.calls[0]["sequence_lengths"] == [2, 1, 3]
    assert primary_view.calls[0]["k_tensor"].shape == (6, 1, 576)
    assert len(AttnWrapperBase.pending_prefill_offload_tasks) == 2
    assert len(AttnWrapperBase.pending_prefill_offload_tensors) == 2


def test_glm5_prefill_packed_offload_rejects_metadata_mismatch(monkeypatch):
    wrapper = object.__new__(GLM5AttnWrapper)
    wrapper.layer_idx = 7
    wrapper.prepack_seq_lengths = [2, 1]
    wrapper.cur_batch = [101, 202]
    wrapper.core_engine = types.SimpleNamespace(
        host_paged_kv_worker_view=object()
    )

    with pytest.raises(RuntimeError, match="primary KV token count mismatch"):
        wrapper._offload_prepacked_kv(torch.zeros(4, 576))


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


def test_prefill_offload_retirement_does_not_synchronize_device(monkeypatch):
    class DummyTask:
        def wait(self):
            return None

    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("device-wide synchronization is redundant")
        ),
    )
    monkeypatch.setattr(
        AttnWrapperBase, "pending_prefill_offload_tasks", [DummyTask()]
    )
    monkeypatch.setattr(
        AttnWrapperBase, "pending_prefill_offload_tensors", [torch.zeros(1)]
    )
    monkeypatch.setattr(
        AttnWrapperBase, "pending_prefill_offload_layer_idx", 3
    )

    assert AttnWrapperBase.retire_pending_prefill_offloads(
        device=torch.device("cuda:0")
    ) == 1


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
