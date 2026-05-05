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
)
import batchgen.models.glm.glm5.model as glm5_model
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
    monkeypatch.setattr(
        GLM5AttnWrapper,
        "_dsa_cuda_graph_page_tables_match",
        lambda self, primary, aux: True,
    )

    old_debug = AttnWrapperBase.batchgen_debug
    AttnWrapperBase.batchgen_debug = {"glm5_dsa_graph_compare": True}
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

    assert actual is expected
    assert calls == {"return_debug": True, "compare": True}


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


def test_glm5_dsa_cuda_graph_replay_gate_requires_fixed_flashmla_length():
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
    assert not _glm5_dsa_cuda_graph_can_replay(
        torch.tensor([1, 4], dtype=torch.int32),
        max_seqlen=4,
        index_topk=index_topk,
        captured_max_seqlen=8,
    )
    assert not _glm5_dsa_cuda_graph_can_replay(
        torch.tensor([2, 4, 7], dtype=torch.int32),
        max_seqlen=7,
        index_topk=index_topk,
        captured_max_seqlen=8,
    )
    assert not _glm5_dsa_cuda_graph_can_replay(
        torch.tensor([1, 4], dtype=torch.int32),
        max_seqlen=0,
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


def test_glm5_whole_model_graph_policy_is_opt_in_and_glm_only():
    env = {"BATCHGEN_GLM5_WHOLE_MODEL_CUDA_GRAPH": "1"}

    assert glm5_whole_model_cuda_graph_requested_for_model(
        "zai-org/GLM-5-FP8", environ=env
    )
    assert not glm5_whole_model_cuda_graph_requested_for_model(
        "gpt-oss-120b", environ=env
    )
    assert not glm5_whole_model_cuda_graph_requested_for_model(
        "zai-org/GLM-5-FP8", environ={}
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


def test_glm5_graph_policy_tracks_segmented_and_any_requests():
    dsa_env = {"BATCHGEN_GLM5_DSA_CUDA_GRAPH": "1"}
    moe_env = {"BATCHGEN_GLM5_MOE_CUDA_GRAPH": "1"}
    whole_env = {"BATCHGEN_GLM5_WHOLE_MODEL_CUDA_GRAPH": "1"}
    compare_env = {"BATCHGEN_GLM5_WHOLE_MODEL_GRAPH_COMPARE": "1"}
    model_name = "zai-org/GLM-5-FP8"

    assert glm5_dsa_cuda_graph_requested_for_model(model_name, environ=dsa_env)
    assert glm5_moe_cuda_graph_requested_for_model(model_name, environ=moe_env)
    assert glm5_segmented_cuda_graph_requested_for_model(model_name, environ=dsa_env)
    assert glm5_segmented_cuda_graph_requested_for_model(model_name, environ=moe_env)
    assert not glm5_segmented_cuda_graph_requested_for_model(
        model_name, environ=whole_env
    )
    assert glm5_any_cuda_graph_requested_for_model(model_name, environ=dsa_env)
    assert glm5_any_cuda_graph_requested_for_model(model_name, environ=moe_env)
    assert glm5_any_cuda_graph_requested_for_model(model_name, environ=whole_env)
    assert glm5_any_cuda_graph_requested_for_model(model_name, environ=compare_env)
    assert not glm5_any_cuda_graph_requested_for_model("gpt-oss-120b", environ=whole_env)


def test_glm5_power2_graph_buckets_cover_local_batches_to_32():
    buckets = GLM5_POWER_OF_TWO_BUCKETS_32

    assert buckets == [1, 2, 4, 8, 16, 32]
    assert [
        glm5_cuda_graph_bucket_for_batch_size(batch_size, buckets)
        for batch_size in [0, 1, 2, 3, 4, 5, 8, 9, 16, 17, 32, 33]
    ] == [None, 1, 2, 4, 4, 8, 8, 16, 16, 32, 32, None]


def test_glm5_moe_power2_bucket_32_represents_global_512_rows():
    buckets = GLM5_POWER_OF_TWO_BUCKETS_32

    assert glm5_moe_graph_bucket_capacity(
        max_rank_batch_size=16,
        world_size=16,
        bucket_sizes=buckets,
    ) == (16, 256)
    assert glm5_moe_graph_bucket_capacity(
        max_rank_batch_size=17,
        world_size=16,
        bucket_sizes=buckets,
    ) == (32, 512)
    assert glm5_moe_graph_bucket_capacity(
        max_rank_batch_size=32,
        world_size=16,
        bucket_sizes=buckets,
    ) == (32, 512)
    assert glm5_moe_graph_bucket_capacity(
        max_rank_batch_size=33,
        world_size=16,
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


def test_glm5_segmented_graph_bucket_changes_do_not_request_recapture(monkeypatch):
    from batchgen.batchgen_worker import BatchGenWorker

    class FakeManager:
        def __init__(self, buckets):
            self._buckets = set(buckets)

        def has_bucket_for_all_segments(self, batch_size):
            return batch_size in self._buckets

    configured_buckets = [1, 2, 3, 7, 12, 24, 40, 80]
    worker = object.__new__(BatchGenWorker)
    worker.model_name = "zai-org/GLM-5-FP8"
    worker.args = types.SimpleNamespace(
        cuda_graph_max_bucket_size=80,
        cuda_graph_num_buckets=8,
    )
    worker._batchgen_debug = {}
    worker._glm5_dsa_graph_failed_buckets = set()
    worker._glm5_moe_graph_failed_buckets = set()
    worker._current_decode_local_batch_size = 17
    worker._current_decode_max_rank_batch_size = 17
    worker._cuda_graph_manager = FakeManager(configured_buckets)
    worker._glm5_moe_cuda_graph_manager = FakeManager(configured_buckets)
    monkeypatch.setenv("BATCHGEN_GLM5_DSA_CUDA_GRAPH", "1")
    monkeypatch.setenv("BATCHGEN_GLM5_MOE_CUDA_GRAPH", "1")
    monkeypatch.setattr(
        worker,
        "_glm5_dsa_graph_page_table_storage_changed",
        lambda: False,
    )

    assert not worker._glm5_dsa_graph_current_bucket_missing()
    assert not worker._glm5_moe_graph_current_bucket_missing()

    worker._current_decode_local_batch_size = 33
    worker._current_decode_max_rank_batch_size = 33

    assert not worker._glm5_dsa_graph_current_bucket_missing()
    assert not worker._glm5_moe_graph_current_bucket_missing()


def test_glm5_segmented_graph_existing_manager_missing_configured_bucket_requests_setup(
    monkeypatch,
):
    from batchgen.batchgen_worker import BatchGenWorker

    class FakeManager:
        def __init__(self, buckets):
            self._buckets = set(buckets)

        def has_bucket_for_all_segments(self, batch_size):
            return batch_size in self._buckets

    worker = object.__new__(BatchGenWorker)
    worker.model_name = "zai-org/GLM-5-FP8"
    worker.args = types.SimpleNamespace(
        cuda_graph_max_bucket_size=80,
        cuda_graph_num_buckets=8,
    )
    worker._batchgen_debug = {}
    worker._glm5_dsa_graph_failed_buckets = set()
    worker._glm5_moe_graph_failed_buckets = set()
    worker._current_decode_local_batch_size = 17
    worker._current_decode_max_rank_batch_size = 17
    worker._cuda_graph_manager = FakeManager([1, 2, 3, 7, 12, 24, 80])
    worker._glm5_moe_cuda_graph_manager = FakeManager([1, 2, 3, 7, 12, 24, 80])
    monkeypatch.setenv("BATCHGEN_GLM5_DSA_CUDA_GRAPH", "1")
    monkeypatch.setenv("BATCHGEN_GLM5_MOE_CUDA_GRAPH", "1")
    monkeypatch.setattr(
        worker,
        "_glm5_dsa_graph_page_table_storage_changed",
        lambda: False,
    )

    assert worker._glm5_dsa_graph_current_bucket_missing()
    assert worker._glm5_moe_graph_current_bucket_missing()


def test_glm5_segmented_graph_setup_missing_only_when_manager_absent_or_storage_changes(
    monkeypatch,
):
    from batchgen.batchgen_worker import BatchGenWorker

    worker = object.__new__(BatchGenWorker)
    worker.model_name = "zai-org/GLM-5-FP8"
    worker._batchgen_debug = {}
    worker._current_decode_local_batch_size = 8
    worker._current_decode_max_rank_batch_size = 8
    worker._cuda_graph_manager = None
    worker._glm5_moe_cuda_graph_manager = None
    monkeypatch.setenv("BATCHGEN_GLM5_DSA_CUDA_GRAPH", "1")
    monkeypatch.setenv("BATCHGEN_GLM5_MOE_CUDA_GRAPH", "1")

    assert worker._glm5_dsa_graph_current_bucket_missing()
    assert worker._glm5_moe_graph_current_bucket_missing()

    worker._cuda_graph_manager = object()
    worker._glm5_moe_cuda_graph_manager = object()
    monkeypatch.setattr(
        worker,
        "_glm5_dsa_graph_page_table_storage_changed",
        lambda: True,
    )

    assert worker._glm5_dsa_graph_current_bucket_missing()
    assert worker._cuda_graph_manager is None


def test_glm5_segmented_graph_single_capture_per_batch_after_manager_clear(
    monkeypatch,
):
    from batchgen.batchgen_worker import BatchGenWorker

    worker = object.__new__(BatchGenWorker)
    worker.model_name = "zai-org/GLM-5-FP8"
    worker._batchgen_debug = {}
    worker._current_decode_local_batch_size = 8
    worker._current_decode_max_rank_batch_size = 8
    worker._cuda_graph_manager = None
    worker._glm5_moe_cuda_graph_manager = None
    worker._glm5_dsa_graph_capture_attempted_for_batch = True
    worker._glm5_moe_graph_capture_attempted_for_batch = True
    monkeypatch.setenv("BATCHGEN_GLM5_DSA_CUDA_GRAPH", "1")
    monkeypatch.setenv("BATCHGEN_GLM5_MOE_CUDA_GRAPH", "1")
    monkeypatch.setattr(
        worker,
        "_glm5_dsa_graph_page_table_storage_changed",
        lambda: False,
    )

    assert not worker._glm5_segmented_graph_initial_capture_missing()
    assert not worker._glm5_dsa_graph_current_bucket_missing()
    assert not worker._glm5_moe_graph_current_bucket_missing()


def test_glm5_segmented_graph_blocks_generic_warmup_after_capture_attempts(
    monkeypatch,
):
    from batchgen.batchgen_worker import BatchGenWorker

    worker = object.__new__(BatchGenWorker)
    worker.model_name = "zai-org/GLM-5-FP8"
    worker._batchgen_debug = {}
    monkeypatch.setenv("BATCHGEN_GLM5_DSA_CUDA_GRAPH", "1")
    monkeypatch.setenv("BATCHGEN_GLM5_MOE_CUDA_GRAPH", "1")

    worker._glm5_dsa_graph_capture_attempted_for_batch = True
    worker._glm5_moe_graph_capture_attempted_for_batch = True

    assert worker._glm5_segmented_graph_capture_already_attempted_for_requested_paths()

    worker._glm5_moe_graph_capture_attempted_for_batch = False

    assert not worker._glm5_segmented_graph_capture_already_attempted_for_requested_paths()


def test_glm5_setup_cuda_graphs_does_not_recapture_after_manager_clear(
    monkeypatch,
):
    from batchgen.batchgen_worker import BatchGenWorker

    worker = object.__new__(BatchGenWorker)
    worker.rank = 0
    worker.model_name = "zai-org/GLM-5-FP8"
    worker.args = types.SimpleNamespace(
        cuda_graph_max_bucket_size=80,
        cuda_graph_num_buckets=8,
    )
    worker.model_config = types.SimpleNamespace(max_position_embeddings=131072)
    worker.torch_device = torch.device("cpu")
    worker._batchgen_debug = {}
    worker._cuda_graph_manager = None
    worker._glm5_moe_cuda_graph_manager = None
    worker._glm5_dsa_graph_capture_attempted_for_batch = True
    worker._glm5_moe_graph_capture_attempted_for_batch = True
    worker._current_decode_local_batch_size = 8
    worker._current_decode_max_rank_batch_size = 8
    monkeypatch.setenv("BATCHGEN_GLM5_DSA_CUDA_GRAPH", "1")
    monkeypatch.setenv("BATCHGEN_GLM5_MOE_CUDA_GRAPH", "1")

    moe_setup_calls = []
    monkeypatch.setattr(
        worker,
        "_setup_glm5_moe_cuda_graphs",
        lambda bucket_sizes: moe_setup_calls.append(tuple(bucket_sizes)),
    )

    worker._setup_cuda_graphs(types.SimpleNamespace())

    assert worker._cuda_graph_manager is None
    assert moe_setup_calls


def test_glm5_moe_setup_does_not_recapture_after_manager_clear(monkeypatch):
    from batchgen.batchgen_worker import BatchGenWorker

    worker = object.__new__(BatchGenWorker)
    worker.rank = 0
    worker.model_name = "zai-org/GLM-5-FP8"
    worker._batchgen_debug = {}
    worker._glm5_moe_cuda_graph_manager = None
    worker._glm5_moe_graph_capture_attempted_for_batch = True
    worker._current_decode_max_rank_batch_size = 8
    monkeypatch.setenv("BATCHGEN_GLM5_MOE_CUDA_GRAPH", "1")

    worker._setup_glm5_moe_cuda_graphs([1, 2, 4, 8])

    assert worker._glm5_moe_cuda_graph_manager is None


def test_glm5_graph_path_reason_marks_manager_cleared_after_capture(
    monkeypatch,
):
    from batchgen.batchgen_worker import BatchGenWorker

    worker = object.__new__(BatchGenWorker)
    worker.model_name = "zai-org/GLM-5-FP8"
    worker._batchgen_debug = {}
    worker._cuda_graph_manager = None
    worker._glm5_moe_cuda_graph_manager = None
    worker._glm5_dsa_graph_capture_attempted_for_batch = True
    worker._glm5_moe_graph_capture_attempted_for_batch = True
    monkeypatch.setenv("BATCHGEN_GLM5_DSA_CUDA_GRAPH", "1")
    monkeypatch.setenv("BATCHGEN_GLM5_MOE_CUDA_GRAPH", "1")

    assert worker._glm5_dsa_graph_path_state(8, None) == (
        "eager",
        None,
        "no_manager_after_initial_capture",
    )
    assert worker._glm5_moe_graph_path_state(8) == (
        "eager",
        None,
        "no_manager_after_initial_capture",
    )


def test_glm5_dsa_graph_page_table_change_after_capture_falls_back_eager(
    monkeypatch,
):
    from batchgen.batchgen_worker import BatchGenWorker

    worker = object.__new__(BatchGenWorker)
    worker.rank = 0
    worker.model_name = "zai-org/GLM-5-FP8"
    worker._batchgen_debug = {}
    worker._current_decode_local_batch_size = 8
    worker._cuda_graph_manager = object()
    worker._glm5_dsa_graph_capture_attempted_for_batch = True
    worker._glm5_dsa_graph_page_table_change_after_capture_logged = False
    monkeypatch.setenv("BATCHGEN_GLM5_DSA_CUDA_GRAPH", "1")
    monkeypatch.setattr(
        worker,
        "_glm5_dsa_graph_page_table_storage_changed",
        lambda: True,
    )

    assert not worker._glm5_dsa_graph_current_bucket_missing()
    assert worker._cuda_graph_manager is None
    assert worker._glm5_dsa_graph_page_table_change_after_capture_logged


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


def test_glm5_graph_path_state_reports_over_bucket_eager(monkeypatch):
    from batchgen.batchgen_worker import BatchGenWorker

    worker = object.__new__(BatchGenWorker)
    worker.model_name = "zai-org/GLM-5-FP8"
    worker._batchgen_debug = {}
    worker._cuda_graph_manager = types.SimpleNamespace(
        bucketing=BatchSizeBucketing([1, 2]),
    )
    worker._glm5_moe_cuda_graph_manager = types.SimpleNamespace(
        bucketing=BatchSizeBucketing([1, 2]),
    )
    worker._glm5_moe_graph_failed_buckets = set()
    monkeypatch.setenv("BATCHGEN_GLM5_DSA_CUDA_GRAPH", "1")
    monkeypatch.setenv("BATCHGEN_GLM5_MOE_CUDA_GRAPH", "1")

    assert worker._glm5_dsa_graph_path_state(3, object()) == (
        "eager",
        None,
        "over_bucket",
    )
    assert worker._glm5_moe_graph_path_state(3) == (
        "eager",
        None,
        "over_bucket",
    )


def test_glm5_whole_model_warmup_policy_allows_capture_with_queued_prefill():
    env = {"BATCHGEN_GLM5_WHOLE_MODEL_CUDA_GRAPH": "1"}

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
