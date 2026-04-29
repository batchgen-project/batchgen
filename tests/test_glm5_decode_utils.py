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
from batchgen.models.glm.glm5.wrappers import (
    _fail_if_glm5_dsa_cuda_graph_required_without_replay,
)


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
    page_size = 64
    blocked_k = torch.zeros(4, page_size, 1, 1, dtype=torch.float32)

    blocked_k[0, :, 0, 0] = 1000 + torch.arange(page_size)
    blocked_k[1, 0, 0, 0] = 2000
    blocked_k[1, 1:, 0, 0] = 9000 + torch.arange(page_size - 1)
    blocked_k[2, :, 0, 0] = 3000 + torch.arange(page_size)
    blocked_k[3, :, 0, 0] = 4000 + torch.arange(page_size)

    block_table = torch.tensor([[0, 1, -1], [2, 3, -1]], dtype=torch.int64)
    cache_seqlens = torch.tensor([65, 128], dtype=torch.int32)

    clamped_indices = build_clamped_dense_token_indices(
        cache_seqlens,
        max_seqlen=128,
        device=torch.device("cpu"),
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
    page_size = 4
    blocked_k = torch.zeros(2, page_size, 1, 1, dtype=torch.float32)
    blocked_k[0, :, 0, 0] = torch.tensor([100.0, 101.0, 102.0, 103.0])
    blocked_k[1, :, 0, 0] = torch.tensor([200.0, 201.0, 202.0, 203.0])

    slot_order_block_table = torch.tensor([[1, -1], [0, -1]], dtype=torch.int64)
    top_k_indices = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)

    wrong = sparse_gather_from_paged_kv(
        blocked_k, slot_order_block_table, top_k_indices, page_size
    ).squeeze(-1).squeeze(-1)

    reordered = reorder_block_table_to_batch_slots(
        slot_order_block_table,
        torch.tensor([1, 0], dtype=torch.int32),
    )
    fixed = sparse_gather_from_paged_kv(
        blocked_k, reordered, top_k_indices, page_size
    ).squeeze(-1).squeeze(-1)

    assert wrong.tolist() == [[200.0, 201.0], [100.0, 101.0]]
    assert fixed.tolist() == [[100.0, 101.0], [200.0, 201.0]]


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
