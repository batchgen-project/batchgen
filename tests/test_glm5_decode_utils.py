import torch

from batchgen.attention.dsa.sparse_gather import sparse_gather_from_paged_kv
from batchgen.models.glm.glm5.decode_utils import (
    build_clamped_dense_token_indices,
    clamp_token_indices_to_seqlens,
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
