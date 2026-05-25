import torch

from batchgen.attention.gqa import fa_extend


def test_gqa_extend_fa_passes_paged_extend_metadata(monkeypatch):
    calls = {}

    def fake_flash_with_kvcache(*args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return torch.ones_like(args[0])

    monkeypatch.setattr(fa_extend, "_USE_FA3", True)
    monkeypatch.setattr(
        fa_extend, "_flash_with_kvcache", fake_flash_with_kvcache
    )

    q = torch.zeros(5, 4, 8)
    k_cache = torch.zeros(3, 64, 1, 8)
    v_cache = torch.zeros(3, 64, 1, 8)
    cache_seqlens = torch.tensor([66, 67], dtype=torch.int32)
    page_table = torch.tensor([[0, 1], [2, -1]], dtype=torch.int32)
    cu_q = torch.tensor([0, 2, 5], dtype=torch.int32)
    cu_k = torch.tensor([0, 66, 133], dtype=torch.int32)

    output, lse = fa_extend.gqa_extend_fa(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        cache_seqlens=cache_seqlens,
        page_table=page_table,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=3,
        sliding_window=128,
    )

    assert lse is None
    assert torch.equal(output, torch.ones_like(q))
    assert calls["args"] == (q, k_cache, v_cache)
    assert calls["kwargs"]["page_table"] is page_table
    assert calls["kwargs"]["cache_seqlens"] is cache_seqlens
    assert calls["kwargs"]["cu_seqlens_q"] is cu_q
    assert calls["kwargs"]["cu_seqlens_k_new"] is cu_k
    assert calls["kwargs"]["max_seqlen_q"] == 3
    assert calls["kwargs"]["causal"] is True
    assert calls["kwargs"]["window_size"] == (127, 0)
    assert calls["kwargs"]["return_softmax_lse"] is False
