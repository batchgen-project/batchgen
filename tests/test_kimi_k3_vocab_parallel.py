"""Vocab-parallel embed_tokens / lm_head (DECODE_CONCURRENCY_PLAN.md slice 2).

The collectives are stubbed (single process); the tests pin the shard
bookkeeping: masked lookup outside this rank's rows contributes zero (the TP
all_reduce then sums exactly one non-zero contribution per token), and the
gathered head slices reassemble into the full-vocab logits in rank order.
"""
import pytest

torch = pytest.importorskip("torch")

from batchgen.models.moonshotai.kimi_linear import vocab_parallel as vp


def test_shard_bounds():
    assert vp.vocab_shard_bounds(163840, 8, 3) == (3 * 20480, 4 * 20480)
    with pytest.raises(RuntimeError):
        vp.vocab_shard_bounds(163841, 8, 0)


def test_embedding_masks_foreign_rows_and_sum_over_ranks_is_the_full_lookup(monkeypatch):
    torch.manual_seed(0)
    V, H, G = 24, 5, 4
    full = torch.randn(V, H)
    ids = torch.tensor([[0], [7], [23], [12]])
    monkeypatch.setattr(vp.dist if hasattr(vp, "dist") else torch.distributed,
                        "all_reduce", lambda t, group=None: None)
    total = torch.zeros(ids.shape + (H,))
    for c in range(G):
        s, e = vp.vocab_shard_bounds(V, G, c)
        emb = vp.VocabParallelEmbedding(full[s:e].clone(), s, tp_group=None)
        out = emb(ids)
        # rows this rank does not own are exactly zero
        owned = (ids >= s) & (ids < e)
        assert torch.equal(out[~owned], torch.zeros_like(out[~owned]))
        total += out
    torch.testing.assert_close(total, torch.nn.functional.embedding(ids, full))


def test_gathered_logits_reassemble_in_rank_order():
    torch.manual_seed(0)
    V, G, rows = 16, 4, 3
    full = torch.randn(rows, 1, V)
    gathered = torch.stack([full[..., c * 4:(c + 1) * 4] for c in range(G)], dim=0)
    assert gathered.shape == (G, rows, 1, 4)
    out = vp.assemble_gathered_logits(gathered, V)
    torch.testing.assert_close(out, full)


def test_lm_head_keeps_the_linear_call_contract(monkeypatch):
    torch.manual_seed(0)
    V, H, G, rows = 16, 6, 4, 3
    W = torch.randn(V, H)
    h = torch.randn(rows, 1, H)
    ref = torch.nn.functional.linear(h, W)
    shards = [W[c * 4:(c + 1) * 4] for c in range(G)]

    def fake_all_gather(out, local, group=None):
        # every rank's slice, rank-major
        for c in range(G):
            out[c].copy_(torch.nn.functional.linear(h, shards[c]))
    monkeypatch.setattr(torch.distributed, "all_gather_into_tensor", fake_all_gather)
    head = vp.VocabParallelLMHead(shards[1].clone(), V, G, tp_group=None)
    assert head.out_features == V and head.in_features == H and head.bias is None
    torch.testing.assert_close(head(h), ref)
