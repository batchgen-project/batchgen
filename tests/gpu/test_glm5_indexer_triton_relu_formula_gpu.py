"""GPU twin of ``tests/test_glm5_indexer_triton_relu_formula.py``.

Runs the compiled bf16 Triton indexer scorers (the GLM-5 / GLM-5.1 decode
path) at production head geometry and checks their aggregate against the fp64
reference ``sum_h g_h * relu(q_h . k_t)``, then checks the top-k they select
against the reference top-k.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA device and Triton"
)

H, D, TOPK = 32, 128, 2048


def _reference(q, k_rows, gates, lens):
    B, max_seqlen = k_rows.shape[0], k_rows.shape[1]
    out = torch.full((B, max_seqlen), float("-inf"), dtype=torch.float64, device=q.device)
    for b in range(B):
        n = int(lens[b])
        per_head = torch.relu(q[b].double() @ k_rows[b, :n].double().T)
        out[b, :n] = (gates[b].double()[:, None] * per_head).sum(dim=0)
    return out


def _inputs(B, max_seqlen, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(B, H, D, generator=gen, device="cuda").to(torch.bfloat16)
    k = torch.randn(B, max_seqlen, D, generator=gen, device="cuda").to(torch.bfloat16)
    gates = torch.randn(B, H, generator=gen, device="cuda") * (H ** -0.5) * (D ** -0.5)
    return q, k, gates


def _check(agg, top_k, ref, lens):
    finite = torch.isfinite(ref)
    assert torch.equal(torch.isfinite(agg), finite)
    torch.testing.assert_close(agg.double()[finite], ref[finite], rtol=2e-3, atol=2e-3)
    ref_top = torch.topk(ref, TOPK, dim=-1).indices
    for b in range(ref.shape[0]):
        got = set(top_k[b].tolist())
        want = set(ref_top[b].tolist())
        # fp32 accumulation vs fp64 may swap exact near-ties at the threshold.
        assert len(got & want) / TOPK >= 0.995, (b, len(got & want))


@pytest.mark.parametrize("max_seqlen", [4096, 16384])
def test_dense_triton_scorer_matches_relu_reference(max_seqlen):
    from batchgen_kernels.attention.dsa.fused_indexer_score import fused_score_and_topk_out

    B = 4
    q, k, gates = _inputs(B, max_seqlen, seed=max_seqlen)
    lens = torch.tensor([max_seqlen, max_seqlen - 17, TOPK + 1, max_seqlen // 2],
                        dtype=torch.int32, device="cuda")
    agg = torch.empty(B, max_seqlen, dtype=torch.float32, device="cuda")
    top_k = torch.empty(B, TOPK, dtype=torch.int32, device="cuda")
    fused_score_and_topk_out(q, k, gates, lens, agg, top_k, topk=TOPK)
    torch.cuda.synchronize()
    _check(agg, top_k, _reference(q, k, gates, lens), lens)


def test_graph_slot_triton_scorer_matches_relu_reference():
    from batchgen_kernels.attention.dsa.fused_indexer_score import (
        fused_paged_score_and_topk_with_slots_out,
    )

    B, max_seqlen, page_size = 4, 8192, 64
    q, k, gates = _inputs(B, max_seqlen, seed=7)
    lens = torch.tensor([8192, 5000, TOPK + 3, 7777], dtype=torch.int32, device="cuda")
    pages = max_seqlen // page_size
    num_pages = B * pages + 5
    perm = torch.randperm(num_pages, device="cuda")[: B * pages].view(B, pages).int()
    cache = torch.zeros(num_pages, page_size, 1, D, dtype=torch.bfloat16, device="cuda")
    for b in range(B):
        cache[perm[b].long()] = k[b].view(pages, page_size, 1, D)
    slot_indices = torch.tensor([2, 0, 3, 1], dtype=torch.int32, device="cuda")
    slot_table = torch.empty_like(perm)
    slot_table[slot_indices.long()] = perm
    agg = torch.empty(B, max_seqlen, dtype=torch.float32, device="cuda")
    top_k = torch.empty(B, TOPK, dtype=torch.int32, device="cuda")
    fused_paged_score_and_topk_with_slots_out(
        q, cache, slot_table, slot_indices, gates, lens, agg, top_k,
        topk=TOPK, page_size=page_size, max_seqlen=max_seqlen,
    )
    torch.cuda.synchronize()
    _check(agg, top_k, _reference(q, k, gates, lens), lens)
