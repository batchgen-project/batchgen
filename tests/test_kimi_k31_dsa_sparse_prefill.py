# ---------------------------------------------------------------------------- #
#  BatchGen — Kimi-K3.1                                                        #
#  Copyright (c) 2025-2026 BatchGen Team                                       #
#  Licensed under the Apache License, Version 2.0                              #
# ---------------------------------------------------------------------------- #
"""Kimi-K3.1 DSA sparse prefill (``kimi_linear/dsa_sparse_prefill.py``), on CPU.

The module reaches three GPU launches through seams: the causal scorer +
top-k (``deep_gemm.fp8_mqa_logits`` + ``fast_topk_2048_out``), the head gates
and FlashAttention-3 with ``qv``. Each is replaced by the CPU reference in
``tests/kimi_k31_dsa_refs.py``; the orchestration under test is the
production one.

The tests prove, on CPU:
  * packed causal ranges stay inside each packed row and include the query;
  * a packed batch whose rows all fit in ``index_topk`` gives the dense K3
    prefill's output when forced down the sparse path (the sparse plumbing is
    the dense math);
  * the selection the module hands attention is the reference oracle's
    (``kimi_k3/dsa_indexer.py``) top-k, so the production Q / K / gate wiring
    is the model's;
  * every query token attends to exactly its selected keys, against an
    independent expanded-head attention computed straight from the weights;
  * the wrapper keeps the dense prefill when no row exceeds ``index_topk``,
    takes the sparse one otherwise, and writes the same index keys either way.

GPU-deferred: ``fp8_mqa_logits`` + ``fast_topk_2048_out`` equal the reference
ordering within FP8 tolerance, and FA3 ``qv`` with one row per query token at
K3's TP8 head count.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.kimi_k31_dsa_refs import (  # noqa: E402
    QUANT,
    fa3_ref,
    head_gates_ref,
    install_fla_shim,
    install_flash_attn_varlen,
    prefill_topk_ref,
)

install_fla_shim()

from batchgen.models.moonshotai.kimi_k3.dsa_indexer import dsa_attention_bias  # noqa: E402
from batchgen.models.moonshotai.kimi_linear import (  # noqa: E402
    dsa_index_rotation as rot,
    dsa_prefill,
    dsa_sparse_prefill as sp,
)
from batchgen.models.moonshotai.kimi_linear.config import KimiLinearConfig  # noqa: E402
from batchgen.models.moonshotai.kimi_linear.serving_modules import (  # noqa: E402
    mla_prefill_nope_prepacked,
)
from batchgen.models.wrappers import AttnWrapperBase  # noqa: E402

INDEX_TOPK = 8
D_IDX = 128
H_IDX = 4
HIDDEN = 64


class TopkRecorder:
    """The CPU scorer + top-k reference, recording what it was handed."""

    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        out = prefill_topk_ref(**kwargs)
        self.calls.append(dict(kwargs, result=out.clone()))
        return out


@pytest.fixture()
def seams(monkeypatch):
    recorder = TopkRecorder()
    monkeypatch.setattr(sp, "_prefill_index_topk", lambda: recorder)
    monkeypatch.setattr(sp, "_head_gates", lambda: head_gates_ref)
    monkeypatch.setattr(sp, "_fa3_with_kvcache", lambda: fa3_ref)
    monkeypatch.setattr(sp, "_sparse_fa3_num_splits", lambda: 1)
    monkeypatch.setattr(rot, "_hadamard_predicate", lambda: QUANT.indexer_applies_hadamard)
    monkeypatch.setattr(
        dsa_prefill, "_quantize_indexer_k_fp8", lambda: QUANT.quantize_indexer_k_fp8)
    install_flash_attn_varlen(monkeypatch)
    return recorder


def _config():
    text = {
        "vocab_size": 256, "hidden_size": HIDDEN, "intermediate_size": 128,
        "num_hidden_layers": 2, "rms_norm_eps": 1e-5,
        "num_attention_heads": 4, "num_key_value_heads": 4,
        "q_lora_rank": 32, "kv_lora_rank": 48,
        "qk_nope_head_dim": 16, "qk_rope_head_dim": 8, "v_head_dim": 16,
        "mla_use_nope": True, "mla_use_output_gate": True,
        "linear_attn_config": {
            "kda_layers": [1], "full_attn_layers": [2], "num_heads": 4,
            "head_dim": 16, "short_conv_kernel_size": 4,
            "use_full_rank_gate": True, "gate_lower_bound": -5.0,
        },
        "dsa_config": {
            "dsa_layers": [2], "index_topk": INDEX_TOPK, "index_head_dim": D_IDX,
            "index_num_heads": H_IDX, "index_rope_dim": 0,
            "index_k_norm_type": "layernorm", "index_topk_freq": 1,
            "index_skip_topk_offset": 1,
        },
    }
    return KimiLinearConfig.from_hf_dict({"model_type": "kimi_k31", "text_config": text})


def _module(seed=7):
    from batchgen.models.moonshotai.kimi_linear.model import KimiMLAAttention

    attn = KimiMLAAttention(_config(), 1).eval()
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in sorted(attn.named_parameters()):
            if name.endswith("layernorm.weight") or name.endswith("k_norm.weight"):
                p.fill_(1.0)
            elif name.endswith("k_norm.bias"):
                p.normal_(0.0, 0.1, generator=g)
            else:
                p.normal_(0.0, 0.2, generator=g)
    return attn


def _packed(lengths, seed=3):
    hidden = torch.randn(sum(lengths), HIDDEN, generator=torch.Generator().manual_seed(seed))
    cu = torch.tensor([0] + list(torch.tensor(lengths).cumsum(0).tolist()), dtype=torch.int32)
    return hidden, cu


def _sparse(module, lengths, seed=3):
    hidden, cu = _packed(lengths, seed)
    with torch.no_grad():
        keys = dsa_prefill.prefill_index_keys(module.indexer, hidden)
        out, offload = sp.dsa_sparse_prefill(module, hidden, keys, cu, max(lengths))
    return out, offload, hidden, cu, keys


# --------------------------------------------------------------------------- #
#  Packed causal ranges                                                       #
# --------------------------------------------------------------------------- #

def test_packed_causal_ranges_stay_inside_each_row_and_include_the_query():
    starts, ends = sp.packed_causal_ranges(torch.tensor([0, 3, 4, 9]), 9)
    assert starts.tolist() == [0, 0, 0, 3, 4, 4, 4, 4, 4]
    assert ends.tolist() == [1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert starts.dtype is ends.dtype is torch.int32
    assert starts.is_contiguous() and ends.is_contiguous()


@pytest.mark.parametrize("cu, total, match", [
    ([0, 3, 5], 6, "must run 0..6"),
    ([1, 3, 6], 6, "must run 0..6"),
    ([0, 3, 3, 6], 6, "at least one token"),
    ([0], 0, "must run"),
])
def test_packed_causal_ranges_refuse_a_malformed_layout(cu, total, match):
    with pytest.raises(ValueError, match=match):
        sp.packed_causal_ranges(torch.tensor(cu), total)


def test_offset_to_packed_global_keeps_padding():
    rel = torch.tensor([[0, 2, -1], [1, 0, -1]], dtype=torch.int32)
    got = sp.offset_to_packed_global(rel, torch.tensor([5, 9], dtype=torch.int32))
    assert got.tolist() == [[5, 7, -1], [10, 9, -1]]


def test_the_production_scorer_refuses_a_top_k_other_than_2048():
    with pytest.raises(NotImplementedError, match="fast_topk_2048"):
        sp._score_and_select_prefill(
            q_idx=torch.zeros(1, 1, 128), index_keys=torch.zeros(1, 128),
            head_gates=torch.zeros(1, 1), causal_starts=torch.zeros(1, dtype=torch.int32),
            causal_ends=torch.ones(1, dtype=torch.int32), max_seqlen=1, topk=1024)


def test_index_keys_of_the_wrong_shape_are_refused(seams):
    module = _module()
    hidden, cu = _packed([3, 2])
    with pytest.raises(ValueError, match="index_keys must be"):
        sp.dsa_sparse_prefill(module, hidden, torch.zeros(5, 64), cu, 3)


# --------------------------------------------------------------------------- #
#  Short rows: sparse == dense                                                #
# --------------------------------------------------------------------------- #

def test_rows_within_index_topk_on_the_sparse_path_equal_the_dense_prefill(seams):
    module = _module()
    lengths = [INDEX_TOPK, 3, 5]
    out, offload, hidden, cu, _ = _sparse(module, lengths)
    with torch.no_grad():
        dense_out, dense_offload = mla_prefill_nope_prepacked(
            module, hidden, None, cu, max(lengths), len(lengths))
    assert torch.equal(offload, dense_offload)
    assert torch.allclose(out, dense_out, rtol=1e-4, atol=1e-5)
    # the selection is every causal key, packed-global, then padding
    picked = seams.calls[0]["result"]
    starts, ends = sp.packed_causal_ranges(cu, hidden.shape[0])
    for i in range(hidden.shape[0]):
        n = int(ends[i] - starts[i])
        assert picked[i, :n].tolist() == list(range(int(starts[i]), int(ends[i])))
        assert bool((picked[i, n:] == -1).all())


# --------------------------------------------------------------------------- #
#  Long rows: the oracle's selection, and attention over exactly it           #
# --------------------------------------------------------------------------- #

def _oracle_scores(module, hidden, cu):
    """Per packed row: the reference oracle's aggregate scores [S, S] (fp32,
    causal-masked to -inf), computed from the index keys the serving path
    stores (bf16) so both sides score the same keys."""
    idx = module.indexer
    rows = []
    with torch.no_grad():
        for s, e in zip(cu.tolist(), cu.tolist()[1:]):
            h = hidden[s:e].unsqueeze(0)
            q_a = module.q_a_layernorm(module.q_a_proj(h))
            keys = dsa_prefill.prefill_index_keys(idx, hidden[s:e]).float().unsqueeze(0)
            n = e - s
            mask = torch.full((n, n), torch.finfo(torch.float32).min).triu(1)[None, None]
            scores = idx.scores(q_a, h, keys, mask)[0]
            rows.append(scores.masked_fill(mask[0, 0] < 0, float("-inf")))
    return rows


def test_the_selection_is_the_reference_oracles_top_k(seams):
    module = _module()
    lengths = [21, 5, 14]
    _, _, hidden, cu, _ = _sparse(module, lengths)
    picked = seams.calls[0]["result"]
    for r, scores in enumerate(_oracle_scores(module, hidden, cu)):
        base = int(cu[r])
        for q in range(scores.shape[0]):
            row = picked[base + q]
            got = sorted(int(t) - base for t in row.tolist() if t >= 0)
            visible = q + 1
            assert len(got) == min(visible, INDEX_TOPK)
            assert all(0 <= t <= q for t in got)
            if visible <= INDEX_TOPK:
                assert got == list(range(visible))
                continue
            # tie-aware: everything clearly above the k-th score is picked,
            # nothing clearly below it is
            s = scores[q, :visible]
            kth = s.topk(INDEX_TOPK).values[-1]
            tol = 1e-2 * max(1.0, float(kth.abs()))
            must = {int(t) for t in torch.nonzero(s > kth + tol).flatten()}
            may = {int(t) for t in torch.nonzero(s >= kth - tol).flatten()}
            assert must <= set(got) <= may, (r, q)


def test_the_oracle_selection_is_the_dsa_attention_bias(seams):
    """Cross-check the tie-aware comparison above against the oracle's own
    entry point: the bias it builds keeps exactly the selected keys."""
    module = _module()
    lengths = [17]
    _, _, hidden, cu, _ = _sparse(module, lengths)
    picked = seams.calls[0]["result"]
    with torch.no_grad():
        h = hidden.unsqueeze(0)
        q_a = module.q_a_layernorm(module.q_a_proj(h))
        n = hidden.shape[0]
        mask = torch.full((n, n), torch.finfo(torch.float32).min).triu(1)[None, None]
        bias = dsa_attention_bias(module.indexer, q_a, h, mask)[0, 0]
    agree = 0
    for q in range(n):
        oracle = set(torch.nonzero(bias[q] == 0).flatten().tolist())
        got = {int(t) for t in picked[q].tolist() if t >= 0}
        assert len(oracle) == len(got) == min(q + 1, INDEX_TOPK)
        agree += oracle == got
    # bf16 keys vs fp32 keys may swap a near-tie; most rows agree exactly
    assert agree >= n - 2


def _expanded_attention(module, hidden, q_tok, keys_tok):
    """One query token over a given key set, in the EXPANDED (per-head K/V)
    form, straight from the weights — independent of the absorbed FA3 form."""
    m = module
    with torch.no_grad():
        h = hidden[q_tok:q_tok + 1]
        q = m.q_b_proj(m.q_a_layernorm(m.q_a_proj(h))).view(m.num_heads, m.q_head_dim)
        ckv = m.kv_a_proj_with_mqa(hidden[keys_tok])
        c, k_pe = ckv[:, : m.kv_lora_rank], ckv[:, m.kv_lora_rank:]
        kv = m.kv_b_proj(m.kv_a_layernorm(c)).view(len(keys_tok), m.num_heads, -1)
        k_nope, v = kv[..., : m.qk_nope_head_dim], kv[..., m.qk_nope_head_dim:]
        k = torch.cat([k_nope, k_pe[:, None, :].expand(-1, m.num_heads, -1)], -1)
        score = torch.einsum("hd,khd->hk", q, k) * m.scaling
        o = torch.einsum("hk,khd->hd", torch.softmax(score, -1), v).reshape(1, -1)
        o = o * m.g_proj(h).sigmoid()
        return m.o_proj(o)[0]


def test_every_query_attends_to_exactly_its_selected_keys(seams):
    module = _module()
    lengths = [21, 5, 14]
    out, _, hidden, _, _ = _sparse(module, lengths)
    picked = seams.calls[0]["result"]
    for t in range(hidden.shape[0]):
        keys = [int(k) for k in picked[t].tolist() if k >= 0]
        expected = _expanded_attention(module, hidden, t, keys)
        assert torch.allclose(out[t], expected, rtol=1e-4, atol=1e-5), t


def test_a_long_query_is_not_its_dense_prefill(seams):
    """Guard against a vacuous suite: past ``index_topk`` the sparse output
    differs from the dense one."""
    module = _module()
    lengths = [21]
    out, _, hidden, cu, _ = _sparse(module, lengths)
    with torch.no_grad():
        dense, _ = mla_prefill_nope_prepacked(module, hidden, None, cu, 21, 1)
    assert torch.allclose(out[:INDEX_TOPK], dense[:INDEX_TOPK], rtol=1e-4, atol=1e-5)
    assert not torch.allclose(out[INDEX_TOPK:], dense[INDEX_TOPK:], rtol=1e-3, atol=1e-4)


def test_the_scorer_receives_the_stored_keys_and_the_models_gates(seams):
    module = _module()
    lengths = [12, 9]
    _, _, hidden, cu, keys = _sparse(module, lengths)
    call = seams.calls[0]
    assert torch.equal(call["index_keys"], keys.to(torch.bfloat16))
    assert call["q_idx"].dtype is torch.bfloat16
    assert call["q_idx"].shape == (hidden.shape[0], H_IDX, D_IDX)
    with torch.no_grad():
        gates = head_gates_ref(hidden, module.indexer.weights_proj.weight, H_IDX, D_IDX)
        q_a = module.q_a_layernorm(module.q_a_proj(hidden))
        q_idx = F.linear(q_a, module.indexer.wq_b.weight).view(-1, H_IDX, D_IDX)
    assert torch.equal(call["head_gates"], gates)
    assert torch.equal(call["q_idx"], q_idx.to(torch.bfloat16))
    assert call["max_seqlen"] == 12 and call["topk"] == INDEX_TOPK


# --------------------------------------------------------------------------- #
#  Wrapper: dense below index_topk, sparse above, keys written either way     #
# --------------------------------------------------------------------------- #

def _run_wrapper(monkeypatch, module, lengths):
    from batchgen.models.moonshotai.kimi_linear.wrappers import KimiLinearAttnWrapper

    hidden, cu = _packed(lengths)
    monkeypatch.setattr(AttnWrapperBase, "prefill_pool_batch", None, raising=False)
    monkeypatch.setattr(AttnWrapperBase, "prepack_mode", True, raising=False)
    monkeypatch.setattr(AttnWrapperBase, "prepack_cu_seqlens", cu, raising=False)
    monkeypatch.setattr(AttnWrapperBase, "prepack_max_seqlen", max(lengths), raising=False)
    monkeypatch.setattr(AttnWrapperBase, "prepack_num_sequences", len(lengths), raising=False)
    monkeypatch.setattr(AttnWrapperBase, "position_ids", None, raising=False)
    monkeypatch.setattr(AttnWrapperBase, "retire_pending_prefill_offloads_before_layer",
                        staticmethod(lambda *a, **k: None), raising=False)
    trace = []
    real_sparse = sp.dsa_sparse_prefill

    def sparse(*args, **kwargs):
        trace.append("sparse")
        return real_sparse(*args, **kwargs)

    def dense(*args, **kwargs):
        trace.append("dense")
        return mla_prefill_nope_prepacked(module, *args, **kwargs)

    monkeypatch.setattr(sp, "dsa_sparse_prefill", sparse)
    monkeypatch.setattr(module, "mla_prefill_nope_prepacked", dense, raising=False)
    wrapper = SimpleNamespace(
        module=module, layer_idx=1,
        _prefill_needs_selection=KimiLinearAttnWrapper._prefill_needs_selection,
        _pool_prefill_kwargs=lambda: {},
        _offload_prepacked_kv=lambda kv, cu_: trace.append(("latent", kv.shape[0])),
        _write_prefill_index_keys=lambda keys, cu_: trace.append(("index", keys.clone())),
    )
    with torch.no_grad():
        out = KimiLinearAttnWrapper._forward_prefill(wrapper, hidden.unsqueeze(0))
    return out, trace, hidden


def test_the_wrapper_keeps_the_dense_prefill_when_every_row_fits(seams, monkeypatch):
    module = _module()
    out, trace, hidden = _run_wrapper(monkeypatch, module, [INDEX_TOPK, 4])
    assert trace[0] == "dense" and "sparse" not in trace
    assert trace[1] == ("latent", hidden.shape[0])
    with torch.no_grad():
        expected = dsa_prefill.prefill_index_keys(module.indexer, hidden)
    assert trace[2][0] == "index" and torch.equal(trace[2][1], expected)
    assert out.shape == (1, hidden.shape[0], HIDDEN)


def test_the_wrapper_takes_the_sparse_prefill_past_index_topk(seams, monkeypatch):
    module = _module()
    out, trace, hidden = _run_wrapper(monkeypatch, module, [INDEX_TOPK + 1, 4])
    assert trace[0] == "sparse" and "dense" not in trace
    assert trace[1] == ("latent", hidden.shape[0])
    # the keys the sparse prefill scored are the keys written for decode
    assert torch.equal(seams.calls[0]["index_keys"], trace[2][1].to(torch.bfloat16))
    assert out.shape == (1, hidden.shape[0], HIDDEN)
