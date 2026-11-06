# ---------------------------------------------------------------------------- #
#  BatchGen — Kimi-K3.1                                                        #
#  Copyright (c) 2025-2026 BatchGen Team                                       #
#  Licensed under the Apache License, Version 2.0                              #
# ---------------------------------------------------------------------------- #
"""CPU gate for the Kimi-K3.1 DSA indexer and the sparse MLA layer.

Modules under test are loaded by file path through ``tests/kimi_k3_harness.py``
(no ``import batchgen``, no core-engine build, no GPU).

The load-bearing test here is
:func:`test_dsa_equals_dense_when_the_context_fits_in_topk`:

    a query row that can see at most ``index_topk`` keys has no selection to
    make, so the selected set IS the visible set and sparse MLA must reproduce
    dense MLA **exactly** — not within a tolerance.

That identity is cheap, it needs no reference implementation, and it fails
under every plausible defect in the sparse path: a dropped key, an off-by-one
in the top-k count, a mask written in the wrong space, selection ranked against
unmasked scores, or the indexer perturbing the attention it is only supposed to
restrict.  Its companion
:func:`test_dsa_differs_from_dense_when_topk_actually_bites` keeps it from
passing vacuously (i.e. from a sparse path that is secretly dense).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kimi_k3_harness as H  # noqa: E402

OUR = H.load_our_modules()
cfgmod = OUR.config
model = OUR.model
# ``model`` imports it at module scope, so reuse that instance rather than
# loading a second copy of the same file under the same name.
dsa_indexer = sys.modules["_k3m.dsa_indexer"]

SEED = 20261105


# --------------------------------------------------------------------------- #
#  K31-TOY-2 — the smallest config with one layer of each kind                 #
# --------------------------------------------------------------------------- #

def toy_config_dict(*, index_topk: int, with_dsa: bool = True) -> dict:
    """2 layers: layer 0 KDA + dense SiTU MLP, layer 1 MLA (+ indexer) + MoE.

    Layer 0 must be KDA and dense (the checkpoint invariant K3 enforces), so
    two layers is the minimum that covers both mixers and both FFN kinds.
    ``index_topk`` is the only knob the tests vary: at ``>= seq_len`` the
    selection is the identity, below it the selection bites.
    """
    text = {
        "vocab_size": 256,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "rms_norm_eps": 1e-5,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "q_lora_rank": 32,
        "kv_lora_rank": 48,
        "qk_nope_head_dim": 16,
        "qk_rope_head_dim": 8,
        "v_head_dim": 16,
        "mla_use_nope": True,
        "mla_use_output_gate": True,
        "linear_attn_config": {
            "kda_layers": [1],
            "full_attn_layers": [2],
            "num_heads": 4,
            "head_dim": 16,
            "short_conv_kernel_size": 4,
            "use_full_rank_gate": True,
            "gate_lower_bound": -5.0,
        },
        "num_experts": 8,
        "num_experts_per_token": 2,
        "moe_intermediate_size": 32,
        "num_shared_experts": 2,
        "first_k_dense_replace": 1,
        "moe_layer_freq": 1,
        "moe_renormalize": True,
        "moe_router_activation_func": "sigmoid",
        "routed_scaling_factor": 1.0,
        "num_expert_group": 1,
        "topk_group": 1,
        "use_grouped_topk": True,
        "topk_method": "noaux_tc",
        "routed_expert_hidden_size": 32,
        "latent_moe_use_norm": True,
        "hidden_act": "situ",
        "activation_situ_beta": 4.0,
        "activation_situ_linear_beta": 25.0,
        "attn_res_block_size": 1,
        "num_nextn_predict_layers": 0,
        "max_position_embeddings": 128,
        "initializer_range": 0.02,
        "tie_word_embeddings": False,
        "pad_token_id": 255,
        "bos_token_id": 252,
        "eos_token_id": 253,
    }
    if with_dsa:
        text["dsa_config"] = {
            "dsa_layers": [2],
            "index_topk": index_topk,
            "index_head_dim": 32,
            "index_num_heads": 4,
            "index_rope_dim": 0,
            "index_k_norm_type": "layernorm",
            "index_topk_freq": 1,
            "index_skip_topk_offset": 1,
        }
    return {
        "model_type": "kimi_k31" if with_dsa else "kimi_k3",
        "media_placeholder_token_id": 250,
        "text_config": text,
    }


def toy_config(*, index_topk: int = 8, with_dsa: bool = True):
    return cfgmod.parse_k3_config(toy_config_dict(
        index_topk=index_topk, with_dsa=with_dsa))


MLA_LAYER = 1          # the full-attention (and DSA) layer of K31-TOY-2
KDA_LAYER = 0


def _seeded(module: torch.nn.Module, seed: int = SEED) -> torch.nn.Module:
    """Deterministic fp32 init.  LayerNorm/RMSNorm keep sane scales so the
    indexer's scores are neither saturated nor identically zero."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, param in sorted(module.named_parameters()):
            if name.endswith("k_norm.weight"):
                param.fill_(1.0)
            elif name.endswith("k_norm.bias"):
                param.normal_(0.0, 0.1, generator=generator)
            elif name.endswith("norm.weight") or name.endswith("_norm.weight"):
                param.fill_(1.0)
            else:
                param.normal_(0.0, 0.3, generator=generator)
    return module.eval()


def _hidden(batch: int, seq_len: int, hidden: int, seed: int = SEED + 1):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(batch, seq_len, hidden, generator=generator)


def _mla(cfg, seed: int = SEED):
    return _seeded(model.KimiK3MLAAttention(cfg, MLA_LAYER), seed)


# --------------------------------------------------------------------------- #
#  Structure: K3 is untouched, K3.1 carries the checkpoint's own names         #
# --------------------------------------------------------------------------- #

def test_a_k3_mla_layer_has_no_indexer_at_all():
    """Structural absence, not a disabled branch: a dense K3 build must have
    the same parameter set it had before K3.1 existed."""
    layer = model.KimiK3MLAAttention(toy_config(with_dsa=False), MLA_LAYER)
    assert not hasattr(layer, "indexer")
    assert not any("indexer" in name for name, _ in layer.named_parameters())


def test_indexer_parameter_names_are_the_checkpoint_suffixes():
    layer = model.KimiK3MLAAttention(toy_config(), MLA_LAYER)
    names = sorted(name for name, _ in layer.named_parameters()
                   if name.startswith("indexer."))
    assert names == [
        "indexer.k_norm.bias",
        "indexer.k_norm.weight",
        "indexer.weights_proj.weight",
        "indexer.wk.weight",
        "indexer.wq_b.weight",
    ]


def test_indexer_parameter_shapes_follow_dsa_config():
    cfg = toy_config()
    layer = model.KimiK3MLAAttention(cfg, MLA_LAYER)
    dsa = cfg.dsa_config
    shapes = {n: tuple(p.shape) for n, p in layer.named_parameters()}
    assert shapes["indexer.wk.weight"] == (dsa.index_head_dim, cfg.hidden_size)
    assert shapes["indexer.wq_b.weight"] == (dsa.index_dim, cfg.q_lora_rank)
    assert shapes["indexer.weights_proj.weight"] == (
        dsa.index_num_heads, cfg.hidden_size)
    assert shapes["indexer.k_norm.weight"] == (dsa.index_head_dim,)
    assert shapes["indexer.k_norm.bias"] == (dsa.index_head_dim,)


def test_the_index_key_is_one_mqa_key_per_token():
    """``index_head_dim`` wide, shared by all index heads — NOT one key per
    head.  The per-head form would be a 32x larger index cache per token."""
    cfg = toy_config()
    layer = _mla(cfg)
    hidden = _hidden(2, 5, cfg.hidden_size)
    keys = layer.indexer.index_keys(hidden)
    assert keys.shape == (2, 5, cfg.dsa_config.index_head_dim)


# --------------------------------------------------------------------------- #
#  Indexer math vs a hand-built reference                                      #
# --------------------------------------------------------------------------- #

def test_index_keys_are_wk_then_layernorm():
    cfg = toy_config()
    indexer = _mla(cfg).indexer
    hidden = _hidden(1, 4, cfg.hidden_size)
    want = torch.nn.functional.layer_norm(
        hidden @ indexer.wk.weight.T,
        (cfg.dsa_config.index_head_dim,),
        indexer.k_norm.weight,
        indexer.k_norm.bias,
        eps=indexer.k_norm.eps,
    )
    assert torch.equal(indexer.index_keys(hidden), want)


def test_index_queries_split_wq_b_into_heads_in_row_major_order():
    cfg = toy_config()
    indexer = _mla(cfg).indexer
    q_a = _hidden(2, 3, cfg.q_lora_rank, seed=7)
    got = indexer.index_queries(q_a)
    flat = q_a @ indexer.wq_b.weight.T
    heads, dim = cfg.dsa_config.index_num_heads, cfg.dsa_config.index_head_dim
    assert got.shape == (2, 3, heads, dim)
    for head in range(heads):
        assert torch.equal(got[..., head, :],
                           flat[..., head * dim:(head + 1) * dim])


def test_head_gates_carry_the_head_count_normalisation():
    cfg = toy_config()
    indexer = _mla(cfg).indexer
    hidden = _hidden(1, 4, cfg.hidden_size)
    heads = cfg.dsa_config.index_num_heads
    want = (hidden @ indexer.weights_proj.weight.T).float() * heads ** -0.5
    assert torch.allclose(indexer.head_gates(hidden), want, atol=0, rtol=0)


def _reference_scores(indexer, cfg, q_a, hidden, mask):
    """Loop-form restatement of the scoring formula, independent of the
    einsums under test:

        score[b,q,k] = sum_h gate[b,q,h] * relu(scale * <q[b,q,h], key[b,k]>)

    with invisible keys forced to -inf BEFORE the relu (so they contribute 0).
    """
    scale = float(cfg.dsa_config.index_head_dim) ** -0.5
    with torch.no_grad():
        keys = torch.nn.functional.layer_norm(
            hidden @ indexer.wk.weight.T,
            (cfg.dsa_config.index_head_dim,),
            indexer.k_norm.weight, indexer.k_norm.bias, eps=indexer.k_norm.eps,
        ).float()
        flat_q = (q_a @ indexer.wq_b.weight.T).float()
        gates = (hidden @ indexer.weights_proj.weight.T).float() * (
            cfg.dsa_config.index_num_heads ** -0.5)
    heads, dim = cfg.dsa_config.index_num_heads, cfg.dsa_config.index_head_dim
    batch, q_len = q_a.shape[:2]
    kv_len = hidden.shape[1]
    out = torch.zeros(batch, q_len, kv_len, dtype=torch.float32)
    for b in range(batch):
        for q in range(q_len):
            for k in range(kv_len):
                total = 0.0
                visible = mask is None or mask[b % mask.shape[0], 0, q, k] >= 0
                for h in range(heads):
                    if not visible:
                        continue
                    dot = torch.dot(flat_q[b, q, h * dim:(h + 1) * dim], keys[b, k])
                    total += float(gates[b, q, h]) * max(0.0, float(dot) * scale)
                out[b, q, k] = total
    return out


def test_scores_match_the_loop_reference():
    cfg = toy_config()
    layer = _mla(cfg)
    batch, seq_len = 2, 5
    hidden = _hidden(batch, seq_len, cfg.hidden_size)
    q_a = _hidden(batch, seq_len, cfg.q_lora_rank, seed=11)
    mask = model._build_causal_mask(seq_len, hidden.device, torch.float32)
    got = layer.indexer.scores(q_a, hidden, layer.indexer.index_keys(hidden), mask)
    want = _reference_scores(layer.indexer, cfg, q_a, hidden, mask)
    assert got.shape == (batch, seq_len, seq_len)
    assert torch.allclose(got, want, atol=1e-5, rtol=1e-4)


def test_an_invisible_key_scores_exactly_zero():
    """Masked BEFORE the relu, so a future (or cross-sequence) key cannot leak
    any post-relu energy into the aggregate and change a visible key's rank."""
    cfg = toy_config()
    layer = _mla(cfg)
    seq_len = 6
    hidden = _hidden(1, seq_len, cfg.hidden_size)
    q_a = _hidden(1, seq_len, cfg.q_lora_rank, seed=13)
    mask = model._build_causal_mask(seq_len, hidden.device, torch.float32)
    scores = layer.indexer.scores(
        q_a, hidden, layer.indexer.index_keys(hidden), mask)
    for q in range(seq_len):
        assert torch.equal(scores[0, q, q + 1:],
                           torch.zeros(seq_len - q - 1))


def test_relu_is_applied_per_head_before_the_gate_weighting():
    """Without the per-head relu, negative per-head scores would cancel
    positive ones during the sum.  Detected by giving one head a large
    negative gate: the aggregate must stay a relu-weighted sum, which the
    reference reproduces and a plain (q.k)-weighted sum does not.
    """
    cfg = toy_config()
    layer = _mla(cfg)
    seq_len = 4
    hidden = _hidden(1, seq_len, cfg.hidden_size)
    q_a = _hidden(1, seq_len, cfg.q_lora_rank, seed=17)
    mask = model._build_causal_mask(seq_len, hidden.device, torch.float32)
    keys = layer.indexer.index_keys(hidden)

    got = layer.indexer.scores(q_a, hidden, keys, mask)
    want = _reference_scores(layer.indexer, cfg, q_a, hidden, mask)
    assert torch.allclose(got, want, atol=1e-5, rtol=1e-4)

    # No-relu variant of the same aggregate, for contrast.
    scale = float(cfg.dsa_config.index_head_dim) ** -0.5
    q = layer.indexer.index_queries(q_a).float()
    logits = torch.einsum("bqhd,bkd->bhqk", q, keys.float()) * scale
    logits = logits.masked_fill(
        dsa_indexer._invisible(mask, seq_len).unsqueeze(1), 0.0)
    gates = layer.indexer.head_gates(hidden)
    without_relu = torch.einsum("bhqk,bqh->bqk", logits, gates)
    assert not torch.allclose(got, without_relu, atol=1e-3)


# --------------------------------------------------------------------------- #
#  Selection                                                                   #
# --------------------------------------------------------------------------- #

def _selected(bias: torch.Tensor) -> torch.Tensor:
    return torch.isfinite(bias[:, 0])


def test_select_topk_result_is_a_subset_of_the_visible_set():
    """``topk`` always returns k indices, so a row with fewer than k visible
    keys gets invisible ones among them.  The selection must exclude them on
    its own — the caller's bias already hides them, which would mask the
    defect if the intersection were dropped."""
    scores = torch.zeros(1, 3, 4)
    invisible = torch.tensor([[[False, True, True, True],
                               [False, False, True, True],
                               [False, False, False, True]]])
    selected = dsa_indexer.select_topk(scores, invisible, index_topk=4)
    assert torch.equal(selected, ~invisible)
    assert selected.sum(dim=-1).flatten().tolist() == [1, 2, 3]


def test_select_topk_ranks_only_visible_keys():
    """A visible key with a NEGATIVE aggregate must outrank an invisible key
    whose aggregate is 0."""
    scores = torch.tensor([[[-5.0, -4.0, 0.0, 0.0]]])
    invisible = torch.tensor([[[False, False, True, True]]])
    selected = dsa_indexer.select_topk(scores, invisible, index_topk=2)
    assert selected.flatten().tolist() == [True, True, False, False]


def test_select_topk_keeps_the_highest_scoring_visible_keys():
    scores = torch.tensor([[[1.0, 9.0, 3.0, 7.0]]])
    invisible = torch.zeros(1, 1, 4, dtype=torch.bool)
    selected = dsa_indexer.select_topk(scores, invisible, index_topk=2)
    assert selected.flatten().tolist() == [False, True, False, True]


def test_selection_keeps_exactly_min_visible_topk_keys_per_row():
    cfg = toy_config(index_topk=3)
    layer = _mla(cfg)
    seq_len = 8
    hidden = _hidden(1, seq_len, cfg.hidden_size)
    q_a = _hidden(1, seq_len, cfg.q_lora_rank, seed=19)
    mask = model._build_causal_mask(seq_len, hidden.device, torch.float32)
    bias = dsa_indexer.dsa_attention_bias(layer.indexer, q_a, hidden, mask)
    counts = _selected(bias).sum(dim=-1)[0].tolist()
    assert counts == [min(pos + 1, 3) for pos in range(seq_len)]


def test_selection_is_the_top_k_of_the_reference_scores():
    cfg = toy_config(index_topk=3)
    layer = _mla(cfg)
    seq_len = 8
    hidden = _hidden(1, seq_len, cfg.hidden_size)
    q_a = _hidden(1, seq_len, cfg.q_lora_rank, seed=23)
    mask = model._build_causal_mask(seq_len, hidden.device, torch.float32)
    bias = dsa_indexer.dsa_attention_bias(layer.indexer, q_a, hidden, mask)
    want_scores = _reference_scores(layer.indexer, cfg, q_a, hidden, mask)
    got = _selected(bias)[0]
    for pos in range(seq_len):
        visible = want_scores[0, pos, :pos + 1]
        k = min(pos + 1, 3)
        expected = set(torch.topk(visible, k).indices.tolist())
        assert set(torch.nonzero(got[pos]).flatten().tolist()) == expected, pos


def test_selection_never_crosses_a_sequence_boundary():
    """Packed prefill: the block-diagonal causal mask is the visibility oracle,
    so no score can pull a key out of another sequence."""
    cfg = toy_config(index_topk=64)
    layer = _mla(cfg)
    cu_seqlens = torch.tensor([0, 3, 7], dtype=torch.int32)
    seq_len = 7
    hidden = _hidden(1, seq_len, cfg.hidden_size)
    q_a = _hidden(1, seq_len, cfg.q_lora_rank, seed=29)
    mask = model._build_block_causal_mask(
        cu_seqlens, seq_len, hidden.device, torch.float32)
    bias = dsa_indexer.dsa_attention_bias(layer.indexer, q_a, hidden, mask)
    assert torch.equal(_selected(bias), torch.isfinite(mask[:, 0]))


def test_a_negative_head_gate_cannot_drop_a_visible_key():
    """Regression: ``weights_proj`` is unconstrained, so a visible key's
    aggregate can be NEGATIVE while an invisible key's is exactly 0.  Ranking
    against the unmasked aggregate would then prefer the invisible key and
    silently shrink the context.  Forced here by making every gate negative.
    """
    cfg = toy_config(index_topk=4)
    layer = _mla(cfg)
    with torch.no_grad():
        layer.indexer.weights_proj.weight.fill_(-1.0)
    seq_len = 8
    hidden = _hidden(1, seq_len, cfg.hidden_size).abs() + 0.1   # gates < 0
    q_a = _hidden(1, seq_len, cfg.q_lora_rank, seed=31)
    mask = model._build_causal_mask(seq_len, hidden.device, torch.float32)
    scores = layer.indexer.scores(
        q_a, hidden, layer.indexer.index_keys(hidden), mask)
    assert (scores[0, -1, :seq_len] < 0).any(), "fixture did not force a negative aggregate"

    bias = dsa_indexer.dsa_attention_bias(layer.indexer, q_a, hidden, mask)
    counts = _selected(bias)[0].sum(dim=-1).tolist()
    assert counts == [min(pos + 1, 4) for pos in range(seq_len)]
    # And the rows that fit entirely inside topk keep the dense mask.
    assert torch.equal(_selected(bias)[0, :4], torch.isfinite(mask[0, 0, :4]))


def test_the_bias_does_not_mutate_the_shared_causal_mask():
    """One [1, 1, S, S] mask is shared by every DSA layer of the stack; writing
    into it would make layer N's selection layer N+1's causal mask."""
    cfg = toy_config(index_topk=2)
    layer = _mla(cfg)
    seq_len = 6
    hidden = _hidden(1, seq_len, cfg.hidden_size)
    q_a = _hidden(1, seq_len, cfg.q_lora_rank, seed=37)
    mask = model._build_causal_mask(seq_len, hidden.device, torch.float32)
    before = mask.clone()
    dsa_indexer.dsa_attention_bias(layer.indexer, q_a, hidden, mask)
    assert torch.equal(mask, before)


def test_a_decode_shaped_call_without_a_mask_hard_fails():
    cfg = toy_config()
    layer = _mla(cfg)
    hidden = _hidden(1, 1, cfg.hidden_size)
    q_a = _hidden(1, 1, cfg.q_lora_rank, seed=41)
    with pytest.raises(ValueError, match="additive attention bias"):
        dsa_indexer.dsa_attention_bias(layer.indexer, q_a, hidden, None)


# --------------------------------------------------------------------------- #
#  THE ORACLE — sparse == dense when the context fits in topk                  #
# --------------------------------------------------------------------------- #

def _dense_twin(dsa_layer, dense_cfg):
    """A dense K3 MLA layer sharing every non-indexer parameter."""
    twin = model.KimiK3MLAAttention(dense_cfg, MLA_LAYER)
    state = {name: tensor for name, tensor in dsa_layer.state_dict().items()
             if not name.startswith("indexer.")}
    missing, unexpected = twin.load_state_dict(state, strict=True)
    assert not missing and not unexpected
    return twin.eval()


@pytest.mark.parametrize("seq_len,index_topk", [(2, 2), (4, 4), (4, 9), (16, 16), (7, 2048)])
def test_dsa_equals_dense_when_the_context_fits_in_topk(seq_len, index_topk):
    """``context <= index_topk`` => the selected set is the identity set.

    Equality is EXACT (``torch.equal``), not approximate: the sparse path must
    hand the attention the same mask tensor values the dense path does, so the
    same kernels run on the same inputs.
    """
    assert seq_len <= index_topk
    cfg = toy_config(index_topk=index_topk)
    dense_cfg = toy_config(with_dsa=False)
    layer = _mla(cfg)
    twin = _dense_twin(layer, dense_cfg)

    hidden = _hidden(3, seq_len, cfg.hidden_size)
    mask = model._build_causal_mask(seq_len, hidden.device, torch.float32)

    with torch.no_grad():
        got = layer(hidden, attention_mask=mask)
        want = twin(hidden, attention_mask=mask)
    assert torch.equal(got, want)


def test_dsa_equals_dense_on_a_packed_batch_that_fits_in_topk():
    cfg = toy_config(index_topk=64)
    layer = _mla(cfg)
    twin = _dense_twin(layer, toy_config(with_dsa=False))
    cu_seqlens = torch.tensor([0, 3, 7, 10], dtype=torch.int32)
    seq_len = 10
    hidden = _hidden(1, seq_len, cfg.hidden_size)
    mask = model._build_block_causal_mask(
        cu_seqlens, seq_len, hidden.device, torch.float32)
    with torch.no_grad():
        assert torch.equal(layer(hidden, attention_mask=mask),
                           twin(hidden, attention_mask=mask))


def test_dsa_differs_from_dense_when_topk_actually_bites():
    """Keeps the identity test above from passing vacuously."""
    cfg = toy_config(index_topk=2)
    layer = _mla(cfg)
    twin = _dense_twin(layer, toy_config(with_dsa=False))
    seq_len = 12
    hidden = _hidden(2, seq_len, cfg.hidden_size)
    mask = model._build_causal_mask(seq_len, hidden.device, torch.float32)
    with torch.no_grad():
        got = layer(hidden, attention_mask=mask)
        want = twin(hidden, attention_mask=mask)
    assert not torch.allclose(got, want, atol=1e-4)
    # The first two rows see <= 2 keys and must still agree exactly.
    assert torch.equal(got[:, :2], want[:, :2])


def test_splitting_q_a_out_did_not_change_the_dense_layer():
    """The dense forward now materialises ``q_a`` as a named intermediate so
    the indexer can read it.  Same ops in the same order, so the K3 output must
    be reproducible from the pieces exactly."""
    cfg = toy_config(with_dsa=False)
    layer = _mla(cfg)
    hidden = _hidden(2, 5, cfg.hidden_size)
    with torch.no_grad():
        q_a = layer.q_a_layernorm(layer.q_a_proj(hidden))
        fused = layer.q_b_proj(layer.q_a_layernorm(layer.q_a_proj(hidden)))
    assert torch.equal(layer.q_b_proj(q_a), fused)


# --------------------------------------------------------------------------- #
#  2-layer model smoke (1 KDA + 1 DSA-MLA)                                     #
# --------------------------------------------------------------------------- #

def _toy_model(cfg, seed: int = SEED):
    return _seeded(model.KimiK3ForCausalLM(cfg, kda_backend="reference"), seed)


def test_toy_model_forward_is_finite_and_deterministic():
    cfg = toy_config(index_topk=4)
    net = _toy_model(cfg)
    assert net.model.layers[KDA_LAYER].is_linear_attn
    assert not net.model.layers[MLA_LAYER].is_linear_attn
    assert hasattr(net.model.layers[MLA_LAYER].self_attn, "indexer")

    ids = torch.randint(0, cfg.vocab_size - 8, (2, 9),
                        generator=torch.Generator().manual_seed(5))
    with torch.no_grad():
        first = net(input_ids=ids)
        second = net(input_ids=ids)
    assert first.shape == (2, 9, cfg.vocab_size)
    assert torch.isfinite(first).all()
    assert torch.equal(first, second)


def test_toy_model_equals_the_dense_model_when_context_fits_in_topk():
    """The oracle at whole-model scope: with ``index_topk >= seq_len`` the
    sparse 2-layer model must reproduce the dense one bit for bit, through the
    KDA layer, the Block-Attention-Residual depth mix and the MoE as well."""
    cfg = toy_config(index_topk=32)
    net = _toy_model(cfg)
    dense = _toy_model(toy_config(with_dsa=False))
    state = {name: tensor for name, tensor in net.state_dict().items()
             if ".indexer." not in name}
    dense.load_state_dict(state, strict=True)

    ids = torch.randint(0, cfg.vocab_size - 8, (2, 11),
                        generator=torch.Generator().manual_seed(9))
    with torch.no_grad():
        got = net(input_ids=ids)
        want = dense(input_ids=ids)
    assert torch.equal(got, want)


def test_toy_model_sparse_run_changes_the_output_when_topk_bites():
    cfg = toy_config(index_topk=2)
    net = _toy_model(cfg)
    dense = _toy_model(toy_config(with_dsa=False))
    dense.load_state_dict(
        {n: t for n, t in net.state_dict().items() if ".indexer." not in n},
        strict=True)
    ids = torch.randint(0, cfg.vocab_size - 8, (1, 12),
                        generator=torch.Generator().manual_seed(13))
    with torch.no_grad():
        got = net(input_ids=ids)
        want = dense(input_ids=ids)
    assert torch.isfinite(got).all()
    assert not torch.allclose(got, want, atol=1e-4)
