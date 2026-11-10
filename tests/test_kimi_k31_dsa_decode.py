# ---------------------------------------------------------------------------- #
#  BatchGen — Kimi-K3.1                                                        #
#  Copyright (c) 2025-2026 BatchGen Team                                       #
#  Licensed under the Apache License, Version 2.0                              #
# ---------------------------------------------------------------------------- #
"""Slices 2-3 of the Kimi-K3.1 DSA serving path: sparse decode, on CPU.

``kimi_linear/dsa_decode.py`` drives five GPU kernels through module-level
seams. Here each seam is replaced by a CPU reference built from the kernel
package's OWN CPU statements wherever one exists:

==========================  ==================================================
seam                        CPU reference
==========================  ==================================================
fused index-K writer        ``plan_norm_rope_write`` (preconditions) +
                            ``indexer_k_norm_rope_reference`` +
                            ``quantize_indexer_k_fp8`` +
                            ``page_split_write_reference``
page-split paged scorer     ``quantize_q_e4m3_reference`` + a dequantizing
                            reader (round-trip-tested against the writer
                            reference) + ``topk_from_logits_reference``
head gates                  ``fused_indexer_score.compute_head_gates``'s two
                            lines (that module imports triton)
selection transform         the Triton kernel's arithmetic, line for line
FA3 / FlashMLA              plain softmax attention over the addressed tokens
==========================  ==================================================

So the orchestration under test is the production one; only the arithmetic
inside each launch is a reference. GPU-deferred: each compiled kernel equals
its reference (the on-node checks), FA3's ``qv`` MLA form equals FlashMLA's
576-wide latent attention to bf16 rounding.

The tests prove, on CPU:
  * an all-short batch IS the dense K3 decode (same call, bit-identical);
  * the index key of the new token is written before scoring, at the right
    sequence's page, through the aux slot indirection;
  * long rows select exactly the emulated page-split top-k;
  * sparse attention over that selection equals an independent reference
    computed straight from the weights and the latent cache;
  * a SHORT row decoded on the sparse path equals the dense decode of that row
    (the sparse plumbing is the dense math);
  * the FP8 page-split ordering agrees with the ReLU-gated fp32 oracle;
  * the wrapper appends both Host regions, and refuses to append one alone.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


from tests.kimi_k31_dsa_refs import (  # noqa: E402
    PML,
    QUANT,
    fa3_ref,
    head_gates_ref,
    install_fla_shim,
)

install_fla_shim()

from batchgen.models.moonshotai.kimi_linear import (  # noqa: E402
    dsa_decode,
    dsa_index_rotation as rot,
)
from batchgen.models.moonshotai.kimi_linear.config import KimiLinearConfig  # noqa: E402
from batchgen.models.wrappers import AttnWrapperBase  # noqa: E402

PAGE = 4          # small pages, so selections cross page boundaries
INDEX_TOPK = 8
D_IDX = 128
H_IDX = 4


# --------------------------------------------------------------------------- #
#  CPU references for each kernel seam                                        #
# --------------------------------------------------------------------------- #

def _page_bytes(aux_cache):
    return int(aux_cache.shape[1]) * int(aux_cache.shape[3])


def _read_page_split(aux_cache, slots):
    """Inverse of ``page_split_write_reference``: dequantized keys at slots."""
    num_pages = aux_cache.shape[0]
    page_bytes = _page_bytes(aux_cache)
    flat = aux_cache.view(torch.uint8).reshape(num_pages * page_bytes)
    slots = slots.to(torch.long)
    page_idx, offset = slots // PAGE, slots % PAGE
    k_base = page_idx * page_bytes + offset * D_IDX
    k_bytes = flat[(k_base[:, None] + torch.arange(D_IDX)[None, :]).reshape(-1)]
    k_fp8 = k_bytes.view(torch.float8_e4m3fn).reshape(-1, D_IDX)
    s_base = page_idx * page_bytes + PAGE * D_IDX + offset * 4
    s_bytes = flat[(s_base[:, None] + torch.arange(4)[None, :]).reshape(-1)]
    scales = s_bytes.view(torch.float32).reshape(-1)
    return k_fp8, scales


def _physical_slots(page_table_row, positions):
    positions = positions.to(torch.long)
    return page_table_row[positions // PAGE].to(torch.long) * PAGE + positions % PAGE


class FusedWriterRef:
    def __init__(self):
        self.calls = []

    def __call__(self, *, aux_cache, k_raw, ln_weight, ln_bias, ln_eps, cos_cache,
                 sin_cache, positions, page_table, slot_indices, token_indices,
                 page_size_tokens, apply_hadamard, k_fp8_out, scales_out,
                 num_valid_tokens=None):
        QUANT.plan_norm_rope_write(aux_cache, k_raw, ln_weight, ln_bias,
                                   cos_cache, sin_cache, positions, page_size_tokens)
        keys = QUANT.indexer_k_norm_rope_reference(
            k_raw, ln_weight, ln_bias, ln_eps, cos_cache, sin_cache, positions,
            apply_hadamard)
        k_fp8, scales = QUANT.quantize_indexer_k_fp8(keys)
        # rows at or beyond num_valid_tokens write nothing (graph padding);
        # the host-append sources still cover every row
        rows = k_raw.shape[0] if num_valid_tokens is None else int(num_valid_tokens.reshape(()))
        if rows:
            slots = torch.stack([
                _physical_slots(page_table[int(s)], token_indices[i:i + 1])[0]
                for i, s in enumerate(slot_indices[:rows].tolist())
            ])
            QUANT.page_split_write_reference(
                aux_cache.view(torch.uint8).reshape(aux_cache.shape[0], _page_bytes(aux_cache)),
                slots, k_fp8[:rows], scales[:rows], PAGE)
        k_fp8_out.view(-1, D_IDX).copy_(k_fp8)
        scales_out.copy_(scales)
        self.calls.append(dict(slot_indices=slot_indices.clone(),
                               token_indices=token_indices.clone(),
                               apply_hadamard=apply_hadamard,
                               num_valid_tokens=num_valid_tokens))
        return k_fp8_out, scales_out


def paged_scorer_ref(*, q, aux_blocked_k, block_table, head_gates, cache_seqlens,
                     topk, max_seqlen, index_head_dim, page_size):
    q8 = PML.quantize_q_e4m3_reference(q).float()[:, 0]          # [B, H, D]
    logits = torch.zeros(q.shape[0], int(max_seqlen))
    for b in range(q.shape[0]):
        length = int(cache_seqlens[b])
        k_fp8, scales = _read_page_split(
            aux_blocked_k, _physical_slots(block_table[b], torch.arange(length)))
        k = k_fp8.float() * scales[:, None]
        per_head = torch.relu(q8[b] @ k.T)                          # [H, L]
        logits[b, :length] = (head_gates[b][:, None] * per_head).sum(0)
    return PML.topk_from_logits_reference(logits, cache_seqlens.to(torch.int32), topk)


def transform_ref(page_table, cache_seqlens, long_topk, physical, lengths, *, page_size,
                  primary_slot_indices=None, num_valid_tokens=None):
    """``_transform_selected_positions_kernel``'s arithmetic, line for line."""
    batch, index_topk = long_topk.shape
    for row in range(batch):
        slot = row if primary_slot_indices is None else int(primary_slot_indices[row])
        seqlen = int(cache_seqlens[row])
        selected = min(seqlen, index_topk)
        if num_valid_tokens is not None and row >= int(num_valid_tokens.reshape(())):
            selected = 0
        lengths[row] = selected
        for off in range(index_topk):
            logical = int(long_topk[row, off]) if seqlen > index_topk else off
            ok = off < selected and 0 <= logical < seqlen
            if ok:
                page = int(page_table[slot, logical // page_size])
                ok = page >= 0
            physical[row, off] = page * page_size + logical % page_size if ok else -1
    return physical, lengths


def _attend(q, k, v, scale):
    weights = torch.softmax((q @ k.T) * scale, dim=-1)
    return weights @ v


def _install_flash_mla(monkeypatch):
    """A CPU ``flash_mla`` for the UNCHANGED dense K3 decode."""
    def get_mla_metadata(cache_seqlens, num_heads, _):
        return None, None

    def flash_mla_with_kvcache(query, blocked_k, block_table, cache_seqlens, d_v,
                               meta, splits, scale, causal):
        batch, _, heads, _ = query.shape
        out = torch.zeros(batch, 1, heads, d_v)
        for b in range(batch):
            length = int(cache_seqlens[b])
            slots = _physical_slots(block_table[b], torch.arange(length))
            latent = blocked_k.view(-1, blocked_k.shape[-1])[slots].float()
            for h in range(heads):
                out[b, 0, h] = _attend(query[b, 0, h].float(), latent, latent[:, :d_v], scale)
        return out.to(query.dtype), None

    module = types.ModuleType("flash_mla")
    module.get_mla_metadata = get_mla_metadata
    module.flash_mla_with_kvcache = flash_mla_with_kvcache
    monkeypatch.setitem(sys.modules, "flash_mla", module)


@pytest.fixture()
def writer(monkeypatch):
    ref = FusedWriterRef()
    monkeypatch.setattr(dsa_decode, "_fused_index_k_writer", lambda: ref)
    monkeypatch.setattr(dsa_decode, "_paged_scorer", lambda: paged_scorer_ref)
    monkeypatch.setattr(dsa_decode, "_head_gates", lambda: head_gates_ref)
    monkeypatch.setattr(dsa_decode, "_selected_positions", lambda: transform_ref)
    monkeypatch.setattr(dsa_decode, "_fa3_with_kvcache", lambda: fa3_ref)
    monkeypatch.setattr(dsa_decode, "_sparse_fa3_num_splits", lambda: 1)
    monkeypatch.setattr(rot, "_hadamard_predicate", lambda: QUANT.indexer_applies_hadamard)
    _install_flash_mla(monkeypatch)
    return ref


# --------------------------------------------------------------------------- #
#  Model + caches                                                             #
# --------------------------------------------------------------------------- #

def _config():
    text = {
        "vocab_size": 256, "hidden_size": 64, "intermediate_size": 128,
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


LATENT = 48 + 8


class FakePagedKV:
    """Just enough of a GPU paged KV manager for one layer on CPU."""

    def __init__(self, *, num_pages, dim, dtype, page_table, seq_id_to_slot=None):
        self.config = SimpleNamespace(page_size_tokens=PAGE)
        self.device = torch.device("cpu")
        self.blocked = torch.zeros(num_pages, PAGE, 1, dim, dtype=dtype)
        self.page_table = page_table.to(torch.int32)
        self._gpu_page_table_manager = SimpleNamespace(seq_id_to_slot=seq_id_to_slot or {})
        self.writes = []

    def get_layer_kv_with_page_table(self, layer_idx):
        return self.blocked, None, self.page_table

    def get_layer_kv_cache(self, layer_idx):
        return self.blocked, None

    def update_layer_decode_new_token(self, *, k_tensor, v_tensor, sequence_lengths,
                                      layer_idx, batch_slice=None, slot_indices=None):
        assert slot_indices is None and batch_slice is None
        flat = self.blocked.view(-1, self.blocked.shape[-1])
        for b in range(k_tensor.shape[0]):
            slot = _physical_slots(self.page_table[b], sequence_lengths[b:b + 1])[0]
            flat[slot] = k_tensor[b, 0, 0].to(flat.dtype)
        self.writes.append(sequence_lengths.clone())


def _caches(lengths, *, aux_slot_perm=None, seed=11):
    """Primary rows in batch order; aux rows in a PERMUTED slot order.

    History tokens ``[0, len - 1)`` are pre-filled: latent from random values,
    index keys through the shipped quantize + page-split write references.
    """
    g = torch.Generator().manual_seed(seed)
    batch = len(lengths)
    max_pages = max((n + PAGE - 1) // PAGE for n in lengths)
    num_pages = batch * max_pages + 3
    pages = torch.randperm(num_pages, generator=g)[: batch * max_pages].view(batch, max_pages)
    primary = FakePagedKV(num_pages=num_pages, dim=LATENT, dtype=torch.float32,
                          page_table=pages)
    perm = list(range(batch)) if aux_slot_perm is None else list(aux_slot_perm)
    seq_ids = [100 + b for b in range(batch)]
    aux_table = torch.zeros(batch, max_pages, dtype=torch.int32)
    for b in range(batch):
        aux_table[perm[b]] = pages[b]              # mirrored pages, permuted rows
    aux = FakePagedKV(num_pages=num_pages, dim=D_IDX + 4, dtype=torch.uint8,
                      page_table=aux_table,
                      seq_id_to_slot={seq_ids[b]: perm[b] for b in range(batch)})
    flat_primary = primary.blocked.view(-1, LATENT)
    aux_u8 = aux.blocked.view(torch.uint8).reshape(num_pages, PAGE * (D_IDX + 4))
    for b, n in enumerate(lengths):
        hist = torch.arange(n - 1)
        slots = _physical_slots(pages[b], hist)
        flat_primary[slots] = torch.randn(len(hist), LATENT, generator=g)
        keys = torch.randn(len(hist), D_IDX, generator=g).to(torch.bfloat16)
        k_fp8, scales = QUANT.quantize_indexer_k_fp8(keys)
        QUANT.page_split_write_reference(aux_u8, slots, k_fp8, scales, PAGE)
    return primary, aux, seq_ids


def _decode(module, lengths, *, aux_slot_perm=None, seed=11, hidden_seed=3):
    primary, aux, seq_ids = _caches(lengths, aux_slot_perm=aux_slot_perm, seed=seed)
    batch = len(lengths)
    hidden = torch.randn(batch, 1, 64, generator=torch.Generator().manual_seed(hidden_seed))
    positions = torch.tensor([[n - 1] for n in lengths])
    cache_seqlens = torch.tensor(lengths, dtype=torch.int32)
    with torch.no_grad():
        result = dsa_decode.dsa_decode_forward(
            module, hidden, positions, cache_seqlens, max(lengths),
            primary, aux, 1, seq_ids)
    return result, primary, aux, hidden, positions, cache_seqlens


# --------------------------------------------------------------------------- #
#  Self-checks of the references                                              #
# --------------------------------------------------------------------------- #

def test_page_split_reader_inverts_the_shipped_writer():
    num_pages = 3
    buf = torch.zeros(num_pages, PAGE, 1, D_IDX + 4, dtype=torch.uint8)
    keys = torch.randn(5, D_IDX).to(torch.bfloat16)
    k_fp8, scales = QUANT.quantize_indexer_k_fp8(keys)
    slots = torch.tensor([0, 5, 6, 11, 2])
    QUANT.page_split_write_reference(
        buf.view(torch.uint8).reshape(num_pages, PAGE * (D_IDX + 4)), slots, k_fp8, scales, PAGE)
    got_fp8, got_scales = _read_page_split(buf, slots)
    assert torch.equal(got_fp8.view(torch.uint8), k_fp8.view(torch.uint8))
    assert torch.equal(got_scales, scales)


# --------------------------------------------------------------------------- #
#  All-short: the dense K3 decode, exactly                                    #
# --------------------------------------------------------------------------- #

def test_an_all_short_batch_is_the_dense_k3_decode(writer):
    module = _module()
    lengths = [3, INDEX_TOPK, 5]                     # every row <= index_topk
    result, primary, aux, hidden, positions, seqlens = _decode(module, lengths)
    assert result.mode == "dense"

    twin_primary, _, _ = _caches(lengths)
    from batchgen.models.moonshotai.kimi_linear.serving_modules import (
        mla_decoding_nope_with_pagekv,
    )
    with torch.no_grad():
        dense_out, dense_k = mla_decoding_nope_with_pagekv(
            module, hidden, positions, seqlens, max(lengths), None, twin_primary, 1, None)
    assert torch.equal(result.attn_output, dense_out)
    assert torch.equal(result.k_tensor, dense_k)
    assert torch.equal(primary.blocked, twin_primary.blocked)


def test_the_index_key_is_written_even_when_nothing_is_scored(writer):
    """Later steps score this token; the aux cache must stay complete."""
    module = _module()
    lengths = [3, 5]
    result, _, aux, hidden, positions, _ = _decode(module, lengths)
    assert result.mode == "dense"
    assert len(writer.calls) == 1
    assert result.index_k_fp8.shape == (2, 1, 1, D_IDX)
    assert result.index_scales.shape == (2,)
    # Read back at each row's new-token position through ITS aux slot.
    for b, n in enumerate(lengths):
        slot = aux._gpu_page_table_manager.seq_id_to_slot[100 + b]
        k_fp8, scales = _read_page_split(
            aux.blocked, _physical_slots(aux.page_table[slot], torch.tensor([n - 1])))
        assert torch.equal(k_fp8.view(torch.uint8),
                           result.index_k_fp8[b].reshape(1, D_IDX).view(torch.uint8))
        assert torch.equal(scales, result.index_scales[b:b + 1])


def test_the_written_key_is_the_layernorm_key_with_no_rotation(writer):
    module = _module()
    result, *_rest = _decode(module, [4])
    hidden = _rest[2]
    with torch.no_grad():
        k_raw = F.linear(hidden[:, 0], module.indexer.wk.weight).to(torch.bfloat16)
        expected = module.indexer.k_norm.float()(k_raw.float())
    dequant = result.index_k_fp8[:, 0, 0].float() * result.index_scales[:, None]
    assert torch.allclose(dequant, expected, rtol=0.08, atol=2e-2)
    assert writer.calls[0]["apply_hadamard"] is False


# --------------------------------------------------------------------------- #
#  Aux slot indirection                                                       #
# --------------------------------------------------------------------------- #

def test_keys_land_on_each_sequences_own_aux_pages_through_a_permuted_slot_map(writer):
    module = _module()
    lengths = [5, 3, 6]
    result, _, aux, *_ = _decode(module, lengths, aux_slot_perm=[2, 0, 1])
    for b, n in enumerate(lengths):
        slot = aux._gpu_page_table_manager.seq_id_to_slot[100 + b]
        assert slot == [2, 0, 1][b]
        k_fp8, _ = _read_page_split(
            aux.blocked, _physical_slots(aux.page_table[slot], torch.tensor([n - 1])))
        assert torch.equal(k_fp8.view(torch.uint8),
                           result.index_k_fp8[b].reshape(1, D_IDX).view(torch.uint8))
    assert writer.calls[0]["slot_indices"].tolist() == [2, 0, 1]
    assert writer.calls[0]["token_indices"].tolist() == [4, 2, 5]


# --------------------------------------------------------------------------- #
#  Long rows: selection + sparse attention                                    #
# --------------------------------------------------------------------------- #

def _independent_selection(module, aux, hidden, seqlens, row, seq_id):
    """The page-split scorer's top-k for one row, recomputed from the module's
    weights and the aux cache bytes — not through dsa_decode."""
    idx = module.indexer
    with torch.no_grad():
        q_a = module.q_a_layernorm(F.linear(hidden[row:row + 1, 0], module.q_a_proj.weight))
        q = F.linear(q_a, idx.wq_b.weight).view(1, H_IDX, D_IDX).to(torch.bfloat16)
        gates = head_gates_ref(hidden[row:row + 1, 0], idx.weights_proj.weight, H_IDX, D_IDX)
    slot = aux._gpu_page_table_manager.seq_id_to_slot[seq_id]
    out = paged_scorer_ref(
        q=q, aux_blocked_k=aux.blocked, block_table=aux.page_table[slot:slot + 1],
        head_gates=gates, cache_seqlens=seqlens[row:row + 1], topk=INDEX_TOPK,
        max_seqlen=int(seqlens[row]), index_head_dim=D_IDX, page_size=PAGE)
    return sorted(out[0].tolist())


def _independent_attention(module, primary, hidden, row, tokens):
    """Reference MLA attention for one decode row over a given set of logical
    tokens, straight from the weights and the latent cache."""
    m = module
    with torch.no_grad():
        h = hidden[row:row + 1, 0]
        q = F.linear(m.q_a_layernorm(F.linear(h, m.q_a_proj.weight)), m.q_b_proj.weight)
        q = q.view(m.num_heads, m.q_head_dim)
        q_nope, q_pe = q[:, : m.qk_nope_head_dim], q[:, m.qk_nope_head_dim:]
        kv_b = m.kv_b_proj.weight.view(m.num_heads, -1, m.kv_lora_rank)
        q_abs = torch.einsum("hd,hdc->hc", q_nope, kv_b[:, : m.qk_nope_head_dim])
        slots = _physical_slots(primary.page_table[row], torch.tensor(tokens))
        latent = primary.blocked.view(-1, LATENT)[slots].float()
        c, k_pe = latent[:, : m.kv_lora_rank], latent[:, m.kv_lora_rank:]
        heads = []
        for hh in range(m.num_heads):
            score = (q_abs[hh] @ c.T + q_pe[hh] @ k_pe.T) * m.scaling
            heads.append(torch.softmax(score, -1) @ c)
        o = torch.einsum("hc,hdc->hd", torch.stack(heads), kv_b[:, m.qk_nope_head_dim:])
        o = o.reshape(1, -1) * F.linear(h, m.g_proj.weight).sigmoid()
        return F.linear(o, m.o_proj.weight)


def test_long_rows_select_the_page_split_top_k_and_attend_only_to_it(writer):
    module = _module()
    lengths = [20, 23]
    result, primary, aux, hidden, _, seqlens = _decode(module, lengths, aux_slot_perm=[1, 0])
    assert result.mode == "sparse"
    for row, n in enumerate(lengths):
        tokens = _independent_selection(module, aux, hidden, seqlens, row, 100 + row)
        assert len(tokens) == INDEX_TOPK and all(0 <= t < n for t in tokens)
        expected = _independent_attention(module, primary, hidden, row, tokens)
        assert torch.allclose(result.attn_output[row, 0], expected[0], rtol=1e-4, atol=1e-5)


def test_the_new_token_is_scorable_on_its_own_step(writer):
    """The key is written before the scorer reads the cache: with a selection
    that must include the newest token, it is found."""
    module = _module()
    lengths = [20]
    result, primary, aux, hidden, _, seqlens = _decode(module, lengths)
    tokens = _independent_selection(module, aux, hidden, seqlens, 0, 100)
    # The independent selection reads the cache AFTER the step; the step's own
    # selection must equal it, which is only possible if the new key was
    # already in place when the step scored.
    expected = _independent_attention(module, primary, hidden, 0, tokens)
    assert torch.allclose(result.attn_output[0, 0], expected[0], rtol=1e-4, atol=1e-5)


def test_a_short_row_on_the_sparse_path_equals_its_dense_decode(writer):
    """Mixed batch: the short row selects every key and must reproduce the
    dense FlashMLA decode of that row — the sparse plumbing (absorb, the
    selection transform, FA3's qv form, out-absorb, gate) IS the dense math."""
    module = _module()
    lengths = [6, 21]
    result, primary, aux, hidden, positions, seqlens = _decode(module, lengths)
    assert result.mode == "sparse"
    dense = _independent_attention(module, primary, hidden, 0, list(range(6)))
    assert torch.allclose(result.attn_output[0, 0], dense[0], rtol=1e-4, atol=1e-5)

    twin_primary, _, _ = _caches(lengths)
    from batchgen.models.moonshotai.kimi_linear.serving_modules import (
        mla_decoding_nope_with_pagekv,
    )
    with torch.no_grad():
        dense_out, _ = mla_decoding_nope_with_pagekv(
            module, hidden[:1], positions[:1], seqlens[:1], 6, None,
            _row_view(twin_primary, 0), 1, None)
    assert torch.allclose(result.attn_output[0], dense_out[0], rtol=1e-4, atol=1e-5)


def _row_view(manager, row):
    view = FakePagedKV(num_pages=manager.blocked.shape[0], dim=LATENT,
                       dtype=torch.float32, page_table=manager.page_table[row:row + 1])
    view.blocked = manager.blocked
    return view


# --------------------------------------------------------------------------- #
#  FP8 page-split ordering vs the ReLU-gated fp32 oracle                      #
# --------------------------------------------------------------------------- #

def test_fp8_page_split_ordering_matches_the_fp32_oracle():
    """The selection the kernel makes from e4m3 q and e4m3+scale keys, against
    the reference ``sum_h gate_h relu(scale q.k)`` in fp32 (the oracle's own
    ``KimiK3Indexer.scores``). Exact order equality is not expected at e4m3
    resolution; the top-k SETS must overlap almost entirely, and the
    clear winners must agree."""
    module = _module(seed=21)
    idx = module.indexer
    g = torch.Generator().manual_seed(4)
    length, topk = 512, 64
    hidden_hist = torch.randn(length, 64, generator=g)
    hidden_q = torch.randn(1, 64, generator=g)
    with torch.no_grad():
        q_a = module.q_a_layernorm(F.linear(hidden_q, module.q_a_proj.weight))
        keys = idx.index_keys(hidden_hist)
        oracle = idx.scores(q_a.view(1, 1, -1), hidden_q.view(1, 1, -1),
                            keys.view(1, length, -1))[0, 0]
        q = F.linear(q_a, idx.wq_b.weight).view(1, H_IDX, D_IDX).to(torch.bfloat16)
        gates = head_gates_ref(hidden_q, idx.weights_proj.weight, H_IDX, D_IDX)
    k_fp8, scales = QUANT.quantize_indexer_k_fp8(keys.to(torch.bfloat16))
    q8 = PML.quantize_q_e4m3_reference(q).float()[0, 0]
    k_deq = k_fp8.float() * scales[:, None]
    fp8_scores = (gates[0][:, None] * torch.relu(q8 @ k_deq.T)).sum(0)

    oracle_top = set(torch.topk(oracle, topk).indices.tolist())
    fp8_top = set(torch.topk(fp8_scores, topk).indices.tolist())
    overlap = len(oracle_top & fp8_top) / topk
    assert overlap >= 0.85, overlap
    # The clearest winners (top quarter) agree outright.
    clear = set(torch.topk(oracle, topk // 4).indices.tolist())
    assert clear <= fp8_top
    # And the kernel's math orders by the same scalar, up to the positive
    # head-count/dim factors relu(c x) = c relu(x) lets it fold in.
    corr = torch.corrcoef(torch.stack([oracle, fp8_scores]))[0, 1]
    assert corr > 0.98, float(corr)


# --------------------------------------------------------------------------- #
#  Refusals and edges                                                         #
# --------------------------------------------------------------------------- #

def test_missing_aux_manager_is_refused(writer):
    primary, _, seq_ids = _caches([3])
    with pytest.raises(RuntimeError, match="both the primary"):
        dsa_decode.dsa_decode_forward(
            _module(), torch.randn(1, 1, 64), torch.tensor([[2]]),
            torch.tensor([3], dtype=torch.int32), 3, primary, None, 1, seq_ids)


def test_multi_token_decode_is_refused(writer):
    primary, aux, seq_ids = _caches([3])
    with pytest.raises(ValueError, match="q_len=1"):
        dsa_decode.dsa_decode_forward(
            _module(), torch.randn(1, 2, 64), torch.tensor([[2]]),
            torch.tensor([3], dtype=torch.int32), 3, primary, aux, 1, seq_ids)


def test_a_sequence_without_an_aux_slot_is_refused(writer):
    primary, aux, _ = _caches([3])
    with pytest.raises(RuntimeError, match="no aux page-table slot"):
        dsa_decode.dsa_decode_forward(
            _module(), torch.randn(1, 1, 64), torch.tensor([[2]]),
            torch.tensor([3], dtype=torch.int32), 3, primary, aux, 1, [999])


def test_an_empty_batch_writes_nothing(writer):
    primary, aux, _ = _caches([3])
    result = dsa_decode.dsa_decode_forward(
        _module(), torch.empty(0, 1, 64), torch.empty(0, 1, dtype=torch.long),
        torch.empty(0, dtype=torch.int32), 0, primary, aux, 1, [])
    assert result.mode == "empty"
    assert writer.calls == []
    assert result.index_k_fp8 is None


def test_helpers():
    assert dsa_decode.batch_slot_indices([5, 7], {5: 1, 7: 0}, 2, "cpu").tolist() == [1, 0]
    with pytest.raises(RuntimeError, match="mismatch"):
        dsa_decode.batch_slot_indices([5], {5: 1}, 2, "cpu")
    table = torch.tensor([[1, 2], [3, 4], [5, 6]])
    ordered = dsa_decode.rows_in_batch_order(table, torch.tensor([2, 0]))
    assert ordered.tolist() == [[5, 6], [1, 2]] and ordered.dtype is torch.int32
    clamped = dsa_decode.clamp_to_last_token(torch.tensor([[0, 9]]), torch.tensor([4]))
    assert clamped.tolist() == [[0, 3]]


# --------------------------------------------------------------------------- #
#  Wrapper                                                                    #
# --------------------------------------------------------------------------- #

def _bind_decode(monkeypatch, primary, aux, seq_ids, lengths, *, primary_cb, aux_cb):
    monkeypatch.setattr(AttnWrapperBase, "position_ids",
                        torch.tensor([[n - 1] for n in lengths]), raising=False)
    monkeypatch.setattr(AttnWrapperBase, "cache_seqlens",
                        torch.tensor(lengths, dtype=torch.int32), raising=False)
    monkeypatch.setattr(AttnWrapperBase, "max_seqlen", max(lengths), raising=False)
    monkeypatch.setattr(AttnWrapperBase, "gpu_paged_kv_manager", primary, raising=False)
    monkeypatch.setattr(AttnWrapperBase, "gpu_paged_kv_manager_aux", aux, raising=False)
    monkeypatch.setattr(AttnWrapperBase, "cur_batch", seq_ids, raising=False)
    monkeypatch.setattr(AttnWrapperBase, "kv_append_callback", primary_cb, raising=False)
    monkeypatch.setattr(AttnWrapperBase, "kv_append_callback_aux", aux_cb, raising=False)


def test_wrapper_appends_both_host_regions(writer, monkeypatch):
    from batchgen.models.moonshotai.kimi_linear.wrappers import KimiLinearAttnWrapper

    lengths = [3, 21]
    primary, aux, seq_ids = _caches(lengths)
    appended = []
    _bind_decode(
        monkeypatch, primary, aux, seq_ids, lengths,
        primary_cb=lambda l, k, v: appended.append(("latent", l, tuple(k.shape))),
        aux_cb=lambda l, k, v, s: appended.append(("index", l, k.dtype, tuple(s.shape))),
    )
    module = _module()
    with torch.no_grad():
        out = KimiLinearAttnWrapper._forward_decode(
            SimpleNamespace(module=module, layer_idx=1,
                            _forward_decode_dsa=lambda h: KimiLinearAttnWrapper
                            ._forward_decode_dsa(SimpleNamespace(module=module, layer_idx=1), h)),
            torch.randn(2, 1, 64))
    assert out.shape == (2, 1, 64)
    assert appended == [
        ("latent", 1, (2, 1, 1, LATENT)),
        ("index", 1, torch.float8_e4m3fn, (2,)),
    ]


def test_wrapper_refuses_to_append_the_latent_alone(writer, monkeypatch):
    from batchgen.models.moonshotai.kimi_linear.wrappers import KimiLinearAttnWrapper

    lengths = [3]
    primary, aux, seq_ids = _caches(lengths)
    _bind_decode(monkeypatch, primary, aux, seq_ids, lengths,
                 primary_cb=lambda *a: None, aux_cb=None)
    with torch.no_grad(), pytest.raises(RuntimeError, match="no auxiliary one"):
        KimiLinearAttnWrapper._forward_decode_dsa(
            SimpleNamespace(module=_module(), layer_idx=1), torch.randn(1, 1, 64))
