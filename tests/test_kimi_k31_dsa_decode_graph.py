# ---------------------------------------------------------------------------- #
#  BatchGen — Kimi-K3.1                                                        #
#  Copyright (c) 2025-2026 BatchGen Team                                       #
#  Licensed under the Apache License, Version 2.0                              #
# ---------------------------------------------------------------------------- #
"""Slice 5 of the Kimi-K3.1 DSA serving path: the CUDA-graph decode step.

``dsa_decode.dsa_decode_graph_safe`` is what a captured decode span runs on a
DSA layer. It cannot be captured on CPU, but its arithmetic can run eagerly
against the same CPU kernel references the eager-step tests use, which is
what this module does. Two more references join them here: the graph-capturable
scorer (``paged_mqa_score_and_topk_out``: slot indirection + padding rows) and
the static-table latent writer (``run_paged_kv_token_update_fused``).

The tests prove, on CPU:
  * on a batch with long rows, the graph step equals the eager step row for
    row — output, both cache writes, and the host-append sources;
  * on an all-short batch (where the eager step takes the dense K3 decode)
    the graph step's sparse path gives the same attention;
  * bucket padding rows write nothing to either cache;
  * a graph whose context capacity is below ``index_topk`` scores at that
    capacity and is still the dense attention;
  * the span segment captures the DSA step on a DSA layer (not the dense
    inline), declares the index-key outputs, and the driver hands them to the
    auxiliary Host append — refusing to append the latent alone;
  * statics carry the auxiliary table in the latent table's batch order, and
    the KV signature moves when the auxiliary cache moves;
  * the whole-model graph stages each DSA layer's index keys.

GPU-deferred: capture/replay itself, DeepGEMM's paged logits under capture,
FA3 with ``num_valid_tokens`` padding rows of length 0.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.kimi_k31_dsa_refs import (  # noqa: E402
    QUANT,
    install_sparse_attention_ref,
    head_gates_ref,
)
from tests.test_kimi_k31_dsa_decode import (  # noqa: E402
    D_IDX,
    LATENT,
    PAGE,
    FusedWriterRef,
    _caches,
    _config,
    _install_flash_mla,
    _module,
    _physical_slots,
    _read_page_split,
    paged_scorer_ref,
    transform_ref,
)

from batchgen.models.moonshotai.kimi_linear import (  # noqa: E402
    cuda_graph_segments as cgs,
    dsa_decode,
    dsa_index_rotation as rot,
)
from batchgen.models.wrappers import AttnWrapperBase  # noqa: E402


# --------------------------------------------------------------------------- #
#  Graph-only kernel references                                               #
# --------------------------------------------------------------------------- #

def paged_scorer_out_ref(*, q, aux_blocked_k, aux_page_table, aux_slot_indices, head_gates,
                         cache_seqlens, q_fp8, block_table, context_lens, top_k_indices,
                         topk, max_seqlen, index_head_dim, page_size,
                         num_valid_tokens=None, q_prequantized=False):
    """``paged_mqa_score_and_topk_out``: the prologue's slot indirection + the
    same scoring as the eager scorer reference; rows at or beyond
    ``num_valid_tokens`` select nothing."""
    batch = q.shape[0]
    rows = batch if num_valid_tokens is None else int(num_valid_tokens.reshape(()))
    assert int(topk) <= int(max_seqlen)
    assert tuple(block_table.shape) == (batch, aux_page_table.shape[1])
    assert tuple(context_lens.shape) == (batch, 1)
    assert tuple(q_fp8.shape) == (batch, 1, q.shape[1], q.shape[2])
    top_k_indices.fill_(-1)
    for b in range(rows):
        slot = int(aux_slot_indices[b])
        out = paged_scorer_ref(
            q=q[b:b + 1], aux_blocked_k=aux_blocked_k,
            block_table=aux_page_table[slot:slot + 1], head_gates=head_gates[b:b + 1],
            cache_seqlens=cache_seqlens[b:b + 1], topk=topk, max_seqlen=max_seqlen,
            index_head_dim=index_head_dim, page_size=page_size)
        top_k_indices[b] = out[0].to(top_k_indices.dtype)
    return top_k_indices


class LatentWriterRef:
    """``run_paged_kv_token_update_fused`` over a static table."""

    def __init__(self):
        self.calls = 0

    def __call__(self, *, k_cache, k_tokens, page_table, slot_indices, token_indices,
                 page_size_tokens, num_valid_tokens):
        self.calls += 1
        flat = k_cache.view(-1, k_cache.shape[-1])
        for b in range(int(num_valid_tokens.reshape(()))):
            slot = _physical_slots(page_table[int(slot_indices[b])], token_indices[b:b + 1])[0]
            flat[slot] = k_tokens[b].to(flat.dtype)


@pytest.fixture()
def refs(monkeypatch):
    writer = FusedWriterRef()
    latent = LatentWriterRef()
    monkeypatch.setattr(dsa_decode, "_fused_index_k_writer", lambda: writer)
    monkeypatch.setattr(dsa_decode, "_paged_scorer", lambda: paged_scorer_ref)
    monkeypatch.setattr(dsa_decode, "_paged_scorer_out", lambda: paged_scorer_out_ref)
    monkeypatch.setattr(dsa_decode, "_latent_token_writer", lambda: latent)
    monkeypatch.setattr(dsa_decode, "_head_gates", lambda: head_gates_ref)
    monkeypatch.setattr(dsa_decode, "_selected_positions", lambda: transform_ref)
    install_sparse_attention_ref(monkeypatch)
    monkeypatch.setattr(rot, "_hadamard_predicate", lambda: QUANT.indexer_applies_hadamard)
    _install_flash_mla(monkeypatch)
    return SimpleNamespace(writer=writer, latent=latent)


# --------------------------------------------------------------------------- #
#  Static graph inputs from the eager caches                                  #
# --------------------------------------------------------------------------- #

def _clone_caches(primary, aux):
    import copy

    p2, a2 = copy.copy(primary), copy.copy(aux)
    p2.blocked = primary.blocked.clone()
    a2.blocked = aux.blocked.clone()
    p2.writes, a2.writes = [], []
    return p2, a2


def _statics(primary, aux, seq_ids, lengths, bucket):
    """What ``_refresh_statics`` binds: batch-ordered rows of both tables
    (the aux rows gathered through ``seq_id_to_slot``), padding rows at page 0
    / seqlen 1 / write-skipped."""
    bsz = len(lengths)
    max_pages, max_aux = primary.page_table.shape[1], aux.page_table.shape[1]
    statics = cgs._BucketStatics(bucket, max_pages, "cpu", None, max_aux_pages=max_aux)
    aux_rows = torch.stack([
        aux.page_table[aux._gpu_page_table_manager.seq_id_to_slot[s]] for s in seq_ids
    ])
    statics.refresh(
        bsz,
        torch.tensor(lengths, dtype=torch.int32),
        torch.tensor([n - 1 for n in lengths], dtype=torch.int32),
        primary.page_table[:bsz],
        aux_rows,
    )
    return statics


def _graph_step(module, primary, aux, statics, hidden):
    with torch.no_grad():
        return dsa_decode.dsa_decode_graph_safe(
            module, hidden,
            layer_idx=1,
            page_table=statics.page_table,
            slot_indices=statics.slot_indices,
            token_indices=statics.token_indices,
            cache_seqlens=statics.cache_seqlens,
            num_valid_tokens=statics.num_valid_tokens,
            page_size_tokens=PAGE,
            blocked_k=primary.blocked,
            aux_blocked_k=aux.blocked,
            aux_page_table=statics.aux_page_table,
            aux_page_size=PAGE,
            max_seqlen=statics.aux_page_table.shape[1] * PAGE,
        )


def _eager_step(module, primary, aux, seq_ids, lengths, hidden):
    with torch.no_grad():
        return dsa_decode.dsa_decode_forward(
            module, hidden, torch.tensor([[n - 1] for n in lengths]),
            torch.tensor(lengths, dtype=torch.int32), max(lengths),
            primary, aux, 1, seq_ids)


def _setup(lengths, bucket, *, perm=None, seed=11, hidden_seed=3):
    module = _module()
    primary, aux, seq_ids = _caches(lengths, aux_slot_perm=perm, seed=seed)
    hidden = torch.randn(bucket, 1, 64, generator=torch.Generator().manual_seed(hidden_seed))
    return module, primary, aux, seq_ids, hidden


# --------------------------------------------------------------------------- #
#  The graph step == the eager step                                           #
# --------------------------------------------------------------------------- #

def test_on_a_batch_with_long_rows_the_graph_step_is_the_eager_step(refs):
    """Same rows, same shapes (bucket == batch): bit-identical, caches included."""
    lengths = [20, 5, 23]
    module, primary, aux, seq_ids, hidden = _setup(lengths, bucket=3, perm=[2, 0, 1])
    g_primary, g_aux = _clone_caches(primary, aux)
    statics = _statics(g_primary, g_aux, seq_ids, lengths, bucket=3)

    graph = _graph_step(module, g_primary, g_aux, statics, hidden)
    eager = _eager_step(module, primary, aux, seq_ids, lengths, hidden)
    assert eager.mode == "sparse"

    assert torch.equal(graph.attn_output, eager.attn_output)
    assert torch.equal(graph.k_tensor, eager.k_tensor)
    assert torch.equal(graph.index_k_fp8.view(torch.uint8), eager.index_k_fp8.view(torch.uint8))
    assert torch.equal(graph.index_scales, eager.index_scales)
    # both caches hold exactly the same bytes afterwards
    assert torch.equal(g_primary.blocked, primary.blocked)
    assert torch.equal(g_aux.blocked, aux.blocked)


def test_padding_rows_do_not_change_the_real_rows(refs):
    lengths = [20, 5, 23]
    module, primary, aux, seq_ids, hidden = _setup(lengths, bucket=4, perm=[2, 0, 1])
    g_primary, g_aux = _clone_caches(primary, aux)
    statics = _statics(g_primary, g_aux, seq_ids, lengths, bucket=4)
    graph = _graph_step(module, g_primary, g_aux, statics, hidden)
    eager = _eager_step(module, primary, aux, seq_ids, lengths, hidden[:3])
    assert graph.attn_output.shape == (4, 1, 64)
    assert torch.allclose(graph.attn_output[:3], eager.attn_output, rtol=1e-5, atol=1e-6)
    assert torch.allclose(graph.k_tensor[:3], eager.k_tensor, rtol=1e-6, atol=1e-6)
    assert torch.allclose(g_primary.blocked, primary.blocked, rtol=1e-6, atol=1e-6)


def test_on_an_all_short_batch_the_sparse_graph_step_is_the_dense_decode(refs):
    lengths = [5, 3, 8]
    module, primary, aux, seq_ids, hidden = _setup(lengths, bucket=3)
    g_primary, g_aux = _clone_caches(primary, aux)
    statics = _statics(g_primary, g_aux, seq_ids, lengths, bucket=3)

    graph = _graph_step(module, g_primary, g_aux, statics, hidden)
    eager = _eager_step(module, primary, aux, seq_ids, lengths, hidden)
    assert eager.mode == "dense"
    assert torch.allclose(graph.attn_output, eager.attn_output, rtol=1e-4, atol=1e-5)
    assert torch.equal(g_primary.blocked, primary.blocked)
    assert torch.equal(g_aux.blocked, aux.blocked)


def test_padding_rows_write_nothing_to_either_cache(refs):
    lengths = [20, 9]
    module, primary, aux, seq_ids, hidden = _setup(lengths, bucket=4)
    before_p, before_a = primary.blocked.clone(), aux.blocked.clone()
    statics = _statics(primary, aux, seq_ids, lengths, bucket=4)
    _graph_step(module, primary, aux, statics, hidden)
    # exactly the two real rows' slots changed, in each cache
    changed_p = (primary.blocked != before_p).any(-1).flatten().nonzero().flatten()
    expected = sorted(int(_physical_slots(primary.page_table[b], torch.tensor([n - 1]))[0])
                      for b, n in enumerate(lengths))
    assert sorted(changed_p.tolist()) == expected
    changed_pages = (aux.blocked != before_a).flatten(1).any(-1).nonzero().flatten().tolist()
    real_pages = {int(primary.page_table[b][(n - 1) // PAGE]) for b, n in enumerate(lengths)}
    assert set(changed_pages) == real_pages
    assert refs.writer.calls[-1]["num_valid_tokens"] is statics.num_valid_tokens


def test_the_new_index_key_lands_through_the_static_aux_table(refs):
    lengths = [20, 23]
    module, primary, aux, seq_ids, hidden = _setup(lengths, bucket=2, perm=[1, 0])
    statics = _statics(primary, aux, seq_ids, lengths, bucket=2)
    step = _graph_step(module, primary, aux, statics, hidden)
    for b, n in enumerate(lengths):
        slot = aux._gpu_page_table_manager.seq_id_to_slot[seq_ids[b]]
        phys = _physical_slots(aux.page_table[slot], torch.tensor([n - 1]))
        k_fp8, scale = _read_page_split(aux.blocked, phys)
        assert torch.equal(k_fp8.view(torch.uint8), step.index_k_fp8[b].reshape(1, D_IDX).view(torch.uint8))
        assert torch.equal(scale, step.index_scales[b:b + 1])


def test_a_capacity_below_index_topk_scores_at_the_capacity_and_is_dense(refs, monkeypatch):
    """Every row is short when the whole graph capacity is below
    ``index_topk``; the scorer must be asked for at most that many keys."""
    lengths = [5, 3]
    module, primary, aux, seq_ids, hidden = _setup(lengths, bucket=2)
    g_primary, g_aux = _clone_caches(primary, aux)
    statics = _statics(g_primary, g_aux, seq_ids, lengths, bucket=2)
    seen = {}

    def spy(**kwargs):
        seen.update(topk=kwargs["topk"], max_seqlen=kwargs["max_seqlen"])
        return paged_scorer_out_ref(**kwargs)

    monkeypatch.setattr(dsa_decode, "_paged_scorer_out", lambda: spy)
    with torch.no_grad():
        graph = dsa_decode.dsa_decode_graph_safe(
            module, hidden, layer_idx=1, page_table=statics.page_table,
            slot_indices=statics.slot_indices, token_indices=statics.token_indices,
            cache_seqlens=statics.cache_seqlens, num_valid_tokens=statics.num_valid_tokens,
            page_size_tokens=PAGE, blocked_k=g_primary.blocked, aux_blocked_k=g_aux.blocked,
            aux_page_table=statics.aux_page_table, aux_page_size=PAGE, max_seqlen=6)
    assert seen == {"topk": 6, "max_seqlen": 6}
    eager = _eager_step(module, primary, aux, seq_ids, lengths, hidden)
    assert eager.mode == "dense"
    assert torch.allclose(graph.attn_output, eager.attn_output, rtol=1e-4, atol=1e-5)


def test_graph_score_width():
    assert dsa_decode.graph_score_width(2048, 1 << 20) == 2048
    assert dsa_decode.graph_score_width(2048, 1024) == 1024
    with pytest.raises(ValueError, match="positive"):
        dsa_decode.graph_score_width(2048, 0)


# --------------------------------------------------------------------------- #
#  Statics                                                                    #
# --------------------------------------------------------------------------- #

def test_statics_carry_the_aux_table_in_batch_order_and_pad_it():
    statics = cgs._BucketStatics(4, 3, "cpu", None, max_aux_pages=5)
    assert statics.aux_page_table.shape == (4, 5)
    aux_rows = torch.arange(10, dtype=torch.int32).view(2, 5)
    statics.refresh(2, torch.tensor([3, 4], dtype=torch.int32),
                    torch.tensor([2, 3], dtype=torch.int32),
                    torch.ones(2, 3, dtype=torch.int32), aux_rows)
    assert torch.equal(statics.aux_page_table[:2], aux_rows)
    assert int(statics.aux_page_table[2:].abs().sum()) == 0
    statics.arm_for_capture()
    assert int(statics.aux_page_table.abs().sum()) == 0


def test_statics_refuse_a_mismatched_aux_binding():
    with_aux = cgs._BucketStatics(2, 3, "cpu", None, max_aux_pages=5)
    without = cgs._BucketStatics(2, 3, "cpu", None)
    assert without.aux_page_table is None
    args = (1, torch.ones(1, dtype=torch.int32), torch.zeros(1, dtype=torch.int32),
            torch.ones(1, 3, dtype=torch.int32))
    with pytest.raises(RuntimeError, match="gave no auxiliary rows"):
        with_aux.refresh(*args)
    with pytest.raises(RuntimeError, match="gave auxiliary rows"):
        without.refresh(*args, torch.ones(1, 5, dtype=torch.int32))
    # an empty rank has no rows to give: every row is padding, zeroed
    with_aux.aux_page_table.fill_(7)
    empty = torch.zeros(0, dtype=torch.int32)
    with_aux.refresh(0, empty, empty, with_aux.page_table[:0])
    assert int(with_aux.aux_page_table.abs().sum()) == 0
    assert int(with_aux.num_valid_tokens) == 0


# --------------------------------------------------------------------------- #
#  The span segment                                                           #
# --------------------------------------------------------------------------- #

def _decoder_layer():
    from batchgen.models.moonshotai.kimi_linear.model import KimiDecoderLayer

    layer = KimiDecoderLayer(_config(), 1).eval()
    layer.self_attn.load_state_dict(_module().state_dict())
    g = torch.Generator().manual_seed(5)
    with torch.no_grad():
        for name, p in layer.named_parameters():
            if name.startswith("self_attn."):
                continue
            if "layernorm" in name:
                p.fill_(1.0)
            else:
                p.normal_(0.0, 0.1, generator=g)
    return layer


def test_the_span_segment_declares_the_dsa_outputs_and_needs_aux_geometry():
    layer = _decoder_layer()
    with pytest.raises(ValueError, match="auxiliary cache geometry"):
        cgs.KimiLinearSpanSegment(layer, 1, {}, PAGE, torch.float32)
    seg = cgs.KimiLinearSpanSegment(layer, 1, {}, PAGE, torch.float32,
                                    aux_page_size_tokens=PAGE, max_aux_context=64)
    assert seg.is_dsa and not seg.is_kda
    specs = seg.get_static_output_specs(4)
    assert specs["index_k"].dtype is torch.float8_e4m3fn
    assert specs["index_k"].shape == ("batch_size", 1, 1, D_IDX)
    assert specs["index_scales"].dtype is torch.float32
    assert specs["k_tensor"].shape == ("batch_size", 1, 1, LATENT)


def test_the_span_segment_runs_the_dsa_step_not_the_dense_inline(refs, monkeypatch):
    lengths = [20, 6]
    layer = _decoder_layer()
    _, primary, aux, seq_ids, hidden = _setup(lengths, bucket=4)
    g_primary, g_aux = _clone_caches(primary, aux)
    statics = _statics(g_primary, g_aux, seq_ids, lengths, bucket=4)
    monkeypatch.setattr(AttnWrapperBase, "gpu_paged_kv_manager", g_primary, raising=False)
    monkeypatch.setattr(AttnWrapperBase, "gpu_paged_kv_manager_aux", g_aux, raising=False)

    def dense_inline(*a, **k):
        raise AssertionError("a DSA layer must not capture the dense NoPE-MLA inline")

    monkeypatch.setattr(cgs, "_mla_decode_graph_safe", dense_inline)
    seg = cgs.KimiLinearSpanSegment(layer, 1, {4: statics}, PAGE, torch.float32,
                                    aux_page_size_tokens=PAGE,
                                    max_aux_context=statics.aux_page_table.shape[1] * PAGE)
    seg.setup_static_buffers(4)
    with torch.no_grad():
        out = seg.forward(hidden)
    assert not seg.fold_ffn             # an MoE layer: the span ends at the post-norm
    assert set(out) == {"normed", "residual", "k_tensor", "index_k", "index_scales"}

    # the eager layer: same layer norms around the eager DSA step
    with torch.no_grad():
        normed = layer.input_layernorm(hidden[:2])
        eager = _eager_step(layer.self_attn, primary, aux, seq_ids, lengths, normed)
        residual = hidden[:2] + eager.attn_output
    assert torch.allclose(out["residual"][:2], residual, rtol=1e-4, atol=1e-5)
    assert torch.allclose(out["normed"][:2], layer.post_attention_layernorm(residual),
                          rtol=1e-4, atol=1e-5)
    assert torch.allclose(out["index_scales"][:2], eager.index_scales, rtol=1e-6, atol=0)


# --------------------------------------------------------------------------- #
#  The driver: Host appends, signature                                        #
# --------------------------------------------------------------------------- #

def _driver(bsz):
    driver = cgs.KimiLinearDecodeGraph.__new__(cgs.KimiLinearDecodeGraph)
    driver._bsz = bsz
    driver._has_dsa = True
    driver.whole_segment = None
    return driver


def test_the_driver_appends_index_keys_to_the_aux_region(monkeypatch):
    calls = []
    monkeypatch.setattr(AttnWrapperBase, "kv_append_callback",
                        lambda *a: None, raising=False)
    monkeypatch.setattr(AttnWrapperBase, "kv_append_callback_aux",
                        lambda *a: calls.append(a), raising=False)
    index_k = torch.zeros(4, 1, 1, D_IDX, dtype=torch.float8_e4m3fn)
    scales = torch.arange(4, dtype=torch.float32)
    _driver(3)._append_index_keys(7, index_k, scales)
    (layer, k, v, s), = calls
    assert layer == 7 and v is None
    assert k.shape == (3, 1, 1, D_IDX) and torch.equal(s, scales[:3])
    # cloned: the static outputs are overwritten by the next replay
    assert k.data_ptr() != index_k.data_ptr() and s.data_ptr() != scales.data_ptr()


def test_the_driver_refuses_to_append_the_latent_alone(monkeypatch):
    monkeypatch.setattr(AttnWrapperBase, "kv_append_callback",
                        lambda *a: None, raising=False)
    monkeypatch.setattr(AttnWrapperBase, "kv_append_callback_aux", None, raising=False)
    with pytest.raises(RuntimeError, match="no auxiliary one"):
        _driver(1)._append_index_keys(
            3, torch.zeros(1, 1, 1, D_IDX, dtype=torch.float8_e4m3fn), torch.zeros(1))


def test_without_host_callbacks_nothing_is_appended(monkeypatch):
    monkeypatch.setattr(AttnWrapperBase, "kv_append_callback", None, raising=False)
    monkeypatch.setattr(AttnWrapperBase, "kv_append_callback_aux", None, raising=False)
    _driver(1)._append_index_keys(
        3, torch.zeros(1, 1, 1, D_IDX, dtype=torch.float8_e4m3fn), torch.zeros(1))


def test_a_dsa_graph_needs_a_bound_aux_manager(monkeypatch):
    monkeypatch.setattr(AttnWrapperBase, "gpu_paged_kv_manager_aux", None, raising=False)
    with pytest.raises(RuntimeError, match="no auxiliary"):
        _driver(1)._aux_graph_page_table([1])
    no_dsa = _driver(1)
    no_dsa._has_dsa = False
    assert no_dsa._aux_graph_page_table([1]) is None


class _SigManager:
    def __init__(self, pages):
        self.config = SimpleNamespace(page_size_tokens=PAGE)
        self.k = torch.zeros(pages, PAGE, 1, 8)
        self.table = torch.zeros(2, 3, dtype=torch.int32)

    def get_cuda_graph_page_table_storage(self):
        return self.table

    def get_kv_tensors(self):
        return self.k, None


def test_the_kv_signature_moves_with_the_aux_cache(monkeypatch):
    monkeypatch.setattr(cgs.KimiLinearKDAWrapper, "state_manager", None, raising=False)
    primary = _SigManager(4)
    monkeypatch.setattr(AttnWrapperBase, "gpu_paged_kv_manager_aux", _SigManager(4),
                        raising=False)
    driver = _driver(1)
    first = driver._signature(primary)
    assert first == driver._signature(primary)
    moved = _SigManager(6)
    moved.table = AttnWrapperBase.gpu_paged_kv_manager_aux.table   # only the cache moved
    monkeypatch.setattr(AttnWrapperBase, "gpu_paged_kv_manager_aux", moved, raising=False)
    assert driver._signature(primary) != first
    driver._has_dsa = False
    assert driver._signature(primary)[-1] == ()


# --------------------------------------------------------------------------- #
#  The whole-model graph                                                      #
# --------------------------------------------------------------------------- #

def test_the_whole_model_graph_stages_and_appends_each_dsa_layers_index_keys(monkeypatch):
    from batchgen.models.moonshotai.kimi_linear.whole_model_cuda_graph_segments import (
        KimiLinearWholeModelSegment,
    )

    segs = [SimpleNamespace(is_kda=True), SimpleNamespace(is_kda=False, is_dsa=True, kv_dim=LATENT),
            SimpleNamespace(is_kda=True), SimpleNamespace(is_kda=False, is_dsa=True, kv_dim=LATENT)]
    model = SimpleNamespace(model=SimpleNamespace(layers=[None] * 4))
    whole = KimiLinearWholeModelSegment(
        model=model, layer_segments=segs, moe_segments={}, statics={}, tp_size=1, tp_rank=0,
        device=torch.device("cpu"), hidden_size=64, dtype=torch.float32, max_bucket_size=4)
    assert whole.index_key_layers == (1, 3)
    whole.layer_segments = []          # setup only allocates the staging buffers here
    whole.moe_segments = {}
    whole._make_bucket_maps = lambda bucket: None
    whole.setup_static_buffers(4)
    assert whole.index_key_buffer.shape == (2, 4, 1, 1, D_IDX)
    assert whole.index_key_buffer.dtype is torch.float8_e4m3fn
    assert whole.index_scale_buffer.shape == (2, 4)

    k3 = torch.full((4, 1, 1, D_IDX), 2.0).to(torch.float8_e4m3fn)
    whole._copy_index_keys(3, k3, torch.full((4,), 5.0))
    with pytest.raises(RuntimeError, match="no index-key staging"):
        whole._copy_index_keys(2, k3, torch.zeros(4))

    calls = []
    monkeypatch.setattr(AttnWrapperBase, "kv_append_callback", lambda *a: None, raising=False)
    monkeypatch.setattr(AttnWrapperBase, "kv_append_callback_aux",
                        lambda *a: calls.append(a), raising=False)
    driver = _driver(2)
    driver.whole_segment = whole
    driver._append_whole_index_keys()
    assert [c[0] for c in calls] == [1, 3]
    assert torch.equal(calls[1][3], torch.full((2,), 5.0))
    assert calls[1][1].shape == (2, 1, 1, D_IDX)
