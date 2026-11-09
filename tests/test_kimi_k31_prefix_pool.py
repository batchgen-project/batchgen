# ---------------------------------------------------------------------------- #
#  BatchGen — Kimi-K3.1                                                        #
#  Copyright (c) 2025-2026 BatchGen Team                                       #
#  Licensed under the Apache License, Version 2.0                              #
# ---------------------------------------------------------------------------- #
"""Kimi-K3.1 DSA prefill over a CACHED prefix, end to end on CPU.

A prefix hit must skip recomputation on BOTH regions -- the MLA latent and the
page-split FP8 indexer-K region -- and the suffix must still select over the
cached keys as well as its own. These tests prove that where it can be proved
without a GPU: the orchestration is the production one, every GPU launch is
replaced by the reference in ``tests/kimi_k31_dsa_refs.py``, and the Host
regions are emulated by the kernel package's own torch-only layout statements.

What is held to what:

* a pooled row's selection and output are EXACTLY the no-cache recompute's for
  the same logical context -- the whole prefix/suffix split is addressing, not
  arithmetic, so there is no tolerance to spend here;
* a prefix loaded from the auxiliary region is scored against exactly the keys
  those stored bytes represent, which are the keys DECODE reads back for the
  same tokens: the page-split round trip is byte-exact and the pool row is its
  dequantization, nothing more;
* a computed prefix's auxiliary bytes and a cached prefix's auxiliary bytes are
  the same bytes, so the hit and the recompute put decode in the same state;
* the pooled offload writes both regions at the same row starts and token
  counts, because one page table addresses both.

GPU-deferred (named in the report, not simulated here): ``fp8_mqa_logits`` +
``fast_topk_2048_out`` over a gathered pooled key list, the H2D of page-split
pages through ``async_load_prefix_pages_to_device``, and the aux range offload
landing on the same Host pages as the latent.
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
    install_fla_shim,
    install_flash_attn_varlen,
    prefill_topk_ref,
)

install_fla_shim()

from batchgen.models.moonshotai.kimi_linear import (  # noqa: E402
    dsa_index_rotation as rot,
    dsa_prefill,
    dsa_sparse_prefill as sp,
)
from batchgen.models.moonshotai.kimi_linear.serving_modules import (  # noqa: E402
    dsa_materialize_pool_segment_index_keys,
    mla_materialize_pool_segment,
)
from batchgen.models.wrappers import AttnWrapperBase  # noqa: E402
from batchgen.prefix_reuse.pool import PrefillPrefixPool  # noqa: E402

from tests.test_kimi_k31_dsa_sparse_prefill import (  # noqa: E402
    _module,
    _packed,
)

PAGE_TOKENS = 4


class _TopkRecorder:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        out = prefill_topk_ref(**kwargs)
        self.calls.append(dict(kwargs, result=out.clone()))
        return out


@pytest.fixture()
def seams(monkeypatch):
    recorder = _TopkRecorder()
    monkeypatch.setattr(sp, "_prefill_index_topk", lambda: recorder)
    monkeypatch.setattr(sp, "_head_gates", lambda: head_gates_ref)
    install_sparse_attention_ref(monkeypatch)
    monkeypatch.setattr(
        rot, "_hadamard_predicate", lambda: QUANT.indexer_applies_hadamard)
    monkeypatch.setattr(
        dsa_prefill, "_quantize_indexer_k_fp8",
        lambda: QUANT.quantize_indexer_k_fp8)
    install_flash_attn_varlen(monkeypatch)
    return recorder


# --------------------------------------------------------------------------- #
#  A K3.1-shaped pool: one latent row per DSA layer, no V, one indexer row    #
# --------------------------------------------------------------------------- #


def _pool(module, *, num_pages=16, layer_idx=1):
    pool = PrefillPrefixPool(
        num_layers=1,
        num_pages=num_pages,
        page_tokens=PAGE_TOKENS,
        num_kv_heads=1,
        head_dim=module.kv_lora_rank + module.qk_rope_head_dim,
        dtype=torch.float32,
        device="cpu",
        with_v=False,
        indexer_layers=[layer_idx],
        indexer_dim=module.indexer.index_head_dim,
    )
    pool.k.zero_()
    pool.indexer.zero_()
    return pool


def _chunk(pool, *, prefix_pages, prefix_tokens, suffix_tokens):
    """One row continuing a resident prefix: the chunk the executor builds.

    Only the fields the DSA pool prefill reads, with the values
    ``PoolChunkExecutor.build`` gives them: ancestor pages then its own, the
    row's absolute positions, and its own tokens' pool slots.
    """
    own = pool.alloc(-(-suffix_tokens // PAGE_TOKENS))
    table = list(prefix_pages) + list(own)
    return SimpleNamespace(
        page_table=torch.tensor([table], dtype=torch.int32),
        cache_seqlens=torch.tensor(
            [prefix_tokens + suffix_tokens], dtype=torch.int32),
        cu_seqlens_q=torch.tensor([0, suffix_tokens], dtype=torch.int32),
        slots=pool.slots(own, 0, suffix_tokens),
        max_cache_seqlen=prefix_tokens + suffix_tokens,
        position_ids=torch.arange(
            prefix_tokens, prefix_tokens + suffix_tokens, dtype=torch.int64),
    ), own


def _pool_ctx(chunk, pool):
    return sp.build_dsa_pool_prefill(
        page_table=chunk.page_table,
        cache_seqlens=chunk.cache_seqlens,
        cu_seqlens_q=chunk.cu_seqlens_q,
        position_ids=chunk.position_ids,
        slots=chunk.slots,
        page_tokens=pool.page_tokens,
        max_cache_seqlen=chunk.max_cache_seqlen,
    )


def _recompute(module, hidden, total):
    """The no-cache arm: one packed row of the whole prompt."""
    cu = torch.tensor([0, total], dtype=torch.int32)
    with torch.no_grad():
        keys = dsa_prefill.prefill_index_keys(module.indexer, hidden)
        out, latent = sp.dsa_sparse_prefill(module, hidden, keys, cu, total)
    return out, latent, keys


def _write_prefix_into_pool(module, pool, pages, latent, keys, layer_idx=1):
    """Populate the pool's prefix pages through the MODEL's load seams."""
    slots = pool.slots(pages, 0, latent.shape[0])
    mla_materialize_pool_segment(
        module, latent, pool=pool, pool_layer=0, slots=slots)
    dsa_materialize_pool_segment_index_keys(
        module, keys, pool=pool, indexer_row=pool.indexer_row[layer_idx],
        slots=slots)
    return slots


def _pooled_suffix(module, pool, chunk, hidden_suffix):
    with torch.no_grad():
        keys = dsa_prefill.prefill_index_keys(module.indexer, hidden_suffix)
        out, latent = sp.dsa_sparse_prefill(
            module,
            hidden_suffix,
            keys,
            chunk.cu_seqlens_q,
            int(chunk.cu_seqlens_q[-1]),
            pool=_pool_ctx(chunk, pool),
            latent_pool=pool.k[0],
            indexer_pool=pool.indexer[pool.indexer_row[1]],
        )
    return out, latent, keys


# --------------------------------------------------------------------------- #
#  1. The chunk's addressing                                                  #
# --------------------------------------------------------------------------- #


def test_the_key_list_is_the_rows_whole_context_in_order(seams):
    module = _module()
    pool = _pool(module)
    chunk, _ = _chunk(
        pool, prefix_pages=[5, 6], prefix_tokens=2 * PAGE_TOKENS,
        suffix_tokens=5,
    )
    ctx = _pool_ctx(chunk, pool)
    # Prefix pages first, then the row's own: slot = page * page_tokens + off.
    expected = (
        [5 * PAGE_TOKENS + off for off in range(PAGE_TOKENS)]
        + [6 * PAGE_TOKENS + off for off in range(PAGE_TOKENS)]
        + chunk.slots.tolist()[:5]
    )
    assert ctx.key_slots.tolist() == expected
    # Every suffix query sees its whole context, cached prefix included.
    assert ctx.causal_starts.tolist() == [0] * 5
    assert ctx.causal_ends.tolist() == [9, 10, 11, 12, 13]


def test_the_selection_is_mapped_back_to_pool_slots(seams):
    key_slots = torch.tensor([20, 21, 40, 41], dtype=torch.long)
    picked = torch.tensor([[0, 3, -1], [2, 1, -1]], dtype=torch.int32)
    mapped = sp.map_topk_to_pool_slots(picked, key_slots)
    assert mapped.tolist() == [[20, 41, -1], [40, 21, -1]]
    assert mapped.dtype is torch.int32


@pytest.mark.parametrize("rows", [
    {"pool": None, "latent": True, "indexer": True},
    {"pool": True, "latent": False, "indexer": True},
    {"pool": True, "latent": True, "indexer": False},
])
def test_a_half_wired_pool_call_is_refused_by_name(seams, rows):
    module = _module()
    pool = _pool(module)
    chunk, _ = _chunk(
        pool, prefix_pages=[5], prefix_tokens=PAGE_TOKENS, suffix_tokens=3)
    hidden, cu = _packed([3])
    keys = dsa_prefill.prefill_index_keys(module.indexer, hidden)
    with pytest.raises(RuntimeError):
        sp.dsa_sparse_prefill(
            module, hidden, keys, cu, 3,
            pool=_pool_ctx(chunk, pool) if rows["pool"] else None,
            latent_pool=pool.k[0] if rows["latent"] else None,
            indexer_pool=(
                pool.indexer[pool.indexer_row[1]] if rows["indexer"] else None
            ),
        )


# --------------------------------------------------------------------------- #
#  2. (c) A cached prefix gives the no-cache selection and output             #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("prefix_tokens, suffix_tokens", [
    (2 * PAGE_TOKENS, 5),    # suffix shorter than index_topk
    (3 * PAGE_TOKENS, 9),    # every suffix query past index_topk
    (PAGE_TOKENS, 1),        # a one-token suffix continuing one page
])
def test_a_pooled_prefix_hit_equals_the_no_cache_recompute(
    seams, prefix_tokens, suffix_tokens
):
    """The SAME logical context, computed in one packed row or split into a
    resident prefix plus a suffix row: identical selection and identical
    attention output. Exactly equal, not close: the split is addressing."""
    module = _module()
    total = prefix_tokens + suffix_tokens
    hidden, _ = _packed([total], seed=11)
    dense_out, dense_latent, dense_keys = _recompute(module, hidden, total)
    dense_pick = seams.calls[-1]["result"]

    pool = _pool(module)
    prefix_pages = pool.alloc(prefix_tokens // PAGE_TOKENS)
    _write_prefix_into_pool(
        module, pool, prefix_pages,
        dense_latent[:prefix_tokens], dense_keys[:prefix_tokens],
    )
    chunk, _ = _chunk(
        pool, prefix_pages=prefix_pages, prefix_tokens=prefix_tokens,
        suffix_tokens=suffix_tokens,
    )
    ctx = _pool_ctx(chunk, pool)
    pooled_out, pooled_latent, pooled_keys = _pooled_suffix(
        module, pool, chunk, hidden[prefix_tokens:]
    )
    pooled_pick = seams.calls[-1]["result"]

    # The suffix recomputes only its own tokens. The projections run over 5
    # rows instead of 13, which is the same arithmetic in a different GEMM
    # shape, so these are equal to within the last bits rather than bitwise.
    assert pooled_latent.shape[0] == suffix_tokens
    assert torch.allclose(
        pooled_latent, dense_latent[prefix_tokens:], rtol=1e-6, atol=1e-6)
    assert torch.allclose(
        pooled_keys.float(), dense_keys[prefix_tokens:].float(),
        rtol=1e-6, atol=1e-6)
    # And selects the same logical keys: pool slots back to context positions.
    slot_of_position = ctx.key_slots.tolist()
    position_of_slot = {slot: pos for pos, slot in enumerate(slot_of_position)}
    for query in range(suffix_tokens):
        cached = sorted(
            position_of_slot[int(slot)]
            for slot in pooled_pick[query].tolist() if slot >= 0
        )
        recomputed = sorted(
            int(token) for token in dense_pick[prefix_tokens + query].tolist()
            if token >= 0
        )
        assert cached == recomputed, query
    assert torch.allclose(
        pooled_out, dense_out[prefix_tokens:], rtol=1e-4, atol=1e-5)


def test_the_pooled_prefix_is_not_recomputed(seams):
    """Guard against a vacuous suite: the pooled arm scores a key list it did
    not compute, so the scorer is handed MORE keys than the row has queries."""
    module = _module()
    prefix_tokens, suffix_tokens = 2 * PAGE_TOKENS, 5
    hidden, _ = _packed([prefix_tokens + suffix_tokens], seed=11)
    _, latent, keys = _recompute(module, hidden, prefix_tokens + suffix_tokens)

    pool = _pool(module)
    pages = pool.alloc(prefix_tokens // PAGE_TOKENS)
    _write_prefix_into_pool(
        module, pool, pages, latent[:prefix_tokens], keys[:prefix_tokens])
    chunk, _ = _chunk(
        pool, prefix_pages=pages, prefix_tokens=prefix_tokens,
        suffix_tokens=suffix_tokens,
    )
    _pooled_suffix(module, pool, chunk, hidden[prefix_tokens:])
    call = seams.calls[-1]
    assert call["index_keys"].shape[0] == prefix_tokens + suffix_tokens
    assert call["q_idx"].shape[0] == suffix_tokens
    # The cached half of the gathered key list is the stored prefix verbatim.
    assert torch.equal(
        call["index_keys"][:prefix_tokens].to(torch.float32),
        keys[:prefix_tokens].to(torch.float32),
    )


def test_a_prefix_loaded_through_the_aux_region_selects_the_same_keys(seams):
    """The real hit: the prefix's keys come back from the page-split FP8
    region, so what the suffix scores for the cached span is what the region
    HOLDS. Against a recompute over that same logical context -- the whole
    prompt, with the prefix keys the stored ones -- the selection is
    identical, which is the statement that the load is transparent."""
    module = _module()
    prefix_tokens, suffix_tokens = 2 * PAGE_TOKENS, 7
    total = prefix_tokens + suffix_tokens
    hidden, _ = _packed([total], seed=13)
    with torch.no_grad():
        keys = dsa_prefill.prefill_index_keys(module.indexer, hidden)

    host_pages = [9, 10]
    buf, _, _ = _host_aux_region(keys[:prefix_tokens], host_pages)
    stored = QUANT.dequantize_page_split_pages(
        buf[host_pages], PAGE_TOKENS, keys.shape[-1], prefix_tokens
    ).to(keys.dtype)
    # A round trip is lossy, so the arms must agree on WHICH keys they score:
    # the cached ones, as stored.
    assert not torch.equal(stored, keys[:prefix_tokens])
    as_cached = torch.cat([stored, keys[prefix_tokens:]])

    cu = torch.tensor([0, total], dtype=torch.int32)
    with torch.no_grad():
        reference_out, reference_latent = sp.dsa_sparse_prefill(
            module, hidden, as_cached, cu, total)
    reference_pick = seams.calls[-1]["result"]

    pool = _pool(module)
    prefix_pages = pool.alloc(prefix_tokens // PAGE_TOKENS)
    _write_prefix_into_pool(
        module, pool, prefix_pages, reference_latent[:prefix_tokens], stored)
    chunk, _ = _chunk(
        pool, prefix_pages=prefix_pages, prefix_tokens=prefix_tokens,
        suffix_tokens=suffix_tokens,
    )
    ctx = _pool_ctx(chunk, pool)
    pooled_out, _, _ = _pooled_suffix(
        module, pool, chunk, hidden[prefix_tokens:])
    pooled_pick = seams.calls[-1]["result"]

    position_of_slot = {
        int(slot): pos for pos, slot in enumerate(ctx.key_slots.tolist())
    }
    for query in range(suffix_tokens):
        cached = sorted(
            position_of_slot[int(slot)]
            for slot in pooled_pick[query].tolist() if slot >= 0
        )
        recomputed = sorted(
            int(token)
            for token in reference_pick[prefix_tokens + query].tolist()
            if token >= 0
        )
        assert cached == recomputed, query
    assert torch.allclose(
        pooled_out, reference_out[prefix_tokens:], rtol=1e-4, atol=1e-5)


def test_an_empty_prefix_pool_row_reproduces_the_packed_path(seams):
    """A pooled row with NO ancestor must equal the plain packed prefill: the
    pooled form is the packed one with every prefix empty."""
    module = _module()
    tokens = 7
    hidden, _ = _packed([tokens], seed=5)
    packed_out, packed_latent, _ = _recompute(module, hidden, tokens)

    pool = _pool(module)
    chunk, _ = _chunk(
        pool, prefix_pages=[], prefix_tokens=0, suffix_tokens=tokens)
    pooled_out, pooled_latent, _ = _pooled_suffix(module, pool, chunk, hidden)
    assert torch.equal(pooled_latent, packed_latent)
    assert torch.equal(pooled_out, packed_out)
    # The row's own tokens landed in its own pages, in order.
    written = pool.k[0].view(-1, pool.k.shape[-1]).index_select(0, chunk.slots)
    assert torch.equal(written, packed_latent.to(written.dtype))


# --------------------------------------------------------------------------- #
#  3. (b)+(d) The page-split round trip, and what decode then reads           #
# --------------------------------------------------------------------------- #


def _host_aux_region(keys, pages, *, page_tokens=PAGE_TOKENS):
    """Emulate the auxiliary region: quantize and scatter into page-split
    pages exactly as the Host write does, through the kernel's own oracle."""
    data_dim = keys.shape[-1]
    buf = torch.zeros(
        max(pages) + 1, page_tokens * (data_dim + QUANT.SCALE_BYTES_PER_TOKEN),
        dtype=torch.uint8,
    )
    k_fp8, scales = QUANT.quantize_indexer_k_fp8(keys)
    slots = torch.tensor(
        [page * page_tokens + off for page in pages
         for off in range(page_tokens)][:keys.shape[0]],
        dtype=torch.long,
    )
    QUANT.page_split_write_reference(buf, slots, k_fp8, scales, page_tokens)
    return buf, k_fp8, scales


def test_the_page_split_round_trip_is_byte_exact(seams):
    module = _module()
    tokens = 2 * PAGE_TOKENS
    hidden, _ = _packed([tokens], seed=23)
    keys = dsa_prefill.prefill_index_keys(module.indexer, hidden)
    pages = [3, 4]
    buf, k_fp8, scales = _host_aux_region(keys, pages)

    block = buf[pages]
    read_fp8, read_scales = QUANT.page_split_read_reference(
        block, PAGE_TOKENS, keys.shape[-1], tokens)
    assert torch.equal(read_fp8.view(torch.uint8), k_fp8.view(torch.uint8))
    assert torch.equal(read_scales, scales)
    # And the dequantization is exactly what the stored record means.
    dequantized = QUANT.dequantize_page_split_pages(
        block, PAGE_TOKENS, keys.shape[-1], tokens, dtype=torch.float32)
    assert torch.equal(dequantized, read_fp8.float() * read_scales[:, None])


def test_a_loaded_prefix_row_holds_exactly_what_the_region_stores(seams):
    """(d) The keys the suffix scores for the cached span are the keys DECODE
    reads back for those tokens: the pool row is the region's dequantization,
    nothing re-derived."""
    module = _module()
    tokens = 2 * PAGE_TOKENS
    hidden, _ = _packed([tokens], seed=23)
    keys = dsa_prefill.prefill_index_keys(module.indexer, hidden)
    pages = [3, 4]
    buf, _, _ = _host_aux_region(keys, pages)
    stored = QUANT.dequantize_page_split_pages(
        buf[pages], PAGE_TOKENS, keys.shape[-1], tokens)

    pool = _pool(module)
    pool_pages = pool.alloc(tokens // PAGE_TOKENS)
    slots = pool.slots(pool_pages, 0, tokens)
    dsa_materialize_pool_segment_index_keys(
        module, stored, pool=pool, indexer_row=pool.indexer_row[1],
        slots=slots)
    row = pool.indexer[pool.indexer_row[1]]
    loaded = row.view(-1, row.shape[-1]).index_select(0, slots)
    assert torch.equal(loaded, stored.to(loaded.dtype))
    # Re-storing what was loaded is a fixed point: the hit and a recompute of
    # the same tokens leave the region holding the same bytes.
    again, _, _ = _host_aux_region(stored.to(keys.dtype), pages)
    assert torch.equal(again[pages], buf[pages])


def test_a_materialized_latent_row_is_the_stored_latent(seams):
    """A K3.1 pool holds the latent itself (no V row, no projection), so the
    load is the stored bytes in the slots the computed path writes."""
    module = _module()
    tokens = PAGE_TOKENS
    hidden, _ = _packed([tokens], seed=31)
    _, latent, _ = _recompute(module, hidden, tokens)

    pool = _pool(module)
    pages = pool.alloc(1)
    slots = pool.slots(pages, 0, tokens)
    written = mla_materialize_pool_segment(
        module, latent, pool=pool, pool_layer=0, slots=slots)
    assert written == tokens
    row = pool.k[0].view(-1, pool.k.shape[-1]).index_select(0, slots)
    assert torch.equal(row, latent.to(row.dtype))


@pytest.mark.parametrize("bad", ["slots", "width"])
def test_materializing_index_keys_refuses_a_mismatch(seams, bad):
    module = _module()
    pool = _pool(module)
    pages = pool.alloc(1)
    slots = pool.slots(pages, 0, PAGE_TOKENS)
    width = module.indexer.index_head_dim
    keys = torch.zeros(
        PAGE_TOKENS if bad == "width" else PAGE_TOKENS - 1,
        width - 1 if bad == "width" else width,
    )
    with pytest.raises(ValueError, match="index key"):
        dsa_materialize_pool_segment_index_keys(
            module, keys, pool=pool, indexer_row=pool.indexer_row[1],
            slots=slots)


def test_a_pool_without_indexer_rows_refuses_an_index_key_load(seams):
    module = _module()
    pool = _pool(module)
    pool.indexer = None
    with pytest.raises(RuntimeError, match="no indexer rows"):
        dsa_materialize_pool_segment_index_keys(
            module, torch.zeros(1, module.indexer.index_head_dim),
            pool=pool, indexer_row=0, slots=torch.zeros(1, dtype=torch.long))


# --------------------------------------------------------------------------- #
#  4. The pooled offload writes both regions identically                      #
# --------------------------------------------------------------------------- #


class _RowView:
    """Records one region's row writes."""

    def __init__(self, *, page_split=False):
        self.is_page_split = page_split
        self.calls = []

    def async_offload_layer_kv_range_to_host(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(wait=lambda: None)


def _bind_pool_offload(monkeypatch, *, aux, bounds, starts, ids):
    monkeypatch.setattr(
        AttnWrapperBase, "prefill_pool_batch",
        SimpleNamespace(row_bounds=bounds, items=()), raising=False)
    monkeypatch.setattr(
        AttnWrapperBase, "prefill_pool_row_starts", starts, raising=False)
    monkeypatch.setattr(AttnWrapperBase, "cur_batch", ids, raising=False)
    monkeypatch.setattr(
        AttnWrapperBase, "prefill_pool_offload_recorder", None, raising=False)
    monkeypatch.setattr(
        AttnWrapperBase, "host_paged_kv_worker_view_aux", aux, raising=False)
    pinned, tracked = [], []
    monkeypatch.setattr(
        AttnWrapperBase, "pin_prefill_offload_tensor",
        staticmethod(lambda tensor, layer: pinned.append(layer)), raising=False)
    monkeypatch.setattr(
        AttnWrapperBase, "track_prefill_offload_task",
        staticmethod(lambda task, layer: tracked.append(layer)), raising=False)
    return pinned, tracked


def test_the_pooled_offload_writes_both_regions_at_the_same_positions(
    seams, monkeypatch
):
    from batchgen.models.moonshotai.kimi_linear.wrappers import (
        KimiLinearAttnWrapper,
    )

    module = _module()
    latent_view = _RowView()
    aux_view = _RowView(page_split=True)
    bounds = (0, 3, 7)
    starts = [0, 12]
    pinned, tracked = _bind_pool_offload(
        monkeypatch, aux=aux_view, bounds=bounds, starts=starts, ids=[-11, 44])
    hidden, _ = _packed([7], seed=17)
    with torch.no_grad():
        keys = dsa_prefill.prefill_index_keys(module.indexer, hidden)
        latent = torch.cat(
            torch.split(
                module.kv_a_proj_with_mqa(hidden),
                [module.kv_lora_rank, module.qk_rope_head_dim], dim=-1,
            ), dim=-1,
        )
    wrapper = SimpleNamespace(
        layer_idx=5,
        core_engine=SimpleNamespace(host_paged_kv_worker_view=latent_view),
    )
    KimiLinearAttnWrapper._offload_prefill_pool_rows(wrapper, latent, keys)

    assert len(latent_view.calls) == len(aux_view.calls) == 2
    for row, (primary, auxiliary) in enumerate(
        zip(latent_view.calls, aux_view.calls)
    ):
        tokens = bounds[row + 1] - bounds[row]
        for call in (primary, auxiliary):
            assert call["layer_idx"] == 5
            assert call["sequence_ids"] == [[-11, 44][row]]
            assert call["raw_start_positions"] == [starts[row]]
            assert call["token_counts"] == [tokens]
            assert call["v_tensor"] is None
        assert primary.get("scale_tensor") is None
        assert auxiliary["k_tensor"].shape == (1, tokens, 1, keys.shape[-1])
        assert auxiliary["k_tensor"].dtype is torch.float8_e4m3fn
        assert int(auxiliary["scale_tensor"].numel()) == tokens
        # The bytes are the shipped recipe's, on this row's own keys.
        expect_fp8, expect_scales = QUANT.quantize_indexer_k_fp8(
            keys[bounds[row]:bounds[row + 1]].unsqueeze(1)
        )
        assert torch.equal(
            auxiliary["k_tensor"].squeeze(0).view(torch.uint8),
            expect_fp8.view(torch.uint8),
        )
        assert torch.equal(
            auxiliary["scale_tensor"].reshape(-1), expect_scales.reshape(-1))
    # Both sources pinned and both tasks tracked, on this layer.
    assert pinned == [5, 5, 5, 5, 5, 5] and tracked == [5, 5, 5, 5]


@pytest.mark.parametrize("aux, match", [
    (None, "no auxiliary Host"),
    (_RowView(page_split=False), "not page-split"),
])
def test_the_pooled_offload_refuses_a_wrong_aux_region(
    seams, monkeypatch, aux, match
):
    from batchgen.models.moonshotai.kimi_linear.wrappers import (
        KimiLinearAttnWrapper,
    )

    module = _module()
    latent_view = _RowView()
    _bind_pool_offload(
        monkeypatch, aux=aux, bounds=(0, 2), starts=[0], ids=[-11])
    wrapper = SimpleNamespace(
        layer_idx=5,
        core_engine=SimpleNamespace(host_paged_kv_worker_view=latent_view),
    )
    width = module.kv_lora_rank + module.qk_rope_head_dim
    with pytest.raises(RuntimeError, match=match):
        KimiLinearAttnWrapper._offload_prefill_pool_rows(
            wrapper, torch.zeros(2, width),
            torch.zeros(2, module.indexer.index_head_dim),
        )
    assert latent_view.calls == []


def test_a_dense_pool_row_still_offloads_the_latent_alone(seams, monkeypatch):
    """K3 (no indexer) keeps the single-region pooled offload untouched."""
    from batchgen.models.moonshotai.kimi_linear.wrappers import (
        KimiLinearAttnWrapper,
    )

    latent_view = _RowView()
    _bind_pool_offload(
        monkeypatch, aux=None, bounds=(0, 2), starts=[7], ids=[-11])
    wrapper = SimpleNamespace(
        layer_idx=2,
        core_engine=SimpleNamespace(host_paged_kv_worker_view=latent_view),
    )
    KimiLinearAttnWrapper._offload_prefill_pool_rows(
        wrapper, torch.zeros(2, 56))
    assert len(latent_view.calls) == 1
    assert latent_view.calls[0].get("scale_tensor") is None
