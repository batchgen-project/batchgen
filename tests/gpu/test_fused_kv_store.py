# ---------------------------------------------------------------------------- #
#  BatchGen — GLM-5                                                            #
#  Copyright (c) 2025-2026 BatchGen Team                                       #
#  Licensed under the Apache License, Version 2.0                              #
# ---------------------------------------------------------------------------- #
"""REMOTE-ONLY GPU exactness gate for the fused paged-KV store (slice 1b-iii).

The fused norm+rope kernel's lane 0 now stores the processed row straight
into the paged KV cache, replacing the separate
``_paged_cache_update_with_page_table_kernel`` launch. The contract is
byte-for-byte: the cache after the fused call must equal the cache after
"fused kernel WITHOUT the store, then the separate update kernel" — the
exact pre-change sequence — including the two negative gates:

  N1  a slot<0 sentinel row writes nothing;
  N2  a row masked by ``row_valid_mask`` writes nothing (the separate
      kernel got this for free because callers hand it the already-gated
      slot tensor; the fused kernel must hold it even for an UNGATED slot).
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("triton", reason="module under test imports triton")

STAGE_ENV = "BATCHGEN_FUSED_ROPE_GPU_STAGE"

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="staged fused paged-KV-store CUDA exactness test",
)

KV_LORA, ROPE, HEADS = 512, 64, 64
TOTAL = KV_LORA + ROPE
PAGE = 64
EPS = 1e-6


def _production():
    pytest.importorskip("batchgen_kernels")
    import importlib

    frr = importlib.import_module("batchgen_kernels.triton.fused_rmsnorm_rope")
    kvc = importlib.import_module("batchgen_kernels.triton.kv_cache")
    return frr, kvc


def _inputs(bsz: int, seed: int = 20261023):
    g = torch.Generator(device="cuda")
    g.manual_seed(seed + bsz)
    kv = torch.randn((bsz, 1, TOTAL), device="cuda", generator=g).to(torch.bfloat16)
    q_pe = torch.randn((bsz, HEADS, 1, ROPE), device="cuda", generator=g).to(
        torch.bfloat16
    )
    cos = torch.randn((4096, ROPE), device="cuda", generator=g)
    sin = torch.randn((4096, ROPE), device="cuda", generator=g)
    positions = torch.randint(
        0, 4096, (bsz, 1), device="cuda", generator=g, dtype=torch.int64
    )
    weight = torch.randn((KV_LORA,), device="cuda", generator=g).to(torch.bfloat16)
    return kv, q_pe, cos, sin, positions, weight


def _paged_env(bsz: int, seed: int = 7):
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    max_pages_per_seq = 4
    # One page per (row, page-slot): no two rows share a destination, so the
    # byte comparison is order-independent.
    num_pages = bsz * max_pages_per_seq
    # Poison so an unwritten byte is visible.
    cache = (
        torch.randn((num_pages, PAGE, 1, TOTAL), device="cuda", generator=g) * 17.0
    ).to(torch.bfloat16)
    page_table = torch.arange(
        num_pages, device="cuda", dtype=torch.int32
    ).reshape(bsz, max_pages_per_seq)
    slots = torch.arange(bsz, device="cuda", dtype=torch.int32)
    toks = torch.randint(
        0, PAGE * max_pages_per_seq, (bsz,), device="cuda", generator=g
    ).to(torch.int32)
    return cache, page_table, slots, toks


def _run_pre_change(frr, kvc, args, cache, page_table, slots, toks, mask=None):
    kv, q_pe, cos, sin, positions, weight = args
    bsz = kv.shape[0]
    q = q_pe.clone()
    out = torch.empty((bsz, 1, TOTAL), dtype=torch.bfloat16, device="cuda")
    processed = frr.fused_rmsnorm_rope_with_q_native(
        kv, q, cos, sin, positions, weight, KV_LORA, ROPE, eps=EPS,
        out=out, row_valid_mask=mask,
    )
    kvc.run_paged_kv_token_update_fused(
        k_cache=cache,
        k_tokens=processed.view(bsz, -1),
        page_table=page_table,
        slot_indices=slots,
        token_indices=toks,
        page_size_tokens=PAGE,
    )
    torch.cuda.synchronize()
    return processed, q


def _run_fused(frr, args, cache, page_table, slots, toks, mask=None):
    kv, q_pe, cos, sin, positions, weight = args
    bsz = kv.shape[0]
    q = q_pe.clone()
    out = torch.empty((bsz, 1, TOTAL), dtype=torch.bfloat16, device="cuda")
    processed = frr.fused_rmsnorm_rope_with_q_native(
        kv, q, cos, sin, positions, weight, KV_LORA, ROPE, eps=EPS,
        out=out, row_valid_mask=mask,
        cache_tensor=cache, page_table=page_table, slot_indices=slots,
        token_indices=toks, page_size_tokens=PAGE,
    )
    torch.cuda.synchronize()
    return processed, q


@pytest.mark.parametrize("bsz", [1, 7, 64, 128])
def test_fused_store_matches_separate_kernel_bytewise(bsz):
    frr, kvc = _production()
    args = _inputs(bsz)
    cache_ref, pt, slots, toks = _paged_env(bsz)
    cache_new = cache_ref.clone()

    p_ref, q_ref = _run_pre_change(frr, kvc, args, cache_ref, pt, slots, toks)
    p_new, q_new = _run_fused(frr, args, cache_new, pt, slots, toks)

    assert torch.equal(p_new, p_ref)
    assert torch.equal(q_new, q_ref)
    assert torch.equal(cache_new, cache_ref), (
        f"bsz={bsz}: fused paged store differs from the separate kernel "
        f"({int((cache_new != cache_ref).sum())} bytes-elements)"
    )


def test_sentinel_row_writes_nothing():
    frr, kvc = _production()
    bsz = 8
    args = _inputs(bsz)
    cache_ref, pt, slots, toks = _paged_env(bsz)
    slots = slots.clone()
    slots[3] = -1
    slots[6] = -1
    cache_new = cache_ref.clone()
    before = cache_ref.clone()

    _run_pre_change(frr, kvc, args, cache_ref, pt, slots, toks)
    _run_fused(frr, args, cache_new, pt, slots, toks)

    assert torch.equal(cache_new, cache_ref)
    # And the sentinel rows' would-be destinations kept their poison: compare
    # the whole cache against "written rows only" reconstruction.
    valid = [i for i in range(bsz) if i not in (3, 6)]
    touched_pages = {
        int(pt[i, int(toks[i].item()) // PAGE].item()) for i in valid
    }
    for p in range(cache_ref.shape[0]):
        if p not in touched_pages:
            assert torch.equal(cache_new[p], before[p])


def test_masked_row_with_live_slot_writes_nothing():
    """The fused gate must hold even when the slot tensor is NOT pre-gated."""
    frr, kvc = _production()
    bsz = 8
    args = _inputs(bsz)
    cache_before, pt, slots, toks = _paged_env(bsz)
    cache_new = cache_before.clone()

    mask = torch.ones(bsz, dtype=torch.bool, device="cuda")
    mask[2] = False  # live slot, masked row

    _run_fused(frr, args, cache_new, pt, slots, toks, mask=mask)

    page2 = int(pt[2, int(toks[2].item()) // PAGE].item())
    off2 = int(toks[2].item()) % PAGE
    assert torch.equal(cache_new[page2, off2], cache_before[page2, off2]), (
        "masked row with a live slot must not write the cache"
    )
    # Every unmasked row must still have written its destination.
    for i in range(bsz):
        if i == 2:
            continue
        pg = int(pt[i, int(toks[i].item()) // PAGE].item())
        off = int(toks[i].item()) % PAGE
        assert not torch.equal(cache_new[pg, off], cache_before[pg, off])
