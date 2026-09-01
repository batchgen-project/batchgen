# ---------------------------------------------------------------------------- #
#  BatchGen — Kimi-Linear                                                       #
#  copyright (c) EfficientMoE team 2025                                         #
#  Licensed under the Apache License, Version 2.0                               #
# ---------------------------------------------------------------------------- #
"""M2b Tier-1 (CPU): resident-EP MoE intra-group scatter/gather round-trip.

Validates piece 4 without real collectives, by simulating a decode group's G
ranks in one process: the group's B_grp replicated rows are scattered into G
distinct contiguous slices (DP-32 contract restored), passed through a stubbed
row-preserving ``resident.forward``, then reassembled to the full B_grp on
every rank. Asserts:

  * the scatter is a distinct partition covering all rows (order preserved);
  * the gather reproduces the reference transform bit-exact;
  * an empty rank (B_grp < G) still participates and reassembly stays correct;
  * ntp == ceil(B_grp / G).

NON-VACUITY: reassembling with a wrong split differs from the reference.

Run: python -m pytest tests/test_kimi_linear_m2b_moe_scatter_gather.py -x -q -rA
"""

from __future__ import annotations

import math

import pytest
import torch

from batchgen.models.moonshotai.kimi_linear.moe_tp_reshard import (
    all_gather_rows_into,
    balanced_row_split,
    reassemble_rows,
    scatter_rows,
)

H = 8


def _stub_resident_forward(x_local):
    """Row-preserving stand-in for ResidentEPMoELayer.forward (a bias-free MLP
    maps 0->0, so an empty rank yields (0,H); a real transform is affine here)."""
    return x_local * 3.0 + 1.0


def _simulate_group(x, G):
    """Run scatter -> stub forward -> padded gather across G virtual ranks."""
    B = x.shape[0]
    splits = balanced_row_split(B, G)
    ntp = max((e - s) for s, e in splits)          # == ceil(B/G)
    per_rank_slices = [scatter_rows(x, G, g) for g in range(G)]
    routed = [_stub_resident_forward(s) for s in per_rank_slices]
    # Emulate all_gather_into_tensor: zero-pad each rank to ntp then stack.
    gathered = x.new_zeros((G, ntp, H))
    for g, r in enumerate(routed):
        if r.shape[0] > 0:
            gathered[g, : r.shape[0]] = r
    out = reassemble_rows(gathered, B, G)
    return out, per_rank_slices, splits, ntp


@pytest.mark.parametrize("G", [1, 2, 4, 8])
@pytest.mark.parametrize("B", [0, 1, 3, 8, 10, 17, 32])
def test_scatter_gather_roundtrip(B, G):
    torch.manual_seed(B * 100 + G)
    x = torch.randn(B, H, dtype=torch.float32)
    out, slices, splits, ntp = _simulate_group(x, G)

    # ntp is the post-scatter share.
    assert ntp == math.ceil(B / G) if B > 0 else ntp == 0

    # Distinct partition covering every row in order.
    covered = torch.cat(slices, dim=0) if slices else x.new_zeros((0, H))
    assert covered.shape[0] == B
    if B > 0:
        assert torch.equal(covered, x)             # concat restores order

    # Round-trip == reference transform, bit-exact.
    ref = _stub_resident_forward(x)
    assert torch.equal(out, ref), (B, G)


@pytest.mark.parametrize("G", [4, 8])
def test_empty_rank_still_participates(G):
    """B_grp < G: tail ranks get 0 rows but the gather is still correct."""
    B = G - 1                                       # guarantees >=1 empty rank
    torch.manual_seed(7)
    x = torch.randn(B, H, dtype=torch.float32)
    out, slices, splits, ntp = _simulate_group(x, G)

    empties = [g for g, (s, e) in enumerate(splits) if e - s == 0]
    assert empties, "expected at least one empty rank for B_grp<G"
    # An empty rank's stub forward yields (0,H) yet the reassembly is exact.
    for g in empties:
        assert slices[g].shape[0] == 0
    assert torch.equal(out, _stub_resident_forward(x))


def test_balanced_row_split_properties():
    for B in (0, 1, 5, 8, 10, 31):
        for G in (1, 2, 4, 8):
            splits = balanced_row_split(B, G)
            assert len(splits) == G
            assert splits[0][0] == 0
            assert splits[-1][1] == B
            # contiguous, non-overlapping, covers [0,B)
            for (s, e), (ns, _) in zip(splits, splits[1:]):
                assert e == ns and e >= s
            counts = [e - s for s, e in splits]
            assert sum(counts) == B
            if B > 0:
                assert max(counts) == math.ceil(B / G)
                assert max(counts) - min(counts) <= 1   # balanced


def test_scatter_gather_nonvacuity_wrong_split():
    """Reassembling with a shifted split must corrupt the round-trip."""
    B, G = 10, 4
    torch.manual_seed(3)
    x = torch.randn(B, H, dtype=torch.float32)
    _, _, splits, ntp = _simulate_group(x, G)
    routed = [_stub_resident_forward(scatter_rows(x, G, g)) for g in range(G)]
    gathered = x.new_zeros((G, ntp, H))
    for g, r in enumerate(routed):
        if r.shape[0] > 0:
            gathered[g, : r.shape[0]] = r

    # WRONG: swap two ranks' contributions -> the contiguous split no longer
    # reconstructs the original row order.
    bad = gathered.clone()
    bad[0], bad[1] = gathered[1].clone(), gathered[0].clone()
    out_bad = reassemble_rows(bad, B, G)
    ref = _stub_resident_forward(x)
    assert not torch.equal(out_bad, ref), "non-vacuity failed: wrong split matched"


@pytest.mark.parametrize(("B", "G"), [(17, 8), (3, 8), (16, 4)])
def test_chunked_gather_into_reassembles_exact_rows(monkeypatch, B, G):
    x = torch.arange(B * H, dtype=torch.float32).view(B, H)
    local = [scatter_rows(x, G, rank) for rank in range(G)]
    splits = balanced_row_split(B, G)
    ntp = max(end - start for start, end in splits)
    chunk_rows = 2

    for group_rank in range(G):
        call_index = 0

        def fake_all_gather(gathered, send, group):
            nonlocal call_index
            assert group == "fake-group"
            start = call_index * chunk_rows
            count = send.shape[0]
            for rank, rows in enumerate(local):
                chunk = rows.new_zeros((count, H))
                valid_end = min(start + count, rows.shape[0])
                if valid_end > start:
                    chunk[:valid_end - start].copy_(rows[start:valid_end])
                gathered[rank * count:(rank + 1) * count].copy_(chunk)
            call_index += 1

        monkeypatch.setattr(
            torch.distributed,
            "all_gather_into_tensor",
            fake_all_gather,
        )
        output = torch.empty_like(x)
        all_gather_rows_into(
            output,
            local[group_rank],
            B,
            G,
            group_rank,
            "fake-group",
            chunk_rows=chunk_rows,
        )
        assert call_index == math.ceil(ntp / chunk_rows)
        assert torch.equal(output, x)
