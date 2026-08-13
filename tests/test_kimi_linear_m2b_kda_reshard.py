# ---------------------------------------------------------------------------- #
#  BatchGen — Kimi-Linear                                                       #
#  copyright (c) EfficientMoE team 2025                                         #
#  Licensed under the Apache License, Version 2.0                               #
# ---------------------------------------------------------------------------- #
"""M2b Tier-1 (CPU): prefill->decode KDA state reshard parity.

Validates piece 3's slice/gather arithmetic: the full 96-head recurrent + conv
state head-shards into G slices of 96/G and gathers back BIT-EXACT (P0.6
head-independence), and the rank->group move plan routes shard i to rank
group_base+i while preserving values. Matches the M2a head-slice convention
(test_kimi_k3_kda_head_parallel_parity): recurrent slices the head axis,
conv slices the conv_dim (heads*head_dim) axis.

NON-VACUITY: a wrong (rotated) head slice makes the gather differ.

Run: python -m pytest tests/test_kimi_linear_m2b_kda_reshard.py -x -q -rA
"""

from __future__ import annotations

import pytest
import torch

from batchgen.kv_cache.kda_state_reshard import (
    build_reshard_move_plan,
    gather_conv,
    gather_recurrent,
    head_shard_conv,
    head_shard_recurrent,
)

NUM_HEADS = 96
HEAD_DIM = 16          # small (K,K) state matrices keep the CPU test fast
NUM_LAYERS = 3         # the "L" in [L, 96, K, K]
CONV_W = 4             # W; conv pools hold W-1 raw inputs per slot


def _full_recurrent():
    torch.manual_seed(0)
    return torch.randn(NUM_LAYERS, NUM_HEADS, HEAD_DIM, HEAD_DIM,
                       dtype=torch.float32)


def _full_conv():
    torch.manual_seed(1)
    return torch.randn(NUM_LAYERS, NUM_HEADS * HEAD_DIM, CONV_W - 1,
                       dtype=torch.bfloat16)


@pytest.mark.parametrize("G", [2, 4, 8])
def test_recurrent_head_shard_gather_bit_exact(G):
    full = _full_recurrent()
    shards = [head_shard_recurrent(full, G, g, NUM_HEADS) for g in range(G)]
    for s in shards:
        assert s.shape == (NUM_LAYERS, NUM_HEADS // G, HEAD_DIM, HEAD_DIM)
    rebuilt = gather_recurrent(shards)
    max_d = (rebuilt.float() - full.float()).abs().max().item()
    assert max_d == 0.0, f"recurrent reshard not bit-exact (G={G}): {max_d}"


@pytest.mark.parametrize("G", [2, 4, 8])
def test_conv_head_shard_gather_bit_exact(G):
    full = _full_conv()
    shards = [head_shard_conv(full, G, g, NUM_HEADS, HEAD_DIM)
              for g in range(G)]
    for s in shards:
        assert s.shape == (NUM_LAYERS, (NUM_HEADS // G) * HEAD_DIM, CONV_W - 1)
    rebuilt = gather_conv(shards)
    max_d = (rebuilt.float() - full.float()).abs().max().item()
    assert max_d == 0.0, f"conv reshard not bit-exact (G={G}): {max_d}"


def test_recurrent_nonvacuity_wrong_slice():
    """A rotated (wrong) head block must break the gather — the test is real."""
    G = 8
    full = _full_recurrent()
    Hl = NUM_HEADS // G
    wrong = []
    for g in range(G):
        bad_g = (g + 1) % G            # shard g takes the NEXT block's heads
        wrong.append(full.narrow(full.ndim - 3, bad_g * Hl, Hl))
    rebuilt = gather_recurrent(wrong)
    max_d = (rebuilt.float() - full.float()).abs().max().item()
    assert max_d > 0.0, "non-vacuity failed: wrong head slice still matched"


def test_conv_nonvacuity_wrong_slice():
    G = 8
    full = _full_conv()
    blk = (NUM_HEADS // G) * HEAD_DIM
    wrong = []
    for g in range(G):
        bad_g = (g + 1) % G
        wrong.append(full.narrow(full.ndim - 2, bad_g * blk, blk))
    rebuilt = gather_conv(wrong)
    max_d = (rebuilt.float() - full.float()).abs().max().item()
    assert max_d > 0.0, "non-vacuity failed: wrong conv slice still matched"


def test_move_plan_routes_shard_to_group_rank():
    G = 8
    prefill_rank = 5
    group_id = 2
    plan = build_reshard_move_plan(prefill_rank, group_id, G)
    assert len(plan) == G
    group_base = group_id * G          # 16
    for i, (src, dst, shard_idx) in enumerate(plan):
        assert src == prefill_rank
        assert shard_idx == i
        assert dst == group_base + i   # shard i -> rank group_base+i


def test_rank_to_group_move_preserves_values():
    """Head-slice -> (byte-preserving transport) -> gather round-trips exactly."""
    G = 8
    prefill_rank = 5
    group_id = 1
    full = _full_recurrent()
    shards = [head_shard_recurrent(full, G, g, NUM_HEADS) for g in range(G)]
    plan = build_reshard_move_plan(prefill_rank, group_id, G)

    # Deliver shard `shard_idx` to rank `dst`; place it at that rank's group
    # position. Gloo send/recv preserves bytes, so transport == identity here.
    delivered = [None] * G
    for src, dst, shard_idx in plan:
        pos = dst - group_id * G
        delivered[pos] = shards[shard_idx].clone()
    rebuilt = gather_recurrent(delivered)
    assert (rebuilt.float() - full.float()).abs().max().item() == 0.0


def test_shard_validates_shapes():
    full = _full_recurrent()
    with pytest.raises(ValueError):
        head_shard_recurrent(full, 7, 0, NUM_HEADS)     # 96 % 7 != 0
    with pytest.raises(ValueError):
        head_shard_recurrent(full, 8, 8, NUM_HEADS)     # rank out of range
    with pytest.raises(ValueError):
        head_shard_recurrent(full, 8, 0, NUM_HEADS + 1)  # wrong num_heads
