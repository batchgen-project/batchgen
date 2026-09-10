# ---------------------------------------------------------------------------- #
#  BatchGen — Kimi-Linear                                                        #
#  copyright (c) EfficientMoE team 2025                                          #
#  Licensed under the Apache License, Version 2.0                                #
# ---------------------------------------------------------------------------- #
"""M2a regression: A_log head-shard must survive the (1, 1, H, 1) checkpoint shape.

The Kimi-Linear-48B checkpoint ships ``A_log`` with shape ``(1, 1, kda_num_heads,
1)`` — the heads live on axis 2, not axis 0. ``_head_shard_kda_tensor`` originally
sliced dim 0 (``tensor[lo:hi]``); on that shape a dim-0 slice returns 0 rows for
every rank whose ``lo`` exceeds the size-1 leading axis (i.e. every rank but 0),
producing a 0-element / null-pointer tensor. The fla ``kda_gate_chunk_cumsum``
kernel then does ``tl.load(A_log + i_h)`` off the null pointer -> CUDA illegal
memory access at G=8 prefill (bug_log 2026-08-14).

This pins the flattened-head-axis slice. CPU-only (pure tensor slicing); the
method reads only the four ``_attn_tp_*`` attributes a stub supplies.
"""
import types

import pytest
import torch

from batchgen.models.moonshotai.kimi_linear.Parallel_Strategy_Manager import (
    KimiLinearParallelStrategyManager as PSM,
)


def _shard(name, tensor, G, Hl, rank, hd=128, *, is_k3=False):
    stub = types.SimpleNamespace(
        _attn_tp_head_dim=hd,
        _attn_tp_hl=Hl,
        _attn_tp_size=G,
        _attn_tp_rank=rank,
        _is_k3=is_k3,
    )
    return PSM._head_shard_kda_tensor(stub, name, tensor)


def test_alog_checkpoint_shape_shards_non_null_per_rank():
    """(1, 1, H, 1) -> every rank gets a contiguous (Hl,) slice of ITS heads."""
    G, Hl = 8, 4
    H = G * Hl  # 32
    A = torch.arange(H, dtype=torch.float32).reshape(1, 1, H, 1)
    for r in range(G):
        s = _shard("A_log", A, G, Hl, r).contiguous()
        assert s.numel() == Hl, (r, tuple(s.shape))
        assert s.data_ptr() != 0, f"rank {r} A_log is a null-pointer tensor"
        assert torch.equal(
            s.flatten(),
            torch.arange(r * Hl, (r + 1) * Hl, dtype=torch.float32),
        ), f"rank {r} got the wrong heads: {s.flatten().tolist()}"


def test_alog_flat_shape_unchanged():
    """A plain 1-D (H,) A_log shards identically (no regression for that layout)."""
    G, Hl = 4, 8
    H = G * Hl
    A = torch.arange(H, dtype=torch.float32)
    for r in range(G):
        s = _shard("A_log", A, G, Hl, r)
        assert torch.equal(
            s.flatten(),
            torch.arange(r * Hl, (r + 1) * Hl, dtype=torch.float32),
        )


def test_alog_wrong_numel_hard_fails():
    """A_log whose flattened size != kda_num_heads must raise, never silently pass."""
    A = torch.zeros(1, 1, 30, 1)  # 30 not = 8*4
    with pytest.raises(ValueError):
        _shard("A_log", A, 8, 4, 0)


def test_k3_padded_alog_shards_live_heads_not_padding_or_rank_zero():
    """K3's [128] checkpoint row is [96 live heads] plus zero padding."""
    G, Hl = 8, 12
    A = torch.arange(128, dtype=torch.float32)
    for r in range(G):
        s = _shard("A_log", A, G, Hl, r, is_k3=True)
        assert torch.equal(
            s.flatten(),
            torch.arange(r * Hl, (r + 1) * Hl, dtype=torch.float32),
        ), f"rank {r} did not receive its K3 A_log head slice"


def test_bproj_still_dim0_sharded():
    """b_proj is (H, hidden) with heads on dim 0 — its slice must be unchanged."""
    G, Hl = 8, 4
    H = G * Hl
    W = torch.arange(H * 5, dtype=torch.float32).reshape(H, 5)
    for r in range(G):
        s = _shard("b_proj", W, G, Hl, r)
        assert tuple(s.shape) == (Hl, 5)
        assert torch.equal(s, W[r * Hl:(r + 1) * Hl])
