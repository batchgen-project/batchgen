# ---------------------------------------------------------------------------- #
#  BatchGen — GLM-5                                                            #
#  Copyright (c) 2025-2026 BatchGen Team                                       #
#  Licensed under the Apache License, Version 2.0                              #
# ---------------------------------------------------------------------------- #
"""Launch-width contract of ``fused_rmsnorm_rope_with_q_native``.

The kv-lora RMSNorm reduction is launched single-pass whenever the rank fits
one block: ``BLOCK_SIZE = min(512, next_pow2(kv_lora_rank))``.  This test
records the actual launch kwargs through a stub kernel (no GPU needed beyond
a Triton import), so a regression back to the old 64-wide serial loop — or an
accidental unbounded width — fails here instead of only showing up as a
per-layer timing drift on the node.
"""

from __future__ import annotations

import importlib

import pytest
import torch

pytest.importorskip("triton", reason="module under test imports triton")

frr = importlib.import_module("batchgen_kernels.triton.fused_rmsnorm_rope")


class _RecordingKernel:
    """Stands in for the @triton.jit object: captures the launch kwargs."""

    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        def _launch(*args, **kwargs):
            self.calls.append({"grid": grid, "kwargs": kwargs})

        return _launch


def _call(monkeypatch, kv_lora_rank: int, rope_dim: int = 64):
    recorder = _RecordingKernel()
    monkeypatch.setattr(
        frr, "fused_rmsnorm_rope_with_q_native_kernel", recorder
    )
    bsz, heads = 4, 8
    total = kv_lora_rank + rope_dim
    kv = torch.zeros((bsz, 1, total), dtype=torch.bfloat16)
    q_pe = torch.zeros((bsz, heads, 1, rope_dim), dtype=torch.bfloat16)
    cos = torch.zeros((16, rope_dim), dtype=torch.float32)
    sin = torch.zeros((16, rope_dim), dtype=torch.float32)
    positions = torch.zeros((bsz, 1), dtype=torch.int64)
    weight = torch.zeros((kv_lora_rank,), dtype=torch.bfloat16)
    frr.fused_rmsnorm_rope_with_q_native(
        kv, q_pe, cos, sin, positions, weight, kv_lora_rank, rope_dim
    )
    assert len(recorder.calls) == 1
    return recorder.calls[0]


def test_glm52_rank_launches_single_pass(monkeypatch):
    call = _call(monkeypatch, kv_lora_rank=512)
    assert call["kwargs"]["BLOCK_SIZE"] == 512


@pytest.mark.parametrize(
    ("kv_lora_rank", "expected"),
    [
        (128, 128),   # exact power of two below the cap
        (100, 128),   # rounded up to one pass
        (513, 512),   # past the cap: capped, two-iteration loop
        (2048, 512),  # far past the cap: still capped
    ],
)
def test_width_is_next_pow2_capped_at_512(monkeypatch, kv_lora_rank, expected):
    call = _call(monkeypatch, kv_lora_rank=kv_lora_rank)
    assert call["kwargs"]["BLOCK_SIZE"] == expected


def test_head_tiling_unchanged(monkeypatch):
    """The width change must not touch the head-axis tiling policy."""
    call = _call(monkeypatch, kv_lora_rank=512)
    block_h, block_pairs, num_tiles = frr.select_q_rope_tiling(4, 8, 32)
    assert call["kwargs"]["BLOCK_H"] == block_h
    assert call["kwargs"]["BLOCK_PAIRS"] == block_pairs
    assert call["grid"] == (4, 1 + num_tiles)
