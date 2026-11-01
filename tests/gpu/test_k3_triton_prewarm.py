# ---------------------------------------------------------------------------- #
#  BatchGen — Kimi-K3                                                          #
#  Copyright (c) 2025-2026 BatchGen Team                                       #
#  Licensed under the Apache License, Version 2.0                              #
# ---------------------------------------------------------------------------- #
"""After the K3 Triton prewarm, random prefill shapes compile nothing.

One KDA layer with the TP8-local K3 shapes (12 heads x 128, hidden 7168,
FP32 conv weights) runs the prewarm's KDA prefill table through the real
``kda_prefill_serving``; the late-compile guard is then armed strict and
random packed micro-batches, short and over the segment cap, run through the
same entry point. A ``LateTritonCompileError`` names the kernel and the key
the table missed; add a shape class to ``KDA_PREFILL_SHAPES`` for it.

Run on a cold ``TRITON_CACHE_DIR`` to exercise compilation; a warm cache only
proves the in-process coverage, which is what the guard checks either way.
"""

from __future__ import annotations

import os
import random
import types

import pytest
import torch


if os.environ.get("K3_GPU_STAGE") == "1" and not torch.cuda.is_available():
    raise RuntimeError("K3_GPU_STAGE=1 requires a visible CUDA device")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="staged Kimi-K3 CUDA test"
)

HIDDEN, HEADS, HEAD_DIM, CONV = 7168, 12, 128, 4
SEGMENT_TOKENS = 4096
MAX_PREFILL_TOKENS = 131_072  # ModuleBatchingConfig.prefill_micro_batch_token_cap default


def _layer():
    """``KimiKDAAttention`` at one TP8 rank's K3 shapes.

    BF16 weights except A_log, dt_bias and the conv weights, which K3 ships
    in FP32. The live prewarm drives the loaded modules, so a dtype guessed
    wrong here weakens this test only, not the startup coverage.
    """
    from batchgen.models.moonshotai.kimi_linear.model import KimiKDAAttention
    from batchgen.models.moonshotai.kimi_linear.serving_modules import (
        kda_prefill_serving,
    )

    torch.manual_seed(20261101)
    cfg = types.SimpleNamespace(
        hidden_size=HIDDEN, kda_conv_size=CONV, kda_head_dim=HEAD_DIM,
        kda_num_heads=HEADS, kda_use_full_rank_gate=True,
        kda_gate_lower_bound=-5.0, rms_norm_eps=1e-5,
    )
    layer = KimiKDAAttention(cfg, 0)
    with torch.no_grad():
        layer.dt_bias.copy_(torch.randn_like(layer.dt_bias) * 0.1)
    layer = layer.cuda()
    for name, param in layer.named_parameters():
        fp32 = name in ("A_log", "dt_bias") or "conv1d" in name
        param.data = param.data.to(torch.float32 if fp32 else torch.bfloat16)
    layer.attn_tp_size = 1
    layer.kda_prefill_serving = types.MethodType(kda_prefill_serving, layer)
    return layer


def _template():
    """Per-slot pool layout of the live KDA state manager."""
    proj = HEADS * HEAD_DIM
    conv = lambda: torch.zeros(1, proj, CONV - 1, dtype=torch.bfloat16,  # noqa: E731
                               device="cuda")
    return types.SimpleNamespace(
        conv_q=conv(), conv_k=conv(), conv_v=conv(),
        recurrent_pool=torch.zeros(1, HEADS, HEAD_DIM, HEAD_DIM,
                                   dtype=torch.float32, device="cuda"),
    )


def _random_lengths(rng):
    count = rng.choice((1, 2, 3, 7, 16, 17, 31))
    ranges = ((1, 4), (1, 300), (1, 2000), (SEGMENT_TOKENS - 70,
              SEGMENT_TOKENS + 70), (1, 2 * SEGMENT_TOKENS))
    return [rng.randint(*rng.choice(ranges)) for _ in range(count)]


def _capped_lengths(rng):
    """A micro-batch never exceeds the prefill micro-batch token cap."""
    lengths, total = [], 0
    for n in _random_lengths(rng):
        if total + n > MAX_PREFILL_TOKENS:
            break
        lengths.append(n)
        total += n
    return lengths or [1]


def test_random_prefill_shapes_compile_nothing_after_prewarm():
    from batchgen.models.moonshotai.kimi_linear import triton_prewarm as tp

    recorder = tp.get_recorder()
    if not recorder.install():
        pytest.skip("this Triton exposes no compile hooks")
    layer, template = _layer(), _template()
    entries = tp.kda_prefill_entries(layer, template, SEGMENT_TOKENS,
                                     max_prefill_tokens=MAX_PREFILL_TOKENS)
    with torch.inference_mode():
        reports = tp.run_prewarm(entries, recorder=recorder)
    assert reports[0].failed == 0

    rng = random.Random(20261101)
    recorder.guard.arm(strict=True)
    try:
        with torch.inference_mode():
            for _ in range(32):
                lengths = _capped_lengths(rng)
                shape = tp.KDAPrefillShape(
                    "random", tuple((0, n) for n in lengths),
                    slot_offset=rng.choice((0, 0, 1)),
                )
                tp._run_kda_prefill(layer, template, shape, SEGMENT_TOKENS)
                torch.cuda.synchronize()
    finally:
        recorder.guard.disarm()
