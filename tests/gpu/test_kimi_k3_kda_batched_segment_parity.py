# ---------------------------------------------------------------------------- #
#  BatchGen — Kimi-K3                                                          #
#  copyright (c) EfficientMoE team 2025                                        #
#  Licensed under the Apache License, Version 2.0                              #
# ---------------------------------------------------------------------------- #
"""fla chunk_kda: the batch-1 call is bit-identical to the varlen call for a
single sequence, with the exact kwargs kda_prefill_serving passes."""

from __future__ import annotations

import os

import pytest
import torch


if os.environ.get("K3_GPU_STAGE") == "1" and not torch.cuda.is_available():
    raise RuntimeError("K3_GPU_STAGE=1 requires a visible CUDA device")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="staged Kimi-K3 CUDA test"
)


@pytest.mark.parametrize("seg", [64, 4096, 8192])
def test_batched_call_matches_varlen_single_sequence(seg):
    from fla.ops.kda import chunk_kda

    H, D = 12, 128
    gen = torch.Generator(device="cuda").manual_seed(20260903 + seg)
    q = torch.randn((1, seg, H, D), device="cuda", dtype=torch.bfloat16, generator=gen)
    k = torch.randn((1, seg, H, D), device="cuda", dtype=torch.bfloat16, generator=gen)
    v = torch.randn((1, seg, H, D), device="cuda", dtype=torch.bfloat16, generator=gen)
    f = torch.randn((1, seg, H, D), device="cuda", dtype=torch.bfloat16, generator=gen)
    beta = torch.randn((1, seg, H), device="cuda", dtype=torch.bfloat16, generator=gen)
    h0 = torch.randn((1, H, D, D), device="cuda", dtype=torch.float32, generator=gen)
    A_log = torch.randn((H,), device="cuda", dtype=torch.float32, generator=gen) * 0.5
    dt_bias = torch.randn((H,), device="cuda", dtype=torch.float32, generator=gen) * 0.1
    kwargs = dict(
        A_log=A_log, dt_bias=dt_bias,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        lower_bound=-5.0,
        output_final_state=True,
    )
    cu = torch.tensor([0, seg], dtype=torch.long, device="cuda")
    o_v, s_v = chunk_kda(q=q, k=k, v=v, g=f, beta=beta, initial_state=h0, cu_seqlens=cu, **kwargs)
    o_b, s_b = chunk_kda(q=q, k=k, v=v, g=f, beta=beta, initial_state=h0, **kwargs)
    torch.cuda.synchronize()
    assert torch.equal(o_v, o_b)
    assert torch.equal(s_v, s_b)
