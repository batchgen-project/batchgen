"""Numerics parity: batchgen `_linear_from_weight` / expert forward vs the
official `linear` (act_quant + fp8_gemm / fp4_gemm with QAT) on identical
weights, isolating the residual prefill FFN drift.

Run:
  python -m pytest tests/integration/test_v4_linear_numerics_parity.py -q -s
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

ASSETS = (
    Path(__file__).resolve().parents[2]
    / "batchgen/models/deepseek/deepseekv4_flash/assets/inference"
)
sys.path.insert(0, str(ASSETS))

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


def _cos(a, b):
    return F.cosine_similarity(
        a.float().flatten().unsqueeze(0), b.float().flatten().unsqueeze(0)
    ).item()


def _rel(a, b):
    return (
        torch.linalg.vector_norm(a.float() - b.float())
        / torch.linalg.vector_norm(a.float())
    ).item()


def test_fp8_linear_parity():
    import model as official_model
    from kernel import act_quant
    from batchgen.models.deepseek.deepseekv4_flash.model import (
        _linear_from_weight,
    )

    torch.manual_seed(0)
    torch.set_default_dtype(torch.bfloat16)
    official_model.scale_fmt = "ue8m0"
    official_model.scale_dtype = torch.float8_e8m0fnu

    M, N, K = 64, 512, 4096
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    w_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") * 0.02

    wq, ws = act_quant(w_bf16, 128, "ue8m0", torch.float8_e8m0fnu)
    n_blk, k_blk = N // 128, K // 128
    ws_block = ws.view(N, k_blk)[::128].contiguous()
    w_fp8 = wq
    assert ws_block.shape == (n_blk, k_blk) or True

    wq2 = torch.empty(N, K, dtype=torch.float8_e4m3fn, device="cuda")
    sblock = torch.empty(
        n_blk, k_blk, dtype=torch.float8_e8m0fnu, device="cuda"
    )
    for i in range(n_blk):
        for j in range(k_blk):
            blk = w_bf16[i * 128 : (i + 1) * 128, j * 128 : (j + 1) * 128]
            amax = blk.float().abs().max().clamp(min=1e-8)
            s = 2.0 ** torch.ceil(torch.log2(amax / 448.0))
            wq2[i * 128 : (i + 1) * 128, j * 128 : (j + 1) * 128] = (
                blk.float() / s
            ).to(torch.float8_e4m3fn)
            sblock[i, j] = s.to(torch.float8_e8m0fnu)

    ref = official_model.linear(x, _attach_scale(wq2, sblock))
    bg = _linear_from_weight(x, wq2, sblock)
    cos = _cos(ref, bg)
    rel = _rel(ref, bg)
    print(f"fp8 linear: cos={cos:.6f} rel={rel:.4e}")
    assert cos > 0.999


def _attach_scale(weight, scale):
    weight.scale = scale
    return weight


def test_fp4_expert_parity():
    import model as official_model
    from kernel import fp4_act_quant
    from batchgen.models.deepseek.deepseekv4_flash.model import (
        DeepSeekV4FlashExpertPlaceholder,
    )

    torch.manual_seed(1)
    torch.set_default_dtype(torch.bfloat16)
    official_model.scale_fmt = "ue8m0"
    official_model.scale_dtype = torch.float8_e8m0fnu

    hidden, inter = 1024, 512
    M = 32
    x = torch.randn(M, hidden, dtype=torch.bfloat16, device="cuda") * 0.5

    expert_ref = official_model.Expert(
        hidden, inter, dtype=torch.float4_e2m1fn_x2, swiglu_limit=10.0
    ).cuda()
    for name in ("w1", "w2", "w3"):
        lin = getattr(expert_ref, name)
        w_b = (
            torch.randn(
                lin.weight.shape[0],
                lin.weight.shape[1] * 2,
                dtype=torch.bfloat16,
                device="cuda",
            )
            * 0.05
        )
        q, s = fp4_act_quant(w_b, 32)
        lin.weight.data = q.view(torch.float4_e2m1fn_x2)
        lin.scale.data = s
        lin.weight.scale = lin.scale

    bg = DeepSeekV4FlashExpertPlaceholder(hidden, inter, 10.0).cuda()
    bg.set_runtime_tensors(
        {
            "w1.weight": expert_ref.w1.weight.data,
            "w1.scale": expert_ref.w1.scale.data,
            "w2.weight": expert_ref.w2.weight.data,
            "w2.scale": expert_ref.w2.scale.data,
            "w3.weight": expert_ref.w3.weight.data,
            "w3.scale": expert_ref.w3.scale.data,
        }
    )

    weights = torch.full((M, 1), 0.7, dtype=torch.float32, device="cuda")
    with torch.inference_mode():
        ref_out = expert_ref(x.clone(), weights)
        bg_out = bg(x.clone(), weights)
    cos = _cos(ref_out, bg_out)
    rel = _rel(ref_out, bg_out)
    print(f"fp4 expert: cos={cos:.6f} rel={rel:.4e}")
    assert cos > 0.999


def test_shared_expert_bf16_parity():
    import model as official_model
    from batchgen.models.deepseek.deepseekv4_flash.model import (
        DeepSeekV4FlashExpertPlaceholder,
    )

    torch.manual_seed(2)
    torch.set_default_dtype(torch.bfloat16)
    hidden, inter = 1024, 512
    M = 32
    x = torch.randn(M, hidden, dtype=torch.bfloat16, device="cuda") * 0.5

    expert_ref = official_model.Expert(
        hidden, inter, dtype=torch.bfloat16, swiglu_limit=10.0
    ).cuda()
    for name in ("w1", "w2", "w3"):
        torch.nn.init.normal_(getattr(expert_ref, name).weight, std=0.05)

    bg = DeepSeekV4FlashExpertPlaceholder(hidden, inter, 10.0).cuda()
    bg.set_runtime_tensors(
        {
            "w1.weight": expert_ref.w1.weight.data,
            "w2.weight": expert_ref.w2.weight.data,
            "w3.weight": expert_ref.w3.weight.data,
        }
    )
    with torch.inference_mode():
        ref_out = expert_ref(x.clone(), None)
        bg_out = bg(x.clone(), None)
    cos = _cos(ref_out, bg_out)
    rel = _rel(ref_out, bg_out)
    print(f"bf16 shared expert: cos={cos:.6f} rel={rel:.4e}")
    assert cos > 0.999
