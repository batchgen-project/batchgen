"""GPU parity for the merged ``[q_b | wq_b]`` GLM-5 DSA decode GEMM (Group B).

Three claims, each at the real GLM-5.2 shapes (q_lora_rank 2048, q_b 64 x 256,
wq_b 32 x 128) and the decode bucket sizes:

1. **Bitwise, merged vs separate DeepGEMM** -- the acceptance criterion. One
   ``w8a8_deepgemm`` over the concatenated operand equals two separate
   ``w8a8_deepgemm`` calls on the same operands, ``torch.equal`` on both
   output slices, on the masked (captured) and the dense (eager) path. Every
   member starts on a 128-row boundary, so each output element sees the same
   k-block sequence and the same weight scale.
2. **Bitwise, strided vs contiguous RoPE entry point** -- the indexer RoPE
   reading the wq_b column slice in place produces the same bf16 and e4m3
   bytes as the original kernel on a contiguous copy, both rotation arms.
3. **Tolerance, merged wq_b vs the custom WGMMA path it replaces** -- an
   intentional numerics change (checkpoint block scales instead of a
   dequantize / per-32-row re-quantize round trip, and 128-block instead of
   per-row activation scales). Both are measured against the same fp32
   reference and reported; each must meet the FP8 GEMM ``calc_diff`` bound
   (``knowledge/agent-memory/rules/kernel-validation.md``).

Run on an SM90 node:
    pytest tests/gpu/test_glm52_q_b_wq_b_merge_parity.py -v -s
"""

from __future__ import annotations

import pytest
import torch


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="merged [q_b | wq_b] parity needs an SM90 CUDA device",
)

_Q_LORA_RANK = 2048
_Q_B_ROWS = 64 * 256
_WQ_B_ROWS = 32 * 128
_INDEX_HEADS, _INDEX_HEAD_DIM = 32, 128
_BUCKETS = (1, 3, 8, 32, 64, 128)
_FP8_DIFF_THRESHOLD = 1e-3


def _calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    x, y = x.double(), y.double()
    denom = (x * x + y * y).sum()
    if denom == 0:
        return 0.0
    return float(1 - 2 * (x * y).sum() / denom)


def _block_quantize_weight(w: torch.Tensor):
    """bf16 [N, K] -> (e4m3 [N, K], fp32 [N/128, K/128]), the checkpoint layout."""
    n, k = w.shape
    blocks = w.float().reshape(n // 128, 128, k // 128, 128)
    amax = blocks.abs().amax(dim=(1, 3)).clamp(min=1e-4)
    scale = amax / 448.0
    q = (blocks / scale[:, None, :, None]).to(torch.float8_e4m3fn)
    return q.reshape(n, k).contiguous(), scale.contiguous()


@pytest.fixture(scope="module")
def operands():
    from batchgen.models.glm.glm5.q_b_wq_b_merge import build_q_b_wq_b_merged

    gen = torch.Generator(device="cuda").manual_seed(0)
    q_b_bf16 = (torch.randn(_Q_B_ROWS, _Q_LORA_RANK, device="cuda", generator=gen) * 0.02).bfloat16()
    wq_b_bf16 = (torch.randn(_WQ_B_ROWS, _Q_LORA_RANK, device="cuda", generator=gen) * 0.02).bfloat16()
    q_b, q_b_s = _block_quantize_weight(q_b_bf16)
    wq_b, wq_b_s = _block_quantize_weight(wq_b_bf16)
    merged = build_q_b_wq_b_merged(q_b, q_b_s, wq_b, wq_b_s)
    return q_b, q_b_s, wq_b, wq_b_s, merged


@pytest.mark.parametrize("masked", (True, False), ids=("masked", "dense"))
@pytest.mark.parametrize("bucket", _BUCKETS)
def test_merged_deepgemm_is_bitwise_the_two_separate_gemms(operands, bucket, masked):
    from batchgen.attention.mla.fa3_backend import act_quant
    from batchgen.gemm.w8a8_deepgemm import w8a8_deepgemm

    q_b, q_b_s, wq_b, wq_b_s, merged = operands
    gen = torch.Generator(device="cuda").manual_seed(bucket)
    x = torch.randn(bucket, _Q_LORA_RANK, device="cuda", generator=gen).bfloat16()
    valid = max(1, bucket - 1) if masked else bucket
    nv = torch.tensor([valid], dtype=torch.int32, device="cuda") if masked else None
    a, a_s = act_quant(x, num_valid_tokens=nv, scale_tma_aligned=masked)
    kw = dict(num_valid_tokens=nv, expected_m=bucket) if masked else {}

    out_m = torch.zeros(bucket, _Q_B_ROWS + _WQ_B_ROWS, dtype=torch.bfloat16, device="cuda")
    out_q = torch.zeros(bucket, _Q_B_ROWS, dtype=torch.bfloat16, device="cuda")
    out_w = torch.zeros(bucket, _WQ_B_ROWS, dtype=torch.bfloat16, device="cuda")
    w8a8_deepgemm(a, a_s, merged.weight, merged.scale, out=out_m, **kw)
    w8a8_deepgemm(a, a_s, q_b, q_b_s, out=out_q, **kw)
    w8a8_deepgemm(a, a_s, wq_b, wq_b_s, out=out_w, **kw)
    torch.cuda.synchronize()
    assert torch.equal(out_m[:valid, :_Q_B_ROWS], out_q[:valid])
    assert torch.equal(out_m[:valid, _Q_B_ROWS:], out_w[:valid])


@pytest.mark.parametrize("apply_hadamard", (False, True))
@pytest.mark.parametrize("bucket", _BUCKETS)
def test_strided_rope_entry_is_bitwise_the_contiguous_entry(bucket, apply_hadamard):
    from batchgen_kernels.attention.dsa.fused_indexer_score import rope_hadamard_q_out

    gen = torch.Generator(device="cuda").manual_seed(100 + bucket)
    merged = (torch.randn(bucket, _Q_B_ROWS + _WQ_B_ROWS, device="cuda", generator=gen) * 3).bfloat16()
    before = merged.clone()
    q = merged[:, _Q_B_ROWS:].unflatten(-1, (_INDEX_HEADS, _INDEX_HEAD_DIM))
    cos = torch.rand(4096, 64, device="cuda", generator=gen)
    sin = torch.rand(4096, 64, device="cuda", generator=gen)
    positions = torch.randint(0, 4096, (bucket,), device="cuda", generator=gen)
    positions = positions.view(bucket, 1).expand(bucket, _INDEX_HEADS).reshape(-1).contiguous()

    def run(src, row_stride):
        out = torch.empty(bucket, _INDEX_HEADS, _INDEX_HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        fp8 = torch.empty(bucket, 1, _INDEX_HEADS, _INDEX_HEAD_DIM, dtype=torch.float8_e4m3fn, device="cuda")
        rope_hadamard_q_out(
            src, cos, sin, positions, out, out_fp8=fp8,
            apply_hadamard=apply_hadamard, q_row_stride=row_stride,
        )
        return out, fp8

    out_s, fp8_s = run(q, q.stride(0))
    out_c, fp8_c = run(q.contiguous(), None)
    torch.cuda.synchronize()
    assert torch.equal(out_s, out_c)
    assert torch.equal(fp8_s.view(torch.uint8), fp8_c.view(torch.uint8))
    assert torch.equal(merged, before), "the strided entry must only read its source"


@pytest.mark.parametrize("bucket", (1, 8, 64))
def test_merged_wq_b_vs_custom_wgmma_tolerance_report(operands, bucket):
    """Reported, and bounded by the FP8 calc_diff rule -- NOT bitwise."""
    from batchgen.attention.mla.fa3_backend import act_quant
    from batchgen.gemm.w8a8_deepgemm import w8a8_deepgemm
    from batchgen.models.glm.glm5.wrappers import glm5_fp8_dequantization
    from batchgen_kernels.attention.dsa.fused_indexer_kv_proj_cuda import build_module
    from batchgen_kernels.attention.dsa.fused_indexer_score import (
        FP8WqbWeightsCUDA,
        cuda_wq_b_proj,
    )

    _, _, wq_b, wq_b_s, merged = operands
    gen = torch.Generator(device="cuda").manual_seed(200 + bucket)
    x = torch.randn(bucket, _Q_LORA_RANK, device="cuda", generator=gen).bfloat16()

    # The separate version exactly as wrappers.initialize_fused_kernels builds it.
    module = build_module()
    wq_b_bf16 = glm5_fp8_dequantization(wq_b, wq_b_s)
    custom = cuda_wq_b_proj(x, FP8WqbWeightsCUDA(wq_b_bf16, module), module)

    a, a_s = act_quant(x)
    merged_out = w8a8_deepgemm(a, a_s, merged.weight, merged.scale)[:, _Q_B_ROWS:]

    reference = x.float() @ wq_b_bf16.float().t()
    d_custom = _calc_diff(custom, reference)
    d_merged = _calc_diff(merged_out, reference)
    d_cross = _calc_diff(merged_out, custom)
    print(
        f"\n[q_b|wq_b merge] bucket={bucket} calc_diff vs fp32 ref: "
        f"custom={d_custom:.3e} merged={d_merged:.3e}; merged-vs-custom={d_cross:.3e}"
    )
    assert d_custom < _FP8_DIFF_THRESHOLD
    assert d_merged < _FP8_DIFF_THRESHOLD
    assert d_cross < _FP8_DIFF_THRESHOLD
