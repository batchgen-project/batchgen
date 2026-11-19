"""GPU gate for the fused SiLU*up + FP8 quant arm of the grouped prefill.

The fused ragged kernel replaces the (silu, mul_, act_quant_ragged) chain
between the two S1 GEMMs and S3. Three things are checked on a real device:

* numerics: the dequantized activations of both arms agree with each other
  and with the fp32 reference under the FP8 similarity bound, and the two
  arms' group scales agree;
* layout: alignment holes and the rows of empty experts are neither read
  nor written, and the row-major scales are the transpose of the chain's;
* consumption: the contiguous S3 GEMM gives the same result from the
  row-major scales as from the same scales routed through the column-major
  shim, so the layout switch alone cannot change an output.

Run on a GPU node with Triton and deep_gemm:
    PYTHONPATH=<pyroot> python -m pytest tests/gpu/test_glm5_s1_act_quant_fused_parity.py -v
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA device"
)

QUANT_BLOCK = 128
FP8_DIFF_THRESHOLD = 1e-3  # kernel-validation rule: DeepGEMM calc_diff
HOLE = 0x5A


def calc_diff(x, y):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    return float(1 - 2 * (x * y).sum() / denominator)


def _layout(counts, align, device):
    counts = torch.tensor(counts, dtype=torch.int32, device=device)
    padded = ((counts + align - 1) // align) * align
    cu = torch.zeros(counts.numel() + 1, dtype=torch.int32, device=device)
    cu[1:] = torch.cumsum(padded, 0)
    return counts, cu, int(cu[-1])


def _live_mask(counts, cu, rows):
    mask = torch.zeros(rows, dtype=torch.bool, device=counts.device)
    for start, count in zip(cu[:-1].tolist(), counts.tolist()):
        mask[start:start + count] = True
    return mask


def _inputs(rows, inter, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    gate = torch.randn(rows, inter, generator=gen, device="cuda") * 2.0
    up = torch.randn(rows, inter, generator=gen, device="cuda") * 0.5
    # a few wide groups so the per-group scales span orders of magnitude
    gate[:, :QUANT_BLOCK] *= 8.0
    up[:, -QUANT_BLOCK:] *= 1e-3
    return gate.to(torch.bfloat16), up.to(torch.bfloat16)


def _run_chain(gate, up, counts, cu, align):
    from batchgen.models.glm.glm5.moe_ragged import (
        act_quant_ragged,
        make_quant_buffers,
    )

    rows, inter = gate.shape
    act = gate.clone()
    torch.nn.functional.silu(act, inplace=True)
    act.mul_(up)
    out, scale = make_quant_buffers(rows, inter, gate.device)
    out.fill_(HOLE)
    act_quant_ragged(act, counts, cu, out, scale, row_align=align)
    return out, scale


def _run_fused(gate, up, counts, cu):
    from batchgen.models.glm.glm5.moe_ragged import silu_mul_quant_ragged

    rows, inter = gate.shape
    out = torch.full((rows, inter), HOLE, dtype=torch.uint8, device=gate.device)
    scale_rows = torch.zeros(
        rows, inter // QUANT_BLOCK, dtype=torch.float32, device=gate.device
    )
    silu_mul_quant_ragged(
        gate, up, out.view(torch.float8_e4m3fn), scale_rows, counts, cu
    )
    return out, scale_rows


def _dequant(out_u8, scale_rows):
    values = out_u8.view(torch.float8_e4m3fn).float()
    return values * scale_rows.repeat_interleave(QUANT_BLOCK, dim=1)


CASES = {
    "uniform": [256] * 8,
    "ragged": [1, 0, 127, 128, 129, 3, 0, 700, 61, 512],
    "single_row_experts": [1] * 16,
    "sparse": [0, 0, 5, 0, 0, 0, 2048, 0],
}


@pytest.mark.parametrize("align", [64, 128])
@pytest.mark.parametrize("name", sorted(CASES))
def test_fused_arm_matches_the_chain_and_the_reference(name, align):
    pytest.importorskip("triton")
    counts, cu, rows = _layout(CASES[name], align, "cuda")
    inter = 2048
    gate, up = _inputs(rows, inter, seed=len(name) * 31 + align)
    live = _live_mask(counts, cu, rows)

    chain_out, chain_scale = _run_chain(gate, up, counts, cu, align)
    fused_out, fused_scale = _run_fused(gate, up, counts, cu)
    torch.cuda.synchronize()

    # Holes: untouched by both arms (sentinel bytes survive, scales stay 0).
    assert bool((fused_out[~live] == HOLE).all())
    assert bool((fused_scale[~live] == 0).all())

    chain_scale_rows = chain_scale.t().contiguous()
    reference = torch.nn.functional.silu(gate.float()) * up.float()
    fused_deq = _dequant(fused_out, fused_scale)[live]
    chain_deq = _dequant(chain_out, chain_scale_rows)[live]

    assert calc_diff(fused_deq, reference[live]) < FP8_DIFF_THRESHOLD
    assert calc_diff(chain_deq, reference[live]) < FP8_DIFF_THRESHOLD
    assert calc_diff(fused_deq, chain_deq) < FP8_DIFF_THRESHOLD

    # The group scale is absmax/448 of the same bf16 product on both arms; a
    # one-ulp difference of the activation is the only allowed source of drift.
    torch.testing.assert_close(
        fused_scale[live], chain_scale_rows[live], rtol=1.6e-2, atol=1e-12
    )
    # The fused arm is not allowed to be a worse quantizer than the chain.
    fused_err = (fused_deq - reference[live]).abs().mean()
    chain_err = (chain_deq - reference[live]).abs().mean()
    assert float(fused_err) <= float(chain_err) * 1.05


def test_s3_gemm_output_is_independent_of_the_scale_layout():
    pytest.importorskip("triton")
    pytest.importorskip("deep_gemm")
    from batchgen.moe.grouped_fp8_deepgemm_moe import (
        build_grouped_layout,
        deepgemm_m_alignment,
        grouped_fp8_deepgemm_gemm,
    )

    align = deepgemm_m_alignment()
    counts, cu, rows = _layout([300, 0, 128, 77, 1024, 5], align, "cuda")
    inter, hidden, groups = 2048, 1024, counts.numel()
    gate, up = _inputs(rows, inter, seed=7)
    fused_out, scale_rows = _run_fused(gate, up, counts, cu)

    gen = torch.Generator(device="cuda").manual_seed(11)
    weight = torch.randn(groups, hidden, inter, generator=gen, device="cuda") * 0.05
    b = weight.to(torch.float8_e4m3fn).contiguous()
    sfb = torch.ones(
        groups, hidden // QUANT_BLOCK, inter // QUANT_BLOCK,
        dtype=torch.float32, device="cuda",
    )
    layout = build_grouped_layout(counts, cu, rows, alignment=align)

    def run(scale, rows_major):
        out = torch.zeros(rows, hidden, dtype=torch.bfloat16, device="cuda")
        grouped_fp8_deepgemm_gemm(
            fused_out.view(torch.float8_e4m3fn), scale, b, sfb,
            output=out, grouped_layout=layout, x_scale_rows_major=rows_major,
        )
        return out

    direct = run(scale_rows, True)
    via_shim = run(scale_rows.t().contiguous(), False)
    torch.cuda.synchronize()
    live = _live_mask(counts, cu, rows)
    assert bool(torch.isfinite(direct[live].float()).all())
    assert torch.equal(direct[live], via_shim[live])
