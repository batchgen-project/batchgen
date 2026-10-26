"""DeepGEMM masked routed-GEMM chain — GPU numerical parity gate.

Skipped without CUDA; this is the node-side half of
``test_glm52_deepgemm_masked_moe.py``. Run it on the machine that serves
GLM-5.2, from the worktree, with the DeepEP decode runtime's own interpreter::

    pytest -s tests/test_glm52_deepgemm_masked_parity_gpu.py

Gate
----
FP8 e4m3 carries 3 mantissa bits, so a per-element tolerance is meaningless;
the accepted measure is DeepGEMM's own ``calc_diff`` (1 - cosine similarity
over f64) against an f32 reference that dequantizes both operands and
multiplies in f32. Threshold ``1e-3``, from
``knowledge/agent-memory/rules/kernel-validation.md``. Nothing here is
bit-identical to the CuTe S1/S3 pair it replaces and nothing is expected to
be: the model-level gate is a WI-5-style accuracy comparison, and this file's
job is to prove the chain computes the right function at every row count the
receive layout can present.

Row counts
----------
The masked kernel's m-block is 64 and the receive stride is a multiple of 64,
so every expert's live extent falls inside one or more 64-row blocks with a
partial tail. The counts swept are therefore the tile edges either side of the
block and of the old 16-row tile (the 2026-07-06 FP8 MoE bench saw calc_diff
spikes at gate/up tile edges around 15/16/17 and 63/64), the empty expert, the
full stride, and ragged mixes of all of them in one launch.
"""
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA device"
)

from batchgen.models.glm.glm5.moe_deepgemm_masked import (  # noqa: E402
    MASKED_BLOCK_M,
    SCALE_BLOCK,
    DeepGemmMaskedExpertWeights,
    expected_m_per_expert,
    masked_act_quant_fn,
    run_deepgemm_masked_moe,
)

FP8 = torch.float8_e4m3fn
FP8_MAX = 448.0
DIFF_THRESHOLD = 1e-3

# GLM-5.2's routed shape at a small local-expert count: the K/N extents are the
# real ones (that is what sets the tiling and the scale-block counts); only the
# expert count and the receive stride are shrunk so the test fits in seconds.
E_LOCAL = 8
HIDDEN = 6144
INTERMEDIATE = 2048
STRIDE = 2 * MASKED_BLOCK_M            # 128 rows per expert

EDGE_COUNTS = (0, 1, 15, 16, 17, 31, 32, 63, 64, 65, 127, 128)


def calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    """DeepGEMM's similarity measure: 1 - cosine similarity, in f64."""
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    if denominator == 0:
        return 0.0
    return float(1 - 2 * (x * y).sum() / denominator)


def quant_rows(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token 128-group FP8 quant of ``[..., K]`` -> (fp8, f32 scales)."""
    *lead, k = x.shape
    groups = k // SCALE_BLOCK
    blocks = x.reshape(*lead, groups, SCALE_BLOCK).float()
    scale = blocks.abs().amax(-1).clamp_min(1e-10) / FP8_MAX
    q = (blocks / scale.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX).to(FP8)
    return q.reshape(*lead, k), scale


def quant_weight(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-128x128-block FP8 quant of ``[E, N, K]`` -> (fp8, f32 scales)."""
    e, n, k = w.shape
    nb, kb = n // SCALE_BLOCK, k // SCALE_BLOCK
    tiles = w.reshape(e, nb, SCALE_BLOCK, kb, SCALE_BLOCK).float()
    scale = tiles.abs().amax((2, 4)).clamp_min(1e-10) / FP8_MAX
    q = (tiles / scale[:, :, None, :, None]).clamp(-FP8_MAX, FP8_MAX).to(FP8)
    return q.reshape(e, n, k).contiguous(), scale.contiguous()


def dequant_rows(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    *lead, k = q.shape
    groups = k // SCALE_BLOCK
    return (q.float().reshape(*lead, groups, SCALE_BLOCK)
            * scale.unsqueeze(-1)).reshape(*lead, k)


def dequant_weight(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    e, n, k = q.shape
    nb, kb = n // SCALE_BLOCK, k // SCALE_BLOCK
    tiles = q.float().reshape(e, nb, SCALE_BLOCK, kb, SCALE_BLOCK)
    return (tiles * scale[:, :, None, :, None]).reshape(e, n, k)


def mn_major(groups: int, rows: int, blocks: int, device) -> torch.Tensor:
    """The masked kernel's A-scale buffer: [G, B, R] contiguous as [G, R, B]."""
    return torch.zeros(
        groups, blocks, rows, dtype=torch.float32, device=device
    ).transpose(1, 2)


def masked_reference(
    a_q, a_s, b_q, b_s, masked_m, out_dtype=torch.float32
) -> torch.Tensor:
    """Per expert, the live rows only, dequantized and multiplied in f32."""
    e, rows, _ = a_q.shape
    n = b_q.shape[1]
    ref = torch.zeros(e, rows, n, dtype=out_dtype, device=a_q.device)
    b = dequant_weight(b_q, b_s)
    for i in range(e):
        live = int(masked_m[i])
        if live == 0:
            continue
        a = dequant_rows(a_q[i, :live], a_s[i, :live])
        ref[i, :live] = (a @ b[i].t()).to(out_dtype)
    return ref


def act_reference(gate, up, masked_m):
    """The activation's own numerics: SiLU in f32, cast back to the input
    dtype before the multiply, then a per-128-group e4m3 quant. Padded rows
    are left untouched, as the kernel leaves them."""
    e, rows, inter = gate.shape
    groups = inter // SCALE_BLOCK
    q = torch.zeros(e, rows, inter, dtype=FP8, device=gate.device)
    scale = torch.zeros(e, rows, groups, dtype=torch.float32,
                        device=gate.device)
    for i in range(e):
        live = int(masked_m[i])
        if live == 0:
            continue
        g = gate[i, :live].float()
        g = (g / (1 + torch.exp(-g))).to(gate.dtype)
        prod = (up[i, :live] * g).reshape(live, groups, SCALE_BLOCK)
        amax = prod.abs().amax(-1).float().clamp_min(1e-10)
        s = amax / FP8_MAX
        q[i, :live] = (prod.float() / s.unsqueeze(-1)).clamp(
            -FP8_MAX, FP8_MAX).to(FP8).reshape(live, inter)
        scale[i, :live] = s
    return q, scale


@pytest.fixture(scope="module")
def fixture_layer():
    """One routed layer's weights plus the buffers the segment owns."""
    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(20261021)
    e, h, i, s = E_LOCAL, HIDDEN, INTERMEDIATE, STRIDE

    def rand(*shape):
        return torch.randn(*shape, generator=gen, device=device,
                           dtype=torch.float32) * 0.1

    # One fused [E, 2I, H] stack: gate rows [0, I), up rows [I, 2I). Quantize
    # the two projections as before and then lay them out fused, so the values
    # are identical to the per-projection reference and only the layout moves.
    gate_ref, up_ref = rand(e, i, h), rand(e, i, h)
    gate_q, gate_s = quant_weight(gate_ref)
    up_q, up_s = quant_weight(up_ref)
    down_q, down_s = quant_weight(rand(e, h, i))
    gate_up_q = torch.empty(e, 2 * i, h, dtype=gate_q.dtype, device=device)
    gate_up_q[:, :i, :] = gate_q
    gate_up_q[:, i:, :] = up_q
    gate_up_s = torch.empty(
        e, (2 * i) // SCALE_BLOCK, h // SCALE_BLOCK,
        dtype=gate_s.dtype, device=device)
    gate_up_s[:, : i // SCALE_BLOCK, :] = gate_s
    gate_up_s[:, i // SCALE_BLOCK:, :] = up_s
    weights = DeepGemmMaskedExpertWeights(
        gate_up_w=gate_up_q, gate_up_scale=gate_up_s,
        down_w=down_q, down_scale=down_s,
    )
    weights.validate(num_experts=e, hidden_size=h, intermediate_size=i)

    x_bf16 = (rand(e, s, h)).to(torch.bfloat16)
    x_q, x_s = quant_rows(x_bf16)
    # the dispatch's layout: values row-major, scales MN-major
    x_fp8 = x_q.contiguous()
    x_scales = mn_major(e, s, h // SCALE_BLOCK, device)
    x_scales.copy_(x_s)

    buffers = dict(
        x_fp8=x_fp8,
        x_scales=x_scales,
        gate_up_out=torch.zeros(
            e, s, 2 * i, dtype=torch.bfloat16, device=device),
        act_fp8=torch.zeros(e, s, i, dtype=FP8, device=device),
        act_scales=mn_major(e, s, i // SCALE_BLOCK, device),
        out=torch.zeros(e, s, h, dtype=torch.bfloat16, device=device),
    )
    return weights, buffers, x_s


def _run(weights, buffers, masked_m):
    # The scheduling hint the serving path would pass at this geometry: the
    # mean rows per expert over the receive space, rounded up. It steers only
    # the masked kernel's work distribution, never the result.
    expected_m = expected_m_per_expert(
        local_rows=STRIDE, world_size=1, topk=1,
        num_experts_global=E_LOCAL)
    run_deepgemm_masked_moe(
        weights=weights, masked_m=masked_m, expected_m=expected_m, **buffers)
    torch.cuda.synchronize()


def _counts(vector, device):
    return torch.tensor(vector, dtype=torch.int32, device=device)


def _halves(weights, buffers):
    """The gate/up views of the fused stack and of R1's fused output.

    Strided views, never copies: the whole point of the fused layout is that
    the halves are addressable in place.
    """
    i = buffers["gate_up_out"].shape[2] // 2
    nb = i // SCALE_BLOCK
    return dict(
        gate_w=weights.gate_up_w[:, :i, :],
        gate_s=weights.gate_up_scale[:, :nb, :],
        up_w=weights.gate_up_w[:, i:, :],
        up_s=weights.gate_up_scale[:, nb:, :],
        gate_out=buffers["gate_up_out"][:, :, :i],
        up_out=buffers["gate_up_out"][:, :, i:],
    )


@pytest.mark.parametrize("rows", EDGE_COUNTS)
def test_r1_matches_the_dequantized_reference_at_every_tile_edge(
        fixture_layer, rows):
    """R1 gate and R1 up, at a uniform row count per expert."""
    weights, buffers, x_s = fixture_layer
    device = buffers["out"].device
    masked_m = _counts([rows] * E_LOCAL, device)
    _run(weights, buffers, masked_m)

    hv = _halves(weights, buffers)
    for name, w, s, got in (
        ("gate", hv["gate_w"], hv["gate_s"], hv["gate_out"]),
        ("up", hv["up_w"], hv["up_s"], hv["up_out"]),
    ):
        ref = masked_reference(buffers["x_fp8"], x_s, w, s, masked_m)
        live = _live_mask(masked_m, got.shape[1], device)
        diff = calc_diff(got.float()[live], ref[live]) if rows else 0.0
        print(f"R1 {name:4s} rows={rows:4d} calc_diff={diff:.3e}")
        assert diff < DIFF_THRESHOLD, (name, rows, diff)


@pytest.mark.parametrize("rows", EDGE_COUNTS)
def test_the_masked_activation_matches_its_reference(fixture_layer, rows):
    """The activation's quantized output and its group scales, live rows only.

    Checked on the R1 output the chain actually produced, so a disagreement
    here is the activation's own and not inherited from the GEMM.
    """
    weights, buffers, _x_s = fixture_layer
    device = buffers["out"].device
    masked_m = _counts([rows] * E_LOCAL, device)
    _run(weights, buffers, masked_m)

    hv = _halves(weights, buffers)
    ref_q, ref_s = act_reference(hv["gate_out"], hv["up_out"], masked_m)
    live = _live_mask(masked_m, buffers["act_fp8"].shape[1], device)
    got_q = buffers["act_fp8"].float()[live]
    exp_q = ref_q.float()[live]
    q_diff = calc_diff(got_q, exp_q) if rows else 0.0
    exact = float((got_q == exp_q).float().mean()) if rows else 1.0
    got_s = buffers["act_scales"][live]
    s_diff = calc_diff(got_s, ref_s[live]) if rows else 0.0
    print(f"ACT rows={rows:4d} q_diff={q_diff:.3e} exact={exact:.6f} "
          f"scale_diff={s_diff:.3e}")
    assert q_diff < DIFF_THRESHOLD, (rows, q_diff)
    assert s_diff < DIFF_THRESHOLD, (rows, s_diff)


@pytest.mark.parametrize("rows", EDGE_COUNTS)
def test_r2_matches_the_dequantized_reference_at_every_tile_edge(
        fixture_layer, rows):
    """R2 against the activation's own output, i.e. the whole chain."""
    weights, buffers, _x_s = fixture_layer
    device = buffers["out"].device
    masked_m = _counts([rows] * E_LOCAL, device)
    _run(weights, buffers, masked_m)

    ref = masked_reference(
        buffers["act_fp8"], buffers["act_scales"],
        weights.down_w, weights.down_scale, masked_m)
    live = _live_mask(masked_m, buffers["out"].shape[1], device)
    diff = calc_diff(buffers["out"].float()[live], ref[live]) if rows else 0.0
    print(f"R2        rows={rows:4d} calc_diff={diff:.3e}")
    assert diff < DIFF_THRESHOLD, (rows, diff)


RAGGED = (
    [0, 1, 15, 16, 17, 63, 64, 65],
    [64, 0, 128, 1, 127, 16, 65, 63],
    [1] * 7 + [128],
    [128] * 7 + [1],
)


@pytest.mark.parametrize("vector", RAGGED)
def test_the_chain_matches_the_reference_on_a_ragged_mix(
        fixture_layer, vector):
    """One launch where every expert sits at a different tile edge — the shape
    the recorded decode routing actually presents."""
    weights, buffers, x_s = fixture_layer
    device = buffers["out"].device
    masked_m = _counts(vector, device)
    _run(weights, buffers, masked_m)

    live = _live_mask(masked_m, buffers["out"].shape[1], device)
    hv = _halves(weights, buffers)
    gate_ref = masked_reference(
        buffers["x_fp8"], x_s, hv["gate_w"], hv["gate_s"], masked_m)
    up_ref = masked_reference(
        buffers["x_fp8"], x_s, hv["up_w"], hv["up_s"], masked_m)
    down_ref = masked_reference(
        buffers["act_fp8"], buffers["act_scales"],
        weights.down_w, weights.down_scale, masked_m)
    diffs = {
        "gate": calc_diff(hv["gate_out"].float()[live], gate_ref[live]),
        "up": calc_diff(hv["up_out"].float()[live], up_ref[live]),
        "down": calc_diff(buffers["out"].float()[live], down_ref[live]),
    }
    print(f"RAGGED {vector} " +
          " ".join(f"{k}={v:.3e}" for k, v in diffs.items()))
    for name, diff in diffs.items():
        assert diff < DIFF_THRESHOLD, (name, vector, diff)


def test_the_live_output_is_independent_of_the_padded_rows(fixture_layer):
    """M-independence, on device: poison every row above the live count in
    every buffer the chain reads, and the live rows of every output must come
    back BIT-identical. This is what makes the captured graph's result depend
    on ``recv_count`` alone."""
    weights, buffers, _x_s = fixture_layer
    device = buffers["out"].device
    vector = [0, 1, 15, 16, 17, 63, 64, 65]
    masked_m = _counts(vector, device)
    _run(weights, buffers, masked_m)
    live = _live_mask(masked_m, buffers["out"].shape[1], device)
    clean = {k: buffers[k].clone() for k in
             ("gate_up_out", "act_fp8", "act_scales", "out")}
    # the module-scoped A operand is restored below, so the order of the tests
    # in this file carries no state
    x_backup = (buffers["x_fp8"].clone(), buffers["x_scales"].clone())

    nan = float("nan")
    for i, rows in enumerate(vector):
        buffers["x_fp8"][i, rows:] = torch.full_like(
            buffers["x_fp8"][i, rows:], FP8_MAX)
        buffers["x_scales"][i, rows:] = nan
        buffers["gate_up_out"][i, rows:] = nan
        buffers["act_scales"][i, rows:] = nan
        buffers["out"][i, rows:] = nan
    _run(weights, buffers, masked_m)
    buffers["x_fp8"].copy_(x_backup[0])
    buffers["x_scales"].copy_(x_backup[1])

    for key, before in clean.items():
        got = buffers[key]
        if got.dtype == FP8:
            got, before = got.view(torch.uint8), before.view(torch.uint8)
        assert torch.equal(got[live], before[live]), key


def test_the_activation_leaves_the_padded_rows_untouched(fixture_layer):
    """The activation writes per live row, so a padded row keeps whatever the
    previous step left -- which is what the GEMM's row-wise independence then
    makes unreachable."""
    weights, buffers, _x_s = fixture_layer
    device = buffers["out"].device
    vector = [0, 1, 16, 17, 63, 64, 65, 128]
    masked_m = _counts(vector, device)
    act = masked_act_quant_fn()

    sentinel_q = torch.full_like(buffers["act_fp8"].view(torch.uint8), 0x5A)
    buffers["act_fp8"].view(torch.uint8).copy_(sentinel_q)
    buffers["act_scales"].fill_(-7.5)
    buffers["gate_up_out"].normal_()
    hv = _halves(weights, buffers)
    act(hv["gate_out"], hv["up_out"], buffers["act_fp8"],
        buffers["act_scales"], masked_m, group_size=SCALE_BLOCK)
    torch.cuda.synchronize()

    for i, rows in enumerate(vector):
        tail_q = buffers["act_fp8"][i, rows:].view(torch.uint8)
        assert torch.all(tail_q == 0x5A), i
        assert torch.all(buffers["act_scales"][i, rows:] == -7.5), i


def _live_mask(masked_m, rows: int, device) -> torch.Tensor:
    """[E, rows] bool, True on the rows below each expert's live count."""
    index = torch.arange(rows, device=device).unsqueeze(0)
    return index < masked_m.to(torch.int64).unsqueeze(1)
