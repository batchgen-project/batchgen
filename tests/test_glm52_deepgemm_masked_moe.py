"""DeepGEMM masked m-grouped routed GEMMs — CPU contract tests.

The numerics are a node gate (``test_glm52_deepgemm_masked_parity_gpu.py``).
What is checkable here is the call contract the node run depends on, which is
where a layout mistake would otherwise surface as a wrong-but-plausible logit:

* the ``nt`` weight layout — our routed stacks are ALREADY ``[E, N, K]`` FP8
  with ``[E, N/128, K/128]`` f32 scales for their own projection at the GLM-5.2
  shape, so the swap needs no load-time conversion and no second resident
  copy; a shape where the CuTe scale padding would bite is refused, loudly,
  instead of being handed over mis-strided;
* the A-scale layout — the masked kernel loads MN-major, 4-float-aligned
  scales, which is exactly what the DeepEP dispatch returns and what the masked
  activation writes, so the path carries no scale-preparation launch; anything
  else fails with the strides it wanted named;
* ``masked_m`` threading — every launch of the layer gets the dispatch's own
  device ``recv_count`` tensor, never a copy and never a host value;
* ``expected_m`` — the reference row-count hint, a host int derived from the
  captured bucket;
* graph stability — the runner allocates nothing, reshapes nothing and reads
  nothing back to the host, so the only per-step input of the captured chain is
  the contents of two device tensors;
* the activation launcher — its grid is a function of the captured shapes only
  (the live row count is consumed on the device), and it validates its operands
  rather than trusting them.
"""
import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from batchgen.models.glm.glm5.moe_deepgemm_masked import (
    MASKED_BLOCK_M,
    SCALE_BLOCK,
    DeepGemmMaskedExpertWeights,
    expected_m_per_expert,
    is_mn_major_tma_aligned,
    masked_act_quant_fn,
    masked_gemm_fn,
    require_mn_major_tma_aligned,
    run_deepgemm_masked_moe,
)

FP8 = torch.float8_e4m3fn

# Small stand-in geometry (one scale block per axis is enough to pin layout).
E, H, I, S = 4, 256, 128, 64


def _mn_major(groups, rows, blocks, *, device="cpu"):
    """The masked kernel's A-scale form: [G, B, R] contiguous, seen as
    [G, R, B]. This is DeepEP's own dispatch layout."""
    return torch.zeros(
        groups, blocks, rows, dtype=torch.float32, device=device
    ).transpose(1, 2)


def _weights(e=E, h=H, i=I, device="cpu"):
    return DeepGemmMaskedExpertWeights(
        gate_w=torch.zeros(e, i, h, dtype=torch.uint8, device=device).view(FP8),
        gate_scale=torch.ones(
            e, i // SCALE_BLOCK, h // SCALE_BLOCK, device=device),
        up_w=torch.zeros(e, i, h, dtype=torch.uint8, device=device).view(FP8),
        up_scale=torch.ones(
            e, i // SCALE_BLOCK, h // SCALE_BLOCK, device=device),
        down_w=torch.zeros(e, h, i, dtype=torch.uint8, device=device).view(FP8),
        down_scale=torch.ones(
            e, h // SCALE_BLOCK, i // SCALE_BLOCK, device=device),
    )


# --------------------------------------------------------------------------
# expected_m
# --------------------------------------------------------------------------
def test_expected_m_is_the_reference_formula():
    """(local rows x ranks x topk + E_global) // E_global — the mean routed
    rows per global expert, rounded up. 64 rows/rank over 8 ranks at top-8 into
    256 experts is 17, which is what the shape probe ran the node timings at."""
    assert expected_m_per_expert(
        local_rows=64, world_size=8, topk=8, num_experts_global=256) == 17
    # it tracks the bucket, and it is a plain host int
    value = expected_m_per_expert(
        local_rows=128, world_size=8, topk=8, num_experts_global=256)
    assert value == 33 and type(value) is int


@pytest.mark.parametrize("kwargs", [
    dict(local_rows=0, world_size=8, topk=8, num_experts_global=256),
    dict(local_rows=64, world_size=0, topk=8, num_experts_global=256),
    dict(local_rows=64, world_size=8, topk=0, num_experts_global=256),
    dict(local_rows=64, world_size=8, topk=8, num_experts_global=0),
])
def test_expected_m_refuses_a_degenerate_shape(kwargs):
    with pytest.raises(ValueError, match="positive"):
        expected_m_per_expert(**kwargs)


# --------------------------------------------------------------------------
# A-scale layout
# --------------------------------------------------------------------------
def test_the_dispatch_scale_layout_needs_no_preparation():
    """DeepEP allocates [E, K/128, S] contiguous and returns its (1, 2)
    transpose. That IS the masked kernel's MN-major form, so the segment can
    hand it over directly — this is the claim the dropped repack rests on."""
    scales = _mn_major(E, S, H // SCALE_BLOCK)
    assert scales.shape == (E, S, H // SCALE_BLOCK)
    assert scales.stride() == ((H // SCALE_BLOCK) * S, 1, S)
    assert is_mn_major_tma_aligned(scales)
    require_mn_major_tma_aligned(scales, "dispatch")


def test_a_k_major_scale_tensor_is_refused_with_the_strides_it_wanted():
    """The shape alone is not the contract: a contiguous [E, S, K/128] has the
    right shape and the wrong strides, and would be silently transposed (and
    reallocated, inside a capture) by DeepGEMM's own alignment helper."""
    k_major = torch.zeros(E, S, H // SCALE_BLOCK)
    assert not is_mn_major_tma_aligned(k_major)
    with pytest.raises(RuntimeError, match="MN-major"):
        require_mn_major_tma_aligned(k_major, "dispatch")
    # and the error names the strides it wanted, so the fix is unambiguous
    with pytest.raises(RuntimeError, match=r"strides \(128, 1, 64\)"):
        require_mn_major_tma_aligned(torch.zeros(E, 64, 2), "dispatch")


def test_an_unaligned_row_extent_is_refused():
    """The masked kernel's A-scale TMA load needs the MN extent padded to 4
    floats; the receive stride is a multiple of the 64-row block, so a real
    DeepEP layout always is, and a hand-built one that is not must fail."""
    scales = _mn_major(E, 6, H // SCALE_BLOCK)
    assert not is_mn_major_tma_aligned(scales)
    with pytest.raises(RuntimeError, match="padded to 4 floats"):
        require_mn_major_tma_aligned(scales, "dispatch")
    assert MASKED_BLOCK_M % 4 == 0


@pytest.mark.parametrize("bad,match", [
    (torch.zeros(E, S), "logical"),
    (torch.zeros(E, 1, S, dtype=torch.float16), "f32"),
])
def test_scale_dim_and_dtype_are_refused(bad, match):
    assert not is_mn_major_tma_aligned(bad)
    with pytest.raises(RuntimeError, match=match):
        require_mn_major_tma_aligned(bad, "dispatch")


# --------------------------------------------------------------------------
# Weight layout — the load-time question
# --------------------------------------------------------------------------
def test_the_glm52_routed_stacks_are_already_the_masked_nt_layout():
    """GLM-5.2's loader stacks produce exactly the masked ``nt`` operands.

    ``_init_fp8_blockwise_weights`` builds gate/up as ``[E, N, K]`` and down as
    ``[E, K, N]`` FP8, with scales ``[E, N/128, K/128_pad4]`` and
    ``[E, K/128, N/128_pad4]`` where ``pad4`` rounds the trailing block axis up
    to 4. At H=6144 / I=2048 both block counts (48 and 16) are already
    multiples of 4, so ``pad4`` is the identity and the stacks need no
    conversion and no second resident copy. Meta tensors: this is a statement
    about shape, dtype and contiguity, not about values.
    """
    e, h, i = 32, 6144, 2048
    k_blocks, n_blocks = h // SCALE_BLOCK, i // SCALE_BLOCK
    pad4 = lambda v: (v + 3) // 4 * 4            # noqa: E731 - loader's own
    assert pad4(k_blocks) == k_blocks == 48
    assert pad4(n_blocks) == n_blocks == 16

    def meta(*shape, dtype=torch.float32):
        return torch.empty(*shape, dtype=dtype, device="meta")

    weights = DeepGemmMaskedExpertWeights(
        gate_w=meta(e, i, h, dtype=torch.uint8).view(FP8),
        gate_scale=meta(e, n_blocks, pad4(k_blocks)),
        up_w=meta(e, i, h, dtype=torch.uint8).view(FP8),
        up_scale=meta(e, n_blocks, pad4(k_blocks)),
        down_w=meta(e, h, i, dtype=torch.uint8).view(FP8),
        down_scale=meta(e, k_blocks, pad4(n_blocks)),
    )
    weights.validate(num_experts=e, hidden_size=h, intermediate_size=i)


def test_a_cute_padded_scale_stack_is_refused_not_mis_strided():
    """A shape where the loader's pad-to-4 is NOT the identity would hand the
    masked GEMM a wider-than-declared scale block. It must fail at binding with
    the padding named, never be passed through."""
    e, h, i = 2, 384, 128                        # 3 k-blocks -> padded to 4
    k_blocks = h // SCALE_BLOCK
    assert (k_blocks + 3) // 4 * 4 != k_blocks
    weights = DeepGemmMaskedExpertWeights(
        gate_w=torch.zeros(e, i, h, dtype=torch.uint8).view(FP8),
        gate_scale=torch.ones(e, i // SCALE_BLOCK, 4),
        up_w=torch.zeros(e, i, h, dtype=torch.uint8).view(FP8),
        up_scale=torch.ones(e, i // SCALE_BLOCK, 4),
        down_w=torch.zeros(e, h, i, dtype=torch.uint8).view(FP8),
        down_scale=torch.ones(e, k_blocks, i // SCALE_BLOCK),
    )
    with pytest.raises(RuntimeError, match="pad the trailing block axis"):
        weights.validate(num_experts=e, hidden_size=h, intermediate_size=i)


def test_weight_validation_fences_dtype_shape_and_contiguity():
    base = _weights()
    good = dict(num_experts=E, hidden_size=H, intermediate_size=I)
    base.validate(**good)

    bf16 = DeepGemmMaskedExpertWeights(
        **{**base.__dict__, "gate_w": torch.zeros(E, I, H, dtype=torch.bfloat16)})
    with pytest.raises(RuntimeError, match="float8_e4m3fn weights"):
        bf16.validate(**good)

    swapped = DeepGemmMaskedExpertWeights(
        **{**base.__dict__, "down_w": base.gate_w})
    with pytest.raises(RuntimeError, match=r"down_w.*\[E, N, K\]"):
        swapped.validate(**good)

    strided = DeepGemmMaskedExpertWeights(
        **{**base.__dict__,
           "up_w": torch.zeros(E, 2 * I, H, dtype=torch.uint8)
                        .view(FP8)[:, :I, :]})
    with pytest.raises(RuntimeError, match="up_w: the masked GEMM takes a contiguous"):
        strided.validate(**good)

    with pytest.raises(RuntimeError, match="block-scaled"):
        base.validate(num_experts=E, hidden_size=H + 1, intermediate_size=I)


# --------------------------------------------------------------------------
# Kernel resolution — the deep_gemm seam
# --------------------------------------------------------------------------
@pytest.fixture
def fake_deep_gemm(monkeypatch):
    """A stand-in ``deep_gemm`` that records every masked call."""
    calls = []

    def fp8_m_grouped_gemm_nt_masked(a, b, d, masked_m, expected_m):
        calls.append(SimpleNamespace(
            a=a, b=b, d=d, masked_m=masked_m, expected_m=expected_m))
        return None

    module = ModuleType("deep_gemm")
    module.fp8_m_grouped_gemm_nt_masked = fp8_m_grouped_gemm_nt_masked
    module.get_mn_major_tma_aligned_tensor = lambda t: t
    monkeypatch.setitem(sys.modules, "deep_gemm", module)
    return SimpleNamespace(module=module, calls=calls)


def test_the_masked_gemm_is_resolved_from_deep_gemm(fake_deep_gemm):
    assert masked_gemm_fn() is fake_deep_gemm.module.fp8_m_grouped_gemm_nt_masked


def test_a_deep_gemm_without_the_masked_entry_point_fails_loud(monkeypatch):
    stub = ModuleType("deep_gemm")
    stub.fp8_m_grouped_gemm_nt_contiguous = lambda *a: None
    monkeypatch.setitem(sys.modules, "deep_gemm", stub)
    with pytest.raises(RuntimeError, match="no fp8_m_grouped_gemm_nt_masked"):
        masked_gemm_fn()


def test_a_missing_deep_gemm_fails_loud(monkeypatch):
    monkeypatch.setitem(sys.modules, "deep_gemm", None)
    with pytest.raises(RuntimeError, match="deep_gemm is not importable"):
        masked_gemm_fn()


# --------------------------------------------------------------------------
# The runner
# --------------------------------------------------------------------------
def _runner_buffers(device="cpu"):
    return dict(
        x_fp8=torch.zeros(E, S, H, dtype=torch.uint8, device=device).view(FP8),
        x_scales=_mn_major(E, S, H // SCALE_BLOCK, device=device),
        gate_out=torch.zeros(E, S, I, dtype=torch.bfloat16, device=device),
        up_out=torch.zeros(E, S, I, dtype=torch.bfloat16, device=device),
        act_fp8=torch.zeros(E, S, I, dtype=torch.uint8, device=device).view(FP8),
        act_scales=_mn_major(E, S, I // SCALE_BLOCK, device=device),
        out=torch.zeros(E, S, H, dtype=torch.bfloat16, device=device),
    )


def _fake_act(monkeypatch):
    seen = []

    def act(gate, up, out, out_scale, masked_m, *, group_size):
        seen.append(SimpleNamespace(gate=gate, up=up, out=out,
                                    out_scale=out_scale, masked_m=masked_m,
                                    group_size=group_size))

    monkeypatch.setattr(
        "batchgen.models.glm.glm5.moe_deepgemm_masked.masked_act_quant_fn",
        lambda: act)
    return seen


def test_the_runner_threads_the_full_masked_call_contract(
        fake_deep_gemm, monkeypatch):
    acts = _fake_act(monkeypatch)
    bufs = _runner_buffers()
    weights = _weights()
    masked_m = torch.tensor([1, 16, 64, 0], dtype=torch.int32)

    returned = run_deepgemm_masked_moe(
        weights=weights, masked_m=masked_m, expected_m=17, **bufs)

    assert returned is bufs["out"]
    gate, up, down = fake_deep_gemm.calls
    assert len(acts) == 1
    act = acts[0]

    # R1: both projections read the SAME A operand, as delivered
    for call, (w, s) in ((gate, (weights.gate_w, weights.gate_scale)),
                         (up, (weights.up_w, weights.up_scale))):
        assert call.a[0] is bufs["x_fp8"] and call.a[1] is bufs["x_scales"]
        assert call.b == (w, s)
    assert gate.d is bufs["gate_out"] and up.d is bufs["up_out"]

    # act: reads both halves of R1, writes R2's A operand pair
    assert act.gate is bufs["gate_out"] and act.up is bufs["up_out"]
    assert act.out is bufs["act_fp8"] and act.out_scale is bufs["act_scales"]
    assert act.group_size == SCALE_BLOCK

    # R2: the activation's outputs, the down stack, the caller's output buffer
    assert down.a[0] is bufs["act_fp8"] and down.a[1] is bufs["act_scales"]
    assert down.b == (weights.down_w, weights.down_scale)
    assert down.d is bufs["out"]

    # masked_m is the SAME device tensor everywhere, never a copy or a host int
    for call in (gate, up, down):
        assert call.masked_m is masked_m
        assert call.expected_m == 17 and type(call.expected_m) is int
    assert act.masked_m is masked_m


def test_the_runner_allocates_nothing(fake_deep_gemm, monkeypatch):
    """Graph stability: a tensor minted per call would bake a per-call address
    into the captured graph (or, worse, be minted DURING the capture). Every
    buffer of the chain is the caller's."""
    _fake_act(monkeypatch)
    bufs = _runner_buffers()
    weights = _weights()
    masked_m = torch.ones(E, dtype=torch.int32)

    def refuse(*a, **kw):
        raise AssertionError("the masked chain must allocate nothing")

    for name in ("empty", "zeros", "ones", "empty_like", "zeros_like",
                 "full", "cat", "stack"):
        monkeypatch.setattr(torch, name, refuse)
    monkeypatch.setattr(torch.Tensor, "contiguous", refuse)

    run_deepgemm_masked_moe(
        weights=weights, masked_m=masked_m, expected_m=17, **bufs)


def test_the_runner_reads_nothing_back_to_the_host(fake_deep_gemm, monkeypatch):
    """No ``.item()`` / ``.cpu()`` / ``.tolist()``: the live row counts stay on
    the device, which is what lets the captured graph depend on them."""
    _fake_act(monkeypatch)
    bufs = _runner_buffers()

    def refuse(*a, **kw):
        raise AssertionError("the masked chain must not synchronize")

    for name in ("item", "tolist", "cpu", "numpy"):
        monkeypatch.setattr(torch.Tensor, name, refuse)

    run_deepgemm_masked_moe(
        weights=_weights(), masked_m=torch.ones(E, dtype=torch.int32),
        expected_m=17, **bufs)


def test_the_runner_refuses_a_device_expected_m(fake_deep_gemm, monkeypatch):
    _fake_act(monkeypatch)
    bufs = _runner_buffers()
    with pytest.raises(RuntimeError, match="expected_m must be a host int"):
        run_deepgemm_masked_moe(
            weights=_weights(), masked_m=torch.ones(E, dtype=torch.int32),
            expected_m=torch.tensor(17), **bufs)


def test_the_runner_refuses_scales_that_would_need_a_repack(
        fake_deep_gemm, monkeypatch):
    _fake_act(monkeypatch)
    bufs = _runner_buffers()
    bufs["x_scales"] = torch.zeros(E, S, H // SCALE_BLOCK)
    with pytest.raises(RuntimeError, match="dispatch activation scales"):
        run_deepgemm_masked_moe(
            weights=_weights(), masked_m=torch.ones(E, dtype=torch.int32),
            expected_m=17, **bufs)
    assert fake_deep_gemm.calls == [], "it must refuse BEFORE any launch"

    bufs = _runner_buffers()
    bufs["act_scales"] = torch.zeros(E, S, I // SCALE_BLOCK)
    with pytest.raises(RuntimeError, match="down-projection input scales"):
        run_deepgemm_masked_moe(
            weights=_weights(), masked_m=torch.ones(E, dtype=torch.int32),
            expected_m=17, **bufs)


# --------------------------------------------------------------------------
# The activation launcher
# --------------------------------------------------------------------------
class _FakeKernel:
    def __init__(self, fn, launches):
        self.fn = fn
        self.launches = launches

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.launches.append(SimpleNamespace(grid=grid, args=args,
                                                 kwargs=kwargs))
        return launch


@pytest.fixture
def act_module(monkeypatch):
    """Import the Triton activation module against a recording stand-in.

    Triton is not installed on a CPU box and the kernel body cannot run there
    anyway, but the LAUNCHER is plain Python: its validation and, above all,
    its grid — which must be a function of the captured shapes only — are
    exactly what has to hold for the captured chain to be correct.
    """
    launches = []
    triton = ModuleType("triton")
    triton.jit = lambda fn: _FakeKernel(fn, launches)
    triton.cdiv = lambda a, b: -(-a // b)
    language = ModuleType("triton.language")
    language.constexpr = object()
    triton.language = language
    monkeypatch.setitem(sys.modules, "triton", triton)
    monkeypatch.setitem(sys.modules, "triton.language", language)
    # Loaded from its file, not through ``batchgen_kernels.triton``: that
    # package's __init__ eagerly imports every kernel module, which is exactly
    # the Triton import the runtime defers.
    path = (Path(__file__).resolve().parents[1] / "batchgen_kernels" /
            "triton" / "moe_masked_act_quant.py")
    spec = importlib.util.spec_from_file_location(
        "_moe_masked_act_quant_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return SimpleNamespace(module=module, launches=launches)


def _act_args(e=E, s=S, i=I):
    return dict(
        gate=torch.zeros(e, s, i, dtype=torch.bfloat16),
        up=torch.zeros(e, s, i, dtype=torch.bfloat16),
        out=torch.zeros(e, s, i, dtype=torch.uint8).view(FP8),
        out_scale=_mn_major(e, s, i // SCALE_BLOCK),
        masked_m=torch.ones(e, dtype=torch.int32),
    )


def test_the_activation_grid_is_independent_of_the_live_row_count(act_module):
    """CUDA-graph safety: the launch geometry comes from the PADDED extent and
    the live count is consumed on the device, inside the row loop."""
    args = _act_args()
    act = act_module.module.silu_mul_masked_group_quant
    act(args["gate"], args["up"], args["out"], args["out_scale"],
        args["masked_m"], group_size=SCALE_BLOCK)
    first = act_module.launches[-1].grid

    args["masked_m"] = torch.full((E,), S, dtype=torch.int32)
    act(args["gate"], args["up"], args["out"], args["out_scale"],
        args["masked_m"], group_size=SCALE_BLOCK)
    assert act_module.launches[-1].grid == first
    # (I / group, row blocks per expert, experts)
    assert first == (I // SCALE_BLOCK, 32, E)


def test_the_activation_passes_the_scale_strides_it_was_given(act_module):
    args = _act_args()
    act_module.module.silu_mul_masked_group_quant(
        args["gate"], args["up"], args["out"], args["out_scale"],
        args["masked_m"], group_size=SCALE_BLOCK)
    launch = act_module.launches[-1]
    # (gate, s0, s1, up, s0, s1, out, s0, s1, scale, s0, s1, s2, masked_m,
    #  size_n, fp8_max, fp8_min) -- every operand's own strides, so the
    #  MN-major scale block is addressed, not assumed contiguous
    got = launch.args
    assert got[0] is args["gate"] and got[1:3] == args["gate"].stride()[:2]
    assert got[3] is args["up"] and got[4:6] == args["up"].stride()[:2]
    assert got[6] is args["out"] and got[7:9] == args["out"].stride()[:2]
    assert got[9] is args["out_scale"]
    assert got[10:13] == args["out_scale"].stride()
    assert got[13] is args["masked_m"]
    assert got[14] == I
    assert (got[15], got[16]) == (448.0, -448.0)
    assert launch.kwargs["BLOCK_N"] == SCALE_BLOCK
    assert launch.kwargs["num_warps"] == 1


@pytest.mark.parametrize("mutate,match", [
    (lambda a: a.update(out=a["out"].to(torch.bfloat16)), "writes"),
    (lambda a: a.update(out_scale=a["out_scale"].to(torch.float16)),
     "f32 group scales"),
    (lambda a: a.update(masked_m=a["masked_m"].to(torch.int64)),
     "int32 masked_m"),
    (lambda a: a.update(masked_m=torch.ones(E + 1, dtype=torch.int32)),
     "masked_m"),
    (lambda a: a.update(up=torch.zeros(E, S, I * 2, dtype=torch.bfloat16)),
     "same"),
    (lambda a: a.update(gate=a["gate"][:, :, :I - 1],
                        up=a["up"][:, :, :I - 1],
                        out=a["out"][:, :, :I - 1]), "multiple of the quant"),
    (lambda a: a.update(out_scale=torch.zeros(E, S, 2)), "out_scale"),
])
def test_the_activation_validates_its_operands(act_module, mutate, match):
    args = _act_args()
    mutate(args)
    with pytest.raises(ValueError, match=match):
        act_module.module.silu_mul_masked_group_quant(
            args["gate"], args["up"], args["out"], args["out_scale"],
            args["masked_m"], group_size=SCALE_BLOCK)


@pytest.mark.skipif(
    importlib.util.find_spec("triton") is None, reason="needs Triton")
def test_the_runner_resolves_the_shipped_activation_kernel():
    """The seam the fakes stand in for points at the shipped kernel."""
    from batchgen_kernels.triton.moe_masked_act_quant import (
        silu_mul_masked_group_quant,
    )

    assert masked_act_quant_fn() is silu_mul_masked_group_quant
