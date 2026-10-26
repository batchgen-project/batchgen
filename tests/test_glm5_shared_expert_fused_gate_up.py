"""GLM-5 decode shared expert: fused gate/up chain.

The persistent FP8 shared expert runs as four launches -- quantize(x), ONE
DeepGEMM over the stacked [gate; up] weight, a fused SiLU*up + quantize, the
down DeepGEMM -- instead of quantize/transpose/GEMM for each of gate, up and
down plus a SiLU, a multiply and a staging copy.

CPU contract (always runs):
  * the loader stores gate and up as the row halves of one allocation and the
    wrapper binds it only when that aliasing, the scale-block geometry and the
    scales hold; every violation raises instead of running a different GEMM;
  * ``out=`` is refused by every chain that cannot write into a caller buffer;
  * the MoE graph segment refuses a shared expert without the binding.

GPU (CUDA + DeepGEMM; must RUN on the node, a skip there is a gate failure):
  * (the fused SiLU*up + quantize kernel itself: test_silu_mul_fp8_quantize.py)
  * the fused GEMM's gate / up columns are bit-identical to the two separate
    DeepGEMM products at GLM-5.2 shapes;
  * the whole fused chain is bit-identical to the unfused chain, eagerly and
    under CUDA-graph replay on a side stream, with and without ``out=``.
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from batchgen.models.glm.glm5.wrappers import GLM5ExpertWrapper


ROOT = Path(__file__).resolve().parents[1]
PSM_SRC = (ROOT / "batchgen" / "models" / "glm" / "glm5"
           / "Parallel_Strategy_Manager.py").read_text()

H, INTER = 6144, 2048          # GLM-5.2 hidden size / shared-expert intermediate


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _fp8_weight(rows, cols, gen, device):
    """Random E4M3 bytes without NaN encodings (0x7F / 0xFF)."""
    b = torch.randint(0, 256, (rows, cols), generator=gen, dtype=torch.int32)
    b = b.to(torch.uint8)
    b[(b & 0x7F) == 0x7F] = 0x7E
    return b.to(device).view(torch.float8_e4m3fn)


def _scales(rows, cols, gen, device):
    return (torch.rand(rows // 128, cols // 128, generator=gen) * 4e-3 + 1e-4).to(device)


class _Module:
    """Stand-in for Glm5Expert after the loader ran: three projections whose
    gate/up weights are views of ``gate_up_weight`` (or not, for negatives)."""

    def __init__(self, gate, up, down, gate_up=None):
        self.gate_proj = SimpleNamespace(weight=SimpleNamespace(data=gate))
        self.up_proj = SimpleNamespace(weight=SimpleNamespace(data=up))
        self.down_proj = SimpleNamespace(weight=SimpleNamespace(data=down))
        if gate_up is not None:
            self.gate_up_weight = gate_up


def _wrapper(module, scales, *, is_fp8=True, persistent=True):
    w = GLM5ExpertWrapper.__new__(GLM5ExpertWrapper)
    torch.nn.Module.__init__(w)
    w.__dict__["module"] = module
    w.layer_idx = 7
    w.expert_idx = -1
    w.persistent = persistent
    w.module_key = "shared_expert_7"
    w.weight_dequant_scale = dict(scales)
    w.is_fp8 = is_fp8
    w.cached_gate = w.cached_up = w.cached_down = None
    w.cached_gate_up = w.cached_gate_up_scale = None
    return w


def _loaded(inter=INTER, hidden=H, device="cpu", seed=0):
    """What ``_load_shared_expert_module`` + ``_config_expert_module`` build."""
    g = torch.Generator().manual_seed(seed)
    gate_up = torch.cat([_fp8_weight(inter, hidden, g, device).view(torch.uint8),
                         _fp8_weight(inter, hidden, g, device).view(torch.uint8)]
                        ).view(torch.float8_e4m3fn)
    down = _fp8_weight(hidden, inter, g, device)
    scales = {
        "gate_proj.weight_scale_inv": _scales(inter, hidden, g, device),
        "up_proj.weight_scale_inv": _scales(inter, hidden, g, device),
        "down_proj.weight_scale_inv": _scales(hidden, inter, g, device),
    }
    module = _Module(gate_up[:inter], gate_up[inter:], down, gate_up)
    return module, scales


# --------------------------------------------------------------------------
# CPU contract
# --------------------------------------------------------------------------
def test_register_binds_the_loader_allocation_and_stacks_the_block_scales():
    module, scales = _loaded(inter=256, hidden=384)
    w = _wrapper(module, scales)
    w._register_fp8_weights()
    assert w.cached_gate_up is module.gate_up_weight
    assert w.cached_gate.data_ptr() == w.cached_gate_up.data_ptr()
    assert w.cached_up.data_ptr() == w.cached_gate_up[256:].data_ptr()
    want = torch.cat([scales["gate_proj.weight_scale_inv"],
                      scales["up_proj.weight_scale_inv"]])
    assert torch.equal(w.cached_gate_up_scale, want)
    assert w.cached_gate_up_scale.is_contiguous()
    # the per-projection scales stay bound for the unfused chain
    assert w.weight_dequant_scale["gate_proj.weight_scale_inv"] is scales[
        "gate_proj.weight_scale_inv"]
    w._unregister_fp8_weights()
    assert w.cached_gate_up is None and w.cached_gate_up_scale is None


def test_a_module_without_the_fused_allocation_keeps_the_unfused_chain():
    module, scales = _loaded(inter=256, hidden=384)
    del module.gate_up_weight
    w = _wrapper(module, scales)
    w._register_fp8_weights()
    assert w.cached_gate_up is None
    with pytest.raises(ValueError, match="out= requires the fused gate/up binding"):
        w._forward_impl(torch.zeros(2, 384, dtype=torch.bfloat16),
                        out=torch.zeros(2, 384, dtype=torch.bfloat16))


@pytest.mark.parametrize("case", [
    "separate_allocations", "swapped_halves", "wrong_shape",
    "partial_scale_block", "missing_up_scale", "wrong_scale_shape",
    "wrong_scale_dtype",
])
def test_bind_refuses_every_geometry_that_would_change_the_gemm(case):
    inter, hidden = 256, 384
    module, scales = _loaded(inter=inter, hidden=hidden)
    gate_up = module.gate_up_weight
    if case == "separate_allocations":
        module.gate_proj.weight.data = gate_up[:inter].clone()
    elif case == "swapped_halves":
        module.gate_proj.weight.data = gate_up[inter:]
        module.up_proj.weight.data = gate_up[:inter]
    elif case == "wrong_shape":
        module.gate_up_weight = gate_up[: 2 * inter - 128]
    elif case == "partial_scale_block":
        inter = 192
        module, scales = _loaded(inter=inter, hidden=hidden)
        scales["gate_proj.weight_scale_inv"] = torch.ones(2, 3)
        scales["up_proj.weight_scale_inv"] = torch.ones(2, 3)
    elif case == "missing_up_scale":
        del scales["up_proj.weight_scale_inv"]
    elif case == "wrong_scale_shape":
        scales["up_proj.weight_scale_inv"] = scales["up_proj.weight_scale_inv"][:1]
    elif case == "wrong_scale_dtype":
        scales["gate_proj.weight_scale_inv"] = scales[
            "gate_proj.weight_scale_inv"].double()
    w = _wrapper(module, scales)
    with pytest.raises(RuntimeError):
        w._register_fp8_weights()
    assert w.cached_gate_up is None


def test_a_bf16_expert_never_binds_the_fp8_chain():
    module, scales = _loaded(inter=256, hidden=384)
    w = _wrapper(module, scales, is_fp8=False)
    w._register_fp8_weights()
    assert w.cached_gate_up is None
    with pytest.raises(RuntimeError, match="requires FP8"):
        w._bind_fused_gate_up(module.gate_up_weight)


def test_out_is_refused_by_a_nonpersistent_expert():
    module, scales = _loaded(inter=256, hidden=384)
    w = _wrapper(module, scales, persistent=False)
    with pytest.raises(ValueError, match="persistent"):
        w.forward(torch.zeros(1, 384, dtype=torch.bfloat16),
                  out=torch.zeros(1, 384, dtype=torch.bfloat16))


def test_the_loader_stores_gate_and_up_as_halves_of_one_allocation():
    """Run the real ``_load_shared_expert_module`` body against fakes."""
    tree = ast.parse(PSM_SRC)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == "GLM5ParallelStrategyManager")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
              and n.name == "_load_shared_expert_module")
    ns = {"torch": torch}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "psm", "exec"), ns)
    load = ns["_load_shared_expert_module"]

    g = torch.Generator().manual_seed(3)
    host = {
        3: {"gate_proj.weight": _fp8_weight(256, 384, g, "cpu"),
            "up_proj.weight": _fp8_weight(256, 384, g, "cpu"),
            "down_proj.weight": _fp8_weight(384, 256, g, "cpu")},
    }
    shared = torch.nn.Module()
    for name, shape in (("gate_proj", (256, 384)), ("up_proj", (256, 384)),
                        ("down_proj", (384, 256))):
        lin = torch.nn.Linear(shape[1], shape[0], bias=False)
        setattr(shared, name, lin)
    layers = [SimpleNamespace(mlp=None)] * 3 + [SimpleNamespace(mlp=SimpleNamespace(
        shared_experts=shared))]
    psm = SimpleNamespace(
        engine_config=SimpleNamespace(Basic_Config=SimpleNamespace(device_torch="cpu")),
        FIRST_K_DENSE=3,
        model=SimpleNamespace(model=SimpleNamespace(layers=layers)),
        core_engine=SimpleNamespace(get_tensor=lambda key: host[int(key.rsplit("_", 1)[1])]),
    )
    load(psm)
    gu = shared.gate_up_weight
    assert gu.shape == (512, 384) and gu.is_contiguous()
    assert shared.gate_proj.weight.data_ptr() == gu.data_ptr()
    assert shared.up_proj.weight.data_ptr() == gu[256:].data_ptr()
    u8 = lambda t: t.view(torch.uint8)
    assert torch.equal(u8(shared.gate_proj.weight.data), u8(host[3]["gate_proj.weight"]))
    assert torch.equal(u8(shared.up_proj.weight.data), u8(host[3]["up_proj.weight"]))
    assert torch.equal(u8(shared.down_proj.weight.data), u8(host[3]["down_proj.weight"]))

    host[3]["up_proj.weight"] = _fp8_weight(128, 384, g, "cpu")
    with pytest.raises(RuntimeError, match="same-shape"):
        load(psm)


def test_the_graph_segment_requires_the_fused_binding():
    from batchgen.models.glm.glm5.moe_cuda_graph_segments import Glm5MoEGraphSegment

    seg = Glm5MoEGraphSegment.__new__(Glm5MoEGraphSegment)
    shared = SimpleNamespace(persistent=True, is_fp8=True, cached_gate=1,
                             cached_up=1, cached_down=1, cached_gate_up=None)
    seg.moe = SimpleNamespace(layer_idx=4, shared_experts=shared)
    with pytest.raises(RuntimeError, match="fused gate/up"):
        seg._validate_shared_expert_graph_safe()
    shared.cached_gate_up = object()
    seg._validate_shared_expert_graph_safe()


# --------------------------------------------------------------------------
# GPU bit-identity
# --------------------------------------------------------------------------
def _cuda_ready():
    if not torch.cuda.is_available():
        return "CUDA unavailable"
    try:
        import deep_gemm  # noqa: F401
        import triton  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment dependent
        return f"deep_gemm/triton unavailable: {exc!r}"
    return None


_skip = pytest.mark.skipif(_cuda_ready() is not None, reason=str(_cuda_ready()))
ROWS = (1, 3, 4, 7, 8, 16, 31, 32, 61, 64, 96, 128, 256)


def _mismatch(a, b):
    a8, b8 = a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
    return int((a8 != b8).sum())


def _gpu_wrapper(seed):
    module, scales = _loaded(device="cuda", seed=seed)
    w = _wrapper(module, scales)
    w._register_fp8_weights()
    assert w.cached_gate_up is not None
    return w


@_skip
@pytest.mark.parametrize("rows", ROWS)
def test_fused_gate_up_gemm_columns_equal_the_separate_products(rows):
    import deep_gemm
    from batchgen_kernels.triton.fp8_quantize import (
        per_token_blocked_quantize_bf16_to_fp8_1d,
    )
    w = _gpu_wrapper(seed=11)
    x = (torch.randn(rows, H, device="cuda") * 0.5).to(torch.bfloat16)
    xq, xs = per_token_blocked_quantize_bf16_to_fp8_1d(x, scale_tma_aligned=True)
    fused = torch.empty(rows, 2 * INTER, dtype=torch.bfloat16, device="cuda")
    deep_gemm.fp8_gemm_nt((xq, xs), (w.cached_gate_up, w.cached_gate_up_scale), fused)
    xq_r, xs_r = per_token_blocked_quantize_bf16_to_fp8_1d(x)   # unfused layout
    gate = torch.empty(rows, INTER, dtype=torch.bfloat16, device="cuda")
    up = torch.empty_like(gate)
    deep_gemm.fp8_gemm_nt((xq_r, xs_r), (w.cached_gate,
                          w.weight_dequant_scale["gate_proj.weight_scale_inv"]), gate)
    deep_gemm.fp8_gemm_nt((xq_r, xs_r), (w.cached_up,
                          w.weight_dequant_scale["up_proj.weight_scale_inv"]), up)
    ng, nu = _mismatch(fused[:, :INTER], gate), _mismatch(fused[:, INTER:], up)
    assert ng == 0 and nu == 0, f"rows={rows}: gate {ng} / up {nu} bf16 elements differ"


@_skip
@pytest.mark.parametrize("rows", ROWS)
def test_the_fused_chain_is_bit_identical_to_the_unfused_chain(rows):
    w = _gpu_wrapper(seed=21)
    x = (torch.randn(rows, H, device="cuda") * 0.5).to(torch.bfloat16)
    ref = w._forward_unfused(x)
    got = w._forward_impl(x)
    assert _mismatch(got, ref) == 0
    out = torch.full((rows, H), float("nan"), dtype=torch.bfloat16, device="cuda")
    ret = w.forward(x, out=out)
    assert ret is out
    assert _mismatch(out, ref) == 0


@_skip
def test_the_fused_chain_replays_bit_identically_from_a_side_stream_graph():
    """The decode path: captured inside a CUDA graph, issued on a side stream,
    writing into a buffer that outlives the capture."""
    w = _gpu_wrapper(seed=31)
    rows = 64
    x = (torch.randn(rows, H, device="cuda") * 0.5).to(torch.bfloat16)
    out = torch.empty(rows, H, dtype=torch.bfloat16, device="cuda")
    ref = w._forward_unfused(x)
    side = torch.cuda.Stream()
    for _ in range(2):                       # warm Triton / DeepGEMM JIT
        w.forward(x, out=out)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        main = torch.cuda.current_stream()
        side.wait_stream(main)
        with torch.cuda.stream(side):
            w.forward(x, out=out)
        main.wait_stream(side)
    for seed in range(3):
        x.copy_((torch.randn(rows, H, device="cuda", generator=torch.Generator(
            device="cuda").manual_seed(seed)) * 0.5).to(torch.bfloat16))
        out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert _mismatch(out, w._forward_unfused(x)) == 0
    assert ref.shape == out.shape
