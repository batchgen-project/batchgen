"""`w8a16_gemm` must quantize straight into the MN-major TMA-aligned scale.

DeepGEMM's `fp8_gemm_nt` normalizes the LHS scale through
`get_mn_major_tma_aligned_tensor`
(`csrc/jit_kernels/impls/smxx_layout.hpp:114`). That function early-returns the
caller's tensor untouched when it is already MN-major and TMA aligned
(`smxx_layout.hpp:117-119`):

    if ((batched_sf.stride(0) == tma_aligned_mn * sf_k or dim == 2)
        and batched_sf.stride(1) == 1
        and batched_sf.stride(2) == tma_aligned_mn)
        return (dim == 2) ? batched_sf.squeeze(0) : batched_sf;

otherwise it allocates and launches a `transpose_fp32` kernel
(`smxx_layout.hpp:121-144`). `tma_aligned_mn` is
`get_tma_aligned_size(mn, 4) == align(mn, 16 // 4) == align(mn, 4)`
(`csrc/utils/math.hpp:22-26`).

So producing the aligned layout inside the quantizer removes one transpose
launch per `w8a16_gemm` call. The quantized VALUES are untouched: only the
store addresses of the fp32 scale change.

Everything here is source-level or pure-CPU tensor algebra — no GPU, no Triton,
no DeepGEMM import.
"""

import ast
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
FA3 = ROOT / "batchgen" / "attention" / "mla" / "fa3_backend.py"
FP8_QUANTIZE = ROOT / "batchgen_kernels" / "triton" / "fp8_quantize.py"

# DeepGEMM: kNumTMAAlignmentBytes / sizeof(float) (csrc/utils/math.hpp:22-26).
TMA_ALIGN_ROWS = 16 // 4


def _tree(path):
    return ast.parse(path.read_text(), filename=str(path))


def _func(path, name):
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in {path}")


def _calls(node, name):
    found = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name) and sub.func.id == name:
            found.append(sub)
    return found


def _subscript_calls(node, value_name):
    """Triton launches: `kernel[grid](...)` -> Call whose func is a Subscript."""
    found = []
    for sub in ast.walk(node):
        if (
            isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Subscript)
            and isinstance(sub.func.value, ast.Name)
            and sub.func.value.id == value_name
        ):
            found.append(sub)
    return found


def _aligned_m(rows: int) -> int:
    return -(-rows // TMA_ALIGN_ROWS) * TMA_ALIGN_ROWS


# --------------------------------------------------------------------------
# 1. Structural: the call site asks for the aligned layout.
# --------------------------------------------------------------------------

def test_w8a16_gemm_requests_tma_aligned_scale():
    fn = _func(FA3, "w8a16_gemm")
    calls = _calls(fn, "act_quant")
    assert len(calls) == 1, f"expected exactly one act_quant call, got {len(calls)}"
    kwargs = {kw.arg: kw.value for kw in calls[0].keywords}
    assert "scale_tma_aligned" in kwargs, (
        "w8a16_gemm must pass scale_tma_aligned=True, otherwise deep_gemm.fp8_gemm_nt "
        "launches its own transpose_fp32 on the [m, k/128] scale"
    )
    value = kwargs["scale_tma_aligned"]
    assert isinstance(value, ast.Constant) and value.value is True, (
        f"scale_tma_aligned must be the literal True, got {ast.dump(value)}"
    )


def test_w8a16_gemm_flattens_activation_before_quantizing():
    """act_quant rejects scale_tma_aligned on non-2D input (fa3_backend.py:428).

    w8a16_gemm views the activation to 2D first, so the aligned request is
    always legal there. Pin that view so a future 3D pass-through cannot
    silently turn the aligned request into a hard failure.
    """
    fn = _func(FA3, "w8a16_gemm")
    src = ast.get_source_segment(FA3.read_text(), fn)
    assert "x = activation_bf16.view(-1, activation_bf16.size(-1))" in src


# --------------------------------------------------------------------------
# 2. Structural: the legacy fallback kernel places scales by stride.
# --------------------------------------------------------------------------

def test_act_quant_kernel_2d_scale_store_is_stride_driven():
    """The fallback Triton kernel must honour the scale tensor's strides.

    The aligned scale is a non-contiguous view
    (`[num_blocks, align(M, 4)].transpose(0, 1)[:M, :]`, strides
    `(1, align(M, 4))`). A hard-coded `pid_m * num_blocks + pid_n` offset
    silently writes every element to the wrong place.
    """
    kernel = _func(FA3, "act_quant_kernel_2d")
    arg_names = [a.arg for a in kernel.args.args]
    assert "scale_stride_m" in arg_names and "scale_stride_b" in arg_names, arg_names

    body = ast.get_source_segment(FA3.read_text(), kernel)
    assert "pid_m * scale_stride_m + pid_n * scale_stride_b" in body
    assert "num_blocks_n" not in body, (
        "the row-major scale offset must be gone from act_quant_kernel_2d"
    )


def test_act_quant_launch_sites_pass_scale_strides():
    fn = _func(FA3, "act_quant")
    launches = _subscript_calls(fn, "act_quant_kernel_2d")
    assert len(launches) == 2, f"expected 2 fallback launches, got {len(launches)}"
    src = FA3.read_text()
    for call in launches:
        positional = [ast.get_source_segment(src, a) for a in call.args]
        assert any(a is not None and ".stride(0)" in a for a in positional), positional
        assert any(a is not None and ".stride(1)" in a for a in positional), positional


# --------------------------------------------------------------------------
# 3. The alignment constant in both producers matches DeepGEMM's.
# --------------------------------------------------------------------------

def _aligned_m_expressions(path):
    """Every `aligned_m = <expr>` assignment in the module, as source text."""
    src = path.read_text()
    out = []
    for node in ast.walk(_tree(path)):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "aligned_m"
        ):
            out.append(ast.get_source_segment(src, node.value))
    return out


@pytest.mark.parametrize("path", [FA3, FP8_QUANTIZE])
def test_aligned_m_matches_deep_gemm_get_tma_aligned_size(path):
    exprs = _aligned_m_expressions(path)
    assert exprs, f"no aligned_m assignment found in {path}"
    for expr in exprs:
        code = compile(ast.Expression(ast.parse(expr, mode="eval").body), "<aligned_m>", "eval")
        for rows in (0, 1, 2, 3, 4, 5, 7, 8, 63, 64, 65, 100, 4095, 4096, 100_000):
            got = eval(code, {"__builtins__": {}}, {"M": rows})  # noqa: S307 - arithmetic only
            assert got == _aligned_m(rows), f"{expr} at M={rows}: {got} != {_aligned_m(rows)}"


# --------------------------------------------------------------------------
# 4. CPU: the aligned allocation satisfies DeepGEMM's skip predicate.
# --------------------------------------------------------------------------

def _allocate_aligned_scale(M: int, num_blocks: int) -> torch.Tensor:
    """Byte-for-byte the allocation in fp8_quantize.py:369-377 / fa3_backend.py."""
    aligned_m = ((M + 3) // 4) * 4
    return torch.empty((num_blocks, aligned_m), dtype=torch.float32).transpose(0, 1)[:M, :]


@pytest.mark.parametrize("M", [1, 2, 3, 4, 5, 8, 15, 16, 17, 64, 65, 127, 128, 1000])
@pytest.mark.parametrize("K", [2048, 6144])
def test_aligned_scale_hits_deep_gemm_early_return(M, K):
    num_blocks = K // 128
    scale = _allocate_aligned_scale(M, num_blocks)

    assert tuple(scale.shape) == (M, num_blocks)
    assert scale.dtype == torch.float32

    # smxx_layout.hpp:117-119, with `dim == 2` short-circuiting the stride(0) arm.
    tma_aligned_mn = _aligned_m(M)
    assert scale.stride(0) == 1, scale.stride()
    assert scale.stride(1) == tma_aligned_mn, (scale.stride(), tma_aligned_mn)


@pytest.mark.parametrize("M", [1, 3, 4, 7, 64, 65])
def test_row_major_scale_would_not_hit_early_return(M):
    """Control: today's row-major scale fails the predicate (hence the transpose)."""
    num_blocks = 48
    scale = torch.empty((M, num_blocks), dtype=torch.float32)
    aligned = scale.stride(0) == 1 and scale.stride(1) == _aligned_m(M)
    assert not aligned, "row-major scale must NOT look already-aligned"


# --------------------------------------------------------------------------
# 5. CPU: layout-only change — the stride-driven offsets place the same values.
# --------------------------------------------------------------------------

def _store_by_strides(base_1d: torch.Tensor, values: torch.Tensor,
                      stride_m: int, stride_b: int) -> None:
    """Model the kernel's address arithmetic: `pid_m * stride_m + pid_n * stride_b`."""
    M, num_blocks = values.shape
    rows = torch.arange(M).view(M, 1)
    cols = torch.arange(num_blocks).view(1, num_blocks)
    offsets = rows * stride_m + cols * stride_b
    base_1d[offsets.reshape(-1)] = values.reshape(-1)


@pytest.mark.parametrize("M", [1, 3, 4, 7, 64, 65, 129])
@pytest.mark.parametrize("num_blocks", [16, 48])
def test_stride_driven_store_is_value_identical_in_both_layouts(M, num_blocks):
    """Same scale values, two layouts, one offset formula.

    This is the correctness claim behind the change: the quantizer computes the
    identical fp32 scale per (token, K-block) and only the destination address
    differs. Bit-identical, not approximately equal.
    """
    torch.manual_seed(1234)
    values = torch.rand(M, num_blocks, dtype=torch.float32)

    # Row-major: strides (num_blocks, 1) — the pre-change behaviour, and what
    # `pid_m * num_blocks_n + pid_n` hard-coded.
    row_major = torch.zeros((M, num_blocks), dtype=torch.float32)
    _store_by_strides(row_major.view(-1), values, row_major.stride(0), row_major.stride(1))
    assert torch.equal(row_major, values)

    # MN-major TMA aligned: strides (1, align(M, 4)).
    aligned_m = _aligned_m(M)
    base = torch.zeros((num_blocks, aligned_m), dtype=torch.float32)
    view = base.transpose(0, 1)[:M, :]
    _store_by_strides(base.view(-1), values, view.stride(0), view.stride(1))
    assert torch.equal(view, values)

    # The two layouts are element-wise identical through their logical index.
    assert torch.equal(view, row_major)
    assert view.view(torch.int32).equal(row_major.view(torch.int32))


@pytest.mark.parametrize("M", [1, 3, 7, 65])
def test_aligned_store_does_not_touch_padding_rows(M):
    """Rows `M..align(M, 4)` of the base buffer stay untouched.

    DeepGEMM reads only `[0, mn)` along the MN axis, so the pad is never read;
    pin that the quantizer never writes outside the logical view either.
    """
    num_blocks = 48
    aligned_m = _aligned_m(M)
    base = torch.full((num_blocks, aligned_m), float("nan"), dtype=torch.float32)
    view = base.transpose(0, 1)[:M, :]
    values = torch.rand(M, num_blocks, dtype=torch.float32)
    _store_by_strides(base.view(-1), values, view.stride(0), view.stride(1))

    pad = base[:, M:]
    assert pad.numel() == num_blocks * (aligned_m - M)
    if pad.numel():
        assert bool(torch.isnan(pad).all()), "padding was overwritten"
