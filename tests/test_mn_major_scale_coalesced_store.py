"""The MN-major TMA-aligned scale store must be coalesced, and must not move a
single scale value.

`per_token_blocked_quantize_bf16_to_fp8_1d(..., scale_tma_aligned=True)` returns
a logically `[M, num_blocks]` fp32 scale whose PHYSICAL layout is MN-major with
a TMA-aligned leading dimension: `strides == (1, align(M, 4))`. DeepGEMM's
`get_mn_major_tma_aligned_tensor` early-returns exactly that signature
(`csrc/jit_kernels/impls/smxx_layout.hpp:117-119`), which is why the quantizer
produces it.

The row-major kernel is partitioned one program per TOKEN and stores all
`NUM_BLOCKS` of that token's scales in one go. On the row-major layout
(`strides == (num_blocks, 1)`) that is one contiguous run. On the MN-major
layout the contiguous axis is the TOKEN axis instead, so the same partitioning
degenerates into `NUM_BLOCKS` separate 4-byte stores strided by `align(M, 4)`.

`_per_token_blocked_quantize_mn_major_kernel` flips the partitioning: one
program per (token tile, k-block), so its scale output is `TILE_M` consecutive
fp32 words along the stride-1 token axis.

What is pinned here:
  * the two layouts are driven by their own partitionings (AST),
  * the MN-major store's VECTOR axis is the token axis and its SCALAR axis is
    the k-block (AST) -- i.e. the contiguous axis really is the stride-1 one,
  * both kernels take scales and fp8 payload from the one shared
    `_blocked_scale_and_quantize` helper and neither re-derives the math (AST),
    which is what makes "values bit-identical, only addresses move" structural,
  * the address model: element `[m, b]` lands where the tensor's own strides say
    it must, and each MN-major program's addresses are CONSECUTIVE (CPU),
  * bit-identical `out` bytes and bit-identical scale values between the two
    layouts, including the `num_valid_tokens` poison rows (GPU-gated).

The AST/CPU parts need no GPU, no Triton and no DeepGEMM import.
"""

import ast
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
FP8_QUANTIZE = ROOT / "batchgen_kernels" / "triton" / "fp8_quantize.py"

# DeepGEMM: kNumTMAAlignmentBytes / sizeof(float) (csrc/utils/math.hpp:22-26).
TMA_ALIGN_ROWS = 16 // 4


def _tree():
    return ast.parse(FP8_QUANTIZE.read_text(), filename=str(FP8_QUANTIZE))


def _func(name):
    for node in ast.walk(_tree()):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in {FP8_QUANTIZE}")


def _src(node):
    return ast.unparse(node)


def _assigned(func, target_name):
    """Source text of the last `target_name = ...` inside `func`."""
    found = None
    for sub in ast.walk(func):
        if isinstance(sub, ast.Assign):
            for tgt in sub.targets:
                if isinstance(tgt, ast.Name) and tgt.id == target_name:
                    found = _src(sub.value)
    assert found is not None, f"no assignment to {target_name} in {func.name}"
    return found


def _pure_python_symbols():
    """`mn_major_scale_tile_m` / `MN_MAJOR_MAX_TILE_M` from the REAL source.

    Prefers a normal import. `batchgen_kernels.triton` imports Triton at package
    import time, which a CPU dev box does not have, so fall back to exec'ing
    just these two (Triton-free) nodes out of the module AST. Either way the
    code under test is the shipped source, not a copy.
    """
    try:
        from batchgen_kernels.triton.fp8_quantize import (  # noqa: PLC0415
            MN_MAJOR_MAX_TILE_M,
            mn_major_scale_tile_m,
        )

        return MN_MAJOR_MAX_TILE_M, mn_major_scale_tile_m
    except ImportError:
        pass

    wanted = {"MN_MAJOR_MAX_TILE_M", "mn_major_scale_tile_m"}
    picked = []
    for node in _tree().body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            picked.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in wanted for t in node.targets
        ):
            picked.append(node)
    assert len(picked) == 2, f"expected both of {wanted} at module scope, got {len(picked)}"
    ns = {}
    exec(compile(ast.Module(body=picked, type_ignores=[]), str(FP8_QUANTIZE), "exec"), ns)
    return ns["MN_MAJOR_MAX_TILE_M"], ns["mn_major_scale_tile_m"]


MN_MAJOR_MAX_TILE_M, mn_major_scale_tile_m = _pure_python_symbols()


def _subscript_launches(func, kernel_name):
    """Source text of every `kernel_name[grid](...)` launch inside `func`."""
    out = []
    for sub in ast.walk(func):
        if not isinstance(sub, ast.Call):
            continue
        callee = sub.func
        if not isinstance(callee, ast.Subscript):
            continue
        if isinstance(callee.value, ast.Name) and callee.value.id == kernel_name:
            out.append((_src(callee.slice), _src(sub)))
    return out


# ---------------------------------------------------------------------------
# Shared-math pin: the two partitionings must not own two copies of the math.
# ---------------------------------------------------------------------------


def test_scale_math_lives_in_one_shared_helper():
    helper = _func("_blocked_scale_and_quantize")
    body = _src(helper)
    # The helper is the only place the amax reduction and the fp8 clamp live.
    assert "tl.max(tl.abs(q_float), axis=1)" in body
    assert "FP8_SAFE_MAX_INV" in body
    assert "q_float / scale[:, None]" in body


@pytest.mark.parametrize(
    "kernel",
    ["_per_token_blocked_quantize_1d_kernel", "_per_token_blocked_quantize_mn_major_kernel"],
)
def test_both_kernels_delegate_the_value_math(kernel):
    body = _src(_func(kernel))
    assert "_blocked_scale_and_quantize(q_float)" in body, (
        f"{kernel} must take its scale/payload from the shared helper"
    )
    # No second copy of the reduction or the clamp chain.
    assert "tl.max(tl.abs(" not in body, f"{kernel} re-derives amax"
    assert "FP8_SAFE_MAX_INV" not in body, f"{kernel} re-derives the scale formula"


# ---------------------------------------------------------------------------
# Partitioning pins.
# ---------------------------------------------------------------------------


def _aligned_branch(launcher):
    """The `if scale_tma_aligned:` block of the launcher, and everything after it."""
    for i, stmt in enumerate(launcher.body):
        if (
            isinstance(stmt, ast.If)
            and isinstance(stmt.test, ast.Name)
            and stmt.test.id == "scale_tma_aligned"
        ):
            return stmt, launcher.body[i + 1:]
    raise AssertionError("no `if scale_tma_aligned:` branch in the launcher")


def _as_scope(stmts, name):
    node = ast.parse("def _s():\n    pass\n").body[0]
    node.name = name
    node.body = list(stmts)
    return node


def test_aligned_branch_uses_the_mn_major_kernel_with_a_2d_grid():
    launcher = _func("per_token_blocked_quantize_bf16_to_fp8_1d")
    aligned_if, after = _aligned_branch(launcher)
    aligned = _as_scope(aligned_if.body, "aligned")
    fallthrough = _as_scope(after, "row_major")

    # The aligned branch must launch ONLY the MN-major kernel...
    assert len(_subscript_launches(aligned, "_per_token_blocked_quantize_mn_major_kernel")) == 1
    assert _subscript_launches(aligned, "_per_token_blocked_quantize_1d_kernel") == []
    # ...and the row-major fall-through ONLY the original kernel.
    assert len(_subscript_launches(fallthrough, "_per_token_blocked_quantize_1d_kernel")) == 1
    assert _subscript_launches(fallthrough, "_per_token_blocked_quantize_mn_major_kernel") == []

    # MN-major: 2D grid over (token tiles, k-blocks).
    mn_grid = _assigned(aligned, "grid").replace(" ", "")
    assert "num_blocks" in mn_grid, f"MN-major grid must span k-blocks, got {mn_grid}"
    assert "tile_m" in mn_grid, f"MN-major grid must tile the token axis, got {mn_grid}"
    assert "mn_major_scale_tile_m(M)" in _assigned(aligned, "tile_m").replace(" ", "")

    # Row-major: unchanged 1 program per token.
    rm_grid = _assigned(fallthrough, "grid").replace(" ", "")
    assert rm_grid == "(M,)", f"row-major grid changed: {rm_grid}"


def test_mn_major_scale_store_is_contiguous_along_the_token_axis():
    kernel = _func("_per_token_blocked_quantize_mn_major_kernel")
    expr = _assigned(kernel, "scale_offsets").replace(" ", "")
    # The VECTOR term must ride the token axis (scale_stride_m, == 1 for the
    # MN-major layout) and the SCALAR term the k-block axis. Swapping these is
    # precisely the uncoalesced store this kernel exists to avoid.
    assert "rows*scale_stride_m_i64" in expr, expr
    assert "pid_b*scale_stride_b" in expr, expr
    assert "block_ids" not in expr, expr


def test_row_major_kernel_documents_why_it_is_not_for_the_aligned_layout():
    kernel = _func("_per_token_blocked_quantize_1d_kernel")
    doc = ast.get_docstring(kernel) or ""
    assert "_per_token_blocked_quantize_mn_major_kernel" in doc, (
        "the row-major kernel must point at the MN-major kernel"
    )


def test_launcher_asserts_the_deepgemm_stride_signature():
    body = _src(_func("per_token_blocked_quantize_bf16_to_fp8_1d"))
    assert "scale.stride(0) == 1" in body
    assert "scale.stride(1) == aligned_m" in body


# ---------------------------------------------------------------------------
# Tile-size helper.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "M, expected",
    [
        (1, 1),
        (2, 2),
        (3, 4),
        (4, 4),
        (5, 8),
        (8, 8),
        (9, 16),
        (16, 16),
        (17, 32),
        (32, 32),
        (33, 32),
        (64, 32),
        (100_000, 32),
    ],
)
def test_mn_major_scale_tile_m(M, expected):
    assert mn_major_scale_tile_m(M) == expected


def test_mn_major_scale_tile_m_is_always_a_capped_power_of_two():
    for M in range(1, 200):
        tile = mn_major_scale_tile_m(M)
        assert tile & (tile - 1) == 0, f"M={M} tile={tile} is not a power of two"
        assert 1 <= tile <= MN_MAJOR_MAX_TILE_M
        assert tile <= max(1, 2 ** (M - 1).bit_length())


def test_mn_major_scale_tile_m_rejects_empty():
    with pytest.raises(ValueError):
        mn_major_scale_tile_m(0)


# ---------------------------------------------------------------------------
# Address model (CPU). Mirrors the kernel indexing pinned by the AST tests
# above and checks it against the tensor's own strides.
# ---------------------------------------------------------------------------


def _aligned_scale(M, num_blocks):
    aligned_m = ((M + TMA_ALIGN_ROWS - 1) // TMA_ALIGN_ROWS) * TMA_ALIGN_ROWS
    return torch.empty((num_blocks, aligned_m), dtype=torch.float32).transpose(0, 1)[:M, :]


@pytest.mark.parametrize("M, dim", [(1, 2048), (4, 6144), (17, 6144), (64, 6144), (64, 12288)])
def test_mn_major_program_addresses_match_the_tensor_strides(M, dim):
    num_blocks = (dim + 127) // 128
    scale = _aligned_scale(M, num_blocks)
    stride_m, stride_b = scale.stride()
    assert stride_m == 1

    tile_m = mn_major_scale_tile_m(M)
    num_tiles = (M + tile_m - 1) // tile_m

    seen = {}
    for pid_t in range(num_tiles):
        for pid_b in range(num_blocks):
            rows = [pid_t * tile_m + i for i in range(tile_m)]
            addrs = [r * stride_m + pid_b * stride_b for r in rows if r < M]
            # Coalescing property: one contiguous run per program.
            assert addrs == list(range(addrs[0], addrs[0] + len(addrs))), (
                f"program ({pid_t},{pid_b}) addresses are not consecutive: {addrs}"
            )
            for r, a in zip([r for r in rows if r < M], addrs):
                assert (r, pid_b) not in seen, f"element [{r},{pid_b}] written twice"
                seen[(r, pid_b)] = a

    # Every logical element covered exactly once, at the address torch itself
    # computes for scale[m, b].
    assert len(seen) == M * num_blocks
    for (m, b), addr in seen.items():
        assert addr == m * stride_m + b * stride_b


@pytest.mark.parametrize("M, dim", [(64, 6144), (17, 6144)])
def test_the_row_major_partitioning_would_be_strided_on_the_aligned_layout(M, dim):
    """Pins the defect being fixed, so a revert cannot pass silently."""
    num_blocks = (dim + 127) // 128
    scale = _aligned_scale(M, num_blocks)
    stride_m, stride_b = scale.stride()

    # Row-major partitioning: one program per token writes all num_blocks scales.
    addrs = [0 * stride_m + b * stride_b for b in range(num_blocks)]
    assert addrs != list(range(addrs[0], addrs[0] + len(addrs)))
    assert addrs[1] - addrs[0] == stride_b >= 4, (
        "on the MN-major layout a per-token scale store is strided by the "
        "aligned leading dimension, not contiguous"
    )


# ---------------------------------------------------------------------------
# Bit-identity (GPU).
# ---------------------------------------------------------------------------

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA + Triton")


def _quantize(x, **kw):
    from batchgen_kernels.triton.fp8_quantize import per_token_blocked_quantize_bf16_to_fp8_1d

    return per_token_blocked_quantize_bf16_to_fp8_1d(x, **kw)


@cuda_only
@pytest.mark.parametrize("M, dim", [(1, 2048), (4, 6144), (17, 6144), (64, 6144), (1024, 6144)])
def test_aligned_and_row_major_are_bit_identical(M, dim):
    torch.manual_seed(1234 + M + dim)
    x = (torch.randn(M, dim, device="cuda", dtype=torch.float32) * 4.0).to(torch.bfloat16)

    out_rm, scale_rm = _quantize(x, scale_tma_aligned=False)
    out_al, scale_al = _quantize(x, scale_tma_aligned=True)

    assert scale_rm.stride() == (scale_rm.size(1), 1)
    aligned_m = ((M + TMA_ALIGN_ROWS - 1) // TMA_ALIGN_ROWS) * TMA_ALIGN_ROWS
    assert scale_al.stride() == (1, aligned_m)
    assert scale_al.shape == scale_rm.shape == (M, (dim + 127) // 128)

    # fp8 payload: exact bytes.
    assert torch.equal(out_rm.view(torch.int8), out_al.view(torch.int8))
    # scale: exact bit patterns of the LOGICAL [m, b] contents.
    assert torch.equal(
        scale_rm.contiguous().view(torch.int32),
        scale_al.contiguous().view(torch.int32),
    )


@cuda_only
@pytest.mark.parametrize("M, n_valid", [(64, 1), (64, 31), (64, 32), (64, 33), (64, 64), (17, 5)])
def test_valid_token_poison_is_bit_identical_across_layouts(M, n_valid):
    """`n_valid` is swept across a TILE_M boundary: the MN-major kernel selects
    per row instead of returning early, so a tile that straddles the boundary
    must still poison exactly the rows past it."""
    dim = 6144
    torch.manual_seed(99 + M + n_valid)
    x = (torch.randn(M, dim, device="cuda", dtype=torch.float32) * 4.0).to(torch.bfloat16)
    nvt = torch.tensor([n_valid], device="cuda", dtype=torch.int32)

    out_rm, scale_rm = _quantize(x, num_valid_tokens=nvt, scale_tma_aligned=False)
    out_al, scale_al = _quantize(x, num_valid_tokens=nvt, scale_tma_aligned=True)

    assert torch.equal(out_rm.view(torch.int8), out_al.view(torch.int8))
    assert torch.equal(
        scale_rm.contiguous().view(torch.int32),
        scale_al.contiguous().view(torch.int32),
    )

    # And the poison itself is what the aligned consumer expects.
    assert torch.all(out_al[n_valid:].view(torch.int8) == 0)
    assert torch.all(scale_al[n_valid:] == torch.tensor(1e-12, dtype=torch.float32, device="cuda"))
    assert torch.all(scale_al[:n_valid] > 0)
