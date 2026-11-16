"""The prefill S1 act+quant arm: selector contract and scale plumbing.

CPU-only coverage for the fused SiLU*up + FP8 quant arm of the m-grouped
prefill path: the batch-level selector's allowlist and refusals, the GEMM
wrapper's two SFA layouts (the row-major one the fused kernel writes, and
the column-major one the transpose shim serves), and source pins that keep
the dispatcher's call structure honest. The kernel's numerics run on the
GPU gate, not here; its argument validation runs wherever triton imports.
"""

import ast
from pathlib import Path

import pytest
import torch

import batchgen.moe.grouped_fp8_deepgemm_moe as deepgemm_moe

# glm5/model.py imports triton at module level, so the selector is exec'd in
# isolation with a stubbed debug dict instead of imported (the pattern the
# other CPU-only glm5 unit tests use).
_MODEL_PATH = (
    Path(__file__).resolve().parents[2]
    / "batchgen" / "models" / "glm" / "glm5" / "model.py"
)


def _isolated_selector(debug_value):
    tree = ast.parse(_MODEL_PATH.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_glm5_prefill_s1_act_quant_backend"
    )
    namespace = {"_glm5_moe_debug_dict": lambda: debug_value or {}}
    exec(  # noqa: S102 - exec of first-party source under test
        compile(ast.Module([function], []), str(_MODEL_PATH), "exec"),
        namespace,
    )
    return namespace["_glm5_prefill_s1_act_quant_backend"]


def test_default_is_the_unfused_chain():
    select = _isolated_selector(None)
    assert select("deepgemm") == "unfused"
    assert select("cute_ptrs") == "unfused"


def test_fused_arm_is_selected_and_normalized():
    select = _isolated_selector({"prefill_s1_act_quant": " FuSeD "})
    assert select("deepgemm") == "fused"
    select = _isolated_selector({"prefill_s1_act_quant": "unfused"})
    assert select("deepgemm") == "unfused"
    assert select("cute_ptrs") == "unfused"


def test_fused_arm_refuses_the_ptrs_backend():
    # cute_ptrs fuses S1 itself and feeds a column-major-scale S3 kernel:
    # half-applying the arm would silently compare identical runs.
    select = _isolated_selector({"prefill_s1_act_quant": "fused"})
    with pytest.raises(ValueError, match="deepgemm"):
        select("cute_ptrs")


def test_unknown_spellings_are_refused():
    select = _isolated_selector({"prefill_s1_act_quant": "fussed"})
    with pytest.raises(ValueError, match="prefill_s1_act_quant"):
        select("deepgemm")
    select = _isolated_selector({"prefill_s1_act_quant": 1})
    with pytest.raises(ValueError, match="must be a string"):
        select("deepgemm")


# --------------------------------------------------------------------------
# grouped_fp8_deepgemm_gemm: SFA layout plumbing (fake GEMM symbol, CPU).
# --------------------------------------------------------------------------

_ROWS, _K, _N = 4, 256, 128


def _call_gemm(monkeypatch, x_scale, *, rows_major):
    captured = {}

    def fake_symbol(a_pair, b_pair, d, grouped_layout):
        captured["a"], captured["sfa"] = a_pair

    monkeypatch.setattr(deepgemm_moe, "_deepgemm", (fake_symbol, 128))
    deepgemm_moe.grouped_fp8_deepgemm_gemm(
        torch.zeros(_ROWS + 2, _K, dtype=torch.float8_e4m3fn),
        x_scale,
        None,
        None,
        output=torch.zeros(_ROWS, _N, dtype=torch.bfloat16),
        grouped_layout=torch.zeros(_ROWS, dtype=torch.int32),
        x_scale_rows_major=rows_major,
    )
    return captured


def test_rows_major_scale_is_passed_without_a_copy(monkeypatch):
    scale = torch.zeros(_ROWS + 2, _K // 128, dtype=torch.float32)
    captured = _call_gemm(monkeypatch, scale, rows_major=True)
    sfa = captured["sfa"]
    assert sfa.shape == (_ROWS, _K // 128)
    # The whole point of the fused arm's layout: no transpose shim, the GEMM
    # reads the quant kernel's buffer in place.
    assert sfa.data_ptr() == scale.data_ptr()


def test_column_major_scale_still_goes_through_the_shim(monkeypatch):
    scale = torch.arange(
        (_K // 128) * (_ROWS + 2), dtype=torch.float32
    ).reshape(_K // 128, _ROWS + 2)
    captured = _call_gemm(monkeypatch, scale, rows_major=False)
    sfa = captured["sfa"]
    assert sfa.shape == (_ROWS, _K // 128)
    assert sfa.is_contiguous()
    torch.testing.assert_close(sfa, scale[:, :_ROWS].t().contiguous())
    assert sfa.data_ptr() != scale.data_ptr()  # the shim copies


def test_rows_major_scale_layout_is_validated(monkeypatch):
    transposed = torch.zeros(_K // 128, _ROWS + 2, dtype=torch.float32).t()
    with pytest.raises(ValueError, match="contiguous last dim"):
        _call_gemm(monkeypatch, transposed, rows_major=True)
    short = torch.zeros(_ROWS - 1, _K // 128, dtype=torch.float32)
    with pytest.raises(ValueError, match="fewer rows"):
        _call_gemm(monkeypatch, short, rows_major=True)


# --------------------------------------------------------------------------
# Source pins: the dispatcher's call structure.
# --------------------------------------------------------------------------


def _forward_impl_ast():
    tree = ast.parse(_MODEL_PATH.read_text())
    moe = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Glm5MoE"
    )
    return next(
        node
        for node in moe.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_forward_prefill_grouped_impl"
    )


def _calls_named(function, name):
    return [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id == name)
            or (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == name
            )
        )
    ]


def test_gemm_call_count_matches_the_coverage_pin():
    # The deep_gemm inventory pins exactly 3 grouped_fp8_deepgemm_gemm call
    # sites in this method; the fused arm must share the S3 call, not add
    # a fourth site.
    calls = _calls_named(_forward_impl_ast(), "grouped_fp8_deepgemm_gemm")
    assert len(calls) == 3
    rows_major = [
        call
        for call in calls
        if any(kw.arg == "x_scale_rows_major" for kw in call.keywords)
    ]
    assert len(rows_major) == 1, (
        "exactly the shared S3 call site selects the SFA layout per arm"
    )


def test_both_act_arms_are_reachable_from_the_dispatcher():
    function = _forward_impl_ast()
    assert len(_calls_named(function, "_glm5_silu_mul_quant_ragged")) == 1
    # The unfused chain stays the other selectable arm of the same site.
    assert len(_calls_named(function, "_glm5_act_quant_ragged")) >= 2


# --------------------------------------------------------------------------
# Kernel wrapper validation (runs wherever triton is importable).
# --------------------------------------------------------------------------


def test_ragged_kernel_wrapper_validates_layouts():
    pytest.importorskip("triton")
    from batchgen_kernels.triton.moe_masked_act_quant import (
        silu_mul_ragged_group_quant,
    )

    rows, inter = 8, 256
    gate = torch.zeros(rows, inter, dtype=torch.bfloat16)
    up = torch.zeros(rows, inter, dtype=torch.bfloat16)
    out = torch.zeros(rows, inter, dtype=torch.float8_e4m3fn)
    scale = torch.zeros(rows, inter // 128, dtype=torch.float32)
    counts = torch.zeros(2, dtype=torch.int32)
    cu = torch.zeros(3, dtype=torch.int32)

    with pytest.raises(ValueError, match="row-major"):
        silu_mul_ragged_group_quant(
            gate, up, out,
            torch.zeros(inter // 128, rows, dtype=torch.float32).t(),
            counts, cu,
        )
    with pytest.raises(ValueError, match="same \\[m, I\\] extent"):
        silu_mul_ragged_group_quant(
            gate, up[: rows - 1], out, scale, counts, cu
        )
    with pytest.raises(ValueError, match="counts \\[E\\] and cu_seqlens"):
        silu_mul_ragged_group_quant(
            gate, up, out, scale, counts, torch.zeros(2, dtype=torch.int32)
        )
    with pytest.raises(ValueError, match="int32"):
        silu_mul_ragged_group_quant(
            gate, up, out, scale, counts.long(), cu
        )
