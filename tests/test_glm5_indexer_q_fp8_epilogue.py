"""The indexer-Q fp8 cast lives in the RoPE+Hadamard epilogue, not in ATen.

``fp8_paged_mqa_logits`` reads indexer ``q`` as plain e4m3: there is no q-scale
channel (the per-token fp32 scales in the paged cache cover K only). So the
only thing between the bf16 RoPE+Hadamard output and the scorer used to be two
whole-tensor ATen launches -- ``clamp(-448, 448)`` and a bf16 -> e4m3
``copy_`` -- and both are now the fused kernel's epilogue.

That fold has to be BYTE-identical, not merely close, because the scorer's
top-k is a hard selection: one flipped e4m3 byte can reorder two logits and
change which 2048 of the context the attention reads. The identity rests on
four facts pinned here:

* 448 is the finite maximum of e4m3 and bf16 represents it exactly, so the
  clamp is lossless and the cast that follows it is always in range;
* the epilogue casts the ROUNDED bf16 result, not the fp32 accumulator, so no
  double rounding can creep in;
* in range, torch's cast and the hardware's ``__NV_SATFINITE`` convert both
  round to nearest even, so they agree bit for bit;
* ATen's ``clamp`` passes NaN through and torch's cast then keeps the sign bit,
  while ``__NV_SATFINITE`` drops it -- so the epilogue reattaches it instead of
  trusting the hardware convert.

The duplicated ``FusedRopeHadamardParams`` is the other footgun: the struct is
written out twice, once in the ``.cu`` and once in the binding, with no shared
header. A field added to one and not the other, or added at a different index,
is a silent ABI mismatch that compiles and then reads garbage pointers. Field
order is pinned across both copies here.

These are source-contract tests. Importing the segment module pulls in Triton,
FA3 and DeepGEMM and the kernels need an SM90 device to build, so they parse
the files instead and run anywhere.
"""

from __future__ import annotations

import ast
import pathlib
import re

import pytest


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_CSRC = _REPO_ROOT / "batchgen_kernels/attention/dsa/indexer/csrc"
_KERNEL_SRC = _CSRC / "fused_rope_hadamard.cu"
_BINDING_SRC = _CSRC / "fused_rope_hadamard_binding.cpp"
_INDEXER_INIT = _REPO_ROOT / "batchgen_kernels/attention/dsa/indexer/__init__.py"
_SCORE_SRC = _REPO_ROOT / "batchgen_kernels/attention/dsa/fused_indexer_score.py"
_SCORER_SRC = _REPO_ROOT / "batchgen_kernels/attention/dsa/paged_mqa_scorer.py"
_SEGMENTS_SRC = _REPO_ROOT / "batchgen/models/glm/glm5/cuda_graph_segments.py"
_K_QUANT_SRC = _REPO_ROOT / "batchgen_kernels/attention/dsa/indexer_k_quant.py"

_SEGMENT_CLASSES = ("Glm5DsaAttnSegment", "Glm5FullDsaAttnSegment")

# e4m3's finite maximum. Pinned as a number here so a typo in either the .cu
# literal or the Python constant fails instead of silently widening the range.
_FP8_MAX = 448.0


def _tree(path: pathlib.Path) -> ast.Module:
    return ast.parse(path.read_text())


def _top_level(tree: ast.Module, name: str):
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"no top-level `{name}`")


def _method(tree: ast.Module, class_name: str, method_name: str) -> ast.FunctionDef:
    cls = _top_level(tree, class_name)
    for stmt in cls.body:
        if isinstance(stmt, ast.FunctionDef) and stmt.name == method_name:
            return stmt
    raise AssertionError(f"no `{class_name}.{method_name}`")


def _calls(node, callee: str) -> list[ast.Call]:
    return [
        sub
        for sub in ast.walk(node)
        if isinstance(sub, ast.Call) and ast.unparse(sub.func) == callee
    ]


def _one_call(node, callee: str) -> ast.Call:
    found = _calls(node, callee)
    assert len(found) == 1, f"expected exactly one `{callee}` call, got {len(found)}"
    return found[0]


def _kwargs(call: ast.Call) -> dict[str, str]:
    return {kw.arg: ast.unparse(kw.value) for kw in call.keywords if kw.arg}


def _params_struct_fields(text: str) -> list[str]:
    """Field names of ``FusedRopeHadamardParams``, in declaration order."""
    match = re.search(
        r"struct\s+FusedRopeHadamardParams\s*\{(.*?)\n\};", text, flags=re.S
    )
    assert match is not None, "no FusedRopeHadamardParams definition"
    fields: list[str] = []
    for line in match.group(1).splitlines():
        line = re.sub(r"//.*", "", line).strip()
        if not line.endswith(";") or line.startswith("using "):
            continue
        name = re.sub(r"\s*(\[[^\]]*\])?;$", "", line).split()[-1]
        fields.append(name.lstrip("*&"))
    return fields


# ──────────────────────────────────────────────────────────────────────────────
# The two struct copies stay ABI-identical
# ──────────────────────────────────────────────────────────────────────────────

def test_params_struct_field_order_matches_across_the_two_copies():
    kernel_fields = _params_struct_fields(_KERNEL_SRC.read_text())
    binding_fields = _params_struct_fields(_BINDING_SRC.read_text())
    assert kernel_fields == binding_fields, (
        "FusedRopeHadamardParams is declared twice with no shared header; a "
        "field order or membership difference is a silent ABI mismatch"
    )


def test_params_struct_carries_the_optional_fp8_output_pointer():
    fields = _params_struct_fields(_KERNEL_SRC.read_text())
    assert "out_fp8_ptr" in fields
    # Directly after the bf16 output pointer: pointers stay grouped, so the
    # trailing scalars keep their offsets.
    assert fields.index("out_fp8_ptr") == fields.index("out_ptr") + 1


# ──────────────────────────────────────────────────────────────────────────────
# The epilogue's clamp-and-cast semantics
# ──────────────────────────────────────────────────────────────────────────────

def test_clamp_constant_is_the_exact_e4m3_finite_maximum():
    text = _KERNEL_SRC.read_text()
    match = re.search(r"constexpr\s+float\s+kQFp8Max\s*=\s*([0-9.]+)f?\s*;", text)
    assert match is not None, "no kQFp8Max constant"
    assert float(match.group(1)) == _FP8_MAX
    # And the clamp actually uses it, symmetrically.
    assert "fminf(fmaxf(v, -kQFp8Max), kQFp8Max)" in text


def test_in_tree_fp8_max_constants_agree():
    text = _K_QUANT_SRC.read_text()
    match = re.search(r"^FP8_MAX\s*=\s*([0-9.]+)\s*$", text, flags=re.M)
    assert match is not None, "no FP8_MAX in indexer_k_quant"
    assert float(match.group(1)) == _FP8_MAX


def test_cast_is_saturating_e4m3_and_nan_keeps_its_sign():
    text = _KERNEL_SRC.read_text()
    assert "__nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3)" in text
    # NaN is special-cased the way ATen's clamp is, and re-encoded with the
    # sign bit torch's cast would have kept.
    assert "if (v != v)" in text
    assert "0x80u" in text and "0x7fu" in text


def test_fp8_byte_is_derived_from_the_rounded_bf16_value():
    text = _KERNEL_SRC.read_text()
    match = re.search(
        r"void\s+store_output_bf16_and_e4m3\((.*?)\n\}", text, flags=re.S
    )
    assert match is not None, "no store_output_bf16_and_e4m3"
    body = match.group(1)
    assert "clamped_bf16_to_e4m3(float(out_vals_store[c][i]))" in body, (
        "the e4m3 byte must come from the stored bf16 value, not the fp32 "
        "accumulator, or the two paths round twice differently"
    )
    # bf16 is still produced exactly as the shared ``store_output`` does it.
    assert "out_vals_store[c][i] = out_vals[c][i] * scale;" in body


def test_fp8_store_is_skipped_when_no_buffer_is_given():
    text = _KERNEL_SRC.read_text()
    assert "if (out_fp8 == nullptr) { return; }" in text
    assert "params.out_fp8_ptr == nullptr" in text


def test_binding_validates_the_fp8_buffer_it_is_handed():
    text = _BINDING_SRC.read_text()
    assert "at::ScalarType::Float8_e4m3fn" in text
    assert 'TORCH_CHECK(q_fp8.is_contiguous(), "out_fp8 must be contiguous")' in text
    assert "q_fp8.numel() == x_2d.numel()" in text
    # Default-constructed params keep the pointer null, so the old 6-argument
    # call site is unchanged.
    assert "memset(&params, 0, sizeof(params));" in text


# ──────────────────────────────────────────────────────────────────────────────
# Python wiring: the wrapper forwards the buffer, the scorer stops casting
# ──────────────────────────────────────────────────────────────────────────────

def test_indexer_out_wrapper_forwards_out_fp8():
    fn = _top_level(_tree(_INDEXER_INIT), "fused_rope_hadamard_out")
    args = [a.arg for a in fn.args.args]
    assert args[-1] == "out_fp8"
    assert fn.args.defaults and ast.unparse(fn.args.defaults[-1]) == "None"
    forwarded = ast.unparse(
        _one_call(fn, "_fused_rope_hadamard_cuda.fused_rope_hadamard_out")
    )
    assert "out_fp8" in forwarded


def test_q_wrapper_takes_out_fp8_and_checks_its_dtype_and_size():
    fn = _top_level(_tree(_SCORE_SRC), "rope_hadamard_q_out")
    args = [a.arg for a in fn.args.args]
    assert args[-1] == "out_fp8"
    src = ast.unparse(fn)
    assert "torch.float8_e4m3fn" in src
    assert "out_fp8.numel() != q.numel()" in src
    assert "out_fp8.is_contiguous()" in src


def test_scorer_only_casts_q_when_it_was_not_prequantized():
    tree = _tree(_SCORER_SRC)
    fn = _top_level(tree, "paged_mqa_score_and_topk_out")
    assert "q_prequantized" in [a.arg for a in fn.args.kwonlyargs]

    guards = [
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.If) and ast.unparse(node.test) == "not q_prequantized"
    ]
    assert len(guards) == 1, "expected one `if not q_prequantized:` guard"
    assert _calls(guards[0], "saturate_q_to_e4m3_range"), (
        "the clamp must be the thing the guard removes"
    )
    # The guard is the ONLY place this scorer entry point casts q, so a
    # prequantized call issues neither the clamp nor the copy. Both launches
    # are inside it: the clamp builds the temporary, the copy_ narrows it.
    copy_callee = "q_fp8.view(plan.batch, plan.num_heads, plan.head_dim).copy_"
    assert len(_calls(fn, "saturate_q_to_e4m3_range")) == 1
    assert len(_calls(fn, copy_callee)) == 1
    assert len(_calls(guards[0], copy_callee)) == 1

    # The eager entry point is untouched and still casts for itself.
    eager = _top_level(tree, "paged_mqa_score_and_topk")
    assert "q_prequantized" not in [a.arg for a in eager.args.kwonlyargs]


# ──────────────────────────────────────────────────────────────────────────────
# Both graph segments produce the fp8 q in the epilogue and tell the scorer so
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("segment", _SEGMENT_CLASSES)
def test_segment_hands_the_fp8_buffer_to_the_rope_epilogue(segment):
    fn = _method(_tree(_SEGMENTS_SRC), segment, "forward")
    kwargs = _kwargs(_one_call(fn, "rope_hadamard_q_out"))
    assert "out_fp8" in kwargs, f"{segment}.forward does not fill q_index_fp8"
    # Only the page-split FP8 cache has a real q_index_fp8 buffer; the bf16 AoS
    # scorer gets a zero-extent placeholder, which must stay unwritten.
    assert kwargs["out_fp8"] == (
        "buffers.q_index_fp8 if self.aux_is_page_split else None"
    )


@pytest.mark.parametrize("segment", _SEGMENT_CLASSES)
def test_segment_marks_q_prequantized_on_the_page_split_arm(segment):
    fn = _method(_tree(_SEGMENTS_SRC), segment, "_score_and_topk")
    call = _one_call(fn, "paged_mqa_score_and_topk_out")
    kwargs = _kwargs(call)
    assert kwargs.get("q_prequantized") == "True"
    assert kwargs["q_fp8"] == "buffers.q_index_fp8"
    assert kwargs["q"] == "buffers.q_index"
