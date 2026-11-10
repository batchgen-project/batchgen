"""Static contract for K3 whole-model dense MLA decode schedule reuse.

The CUDA path is exercised on the remote GPU gates.  This CPU-only contract
prevents a later edit from moving schedule generation back inside every MLA
layer of the whole-model graph, or from calling a decode kernel around the
capability-selected seam (``batchgen.attention.mla.paged_decode``).
"""

import ast
from pathlib import Path


_MODEL_ROOT = Path(__file__).parents[1] / "batchgen/models/moonshotai/kimi_linear"


def _method(tree, class_name, method_name):
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if child.name == method_name:
                        return child
    raise AssertionError(f"missing {class_name}.{method_name}")


def _call_names(node, name):
    return [
        call
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == name
    ]


def _kernel_imports(tree):
    """Modules of the decode-kernel packages a source imports directly."""
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
        elif isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
    return {
        name for name in names
        if name.split(".")[0] in ("flash_mla", "flashinfer")
    }


def test_whole_model_generates_one_mla_schedule_per_forward():
    source = (_MODEL_ROOT / "whole_model_cuda_graph_segments.py").read_text()
    tree = ast.parse(source)
    forward = _method(tree, "KimiLinearWholeModelSegment", "forward")
    schedule = _method(
        tree, "KimiLinearWholeModelSegment", "_get_mla_schedule"
    )

    assert len(
        [
            call
            for call in ast.walk(forward)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "_get_mla_schedule"
        ]
    ) == 1
    assert len(_call_names(schedule, "decode_schedule")) == 1
    # The schedule is the seam's: no kernel package is reached directly.
    assert _kernel_imports(tree) == set()


def test_whole_model_reuses_one_static_kda_cu_vector_per_bucket():
    source = (_MODEL_ROOT / "whole_model_cuda_graph_segments.py").read_text()
    tree = ast.parse(source)
    forward = _method(tree, "KimiLinearWholeModelSegment", "forward")
    helper = _method(
        tree, "KimiLinearWholeModelSegment", "_get_kda_cu_seqlens"
    )

    assert len(
        [
            call
            for call in ast.walk(forward)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "_get_kda_cu_seqlens"
        ]
    ) == 1
    assert len(
        [
            call
            for call in ast.walk(helper)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "arange"
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "torch"
        ]
    ) == 1


def test_layer_graph_decodes_through_the_seam_with_an_optional_schedule():
    source = (_MODEL_ROOT / "cuda_graph_segments.py").read_text()
    tree = ast.parse(source)
    mla = _method(tree, "KimiLinearSpanSegment", "_graph_attention")
    mla_safe = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_mla_decode_graph_safe"
    )

    # The whole-model path hands its shared schedule through; a standalone
    # per-layer capture passes None and the seam records its own.
    assert any(
        isinstance(node, ast.Name) and node.id == "mla_schedule"
        for node in ast.walk(mla)
    )
    (decode,) = _call_names(mla_safe, "mla_paged_decode")
    assert [kw.arg for kw in decode.keywords] == ["schedule"]
    # Golden rule 7: the seam alone picks the kernel by capability.
    assert "get_device_capability" not in source
    assert _kernel_imports(tree) == set()
    assert "NotImplementedError" not in ast.unparse(mla_safe)
