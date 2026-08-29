"""Static contract for K3 whole-model FlashMLA scheduler reuse.

The CUDA path is exercised on the remote H200 gate.  This CPU-only contract
prevents a later edit from moving metadata generation back inside every MLA
layer of the whole-model graph.
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


def test_whole_model_generates_one_flashmla_metadata_pair_per_forward():
    source = (_MODEL_ROOT / "whole_model_cuda_graph_segments.py").read_text()
    tree = ast.parse(source)
    forward = _method(tree, "KimiLinearWholeModelSegment", "forward")
    metadata = _method(
        tree, "KimiLinearWholeModelSegment", "_get_flashmla_metadata"
    )

    assert len(
        [
            call
            for call in ast.walk(forward)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "_get_flashmla_metadata"
        ]
    ) == 1
    assert len(_call_names(metadata, "get_mla_metadata")) == 1


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
    assert len(_call_names(helper, "arange")) == 1


def test_layer_graph_keeps_standalone_flashmla_fallback():
    source = (_MODEL_ROOT / "cuda_graph_segments.py").read_text()
    tree = ast.parse(source)
    mla = _method(tree, "KimiLinearSpanSegment", "_graph_attention")
    mla_safe = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_mla_decode_graph_safe"
    )

    # Standalone per-layer captures still own metadata.  The whole-model path
    # supplies both optional arguments and therefore takes the shared branch.
    assert any(
        isinstance(node, ast.Name) and node.id == "flashmla_metadata"
        for node in ast.walk(mla)
    )
    assert any(
        isinstance(node, ast.Name) and node.id == "flashmla_num_splits"
        for node in ast.walk(mla)
    )
    assert len(_call_names(mla_safe, "get_mla_metadata")) == 1
