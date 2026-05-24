"""Tests for `batchgen.cuda_graph.composition`.

CaptureContext save/restore + compose_sequential shape contracts. Does not
require CUDA. The integration-level T2 test (real GLM-5 segments via the
adapter) lives in `test_segment_parity.py` and is GPU-gated.
"""

from __future__ import annotations

from typing import Dict, List

import pytest
import torch

from batchgen.cuda_graph.composition import (
    CaptureContext,
    allocate_kv_staging,
    compose_layer_from_segments,
    compose_sequential,
)
from batchgen.cuda_graph.graph_manager import TensorSpec


class _StubSegment:
    """Minimal real-protocol CapturableSegment for shape testing."""

    def __init__(self, name: str, in_specs: Dict[str, TensorSpec], out_specs: Dict[str, TensorSpec]):
        self.name = name
        self._in = in_specs
        self._out = out_specs
        self.setup_calls: List[int] = []
        self.release_calls: List[int] = []

    def get_static_input_specs(self, bucket_size: int): return dict(self._in)
    def get_static_output_specs(self, bucket_size: int): return dict(self._out)
    def setup_static_buffers(self, bucket_size: int): self.setup_calls.append(bucket_size)
    def release_static_buffers(self, bucket_size: int): self.release_calls.append(bucket_size)
    def forward(self, **inputs): return {"out": inputs.get("x")}


# ---- CaptureContext --------------------------------------------------------

def test_capture_context_binds_and_restores():
    class T:
        a = 1
        b = "x"
    with CaptureContext(T, {"a": 99, "b": "y"}):
        assert T.a == 99 and T.b == "y"
    assert T.a == 1 and T.b == "x"


def test_capture_context_restores_on_exception():
    class T:
        a = 1
    try:
        with CaptureContext(T, {"a": 99}):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert T.a == 1


def test_capture_context_rejects_missing_attribute():
    class T:
        a = 1
    with pytest.raises(AttributeError, match="no attribute"):
        with CaptureContext(T, {"missing": 99}):
            pass


# ---- compose_sequential ---------------------------------------------------

def test_compose_sequential_unions_input_specs():
    s1 = _StubSegment("a", {"x": TensorSpec(("batch_size",), torch.float32)}, {})
    s2 = _StubSegment("b", {"y": TensorSpec(("batch_size",), torch.float32)}, {})
    composed = compose_sequential(
        inner=[s1, s2],
        extra_input_specs={"z": TensorSpec(("batch_size",), torch.float32)},
        output_specs={"out": TensorSpec(("batch_size",), torch.float32)},
        forward_fn=lambda inner, **kw: {"out": kw["x"]},
    )
    specs = composed.get_static_input_specs(4)
    assert set(specs.keys()) == {"x", "y", "z"}


def test_compose_sequential_rejects_conflicting_specs():
    s1 = _StubSegment("a", {"x": TensorSpec(("batch_size",), torch.float32)}, {})
    s2 = _StubSegment("b", {"x": TensorSpec(("batch_size",), torch.bfloat16)}, {})
    composed = compose_sequential(
        inner=[s1, s2], extra_input_specs={}, output_specs={},
        forward_fn=lambda inner, **kw: {},
    )
    with pytest.raises(ValueError, match="conflicting spec"):
        composed.get_static_input_specs(4)


def test_compose_sequential_cascades_setup_and_release():
    s1 = _StubSegment("a", {}, {})
    s2 = _StubSegment("b", {}, {})
    composed = compose_sequential(
        inner=[s1, s2], extra_input_specs={}, output_specs={},
        forward_fn=lambda inner, **kw: {},
    )
    composed.setup_static_buffers(4)
    assert s1.setup_calls == [4] and s2.setup_calls == [4]
    composed.release_static_buffers(4)
    assert s1.release_calls == [4] and s2.release_calls == [4]


def test_compose_sequential_requires_inner():
    with pytest.raises(ValueError):
        compose_sequential(inner=[], extra_input_specs={}, output_specs={}, forward_fn=lambda inner, **kw: {})


# ---- compose_layer_from_segments ------------------------------------------

def test_compose_layer_attn_only():
    attn = _StubSegment(
        "attn",
        {"hidden_states": TensorSpec(("batch_size", 64), torch.bfloat16)},
        {"attn_output": TensorSpec(("batch_size", 64), torch.bfloat16)},
    )
    def glue(*, attn, moe, hidden_states):
        return {"hidden_states": hidden_states}

    layer = compose_layer_from_segments(
        attn_segment=attn,
        moe_segment=None,
        glue=glue,
        output_specs={"hidden_states": TensorSpec(("batch_size", 64), torch.bfloat16)},
    )
    specs = layer.get_static_input_specs(4)
    assert "hidden_states" in specs


# ---- allocate_kv_staging --------------------------------------------------

def test_allocate_kv_staging_shapes():
    out = allocate_kv_staging(
        num_layers=4,
        max_bucket=8,
        kv_staging_dim={"primary": 64, "aux": 32},
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )
    assert out["primary"].shape == (4, 8, 1, 1, 64)
    assert out["aux"].shape == (4, 8, 1, 1, 32)
    assert out["primary"].dtype is torch.bfloat16


def test_allocate_kv_staging_rejects_nonpositive():
    with pytest.raises(ValueError):
        allocate_kv_staging(
            num_layers=0, max_bucket=8,
            kv_staging_dim={"p": 1}, dtype=torch.bfloat16, device=torch.device("cpu"),
        )
