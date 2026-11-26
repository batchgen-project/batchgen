"""GLM-5 layer-body unification contract (step 1 of the three-step extraction).

The decoder-layer body exists ONCE, as ``run_glm5_layer_body``
(batchgen/models/glm/glm5/layer_body.py):

* the CUDA-graph layer segment CAPTURES that function instead of spelling the
  op sequence out again, and
* the eager ``Glm5DecoderLayer.forward`` calls the same function outside
  capture.

Three kinds of evidence, all CPU-only:

1. **Identity** -- the callable the segment module runs IS the shared body
   function (not a copy).
2. **Source structure** -- neither the segment's ``forward`` nor the eager
   layer's ``forward`` contains the body's op-sequence text any more (no
   norm/quant kernel call sites); each contains exactly one call to the
   shared body. The eager side is checked at the AST level because
   ``glm5/model.py`` imports triton at module scope and is not importable on
   a CPU-only box.
3. **Behaviour** -- the graph operator set (via the segment) and an
   eager-shaped operator set (fusions off, no row mask) produce bit-equal
   hidden states through the one body, with the norm kernel stubbed by a
   CPU reference. Token-level equality of the real kernels with graphs on
   and off stays a real-machine gate.
"""

import ast
import pathlib
import types

import pytest
import torch

import batchgen.attention.fused_kernels as fused_kernels
import batchgen.models.glm.glm5.layer_body as layer_body_module
import batchgen.models.glm.glm5.layer_cuda_graph_segments as layer_segments_module
from batchgen.models.glm.glm5.layer_body import (
    Glm5LayerBodyOps,
    run_glm5_layer_body,
)
from batchgen.models.glm.glm5.layer_cuda_graph_segments import (
    Glm5DecoderLayerGraphSegment,
)

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_MODEL_SRC = _REPO_ROOT / "batchgen/models/glm/glm5/model.py"
_LAYER_SEG_SRC = _REPO_ROOT / "batchgen/models/glm/glm5/layer_cuda_graph_segments.py"

_BODY_KERNEL_CALLS = {
    "cuda_add_rmsnorm",
    "cuda_add_rmsnorm_fp8_quant",
    "act_quant",
    "cuda_rmsnorm",
}

_H = 8


# --------------------------------------------------------------------------- #
# AST helpers
# --------------------------------------------------------------------------- #

def _method_def(path: pathlib.Path, class_name: str, func_name: str) -> ast.AST:
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for sub in node.body:
                if (
                    isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and sub.name == func_name
                ):
                    return sub
            raise AssertionError(
                f"{path.name}: no def {func_name} in {class_name}"
            )
    raise AssertionError(f"{path.name}: no class {class_name}")


def _call_names(func_node: ast.AST) -> list:
    names = []
    for node in ast.walk(func_node):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                names.append(func.id)
            elif isinstance(func, ast.Attribute):
                names.append(func.attr)
    return names


# --------------------------------------------------------------------------- #
# 1. Identity: the segment runs THE shared function
# --------------------------------------------------------------------------- #

def test_the_segment_module_runs_the_shared_body_function():
    assert (
        layer_segments_module.run_glm5_layer_body
        is layer_body_module.run_glm5_layer_body
    )
    assert (
        layer_segments_module.Glm5LayerBodyOps
        is layer_body_module.Glm5LayerBodyOps
    )


# --------------------------------------------------------------------------- #
# 2. Source structure: the op sequence lives in the body alone
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "path,class_name",
    [
        (_LAYER_SEG_SRC, "Glm5DecoderLayerGraphSegment"),
        (_MODEL_SRC, "Glm5DecoderLayer"),
    ],
)
def test_forward_calls_the_shared_body_and_spells_no_kernel(path, class_name):
    forward = _method_def(path, class_name, "forward")
    names = _call_names(forward)
    assert names.count("run_glm5_layer_body") == 1, (
        f"{class_name}.forward must run the layer through the ONE shared "
        f"body function exactly once, found {names.count('run_glm5_layer_body')}"
    )
    spelled = sorted(set(names) & _BODY_KERNEL_CALLS)
    assert not spelled, (
        f"{class_name}.forward still spells out the layer body's kernel "
        f"call sites {spelled}; the op sequence belongs to "
        "run_glm5_layer_body only"
    )


def test_the_eager_layer_imports_the_body_from_the_shared_module():
    tree = ast.parse(_MODEL_SRC.read_text())
    imports = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "layer_body"
    ]
    assert imports, "glm5/model.py must import from .layer_body"
    imported = {alias.name for node in imports for alias in node.names}
    assert {"Glm5LayerBodyOps", "run_glm5_layer_body"} <= imported


# --------------------------------------------------------------------------- #
# 3. Behaviour: both operator sets agree through the one body
# --------------------------------------------------------------------------- #

def _rmsnorm_ref(x, weight, eps):
    x32 = x.float()
    var = x32.pow(2).mean(dim=-1, keepdim=True)
    return (x32 * torch.rsqrt(var + eps) * weight.float()).to(x.dtype)


def _fake_add_rmsnorm(residual, hidden, weight, eps,
                      num_valid_tokens=None, row_valid_mask=None):
    hid = hidden.float()
    if row_valid_mask is not None:
        keep = (row_valid_mask.float() != 0).view(
            -1, *([1] * (hidden.dim() - 1))
        )
        hid = torch.where(keep, hid, torch.zeros_like(hid))
    updated = (residual.float() + hid).to(residual.dtype)
    return _rmsnorm_ref(updated, weight, eps), updated


class _Norm:
    def __init__(self):
        self.weight = torch.linspace(0.5, 1.5, _H, dtype=torch.bfloat16)
        self.eps = 1e-5

    def __call__(self, x):
        return _rmsnorm_ref(x, self.weight, self.eps)


def _attn_math(x):
    return (x * 0.5 + 0.125).to(torch.bfloat16)


def _ffn_math(x):
    return (x * 2.0 + 0.25).to(torch.bfloat16)


def _make_shared_layer():
    return types.SimpleNamespace(
        layer_idx=0,
        hidden_size=_H,
        input_layernorm=_Norm(),
        post_attention_layernorm=_Norm(),
        mlp=_ffn_math,
    )


class _FakeDsaSegment:
    consumes_fused_hidden_fp8 = False
    max_seqlen = 16
    index_topk = 8
    primary_blocked_k = torch.zeros(1, 1, 1, 6)
    aux_blocked_k = None

    def attn_output_row_mask(self, batch_size):
        # All-ones mask: predicate semantics keep every row, so the graph
        # operator set must match the maskless eager one bit for bit.
        return torch.ones(batch_size, dtype=torch.bfloat16)

    def forward(self, **kwargs):
        batch = kwargs["hidden_states"].shape[0]
        return {
            "attn_output": _attn_math(kwargs["hidden_states"]),
            "primary_k_tensor": torch.zeros(batch, 1, 1, 6),
        }


class _FakeMoeSegment:
    consumes_fused_padded_fp8 = False

    def forward(self, *, padded, rank_token_counts,
                padded_fp8=None, padded_scale=None):
        assert padded_fp8 is None and padded_scale is None
        return {"moe_output": _ffn_math(padded)}


def _eager_ops():
    def attn(*, hidden_states, hidden_fp8, hidden_scale):
        assert hidden_fp8 is None and hidden_scale is None
        return {
            "attn_output": _attn_math(hidden_states),
            "attn_weights": None,
            "present": ("present",),
        }

    def ffn(hidden_states, hidden_fp8, hidden_scale):
        assert hidden_fp8 is None and hidden_scale is None
        return _ffn_math(hidden_states)

    return Glm5LayerBodyOps(attn=attn, ffn=ffn)


@pytest.fixture
def ref_add_rmsnorm(monkeypatch):
    monkeypatch.setattr(fused_kernels, "cuda_add_rmsnorm", _fake_add_rmsnorm)


def test_eager_and_graph_operator_sets_agree_through_the_one_body(
    ref_add_rmsnorm,
):
    hidden = torch.linspace(-1.0, 1.0, 2 * _H, dtype=torch.bfloat16).view(
        2, 1, _H
    )

    eager_out = run_glm5_layer_body(
        layer=_make_shared_layer(),
        ops=_eager_ops(),
        hidden_states=hidden.clone(),
    )
    assert eager_out["present"] == ("present",)
    assert eager_out["attn_weights"] is None

    segment = Glm5DecoderLayerGraphSegment(
        layer=_make_shared_layer(),
        dsa_segment=_FakeDsaSegment(),
        moe_segment=_FakeMoeSegment(),
        device=torch.device("cpu"),
        world_size=1,
    )
    graph_out = segment.forward(
        hidden_states=hidden.clone(),
        position_ids=torch.zeros(2, 1, dtype=torch.int64),
        cache_seqlens=torch.ones(2, dtype=torch.int32),
        primary_slot_indices=torch.zeros(2, dtype=torch.int32),
        aux_slot_indices=torch.zeros(2, dtype=torch.int32),
        num_valid_tokens=torch.tensor([2], dtype=torch.int32),
        rank_token_counts=torch.tensor([2], dtype=torch.int64),
        residual=None,
        defer_residual_add=False,
    )
    assert "primary_k_tensor" in graph_out

    assert torch.equal(
        eager_out["hidden_states"], graph_out["hidden_states"]
    ), (
        "the eager and graph operator sets diverged through the shared "
        "layer body"
    )
