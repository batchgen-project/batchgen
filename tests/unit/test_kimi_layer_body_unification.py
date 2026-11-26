"""Kimi layer-body unification contract (step 2 of the three-step extraction).

Kimi has exactly TWO decoder-layer bodies, one per architecture, each written
ONCE in ``batchgen/models/moonshotai/kimi_linear/layer_body.py``:

* ``run_kimi_layer_body`` — the plain pre-norm body;
* ``run_kimi_block_residual_layer_body`` — the K3/K3.1 Block Attention
  Residual body.

The CUDA-graph span segment CAPTURES those functions, the compare-mode
reference (``_eager_span``) calls them with its own operator set, and the
eager ``KimiDecoderLayer.forward`` / ``_forward_attn_residual`` call the same
functions outside capture. The five former spellings are gone.

Three kinds of evidence, all CPU-only:

1. **Identity** — the module the segment runs imports THE shared body
   functions (not copies).
2. **Source structure** — none of the former copy sites contains the body's
   op-sequence text any more (no norm / depth-mix call sites); each contains
   only calls to the shared bodies. The eager side is checked at the AST
   level because ``kimi_linear/model.py`` imports ``fla`` at module scope and
   is not importable on a CPU-only box.
3. **Behaviour** — a runtime oracle replays the PRE-change call sequences
   (the old segment forward, the old ``_forward_block_residual`` in both its
   ``write_boundary`` modes, the old eager bodies) and the one body under
   each operator set must match them bit for bit, static-buffer side effects
   included. Token-level equality of the real kernels with graphs on and off
   stays a real-machine gate.
"""

import ast
import pathlib
import types

import pytest
import torch

import batchgen.models.moonshotai.kimi_linear.cuda_graph_segments as cgs
import batchgen.models.moonshotai.kimi_linear.layer_body as layer_body_module
from batchgen.models.moonshotai.kimi_linear.block_residual import (
    BlockResidualBuffer,
    apply_attn_res,
    gather_attn_residual_rows,
)
from batchgen.models.moonshotai.kimi_linear.cuda_graph_segments import (
    KimiLinearSpanSegment,
)
from batchgen.models.moonshotai.kimi_linear.layer_body import (
    KimiLayerBodyOps,
    run_kimi_block_residual_layer_body,
    run_kimi_layer_body,
)

_KIMI_SRC = (
    pathlib.Path(__file__).resolve().parents[2]
    / "batchgen/models/moonshotai/kimi_linear"
)
_MODEL_SRC = _KIMI_SRC / "model.py"
_SEGMENTS_SRC = _KIMI_SRC / "cuda_graph_segments.py"

# The body's own op-sequence call sites: a former copy site that still calls
# any of these is spelling the layer body out again.
_BODY_OP_CALLS = {
    "input_layernorm",
    "post_attention_layernorm",
    "apply_attn_res",
    "_apply_attn_res",
}

_H = 8
_BLOCK = 2
_COLUMNS = 3


# --------------------------------------------------------------------------- #
# AST helpers
# --------------------------------------------------------------------------- #

def _method_def(path, class_name, func_name):
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


def _call_names(func_node):
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
# 1. Identity: the segment module runs THE shared functions
# --------------------------------------------------------------------------- #

def test_the_segment_module_runs_the_shared_body_functions():
    assert cgs.run_kimi_layer_body is layer_body_module.run_kimi_layer_body
    assert (
        cgs.run_kimi_block_residual_layer_body
        is layer_body_module.run_kimi_block_residual_layer_body
    )
    assert cgs.KimiLayerBodyOps is layer_body_module.KimiLayerBodyOps


# --------------------------------------------------------------------------- #
# 2. Source structure: the op sequences live in the two bodies alone
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "path,class_name,func_name,expected_bodies",
    [
        (
            _SEGMENTS_SRC,
            "KimiLinearSpanSegment",
            "forward",
            {"run_kimi_layer_body": 1,
             "run_kimi_block_residual_layer_body": 1},
        ),
        (
            _SEGMENTS_SRC,
            "KimiLinearDecodeGraph",
            "_eager_span",
            {"run_kimi_layer_body": 1,
             "run_kimi_block_residual_layer_body": 1},
        ),
        (
            _MODEL_SRC,
            "KimiDecoderLayer",
            "forward",
            {"run_kimi_layer_body": 1,
             "run_kimi_block_residual_layer_body": 0},
        ),
        (
            _MODEL_SRC,
            "KimiDecoderLayer",
            "_forward_attn_residual",
            {"run_kimi_layer_body": 0,
             "run_kimi_block_residual_layer_body": 1},
        ),
    ],
)
def test_former_copy_sites_call_the_shared_bodies_and_spell_no_ops(
    path, class_name, func_name, expected_bodies
):
    func = _method_def(path, class_name, func_name)
    names = _call_names(func)
    for body_name, count in expected_bodies.items():
        assert names.count(body_name) == count, (
            f"{class_name}.{func_name} must call {body_name} exactly "
            f"{count} time(s), found {names.count(body_name)}"
        )
    spelled = sorted(set(names) & _BODY_OP_CALLS)
    assert not spelled, (
        f"{class_name}.{func_name} still spells out the layer body's op "
        f"call sites {spelled}; the op sequence belongs to layer_body.py only"
    )


def test_the_former_second_spelling_is_gone():
    """``_forward_block_residual`` was the segment-side second copy of the
    block-residual body (shared by capture and compare through its
    ``write_boundary`` switch); the body now lives in layer_body.py alone."""
    assert not hasattr(KimiLinearSpanSegment, "_forward_block_residual")
    assert "_forward_block_residual" not in _SEGMENTS_SRC.read_text()


def test_the_eager_layer_imports_the_bodies_from_the_shared_module():
    tree = ast.parse(_MODEL_SRC.read_text())
    imports = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "layer_body"
    ]
    assert imports, "kimi_linear/model.py must import from .layer_body"
    imported = {alias.name for node in imports for alias in node.names}
    assert {
        "KimiLayerBodyOps",
        "run_kimi_layer_body",
        "run_kimi_block_residual_layer_body",
    } <= imported


# --------------------------------------------------------------------------- #
# 3. Behaviour: the one body replays every pre-change call sequence
# --------------------------------------------------------------------------- #

class _Norm:
    """CPU RMSNorm with the attributes apply_attn_res reads."""

    def __init__(self, seed):
        gen = torch.Generator().manual_seed(seed)
        self.weight = torch.randn(_H, generator=gen) * 0.02 + 1.0
        self.variance_epsilon = 1e-6

    def __call__(self, x):
        x32 = x.float()
        out = x32 * torch.rsqrt(
            x32.pow(2).mean(-1, keepdim=True) + self.variance_epsilon
        )
        return (out * self.weight).to(x.dtype)


def _attn_math(x):
    return x * 0.5 + 0.125


def _ffn_math(x):
    return x * 2.0 + 0.25


def _make_layer(layer_idx, *, fold_ffn, block_residual):
    layer = types.SimpleNamespace(
        layer_idx=layer_idx,
        hidden_size=_H,
        is_linear_attn=False,
        self_attn=types.SimpleNamespace(kv_lora_rank=4, qk_rope_head_dim=0),
        input_layernorm=_Norm(11 + layer_idx),
        post_attention_layernorm=_Norm(23 + layer_idx),
        mlp=_ffn_math,
    )
    if not fold_ffn:
        # The segment constructor derives fold_ffn from the presence of the
        # MoE block, exactly as production layers do.
        layer.block_sparse_moe = object()
    if block_residual:
        layer.use_attn_residuals = True
        layer.attn_res_block_size = _BLOCK
        layer.self_attention_res_norm = _Norm(31 + layer_idx)
        layer.mlp_res_norm = _Norm(43 + layer_idx)
        layer.self_attention_res_proj = torch.nn.Linear(_H, 1, bias=False)
        layer.mlp_res_proj = torch.nn.Linear(_H, 1, bias=False)
        gen = torch.Generator().manual_seed(53 + layer_idx)
        with torch.no_grad():
            layer.self_attention_res_proj.weight.copy_(
                torch.randn(1, _H, generator=gen)
            )
            layer.mlp_res_proj.weight.copy_(torch.randn(1, _H, generator=gen))
    return layer


def _make_segment(layer):
    segment = KimiLinearSpanSegment(
        layer,
        layer.layer_idx,
        statics={},
        page_size_tokens=16,
        dtype=torch.float32,
    )

    def fake_graph_attention(normed, mla_schedule=None, kda_cu_seqlens=None):
        batch = normed.shape[0]
        return _attn_math(normed), {
            "k_tensor": normed[..., :4].reshape(batch, 1, 1, 4)
        }

    segment._graph_attention = fake_graph_attention
    return segment


def _hidden(batch=2, seq=1, seed=7):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(batch, seq, _H, generator=gen) * 0.5


def _bank(batch=2, seq=1, columns=_COLUMNS, seed=77):
    gen = torch.Generator().manual_seed(seed)
    buf = torch.zeros(batch * seq, columns, _H)
    buf[:, :].copy_(torch.randn(batch * seq, columns, _H, generator=gen) * 0.3)
    return buf


# ---- the PRE-change call sequences, replayed verbatim ---------------------- #

def _ref_old_segment_plain(layer, hidden_states, attention, fold_ffn):
    """KimiLinearSpanSegment.forward, plain arm, as it was before step 2."""
    normed = layer.input_layernorm(hidden_states)
    attn_out, kv_outputs = attention(normed)
    residual = hidden_states + attn_out
    normed_out = layer.post_attention_layernorm(residual)
    if fold_ffn:
        out = {"hidden": residual + layer.mlp(normed_out)}
    else:
        out = {"normed": normed_out, "residual": residual}
    out.update(kv_outputs)
    return out


def _ref_old_eager_plain(layer, hidden_states, run_attn, run_ffn):
    """KimiDecoderLayer.forward, plain arm, as it was before step 2."""
    residual = hidden_states
    hidden_states = layer.input_layernorm(hidden_states)
    hidden_states = run_attn(hidden_states)
    hidden_states = residual + hidden_states
    residual = hidden_states
    hidden_states = layer.post_attention_layernorm(hidden_states)
    hidden_states = run_ffn(hidden_states)
    return (residual + hidden_states,)


def _ref_old_forward_block_residual(
    segment, hidden_states, attention, block_residual, *, write_boundary
):
    """KimiLinearSpanSegment._forward_block_residual as it was before step 2
    (the captured path under ``write_boundary=True``; compare under False)."""
    layer = segment.layer
    hidden_size = segment.hidden_size
    flat_prefix = hidden_states.reshape(-1, hidden_size)
    active = block_residual[:, : segment.num_blocks_before]

    attn_input = hidden_states
    if segment.num_blocks_before:
        attn_input = apply_attn_res(
            flat_prefix,
            active,
            layer.self_attention_res_proj,
            layer.self_attention_res_norm,
        ).view_as(hidden_states)

    prefix_sum = hidden_states
    if segment.is_block_boundary:
        if write_boundary:
            block_residual[:, segment.num_blocks_before].copy_(flat_prefix)
            active = block_residual[:, : segment.num_blocks_after]
        else:
            active = torch.cat((active, flat_prefix.unsqueeze(1)), dim=1)
        prefix_sum = None

    normed = layer.input_layernorm(attn_input)
    attn_out, kv_outputs = attention(normed)
    prefix_sum = attn_out if prefix_sum is None else prefix_sum + attn_out

    ffn_input = apply_attn_res(
        prefix_sum.reshape(-1, hidden_size),
        active,
        layer.mlp_res_proj,
        layer.mlp_res_norm,
    ).view_as(hidden_states)
    normed_out = layer.post_attention_layernorm(ffn_input)
    if segment.fold_ffn:
        out = {"hidden": prefix_sum + layer.mlp(normed_out)}
    else:
        out = {"normed": normed_out, "residual": prefix_sum}
    out.update(kv_outputs)
    return out


def _ref_old_eager_attn_residual(
    layer, hidden_states, block_residual, run_attn, run_ffn, records
):
    """KimiDecoderLayer._forward_attn_residual as it was before step 2
    (in-place merges, growing boundary append, probe stages recorded)."""
    batch_size, seq_len, hidden_size = hidden_states.shape
    records.append("entry")
    prefix_sum = hidden_states

    if block_residual is not None and block_residual.shape[1] > 0:
        hidden_states = apply_attn_res(
            prefix_sum.view(-1, hidden_size),
            block_residual,
            layer.self_attention_res_proj,
            layer.self_attention_res_norm,
        ).view(batch_size, -1, hidden_size)
        records.append("post_input_depth_mix")

    if layer.layer_idx % layer.attn_res_block_size == 0:
        snapshot = gather_attn_residual_rows(
            prefix_sum.view(-1, hidden_size),
            block_residual,
            layer.self_attention_res_norm,
        )
        block_residual = BlockResidualBuffer.append(block_residual, snapshot)
        prefix_sum = None

    hidden_states = layer.input_layernorm(hidden_states)
    hidden_states = run_attn(hidden_states)
    records.append("post_attention")

    if prefix_sum is not None:
        prefix_sum.add_(hidden_states)
    else:
        prefix_sum = hidden_states
    records.append("post_attention_residual")

    hidden_states = apply_attn_res(
        prefix_sum.view(-1, hidden_size),
        block_residual,
        layer.mlp_res_proj,
        layer.mlp_res_norm,
    ).view(batch_size, -1, hidden_size)
    records.append("post_mlp_depth_mix")

    hidden_states = layer.post_attention_layernorm(hidden_states)
    records.append("post_mlp_norm")
    hidden_states = run_ffn(hidden_states)
    records.append("post_ffn")

    prefix_sum.add_(hidden_states)
    records.append("output")
    return prefix_sum, block_residual


# ---- plain body ------------------------------------------------------------ #

@pytest.mark.parametrize("fold_ffn", [True, False], ids=["fold", "halves"])
def test_plain_segment_forward_replays_the_old_sequence(fold_ffn):
    layer = _make_layer(0, fold_ffn=fold_ffn, block_residual=False)
    segment = _make_segment(layer)
    assert segment.fold_ffn is fold_ffn
    hidden = _hidden()

    got = segment.forward(hidden.clone())
    want = _ref_old_segment_plain(
        layer, hidden.clone(), segment._graph_attention, fold_ffn
    )

    assert got.keys() == want.keys()
    for key in want:
        assert torch.equal(got[key], want[key]), (
            f"plain segment output {key!r} diverged from the pre-change "
            "sequence"
        )
    assert "k_tensor" in got, "the attention extras must ride along"


def test_plain_eager_operator_set_replays_the_old_eager_sequence():
    layer = _make_layer(0, fold_ffn=True, block_residual=False)
    hidden = _hidden()

    def eager_attn(normed):
        return _attn_math(normed), {}

    got = run_kimi_layer_body(
        layer=layer,
        ops=KimiLayerBodyOps(attn=eager_attn, ffn=_ffn_math),
        hidden_states=hidden.clone(),
    )
    (want,) = _ref_old_eager_plain(
        layer, hidden.clone(), _attn_math, _ffn_math
    )
    assert torch.equal(got["hidden"], want), (
        "the eager operator set diverged from the pre-change eager sequence"
    )


def test_plain_body_refuses_block_residual_ops():
    layer = _make_layer(0, fold_ffn=True, block_residual=False)
    with pytest.raises(ValueError, match="block residuals"):
        run_kimi_layer_body(
            layer=layer,
            ops=KimiLayerBodyOps(
                attn=lambda normed: (normed, {}),
                ffn=_ffn_math,
                append_boundary=lambda flat, active: active,
            ),
            hidden_states=_hidden(),
        )


def test_block_residual_body_requires_the_boundary_op():
    layer = _make_layer(0, fold_ffn=True, block_residual=True)
    with pytest.raises(ValueError, match="append_boundary"):
        run_kimi_block_residual_layer_body(
            layer=layer,
            ops=KimiLayerBodyOps(attn=lambda normed: (normed, {})),
            hidden_states=_hidden(),
            block_residual=_bank(),
        )


# ---- block-residual body --------------------------------------------------- #

@pytest.mark.parametrize("fold_ffn", [True, False], ids=["fold", "halves"])
@pytest.mark.parametrize(
    "layer_idx", [0, 1, 2], ids=["boundary0", "mid", "boundary2"]
)
def test_block_residual_segment_forward_replays_the_captured_sequence(
    layer_idx, fold_ffn
):
    """The captured path: static-buffer boundary write, out-of-place merges."""
    layer = _make_layer(layer_idx, fold_ffn=fold_ffn, block_residual=True)
    segment = _make_segment(layer)
    assert segment.use_block_residual
    hidden = _hidden(seed=70 + layer_idx)

    got_buf = _bank(seed=90 + layer_idx)
    want_buf = got_buf.clone()
    segment._buf = types.SimpleNamespace(block_residual=got_buf)

    got = segment.forward(hidden.clone())
    want = _ref_old_forward_block_residual(
        segment,
        hidden.clone(),
        segment._graph_attention,
        want_buf,
        write_boundary=True,
    )

    assert "block_residual" not in got, (
        "the residual bank is bucket-owned static state, not a segment output"
    )
    assert got.keys() == want.keys()
    for key in want:
        assert torch.equal(got[key], want[key]), (
            f"captured block-residual output {key!r} diverged from the "
            "pre-change sequence"
        )
    # The boundary side effect on the bucket's static buffer is part of the
    # captured contract: identical column writes, bit for bit.
    assert torch.equal(got_buf, want_buf)
    if segment.is_block_boundary:
        assert torch.equal(
            got_buf[:, segment.num_blocks_before],
            hidden.reshape(-1, _H),
        ), "a boundary layer must write its pre-mix prefix into the bank"


@pytest.mark.parametrize("layer_idx", [1, 2], ids=["mid", "boundary2"])
def test_block_residual_compare_operator_set_replays_the_old_sequence(
    layer_idx,
):
    """The compare path: a PRIVATE torch.cat append, the static bank
    untouched — the eager reference must not advance graph state."""
    layer = _make_layer(layer_idx, fold_ffn=True, block_residual=True)
    segment = _make_segment(layer)
    hidden = _hidden(seed=80 + layer_idx)
    bank = _bank(seed=60 + layer_idx)
    bank_before = bank.clone()

    def compare_append(flat_prefix, active):
        return torch.cat((active, flat_prefix.unsqueeze(1)), dim=1)

    got = run_kimi_block_residual_layer_body(
        layer=layer,
        ops=KimiLayerBodyOps(
            attn=segment._graph_attention,
            ffn=layer.mlp,
            append_boundary=compare_append,
        ),
        hidden_states=hidden.clone(),
        block_residual=bank[:, : segment.num_blocks_before],
    )
    got.pop("block_residual")
    want = _ref_old_forward_block_residual(
        segment,
        hidden.clone(),
        segment._graph_attention,
        bank,
        write_boundary=False,
    )

    assert got.keys() == want.keys()
    for key in want:
        assert torch.equal(got[key], want[key])
    assert torch.equal(bank, bank_before), (
        "compare mode must not mutate the bucket's static residual bank"
    )


@pytest.mark.parametrize(
    "layer_idx", [0, 1, 2], ids=["boundary0", "mid", "boundary2"]
)
def test_block_residual_eager_operator_set_replays_the_old_eager_sequence(
    layer_idx,
):
    """The eager path: in-place prefix merges, the growing boundary append,
    and the probe stages, in the pre-change order."""
    layer = _make_layer(layer_idx, fold_ffn=True, block_residual=True)
    hidden = _hidden(seed=40 + layer_idx)
    nb_before = layer_idx // _BLOCK + 1 - int(layer_idx % _BLOCK == 0)
    bank = _bank(columns=nb_before, seed=50 + layer_idx)

    want_records: list = []
    BlockResidualBuffer.reset()
    want_hidden, want_bank = _ref_old_eager_attn_residual(
        layer, hidden.clone(), bank.clone(), _attn_math, _ffn_math,
        want_records,
    )

    got_records: list = []

    def eager_attn(normed):
        return _attn_math(normed), {}

    def append_boundary(flat_prefix, active):
        snapshot = gather_attn_residual_rows(
            flat_prefix, active, layer.self_attention_res_norm
        )
        return BlockResidualBuffer.append(active, snapshot)

    BlockResidualBuffer.reset()
    got = run_kimi_block_residual_layer_body(
        layer=layer,
        ops=KimiLayerBodyOps(
            attn=eager_attn,
            ffn=_ffn_math,
            append_boundary=append_boundary,
            residual_add_in_place=True,
            record_finite=lambda stage, tensor: got_records.append(stage),
        ),
        hidden_states=hidden.clone(),
        block_residual=bank.clone(),
    )

    assert torch.equal(got["hidden"], want_hidden)
    assert torch.equal(got["block_residual"], want_bank)
    assert got_records == want_records, (
        "the streamed-SP8 probe stages must fire in the pre-change order"
    )


def test_eager_and_graph_operator_sets_agree_through_the_one_body():
    """Same weights, same math, both operator sets, one body: in-place vs
    out-of-place merges and static-write vs cat appends are memory choices,
    never value choices."""
    layer = _make_layer(2, fold_ffn=True, block_residual=True)
    segment = _make_segment(layer)
    hidden = _hidden(seed=99)
    bank = _bank(seed=98)

    segment._buf = types.SimpleNamespace(block_residual=bank.clone())
    graph_out = segment.forward(hidden.clone())

    def eager_attn(normed):
        return _attn_math(normed), {}

    def append_boundary(flat_prefix, active):
        snapshot = gather_attn_residual_rows(
            flat_prefix, active, layer.self_attention_res_norm
        )
        return BlockResidualBuffer.append(active, snapshot)

    BlockResidualBuffer.reset()
    eager_out = run_kimi_block_residual_layer_body(
        layer=layer,
        ops=KimiLayerBodyOps(
            attn=eager_attn,
            ffn=layer.mlp,
            append_boundary=append_boundary,
            residual_add_in_place=True,
        ),
        hidden_states=hidden.clone(),
        block_residual=bank[:, : segment.num_blocks_before].clone(),
    )

    assert torch.equal(graph_out["hidden"], eager_out["hidden"]), (
        "the eager and graph operator sets diverged through the shared body"
    )
