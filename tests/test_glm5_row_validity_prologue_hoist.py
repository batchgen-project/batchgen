"""CPU contract for the hoisted GLM-5 row-validity prologue.

The attention segments used to derive the row-validity mask and its safe
index/seqlen buffers inside every layer, from inputs the whole-model graph
hands all 78 layers unchanged. The values were therefore identical in every
layer, and the block ran 78 times for one step's worth of information.

This file pins the two halves of the hoist:

  * BEHAVIOUR -- the step-level prologue computes exactly what the per-layer
    block computed, including on the rows the two variants disagree about. The
    reference implementations below are the pre-hoist bodies, verbatim, from
    ``cuda_graph_segments.py`` and ``reuse_topk_segment.py``. They are pure
    elementwise torch, so they run on CPU.
  * STRUCTURE -- the per-layer ``forward`` no longer contains the prologue
    ops, the whole-model driver runs the prologue once per distinct buffer set
    before the layer loop, and the two row masks that leave a segment (the
    ``attn_output`` multiply and the ``primary_k_tensor`` select) are untouched.

Both halves are needed: a behavioural test alone would still pass if the block
were left in the layers, and a structural test alone would not catch a mask
that changed meaning. The segment modules cannot be imported on a CPU box
(FA3 / triton), which is why the methods under test are read out of the source
with ``ast`` and executed in a bare namespace -- the same reason
``test_glm5_decode_row_mask_contract.py`` asserts on source.
"""

import ast
import types
from pathlib import Path

import pytest
import torch

from batchgen.models.glm.glm5.whole_model_cuda_graph_segments import (
    Glm5WholeModelSegment,
)


_GLM5 = Path(__file__).parents[1] / "batchgen" / "models" / "glm" / "glm5"
_FULL = "cuda_graph_segments.py"
_REUSE = "reuse_topk_segment.py"

# Field -> how the segments allocate it (see the two setup_static_buffers).
_PROLOGUE_OUTPUTS = (
    "valid_mask",
    "aux_valid_mask",
    "kv_primary_slot_indices",
    "kv_aux_slot_indices",
    "safe_primary_slot_indices",
    "safe_aux_slot_indices",
    "safe_cache_seqlens",
    "valid_rows_bf16",
)


def _tree(name):
    return ast.parse((_GLM5 / name).read_text())


def _class_def(tree, class_name):
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    raise AssertionError(f"class {class_name} not found")


def _method_def(tree, class_name, method_name):
    for node in _class_def(tree, class_name).body:
        if isinstance(node, ast.FunctionDef) and node.name == method_name:
            return node
    raise AssertionError(f"{class_name}.{method_name} not found")


def _load_method(file_name, class_name, method_name):
    """Compile ONE method out of a module that cannot be imported on CPU."""
    node = _method_def(_tree(file_name), class_name, method_name)
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "torch": torch,
        "Optional": __import__("typing").Optional,
        "_Glm5FullDsaSegmentBuffers": object,
    }
    exec(compile(module, f"<{file_name}:{class_name}.{method_name}>", "exec"), namespace)
    return namespace[method_name]


def _op_counts(node, *, production_path=False):
    """Count ``torch.*`` calls, i.e. launches.

    ``production_path`` drops the ``num_valid_tokens is None`` branch: the
    whole-model graph always passes a row count, so that branch is a
    legacy/per-layer fallback and is not part of the per-step launch budget.
    """
    skipped = set()
    if production_path:
        for sub in ast.walk(node):
            if (
                isinstance(sub, ast.If)
                and ast.unparse(sub.test) == "num_valid_tokens is None"
            ):
                for stmt in sub.body:
                    skipped.update(id(inner) for inner in ast.walk(stmt))
    counts = {}
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call) and id(sub) not in skipped:
            name = ast.unparse(sub.func)
            if name.startswith("torch."):
                counts[name] = counts.get(name, 0) + 1
    return counts


def _where_out_targets(node):
    targets = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call) and ast.unparse(sub.func) == "torch.where":
            for kw in sub.keywords:
                if kw.arg == "out":
                    targets.append(ast.unparse(kw.value))
    return targets


def _make_buffers(bucket_size):
    """The prologue's slice of a segment buffer set, allocated as production does."""
    i32 = dict(dtype=torch.int32)
    return types.SimpleNamespace(
        row_indices=torch.arange(bucket_size, **i32),
        valid_rows_ones=torch.ones(bucket_size, dtype=torch.bfloat16),
        valid_rows_zeros=torch.zeros(bucket_size, dtype=torch.bfloat16),
        safe_slot_zeros=torch.zeros(bucket_size, **i32),
        skip_slot_neg_ones=torch.full((bucket_size,), -1, **i32),
        safe_seqlen_zeros=torch.zeros(bucket_size, **i32),
        valid_mask=torch.zeros(bucket_size, dtype=torch.bool),
        aux_valid_mask=torch.zeros(bucket_size, dtype=torch.bool),
        valid_rows_bf16=torch.zeros(bucket_size, dtype=torch.bfloat16),
        kv_primary_slot_indices=torch.zeros(bucket_size, **i32),
        kv_aux_slot_indices=torch.zeros(bucket_size, **i32),
        safe_primary_slot_indices=torch.zeros(bucket_size, **i32),
        safe_aux_slot_indices=torch.zeros(bucket_size, **i32),
        safe_cache_seqlens=torch.zeros(bucket_size, **i32),
    )


# --------------------------------------------------------------------------- #
# Pre-hoist reference bodies, verbatim (main f788c54c:
# cuda_graph_segments.py:1373-1416 and reuse_topk_segment.py:382-403).
# --------------------------------------------------------------------------- #
def _reference_full_prologue(
    buffers,
    *,
    cache_seqlens,
    primary_slot_indices,
    aux_slot_indices,
    num_valid_tokens,
):
    if num_valid_tokens is None:
        torch.ge(primary_slot_indices, 0, out=buffers.valid_mask)
    else:
        torch.lt(buffers.row_indices, num_valid_tokens, out=buffers.valid_mask)
        torch.ge(primary_slot_indices, 0, out=buffers.aux_valid_mask)
        torch.logical_and(buffers.valid_mask, buffers.aux_valid_mask, out=buffers.valid_mask)
    torch.ge(aux_slot_indices, 0, out=buffers.aux_valid_mask)
    torch.logical_and(buffers.valid_mask, buffers.aux_valid_mask, out=buffers.valid_mask)
    torch.where(
        buffers.valid_mask,
        primary_slot_indices,
        buffers.skip_slot_neg_ones,
        out=buffers.kv_primary_slot_indices,
    )
    torch.where(
        buffers.valid_mask,
        aux_slot_indices,
        buffers.skip_slot_neg_ones,
        out=buffers.kv_aux_slot_indices,
    )
    torch.where(
        buffers.valid_mask,
        primary_slot_indices,
        buffers.safe_slot_zeros,
        out=buffers.safe_primary_slot_indices,
    )
    torch.where(
        buffers.valid_mask,
        aux_slot_indices,
        buffers.safe_slot_zeros,
        out=buffers.safe_aux_slot_indices,
    )
    torch.where(
        buffers.valid_mask,
        cache_seqlens,
        buffers.safe_seqlen_zeros,
        out=buffers.safe_cache_seqlens,
    )
    torch.where(
        buffers.valid_mask,
        buffers.valid_rows_ones,
        buffers.valid_rows_zeros,
        out=buffers.valid_rows_bf16,
    )


def _reference_reuse_prologue(
    buffers,
    *,
    cache_seqlens,
    primary_slot_indices,
    aux_slot_indices,
    num_valid_tokens,
):
    if num_valid_tokens is None:
        torch.ge(primary_slot_indices, 0, out=buffers.valid_mask)
    else:
        torch.lt(buffers.row_indices, num_valid_tokens, out=buffers.valid_mask)
        torch.ge(primary_slot_indices, 0, out=buffers.aux_valid_mask)
        torch.logical_and(buffers.valid_mask, buffers.aux_valid_mask, out=buffers.valid_mask)
    torch.where(
        buffers.valid_mask, primary_slot_indices, buffers.skip_slot_neg_ones,
        out=buffers.kv_primary_slot_indices,
    )
    torch.where(
        buffers.valid_mask, primary_slot_indices, buffers.safe_slot_zeros,
        out=buffers.safe_primary_slot_indices,
    )
    torch.where(
        buffers.valid_mask, cache_seqlens, buffers.safe_seqlen_zeros,
        out=buffers.safe_cache_seqlens,
    )
    torch.where(
        buffers.valid_mask, buffers.valid_rows_ones, buffers.valid_rows_zeros,
        out=buffers.valid_rows_bf16,
    )


# ``aux_slot_indices`` differing from ``primary_slot_indices`` is the case the
# two variants must disagree about, so it gets its own row here.
_CASES = {
    "all_valid": ([0, 1, 2, 3], [0, 1, 2, 3], [7, 9, 11, 13], 4),
    "padded_tail": ([0, 1, -1, -1], [0, 1, -1, -1], [7, 9, 0, 0], 2),
    "live_rows_without_a_slot": ([0, -1, 2, -1], [0, -1, 2, -1], [7, 0, 11, 0], 4),
    "aux_missing_only": ([0, 1, 2, 3], [0, -1, 2, -1], [7, 9, 11, 13], 4),
    "primary_missing_only": ([0, -1, 2, 3], [0, 1, 2, 3], [7, 9, 11, 13], 4),
    "no_valid_rows": ([-1, -1, -1, -1], [-1, -1, -1, -1], [0, 0, 0, 0], 0),
    "zero_num_valid_tokens": ([0, 1, 2, 3], [0, 1, 2, 3], [7, 9, 11, 13], 0),
    "capture_dummy_row": ([0, -1, -1, -1], [0, -1, -1, -1], [1, 0, 0, 0], 1),
}


@pytest.mark.parametrize("case", sorted(_CASES))
@pytest.mark.parametrize("with_num_valid_tokens", [True, False])
@pytest.mark.parametrize(
    "variant,file_name,class_name,reference",
    [
        ("full", _FULL, "Glm5FullDsaAttnSegment", _reference_full_prologue),
        ("reuse", _REUSE, "Glm5ReuseTopkAttnSegment", _reference_reuse_prologue),
    ],
)
def test_hoisted_prologue_matches_the_per_layer_block(
    variant, file_name, class_name, reference, case, with_num_valid_tokens
):
    primary, aux, seqlens, nvt = _CASES[case]
    bucket_size = len(primary)
    kwargs = dict(
        cache_seqlens=torch.tensor(seqlens, dtype=torch.int32),
        primary_slot_indices=torch.tensor(primary, dtype=torch.int32),
        aux_slot_indices=torch.tensor(aux, dtype=torch.int32),
        num_valid_tokens=(
            torch.tensor([nvt], dtype=torch.int32) if with_num_valid_tokens else None
        ),
    )

    compute = _load_method(file_name, class_name, "compute_row_validity")
    got = _make_buffers(bucket_size)
    compute(object(), got, **kwargs)

    expected = _make_buffers(bucket_size)
    reference(expected, **kwargs)

    for field in _PROLOGUE_OUTPUTS:
        assert torch.equal(getattr(got, field), getattr(expected, field)), (
            f"{variant}/{case}/nvt={with_num_valid_tokens}: {field} changed "
            "meaning; the hoist must be a launch-count change only"
        )


def test_the_two_variants_really_do_disagree_on_the_aux_slot():
    """Guard the guard: one shared mask would be a semantic change.

    If this ever stops holding, the variant split in the hoisted prologue can
    be collapsed -- until then, a full and a skip layer must not share one
    computed mask, which is what the driver's collision check enforces.
    """
    kwargs = dict(
        cache_seqlens=torch.tensor([7, 9], dtype=torch.int32),
        primary_slot_indices=torch.tensor([0, 1], dtype=torch.int32),
        aux_slot_indices=torch.tensor([0, -1], dtype=torch.int32),
        num_valid_tokens=torch.tensor([2], dtype=torch.int32),
    )
    full = _make_buffers(2)
    reuse = _make_buffers(2)
    _reference_full_prologue(full, **kwargs)
    _reference_reuse_prologue(reuse, **kwargs)

    assert full.valid_mask.tolist() == [True, False]
    assert reuse.valid_mask.tolist() == [True, True]


# --------------------------------------------------------------------------- #
# Structure: the ops left the per-layer path and the invariants stayed.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "file_name,class_name,expected,total",
    [
        # 11 launches for an indexer layer, 7 for a skip layer -- now once per
        # step each (11 + 7 = 18) instead of 21x11 + 57x7 = 630.
        (_FULL, "Glm5FullDsaAttnSegment",
         {"torch.lt": 1, "torch.ge": 2, "torch.logical_and": 2, "torch.where": 6}, 11),
        (_REUSE, "Glm5ReuseTopkAttnSegment",
         {"torch.lt": 1, "torch.ge": 1, "torch.logical_and": 1, "torch.where": 4}, 7),
    ],
)
def test_prologue_launch_count_per_variant(file_name, class_name, expected, total):
    node = _method_def(_tree(file_name), class_name, "compute_row_validity")
    counts = _op_counts(node, production_path=True)
    assert counts == expected, (
        f"{class_name}.compute_row_validity op count changed; the step-level "
        "launch budget and the expected per-step delta are derived from it"
    )
    assert sum(counts.values()) == total


@pytest.mark.parametrize(
    "file_name,class_name",
    [(_FULL, "Glm5FullDsaAttnSegment"), (_REUSE, "Glm5ReuseTopkAttnSegment")],
)
def test_forward_no_longer_recomputes_row_validity(file_name, class_name):
    node = _method_def(_tree(file_name), class_name, "forward")
    counts = _op_counts(node)
    for dead in ("torch.lt", "torch.ge", "torch.logical_and"):
        assert dead not in counts, (
            f"{class_name}.forward still calls {dead}; the row-validity mask "
            "is step-level and must not be rebuilt per layer"
        )
    # No prologue `where` may leak back into the layer path. The
    # primary_k_tensor select used to be the one legitimate exception here;
    # it has since been folded into the epilogue of the kernel that produces
    # the value (same select-not-multiply semantics, one launch fewer), so the
    # layer path must now contain NO torch.where at all. The fold itself is
    # pinned by tests/test_glm5_fold_elementwise_into_neighbours.py.
    assert _where_out_targets(node) == [], (
        f"{class_name}.forward must contain no torch.where: the prologue is "
        f"hoisted and the primary_k_tensor select lives in its producer, got "
        f"{_where_out_targets(node)}"
    )
    assert any(
        isinstance(sub, ast.Call)
        and isinstance(sub.func, ast.Attribute)
        and sub.func.attr == "compute_row_validity"
        for sub in ast.walk(node)
    ), f"{class_name}.forward must still be able to compute the prologue itself"


@pytest.mark.parametrize(
    "file_name,class_name",
    [(_FULL, "Glm5FullDsaAttnSegment"), (_REUSE, "Glm5ReuseTopkAttnSegment")],
)
def test_the_load_bearing_attn_output_mask_survived_the_hoist(file_name, class_name):
    node = _method_def(_tree(file_name), class_name, "forward")
    masked = [
        ast.unparse(sub.func.value)
        for sub in ast.walk(node)
        if isinstance(sub, ast.Call)
        and isinstance(sub.func, ast.Attribute)
        and sub.func.attr == "mul_"
        and "valid_rows_bf16" in ast.unparse(sub)
    ]
    assert masked == ["outputs.attn_output"], (
        f"{class_name}: invalid rows must still be inert before the MoE "
        f"router mixes rows, got {masked}"
    )


def test_driver_runs_the_prologue_before_the_layer_loop():
    node = _method_def(
        ast.parse((_GLM5 / "whole_model_cuda_graph_segments.py").read_text()),
        "Glm5WholeModelSegment",
        "run_model_with_probes",
    )
    body = ast.unparse(node)
    assert "_run_row_validity_prologues" in body
    assert "row_validity_ready=True" in body
    loop_pos = body.index("for layer_idx, layer_segment in enumerate(self.layer_segments)")
    assert body.index("_run_row_validity_prologues") < loop_pos, (
        "the prologue must be recorded into the graph before the first layer "
        "reads the buffers it writes"
    )


# --------------------------------------------------------------------------- #
# Driver: once per distinct buffer set, and loud about the unsafe shapes.
# --------------------------------------------------------------------------- #
class _FakeIndexer:
    index_head_dim = 128
    index_topk = 2048


class _FakeLayer:
    self_attn = types.SimpleNamespace(
        module=types.SimpleNamespace(
            kv_lora_rank=512, qk_rope_head_dim=64, indexer=_FakeIndexer()
        )
    )


class _FakeLayerSegment:
    def __init__(self, variant, buffers_id):
        self._key = (variant, buffers_id)
        self.runs = []

    def row_validity_prologue_key(self, bucket_size):
        return self._key

    def run_row_validity_prologue(self, bucket_size, **kwargs):
        self.runs.append((bucket_size, kwargs))


def _driver(layer_segments):
    return Glm5WholeModelSegment(
        model=types.SimpleNamespace(
            model=types.SimpleNamespace(layers=[_FakeLayer() for _ in layer_segments])
        ),
        device=torch.device("cpu"),
        world_size=1,
        max_pages_per_seq=8,
        max_aux_pages_per_seq=8,
        vocab_size=16,
        hidden_size=8,
        max_bucket_size=4,
        max_seqlen=64,
        layer_segments=layer_segments,
    )


def _prologue_kwargs(bucket_size=4):
    return dict(
        cache_seqlens=torch.zeros(bucket_size, dtype=torch.int32),
        primary_slot_indices=torch.zeros(bucket_size, dtype=torch.int32),
        aux_slot_indices=torch.zeros(bucket_size, dtype=torch.int32),
        num_valid_tokens=torch.tensor([bucket_size], dtype=torch.int32),
    )


def test_prologue_runs_once_per_shared_buffer_set_not_once_per_layer():
    # 3 indexer layers sharing one buffer set + 5 skip layers sharing another:
    # the production shape, scaled down from 21 + 57.
    full = [_FakeLayerSegment("full", 1) for _ in range(3)]
    reuse = [_FakeLayerSegment("reuse", 2) for _ in range(5)]
    layer_segments = [full[0], reuse[0], reuse[1], full[1], reuse[2],
                      full[2], reuse[3], reuse[4]]
    driver = _driver(layer_segments)

    ran = driver._run_row_validity_prologues(4, **_prologue_kwargs())

    assert ran == (("full", 1), ("reuse", 2))
    assert [len(ls.runs) for ls in layer_segments] == [1, 1, 0, 0, 0, 0, 0, 0]
    assert sum(len(ls.runs) for ls in layer_segments) == 2


def test_prologue_rejects_one_buffer_set_claimed_by_both_variants():
    driver = _driver([_FakeLayerSegment("full", 1), _FakeLayerSegment("reuse", 1)])
    with pytest.raises(RuntimeError, match="variants' masks differ on the aux slot"):
        driver._run_row_validity_prologues(4, **_prologue_kwargs())


def test_prologue_rejects_a_layer_segment_without_the_hooks():
    driver = _driver([_FakeLayerSegment("full", 1), object()])
    with pytest.raises(RuntimeError, match="row-validity prologue hooks|row_validity_prologue_key"):
        driver._run_row_validity_prologues(4, **_prologue_kwargs())
