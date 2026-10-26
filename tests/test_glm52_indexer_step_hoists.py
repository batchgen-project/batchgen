"""CPU contract for the three GLM-5.2 decode indexer step-level hoists.

GLM-5.2 runs a DSA indexer on 21 of 78 decoder layers; the other 57 reuse the
most recent selection. Three pieces of work inside the whole-model graph were
being re-recorded per layer although they are step-level:

  R1  the paged-MQA scorer's input preparation (one Triton launch) and
      DeepGEMM's ``get_paged_mqa_logits_metadata`` -- 21x each per step, from
      step-level inputs, into buffers all 21 indexer layers share.
  R2  the selected-page-table transform -- 78x per step, of which the 57
      reuse layers transform the SAME borrowed selection within each run.
  R5  the int32 narrowing of ``position_ids`` (78x) and the per-index-head
      position broadcast (21x), both pure functions of one graph input.

This file pins all three the way
``test_glm5_row_validity_prologue_hoist.py`` pins the row-validity hoist:
BEHAVIOUR (the hoisted code computes what the per-layer code computed, and
the driver's dedupe really collapses the launches) and STRUCTURE (the ops
left the per-layer path, and each hoisted buffer is written before its first
consumer). The segment modules cannot be imported on a CPU box (FA3 /
triton), so the methods under test are read out of the source with ``ast``
and executed in a bare namespace.
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
_KERNELS = Path(__file__).parents[1] / "batchgen_kernels" / "attention" / "dsa"
_FULL = "cuda_graph_segments.py"
_REUSE = "reuse_topk_segment.py"
_DRIVER = "whole_model_cuda_graph_segments.py"
_LAYER = "layer_cuda_graph_segments.py"


def _tree(name, root=_GLM5):
    return ast.parse((root / name).read_text())


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


def _top_level(tree, name):
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name:
            return node
    raise AssertionError(f"{name} not found")


def _load_method(file_name, class_name, method_name, root=_GLM5):
    """Compile ONE method out of a module that cannot be imported on CPU."""
    node = _method_def(_tree(file_name, root), class_name, method_name)
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "torch": torch,
        "Optional": __import__("typing").Optional,
        "_Glm5FullDsaSegmentBuffers": object,
    }
    exec(compile(module, f"<{file_name}:{class_name}.{method_name}>", "exec"), namespace)
    return namespace[method_name]


def _call_names(node):
    return [
        ast.unparse(sub.func)
        for sub in ast.walk(node)
        if isinstance(sub, ast.Call)
    ]


def _attr_calls(node, attr):
    """Every ``<something>.<attr>(...)`` call, as the receiver's source."""
    return [
        ast.unparse(sub.func.value)
        for sub in ast.walk(node)
        if isinstance(sub, ast.Call)
        and isinstance(sub.func, ast.Attribute)
        and sub.func.attr == attr
    ]


def _stmt_index(fn, predicate):
    """Index of the first top-level statement whose source matches."""
    for i, stmt in enumerate(fn.body):
        if predicate(ast.unparse(stmt)):
            return i
    return None


# --------------------------------------------------------------------------- #
# Driver scaffolding (same shape as the row-validity hoist file's).
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
    """A layer segment with just the hoist hooks the driver calls.

    ``prep_key=None`` is a segment that never scores (a skip layer, a bf16
    AoS aux cache, an ``all_short`` graph). ``selected_key=None`` is a layer
    that selects its own top-k.
    """

    def __init__(self, *, selects, prep_key=None, selected_key=None):
        self.selects_top_k = selects
        self._prep_key = prep_key
        self._selected_key = selected_key
        self.preps = []

    def paged_mqa_prep_key(self, bucket_size):
        return self._prep_key

    def run_paged_mqa_prep(self, bucket_size, **kwargs):
        self.preps.append((bucket_size, kwargs))

    def selected_positions_key(self, bucket_size):
        return self._selected_key


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


# 21 indexer layers, 57 skip layers, in runs of deliberately uneven length.
# One run is EMPTY, i.e. two indexer layers are adjacent -- that must not be
# mistaken for a shareable transform.
_UNEVEN_RUNS = (1, 2, 3, 4, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2)
# Same totals with every indexer layer followed by at least one skip layer.
_EVEN_RUNS = (3,) * 15 + (2,) * 6


def _glm52_shape(
    run_lengths=_UNEVEN_RUNS,
    full_key=("bt", "ctx", "dg"),
    reuse_key=("page", "topk", "reuse"),
):
    """A GLM-5.2-shaped layer list: 21 indexer layers, 57 skip layers."""
    assert len(run_lengths) == 21 and sum(run_lengths) == 57
    segments = []
    for length in run_lengths:
        segments.append(_FakeLayerSegment(selects=True, prep_key=full_key))
        segments.extend(
            _FakeLayerSegment(selects=False, selected_key=reuse_key)
            for _ in range(length)
        )
    assert len(segments) == 78
    return segments


# --------------------------------------------------------------------------- #
# R1: the paged-MQA preparation and the DeepGEMM schedule, once per step.
# --------------------------------------------------------------------------- #
def test_paged_mqa_prep_runs_once_per_step_not_once_per_indexer_layer():
    segments = _glm52_shape()
    driver = _driver(segments)

    num_valid_tokens = torch.tensor([4], dtype=torch.int32)
    ran = driver._run_paged_mqa_preps(4, num_valid_tokens=num_valid_tokens)

    assert ran == (("bt", "ctx", "dg"),)
    assert sum(len(s.preps) for s in segments) == 1, (
        "the preparation launch and the DeepGEMM metadata launch are "
        "step-level; 21 indexer layers must not re-record them"
    )
    # The one that ran is the FIRST indexer layer, so the inputs are ready
    # before any layer scores.
    assert len(segments[0].preps) == 1
    (bucket, kwargs), = segments[0].preps
    assert bucket == 4
    assert kwargs == {"num_valid_tokens": num_valid_tokens}


def test_paged_mqa_prep_skips_every_segment_that_never_scores():
    """A skip layer, a bf16 AoS aux cache and an all_short graph all report
    ``None``: there is no preparation to run, not a shared one to dedupe."""
    segments = [
        _FakeLayerSegment(selects=True, prep_key=None),
        _FakeLayerSegment(selects=False, selected_key=("page",)),
        _FakeLayerSegment(selects=True, prep_key=None),
    ]
    driver = _driver(segments)

    assert driver._run_paged_mqa_preps(4, num_valid_tokens=None) == ()
    assert [len(s.preps) for s in segments] == [0, 0, 0]


def test_paged_mqa_prep_runs_once_per_distinct_buffer_set():
    """Dedupe is by the OUTPUT buffers' identity, not by layer or by variant:
    two groups of indexer layers with their own buffer sets need one each."""
    group_a = [_FakeLayerSegment(selects=True, prep_key=("a",)) for _ in range(3)]
    group_b = [_FakeLayerSegment(selects=True, prep_key=("b",)) for _ in range(2)]
    segments = [group_a[0], group_b[0], group_a[1], group_a[2], group_b[1]]
    driver = _driver(segments)

    assert driver._run_paged_mqa_preps(4, num_valid_tokens=None) == (("a",), ("b",))
    assert [len(s.preps) for s in segments] == [1, 1, 0, 0, 0]


def test_paged_mqa_prep_rejects_a_layer_segment_without_the_hooks():
    driver = _driver([_FakeLayerSegment(selects=True, prep_key=("a",)), object()])
    with pytest.raises(RuntimeError, match="paged_mqa_prep_key"):
        driver._run_paged_mqa_preps(4, num_valid_tokens=None)


def test_prep_key_is_none_exactly_when_the_segment_cannot_score():
    """The segment's own discriminator, executed: only a page-split aux cache
    on a non-all_short graph has a preparation to share."""
    key_fn = _load_method(_FULL, "Glm5FullDsaAttnSegment", "paged_mqa_prep_key")
    buffers = types.SimpleNamespace(
        aux_block_table=torch.zeros(1),
        aux_context_lens=torch.zeros(1),
        aux_deep_gemm_context_lens=torch.zeros(1),
    )

    def _self(page_split, all_short):
        return types.SimpleNamespace(
            aux_is_page_split=page_split,
            all_short=all_short,
            _row_validity_buffers=lambda bucket_size: buffers,
        )

    assert key_fn(_self(False, False), 4) is None, "bf16 AoS scorer needs no prep"
    assert key_fn(_self(True, True), 4) is None, "an all_short graph never scores"
    assert key_fn(_self(False, True), 4) is None
    key = key_fn(_self(True, False), 4)
    assert key == (
        id(buffers.aux_block_table),
        id(buffers.aux_context_lens),
        id(buffers.aux_deep_gemm_context_lens),
    )


def test_a_skip_layer_has_no_paged_mqa_prep_at_all():
    key_fn = _load_method(_REUSE, "Glm5ReuseTopkAttnSegment", "paged_mqa_prep_key")
    assert key_fn(object(), 4) is None
    # And running it anyway is a named refusal, not an inherited deref of an
    # aux page table this segment has no field for.
    run_fn = _load_method(_REUSE, "Glm5ReuseTopkAttnSegment", "run_paged_mqa_prep")
    with pytest.raises(RuntimeError, match="never scores"):
        run_fn(types.SimpleNamespace(layer_idx=7), 4, num_valid_tokens=None)


def test_the_prep_is_launched_from_exactly_one_place_in_the_segments():
    """``paged_mqa_step_prologue_out`` is the step-level entry point, so the
    only caller inside a segment must be the step-level hook."""
    tree = _tree(_FULL)
    callers = [
        (node.name, method.name)
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        for method in node.body
        if isinstance(method, ast.FunctionDef)
        and "paged_mqa_step_prologue_out" in _call_names(method)
    ]
    assert callers == [("Glm5FullDsaAttnSegment", "run_paged_mqa_prep")]


def test_the_per_layer_scorer_half_is_used_only_with_a_ready_prep():
    """``_score_and_topk`` must take the prepared path exactly under
    ``paged_mqa_prep_ready``, and fail loudly rather than score with a
    schedule the driver never built."""
    fn = _method_def(_tree(_FULL), "Glm5FullDsaAttnSegment", "_score_and_topk")
    assert "paged_mqa_prep_ready" in {a.arg for a in fn.args.kwonlyargs}
    guarded = [
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "paged_mqa_prep_ready"
    ]
    assert len(guarded) == 1
    body = ast.unparse(guarded[0])
    assert "paged_mqa_score_and_topk_prepared_out" in body
    assert "raise RuntimeError" in body
    assert "aux_schedule_meta is None" in body
    # The self-contained whole-layer form survives only OUTSIDE that branch,
    # for the per-layer graph path that has nothing to share.
    assert "paged_mqa_score_and_topk_out" not in body
    assert "paged_mqa_score_and_topk_out" in ast.unparse(fn)


def test_forward_hands_the_prep_flag_through_to_the_scorer():
    fwd = _method_def(_tree(_FULL), "Glm5FullDsaAttnSegment", "forward")
    assert "paged_mqa_prep_ready" in {a.arg for a in fwd.args.kwonlyargs}
    calls = [
        sub
        for sub in ast.walk(fwd)
        if isinstance(sub, ast.Call) and ast.unparse(sub.func) == "self._score_and_topk"
    ]
    assert len(calls) == 1
    passed = {
        kw.arg: ast.unparse(kw.value) for kw in calls[0].keywords
    }
    assert passed["paged_mqa_prep_ready"] == "paged_mqa_prep_ready"


# --------------------------------------------------------------------------- #
# R2: the selected-page-table transform, once per reuse run.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "run_lengths,expected_reuse_launches",
    [
        # Every indexer layer followed by skip layers: 21 runs, so the 57
        # skip-layer transforms collapse to 21 and the step records 42.
        (_EVEN_RUNS, 21),
        # One empty run (two adjacent indexer layers): 20 non-empty runs.
        (_UNEVEN_RUNS, 20),
    ],
)
def test_transform_runs_once_per_reuse_run_not_once_per_layer(
    run_lengths, expected_reuse_launches
):
    segments = _glm52_shape(run_lengths)
    driver = _driver(segments)

    flags = driver._selected_positions_ready(4)

    assert len(flags) == 78
    reuse_launches = sum(
        1
        for segment, ready in zip(segments, flags)
        if not segment.selects_top_k and not ready
    )
    assert reuse_launches == expected_reuse_launches, flags
    # One launch per non-empty reuse run, down from one per skip layer.
    assert reuse_launches == sum(1 for length in run_lengths if length)
    # Every indexer layer still transforms, so the step records 21 + runs.
    assert all(
        not ready for segment, ready in zip(segments, flags) if segment.selects_top_k
    ), "a selecting layer must never skip its own transform"
    assert flags.count(False) == 21 + expected_reuse_launches
    assert flags.count(False) < 78


def test_two_adjacent_indexer_layers_both_transform():
    """The trap in sharing by buffer identity alone: all indexer layers share
    ONE top-k buffer, so two adjacent selecting layers have identical buffer
    identities while the CONTENT changed between them."""
    segments = [
        _FakeLayerSegment(selects=True, prep_key=("a",)),
        _FakeLayerSegment(selects=True, prep_key=("a",)),
        _FakeLayerSegment(selects=False, selected_key=("k",)),
    ]
    driver = _driver(segments)

    assert driver._selected_positions_ready(4) == (False, False, False)


def test_a_reuse_layer_after_a_selecting_layer_always_transforms():
    segments = [
        _FakeLayerSegment(selects=True, prep_key=("a",)),
        _FakeLayerSegment(selects=False, selected_key=("k",)),
        _FakeLayerSegment(selects=False, selected_key=("k",)),
        _FakeLayerSegment(selects=True, prep_key=("a",)),
        _FakeLayerSegment(selects=False, selected_key=("k",)),
        _FakeLayerSegment(selects=False, selected_key=("k",)),
    ]
    driver = _driver(segments)

    assert driver._selected_positions_ready(4) == (
        False, False, True, False, False, True,
    )


def test_a_changed_transform_input_re_runs_the_transform():
    """The key is checked, not just the run structure: a reuse layer that
    would read or write a different tensor is not a repetition."""
    segments = [
        _FakeLayerSegment(selects=True, prep_key=("a",)),
        _FakeLayerSegment(selects=False, selected_key=("k", "set-1")),
        _FakeLayerSegment(selects=False, selected_key=("k", "set-2")),
        _FakeLayerSegment(selects=False, selected_key=("k", "set-2")),
    ]
    driver = _driver(segments)

    assert driver._selected_positions_ready(4) == (False, False, False, True)


def test_selected_positions_rejects_a_layer_segment_without_the_hook():
    driver = _driver([_FakeLayerSegment(selects=True), object()])
    with pytest.raises(RuntimeError, match="selects_top_k"):
        driver._selected_positions_ready(4)


def test_the_two_variants_declare_opposite_top_k_ownership():
    """The run boundary is derived from this flag, so it must stay opposite:
    an indexer layer writes the shared top-k, a skip layer only reads it."""
    full = _class_def(_tree(_FULL), "Glm5FullDsaAttnSegment")
    reuse = _class_def(_tree(_REUSE), "Glm5ReuseTopkAttnSegment")

    def _flag(cls):
        for stmt in cls.body:
            if (
                isinstance(stmt, ast.Assign)
                and [ast.unparse(t) for t in stmt.targets] == ["SELECTS_TOP_K"]
            ):
                return ast.literal_eval(stmt.value)
        raise AssertionError(f"{cls.name} does not declare SELECTS_TOP_K")

    assert _flag(full) is True
    assert _flag(reuse) is False


def test_the_reuse_transform_is_guarded_and_the_indexer_one_is_not():
    reuse_fwd = _method_def(_tree(_REUSE), "Glm5ReuseTopkAttnSegment", "forward")
    guards = [
        node
        for node in ast.walk(reuse_fwd)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "not selected_positions_ready"
    ]
    assert len(guards) == 1
    assert "transform_selected_positions_out" in _call_names(guards[0])
    assert (
        len(
            [
                name
                for name in _call_names(reuse_fwd)
                if name == "transform_selected_positions_out"
            ]
        )
        == 1
    ), "exactly one transform call site, and it is the guarded one"

    full_fwd = _method_def(_tree(_FULL), "Glm5FullDsaAttnSegment", "forward")
    assert not [
        node
        for node in ast.walk(full_fwd)
        if isinstance(node, ast.If)
        and "selected_positions_ready" in ast.unparse(node.test)
        and "transform_selected_positions_out" in _call_names(node)
    ], "an indexer layer's transform must not be skippable"
    # Instead it refuses the flag outright, before any launch.
    raises = [
        node
        for node in ast.walk(full_fwd)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "selected_positions_ready"
    ]
    assert len(raises) == 1
    assert "raise ValueError" in ast.unparse(raises[0])
    assert full_fwd.body.index(raises[0]) == 0, (
        "the refusal must come before the segment records anything"
    )


def test_the_reuse_transform_reads_only_step_level_and_borrowed_inputs():
    """Why the hoist is sound: within a run, every input is either a
    step-level static or the borrowed top-k no layer in the run writes."""
    reuse_fwd = _method_def(_tree(_REUSE), "Glm5ReuseTopkAttnSegment", "forward")
    (call,) = [
        sub
        for sub in ast.walk(reuse_fwd)
        if isinstance(sub, ast.Call)
        and ast.unparse(sub.func) == "transform_selected_positions_out"
    ]
    assert [ast.unparse(a) for a in call.args] == [
        "self.primary_page_table",
        "buffers.safe_cache_seqlens",
        "buffers.top_k_indices",
        "buffers.selected_token_ids",
        "buffers.selected_lengths",
    ]
    kwargs = {kw.arg: ast.unparse(kw.value) for kw in call.keywords}
    assert kwargs["primary_slot_indices"] == "buffers.safe_primary_slot_indices"
    # And the key the driver dedupes on covers every one of them.
    key_src = ast.unparse(
        _method_def(_tree(_REUSE), "Glm5ReuseTopkAttnSegment", "selected_positions_key")
    )
    for covered in (
        "self.primary_page_table",
        "buffers.top_k_indices",
        "buffers.safe_cache_seqlens",
        "buffers.safe_primary_slot_indices",
        "buffers.selected_token_ids",
        "buffers.selected_lengths",
    ):
        assert f"id({covered})" in key_src, covered


# --------------------------------------------------------------------------- #
# R5: the step's derived position tensors.
# --------------------------------------------------------------------------- #
_INDEX_HEADS = 32


def _step_buffers(bucket_size, *, index_heads=_INDEX_HEADS):
    return types.SimpleNamespace(
        token_indices=torch.zeros(bucket_size, dtype=torch.int32),
        positions_expanded=torch.zeros(
            bucket_size, index_heads, dtype=torch.int64
        ),
    )


@pytest.mark.parametrize("bucket_size", [1, 4, 32])
def test_hoisted_step_statics_match_the_per_layer_expressions(bucket_size):
    """The pre-hoist per-layer bodies, verbatim: ``token_indices =
    position_ids.view(batch).to(dtype=torch.int32)`` and
    ``positions_expanded.copy_(position_ids.view(batch, 1).expand(batch,
    index_n_heads))``."""
    position_ids = torch.arange(bucket_size, dtype=torch.int64).view(bucket_size, 1) * 7

    compute = _load_method(_FULL, "Glm5FullDsaAttnSegment", "compute_step_statics")
    got = _step_buffers(bucket_size)
    compute(types.SimpleNamespace(all_short=False), got, position_ids=position_ids)

    expected_tokens = position_ids.view(bucket_size).to(dtype=torch.int32)
    expected_positions = torch.zeros(bucket_size, _INDEX_HEADS, dtype=torch.int64)
    expected_positions.copy_(
        position_ids.view(bucket_size, 1).expand(bucket_size, _INDEX_HEADS)
    )

    assert torch.equal(got.token_indices, expected_tokens)
    assert got.token_indices.dtype == torch.int32
    assert torch.equal(got.positions_expanded, expected_positions)


def test_an_all_short_graph_leaves_the_indexer_position_broadcast_alone():
    """``positions_expanded`` feeds the indexer Q RoPE, which an all_short
    graph does not run -- writing it would ADD a launch there."""
    compute = _load_method(_FULL, "Glm5FullDsaAttnSegment", "compute_step_statics")
    got = _step_buffers(4)
    compute(
        types.SimpleNamespace(all_short=True),
        got,
        position_ids=torch.arange(4, dtype=torch.int64).view(4, 1),
    )
    assert torch.equal(got.token_indices, torch.arange(4, dtype=torch.int32))
    assert torch.equal(got.positions_expanded, torch.zeros(4, _INDEX_HEADS, dtype=torch.int64))


def test_a_skip_layer_narrows_positions_but_never_writes_the_placeholder():
    """The skip variant's ``positions_expanded`` is a 1-element placeholder,
    so its step statics must cover ``token_indices`` only."""
    compute = _load_method(_REUSE, "Glm5ReuseTopkAttnSegment", "compute_step_statics")
    got = types.SimpleNamespace(
        token_indices=torch.zeros(4, dtype=torch.int32),
        positions_expanded=torch.ones(1, dtype=torch.int32),
    )
    compute(
        types.SimpleNamespace(all_short=False),
        got,
        position_ids=(torch.arange(4, dtype=torch.int64) * 3).view(4, 1),
    )
    assert got.token_indices.tolist() == [0, 3, 6, 9]
    assert got.positions_expanded.tolist() == [1], (
        "the skip variant must not write the indexer-only placeholder"
    )


@pytest.mark.parametrize(
    "file_name,class_name,copies",
    [
        (_FULL, "Glm5FullDsaAttnSegment", ["buffers.token_indices", "buffers.positions_expanded"]),
        (_REUSE, "Glm5ReuseTopkAttnSegment", ["buffers.token_indices"]),
    ],
)
def test_step_statics_launch_budget_per_variant(file_name, class_name, copies):
    """One launch per derived tensor, and no extra temporary: a cross-dtype
    ``copy_`` narrows inside the copy kernel, so a ``.to(...)`` feeding the
    copy would be a second launch per step."""
    node = _method_def(_tree(file_name), class_name, "compute_step_statics")
    assert _attr_calls(node, "copy_") == copies
    assert "to(dtype=torch.int32)" not in ast.unparse(node)
    assert not [
        name for name in _call_names(node) if name.startswith("torch.")
    ], (
        "every op here writes a caller-owned buffer; a torch.* call would be "
        "an allocation"
    )


@pytest.mark.parametrize(
    "file_name,class_name",
    [(_FULL, "Glm5FullDsaAttnSegment"), (_REUSE, "Glm5ReuseTopkAttnSegment")],
)
def test_forward_no_longer_derives_the_step_positions(file_name, class_name):
    node = _method_def(_tree(file_name), class_name, "forward")
    src = ast.unparse(node)
    assert "to(dtype=torch.int32)" not in src, (
        f"{class_name}.forward still narrows position_ids per layer"
    )
    assert "positions_expanded" not in _attr_calls(node, "copy_") and not [
        receiver
        for receiver in _attr_calls(node, "copy_")
        if "positions_expanded" in receiver
    ], f"{class_name}.forward still broadcasts position_ids per layer"
    assert "expand(" not in src


# --------------------------------------------------------------------------- #
# Ordering: each hoisted buffer is written before its first consumer.
# --------------------------------------------------------------------------- #
def _call_lineno(fn, callee):
    (call,) = [
        sub
        for sub in ast.walk(fn)
        if isinstance(sub, ast.Call) and ast.unparse(sub.func) == callee
    ]
    return call.lineno


def test_the_driver_runs_both_prologues_before_the_layer_loop_in_order():
    fn = _method_def(_tree(_DRIVER), "Glm5WholeModelSegment", "run_model_with_probes")
    validity = _call_lineno(fn, "self._run_row_validity_prologues")
    paged = _call_lineno(fn, "self._run_paged_mqa_preps")
    ready = _call_lineno(fn, "self._selected_positions_ready")
    loops = [
        node.lineno
        for node in ast.walk(fn)
        if isinstance(node, ast.For)
        and ast.unparse(node.target) == "(layer_idx, layer_segment)"
        and "self.layer_segments" in ast.unparse(node.iter)
    ]
    assert loops, "the layer loop moved; this ordering pin must follow it"
    loop = min(loops)
    # The paged-MQA preparation READS the validity prologue's safe slot and
    # seqlen buffers, so the order between the two is load-bearing.
    assert validity < paged < loop
    assert ready < loop


def test_the_driver_tells_every_layer_the_step_work_is_done():
    fn = _method_def(_tree(_DRIVER), "Glm5WholeModelSegment", "run_model_with_probes")
    (call,) = [
        sub
        for sub in ast.walk(fn)
        if isinstance(sub, ast.Call) and ast.unparse(sub.func) == "layer_segment.forward"
    ]
    passed = {kw.arg: ast.unparse(kw.value) for kw in call.keywords}
    assert passed["row_validity_ready"] == "True"
    assert passed["paged_mqa_prep_ready"] == "True"
    assert passed["selected_positions_ready"] == "selected_positions_ready[layer_idx]"


def test_the_step_prologue_computes_validity_before_the_positions():
    """One hook, two blocks, in the order the second block's buffers need:
    nothing in ``compute_step_statics`` may read a validity static that has
    not been written yet."""
    for file_name, class_name in (
        (_FULL, "Glm5FullDsaAttnSegment"),
    ):
        fn = _method_def(_tree(file_name), class_name, "run_row_validity_prologue")
        names = _call_names(fn)
        assert "self.compute_row_validity" in names
        assert "self.compute_step_statics" in names
        assert names.index("self.compute_row_validity") < names.index(
            "self.compute_step_statics"
        )
        assert "position_ids" in {a.arg for a in fn.args.kwonlyargs}


def test_the_paged_prep_reads_the_validity_prologues_outputs():
    fn = _method_def(_tree(_FULL), "Glm5FullDsaAttnSegment", "run_paged_mqa_prep")
    (call,) = [
        sub
        for sub in ast.walk(fn)
        if isinstance(sub, ast.Call)
        and ast.unparse(sub.func) == "paged_mqa_step_prologue_out"
    ]
    passed = {kw.arg: ast.unparse(kw.value) for kw in call.keywords}
    # Exactly the safe (row-validity-masked) forms the per-layer call used.
    assert passed["aux_slot_indices"] == "buffers.safe_aux_slot_indices"
    assert passed["cache_seqlens"] == "buffers.safe_cache_seqlens"
    assert passed["aux_page_table"] == "self.aux_page_table"
    assert passed["block_table"] == "buffers.aux_block_table"
    assert passed["context_lens"] == "buffers.aux_context_lens"
    assert passed["deep_gemm_context_lens"] == "buffers.aux_deep_gemm_context_lens"
    # The DeepGEMM-owned schedule is kept on the SHARED buffer set, which is
    # both how every indexer layer finds it and how the reference stays alive
    # for the whole layer loop.
    assert "buffers.aux_schedule_meta" in [
        ast.unparse(target)
        for stmt in fn.body
        if isinstance(stmt, ast.Assign)
        for target in stmt.targets
    ]


@pytest.mark.parametrize(
    "file_name,class_name",
    [(_FULL, "Glm5FullDsaAttnSegment"), (_REUSE, "Glm5ReuseTopkAttnSegment")],
)
def test_a_layer_can_still_compute_the_whole_step_prologue_itself(file_name, class_name):
    """The per-layer graph path has no driver, so a forward that was not told
    the prologue ran must do all of it -- and must do it before the first
    consumer of either block."""
    fn = _method_def(_tree(file_name), class_name, "forward")
    fallback = [
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.If) and ast.unparse(node.test) == "not row_validity_ready"
    ]
    assert len(fallback) == 1
    names = _call_names(fallback[0])
    assert "self.compute_row_validity" in names
    assert "self.compute_step_statics" in names
    # ...and nothing reads a hoisted buffer before that block.
    fallback_at = _stmt_index(fn, lambda s: s.startswith("if not row_validity_ready:"))
    assert fallback_at is not None
    for i, stmt in enumerate(fn.body[:fallback_at]):
        src = ast.unparse(stmt)
        for hoisted in (
            "buffers.token_indices",
            "buffers.positions_expanded",
            "buffers.aux_schedule_meta",
            "buffers.safe_cache_seqlens",
        ):
            assert hoisted not in src, (
                f"{class_name}.forward reads {hoisted} at statement {i}, "
                "before the step prologue that writes it"
            )


def test_the_layer_segment_forwards_every_hoist_flag_and_input():
    """The driver talks to the DSA segments through the decoder-layer
    segment, so a dropped kwarg there would silently un-hoist the work."""
    tree = _tree(_LAYER)
    prologue = _method_def(tree, "Glm5DecoderLayerGraphSegment", "run_row_validity_prologue")
    assert "position_ids" in {a.arg for a in prologue.args.kwonlyargs}
    (inner,) = [
        sub
        for sub in ast.walk(prologue)
        if isinstance(sub, ast.Call)
        and ast.unparse(sub.func) == "self.dsa_segment.run_row_validity_prologue"
    ]
    assert {kw.arg: ast.unparse(kw.value) for kw in inner.keywords}["position_ids"] == (
        "position_ids"
    )

    for name in ("paged_mqa_prep_key", "run_paged_mqa_prep", "selected_positions_key"):
        _method_def(tree, "Glm5DecoderLayerGraphSegment", name)
    assert any(
        isinstance(node, ast.FunctionDef) and node.name == "selects_top_k"
        for node in _class_def(tree, "Glm5DecoderLayerGraphSegment").body
    )

    fwd = _method_def(tree, "Glm5DecoderLayerGraphSegment", "forward")
    params = {a.arg for a in fwd.args.kwonlyargs}
    assert {"paged_mqa_prep_ready", "selected_positions_ready"} <= params
    (dsa_call,) = [
        sub
        for sub in ast.walk(fwd)
        if isinstance(sub, ast.Call)
        and ast.unparse(sub.func) == "self.dsa_segment.forward"
    ]
    passed = {kw.arg: ast.unparse(kw.value) for kw in dsa_call.keywords}
    for flag in ("row_validity_ready", "paged_mqa_prep_ready", "selected_positions_ready"):
        assert passed[flag] == flag
