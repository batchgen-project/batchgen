"""The worker's decode CUDA-graph state machine, as owned by ``GraphRuntime``.

These tests pin what the worker's 29 loose graph attributes guaranteed before
S2 moved them here:

* the initial value of every field, because the seven that the worker never
  set in ``__init__`` were read through ``getattr(self, name, <falsy>)`` --
  the field initializers ARE those defaults, and a direct read is only
  equivalent while that holds;
* the capture-once counter surviving the deep free that resets everything
  else;
* failed-bucket quarantine;
* the single owner of bucket-size generation;
* the startup adapter backstop and the shutdown drop order.

Three of today's asymmetries are pinned here as PINNED, NOT ENDORSED: they
are reproduced exactly because S2 is behavior-neutral, and each is a separate
fix.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from batchgen.cuda_graph.adapter import GraphMode, ModelCudaGraphAdapter
from batchgen.cuda_graph.buckets import generate_bucket_sizes
from batchgen.worker.graph_runtime import GraphRuntime

_ROOT = Path(__file__).resolve().parents[2]
_WORKER_SRC = _ROOT / "batchgen" / "batchgen_worker.py"


# The seven fields the worker never assigned in ``__init__``: every read of
# them before the first capture went through a getattr default.
LATE_INIT = {
    "whole_model_bucketing": None,
    "whole_model_segment": None,
    "whole_model_segment_name": None,
    "k25_whole_model_graph": False,
    "glm5_whole_model_all_short": False,
    "glm5_whole_model_index_topk": 0,
    "glm5_whole_model_capture_input_ids": None,
}

# Every field, with the value a fresh runtime starts at.
INITIAL = {
    "cuda_graph_manager": None,
    "cuda_graph_adapter": None,
    "cuda_graph_adapter_dual": True,
    "glm5_moe_cuda_graph_manager": None,
    "glm5_layer_cuda_graph_manager": None,
    "whole_model_graph": False,
    "decode_graph_captures": 0,
    "decode_graph_span_warn_monotonic": None,
    "glm5_whole_model_graph": False,
    "glm5_whole_model_graph_failed_buckets": set(),
    "glm5_whole_model_graph_capture_attempted_for_batch": False,
    "glm5_whole_model_graph_state_change_after_capture_logged": False,
    "glm5_whole_model_graph_signature": None,
    "glm5_whole_model_graph_unavailable_reason": None,
    "glm5_layer_graph_failed_buckets": set(),
    "glm5_layer_graph_capture_attempted_for_batch": False,
    "glm5_layer_graph_signature": None,
    "glm5_layer_graph_max_seqlen": None,
    "glm5_moe_graph_failed_buckets": set(),
    "glm5_dsa_graph_capture_attempted_for_batch": False,
    "glm5_moe_graph_capture_attempted_for_batch": False,
    "glm5_dsa_graph_page_table_change_after_capture_logged": False,
    **LATE_INIT,
}


def _worker_method(name: str) -> ast.FunctionDef:
    tree = ast.parse(_WORKER_SRC.read_text())
    return next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _graph_fields_assigned_in(method: ast.FunctionDef) -> set:
    """``self._graph.<field>`` assignment targets inside ``method``."""
    fields = set()
    for node in ast.walk(method):
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            targets = [node.target]
        for target in targets:
            if (
                isinstance(target, ast.Attribute)
                and ast.unparse(target).startswith("self._graph.")
            ):
                fields.add(target.attr)
    return fields


# --------------------------------------------------------------------------- #
# the state table
# --------------------------------------------------------------------------- #


def test_the_runtime_owns_exactly_the_twentynine_graph_fields():
    graph = GraphRuntime()
    own = {
        name
        for name in vars(graph)
        if not name.startswith("_")
    }
    assert own == set(INITIAL), (own ^ set(INITIAL))
    assert len(own) == 29, len(own)


@pytest.mark.parametrize("name,expected", sorted(INITIAL.items(), key=lambda kv: kv[0]))
def test_each_field_starts_at_the_value_its_old_getattr_default_returned(
    name, expected
):
    assert getattr(GraphRuntime(), name) == expected


def test_the_late_init_seven_are_falsy_so_a_direct_read_replaces_getattr():
    """The worker now reads these without a default; absent used to mean falsy."""
    graph = GraphRuntime()
    for name in LATE_INIT:
        assert not getattr(graph, name), name


def test_the_segment_name_default_is_spelled_at_the_read_not_in_the_field():
    """``whole_model_segment_name`` is the one non-falsy old default.

    The worker's single read was ``getattr(self, "_whole_model_segment_name",
    "glm5_whole_model")``. The field starts at None, so the read site must
    spell that default itself -- a getattr against an initialized field would
    hand replay a None segment name.
    """
    assert GraphRuntime().whole_model_segment_name is None
    src = _WORKER_SRC.read_text()
    assert (
        'self._graph.whole_model_segment_name or "glm5_whole_model"' in src
    ), "the read site must keep the absent=>default semantics"
    assert '"_whole_model_segment_name"' not in src


def test_a_set_field_is_per_instance_not_shared():
    a, b = GraphRuntime(), GraphRuntime()
    a.glm5_whole_model_graph_failed_buckets.add(64)
    assert b.glm5_whole_model_graph_failed_buckets == set()


# --------------------------------------------------------------------------- #
# capture-once guard
# --------------------------------------------------------------------------- #


def test_the_capture_counter_increments_per_capture_episode():
    graph = GraphRuntime()
    assert graph.decode_graph_captures == 0
    graph.decode_graph_captures += 1
    assert graph.decode_graph_captures == 1


def test_the_capture_counter_survives_the_deep_free_that_resets_the_rest():
    """PINNED, NOT ENDORSED (asymmetry, deliberate): the counter is the guard.

    ``deep_free_model_memory`` resets the graph state field by field. The
    capture-once counter is deliberately NOT in that list -- it is what makes
    a second capture on a reused instance fatal.
    """
    reset = _graph_fields_assigned_in(_worker_method("deep_free_model_memory"))

    assert "decode_graph_captures" not in reset
    assert reset <= set(INITIAL), reset - set(INITIAL)
    # Everything the deep free does reset returns to its initial value.
    graph = GraphRuntime()
    for name in reset:
        assert getattr(graph, name) == INITIAL[name], name


def test_three_fields_are_left_stale_by_the_deep_free():
    """PINNED, NOT ENDORSED: the K2.5 flags and the span-warn stamp.

    ``k25_whole_model_graph`` / ``whole_model_segment_name`` are written only
    by the K2.5 capture and reset by neither the release helper nor the deep
    free, so a deep free followed by a capture in another mode reads a stale
    True / stale name. Reproduced exactly; fixing it is its own slice.
    """
    reset = _graph_fields_assigned_in(_worker_method("deep_free_model_memory"))
    for name in ("k25_whole_model_graph", "whole_model_segment_name"):
        assert name not in reset, name
    released = _graph_fields_assigned_in(
        _worker_method("_release_glm5_whole_model_graph_state")
    )
    for name in ("k25_whole_model_graph", "whole_model_segment_name"):
        assert name not in released, name


def test_the_oom_rollback_quarantines_only_the_failing_bucket():
    """PINNED, NOT ENDORSED: the capture-OOM bookkeeping gap.

    The GLM-5 capture-OOM handler nulls the whole manager but adds only the
    bucket that failed to the failed set, so buckets already captured in that
    episode are dropped with the manager yet stay eligible next warmup.
    """
    setup = _worker_method("_setup_cuda_graphs")
    handler = next(
        node
        for node in ast.walk(setup)
        if isinstance(node, ast.ExceptHandler)
        and "OutOfMemoryError" in ast.unparse(node.type)
    )
    adds = [
        ast.unparse(node)
        for node in ast.walk(handler)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func)
        == "self._graph.glm5_whole_model_graph_failed_buckets.add"
    ]
    assert adds == [
        "self._graph.glm5_whole_model_graph_failed_buckets.add(capture_bucket)"
    ], adds
    assert "cuda_graph_manager" in _graph_fields_assigned_in(handler)


# --------------------------------------------------------------------------- #
# failed-bucket quarantine
# --------------------------------------------------------------------------- #


def test_a_failed_bucket_is_added_and_filters_the_capture_list():
    graph = GraphRuntime()
    graph.glm5_whole_model_graph_failed_buckets.add(64)
    graph.glm5_whole_model_graph_failed_buckets.add(64)

    assert graph.glm5_whole_model_graph_failed_buckets == {64}
    bucket_sizes = [1, 16, 64, 128]
    assert [
        bucket
        for bucket in bucket_sizes
        if bucket not in graph.glm5_whole_model_graph_failed_buckets
    ] == [1, 16, 128]


def test_the_capture_loop_filters_against_the_runtimes_failed_set():
    src = _WORKER_SRC.read_text()
    assert (
        "failed_buckets = self._graph.glm5_whole_model_graph_failed_buckets" in src
    )


# --------------------------------------------------------------------------- #
# bucket sizes: one owner
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "max_bucket,num_buckets,expected",
    [
        (256, 9, [1, 2, 4, 8, 16, 32, 64, 128, 256]),
        (256, 16, [1, 2, 3, 4, 6, 10, 14, 20, 28, 40, 56, 80, 128, 160, 192, 256]),
        (16, 16, list(range(1, 17))),
        (128, 16, [1, 2, 3, 4, 5, 7, 10, 14, 20, 24, 32, 48, 64, 80, 96, 128]),
        (128, 1, [128]),
    ],
)
def test_bucket_sizes_match_the_generator_the_worker_used_to_carry(
    max_bucket, num_buckets, expected
):
    """The values the deleted worker copy produced, by its own docstring."""
    assert GraphRuntime().generate_bucket_sizes(max_bucket, num_buckets) == expected
    assert generate_bucket_sizes(max_bucket, num_buckets) == expected


def test_the_runtime_delegates_to_the_single_generator():
    graph = GraphRuntime()
    for max_bucket in (1, 7, 16, 100, 128, 256, 1024):
        for num_buckets in (1, 2, 5, 9, 16, 32):
            assert graph.generate_bucket_sizes(
                max_bucket, num_buckets
            ) == generate_bucket_sizes(max_bucket, num_buckets)


def test_the_package_implements_the_generator_exactly_once():
    """One implementation; the runtime's same-named method only delegates.

    The worker used to carry a second copy of the geometric spacing (its
    ``_generate_bucket_sizes`` static method) while ``buckets.py``'s docstring
    already claimed the move had happened.
    """
    implementations, delegators = [], []
    for path in sorted((_ROOT / "batchgen").rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if not (
                isinstance(node, ast.FunctionDef)
                and node.name in {"generate_bucket_sizes", "_generate_bucket_sizes"}
            ):
                continue
            where = f"{path.relative_to(_ROOT)}:{node.lineno}"
            body = [
                stmt
                for stmt in node.body
                if not (
                    isinstance(stmt, ast.Expr)
                    and isinstance(stmt.value, ast.Constant)
                )
            ]
            is_delegation = (
                len(body) == 2
                and isinstance(body[0], ast.ImportFrom)
                and isinstance(body[1], ast.Return)
                and isinstance(body[1].value, ast.Call)
            )
            (delegators if is_delegation else implementations).append(where)
    assert len(implementations) == 1, implementations
    assert implementations[0].startswith(
        "batchgen/cuda_graph/buckets.py"
    ), implementations
    assert len(delegators) == 1, delegators
    assert delegators[0].startswith(
        "batchgen/worker/graph_runtime.py"
    ), delegators

    # The algorithm's inner helper exists only inside that one implementation.
    helpers = [
        f"{path.relative_to(_ROOT)}:{node.lineno}"
        for path in sorted((_ROOT / "batchgen").rglob("*.py"))
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.FunctionDef) and node.name == "_round_nice"
    ]
    assert len(helpers) == 1, helpers
    assert helpers[0].startswith("batchgen/cuda_graph/buckets.py"), helpers


# --------------------------------------------------------------------------- #
# adapter handle
# --------------------------------------------------------------------------- #


class _RealAdapter(ModelCudaGraphAdapter):
    def __init__(self):
        self.released = []

    def advertised_modes(self):
        return [GraphMode.WHOLE_MODEL]

    def is_supported(self, *a, **k):
        return True

    def select_buckets(self, *a, **k):
        return []

    def build_segments(self, *a, **k):
        return None

    def capture_signature(self, *a, **k):
        return ()

    def capture_inputs_for(self, *a, **k):
        return {}

    def eligibility(self, *a, **k):
        return None

    def prepare_replay_inputs(self, *a, **k):
        return {}

    def stage_post_graph_kv(self, *a, **k):
        return None

    def run_eager_reference(self, *a, **k):
        return {}


def test_a_model_without_an_adapter_leaves_the_handle_none():
    graph = GraphRuntime()
    graph.adopt_adapter(SimpleNamespace(), model_name="no-adapter-model")
    assert graph.cuda_graph_adapter is None


def test_a_real_adapter_is_adopted():
    graph = GraphRuntime()
    adapter = _RealAdapter()
    graph.adopt_adapter(
        SimpleNamespace(get_cuda_graph_adapter=lambda: adapter),
        model_name="real-model",
    )
    assert graph.cuda_graph_adapter is adapter


def test_a_duck_typed_adapter_is_rejected_at_adoption():
    class LooksRight:
        def advertised_modes(self):
            return [GraphMode.EAGER]

    graph = GraphRuntime()
    with pytest.raises(RuntimeError) as excinfo:
        graph.adopt_adapter(
            SimpleNamespace(get_cuda_graph_adapter=lambda: LooksRight()),
            model_name="duck-model",
        )
    message = str(excinfo.value)
    assert "'duck-model'" in message
    assert "ModelCudaGraphAdapter" in message
    assert "batchgen.cuda_graph.adapter.ModelCudaGraphAdapter" in message


# --------------------------------------------------------------------------- #
# per-step seam objects
# --------------------------------------------------------------------------- #


def test_make_batch_state_builds_the_adapter_snapshot():
    state = GraphRuntime().make_batch_state(
        local_bsz=2,
        max_rank_bsz=4,
        rank_token_counts=None,
        cache_seqlens=None,
        position_ids=None,
        max_seqlen=17,
        cur_batch_sequence_ids=("a", "b"),
        gpu_kv_manager=None,
        decode_iter=0,
        input_ids=None,
        device=None,
    )
    assert (state.local_bsz, state.max_rank_bsz, state.max_seqlen) == (2, 4, 17)
    assert state.cur_batch_sequence_ids == ("a", "b")


# --------------------------------------------------------------------------- #
# shutdown
# --------------------------------------------------------------------------- #


def test_drop_refs_clears_exactly_the_five_retaining_refs():
    graph = GraphRuntime()
    graph.cuda_graph_adapter = _RealAdapter()
    graph.cuda_graph_manager = object()
    graph.glm5_moe_cuda_graph_manager = object()
    graph.glm5_layer_cuda_graph_manager = object()
    graph.whole_model_segment = object()
    graph.whole_model_graph = True
    graph.decode_graph_captures = 1

    graph.drop_refs_for_shutdown()

    for name in (
        "cuda_graph_adapter",
        "cuda_graph_manager",
        "glm5_moe_cuda_graph_manager",
        "glm5_layer_cuda_graph_manager",
        "whole_model_segment",
    ):
        assert getattr(graph, name) is None, name
    # Nothing else is touched: the ordered shutdown nulled these five only.
    assert graph.whole_model_graph is True
    assert graph.decode_graph_captures == 1


def test_drop_refs_is_idempotent():
    graph = GraphRuntime()
    graph.cuda_graph_manager = object()
    graph.drop_refs_for_shutdown()
    graph.drop_refs_for_shutdown()
    assert graph.cuda_graph_manager is None


def test_the_adapter_ref_is_dropped_before_the_manager_and_the_segment():
    """Order is load-bearing: the adapter's ``_ctx`` holds the other refs."""
    source = Path(
        GraphRuntime.__module__.replace(".", "/") + ".py"
    )
    tree = ast.parse((_ROOT / source).read_text())
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "drop_refs_for_shutdown"
    )
    order = [
        node.targets[0].attr
        for node in method.body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Attribute)
    ]
    assert order == [
        "cuda_graph_adapter",
        "cuda_graph_manager",
        "glm5_moe_cuda_graph_manager",
        "glm5_layer_cuda_graph_manager",
        "whole_model_segment",
    ], order


def test_the_ordered_shutdown_calls_the_runtime_instead_of_five_setattrs():
    loop = (_ROOT / "batchgen" / "server_worker_main_loop.py").read_text()
    assert "graph.drop_refs_for_shutdown()" in loop
    for old in (
        '"_cuda_graph_adapter"',
        '"_cuda_graph_manager"',
        '"_glm5_moe_cuda_graph_manager"',
        '"_glm5_layer_cuda_graph_manager"',
        '"_whole_model_segment"',
    ):
        assert old not in loop, old
