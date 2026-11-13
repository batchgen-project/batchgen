"""CPU contract tests for the GLM-5 decode graph's captured context span.

The span used to be whatever the FIRST decode batch needed: the page tables
were sized from the worker's per-batch ``max_input_length`` /
``max_decoding_length`` whenever the model config carried no
``max_position_embeddings``, and the capture then took the ``min`` over those
tables. A later, longer turn had no replayable graph and every rank died with
``reason=max_seqlen_exceeds_capture``.

These tests pin the replacement: the span comes from the DEPLOYMENT bound
(``model_context_length``, what admission clamps every sequence to), trimmed
only by the graph's context-proportional statics budget, and a step past the
span falls back to eager decode with a rate-limited WARN.

``batchgen_worker.py`` does not import on a CPU dev box (Triton / FA3 /
DeepGEMM), so the methods under test are compiled from its AST against stub
globals and called with a stub ``self`` -- the same approach as
``test_glm5_worker_fa3_wiring.py``.
"""

import ast
import logging
import pathlib
import time
import types
import typing

import pytest

from batchgen.config.graph_span_policy import (
    GRAPH_CONTEXT_SPAN_ALIGN,
    GRAPH_CONTEXT_SPAN_FLOOR_TOKENS,
    GRAPH_CONTEXT_STATICS_BUDGET_BYTES,
    GRAPH_CONTEXT_STATICS_BYTES_PER_TOKEN_PER_ROW,
    graph_context_span_cap_tokens,
)
from batchgen.worker.kv_manager import KVCacheManager, PageTableCapacityRequest
from tests.worker.worker_fakes import make_graph_state
from tests.worker_method_source import (
    STEP_OBSERVABILITY_SOURCE,
    delegated_method_names,
    worker_shaped_method_node,
)


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_WORKER_SRC = _REPO_ROOT / "batchgen/batchgen_worker.py"
_WORKER_TREE = ast.parse(_WORKER_SRC.read_text())


def _worker_class() -> ast.ClassDef:
    for node in _WORKER_TREE.body:
        if isinstance(node, ast.ClassDef) and node.name == "BatchGenWorker":
            return node
    raise AssertionError("no top-level class `BatchGenWorker`")


def _worker_method(name: str) -> ast.FunctionDef:
    # S7/S10a: the span warning (StepObservabilityEngine) and the KV pool
    # sizing (KVCapacityEngine) run in their engines (worker delegates).
    if name in delegated_method_names():
        return worker_shaped_method_node(name)
    for stmt in _worker_class().body:
        if isinstance(stmt, ast.FunctionDef) and stmt.name == name:
            return stmt
    raise AssertionError(f"no `BatchGenWorker.{name}`")


def _assigned_value(body, name: str):
    """Evaluate the single assignment of ``name`` in ``body``.

    The constants under test are plain literals and one ``frozenset({...})``,
    so evaluating the unparsed right-hand side needs no worker import.
    """
    for stmt in body:
        if isinstance(stmt, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in stmt.targets
        ):
            return eval(ast.unparse(stmt.value), {"frozenset": frozenset})
    raise AssertionError(f"no assignment of `{name}`")


def _module_constant(name: str):
    return _assigned_value(_WORKER_TREE.body, name)


def _class_constant(name: str):
    return _assigned_value(_worker_class().body, name)


BUDGET_BYTES = GRAPH_CONTEXT_STATICS_BUDGET_BYTES
BYTES_PER_TOKEN_PER_ROW = GRAPH_CONTEXT_STATICS_BYTES_PER_TOKEN_PER_ROW
SPAN_ALIGN = GRAPH_CONTEXT_SPAN_ALIGN
SPAN_FLOOR = GRAPH_CONTEXT_SPAN_FLOOR_TOKENS
# S7: the rate-limit constant moved with the warning into the engine module.
WARN_INTERVAL_S = _assigned_value(
    ast.parse(STEP_OBSERVABILITY_SOURCE.read_text()).body, "_GRAPH_SPAN_WARN_INTERVAL_S"
)
EAGER_SERVABLE_REASONS = _class_constant(
    "_WHOLE_GRAPH_EAGER_SERVABLE_REASONS"
)


class _StubAttnWrapperBase:
    max_seqlen = 0


def _compile(name: str, extra_globals=None):
    """Compile one worker method against stub globals and return it."""
    fn = _worker_method(name)
    # A `@staticmethod` decorator would wrap the result; the tests call these
    # as plain functions, so the decorator list is dropped.
    fn.decorator_list = []
    namespace = {
        "time": time,
        "logging": logging,
        "Sequence": typing.Sequence,
        "Optional": typing.Optional,
        "PageTableCapacityRequest": PageTableCapacityRequest,
        "AttnWrapperBase": _StubAttnWrapperBase,
        "GRAPH_CONTEXT_STATICS_BUDGET_BYTES": BUDGET_BYTES,
        "_GRAPH_SPAN_WARN_INTERVAL_S": WARN_INTERVAL_S,
    }
    namespace.update(extra_globals or {})
    module = ast.Module(body=[fn], type_ignores=[])
    exec(compile(module, str(_WORKER_SRC), "exec"), namespace)
    return namespace[name]


def _page_table(num_pages_per_sequence: int):
    """Minimal stand-in: the capacity helper reads only ``shape[1]``."""
    return types.SimpleNamespace(shape=(1, int(num_pages_per_sequence)))


# --------------------------------------------------------------------------- #
# The deployment bound
# --------------------------------------------------------------------------- #

@pytest.fixture
def bound_fn():
    return _compile("_deployment_context_bound_tokens")


@pytest.mark.parametrize(
    "model_context_length, expected",
    [
        (1048576, 1048576),  # GLM-5.2 / GLM-5.3
        (202752, 202752),  # GLM-5 / GLM-5.1
        (131072, 131072),  # the `_initialize_core_components` default
        (None, 0),  # before core init
        (0, 0),
    ],
)
def test_deployment_bound_is_the_model_context_length(
    bound_fn, model_context_length, expected
):
    worker = types.SimpleNamespace(model_context_length=model_context_length)
    assert bound_fn(worker) == expected


# --------------------------------------------------------------------------- #
# Page-table token capacity is sized from the deployment bound
# --------------------------------------------------------------------------- #


def _capacity_request(
    *,
    model_context_length,
    max_input_length=0,
    max_decoding_length=0,
    model_config=None,
    cuda_graph_max_bucket_size=128,
):
    make = _compile("_make_page_table_capacity_request")
    worker = types.SimpleNamespace(
        max_input_length=max_input_length,
        max_decoding_length=max_decoding_length,
        engine_config=None,
        model_config=model_config,
        args=types.SimpleNamespace(
            cuda_graph_max_bucket_size=cuda_graph_max_bucket_size
        ),
        _deployment_context_bound_tokens=lambda: int(model_context_length or 0),
    )
    return make(worker)


@pytest.mark.parametrize(
    "model_context_length, max_input_length, max_decoding_length, expected",
    [
        # The reproduced failure: a short warm batch (prompt 64, 32768 decode)
        # used to fix the span at 32832 for the life of the process.
        (1048576, 64, 32768, 1048576),
        # A long first batch does not shrink it either.
        (1048576, 65536, 983040, 1048576),
        (202752, 128, 32, 202752),
        # Batch hints cannot widen a deployment-sized row.
        (131072, 200000, 0, 131072),
        # An explicitly lower deployment bound has no hidden floor.
        (4096, 0, 0, 4096),
    ],
)
def test_token_capacity_comes_from_the_deployment_bound(
    model_context_length, max_input_length, max_decoding_length, expected
):
    request = _capacity_request(
        model_context_length=model_context_length,
        max_input_length=max_input_length,
        max_decoding_length=max_decoding_length,
    )
    assert KVCacheManager.page_table_token_capacity(request) == expected


def test_deployment_bound_survives_a_config_without_max_position_embeddings():
    # The regression: `getattr(model_config, "max_position_embeddings", None)`
    # returned None for such a config, which dropped the bound from the
    # candidate set and left the span at the first batch's prompt + decode.
    request = _capacity_request(
        model_context_length=1048576,
        max_input_length=64,
        max_decoding_length=32768,
        model_config=types.SimpleNamespace(),
    )
    assert request.model_max_position_embeddings == 1048576
    assert KVCacheManager.page_table_token_capacity(request) == 1048576


def test_model_config_is_the_fallback_before_core_init():
    # `model_context_length` is only unset before `_initialize_core_components`
    # has run; the config's own field still answers then.
    request = _capacity_request(
        model_context_length=0,
        model_config=types.SimpleNamespace(max_position_embeddings=202752),
    )
    assert request.model_max_position_embeddings == 202752
    assert KVCacheManager.page_table_token_capacity(request) == 202752


# --------------------------------------------------------------------------- #
# The context-proportional statics budget
# --------------------------------------------------------------------------- #


@pytest.fixture
def span_cap():
    return graph_context_span_cap_tokens


def test_page_split_span_cap_is_charged_on_the_top_bucket_only(span_cap):
    # DeepGEMM's logits block comes from the graph-private pool and is freed
    # after each layer's top-k, so one top-bucket block covers every layer and
    # every bucket.
    buckets = [1, 2, 4, 8, 16, 32, 64, 128]
    expected = BUDGET_BYTES // (128 * BYTES_PER_TOKEN_PER_ROW)
    assert expected % SPAN_ALIGN == 0
    assert span_cap(bucket_sizes=buckets, aux_is_page_split=True) == expected
    # The release configuration buys the whole GLM-5.3 context.
    assert expected == 1048576


def test_aos_span_cap_is_charged_on_every_bucket(span_cap):
    # `agg_scores` is a per-bucket static on the bf16 AoS scorer, so the rows
    # add up over the bucket set and the same budget buys ~4x less span.
    buckets = [1, 2, 4, 8, 16, 32, 64, 128]
    rows = sum(buckets)
    expected = BUDGET_BYTES // (rows * BYTES_PER_TOKEN_PER_ROW)
    expected -= expected % SPAN_ALIGN
    assert span_cap(bucket_sizes=buckets, aux_is_page_split=False) == expected
    assert expected < span_cap(bucket_sizes=buckets, aux_is_page_split=True)


def test_span_cap_is_256_aligned(span_cap):
    # A span that is not 256-aligned buys nothing: DeepGEMM rounds the logits
    # row stride up to 256.
    for budget in (BUDGET_BYTES, BUDGET_BYTES // 3, BUDGET_BYTES * 7 // 5):
        cap = span_cap(
            bucket_sizes=[1, 2, 4, 8, 48],
            aux_is_page_split=True,
            budget_bytes=budget,
        )
        assert cap % SPAN_ALIGN == 0 or cap == SPAN_FLOOR


def test_span_cap_never_trims_below_the_floor(span_cap):
    assert (
        span_cap(
            bucket_sizes=[256],
            aux_is_page_split=True,
            budget_bytes=1024,
        )
        == SPAN_FLOOR
    )


def test_span_cap_rejects_an_empty_bucket_set(span_cap):
    with pytest.raises(ValueError, match="at least one positive bucket size"):
        span_cap(bucket_sizes=[], aux_is_page_split=True)


def test_prewarm_geometry_applies_the_same_span_cap():
    """The prewarm compiles the span-dependent kernels before capture.

    Its ``graph_max_seqlen`` must take the ``min`` over every term the worker
    does, cap included, or the warm template and the captured one differ.
    """
    geometry_src = (
        _REPO_ROOT / "batchgen/models/glm/glm5/triton_prewarm.py"
    ).read_text()
    tree = ast.parse(geometry_src)
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Glm5DecodeGeometry"
    )
    span_prop = next(
        stmt
        for stmt in cls.body
        if isinstance(stmt, ast.FunctionDef) and stmt.name == "graph_max_seqlen"
    )
    src = ast.unparse(span_prop)
    assert "self.span_cap_tokens" in src
    # And the field is part of the geometry identity, so a bucket-set change
    # re-runs the phase instead of reusing a warm for a different span.
    key = next(
        stmt
        for stmt in cls.body
        if isinstance(stmt, ast.FunctionDef) and stmt.name == "key"
    )
    assert "self.span_cap_tokens" in ast.unparse(key)
    # One owner for the policy: the prewarm calls the shared helper.
    assert "graph_context_span_cap_tokens(" in geometry_src
    assert "from batchgen.config.graph_span_policy import" in geometry_src


# --------------------------------------------------------------------------- #
# The captured span
# --------------------------------------------------------------------------- #


@pytest.fixture
def capacity_tokens():
    return _compile("_glm5_dsa_graph_score_capacity_tokens")


def test_captured_span_is_the_deployment_bound_when_tables_allow_it(
    capacity_tokens,
):
    # Page tables sized from the same bound (16384 pages x 64 tokens), so the
    # `min` lands on the bound itself rather than on a first-batch table.
    assert (
        capacity_tokens(
            _page_table(16384),
            64,
            _page_table(16384),
            64,
            deployment_context_tokens=1048576,
            span_cap_tokens=1048576,
        )
        == 1048576
    )


def test_captured_span_is_trimmed_by_the_statics_budget(capacity_tokens):
    assert (
        capacity_tokens(
            _page_table(16384),
            64,
            _page_table(16384),
            64,
            deployment_context_tokens=1048576,
            span_cap_tokens=262144,
        )
        == 262144
    )


def test_captured_span_cannot_exceed_a_page_table_column_count(capacity_tokens):
    # Columns the table does not have are unaddressable, so the hardware limit
    # stays in the `min`.
    assert (
        capacity_tokens(
            _page_table(512),
            64,
            _page_table(16384),
            64,
            deployment_context_tokens=1048576,
            span_cap_tokens=1048576,
        )
        == 32768
    )


def test_captured_span_requires_the_deployment_bound(capacity_tokens):
    # The old signature defaulted this to None and silently dropped it, which
    # is exactly how the span collapsed onto the first batch's page tables.
    for missing in (0, -1):
        with pytest.raises(RuntimeError, match="positive deployment context"):
            capacity_tokens(
                _page_table(16384),
                64,
                _page_table(16384),
                64,
                deployment_context_tokens=missing,
            )


def test_captured_span_helper_has_no_default_for_the_bound():
    fn = _worker_method("_glm5_dsa_graph_score_capacity_tokens")
    names = [a.arg for a in fn.args.kwonlyargs]
    defaults = dict(zip(names, fn.args.kw_defaults))
    assert names == ["deployment_context_tokens", "span_cap_tokens"]
    assert defaults["deployment_context_tokens"] is None  # i.e. required


# --------------------------------------------------------------------------- #
# A step past the span falls back to eager instead of raising
# --------------------------------------------------------------------------- #


def _path_state_worker(*, captured_span: int, bucket: int = 2):
    bucketing = types.SimpleNamespace(get_padded_size=lambda n: bucket)
    manager = types.SimpleNamespace(
        bucketing=bucketing,
        has_bucket_for_all_segments=lambda n: True,
    )
    return types.SimpleNamespace(
        model_name="glm-5.3",
        _glm5_whole_model_graph_requested_for_current_batch=lambda: True,
        _glm5_whole_model_graph_capture_signature=lambda bucket_: "sig",
        _graph=make_graph_state(
            glm5_whole_model_graph_unavailable_reason=None,
            cuda_graph_manager=manager,
            glm5_whole_model_graph=True,
            glm5_whole_model_graph_failed_buckets=set(),
            whole_model_segment=types.SimpleNamespace(max_seqlen=captured_span),
            glm5_whole_model_all_short=False,
            glm5_whole_model_index_topk=2048,
            glm5_whole_model_graph_signature="sig",
        ),
    )


def test_step_past_the_span_reports_the_exceeds_capture_state(monkeypatch):
    path_state = _compile("_glm5_whole_graph_path_state")
    monkeypatch.setattr(_StubAttnWrapperBase, "max_seqlen", 65536)
    worker = _path_state_worker(captured_span=32832)

    assert path_state(worker, 2) == ("eager", 2, "max_seqlen_exceeds_capture")


def test_step_inside_the_span_replays_the_graph(monkeypatch):
    path_state = _compile("_glm5_whole_graph_path_state")
    monkeypatch.setattr(_StubAttnWrapperBase, "max_seqlen", 65536)
    worker = _path_state_worker(captured_span=1048576)

    assert path_state(worker, 2) == ("graph", 2, "captured")


def test_exceeds_capture_is_the_one_eager_servable_reason():
    # Eager decode is length-agnostic, so this state is correctness-neutral.
    # Every other reason means the capture is missing or stale, which must stay
    # fail-loud rather than silently degrade a whole run to eager.
    assert EAGER_SERVABLE_REASONS == frozenset({"max_seqlen_exceeds_capture"})


def test_decode_step_only_raises_for_a_non_eager_servable_reason():
    """The required-graph fatal must sit under the eager-servable branch.

    This is the line that killed every rank: the decode step raised whenever
    no replayable graph existed, including for a step merely longer than the
    capture.
    """
    raises = [
        node
        for node in ast.walk(_worker_class())
        if isinstance(node, ast.Raise)
        and "no replayable" in ast.unparse(node)
    ]
    assert len(raises) == 1, [ast.unparse(r) for r in raises]
    guards = [
        node
        for node in ast.walk(_worker_class())
        if isinstance(node, ast.If)
        and "_WHOLE_GRAPH_EAGER_SERVABLE_REASONS" in ast.unparse(node.test)
    ]
    assert len(guards) == 1
    # The raise is in the `else`, i.e. it is reached only for reasons eager
    # decode cannot stand in for.
    assert raises[0] in list(ast.walk(ast.Module(body=guards[0].orelse, type_ignores=[])))
    assert raises[0] not in list(
        ast.walk(ast.Module(body=guards[0].body, type_ignores=[]))
    )


def test_capture_no_longer_raises_when_the_batch_is_past_the_span():
    # Capturing at the span still serves every shorter step, so a batch that
    # is already longer is a WARN plus an eager step, not a capture error.
    owners = [
        stmt
        for stmt in _worker_class().body
        if isinstance(stmt, ast.FunctionDef)
        and "self._glm5_dsa_graph_score_capacity_tokens" in ast.unparse(stmt)
    ]
    assert len(owners) == 1
    src = ast.unparse(owners[0])
    assert "exceeds model/KV capacity" not in src
    assert "self._warn_decode_graph_span_exceeded(" in src
    # The AST-wiring contract in test_glm5_worker_fa3_wiring.py pins this name.
    assert "graph_max_seqlen = int(capacity_seqlen)" in src


# --------------------------------------------------------------------------- #
# The WARN is rate-limited
# --------------------------------------------------------------------------- #


def test_span_warn_names_both_lengths_and_is_rate_limited(caplog):
    warn = _compile("_warn_decode_graph_span_exceeded")
    worker = types.SimpleNamespace(
        rank=3, _graph=make_graph_state(decode_graph_span_warn_monotonic=None)
    )

    with caplog.at_level(logging.WARNING):
        warn(worker, 65536, 32832)
        warn(worker, 70000, 32832)

    records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(records) == 1, [r.getMessage() for r in records]
    message = records[0].getMessage()
    assert "32832" in message
    assert "65536" in message
    assert "eager decode" in message

    # Past the interval it speaks again.
    worker._graph.decode_graph_span_warn_monotonic = (
        time.monotonic() - WARN_INTERVAL_S - 1.0
    )
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        warn(worker, 70000, 32832)
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1
