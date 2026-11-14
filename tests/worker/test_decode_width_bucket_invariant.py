"""Decode width is single-sourced and covered by the whole-model graph ladder.

Invariants pinned here:

* Every decode configure sizes its per-rank buffers, and the decode admission
  cap (``_decode_padding_bsz``), from ``_max_decode_rank_bsz()``; no caller
  passes its own width and no wave-size estimate clamps it.
* On a deployment that requests whole-model decode graphs, the bucket ladder
  contains that width as a bucket and tops out at or above it; a plan or a
  decode step wider than the top bucket raises
  ``DecodeWidthExceedsGraphBucketError`` instead of running eager.
* Deployments that do not request graphs are untouched.

The worker module cannot be imported on a CPU-only machine, so its methods
are compiled from the real source (``tests.worker_method_source``).
"""

from __future__ import annotations

import ast
import logging
import os
from types import SimpleNamespace
from typing import Optional

import pytest

from batchgen.config.cuda_graph_flag_policy import (
    cli_enables_whole_model_graph,
    glm5_whole_model_cuda_graph_requested,
)
from batchgen.cuda_graph.buckets import (
    DecodeWidthExceedsGraphBucketError,
    check_decode_width_within_buckets,
    decode_graph_bucket_sizes,
    generate_bucket_sizes,
)
from batchgen.cuda_graph.graph_manager import BatchSizeBucketing
from batchgen.models.glm.glm5.cuda_graph_policy import (
    glm5_whole_model_cuda_graph_requested_for_model,
)
from batchgen.worker.kv_manager import KVCacheManager, PageTableCapacityRequest
from tests.worker.worker_fakes import attach_graph_state
from tests.worker_method_source import (
    WORKER_SOURCE,
    load_worker_methods,
    load_worker_shaped_methods,
)

GLM5 = "zai-org/GLM-5-FP8"
GPT_OSS = "openai/gpt-oss-120b"

_METHODS = load_worker_methods(
    [
        "_whole_model_decode_graphs_requested",
        "_glm5_whole_model_graph_requested_for_current_batch",
        "_decode_graph_bucket_sizes",
        "_decode_graph_top_bucket",
        "_check_decode_width_within_graph_buckets",
        "_assert_decode_step_within_graph_buckets",
        "_load_decode_model",
    ],
    {
        "os": os,
        "logging": logging,
        "Optional": Optional,
        "_MAX_DECODE_RANK_BSZ": 128,
        "cli_enables_whole_model_graph": cli_enables_whole_model_graph,
        "glm5_whole_model_cuda_graph_requested_for_model": (
            glm5_whole_model_cuda_graph_requested_for_model
        ),
        "decode_graph_bucket_sizes": decode_graph_bucket_sizes,
        "check_decode_width_within_buckets": check_decode_width_within_buckets,
        "DecodeWidthExceedsGraphBucketError": DecodeWidthExceedsGraphBucketError,
        # rb5: the branch re-homed the predicate behind cuda_graph_flag_policy
        # (S4b); the body reads this spelling, not the _for_model one.
        "glm5_whole_model_cuda_graph_requested": glm5_whole_model_cuda_graph_requested,
    },
)
# S10a: _max_decode_rank_bsz runs in KVCapacityEngine; lift the shaped body
# so the worker delegator's engine import never runs.
_METHODS.update(load_worker_shaped_methods(
    ["_max_decode_rank_bsz"],
    {
        "os": os,
        "logging": logging,
        "_MAX_DECODE_RANK_BSZ": 128,
        "decode_graph_bucket_sizes": decode_graph_bucket_sizes,
    },
))
_Worker = type("_Worker", (), dict(_METHODS))


def _worker(
    *,
    model_name=GLM5,
    enable_cuda_graph=True,
    disable_cuda_graphs=False,
    max_bucket=128,
    num_buckets=16,
    attn_tp_size=1,
    planned=0,
    reuse=True,
):
    worker = _Worker()
    attach_graph_state(worker)
    worker.rank = 1
    worker.model_name = model_name
    worker.args = SimpleNamespace(
        enable_cuda_graph=enable_cuda_graph,
        disable_cuda_graphs=disable_cuda_graphs,
        cuda_graph_max_bucket_size=max_bucket,
        cuda_graph_num_buckets=num_buckets,
    )
    worker.engine_config = SimpleNamespace(
        Module_Batching_Config=SimpleNamespace(MoE_decoding_micro_batch_size=planned)
    )
    worker._decode_attn_tp_size = lambda: attn_tp_size
    worker.events = []
    worker._desired_phase_key = lambda phase: ("key", phase)

    def request_reuse():
        worker.events.append("reuse_request")
        return reuse

    worker._request_phase_instance_reuse = request_reuse

    def transition(key, comm=None, max_num_seq=None):
        worker.events.append(("transition", key, max_num_seq))
        return True, False

    worker._transition_phase = transition
    worker._log_phase_memory_ledger = lambda label: None
    worker._assert_phase_switch_clean = lambda *a, **k: None
    worker.set_phase = lambda phase: worker.events.append(("phase", phase))
    worker._install_decode_weight_copy_queue = lambda: None
    worker._remap_reused_kv_pool = lambda: None
    return worker


@pytest.fixture(autouse=True)
def _no_width_env(monkeypatch):
    monkeypatch.delenv("BATCHGEN_MAX_DECODE_RANK_BSZ", raising=False)
    monkeypatch.delenv("BATCHGEN_MAX_RANK_BSZ", raising=False)


# ---- single width source ----------------------------------------------------


def test_raising_the_bucket_flag_widens_decode_on_graph_deployments():
    """POIS 2026-11-12: the bucket flag is the width knob on whole-model-graph
    deployments — raising it raises the admission width with it, so capture
    above the width cannot exist."""
    assert _worker(max_bucket=256)._max_decode_rank_bsz() == 256


def test_the_bucket_flag_does_not_widen_eager_deployments():
    assert _worker(max_bucket=256, enable_cuda_graph=False)._max_decode_rank_bsz() == 128
    assert _worker(max_bucket=256, disable_cuda_graphs=True)._max_decode_rank_bsz() == 128


def test_an_explicit_env_cap_beats_the_bucket_flag(monkeypatch):
    monkeypatch.setenv("BATCHGEN_MAX_DECODE_RANK_BSZ", "128")
    assert _worker(max_bucket=256)._max_decode_rank_bsz() == 128


def test_a_wider_tp_plan_beats_the_bucket_flag():
    assert _worker(max_bucket=256, attn_tp_size=8, planned=320)._max_decode_rank_bsz() == 320





@pytest.mark.parametrize(
    "kwargs,width",
    [
        ({}, 128),  # GLM-5 whole-model graph, pure DP
        ({"reuse": False}, 128),  # manager declines reuse: same width
        ({"enable_cuda_graph": False}, 128),  # eager deployment
        ({"model_name": GPT_OSS}, 128),
        ({"attn_tp_size": 8, "planned": 160}, 160),  # TP group, planned lift
        ({"attn_tp_size": 8, "planned": 96}, 128),  # planned below the guard
        ({"attn_tp_size": 1, "planned": 160}, 128),  # planned only lifts TP groups
    ],
)
def test_decode_configure_width_is_the_single_source(kwargs, width):
    worker = _worker(**kwargs)
    assert worker._max_decode_rank_bsz() == width

    worker._load_decode_model(comm="comm")

    assert worker._decode_padding_bsz == width
    assert ("transition", ("key", "decode"), width) in worker.events
    # The reuse request still runs: it configures the manager, not the width.
    assert worker.events[0] == "reuse_request"


def test_decode_configure_takes_no_width_argument():
    worker = _worker()
    with pytest.raises(TypeError):
        worker._load_decode_model(64, comm="comm")


def _worker_tree():
    return ast.parse(WORKER_SOURCE.read_text())


def _worker_class(tree):
    return next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "BatchGenWorker"
    )


def test_wave_size_estimate_clamp_is_gone():
    worker_src = WORKER_SOURCE.read_text()
    decode_src = (WORKER_SOURCE.parent / "worker" / "decode.py").read_text()
    assert "estimate_max_decode_replica_batch" not in worker_src
    assert "estimate_max_decode_replica_batch" not in decode_src
    assert "max_num_seq_estimate" not in worker_src


def _assigned_from_width_helper(function, name):
    for node in ast.walk(function):
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)
        ):
            value = node.value
            return (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Attribute)
                and value.func.attr == "_max_decode_rank_bsz"
            )
    return False


def test_every_decode_configure_width_derives_from_the_width_helper():
    cls = _worker_class(_worker_tree())
    sites = []
    for function in cls.body:
        if not isinstance(function, ast.FunctionDef):
            continue
        for node in ast.walk(function):
            # Every call that passes a decode width to a phase transition.
            # A forward of the function's own ``max_num_seq`` parameter
            # (``_transition_phase`` into the enter context) is a pass-through,
            # not a width source.
            params = {a.arg for a in function.args.args + function.args.kwonlyargs}
            if isinstance(node, ast.Call):
                for kw in node.keywords:
                    if kw.arg == "max_num_seq" and not (
                        isinstance(kw.value, ast.Name) and kw.value.id in params
                    ):
                        sites.append((function.name, kw.value))
            # Every write of the decode admission cap.
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if (
                        isinstance(target, ast.Attribute)
                        and target.attr == "_decode_padding_bsz"
                    ):
                        sites.append((function.name, node.value))
    assert {name for name, _ in sites} == {
        "_load_decode_model", "_build_persistent_phase_instances",
    }
    functions = {f.name: f for f in cls.body if isinstance(f, ast.FunctionDef)}
    for name, value in sites:
        assert isinstance(value, ast.Name), (name, ast.unparse(value))
        assert _assigned_from_width_helper(functions[name], value.id), (
            name, value.id,
        )


def test_every_decode_model_load_lets_the_loader_size_the_width():
    # S9 moved one call site (the lifecycle rebuild) into
    # WorkerLifecycleEngine; the invariant spans both classes.
    engine_tree = ast.parse(
        (WORKER_SOURCE.parent / "worker" / "worker_lifecycle_engine.py").read_text()
    )
    calls = [
        node
        for tree in (_worker_class(_worker_tree()), engine_tree)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_load_decode_model"
    ]
    assert len(calls) == 3
    for call in calls:
        assert len(call.args) + len(call.keywords) == 1
        assert ast.unparse(call.args[0] if call.args else call.keywords[0].value) == (
            "self.comm"
        )


# ---- ladder covers the width ------------------------------------------------


def test_ladder_is_unchanged_at_the_default_width():
    worker = _worker()
    assert worker._decode_graph_bucket_sizes() == generate_bucket_sizes(128, 16)
    assert worker._decode_graph_top_bucket() == 128


def test_ladder_top_is_raised_to_a_wider_planned_width():
    worker = _worker(attn_tp_size=8, planned=160)
    sizes = worker._decode_graph_bucket_sizes()
    assert max(sizes) == 160 and 160 in sizes
    assert worker._decode_graph_top_bucket() == 160


def test_page_table_slots_cover_the_derived_top_bucket():
    request = PageTableCapacityRequest(
        engine_module_global_batch_size=None,
        engine_module_attn_decoding_micro_batch_size=None,
        engine_basic_num_queries=None,
        model_max_position_embeddings=4096,
        args_cuda_graph_max_bucket_size=128,
        decode_graph_top_bucket=160,
    )
    assert KVCacheManager.page_table_slot_capacity(request) == 160


def test_page_table_request_carries_the_top_bucket_only_for_graph_deployments():
    # S10a: _make_page_table_capacity_request runs in KVCapacityEngine.
    source = (
        WORKER_SOURCE.parent / "worker" / "kv_capacity_engine.py"
    ).read_text()
    assert "decode_graph_top_bucket=decode_graph_top_bucket" in source
    assert _worker(enable_cuda_graph=False)._decode_graph_top_bucket() is None
    assert _worker(disable_cuda_graphs=True)._decode_graph_top_bucket() is None


def test_capture_logs_and_checks_the_derived_ladder():
    cls = _worker_class(_worker_tree())
    setup = next(
        f for f in cls.body
        if isinstance(f, ast.FunctionDef) and f.name == "_setup_cuda_graphs"
    )
    body = ast.unparse(setup)
    assert "bucket_sizes = self._decode_graph_bucket_sizes()" in body
    assert "check_decode_width_within_buckets(decode_width, bucket_sizes)" in body
    # No second ladder is generated from the raw flag.
    assert "generate_bucket_sizes(" not in body
    # S2: the one ladder construction goes through GraphRuntime.
    assert body.count("self._graph.new_bucketing(") == 1


# ---- configure-time invariant ----------------------------------------------


def test_configure_invariant_trips_when_the_ladder_is_narrower_than_the_width():
    worker = _worker()
    worker._decode_graph_bucket_sizes = lambda: generate_bucket_sizes(64, 16)
    with pytest.raises(
        DecodeWidthExceedsGraphBucketError,
        match=r"per-rank decode width 128 exceeds the largest whole-model "
        r"decode graph bucket 64 .*Raise --cuda-graph-max-bucket-size to at "
        r"least 128",
    ):
        worker._load_decode_model(comm="comm")
    # Fails before any decode instance is configured.
    assert not any(
        isinstance(e, tuple) and e[0] == "transition" for e in worker.events
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"enable_cuda_graph": False},
        {"disable_cuda_graphs": True},
        {"model_name": "moonshotai/Kimi-K3"},
    ],
)
def test_eager_deployments_are_untouched(kwargs):
    worker = _worker(**kwargs)
    worker._decode_graph_bucket_sizes = lambda: [1, 2]  # would trip if consulted
    assert not worker._whole_model_decode_graphs_requested()
    worker._check_decode_width_within_graph_buckets(4096)
    worker._load_decode_model(comm="comm")
    assert worker._decode_padding_bsz == 128
    worker._assert_decode_step_within_graph_buckets(4096)


# ---- decode-step invariant: the eager fallback above the bucket is gone ----


def _captured(worker, buckets):
    worker._graph.whole_model_graph = True
    worker._graph.cuda_graph_manager = object()
    worker._graph.whole_model_bucketing = BatchSizeBucketing(buckets)
    worker._decode_padding_bsz = 128


@pytest.mark.parametrize("model_name", [GLM5, GPT_OSS])
def test_decode_step_above_the_captured_top_bucket_raises(model_name):
    worker = _worker(model_name=model_name)
    _captured(worker, [1, 64, 128])
    worker._assert_decode_step_within_graph_buckets(128)
    with pytest.raises(
        DecodeWidthExceedsGraphBucketError,
        match=r"decode step max per-rank batch 129 exceeds the largest "
        r"whole-model decode graph bucket 128 .*refusing to run the step eager",
    ):
        worker._assert_decode_step_within_graph_buckets(129)


def test_decode_step_without_a_live_graph_checks_the_derived_ladder():
    worker = _worker()
    worker._graph.whole_model_graph = False
    worker._graph.cuda_graph_manager = None
    worker._graph.whole_model_bucketing = None
    worker._decode_padding_bsz = 128
    worker._assert_decode_step_within_graph_buckets(128)
    with pytest.raises(DecodeWidthExceedsGraphBucketError):
        worker._assert_decode_step_within_graph_buckets(129)


def test_decode_loop_has_no_over_bucket_eager_exemption():
    # S4b: the decode loop body runs in DecoupledGpuDecode (decode_engine.py);
    # the worker keeps no copy of the over-bucket exemption either.
    assert "_glm5_whole_graph_over_bucket" not in WORKER_SOURCE.read_text()
    source = (WORKER_SOURCE.parent / "worker" / "decode_engine.py").read_text()
    assert "_glm5_whole_graph_over_bucket" not in source
    assert "self._assert_decode_step_within_graph_buckets(_max_bs)" in source
    # The check precedes the graph/eager dispatch decision.
    assert source.index(
        "self._assert_decode_step_within_graph_buckets(_max_bs)"
    ) < source.index("_use_graph = (")
