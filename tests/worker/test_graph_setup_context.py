"""Run the real context/delegator seam with CPU capture collaborators."""

from dataclasses import FrozenInstanceError, fields
import inspect
import gc
import logging
import math
import sys
from types import MethodType, ModuleType, SimpleNamespace
from typing import Sequence

import pytest
import torch

from batchgen.worker.graph_runtime import (
    GRAPH_SETUP_UNCHANGED,
    GraphRuntime,
    GraphSetupContext,
    GraphSetupOutcome,
)
from tests.worker_method_source import load_worker_methods
from batchgen.sequence import SequenceStatus


class RecordingWorker(SimpleNamespace):
    def __setattr__(self, name, value):
        if name in {field.name for field in fields(GraphSetupOutcome)}:
            self.events.append(("bind", name, value))
        super().__setattr__(name, value)


def _worker(monkeypatch, *, probe_requested=True, failure=None, prior_probe=None):
    events = []
    probe = object()

    def module(name, **values):
        fake = ModuleType(name)
        fake.__dict__.update(values)
        monkeypatch.setitem(sys.modules, name, fake)

    class Segment:
        def __init__(self, *args, **kwargs):
            self.__dict__.update(kwargs)

        def set_capture_inputs(self, **kwargs):
            events.append(("inputs", kwargs))

    class Manager:
        def __init__(self, bucketing, device):
            self.bucketing = bucketing

        def register_segment(self, name, segment):
            self.segment = segment

        def warmup_and_capture_buckets(self, buckets):
            # Segment spin nodes must see the installed probe during capture;
            # the worker binding is deliberately deferred until normal return.
            assert getattr(self.segment, "installed_probe", None) is (probe if probe_requested else None)
            assert getattr(worker, "_criticality_probe", None) is (None if "release" in events else prior_probe)
            assert worker._criticality_probe_recapture is True
            assert not hasattr(worker, "_decode_graph_ladder_logged")
            events.append(("capture", tuple(buckets)))
            if failure == "capture":
                raise torch.OutOfMemoryError("capture failed")

        def warmup_and_capture_all(self):
            events.append(("capture_all", tuple(self.bucketing.bucket_sizes)))

        def drop_bucket(self, bucket):
            events.append(("drop", bucket))

        def get_capture_stats(self):
            return {"total_capture_time_ms": 0.0}

    class KVManager:
        config = SimpleNamespace(page_size_tokens=64, kv_dtype=torch.bfloat16)

        def ensure_cuda_graph_page_table(self, sequence_ids):
            events.append(("ensure", tuple(sequence_ids)))

        def get_cuda_graph_page_table_storage(self):
            return torch.zeros((1, 4), dtype=torch.int32)

        def get_kv_tensors(self):
            return [torch.zeros((1, 4))], None

        def resolve_physical_layer(self, layer):
            return layer

    module(
        "batchgen.models.openai.gpt_oss_120b.cuda_graph_segments",
        **{name: Segment for name in (
            "FullAttnSegment", "MoESegment", "MoEComputeSegment",
            "SharedMoEBufferPool", "WholeModelSegment",
        )},
    )
    attention = SimpleNamespace(cur_batch=[7], max_seqlen=128)
    module("batchgen.models.wrappers.attention", AttnWrapperBase=attention)
    module(
        "batchgen.models.glm.glm5.whole_model_cuda_graph_segments",
        Glm5WholeModelSegment=Segment,
        make_glm5_whole_model_graph_segment_name=lambda: "whole_model",
    )
    module("batchgen.models.glm.glm5.cuda_graph_segments", Glm5FullDsaAttnSegment=Segment)
    module("batchgen.models.glm.glm5.layer_cuda_graph_segments", Glm5DecoderLayerGraphSegment=Segment)
    module("batchgen.models.glm.glm5.model", Glm5MoE=type("MoE", (), {}), resolve_glm5_3d_mtp=int)
    module(
        "batchgen.models.glm.glm5.moe_cuda_graph_segments",
        Glm5MoEGraphBufferPool=Segment, Glm5MoEGraphSegment=Segment,
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_: events.append("sync"))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: events.append("empty_cache"))
    monkeypatch.setattr(torch.distributed, "barrier", lambda: events.append("barrier"))

    graph = GraphRuntime()
    graph.new_manager = Manager
    indexer = SimpleNamespace(
        rope_head_dim=4, index_topk=8,
        rotary_emb=lambda *args, **kwargs: (None, None),
    )
    wrapper = SimpleNamespace(
        module=SimpleNamespace(indexer=indexer, num_heads=4, rotary_emb=indexer.rotary_emb),
        sinks=None, persistent=False, num_kv_heads=1, head_dim=4,
        _fp8_absorb_weights=object(), _fp8_qkv_a_proj=object(),
        _fp8_qkv_a_scale=object(), _fused_wqb_weights=object(),
        _indexer_cuda_module=object(),
    )
    worker = RecordingWorker(
        events=events, rank=0, world_size=1, model_name="glm5",
        torch_device=torch.device("cpu"),
        args=SimpleNamespace(cuda_graph_max_bucket_size=4, cuda_graph_num_buckets=3),
        model_config=SimpleNamespace(max_position_embeddings=256),
        model=SimpleNamespace(
            model=SimpleNamespace(layers=[SimpleNamespace(self_attn=wrapper, mlp=None)]),
            config=SimpleNamespace(vocab_size=16, hidden_size=4, max_position_embeddings=256),
        ),
        core_engine=SimpleNamespace(), _graph=graph,
        glm5_disable_decode_host_kv_staging=False,
        _criticality_probe_request=object() if probe_requested else None,
        _criticality_probe_recapture=True,
        _current_decode_local_batch_size=1,
        _current_decode_max_rank_batch_size=1,
        _current_decode_rank_token_counts=torch.ones((1,), dtype=torch.int64),
        _decode_graph_bucket_sizes=lambda: [1, 2, 4],
        _max_decode_rank_bsz=lambda: 4,
        _glm5_whole_model_graph_requested_for_current_batch=lambda: True,
        _assert_decode_graph_capture_allowed=lambda: events.append("allowed"),
        model_context_length=256, global_batch=None, _debug_state_values=None,
        _bind_graph_capture_kv_managers=lambda *args: events.append("bind_kv"),
        _bind_graph_capture_kv_manager=lambda *args: events.append("bind_kv"),
        _glm5_whole_model_graph_capture_signature=lambda: (len(events),),
        _glm5_whole_model_graph_timing_requested_for_current_batch=lambda: False,
    )

    if prior_probe is not None:
        worker._criticality_probe = prior_probe
        events.clear()

    def install(*args, **kwargs):
        assert getattr(worker, "_criticality_probe", None) is (None if "release" in events else prior_probe)
        events.append("install")
        if failure == "install":
            raise ValueError("install failed")
        kwargs["whole_model_segment"].installed_probe = probe
        return probe

    helper_names = [
        "_glm5_dsa_graph_score_capacity_tokens", "_deployment_context_bound_tokens",
        "_warn_decode_graph_span_exceeded", "_glm5_dsa_graph_required_tokens",
        "_release_glm5_whole_model_graph_state", "_glm5_whole_model_graph_probe_layers",
        "_make_glm5_whole_model_capture_inputs", "_glm5_whole_model_graph_debug_value",
    ]
    helpers = load_worker_methods(helper_names, {
        "torch": torch, "gc": gc, "logging": logging, "math": math,
        "Sequence": Sequence, "SequenceStatus": SequenceStatus,
    })
    for name, helper in helpers.items():
        if isinstance(helper, staticmethod):
            setattr(worker, name, helper.__func__)
        else:
            setattr(worker, name, MethodType(helper, worker))
    worker._debug_values = lambda: worker._debug_state_values
    real_capture_inputs = worker._make_glm5_whole_model_capture_inputs

    def capture_inputs(**kwargs):
        events.append(("make_inputs", kwargs["bucket"]))
        if failure == "helper":
            raise torch.OutOfMemoryError("input helper failed")
        return real_capture_inputs(**kwargs)

    def record():
        events.append("record")
        graph.decode_graph_captures += 1

    def release(**kwargs):
        events.append("release")
        assert worker._criticality_probe is None
        assert graph.cuda_graph_manager is None
        assert graph.whole_model_segment is None
        assert graph.glm5_whole_model_graph is False

    graph.cuda_graph_adapter = SimpleNamespace(
        install_criticality_probe=install, release_all=release,
        capture_signature=lambda **kwargs: (),
        record_capture=lambda **kwargs: None,
    )
    worker._make_glm5_whole_model_capture_inputs = capture_inputs
    worker._record_decode_graph_capture = record
    setup = load_worker_methods(["_setup_cuda_graphs"], {})["_setup_cuda_graphs"]
    manager = SimpleNamespace(
        primary=KVManager(), auxiliary=KVManager(), config=KVManager.config,
    )
    return worker, lambda: setup(worker, manager), probe


@pytest.mark.parametrize("probe_requested", [False, True])
def test_success_installs_probe_before_capture_and_binds_outcome_after_record(monkeypatch, probe_requested):
    worker, run, probe = _worker(monkeypatch, probe_requested=probe_requested)
    returned = []
    real_setup = worker._graph._setup_cuda_graphs

    def observe_outcome(ctx, gpu_manager, **helpers):
        assert not inspect.isgeneratorfunction(real_setup)
        outcome = real_setup(ctx, gpu_manager, **helpers)
        returned.append(outcome)
        return outcome

    worker._graph._setup_cuda_graphs = observe_outcome
    assert run() is None  # The public worker method keeps its return contract.
    assert returned == [GraphSetupOutcome(
        _decode_graph_ladder_logged=True,
        _criticality_probe=probe if probe_requested else None,
        _criticality_probe_recapture=False,
    )]
    assert [e for e in worker.events if isinstance(e, tuple) and e[0] == "capture"] == [
        ("capture", (4,)), ("capture", (2,)), ("capture", (1,)),
    ]
    assert worker._criticality_probe is (probe if probe_requested else None)
    assert worker._criticality_probe_recapture is False
    assert worker._graph.decode_graph_captures == 1
    binds = [e for e in worker.events if isinstance(e, tuple) and e[0] == "bind"]
    assert binds == [
        ("bind", "_decode_graph_ladder_logged", True),
        ("bind", "_criticality_probe", probe if probe_requested else None),
        ("bind", "_criticality_probe_recapture", False),
    ]
    assert worker.events.index("record") < worker.events.index(binds[0])


@pytest.mark.parametrize("failure", ["capture", "helper"])
def test_oom_rolls_back_graph_but_does_not_bind_outcome_before_reraise(monkeypatch, failure):
    prior_probe = object()
    worker, run, _ = _worker(monkeypatch, failure=failure, prior_probe=prior_probe)
    with pytest.raises(torch.OutOfMemoryError, match="failed"):
        run()
    assert worker._criticality_probe is prior_probe
    assert worker._criticality_probe_recapture is True
    assert not hasattr(worker, "_decode_graph_ladder_logged")
    assert worker._graph.glm5_whole_model_graph_failed_buckets == {4}
    assert worker._graph.cuda_graph_manager is None
    assert worker._graph.whole_model_segment is None
    assert worker._graph.decode_graph_captures == 0
    assert worker.events[-2:] == [("drop", 4), "empty_cache"]
    assert not [e for e in worker.events if isinstance(e, tuple) and e[0] == "bind"]


def test_install_failure_keeps_prior_probe_and_leaves_ladder_unbound(monkeypatch):
    prior_probe = object()
    worker, run, _ = _worker(monkeypatch, failure="install", prior_probe=prior_probe)
    with pytest.raises(ValueError, match="install failed"):
        run()
    assert worker._criticality_probe is prior_probe
    assert worker._criticality_probe_recapture is True
    assert not hasattr(worker, "_decode_graph_ladder_logged")
    assert not [e for e in worker.events if isinstance(e, tuple) and e[0] == "bind"]
    assert "record" not in worker.events


def test_early_return_does_not_create_or_clear_probe_attributes(monkeypatch):
    worker, run, _ = _worker(monkeypatch)
    worker._current_decode_max_rank_batch_size = 0
    assert run() is None
    assert not hasattr(worker, "_criticality_probe")
    assert worker._criticality_probe_recapture is True
    assert worker.events == [("bind", "_decode_graph_ladder_logged", True)]
    run()
    assert worker.events == [("bind", "_decode_graph_ladder_logged", True)]


def test_changed_state_without_armed_recapture_preserves_probe(monkeypatch):
    worker, run, probe = _worker(monkeypatch)
    worker._criticality_probe = probe
    worker._criticality_probe_recapture = False
    worker._graph.cuda_graph_manager = SimpleNamespace(has_bucket_for_all_segments=lambda _: False)
    worker._graph.glm5_whole_model_graph = True
    worker._graph.glm5_whole_model_graph_capture_attempted_for_batch = True
    worker.events.clear()
    run()
    assert worker.events == [("bind", "_decode_graph_ladder_logged", True)]
    assert worker._criticality_probe is probe


def test_gpt_oss_capture_keeps_probe_fields_untouched(monkeypatch):
    worker, run, probe = _worker(monkeypatch)
    worker.model_name = "openai/gpt-oss-120b"
    worker._criticality_probe = probe
    worker._glm5_whole_model_graph_requested_for_current_batch = lambda: False
    worker.events.clear()
    run()
    assert worker.events.index("allowed") < worker.events.index(("capture_all", (1, 2, 4)))
    assert worker.events.index(("capture_all", (1, 2, 4))) < worker.events.index("record")
    assert worker._criticality_probe is probe
    assert worker._criticality_probe_recapture is True
    assert worker._graph.whole_model_graph is True
    assert worker._graph.decode_graph_captures == 1


def test_armed_recapture_releases_before_install_and_clears_after_record(monkeypatch):
    worker, run, probe = _worker(monkeypatch)
    worker._graph.cuda_graph_manager = SimpleNamespace(has_bucket_for_all_segments=lambda _: False)
    worker._graph.glm5_whole_model_graph = True
    worker._graph.glm5_whole_model_graph_capture_attempted_for_batch = True
    run()
    assert worker.events.index("release") < worker.events.index("install")
    assert worker.events.index("install") < worker.events.index("record")
    assert worker._criticality_probe is probe


def test_context_and_outcome_are_frozen_and_context_contains_no_callbacks(monkeypatch):
    worker, run, _ = _worker(monkeypatch)
    seen = []

    def setup(ctx, gpu_manager, **helpers):
        seen.append(ctx)
        assert gpu_manager.primary is not None
        for name, helper in helpers.items():
            assert helper is getattr(worker, name)
        return GraphSetupOutcome()

    worker._graph._setup_cuda_graphs = setup
    run()
    ctx = seen[0]
    assert isinstance(ctx, GraphSetupContext)
    assert ctx._graph is worker._graph
    assert all(getattr(ctx, field.name) is not worker for field in fields(ctx))
    assert all(not callable(getattr(ctx, field.name)) for field in fields(ctx))
    with pytest.raises(FrozenInstanceError):
        ctx.rank = 99
    outcome = GraphSetupOutcome(_criticality_probe=None)
    assert outcome._decode_graph_ladder_logged is GRAPH_SETUP_UNCHANGED
    with pytest.raises(FrozenInstanceError):
        outcome._criticality_probe = object()


@pytest.mark.parametrize("failure", [None, "capture"])
def test_span_warning_helper_keeps_its_live_cache_even_on_failure(monkeypatch, failure):
    worker, run, _ = _worker(monkeypatch, failure=failure)
    attention = sys.modules["batchgen.models.wrappers.attention"].AttnWrapperBase
    attention.max_seqlen = 512
    if failure:
        with pytest.raises(torch.OutOfMemoryError, match="failed"):
            run()
        assert not hasattr(worker, "_criticality_probe")
        assert not hasattr(worker, "_decode_graph_ladder_logged")
    else:
        run()
        assert [e[1] for e in worker.events if isinstance(e, tuple) and e[0] == "bind"] == [
            "_decode_graph_ladder_logged",
            "_criticality_probe", "_criticality_probe_recapture",
        ]
    from batchgen.worker.step_observability_engine import StepObservabilityEngine
    assert isinstance(worker._step_observability_engine, StepObservabilityEngine)
    # The restored helper binds its cache and warning state before capture.
    assert worker._graph.decode_graph_span_warn_monotonic is not None


def test_existing_graph_and_signature_return_without_touching_probe(monkeypatch):
    prior_probe = object()
    worker, run, _ = _worker(monkeypatch, prior_probe=prior_probe)
    manager = SimpleNamespace(has_bucket_for_all_segments=lambda _: True)
    worker._graph.cuda_graph_manager = manager
    worker._graph.glm5_whole_model_graph = True
    worker._graph.glm5_whole_model_graph_signature = ("same",)
    worker._glm5_whole_model_graph_capture_signature = lambda: ("same",)
    run()
    assert worker._graph.cuda_graph_manager is manager
    assert worker._graph.glm5_whole_model_graph_capture_attempted_for_batch is True
    assert worker._criticality_probe is prior_probe
    assert worker._criticality_probe_recapture is True
    assert worker.events == [("bind", "_decode_graph_ladder_logged", True)]


def test_real_capture_guard_reads_recapture_before_its_deferred_clear(monkeypatch):
    worker, run, _ = _worker(monkeypatch)
    guard = load_worker_methods(["_assert_decode_graph_capture_allowed"], {})[
        "_assert_decode_graph_capture_allowed"
    ]
    worker.parallel_manager = SimpleNamespace(holds_instance=lambda _: True)
    worker._graph.decode_graph_captures = 1
    worker._assert_decode_graph_capture_allowed = lambda: guard(worker)
    run()
    assert worker._graph.decode_graph_captures == 2
    assert worker._criticality_probe_recapture is False
    with pytest.raises(RuntimeError, match="captured once"):
        guard(worker)


@pytest.mark.parametrize("value, expected", [(None, ()), ("all", (0, 1)), ("1, 0", (1, 0)), ([1], (1,))])
def test_probe_layers_use_the_worker_debug_values(monkeypatch, value, expected):
    worker, run, _ = _worker(monkeypatch)
    worker._debug_state_values = {"glm5_whole_model_graph_probe_layers": value}
    seen = []
    real = worker._glm5_whole_model_graph_probe_layers

    def observe(num_layers):
        seen.append(real(2))
        return seen[-1]

    worker._glm5_whole_model_graph_probe_layers = observe
    run()
    assert seen == [expected]
