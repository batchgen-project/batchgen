"""Default-lifecycle decode graph capture happens before readiness.

Invariants pinned here:

* On the default lifecycle, ``prepare_deployment`` hands the startup
  transient probe a capture callable exactly when the deployment requests
  whole-model decode graphs AND the strategy manager retains its decode
  instance across the startup switch to prefill (the worker's own exit-mode
  decision suspends it). The capture runs once, before readiness, and the
  probe publishes the capture flavor as a measured value.
* The first decode configure reuses those graphs: across the startup capture
  and two decode intervals the graph setup runs once and the per-process
  capture counter reads one. Re-entering setup with the graphs present
  captures nothing; with them torn down it raises the capture-once guard.
* A deployment whose manager does not retain the decode instance (a
  build-always manager, a rank that streams part of its routed experts) and
  every eager deployment pass no capture and publish the existing named
  unavailable capture flavor, in the same startup order as before.
* A requested startup capture that produces no graph fails the startup
  instead of silently moving the capture back to the first decode configure.

The worker module cannot be imported on a CPU-only machine, so its methods
are compiled from the real source (``tests.worker_method_source``).
"""

from __future__ import annotations

import logging
import os
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Callable, Optional

import pytest

import batchgen.cuda_graph as cuda_graph_module
from batchgen.config.cuda_graph_flag_policy import (
    cli_enables_whole_model_graph,
    glm5_whole_model_cuda_graph_requested,
)
from batchgen.config.model_name_utils import is_kimi_k25_backend_model
from batchgen.cuda_graph import BatchSizeBucketing
from batchgen.cuda_graph.buckets import check_decode_width_within_buckets
from batchgen.models.glm.glm5.cuda_graph_policy import (
    glm5_whole_model_cuda_graph_requested_for_model,
)
from batchgen.models.wrappers import AttnWrapperBase
from batchgen.phase_lifecycle import ExitMode, PhaseKey
from batchgen.worker import startup_transient as st
from tests.unit.test_startup_transient import FakeCuda
from tests.worker.worker_fakes import make_graph_state
from tests.worker_method_source import load_worker_methods

MiB = 1024 * 1024
GLM5 = "zai-org/GLM-5.3-FP8"
GPT_OSS = "openai/gpt-oss-120b"
K25 = "moonshotai/Kimi-K2.5"
BUCKETS = [1, 2, 4, 8]
NO_STARTUP_CAPTURE = "this lifecycle captures no decode graph at startup"

_METHOD_NAMES = [
    "prepare_deployment",
    "_startup_decode_graph_capture",
    "_capture_startup_decode_graphs",
    "_phase_exit_mode",
    "_probe_startup_transient",
    "_decode_graph_state_present",
    "_assert_decode_graph_state_kept",
    "_warmup_cuda_graphs",
    "_setup_cuda_graphs",
    "_decode_graph_capture_needed_at_configure",
    "_assert_decode_graph_capture_allowed",
    "_record_decode_graph_capture",
    "_glm5_whole_model_graph_current_bucket_missing",
    "_whole_model_decode_graphs_requested",
    "_glm5_whole_model_graph_requested_for_current_batch",
]


def _load(cuda=None, dist=None):
    torch = SimpleNamespace(cuda=cuda if cuda is not None else SimpleNamespace())
    torch.cuda.is_available = lambda: cuda is not None
    return load_worker_methods(_METHOD_NAMES, {
        "os": os,
        "logging": logging,
        "Optional": Optional,
        "Callable": Callable,
        "torch": torch,
        "dist": dist or SimpleNamespace(is_initialized=lambda: False),
        "_startup_transient": st,
        "ExitMode": ExitMode,
        "PhaseKey": PhaseKey,
        "AttnWrapperBase": AttnWrapperBase,
        "is_kimi_k25_backend_model": is_kimi_k25_backend_model,
        "cli_enables_whole_model_graph": cli_enables_whole_model_graph,
        "glm5_whole_model_cuda_graph_requested_for_model": (
            glm5_whole_model_cuda_graph_requested_for_model
        ),
        # rb5: the branch re-homed the predicate behind cuda_graph_flag_policy
        # (S4b); the body reads this spelling, not the _for_model one.
        "glm5_whole_model_cuda_graph_requested": glm5_whole_model_cuda_graph_requested,
        "check_decode_width_within_buckets": check_decode_width_within_buckets,
    })


class Manager:
    """The strategy-manager surface the startup seam reads."""

    def __init__(self, retains_decode: bool, *, probe_step=None):
        self.retains_decode = retains_decode
        if probe_step is not None:
            self.startup_decode_probe_step = probe_step

    def holds_instance(self, phase):
        if phase not in ("prefill", "decode"):
            raise ValueError(phase)
        return phase == "decode" and self.retains_decode


class FakeGraphManager:
    """A captured ladder: every bucket of it is present for every segment."""

    def __init__(self, bucketing, device=None):
        self.bucketing = bucketing

    def has_bucket_for_all_segments(self, batch_size):
        return True


def _record_capture(worker, gpu_manager):
    """What the real whole-model capture tail of ``_setup_cuda_graphs`` leaves
    set once its ladder is captured (manager, flags, signature, counter)."""
    bucketing = BatchSizeBucketing(BUCKETS)
    worker._graph.cuda_graph_manager = FakeGraphManager(bucketing)
    worker._graph.whole_model_graph = True
    worker._graph.glm5_whole_model_graph = True
    worker._graph.whole_model_bucketing = bucketing
    worker._graph.whole_model_segment = SimpleNamespace(max_seqlen=1 << 17)
    worker._graph.glm5_whole_model_graph_capture_attempted_for_batch = True
    worker._graph.glm5_whole_model_graph_signature = worker._glm5_whole_model_graph_capture_signature()
    worker._graph.decode_graph_captures += 1


def _worker(model_name, *, retains_decode, graphs, cuda=None, dist=None,
            capture_produces_graphs=True, probe_step=None):
    """A worker whose startup and graph-decision methods are the real ones.

    ``_setup_cuda_graphs`` is a spy: its first call records what the real
    capture tail leaves behind; every later call runs the REAL method, which
    must never capture again for a retained instance.
    """
    methods = _load(cuda=cuda, dist=dist)
    events = []
    real_setup = methods.pop("_setup_cuda_graphs")
    Worker = type("Worker", (), dict(methods))
    worker = Worker()

    def setup(gpu_manager):
        worker.setup_calls += 1
        events.append("graph_setup")
        if worker.setup_calls == 1:
            if capture_produces_graphs:
                _record_capture(worker, gpu_manager)
            return
        real_setup(worker, gpu_manager)

    def load_decode_model(comm=None):
        events.append("decode_build")
        worker._live_phase_key = PhaseKey(
            "decode", worker._prefill_moe_mode_in_effect, ())
        worker._phase = "decode"
        worker._decode_padding_bsz = 8

    def transition(key):
        # The real _transition_phase asks the same exit-mode question.
        events.append(("to_prefill", worker._phase_exit_mode(worker._live_phase_key, key)))
        worker._live_phase_key = key

    def sync_counts(batch, *, reason):
        events.append(("sync_counts", reason))
        worker._current_decode_local_batch_size = len(batch)
        worker._current_decode_max_rank_batch_size = 0

    worker.__dict__.update(
        rank=0, local_rank=0, world_size=1, comm=object(), model_name=model_name,
        args=SimpleNamespace(
            enable_cuda_graph=graphs, disable_cuda_graphs=not graphs,
            cuda_graph_max_bucket_size=8, cuda_graph_num_buckets=4,
        ),
        engine_config=SimpleNamespace(Basic_Config=SimpleNamespace(enable_cuda_graphs=graphs)),
        parallel_manager=Manager(retains_decode, probe_step=probe_step),
        torch_device="cpu", model_context_length=4096, startup_transient=None,
        _core_initialized=True, _live_phase_key=None, _phase=None,
        _prefill_moe_mode_in_effect=None, _decode_padding_bsz=None,
        _decode_graph_ladder_logged=True,
        # S2: the decode-graph state is GraphRuntime's (its defaults are the
        # former None/False/0/set() fixtures).
        _graph=make_graph_state(),
        _criticality_probe_recapture=False,
        _current_decode_local_batch_size=0, _current_decode_max_rank_batch_size=0,
        setup_calls=0, events=events,
        _startup_phase=lambda name: nullcontext(),
        _ensure_pynccl_communicator=lambda: events.append("comm"),
        _persistent_phase_enabled=lambda: False,
        _desired_phase_key=lambda phase: PhaseKey(phase, "streamed", ()),
        _load_decode_model=load_decode_model,
        _init_gpu_kv_with_actual_size=lambda: events.append("pool_sized"),
        _transition_phase=transition,
        set_phase=lambda phase: setattr(worker, "_phase", phase),
        _install_prefill_weight_copy_pipeline=lambda flag: None,
        _sync_decode_moe_rank_counts=sync_counts,
        _bind_decode_attention_metadata_for_graph_config=(
            lambda batch: events.append(("bind_metadata", tuple(batch)))),
        _get_cuda_graph_gpu_manager=lambda: object(),
        _finalize_gpu_kv_pool_after_capture=lambda gpu_manager: events.append("pool_finalized"),
        _setup_cuda_graphs=setup,
        model_config=SimpleNamespace(),
        _glm5_whole_model_graph_capture_signature=lambda bucket=None: ("page-tables", None),
        _decode_graph_bucket_sizes=lambda: list(BUCKETS),
        _max_decode_rank_bsz=lambda: 8,
    )
    return worker


def _decode_interval(worker, *, max_rank_batch):
    """One decode configure of the reuse lifecycle, as generate() runs it:
    the reuse switch asserts the graphs survived, then the real warmup
    trigger decides whether this configure captures."""
    worker._assert_decode_graph_state_kept("prefill->decode")
    worker._current_decode_local_batch_size = max_rank_batch
    worker._current_decode_max_rank_batch_size = max_rank_batch
    if worker._decode_graph_capture_needed_at_configure(has_queueing=False):
        worker._warmup_cuda_graphs()


@pytest.fixture(autouse=True)
def _attention_statics(monkeypatch):
    monkeypatch.setattr(AttnWrapperBase, "max_seqlen", 512, raising=False)
    monkeypatch.setattr(AttnWrapperBase, "cur_batch", [], raising=False)
    monkeypatch.setattr(cuda_graph_module, "CUDAGraphManager", FakeGraphManager)


# ---- the retained GLM-5 instance captures at startup, once --------------------

def test_retained_instance_captures_inside_the_probe_before_the_prefill_switch():
    worker = _worker(GLM5, retains_decode=True, graphs=True)
    worker.prepare_deployment()
    assert worker.events == [
        "comm", "decode_build", "pool_sized",
        ("sync_counts", "startup_graph_capture"), ("bind_metadata", ()),
        "graph_setup", "pool_finalized",
        ("to_prefill", ExitMode.SUSPEND),
    ]
    assert worker._graph.decode_graph_captures == 1
    assert worker._graph.cuda_graph_manager is not None
    # No CUDA here: the capture still ran exactly once, and the flavor names why
    # it carries no value.
    assert worker.startup_transient[1]["flavor"] == st.FLAVOR_CAPTURE
    assert worker.startup_transient[1]["seam"] == st.SEAM_DEPLOYMENT_PREPARE
    assert worker.startup_transient[1]["unavailable"] == "CUDA is unavailable"


def test_first_two_decode_intervals_reuse_the_startup_graphs():
    worker = _worker(GLM5, retains_decode=True, graphs=True)
    worker.prepare_deployment()
    _decode_interval(worker, max_rank_batch=3)
    _decode_interval(worker, max_rank_batch=8)
    assert worker.setup_calls == 1
    assert worker._graph.decode_graph_captures == 1
    assert worker.events.count("graph_setup") == 1


def test_without_the_startup_capture_the_first_configure_would_capture():
    # Control for the test above: the trigger is live, so "no capture at the
    # first configure" is the startup capture's doing, not a dead trigger.
    worker = _worker(GLM5, retains_decode=True, graphs=True)
    worker._current_decode_max_rank_batch_size = 3
    assert worker._decode_graph_capture_needed_at_configure(has_queueing=False)


def test_real_setup_reentered_with_the_graphs_present_captures_nothing():
    worker = _worker(GLM5, retains_decode=True, graphs=True)
    worker.prepare_deployment()
    worker._current_decode_local_batch_size = 0
    worker._current_decode_max_rank_batch_size = 2
    worker._warmup_cuda_graphs()
    assert worker.setup_calls == 2
    assert worker._graph.decode_graph_captures == 1


def test_real_setup_refuses_to_recapture_a_torn_down_retained_instance():
    worker = _worker(GLM5, retains_decode=True, graphs=True)
    worker.prepare_deployment()
    worker._graph.cuda_graph_manager = None
    worker._graph.glm5_whole_model_graph = False
    worker._graph.whole_model_segment = None
    # The reuse switch names the teardown first ...
    with pytest.raises(RuntimeError, match="tore down the captured decode graph state"):
        worker._assert_decode_graph_state_kept("prefill->decode")
    # ... and should a configure still reach the setup, it refuses to capture.
    worker._current_decode_max_rank_batch_size = 2
    assert worker._decode_graph_capture_needed_at_configure(has_queueing=False)
    with pytest.raises(RuntimeError, match="its graphs are captured once per process"):
        worker._warmup_cuda_graphs()
    assert worker.setup_calls == 2
    assert worker._graph.decode_graph_captures == 1


def test_capture_flavor_is_measured_on_the_default_lifecycle():
    cuda = FakeCuda()
    gathered = []

    def all_gather_object(out, local):
        gathered.append(local)
        out[:] = [local]

    worker = _worker(
        GLM5, retains_decode=True, graphs=True, cuda=cuda,
        dist=SimpleNamespace(is_initialized=lambda: True, all_gather_object=all_gather_object),
        probe_step=lambda padding_bsz: cuda.transient(16 * MiB),
    )
    original = worker._warmup_cuda_graphs

    def capture_with_footprint():
        cuda.transient(64 * MiB, reserved_growth=32 * MiB, external_growth=8 * MiB)
        original()

    worker._warmup_cuda_graphs = capture_with_footprint
    worker.prepare_deployment()
    eager, captured = worker.startup_transient
    assert len(gathered) == 1
    assert captured["flavor"] == st.FLAVOR_CAPTURE
    assert captured["seam"] == st.SEAM_DEPLOYMENT_PREPARE
    assert captured["padding_bsz"] == 8
    assert captured["unavailable"] is None
    assert captured["allocator_gib"] * st.GIB == 64 * MiB
    assert captured["external_gib"] * st.GIB == 8 * MiB
    assert eager["unavailable"] is None
    assert worker._graph.decode_graph_captures == 1


def test_a_requested_startup_capture_that_produces_no_graph_fails_the_startup():
    worker = _worker(GLM5, retains_decode=True, graphs=True, capture_produces_graphs=False)
    with pytest.raises(RuntimeError, match="startup decode graph capture"):
        worker.prepare_deployment()
    # The worker never reached the prefill switch, so it never reports ready.
    assert not any(isinstance(e, tuple) and e[0] == "to_prefill" for e in worker.events)


def test_the_seam_refuses_to_decide_outside_the_startup_decode_phase():
    worker = _worker(GLM5, retains_decode=True, graphs=True)
    with pytest.raises(RuntimeError, match="live decode phase"):
        worker._startup_decode_graph_capture(PhaseKey("prefill", "streamed", ()))


# ---- refusal matrix: no retained instance or no graphs -> today's behavior -----

@pytest.mark.parametrize(
    "model_name, retains_decode, graphs, exit_mode",
    [
        # GLM-5 rank that streams part of its routed experts: no held instance.
        (GLM5, False, True, ExitMode.RETIRE),
        # Build-always managers under --enable-cuda-graph.
        (GPT_OSS, False, True, ExitMode.RETIRE),
        (K25, False, True, ExitMode.RETIRE),
        # Eager deployments, retained or not.
        (GLM5, True, False, ExitMode.SUSPEND),
        (GPT_OSS, False, False, ExitMode.RETIRE),
    ],
)
def test_no_startup_capture_without_a_retained_instance_and_requested_graphs(
    model_name, retains_decode, graphs, exit_mode,
):
    worker = _worker(model_name, retains_decode=retains_decode, graphs=graphs)
    worker.prepare_deployment()
    assert worker.events == [
        "comm", "decode_build", "pool_sized", ("to_prefill", exit_mode),
    ]
    assert worker.setup_calls == 0
    assert worker._graph.decode_graph_captures == 0
    assert worker._graph.cuda_graph_manager is None
    assert worker.startup_transient[1] == st.StartupTransient.missing(
        st.FLAVOR_CAPTURE, st.SEAM_DEPLOYMENT_PREPARE, 8, NO_STARTUP_CAPTURE,
    ).as_dict()


@pytest.mark.parametrize("model_name", [GLM5, GPT_OSS])
def test_a_rebuilt_instance_keeps_capturing_at_its_first_decode_configure(model_name):
    # The lazy capture stays where it was for a manager that rebuilds its
    # decode instance: the first configure's trigger still fires.
    worker = _worker(model_name, retains_decode=False, graphs=True)
    worker.prepare_deployment()
    worker._current_decode_max_rank_batch_size = 3
    assert worker._decode_graph_capture_needed_at_configure(has_queueing=False)
