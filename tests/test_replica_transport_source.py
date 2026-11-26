"""CPU-only source checks for the node replica weight transport (slice 2).

The producer consumes the installed plan with one record state machine:
HostPrivate is the historic flow byte for byte, HostShared adds publish and
the peer-gated release after landing, Replica pulls the owner's landed slot
over the per-owner stream after barrier A. The uninstalled engine is the
single-member group, and no code path branches on group size at copy time.
"""

import ast
import dataclasses
import logging
import os
import socket
import sys
import time
import types
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import pytest


ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "core/HtoD_Engine/HtoD_Engine.cu"
BUFFER = ROOT / "core/GPU_Weight_Buffer/GPU_Weight_Buffer.cpp"
WORKER = ROOT / "batchgen/batchgen_worker.py"
# The install body is a module-level function of a frozen context since D1;
# the worker keeps a same-name delegator that builds the context and binds
# the returned outcome.
TRANSPORT_ENGINE = ROOT / "batchgen/worker/replica_transport_engine.py"


def _producer(source: str) -> str:
    return source[source.index("void HtoD_Engine::HtoD_Worker()"):]


def test_replica_pull_waits_for_owner_barrier_a_with_bounded_stall():
    body = _producer(ENGINE.read_text())
    gate = body.index("Barrier::A)")
    assert body.index("GPU_Weight_Buffer::kStallTimeout", gate) > gate
    assert "has not passed barrier A of frame" in body
    # the wait is a polled state, never a blocking sleep in the producer
    assert "progressed = false;" in body


def test_replica_pull_is_one_device_copy_on_the_owner_stream():
    body = _producer(ENGINE.read_text())
    pull = body.index("plan.pull_streams[role->owner_member]")
    copy = body.index("cudaMemcpyDeviceToDevice", pull - 2000)
    assert abs(copy - pull) < 2500
    # L-E2: no blocking sync per record; the landing waits on the copy's
    # event so the stream stays bandwidth-bound (the per-record sync made
    # the wall track module count; ABBA 2026-11-11).
    replica_block = body[body.index("Pull the owner's landed slot"):
                         body.index("set_producer_state_(ProducerState::kAcquiring")]
    assert "cudaStreamSynchronize" not in replica_block
    assert "cudaEventRecord(landing.event, stream)" in replica_block
    # and never acquires host storage or a daemon lease on this path
    assert "get_module_weights_storage" not in replica_block
    assert "release_module" not in replica_block


def test_host_shared_landing_publishes_arms_gate_and_posts_frames():
    source = ENGINE.read_text()
    # L-E2: the landing actions run in complete_landing_ once the module's
    # copy event is done; publish/marks never precede the copied bytes.
    body = source[source.index("void HtoD_Engine::complete_landing_"):
                  source.index("bool HtoD_Engine::drain_landings_")]
    sync = body.index("cudaEventSynchronize(landing.event)")
    complete = body.index("weights_copy_complete(")
    assert sync < complete
    assert "arm_peer_release_gate" in body
    assert "publish_slot" in body
    assert "replica_mark_owned_landed_" in body
    assert "replica_mark_terminal_" in body
    assert complete < body.index("publish_slot")
    # produce-only (HostShared skip or release-on-landing) fails loud
    assert "produce-only" in body or "produce-only" in _producer(source)


def test_failed_producer_raises_its_abort_word():
    source = ENGINE.read_text()
    fail = source[source.index("void HtoD_Engine::fail_producer_"):]
    assert fail.index("block.abort(") < fail.index("record_producer_failure")


def test_reclaim_keeps_peer_gated_slots_pending():
    source = BUFFER.read_text()
    body = source[source.index("void GPU_Weight_Buffer::reclaimCompletedReleasesLocked()"):
                  source.index("void GPU_Weight_Buffer::reset_weight_stream_profile")]
    assert "peer_release_gate_" in body
    assert "still_gated" in body
    assert "it->event = nullptr;" in body
    drain = source[source.index("void GPU_Weight_Buffer::drainPendingReleasesLocked()"):
                   source.index("void GPU_Weight_Buffer::bumpProgressLocked()")]
    # teardown forces reclaim after the group barrier
    assert "[](const std::string&, int64_t) { return true; }" in drain


def test_worker_installs_between_queue_set_and_producer_start():
    source = WORKER.read_text()
    body = source[source.index("def _install_prefill_weight_copy_pipeline"):]
    set_queue = body.index("set_weight_copy_queue(self.weight_copy_task)")
    install = body.index("_install_replica_transport()")
    start = body.index("start_h2d_worker()")
    assert set_queue < install < start


def test_worker_tears_down_with_a_group_barrier_before_uninstall():
    source = WORKER.read_text()
    body = source[source.index("def _uninstall_replica_transport"):]
    barrier = body.index("dist.barrier()")
    uninstall = body.index("uninstall_replica_plan()")
    assert barrier < uninstall
    exit_body = source[source.index("def _prepare_phase_exit"):]
    assert exit_body.index("stop_h2d_worker()") < exit_body.index(
        "_uninstall_replica_transport()"
    )


def test_ring_segments_are_shared_by_vmm_fd_not_torch_ipc():
    """Slice 2c: torch CUDA-IPC rebuild enabled whole-device peer access and
    left foreign contexts (minutes and GiBs on a serving worker); segments
    now travel as exported VMM fds over SCM_RIGHTS and import maps only the
    segment range read-only."""
    engine_src = TRANSPORT_ENGINE.read_text()
    install = engine_src[engine_src.index("def _install_replica_transport"):]
    assert "reduce_tensor" not in install
    assert "ring_segment_export_fd" in install
    assert "send_fds" in install and "recv_fds" in install
    # every fd is closed after the engine imported the mappings
    assert install.index("install_replica_plan(") < install.index("os.close")
    engine = ENGINE.read_text()
    assert "VmmPeerImport" in engine
    assert "cudaIpcOpenMemHandle" not in engine
    vmm = (ROOT / "core/GPU_Weight_Buffer/vmm_ring_segment.h").read_text()
    assert "CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR" in vmm
    # the importer grants only its own device read access to the one range
    imp = vmm[vmm.index("class VmmPeerImport"):]
    assert "CU_MEM_ACCESS_FLAGS_PROT_READ;" in imp
    assert "cuMemUnmap" in imp  # uninstall returns the mapping


def test_barrier_words_are_per_member_and_type():
    """The producer interleaves types whose rings run at different depths
    (attn ring 2 trails the forward; routed prefetches frames), so frame
    completion is monotone only within a type. A single per-member word
    regressed on GLM dense frames: the owner landed routed frame 3 before
    attn frame 2 and every attn_2 consumer read "owner passed barrier A
    without publishing" while the owner threw "cannot move backwards"."""
    block = (ROOT / "core/GPU_Weight_Buffer/replica_control_block.h").read_text()
    assert "void advance(size_t member, size_t type, Barrier barrier" in block
    assert "size_t num_types() const;" in block
    engine = ENGINE.read_text()
    assert "owned_per_type_pos[type]" in engine
    assert "total_per_type_pos[type]" in engine
    # the pull gate reads the owner's word for the record's own type
    assert "role->owner_member), type,\n" in engine or \
        "static_cast<size_t>(role->owner_member), type," in engine


def test_worker_declines_rather_than_crashes_non_participants():
    source = TRANSPORT_ENGINE.read_text()
    body = source[source.index("def _install_replica_transport"):]
    for reason in (
        "daemon-backed weight source",
        "hierarchical_gdr weight source",
        "does not fit the layer grammar",
        "replica group has one member",
        "two-frame rendezvous needs",
    ):
        assert reason in body, reason
    assert "raise RuntimeError" not in body.split("ring capacity")[1][:400]


def test_produce_only_pass_alignment_is_wired():
    """Slice 2b: a rank short of the group's pass count pads with
    produce-only passes — it publishes its owned records, declines its
    pulls, and never strands peers on barrier A (the BCT deadlock)."""
    table = (ROOT / "core/GPU_Weight_Buffer/weight_slot_table.h").read_text()
    assert "void mark_produce_only(int64_t pass)" in table
    assert "void decline_produce_only" in table
    assert "produce_only_passes_.erase(pass)" in table
    assert "produce_only_passes_.clear()" in table

    buffer = (ROOT / "core/GPU_Weight_Buffer/GPU_Weight_Buffer.cpp").read_text()
    assert "begin_pass(bool produce_only)" in buffer
    assert "release_unconsumed" in buffer
    # Pass-end drain honors the peer gate instead of force-reclaiming slots
    # a slower member may still be reading.
    assert "drainPendingReleasesGatedLocked(lock)" in buffer

    engine = (ROOT / "core/HtoD_Engine/HtoD_Engine.cu").read_text()
    assert "pass_produce_only(next.pass)" in engine
    assert "decline_produce_only" in engine
    assert "release_unconsumed" in engine

    worker = WORKER.read_text()
    assert 'replica = getattr(self, "_replica_plan_installed", False)' in worker
    assert "produce_only=replica_padding" in worker
    assert "cannot" in worker and "_carried_produce_only" in worker


# --------------------------------------------------------------------------- #
#  D1: the install is a function of its frozen context (no stand-in worker)
# --------------------------------------------------------------------------- #

CONTEXT_FIELDS = {
    "rank",
    "world_size",
    "local_rank",
    "local_world_size",
    "parallel_manager",
    "daemon_weight_source",
    "cross_node_broadcast_plan",
    "weight_copy_task",
    "weight_copy_fingerprint",
    "core_engine",
    "num_prefill_module_buffer",
    "replica_block_cache",
    "shm_prefix",
}


def _load_transport(dist):
    """Compile the context, the outcome and the install from the real source.

    The engine module imports torch and the native core engine at module
    scope, so a CPU test lifts the three definitions by AST instead; the
    ``dist`` stub stands in for ``torch.distributed`` and ``torch`` itself is
    never reached on the decline paths these tests drive.
    """
    tree = ast.parse(TRANSPORT_ENGINE.read_text(), filename=str(TRANSPORT_ENGINE))
    wanted = {
        "ReplicaTransportContext",
        "ReplicaTransportInstallOutcome",
        "_install_replica_transport",
    }
    nodes = [node for node in tree.body if getattr(node, "name", None) in wanted]
    assert {node.name for node in nodes} == wanted
    namespace = {
        "dataclass": dataclasses.dataclass,
        "Any": Any,
        "Callable": Callable,
        "Optional": Optional,
        "Tuple": Tuple,
        "logging": logging,
        "os": os,
        "socket": socket,
        "time": time,
        "torch": None,
        "dist": dist,
        "core_engine": None,
    }
    module = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(TRANSPORT_ENGINE), "exec"), namespace)  # noqa: S102
    return namespace


@pytest.fixture
def fake_worker_module(monkeypatch):
    """The install's late ``allocate_node_shared_tensor`` import, unreachable
    on every decline path these CPU tests drive."""
    import batchgen  # noqa: F401  (parent package; CPU-importable)

    module = types.ModuleType("batchgen.batchgen_worker")

    def allocate_node_shared_tensor(**_kwargs):
        raise AssertionError("decline paths never allocate the control block")

    module.allocate_node_shared_tensor = allocate_node_shared_tensor
    monkeypatch.setitem(sys.modules, "batchgen.batchgen_worker", module)


def _context(namespace, **overrides):
    values = dict(
        rank=0,
        world_size=2,
        local_rank=0,
        local_world_size=2,
        parallel_manager=types.SimpleNamespace(
            _distributed_weight_transport=None
        ),
        daemon_weight_source=False,
        cross_node_broadcast_plan=None,
        weight_copy_task={"attention": ["layers.0.attn"]},
        weight_copy_fingerprint="fp",
        core_engine=None,
        num_prefill_module_buffer={},
        replica_block_cache={},
        shm_prefix="t_",
    )
    values.update(overrides)
    return namespace["ReplicaTransportContext"](**values)


def _no_exchange(_pid, _fd):
    raise AssertionError("decline paths never exchange a memfd identity")


def test_install_declines_without_distributed(fake_worker_module, caplog):
    namespace = _load_transport(
        types.SimpleNamespace(is_initialized=lambda: False)
    )
    ctx = _context(namespace)
    with caplog.at_level(logging.INFO):
        outcome = namespace["_install_replica_transport"](ctx, _no_exchange)
    assert outcome.replica_plan_installed is False
    assert outcome.staggered_weight_copy_task is None
    assert "torch.distributed is not initialized" in caplog.text


def test_install_declines_daemon_weight_source(fake_worker_module):
    namespace = _load_transport(
        types.SimpleNamespace(is_initialized=lambda: True)
    )
    ctx = _context(namespace, daemon_weight_source=True)
    outcome = namespace["_install_replica_transport"](ctx, _no_exchange)
    assert outcome.replica_plan_installed is False
    assert outcome.staggered_weight_copy_task is None


def test_install_declines_hierarchical_source_without_seam(fake_worker_module):
    namespace = _load_transport(
        types.SimpleNamespace(is_initialized=lambda: True)
    )
    ctx = _context(
        namespace,
        parallel_manager=types.SimpleNamespace(
            _distributed_weight_transport="hierarchical_gdr"
        ),
    )
    outcome = namespace["_install_replica_transport"](ctx, _no_exchange)
    assert outcome.replica_plan_installed is False


def test_install_declines_task_outside_the_layer_grammar(fake_worker_module):
    # Real grammar, real decline: an empty task has no frame plan.
    namespace = _load_transport(
        types.SimpleNamespace(is_initialized=lambda: True)
    )
    ctx = _context(namespace, weight_copy_task={})
    outcome = namespace["_install_replica_transport"](ctx, _no_exchange)
    assert outcome.replica_plan_installed is False


def test_seam_requested_without_a_plan_raises(fake_worker_module):
    # Under the seam a silent decline would strand peers on records this
    # member never fetches, so the install fails loud instead.
    namespace = _load_transport(
        types.SimpleNamespace(is_initialized=lambda: False)
    )
    ctx = _context(
        namespace,
        parallel_manager=types.SimpleNamespace(
            _distributed_weight_transport="hierarchical_gdr",
            cross_seam_enabled=lambda: True,
        ),
        cross_node_broadcast_plan=None,
    )
    with pytest.raises(RuntimeError, match="cross-node replica seam requested"):
        namespace["_install_replica_transport"](ctx, _no_exchange)


def test_context_is_frozen_and_declares_exactly_the_install_reads(
    fake_worker_module,
):
    namespace = _load_transport(
        types.SimpleNamespace(is_initialized=lambda: False)
    )
    context_type = namespace["ReplicaTransportContext"]
    assert {f.name for f in dataclasses.fields(context_type)} == CONTEXT_FIELDS
    ctx = _context(namespace)
    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.rank = 1
    outcome_type = namespace["ReplicaTransportInstallOutcome"]
    assert {f.name for f in dataclasses.fields(outcome_type)} == {
        "replica_plan_installed",
        "staggered_weight_copy_task",
    }


def test_engine_file_never_names_self_or_worker():
    """D1 permanence: the engine reads a context, never a worker.

    The boundary checker enforces the same property repo-wide; this local
    guard fails the suite the moment a ``self``/``worker`` parameter or a
    ``self = worker`` rebinding comes back to this file.
    """
    tree = ast.parse(TRANSPORT_ENGINE.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.arg):
            assert node.arg not in {"self", "worker"}, ast.dump(node)
        if isinstance(node, ast.Name):
            assert node.id not in {"self", "worker"}, node.lineno


def test_worker_delegator_builds_the_context_and_binds_the_outcome():
    source = WORKER.read_text()
    body = source[source.index("def _install_replica_transport"):]
    body = body[:body.index("\n\tdef ")]
    build = body.index("ReplicaTransportContext(")
    call = body.index("outcome = _install_replica_transport(")
    assert build < call
    # The one value the engine needs mid-install travels as an explicit
    # argument, never as a context field holding a bound worker method.
    assert "exchange=self._exchange_node_memfd_identity" in body
    # The delegator performs exactly the binds the worker method used to:
    # the staggered schedule and the installed flag.
    stagger_bind = body.index(
        "self.weight_copy_task = outcome.staggered_weight_copy_task"
    )
    flag_bind = body.index("self._replica_plan_installed = True")
    assert call < stagger_bind < flag_bind
