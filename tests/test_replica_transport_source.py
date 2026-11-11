"""CPU-only source checks for the node replica weight transport (slice 2).

The producer consumes the installed plan with one record state machine:
HostPrivate is the historic flow byte for byte, HostShared adds publish and
the peer-gated release after landing, Replica pulls the owner's landed slot
over the per-owner stream after barrier A. The uninstalled engine is the
single-member group, and no code path branches on group size at copy time.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "core/HtoD_Engine/HtoD_Engine.cu"
BUFFER = ROOT / "core/GPU_Weight_Buffer/GPU_Weight_Buffer.cpp"
WORKER = ROOT / "batchgen/batchgen_worker.py"


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
    # paced like the host path: the producer synchronizes its own copy
    assert body.index("cudaStreamSynchronize(stream)", pull) - pull < 800
    # and never acquires host storage or a daemon lease on this path
    replica_block = body[body.index("Pull the owner's landed slot"):
                         body.index("set_producer_state_(ProducerState::kAcquiring")]
    assert "get_module_weights_storage" not in replica_block
    assert "release_module" not in replica_block


def test_host_shared_landing_publishes_arms_gate_and_posts_frames():
    body = _producer(ENGINE.read_text())
    landing = body.index("weights_copy_complete(\n                        module_type, module_name, buffer_idx);")
    tail = body[landing:landing + 2600]
    assert "arm_peer_release_gate" in tail
    assert "publish_slot" in tail
    assert "replica_mark_owned_landed_" in tail
    assert "replica_mark_terminal_" in tail
    # produce-only (HostShared skip or release-on-landing) fails loud
    assert "produce-only" in body


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
    assert "[](int64_t) { return true; }" in drain


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
    worker = WORKER.read_text()
    install = worker[worker.index("def _install_replica_transport"):
                     worker.index("def _install_prefill_weight_copy_pipeline")]
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


def test_worker_declines_rather_than_crashes_non_participants():
    source = WORKER.read_text()
    body = source[source.index("def _install_replica_transport"):
                  source.index("def _install_prefill_weight_copy_pipeline")]
    for reason in (
        "daemon-backed weight source",
        "hierarchical_gdr weight source",
        "does not fit the layer grammar",
        "replica group has one member",
        "two-frame rendezvous needs",
    ):
        assert reason in body, reason
    assert "raise RuntimeError" not in body.split("ring capacity")[1][:400]
