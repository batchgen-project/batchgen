"""CPU-only coverage for GLM-5 decode host-KV causal controls.

The control is a causal-profiling knob: it removes ONLY the two post-forward
host-KV append launches so a fixed-work remote run can attribute the residual to
them. Everything that shapes the step's synchronization structure (capacity
growth, the single event record + synchronize) must survive, and every consumer
of host KV must fail closed once the writeback has been skipped.

The second half of this module covers the GLM-5 whole-model-graph boundary
host-KV writeback, which shares the deferred-KV plumbing but is a *production*
path: it defers the per-token append into one exact-range copy per host view at
the page boundary and at decode cleanup. Its bounded causal control preserves
dirty-range recording but suppresses only those final exact-range copies.
"""

import pytest
import torch
import types

from batchgen.sequence import SequenceStatus


FLAG = "glm5_suppress_decode_host_kv_writeback"
BOUNDARY_FLAG = "glm5_suppress_boundary_kv_writeback"
_COMPLETED = SequenceStatus.COMPLETED
_IN_DECODE = SequenceStatus.IN_DECODE


class _FakeEvent:
    def __init__(self, trace):
        self._trace = trace

    def record(self, stream):
        self._trace.append(("event_record", stream))

    def synchronize(self):
        self._trace.append(("event_synchronize",))


class _FakeHostKVView:
    """Records append launches; the presence of the batched kernel picks UVA."""

    def __init__(self, trace, name):
        self._trace = trace
        self._name = name

    def async_append_decode_kv_to_host_batched_kernel(self, entries, sequence_ids, sequence_lengths):
        self._trace.append((f"{self._name}_batched_append", list(sequence_ids)))
        return object()

    def async_append_decode_kv_to_host(self, **kwargs):  # pragma: no cover - fallback path
        self._trace.append((f"{self._name}_per_layer_append", list(kwargs["sequence_ids"])))
        return object()


def _make_flush_worker(monkeypatch, trace, suppress, with_batch_info=True):
    from batchgen.batchgen_worker import BatchGenWorker

    monkeypatch.delenv("BATCHGEN_KV_OFFLOAD_UVA_KERNEL", raising=False)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: "stream")

    worker = object.__new__(BatchGenWorker)
    worker.rank = 0
    worker.torch_device = "cuda:0"
    worker._kv_offload_event = _FakeEvent(trace)
    worker._pending_kv_append_tasks = []
    worker._suppress_decode_host_kv_writeback = suppress
    worker._host_kv_stale_global_ids = set()

    def _capacity(sequence_ids, sequence_lengths):
        trace.append(("ensure_capacity", list(sequence_ids), list(sequence_lengths)))

    worker._ensure_host_kv_append_capacity = _capacity

    primary_view = _FakeHostKVView(trace, "primary")
    aux_view = _FakeHostKVView(trace, "aux")
    worker._deferred_kv_entries = [(0, torch.zeros(2, 1, 8), torch.zeros(2, 1, 8))]
    worker._deferred_kv_entries_aux = [(0, torch.zeros(2, 1, 4), None)]
    worker._deferred_kv_worker_view = primary_view
    worker._deferred_kv_worker_view_aux = aux_view
    worker._deferred_kv_batch = ([11, 12], [63, 64]) if with_batch_info else None
    return worker


def _assert_deferred_state_cleared(worker):
    assert worker._deferred_kv_entries == []
    assert worker._deferred_kv_entries_aux == []
    assert worker._deferred_kv_batch is None
    assert worker._deferred_kv_worker_view is None
    assert worker._deferred_kv_worker_view_aux is None


def test_suppressed_flush_keeps_sync_structure_and_skips_appends(monkeypatch):
    trace = []
    worker = _make_flush_worker(monkeypatch, trace, suppress=True)

    worker._flush_deferred_kv_to_host()

    # Capacity growth still runs, and exactly one record + synchronize remain so
    # the caller's token D2H copy is drained the same way as the control run.
    assert trace == [
        ("ensure_capacity", [11, 12], [63, 64]),
        ("event_record", "stream"),
        ("event_synchronize",),
    ]
    assert worker._kv_offload_synced_this_step is True
    # The only removed work: both append launches.
    assert worker._pending_kv_append_tasks == []
    assert getattr(worker, "_pending_kv_append_tensors", []) == []
    # The caller marks the globally identical decode set after this local flush;
    # the flush itself must not create rank-local stale-ID state.
    assert worker._host_kv_stale_global_ids == set()
    _assert_deferred_state_cleared(worker)


def test_default_off_flush_launches_primary_and_aux_appends(monkeypatch):
    trace = []
    worker = _make_flush_worker(monkeypatch, trace, suppress=False)

    worker._flush_deferred_kv_to_host()

    assert ("primary_batched_append", [11, 12]) in trace
    assert ("aux_batched_append", [11, 12]) in trace
    assert len(worker._pending_kv_append_tasks) == 2
    assert worker._host_kv_stale_global_ids == set()
    _assert_deferred_state_cleared(worker)


def test_suppressed_flush_without_batch_metadata_fails_closed(monkeypatch):
    trace = []
    worker = _make_flush_worker(monkeypatch, trace, suppress=True, with_batch_info=False)
    # Aux-only shape: the flush reaches the sync with no sequence ids to mark.
    worker._deferred_kv_entries = []
    worker._deferred_kv_worker_view = None

    with pytest.raises(RuntimeError, match=FLAG):
        worker._flush_deferred_kv_to_host()

    assert worker._host_kv_stale_global_ids == set()
    assert ("aux_batched_append", [11, 12]) not in trace


def _stale_worker(**attrs):
    from batchgen.batchgen_worker import BatchGenWorker

    worker = object.__new__(BatchGenWorker)
    worker.rank = 0
    worker._host_kv_stale_global_ids = {11, 12}
    worker._suppress_decode_host_kv_writeback = True
    for key, value in attrs.items():
        setattr(worker, key, value)
    return worker


def test_stale_guard_blocks_dual_load_pointer_prep_before_page_table_rebuild():
    from batchgen.kv_cache.dual_kv_cache_coordinator import DualKVCacheCoordinator

    worker = _stale_worker()
    # Bare coordinator: any real use would explode, so reaching the guard first
    # is what keeps the test meaningful.
    coordinator = object.__new__(DualKVCacheCoordinator)

    with pytest.raises(RuntimeError, match="stale host KV"):
        worker._prepare_dual_kv_load_pointers(coordinator, [12])


def test_stale_guard_blocks_host_kv_load_before_touching_worker_view():
    worker = _stale_worker(core_engine=None)

    with pytest.raises(RuntimeError, match="stale host KV"):
        worker._load_host_kv_to_gpu(manager=None, global_sequence_ids=[12])


class _RecordingManager:
    is_initialized = True

    def __init__(self):
        self.freed = []
        self._sequences = {}

    def free_pages_for_sequences(self, ids):  # pragma: no cover - must not run
        self.freed.append(list(ids))


def _global_batch_stub(uuid_to_gid):
    return types.SimpleNamespace(
        get_sequence=lambda uuid: types.SimpleNamespace(global_idx=uuid_to_gid[uuid])
    )


def test_legacy_onhold_helper_guard_fires_before_freeing_gpu_pages():
    manager = _RecordingManager()
    worker = _stale_worker(
        global_batch=_global_batch_stub({"a": 12}),
        gpu_paged_kv_cache_manager=manager,
        _uuid_to_local_map={"a": 0},
        _sequences_with_gpu_kv={"a"},
    )

    with pytest.raises(RuntimeError, match="stale host KV"):
        worker._put_sequences_onhold(["a"])

    assert manager.freed == []


def test_onhold_watermark_path_guard_fires_before_freeing_gpu_pages():
    manager = _RecordingManager()
    synced = []
    worker = _stale_worker(
        global_batch=_global_batch_stub({"a": 12}),
        gpu_paged_kv_cache_manager=manager,
        _uuid_to_local_map={"a": 0},
        _sync_sequence_metadata=synced.append,
    )

    with pytest.raises(RuntimeError, match="stale host KV"):
        worker._put_sequences_on_hold(["a"])

    assert manager.freed == []
    assert synced == []


def test_onhold_guard_is_inert_without_stale_ids():
    manager = _RecordingManager()
    worker = _stale_worker(
        global_batch=_global_batch_stub({"a": 12}),
        gpu_paged_kv_cache_manager=manager,
        _uuid_to_local_map={},
        _sequences_with_gpu_kv=set(),
    )
    worker._host_kv_stale_global_ids = set()

    worker._put_sequences_onhold(["a"])

    assert manager.freed == []


def test_rebalance_allows_initial_pass_but_rejects_once_host_kv_is_stale():
    worker = _stale_worker(enable_decode_preemption=True)
    worker._host_kv_stale_global_ids = set()
    worker._plan_kv_migration = lambda: []

    # Initial prefill rebalance: nothing suppressed yet, host KV still faithful.
    worker._rebalance_host_kv()

    worker._host_kv_stale_global_ids = {11}
    with pytest.raises(RuntimeError, match="stale host KV"):
        worker._rebalance_host_kv()


@pytest.mark.parametrize(
    "field",
    ["onhold_uuids", "new_load_uuids", "host_evicted_uuids"],
)
def test_boundary_decision_guard_rejects_prohibited_transitions(field):
    worker = _stale_worker()
    decisions = types.SimpleNamespace(
        onhold_uuids=[], new_load_uuids=[], host_evicted_uuids=[]
    )
    setattr(decisions, field, ["a"])

    with pytest.raises(RuntimeError, match=FLAG) as excinfo:
        worker._assert_boundary_decisions_allowed_with_stale_host_kv(decisions)
    assert field in str(excinfo.value)


def test_boundary_decision_guard_allows_pure_decode_and_completion():
    worker = _stale_worker()
    decisions = types.SimpleNamespace(
        onhold_uuids=[], new_load_uuids=[], host_evicted_uuids=[]
    )

    worker._assert_boundary_decisions_allowed_with_stale_host_kv(decisions)

    worker._host_kv_stale_global_ids = set()
    decisions.new_load_uuids = ["a"]
    worker._assert_boundary_decisions_allowed_with_stale_host_kv(decisions)


def _debug_worker(enable_host_kv_eviction=True):
    from batchgen.batchgen_worker import BatchGenWorker

    worker = object.__new__(BatchGenWorker)
    worker.rank = 0
    worker.enable_host_kv_eviction = enable_host_kv_eviction
    worker._suppress_decode_host_kv_writeback = False
    worker._suppress_boundary_kv_writeback = False
    worker._host_kv_stale_global_ids = set()
    return worker


def test_flag_defaults_off_and_resolves_from_batchgen_debug(monkeypatch):
    import batchgen.batchgen_worker as worker_mod

    monkeypatch.setattr(worker_mod, "BATCHGEN_SYNC_KV", False)
    worker = _debug_worker()

    worker.set_batchgen_debug(None)
    assert worker._suppress_decode_host_kv_writeback is False

    worker.set_batchgen_debug({"glm5_moe_mode": "graph"})
    assert worker._suppress_decode_host_kv_writeback is False

    worker.set_batchgen_debug({FLAG: True})
    assert worker._suppress_decode_host_kv_writeback is True

    worker.set_batchgen_debug({FLAG: "1"})
    assert worker._suppress_decode_host_kv_writeback is True

    worker.set_batchgen_debug({FLAG: False})
    assert worker._suppress_decode_host_kv_writeback is False


def test_flag_rejects_sync_kv_but_allows_always_on_eviction_config(monkeypatch):
    import batchgen.batchgen_worker as worker_mod

    monkeypatch.setattr(worker_mod, "BATCHGEN_SYNC_KV", True)
    worker = _debug_worker()
    with pytest.raises(RuntimeError, match="BATCHGEN_SYNC_KV"):
        worker.set_batchgen_debug({FLAG: True})
    assert worker._suppress_decode_host_kv_writeback is False
    # Default-off batches are unaffected by the incompatible configuration.
    worker.set_batchgen_debug({"glm5_moe_mode": "graph"})
    assert worker._suppress_decode_host_kv_writeback is False

    monkeypatch.setattr(worker_mod, "BATCHGEN_SYNC_KV", False)
    worker = _debug_worker(enable_host_kv_eviction=True)
    worker.set_batchgen_debug({FLAG: True})
    assert worker._suppress_decode_host_kv_writeback is True


def test_global_decode_set_is_marked_stale_symmetrically():
    worker = _stale_worker(
        global_batch=_global_batch_stub({"a": 11, "b": 12}),
    )
    worker._host_kv_stale_global_ids = set()

    worker._mark_suppressed_decode_host_kv_stale(["a", "b"])

    assert worker._host_kv_stale_global_ids == {11, 12}


def test_reset_helper_clears_suppression_state():
    worker = _stale_worker()
    worker._suppress_boundary_kv_writeback = True

    worker._reset_decode_host_kv_writeback_debug_state()

    assert worker._suppress_decode_host_kv_writeback is False
    assert worker._suppress_boundary_kv_writeback is False
    assert worker._host_kv_stale_global_ids == set()


def test_boundary_flag_activation_marker_is_transition_gated_and_reset_clears_state(
    caplog,
):
    worker = _debug_worker()
    worker.rank = 3
    # Hot reload rebinds methods without rerunning BatchGenWorker.__init__.
    del worker._suppress_boundary_kv_writeback

    worker.set_batchgen_debug({BOUNDARY_FLAG: True})
    worker.set_batchgen_debug({BOUNDARY_FLAG: True})

    assert worker._suppress_boundary_kv_writeback is True
    markers = [
        record.getMessage()
        for record in caplog.records
        if BOUNDARY_FLAG in record.getMessage() and "SUPPRESSED" in record.getMessage()
    ]
    assert len(markers) == 1
    assert "rank=3" in markers[0]

    worker._host_kv_stale_global_ids = {11, 12}
    worker._reset_decode_host_kv_writeback_debug_state()

    assert worker._suppress_decode_host_kv_writeback is False
    assert worker._suppress_boundary_kv_writeback is False
    assert worker._host_kv_stale_global_ids == set()


# --------------------------------------------------------------------------
# Pool admission: batchgen_debug must be settled before anything mutates.
# The causal control is batch-scoped, so a mid-group flag change would mix two
# host-KV validity regimes inside one decode microbatch.
# --------------------------------------------------------------------------


def _pool_worker(
    monkeypatch,
    debug=None,
    suppress=False,
    boundary_suppress=False,
    stale=(),
    sequences=(),
):
    import batchgen.batchgen_worker as worker_mod
    from batchgen.sequence import SequenceBatch, SequenceEntry

    monkeypatch.setattr(worker_mod, "BATCHGEN_SYNC_KV", False)

    worker = object.__new__(worker_mod.BatchGenWorker)
    worker.rank = 0
    worker._batchgen_debug = debug
    worker._suppress_decode_host_kv_writeback = suppress
    worker._suppress_boundary_kv_writeback = boundary_suppress
    worker._host_kv_stale_global_ids = set(stale)
    worker.global_batch = SequenceBatch()
    for idx, (uuid, seq_debug, status) in enumerate(sequences):
        seq = SequenceEntry(
            uuid=uuid, global_idx=idx, prompt_length=4, max_decode_length=8
        )
        seq.batchgen_debug = seq_debug
        # Set before add_sequence so the batch's status index stays coherent.
        seq.status = status
        worker.global_batch.add_sequence(seq)
    return worker


def _entries(*debugs):
    return [
        {"request_id": f"r{i}", "text": "hi", "batchgen_debug": debug}
        for i, debug in enumerate(debugs)
    ]


def test_first_pool_activation_arms_causal_control_from_a_drained_pool(
    monkeypatch, caplog
):
    # Leftovers from a previous (fully completed) group must not survive.
    worker = _pool_worker(
        monkeypatch,
        debug={"glm5_moe_mode": "graph"},
        suppress=True,
        stale={11, 12},
        sequences=[("old", {"glm5_moe_mode": "graph"}, _COMPLETED)],
    )
    worker.rank = 3

    worker._resolve_admission_batchgen_debug(_entries({FLAG: True}, {FLAG: True}))
    worker.set_batchgen_debug({FLAG: True})

    assert worker._batchgen_debug == {FLAG: True}
    assert worker._suppress_decode_host_kv_writeback is True
    assert worker._host_kv_stale_global_ids == set()
    markers = [
        record.getMessage()
        for record in caplog.records
        if FLAG in record.getMessage() and "SUPPRESSED" in record.getMessage()
    ]
    assert len(markers) == 1
    assert "rank=3" in markers[0]


@pytest.mark.parametrize("incoming", [(None, {}), ({}, None)])
def test_first_pool_activation_treats_none_and_empty_debug_alike(monkeypatch, incoming):
    worker = _pool_worker(
        monkeypatch,
        debug={FLAG: True},
        suppress=True,
        boundary_suppress=True,
        stale={11},
    )

    worker._resolve_admission_batchgen_debug(_entries(*incoming))

    assert worker._batchgen_debug is None
    assert worker._suppress_decode_host_kv_writeback is False
    assert worker._suppress_boundary_kv_writeback is False
    assert worker._host_kv_stale_global_ids == set()


def test_sequential_completed_group_switch_resets_causal_state(monkeypatch):
    # A suppressed group that has fully completed releases the control, so the
    # next group starts from a clean (non-stale) host KV view.
    worker = _pool_worker(
        monkeypatch,
        debug={FLAG: True},
        suppress=True,
        boundary_suppress=True,
        stale={11, 12},
        sequences=[
            ("done-a", {FLAG: True}, _COMPLETED),
            ("done-b", {FLAG: True}, _COMPLETED),
        ],
    )

    worker._resolve_admission_batchgen_debug(_entries({"glm5_moe_mode": "eager"}))

    assert worker._batchgen_debug == {"glm5_moe_mode": "eager"}
    assert worker._suppress_decode_host_kv_writeback is False
    assert worker._suppress_boundary_kv_writeback is False
    assert worker._host_kv_stale_global_ids == set()


def test_same_debug_admission_into_active_group_preserves_stale_ids(monkeypatch):
    worker = _pool_worker(
        monkeypatch,
        debug={FLAG: True},
        suppress=True,
        stale={11, 12},
        sequences=[
            ("live", {FLAG: True}, _IN_DECODE),
            ("done", {FLAG: True}, _COMPLETED),
        ],
    )
    reconfigured = []
    worker.set_batchgen_debug = reconfigured.append

    # Equal-but-distinct dict: the group is matched by value, not identity.
    worker._resolve_admission_batchgen_debug(_entries({FLAG: True}))

    # No reset and no re-arm: the running group keeps its stale-ID bookkeeping,
    # which the host-KV consumer guards depend on.
    assert reconfigured == []
    assert worker._host_kv_stale_global_ids == {11, 12}
    assert worker._suppress_decode_host_kv_writeback is True


def test_mixed_incoming_debug_rejected_before_sequences_or_tokenization(monkeypatch):
    worker = _pool_worker(monkeypatch)
    tokenized = []
    worker._tokenize_admitted_sequences = tokenized.append

    with pytest.raises(RuntimeError, match="batch-level"):
        worker._admit_sequences_from_message(
            {"entries": _entries({FLAG: True}, None)}
        )

    assert len(worker.global_batch) == 0
    assert tokenized == []
    assert worker._batchgen_debug is None


@pytest.mark.parametrize("invalid", [FLAG, [], 0])
def test_non_dict_incoming_debug_rejected(monkeypatch, invalid):
    worker = _pool_worker(monkeypatch)

    with pytest.raises(RuntimeError, match="must be a dict"):
        worker._resolve_admission_batchgen_debug(_entries(invalid))


def test_active_group_rejects_admission_with_different_debug(monkeypatch):
    worker = _pool_worker(
        monkeypatch,
        debug={FLAG: True},
        suppress=True,
        stale={11},
        sequences=[("live", {FLAG: True}, _IN_DECODE)],
    )

    with pytest.raises(RuntimeError, match="running group"):
        worker._resolve_admission_batchgen_debug(_entries(None))

    assert worker._suppress_decode_host_kv_writeback is True
    assert worker._host_kv_stale_global_ids == {11}


def test_active_sequences_disagreeing_with_each_other_are_rejected(monkeypatch):
    worker = _pool_worker(
        monkeypatch,
        debug={FLAG: True},
        suppress=True,
        sequences=[
            ("live-a", {FLAG: True}, _IN_DECODE),
            ("live-b", None, _IN_DECODE),
        ],
    )

    with pytest.raises(RuntimeError, match="disagree"):
        worker._resolve_admission_batchgen_debug(_entries({FLAG: True}))


def test_worker_state_drift_from_active_group_is_rejected(monkeypatch):
    # Worker-level debug lost the flag while a suppressed group is still live.
    worker = _pool_worker(
        monkeypatch,
        debug=None,
        suppress=True,
        stale={11},
        sequences=[("live", {FLAG: True}, _IN_DECODE)],
    )

    with pytest.raises(RuntimeError, match="worker has"):
        worker._resolve_admission_batchgen_debug(_entries({FLAG: True}))


# ==========================================================================
# GLM-5 whole-model graph boundary host-KV writeback.
# ==========================================================================


class _FakeDirectHostKVView:
    """Host worker view exposing the exact-range boundary copy."""

    def __init__(self, name, trace, *, error=None):
        self._name = name
        self._trace = trace
        self._error = error
        self.task = object()

    def async_copy_dirty_kv_token_ranges_to_host(
        self,
        *,
        sequence_ids,
        active_page_counts,
        dirty_token_ranges,
        k_device_page_ptrs,
        v_device_page_ptrs,
    ):
        self._trace.append(
            (
                f"{self._name}_copy",
                sequence_ids.tolist(),
                active_page_counts.tolist(),
                dirty_token_ranges.tolist(),
                k_device_page_ptrs.tolist(),
                None if v_device_page_ptrs is None else v_device_page_ptrs.tolist(),
            )
        )
        if self._error is not None:
            raise self._error
        return self.task


class _MappedHostKVView:
    """Legacy mapped view: append-only, no exact-range capability."""

    def async_append_decode_kv_to_host_batched_kernel(self, **kwargs):
        raise AssertionError("append must not run on the boundary path")


class _FakeGPUManager:
    """Minimal stand-in for one side of the dual GPU paged-KV manager."""

    def __init__(self, slot_order, base_ptr, *, num_layers=2, max_pages=2,
                 page_counts=None, has_v=False):
        self._gpu_page_table_manager = types.SimpleNamespace(
            slot_to_seq_id=list(slot_order)
        )
        n = len(slot_order)
        self._k = (
            base_ptr + torch.arange(num_layers * n * max_pages, dtype=torch.int64)
        ).reshape(num_layers, n, max_pages)
        self._v = self._k + 10_000 if has_v else None
        self._counts = torch.tensor(
            list(page_counts) if page_counts is not None else [max_pages] * n,
            dtype=torch.int32,
        )

    def get_padded_3d_page_pointers(self):
        return self._k, self._v

    def export_active_sequence_page_counts(self):
        return self._counts


def _dual(primary, auxiliary):
    return types.SimpleNamespace(primary=primary, auxiliary=auxiliary)


def _boundary_worker(monkeypatch, **overrides):
    import batchgen.batchgen_worker as worker_mod

    monkeypatch.setattr(worker_mod, "BATCHGEN_SYNC_KV", False)

    worker = object.__new__(worker_mod.BatchGenWorker)
    worker.rank = 0
    worker._suppress_decode_host_kv_writeback = False
    worker._suppress_boundary_kv_writeback = False
    worker._host_kv_stale_global_ids = set()
    worker._glm5_whole_model_graph = True
    worker.core_engine = types.SimpleNamespace(gpu_paged_kv_manager_aux=None)
    worker.global_batch = None
    worker._pending_kv_append_tasks = []
    worker._pending_kv_append_tensors = []
    worker._boundary_kv_dirty_ranges = {}
    worker._reset_boundary_kv_dirty_state()
    for key, value in overrides.items():
        setattr(worker, key, value)
    return worker


def _arm_deferred(worker, trace, *, ids=(11, 12), write_positions=(63, 64),
                  primary_view=None, aux_view=None):
    worker._deferred_kv_batch = (list(ids), list(write_positions))
    worker._deferred_kv_entries = [(0, torch.zeros(2, 1, 8), None)]
    worker._deferred_kv_entries_aux = [(0, torch.zeros(2, 1, 4), None)]
    worker._deferred_kv_worker_view = (
        primary_view if primary_view is not None
        else _FakeDirectHostKVView("primary", trace)
    )
    worker._deferred_kv_worker_view_aux = (
        aux_view if aux_view is not None
        else _FakeDirectHostKVView("aux", trace)
    )


def _install_wait_stub(worker, trace, *, error=None):
    def _wait(*, sync_distributed_errors=False, defer_errors=False):
        deferred = list(
            getattr(worker, "_deferred_kv_append_wait_errors", [])
        )
        if hasattr(worker, "_deferred_kv_append_wait_errors"):
            worker._deferred_kv_append_wait_errors.clear()
        trace.append(
            ("wait", sync_distributed_errors, list(worker._pending_kv_append_tasks))
        )
        if error is not None:
            raise error
        num = len(worker._pending_kv_append_tasks)
        worker._pending_kv_append_tasks.clear()
        worker._pending_kv_append_tensors.clear()
        if deferred:
            raise RuntimeError(
                f"KV append/offload failed on at least one rank: {deferred}"
            )
        return num

    worker._wait_pending_kv_append_tasks = _wait


def _assert_deferred_consumed(worker):
    assert worker._deferred_kv_entries == []
    assert worker._deferred_kv_entries_aux == []
    assert worker._deferred_kv_batch is None
    assert worker._deferred_kv_worker_view is None
    assert worker._deferred_kv_worker_view_aux is None


def test_recording_merges_exact_write_positions_across_tokens(monkeypatch):
    trace = []
    worker = _boundary_worker(monkeypatch)
    gpu = _dual(_FakeGPUManager([11, 12], 0x1000), _FakeGPUManager([11, 12], 0x9000))

    _arm_deferred(worker, trace, write_positions=(63, 64))
    assert worker._record_glm5_boundary_dirty_kv(gpu) is True
    # write_pos = current_context_length - 1, recorded as [write_pos, write_pos+1).
    assert worker._boundary_kv_dirty_ranges == {11: [63, 64], 12: [64, 65]}
    # Nothing is staged or appended for this token.
    _assert_deferred_consumed(worker)
    assert trace == []
    assert worker._pending_kv_append_tasks == []

    _arm_deferred(worker, trace, write_positions=(64, 65))
    assert worker._record_glm5_boundary_dirty_kv(gpu) is True
    _arm_deferred(worker, trace, write_positions=(65, 66))
    assert worker._record_glm5_boundary_dirty_kv(gpu) is True

    # Merged hull is exactly the tokens produced since the last flush.
    assert worker._boundary_kv_dirty_ranges == {11: [63, 66], 12: [64, 67]}


def test_recording_tracks_a_sequence_that_joins_mid_interval(monkeypatch):
    trace = []
    worker = _boundary_worker(monkeypatch)
    gpu = _dual(_FakeGPUManager([11, 12], 0x1000), _FakeGPUManager([11, 12], 0x9000))

    _arm_deferred(worker, trace, ids=(11,), write_positions=(63,))
    worker._record_glm5_boundary_dirty_kv(gpu)
    _arm_deferred(worker, trace, ids=(11, 12), write_positions=(64, 20))
    worker._record_glm5_boundary_dirty_kv(gpu)

    assert worker._boundary_kv_dirty_ranges == {11: [63, 65], 12: [20, 21]}


@pytest.mark.parametrize(
    "missing",
    [
        "sync_kv",
        "non_glm5_or_eager_fallback",
        "no_deferred_batch",
        "primary_view_without_capability",
        "aux_view_without_capability",
        "missing_aux_view",
    ],
)
def test_missing_capability_keeps_per_token_stage_and_append(monkeypatch, missing):
    import batchgen.batchgen_worker as worker_mod

    trace = []
    worker = _boundary_worker(monkeypatch)
    gpu = _dual(_FakeGPUManager([11, 12], 0x1000), _FakeGPUManager([11, 12], 0x9000))
    _arm_deferred(worker, trace)

    if missing == "sync_kv":
        monkeypatch.setattr(worker_mod, "BATCHGEN_SYNC_KV", True)
    elif missing == "non_glm5_or_eager_fallback":
        worker._glm5_whole_model_graph = False
    elif missing == "no_deferred_batch":
        worker._deferred_kv_batch = None
    elif missing == "primary_view_without_capability":
        worker._deferred_kv_worker_view = _MappedHostKVView()
    elif missing == "aux_view_without_capability":
        worker._deferred_kv_worker_view_aux = _MappedHostKVView()
    else:
        worker._deferred_kv_worker_view_aux = None

    assert worker._record_glm5_boundary_dirty_kv(gpu) is False

    # Deferred metadata is left completely intact for the existing path.
    assert worker._deferred_kv_entries != []
    assert worker._boundary_kv_dirty_ranges == {}
    assert trace == []


@pytest.mark.parametrize(
    "gpu",
    [
        pytest.param(
            _dual(_FakeGPUManager([11], 0x1000), None), id="missing_aux_manager"
        ),
        pytest.param(
            _dual(_FakeGPUManager([11], 0x1000), types.SimpleNamespace()),
            id="aux_manager_without_pointer_api",
        ),
        pytest.param(
            _dual(types.SimpleNamespace(), _FakeGPUManager([11], 0x9000)),
            id="primary_manager_without_pointer_api",
        ),
    ],
)
def test_missing_dual_manager_capability_keeps_per_token_path(monkeypatch, gpu):
    trace = []
    worker = _boundary_worker(monkeypatch)
    _arm_deferred(worker, trace, ids=(11,), write_positions=(63,))

    assert worker._record_glm5_boundary_dirty_kv(gpu) is False
    assert worker._boundary_kv_dirty_ranges == {}
    assert worker._deferred_kv_batch == ([11], [63])


def test_suppression_control_never_becomes_boundary_writeback(monkeypatch):
    trace = []
    worker = _boundary_worker(monkeypatch, _suppress_decode_host_kv_writeback=True)
    gpu = _dual(_FakeGPUManager([11, 12], 0x1000), _FakeGPUManager([11, 12], 0x9000))
    _arm_deferred(worker, trace)

    # Full capability present, but the diagnostic keeps its own causal path:
    # the deferred metadata stays for _flush_deferred_kv_to_host to drop.
    assert worker._record_glm5_boundary_dirty_kv(gpu) is False
    assert worker._boundary_kv_dirty_ranges == {}
    assert worker._deferred_kv_batch == ([11, 12], [63, 64])
    assert trace == []


def test_boundary_suppression_preserves_target_and_record_fast_path(monkeypatch):
    trace = []
    worker = _boundary_worker(
        monkeypatch, _suppress_boundary_kv_writeback=True
    )
    gpu = _dual(
        _FakeGPUManager([11, 12], 0x1000),
        _FakeGPUManager([11, 12], 0x9000),
    )
    _arm_deferred(worker, trace)

    assert worker._glm5_boundary_kv_writeback_targets(gpu) is not None
    assert worker._record_glm5_boundary_dirty_kv(gpu) is True
    assert worker._boundary_kv_dirty_ranges == {
        11: [63, 64],
        12: [64, 65],
    }
    _assert_deferred_consumed(worker)
    assert trace == []


def _armed_flush_worker(monkeypatch, trace, *, primary=None, aux=None,
                        primary_view=None, aux_view=None,
                        write_positions=(63, 64), ids=(11, 12)):
    worker = _boundary_worker(monkeypatch)
    gpu = _dual(
        primary if primary is not None else _FakeGPUManager([11, 12], 0x1000),
        aux if aux is not None else _FakeGPUManager([11, 12], 0x9000, max_pages=3),
    )
    _arm_deferred(
        worker, trace, ids=ids, write_positions=write_positions,
        primary_view=primary_view, aux_view=aux_view,
    )
    assert worker._record_glm5_boundary_dirty_kv(gpu) is True
    return worker, gpu


def test_flush_launches_exact_ranges_on_both_views_then_waits(monkeypatch):
    trace = []
    primary = _FakeGPUManager([11, 12], 0x1000)
    aux = _FakeGPUManager([11, 12], 0x9000, max_pages=3)
    primary_view = _FakeDirectHostKVView("primary", trace)
    aux_view = _FakeDirectHostKVView("aux", trace)
    worker, _ = _armed_flush_worker(
        monkeypatch, trace, primary=primary, aux=aux,
        primary_view=primary_view, aux_view=aux_view,
    )
    _install_wait_stub(worker, trace)

    assert worker._flush_boundary_dirty_kv_ranges() == 2

    primary_call, aux_call, wait_call = trace
    # Slot order, ranges, counts and pointers are all materialized in the SAME
    # page-table slot order, per side.
    assert primary_call == (
        "primary_copy",
        [11, 12],
        [2, 2],
        [[63, 64], [64, 65]],
        primary._k.tolist(),
        None,
    )
    assert aux_call == (
        "aux_copy",
        [11, 12],
        [3, 3],
        [[63, 64], [64, 65]],
        aux._k.tolist(),
        None,
    )
    # Both tasks are surfaced through the existing collective waiter.
    assert wait_call == (
        "wait", True, [primary_view.task, aux_view.task],
    )
    assert worker._boundary_kv_dirty_ranges == {}
    assert worker._boundary_kv_primary_manager is None
    assert worker._boundary_kv_aux_manager is None
    assert worker._boundary_kv_primary_view is None
    assert worker._boundary_kv_aux_view is None


def test_suppressed_boundary_flush_skips_copies_but_waits_and_marks_stale(
    monkeypatch,
):
    trace = []
    worker, _ = _armed_flush_worker(monkeypatch, trace)
    worker._suppress_boundary_kv_writeback = True
    _install_wait_stub(worker, trace)

    assert worker._flush_boundary_dirty_kv_ranges() == 0

    assert trace == [("wait", True, [])]
    assert worker._host_kv_stale_global_ids == {11, 12}
    assert worker._boundary_kv_dirty_ranges == {}
    assert worker._boundary_kv_primary_manager is None
    assert worker._boundary_kv_aux_manager is None
    assert worker._boundary_kv_primary_view is None
    assert worker._boundary_kv_aux_view is None


def test_suppressed_boundary_wait_failure_retains_dirty_and_does_not_mark_stale(
    monkeypatch,
):
    trace = []
    worker, _ = _armed_flush_worker(monkeypatch, trace)
    worker._suppress_boundary_kv_writeback = True
    _install_wait_stub(worker, trace, error=RuntimeError("collective wait failed"))

    with pytest.raises(RuntimeError, match="collective wait failed"):
        worker._flush_boundary_dirty_kv_ranges()

    assert trace == [("wait", True, [])]
    assert worker._host_kv_stale_global_ids == set()
    assert worker._boundary_kv_dirty_ranges == {
        11: [63, 64],
        12: [64, 65],
    }
    assert worker._boundary_kv_primary_manager is not None
    assert worker._boundary_kv_aux_manager is not None
    assert worker._boundary_kv_primary_view is not None
    assert worker._boundary_kv_aux_view is not None


def test_boundary_suppression_stale_guard_allows_completion_only_and_rejects_transitions():
    worker = _stale_worker()
    worker._suppress_decode_host_kv_writeback = False
    worker._suppress_boundary_kv_writeback = True
    decisions = types.SimpleNamespace(
        completed_uuids=["done"],
        onhold_uuids=[],
        new_load_uuids=[],
        host_evicted_uuids=[],
    )

    worker._assert_boundary_decisions_allowed_with_stale_host_kv(decisions)

    for field in ("onhold_uuids", "new_load_uuids", "host_evicted_uuids"):
        setattr(decisions, field, ["blocked"])
        with pytest.raises(RuntimeError, match=BOUNDARY_FLAG):
            worker._assert_boundary_decisions_allowed_with_stale_host_kv(decisions)
        setattr(decisions, field, [])


def test_flush_ensures_host_capacity_before_materializing_page_tables(monkeypatch):
    trace = []
    worker, _ = _armed_flush_worker(monkeypatch, trace)
    _install_wait_stub(worker, trace)

    def _ensure(sequence_ids, write_positions):
        trace.append(("ensure_capacity", sequence_ids, write_positions))

    worker._ensure_host_kv_append_capacity = _ensure
    worker._flush_boundary_dirty_kv_ranges()

    assert trace[0] == ("ensure_capacity", [11, 12], [63, 64])
    assert trace[1][0] == "primary_copy"


def test_flush_emits_empty_ranges_for_active_but_undirtied_slots(monkeypatch):
    trace = []
    primary = _FakeGPUManager([11, 12, 13], 0x1000)
    aux = _FakeGPUManager([11, 12, 13], 0x9000)
    worker, _ = _armed_flush_worker(
        monkeypatch, trace, primary=primary, aux=aux, ids=(12,),
        write_positions=(7,),
    )
    _install_wait_stub(worker, trace)

    worker._flush_boundary_dirty_kv_ranges()

    assert trace[0][1] == [11, 12, 13]
    assert trace[0][3] == [[0, 0], [7, 8], [0, 0]]


def test_flush_forwards_v_pointers_when_the_cache_has_them(monkeypatch):
    trace = []
    primary = _FakeGPUManager([11], 0x1000, has_v=True)
    aux = _FakeGPUManager([11], 0x9000)
    worker, _ = _armed_flush_worker(
        monkeypatch, trace, primary=primary, aux=aux, ids=(11,),
        write_positions=(5,),
    )
    _install_wait_stub(worker, trace)

    worker._flush_boundary_dirty_kv_ranges()

    assert trace[0][5] == primary._v.tolist()
    assert trace[1][5] is None


def test_flush_without_dirty_state_still_waits_pending_appends(monkeypatch):
    trace = []
    worker = _boundary_worker(monkeypatch)
    sentinel = object()
    worker._pending_kv_append_tasks = [sentinel]
    _install_wait_stub(worker, trace)

    assert worker._flush_boundary_dirty_kv_ranges() == 1
    assert trace == [("wait", True, [sentinel])]


def test_flush_waits_per_token_appends_together_with_boundary_copies(monkeypatch):
    trace = []
    primary_view = _FakeDirectHostKVView("primary", trace)
    aux_view = _FakeDirectHostKVView("aux", trace)
    worker, _ = _armed_flush_worker(
        monkeypatch, trace, primary_view=primary_view, aux_view=aux_view
    )
    leftover = object()
    worker._pending_kv_append_tasks.append(leftover)
    _install_wait_stub(worker, trace)

    assert worker._flush_boundary_dirty_kv_ranges() == 3
    assert trace[-1] == (
        "wait", True, [leftover, primary_view.task, aux_view.task],
    )


def test_slot_order_mismatch_fails_closed_before_any_launch(monkeypatch):
    trace = []
    worker, _ = _armed_flush_worker(
        monkeypatch,
        trace,
        primary=_FakeGPUManager([11, 12], 0x1000),
        aux=_FakeGPUManager([12, 11], 0x9000),
    )
    _install_wait_stub(worker, trace)

    with pytest.raises(RuntimeError, match="slot-order mismatch"):
        worker._flush_boundary_dirty_kv_ranges()

    # No copy launched, but every rank still enters the collective waiter so
    # the local validation error cannot deadlock peers.
    assert trace == [("wait", True, [])]
    assert worker._boundary_kv_dirty_ranges == {11: [63, 64], 12: [64, 65]}


def test_dirty_id_missing_from_slot_order_fails_closed(monkeypatch):
    trace = []
    worker, _ = _armed_flush_worker(
        monkeypatch,
        trace,
        primary=_FakeGPUManager([11], 0x1000),
        aux=_FakeGPUManager([11], 0x9000),
    )
    _install_wait_stub(worker, trace)

    with pytest.raises(RuntimeError, match="no GPU page-table slot"):
        worker._flush_boundary_dirty_kv_ranges()

    assert trace == [("wait", True, [])]
    assert worker._boundary_kv_dirty_ranges == {11: [63, 64], 12: [64, 65]}


@pytest.mark.parametrize(
    "attr",
    [
        "_boundary_kv_primary_manager",
        "_boundary_kv_aux_manager",
        "_boundary_kv_primary_view",
        "_boundary_kv_aux_view",
    ],
)
def test_lost_manager_or_view_fails_closed_with_dirty_ranges(monkeypatch, attr):
    trace = []
    worker, _ = _armed_flush_worker(monkeypatch, trace)
    _install_wait_stub(worker, trace)
    setattr(worker, attr, None)

    with pytest.raises(RuntimeError, match="dual GPU manager / host view"):
        worker._flush_boundary_dirty_kv_ranges()

    assert trace == [("wait", True, [])]
    assert worker._boundary_kv_dirty_ranges != {}


def test_page_count_width_mismatch_fails_closed(monkeypatch):
    trace = []
    aux = _FakeGPUManager([11, 12], 0x9000)
    aux._counts = torch.tensor([2], dtype=torch.int32)
    worker, _ = _armed_flush_worker(monkeypatch, trace, aux=aux)
    _install_wait_stub(worker, trace)

    with pytest.raises(RuntimeError, match="active page counts"):
        worker._flush_boundary_dirty_kv_ranges()

    # Both sides are validated before either launches, so an aux-side defect
    # leaves no unwaited primary copy in flight and the ranges are still owed.
    assert trace == [("wait", True, [])]
    assert worker._boundary_kv_dirty_ranges != {}


def test_pointer_rank_mismatch_fails_closed(monkeypatch):
    trace = []
    primary = _FakeGPUManager([11, 12], 0x1000)
    primary._k = primary._k[0]
    worker, _ = _armed_flush_worker(monkeypatch, trace, primary=primary)
    _install_wait_stub(worker, trace)

    with pytest.raises(RuntimeError, match="unusable primary page pointers"):
        worker._flush_boundary_dirty_kv_ranges()

    assert trace == [("wait", True, [])]
    assert worker._boundary_kv_dirty_ranges != {}


def test_launch_failure_keeps_dirty_ranges(monkeypatch):
    trace = []
    worker, _ = _armed_flush_worker(
        monkeypatch,
        trace,
        aux_view=_FakeDirectHostKVView(
            "aux", trace, error=RuntimeError("contains null pointer")
        ),
    )
    _install_wait_stub(worker, trace)

    with pytest.raises(RuntimeError, match="contains null pointer"):
        worker._flush_boundary_dirty_kv_ranges()

    # The primary task launched before aux failed, and was drained by the
    # collective error path before the exception escaped.
    assert trace[-1][0] == "wait"
    assert len(trace[-1][2]) == 1
    assert worker._boundary_kv_dirty_ranges == {11: [63, 64], 12: [64, 65]}


def test_collective_wait_failure_keeps_dirty_ranges(monkeypatch):
    trace = []
    worker, _ = _armed_flush_worker(monkeypatch, trace)
    _install_wait_stub(
        worker, trace, error=RuntimeError("KV append/offload failed on at least one rank")
    )

    with pytest.raises(RuntimeError, match="at least one rank"):
        worker._flush_boundary_dirty_kv_ranges()

    assert worker._boundary_kv_dirty_ranges == {11: [63, 64], 12: [64, 65]}


def test_successful_flush_clears_and_next_interval_starts_fresh(monkeypatch):
    trace = []
    worker, gpu = _armed_flush_worker(monkeypatch, trace)
    _install_wait_stub(worker, trace)

    worker._flush_boundary_dirty_kv_ranges()
    assert worker._boundary_kv_dirty_ranges == {}

    _arm_deferred(worker, trace, write_positions=(70, 71))
    worker._record_glm5_boundary_dirty_kv(gpu)

    assert worker._boundary_kv_dirty_ranges == {11: [70, 71], 12: [71, 72]}


def test_reset_refuses_to_discard_unwritten_ranges_and_clears_after_cleanup(
    monkeypatch,
):
    trace = []
    worker, _ = _armed_flush_worker(monkeypatch, trace)
    _install_wait_stub(worker, trace)

    # A new batch must not be able to drop ranges the cleanup flush never wrote.
    with pytest.raises(RuntimeError, match="never written back"):
        worker._reset_boundary_kv_dirty_state()

    worker._flush_boundary_dirty_kv_ranges()
    worker._reset_boundary_kv_dirty_state()

    assert worker._boundary_kv_dirty_ranges == {}
    assert worker._boundary_kv_primary_manager is None
    assert worker._boundary_kv_aux_manager is None
    assert worker._boundary_kv_primary_view is None
    assert worker._boundary_kv_aux_view is None
