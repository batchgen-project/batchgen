"""CPU-only coverage for ``batchgen_debug.glm5_suppress_decode_host_kv_writeback``.

The control is a causal-profiling knob: it removes ONLY the two post-forward
host-KV append launches so a fixed-work remote run can attribute the residual to
them. Everything that shapes the step's synchronization structure (capacity
growth, the single event record + synchronize) must survive, and every consumer
of host KV must fail closed once the writeback has been skipped.
"""

import pytest
import torch
import types

from batchgen.sequence import SequenceStatus


FLAG = "glm5_suppress_decode_host_kv_writeback"
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

    worker._reset_decode_host_kv_writeback_debug_state()

    assert worker._suppress_decode_host_kv_writeback is False
    assert worker._host_kv_stale_global_ids == set()


# --------------------------------------------------------------------------
# Pool admission: batchgen_debug must be settled before anything mutates.
# The causal control is batch-scoped, so a mid-group flag change would mix two
# host-KV validity regimes inside one decode microbatch.
# --------------------------------------------------------------------------


def _pool_worker(monkeypatch, debug=None, suppress=False, stale=(), sequences=()):
    import batchgen.batchgen_worker as worker_mod
    from batchgen.sequence import SequenceBatch, SequenceEntry

    monkeypatch.setattr(worker_mod, "BATCHGEN_SYNC_KV", False)

    worker = object.__new__(worker_mod.BatchGenWorker)
    worker.rank = 0
    worker._batchgen_debug = debug
    worker._suppress_decode_host_kv_writeback = suppress
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
    worker = _pool_worker(monkeypatch, debug={FLAG: True}, suppress=True, stale={11})

    worker._resolve_admission_batchgen_debug(_entries(*incoming))

    assert worker._batchgen_debug is None
    assert worker._suppress_decode_host_kv_writeback is False
    assert worker._host_kv_stale_global_ids == set()


def test_sequential_completed_group_switch_resets_causal_state(monkeypatch):
    # A suppressed group that has fully completed releases the control, so the
    # next group starts from a clean (non-stale) host KV view.
    worker = _pool_worker(
        monkeypatch,
        debug={FLAG: True},
        suppress=True,
        stale={11, 12},
        sequences=[
            ("done-a", {FLAG: True}, _COMPLETED),
            ("done-b", {FLAG: True}, _COMPLETED),
        ],
    )

    worker._resolve_admission_batchgen_debug(_entries({"glm5_moe_mode": "eager"}))

    assert worker._batchgen_debug == {"glm5_moe_mode": "eager"}
    assert worker._suppress_decode_host_kv_writeback is False
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
