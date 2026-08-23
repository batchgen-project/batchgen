import ast
import copy
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _function(path, class_name, function_name):
    tree = ast.parse(path.read_text())
    klass = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return next(
        node
        for node in klass.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )


def _isolated_method(path, class_name, function_name, globals_=None):
    method = copy.deepcopy(_function(path, class_name, function_name))
    module = ast.Module(
        body=[
            ast.ClassDef(
                name="Isolated",
                bases=[],
                keywords=[],
                body=[method],
                decorator_list=[],
            )
        ],
        type_ignores=[],
    )
    namespace = dict(globals_ or {})
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return getattr(namespace["Isolated"], function_name)


def _call_guards(function):
    """Map each ``obj.name()`` call to the ``if`` tests enclosing it.

    Returns ``{attribute_name: [tuple_of_enclosing_tests, ...]}``; an empty
    tuple means the call runs unconditionally in the function body.
    """
    found = {}

    def visit(node, stack):
        if isinstance(node, ast.If):
            visit(node.test, stack)
            for child in node.body:
                visit(child, stack + (ast.unparse(node.test),))
            for child in node.orelse:
                visit(child, stack + ("else: " + ast.unparse(node.test),))
            return
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            found.setdefault(node.func.attr, []).append(stack)
        for child in ast.iter_child_nodes(node):
            visit(child, stack)

    for statement in function.body:
        visit(statement, ())
    return found


def _isolated_function(path, function_name):
    tree = ast.parse(path.read_text())
    function = copy.deepcopy(
        next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == function_name
        )
    )
    module = ast.Module(body=[function], type_ignores=[])
    namespace = {}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[function_name]


def test_prefill_mode_accepts_streamed_resident_ep_or_streamed_sp8():
    path = (
        ROOT
        / "batchgen"
        / "models"
        / "moonshotai"
        / "kimi_linear"
        / "Parallel_Strategy_Manager.py"
    )
    manager = type("Manager", (), {})()
    manager._is_k3 = True
    manager._prefill_moe_mode = "streamed"
    manager.set_prefill_moe_mode = _isolated_method(
        path, "KimiLinearParallelStrategyManager", "set_prefill_moe_mode"
    ).__get__(manager)
    manager.prefill_uses_resident_ep = _isolated_method(
        path, "KimiLinearParallelStrategyManager", "prefill_uses_resident_ep"
    ).__get__(manager)
    manager.prefill_uses_streamed_sp8 = _isolated_method(
        path, "KimiLinearParallelStrategyManager", "prefill_uses_streamed_sp8"
    ).__get__(manager)

    manager.set_prefill_moe_mode("resident_ep")
    assert manager.prefill_uses_resident_ep()
    manager.set_prefill_moe_mode("streamed_sp8")
    assert manager.prefill_uses_streamed_sp8()
    manager.set_prefill_moe_mode(None)
    assert not manager.prefill_uses_resident_ep()
    assert not manager.prefill_uses_streamed_sp8()
    with pytest.raises(ValueError, match="streamed.*resident_ep.*streamed_sp8"):
        manager.set_prefill_moe_mode("unknown")


def test_distributed_store_defaults_to_sp8_and_rejects_replicated_prefill():
    path = (
        ROOT
        / "batchgen"
        / "models"
        / "moonshotai"
        / "kimi_linear"
        / "Parallel_Strategy_Manager.py"
    )
    manager = type("Manager", (), {})()
    manager._distributed_weight_sharded = True
    manager._is_k3 = True
    manager._prefill_moe_mode = "streamed_sp8"
    default_mode = _isolated_method(
        path,
        "KimiLinearParallelStrategyManager",
        "default_prefill_moe_mode",
    ).__get__(manager)
    set_mode = _isolated_method(
        path,
        "KimiLinearParallelStrategyManager",
        "set_prefill_moe_mode",
    ).__get__(manager)

    assert default_mode() == "streamed_sp8"
    set_mode("streamed_sp8")
    with pytest.raises(ValueError, match="require.*streamed_sp8"):
        set_mode("streamed")
    with pytest.raises(ValueError, match="require.*streamed_sp8"):
        set_mode("resident_ep")


def test_kda_prefill_capacity_is_scoped_to_the_tp8_node():
    path = (
        ROOT
        / "batchgen"
        / "models"
        / "moonshotai"
        / "kimi_linear"
        / "Parallel_Strategy_Manager.py"
    )
    manager = type("Manager", (), {})()
    manager._kda_pool_slots = 4
    manager._attn_tp_size = 8
    fake_wrapper = type("KDAWrapper", (), {"state_manager": None})
    method = _isolated_method(
        path,
        "KimiLinearParallelStrategyManager",
        "prefill_sequence_limits",
        {"KimiLinearKDAWrapper": fake_wrapper},
    ).__get__(manager)

    assert method() == {"max_sequences_per_node": 4}

    manager._attn_tp_size = 1
    assert method() == {"max_sequences_per_rank": 4}


def test_prefill_only_completion_releases_kda_without_paged_gpu_kv():
    path = ROOT / "batchgen" / "batchgen_worker.py"
    prefilled = object()
    seq = SimpleNamespace(
        uuid="u",
        global_idx=17,
        status=prefilled,
        decoded_length=1,
        max_decode_length=1,
        gpu_pages_allocated=0,
        host_pages_allocated=1,
        host_token_capacity=64,
    )
    batch = SimpleNamespace(get_sequence=lambda uuid: seq if uuid == "u" else None)
    calls = SimpleNamespace(kda=[], gpu=[], host=[], report=[])
    worker = SimpleNamespace(
        rank=1,
        global_batch=batch,
        _uuid_to_local_map={"u": 0},
        _sequences_with_gpu_kv=set(),
        _response_queue=object(),
        _sync_sequence_metadata=lambda uuids: None,
        _submit_completed_to_incremental_writer=lambda uuids: None,
        _gather_completed_tokens=lambda uuids: {"u": "<|sep|>"},
        _release_gpu_kv_pages=lambda local_ids: calls.gpu.append(local_ids),
        _get_local_indices_for_uuids=lambda uuids: [0],
        _release_kda_state_slots=lambda global_ids: calls.kda.append(global_ids),
        _release_host_kv_pages_for_batch=lambda uuids: calls.host.append(uuids),
        _update_batch_status=lambda uuids, status: None,
        _report_completion=lambda uuid, gathered_text=None: calls.report.append(
            (uuid, gathered_text)
        ),
    )
    method = _isolated_method(
        path,
        "BatchGenWorker",
        "_finish_prefill_completed_sequences",
        {
            "List": list,
            "SequenceStatus": SimpleNamespace(PREFILLED=prefilled, COMPLETED=object()),
        },
    ).__get__(worker)

    assert method(["u"]) == ["u"]
    assert calls.gpu == []
    assert calls.kda == [[17]]
    assert calls.host == [["u"]]
    assert calls.report == [("u", "<|sep|>")]


def test_prefill_completion_does_not_double_release_kda_with_gpu_kv():
    path = ROOT / "batchgen" / "batchgen_worker.py"
    prefilled = object()
    seq = SimpleNamespace(
        uuid="u",
        global_idx=17,
        status=prefilled,
        decoded_length=1,
        max_decode_length=1,
        gpu_pages_allocated=1,
        host_pages_allocated=1,
        host_token_capacity=64,
    )
    batch = SimpleNamespace(get_sequence=lambda uuid: seq if uuid == "u" else None)
    calls = SimpleNamespace(kda=[], gpu=[])
    worker = SimpleNamespace(
        rank=1,
        global_batch=batch,
        _uuid_to_local_map={"u": 0},
        _sequences_with_gpu_kv={"u"},
        _response_queue=object(),
        _sync_sequence_metadata=lambda uuids: None,
        _submit_completed_to_incremental_writer=lambda uuids: None,
        _gather_completed_tokens=lambda uuids: {"u": "<|sep|>"},
        _release_gpu_kv_pages=lambda local_ids: calls.gpu.append(local_ids),
        _get_local_indices_for_uuids=lambda uuids: [0],
        _release_kda_state_slots=lambda global_ids: calls.kda.append(global_ids),
        _release_host_kv_pages_for_batch=lambda uuids: None,
        _update_batch_status=lambda uuids, status: None,
        _report_completion=lambda uuid, gathered_text=None: None,
    )
    method = _isolated_method(
        path,
        "BatchGenWorker",
        "_finish_prefill_completed_sequences",
        {
            "List": list,
            "SequenceStatus": SimpleNamespace(PREFILLED=prefilled, COMPLETED=object()),
        },
    ).__get__(worker)

    assert method(["u"]) == ["u"]
    assert calls.gpu == [[0]]
    assert calls.kda == []


def test_distributed_store_requires_boolean_worker_sharding(tmp_path):
    from batchgen.models.moonshotai.kimi_linear.distributed_weight_store import (
        load_distributed_weight_config,
    )

    store = tmp_path / "store.bin"
    store.write_bytes(b"store")
    metadata = tmp_path / "metadata.tsv"
    metadata.write_text("H\n")
    base = {
        "node_rank": 0,
        "node_ips": ["n0", "n1", "n2", "n3"],
        "workers": 8,
        "store_path": str(store),
        "metadata_path": str(metadata),
        "daemon_socket": str(tmp_path / "daemon.sock"),
        "summary_path": str(tmp_path / "summary.json"),
        "store_bytes": store.stat().st_size,
        "replicated_bytes": 0,
        "module_bytes": 1,
    }
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps({**base, "worker_sharded": "true"}))
    with pytest.raises(ValueError, match="worker_sharded=true"):
        load_distributed_weight_config(invalid)

    valid = tmp_path / "valid.json"
    valid.write_text(json.dumps({**base, "worker_sharded": True}))
    assert load_distributed_weight_config(valid)["workers"] == 8


def test_resident_prefill_is_refused_for_non_k3():
    path = (
        ROOT
        / "batchgen"
        / "models"
        / "moonshotai"
        / "kimi_linear"
        / "Parallel_Strategy_Manager.py"
    )
    manager = type("Manager", (), {})()
    manager._is_k3 = False
    manager._prefill_moe_mode = "streamed"
    manager.set_prefill_moe_mode = _isolated_method(
        path, "KimiLinearParallelStrategyManager", "set_prefill_moe_mode"
    ).__get__(manager)
    with pytest.raises(ValueError, match="only for Kimi-K3"):
        manager.set_prefill_moe_mode("resident_ep")
    with pytest.raises(ValueError, match="only for Kimi-K3"):
        manager.set_prefill_moe_mode("streamed_sp8")


def test_streamed_sp8_builds_rank_local_112_expert_schedule():
    path = (
        ROOT
        / "batchgen"
        / "models"
        / "moonshotai"
        / "kimi_linear"
        / "Parallel_Strategy_Manager.py"
    )
    manager = _sp8_schedule_manager(hierarchical_gdr=False)
    build = _isolated_method(
        path,
        "KimiLinearParallelStrategyManager",
        "_build_weight_copy_task",
    ).__get__(manager)

    tasks = build()["routed_expert"]

    assert len(tasks) == 112
    assert tasks[0] == "routed_expert_0_336"
    assert tasks[-1] == "routed_expert_0_447"


def _sp8_schedule_manager(*, hierarchical_gdr, cross_source=False):
    """A TP8 rank-3 manager stub for ``_build_weight_copy_task``."""
    manager = type("Manager", (), {})()
    manager._stream_all_modules = False
    manager._attn_tp_size = 8
    manager._attn_tp_rank = 3
    manager._hierarchical_gdr = hierarchical_gdr
    manager._cross_weight_source = cross_source
    manager.prefill_uses_streamed_sp8 = lambda: True
    moe = type("MoE", (), {
        "experts": [object()] * 896,
        "shared_experts": None,
    })()
    layer = type("Layer", (), {"block_sparse_moe": moe})()
    manager.model = type("Model", (), {
        "model": type("Inner", (), {"layers": [layer]})(),
    })()
    manager.loaded_model_config = type("Config", (), {
        "num_hidden_layers": 1,
        "is_kda_layer": lambda self, _: False,
    })()
    return manager


def test_hierarchical_gdr_schedules_host_ingress_on_source_ranks_only():
    path = (
        ROOT
        / "batchgen"
        / "models"
        / "moonshotai"
        / "kimi_linear"
        / "Parallel_Strategy_Manager.py"
    )

    def routed(manager):
        return _isolated_method(
            path,
            "KimiLinearParallelStrategyManager",
            "_build_weight_copy_task",
        ).__get__(manager)()["routed_expert"]

    # A source rank keeps exactly its contiguous 112-expert shard: the eight
    # sources between them still cover all 896 experts of the layer.
    source = routed(
        _sp8_schedule_manager(hierarchical_gdr=True, cross_source=True)
    )
    assert len(source) == 112
    assert source[0] == "routed_expert_0_336"
    assert source[-1] == "routed_expert_0_447"

    # The 24 non-source ranks receive the shard over the cross-node broadcast
    # and must request nothing from the host store.
    assert routed(
        _sp8_schedule_manager(hierarchical_gdr=True, cross_source=False)
    ) == []


def test_hierarchical_gdr_builds_the_exact_cross_node_group_and_root_map():
    dist = pytest.importorskip("torch.distributed")
    path = (
        ROOT
        / "batchgen"
        / "models"
        / "moonshotai"
        / "kimi_linear"
        / "Parallel_Strategy_Manager.py"
    )
    build = _isolated_method(
        path,
        "KimiLinearParallelStrategyManager",
        "_build_cross_weight_group",
    )
    created = []
    original = dist.new_group

    def fake_new_group(ranks=None, **kwargs):
        created.append(tuple(ranks))
        return ("group", tuple(ranks))

    def manager_for(global_rank, hierarchical_gdr=True):
        manager = type("Manager", (), {})()
        manager._hierarchical_gdr = hierarchical_gdr
        manager._attn_tp_size = 8
        manager._attn_tp_rank = global_rank % 8
        manager.world_size = 32
        manager.global_rank = global_rank
        manager._cross_weight_group = None
        manager._cross_weight_root = None
        manager._cross_weight_source = False
        return manager

    dist.new_group = fake_new_group
    try:
        sources = []
        for global_rank in range(32):
            manager = manager_for(global_rank)
            created.clear()
            build.__get__(manager)()

            # Every rank creates all eight groups, in one identical order.
            assert created == [
                (g, g + 8, g + 16, g + 24) for g in range(8)
            ]
            g = global_rank % 8
            assert manager._cross_weight_group == (
                "group", (g, g + 8, g + 16, g + 24)
            )
            assert manager._cross_weight_root == (g // 2) * 8 + g
            if manager._cross_weight_source:
                sources.append(global_rank)

        # Two sources per node, so every node drives an egress stream.
        assert sources == [0, 1, 10, 11, 20, 21, 30, 31]

        # host_rdma builds nothing at all, so the attention and node-local
        # weight groups keep their existing creation order and count.
        manager = manager_for(5, hierarchical_gdr=False)
        created.clear()
        build.__get__(manager)()
        assert created == []
        assert manager._cross_weight_group is None
        assert manager._cross_weight_root is None
        assert manager._cross_weight_source is False
    finally:
        dist.new_group = original


def test_streamed_sp8_acquire_batch_tracks_expert_ring_depth():
    path = (
        ROOT
        / "batchgen"
        / "models"
        / "moonshotai"
        / "kimi_linear"
        / "Parallel_Strategy_Manager.py"
    )
    function = _function(
        path,
        "KimiLinearParallelStrategyManager",
        "_init_streamed_sp8_prefill",
    )
    source = ast.unparse(function)

    assert "num_prefill_module_buffer['routed_expert']" in source
    assert "acquire_batch_size=expert_ring_depth" in source
    assert "acquire_batch_size=16" not in source


def test_streamed_sp8_forward_uses_only_local_row_collective():
    path = (
        ROOT
        / "batchgen"
        / "models"
        / "moonshotai"
        / "kimi_linear"
        / "serving_modules.py"
    )
    tree = ast.parse(path.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "moe_forward_serving"
    )
    sp8_if = next(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.If)
        and any(
            isinstance(name, ast.Name) and name.id == "streamed_sp8"
            for name in ast.walk(node.test)
        )
    )
    calls = set()
    for node in ast.walk(sp8_if):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Attribute):
            calls.add(node.func.attr)
        elif isinstance(node.func, ast.Name):
            calls.add(node.func.id)
    assert "all_gather_rows" in calls
    assert "all_reduce" not in calls
    assert "all_gather" not in calls


def test_streamed_sp8_weight_batch_uses_non_evicting_phase():
    path = ROOT / "batchgen" / "moe" / "streamed_sp8_mxfp4.py"
    function = _function(
        path, "StreamedSP8LayerBuffer", "_acquire_local_shard"
    )
    phases = [
        node.value
        for node in ast.walk(function)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value.startswith("prefill")
    ]
    assert "prefill_sp8" in phases
    assert "prefill" not in phases


def test_streamed_sp8_reusable_buffers_are_not_inference_tensors():
    path = ROOT / "batchgen" / "moe" / "streamed_sp8_mxfp4.py"
    function = _function(path, "StreamedSP8LayerBuffer", "_allocate")
    source = ast.unparse(function)

    assert "with torch.inference_mode(False)" in source


class _FakeEvent:
    def __init__(self, trace):
        self._trace = trace

    def record(self, stream):
        self._trace.append(
            SimpleNamespace(
                op="record",
                event=self,
                stream=stream.name,
                thread=threading.get_ident(),
            )
        )


class _FakeStream:
    def __init__(self, name, trace):
        self.name = name
        self._trace = trace

    def _log(self, op, event=None):
        self._trace.append(
            SimpleNamespace(
                op=op,
                event=event,
                stream=self.name,
                thread=threading.get_ident(),
            )
        )

    def wait_event(self, event):
        self._log("wait", event)

    def synchronize(self):
        self._log("synchronize")


def _mock_sp8_buffer(trace, method_names, *, acquire=None):
    """Bind real ``StreamedSP8LayerBuffer`` methods onto fake CUDA streams."""
    path = ROOT / "batchgen" / "moe" / "streamed_sp8_mxfp4.py"
    compute_stream = _FakeStream("compute", trace)
    prefetch_stream = _FakeStream("prefetch", trace)

    def record_call(op):
        def call(*_args, stream=None, **_kwargs):
            trace.append(
                SimpleNamespace(
                    op=op,
                    event=None,
                    stream=(stream or compute_stream).name,
                    thread=threading.get_ident(),
                )
            )

        return call

    buffer = type("Buffer", (), {})()
    buffer.device = "cuda:0"
    buffer._pending = None
    buffer._next_layer = {3: 4}
    buffer._prefetch_stream = prefetch_stream
    buffer._local_free = _FakeEvent(trace)
    buffer._acquire_local_shard = acquire or record_call("acquire")
    buffer._assemble_compute_shard = record_call("assemble")
    buffer.shard = SimpleNamespace(name="shard")
    buffer._make_shard = lambda: buffer.shard
    globals_ = {
        "torch": SimpleNamespace(
            cuda=SimpleNamespace(
                Event=lambda: _FakeEvent(trace),
                current_stream=lambda device: compute_stream,
                set_device=lambda device: None,
            )
        ),
        "threading": threading,
        "SimpleNamespace": SimpleNamespace,
    }
    for name in method_names:
        setattr(
            buffer,
            name,
            _isolated_method(
                path, "StreamedSP8LayerBuffer", name, globals_
            ).__get__(buffer),
        )
    return buffer


def test_streamed_sp8_ingress_starts_before_full_overwrite_is_permitted():
    trace = []
    model_thread = threading.get_ident()
    started = threading.Event()
    resume = threading.Event()

    def acquire(layer_idx, stream=None):
        trace.append(
            SimpleNamespace(
                op="acquire",
                event=None,
                stream=stream.name,
                thread=threading.get_ident(),
            )
        )
        started.set()
        assert resume.wait(5)

    buffer = _mock_sp8_buffer(
        trace,
        ("begin_prefetch_next", "allow_full_overwrite"),
        acquire=acquire,
    )

    buffer.begin_prefetch_next(3)
    pending = buffer._pending

    # Overlap: remote ingress runs without waiting for any current-layer
    # kernel, i.e. it is not held behind a compute-stream event.
    assert started.wait(5)
    assert [(entry.op, entry.stream) for entry in trace] == [
        ("wait", "prefetch"),
        ("acquire", "prefetch"),
    ]
    # Safety: the ingress that overwrites ``self.local`` is still ordered after
    # the PREVIOUS all-gather, which on the first boundary ran on the compute
    # stream and is therefore not covered by prefetch-stream ordering.
    assert trace[0].event is buffer._local_free
    assert trace[1].thread != model_thread
    resume.set()

    # Safety: the local->compute copy that overwrites the shard the current
    # layer's kernels read is parked on the handshake until the model thread
    # grants permission.
    pending.thread.join(0.1)
    assert pending.thread.is_alive()
    assert not any(entry.op == "assemble" for entry in trace)

    buffer.allow_full_overwrite()
    pending.thread.join(5)
    assert not pending.thread.is_alive()
    assert pending.error is None
    assert [(entry.op, entry.stream) for entry in trace] == [
        ("wait", "prefetch"),
        ("acquire", "prefetch"),
        ("record", "compute"),
        ("wait", "prefetch"),
        ("assemble", "prefetch"),
        ("record", "prefetch"),
    ]
    # The covering event is recorded on the model thread at release time — not
    # when the prefetch started — so it spans the whole current-layer MoE, and
    # the overwriting gather waits on exactly that event.
    assert trace[2].event is pending.compute_done
    assert trace[2].thread == model_thread
    assert trace[3].event is pending.compute_done
    # ``pending.ready`` still orders the next compute after the prefetch.
    assert trace[5].event is pending.ready
    assert pending.ready is not pending.compute_done

    # Releasing twice must not re-record the event behind the worker's wait.
    buffer.allow_full_overwrite()
    assert len(trace) == 6


def test_streamed_sp8_load_joins_the_pending_layer():
    trace = []
    buffer = _mock_sp8_buffer(
        trace,
        (
            "begin_prefetch_next",
            "allow_full_overwrite",
            "_wait_pending",
            "load",
        ),
    )

    buffer.begin_prefetch_next(3)
    pending = buffer._pending
    buffer.allow_full_overwrite()

    assert buffer.load(4) is buffer.shard
    assert buffer._pending is None
    assert not pending.thread.is_alive()
    # The compute stream is ordered after the prefetch's full-layer writes.
    last = trace[-1]
    assert (last.op, last.stream, last.event) == ("wait", "compute", pending.ready)

    # Loading any layer other than the pending one is a scheduling bug.
    buffer.begin_prefetch_next(3)
    with pytest.raises(RuntimeError, match="expected prefetched layer 4, requested 9"):
        buffer.load(9)
    buffer.allow_full_overwrite()
    buffer._wait_pending()


def test_streamed_sp8_load_propagates_prefetch_errors_to_the_model_thread():
    trace = []
    failure = RuntimeError("weight lease timed out")

    def acquire(layer_idx, stream=None):
        raise failure

    buffer = _mock_sp8_buffer(
        trace,
        (
            "begin_prefetch_next",
            "allow_full_overwrite",
            "_wait_pending",
            "load",
        ),
        acquire=acquire,
    )

    buffer.begin_prefetch_next(3)
    buffer.allow_full_overwrite()
    with pytest.raises(RuntimeError, match="prefetch of layer 4 failed") as excinfo:
        buffer.load(4)

    assert excinfo.value.__cause__ is failure
    assert not any(entry.op == "assemble" for entry in trace)


def test_streamed_sp8_close_cannot_strand_a_prefetch_awaiting_permission():
    trace = []
    buffer = _mock_sp8_buffer(
        trace, ("begin_prefetch_next", "_wait_pending", "close")
    )

    # A forward that raised between the two phases never grants permission.
    buffer.begin_prefetch_next(3)
    pending = buffer._pending

    buffer.close()

    assert pending.cancelled
    assert not pending.thread.is_alive()
    assert pending.error is None
    assert buffer._pending is None
    # Teardown cancels the unreleased assembly rather than granting it: the
    # ingress it follows may never have completed on the other ranks.
    assert not any(entry.op == "assemble" for entry in trace)
    assert (trace[-1].op, trace[-1].stream) == ("synchronize", "prefetch")


SP8_HIDDEN = 8
SP8_LATENT = 4
SP8_TOP_K = 2


def _ops(trace):
    """Operation names of a mixed string/tuple trace."""
    return [entry if isinstance(entry, str) else entry[0] for entry in trace]


def _mock_sp8_moe_layer(
    trace,
    rows,
    *,
    num_rows=None,
    tp_size=2,
    tp_rank=0,
    expert_start=0,
    num_local=4,
    row_experts=None,
    profile=False,
):
    """Bind the real ``forward`` onto mock projections, buffer and NCCL.

    ``rows`` is this rank's slice of the node's ``num_rows`` replicated rows;
    ``row_experts`` (per padded local row, ``SP8_TOP_K`` expert ids) drives the
    router so ownership accounting can be checked.
    """
    torch = pytest.importorskip("torch")
    path = ROOT / "batchgen" / "moe" / "streamed_sp8_mxfp4.py"
    hidden, latent = SP8_HIDDEN, SP8_LATENT
    if num_rows is None:
        num_rows = rows

    class FakeHelper:
        def __init__(
            self,
            layer_idx,
            shard,
            down_proj,
            norm,
            up_proj,
            world_size=1,
            expert_start=0,
        ):
            self.shard = shard
            trace.append(("helper", world_size, expert_start))

        def _expert_path(self, x_latent, topk_idx, count):
            trace.append(("expert_path", count, tuple(topk_idx.shape)))
            return torch.zeros((count, 2, self.shard.K_latent)), None

        def _combine_fp32(
            self, expert_out, topk_pos, topk_weight, count, latent_size, top_k
        ):
            trace.append(("combine", count))
            return torch.zeros((count, latent_size))

    def log(op, value):
        trace.append(op)
        return value

    def all_gather_into_tensor(out, src, group=None):
        trace.append(("all_gather", tuple(src.shape), str(src.dtype), group))
        n = src.shape[0]
        for g in range(tp_size):
            out[g * n:(g + 1) * n].copy_(src)

    def reduce_scatter_tensor(out, inp, op=None, group=None):
        trace.append(
            ("reduce_scatter", tuple(inp.shape), str(inp.dtype), op, group)
        )
        n = out.shape[0]
        out.copy_(inp[tp_rank * n:(tp_rank + 1) * n])

    fake_dist = SimpleNamespace(
        all_gather_into_tensor=all_gather_into_tensor,
        reduce_scatter_tensor=reduce_scatter_tensor,
        ReduceOp=SimpleNamespace(SUM="SUM"),
    )

    layer = type("Layer", (), {
        "_prefill_profile_enabled": profile,
        "_prefill_profile_forward_calls": 0,
        "_prefill_profile_input_rows": 0,
        "_prefill_profile_routed_assignments": 0,
        "_prefill_profile_node_routed_assignments": 0,
        "_prefill_profile_expert_shard_size": 0,
        "_prefill_profile_grouped_chunks": 0,
        "_prefill_profile_active_experts": 0,
        "_prefill_profile_wall_s": 0.0,
    })()
    layer.layer_idx = 3
    layer.buffer = SimpleNamespace(
        load=lambda idx: log("load", SimpleNamespace(K_latent=latent)),
        begin_prefetch_next=lambda idx: log("begin", None),
        allow_full_overwrite=lambda: log("allow", None),
        tp_group="tp8",
        tp_size=tp_size,
        expert_start=expert_start,
        experts_per_rank=num_local,
    )
    layer.down_proj = lambda t: torch.zeros((t.shape[0], latent))
    layer.norm = lambda t: log("norm", t)
    layer.up_proj = lambda t: log("up", torch.zeros((t.shape[0], hidden)))
    layer.chunk_rows = 2
    layer.post_chunk_rows = 3
    layer.forward = _isolated_method(
        path,
        "StreamedSP8MXFP4MoELayer",
        "forward",
        {
            "torch": torch,
            "time": time,
            "dist": fake_dist,
            "ResidentEPMXFP4MoELayer": FakeHelper,
        },
    ).__get__(layer)

    def gate(t):
        n = t.shape[0]
        if row_experts is None:
            idx = torch.zeros((n, 1, SP8_TOP_K), dtype=torch.int64)
        else:
            idx = torch.tensor(
                row_experts[:n], dtype=torch.int64
            ).reshape(n, 1, SP8_TOP_K)
        return idx, torch.ones((n, 1, SP8_TOP_K))

    return layer, torch.zeros((rows, hidden)), gate, num_rows


def test_streamed_sp8_forward_begins_ingress_before_grouped_expert_work():
    trace = []
    layer, x, gate, num_rows = _mock_sp8_moe_layer(
        trace, rows=3, num_rows=5, tp_size=2
    )

    output = layer.forward(x, gate, num_rows)
    ops = _ops(trace)

    assert tuple(output.shape) == (3, SP8_HIDDEN)
    # Ingress starts immediately after the current layer is loaded, before any
    # grouped expert work is enqueued.
    assert trace[:2] == ["load", "begin"]
    assert ops.index("begin") < ops.index("expert_path")
    # Permission to overwrite the compute shard is signalled exactly once, and
    # only after every reader of it -- the grouped kernels, their FP32 combine
    # and the node-local reduce-scatter that consumes them -- is enqueued.
    assert ops.count("allow") == 1
    assert ops.index("expert_path") < ops.index("combine")
    assert ops.index("combine") < ops.index("reduce_scatter")
    assert ops.index("reduce_scatter") < ops.index("allow")
    # norm/up-proj read no expert weight, so they may trail the release.
    assert ops.index("allow") < ops.index("norm") < ops.index("up")


def test_streamed_sp8_forward_gathers_only_inside_the_tp_group():
    trace = []
    layer, x, gate, num_rows = _mock_sp8_moe_layer(
        trace, rows=3, num_rows=5, tp_size=2, expert_start=4, num_local=4
    )

    layer.forward(x, gate, num_rows)

    # ntp = ceil(5/2) = 3, so the node lays out 2*3 = 6 rows.
    gathers = [entry for entry in trace if entry[0] == "all_gather"]
    assert [(entry[1], entry[2]) for entry in gathers] == [
        ((3, SP8_LATENT), "torch.float32"),      # latent, not the 7168 hidden
        ((3, SP8_TOP_K), "torch.int32"),         # topk indices
        ((3, SP8_TOP_K), "torch.float32"),       # topk weights
    ]
    # Every collective is the node-local TP group; nothing crosses nodes.
    assert {entry[-1] for entry in gathers} == {"tp8"}

    scatters = [entry for entry in trace if entry[0] == "reduce_scatter"]
    assert len(scatters) == 1
    # The combine is summed in FP32 over all 6 node rows before the single
    # bf16 downcast, and lands as this rank's own 3-row block.
    assert scatters[0][1] == (6, SP8_LATENT)
    assert scatters[0][2] == "torch.float32"
    assert scatters[0][3] == "SUM"
    assert scatters[0][4] == "tp8"

    # The grouped work covers all 6 node rows, chunked at chunk_rows=2, and it
    # runs against THIS rank's expert offset.
    assert [
        entry[1] for entry in trace if entry[0] == "expert_path"
    ] == [2, 2, 2]
    assert [entry for entry in trace if entry[0] == "helper"] == [
        ("helper", 1, 4)
    ]


def test_streamed_sp8_forward_empty_rank_still_enters_every_node_collective():
    trace = []
    # num_rows=1 over tp_size=2: rank 1 owns zero rows after the balanced split.
    layer, x, gate, num_rows = _mock_sp8_moe_layer(
        trace, rows=0, num_rows=1, tp_size=2, tp_rank=1
    )

    output = layer.forward(x, gate, num_rows)
    ops = _ops(trace)

    # It contributes no rows but still owns experts for the OTHER rank's rows,
    # so it must run the gathers, the grouped kernels and the reduce-scatter.
    assert tuple(output.shape) == (0, SP8_HIDDEN)
    assert ops.count("all_gather") == 3
    assert ops.count("reduce_scatter") == 1
    assert ops.count("expert_path") == 1
    assert ops.count("allow") == 1
    # No rows of its own means no post-combine work at all.
    assert "norm" not in ops and "up" not in ops


def test_streamed_sp8_forward_node_with_no_rows_skips_on_every_rank():
    trace = []
    layer, x, gate, num_rows = _mock_sp8_moe_layer(trace, rows=0, num_rows=0)

    output = layer.forward(x, gate, num_rows)

    # The early return keys off the NODE row count, which is identical on all
    # eight ranks, so they skip the node-local collectives together. It still
    # loads and releases, or the next layer's ingress handshake would hang.
    assert tuple(output.shape) == (0, SP8_HIDDEN)
    assert trace == ["load", "begin", "allow"]


def test_streamed_sp8_forward_pads_uneven_rows_out_of_expert_ownership():
    trace = []
    # ntp = 3 but this rank owns 2 real rows; the router still returns experts
    # for the zero pad row, and those must not be dispatched or counted.
    layer, x, gate, num_rows = _mock_sp8_moe_layer(
        trace,
        rows=2,
        num_rows=5,
        tp_size=2,
        tp_rank=1,
        expert_start=0,
        num_local=8,
        row_experts=[[0, 1], [2, 3], [4, 5]],
        profile=True,
    )

    layer.forward(x, gate, num_rows)
    cls = type(layer)

    # The mock replicates this rank's block on both gather slots, so the 6 node
    # rows carry 2 real rows x 2 slots x top_k=2 = 8 owned assignments. Experts
    # 4/5 belong to the pad row and are masked to -1 rather than counted.
    assert cls._prefill_profile_routed_assignments == 8
    assert cls._prefill_profile_active_experts == 4
    # input_rows stays this rank's REAL local rows.
    assert cls._prefill_profile_input_rows == 2
    assert cls._prefill_profile_expert_shard_size == 8


def test_streamed_sp8_profile_counts_only_this_rank_owned_assignments():
    trace = []
    # expert_start=4 with 4 local experts: of each row's 2 assignments only
    # experts 4 and 5 belong here, and the mock repeats the block per slot.
    layer, x, gate, num_rows = _mock_sp8_moe_layer(
        trace,
        rows=3,
        num_rows=6,
        tp_size=2,
        expert_start=4,
        num_local=4,
        row_experts=[[0, 4], [1, 5], [2, 9]],
        profile=True,
    )

    layer.forward(x, gate, num_rows)
    cls = type(layer)

    # Owned assignments are deduplicated by expert range, NOT input_rows*top_k:
    # ownership is skewed, and here this rank computes 4 of the node's 12.
    assert cls._prefill_profile_routed_assignments == 4
    assert cls._prefill_profile_active_experts == 2
    assert cls._prefill_profile_input_rows == 3
    # Group-level evidence: summing routed_assignments over the node's ranks
    # must reproduce node_routed_assignments exactly.
    assert cls._prefill_profile_node_routed_assignments == 6 * SP8_TOP_K
    assert cls._prefill_profile_expert_shard_size == 4
    assert cls._prefill_profile_forward_calls == 1


def test_streamed_sp8_grouped_chunk_is_wide_enough_for_large_token_runs():
    path = ROOT / "batchgen" / "moe" / "streamed_sp8_mxfp4.py"
    init = _function(path, "StreamedSP8MXFP4MoELayer", "__init__")
    defaults = {
        arg.arg: ast.literal_eval(default)
        for arg, default in zip(init.args.kwonlyargs, init.args.kw_defaults)
        if default is not None
    }
    # 256-row chunks over a 112-expert shard leave ~4 assignments per expert.
    # The resident-EP 256 bound answers a different HBM budget (896 experts
    # materialized on every rank), which this path no longer pays.
    assert defaults["chunk_rows"] == 2048
    assert defaults["post_chunk_rows"] == 8192


def test_streamed_sp8_make_shard_exposes_only_the_rank_expert_range():
    torch = pytest.importorskip("torch")
    path = ROOT / "batchgen" / "moe" / "streamed_sp8_mxfp4.py"
    from batchgen.models.moonshotai.kimi_linear.k3.mxfp4_layout import (
        MXFP4_GROUP_SIZE,
        ROUTED_EXPERT_PROJECTIONS,
        routed_expert_module_shapes,
    )

    intermediate, latent, num_local = 32, 32, 3
    shapes = routed_expert_module_shapes(intermediate, latent)
    buffer = type("Buffer", (), {})()
    buffer.experts_per_rank = num_local
    buffer.intermediate_size = intermediate
    buffer.latent_size = latent
    buffer.scale_bf16 = {}
    buffer.compute = {
        name: torch.zeros((num_local, *shape), dtype=torch.uint8)
        for name, shape in shapes.items()
    }
    buffer._expert_offsets = torch.arange(num_local, dtype=torch.int64)
    buffer._ptrs = _isolated_method(
        path, "StreamedSP8LayerBuffer", "_ptrs", {"torch": torch}
    ).__get__(buffer)
    # Both of these are ``@staticmethod`` on the real class, so binding them
    # would shift their arguments.
    for name in ("_expand_e8m0_into", "_offline_marlin_packed_view"):
        setattr(
            buffer,
            name,
            _isolated_method(
                path, "StreamedSP8LayerBuffer", name, {"torch": torch}
            ),
        )
    buffer._make_shard = _isolated_method(
        path,
        "StreamedSP8LayerBuffer",
        "_make_shard",
        {
            "torch": torch,
            "SimpleNamespace": SimpleNamespace,
            "MXFP4_GROUP_SIZE": MXFP4_GROUP_SIZE,
            "ROUTED_EXPERT_PROJECTIONS": ROUTED_EXPERT_PROJECTIONS,
        },
    ).__get__(buffer)

    shard = buffer._make_shard()

    # The grouped kernels see this rank's shard only, never all 896 experts.
    assert shard.num_local == num_local
    for field in (
        "gate_B_ptrs",
        "gate_scales_ptrs",
        "up_B_ptrs",
        "up_scales_ptrs",
        "down_B_ptrs",
        "down_scales_ptrs",
    ):
        assert getattr(shard, field).numel() == num_local, field
    # Pointer array 0 is the compute buffer's base, i.e. LOCAL expert 0 -- the
    # global offset lives in ``buffer.expert_start``, not in the pointers.
    assert int(shard.gate_B_ptrs[0]) == (
        buffer.compute["w1.weight_packed"].data_ptr()
    )


def test_streamed_sp8_assembly_is_a_same_rank_copy_and_frees_the_local_shard():
    path = ROOT / "batchgen" / "moe" / "streamed_sp8_mxfp4.py"
    source = ast.unparse(
        _function(path, "StreamedSP8LayerBuffer", "_assemble_compute_shard")
    )
    # Node-local weight all-gathers were the 12 GiB / 896-tiny-GEMM cost: the
    # eight TP slots own disjoint experts, so nothing is exchanged at all.
    assert "all_gather" not in source
    assert source.index("self.compute[tensor_name].copy_(") < source.index(
        "self._local_free.record(copy_stream)"
    )
    init = ast.unparse(_function(path, "StreamedSP8LayerBuffer", "__init__"))
    assert "self._local_free = torch.cuda.Event()" in init


def test_streamed_sp8_buffers_hold_only_this_rank_expert_shard():
    path = ROOT / "batchgen" / "moe" / "streamed_sp8_mxfp4.py"
    allocate = ast.unparse(
        _function(path, "StreamedSP8LayerBuffer", "_allocate")
    )
    # BOTH roles are 112 experts. A ``self.num_experts`` allocation here is the
    # regression this path exists to prevent.
    assert allocate.count("(self.experts_per_rank, *shape)") == 2
    assert "self.num_experts" not in allocate

    init = ast.unparse(_function(path, "StreamedSP8LayerBuffer", "__init__"))
    assert "self.expert_start = self.tp_rank * self.experts_per_rank" in init
    assert "torch.arange(self.experts_per_rank" in init


SIX_TENSORS = (
    "w1.weight_packed",
    "w1.weight_scale",
    "w3.weight_packed",
    "w3.weight_scale",
    "w2.weight_packed",
    "w2.weight_scale",
)


class _FakeShardTensor:
    def __init__(self, name, trace=None):
        self.name = name
        self.value = 1
        self._trace = trace

    def __getitem__(self, index):
        return SimpleNamespace(copy_=lambda source: None)

    def copy_(self, source):
        self._trace.append(("assemble", self.name, source.name))

    def numel(self):
        return 1024

    def element_size(self):
        return 1

    def fill_(self, value):
        self.value = int(value)

    def item(self):
        return self.value


class _FakeIngressStream:
    def __init__(self, trace):
        self.name = "prefetch"
        self._trace = trace

    def synchronize(self):
        self._trace.append(("synchronize",))


class _FakeStreamContext:
    def __init__(self, stream):
        self._stream = stream

    def __enter__(self):
        return self._stream

    def __exit__(self, *exc_info):
        return False


def _sp8_ingress_buffer(trace, *, cross_group, cross_root, cross_source):
    """Bind the real ingress/assembly methods onto fake streams and NCCL."""
    path = ROOT / "batchgen" / "moe" / "streamed_sp8_mxfp4.py"
    profile = type("Profile", (), {
        "_prefill_profile_enabled": True,
        "_prefill_profile_cross_broadcast_calls": 0,
        "_prefill_profile_cross_broadcast_bytes": 0,
        "_prefill_profile_cross_source": False,
        "_prefill_profile_cross_status_calls": 0,
        "_prefill_profile_cross_status_failures": 0,
    })
    globals_ = {
        "torch": SimpleNamespace(
            cuda=SimpleNamespace(
                stream=_FakeStreamContext,
                current_stream=lambda device: None,
            )
        ),
        "dist": SimpleNamespace(
            broadcast=lambda tensor, root, group=None: trace.append(
                ("broadcast", tensor.name, root, group)
            ),
            all_gather_into_tensor=lambda out, src, group=None: trace.append(
                ("all_gather", src.name, group)
            ),
        ),
        "validate_routed_expert_slot": lambda *args: None,
        "StreamedSP8MXFP4MoELayer": profile,
    }
    buffer = type("Buffer", (), {})()
    buffer.device = "cuda:0"
    buffer.shapes = {name: (2,) for name in SIX_TENSORS}
    buffer.local = {
        name: _FakeShardTensor(name, trace) for name in SIX_TENSORS
    }
    buffer.compute = {
        name: _FakeShardTensor(name, trace) for name in SIX_TENSORS
    }
    buffer._cross_status = _FakeShardTensor("source_status", trace)
    buffer.tp_group = "tp8"
    buffer.expert_start = 336
    buffer.experts_per_rank = 2
    buffer.acquire_batch_size = 2
    buffer.cross_group = cross_group
    buffer.cross_root = cross_root
    buffer.cross_source = cross_source
    buffer._acquires_from_host = cross_group is None or cross_source
    buffer._allocate = lambda: None
    buffer._local_free = SimpleNamespace(
        record=lambda stream: trace.append(("local_free",))
    )
    buffer.core_engine = SimpleNamespace(
        get_weights=lambda name, phase: trace.append(
            ("get_weights", name, phase)
        ) or {tensor: tensor for tensor in SIX_TENSORS},
        free_weights_buffer=lambda name: None,
    )
    for name in (
        "_acquire_local_shard",
        "_broadcast_source_status",
        "_broadcast_local_shard",
        "_assemble_compute_shard",
    ):
        setattr(
            buffer,
            name,
            _isolated_method(
                path, "StreamedSP8LayerBuffer", name, globals_
            ).__get__(buffer),
        )
    return buffer, profile


def _op_span(trace, op):
    positions = [i for i, entry in enumerate(trace) if entry[0] == op]
    return positions[0], positions[-1]


def test_hierarchical_gdr_broadcasts_six_tensors_before_the_local_assembly():
    trace = []
    stream = _FakeIngressStream(trace)
    buffer, profile = _sp8_ingress_buffer(
        trace, cross_group="cross3", cross_root=11, cross_source=True
    )

    buffer._acquire_local_shard(7, stream=stream)
    buffer._assemble_compute_shard(stream=stream)

    ops = [entry[0] for entry in trace]
    assert ops.count("get_weights") == 2
    assert ops.count("broadcast") == 7
    # Six same-rank local->compute copies, and no weight collective at all.
    assert ops.count("assemble") == 6
    assert "all_gather" not in ops
    assert [(entry[1], entry[2]) for entry in trace if entry[0] == "assemble"] == [
        (name, name) for name in SIX_TENSORS
    ]
    # Host ingress -> cross-node replication -> same-rank assembly. The copy
    # reads self.local, so it must follow every broadcast into it.
    assert _op_span(trace, "get_weights")[1] < _op_span(trace, "broadcast")[0]
    assert _op_span(trace, "broadcast")[1] < _op_span(trace, "assemble")[0]

    broadcasts = [
        entry for entry in trace
        if entry[0] == "broadcast" and entry[1] != "source_status"
    ]
    assert [entry[1] for entry in broadcasts] == list(SIX_TENSORS)
    assert {(entry[2], entry[3]) for entry in broadcasts} == {(11, "cross3")}
    assert profile._prefill_profile_cross_broadcast_calls == 6
    assert profile._prefill_profile_cross_broadcast_bytes == 6 * 1024
    assert profile._prefill_profile_cross_source is True
    assert profile._prefill_profile_cross_status_calls == 1
    assert profile._prefill_profile_cross_status_failures == 0


def test_hierarchical_gdr_non_source_rank_receives_without_host_ingress():
    trace = []
    stream = _FakeIngressStream(trace)
    buffer, profile = _sp8_ingress_buffer(
        trace, cross_group="cross3", cross_root=11, cross_source=False
    )

    buffer._acquire_local_shard(7, stream=stream)
    buffer._assemble_compute_shard(stream=stream)

    ops = [entry[0] for entry in trace]
    # 24 of the 32 ranks touch the host store for no expert at all, but they
    # still enter every cross-node broadcast and still assemble their shard.
    assert "get_weights" not in ops
    assert ops.count("broadcast") == 7
    assert ops.count("assemble") == 6
    assert "all_gather" not in ops
    assert _op_span(trace, "broadcast")[1] < _op_span(trace, "assemble")[0]
    assert profile._prefill_profile_cross_broadcast_calls == 6
    assert profile._prefill_profile_cross_source is False
    assert profile._prefill_profile_cross_status_calls == 1
    assert profile._prefill_profile_cross_status_failures == 0


def test_host_rdma_transport_runs_no_cross_node_collective():
    trace = []
    stream = _FakeIngressStream(trace)
    buffer, profile = _sp8_ingress_buffer(
        trace, cross_group=None, cross_root=None, cross_source=False
    )

    buffer._acquire_local_shard(7, stream=stream)
    buffer._assemble_compute_shard(stream=stream)

    ops = [entry[0] for entry in trace]
    assert ops.count("get_weights") == 2
    assert "broadcast" not in ops
    assert ops.count("assemble") == 6
    assert "all_gather" not in ops
    assert profile._prefill_profile_cross_broadcast_calls == 0
    assert profile._prefill_profile_cross_broadcast_bytes == 0
    assert profile._prefill_profile_cross_source is False
    assert profile._prefill_profile_cross_status_calls == 0
    assert profile._prefill_profile_cross_status_failures == 0


def test_hierarchical_gdr_source_failure_announces_status_before_raising():
    trace = []
    stream = _FakeIngressStream(trace)
    buffer, profile = _sp8_ingress_buffer(
        trace, cross_group="cross3", cross_root=11, cross_source=True
    )

    def fail_source(name, phase):
        raise RuntimeError("source load failed")

    buffer.core_engine.get_weights = fail_source
    with pytest.raises(RuntimeError, match="source load failed"):
        buffer._acquire_local_shard(7, stream=stream)

    broadcasts = [entry for entry in trace if entry[0] == "broadcast"]
    assert broadcasts == [("broadcast", "source_status", 11, "cross3")]
    assert "assemble" not in [entry[0] for entry in trace]
    assert profile._prefill_profile_cross_broadcast_calls == 0
    assert profile._prefill_profile_cross_status_calls == 1
    assert profile._prefill_profile_cross_status_failures == 1


def test_cross_node_broadcast_stays_inside_the_early_ingress_phase():
    path = ROOT / "batchgen" / "moe" / "streamed_sp8_mxfp4.py"
    acquire = ast.unparse(
        _function(path, "StreamedSP8LayerBuffer", "_acquire_local_shard")
    )
    # Phase one writes self.local only, so the broadcast inherits the existing
    # WAR event and overlaps the current layer's compute.
    assert "self._broadcast_local_shard(copy_stream)" in acquire
    # Phase two is the parked assembly copy; carrying the broadcast there would
    # serialize the cross-node transfer behind the overwrite handshake.
    assemble = ast.unparse(
        _function(path, "StreamedSP8LayerBuffer", "_assemble_compute_shard")
    )
    assert "_broadcast_local_shard" not in assemble


def test_streamed_sp8_node_empty_check_happens_after_the_load():
    path = ROOT / "batchgen" / "moe" / "streamed_sp8_mxfp4.py"
    function = _function(
        path, "StreamedSP8MXFP4MoELayer", "forward"
    )
    source = ast.unparse(function)
    assert source.index("shard = buffer.load") < source.index(
        "if num_rows == 0"
    )
    # A per-rank ``T == 0`` early return would desynchronize the node-local
    # gathers and the reduce-scatter below it.
    assert "if T == 0" not in source


def test_streamed_sp8_profile_counts_grouped_assignments():
    path = ROOT / "batchgen" / "moe" / "streamed_sp8_mxfp4.py"
    forward = _function(path, "StreamedSP8MXFP4MoELayer", "forward")
    source = ast.unparse(forward)

    assert "_prefill_profile_input_rows += T" in source
    assert (
        "_prefill_profile_routed_assignments += int(owned.sum().item())"
        in source
    )
    # The old accounting counted every assignment of every row this rank held.
    # Post-gather that is the whole node's routing on all eight ranks, so it
    # would over-report the node 8x.
    assert "int(topk_idx.numel())" not in source
    assert (
        "_prefill_profile_node_routed_assignments += num_rows * top_k" in source
    )
    assert "_prefill_profile_grouped_chunks +=" in source
    assert source.count("helper._expert_path") == 1


def test_streamed_sp8_profile_is_emitted_separately_from_legacy_expert_profile():
    path = ROOT / "batchgen" / "batchgen_worker.py"
    function = _function(path, "BatchGenWorker", "_config_prefill_for_batch")
    assert function is not None
    source = path.read_text()

    assert "StreamedSP8MXFP4MoELayer.reset_prefill_profile(True)" in source
    assert '"streamed_sp8": (' in source
    assert '"expert_consumer": (' in source
    assert "torch.cuda.reset_peak_memory_stats(self.local_rank)" in source
    for field in (
        "current_allocated_bytes",
        "current_reserved_bytes",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
        "free_bytes",
        "total_bytes",
    ):
        assert f'"{field}"' in source


def test_distributed_daemon_joins_bootstrap_threads_on_failure():
    path = ROOT / "core" / "Weights_Storage" / "distributed_weight_daemon.cpp"
    source = path.read_text()

    assert "auto join_accept_threads" in source
    assert "close_tcp_listeners" in source
    assert source.count("join_accept_threads();") >= 3
    assert "if (listener >= 0)" in source


def test_hierarchical_gdr_core_transport_fails_closed_without_host_network():
    storage = (
        ROOT / "core" / "Weights_Storage" / "Weights_Storage.cpp"
    ).read_text()
    daemon = (
        ROOT
        / "core"
        / "Weights_Storage"
        / "distributed_weight_daemon.cpp"
    ).read_text()

    assert 'config.value("transport", "host_rdma")' in storage
    assert 'transport != "host_rdma"' in storage
    assert 'transport != "hierarchical_gdr"' in storage
    assert "hierarchical_gdr rank requested non-local host module" in storage

    assert 'config.value("transport", "host_rdma")' in daemon
    assert "if (hierarchical_gdr)" in daemon
    assert "network_ready.store(true)" in daemon
    assert "BootstrapNetwork();" in daemon
    assert "hierarchical_gdr forbids remote host acquire" in daemon
    assert 'summary["transport"] = transport' in daemon


def test_streamed_sp8_reinterprets_offline_marlin_as_int32():
    torch = pytest.importorskip("torch")
    path = ROOT / "batchgen" / "moe" / "streamed_sp8_mxfp4.py"
    method = _isolated_method(
        path,
        "StreamedSP8LayerBuffer",
        "_offline_marlin_packed_view",
        {"torch": torch},
    )
    packed = torch.empty((3, 3072, 1792), dtype=torch.uint8)
    storage = packed.data_ptr()

    actual = method(packed, 3072, 3584)

    assert actual.dtype is torch.int32
    assert actual.shape == (3, 224, 6144)
    assert actual.data_ptr() == storage
    assert actual.numel() * actual.element_size() == packed.numel()


def test_weight_buffer_release_is_fail_safe_for_missing_sp8_lease():
    path = ROOT / "core" / "GPU_Weight_Buffer" / "GPU_Weight_Buffer.cpp"
    source = path.read_text()
    release = source[
        source.index("void GPU_Weight_Buffer::releaseBuffer"):
        source.index("module_weight_tensor_map GPU_Weight_Buffer::get_weights")
    ]
    get_weights = source[
        source.index("module_weight_tensor_map GPU_Weight_Buffer::get_weights"):
        source.index("void GPU_Weight_Buffer::weights_copy_complete")
    ]

    assert "module_in_buffers_.find(module_name)" in release
    assert "module_in_buffers_[module_name]" not in release
    assert "Cannot release missing weight-buffer lease" in release
    assert 'phase == "prefill_sp8"' in get_weights
    assert "std::chrono::seconds(300)" in get_weights
    assert "!hold_layer_batch" in get_weights


def test_worker_reads_batch_level_mode_before_configure_prefill():
    worker_path = ROOT / "batchgen" / "batchgen_worker.py"
    function = _function(
        worker_path, "BatchGenWorker", "_config_prefill_for_batch"
    )
    calls = [
        (
            node.lineno,
            (
                node.func.attr
                if isinstance(node.func, ast.Attribute)
                else node.func.id if isinstance(node.func, ast.Name) else None
            ),
        )
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
    ]
    set_mode = min(
        line for line, name in calls if name == "set_prefill_moe_mode"
    )
    configure = min(
        line for line, name in calls if name == "configure_prefill"
    )
    assert set_mode < configure


def test_prefill_sync_happens_before_decoder_layers():
    worker_path = ROOT / "batchgen" / "batchgen_worker.py"
    function = _function(worker_path, "BatchGenWorker", "prefill_prepacked")
    sync_lines = [
        node.lineno
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_sync_prefill_moe_rank_counts"
    ]
    layer_loops = [
        node.lineno
        for node in ast.walk(function)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Tuple)
        and any(
            isinstance(item, ast.Name) and item.id == "decoder_layer"
            for item in node.target.elts
        )
    ]
    assert sync_lines and layer_loops
    assert min(sync_lines) < min(layer_loops)


def test_prefill_schedule_checks_microbatch_count_before_sync_loop():
    worker_path = ROOT / "batchgen" / "batchgen_worker.py"
    function = _function(worker_path, "BatchGenWorker", "prefill_prepacked")
    gather_lines = [
        node.lineno
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "all_gather_into_tensor"
    ]
    sync_lines = [
        node.lineno
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_sync_prefill_moe_rank_counts"
    ]
    assert gather_lines and sync_lines
    assert min(gather_lines) < min(sync_lines)


def test_compact_resident_prefill_chunks_before_one_all_reduce():
    path = ROOT / "batchgen" / "moe" / "fused_moe_mxfp4_resident.py"
    function = _function(path, "ResidentEPMXFP4MoELayer", "_forward_ep")
    loops = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "chunk_start"
    ]
    assert len(loops) == 1
    all_reduce_calls = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "all_reduce"
    ]
    assert len(all_reduce_calls) == 1
    assert loops[0].lineno < all_reduce_calls[0].lineno
    bounded_calls = [
        node
        for node in ast.walk(loops[0])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_expert_path"
        and any(
            keyword.arg == "dispatch_capacity"
            for keyword in node.keywords
        )
    ]
    assert len(bounded_calls) == 1


def test_compact_resident_prefill_uses_preallocated_output():
    path = ROOT / "batchgen" / "moe" / "fused_moe_mxfp4_resident.py"
    function = _function(path, "ResidentEPMXFP4MoELayer", "_forward_ep")
    prefill_y_reads = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Attribute)
        and node.attr == "_prefill_y"
    ]
    output_allocations = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "empty"
        and any(
            isinstance(arg, ast.Tuple)
            and any(
                isinstance(dim, ast.Name) and dim.id == "num_global"
                for dim in arg.elts
            )
            for arg in node.args
        )
    ]
    assert prefill_y_reads
    assert not output_allocations


def test_compact_resident_prefill_chunk_policy():
    path = ROOT / "batchgen" / "moe" / "fused_moe_mxfp4_resident.py"
    choose = _isolated_function(path, "compact_prefill_chunk_rows")
    assert choose(16384, 2048) == 2048
    assert choose(65536, 2048) == 256
    assert choose(65536, 256) == 256
    with pytest.raises(ValueError, match="positive"):
        choose(65536, 0)


def test_resident_prefill_shared_merge_is_exact_and_in_place():
    path = (
        ROOT
        / "batchgen"
        / "models"
        / "moonshotai"
        / "kimi_linear"
        / "serving_modules.py"
    )
    merge = _isolated_function(path, "_merge_resident_prefill_shared")
    torch = pytest.importorskip("torch")
    torch.manual_seed(260821)
    routed = torch.randn((33, 17), dtype=torch.bfloat16)
    shared = torch.randn_like(routed)
    expected = routed + shared
    storage = routed.data_ptr()

    actual = merge(routed, shared)

    assert actual.data_ptr() == storage
    assert torch.equal(actual, expected)


def test_resident_prefill_sets_dense_and_shared_ffn_tiles():
    path = (
        ROOT
        / "batchgen"
        / "models"
        / "moonshotai"
        / "kimi_linear"
        / "Parallel_Strategy_Manager.py"
    )
    manager = type("Manager", (), {})()
    dense = type("FFN", (), {"_resident_prefill_token_tile": None})()
    shared = type("FFN", (), {"_resident_prefill_token_tile": None})()
    norm = type("KimiRMSNorm", (), {
        "_resident_prefill_token_tile": None,
    })()
    kda = type("KDAWrapper", (), {
        "_resident_prefill_segment_tokens": None,
    })()
    moe = type("MoE", (), {
        "_resident_ep_moe": None,
        "shared_experts": shared,
    })()
    layer = type("Layer", (), {"mlp": dense, "block_sparse_moe": moe})()
    manager.model = type("Model", (), {
        "model": type("Inner", (), {"layers": [layer]})(),
        "modules": lambda self: [norm, kda],
    })()
    manager._set_prefill_memory_tiling = _isolated_method(
        path,
        "KimiLinearParallelStrategyManager",
        "_set_prefill_memory_tiling",
    ).__get__(manager)
    method = _isolated_method(
        path,
        "KimiLinearParallelStrategyManager",
        "_set_resident_ep_prefill_enabled",
    ).__get__(manager)

    method(True)
    assert dense._resident_prefill_token_tile == 512
    assert shared._resident_prefill_token_tile == 512
    assert norm._resident_prefill_token_tile == 512
    assert kda._resident_prefill_segment_tokens == 8192
    assert moe._resident_ep_prefill_enabled is True

    method(False)
    assert dense._resident_prefill_token_tile is None
    assert shared._resident_prefill_token_tile is None
    assert norm._resident_prefill_token_tile is None
    assert kda._resident_prefill_segment_tokens is None
    assert moe._resident_ep_prefill_enabled is False


def test_worker_preallocates_resident_output_before_configure_prefill():
    worker_path = ROOT / "batchgen" / "batchgen_worker.py"
    function = _function(
        worker_path, "BatchGenWorker", "_config_prefill_for_batch"
    )
    calls = [
        (
            node.lineno,
            node.func.attr,
        )
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
    ]
    destroy = min(
        line for line, name in calls if name == "_destroy_gpu_paged_kv_cache"
    )
    sync = min(
        line for line, name in calls if name == "_sync_prefill_moe_rank_counts"
    )
    prepare = min(
        line
        for line, name in calls
        if name == "prepare_resident_ep_prefill_output"
    )
    configure = min(
        line for line, name in calls if name == "configure_prefill"
    )
    assert destroy < sync < prepare < configure


def test_worker_initializes_streamed_sp8_install_state():
    worker_path = ROOT / "batchgen" / "batchgen_worker.py"
    source = ast.unparse(_function(worker_path, "BatchGenWorker", "__init__"))
    assert "self._streamed_sp8_h2d_installed = False" in source
    assert "self._streamed_sp8_weight_copy_fingerprint = None" in source


def test_streamed_sp8_prefill_seeds_once_and_restarts_nonempty_workers():
    worker_path = ROOT / "batchgen" / "batchgen_worker.py"
    function = _function(
        worker_path, "BatchGenWorker", "_config_prefill_for_batch"
    )
    guards = _call_guards(function)

    # Every admission stops the copy engine and resets the profile. Source
    # ranks restart it from the preserved cursor; hierarchical non-sources
    # have an empty schedule and must never create an H2D thread.
    for name in (
        "stop_h2d_worker",
        "reset_weight_stream_profile",
    ):
        assert () in guards[name], name
    start_guards = guards["start_h2d_worker"]
    assert len(start_guards) == 2
    assert all(
        "any(self.weight_copy_task.values())" in stack
        for stack in start_guards
    )
    assert any("sp8_reentry" in stack for stack in start_guards)
    assert any("else: sp8_reentry" in stack for stack in start_guards)

    # Seeding the queue and dropping the prefill buffer would rewind the
    # daemon's free-running circular schedule, so they are first-install only.
    for name in (
        "clear_weight_copy_queue",
        "reset_prefill_buffer",
        "set_weight_copy_queue",
    ):
        assert guards[name], name
        assert all(
            any("sp8_reentry" in test for test in stack)
            for stack in guards[name]
        ), name

    # A re-entry that finds a different schedule must fail loudly rather than
    # stream the wrong experts against the preserved cursor.
    mismatch = next(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.If)
        and "_streamed_sp8_weight_copy_fingerprint" in ast.unparse(node.test)
    )
    assert any(isinstance(node, ast.Raise) for node in ast.walk(mismatch))


def test_resident_decode_preserves_streamed_sp8_weight_pipeline():
    worker_path = ROOT / "batchgen" / "batchgen_worker.py"
    function = _function(worker_path, "BatchGenWorker", "_load_decode_model")
    guards = _call_guards(function)

    assert () in guards["stop_h2d_worker"]
    assert () in guards["clear_kv_copy_queue"]
    # Both of these destroy streamed-SP8 state: one rewinds the queue cursor,
    # the other frees the already-prefetched routed-expert GPU slots.
    for name in ("clear_weight_copy_queue", "reset_decoding_buffer"):
        assert guards[name], name
        assert all(
            any("_streamed_sp8_h2d_installed" in test for test in stack)
            for stack in guards[name]
        ), name

    # A streamed decode task would reseed the same queue with a different
    # phase schedule.  Distributed streamed-SP8 therefore requires the
    # resident-EP decode path and fails before set_weight_copy_queue.
    routed_task = next(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.If)
        and "weight_copy_task.get('routed_expert')" in ast.unparse(node.test)
    )
    installed_guard = next(
        node
        for node in routed_task.body
        if isinstance(node, ast.If)
        and "_streamed_sp8_h2d_installed" in ast.unparse(node.test)
    )
    assert any(isinstance(node, ast.Raise) for node in installed_guard.body)
    assert routed_task.body.index(installed_guard) < next(
        index
        for index, node in enumerate(routed_task.body)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "set_weight_copy_queue"
    )


def test_weight_copy_task_fingerprint_pins_type_and_order():
    worker_path = ROOT / "batchgen" / "batchgen_worker.py"
    fingerprint = _isolated_method(
        worker_path, "BatchGenWorker", "_weight_copy_task_fingerprint"
    )
    base = {
        "attn": [],
        "routed_expert": ["routed_expert_0_336", "routed_expert_0_337"],
    }

    # Dict insertion order is not part of the schedule; list order is.
    assert fingerprint(base) == fingerprint(
        dict(reversed(list(base.items())))
    )
    assert fingerprint(base) != fingerprint(
        {**base, "routed_expert": list(reversed(base["routed_expert"]))}
    )
    assert fingerprint(base) != fingerprint(
        {**base, "routed_expert": base["routed_expert"][:1]}
    )
