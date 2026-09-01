import ast
import copy
import json
import math
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


def _name_call_guards(function, name):
    """Enclosing ``if`` tests for each bare ``name(...)`` call.

    ``_call_guards`` only sees ``obj.method()`` calls; the E0 boundary helper is
    a module-level function, so it needs its own walk.
    """
    found = []

    def visit(node, stack):
        if isinstance(node, ast.If):
            visit(node.test, stack)
            for child in node.body:
                visit(child, stack + (ast.unparse(node.test),))
            for child in node.orelse:
                visit(child, stack + ("else: " + ast.unparse(node.test),))
            return
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == name
        ):
            found.append(stack)
        for child in ast.iter_child_nodes(node):
            visit(child, stack)

    for statement in function.body:
        visit(statement, ())
    return found


def _tracing_profile_mark(trace):
    """Stand-in for the module's ``_profile_mark``: records, never waits."""

    def mark(marks):
        marks.append(SimpleNamespace(order=len(trace)))
        trace.append(("profile_mark",))

    return mark


def _isolated_function(path, function_name, globals_=None):
    tree = ast.parse(path.read_text())
    function = copy.deepcopy(
        next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == function_name
        )
    )
    module = ast.Module(body=[function], type_ignores=[])
    namespace = dict(globals_ or {})
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

    # The decode graph reserves its padding/warmup scratch slot during server
    # startup. Admission consumes the manager's remaining free count, so the
    # H200 plan needs 33 physical items to expose all 32 user sequences.
    fake_wrapper.state_manager = SimpleNamespace(
        get_stats=lambda: SimpleNamespace(num_free_state_items=32)
    )
    manager._kda_pool_slots = 33
    assert method() == {"max_sequences_per_node": 32}

    manager._attn_tp_size = 1
    assert method() == {"max_sequences_per_rank": 32}


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


def test_hierarchical_gdr_cross_node_map_follows_the_node_count():
    """world16 (2 nodes) and world32 (4 nodes) both derive from world_size."""
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

    dist.new_group = fake_new_group
    try:
        for world_size, expected_sources in (
            (16, [0, 1, 2, 3, 12, 13, 14, 15]),
            (32, [0, 1, 10, 11, 20, 21, 30, 31]),
        ):
            num_nodes = world_size // 8
            sources = []
            for global_rank in range(world_size):
                manager = type("Manager", (), {})()
                manager._hierarchical_gdr = True
                manager._attn_tp_size = 8
                manager._attn_tp_rank = global_rank % 8
                manager.world_size = world_size
                manager.global_rank = global_rank
                manager._cross_weight_group = None
                manager._cross_weight_root = None
                manager._cross_weight_source = False
                created.clear()
                build.__get__(manager)()

                # Cross group g stays [g + n*8], one rank per node.
                assert created == [
                    tuple(g + node * 8 for node in range(num_nodes))
                    for g in range(8)
                ]
                g = global_rank % 8
                assert manager._cross_weight_root == (
                    ((g * num_nodes) // 8) * 8 + g
                )
                assert 0 <= manager._cross_weight_root < world_size
                if manager._cross_weight_source:
                    sources.append(global_rank)

            # Eight sources, spread evenly so every node drives egress.
            assert sources == expected_sources
    finally:
        dist.new_group = original


def test_hierarchical_gdr_requires_supported_implicit_launch_order(monkeypatch):
    path = (
        ROOT
        / "batchgen"
        / "models"
        / "moonshotai"
        / "kimi_linear"
        / "Parallel_Strategy_Manager.py"
    )
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            nccl=SimpleNamespace(version=lambda: (2, 27, 5))
        ),
        version=SimpleNamespace(cuda="12.8"),
    )
    check = _isolated_method(
        path,
        "KimiLinearParallelStrategyManager",
        "_require_hierarchical_gdr_runtime",
        {"os": __import__("os"), "torch": fake_torch},
    )

    monkeypatch.delenv("NCCL_LAUNCH_ORDER_IMPLICIT", raising=False)
    with pytest.raises(RuntimeError, match="NCCL_LAUNCH_ORDER_IMPLICIT=1"):
        check()

    monkeypatch.setenv("NCCL_LAUNCH_ORDER_IMPLICIT", "1")
    fake_torch.cuda.nccl.version = lambda: (2, 25, 1)
    with pytest.raises(RuntimeError, match="NCCL >=2.26"):
        check()

    fake_torch.cuda.nccl.version = lambda: (2, 27, 5)
    fake_torch.version.cuda = "12.2"
    with pytest.raises(RuntimeError, match="CUDA >=12.3"):
        check()

    fake_torch.version.cuda = "12.8"
    check()


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


def test_streamed_sp8_layer_uses_planned_collective_stripe_threshold():
    path = (
        ROOT
        / "batchgen"
        / "models"
        / "moonshotai"
        / "kimi_linear"
        / "Parallel_Strategy_Manager.py"
    )
    source = ast.unparse(
        _function(
            path,
            "KimiLinearParallelStrategyManager",
            "_init_streamed_sp8_prefill",
        )
    )

    assert "collective_stripe_threshold_rows=int(getattr(" in source
    assert "'k3_prefill_collective_stripe_threshold_rows'" in source
    assert "32768" in source


def _psm_path():
    return (
        ROOT
        / "batchgen"
        / "models"
        / "moonshotai"
        / "kimi_linear"
        / "Parallel_Strategy_Manager.py"
    )


def test_streamed_sp8_init_attaches_order_wait_and_profiler():
    source = ast.unparse(
        _function(
            _psm_path(),
            "KimiLinearParallelStrategyManager",
            "_init_streamed_sp8_prefill",
        )
    )

    # The wait lives on attention, because the next TP8 launch after the gate
    # opens is the following layer's attention all-reduce, not an MoE call.
    assert (
        "order_wait = self._streamed_sp8_buffer"
        ".order_tp_collective_after_cross_launch" in source
    )
    assert "for module in self._streamed_sp8_attention_modules():" in source
    assert "module._streamed_sp8_order_wait = order_wait" in source
    assert (
        "module._streamed_sp8_profiler = StreamedSP8MXFP4MoELayer" in source
    )
    assert (
        "shared._streamed_sp8_profiler = StreamedSP8MXFP4MoELayer" in source
    )
    assert "dense._streamed_sp8_row_group =" in source
    assert "dense._streamed_sp8_profiler = StreamedSP8MXFP4MoELayer" in source


def _psm_layer(module, *, wrapped):
    """One decoder layer whose ``self_attn`` may or may not be wrapped."""
    moe = type("MoE", (), {})()
    moe._streamed_sp8_prefill_enabled = True
    moe._streamed_sp8_moe = object()
    moe.shared_experts = type("Shared", (), {})()
    moe.shared_experts._streamed_sp8_profiler = object()
    dense = type("Dense", (), {})()
    dense._streamed_sp8_row_group = object()
    dense._streamed_sp8_profiler = object()
    return type("Layer", (), {
        "self_attn": SimpleNamespace(module=module) if wrapped else module,
        "block_sparse_moe": moe,
        "mlp": dense,
    })()


def test_streamed_sp8_release_removes_the_attention_order_wait():
    path = _psm_path()
    manager = type("Manager", (), {})()
    modules = [type("Attn", (), {})() for _ in range(2)]
    for module in modules:
        module._streamed_sp8_order_wait = lambda: None
        module._streamed_sp8_profiler = object()
    # ``self_attn`` is the MLA/KDA streaming wrapper once prefill is
    # configured, but the serving methods live on the module it wraps; cover
    # the unwrapped shape too so the callback lands on the same object either
    # way.
    layers = [
        _psm_layer(modules[0], wrapped=True),
        _psm_layer(modules[1], wrapped=False),
    ]
    manager.model = type("Model", (), {
        "model": type("Inner", (), {"layers": layers})(),
    })()
    closed = []
    manager._streamed_sp8_buffer = SimpleNamespace(
        close=lambda: closed.append("close")
    )
    manager._streamed_sp8_attention_modules = _isolated_method(
        path,
        "KimiLinearParallelStrategyManager",
        "_streamed_sp8_attention_modules",
    ).__get__(manager)
    release = _isolated_method(
        path,
        "KimiLinearParallelStrategyManager",
        "_release_streamed_sp8_prefill",
        {"torch": SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False)
        )},
    ).__get__(manager)

    release()

    assert closed == ["close"]
    assert manager._streamed_sp8_buffer is None
    for layer in layers:
        assert layer.block_sparse_moe._streamed_sp8_moe is None
        assert layer.block_sparse_moe._streamed_sp8_prefill_enabled is False
        assert not hasattr(
            layer.block_sparse_moe.shared_experts,
            "_streamed_sp8_profiler",
        )
        assert not hasattr(layer.mlp, "_streamed_sp8_row_group")
        assert not hasattr(layer.mlp, "_streamed_sp8_profiler")
    # Decode reaches the same attention all-reduce helper. A surviving callback
    # would close over the buffer this release just tore down.
    for module in modules:
        assert not hasattr(module, "_streamed_sp8_order_wait")
        assert not hasattr(module, "_streamed_sp8_profiler")

    # configure_prefill -> configure_decoding releases twice; the second pass
    # has no buffer and no callback left and must still be a no-op.
    release()
    assert closed == ["close"]


def test_streamed_sp8_release_cleans_callbacks_when_close_raises():
    path = _psm_path()
    manager = type("Manager", (), {})()
    module = type("Attn", (), {})()
    module._streamed_sp8_order_wait = lambda: None
    module._streamed_sp8_profiler = object()
    layer = _psm_layer(module, wrapped=True)
    manager.model = type("Model", (), {
        "model": type("Inner", (), {"layers": [layer]})(),
    })()

    def close():
        raise RuntimeError("ingress failed")

    manager._streamed_sp8_buffer = SimpleNamespace(close=close)
    manager._streamed_sp8_attention_modules = _isolated_method(
        path,
        "KimiLinearParallelStrategyManager",
        "_streamed_sp8_attention_modules",
    ).__get__(manager)
    release = _isolated_method(
        path,
        "KimiLinearParallelStrategyManager",
        "_release_streamed_sp8_prefill",
        {"torch": SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False)
        )},
    ).__get__(manager)

    with pytest.raises(RuntimeError, match="ingress failed"):
        release()

    assert manager._streamed_sp8_buffer is None
    assert layer.block_sparse_moe._streamed_sp8_moe is None
    assert layer.block_sparse_moe._streamed_sp8_prefill_enabled is False
    assert not hasattr(
        layer.block_sparse_moe.shared_experts,
        "_streamed_sp8_profiler",
    )
    assert not hasattr(layer.mlp, "_streamed_sp8_row_group")
    assert not hasattr(layer.mlp, "_streamed_sp8_profiler")
    assert not hasattr(module, "_streamed_sp8_order_wait")
    assert not hasattr(module, "_streamed_sp8_profiler")


def test_streamed_sp8_attention_modules_unwraps_and_skips_missing_attention():
    manager = type("Manager", (), {})()
    method = _isolated_method(
        _psm_path(),
        "KimiLinearParallelStrategyManager",
        "_streamed_sp8_attention_modules",
    ).__get__(manager)

    manager.model = None
    assert list(method()) == []

    inner = object()
    layers = [
        type("Wrapped", (), {"self_attn": SimpleNamespace(module=inner)})(),
        type("Bare", (), {})(),
    ]
    manager.model = type("Model", (), {
        "model": type("Inner", (), {"layers": layers})(),
    })()
    assert list(method()) == [inner]


def test_distributed_k3_inherits_the_base_planner_prefill_ring_depth():
    path = (
        ROOT
        / "batchgen"
        / "models"
        / "moonshotai"
        / "kimi_linear"
        / "kimi_initializer.py"
    )
    source = ast.unparse(
        _function(path, "KimiLinearInitializer", "__init__")
    )

    # The routed-expert ring is bounded by the base planner's depth-8 slot
    # budget. Sizing it to a whole 112-expert TP8 shard here would reserve ~12
    # GiB of ring per rank for an ingress that now releases each bounded batch
    # as soon as its copies land.
    assert "num_prefill_module_buffer" not in source


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
    assert "all_gather_rows_add_" in calls
    assert "all_reduce" not in calls
    assert "all_gather" not in calls


def _serving_modules_path():
    return (
        ROOT
        / "batchgen"
        / "models"
        / "moonshotai"
        / "kimi_linear"
        / "serving_modules.py"
    )


def _sp8_serving_branch():
    tree = ast.parse(_serving_modules_path().read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "moe_forward_serving"
    )
    return next(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.If)
        and any(
            isinstance(name, ast.Name) and name.id == "streamed_sp8"
            for name in ast.walk(node.test)
        )
    )


def test_streamed_sp8_serving_opens_the_cross_gate_last_in_the_branch():
    branch = _sp8_serving_branch()
    calls = [
        (
            node.lineno,
            node.func.attr
            if isinstance(node.func, ast.Attribute)
            else node.func.id,
        )
        for node in ast.walk(branch)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Attribute, ast.Name))
    ]

    def last(name):
        return max(line for line, called in calls if called == name)

    # The gate must trail EVERY TP8 collective of the layer: the routed-output
    # all-gather and the shared expert's row-parallel all-reduce both live out
    # here, past the streamed-SP8 layer forward.
    assert (
        last("forward")
        < last("all_gather_rows_add_")
        < last("allow_cross_launch")
    )
    assert last("forward_into") < last("allow_cross_launch")
    # ...and it must be the LAST thing the branch does, so nothing new can be
    # slipped in between the gate and the next layer's attention.
    assert last("allow_cross_launch") == max(line for line, _ in calls)
    # The launch-order wait belongs to attention now, not to the MoE.
    assert "order_tp_collective_after_cross_launch" not in {
        called for _, called in calls
    }


def test_block_residual_merges_reuse_the_surviving_prefix():
    path = (
        ROOT
        / "batchgen"
        / "models"
        / "moonshotai"
        / "kimi_linear"
        / "model.py"
    )
    source = ast.unparse(
        _function(path, "KimiDecoderLayer", "_forward_attn_residual")
    )

    # Exact-64K is 896 MiB per hidden tensor per rank. Both attention and the
    # streamed MoE produce dead outputs beside the surviving prefix, so
    # neither residual merge may allocate a third full hidden tensor.
    assert source.count("prefix_sum.add_(hidden_states)") == 2


def test_streamed_sp8_layer_forward_touches_neither_side_of_the_handshake():
    path = ROOT / "batchgen" / "moe" / "streamed_sp8_mxfp4.py"
    forward = _function(path, "StreamedSP8MXFP4MoELayer", "forward")
    guards = _call_guards(forward)

    # Both halves moved out of the layer: the gate to the end of the serving
    # branch, the wait to the next attention's TP all-reduce. Only the
    # compute-buffer release is still issued from here.
    assert "allow_cross_launch" not in guards
    assert "order_tp_collective_after_cross_launch" not in guards
    assert guards["allow_full_overwrite"]


def test_prefill_attention_waits_for_cross_launch_before_its_tp_all_reduce():
    tree = ast.parse(_serving_modules_path().read_text())
    for name in ("_reduce_mla_tp_output", "kda_prefill_serving"):
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        )
        waits = [
            node.lineno
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_wait_streamed_sp8_cross_launch"
        ]
        reduces = [
            node.lineno
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "all_reduce"
        ]
        # One wait per all-reduce, and it is issued first: this all-reduce is
        # the next TP8 launch after the previous layer's MoE opened the gate.
        assert len(waits) == 1, name
        assert len(reduces) == 1, name
        assert waits[0] < reduces[0], name


def test_mla_tp_reduce_calls_the_installed_order_wait_then_all_reduce(
    monkeypatch,
):
    dist = pytest.importorskip("torch.distributed")
    path = _serving_modules_path()
    trace = []
    monkeypatch.setattr(
        dist,
        "all_reduce",
        lambda output, group=None: trace.append(("all_reduce", group)),
    )
    reduce_output = _isolated_function(
        path,
        "_reduce_mla_tp_output",
        {
            "_wait_streamed_sp8_cross_launch": _isolated_function(
                path, "_wait_streamed_sp8_cross_launch"
            ),
            "_begin_streamed_sp8_profile": _isolated_function(
                path, "_begin_streamed_sp8_profile"
            ),
            "_end_streamed_sp8_profile": _isolated_function(
                path, "_end_streamed_sp8_profile"
            ),
        },
    )
    profiler = SimpleNamespace(
        _prefill_profile_enabled=True,
        begin_profile_span=lambda: trace.append(("profile_begin",)) or "span",
        end_profile_span=lambda name, span: trace.append(
            ("profile_end", name, span)
        ),
    )
    module = SimpleNamespace(
        attn_tp_size=8,
        attn_tp_group="tp8",
        _streamed_sp8_order_wait=lambda: trace.append(("order_wait",)),
        _streamed_sp8_profiler=profiler,
    )
    output = object()

    assert reduce_output(module, output) is output
    assert trace == [
        ("order_wait",),
        ("profile_begin",),
        ("all_reduce", "tp8"),
        ("profile_end", "attention_reduce", "span"),
    ]

    # Decode reaches the SAME helper after the PSM released prefill. With the
    # callback removed there is no wait and no reference to a torn-down buffer.
    trace.clear()
    del module._streamed_sp8_order_wait
    del module._streamed_sp8_profiler
    reduce_output(module, output)
    assert trace == [("all_reduce", "tp8")]

    # attn_tp_size==1 has no TP all-reduce to order against at all.
    trace.clear()
    reduce_output(SimpleNamespace(attn_tp_size=1), output)
    assert trace == []


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


def _mock_sp8_buffer(
    trace, method_names, *, acquire=None, cross_group=None, profile=False
):
    """Bind real ``StreamedSP8LayerBuffer`` methods onto fake CUDA streams."""
    path = ROOT / "batchgen" / "moe" / "streamed_sp8_mxfp4.py"
    compute_stream = _FakeStream("compute", trace)
    prefetch_stream = _FakeStream("prefetch", trace)

    def record_call(op):
        def call(
            *_args,
            stream=None,
            cross_launch_gate=None,
            cross_launch_callback=None,
            profile_host=None,
            **_kwargs,
        ):
            def log(name):
                trace.append(
                    SimpleNamespace(
                        op=name,
                        event=None,
                        stream=(stream or compute_stream).name,
                        thread=threading.get_ident(),
                    )
                )

            # Host/ring ingress runs as soon as the thread starts; the
            # cross-node collectives are held behind the model thread's gate.
            log(op)
            if cross_launch_gate is not None:
                if not cross_launch_gate():
                    return
                log("cross_broadcast")
                if profile_host is not None:
                    profile_host.broadcast_enqueue_s = time.perf_counter()
            if cross_launch_callback is not None:
                cross_launch_callback()

        return call

    buffer = type("Buffer", (), {})()
    buffer.device = "cuda:0"
    buffer.cross_group = cross_group
    buffer._pending = None
    buffer._next_layer = {3: 4}
    buffer._prefetch_stream = prefetch_stream
    buffer._local_free = _FakeEvent(trace)
    buffer._acquire_local_shard = acquire or record_call("acquire")
    buffer._assemble_compute_shard = record_call("assemble")
    buffer.shard = SimpleNamespace(name="shard")
    buffer._make_shard = lambda: buffer.shard
    profile_cls = type("Profile", (), {
        "_prefill_profile_enabled": profile,
        "_prefill_profile_host_records": [],
    })
    globals_ = {
        "torch": SimpleNamespace(
            cuda=SimpleNamespace(
                Event=lambda: _FakeEvent(trace),
                current_stream=lambda device: compute_stream,
                set_device=lambda device: None,
            )
        ),
        "threading": threading,
        "time": time,
        "SimpleNamespace": SimpleNamespace,
        "PREFETCH_HANDOFF_TIMEOUT_S": 300.0,
        "StreamedSP8MXFP4MoELayer": profile_cls,
    }
    method_names = list(method_names)
    if "load" in method_names and "_load_layer_bytes" not in method_names:
        method_names.insert(method_names.index("load"), "_load_layer_bytes")
    for name in method_names:
        setattr(
            buffer,
            name,
            _isolated_method(
                path, "StreamedSP8LayerBuffer", name, globals_
            ).__get__(buffer),
        )
    return buffer


def test_streamed_sp8_transport_only_pass_drives_bytes_without_model_math():
    path = ROOT / "batchgen" / "moe" / "streamed_sp8_mxfp4.py"
    trace = []
    buffer = SimpleNamespace(
        cross_group="cross4",
        _pending=None,
        layer_indices=(1, 3, 7),
        _load_layer_bytes=lambda layer: trace.append(("bytes", layer)),
        begin_prefetch_next=lambda layer: trace.append(("begin", layer)),
        allow_full_overwrite=lambda: trace.append(("overwrite",)),
        allow_cross_launch=lambda: trace.append(("cross",)),
        _make_shard=lambda: pytest.fail("transport-only path expanded a shard"),
    )
    participate = _isolated_method(
        path,
        "StreamedSP8LayerBuffer",
        "participate_empty_prefill_pass",
    ).__get__(buffer)

    participate()

    assert trace == [
        ("bytes", 1), ("begin", 1), ("overwrite",), ("cross",),
        ("bytes", 3), ("begin", 3), ("overwrite",), ("cross",),
        ("bytes", 7), ("begin", 7), ("overwrite",), ("cross",),
    ]


def test_streamed_sp8_transport_only_pass_rejects_host_rdma():
    path = ROOT / "batchgen" / "moe" / "streamed_sp8_mxfp4.py"
    buffer = SimpleNamespace(cross_group=None, _pending=None, layer_indices=())
    participate = _isolated_method(
        path,
        "StreamedSP8LayerBuffer",
        "participate_empty_prefill_pass",
    ).__get__(buffer)

    with pytest.raises(RuntimeError, match="requires hierarchical GDR"):
        participate()


def test_streamed_sp8_ingress_starts_before_full_overwrite_is_permitted():
    trace = []
    model_thread = threading.get_ident()
    started = threading.Event()
    resume = threading.Event()

    def acquire(
        layer_idx,
        stream=None,
        cross_launch_gate=None,
        cross_launch_callback=None,
        **_kwargs,
    ):
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
        # host_rdma has no cross-node communicator to order.
        assert cross_launch_gate is None
        if cross_launch_callback is not None:
            cross_launch_callback()

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

    def acquire(layer_idx, stream=None, **_kwargs):
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
    with pytest.raises(
        RuntimeError, match="prefetch of layer 4 failed"
    ) as excinfo:
        buffer.load(4)

    assert excinfo.value.__cause__ is failure
    assert buffer._pending is None
    assert not any(entry.op == "assemble" for entry in trace)


def test_streamed_sp8_prefetch_failure_does_not_poison_teardown():
    trace = []

    def acquire(layer_idx, stream=None, **_kwargs):
        raise RuntimeError("weight lease timed out")

    buffer = _mock_sp8_buffer(
        trace,
        (
            "begin_prefetch_next",
            "allow_full_overwrite",
            "_wait_pending",
            "load",
            "close",
        ),
        acquire=acquire,
    )

    buffer.begin_prefetch_next(3)
    buffer.allow_full_overwrite()
    with pytest.raises(RuntimeError, match="prefetch of layer 4 failed"):
        buffer.load(4)

    assert buffer._pending is None
    buffer.close()


def test_streamed_sp8_handoff_timeout_matches_ring_lease_bound():
    path = ROOT / "batchgen" / "moe" / "streamed_sp8_mxfp4.py"
    tree = ast.parse(path.read_text())
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "PREFETCH_HANDOFF_TIMEOUT_S"
            for target in node.targets
        )
    )
    assert ast.literal_eval(assignment.value) >= 300.0


def test_distributed_weight_sharding_rejects_streamed_decode_on_every_rank():
    path = (
        ROOT
        / "batchgen"
        / "models"
        / "moonshotai"
        / "kimi_linear"
        / "Parallel_Strategy_Manager.py"
    )
    configure = _isolated_method(
        path, "KimiLinearParallelStrategyManager", "configure_decoding"
    )

    for cross_source in (False, True):
        manager = type("Manager", (), {})()
        manager._distributed_weight_sharded = True
        manager._cross_weight_source = cross_source
        manager._decode_moe_mode = lambda: "streamed"
        manager._stream_all_modules = False
        with pytest.raises(RuntimeError, match="require resident-EP decode"):
            configure.__get__(manager)()


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
    cross_group=None,
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

        def _expert_path(
            self,
            x_latent,
            topk_idx,
            count,
            packed_capacity=None,
            packed_max_rows=None,
        ):
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

    def begin_profile_span(cls):
        return object() if cls._prefill_profile_enabled else None

    def end_profile_span(cls, name, start):
        if start is not None:
            cls._prefill_profile_named_spans[name].append((start, object()))

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
        "_prefill_profile_layer_marks": [],
        "_prefill_profile_load_spans": [],
        "_prefill_profile_named_spans": {
            "grouped_expert_path": [],
            "grouped_combine": [],
        },
        "begin_profile_span": classmethod(begin_profile_span),
        "end_profile_span": classmethod(end_profile_span),
    })()
    layer.layer_idx = 3
    layer.buffer = SimpleNamespace(
        load=lambda idx: log("load", SimpleNamespace(K_latent=latent)),
        begin_prefetch_next=lambda idx: log("begin", None),
        allow_cross_launch=lambda: (
            log("allow_cross", None) if cross_group is not None else None
        ),
        order_tp_collective_after_cross_launch=lambda: (
            log("order_cross", None) if cross_group is not None else None
        ),
        allow_full_overwrite=lambda: log("allow", None),
        cross_group=cross_group,
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
            "compact_dispatch_route_stats_by_chunk": _isolated_function(
                ROOT / "batchgen" / "moe" / "fused_moe_mxfp4_resident.py",
                "compact_dispatch_route_stats_by_chunk",
                {"torch": torch},
            ),
            # The real helper records a CUDA timing event; the mocks run on CPU
            # tensors, so stand in for it and log the boundary instead.
            "_profile_mark": _tracing_profile_mark(trace),
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


def test_hierarchical_gdr_defers_cross_launch_past_every_tp_collective():
    trace = []
    layer, x, gate, num_rows = _mock_sp8_moe_layer(
        trace,
        rows=3,
        num_rows=5,
        tp_size=2,
        cross_group="cross4",
    )

    output = layer.forward(x, gate, num_rows)
    ops = _ops(trace)

    assert tuple(output.shape) == (3, SP8_HIDDEN)
    gather_positions = [
        index for index, op in enumerate(ops) if op == "all_gather"
    ]
    assert len(gather_positions) == 3
    # Host/ring ingress starts right after the load, exactly like host_rdma, so
    # it overlaps the router, down-proj and the gathers themselves.
    assert trace[:2] == ["load", "begin"]
    assert ops.index("begin") < min(gather_positions)
    # E1: releasing the cross-node communicator mid-layer only relocated the
    # peers' payload wait into the TP8 reduce-scatter, because implicit launch
    # ordering turns the host order into launch-completion edges. The layer
    # forward therefore touches NEITHER side of the handshake -- the serving
    # branch opens the gate after its routed all-gather and shared expert, and
    # the next layer's attention all-reduce performs the launch-order wait.
    assert "allow_cross" not in ops
    assert "order_cross" not in ops
    # Every TP8 collective of the layer still runs in the same order, and the
    # compute-buffer release still trails the reduce-scatter that reads it.
    assert max(gather_positions) < ops.index("expert_path")
    assert ops.index("combine") < ops.index("reduce_scatter")
    assert ops.index("reduce_scatter") < ops.index("allow")


def test_hierarchical_gdr_prefetch_hands_off_after_cross_launch():
    trace = []
    buffer = _mock_sp8_buffer(
        trace,
        (
            "begin_prefetch_next",
            "allow_cross_launch",
            "order_tp_collective_after_cross_launch",
            "allow_full_overwrite",
        ),
        cross_group="cross4",
        profile=True,
    )

    buffer.begin_prefetch_next(3)
    pending = buffer._pending
    buffer.allow_cross_launch()
    buffer.order_tp_collective_after_cross_launch()

    assert pending.cross_launch_enqueued.is_set()
    # The model thread waits only until every payload call has been issued.
    # No compute-stream completion wait serializes those payloads before TP8.
    assert [(entry.op, entry.stream) for entry in trace] == [
        ("wait", "prefetch"),
        ("acquire", "prefetch"),
        ("cross_broadcast", "prefetch"),
    ]
    assert pending.thread.is_alive()

    buffer.allow_full_overwrite()
    pending.thread.join(5)
    assert not pending.thread.is_alive()
    assert pending.error is None
    host = pending.profile_host
    assert host.gate_open_s is not None
    assert host.broadcast_enqueue_s >= host.gate_open_s
    assert host.order_wait_end_s >= host.order_wait_begin_s

    # Every layer's attention performs the wait, so a dense layer between two
    # MoE layers re-enters an already satisfied handoff. The measured span must
    # keep the first sample rather than collapse to the zero-cost re-entry.
    stamps = (host.order_wait_begin_s, host.order_wait_end_s)
    buffer.order_tp_collective_after_cross_launch()
    assert (host.order_wait_begin_s, host.order_wait_end_s) == stamps


def test_hierarchical_gdr_acquires_host_shard_before_the_cross_launch_gate():
    trace = []
    model_thread = threading.get_ident()
    acquired = threading.Event()

    def acquire(
        layer_idx,
        stream=None,
        cross_launch_gate=None,
        cross_launch_callback=None,
        **_kwargs,
    ):
        trace.append(
            SimpleNamespace(
                op="acquire",
                event=None,
                stream=stream.name,
                thread=threading.get_ident(),
            )
        )
        acquired.set()
        assert cross_launch_gate()
        trace.append(
            SimpleNamespace(
                op="cross_broadcast",
                event=None,
                stream=stream.name,
                thread=threading.get_ident(),
            )
        )
        cross_launch_callback()

    buffer = _mock_sp8_buffer(
        trace,
        (
            "begin_prefetch_next",
            "allow_cross_launch",
            "order_tp_collective_after_cross_launch",
            "allow_full_overwrite",
            "_wait_pending",
        ),
        acquire=acquire,
        cross_group="cross4",
    )

    buffer.begin_prefetch_next(3)
    pending = buffer._pending

    # Host/ring ingress starts immediately, on the prefetch thread, without
    # waiting for the model thread's TP8 gathers: that is the whole point of
    # starting the prefetch right after ``buffer.load``.
    assert acquired.wait(5)
    assert [(entry.op, entry.stream) for entry in trace] == [
        ("wait", "prefetch"),
        ("acquire", "prefetch"),
    ]
    assert trace[1].thread != model_thread

    # The cross-node collectives are NOT issued yet. Until the gate opens the
    # thread is parked, so the TP8 gathers keep their place ahead of every
    # cross-node launch in the one global host order.
    pending.thread.join(0.1)
    assert pending.thread.is_alive()
    assert not pending.cross_launch_allowed.is_set()
    assert not pending.cross_launch_enqueued.is_set()
    assert not any(entry.op == "cross_broadcast" for entry in trace)

    buffer.allow_cross_launch()
    buffer.order_tp_collective_after_cross_launch()
    assert [entry.op for entry in trace] == [
        "wait",
        "acquire",
        "cross_broadcast",
    ]

    buffer.allow_full_overwrite()
    pending.thread.join(5)
    assert not pending.thread.is_alive()
    assert pending.error is None


def test_hierarchical_gdr_close_cannot_strand_the_cross_launch_gate():
    trace = []
    buffer = _mock_sp8_buffer(
        trace,
        ("begin_prefetch_next", "_wait_pending", "close"),
        cross_group="cross4",
    )

    # A forward that raised between the gathers and the gate never releases it.
    buffer.begin_prefetch_next(3)
    pending = buffer._pending

    buffer.close()

    assert pending.cancelled
    assert not pending.thread.is_alive()
    assert pending.error is None
    assert buffer._pending is None
    # Teardown wakes the parked thread rather than granting the launch: the
    # peers whose broadcasts it would join may never enter them.
    ops = [entry.op for entry in trace]
    assert "cross_broadcast" not in ops
    assert "assemble" not in ops
    # Both handshakes are released, so neither this thread nor a later
    # ``order_tp_collective_after_cross_launch`` can be parked forever.
    assert pending.cross_launch_allowed.is_set()
    assert pending.cross_launch_enqueued.is_set()


def test_hierarchical_gdr_close_cancels_an_overwrite_granted_before_the_gate():
    trace = []
    buffer = _mock_sp8_buffer(
        trace,
        (
            "begin_prefetch_next",
            "allow_full_overwrite",
            "_wait_pending",
            "close",
        ),
        cross_group="cross4",
    )

    buffer.begin_prefetch_next(3)
    pending = buffer._pending
    # The grant order is REVERSED under the new schedule: the MoE forward
    # releases the compute buffer after its reduce-scatter, and only the end of
    # the serving branch opens the cross gate. A forward that raised in between
    # leaves the worker parked on the gate with the overwrite already granted,
    # which the old "ungranted overwrite covers an ungranted gate" shortcut
    # read as nothing to cancel -- and close() then hung on the parked thread.
    buffer.allow_full_overwrite()
    assert pending.overwrite_allowed.is_set()
    pending.thread.join(0.1)
    assert pending.thread.is_alive()

    buffer.close()

    assert pending.cancelled
    assert not pending.thread.is_alive()
    assert pending.error is None
    assert buffer._pending is None
    ops = [entry.op for entry in trace]
    # Cancelled, not granted: the peers are unwinding too, so no cross-node
    # collective is issued and the assembly that follows it is skipped.
    assert "cross_broadcast" not in ops
    assert "assemble" not in ops
    assert pending.cross_launch_allowed.is_set()
    assert pending.cross_launch_enqueued.is_set()
    assert (trace[-1].op, trace[-1].stream) == ("synchronize", "prefetch")


def test_streamed_sp8_close_keeps_a_fully_granted_prefetch():
    trace = []
    buffer = _mock_sp8_buffer(
        trace,
        (
            "begin_prefetch_next",
            "allow_cross_launch",
            "allow_full_overwrite",
            "_wait_pending",
            "close",
        ),
        cross_group="cross4",
    )

    buffer.begin_prefetch_next(3)
    pending = buffer._pending
    buffer.allow_full_overwrite()
    buffer.allow_cross_launch()

    buffer.close()

    # Both handshakes were granted in the new order, so teardown must drain the
    # completed prefetch rather than cancel work the peers already joined.
    assert not pending.cancelled
    assert pending.error is None
    ops = [entry.op for entry in trace]
    assert "cross_broadcast" in ops
    assert "assemble" in ops


def test_hierarchical_gdr_prefetch_error_unblocks_tp_launch_ordering():
    trace = []
    failure = RuntimeError("source shard failed")

    def acquire(layer_idx, stream=None, **_kwargs):
        raise failure

    buffer = _mock_sp8_buffer(
        trace,
        (
            "begin_prefetch_next",
            "order_tp_collective_after_cross_launch",
            "_wait_pending",
            "close",
        ),
        acquire=acquire,
        cross_group="cross4",
    )

    buffer.begin_prefetch_next(3)
    pending = buffer._pending
    with pytest.raises(RuntimeError, match="prefetch of layer 4 failed") as excinfo:
        buffer.order_tp_collective_after_cross_launch()

    pending.thread.join(5)
    assert not pending.thread.is_alive()
    assert excinfo.value.__cause__ is failure
    assert buffer._pending is None
    buffer.close()


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


def test_hierarchical_gdr_node_with_no_rows_leaves_the_gate_to_the_caller():
    trace = []
    layer, x, gate, num_rows = _mock_sp8_moe_layer(
        trace,
        rows=0,
        num_rows=0,
        cross_group="cross4",
    )

    output = layer.forward(x, gate, num_rows)

    assert tuple(output.shape) == (0, SP8_HIDDEN)
    # No reader of the compute shard is enqueued, so the overwrite is released
    # immediately. The gate is NOT opened here: the serving branch still enters
    # the routed all-gather and the shared expert on this node, and it opens
    # the gate after them exactly like a non-empty layer.
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
    assert len(cls._prefill_profile_named_spans["grouped_expert_path"]) == 3
    assert len(cls._prefill_profile_named_spans["grouped_combine"]) == 3


def test_streamed_sp8_profile_does_not_synchronize_each_layer():
    path = ROOT / "batchgen" / "moe" / "streamed_sp8_mxfp4.py"
    forward = _function(path, "StreamedSP8MXFP4MoELayer", "forward")
    snapshot = _function(
        path, "StreamedSP8MXFP4MoELayer", "prefill_profile_snapshot"
    )

    forward_calls = [
        node.func.attr
        for node in ast.walk(forward)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
    ]
    snapshot_calls = [
        node.func.attr
        for node in ast.walk(snapshot)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
    ]
    assert "item" not in forward_calls
    assert "unique" not in forward_calls
    assert "item" in snapshot_calls

    # E0 timing events are RECORDED in the measured path and consumed exactly
    # once, in the snapshot. A per-layer ``elapsed_time``/``synchronize`` would
    # park the host on the overlap this path exists to create.
    path_functions = {
        "forward": forward,
        "_acquire_local_shard": _function(
            path, "StreamedSP8LayerBuffer", "_acquire_local_shard"
        ),
        "_broadcast_local_shard": _function(
            path, "StreamedSP8LayerBuffer", "_broadcast_local_shard"
        ),
        "_profile_mark": next(
            node
            for node in ast.parse(path.read_text()).body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_profile_mark"
        ),
    }
    for name, function in path_functions.items():
        attrs = {
            node.func.attr
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
        }
        assert "elapsed_time" not in attrs, name
        # ``_acquire_local_shard`` keeps its per-batch ring-lease synchronize,
        # which predates E0 and is not a timing-event read.
        if name != "_acquire_local_shard":
            assert "synchronize" not in attrs, name
    assert "elapsed_time" in snapshot_calls
    assert "synchronize" not in snapshot_calls


def test_streamed_sp8_layer_boundaries_follow_the_r10_order():
    trace = []
    layer, x, gate, num_rows = _mock_sp8_moe_layer(
        trace,
        rows=3,
        num_rows=5,
        tp_size=2,
        profile=True,
        cross_group="cross4",
    )

    layer.forward(x, gate, num_rows)
    ops = _ops(trace)
    marks = [index for index, op in enumerate(ops) if op == "profile_mark"]

    # Two load-readiness boundaries followed by the nine layer boundaries.
    assert len(marks) == 11
    assert len(type(layer)._prefill_profile_load_spans) == 1
    load_record = type(layer)._prefill_profile_load_spans[0]
    assert len(load_record) == 2
    assert [event.order for event in load_record] == marks[:2]
    assert marks[0] < ops.index("load") < marks[1] < ops.index("begin")

    # Nine exact layer boundaries, one complete record, published as a unit.
    assert len(type(layer)._prefill_profile_layer_marks) == 1
    record = type(layer)._prefill_profile_layer_marks[0]
    assert len(record) == 9
    assert [event.order for event in record] == marks[2:]
    marks = marks[2:]

    def last(op):
        return max(index for index, name in enumerate(ops) if name == op)

    # b0 layer; b1/b2 gathers; b3/b4 grouped+combine; b5 post-fence;
    # b6/b7 reduce-scatter; b8 post-MoE. The nine-boundary layout and every
    # snapshot key are unchanged; b4->b5 no longer straddles a launch handoff
    # because the handshake moved out of the layer entirely.
    assert marks[0] < marks[1] < ops.index("all_gather")
    assert last("all_gather") < marks[2] < marks[3]
    assert marks[3] < ops.index("expert_path")
    assert last("combine") < marks[4] < marks[5]
    assert marks[5] < marks[6]
    assert marks[6] < ops.index("reduce_scatter") < marks[7]
    assert marks[7] < ops.index("allow")
    assert last("up") < marks[8]


def test_streamed_sp8_empty_node_publishes_no_layer_boundaries():
    trace = []
    layer, x, gate, num_rows = _mock_sp8_moe_layer(
        trace, rows=0, num_rows=0, profile=True, cross_group="cross4"
    )

    layer.forward(x, gate, num_rows)

    # The zero-row node still enters ``load`` in collective lockstep and records
    # that readiness span, but returns before the first compute-layer boundary.
    assert _ops(trace).count("profile_mark") == 2
    assert len(type(layer)._prefill_profile_load_spans) == 1
    assert type(layer)._prefill_profile_layer_marks == []


def test_streamed_sp8_e0_records_are_gated_on_the_profile_flag():
    path = ROOT / "batchgen" / "moe" / "streamed_sp8_mxfp4.py"
    forward = _function(path, "StreamedSP8MXFP4MoELayer", "forward")

    # Every boundary in the layer path, and the record that publishes them,
    # sits behind ``profile``; a disabled prefill records nothing at all.
    guards = _name_call_guards(forward, "_profile_mark")
    assert len(guards) == 11
    assert all(
        any("profile" in test for test in stack) for stack in guards
    ), guards
    publish = _call_guards(forward)["append"]
    assert publish and all(
        any("profile" in test for test in stack) for stack in publish
    ), publish

    broadcast = _function(
        path, "StreamedSP8LayerBuffer", "_broadcast_local_shard"
    )
    broadcast_guards = _name_call_guards(broadcast, "_profile_mark")
    assert len(broadcast_guards) == 2
    assert all(
        any("enabled" in test for test in stack)
        for stack in broadcast_guards
    ), broadcast_guards

    acquire = ast.unparse(
        _function(path, "StreamedSP8LayerBuffer", "_acquire_local_shard")
    )
    assert "enabled = profile._prefill_profile_enabled" in acquire
    assert "profile_host.acquisition_start_s = time.perf_counter()" in acquire
    assert "profile_host.acquisition_end_s = time.perf_counter()" in acquire
    assert "if enabled and source_error is None:" in acquire
    assert "profile_host.broadcast_enqueue_s = time.perf_counter()" in acquire

    allow = ast.unparse(
        _function(path, "StreamedSP8LayerBuffer", "allow_cross_launch")
    )
    assert "pending.profile_host.gate_open_s = time.perf_counter()" in allow
    order = ast.unparse(
        _function(
            path,
            "StreamedSP8LayerBuffer",
            "order_tp_collective_after_cross_launch",
        )
    )
    assert "pending.profile_host.order_wait_begin_s = time.perf_counter()" in order
    assert "pending.profile_host.order_wait_end_s = time.perf_counter()" in order


class _FakeTimingEvent:
    """A recorded E0 event; ``elapsed_time`` is ms to a later boundary."""

    def __init__(self, stamp_ms):
        self.stamp_ms = stamp_ms

    def elapsed_time(self, other):
        return other.stamp_ms - self.stamp_ms


class _NotATensor:
    pass


def _isolated_profile_class(synchronizes):
    """The real class profile state plus reset / stats / snapshot."""
    path = ROOT / "batchgen" / "moe" / "streamed_sp8_mxfp4.py"
    klass = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.ClassDef)
        and node.name == "StreamedSP8MXFP4MoELayer"
    )
    wanted = (
        "reset_prefill_profile",
        "begin_profile_span",
        "end_profile_span",
        "_profile_span_stats",
        "prefill_profile_snapshot",
    )
    body = [
        copy.deepcopy(node)
        for node in klass.body
        if (
            isinstance(node, ast.Assign)
            and all(
                isinstance(target, ast.Name)
                and target.id.startswith("_prefill_profile_")
                for target in node.targets
            )
        )
        or (isinstance(node, ast.FunctionDef) and node.name in wanted)
    ]
    module = ast.Module(
        body=[
            ast.ClassDef(
                name="Profile",
                bases=[],
                keywords=[],
                body=body,
                decorator_list=[],
            )
        ],
        type_ignores=[],
    )
    mark_stamps = []

    def profile_mark(marks):
        stamp = float(len(mark_stamps))
        mark_stamps.append(stamp)
        marks.append(_FakeTimingEvent(stamp))

    namespace = {
        "math": math,
        "_profile_mark": profile_mark,
        "torch": SimpleNamespace(
            Tensor=_NotATensor,
            cuda=SimpleNamespace(
                synchronize=lambda: synchronizes.append("sync")
            ),
        ),
    }
    exec(
        compile(ast.fix_missing_locations(module), str(path), "exec"),
        namespace,
    )
    return namespace["Profile"]


def test_streamed_sp8_external_profile_spans_publish_complete_pairs_only():
    profile = _isolated_profile_class([])

    profile.reset_prefill_profile(False)
    assert profile.begin_profile_span() is None

    profile.reset_prefill_profile(True)
    start = profile.begin_profile_span()
    assert profile._prefill_profile_named_spans["attention_reduce"] == []
    profile.end_profile_span("attention_reduce", start)
    assert len(profile._prefill_profile_named_spans["attention_reduce"]) == 1

    with pytest.raises(ValueError, match="unknown streamed-SP8 profile span"):
        profile.end_profile_span("not-a-span", profile.begin_profile_span())


def test_streamed_sp8_snapshot_aggregates_spans_and_reset_replaces_them():
    synchronizes = []
    profile = _isolated_profile_class(synchronizes)

    profile.reset_prefill_profile(True)
    for stamps in (
        (0.0, 1.0, 4.0, 5.0, 9.0, 12.0, 13.0, 17.0, 20.0),
        (20.0, 22.0, 32.0, 33.0, 40.0, 42.0, 43.0, 50.0, 52.0),
    ):
        profile._prefill_profile_layer_marks.append(
            tuple(_FakeTimingEvent(stamp) for stamp in stamps)
        )
    profile._prefill_profile_broadcast_spans.append(
        (_FakeTimingEvent(0.0), _FakeTimingEvent(7.0))
    )
    profile._prefill_profile_load_spans.extend([
        (_FakeTimingEvent(0.0), _FakeTimingEvent(60.0)),
        (_FakeTimingEvent(10.0), _FakeTimingEvent(30.0)),
    ])
    profile._prefill_profile_host_records.extend([
        SimpleNamespace(
            acquisition_start_s=1.0,
            acquisition_end_s=1.125,
            gate_open_s=2.0,
            order_wait_begin_s=3.0,
            order_wait_end_s=3.750,
            broadcast_enqueue_s=2.1,
        ),
        SimpleNamespace(
            acquisition_start_s=None,
            acquisition_end_s=None,
            gate_open_s=4.0,
            order_wait_begin_s=5.0,
            order_wait_end_s=5.250,
            broadcast_enqueue_s=4.1,
        ),
    ])
    for offset, name in enumerate(profile._prefill_profile_span_names, start=1):
        profile._prefill_profile_named_spans[name].append(
            (_FakeTimingEvent(0.0), _FakeTimingEvent(float(offset)))
        )

    snapshot = profile.prefill_profile_snapshot()

    # The existing end-of-prefill dependency chain has completed every event;
    # the snapshot consumes them without adding another device-wide fence.
    assert synchronizes == []
    # Adjacent boundaries, nearest-rank percentiles, milliseconds throughout.
    assert snapshot["grouped_execution"] == {
        "count": 2, "sum_ms": 11.0, "p50_ms": 4.0, "p95_ms": 7.0
    }
    assert snapshot["moe_pre_gather"] == {
        "count": 2, "sum_ms": 3.0, "p50_ms": 1.0, "p95_ms": 2.0
    }
    assert snapshot["moe_activation_gather"] == {
        "count": 2, "sum_ms": 13.0, "p50_ms": 3.0, "p95_ms": 10.0
    }
    assert snapshot["moe_profile_accounting"] == {
        "count": 2, "sum_ms": 2.0, "p50_ms": 1.0, "p95_ms": 1.0
    }
    assert snapshot["fence_stall"] == {
        "count": 2, "sum_ms": 5.0, "p50_ms": 2.0, "p95_ms": 3.0
    }
    assert snapshot["reduce_scatter"] == {
        "count": 2, "sum_ms": 11.0, "p50_ms": 4.0, "p95_ms": 7.0
    }
    assert snapshot["post_moe"] == {
        "count": 2, "sum_ms": 5.0, "p50_ms": 2.0, "p95_ms": 3.0
    }
    assert snapshot["broadcast_execution"] == {
        "count": 1, "sum_ms": 7.0, "p50_ms": 7.0, "p95_ms": 7.0
    }
    assert snapshot["load_readiness_wait"] == {
        "count": 2, "sum_ms": 80.0, "p50_ms": 20.0, "p95_ms": 60.0
    }
    # Host stamps are seconds on the wire and milliseconds in the snapshot.
    assert snapshot["host_acquisition"] == {
        "count": 1, "sum_ms": 125.0, "p50_ms": 125.0, "p95_ms": 125.0
    }
    assert snapshot["host_launch_handoff"] == {
        "count": 2, "sum_ms": 1000.0, "p50_ms": 250.0, "p95_ms": 750.0
    }
    assert snapshot["host_timestamp_counts"] == {
        "ingress_records": 2,
        "gate_open": 2,
        "broadcast_enqueue": 2,
    }
    for offset, name in enumerate(profile._prefill_profile_span_names, start=1):
        assert snapshot[name] == {
            "count": 1,
            "sum_ms": float(offset),
            "p50_ms": float(offset),
            "p95_ms": float(offset),
        }

    stale = profile._prefill_profile_layer_marks
    stale_loads = profile._prefill_profile_load_spans
    profile.reset_prefill_profile(False)

    # Reset REPLACES the record lists, so a snapshot already handed to the
    # caller keeps the records it aggregated from.
    assert profile._prefill_profile_layer_marks is not stale
    assert profile._prefill_profile_load_spans is not stale_loads
    assert len(stale) == 2
    assert len(stale_loads) == 2
    for key in (
        "broadcast_execution",
        "load_readiness_wait",
        "moe_pre_gather",
        "moe_activation_gather",
        "moe_profile_accounting",
        "fence_stall",
        "grouped_execution",
        "reduce_scatter",
        "post_moe",
        "host_acquisition",
        "host_launch_handoff",
        *profile._prefill_profile_span_names,
    ):
        assert profile.prefill_profile_snapshot()[key] == {
            "count": 0, "sum_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0
        }, key
    assert synchronizes == []


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
    buffer.compute = {
        name: torch.zeros((num_local, *shape), dtype=torch.uint8)
        for name, shape in shapes.items()
    }
    buffer._expert_offsets = torch.arange(num_local, dtype=torch.int64)
    buffer._ptrs = _isolated_method(
        path, "StreamedSP8LayerBuffer", "_ptrs", {"torch": torch}
    ).__get__(buffer)
    # This is a ``@staticmethod`` on the real class, so binding it would shift
    # its arguments.
    buffer._offline_marlin_packed_view = _isolated_method(
        path, "StreamedSP8LayerBuffer", "_offline_marlin_packed_view",
        {"torch": torch},
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
    assert int(shard.gate_scales_ptrs[0]) == (
        buffer.compute["w1.weight_scale"].data_ptr()
    )
    assert shard._tensors == (buffer.compute,)


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


class _FakeIngressEvent:
    def __init__(self, trace):
        self._trace = trace

    def record(self, stream):
        self._trace.append(("h2d_record", stream.name))

    def synchronize(self):
        self._trace.append(("h2d_synchronize",))


class _FakeStreamContext:
    def __init__(self, stream):
        self._stream = stream

    def __enter__(self):
        return self._stream

    def __exit__(self, *exc_info):
        return False


def _sp8_ingress_buffer(
    trace,
    *,
    cross_group,
    cross_root,
    cross_source,
    experts_per_rank=2,
    acquire_batch_size=2,
):
    """Bind the real ingress/assembly methods onto fake streams and NCCL."""
    path = ROOT / "batchgen" / "moe" / "streamed_sp8_mxfp4.py"
    profile = type("Profile", (), {
        "_prefill_profile_enabled": True,
        "_prefill_profile_cross_broadcast_calls": 0,
        "_prefill_profile_cross_broadcast_bytes": 0,
        "_prefill_profile_cross_source": False,
        "_prefill_profile_cross_status_calls": 0,
        "_prefill_profile_cross_status_failures": 0,
        "_prefill_profile_broadcast_spans": [],
        "_prefill_profile_host_records": [],
    })
    globals_ = {
        "SimpleNamespace": SimpleNamespace,
        "torch": SimpleNamespace(
            cuda=SimpleNamespace(
                Event=lambda: _FakeIngressEvent(trace),
                stream=_FakeStreamContext,
                current_stream=lambda device: None,
            )
        ),
        "time": time,
        "_profile_mark": _tracing_profile_mark(trace),
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
    buffer.experts_per_rank = experts_per_rank
    buffer.acquire_batch_size = acquire_batch_size
    buffer.cross_group = cross_group
    buffer.cross_root = cross_root
    buffer.cross_source = cross_source
    buffer._status_stream = _FakeIngressStream(trace)
    buffer._acquires_from_host = cross_group is None or cross_source
    buffer._allocate = lambda: None
    buffer._local_free = SimpleNamespace(
        record=lambda stream: trace.append(("local_free",))
    )
    buffer.core_engine = SimpleNamespace(
        get_weights=lambda name, phase: trace.append(
            ("get_weights", name, phase)
        ) or {tensor: tensor for tensor in SIX_TENSORS},
        free_weights_buffer=lambda name: trace.append(("free_weights", name)),
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
    # Four experts over a depth-2 ring: the source must complete the shard in
    # two bounded batches, so the ring is NOT required to hold it whole.
    buffer, profile = _sp8_ingress_buffer(
        trace,
        cross_group="cross3",
        cross_root=11,
        cross_source=True,
        experts_per_rank=4,
        acquire_batch_size=2,
    )

    buffer._acquire_local_shard(
        7,
        stream=stream,
        cross_launch_gate=lambda: trace.append(("cross_gate",)) or True,
        cross_launch_callback=lambda: trace.append(("cross_launch_handoff",)),
    )
    buffer._assemble_compute_shard(stream=stream)

    ops = [entry[0] for entry in trace]
    assert ops.count("get_weights") == 4
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
    # The host shard is pulled BEFORE the gate is consulted, so ingress starts
    # early; the status broadcast and all six payload broadcasts are issued
    # only after the model thread released it.
    assert _op_span(trace, "get_weights")[1] < _op_span(trace, "cross_gate")[0]
    assert _op_span(trace, "cross_gate")[1] < _op_span(trace, "broadcast")[0]
    assert _op_span(trace, "broadcast")[1] < _op_span(
        trace, "cross_launch_handoff"
    )[0]

    # Each bounded batch is synchronized and released on its own, and the first
    # batch's slots are freed BEFORE the second batch acquires -- that is what
    # lets a depth-2 ring carry a four-expert shard.
    assert ops.count("synchronize") == 2
    assert ops.count("free_weights") == 4
    assert trace[:10] == [
        ("get_weights", "routed_expert_7_336", "prefill_sp8"),
        ("get_weights", "routed_expert_7_337", "prefill_sp8"),
        ("synchronize",),
        ("free_weights", "routed_expert_7_336"),
        ("free_weights", "routed_expert_7_337"),
        ("get_weights", "routed_expert_7_338", "prefill_sp8"),
        ("get_weights", "routed_expert_7_339", "prefill_sp8"),
        ("synchronize",),
        ("free_weights", "routed_expert_7_338"),
        ("free_weights", "routed_expert_7_339"),
    ]
    # Every lease is released before the cross-node phase begins, so no slot is
    # held across the gate, the status broadcast or the payload broadcasts.
    assert _op_span(trace, "free_weights")[1] < _op_span(trace, "cross_gate")[0]
    # Releases are not deferred behind an H2D completion event any more: the
    # per-batch stream synchronize is the whole ordering.
    assert "h2d_record" not in ops
    assert "h2d_synchronize" not in ops

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

    # E0: exactly one complete broadcast pair, and it BRACKETS the six payload
    # broadcasts -- the status broadcast and the assembly stay outside it.
    assert len(profile._prefill_profile_broadcast_spans) == 1
    opened, closed = profile._prefill_profile_broadcast_spans[0]
    payload = [
        index
        for index, entry in enumerate(trace)
        if entry[0] == "broadcast" and entry[1] != "source_status"
    ]
    assert opened.order < payload[0]
    assert payload[-1] < closed.order
    assert closed.order < _op_span(trace, "assemble")[0]
    # The synchronous ingress records acquisition and enqueue timestamps; it
    # has no async gate or model-thread ordering wait.
    assert len(profile._prefill_profile_host_records) == 1
    host = profile._prefill_profile_host_records[0]
    assert host.acquisition_end_s >= host.acquisition_start_s
    assert host.broadcast_enqueue_s is not None
    assert host.gate_open_s is None
    assert host.order_wait_begin_s is None
    assert host.order_wait_end_s is None


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
    # A synchronous non-source rank pulls nothing from host but still records
    # that its cross-node payloads were enqueued.
    assert len(profile._prefill_profile_broadcast_spans) == 1
    assert len(profile._prefill_profile_host_records) == 1
    host = profile._prefill_profile_host_records[0]
    assert host.acquisition_start_s is None
    assert host.acquisition_end_s is None
    assert host.broadcast_enqueue_s is not None


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
    # No cross-node communicator exists, so neither the broadcast pair nor the
    # ordering handoff has anything to time -- but the host pull still does.
    assert "profile_mark" not in ops
    assert profile._prefill_profile_broadcast_spans == []
    assert len(profile._prefill_profile_host_records) == 1
    host = profile._prefill_profile_host_records[0]
    assert host.acquisition_end_s >= host.acquisition_start_s
    assert host.broadcast_enqueue_s is None


def test_streamed_sp8_synchronous_first_layer_still_times_the_handoff():
    trace = []
    stream = _FakeIngressStream(trace)
    # ``load`` acquires the first layer inline, with no gate to wait on.
    buffer, profile = _sp8_ingress_buffer(
        trace, cross_group="cross3", cross_root=11, cross_source=True
    )

    buffer._acquire_local_shard(7, stream=stream)

    assert len(profile._prefill_profile_host_records) == 1
    host = profile._prefill_profile_host_records[0]
    assert host.acquisition_end_s >= host.acquisition_start_s
    assert host.broadcast_enqueue_s is not None
    assert host.gate_open_s is None
    assert len(profile._prefill_profile_broadcast_spans) == 1
    opened, closed = profile._prefill_profile_broadcast_spans[0]
    assert opened.order < closed.order


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
        buffer._acquire_local_shard(
            7,
            stream=stream,
            cross_launch_gate=lambda: trace.append(("cross_gate",)) or True,
        )

    broadcasts = [entry for entry in trace if entry[0] == "broadcast"]
    assert broadcasts == [("broadcast", "source_status", 11, "cross3")]
    assert "assemble" not in [entry[0] for entry in trace]
    # Even the failure announcement is a cross-node collective, so it stays
    # behind the gate; announcing it early would reorder this rank's
    # communicator launches against its peers'.
    assert _op_span(trace, "cross_gate")[1] < _op_span(trace, "broadcast")[0]
    assert profile._prefill_profile_cross_broadcast_calls == 0
    assert profile._prefill_profile_cross_status_calls == 1
    assert profile._prefill_profile_cross_status_failures == 1
    # A failed ingress publishes nothing: no half broadcast pair, no
    # acquisition sample timing the failure, no ordering handoff.
    assert profile._prefill_profile_broadcast_spans == []
    assert profile._prefill_profile_host_records == []


def test_hierarchical_gdr_cancelled_gate_issues_no_cross_collective():
    trace = []
    stream = _FakeIngressStream(trace)
    buffer, profile = _sp8_ingress_buffer(
        trace, cross_group="cross3", cross_root=11, cross_source=True
    )

    # Teardown cancels the gate. The peers are unwinding too, so this rank must
    # issue neither the status broadcast nor any payload broadcast -- but it
    # must still release the ring leases its host ingress took.
    buffer._acquire_local_shard(
        7,
        stream=stream,
        cross_launch_gate=lambda: trace.append(("cross_gate",)) or False,
        cross_launch_callback=lambda: trace.append(("cross_launch_handoff",)),
    )

    ops = [entry[0] for entry in trace]
    assert ops.count("get_weights") == 2
    assert "broadcast" not in ops
    assert "cross_launch_handoff" not in ops
    assert "assemble" not in ops
    # The leases were already released by their own bounded batch, so a
    # cancelled gate returns early without stranding a single ring slot.
    assert ops.count("free_weights") == 2
    assert _op_span(trace, "free_weights")[1] < _op_span(trace, "cross_gate")[0]
    assert "h2d_record" not in ops
    assert "h2d_synchronize" not in ops
    assert profile._prefill_profile_cross_broadcast_calls == 0
    assert profile._prefill_profile_cross_status_calls == 0
    # A cancelled gate publishes no partial host or event record.
    assert "profile_mark" not in ops
    assert profile._prefill_profile_broadcast_spans == []
    assert profile._prefill_profile_host_records == []


def test_hierarchical_gdr_status_read_is_stream_ordered_after_broadcast():
    path = ROOT / "batchgen" / "moe" / "streamed_sp8_mxfp4.py"
    function = _function(
        path, "StreamedSP8LayerBuffer", "_broadcast_source_status"
    )
    item_calls = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "item"
    ]
    stream_contexts = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.With)
        and "torch.cuda.stream(stream)" in ast.unparse(node)
    ]
    assert len(item_calls) == 1
    assert len(stream_contexts) == 1
    assert item_calls[0] in set(ast.walk(stream_contexts[0]))


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
    assert "_prefill_profile_routed_assignments += owned.sum()" in source
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

    assert (
        "StreamedSP8MXFP4MoELayer.reset_prefill_profile(k3_prefill_profile)"
        in source
    )
    assert (
        "KimiK3MXFP4ExpertWrapper.reset_prefill_profile(k3_prefill_profile)"
        in source
    )
    assert "if streamed_sp8 or k3_prefill_profile:" in source
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


def test_streamed_sp8_detailed_profile_hooks_cover_the_non_load_gap():
    wrapper_source = (ROOT / "batchgen" / "models" / "moonshotai" /
                      "kimi_linear" / "wrappers.py").read_text()
    serving_source = _serving_modules_path().read_text()
    model_source = (ROOT / "batchgen" / "models" / "moonshotai" /
                    "kimi_linear" / "model.py").read_text()

    for name in ("kda_attention", "mla_attention", "mla_kv_offload"):
        assert f'end_profile_span("{name}"' in wrapper_source
    for name in (
        "attention_reduce",
        "routed_output_gather",
        "shared_expert",
        "moe_serving_total",
    ):
        assert f'"{name}"' in serving_source
    assert 'end_profile_span("shared_expert_reduce"' in model_source


def test_distributed_daemon_joins_bootstrap_threads_on_failure():
    path = ROOT / "core" / "Weights_Storage" / "distributed_weight_daemon.cpp"
    source = path.read_text()

    assert "auto join_accept_threads" in source
    assert "close_tcp_listeners" in source
    assert source.count("join_accept_threads();") >= 3
    assert "if (listener >= 0)" in source


def test_distributed_daemon_prepares_for_expected_worker_disconnects():
    daemon = (
        ROOT / "core" / "Weights_Storage" / "distributed_weight_daemon.cpp"
    ).read_text()
    header = (
        ROOT / "core" / "Weights_Storage" / "distributed_weight_daemon.h"
    ).read_text()
    binding = (ROOT / "core" / "batchgen_Binding.cpp").read_text()
    manager = (ROOT / "batchgen" / "server" / "worker_manager.py").read_text()
    stop = manager[manager.index("    def stop(self) -> None:") :]
    stop = stop[: stop.index("    def _get_worker_pids")]

    assert "void PrepareStop();" in header
    assert "void PrepareStop()" in daemon
    assert "PrepareStop();" in daemon
    assert 'def("prepare_stop", &DistributedWeightDaemon::PrepareStop)' in binding
    assert stop.index("distributed_weight_daemon.prepare_stop()") < stop.index(
        "os.kill(pid, signal.SIGTERM)"
    )
    assert stop.index("os.kill(pid, signal.SIGTERM)") < stop.index(
        "distributed_weight_daemon.stop()"
    )


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


def test_distributed_store_core_derives_and_checks_the_runtime_topology():
    storage = (
        ROOT / "core" / "Weights_Storage" / "Weights_Storage.cpp"
    ).read_text()
    daemon = (
        ROOT
        / "core"
        / "Weights_Storage"
        / "distributed_weight_daemon.cpp"
    ).read_text()

    assert "num_nodes != 2 && num_nodes != 4" in daemon
    assert "experts_per_owner = kExperts / num_nodes" in daemon
    assert "(rail - 1) % (num_nodes - 1)" in daemon
    assert "workers != kDefaultWorkers" in daemon
    assert "owner != expert / experts_per_owner" in storage
    assert "distributed routed-expert owner mismatch" in storage


def test_distributed_daemon_selects_validated_configured_rail_devices():
    daemon = (
        ROOT
        / "core"
        / "Weights_Storage"
        / "distributed_weight_daemon.cpp"
    ).read_text()

    assert 'config.contains("rail_devices")' in daemon
    assert "devices.size() != kRails" in daemon
    assert "rail_devices.at(rail_index)" in daemon
    assert (
        '"mlx5_bond_" + std::to_string(rail_index + 1) + ":1"'
        not in daemon
    )


def test_kimi_initializer_cross_checks_store_nodes_against_world_size():
    source = (
        ROOT
        / "batchgen"
        / "models"
        / "moonshotai"
        / "kimi_linear"
        / "kimi_initializer.py"
    ).read_text()

    assert 'self.world_size not in (16, 32)' in source
    assert 'distributed_config["num_nodes"] * 8 != self.world_size' in source
    assert "distributed weight config topology does not match" in source


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


def test_packed_compact_dispatch_avoids_per_chunk_host_sync():
    path = ROOT / "batchgen" / "moe" / "fused_moe_mxfp4_resident.py"
    function = _function(path, "ResidentEPMXFP4MoELayer", "_expert_path")
    compact_branch = next(
        node
        for node in function.body
        if isinstance(node, ast.If)
        and ast.unparse(node.test)
        == "self.compact_dispatch and dispatch_capacity is None"
    )
    compact_body = ast.Module(body=compact_branch.body, type_ignores=[])

    attribute_calls = {
        node.func.attr
        for node in ast.walk(compact_body)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
    }
    assert "tolist" not in attribute_calls
    assert "item" not in attribute_calls
    assert "bincount" not in attribute_calls

    assignments = {
        node.targets[0].id: ast.unparse(node.value)
        for node in ast.walk(compact_body)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    }
    assert assignments["capacity"] == "max(packed_capacity, 1)"
    assert assignments["max_m_tiles"] == (
        "max((packed_max_rows + 15) // 16, 1)"
    )
    assert assignments["mtp"] == "max(packed_max_rows, 1)"


def test_compact_dispatch_plans_all_chunk_route_stats_in_one_transfer():
    path = ROOT / "batchgen" / "moe" / "fused_moe_mxfp4_resident.py"
    torch = pytest.importorskip("torch")
    plan = _isolated_function(
        path,
        "compact_dispatch_route_stats_by_chunk",
        {"torch": torch},
    )
    topk_idx = torch.tensor(
        [
            [10, 11, 99],
            [10, 12, 99],
            [11, 11, 99],
            [9, 99, 99],
            [12, 99, 99],
        ],
        dtype=torch.int32,
    )
    assert plan(topk_idx, 10, 3, 2) == [[2, 4], [2, 2], [1, 1]]
    assert plan(topk_idx[:0], 10, 3, 2) == []
    with pytest.raises(ValueError, match="chunk_rows"):
        plan(topk_idx, 10, 3, 0)
    with pytest.raises(ValueError, match="num_local_experts"):
        plan(topk_idx, 10, 0, 2)


def test_streamed_packed_dispatch_passes_precomputed_chunk_bounds():
    path = ROOT / "batchgen" / "moe" / "streamed_sp8_mxfp4.py"
    tree = ast.parse(path.read_text())
    expert_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_expert_path"
    ]
    assert len(expert_calls) == 2
    assert all(
        {
            keyword.arg
            for keyword in call.keywords
        }.issuperset({"packed_capacity", "packed_max_rows"})
        for call in expert_calls
    )


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


def test_resident_decode_combine_masks_unwritten_nonlocal_rows_before_math():
    path = ROOT / "batchgen" / "moe" / "fused_moe_mxfp4_resident.py"
    torch = pytest.importorskip("torch")
    combine = _isolated_method(
        path,
        "ResidentEPMXFP4MoELayer",
        "_combine_fp32",
        {"torch": torch},
    )
    layer = SimpleNamespace(compact_dispatch=False)

    # Row 0 stands in for an empty local expert's unwritten output.  Non-local
    # routes carry -1 and are clamped to row 0 for the gather; NaN * 0 is NaN,
    # so masking only after multiplication silently corrupts every token.
    tokens, top_k, hidden = 4, 3, 5
    expert_out = torch.full((tokens + 1, hidden), float("nan"))
    expert_out[1:] = torch.arange(tokens * hidden).view(tokens, hidden)
    topk_pos = torch.full((tokens, top_k), -1, dtype=torch.int32)
    topk_pos[:, 0] = torch.arange(1, tokens + 1, dtype=torch.int32)
    topk_weight = torch.tensor(
        [[0.5, 0.3, 0.2]] * tokens, dtype=torch.float32
    )

    output = combine(
        layer,
        expert_out,
        topk_pos,
        topk_weight,
        tokens,
        hidden,
        top_k,
    )
    expected = expert_out[1:].float() * 0.5
    assert torch.equal(output, expected)
    assert torch.isfinite(output).all()

    # A shard with no owned routes must contribute exact zeros to the EP sum,
    # even when its entire expert output buffer is unwritten.
    no_local = combine(
        layer,
        expert_out,
        torch.full_like(topk_pos, -1),
        topk_weight,
        tokens,
        hidden,
        top_k,
    )
    assert torch.equal(no_local, torch.zeros_like(no_local))


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
    assert kda._resident_prefill_segment_tokens == 4096
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


def test_only_hierarchical_gdr_reseeds_streamed_sp8_reentry():
    path = (
        ROOT
        / "batchgen"
        / "models"
        / "moonshotai"
        / "kimi_linear"
        / "Parallel_Strategy_Manager.py"
    )
    method = _isolated_method(
        path,
        "KimiLinearParallelStrategyManager",
        "streamed_sp8_reseeds_h2d_on_reentry",
    )
    manager = SimpleNamespace(
        _hierarchical_gdr=True,
        prefill_uses_streamed_sp8=lambda: True,
    )

    assert method(manager) is True
    manager._hierarchical_gdr = False
    assert method(manager) is False
    manager._hierarchical_gdr = True
    manager.prefill_uses_streamed_sp8 = lambda: False
    assert method(manager) is False


def test_only_hierarchical_gdr_aligns_and_drives_empty_prefill_passes():
    path = (
        ROOT
        / "batchgen"
        / "models"
        / "moonshotai"
        / "kimi_linear"
        / "Parallel_Strategy_Manager.py"
    )
    predicate = _isolated_method(
        path,
        "KimiLinearParallelStrategyManager",
        "streamed_sp8_requires_global_pass_alignment",
    )
    drive = _isolated_method(
        path,
        "KimiLinearParallelStrategyManager",
        "run_streamed_sp8_transport_only_prefill",
    )
    trace = []
    manager = SimpleNamespace(
        _hierarchical_gdr=True,
        _streamed_sp8_buffer=SimpleNamespace(
            participate_empty_prefill_pass=lambda: trace.append("pass")
        ),
        prefill_uses_streamed_sp8=lambda: True,
    )
    manager.streamed_sp8_requires_global_pass_alignment = (
        predicate.__get__(manager)
    )

    assert manager.streamed_sp8_requires_global_pass_alignment() is True
    drive(manager, 2)
    assert trace == ["pass", "pass"]

    manager._hierarchical_gdr = False
    assert manager.streamed_sp8_requires_global_pass_alignment() is False
    with pytest.raises(RuntimeError, match="only valid for hierarchical GDR"):
        drive(manager, 1)
    assert trace == ["pass", "pass"]


@pytest.mark.parametrize(
    "pass_counts",
    (
        (1, 0),       # singleton on a two-node deployment
        (1, 1),       # balanced two-node admission
        (2, 0, 1, 0), # partially populated four-node deployment
    ),
)
def test_worker_aligns_hierarchical_prefill_pass_counts(pass_counts):
    path = ROOT / "batchgen" / "batchgen_worker.py"
    method = _isolated_method(
        path,
        "BatchGenWorker",
        "_streamed_sp8_prefill_pass_alignment",
        {"List": list, "Tuple": tuple},
    )
    global_count = max(pass_counts)

    class FakeCount:
        def __init__(self, value):
            self.value = value

        def item(self):
            return self.value

    for rank_count in pass_counts:
        calls = []
        fake_dist = SimpleNamespace(
            ReduceOp=SimpleNamespace(MAX="MAX"),
            all_reduce=lambda tensor, op: (
                calls.append((tensor.value, op)),
                setattr(tensor, "value", global_count),
            )[-1],
        )
        method.__globals__["torch"] = SimpleNamespace(
            int64="int64",
            tensor=lambda values, dtype, device: FakeCount(values[0]),
        )
        method.__globals__["dist"] = fake_dist
        worker = SimpleNamespace(
            rank=rank_count,
            torch_device="cuda:0",
            parallel_manager=SimpleNamespace(
                streamed_sp8_requires_global_pass_alignment=lambda: True
            ),
            _prefill_model_pass_count=lambda batch: len(batch),
        )

        assert method(worker, list(range(rank_count))) == (
            rank_count,
            global_count,
        )
        assert calls == [(rank_count, "MAX")]


def test_worker_skips_pass_collective_outside_hierarchical_streamed_sp8():
    path = ROOT / "batchgen" / "batchgen_worker.py"
    method = _isolated_method(
        path,
        "BatchGenWorker",
        "_streamed_sp8_prefill_pass_alignment",
        {"List": list, "Tuple": tuple},
    )
    worker = SimpleNamespace(
        parallel_manager=SimpleNamespace(
            streamed_sp8_requires_global_pass_alignment=lambda: False
        ),
        _prefill_model_pass_count=lambda batch: pytest.fail(
            "non-hierarchical path calculated a pass count"
        ),
    )

    assert method(worker, [1]) == (0, 0)


def test_worker_runs_missing_weight_passes_outside_the_local_row_guard():
    path = ROOT / "batchgen" / "batchgen_worker.py"
    generate = _function(path, "BatchGenWorker", "generate")
    guards = _call_guards(generate)

    driver_guards = guards["run_streamed_sp8_transport_only_prefill"]
    assert len(driver_guards) == 1
    assert any(
        "transport_only_passes" in test for test in driver_guards[0]
    )
    assert not any(
        "local_prefill_indices" in test for test in driver_guards[0]
    )

    call_lines = {
        node.func.attr: node.lineno
        for node in ast.walk(generate)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {
            "_config_prefill_for_batch",
            "_streamed_sp8_prefill_pass_alignment",
            "run_streamed_sp8_transport_only_prefill",
            "_unregister_fp8_weights",
        }
    }
    assert (
        call_lines["_config_prefill_for_batch"]
        < call_lines["_streamed_sp8_prefill_pass_alignment"]
        < call_lines["run_streamed_sp8_transport_only_prefill"]
        < call_lines["_unregister_fp8_weights"]
    )


def test_worker_counts_the_same_prefill_plan_that_execution_uses():
    path = ROOT / "batchgen" / "batchgen_worker.py"
    count = _function(path, "BatchGenWorker", "_prefill_model_pass_count")
    execute = _function(path, "BatchGenWorker", "prefill_prepacked")

    def calls_plan(function):
        return sum(
            1
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_plan_prefill_micro_batches"
        )

    assert calls_plan(count) == 1
    assert calls_plan(execute) == 1


def test_streamed_prefill_releases_phase_inactive_resident_decode_shards():
    path = (
        ROOT
        / "batchgen"
        / "models"
        / "moonshotai"
        / "kimi_linear"
        / "Parallel_Strategy_Manager.py"
    )
    trace = []
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(empty_cache=lambda: trace.append("empty_cache"))
    )
    fake_logging = SimpleNamespace(info=lambda *args: trace.append("log"))
    release = _isolated_method(
        path,
        "KimiLinearParallelStrategyManager",
        "_release_resident_ep_decode",
        {"torch": fake_torch, "logging": fake_logging},
    )

    class Shard:
        def __init__(self, size):
            self.size = size

        def nbytes(self):
            return self.size

    residents = [
        SimpleNamespace(shard=Shard(11)),
        SimpleNamespace(shard=Shard(13)),
    ]
    layers = [
        SimpleNamespace(
            block_sparse_moe=SimpleNamespace(_resident_ep_moe=resident)
        )
        for resident in residents
    ]
    graph = SimpleNamespace(release=lambda: trace.append("graph_release"))
    manager = SimpleNamespace(
        _resident_ep_built=True,
        _decode_graph=graph,
        model=SimpleNamespace(model=SimpleNamespace(layers=layers)),
        rank=0,
    )

    assert release(manager) == 24
    assert manager._resident_ep_built is False
    assert manager._decode_graph is None
    assert all(
        layer.block_sparse_moe._resident_ep_moe is None for layer in layers
    )
    assert trace[:2] == ["graph_release", "empty_cache"]
    assert release(manager) == 0
    assert trace.count("empty_cache") == 1


def test_streamed_sp8_installer_selects_transport_specific_reentry():
    worker_path = ROOT / "batchgen" / "batchgen_worker.py"
    function = _function(
        worker_path,
        "BatchGenWorker",
        "_install_prefill_weight_copy_pipeline",
    )
    guards = _call_guards(function)

    # Every installer invocation stops the copy engine and resets the profile.
    # Host-RDMA sources resume the cursor; hierarchical GDR sources reseed it.
    # Hierarchical non-sources stay empty and never create an H2D thread.
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
    assert any(
        any("reseed_reentry" in test for test in stack)
        for stack in start_guards
    )
    assert any(
        any(test.startswith("else:") for test in stack)
        for stack in start_guards
    )

    # Host-RDMA preserves its daemon cursor. Hierarchical GDR has no remote
    # generations, so it is allowed to reseed the queue and GPU ring.
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
        assert any(
            any("reseed_reentry" in test for test in stack)
            for stack in guards[name]
        ), name

    # A re-entry that finds a different schedule must fail loudly rather than
    # stream a task that no longer matches the installed pipeline.
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
    # Decode cannot replace the prefill queue or resize its routed-expert ring.
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


def _weights_storage_source():
    return (
        ROOT / "core" / "Weights_Storage" / "Weights_Storage.cpp"
    ).read_text()


def _cpp_section(source, begin, end):
    """The text of one C++ definition, delimited by two unique markers."""
    start = source.index(begin)
    return source[start:source.index(end, start)]


def _init_distributed_section(source):
    return _cpp_section(
        source,
        "void Weights_Storage::InitDistributed",
        "Weights_Storage::active_lease",
    )


def test_hierarchical_gdr_pin_source_owns_the_slot_it_registers():
    """``(g * num_nodes) / 8`` names the node that owns slot ``g``.

    The compact store's owner column is ``expert / (896 / num_nodes)``. A
    worker may only pin experts its own node owns, otherwise the bytes it
    registers are not even resident in its copy of the store.
    """
    for num_nodes in (2, 4):
        experts_per_owner = 896 // num_nodes
        sources = []
        for g in range(8):
            source_node = (g * num_nodes) // 8
            assert 0 <= source_node < num_nodes
            owners = {
                expert // experts_per_owner
                for expert in range(112 * g, 112 * (g + 1))
            }
            assert owners == {source_node}
            sources.append(source_node)
        # Every node sources an equal share of the eight slots, and the eight
        # slots together still cover all 896 experts exactly once.
        assert sorted(sources) == sources
        assert all(
            sources.count(node) == 8 // num_nodes for node in range(num_nodes)
        )

    init = _init_distributed_section(_weights_storage_source())
    assert (
        "(this->device_id_ * num_nodes) / kKimiK3Workers" in init
    )
    assert (
        "this->pin_expert_begin_ = this->device_id_ * kKimiK3ExpertsPerWorker"
        in init
    )
    assert "this->local_node_rank_ == pin_source_node" in init


def test_hierarchical_gdr_registers_only_the_slot_experts_of_the_source():
    source = _weights_storage_source()
    init = _init_distributed_section(source)

    # 112 experts per TP slot, 896 across the node.
    assert "constexpr int kKimiK3ExpertsPerWorker = kKimiK3Experts / kKimiK3Workers;" in source
    assert "constexpr int kKimiK3Workers = 8;" in source

    # Only a source collects intervals, and only for its own slot.
    assert (
        "if (this->pin_source_ && expert >= this->pin_expert_begin_ &&\n"
        "                expert < this->pin_expert_end_) {" in init
    )
    assert "pin_intervals.emplace_back(" in init

    # host_rdma keeps the whole compact store pinned; a non-source
    # hierarchical worker collects nothing and therefore pins nothing.
    assert "if (this->hierarchical_gdr_) {" in init
    assert "pin_intervals = {{0, this->compact_bytes_}};" in init

    # The full mapping and every local module survive the narrowing, because
    # get_tensor still hands out pageable views of them.
    assert "mmap(nullptr, this->compact_bytes_" in init
    local_insert = _cpp_section(
        init,
        "if (owner < 0 || owner == this->local_node_rank_)",
        "++local_tensors;",
    )
    assert "module_weights_storage_[module_key][tensor_key]" in local_insert
    assert "pin_" not in local_insert

    # The replicated prefix (owner == -1) is never a pin candidate: interval
    # collection lives inside the owned-expert branch only.
    owned_branch = _cpp_section(
        init, "if (owner >= 0) {", "if (owner < 0 || owner =="
    )
    assert "pin_intervals.emplace_back(" in owned_branch
    assert init.count("pin_intervals.emplace_back(") == 1


def test_distributed_pin_intervals_are_aligned_coalesced_and_clamped():
    source = _weights_storage_source()
    merge = _cpp_section(
        source,
        "std::vector<std::pair<int64_t, int64_t>> merge_pin_intervals(",
        "}  // namespace",
    )

    assert "constexpr int64_t kPinAlignment = 2 * 1024 * 1024;" in source
    # Start rounds down, end rounds up.
    assert "(interval.first / kPinAlignment) * kPinAlignment" in merge
    assert (
        "((interval.second + kPinAlignment - 1) / kPinAlignment) *"
        in merge
    )
    # Clamped to the store, so the unaligned tail cannot run past the mapping.
    assert "std::max<int64_t>(begin, 0)" in merge
    assert "std::min<int64_t>(end, limit)" in merge
    # Sorted, then touching or overlapping ranges are merged (``<=``, not
    # ``<``): 2 MiB rounding makes adjacent experts share a boundary page.
    assert "std::sort(aligned.begin(), aligned.end());" in merge
    assert "range.first <= merged.back().second" in merge
    assert "std::max(merged.back().second, range.second)" in merge

    init = _init_distributed_section(source)
    assert "merge_pin_intervals(std::move(pin_intervals)" in init


def test_distributed_unregisters_exactly_the_ranges_it_registered():
    source = _weights_storage_source()
    init = _init_distributed_section(source)
    lines = init.splitlines()

    # Every successful registration is recorded, immediately: a throw from a
    # later one must not strand the earlier pins.
    registrations = [
        index for index, line in enumerate(lines)
        if "cudaHostRegister(" in line
    ]
    assert len(registrations) == 2  # compact ranges, then the staging memfd
    reserve_index = next(
        index for index, line in enumerate(lines)
        if "this->registered_ranges_.reserve(" in line
    )
    assert reserve_index < registrations[0]
    for index in registrations:
        assert any(
            "registered_ranges_.emplace_back" in line
            for line in lines[index:index + 4]
        )
    assert init.count("registered_ranges_.emplace_back") == 2

    destructor = _cpp_section(
        source,
        "Weights_Storage::~Weights_Storage()",
        "void Weights_Storage::InitDistributed",
    )
    # Teardown walks the recorded ranges instead of assuming the whole
    # compact/staging mapping was pinned, and unregisters before munmap.
    assert "this->registered_ranges_.rbegin()" in destructor
    assert "cudaHostUnregister(range->first);" in destructor
    assert "cudaHostUnregister(this->compact_ptr_)" not in destructor
    assert "cudaHostUnregister(this->staging_ptr_)" not in destructor
    assert destructor.index("cudaHostUnregister") < destructor.index("munmap")

    # Partial init: the mappings are unmapped only when they exist, and the
    # recorded range list is whatever InitDistributed got through.
    assert "if (this->compact_ptr_ != nullptr) {" in destructor
    assert "if (this->staging_ptr_ != nullptr) {" in destructor


def test_hierarchical_gdr_get_module_is_fail_closed_outside_the_pinned_slot():
    source = _weights_storage_source()
    get_module = _cpp_section(
        source,
        "Weights_Storage::get_module_weights_storage(std::string module_key)",
        "py::dict Weights_Storage::get_tensor",
    )
    get_tensor = source[source.index("py::dict Weights_Storage::get_tensor"):]

    # A routed-expert copy from a worker that did not pin it would DMA out of
    # unregistered host memory, so refuse before reaching the copy engine.
    assert "if (this->hierarchical_gdr_) {" in get_module
    assert "const int expert = parse_routed_expert(module_key);" in get_module
    assert (
        "if (expert >= 0 &&\n"
        "            !(this->pin_source_ && expert >= this->pin_expert_begin_ &&\n"
        "              expert < this->pin_expert_end_)) {" in get_module
    )
    assert (
        "hierarchical_gdr worker is not the pinned source of" in get_module
    )
    # The host-transport path is untouched.
    assert "hierarchical_gdr rank requested non-local host module" in get_module

    # get_tensor keeps every local module, including the 7/8 of the routed
    # experts this worker never pins: its consumers are one-time pageable
    # Torch copies and decode reads them all.
    assert "module_weights_storage_" in get_tensor
    for guard in (
        "pin_source_",
        "pin_expert_begin_",
        "pin_expert_end_",
        "parse_routed_expert",
        "hierarchical_gdr_",
    ):
        assert guard not in get_tensor, guard


def test_distributed_readiness_log_reports_the_pin_role_and_pinned_bytes():
    init = _init_distributed_section(_weights_storage_source())

    assert '"host_rdma_full"' in init
    assert '(this->pin_source_ ? "gdr_source" : "gdr_replica")' in init
    assert "pin_role={}" in init
    assert "pinned_ranges={}" in init
    assert "pinned_compact={:.3f} GiB" in init
    assert "pin_intervals.size()," in init
    assert (
        "pinned_compact_bytes / (1024.0 * 1024.0 * 1024.0)" in init
    )


class _FakeStoreTensor:
    """A CPU int32 view into the compact store: exact address and pin state."""

    def __init__(self, ptr, nbytes, pinned, *, device="cpu", contiguous=True):
        self._ptr = ptr
        self._nbytes = nbytes
        self._pinned = pinned
        self._contiguous = contiguous
        self.device = SimpleNamespace(type=device)
        self.clones = 0

    def data_ptr(self):
        return self._ptr

    def numel(self):
        return self._nbytes // 4

    def element_size(self):
        return 4

    def is_contiguous(self):
        return self._contiguous

    def is_pinned(self):
        return self._pinned

    def clone(self):
        self.clones += 1
        # A fresh CPU allocation: different address, never registered.
        return _FakeStoreTensor(self._ptr + (1 << 40), self._nbytes, False)


PACKED_BYTES = 1 << 20
SCALE_BYTES = 1 << 15
REGISTER_EDGE = 1 << 31


def _store_pair(
    *,
    packed_pinned,
    scale_pinned,
    gap=0,
    packed_device="cpu",
    packed_contiguous=True,
    scale_contiguous=True,
):
    """``w1.weight_packed`` followed in the store by ``w1.weight_scale``."""
    packed = _FakeStoreTensor(
        REGISTER_EDGE,
        PACKED_BYTES,
        packed_pinned,
        device=packed_device,
        contiguous=packed_contiguous,
    )
    scale = _FakeStoreTensor(
        REGISTER_EDGE + PACKED_BYTES + gap,
        SCALE_BYTES,
        scale_pinned,
        contiguous=scale_contiguous,
    )
    return packed, scale


def _boundary_guard():
    return _isolated_function(
        ROOT / "batchgen" / "moe" / "fused_moe_mxfp4_resident.py",
        "_stage_registration_boundary_packed",
    )


def test_resident_offline_marlin_stages_the_registration_boundary_packed_once():
    stage = _boundary_guard()
    # The measured rank-14 signature: a packed whose START is inside the
    # registered range but whose tail is not, so its immediately adjacent scale
    # already reports pageable.
    packed, scale = _store_pair(packed_pinned=True, scale_pinned=False)

    staged = stage(packed, scale)

    assert staged is not packed
    assert packed.clones == 1
    assert not staged.is_pinned()
    assert staged.numel() * staged.element_size() == PACKED_BYTES
    # Exactly once: the staged copy no longer carries the boundary signature,
    # so re-running the guard on it cannot stage a second copy.
    assert stage(staged, scale) is staged
    assert packed.clones == 1
    assert staged.clones == 0


def test_resident_offline_marlin_leaves_every_other_packed_on_the_direct_path():
    stage = _boundary_guard()
    cases = {
        # Fully inside the registered range (measured: expert 783).
        "fully pinned": _store_pair(packed_pinned=True, scale_pinned=True),
        # Fully outside it (measured: expert 785).
        "pageable": _store_pair(packed_pinned=False, scale_pinned=False),
        # Mixed pin state, but the scale is not this packed's neighbour, so it
        # says nothing about where this packed ends.
        "non-adjacent mixed": _store_pair(
            packed_pinned=True, scale_pinned=False, gap=4096
        ),
        "pageable packed, pinned scale": _store_pair(
            packed_pinned=False, scale_pinned=True
        ),
        "not a host tensor": _store_pair(
            packed_pinned=True, scale_pinned=False, packed_device="cuda"
        ),
        "non-contiguous packed": _store_pair(
            packed_pinned=True, scale_pinned=False, packed_contiguous=False
        ),
        "non-contiguous scale": _store_pair(
            packed_pinned=True, scale_pinned=False, scale_contiguous=False
        ),
    }
    for name, (packed, scale) in cases.items():
        assert stage(packed, scale) is packed, name
        assert packed.clones == 0, name


def test_resident_boundary_guard_is_scoped_to_the_offline_marlin_branch():
    path = ROOT / "batchgen" / "moe" / "fused_moe_mxfp4_resident.py"
    tree = ast.parse(path.read_text())
    repack = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_repack_projection"
    )

    # Only the stored-marlin int32 branch direct-copies the packed weight; the
    # uint8 path repacks on device and must keep its untouched call.
    assert _name_call_guards(repack, "_stage_registration_boundary_packed") == [
        ("packed.dtype == torch.int32",)
    ]
