import ast
import copy
import json
from pathlib import Path

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
    manager = type("Manager", (), {})()
    manager._stream_all_modules = False
    manager._attn_tp_size = 8
    manager._attn_tp_rank = 3
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
    build = _isolated_method(
        path,
        "KimiLinearParallelStrategyManager",
        "_build_weight_copy_task",
    ).__get__(manager)

    tasks = build()["routed_expert"]

    assert len(tasks) == 112
    assert tasks[0] == "routed_expert_0_336"
    assert tasks[-1] == "routed_expert_0_447"


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


def test_streamed_sp8_empty_rank_loads_before_returning():
    path = ROOT / "batchgen" / "moe" / "streamed_sp8_mxfp4.py"
    function = _function(
        path, "StreamedSP8MXFP4MoELayer", "forward"
    )
    source = ast.unparse(function)
    assert source.index("shard = self.buffer.load") < source.index(
        "if T == 0"
    )


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
