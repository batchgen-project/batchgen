import ast
import copy
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


def _isolated_method(path, class_name, function_name):
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
    namespace = {}
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


def test_prefill_mode_accepts_only_streamed_or_resident_ep():
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

    manager.set_prefill_moe_mode("resident_ep")
    assert manager.prefill_uses_resident_ep()
    manager.set_prefill_moe_mode(None)
    assert not manager.prefill_uses_resident_ep()
    with pytest.raises(ValueError, match="streamed.*resident_ep"):
        manager.set_prefill_moe_mode("unknown")


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
    moe = type("MoE", (), {
        "_resident_ep_moe": None,
        "shared_experts": shared,
    })()
    layer = type("Layer", (), {"mlp": dense, "block_sparse_moe": moe})()
    manager.model = type("Model", (), {
        "model": type("Inner", (), {"layers": [layer]})(),
        "modules": lambda self: [norm],
    })()
    method = _isolated_method(
        path,
        "KimiLinearParallelStrategyManager",
        "_set_resident_ep_prefill_enabled",
    ).__get__(manager)

    method(True)
    assert dense._resident_prefill_token_tile == 512
    assert shared._resident_prefill_token_tile == 512
    assert norm._resident_prefill_token_tile == 512
    assert moe._resident_ep_prefill_enabled is True

    method(False)
    assert dense._resident_prefill_token_tile is None
    assert shared._resident_prefill_token_tile is None
    assert norm._resident_prefill_token_tile is None
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
