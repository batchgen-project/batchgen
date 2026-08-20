import ast
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


def test_prefill_mode_accepts_only_streamed_or_resident_ep():
    from batchgen.models.moonshotai.kimi_linear.Parallel_Strategy_Manager import (
        KimiLinearParallelStrategyManager,
    )

    manager = object.__new__(KimiLinearParallelStrategyManager)
    manager._is_k3 = True
    manager._prefill_moe_mode = "streamed"

    manager.set_prefill_moe_mode("resident_ep")
    assert manager.prefill_uses_resident_ep()
    manager.set_prefill_moe_mode(None)
    assert not manager.prefill_uses_resident_ep()
    with pytest.raises(ValueError, match="streamed.*resident_ep"):
        manager.set_prefill_moe_mode("unknown")


def test_resident_prefill_is_refused_for_non_k3():
    from batchgen.models.moonshotai.kimi_linear.Parallel_Strategy_Manager import (
        KimiLinearParallelStrategyManager,
    )

    manager = object.__new__(KimiLinearParallelStrategyManager)
    manager._is_k3 = False
    manager._prefill_moe_mode = "streamed"
    with pytest.raises(ValueError, match="only for Kimi-K3"):
        manager.set_prefill_moe_mode("resident_ep")


def test_worker_reads_batch_level_mode_before_configure_prefill():
    worker_path = ROOT / "batchgen" / "batchgen_worker.py"
    function = _function(
        worker_path, "BatchGenWorker", "_config_prefill_for_batch"
    )
    calls = [
        (node.lineno, getattr(node.func, "attr", None))
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
