import ast
from pathlib import Path


def test_k3_prefill_profile_locals_are_defined_in_prepacked_scope():
    worker_path = (
        Path(__file__).resolve().parents[1] / "batchgen" / "batchgen_worker.py"
    )
    tree = ast.parse(worker_path.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        for child in node.body
        if isinstance(child, ast.FunctionDef)
        and child.name == "prefill_prepacked"
        for node in [child]
    )

    for name in ("_k3_profile_enabled", "_k3_profile_logits"):
        stores = [
            node.lineno
            for node in ast.walk(function)
            if isinstance(node, ast.Name)
            and node.id == name
            and isinstance(node.ctx, ast.Store)
        ]
        loads = [
            node.lineno
            for node in ast.walk(function)
            if isinstance(node, ast.Name)
            and node.id == name
            and isinstance(node.ctx, ast.Load)
        ]
        assert stores, f"{name} has no local initialization"
        assert loads, f"{name} is never consumed"
        assert min(stores) < min(loads), (
            f"{name} is first read at line {min(loads)} before its first "
            f"assignment at line {min(stores)}"
        )


def test_k3_prefill_profile_uses_sequence_debug_in_both_prefill_stages():
    worker_path = (
        Path(__file__).resolve().parents[1] / "batchgen" / "batchgen_worker.py"
    )
    tree = ast.parse(worker_path.read_text())
    worker = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "BatchGenWorker"
    )
    functions = {
        node.name: node
        for node in worker.body
        if isinstance(node, ast.FunctionDef)
    }

    for function_name in ("_config_prefill_for_batch", "prefill_prepacked"):
        calls = [
            node
            for node in ast.walk(functions[function_name])
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_active_batchgen_debug_for_sequences"
        ]
        assert calls, (
            f"{function_name} must resolve batchgen_debug from the active "
            "SequenceEntry objects used by persistent pool admissions"
        )
