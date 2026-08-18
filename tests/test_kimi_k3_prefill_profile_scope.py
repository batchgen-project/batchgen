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
