import ast
from pathlib import Path


def test_k3_prefill_profile_locals_are_defined_in_prepacked_scope():
    # S5b: prefill_prepacked's body moved verbatim to PrefillEngine.
    engine_path = (
        Path(__file__).resolve().parents[1]
        / "batchgen" / "worker" / "prefill_engine.py"
    )
    tree = ast.parse(engine_path.read_text())
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
    # S5b: prefill_prepacked's body moved verbatim to PrefillEngine.
    engine_tree = ast.parse(
        (worker_path.parent / "worker" / "prefill_engine.py").read_text()
    )
    engine = next(
        node
        for node in engine_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "PrefillEngine"
    )
    # S5b/R2: these bodies moved verbatim to PrefillEngine.
    for moved in ("prefill_prepacked", "_config_prefill_for_batch"):
        functions[moved] = next(
            node
            for node in engine.body
            if isinstance(node, ast.FunctionDef) and node.name == moved
        )

    # The invariant is that both prefill stages see batchgen_debug resolved
    # from the admitted SequenceEntry objects, which is the only place a
    # persistent pool admission carries it. prefill_prepacked resolves it
    # itself. _config_prefill_for_batch no longer calls the resolver inline:
    # the phase key owns the resolution and publishes it, and the config stage
    # reads that published view. Pin the chain, not one call site, so the pin
    # keeps guarding the invariant wherever the resolution lives.
    prepacked_calls = [
        node
        for node in ast.walk(functions["prefill_prepacked"])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_active_batchgen_debug_for_sequences"
    ]
    assert prepacked_calls, (
        "prefill_prepacked must resolve batchgen_debug from the active "
        "SequenceEntry objects used by persistent pool admissions"
    )

    key_calls = [
        node
        for node in ast.walk(functions["_desired_phase_key"])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_active_batchgen_debug_for_sequences"
    ]
    assert key_calls, (
        "_desired_phase_key must resolve batchgen_debug from the active "
        "SequenceEntry objects before it publishes the phase debug view"
    )

    published = [
        node
        for node in ast.walk(functions["_desired_phase_key"])
        if isinstance(node, ast.Attribute)
        and node.attr == "_phase_prefill_debug"
        and isinstance(node.ctx, ast.Store)
    ]
    assert published, (
        "_desired_phase_key must publish the resolved view as "
        "_phase_prefill_debug"
    )

    consumed = [
        node
        for node in ast.walk(functions["_config_prefill_for_batch"])
        if isinstance(node, ast.Attribute)
        and node.attr == "_phase_prefill_debug"
        and isinstance(node.ctx, ast.Load)
    ]
    assert consumed, (
        "_config_prefill_for_batch must read the published "
        "_phase_prefill_debug instead of a batch-independent debug dict"
    )
