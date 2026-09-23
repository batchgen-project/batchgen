"""Pool mode releases every cross-wave hit, so no wave materializes one.

The existing path builds a temporary GPU KV manager for a lookup hit; with
the prefix pool on, a wave that the pool declines must reach it hit-free.
"""

import ast
from pathlib import Path
from types import SimpleNamespace

from batchgen.prefix_reuse.prefill import PrefixCacheSequenceState

WORKER = Path(__file__).resolve().parents[2] / "batchgen" / "batchgen_worker.py"


def _method(name):
    tree = ast.parse(WORKER.read_text())
    return next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _calls(node, name):
    return [
        call for call in ast.walk(node)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == name
    ]


def test_hits_are_released_and_prompts_start_at_their_chain_end():
    namespace = {}
    module = ast.Module(body=[_method("_replace_prefix_lookup_hits")], type_ignores=[])
    exec(compile(module, str(WORKER), "exec"), namespace)
    released = []
    worker = SimpleNamespace(
        prefix_cache_coordinator=SimpleNamespace(release_attachment=released.append),
        _prefix_sequence_states={},
    )
    seqs = [SimpleNamespace(global_idx=3), SimpleNamespace(global_idx=5)]
    states = {
        3: PrefixCacheSequenceState(
            lookup_result=SimpleNamespace(attachment_handle=7),
            attached_tokens=128,
            compute_cached_tokens=128,
        ),
        5: PrefixCacheSequenceState(
            lookup_result=None, attached_tokens=0, compute_cached_tokens=0
        ),
    }
    namespace["_replace_prefix_lookup_hits"](worker, seqs, states, [0, 64])
    assert released == [7]
    assert states[3].lookup_result is None
    assert states[3].compute_cached_tokens == 0 and states[3].shared_page_ids == ()
    assert states[5].attached_tokens == 64 and states[5].compute_cached_tokens == 64
    assert worker._prefix_sequence_states == states


def test_every_declined_wave_releases_its_hits_first():
    plan = _method("_plan_prefill_pool_wave")
    declines = 0
    for block in ast.walk(plan):
        body = getattr(block, "body", None)
        if not isinstance(body, list):
            continue
        for index, stmt in enumerate(body):
            if isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Constant) \
                    and stmt.value.value is None:
                declines += 1
                assert any(
                    _calls(prior, "_replace_prefix_lookup_hits") for prior in body[:index]
                ), f"return None at line {stmt.lineno} keeps its lookup hits"
    assert declines == 2
    # Re-entries after eviction are planned, not sent to the existing path.
    assert not any(
        isinstance(node, ast.Attribute) and node.attr == "total_decoded_before_eviction"
        for node in ast.walk(plan)
    )
