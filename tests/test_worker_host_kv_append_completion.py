"""Regression: the deferred host-KV flush must complete its append tasks.

`async_append_decode_kv_to_host` returns a KVAsyncTask backed by a std::async
thread that reads the worker view's page table on its own thread, unlocked.
`_flush_deferred_kv_to_host` used to let up to 256 of those stay in flight
across decode steps, where they overlapped the next admission wave's
register/allocate on the main thread. A torn read lands one token's KV in a
page that now belongs to a freshly prefilled sequence; on the 512x4096 K3
contract every collapsed sequence was in a wave after the first and died at
its 2nd or ~28th token, while the first wave (no page-table mutation in
flight) was clean.

batchgen_worker imports the whole engine, so this is a static check on the
method body: the wait must be guarded by the pending list being non-empty,
never by a count threshold.
"""
import ast
from pathlib import Path

WORKER = Path(__file__).resolve().parents[1] / "batchgen" / "batchgen_worker.py"


def _method(name):
    tree = ast.parse(WORKER.read_text())
    return next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == name)


def test_flush_waits_pending_appends_without_a_count_threshold():
    fn = _method("_flush_deferred_kv_to_host")
    waits = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.If):
            continue
        calls = [c for c in ast.walk(node)
                 if isinstance(c, ast.Call)
                 and isinstance(c.func, ast.Attribute)
                 and c.func.attr == "_wait_pending_kv_append_tasks"]
        if calls:
            waits.append(node.test)
    assert waits, "flush must wait on _pending_kv_append_tasks"
    for test in waits:
        # `if self._pending_kv_append_tasks:` -- a bare truthiness guard
        assert isinstance(test, ast.Attribute) and \
            test.attr == "_pending_kv_append_tasks", (
                "the wait must not be gated on a task-count threshold: "
                + ast.dump(test))
