"""Regression: every decode step must complete the host-KV appends it launched.

`async_append_decode_kv_to_host` returns a KVAsyncTask backed by a std::async
thread that reads the worker view's page table on its own thread, unlocked.
The deferred host-KV flush used to let up to 256 of those stay in flight
across decode steps, where they overlapped the next admission wave's
register/allocate on the main thread. A torn read lands one token's KV in a
page that now belongs to a freshly prefilled sequence; on the 512x4096 K3
contract every collapsed sequence was in a wave after the first and died at
its 2nd or ~28th token, while the first wave (no page-table mutation in
flight) was clean.

`_flush_deferred_kv_to_host` now only launches the appends; the decode loop
drains them right after the sampled-token readback so the wait is timed in a
step-split slot of its own. The drain must still run every step that
launched appends, before anything that can mutate a page table.

batchgen_worker imports the whole engine, so this is a static check on the
method bodies: the wait must be guarded by the pending list being non-empty,
never by a count threshold.
"""
import ast
from pathlib import Path

WORKER = Path(__file__).resolve().parents[1] / "batchgen" / "batchgen_worker.py"


def _method(name):
    tree = ast.parse(WORKER.read_text())
    return next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == name)


def _is_wait_call(node):
    return (isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_wait_pending_kv_append_tasks")


def _stmt_calls(stmt, func_src):
    value = stmt.value if isinstance(stmt, (ast.Expr, ast.Assign)) else None
    return isinstance(value, ast.Call) and ast.unparse(value.func) == func_src


def _single(body, predicate, what):
    hits = [i for i, stmt in enumerate(body) if predicate(stmt)]
    assert len(hits) == 1, f"expected exactly one {what}, got {hits}"
    return hits[0]


def test_flush_only_launches_appends():
    # A wait inside the flush would put the device-sync drain back into the
    # launch slot of the decode step split.
    fn = _method("_flush_deferred_kv_to_host")
    assert not [n for n in ast.walk(fn) if _is_wait_call(n)]


def test_decode_step_drains_appends_after_readback_without_a_count_threshold():
    fn = _method("decoding_continuous")
    loop = next(n for n in fn.body
                if isinstance(n, ast.While) and ast.unparse(n.test) == "decode_uuids")
    body = loop.body

    flush = _single(body, lambda s: _stmt_calls(s, "self._flush_deferred_kv_to_host"),
                    "host-KV append launch")
    # The readback is now a bounded, pollable wait on the same event instead of
    # a blocking `_new_tokens_ready.synchronize()`: a rank whose peer died
    # mid-decode blocks in that sync below Python and never reaches the drain
    # below at all. `_wait_decode_tokens` still performs exactly that sync when
    # no cohort teardown probe is armed, and still returns only once the event
    # has fired on every path that reports COMPLETED, so the drain's ordering
    # requirement is unchanged.
    readback = _single(body, lambda s: _stmt_calls(s, "self._wait_decode_tokens"),
                       "sampled-token readback")
    # The per-step drain is an `if` whose own body is the wait. The boundary
    # block's watermark wait is nested deeper and is a different drain.
    drain = _single(
        body,
        lambda s: isinstance(s, ast.If) and any(
            isinstance(b, ast.Expr) and _is_wait_call(b.value) for b in s.body),
        "per-step append drain",
    )
    bookkeeping = _single(
        body,
        lambda s: isinstance(s, ast.For)
        and ast.unparse(s.iter) == "enumerate(zip(batch, batch_sequences))",
        "per-sequence bookkeeping loop",
    )
    assert flush < readback < drain < bookkeeping

    stmt = body[drain]
    # `if self._pending_kv_append_tasks:` -- a bare truthiness guard
    assert isinstance(stmt.test, ast.Attribute) and \
        stmt.test.attr == "_pending_kv_append_tasks", (
            "the wait must not be gated on a task-count threshold: "
            + ast.dump(stmt.test))
    assert not stmt.orelse
    (wait_stmt,) = stmt.body
    wait = wait_stmt.value
    assert _is_wait_call(wait)
    # The drain stays a purely local wait, and it is the one caller allowed to
    # skip the device sync when its appends are already complete. Anything
    # else appearing here changes what the step waits for.
    assert sorted(
        (k.arg, ast.unparse(k.value)) for k in wait.keywords
    ) == [("defer_errors", "True"), ("ready_fast_path", "True")]

    # Nothing between the launch and the drain can skip the drain -- with one
    # exception, and only because it drains by another route. When the token
    # wait above reports ABANDONED the step's device work is parked behind a
    # collective whose peer is dead, so this unbounded drain (a task join
    # ending in a device sync) would never return: it IS the wedge, one level
    # down. That exit returns, and runs the same `defer_errors=True` drain
    # inside `_quiesce_decode_for_cohort_shutdown` under a deadline, which
    # self-exits rather than unwind with appends still in flight. So the
    # invariant this test defends -- never leave the step with unlocked
    # page-table readers running while teardown releases the mapping -- is
    # still enforced on every path out of here. The ready fast path cannot
    # weaken that: in exactly that state the appends are parked behind the
    # dead device work, so their futures are not ready, the probe says so, and
    # the drain is the same unbounded join-plus-device-sync it is today.
    between = body[flush + 1:drain]
    for stmt in between:
        exits = [n for n in ast.walk(stmt)
                 if isinstance(n, (ast.Break, ast.Continue, ast.Return))]
        if not exits:
            continue
        assert len(exits) == 1 and isinstance(exits[0], ast.Return), ast.dump(stmt)
        quiesces = [n for n in ast.walk(stmt)
                    if isinstance(n, ast.Call)
                    and ast.unparse(n.func)
                    == "self._quiesce_decode_for_cohort_shutdown"]
        assert len(quiesces) == 1, (
            "an exit between the append launch and the per-step drain must "
            "drain through the bounded quiesce instead: " + ast.unparse(stmt))
