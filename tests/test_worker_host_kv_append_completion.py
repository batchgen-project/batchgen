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
drains them at the end of the step, after the one-step-lookahead bookkeeping,
so the host work overlaps the forward and the drain's own wait is the only
thing charged with the forward's tail. The drain must still run every step
that launched appends, before anything that can mutate a page table.

It keeps the per-task thread join and no longer takes a full device sync:
each append task ends by recording an event on its copy stream and blocking
its own thread on it, so the join already implies the copies completed. The
drains that precede page free/reuse keep the device sync.

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


def _is_apply_pending(node):
    return (isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_apply_pending_decode_tokens")


def _decode_loop_body():
    fn = _method("decoding_continuous")
    loop = next(n for n in fn.body
                if isinstance(n, ast.While) and ast.unparse(n.test) == "decode_uuids")
    return fn, loop.body


def test_decode_step_drains_appends_last_without_a_count_threshold():
    _, body = _decode_loop_body()

    flush = _single(body, lambda s: _stmt_calls(s, "self._flush_deferred_kv_to_host"),
                    "host-KV append launch")
    # R4: the host bookkeeps the PREVIOUS step here, between the append launch
    # and the drain, so that work overlaps the forward already queued.
    apply_pending = _single(
        body,
        lambda s: isinstance(s, ast.Expr) and _is_apply_pending(s.value),
        "pending-step apply",
    )
    # The per-step drain is an `if` whose own body is the wait. The boundary
    # block's watermark wait is nested deeper and is a different drain.
    drain = _single(
        body,
        lambda s: isinstance(s, ast.If) and any(
            isinstance(b, ast.Expr) and _is_wait_call(b.value) for b in s.body),
        "per-step append drain",
    )
    # The drain must be LAST of the three: it is the first call that waits on
    # this step's device work, so anything host-only placed after it would be
    # paying for the forward tail twice.
    assert flush < apply_pending < drain

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
    # R5: the per-step drain keeps the thread join and drops the full device
    # sync, which each task's own SynchronizeWithEvent already makes redundant
    # for its copies.
    assert sorted(
        (k.arg, ast.unparse(k.value)) for k in wait.keywords
    ) == [("defer_errors", "True"), ("device_sync", "False")]

    # Nothing between the launch and the drain can skip the drain.
    between = body[flush + 1:drain]
    for node in (n for s in between for n in ast.walk(s)):
        assert not isinstance(node, (ast.Break, ast.Continue, ast.Return)), (
            ast.dump(node))


def test_readback_and_bookkeeping_live_in_the_pending_apply():
    # The wait for the sampled tokens and the per-sequence updates moved out of
    # the loop body into one method, so a step can be bookkept from the next
    # step, from a boundary flush and from the loop exit through one path.
    fn = _method("_apply_pending_decode_tokens")
    syncs = [n for n in ast.walk(fn)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == "synchronize"]
    assert len(syncs) == 1, ast.unparse(fn)
    assert ast.unparse(syncs[0]) == "event_ring[pending.slot].synchronize()"

    # It must take the pending step before reading the ring, so a lost flush is
    # a named error instead of a second read of the same slot.
    take = [n for n in ast.walk(fn)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr == "take"]
    assert len(take) == 1
    # and it must be a no-op when nothing is in flight
    guard = fn.body[1] if isinstance(fn.body[0], ast.Expr) else fn.body[0]
    assert isinstance(guard, ast.If)
    assert ast.unparse(guard.test) == "not lookahead.is_armed"
    assert all(isinstance(s, ast.Return) for s in guard.body)


def test_every_composition_change_flushes_the_pipeline_first():
    # A boundary rebinds the ordered row list and decides retirement from host
    # sequence state; the loop exit must not drop a step's tokens.
    fn, body = _decode_loop_body()
    boundary = _single(
        body,
        lambda s: isinstance(s, ast.If)
        and "force_host_boundary" in ast.unparse(s.test)
        and any(_is_apply_pending(n) for n in ast.walk(s)),
        "page-boundary block with a flush",
    )
    # the flush is the FIRST statement of the boundary block
    first = body[boundary].body[0]
    assert isinstance(first, ast.Expr) and _is_apply_pending(first.value), \
        ast.unparse(first)

    # loop exit: a flush before the final cross-rank drain
    after_loop = fn.body[fn.body.index(
        next(n for n in fn.body if isinstance(n, ast.While)))+1:]
    exit_apply = [i for i, s in enumerate(after_loop)
                  if isinstance(s, ast.Expr) and _is_apply_pending(s.value)]
    exit_drain = [i for i, s in enumerate(after_loop)
                  if isinstance(s, ast.Expr) and _is_wait_call(s.value)]
    assert exit_apply and exit_drain and exit_apply[0] < exit_drain[0]
