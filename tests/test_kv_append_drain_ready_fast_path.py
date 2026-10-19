"""The per-step KV-append drain may skip its device sync only when proven idle.

The drain used to end in ``torch.cuda.synchronize`` on every decode step. That
sync covers two different concerns, and only one of them ever needed it:

* the host-side page-table race between the append threads and this thread's
  register/allocate -- fixed by the thread join (``task.wait()``), which a
  device sync cannot help with;
* a GPU page's D2H read completing before the page is handed to another
  sequence -- already guaranteed per task, because every append task lambda
  ends by recording an event on its copy stream and blocking its own thread on
  it (``SynchronizeWithEvent``). A task's future therefore becomes ready only
  after its own copies have completed.

So when every pending future is ALREADY ready, the joins are no-ops and the
sync has nothing left to establish for a caller that frees no page. That is
the fast path. When any task is still in flight the drain behaves exactly as
it does today -- join all (holding the GIL) and then synchronize the device --
which keeps the measured cohort-abort wedge and release point on precisely the
path that wedges today.

Checks here, all static so they hold without a GPU:

1. the readiness probe is non-blocking and is read before the joins;
2. only the steady-state per-step decode drain opts in; every page-free,
   page-table-mutation and teardown caller keeps the sync unconditionally;
3. the C++ invariant the whole argument rests on -- the two append entry
   points the worker calls still end their task with the event sync, and
   ``done()`` is still ``wait_for(0s)``.
"""
import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "batchgen" / "batchgen_worker.py"
WORKER_VIEW = ROOT / "core" / "KV_Storage" / "host_paged_kv_worker_view.h"
BINDING = ROOT / "core" / "batchgen_Binding.cpp"

# The only append entry points the worker calls. Both are checked below.
APPEND_ENTRY_POINTS = {
    "async_append_decode_kv_to_host": "AsyncAppendDecodeKVToHost",
    "async_append_decode_kv_to_host_batched_kernel":
        "AsyncAppendDecodeKVToHostBatchedKernel",
}


def _tree():
    return ast.parse(WORKER.read_text())


def _method(name):
    return next(n for n in ast.walk(_tree())
                if isinstance(n, ast.FunctionDef) and n.name == name)


# ------------------------------------------------------------------ signature

def test_the_fast_path_is_opt_in_and_defaults_to_todays_behaviour():
    fn = _method("_wait_pending_kv_append_tasks")
    kwonly = {a.arg: d for a, d in zip(fn.args.kwonlyargs, fn.args.kw_defaults)}
    assert "ready_fast_path" in kwonly, "the flag must be explicit"
    assert ast.unparse(kwonly["ready_fast_path"]) == "False", (
        "a caller that forgets the flag must keep the device sync")


def test_readiness_is_probed_before_the_joins_and_never_waits():
    fn = _method("_wait_pending_kv_append_tasks")
    probe = [n for n in ast.walk(fn)
             if isinstance(n, ast.Call)
             and ast.unparse(n.func) == "self._pending_kv_appends_done"]
    assert len(probe) == 1, "exactly one readiness probe per drain"
    join = next(n for n in ast.walk(fn)
                if isinstance(n, ast.Call)
                and ast.unparse(n.func) == "task.wait")
    assert probe[0].lineno < join.lineno, (
        "readiness read after the joins would always answer yes")

    # The probe helper itself must stay a status read.
    helper = _method("_pending_kv_appends_done")
    assert [n for n in ast.walk(helper)
            if isinstance(n, ast.Call) and ast.unparse(n.func) == "task.done"]
    assert not [n for n in ast.walk(helper)
                if isinstance(n, ast.Call)
                and ast.unparse(n.func) in ("task.wait", "task.result")]


def test_the_device_sync_is_skipped_only_on_the_proven_ready_path():
    fn = _method("_wait_pending_kv_append_tasks")
    syncs = [n for n in ast.walk(fn)
             if isinstance(n, ast.Call)
             and ast.unparse(n.func) == "torch.cuda.synchronize"]
    assert len(syncs) == 1, ast.unparse(fn)
    guard = next(
        n for n in ast.walk(fn)
        if isinstance(n, ast.If) and any(c is syncs[0] for c in ast.walk(n))
        and "drain_ready" in ast.unparse(n.test)
    )
    test_src = ast.unparse(guard.test)
    # Today's two conditions must survive verbatim, with the readiness term
    # added: a drain with errors, or with no tasks, still behaves as before.
    assert "num_tasks > 0" in test_src
    assert "not wait_errors" in test_src
    assert "not drain_ready" in test_src

    ready = next(n for n in ast.walk(fn) if isinstance(n, ast.Assign)
                 and [ast.unparse(t) for t in n.targets] == ["drain_ready"])
    ready_src = ast.unparse(ready.value)
    # Readiness requires the opt-in AND a non-empty task set AND the probe.
    assert "ready_fast_path" in ready_src
    assert "num_tasks > 0" in ready_src
    assert "self._pending_kv_appends_done()" in ready_src


# ----------------------------------------------------------------- call sites

def test_only_the_steady_state_step_drain_opts_into_the_fast_path():
    calls = [n for n in ast.walk(_tree())
             if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute)
             and n.func.attr == "_wait_pending_kv_append_tasks"]
    assert len(calls) >= 6, f"expected every drain call site, found {len(calls)}"
    opted = [c for c in calls
             if any(k.arg == "ready_fast_path" and ast.unparse(k.value) == "True"
                    for k in c.keywords)]
    assert len(opted) == 1, (
        "exactly one drain -- the per-step one -- may skip the device sync; "
        f"found {len(opted)}")
    # Every other caller must stay silent about the flag, i.e. keep the
    # default. Those are the ones that free or re-assign GPU pages, mutate a
    # page table, or unwind.
    for call in calls:
        if call in opted:
            continue
        assert not any(k.arg == "ready_fast_path" for k in call.keywords), (
            "non-step drains must not pass ready_fast_path at all: "
            + ast.unparse(call))


def test_the_opted_in_call_is_the_one_inside_the_decode_loop():
    fn = _method("decoding_continuous")
    loop = next(n for n in fn.body
                if isinstance(n, ast.While) and ast.unparse(n.test) == "decode_uuids")
    opted = [n for n in ast.walk(loop)
             if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute)
             and n.func.attr == "_wait_pending_kv_append_tasks"
             and any(k.arg == "ready_fast_path" for k in n.keywords)]
    assert len(opted) == 1, "the fast path belongs to the per-step drain"
    # The bounded teardown quiesce runs the same drain and must NOT opt in: it
    # has already proven readiness its own way and still needs the device quiet.
    quiesce = _method("_quiesce_decode_for_cohort_shutdown")
    for call in ast.walk(quiesce):
        if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                and call.func.attr == "_wait_pending_kv_append_tasks"):
            assert not any(k.arg == "ready_fast_path" for k in call.keywords), (
                "the teardown quiesce must keep the device sync")


def test_the_two_drain_paths_are_counted_for_the_heartbeat():
    # Whether the fast path fires is a deployment property; it must be
    # observable rather than assumed, or a node run cannot attribute a result.
    fn = _method("_wait_pending_kv_append_tasks")
    body = ast.unparse(fn)
    assert "self._kv_drain_ready_fast = getattr(self, '_kv_drain_ready_fast', 0) + 1" in body
    assert "self._kv_drain_ready_slow = getattr(self, '_kv_drain_ready_slow', 0) + 1" in body
    loop_fn = ast.unparse(_method("decoding_continuous"))
    assert "kv_drain_ready=" in loop_fn, "the counts must reach the heartbeat"
    assert "self._kv_drain_ready_fast = 0" in loop_fn, "and be reset there"


# -------------------------------------------------------------- the C++ facts

def test_each_append_task_ends_by_syncing_its_copy_stream():
    """The invariant that makes "future ready" mean "copies completed"."""
    text = WORKER_VIEW.read_text()
    signatures = sorted(
        m.start() for m in re.finditer(r"^\s*KVAsyncTask \w+\(", text, re.M)
    )
    for cpp_name in APPEND_ENTRY_POINTS.values():
        match = re.search(rf"^\s*KVAsyncTask {cpp_name}\(", text, re.M)
        assert match, f"{cpp_name} not found in {WORKER_VIEW.name}"
        start = match.start()
        end = next((s for s in signatures if s > start), len(text))
        lines = [l.rstrip() for l in text[start:end].split("\n") if l.strip()]
        sync_idx = [i for i, l in enumerate(lines)
                    if "this->SynchronizeWithEvent(cuda_stream);" in l]
        assert sync_idx, (
            f"{cpp_name} no longer syncs its copy stream; a ready future stops "
            "implying its copies completed and the fast path must go")
        # The sync must be the lambda's LAST statement -- the two lines after
        # it close the lambda and the method.
        tail = [l.strip() for l in lines[sync_idx[-1] + 1:sync_idx[-1] + 3]]
        assert tail == ["});", "}"], (
            f"{cpp_name} does work after SynchronizeWithEvent: {tail}")


def test_the_sync_helper_records_then_blocks():
    text = WORKER_VIEW.read_text()
    match = re.search(
        r"void SynchronizeWithEvent\(cudaStream_t stream\) const \{(.*?)\n    \}",
        text,
        re.S,
    )
    assert match, "SynchronizeWithEvent helper not found"
    body = match.group(1)
    assert body.index("cudaEventRecord") < body.index("cudaEventSynchronize"), (
        "the helper must record on the copy stream and then block on it")


def test_the_readiness_probe_is_a_zero_timeout_future_poll():
    text = WORKER_VIEW.read_text()
    match = re.search(r"bool done\(\) const \{(.*?)\n    \}", text, re.S)
    assert match, "KVAsyncTask::done not found"
    body = match.group(1)
    assert "wait_for(std::chrono::seconds(0))" in body, (
        "done() must stay non-blocking: " + body)
    assert "std::future_status::ready" in body
    # and it must be exposed to Python
    assert '.def("done", &kv::KVAsyncTask::done)' in BINDING.read_text()


def test_the_blocking_join_is_still_the_gil_holding_one():
    """Why the fallback must stay exactly where it is.

    ``KVAsyncTask::wait`` is bound without a GIL release, so a blocking join
    freezes the abort watchdog. The fast path is only ever taken when the
    futures are already ready, i.e. when that join cannot block. If this
    binding ever releases the GIL, the fallback's cost changes and the
    ready-poll design should be revisited rather than silently kept.
    """
    binding = BINDING.read_text()
    match = re.search(r'\.def\("wait", &kv::KVAsyncTask::wait\)([^;]*)', binding)
    assert match, "KVAsyncTask::wait binding not found"
    assert "gil_scoped_release" not in match.group(1)


def test_worker_calls_no_append_entry_point_beyond_the_checked_ones():
    called = {
        n.func.attr
        for n in ast.walk(_tree())
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and n.func.attr.startswith("async_append_decode_kv")
    }
    assert called <= set(APPEND_ENTRY_POINTS), (
        "a new append entry point must be proven to end in SynchronizeWithEvent "
        f"before the fast path may skip the device sync: {sorted(called)}")
