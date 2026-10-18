"""Regression: the per-step KV-append drain must not take a full device sync.

The drain used to end in ``torch.cuda.synchronize`` on every decode step. That
covered two different concerns, and only one of them needed it:

* the host-side page-table race between the append threads and this thread's
  register/allocate -- fixed by the thread join (``task.wait()``), which a
  device sync cannot help with;
* a GPU page's D2H read completing before the page is handed to another
  sequence -- already guaranteed per task, because every append task lambda
  ends by recording an event on its copy stream and blocking its own thread on
  it (``SynchronizeWithEvent``). The join therefore implies copy completion,
  not merely producer->copy ordering.

What the full sync did add was a host block until the whole graph replay
retired, once per token. The steady-state step path now passes
``device_sync=False``; every drain that precedes page free/reuse or a
page-table mutation keeps the default.

Two static checks, so they hold without a GPU:

1. the Python side -- the flag exists, gates the sync, and only the step path
   turns it off;
2. the C++ side -- the two append entry points the worker actually calls still
   end their task with the event sync. If that ever changes, the Python-side
   argument above is void, and this test is where it must fail.
"""
import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "batchgen" / "batchgen_worker.py"
WORKER_VIEW = ROOT / "core" / "KV_Storage" / "host_paged_kv_worker_view.h"

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


def test_drain_signature_has_a_device_sync_flag_defaulting_to_true():
    fn = _method("_wait_pending_kv_append_tasks")
    kwonly = {a.arg: d for a, d in zip(fn.args.kwonlyargs, fn.args.kw_defaults)}
    assert "device_sync" in kwonly, "the drain must expose the flag explicitly"
    assert ast.unparse(kwonly["device_sync"]) == "True", (
        "a drain that forgets the flag must keep the old, safe behaviour")


def test_the_device_sync_is_gated_on_the_flag():
    fn = _method("_wait_pending_kv_append_tasks")
    syncs = [n for n in ast.walk(fn)
             if isinstance(n, ast.Call)
             and ast.unparse(n.func) == "torch.cuda.synchronize"]
    assert len(syncs) == 1, ast.unparse(fn)
    guard = next(
        n for n in ast.walk(fn)
        if isinstance(n, ast.If) and any(s is syncs[0] for s in ast.walk(n.test) or [])
        or (isinstance(n, ast.If) and any(
            c is syncs[0] for c in ast.walk(n)) and "device_sync" in ast.unparse(n.test))
    )
    assert ast.unparse(guard.test).startswith("device_sync and"), (
        "the flag must be the first term, so a falsy flag short-circuits: "
        + ast.unparse(guard.test))


def test_only_the_steady_state_step_path_disables_the_device_sync():
    calls = [n for n in ast.walk(_tree())
             if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute)
             and n.func.attr == "_wait_pending_kv_append_tasks"]
    assert calls, "no drain call sites found"
    disabled = [c for c in calls
                if any(k.arg == "device_sync" and ast.unparse(k.value) == "False"
                       for k in c.keywords)]
    assert len(disabled) == 1, (
        "exactly one drain -- the per-step one -- may skip the device sync, "
        f"found {len(disabled)}")
    # Every other call site must stay silent about the flag, i.e. keep the
    # default. A boundary that frees pages needs the whole device quiet.
    for call in calls:
        if call in disabled:
            continue
        assert not any(k.arg == "device_sync" for k in call.keywords), (
            "non-step drains must not pass device_sync at all: "
            + ast.unparse(call))


def test_worker_calls_no_append_entry_point_beyond_the_checked_ones():
    called = {
        n.func.attr
        for n in ast.walk(_tree())
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and n.func.attr.startswith("async_append_decode_kv")
    }
    assert called <= set(APPEND_ENTRY_POINTS), (
        "a new append entry point must be proven to end in SynchronizeWithEvent "
        f"before the step path may skip the device sync: {sorted(called)}")


def test_each_append_task_ends_by_syncing_its_copy_stream():
    """The C++ invariant the Python-side drain relies on."""
    text = WORKER_VIEW.read_text()
    starts = {}
    for py_name, cpp_name in APPEND_ENTRY_POINTS.items():
        match = re.search(rf"^\s*KVAsyncTask {cpp_name}\(", text, re.M)
        assert match, f"{cpp_name} not found in {WORKER_VIEW.name}"
        starts[cpp_name] = match.start()

    # Each method body runs to the next KVAsyncTask signature (or EOF).
    signatures = sorted(
        m.start() for m in re.finditer(r"^\s*KVAsyncTask \w+\(", text, re.M)
    )
    for cpp_name, start in starts.items():
        end = next((s for s in signatures if s > start), len(text))
        body = text[start:end]
        lines = [l.rstrip() for l in body.split("\n") if l.strip()]
        sync_idx = [i for i, l in enumerate(lines)
                    if "this->SynchronizeWithEvent(cuda_stream);" in l]
        assert sync_idx, (
            f"{cpp_name} no longer syncs its copy stream; the per-step drain "
            "must get its device sync back")
        # The sync must be the lambda's LAST statement -- the two lines after
        # it close the lambda and the method. That is what makes
        # `task.wait()` mean "this task's copies have completed" rather than
        # "this task was launched".
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
    record = body.index("cudaEventRecord")
    wait = body.index("cudaEventSynchronize")
    assert record < wait, (
        "the helper must record on the copy stream and then block on it; "
        "without the block, task.wait() stops implying copy completion")
