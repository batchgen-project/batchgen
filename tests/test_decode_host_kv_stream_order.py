"""Steady-state decode host-KV stream-order regressions.

These tests extract the shipping worker methods instead of importing the full
GPU worker. GPU integration remains covered by the paged-KV custom-stream
tests; this file pins the Python hot-path contract around that implementation.
"""

import ast
import os
import textwrap
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "batchgen" / "batchgen_worker.py"
HOST_VIEW = ROOT / "core" / "KV_Storage" / "host_paged_kv_worker_view.h"


def _worker_method(name):
    source = WORKER.read_text()
    tree = ast.parse(source)
    worker = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "BatchGenWorker"
    )
    method = next(
        node
        for node in worker.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    lines = source.splitlines(keepends=True)
    return textwrap.dedent("".join(lines[method.lineno - 1 : method.end_lineno]))


class _Tensor:
    shape = (2, 1, 1)

    def dim(self):
        return 3

    def unsqueeze(self, dim):
        assert dim == 2
        return self


class _Task:
    pass


def test_deferred_host_kv_launch_does_not_touch_cuda_from_python(monkeypatch):
    source = _worker_method("_flush_deferred_kv_to_host")

    class _ForbiddenCuda:
        def __getattr__(self, name):
            raise AssertionError(f"Python host-KV flush touched torch.cuda.{name}")

    namespace = {
        "os": os,
        "torch": SimpleNamespace(cuda=_ForbiddenCuda(), Tensor=object),
        "RuntimeError": RuntimeError,
    }
    exec(compile(source, str(WORKER), "exec"), namespace)

    trace = []
    task = _Task()

    class _View:
        def async_append_decode_kv_to_host_batched_kernel(self, **kwargs):
            trace.append(("launch", kwargs))
            return task

    tensor = _Tensor()
    worker = SimpleNamespace(
        _deferred_kv_entries=[(7, tensor, None)],
        _deferred_kv_entries_aux=[],
        _deferred_kv_worker_view=_View(),
        _deferred_kv_worker_view_aux=None,
        _deferred_kv_batch=([11, 12], [63, 64]),
        _pending_kv_append_tasks=[],
        _pending_kv_append_tensors=[],
        _ensure_host_kv_append_capacity=lambda ids, lengths: trace.append(
            ("capacity", ids, lengths)
        ),
        _wait_pending_kv_append_tasks=lambda **kwargs: trace.append(
            ("throttle", kwargs)
        ),
    )
    monkeypatch.delenv("BATCHGEN_KV_OFFLOAD_UVA_KERNEL", raising=False)

    namespace["_flush_deferred_kv_to_host"](worker)

    assert trace[0] == ("capacity", [11, 12], [63, 64])
    assert trace[1][0] == "launch"
    assert trace[1][1]["entries"] == [(7, tensor, None)]
    assert worker._pending_kv_append_tasks == [task]
    assert worker._pending_kv_append_tensors == [tensor]
    assert worker._deferred_kv_entries == []
    assert worker._deferred_kv_batch is None


def test_cpp_append_orders_copy_stream_after_producer_stream():
    source = HOST_VIEW.read_text()
    start = source.index("    KVAsyncTask AsyncAppendDecodeKVToHostBatchedKernel(")
    end = source.index("\n    // ================================================================\n    // Direct host", start)
    method = source[start:end]

    # The producer position is recorded on the ISSUING thread, before the
    # task is launched; the copy stream waits on that event before the
    # kernel; completion is signalled after it.
    record = method.index("auto producer_event = RecordProducerEvent();")
    launch_task = method.index("return LaunchAsyncTask([this")
    wait = method.index(
        "this->WaitForProducerEvent(cuda_stream, *producer_event);"
    )
    launch = method.index("worker_detail::LaunchUvaPageCopyKernel(")
    completion = method.index("this->SynchronizeWithEvent(cuda_stream);")
    assert record < launch_task < wait < launch < completion


def test_cpp_every_async_copy_records_the_producer_on_the_issuing_thread():
    """Regression for the Kimi-K3 decode collapse (2026-09-06): the copy
    streams are non-blocking pool streams and the model runs on the default
    stream (handle 0). The old WaitForProducerStream skipped the wait for a
    null handle, so nothing ordered a d2h offload behind the kernel that
    filled its source, nor an h2d page load behind the K-cache memset."""
    for header in (
        HOST_VIEW,
        ROOT / "core" / "KV_Storage" / "compressed_state_host_manager.h",
    ):
        source = header.read_text()
        assert "WaitForProducerStream" not in source, header.name
        assert "producer_cuda_stream" not in source, header.name
        # RecordProducerEvent must record unconditionally (no null-handle
        # early return) on the caller's current stream.
        start = source.index("RecordProducerEvent() const {")
        body = source[start:source.index("return event;", start)]
        assert "cudaEventRecord(event->get(), stream)" in body, header.name
        assert "return;" not in body, header.name
        assert "== nullptr" not in body, header.name
        records = source.count("auto producer_event = RecordProducerEvent();")
        waits = source.count("WaitForProducerEvent(cuda_stream, *producer_event);")
        assert records == waits, (header.name, records, waits)
    view = HOST_VIEW.read_text()
    # two h2d loads, the prefill offload, the per-layer append, the batched append
    assert view.count("auto producer_event = RecordProducerEvent();") == 5
    state = (ROOT / "core" / "KV_Storage" / "compressed_state_host_manager.h").read_text()
    assert state.count("auto producer_event = RecordProducerEvent();") == 3


def test_decode_waits_only_for_token_event_after_host_kv_launch():
    source = _worker_method("decoding_continuous")
    tree = ast.parse(source)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]

    token_copy = next(
        node
        for node in calls
        if isinstance(node.func, ast.Attribute)
        and node.func.attr == "copy_"
        and ast.unparse(node.func.value).startswith("_new_tokens_pinned[")
    )
    token_record = next(
        node
        for node in calls
        if ast.unparse(node.func) == "_new_tokens_ready.record"
    )
    kv_launch = next(
        node
        for node in calls
        if ast.unparse(node.func) == "self._flush_deferred_kv_to_host"
    )
    token_wait = next(
        node
        for node in calls
        if ast.unparse(node.func) == "_new_tokens_ready.synchronize"
    )

    assert token_copy.lineno < token_record.lineno < kv_launch.lineno < token_wait.lineno
    hot_path = "\n".join(source.splitlines()[token_copy.lineno - 1 : token_wait.end_lineno])
    assert "torch.cuda.synchronize" not in hot_path
    assert "current_stream(self.torch_device).synchronize" not in hot_path
