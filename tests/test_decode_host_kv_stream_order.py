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

    wait = method.index(
        "this->WaitForProducerStream(cuda_stream, producer_cuda_stream);"
    )
    launch = method.index("worker_detail::LaunchUvaPageCopyKernel(")
    completion = method.index("this->SynchronizeWithEvent(cuda_stream);")
    assert wait < launch < completion


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
