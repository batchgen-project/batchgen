"""Static contract: every async host-KV copy is ordered behind the caller's
stream by an event recorded on the ISSUING thread.

Regression for the Kimi-K3 decode collapse (2026-09-06): the copy streams are
non-blocking pool streams and the model runs on the default stream (handle
0). The old WaitForProducerStream skipped the wait for a null handle, so
nothing ordered a d2h offload behind the kernel that filled its source, nor
an h2d page load behind the K-cache memset queued ahead of it. The GPU test
lives in tests/integration/paged_kv/test_host_kv_copy_stream_ordering.py.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOST_VIEW = ROOT / "core" / "KV_Storage" / "host_paged_kv_worker_view.h"
STATE_MANAGER = ROOT / "core" / "KV_Storage" / "compressed_state_host_manager.h"


def test_cpp_every_async_copy_records_the_producer_on_the_issuing_thread():
    for header in (HOST_VIEW, STATE_MANAGER):
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
    # two h2d loads, the prefill offload, the per-layer append, the batched append
    assert HOST_VIEW.read_text().count("auto producer_event = RecordProducerEvent();") == 5
    assert STATE_MANAGER.read_text().count("auto producer_event = RecordProducerEvent();") == 3


def test_cpp_batched_append_records_before_launch_and_waits_before_the_kernel():
    source = HOST_VIEW.read_text()
    start = source.index("    KVAsyncTask AsyncAppendDecodeKVToHostBatchedKernel(")
    method = source[start:start + 20000]
    record = method.index("auto producer_event = RecordProducerEvent();")
    launch_task = method.index("return LaunchAsyncTask([this")
    wait = method.index("this->WaitForProducerEvent(cuda_stream, *producer_event);")
    launch = method.index("worker_detail::LaunchUvaPageCopyKernel(")
    completion = method.index("this->SynchronizeWithEvent(cuda_stream);")
    assert record < launch_task < wait < launch < completion
