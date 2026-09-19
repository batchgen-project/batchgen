import re
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]


def _source(path: str) -> str:
    return (_REPO_ROOT / path).read_text()


def test_async_release_keeps_slots_pending_until_consumer_event_completes():
    source = _source("core/GPU_Weight_Buffer/GPU_Weight_Buffer.cpp")
    release = source[
        source.index("void GPU_Weight_Buffer::releaseBuffersAsync(") :
        source.index("module_weight_tensor_map GPU_Weight_Buffer::get_weights(")
    ]
    reclaim = source[
        source.index("void GPU_Weight_Buffer::reclaimCompletedReleasesLocked()") :
        source.index("void GPU_Weight_Buffer::reset_weight_stream_profile")
    ]
    acquire = source[
        source.index("GPU_Weight_Buffer::acquireEmptyBuffer(") :
        source.index("void GPU_Weight_Buffer::releaseBuffer(")
    ]

    record = release.index("cudaEventRecord(completion_event, consumer_stream)")
    pending = release.index("this->buffer_status_[module_type][buffer_idx] = 2")
    enqueue = release.index("this->pending_releases_.push_back(")
    assert record < pending < enqueue

    query = reclaim.index("cudaEventQuery(it->event)")
    not_ready = reclaim.index("status == cudaErrorNotReady")
    reusable = reclaim.index("status_it->second[buffer_idx] = 0")
    destroy = reclaim.index("cudaEventDestroy(it->event)")
    assert query < not_ready < reusable < destroy
    assert "this->reclaimCompletedReleasesLocked();" in acquire


def test_async_release_binding_records_the_current_cuda_stream():
    source = _source("core/batchgen.cpp")
    method = source[
        source.index("void BatchGen::free_weights_buffers_async(") :
        source.index("torch::Tensor BatchGen::attn(")
    ]
    binding = _source("core/batchgen_Binding.cpp")

    assert "at::cuda::getCurrentCUDAStream(" in method
    assert "releaseBuffersAsync(\n        module_names, consumer_stream)" in method
    assert '.def("free_weights_buffer_async"' in binding
    assert '.def("free_weights_buffers_async"' in binding


def test_glm5_weight_buffer_calls_exist_in_core_header_and_bindings():
    # Keep GLM-5.2's Python wrapper and the JIT-built core extension in sync.
    # A stale extension once hid a missing get_weights_pinned source dependency,
    # then the first non-persistent attention release exposed the absent
    # free_weights_buffer_async binding.
    wrappers = _source("batchgen/models/glm/glm5/wrappers.py")
    header = _source("core/batchgen.h")
    implementation = _source("core/batchgen.cpp")
    binding = _source("core/batchgen_Binding.cpp")

    called = set(
        re.findall(
            r"self\.core_engine\."
            r"(get_weights_pinned|free_weights_buffers?_async)\(",
            wrappers,
        )
    )
    assert called, "no async weight-release core call found in GLM-5.2 wrappers"

    for name in sorted(called):
        return_type = (
            "std::unordered_map<std::string, torch::Tensor>"
            if name == "get_weights_pinned"
            else "void"
        )
        assert f"{return_type} {name}(" in header, (
            f"wrappers.py calls core_engine.{name} but core/batchgen.h "
            f"does not declare it"
        )
        assert f"BatchGen::{name}(" in implementation, (
            f"wrappers.py calls core_engine.{name} but core/batchgen.cpp "
            f"does not define it"
        )
        assert f'.def("{name}"' in binding, (
            f"wrappers.py calls core_engine.{name} but "
            f"core/batchgen_Binding.cpp does not pybind-expose it"
        )


def test_reset_waits_for_pending_consumers_before_replacing_storage():
    source = _source("core/GPU_Weight_Buffer/GPU_Weight_Buffer.cpp")
    synchronize = source[
        source.index("void GPU_Weight_Buffer::synchronizePendingReleasesLocked()") :
        source.index("void GPU_Weight_Buffer::reclaimCompletedReleasesLocked()")
    ]
    prefill_start = source.rindex("void GPU_Weight_Buffer::reset_prefill_buffer()")
    decoding_start = source.rindex("void GPU_Weight_Buffer::reset_decoding_buffer()")
    prefill = source[prefill_start:decoding_start]
    decoding = source[decoding_start:]

    assert "cudaEventSynchronize(pending.event)" in synchronize
    assert prefill.index("synchronizePendingReleasesLocked") < prefill.index(
        'this->buffers_["routed_expert"].clear()'
    )
    assert decoding.index("synchronizePendingReleasesLocked") < decoding.index(
        'this->buffers_["routed_expert"].clear()'
    )
