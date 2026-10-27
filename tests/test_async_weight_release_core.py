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
    # LEASED -> PENDING_RELEASE in the slot table, all or nothing; a refused
    # release destroys the event it no longer needs.
    pending = release.index("this->slots_.release_async(module_names)")
    refused = release.index("cudaEventDestroy(completion_event);\n        throw;")
    enqueue = release.index("this->pending_releases_.push_back(")
    assert record < pending < refused < enqueue

    query = reclaim.index("cudaEventQuery(it->event)")
    not_ready = reclaim.index("status == cudaErrorNotReady")
    # PENDING_RELEASE -> FREE only once the consumer's event completed.
    reusable = reclaim.index("this->slots_.reclaim(module_type, buffer_idx)")
    destroy = reclaim.index("cudaEventDestroy(it->event)")
    assert query < not_ready < reusable < destroy
    assert "this->reclaimCompletedReleasesLocked();" in acquire
    assert "this->slots_.reserve(module_type)" in acquire


def test_slot_table_release_async_is_all_or_nothing():
    table = _source("core/GPU_Weight_Buffer/weight_slot_table.h")
    release_async = table[
        table.index("std::vector<SlotRef> release_async(") :
        table.index("void reclaim(const std::string& module_type")
    ]
    validate = release_async.index("Duplicate module in asynchronous weight release")
    leased = release_async.index("*state != WeightSlotState::kLeased")
    mutate = release_async.index("WeightSlotState::kPendingRelease")
    assert validate < leased < mutate
    # The (name, pass) key leaves the index, so the module may be copied
    # again while its old slot waits for the consumer's event.
    assert "this->index_.erase(Key{module_names[i], *pass})" in release_async


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
    # A reset runs between forwards only: with no pass open no routing skip
    # exists, and after the drain every slot is FREE.
    for reset in (prefill, decoding):
        refuse = reset.index("this->slots_.require_no_open_pass(")
        drain = reset.index("synchronizePendingReleasesLocked")
        assert refuse < drain
        assert drain < reset.index("this->slots_.reset_type(module_type, num_buffer)")
        assert "clear_pending_skips" not in reset


def test_pass_end_drains_and_reclaims_every_pending_release():
    source = _source("core/GPU_Weight_Buffer/GPU_Weight_Buffer.cpp")
    drain = source[
        source.index("void GPU_Weight_Buffer::drainPendingReleasesLocked()") :
        source.index("void GPU_Weight_Buffer::bumpProgressLocked()")
    ]
    # Unlike the reset drain, the slots return to FREE in the table.
    assert drain.index("cudaEventSynchronize(pending.event)") < drain.index(
        "this->slots_.reclaim(module_type, buffer_idx)"
    )
    wait = source[
        source.index("void GPU_Weight_Buffer::wait_pass_drained(int64_t pass)") :
        source.index("WeightSlotTable::PassSummary GPU_Weight_Buffer::end_pass(")
    ]
    assert wait.index("this->waitWithoutStallLocked(") < wait.index(
        "this->drainPendingReleasesLocked();"
    )
