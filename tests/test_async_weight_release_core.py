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

    create = release.index("std::make_shared<CudaEventHandle>()")
    record = release.index("cudaEventRecord(completion_event->get(), consumer_stream)")
    # LEASED -> PENDING_RELEASE in the slot table, all or nothing; a refused
    # release drops the handle, which destroys the event it no longer needs.
    pending = release.index("this->slots_.release_async(module_names)")
    enqueue = release.index("this->pending_releases_.push_back(")
    assert create < record < pending < enqueue
    assert "PendingRelease{std::move(completion_event), std::move(slots)}" in release
    assert "cudaEventDestroy" not in release

    query = reclaim.index("cudaEventQuery(it->event->get())")
    not_ready = reclaim.index("status == cudaErrorNotReady")
    # PENDING_RELEASE -> FREE only once the consumer's event completed.
    reusable = reclaim.index("this->slots_.reclaim(module_type, buffer_idx)")
    dropped = reclaim.index("it = this->pending_releases_.erase(it);")
    assert query < not_ready < reusable < dropped
    # The handle destroys the event once no waiter holds it.
    assert "cudaEventDestroy" not in reclaim
    assert "this->reclaimCompletedReleasesLocked();" in acquire
    assert "this->slots_.reserve(module_type)" in acquire


def test_pending_release_event_is_a_shared_raii_handle():
    header = _source("core/GPU_Weight_Buffer/GPU_Weight_Buffer.h")
    handle = header[
        header.index("class CudaEventHandle {") :
        header.index("struct PendingRelease {")
    ]
    pending = header[
        header.index("struct PendingRelease {") :
        header.index("void notifySlotsLocked();")
    ]
    # The producer copies the handle and waits on the event outside the
    # mutex; a drain, reset or abort may drop the release meanwhile.
    assert "std::shared_ptr<CudaEventHandle> event;" in pending
    assert "cudaEvent_t event;" not in pending
    assert "cudaEventCreateWithFlags(&this->event_," in handle
    assert "cudaEventDisableTiming" in handle
    assert "~CudaEventHandle() { cudaEventDestroy(this->event_); }" in handle
    assert "CudaEventHandle(const CudaEventHandle&) = delete;" in handle
    assert "CudaEventHandle& operator=(const CudaEventHandle&) = delete;" in handle

    # Only the handle destroys release events.
    source = _source("core/GPU_Weight_Buffer/GPU_Weight_Buffer.cpp")
    assert "cudaEventDestroy" not in source
    assert "cudaEventCreateWithFlags" not in source


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

    assert "cudaEventSynchronize(pending.event->get())" in synchronize
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
    assert drain.index("cudaEventSynchronize(pending.event->get())") < drain.index(
        "this->slots_.reclaim(module_type, buffer_idx)"
    )
    wait = source[
        source.index("void GPU_Weight_Buffer::wait_pass_drained(int64_t pass)") :
        source.index("WeightSlotTable::PassSummary GPU_Weight_Buffer::end_pass(")
    ]
    assert wait.index("this->waitWithoutStallLocked(") < wait.index(
        "this->drainPendingReleasesLocked();"
    )
