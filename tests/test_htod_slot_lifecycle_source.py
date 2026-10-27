"""CPU-only source regressions for the weight-streaming slot lifecycle.

The producer thread must never let an exception escape (that is
std::terminate), a cancelled remote acquire must keep the cyclic queue order,
and a node daemon acquire that is merely not ready yet must not fault the
cohort.
"""

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "core/HtoD_Engine/HtoD_Engine.cu"
ENGINE_HEADER = ROOT / "core/HtoD_Engine/HtoD_Engine.h"
BUFFER = ROOT / "core/GPU_Weight_Buffer/GPU_Weight_Buffer.cpp"
BUFFER_HEADER = ROOT / "core/GPU_Weight_Buffer/GPU_Weight_Buffer.h"
STORAGE = ROOT / "core/Weights_Storage/Weights_Storage.cpp"
DAEMON = ROOT / "core/Weights_Storage/distributed_weight_daemon.cpp"
PROTOCOL = ROOT / "core/Weights_Storage/distributed_weights_protocol.h"
QUEUE = ROOT / "core/threadsafe_queue.h"
BATCHGEN = ROOT / "core/batchgen.cpp"
BINDING = ROOT / "core/batchgen_Binding.cpp"


def _section(source, begin, end):
    start = source.index(begin)
    return source[start:source.index(end, start)]


def _worker_body():
    return _section(
        ENGINE.read_text(),
        "void HtoD_Engine::HtoD_Worker()",
        "void HtoD_Engine::set_weight_copy_queue",
    )


def _weight_section():
    body = _worker_body()
    return body[body.index("for (auto& module_type :"):]


def test_storage_fetch_is_inside_the_producer_try():
    weights = _weight_section()
    try_open = weights.index("try {")
    fetch = weights.index("get_module_weights_storage(")
    catch = weights.index("} catch (const std::exception& error) {")
    assert try_open < fetch < catch
    # The fetch takes the stop flag, so a remote acquire can be cancelled.
    assert "get_module_weights_storage(\n" in weights
    assert "module_name, &this->terminate_flag_)" in weights


def test_producer_catch_records_the_failure_and_never_rethrows():
    weights = _weight_section()
    handlers = weights[weights.index("} catch (const std::exception& error) {"):]
    handlers = handlers[: handlers.index("// With every ring slot full")]
    assert "throw;" not in handlers
    assert handlers.count("this->fail_producer_(") == 2
    assert "} catch (...) {" in handlers
    assert handlers.count("break;") == 2

    engine = ENGINE.read_text()
    fail = _section(
        engine,
        "void HtoD_Engine::fail_producer_(",
        "py::dict HtoD_Engine::weight_stream_status()",
    )
    release = fail.index("this->weights_storage_.release_module(module_name)")
    record = fail.index("this->gpu_weight_buffer_.record_producer_failure(")
    assert release < record
    assert "this->producer_failed_ = true;" in fail
    assert "ProducerState::kFailed" in fail


def test_failure_is_visible_to_waiters_and_cleared_by_a_queue_install():
    buffer = BUFFER.read_text()
    record = _section(
        buffer,
        "void GPU_Weight_Buffer::record_producer_failure(",
        "void GPU_Weight_Buffer::clear_producer_failure()",
    )
    assert "this->producer_failure_ = message;" in record
    assert "WeightSlotState::kFilling" in record
    assert "this->slots_.abandon(module_type, buffer_idx)" in record
    assert "this->cv_.notify_all();" in record

    failed = _section(
        buffer,
        "void GPU_Weight_Buffer::throwIfProducerFailedLocked()",
        "GPU_Weight_Buffer::CopyClaim GPU_Weight_Buffer::claim_for_copy(",
    )
    assert '"HtoD producer failed: "' in failed
    get_weights = _section(
        buffer,
        "module_weight_tensor_map GPU_Weight_Buffer::get_weights(",
        "module_weight_tensor_map GPU_Weight_Buffer::get_weights_pinned(",
    )
    pinned = _section(
        buffer,
        "module_weight_tensor_map GPU_Weight_Buffer::get_weights_pinned(",
        "void GPU_Weight_Buffer::throwIfProducerFailedLocked()",
    )
    skip = _section(
        buffer,
        "GPU_Weight_Buffer::skip_modules(",
        "GPU_Weight_Buffer::SlotSnapshot GPU_Weight_Buffer::slot_snapshot()",
    )
    for method in (get_weights, pinned, skip):
        assert "this->throwIfProducerFailedLocked();" in method
    # The wait ends as soon as a failure is recorded.
    for method in (get_weights, pinned):
        assert "this->producer_failure_.has_value()" in method
    # Leasing, not just presence.
    for method in (get_weights, pinned):
        assert "this->slots_.lease(module_name)" in method
    # The rethrow keeps the cause.
    std_catch = get_weights[get_weights.index("catch (const std::exception& e)"):]
    std_catch = std_catch[: std_catch.index("catch (...)")]
    assert "throw std::runtime_error(e.what());" in std_catch

    engine = ENGINE.read_text()
    install = _section(
        engine,
        "void HtoD_Engine::set_weight_copy_queue(",
        "void HtoD_Engine::stop_h2d_worker()",
    )
    assert "this->gpu_weight_buffer_.clear_producer_failure();" in install
    assert "this->producer_failed_ = false;" in install
    assert "++this->queue_generation_;" in install


def test_stop_rethrows_after_join_and_terminate_never_throws():
    engine = ENGINE.read_text()
    stop = _section(
        engine,
        "void HtoD_Engine::stop_h2d_worker()",
        "// Forward declaration of CUDA kernel",
    )
    assert stop.index("this->HtoD_worker_.join();") < stop.index(
        'throw std::runtime_error("HtoD producer failed: "'
    )
    terminate = _section(
        engine, "void HtoD_Engine::Terminate()", "HtoD_Engine::~HtoD_Engine()"
    )
    assert "throw" not in terminate
    assert "producer_failure()" not in terminate


def test_terminate_flag_is_atomic():
    header = ENGINE_HEADER.read_text()
    assert "std::atomic<bool> terminate_flag_{false};" in header
    assert not re.search(r"^\s*bool terminate_flag_;", header, re.MULTILINE)


def test_cancelled_acquire_leaves_the_module_at_the_head():
    weights = _weight_section()
    peek = weights.index(".try_front(module_name)")
    fetch = weights.index("get_module_weights_storage(")
    cancelled = weights.index("if (!fetched.has_value()) {")
    cancel_end = weights.index("break;", cancelled)
    cancel_path = weights[cancelled:cancel_end]
    pop = weights.index(".try_pop(module_name)")
    # Peek before the fetch, pop only after it succeeded.
    assert peek < fetch < cancelled < pop
    assert ".push(" not in cancel_path
    assert ".try_pop(" not in cancel_path
    assert "this->gpu_weight_buffer_.abandon(module_type," in cancel_path
    assert "ProducerState::kParked" in cancel_path
    # The 1 ms pop spin is gone.
    assert "std::chrono::milliseconds(1)" not in weights

    queue = QUEUE.read_text()
    front = _section(queue, "bool try_front(T& value) const", "};")
    assert "data_queue.front()" in front
    assert "data_queue.pop()" not in front


def test_skip_check_follows_the_fetch_and_releases_the_daemon_lease():
    weights = _weight_section()
    fetch = weights.index("get_module_weights_storage(")
    claim = weights.index("this->gpu_weight_buffer_.claim_for_copy(")
    assert fetch < claim
    skipped = weights[claim:weights.index("this->set_producer_state_(ProducerState::kCopying")]
    # A skipped remote module is acquired and released without a copy, so the
    # daemon FIFO still advances, and it goes back on the tail.
    assert "this->weights_storage_.release_module(module_name);" in skipped
    assert "blocking_copy_" not in skipped
    assert ".push(module_name)" in skipped
    assert "this->skipped_modules_" in skipped

    buffer = BUFFER.read_text()
    claim_impl = _section(
        buffer,
        "GPU_Weight_Buffer::CopyClaim GPU_Weight_Buffer::claim_for_copy(",
        "void GPU_Weight_Buffer::abandon(",
    )
    # One buffer-mutex critical section: pending skip, then residency, then
    # naming the slot.
    lock = claim_impl.index("std::lock_guard<std::mutex> lock(this->mutex_);")
    take = claim_impl.index("this->slots_.take_pending_skip(module_name)")
    assign = claim_impl.index("this->slots_.assign(")
    assert lock < take < assign
    assert claim_impl.count("std::lock_guard") == 1


def test_landing_reports_release_on_landing():
    weights = _weight_section()
    complete = weights.index("this->gpu_weight_buffer_.weights_copy_complete(")
    assert "const bool present =" in weights[complete - 80:complete]
    assert "this->released_on_landing_modules_" in weights[complete:]
    buffer = BUFFER.read_text()
    landed = _section(
        buffer,
        "bool GPU_Weight_Buffer::weights_copy_complete(",
        "void GPU_Weight_Buffer::record_producer_failure(",
    )
    assert "this->slots_.landed(module_type, buffer_idx)" in landed
    assert "this->cv_.notify_all();" in landed


def test_eviction_only_frees_present_slots():
    buffer = BUFFER.read_text()
    clear = _section(
        buffer,
        "void GPU_Weight_Buffer::clear_expert_buffer(",
        "// void GPU_Weight_Buffer::reset_prefill_buffer()",
    )
    assert "this->slots_.evict_present_outside(" in clear
    assert "module_in_buffers_" not in clear
    header = BUFFER_HEADER.read_text()
    assert "module_in_buffers_" not in header
    assert "buffer_status_" not in header
    assert "WeightSlotTable slots_;" in header


def test_daemon_acquire_is_bounded_and_retry_is_not_a_failure():
    protocol = PROTOCOL.read_text()
    assert "constexpr std::uint32_t kProtocolVersion = 3;" in protocol
    assert "// Version 3:" in protocol
    assert "constexpr std::int32_t kStatusRetry = 1;" in protocol

    daemon = DAEMON.read_text()
    handler = _section(
        daemon, "void HandleWorker(ClientConnection* connection)",
        "void CloseUnadmitted(",
    )
    send = handler.index("send_exact(fd, &response, sizeof(response));\n                //")
    failure = handler.index("fail(response.error);", send)
    guard = handler[send:failure]
    assert "response.status != 0 && response.status != kStatusRetry" in guard

    acquire = _section(
        daemon,
        "void Acquire(const std::string& module_key, int worker_id,",
        "void Release(const std::string& module_key, int worker_id,",
    )
    assert "constexpr auto kAcquireSlice = std::chrono::milliseconds(250);" in daemon
    assert "constexpr auto kAcquireTimeout = std::chrono::seconds(300);" in daemon
    assert "ring_cv.wait_for(\n            lock, kAcquireSlice," in acquire
    assert "std::chrono::seconds(300)" not in acquire
    assert "response->status = kStatusRetry;" in acquire
    # The 300 s diagnostics span every slice of one pending acquire.
    assert "pending_acquires.at(worker_id)" in acquire
    assert "pending.completions_at_start" in acquire
    assert "remote fetch stalled: no fetch completed in 300 s" in acquire
    assert "ready modules this cohort did not consume" in acquire
    assert "oldest unacquired ready module" in acquire
    assert "OwningWorker(oldest_unacquired->module_key)" in acquire

    summary = _section(daemon, "void WriteSummary()", "std::uint64_t CohortEpoch()")
    assert 'summary["ready_unacquired_by_worker"]' in summary
    assert 'summary["pending_acquire_by_worker"]' in summary


def test_worker_acquire_loops_on_retry_and_can_be_cancelled():
    storage = STORAGE.read_text()
    acquire = _section(
        storage,
        "Weights_Storage::acquire_remote_module(",
        "void Weights_Storage::release_module(",
    )
    loop = acquire.index("while (true) {")
    retry = acquire.index("is_retry_response(response)", loop)
    stop = acquire.index("stop_requested->load()", retry)
    cancelled = acquire.index("return std::nullopt;", stop)
    validate = acquire.index("validate_response(response);", cancelled)
    lease = acquire.index("this->active_leases_[module_key] = lease;")
    assert loop < retry < stop < cancelled < validate < lease
    retry_impl = _section(
        storage, "bool is_retry_response(const Response& response)", "}  // namespace"
    )
    assert "kStatusRetry" in retry_impl
    assert "kProtocolVersion" in retry_impl


def test_bindings_expose_skip_and_status():
    binding = BINDING.read_text()
    assert '.def("skip_weight_modules", &BatchGen::skip_weight_modules,' in binding
    assert '.def("weight_stream_status", &BatchGen::weight_stream_status)' in binding

    batchgen = BATCHGEN.read_text()
    skip = _section(
        batchgen,
        "py::dict BatchGen::skip_weight_modules(",
        "py::dict BatchGen::weight_stream_status()",
    )
    for key in ("freed_present", "release_on_landing", "pending"):
        assert f'result["{key}"]' in skip

    engine = ENGINE.read_text()
    status = _section(
        engine, "py::dict HtoD_Engine::weight_stream_status()", "void HtoD_Engine::HtoD_Worker()"
    )
    for state in ("idle", "acquiring", "copying", "parked", "failed"):
        assert f'"{state}"' in status
    for key in (
        "producer_state",
        "module_in_flight",
        "copied",
        "skipped",
        "released_on_landing",
        "producer_failure",
        "slot_counts",
        "held",
    ):
        assert f'["{key}"]' in status
