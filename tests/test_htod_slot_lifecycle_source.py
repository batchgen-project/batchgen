"""CPU-only source regressions for the weight-streaming slot lifecycle.

The producer thread must never let an exception escape (that is
std::terminate), a cancelled remote acquire must keep the schedule order, a
node daemon acquire that is merely not ready yet must not fault the cohort,
and every slot and daemon lease belongs to exactly one weight pass that the
consumer arms and closes.
"""

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "core/HtoD_Engine/HtoD_Engine.cu"
ENGINE_HEADER = ROOT / "core/HtoD_Engine/HtoD_Engine.h"
BUFFER = ROOT / "core/GPU_Weight_Buffer/GPU_Weight_Buffer.cpp"
BUFFER_HEADER = ROOT / "core/GPU_Weight_Buffer/GPU_Weight_Buffer.h"
STORAGE = ROOT / "core/Weights_Storage/Weights_Storage.cpp"
STORAGE_HEADER = ROOT / "core/Weights_Storage/Weights_Storage.h"
DAEMON = ROOT / "core/Weights_Storage/distributed_weight_daemon.cpp"
PROTOCOL = ROOT / "core/Weights_Storage/distributed_weights_protocol.h"
QUEUE = ROOT / "core/threadsafe_queue.h"
BATCHGEN = ROOT / "core/batchgen.cpp"
BATCHGEN_HEADER = ROOT / "core/batchgen.h"
BINDING = ROOT / "core/batchgen_Binding.cpp"
PACKAGE = ROOT / "batchgen"


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
    handlers = handlers[: handlers.index("if (!progressed) {")]
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
        "WeightProduction GPU_Weight_Buffer::claim_for_copy(",
    )
    assert '"HtoD producer failed: "' in failed

    # get_weights and the pass-end wait share one wait, which throws the
    # failure and ends as soon as one is recorded.
    wait = _section(
        buffer,
        "void GPU_Weight_Buffer::waitWithoutStallLocked(",
        "void GPU_Weight_Buffer::reclaimCompletedReleasesLocked()",
    )
    assert "this->throwIfProducerFailedLocked();" in wait
    assert "this->producer_failure_.has_value()" in wait
    get_weights = _section(
        buffer,
        "module_weight_tensor_map GPU_Weight_Buffer::get_weights(",
        "int64_t GPU_Weight_Buffer::begin_pass()",
    )
    assert "this->waitWithoutStallLocked(lock, module_name," in get_weights
    assert "this->slots_.lease(module_name)" in get_weights
    drained = _section(
        buffer,
        "void GPU_Weight_Buffer::wait_pass_drained(int64_t pass)",
        "WeightSlotTable::PassSummary GPU_Weight_Buffer::end_pass(",
    )
    assert "this->waitWithoutStallLocked(" in drained
    skip = _section(
        buffer,
        "GPU_Weight_Buffer::skip_modules(",
        "GPU_Weight_Buffer::SlotSnapshot GPU_Weight_Buffer::slot_snapshot()",
    )
    assert "this->throwIfProducerFailedLocked();" in skip

    engine = ENGINE.read_text()
    install = _section(
        engine,
        "void HtoD_Engine::set_weight_copy_queue(",
        "void HtoD_Engine::stop_h2d_worker()",
    )
    assert "this->gpu_weight_buffer_.clear_producer_failure();" in install
    assert "this->producer_failed_ = false;" in install


def test_stall_rule_is_the_only_timeout_and_never_a_busy_spin():
    buffer = BUFFER.read_text()
    wait = _section(
        buffer,
        "void GPU_Weight_Buffer::waitWithoutStallLocked(",
        "void GPU_Weight_Buffer::reclaimCompletedReleasesLocked()",
    )
    # The clock restarts on every counted progress, and the wait sleeps on
    # the buffer cv until the stall deadline.
    assert "const auto deadline = last_progress + kStallTimeout;" in wait
    assert "this->cv_.wait_until(lock, deadline," in wait
    assert "this->progress_epoch_ != seen" in wait
    assert "last_progress = std::chrono::steady_clock::now();" in wait
    assert "this->slots_.describe_passes()" in wait
    assert "this->slots_.describe_held()" in wait
    assert "milliseconds" not in wait
    header = BUFFER_HEADER.read_text()
    assert "static constexpr std::chrono::seconds kStallTimeout{300};" in header

    # Landings, claims, retired skips and producer state changes count.
    bump = "this->bumpProgressLocked();"
    for begin, end in (
        ("bool GPU_Weight_Buffer::weights_copy_complete(",
         "void GPU_Weight_Buffer::record_producer_failure("),
        ("WeightProduction GPU_Weight_Buffer::claim_for_copy(",
         "void GPU_Weight_Buffer::abandon("),
        ("void GPU_Weight_Buffer::retire_skip(",
         "void GPU_Weight_Buffer::note_producer_progress()"),
        ("void GPU_Weight_Buffer::note_producer_progress()",
         "void GPU_Weight_Buffer::throwIfProducerFailedLocked()"),
    ):
        assert bump in _section(buffer, begin, end), begin
    engine = ENGINE.read_text()
    state = _section(
        engine,
        "void HtoD_Engine::set_producer_state_(",
        "std::string HtoD_Engine::describe_producer_locked_()",
    )
    # Outside status_mutex_: the buffer mutex never nests inside it.
    assert state.index("}\n") < state.index(
        "this->gpu_weight_buffer_.note_producer_progress();"
    )


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
    claim = weights.index("this->gpu_weight_buffer_.claim_for_copy(")
    pop = weights.index(".try_pop(module_name)")
    push = weights.index(".push(module_name)")
    # Peek before the fetch; claim, pop and push back only after it
    # succeeded, once per module of every pass.
    assert peek < fetch < cancelled < claim < pop < push
    assert ".push(" not in cancel_path
    assert ".try_pop(" not in cancel_path
    assert "claim_for_copy" not in cancel_path
    assert "this->gpu_weight_buffer_.abandon(module_type," in cancel_path
    assert "ProducerState::kParked" in cancel_path
    # The 1 ms pop spin is gone.
    assert "std::chrono::milliseconds(1)" not in weights
    # The queue head must be the table cursor's module.
    assert "module_name != next.module_name" in weights[peek:fetch]

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
    # daemon FIFO still advances, and it goes back on the tail. The cursor
    # moves past it only after the lease is released.
    release = skipped.index("this->weights_storage_.release_module(module_name);")
    retire = skipped.index("this->gpu_weight_buffer_.retire_skip(module_type,")
    assert release < retire
    assert "blocking_copy_" not in skipped
    assert ".push(module_name)" in skipped
    assert "this->skipped_modules_" in skipped
    # A pending skip reserves no ring slot.
    head = weights[:fetch]
    assert "if (next.kind == WeightProduction::kCopy) {" in head
    assert head.index("if (next.kind == WeightProduction::kCopy) {") < head.index(
        "acquireEmptyBuffer(module_type)"
    )

    buffer = BUFFER.read_text()
    claim_impl = _section(
        buffer,
        "WeightProduction GPU_Weight_Buffer::claim_for_copy(",
        "void GPU_Weight_Buffer::abandon(",
    )
    # One buffer-mutex critical section: the table checks the pending skip
    # and names the slot for the cursor's pass.
    lock = claim_impl.index("std::lock_guard<std::mutex> lock(this->mutex_);")
    table = claim_impl.index("this->slots_.claim(module_type, buffer_idx, module_name)")
    assert lock < table
    assert claim_impl.count("std::lock_guard") == 1


def test_lease_is_released_before_the_copy_lands():
    weights = _weight_section()
    copying = weights.index("ProducerState::kCopying")
    release = weights.index("this->weights_storage_.release_module(module_name);", copying)
    complete = weights.index("this->gpu_weight_buffer_.weights_copy_complete(", copying)
    # No FILLING slot of a pass implies no daemon lease of it.
    assert release < complete
    assert "const bool present =" in weights[complete - 80:complete]
    assert "this->released_on_landing_modules_" in weights[complete:]
    buffer = BUFFER.read_text()
    landed = _section(
        buffer,
        "bool GPU_Weight_Buffer::weights_copy_complete(",
        "void GPU_Weight_Buffer::record_producer_failure(",
    )
    assert "this->slots_.landed(module_type, buffer_idx)" in landed


def test_queue_install_requires_a_stopped_producer_and_no_open_pass():
    engine = ENGINE.read_text()
    install = _section(
        engine,
        "void HtoD_Engine::set_weight_copy_queue(",
        "void HtoD_Engine::stop_h2d_worker()",
    )
    refuse = install.index("if (this->HtoD_worker_.joinable()) {")
    table = install.index("this->gpu_weight_buffer_.set_weight_copy_task(")
    queues = install.index("this->weights_copy_task_queue_.clear();")
    assert refuse < table < queues
    assert "queue_generation_" not in engine
    assert "queue_generation_" not in ENGINE_HEADER.read_text()

    batchgen = BATCHGEN.read_text()
    install = _section(
        batchgen,
        "void BatchGen::set_weight_copy_queue(",
        "void BatchGen::reset_decoding_buffer()",
    )
    assert "this->h2d_engine_.set_weight_copy_queue(weight_copy_tasks);" in install
    assert "set_weight_copy_task" not in install

    buffer = BUFFER.read_text()
    for reset in ("reset_prefill_buffer", "reset_decoding_buffer"):
        start = buffer.rindex(f"void GPU_Weight_Buffer::{reset}()")
        body = buffer[start:buffer.index("\n}\n", start)]
        assert body.index("this->slots_.require_no_open_pass(") < body.index(
            "synchronizePendingReleasesLocked"
        )


def test_end_pass_waits_then_checks_daemon_leases_then_closes():
    batchgen = BATCHGEN.read_text()
    end = _section(
        batchgen,
        "py::dict BatchGen::end_weight_pass(int64_t pass_id)",
        "py::dict BatchGen::cancel_weight_pass(int64_t pass_id)",
    )
    release = end.index("py::gil_scoped_release release;")
    wait = end.index("this->gpu_weight_buffer_.wait_pass_drained(pass_id);")
    leases = end.index("this->h2d_engine_.require_no_daemon_lease_for_pass(pass_id);")
    close = end.index("this->gpu_weight_buffer_.end_pass(pass_id);")
    result = end.index("py::dict result;")
    assert release < wait < leases < close < result
    for key in ("pass", "copied", "skipped", "released_on_landing", "carried"):
        assert f'result["{key}"]' in end
    assert "this->h2d_engine_.describe_producer()" in end

    engine = ENGINE.read_text()
    check = _section(
        engine,
        "void HtoD_Engine::require_no_daemon_lease_for_pass(int64_t pass)",
        "py::dict HtoD_Engine::weight_stream_status()",
    )
    lock = check.index("std::lock_guard<std::mutex> lock(this->status_mutex_);")
    leased = check.index("this->weights_storage_.leased_modules()")
    assert lock < leased
    assert "this->pass_in_flight_ > pass" in check

    storage = STORAGE.read_text()
    accessor = _section(
        storage,
        "std::vector<std::string> Weights_Storage::leased_modules() const",
        "void Weights_Storage::release_module(",
    )
    assert "std::lock_guard<std::mutex> index_lock(this->lease_index_mutex_);" in accessor
    assert "daemon_mutex_" not in accessor
    acquire = _section(
        storage,
        "Weights_Storage::acquire_remote_module(",
        "std::vector<std::string> Weights_Storage::leased_modules() const",
    )
    release = _section(
        storage,
        "void Weights_Storage::release_module(",
        "void Weights_Storage::Init(",
    )
    for writer in (acquire, release):
        assert "std::lock_guard<std::mutex> lock(this->daemon_mutex_);" in writer
        assert "std::lock_guard<std::mutex> index_lock(this->lease_index_mutex_);" in writer
    assert "mutable std::mutex lease_index_mutex_;" in STORAGE_HEADER.read_text()


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


def test_bindings_expose_passes_skip_and_status():
    binding = BINDING.read_text()
    assert '.def("skip_weight_modules", &BatchGen::skip_weight_modules,' in binding
    assert '.def("weight_stream_status", &BatchGen::weight_stream_status)' in binding
    assert '.def("begin_weight_pass", &BatchGen::begin_weight_pass)' in binding
    assert '.def("end_weight_pass", &BatchGen::end_weight_pass,\n             py::arg("pass_id"))' in binding
    assert '.def("cancel_weight_pass", &BatchGen::cancel_weight_pass,\n             py::arg("pass_id"))' in binding

    batchgen = BATCHGEN.read_text()
    skip = _section(
        batchgen,
        "py::dict BatchGen::skip_weight_modules(",
        "py::dict BatchGen::weight_stream_status()",
    )
    for key in ("freed_present", "release_on_landing", "pending"):
        assert f'result["{key}"]' in skip
    cancel = _section(
        batchgen,
        "py::dict BatchGen::cancel_weight_pass(int64_t pass_id)",
        "void BatchGen::reset_prefill_buffer()",
    )
    for key in ("pass", "freed_present", "release_on_landing", "pending"):
        assert f'result["{key}"]' in cancel

    engine = ENGINE.read_text()
    states = _section(engine, "const char* const kProducerStates[] = {", "};")
    for state in ("idle", "acquiring", "copying", "parked", "failed"):
        assert f'"{state}"' in states
    status = _section(
        engine, "py::dict HtoD_Engine::weight_stream_status()", "void HtoD_Engine::HtoD_Worker()"
    )
    for key in (
        "producer_state",
        "module_in_flight",
        "pass_in_flight",
        "copied",
        "skipped",
        "released_on_landing",
        "producer_failure",
        "slot_counts",
        "held",
        "open_passes",
        "producer_cursor",
        "passes",
    ):
        assert f'["{key}"]' in status


def test_get_weights_binding_takes_only_the_module_name():
    binding = BINDING.read_text()
    assert (
        '.def("get_weights", &BatchGen::get_weights, py::arg("module_name"),\n'
        "             py::call_guard<py::gil_scoped_release>())"
    ) in binding
    assert re.search(
        r"get_weights\(\s*const std::string& module_name\);",
        BATCHGEN_HEADER.read_text(),
    )
    assert re.search(
        r"BatchGen::get_weights\(\s*const std::string& module_name\) \{",
        BATCHGEN.read_text(),
    )
    assert (
        "module_weight_tensor_map get_weights(const std::string& module_name);"
        in BUFFER_HEADER.read_text()
    )

    # Every Python caller passes the module name alone.
    callers = 0
    for path in PACKAGE.rglob("*.py"):
        for match in re.finditer(r"core_engine\.get_weights\(([^)]*)\)", path.read_text()):
            callers += 1
            assert "," not in match.group(1).strip().rstrip(","), (path, match.group(0))
    assert callers >= 2


DELETED_CORE = (
    "clear_expert_buffer",
    "evict_present_outside",
    "get_weights_pinned",
    "clear_weight_copy_queue",
    "reset_weight_copy_queue",
    "queue_generation_",
    "prefill_sp8",
    "take_pending_skip",
    "CopyClaim",
)
DELETED_PYTHON = (
    "clear_expert_buffer",
    "clear_weight_copy_queue",
    "get_weights_pinned",
    "load_weights_pinned",
    "reset_weight_copy_queue",
    "prefill_sp8",
)
DEAD_MODULES = (
    "models/Wrapper_dep.py",
    "inference_runtime.py",
    "prefill_task.py",
)


def test_deleted_symbols_are_gone_from_core():
    sources = [
        path
        for pattern in ("*.h", "*.cpp", "*.cu")
        for path in (ROOT / "core").rglob(pattern)
    ]
    assert sources
    for path in sources:
        text = path.read_text(errors="replace")
        for symbol in DELETED_CORE:
            assert symbol not in text, (path, symbol)


def test_deleted_symbols_are_gone_from_python_callers():
    for path in PACKAGE.rglob("*.py"):
        text = path.read_text()
        for symbol in DELETED_PYTHON:
            assert symbol not in text, (path, symbol)


def test_dead_python_modules_are_deleted():
    for relative in DEAD_MODULES:
        assert not (PACKAGE / relative).exists(), relative
