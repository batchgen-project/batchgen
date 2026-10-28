"""CPU-only source regressions for the weight-streaming slot lifecycle.

The producer thread must never let an exception escape (that is
std::terminate), a cancelled remote acquire must keep the schedule order, a
node daemon acquire that is merely not ready yet must not fault the cohort,
and every slot and daemon lease belongs to exactly one weight pass that the
consumer arms and closes. Up to kMaxCopiesInFlight modules are in flight, each
with one event after its copies; the producer alone observes the events and
lands the copies oldest first, and it never waits for the node daemon or for
a free slot while a copy is un-landed: with copies in flight it only tries a
remote module, one request the daemon answers at once, and leaves a module
that is not ready at the head of its queue. The daemon posts each schedule
chunk into any free chunk-slot, a chunk an acquire waits for first, and its
summary copies its state and formats it outside the locks the request path
takes.
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


def _land_copies():
    return _section(
        ENGINE.read_text(),
        "size_t HtoD_Engine::land_copies_(bool wait)",
        "py::dict HtoD_Engine::weight_stream_status()",
    )


def test_storage_fetch_is_inside_the_producer_try():
    weights = _weight_section()
    try_open = weights.index("try {")
    fetch = weights.index("get_module_weights_storage(")
    catch = weights.index("} catch (const std::exception& error) {")
    assert try_open < fetch < catch
    # The fetch takes the stop flag, so a remote acquire can be cancelled,
    # and whether it may wait for the node daemon.
    assert "get_module_weights_storage(\n" in weights
    assert "module_name, &this->terminate_flag_, may_wait)" in weights


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
        "size_t HtoD_Engine::land_copies_(bool wait)",
    )
    # No copy may still write into a slot about to be abandoned: the copy
    # stream is synchronized first, without CUDA_CHECK, which would throw.
    sync = fail.index("cudaStreamSynchronize(this->HtoD_stream)")
    walk = fail.index("for (InFlightCopy& copy : this->in_flight_) {")
    slots = fail.index("filling.emplace_back(copy.module_type, copy.buffer_idx);")
    release = fail.index("this->weights_storage_.release_module(copy.module_name)")
    record = fail.index(
        "this->gpu_weight_buffer_.record_producer_failure(filling, failure);"
    )
    dropped = fail.index("this->end_in_flight_(true);", record)
    failed = fail.index("ProducerState::kFailed", dropped)
    assert sync < walk < slots < release < record < dropped < failed
    assert "CUDA_CHECK" not in fail
    assert "throw" not in fail
    # The slot reserved for a module not in flight yet is abandoned too.
    assert "filling.emplace_back(module_type, buffer_idx);" in fail[:walk]
    assert "this->producer_failed_ = true;" in fail


def test_failure_is_visible_to_waiters_and_cleared_by_a_queue_install():
    buffer = BUFFER.read_text()
    record = _section(
        buffer,
        "void GPU_Weight_Buffer::record_producer_failure(",
        "void GPU_Weight_Buffer::clear_producer_failure()",
    )
    assert "this->producer_failure_ = message;" in record
    # Every slot in flight is abandoned, each only while still FILLING.
    loop = record.index("for (const auto& [module_type, buffer_idx] : filling) {")
    filling = record.index("WeightSlotState::kFilling", loop)
    abandon = record.index("this->slots_.abandon(module_type, buffer_idx)", filling)
    assert loop < filling < abandon
    assert "this->cv_.notify_all();" in record
    assert re.search(
        r"void record_producer_failure\(\s*"
        r"const std::vector<std::pair<std::string, int64_t>>& filling,\s*"
        r"const std::string& message\);",
        BUFFER_HEADER.read_text(),
    )

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
    cancelled = weights.index(
        "if (fetched.status == module_fetch::Status::kCancelled) {"
    )
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


def test_a_try_fetch_that_is_not_ready_leaves_the_module_at_the_head():
    weights = _weight_section()
    fetch = weights.index("get_module_weights_storage(")
    cancelled = weights.index(
        "if (fetched.status == module_fetch::Status::kCancelled) {"
    )
    not_ready = weights.index(
        "if (fetched.status == module_fetch::Status::kNotReady) {"
    )
    path = weights[not_ready:weights.index("continue;", not_ready)]
    leased = weights.index("this->in_flight_.back().leased = true;")
    claim = weights.index("this->gpu_weight_buffer_.claim_for_copy(")
    pop = weights.index(".try_pop(module_name)")
    assert fetch < cancelled < not_ready < leased < claim < pop
    # The cancelled fetch's contract: the reserved slot is abandoned and the
    # entry dropped; queue head and cursor stay. Then the next type, not a
    # park: the oldest copy lands first, so the try is retried per landing.
    assert "this->gpu_weight_buffer_.abandon(module_type," in path
    assert "this->end_in_flight_(false);" in path
    for token in (
        ".push(", ".try_pop(", "claim_for_copy", "progressed = true",
        "set_producer_state_", "release_module", "break;",
    ):
        assert token not in path, token


def test_a_module_is_in_flight_from_before_its_fetch():
    weights = _weight_section()
    peek = weights.index(".try_front(module_name)")
    begin = weights.index("this->begin_in_flight_(module_type, module_name, buffer_idx,")
    acquiring = weights.index("this->set_producer_state_(ProducerState::kAcquiring);")
    fetch = weights.index("get_module_weights_storage(")
    leased = weights.index("this->in_flight_.back().leased = true;")
    claim = weights.index("this->gpu_weight_buffer_.claim_for_copy(")
    # Named before the fetch, so any daemon lease is of a module in flight.
    assert peek < begin < acquiring < fetch < leased < claim
    cancelled = weights.index(
        "if (fetched.status == module_fetch::Status::kCancelled) {"
    )
    cancel_path = weights[cancelled:weights.index("break;", cancelled)]
    assert "this->end_in_flight_(false);" in cancel_path


def test_a_fetch_may_wait_for_the_daemon_only_with_nothing_in_flight():
    weights = _weight_section()
    full = weights.index("this->in_flight_.size() >= kMaxCopiesInFlight")
    unparked = weights.index("weights_parked = false;")
    # Read before the module is named in flight: blocked in a daemon acquire
    # the producer could release no lease and land no copy, so a fetch may
    # wait only while nothing, hence no lease, is in flight. With copies in
    # flight a remote module is only tried.
    gate = weights.index("const bool may_wait = this->in_flight_.empty();")
    reserve = weights.index("acquireEmptyBuffer(module_type)")
    begin = weights.index("this->begin_in_flight_(module_type, module_name, buffer_idx,")
    fetch = weights.index("get_module_weights_storage(")
    assert full < unparked < gate < reserve < begin < fetch
    # The one fetch, and its wait argument is that gate.
    assert weights.count("get_module_weights_storage(") == 1
    assert "module_name, &this->terminate_flag_, may_wait)" in weights
    assert "is_remote_module" not in weights
    # Only a fetch that may wait shows as acquiring: a try that finds its
    # module not ready changes no state and counts no consumer progress.
    acquiring = weights.index("this->set_producer_state_(ProducerState::kAcquiring);")
    assert begin < weights.rindex("if (may_wait) {", 0, acquiring) < acquiring < fetch
    assert weights.count("ProducerState::kAcquiring") == 1
    # Progress is counted only once the fetch returned the module.
    not_ready = weights.index("if (fetched.status == module_fetch::Status::kNotReady) {")
    progressed = weights.index("progressed = true;", fetch)
    assert fetch < not_ready < progressed
    assert "progressed = true;" not in weights[unparked:fetch]
    header = ENGINE_HEADER.read_text()
    assert "static constexpr size_t kMaxCopiesInFlight = 4;" in header
    assert "static constexpr size_t kCopyEvents = kMaxCopiesInFlight + 1;" in header


def test_a_module_is_copied_asynchronously_with_one_event():
    weights = _weight_section()
    copying = weights[weights.index("this->set_producer_state_(ProducerState::kCopying"):]
    copying = copying[:copying.index("} catch (const std::exception& error) {")]
    memcpy = copying.index("CUDA_CHECK(cudaMemcpyAsync(slot->second.data_ptr(),")
    loop_end = copying.index("}\n", copying.index("submitted_bytes +=", memcpy))
    record = re.search(
        r"CUDA_CHECK\(cudaEventRecord\(this->in_flight_\.back\(\)\.event,\s*"
        r"this->HtoD_stream\)\);",
        copying,
    )
    assert record is not None
    assert memcpy < loop_end < record.start()
    assert copying.count("cudaEventRecord(") == 1
    # The weight path never synchronizes the copy stream; on-demand and KV
    # copies keep blocking_copy_ and their own stream syncs.
    assert "blocking_copy_" not in weights
    assert "cudaStreamSynchronize" not in weights
    engine = ENGINE.read_text()
    blocking = _section(
        engine, "void HtoD_Engine::blocking_copy_(", "void HtoD_Engine::clear_kv_copy_queue()"
    )
    assert "CUDA_CHECK(cudaStreamSynchronize(this->HtoD_stream));" in blocking

    take = _section(
        engine,
        "cudaEvent_t HtoD_Engine::take_copy_event_()",
        "void HtoD_Engine::require_no_daemon_lease_for_pass(int64_t pass)",
    )
    assert "while (this->copy_events_.size() < kCopyEvents) {" in take
    assert "cudaEventCreateWithFlags(&event, cudaEventDisableTiming)" in take
    destructor = _section(engine, "HtoD_Engine::~HtoD_Engine() {", "torch::Tensor HtoD_Engine::")
    terminate = destructor.index("this->Terminate();")
    destroy = destructor.index("cudaEventDestroy(event);")
    assert terminate < destroy
    assert "for (cudaEvent_t event : this->copy_events_) {" in destructor
    assert engine.count("cudaEventDestroy(") == 1


def test_copies_land_oldest_first_and_only_on_the_producer():
    land = _land_copies()
    loop = land.index("while (!this->in_flight_.empty()) {")
    oldest = land.index("InFlightCopy& copy = this->in_flight_.front();", loop)
    wait = land.index("CUDA_CHECK(cudaEventSynchronize(copy.event));", oldest)
    query = land.index("cudaEventQuery(copy.event)", wait)
    not_ready = land.index("if (status == cudaErrorNotReady) {", query)
    stop = land.index("break;", not_ready)
    assert loop < oldest < wait < query < not_ready < stop
    # Only the first copy of a waiting call is waited for.
    assert "if (wait && landed == 0) {" in land
    # weights_copy_complete is the only FILLING -> PRESENT writer, and only
    # land_copies_ calls it.
    engine = ENGINE.read_text()
    assert engine.count("this->gpu_weight_buffer_.weights_copy_complete(") == 1
    assert "cudaLaunchHostFunc" not in engine
    assert "cudaStreamAddCallback" not in engine


def test_land_before_any_wait_and_land_everything_on_stop():
    body = _worker_body()
    loop = body.index("while (!terminate_flag_) {")
    land = body.index("this->land_copies_(false)", loop)
    on_demand = body.index("on_demand_task_queue_.try_pop", loop)
    modules = body.index("for (auto& module_type :", loop)
    idle = body.index("if (!progressed) {", modules)
    in_flight = body.index("if (!this->in_flight_.empty()) {", idle)
    land_oldest = body.index("this->land_copies_(true);", in_flight)
    park = body.index("this->producer_wake_cv_.wait(lock,", land_oldest)
    slot_wait = body.index("this->gpu_weight_buffer_.wait_for_free_slot(", park)
    assert land < on_demand < modules < idle < in_flight < land_oldest < park < slot_wait

    # After the loop: nothing is submitted, every copy in flight lands, and a
    # failure on the way is recorded, never thrown.
    drain = body[body.rindex("while (!this->in_flight_.empty()) {"):]
    assert body.rindex("while (!this->in_flight_.empty()) {") > slot_wait
    assert "this->land_copies_(true);" in drain
    assert drain.count('this->fail_producer_("", -1, "",') == 2
    assert "throw;" not in drain
    assert "get_module_weights_storage" not in drain
    assert "cudaMemcpyAsync" not in drain


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
    # Its in-flight entry, the newest, is dropped only after both.
    dropped = skipped.index("this->end_in_flight_(false);")
    assert release < retire < dropped
    assert "cudaMemcpyAsync" not in skipped
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
    land = _land_copies()
    observed = land.index("cudaEventQuery(copy.event)")
    release = land.index("this->weights_storage_.release_module(copy.module_name);")
    complete = land.index("this->gpu_weight_buffer_.weights_copy_complete(")
    dropped = land.index("this->end_in_flight_(true);")
    # The lease is released only once the event shows the copy out of its
    # staging slot. No FILLING slot of a pass implies no daemon lease of it,
    # and the entry is dropped only after the landing, so a failure in it
    # still abandons the slot.
    assert observed < release < complete < dropped
    assert "const bool present =" in land[complete - 80:complete]
    assert "this->released_on_landing_modules_" in land[complete:]
    # The copy path itself never lands or releases a copied module.
    weights = _weight_section()
    copying = weights[weights.index("this->set_producer_state_(ProducerState::kCopying"):]
    copying = copying[:copying.index("} catch (const std::exception& error) {")]
    assert "weights_copy_complete" not in copying
    assert "release_module" not in copying
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
    walk = check.index("this->in_flight_.begin(), this->in_flight_.end(),", leased)
    assert lock < leased < walk
    # A lease is allowed only for a module in flight for a later pass.
    assert "copy.module_name == module_name && copy.pass > pass" in check
    # Every in-flight entry is pushed and popped under status_mutex_.
    for begin, end in (
        ("void HtoD_Engine::begin_in_flight_(", "void HtoD_Engine::end_in_flight_("),
        ("void HtoD_Engine::end_in_flight_(", "cudaEvent_t HtoD_Engine::take_copy_event_()"),
    ):
        section = _section(engine, begin, end)
        assert "std::lock_guard<std::mutex> lock(this->status_mutex_);" in section, begin
    assert "module_in_flight_" not in engine
    assert "pass_in_flight_" not in engine

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
    assert "constexpr std::uint32_t kProtocolVersion = 4;" in protocol
    assert "// Version 4:" in protocol
    assert "constexpr std::int32_t kStatusRetry = 1;" in protocol
    assert "constexpr std::uint32_t kAcquireNoWait = 1;" in protocol
    request = _section(protocol, "struct Request {", "};")
    assert "std::uint32_t flags = 0;" in request
    assert "reserved" not in request

    daemon = DAEMON.read_text()
    # Workers and daemon are one build: any other version fails the hello.
    hello = _section(daemon, "std::string AdmitWorker(", "void WaitReadyForWorker(")
    assert "batchgen::distributed_weights::kProtocolVersion ||" in hello
    assert '"invalid distributed weights worker hello (protocol "' in hello
    handler = _section(
        daemon, "void HandleWorker(ClientConnection* connection)",
        "void CloseUnadmitted(",
    )
    send = handler.index("send_exact(fd, &response, sizeof(response));\n                //")
    failure = handler.index("fail(response.error);", send)
    guard = handler[send:failure]
    assert "response.status != 0 && response.status != kStatusRetry" in guard
    # The flag reaches Acquire; an unknown flag bit is an invalid request.
    assert "(operation.flags & ~kAcquireNoWait) != 0" in handler
    assert "(operation.flags & kAcquireNoWait) != 0," in handler

    acquire = _section(
        daemon,
        "void Acquire(const std::string& module_key, int worker_id,",
        "void Release(const std::string& module_key, int worker_id,",
    )
    assert "constexpr auto kAcquireSlice = std::chrono::milliseconds(250);" in daemon
    assert "constexpr auto kAcquireTimeout = std::chrono::seconds(300);" in daemon
    # A no-wait acquire evaluates the slice's predicate once, with no wait;
    # everything before and after is shared.
    evaluate = acquire.index(
        "no_wait ? ready() : ring_cv.wait_for(lock, kAcquireSlice, ready);"
    )
    assert acquire.count("ring_cv.wait_for(") == 1
    assert acquire.count("no_wait") == 2  # the parameter and that branch
    pending = acquire.index("pending_acquires.at(worker_id)")
    polled = acquire.index("pending.last_poll = Clock::now();")
    retry = acquire.index("response->status = kStatusRetry;")
    stalled = acquire.index("remote fetch stalled")
    stats = acquire.index("++stats.acquire_count;")
    assert pending < evaluate < polled < retry < stalled < stats
    # A module that is not ready demands its chunk before any wait.
    not_ready = acquire.index("if (!ready()) {")
    demand = acquire.index(
        "DemandChunkLocked(worker_id, ChunkOf(layer, expert, worker_id));"
    )
    assert pending < not_ready < demand < evaluate
    assert acquire.count("DemandChunkLocked(") == 2  # the call and the helper
    assert "std::chrono::seconds(300)" not in acquire
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


def test_daemon_posts_demanded_chunks_first_into_free_chunk_slots():
    daemon = DAEMON.read_text()
    # The positional ring is gone: no tail, no in-order slot check.
    assert "SlotsFree" not in daemon
    assert "int tail = 0;" not in daemon
    assert "constexpr int kOwnerChunksPerRail = 3;" in daemon
    loop = _section(daemon, "void PrefetchLoop() {", "CohortSnapshot SnapshotCohort()")
    capacity = loop.index("2 * (outstanding + depth / superchunk)")
    # The window refills in schedule order and wraps with the schedule.
    refill = loop.index("while (posting_window.size() < window_capacity) {")
    wrap = loop.index("if (next == schedule.size()) {", refill)
    take = loop.index("schedule_chunks[next / superchunk];", wrap)
    order = loop.index('fail("posting window refill left schedule order");', take)
    deferred = loop.index("for (ChunkKey& wanted : deferred_demand) {", order)
    append = loop.index("posting_window.push_back(pending);", deferred)
    advance = loop.index("next += superchunk;", append)
    assert capacity < refill < wrap < take < order < deferred < append < advance
    # No free chunk-slot: counted, nothing posted.
    empty = loop.index("if (free_chunk_slots.empty()) {", advance)
    assert loop.index("++stats.freelist_empty_polls;", empty) < loop.index("break;", empty)
    # The first demanded entry, else the oldest one under its owner's cap.
    demanded = loop.index("return pending.demanded;", empty)
    capped = loop.index("kOwnerChunksPerRail *", demanded)
    none = loop.index("break;  // every owner at its cap, nothing demanded", capped)
    erase = loop.index("posting_window.erase(selected);", none)
    # Any free chunk-slot, checked free before it is filled.
    pop = loop.index("free_chunk_slots.pop_back();", erase)
    live = loop.index('fail("posted into a live chunk-slot");', pop)
    twice = loop.index('fail("chunk posted twice: "', live)
    fetched = loop.index("module_to_slot.count(module_key) != 0", twice)
    fill = loop.index("slot.state = SlotState::kFetching;", fetched)
    pinned = loop.index("chunk_slot_live[chunk_slot] = superchunk;", fill)
    expedited = loop.index("stats.expedited_posts += pick.demanded;", pinned)
    post = loop.index("chunk.request = ucp_get_nbx(", expedited)
    assert empty < demanded < capped < none < erase < pop < live < twice
    assert twice < fetched < fill < pinned < expedited < post
    # One get per chunk, into its chunk-slot; outstanding caps the flight.
    assert loop.count("ucp_get_nbx(") == 1
    assert "static_cast<std::uint64_t>(chunk.slot_start) *" in loop
    assert "while (static_cast<int>(chunks.size()) < outstanding) {" in loop
    assert loop.count("++owner_in_flight[owner];") == 1
    assert loop.count("--owner_in_flight[iterator->owner];") == 2

    # The last release of a chunk-slot's modules frees it; nothing else does
    # but a reset, which frees all of them.
    release = _section(
        daemon,
        "void Release(const std::string& module_key, int worker_id,",
        "std::uint32_t ExpectedWorkers(",
    )
    last = release.index("if (slot.released_mask == ExpectedWorkers(module_key)) {")
    count = release.index("if (--chunk_slot_live[chunk_slot] == 0) {", last)
    push = release.index("free_chunk_slots.push_back(chunk_slot);", count)
    assert last < count < push
    assert daemon.count("free_chunk_slots.push_back(") == 2
    free_all = _section(
        daemon, "void FreeAllChunkSlotsLocked() {", "void MarkReady(const Chunk& chunk,"
    )
    for token in (
        "free_chunk_slots.clear();",
        "chunk_slot_live.assign(count, 0);",
        "posting_window.clear();",
        "deferred_demand.fill(ChunkKey{});",
    ):
        assert token in free_all, token
    reset = _section(
        daemon, "RingResetReport ResetSlotsLocked() {", "std::uint64_t DrainForReset("
    )
    assert reset.index("module_to_slot.clear();") < reset.index("FreeAllChunkSlotsLocked();")
    # The prefetch thread resets only after draining, and reports the
    # window base as the ring tail.
    drain = loop.index("DrainForReset(&chunks);")
    base = loop.index("posting_window.front().schedule_index", drain)
    reset_call = loop.index("ResetSlotsLocked();", base)
    tail = loop.index("report.ring_tail = static_cast<int>(window_base);", reset_call)
    assert drain < base < reset_call < tail

    # The demand mark: a posted chunk needs nothing, a window entry is
    # marked once, one past the window is left for the refill.
    demand = _section(
        daemon,
        "void DemandChunkLocked(int worker_id, const ChunkKey& key) {",
        "void Release(const std::string& module_key, int worker_id,",
    )
    posted = demand.index("chunk_slot_key[chunk_slot] == key")
    window = demand.index("for (PendingChunk& pending : posting_window) {")
    hit = demand.index("++stats.demand_hits;", window)
    defer = demand.index("deferred = key;", hit)
    miss = demand.index("++stats.demand_misses;", defer)
    assert posted < window < hit < defer < miss

    summary = _section(daemon, "void WriteSummary()", "std::uint64_t CohortEpoch()")
    for key in ("demand_hits", "demand_misses", "expedited_posts", "freelist_empty_polls"):
        assert f'summary["{key}"] = counters.{key};' in summary, key


def test_daemon_summary_formats_snapshots_outside_the_request_locks():
    daemon = DAEMON.read_text()
    # Ready latency is a fixed-size histogram: no per-module sample vector
    # and no sort for the summary.
    assert "std::vector<double> ready_latency_ms" not in daemon
    assert "std::sort" not in daemon
    assert "double percentile(" not in daemon
    assert '#include "latency_histogram.h"' in daemon
    assert "Stats stats;  // under stats_mutex" in daemon

    # The summary holds only its own mutex and takes one snapshot at a time.
    summary = _section(daemon, "void WriteSummary()", "std::uint64_t CohortEpoch()")
    assert summary.count("std::lock_guard") == 1
    assert "std::lock_guard<std::mutex> summary_lock(summary_mutex);" in summary
    assert "unique_lock" not in summary
    for mutex in ("stats_mutex", "ring_mutex", "cohort_mutex"):
        assert mutex not in summary, mutex
    snapshots = [
        summary.index(call)
        for call in ("SnapshotCohort()", "SnapshotRing()", "SnapshotStats()")
    ]
    assert summary.index("summary_lock(summary_mutex)") < min(snapshots)
    assert 'summary["transport"] = transport' in summary
    for key in ('{"p50", ', '{"p95", ', '{"p99", ', '{"count", ', '{"max", '):
        assert key in summary, key

    # Each snapshot only copies under its one mutex.
    cohort = _section(daemon, "CohortSnapshot SnapshotCohort()", "RingSnapshot SnapshotRing()")
    assert cohort.count("std::lock_guard") == 1
    assert "std::lock_guard<std::mutex> lock(cohort_mutex);" in cohort
    ring = _section(daemon, "RingSnapshot SnapshotRing()", "Stats SnapshotStats()")
    assert ring.count("std::lock_guard") == 1
    assert "std::lock_guard<std::mutex> lock(ring_mutex);" in ring
    for token in ("std::to_string", "OwningWorker", "json"):
        assert token not in ring, token
    stats = _section(daemon, "Stats SnapshotStats()", "void WriteSummary()")
    assert stats.count("std::lock_guard") == 1
    assert "std::lock_guard<std::mutex> lock(stats_mutex);" in stats
    for token in ("nlohmann", "summary[", "percentile(", "ofstream", "FailureMessage"):
        assert token not in stats, token

    # A fetched chunk records its latency once, weighted by its modules.
    mark_ready = _section(
        daemon, "void MarkReady(const Chunk& chunk,", "RingResetReport ResetSlotsLocked()"
    )
    assert "stats.ready_latency.record(" in mark_ready
    assert "push_back" not in mark_ready


def test_worker_acquire_loops_on_retry_and_can_be_cancelled():
    storage = STORAGE.read_text()
    acquire = _section(
        storage,
        "Weights_Storage::acquire_remote_module(",
        "void Weights_Storage::release_module(",
    )
    lock = acquire.index("std::lock_guard<std::mutex> lock(this->daemon_mutex_);")
    flags = acquire.index(
        "request.flags =\n        wait ? 0 : batchgen::distributed_weights::kAcquireNoWait;"
    )
    loop = acquire.index("while (true) {")
    retry = acquire.index("is_retry_response(response)", loop)
    # Without wait, one round trip: a retry is answered as not ready.
    no_wait = acquire.index("if (!wait) {", retry)
    not_ready = acquire.index("return std::nullopt;", no_wait)
    stop = acquire.index("stop_requested->load()", retry)
    cancelled = acquire.index("return std::nullopt;", stop)
    validate = acquire.index("validate_response(response);", cancelled)
    lease = acquire.index("this->active_leases_[module_key] = lease;")
    assert lock < flags < loop < retry < no_wait < not_ready < stop < cancelled
    assert cancelled < validate < lease
    retry_impl = _section(
        storage, "bool is_retry_response(const Response& response)", "}  // namespace"
    )
    assert "kStatusRetry" in retry_impl
    assert "kProtocolVersion" in retry_impl

    # Ready with tensors, not ready (without wait) or cancelled (with wait).
    get_module = _section(
        storage,
        "module_fetch Weights_Storage::get_module_weights_storage(",
        "py::dict Weights_Storage::get_tensor(",
    )
    assert "module_fetch{module_fetch::Status::kReady, local->second}" in get_module
    assert "this->acquire_remote_module(module_key, stop_requested, wait)" in get_module
    assert re.search(
        r"wait \? module_fetch::Status::kCancelled\s*: module_fetch::Status::kNotReady",
        get_module,
    )
    header = STORAGE_HEADER.read_text()
    fetch = _section(header, "struct module_fetch {", "\n};")
    for status in ("kReady,", "kNotReady,", "kCancelled,"):
        assert status in fetch, status
    assert "std::unordered_map<std::string, tensor_buffer> tensors;" in fetch
    assert re.search(
        r"module_fetch get_module_weights_storage\(\s*std::string module_key,\s*"
        r"const std::atomic<bool>\* stop_requested = nullptr, bool wait = true\);",
        header,
    )
    assert re.search(
        r"acquire_remote_module\(\s*const std::string& module_key,\s*"
        r"const std::atomic<bool>\* stop_requested, bool wait\);",
        header,
    )


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
        "in_flight",
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
