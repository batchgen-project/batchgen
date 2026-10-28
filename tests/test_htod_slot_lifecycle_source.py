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
that is not ready at the head of its queue. The daemon writes no summary
file: its status and metrics copy snapshots of its state and build the
result outside the locks the request path takes, and every daemon event is
pushed with none of the daemon's mutexes held.
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
DAEMON_HEADER = ROOT / "core/Weights_Storage/distributed_weight_daemon.h"
PROTOCOL = ROOT / "core/Weights_Storage/distributed_weights_protocol.h"
QUEUE = ROOT / "core/threadsafe_queue.h"
BATCHGEN = ROOT / "core/batchgen.cpp"
BATCHGEN_HEADER = ROOT / "core/batchgen.h"
BINDING = ROOT / "core/batchgen_Binding.cpp"
PACKAGE = ROOT / "batchgen"


def _section(source, begin, end):
    start = source.index(begin)
    return source[start:source.index(end, start)]


def _locks_held_at(section, position):
    """The mutex locks declared in section before position and still held there.

    A lock is released once the block that declared it closes, or, for a
    unique_lock, once it is unlocked between its declaration and position.
    """
    held = []
    for match in re.finditer(
        r"std::(lock_guard|unique_lock)<std::mutex> (\w+)\(", section[:position]
    ):
        depth = 0
        released = False
        for char in section[match.end():position]:
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth < 0:
                    released = True
                    break
        if (
            not released
            and match.group(1) == "unique_lock"
            and f"{match.group(2)}.unlock();" in section[match.end():position]
        ):
            released = True
        if not released:
            held.append(match.group(0))
    return held


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
    assert "std::chrono::seconds(300)" not in acquire
    # The 300 s diagnostics span every slice of one pending acquire.
    assert "pending_acquires.at(worker_id)" in acquire
    assert "pending.completions_at_start" in acquire
    assert "remote fetch stalled: no fetch completed in 300 s" in acquire
    assert "ready modules this cohort did not consume" in acquire
    assert "oldest unacquired ready module" in acquire
    assert "OwningWorker(oldest_unacquired->module_key)" in acquire

    # A waiting acquire is reported while it waits: on the retry path, before
    # the timeout, a stall event carries what the ring holds per worker, and
    # Status() reports each worker's pending acquire and unacquired modules.
    assert retry < acquire.index("events.push(std::move(stalled));") < stalled
    assert "stalled.ready_unacquired" in acquire
    assert "stalled.oldest_unacquired_worker" in acquire
    status = _section(daemon, "DaemonStatus Status() {", "DaemonMetrics Metrics() {")
    assert "++status.ready_unacquired_by_worker[worker];" in status
    assert "status.pending_acquires.push_back(" in status


def test_daemon_status_and_metrics_copy_snapshots_outside_the_request_locks():
    daemon = DAEMON.read_text()
    # Ready latency is a fixed-size histogram: no per-module sample vector,
    # no sort and no quantile in the daemon; the consumer reads the buckets.
    assert "std::vector<double> ready_latency_ms" not in daemon
    assert "std::sort" not in daemon
    assert '#include "latency_histogram.h"' in daemon
    assert "Stats stats;  // under stats_mutex" in daemon

    # The daemon writes no periodic summary: its state is read through
    # Status(), Metrics() and the event queue.
    for token in ("WriteSummary", "summary_path", "summary_thread",
                  "summary_mutex", "swap_log", "kSwapLogEntries", "ofstream",
                  "rename(", "percentile(", "#include <iomanip>",
                  "#include <map>"):
        assert token not in daemon, token
    cohort_snapshot = _section(daemon, "struct CohortSnapshot {", "};")
    assert "json" not in cohort_snapshot
    assert "std::uint64_t completed_swaps = 0;" in cohort_snapshot
    finish = _section(daemon, "std::string FinishCohortSwap(double timeout_seconds)",
                      "void PrepareStop()")
    assert "++completed_swaps;" in finish
    # The config no longer names a summary file; a config that still carries
    # the key loads because the daemon never reads it.
    config = _section(daemon, "explicit Impl(const std::string& path)", "~Impl()")
    assert "summary" not in config

    # Each snapshot only copies under its one mutex.
    cohort = _section(daemon, "CohortSnapshot SnapshotCohort()", "RingSnapshot SnapshotRing()")
    assert cohort.count("std::lock_guard") == 1
    assert "std::lock_guard<std::mutex> lock(cohort_mutex);" in cohort
    ring = _section(daemon, "RingSnapshot SnapshotRing()", "Stats SnapshotStats()")
    assert ring.count("std::lock_guard") == 1
    assert "std::lock_guard<std::mutex> lock(ring_mutex);" in ring
    for token in ("std::to_string", "OwningWorker", "json"):
        assert token not in ring, token
    stats = _section(daemon, "Stats SnapshotStats()", "std::uint64_t CohortEpoch()")
    assert stats.count("std::lock_guard") == 1
    assert "std::lock_guard<std::mutex> lock(stats_mutex);" in stats
    for token in ("nlohmann", "percentile(", "FailureMessage"):
        assert token not in stats, token

    # A fetched chunk records its latency once, weighted by its modules.
    mark_ready = _section(
        daemon, "void MarkReady(const Chunk& chunk,", "RingResetReport ResetSlotsLocked()"
    )
    assert "stats.ready_latency.record(" in mark_ready
    assert "push_back" not in mark_ready


# Each daemon function that pushes an event, its bounds, and the kinds it
# pushes, in order.
DAEMON_PUSH_SITES = (
    ("void Start() {", "void BootstrapNetwork()",
     ("kConfig", "kNetworkReady")),
    ("void BootstrapNetwork()", "void WaitReady(double timeout_seconds)",
     ("kNetworkReady",)),
    ("void RecordFailure(", "std::string TransportFatalMessage()",
     ("kFailed",)),
    ("void RecordWorkerLoss(", "void RecordRejectedHello(",
     ("kCohortFaulted",)),
    ("void RecordRejectedHello(", "std::string CohortClosedMessage()",
     ("kHelloRejected",)),
    ("void HandleWorker(ClientConnection* connection)", "void CloseUnadmitted(",
     ("kWorkerAdmitted",)),
    ("void Acquire(const std::string& module_key, int worker_id,",
     "void Release(const std::string& module_key, int worker_id,",
     ("kAcquireStalled",)),
    ("std::uint64_t BeginCohortSwap()", "RingResetReport ResetRing(",
     ("kCohortDraining",)),
    ("std::string FinishCohortSwap(double timeout_seconds)", "void PrepareStop()",
     ("kCohortAdmitting",)),
    ("    void Stop() {", "std::vector<DaemonEvent> PollEvents(",
     ("kStopped",)),
)


def test_daemon_events_are_pushed_after_the_locks_are_released():
    daemon = DAEMON.read_text()
    assert '#include "daemon_event_queue.h"' in daemon
    assert "DaemonEventQueue events;" in daemon
    total = 0
    for begin, end, kinds in DAEMON_PUSH_SITES:
        section = _section(daemon, begin, end)
        pushes = [match.start() for match in re.finditer(r"events\.push\(", section)]
        assert len(pushes) == len(kinds), begin
        total += len(pushes)
        named = re.findall(r"DaemonEventKind::(k\w+)", section)
        assert tuple(named) == kinds, begin
        for push in pushes:
            assert _locks_held_at(section, push) == [], begin
    # No push outside the functions checked above.
    assert daemon.count("events.push(") == total

    # The lock-order comment states the queue mutex is a leaf.
    order = _section(daemon, "// Lock order, outermost first.", "std::mutex cohort_mutex;")
    assert "event push" in order
    assert "The event queue's mutex is a leaf below all of them" in order

    # Pushed where WakeWaiters (which needs no mutex held) already ran.
    for begin, end, _ in DAEMON_PUSH_SITES:
        if begin in ("void RecordFailure(", "void RecordWorkerLoss(",
                     "std::uint64_t BeginCohortSwap()"):
            section = _section(daemon, begin, end)
            assert section.index("WakeWaiters();") < section.index("events.push("), begin

    # The config event precedes every thread; the hierarchical network_ready
    # follows its store.
    start = _section(daemon, "void Start() {", "void BootstrapNetwork()")
    config = start.index("DaemonEventKind::kConfig")
    assert config < start.index("AllocateMappings();")
    assert config < start.index("std::thread(")
    assert "ConfigDetail()" in start
    detail = _section(daemon, "std::string ConfigDetail() const", "void WakeWaiters()")
    for key in ("transport", "acquire_timeout_s", "acquire_slice_ms",
                "stall_warn_after_s"):
        assert f'config["{key}"]' in detail, key

    # A failure event only for the failure the latch keeps; a fault event only
    # for the loss that faulted the cohort.
    failure = _section(daemon, "void RecordFailure(", "std::string TransportFatalMessage()")
    assert failure.index("latched = true;") < failure.index("if (latched) {")
    loss = _section(daemon, "void RecordWorkerLoss(", "void RecordRejectedHello(")
    assert loss.index("fault = cohort_fault;") < loss.index("if (!fault.empty()) {")

    # The swap report is pushed and returned as the same text.
    finish = _section(daemon, "std::string FinishCohortSwap(double timeout_seconds)",
                      "void PrepareStop()")
    assert "std::string dumped = report.dump();" in finish
    assert "return dumped;" in finish
    assert finish.count("report.dump()") == 1

    # The stopped event follows every join and precedes started = false.
    stop = _section(daemon, "    void Stop() {", "std::vector<DaemonEvent> PollEvents(")
    stopped = stop.index("DaemonEventKind::kStopped")
    assert stop.rindex(".join();") < stopped < stop.index("started.store(false);")


def test_daemon_status_and_metrics_only_copy_snapshots():
    daemon = DAEMON.read_text()
    status = _section(daemon, "DaemonStatus Status() {", "DaemonMetrics Metrics() {")
    metrics = _section(daemon, "DaemonMetrics Metrics() {", "std::uint64_t BeginCohortSwap()")
    for section in (status, metrics):
        for token in ("lock_guard", "unique_lock", "ring_mutex", "cohort_mutex",
                      "stats_mutex", "json", "percentile("):
            assert token not in section, token
    assert "const CohortSnapshot cohort = SnapshotCohort();" in status
    assert "const RingSnapshot ring = SnapshotRing();" in status
    assert "SnapshotStats()" not in status
    assert "const Stats counters = SnapshotStats();" in metrics
    assert "SnapshotCohort()" not in metrics
    assert "SnapshotRing()" not in metrics
    assert "latency.counts()" in metrics
    assert "events.dropped()" in metrics
    poll = _section(daemon, "std::vector<DaemonEvent> PollEvents(", "};\n")
    assert "return events.wait(after_seq, timeout_seconds, dropped);" in poll
    assert "mutex" not in poll

    # The overflow bucket's edge is +infinity; the rest are the histogram's.
    edges = _section(
        daemon,
        "std::vector<double> DistributedWeightDaemon::ReadyLatencyBucketEdgesMs()",
        "return edges;",
    )
    assert "LatencyHistogram::upper_edge(index)" in edges
    assert "index < LatencyHistogram::kBuckets - 1" in edges
    assert "std::numeric_limits<double>::infinity()" in edges

    header = DAEMON_HEADER.read_text()
    assert '#include "daemon_event_queue.h"' in header
    assert "nlohmann" not in header
    for declaration in (
        "DaemonStatus Status();",
        "DaemonMetrics Metrics();",
        "std::vector<DaemonEvent> PollEvents(std::uint64_t after_seq,",
        "static std::vector<double> ReadyLatencyBucketEdgesMs();",
    ):
        assert declaration in header, declaration

    binding = BINDING.read_text()
    daemon_class = _section(
        binding,
        'py::class_<DistributedWeightDaemon>(m, "DistributedWeightDaemon")',
        "py::enum_<kv::HostKVPageFormat>",
    )
    assert "return DaemonStatusDict(self.Status());" in daemon_class
    assert "return DaemonMetricsDict(self.Metrics());" in daemon_class
    poll = _section(daemon_class, '.def("poll_events",', '.def_static(')
    release = poll.index("py::gil_scoped_release release;")
    wait = poll.index("self.PollEvents(after_seq, timeout_seconds, &dropped);")
    # The GIL is back before any Python object is built.
    scope_end = poll.index("}", wait)
    build = poll.index("py::list result;")
    assert release < wait < scope_end < build
    assert "return py::make_tuple(result, dropped);" in poll
    assert 'py::arg("after_seq"), py::arg("timeout_seconds")' in poll
    assert (
        '.def_static("ready_latency_bucket_edges_ms",\n'
        "                    &DistributedWeightDaemon::ReadyLatencyBucketEdgesMs);"
    ) in daemon_class


def test_daemon_stall_warning_is_derived_from_the_timeout():
    daemon = DAEMON.read_text()
    assert "constexpr auto kStallWarnAfter = kAcquireTimeout / 10;" in daemon
    assert "static_assert(kStallWarnAfter > kAcquireAbandoned," in daemon
    assert daemon.index("constexpr auto kAcquireTimeout") < daemon.index(
        "constexpr auto kStallWarnAfter"
    )
    assert "std::chrono::seconds(30)" not in daemon
    pending = _section(daemon, "struct PendingAcquire {", "};")
    assert "Clock::time_point next_warn;" in pending

    acquire = _section(
        daemon,
        "void Acquire(const std::string& module_key, int worker_id,",
        "void Release(const std::string& module_key, int worker_id,",
    )
    # A fresh pending acquire schedules its first warning.
    fresh = acquire.index("pending.begin = now;")
    scheduled = acquire.index("pending.next_warn = now + kStallWarnAfter;")
    assert fresh < scheduled < acquire.index("const auto ready = [&]() {")
    # The warning is on the shared retry path, after the slice or the try,
    # so both count toward the same pending acquire.
    evaluate = acquire.index("ring_cv.wait_for(lock, kAcquireSlice, ready);")
    retry = acquire.index("response->status = kStatusRetry;")
    due = acquire.index("if (pending.last_poll < pending.next_warn) {")
    advance = acquire.index("pending.next_warn += kStallWarnAfter;")
    unlock = acquire.index("lock.unlock();", advance)
    push = acquire.index("events.push(std::move(stalled));")
    timeout = acquire.index("remote fetch stalled")
    assert evaluate < retry < due < advance < unlock < push < timeout
    warning = acquire[advance:push]
    for field in ("module_key", "age_s", "progressed", "fetching", "ready",
                  "ready_unacquired", "oldest_unacquired",
                  "oldest_unacquired_worker", "cohort", "message"):
        assert re.search(rf"stalled\.{field}\b", warning), field
    # It still answers retry: the warning never fails the acquire.
    assert "fail(" not in warning
    assert "RecordFailure" not in warning
    assert acquire[push:].index("return;") < acquire[push:].index("}")


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
