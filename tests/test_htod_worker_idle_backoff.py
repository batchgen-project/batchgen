"""CPU-only source regression for the HtoD_Worker idle backoff.

The worker loop used to spin a CPU core at 100% while idle. An iteration that
made no progress parks on a condition variable when no weight type has an
armed pass to produce, sleeps 50 us only while a KV copy waits for a full KV
buffer (GPU_KV_Buffer signals nothing), and otherwise waits in the weight
buffer for a free slot: on the oldest pending release's event, or on the slot
notification every slot-freeing transition and every producer wake sends. A
progressed iteration never waits.
"""

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "core/HtoD_Engine/HtoD_Engine.cu"
BUFFER = ROOT / "core/GPU_Weight_Buffer/GPU_Weight_Buffer.cpp"
BUFFER_HEADER = ROOT / "core/GPU_Weight_Buffer/GPU_Weight_Buffer.h"

IDLE_SLEEP = re.compile(
    r"std::this_thread::sleep_for\(\s*std::chrono::microseconds\(\s*50\s*\)\s*\);"
)


def _worker_body():
    source = ENGINE.read_text()
    start = source.index("void HtoD_Engine::HtoD_Worker()")
    end = source.index("void HtoD_Engine::set_weight_copy_queue", start)
    return source[start:end]


def _idle_tail():
    body = _worker_body()
    return body[body.index("if (!progressed) {"):]


def _section(source, begin, end):
    start = source.index(begin)
    return source[start:source.index(end, start)]


def test_progress_flag_and_wake_count_are_read_each_iteration():
    body = _worker_body()
    loop = body.index("while (!terminate_flag_)")
    wake = body.index("wake_seen = this->producer_wake_;", loop)
    epoch = body.index(
        "const uint64_t slot_epoch_seen = this->gpu_weight_buffer_.slot_epoch();",
        loop,
    )
    reset = body.index("bool progressed = false;", loop)
    parked = body.index("bool weights_parked = true;", loop)
    full = body.index("std::string full_ring_type;", loop)
    # The wake count and the slot epoch are read before any work is looked
    # at, so a wake or a freed slot during the iteration keeps it from
    # parking or waiting.
    assert wake < epoch < reset < body.index("on_demand_task_queue_.try_pop", loop)
    assert parked < body.index("for (auto& module_type :", loop)
    assert full < body.index("for (auto& module_type :", loop)


def test_on_demand_task_marks_progress():
    body = _worker_body()
    pop = body.index("while (on_demand_task_queue_.try_pop(task))")
    task = body.index("task();", pop)
    progressed = body.index("progressed = true;", task)
    assert progressed < body.index("if (!this->kv_copy_task_queue_.empty())", pop)


def test_kv_buffer_acquire_marks_progress():
    body = _worker_body()
    pending = body.index("if (!this->kv_copy_task_queue_.empty())")
    acquire = body.index("this->gpu_kv_buffer_.acquireEmptyBuffer()", pending)
    has_value = body.index("if (optional_buffer.has_value()) {", acquire)
    after = body[has_value + len("if (optional_buffer.has_value()) {") :]
    assert after.lstrip().startswith("progressed = true;")


def test_a_parked_type_is_skipped_without_progress():
    body = _worker_body()
    loop = body.index("for (auto& module_type :")
    peek = body.index("this->gpu_weight_buffer_.next_production(module_type)", loop)
    parked = body.index("next.kind == WeightProduction::kParked", peek)
    skipped = body.index("continue;", parked)
    unparked = body.index("weights_parked = false;", skipped)
    acquire = body.index("gpu_weight_buffer_.acquireEmptyBuffer(module_type)", loop)
    progressed = body.index("progressed = true;", acquire)
    assert peek < parked < skipped < unparked < acquire < progressed
    # A full ring is not progress; the first full type is what the idle
    # iteration waits on.
    full = body[acquire:progressed]
    assert "if (!optional_buffer.has_value()) {" in full
    assert full.index("full_ring_type = module_type;") < full.index("continue;")


def test_progress_is_only_set_on_the_three_paths():
    assert _worker_body().count("progressed = true;") == 3


def test_parked_producer_blocks_on_the_wake_cv_instead_of_sleeping():
    tail = _idle_tail()
    park = tail.index("if (weights_parked && this->kv_copy_task_queue_.empty()) {")
    wait = tail.index("this->producer_wake_cv_.wait(lock,", park)
    predicate = tail[wait:tail.index("});", wait)]
    # Terminate is in the predicate, so a stop never waits for a pass.
    assert "this->terminate_flag_" in predicate
    assert "this->producer_wake_ != wake_seen" in predicate
    assert "wait_for" not in predicate
    sleep = IDLE_SLEEP.search(tail)
    assert sleep is not None
    assert park < wait < sleep.start()
    # The parked branch leaves the iteration before the sleep.
    assert "continue;" in tail[wait:sleep.start()]


def test_every_wake_source_bumps_the_wake_count():
    engine = ENGINE.read_text()
    wake = engine[
        engine.index("void HtoD_Engine::wake_producer() {"):
        engine.index("void HtoD_Engine::blocking_copy_(")
    ]
    assert "++this->producer_wake_;" in wake
    assert "this->producer_wake_cv_.notify_all();" in wake
    # The same wake ends a full-ring wait in the weight buffer, after the
    # wake mutex is released.
    assert wake.index("this->producer_wake_cv_.notify_all();") < wake.index(
        "this->gpu_weight_buffer_.wake_producer();"
    )
    for begin, end in (
        ("void HtoD_Engine::Terminate()", "HtoD_Engine::~HtoD_Engine()"),
        ("void HtoD_Engine::stop_h2d_worker()", "// Forward declaration of CUDA kernel"),
        ("torch::Tensor HtoD_Engine::tensor_on_demand_copy(", "void HtoD_Engine::submit_to_KV_queue("),
        ("void HtoD_Engine::submit_to_KV_queue(", "void HtoD_Engine::wake_producer()"),
    ):
        section = engine[engine.index(begin):engine.index(end, engine.index(begin))]
        assert "this->wake_producer();" in section, begin
    terminate = engine[
        engine.index("void HtoD_Engine::Terminate()"):
        engine.index("HtoD_Engine::~HtoD_Engine()")
    ]
    assert terminate.index("this->terminate_flag_ = true;") < terminate.index(
        "this->wake_producer();"
    ) < terminate.index("this->HtoD_worker_.join();")
    # From ~HtoD_Engine the weight buffer may already be destroyed: only a
    # running producer is woken.
    assert terminate.index("if (this->HtoD_worker_.joinable()) {") < terminate.index(
        "this->wake_producer();"
    )

    batchgen = (ROOT / "core/batchgen.cpp").read_text()
    begin = batchgen[
        batchgen.index("int64_t BatchGen::begin_weight_pass()"):
        batchgen.index("py::dict BatchGen::end_weight_pass(")
    ]
    assert begin.index("this->gpu_weight_buffer_.begin_pass()") < begin.index(
        "this->h2d_engine_.wake_producer();"
    )


def test_the_only_sleep_is_the_full_kv_buffer_path():
    body = _worker_body()
    assert len(IDLE_SLEEP.findall(body)) == 1
    assert body.count("std::chrono::microseconds(") == 1
    assert body.count("sleep_for(") == 1
    tail = _idle_tail()
    park = tail.index("if (weights_parked && this->kv_copy_task_queue_.empty()) {")
    kv_full = tail.index("if (!this->kv_copy_task_queue_.empty()) {", park)
    sleep = IDLE_SLEEP.search(tail)
    slot_wait = tail.index("this->gpu_weight_buffer_.wait_for_free_slot(")
    # Reached only with a KV copy queued; it leaves the iteration before the
    # weight wait, outside any lock.
    assert park < kv_full < sleep.start() < slot_wait
    guarded = tail[kv_full:sleep.start()]
    assert "}" not in guarded
    after_sleep = tail[sleep.end():slot_wait]
    assert after_sleep.lstrip().startswith("continue;")


def test_a_full_weight_ring_waits_in_the_weight_buffer():
    tail = _idle_tail()
    kv_full = tail.index("if (!this->kv_copy_task_queue_.empty()) {")
    try_open = tail.index("try {", kv_full)
    call = tail.index(
        "this->gpu_weight_buffer_.wait_for_free_slot(\n"
        "                    full_ring_type, slot_epoch_seen, this->terminate_flag_);",
        try_open,
    )
    handlers = tail[call:]
    # A CUDA error from the event wait is recorded as a producer failure;
    # nothing leaves the thread.
    assert handlers.index("} catch (const std::exception& error) {") < handlers.index(
        "} catch (...) {"
    )
    assert handlers.count("this->fail_producer_(full_ring_type, -1,") == 2
    assert "throw;" not in handlers
    # The wait counts no consumer progress: the producer's state is left
    # alone and the stall clock keeps running.
    assert "set_producer_state_" not in tail
    assert "note_producer_progress" not in tail


def test_wait_for_free_slot_reclaims_then_waits_on_the_oldest_event_or_the_cv():
    buffer = BUFFER.read_text()
    wait = _section(
        buffer,
        "void GPU_Weight_Buffer::wait_for_free_slot(",
        "void GPU_Weight_Buffer::wake_producer()",
    )
    lock = wait.index("std::unique_lock<std::mutex> lock(this->mutex_);")
    loop = wait.index("while (true) {")
    reclaim = wait.index("this->reclaimCompletedReleasesLocked();", loop)
    done = wait.index(
        "if (stop || this->slot_epoch_ != seen ||\n"
        "            this->slots_.count(module_type, WeightSlotState::kFree) > 0) {",
        reclaim,
    )
    pending = wait.index("if (!this->pending_releases_.empty()) {", done)
    oldest = wait.index("this->pending_releases_.front().event;", pending)
    unlock = wait.index("lock.unlock();", oldest)
    synchronize = wait.index("cudaEventSynchronize(oldest->get());", unlock)
    relock = wait.index("lock.lock();", synchronize)
    checked = wait.index("CUDA_CHECK(status);", relock)
    cv_wait = wait.index("this->slot_cv_.wait(lock,", checked)
    assert lock < loop < reclaim < done < pending < oldest < unlock
    assert unlock < synchronize < relock < checked < cv_wait
    # The handle is copied, so the event outlives a drain that drops it.
    assert "const std::shared_ptr<CudaEventHandle> oldest =" in wait
    predicate = wait[cv_wait:wait.index("});", cv_wait)]
    assert "stop" in predicate
    assert "this->slot_epoch_ != seen" in predicate
    # Producer waits are not consumer progress, never use the consumer cv
    # and never poll.
    assert "bumpProgressLocked" not in wait
    assert "progress_epoch_" not in wait
    assert "this->cv_." not in wait
    assert "sleep_for" not in wait
    assert "wait_for(" not in wait
    assert "wait_until(" not in wait

    header = BUFFER_HEADER.read_text()
    assert "std::condition_variable slot_cv_;" in header
    assert "uint64_t slot_epoch_ = 0;  // under mutex_" in header
    assert re.search(
        r"void wait_for_free_slot\(const std::string& module_type, uint64_t seen,\s*"
        r"const std::atomic<bool>& stop\);",
        header,
    )


def test_every_slot_freeing_transition_notifies_the_producer():
    buffer = BUFFER.read_text()
    notify = "this->notifySlotsLocked();"
    helper = _section(
        buffer,
        "void GPU_Weight_Buffer::notifySlotsLocked()",
        "void GPU_Weight_Buffer::waitWithoutStallLocked(",
    )
    assert "++this->slot_epoch_;" in helper
    assert "this->slot_cv_.notify_all();" in helper
    # Not consumer progress.
    assert "progress_epoch_" not in helper
    assert "this->cv_." not in helper

    for begin, end in (
        ("void GPU_Weight_Buffer::drainPendingReleasesLocked()",
         "void GPU_Weight_Buffer::bumpProgressLocked()"),
        ("void GPU_Weight_Buffer::reclaimCompletedReleasesLocked()",
         "void GPU_Weight_Buffer::reset_weight_stream_profile"),
        ("void GPU_Weight_Buffer::Init()", "void GPU_Weight_Buffer::resize_buffer()"),
        ("void GPU_Weight_Buffer::resize_buffer()",
         "GPU_Weight_Buffer::acquireEmptyBuffer("),
        ("void GPU_Weight_Buffer::releaseBuffer(",
         "void GPU_Weight_Buffer::releaseBuffersAsync("),
        ("void GPU_Weight_Buffer::releaseBuffersAsync(",
         "module_weight_tensor_map GPU_Weight_Buffer::get_weights("),
        ("GPU_Weight_Buffer::SkipCounts GPU_Weight_Buffer::cancel_pass(",
         "void GPU_Weight_Buffer::wait_pass_drained("),
        ("WeightSlotTable::AbortCounts GPU_Weight_Buffer::abort_passes(",
         "WeightSlotTable::Production GPU_Weight_Buffer::next_production("),
        ("void GPU_Weight_Buffer::abandon(",
         "uint64_t GPU_Weight_Buffer::slot_epoch() const"),
        ("void GPU_Weight_Buffer::wake_producer()",
         "bool GPU_Weight_Buffer::weights_copy_complete("),
        ("bool GPU_Weight_Buffer::weights_copy_complete(",
         "void GPU_Weight_Buffer::record_producer_failure("),
        ("void GPU_Weight_Buffer::record_producer_failure(",
         "void GPU_Weight_Buffer::clear_producer_failure()"),
        ("GPU_Weight_Buffer::SkipCounts GPU_Weight_Buffer::skip_modules(",
         "GPU_Weight_Buffer::SlotSnapshot GPU_Weight_Buffer::slot_snapshot()"),
    ):
        assert notify in _section(buffer, begin, end), begin
    for reset in ("reset_prefill_buffer", "reset_decoding_buffer"):
        start = buffer.rindex(f"void GPU_Weight_Buffer::{reset}()")
        body = buffer[start:buffer.index("\n}\n", start)]
        assert body.index("this->slots_.reset_type(module_type, num_buffer);") < body.index(
            notify
        ), reset

    # Landing notifies only when the skip released the slot; a skip only
    # when it freed a PRESENT slot.
    landed = _section(
        buffer,
        "bool GPU_Weight_Buffer::weights_copy_complete(",
        "void GPU_Weight_Buffer::record_producer_failure(",
    )
    assert landed.index("if (!present) {") < landed.index(notify)
    skip = _section(
        buffer,
        "GPU_Weight_Buffer::SkipCounts GPU_Weight_Buffer::skip_modules(",
        "GPU_Weight_Buffer::SlotSnapshot GPU_Weight_Buffer::slot_snapshot()",
    )
    assert skip.index("if (counts.freed_present > 0) {") < skip.index(notify)
    # A queued asynchronous release notifies once it is in the deque, so the
    # producer can wait on its event.
    release = _section(
        buffer,
        "void GPU_Weight_Buffer::releaseBuffersAsync(",
        "module_weight_tensor_map GPU_Weight_Buffer::get_weights(",
    )
    assert release.index("this->pending_releases_.push_back(") < release.index(notify)
