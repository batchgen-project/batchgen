"""CPU-only source regression for the HtoD_Worker idle backoff.

The worker loop used to spin a CPU core at 100% while idle. An iteration that
made no progress parks on a condition variable when no weight type has an
armed pass to produce, and sleeps 50 us only when it waits on something
nothing signals (a full ring, a full KV buffer). A progressed iteration never
waits.
"""

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "core/HtoD_Engine/HtoD_Engine.cu"

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


def test_progress_flag_and_wake_count_are_read_each_iteration():
    body = _worker_body()
    loop = body.index("while (!terminate_flag_)")
    wake = body.index("wake_seen = this->producer_wake_;", loop)
    reset = body.index("bool progressed = false;", loop)
    parked = body.index("bool weights_parked = true;", loop)
    # The wake count is read before any work is looked at, so a wake that
    # arrives during the iteration keeps it from parking.
    assert wake < reset < body.index("on_demand_task_queue_.try_pop", loop)
    assert parked < body.index("for (auto& module_type :", loop)


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
    # A full ring is not progress.
    full = body[acquire:progressed]
    assert "if (!optional_buffer.has_value()) {" in full
    assert "continue;" in full


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

    batchgen = (ROOT / "core/batchgen.cpp").read_text()
    begin = batchgen[
        batchgen.index("int64_t BatchGen::begin_weight_pass()"):
        batchgen.index("py::dict BatchGen::end_weight_pass(")
    ]
    assert begin.index("this->gpu_weight_buffer_.begin_pass()") < begin.index(
        "this->h2d_engine_.wake_producer();"
    )


def test_no_unguarded_microsecond_sleep_in_worker():
    body = _worker_body()
    assert len(IDLE_SLEEP.findall(body)) == 1
    assert body.count("std::chrono::microseconds(") == 1
    # The sleep is the last statement of the idle branch, outside any lock.
    sleep = IDLE_SLEEP.search(body)
    assert re.fullmatch(r"\s*}\s*}\s*};?\s*", body[sleep.end():])
