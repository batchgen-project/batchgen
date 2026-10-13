"""CPU-only source regression for the HtoD_Worker idle-pass backoff.

The worker loop used to spin a CPU core at 100% while idle. It must sleep
50 us only on a pass that made no progress, never on a progressed pass.
"""

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "core/HtoD_Engine/HtoD_Engine.cu"

IDLE_SLEEP = re.compile(
    r"if\s*\(\s*!\s*progressed\s*\)\s*\{\s*"
    r"std::this_thread::sleep_for\(\s*std::chrono::microseconds\(\s*50\s*\)\s*\);\s*\}"
)


def _worker_body():
    source = ENGINE.read_text()
    start = source.index("void HtoD_Engine::HtoD_Worker()")
    end = source.index("void HtoD_Engine::set_weight_copy_queue", start)
    return source[start:end]


def test_progress_flag_reset_each_pass():
    body = _worker_body()
    loop = body.index("while (!terminate_flag_)")
    reset = body.index("bool progressed = false;", loop)
    assert reset < body.index("on_demand_task_queue_.try_pop", loop)


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


def test_weight_buffer_acquire_marks_progress_for_nonempty_queue():
    body = _worker_body()
    loop = body.index("for (auto& module_type :")
    empty_skip = body.index("weights_copy_task_queue_[module_type].empty()", loop)
    acquire = body.index("gpu_weight_buffer_.acquireEmptyBuffer(module_type)", loop)
    assert empty_skip < acquire
    has_value = body.index("if (optional_buffer.has_value()) {", acquire)
    after = body[has_value + len("if (optional_buffer.has_value()) {") :]
    assert after.lstrip().startswith("progressed = true;")


def test_progress_is_only_set_on_the_three_paths():
    assert _worker_body().count("progressed = true;") == 3


def test_idle_sleep_is_50us_after_module_loop_at_loop_tail():
    body = _worker_body()
    matches = list(IDLE_SLEEP.finditer(body))
    assert len(matches) == 1, "expected exactly one guarded 50 us idle sleep"
    sleep = matches[0]

    loop = body.index("while (!terminate_flag_)")
    module_loop = body.index("for (auto& module_type :", loop)

    assert module_loop < sleep.start()
    assert body.rfind("weights_copy_complete(", module_loop, sleep.start()) >= 0
    # The guarded sleep is the final statement of the pass. Only the worker
    # loop and function closers follow it, so it cannot be inside a lock scope.
    assert re.fullmatch(r"\s*}\s*};?\s*", body[sleep.end() :])


def test_no_unguarded_microsecond_sleep_in_worker():
    body = _worker_body()
    unguarded = body.count("std::chrono::microseconds(") - len(
        IDLE_SLEEP.findall(body)
    )
    assert unguarded == 0
