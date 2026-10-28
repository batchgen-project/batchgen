"""CPU-only: the node weight daemon's acquire-wait schedule, compiled and run.

``core/Weights_Storage/pending_acquire.h`` is free of torch, CUDA, pybind, UCX
and JSON, so a small driver is built against it with the system C++ compiler
and fed synthetic clock readings. The driver checks the stall warnings (the
first exactly at kStallWarnAfter, then one per further kStallWarnAfter, nine
before the timeout, whether the worker polls in waiting slices or in fast
no-wait tries), the timeout, an available module ending the wait, and when a
request continues the pending wait or starts a fresh one (another module, or a
gap beyond kAcquireAbandoned, at its exact boundary).
"""

import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
HEADER_DIR = ROOT / "core" / "Weights_Storage"

DRIVER = r"""
#include "pending_acquire.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using std::chrono::milliseconds;
using std::chrono::nanoseconds;
using std::chrono::seconds;
using TimePoint = AcquireClock::time_point;

static int failures = 0;

#define CHECK(cond)                                                       \
    do {                                                                  \
        if (!(cond)) {                                                    \
            std::fprintf(stderr, "line %d: CHECK failed: %s\n", __LINE__, \
                         #cond);                                          \
            ++failures;                                                   \
        }                                                                 \
    } while (0)

// Far from the clock's epoch, as a real steady_clock reading is.
static const TimePoint t0 = TimePoint{} + std::chrono::hours(1000);

struct Run {
    std::vector<TimePoint> warnings;
    TimePoint finished{};
    bool finished_set = false;
    bool consecutive_warnings = false;
};

// One worker asking for `key` every `period` from t0 until the wait ends,
// or for twice the timeout if it never does; the module never arrives.
static Run poll_until_finish(PendingAcquire& pending, nanoseconds period) {
    Run run;
    AcquireStep previous = AcquireStep::kRetry;
    for (TimePoint now = t0; now - t0 <= 2 * kAcquireTimeout; now += period) {
        pending.on_request("L1.E2", now, 7);
        const AcquireStep step = pending.after_poll(false, now);
        if (step == AcquireStep::kRetryWarn) {
            run.consecutive_warnings |= previous == AcquireStep::kRetryWarn;
            run.warnings.push_back(now);
        }
        if (step == AcquireStep::kFinish) {
            run.finished = now;
            run.finished_set = true;
            return run;
        }
        previous = step;
    }
    return run;
}

static void check_full_wait(nanoseconds period) {
    PendingAcquire pending;
    const Run run = poll_until_finish(pending, period);
    CHECK(run.finished_set);
    CHECK(pending.begin == t0);
    CHECK(run.finished - t0 >= kAcquireTimeout);
    CHECK(run.finished - t0 < kAcquireTimeout + period);
    CHECK(run.warnings.size() == 9);
    CHECK(!run.consecutive_warnings);
    for (std::size_t i = 0; i < run.warnings.size(); ++i) {
        const TimePoint due = t0 + kStallWarnAfter * (i + 1);
        CHECK(run.warnings[i] >= due);
        CHECK(run.warnings[i] < due + period);
    }
}

int main() {
    static_assert(kStallWarnAfter == seconds(30), "nine warnings in 300 s");
    static_assert(kStallWarnAfter > kAcquireAbandoned, "live worker");

    // Waiting slices, fast no-wait tries, and a slow poller just inside
    // kAcquireAbandoned all warn on the same schedule.
    check_full_wait(kAcquireSlice);
    check_full_wait(milliseconds(1));
    check_full_wait(kAcquireAbandoned);

    // The first warning is due exactly kStallWarnAfter after the wait began.
    {
        PendingAcquire pending;
        pending.on_request("L1.E2", t0, 0);
        CHECK(pending.after_poll(false, t0) == AcquireStep::kRetry);
        TimePoint now = t0 + kStallWarnAfter - nanoseconds(1);
        for (TimePoint t = t0 + seconds(5); t < now; t += seconds(5)) {
            pending.on_request("L1.E2", t, 0);
            CHECK(pending.after_poll(false, t) == AcquireStep::kRetry);
        }
        pending.on_request("L1.E2", now, 0);
        CHECK(pending.after_poll(false, now) == AcquireStep::kRetry);
        now += nanoseconds(1);
        pending.on_request("L1.E2", now, 0);
        CHECK(pending.after_poll(false, now) == AcquireStep::kRetryWarn);
        CHECK(pending.next_warn == t0 + 2 * kStallWarnAfter);
        now += kAcquireSlice;
        pending.on_request("L1.E2", now, 0);
        CHECK(pending.after_poll(false, now) == AcquireStep::kRetry);
    }

    // An available module ends the wait at once, even past a warning's due
    // time; a module still missing at the timeout ends it too.
    {
        PendingAcquire pending;
        pending.on_request("L1.E2", t0, 0);
        CHECK(pending.after_poll(true, t0) == AcquireStep::kFinish);
        CHECK(pending.last_poll == t0);

        PendingAcquire late;
        late.on_request("L1.E2", t0, 0);
        for (TimePoint t = t0; t < t0 + kStallWarnAfter; t += seconds(1)) {
            late.on_request("L1.E2", t, 0);
            late.after_poll(false, t);
        }
        const TimePoint due = t0 + kStallWarnAfter;
        late.on_request("L1.E2", due, 0);
        CHECK(late.after_poll(true, due) == AcquireStep::kFinish);
    }

    // The same module asked again within kAcquireAbandoned of the last
    // request continues the wait; its begin, warning schedule and completion
    // count are kept.
    {
        PendingAcquire pending;
        pending.on_request("L1.E2", t0, 3);
        pending.after_poll(false, t0);
        const TimePoint next = t0 + kAcquireAbandoned;
        pending.on_request("L1.E2", next, 9);
        CHECK(pending.begin == t0);
        CHECK(pending.next_warn == t0 + kStallWarnAfter);
        CHECK(pending.completions_at_start == 3);
    }

    // A gap one tick beyond kAcquireAbandoned starts a fresh wait: the
    // worker had stopped asking, so its age and warnings restart.
    {
        PendingAcquire pending;
        pending.on_request("L1.E2", t0, 3);
        pending.after_poll(false, t0);
        const TimePoint next = t0 + kAcquireAbandoned + nanoseconds(1);
        pending.on_request("L1.E2", next, 9);
        CHECK(pending.begin == next);
        CHECK(pending.next_warn == next + kStallWarnAfter);
        CHECK(pending.completions_at_start == 9);
    }

    // Another module starts a fresh wait however soon it is asked for.
    {
        PendingAcquire pending;
        pending.on_request("L1.E2", t0, 3);
        pending.after_poll(false, t0);
        const TimePoint next = t0 + milliseconds(1);
        pending.on_request("L1.E3", next, 4);
        CHECK(pending.module_key == "L1.E3");
        CHECK(pending.begin == next);
        CHECK(pending.next_warn == next + kStallWarnAfter);
        CHECK(pending.completions_at_start == 4);
    }

    // A worker that polls only every kStallWarnAfter is an abandoned one:
    // each request is a fresh wait, so it is never warned about.
    {
        PendingAcquire pending;
        for (int i = 0; i < 20; ++i) {
            const TimePoint now = t0 + kStallWarnAfter * i;
            pending.on_request("L1.E2", now, 0);
            CHECK(pending.begin == now);
            CHECK(pending.after_poll(false, now) == AcquireStep::kRetry);
        }
    }

    // A default PendingAcquire (none pending) starts a wait on any request.
    {
        PendingAcquire pending;
        CHECK(pending.module_key.empty());
        pending.on_request("L0.E0", t0, 5);
        CHECK(pending.module_key == "L0.E0");
        CHECK(pending.begin == t0);
        CHECK(pending.completions_at_start == 5);
    }

    if (failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("OK\n");
    return 0;
}
"""


def test_pending_acquire_schedule_driver(tmp_path):
    compiler = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("no system C++ compiler")
    source = tmp_path / "pending_acquire_driver.cpp"
    source.write_text(DRIVER)
    binary = tmp_path / "pending_acquire_driver"
    build = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-Wall",
            "-I",
            str(HEADER_DIR),
            str(source),
            "-o",
            str(binary),
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert build.returncode == 0, build.stderr
    run = subprocess.run(
        [str(binary)], capture_output=True, text=True, timeout=60
    )
    assert run.returncode == 0, run.stdout + run.stderr
    assert run.stdout.strip() == "OK"


def test_pending_acquire_header_is_free_of_runtime_dependencies():
    header = (HEADER_DIR / "pending_acquire.h").read_text()
    includes = [
        line for line in header.splitlines() if line.startswith("#include")
    ]
    assert includes
    for line in includes:
        lowered = line.lower()
        for dependency in ("torch", "cuda", "pybind", "nlohmann", "ucp"):
            assert dependency not in lowered


def test_daemon_acquire_uses_the_tested_schedule():
    daemon = (HEADER_DIR / "distributed_weight_daemon.cpp").read_text()
    assert '#include "pending_acquire.h"' in daemon
    assert "pending.on_request(module_key, now, ring_completions);" in daemon
    assert "pending.after_poll(available, Clock::now())" in daemon
    # The schedule lives in the header only.
    for name in ("kStallWarnAfter =", "kAcquireAbandoned =", "struct PendingAcquire"):
        assert name not in daemon
    assert "next_warn" not in daemon
