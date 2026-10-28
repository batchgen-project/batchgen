"""CPU-only: the node weight daemon's event queue, compiled and run.

``core/Weights_Storage/daemon_event_queue.h`` is free of torch, CUDA, pybind
and JSON, so a small driver is built against it with the system C++ compiler.
The driver checks the dense sequence numbers, drains after a sequence number,
the overflow that drops the oldest events and reports the exact gap, a wait
that returns early on a push and one that returns on its timeout, and that a
drained event is a copy of the pushed one.
"""

import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
HEADER_DIR = ROOT / "core" / "Weights_Storage"

DRIVER = r"""
#include "daemon_event_queue.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <thread>
#include <vector>

using Q = DaemonEventQueue;
using SteadyClock = std::chrono::steady_clock;

static int failures = 0;

#define CHECK(cond)                                                       \
    do {                                                                  \
        if (!(cond)) {                                                    \
            std::fprintf(stderr, "line %d: CHECK failed: %s\n", __LINE__, \
                         #cond);                                          \
            ++failures;                                                   \
        }                                                                 \
    } while (0)

static DaemonEvent numbered(std::uint64_t index) {
    DaemonEvent event;
    event.kind = DaemonEventKind::kWorkerAdmitted;
    event.cohort = 1;
    event.worker = static_cast<int>(index % 8);
    event.message = "event " + std::to_string(index);
    return event;
}

static double seconds_since(SteadyClock::time_point begin) {
    return std::chrono::duration<double>(SteadyClock::now() - begin).count();
}

static void empty() {
    Q queue;
    std::uint64_t dropped = 99;
    CHECK(queue.drain(0, &dropped).empty());
    CHECK(dropped == 0);
    CHECK(queue.last_seq() == 0);
    CHECK(queue.dropped() == 0);
    // A drain without a gap counter is allowed.
    CHECK(queue.drain(0, nullptr).empty());
}

static void dense_sequence_from_one() {
    Q queue;
    for (std::uint64_t index = 1; index <= 10; ++index) {
        CHECK(queue.push(numbered(index)) == index);
    }
    CHECK(queue.last_seq() == 10);
    std::uint64_t dropped = 99;
    const std::vector<DaemonEvent> all = queue.drain(0, &dropped);
    CHECK(dropped == 0);
    CHECK(all.size() == 10);
    for (std::size_t index = 0; index < all.size(); ++index) {
        CHECK(all[index].seq == index + 1);
        CHECK(all[index].wall_time_s > 0.0);
        if (index > 0) {
            CHECK(all[index].wall_time_s >= all[index - 1].wall_time_s);
        }
    }
}

static void drain_after_seq() {
    Q queue;
    for (std::uint64_t index = 1; index <= 10; ++index) {
        queue.push(numbered(index));
    }
    std::uint64_t dropped = 99;
    const std::vector<DaemonEvent> later = queue.drain(6, &dropped);
    CHECK(dropped == 0);
    CHECK(later.size() == 4);
    CHECK(later.front().seq == 7);
    CHECK(later.back().seq == 10);
    CHECK(later.front().message == "event 7");
    // Caught up, or ahead: nothing, and no gap.
    CHECK(queue.drain(10, &dropped).empty());
    CHECK(dropped == 0);
    CHECK(queue.drain(25, &dropped).empty());
    CHECK(dropped == 0);
    // Draining does not consume.
    CHECK(queue.drain(6, &dropped).size() == 4);
}

static void overflow_drops_the_oldest_and_reports_the_gap() {
    Q queue;
    const std::uint64_t total = Q::kCapacity + 44;
    for (std::uint64_t index = 1; index <= total; ++index) {
        CHECK(queue.push(numbered(index)) == index);
    }
    CHECK(queue.last_seq() == total);
    CHECK(queue.dropped() == 44);

    std::uint64_t dropped = 0;
    const std::vector<DaemonEvent> all = queue.drain(0, &dropped);
    CHECK(dropped == 44);
    CHECK(all.size() == Q::kCapacity);
    CHECK(all.front().seq == 45);
    CHECK(all.back().seq == total);
    for (std::size_t index = 1; index < all.size(); ++index) {
        CHECK(all[index].seq == all[index - 1].seq + 1);
    }

    // A consumer that saw seq 30 missed 31..44.
    const std::vector<DaemonEvent> from_30 = queue.drain(30, &dropped);
    CHECK(dropped == 14);
    CHECK(from_30.size() == Q::kCapacity);
    CHECK(from_30.front().seq == 45);
    // One that saw seq 44 missed nothing.
    const std::vector<DaemonEvent> from_44 = queue.drain(44, &dropped);
    CHECK(dropped == 0);
    CHECK(from_44.size() == Q::kCapacity);
    // One inside the retained range missed nothing either.
    const std::vector<DaemonEvent> from_100 = queue.drain(100, &dropped);
    CHECK(dropped == 0);
    CHECK(from_100.size() == total - 100);
    CHECK(from_100.front().seq == 101);
}

static void wait_times_out() {
    Q queue;
    std::uint64_t dropped = 99;
    const SteadyClock::time_point begin = SteadyClock::now();
    CHECK(queue.wait(0, 0.05, &dropped).empty());
    const double waited = seconds_since(begin);
    CHECK(waited >= 0.04);
    CHECK(waited < 5.0);
    CHECK(dropped == 0);

    // A zero, negative or NaN timeout does not wait.
    const SteadyClock::time_point immediate = SteadyClock::now();
    CHECK(queue.wait(0, 0.0, &dropped).empty());
    CHECK(queue.wait(0, -1.0, &dropped).empty());
    CHECK(queue.wait(0, std::numeric_limits<double>::quiet_NaN(), &dropped)
              .empty());
    CHECK(seconds_since(immediate) < 1.0);

    // Events already after after_seq: no wait.
    queue.push(numbered(1));
    const SteadyClock::time_point ready = SteadyClock::now();
    CHECK(queue.wait(0, 30.0, &dropped).size() == 1);
    CHECK(seconds_since(ready) < 1.0);
}

static void wait_returns_early_on_push() {
    Q queue;
    queue.push(numbered(1));
    std::thread pusher([&queue]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        queue.push(numbered(2));
    });
    std::uint64_t dropped = 99;
    const SteadyClock::time_point begin = SteadyClock::now();
    // Waits for an event after seq 1, well short of its 30 s timeout.
    const std::vector<DaemonEvent> later = queue.wait(1, 30.0, &dropped);
    const double waited = seconds_since(begin);
    pusher.join();
    CHECK(later.size() == 1);
    CHECK(later.front().seq == 2);
    CHECK(dropped == 0);
    CHECK(waited < 10.0);
}

static void drained_event_is_a_copy_of_the_pushed_one() {
    Q queue;
    DaemonEvent pushed;
    pushed.kind = DaemonEventKind::kAcquireStalled;
    pushed.cohort = 3;
    pushed.worker = 5;
    pushed.message = "acquire waiting";
    pushed.detail = "{\"k\": 1}";
    pushed.module_key = "routed_expert_7_123";
    pushed.fetching = 4;
    pushed.ready = 28;
    pushed.ready_unacquired = 24;
    pushed.age_s = 30.25;
    pushed.progressed = true;
    pushed.oldest_unacquired = "routed_expert_7_8";
    pushed.oldest_unacquired_worker = 0;
    const std::uint64_t seq = queue.push(pushed);
    std::uint64_t dropped = 99;
    const std::vector<DaemonEvent> drained = queue.drain(0, &dropped);
    CHECK(drained.size() == 1);
    const DaemonEvent& event = drained.front();
    CHECK(event.seq == seq);
    CHECK(event.wall_time_s > 0.0);
    CHECK(event.kind == pushed.kind);
    CHECK(event.cohort == pushed.cohort);
    CHECK(event.worker == pushed.worker);
    CHECK(event.message == pushed.message);
    CHECK(event.detail == pushed.detail);
    CHECK(event.module_key == pushed.module_key);
    CHECK(event.fetching == pushed.fetching);
    CHECK(event.ready == pushed.ready);
    CHECK(event.ready_unacquired == pushed.ready_unacquired);
    CHECK(event.age_s == pushed.age_s);
    CHECK(event.progressed == pushed.progressed);
    CHECK(event.oldest_unacquired == pushed.oldest_unacquired);
    CHECK(event.oldest_unacquired_worker == pushed.oldest_unacquired_worker);
    // The caller's event is untouched by the push.
    CHECK(pushed.seq == 0);
}

static void kind_names() {
    CHECK(std::string(daemon_event_kind_name(DaemonEventKind::kConfig)) ==
          "config");
    CHECK(std::string(daemon_event_kind_name(
              DaemonEventKind::kNetworkReady)) == "network_ready");
    CHECK(std::string(daemon_event_kind_name(DaemonEventKind::kFailed)) ==
          "failed");
    CHECK(std::string(daemon_event_kind_name(
              DaemonEventKind::kCohortDraining)) == "cohort_draining");
    CHECK(std::string(daemon_event_kind_name(
              DaemonEventKind::kCohortAdmitting)) == "cohort_admitting");
    CHECK(std::string(daemon_event_kind_name(
              DaemonEventKind::kCohortFaulted)) == "cohort_faulted");
    CHECK(std::string(daemon_event_kind_name(
              DaemonEventKind::kWorkerAdmitted)) == "worker_admitted");
    CHECK(std::string(daemon_event_kind_name(
              DaemonEventKind::kHelloRejected)) == "hello_rejected");
    CHECK(std::string(daemon_event_kind_name(
              DaemonEventKind::kAcquireStalled)) == "acquire_stalled");
    CHECK(std::string(daemon_event_kind_name(DaemonEventKind::kStopped)) ==
          "stopped");
}

int main() {
    empty();
    dense_sequence_from_one();
    drain_after_seq();
    overflow_drops_the_oldest_and_reports_the_gap();
    wait_times_out();
    wait_returns_early_on_push();
    drained_event_is_a_copy_of_the_pushed_one();
    kind_names();
    if (failures != 0) {
        std::fprintf(stderr, "%d failures\n", failures);
        return 1;
    }
    std::printf("OK\n");
    return 0;
}
"""


def test_daemon_event_queue_sequence_overflow_and_wait(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("no system C++ compiler")
    source = tmp_path / "daemon_event_queue_driver.cpp"
    source.write_text(DRIVER)
    binary = tmp_path / "daemon_event_queue_driver"
    build = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-Wall",
            "-pthread",
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


def test_daemon_event_queue_header_is_free_of_torch_cuda_pybind_and_json():
    header = (HEADER_DIR / "daemon_event_queue.h").read_text()
    includes = [
        line for line in header.splitlines() if line.startswith("#include")
    ]
    assert includes
    for line in includes:
        lowered = line.lower()
        assert "torch" not in lowered
        assert "cuda" not in lowered
        assert "pybind" not in lowered
        assert "nlohmann" not in lowered
