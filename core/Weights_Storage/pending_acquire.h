#pragma once

#include <chrono>
#include <cstdint>
#include <string>

// The wait of one worker's remote-module acquire across its requests, kept
// free of torch, CUDA, pybind, UCX and JSON so its schedule can be compiled
// and tested on its own. The daemon holds one per worker connection under its
// ring mutex.

using AcquireClock = std::chrono::steady_clock;

// One acquire request waits at most one slice, so a worker can stop between
// slices; a kAcquireNoWait request waits none. The timeout spans all requests
// of one pending acquire, waiting or not.
constexpr auto kAcquireSlice = std::chrono::milliseconds(250);
constexpr auto kAcquireTimeout = std::chrono::seconds(300);
// A pending acquire not polled for this long belongs to a worker that stopped
// asking (its producer was cancelled); the next request starts a fresh wait.
constexpr auto kAcquireAbandoned = std::chrono::seconds(5);
// A pending acquire still waiting this long, and each multiple of it, pushes
// one acquire_stalled event, so at most nine warnings precede the timeout. It
// is six times kAcquireAbandoned: a wait that long is a live worker's.
constexpr auto kStallWarnAfter = kAcquireTimeout / 10;
static_assert(kStallWarnAfter > kAcquireAbandoned,
              "a stall warning is about a worker still asking");

// What a request does once it has looked for its module.
enum class AcquireStep {
    kRetry,      // not there yet: tell the worker to ask again
    kRetryWarn,  // the same, and push one acquire_stalled event
    kFinish,     // there, or timed out: the wait ends with this request
};

// Empty key: none.
struct PendingAcquire {
    std::string module_key;
    AcquireClock::time_point begin;
    AcquireClock::time_point last_poll;
    // The next acquire_stalled warning is due at or after this.
    AcquireClock::time_point next_warn;
    std::uint64_t completions_at_start = 0;

    // A request for `key` arrived at `now`. It continues the pending wait
    // when it asks for the same module within kAcquireAbandoned of the last
    // request; otherwise a fresh wait starts here.
    void on_request(const std::string& key, AcquireClock::time_point now,
                    std::uint64_t ring_completions) {
        if (module_key == key && now - last_poll <= kAcquireAbandoned) {
            return;
        }
        module_key = key;
        begin = now;
        next_warn = now + kStallWarnAfter;
        completions_at_start = ring_completions;
    }

    // The request looked at `polled` and found the module `available` or
    // not. A waiting slice and a no-wait try count alike.
    AcquireStep after_poll(bool available, AcquireClock::time_point polled) {
        last_poll = polled;
        if (available || polled - begin >= kAcquireTimeout) {
            return AcquireStep::kFinish;
        }
        if (polled < next_warn) {
            return AcquireStep::kRetry;
        }
        next_warn += kStallWarnAfter;
        return AcquireStep::kRetryWarn;
    }
};
