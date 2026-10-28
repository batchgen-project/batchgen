#pragma once

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

// Events of the node weight daemon, kept free of torch, CUDA, pybind and JSON
// so the queue can be compiled and tested on its own.
//
// The queue is bounded: it keeps the newest kCapacity events and drops the
// oldest on overflow. Every pushed event gets the next sequence number, dense
// from 1, so a consumer that remembers the last number it saw learns exactly
// how many events it missed. Its mutex is a leaf: the daemon pushes with none
// of its own mutexes held, and nothing is called under it.
enum class DaemonEventKind {
    kConfig,
    kNetworkReady,
    kFailed,
    kCohortDraining,
    kCohortAdmitting,
    kCohortFaulted,
    kWorkerAdmitted,
    kHelloRejected,
    kAcquireStalled,
    kStopped,
};

inline const char* daemon_event_kind_name(DaemonEventKind kind) {
    switch (kind) {
        case DaemonEventKind::kConfig:
            return "config";
        case DaemonEventKind::kNetworkReady:
            return "network_ready";
        case DaemonEventKind::kFailed:
            return "failed";
        case DaemonEventKind::kCohortDraining:
            return "cohort_draining";
        case DaemonEventKind::kCohortAdmitting:
            return "cohort_admitting";
        case DaemonEventKind::kCohortFaulted:
            return "cohort_faulted";
        case DaemonEventKind::kWorkerAdmitted:
            return "worker_admitted";
        case DaemonEventKind::kHelloRejected:
            return "hello_rejected";
        case DaemonEventKind::kAcquireStalled:
            return "acquire_stalled";
        case DaemonEventKind::kStopped:
            return "stopped";
    }
    return "unknown";
}

struct DaemonEvent {
    std::uint64_t seq = 0;       // set by push, dense from 1
    double wall_time_s = 0.0;    // set by push, seconds since the Unix epoch
    DaemonEventKind kind = DaemonEventKind::kConfig;
    std::uint64_t cohort = 0;    // 0: not about one cohort
    int worker = -1;             // -1: not about one worker
    std::string message;
    std::string detail;          // JSON text when the kind carries a report
    // kAcquireStalled only: the waiting acquire and the ring when it warned.
    std::string module_key;
    int fetching = 0;
    int ready = 0;
    int ready_unacquired = 0;    // ready slots no worker acquired yet
    double age_s = 0.0;          // since the pending acquire began
    bool progressed = false;     // a fetch completed since it began
    std::string oldest_unacquired;  // module key; empty when none
    int oldest_unacquired_worker = -1;
};

class DaemonEventQueue {
public:
    static constexpr std::size_t kCapacity = 256;
    // One wait is clamped to this, so a poller always returns to its caller
    // within it; a longer wait is a loop of polls.
    static constexpr double kMaxWaitSeconds = 60.0;

    // Stamps seq and wall_time_s and returns the seq.
    std::uint64_t push(DaemonEvent event) {
        std::uint64_t seq = 0;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            seq = ++last_seq_;
            event.seq = seq;
            event.wall_time_s =
                std::chrono::duration<double>(
                    std::chrono::system_clock::now().time_since_epoch())
                    .count();
            if (events_.size() == kCapacity) {
                events_.pop_front();
                ++dropped_;
            }
            events_.push_back(std::move(event));
        }
        cv_.notify_all();
        return seq;
    }

    // The retained events after after_seq, oldest first. *dropped is the
    // number of events after after_seq that were dropped before this drain.
    std::vector<DaemonEvent> drain(std::uint64_t after_seq,
                                   std::uint64_t* dropped) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return DrainLocked(after_seq, dropped);
    }

    // drain(), after waiting until an event after after_seq exists or the
    // timeout passed. A timeout at or below zero, or NaN, does not wait.
    std::vector<DaemonEvent> wait(std::uint64_t after_seq, double timeout_s,
                                  std::uint64_t* dropped) const {
        std::unique_lock<std::mutex> lock(mutex_);
        if (timeout_s > 0.0) {
            const double bounded =
                timeout_s < kMaxWaitSeconds ? timeout_s : kMaxWaitSeconds;
            cv_.wait_for(lock, std::chrono::duration<double>(bounded),
                         [&]() { return last_seq_ > after_seq; });
        }
        return DrainLocked(after_seq, dropped);
    }

    std::uint64_t last_seq() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return last_seq_;
    }

    // Events dropped on overflow since the queue was created.
    std::uint64_t dropped() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return dropped_;
    }

private:
    std::vector<DaemonEvent> DrainLocked(std::uint64_t after_seq,
                                         std::uint64_t* dropped) const {
        // Sequence numbers in the queue are dense, so the oldest retained one
        // tells how many after after_seq are gone.
        const std::uint64_t oldest =
            events_.empty() ? last_seq_ + 1 : events_.front().seq;
        if (dropped != nullptr) {
            *dropped = after_seq + 1 < oldest ? oldest - after_seq - 1 : 0;
        }
        std::vector<DaemonEvent> result;
        if (after_seq >= last_seq_) {
            return result;
        }
        const std::uint64_t skip =
            after_seq < oldest ? 0 : after_seq - oldest + 1;
        result.reserve(events_.size() - static_cast<std::size_t>(skip));
        for (auto it = events_.begin() + static_cast<std::ptrdiff_t>(skip);
             it != events_.end(); ++it) {
            result.push_back(*it);
        }
        return result;
    }

    mutable std::mutex mutex_;
    mutable std::condition_variable cv_;
    std::deque<DaemonEvent> events_;
    std::uint64_t last_seq_ = 0;
    std::uint64_t dropped_ = 0;
};
