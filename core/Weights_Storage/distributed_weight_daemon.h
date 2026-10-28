#pragma once

#include "daemon_event_queue.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// One worker's acquire that is still waiting for its module.
struct DaemonPendingAcquire {
    int worker = -1;
    std::string module_key;
    double age_s = 0.0;
};

// What the daemon is doing now, copied from its state one mutex at a time.
struct DaemonStatus {
    bool started = false;
    bool network_ready = false;
    bool failed = false;  // transport-fatal; only a cold restart clears it
    std::string failure;
    bool stopping = false;
    std::uint64_t cohort_epoch = 0;
    std::string cohort_state;
    std::uint32_t admitted_workers_mask = 0;
    bool cohort_faulted = false;
    std::string cohort_fault;
    std::uint64_t expected_disconnects = 0;
    std::uint64_t rejected_hellos = 0;
    std::string last_rejected_hello;
    std::uint64_t completed_swaps = 0;
    std::vector<DaemonPendingAcquire> pending_acquires;
    // Indexed by worker id: ready slots no worker has acquired yet, by the
    // worker expected to acquire them.
    std::vector<std::uint64_t> ready_unacquired_by_worker;
    std::uint64_t last_event_seq = 0;
};

// Monotonic counters since the daemon was created, except the high-water mark.
struct DaemonMetrics {
    std::uint64_t fetched_modules = 0;
    std::uint64_t fetched_bytes = 0;
    std::uint64_t acquire_count = 0;
    std::uint64_t release_count = 0;
    std::uint64_t cache_hits = 0;
    std::uint64_t duplicate_fetches = 0;
    std::uint64_t wait_count = 0;
    double wait_seconds = 0.0;
    std::vector<std::uint64_t> worker_wait_count;
    std::vector<double> worker_wait_seconds;
    int staging_hwm_modules = 0;
    // Post-to-ready latency of fetched modules: one count per bucket of
    // ReadyLatencyBucketEdgesMs(), plus count, sum and max.
    std::vector<std::uint64_t> ready_latency_bucket_counts;
    std::uint64_t ready_latency_count = 0;
    double ready_latency_sum_ms = 0.0;
    double ready_latency_max_ms = 0.0;
    std::uint64_t events_dropped = 0;
};

class DistributedWeightDaemon {
   public:
    explicit DistributedWeightDaemon(const std::string& config_path);
    ~DistributedWeightDaemon();

    DistributedWeightDaemon(const DistributedWeightDaemon&) = delete;
    DistributedWeightDaemon& operator=(const DistributedWeightDaemon&) =
        delete;

    void Start();
    void WaitReady(double timeout_seconds);
    // Worker cohort swap, used when the server replaces its worker pool while
    // the daemon (and its transport to the other nodes) stays up.
    //
    // CohortEpoch() is the cohort a worker must name in its hello.
    // BeginCohortSwap() runs before the current cohort is torn down: from then
    // on its disconnects are expected, its pending and new acquires fail fast,
    // and no hello is admitted. It is idempotent while a swap is open.
    // FinishCohortSwap() runs once that cohort is reaped: it joins and closes
    // every old connection, drains in-flight fetches, frees every slot and
    // lease, rewinds the prefetch schedule, clears a cohort fault, and admits
    // the next epoch. It returns a JSON report of what it released. A
    // transport failure is never cleared: Finish then throws and only a cold
    // restart recovers.
    std::uint64_t CohortEpoch();
    std::uint64_t BeginCohortSwap();
    std::string FinishCohortSwap(double timeout_seconds);
    void PrepareStop();
    void Stop();

    // Observability. Status() and Metrics() copy state one mutex at a time
    // and never wait behind a request. PollEvents() returns the retained
    // events after after_seq, waiting up to timeout_seconds (clamped to
    // DaemonEventQueue::kMaxWaitSeconds) for one to exist; *dropped is the
    // number after after_seq that were lost to overflow. The daemon must
    // outlive every PollEvents() call.
    DaemonStatus Status();
    DaemonMetrics Metrics();
    std::vector<DaemonEvent> PollEvents(std::uint64_t after_seq,
                                        double timeout_seconds,
                                        std::uint64_t* dropped);
    // The inclusive upper edge of each ready-latency bucket, in ms, in
    // bucket order. The last bucket is overflow and its edge is +infinity.
    static std::vector<double> ReadyLatencyBucketEdgesMs();

   private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
