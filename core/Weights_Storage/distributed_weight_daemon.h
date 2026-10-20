#pragma once

#include <cstdint>
#include <memory>
#include <string>

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

   private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
