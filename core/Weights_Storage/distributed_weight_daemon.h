#pragma once

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
    void Stop();

   private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
