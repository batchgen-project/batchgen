// clang-format off
/* ----------------------------------------------------------------------------  *
 *  BatchGen                                                                      *
 *  Copyright (c) 2025-2026 BatchGen Team                                            *
 *                                                                               *
 *  licensed under the apache license, version 2.0 (the "license");              *
 *  you may not use this file except in compliance with the license.             *
 *                                                                               *
 *  you may obtain a copy of the license at                                      *
 *                                                                               *
 *                  http://www.apache.org/licenses/license-2.0                   *
 *                                                                               *
 *  unless required by applicable law or agreed to in writing, software          *
 *  distributed under the license is distributed on an "as is" basis,            *
 *  without warranties or conditions of any kind, either express or implied.     *
 *  see the license for the specific language governing permissions and          *
 *  limitations under the license.                                               *
 * ---------------------------------------------------------------------------- */
// clang-format on

#pragma once
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <future>
#include <map>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <queue>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <string>
#include <thread>
#include <torch/extension.h>
#include <torch/torch.h>
#include <unordered_map>
#include <vector>

#include "../data_structures.h"
#include "../utils.h"
#include "weight_slot_table.h"

using module_weight_tensor_map = std::unordered_map<std::string, torch::Tensor>;
using weight_buffers = std::vector<module_weight_tensor_map>;
class GPU_Weight_Buffer {
   public:
    using SkipCounts = WeightSkipCounts;

    struct SlotSnapshot {
        // module type -> slot count per WeightSlotState
        std::unordered_map<std::string, std::vector<int64_t>> counts;
        std::vector<WeightSlotTable::HeldSlot> held;
        std::vector<int64_t> open_passes;
        std::map<std::string, WeightSlotTable::Cursor> cursors;
        std::map<int64_t, std::map<std::string, WeightSlotTable::PassCounts>>
            pass_counts;
        bool schedule_lost = false;
    };

    GPU_Weight_Buffer(EngineConfig& engine_config, ModelConfig& model_config);
    ~GPU_Weight_Buffer();

    /* APIs */
    void Init();
    void resize_buffer();

    std::optional<
        std::pair<std::reference_wrapper<module_weight_tensor_map>, int64_t>>
    acquireEmptyBuffer(const std::string& module_name);

    void releaseBuffer(const std::string& module_name);
    void releaseBuffersAsync(const std::vector<std::string>& module_names,
                             cudaStream_t consumer_stream);

    // Waits until module_name is PRESENT in the current pass and leases it;
    // the caller owns its release. Throws at once when it can never be
    // leased in this pass, and after kStallTimeout without producer progress.
    module_weight_tensor_map get_weights(const std::string& module_name);

    /* Passes */
    int64_t begin_pass();
    SkipCounts cancel_pass(int64_t pass);
    // Waits until the producer is past the end of pass, none of its slots is
    // FILLING and every pending asynchronous release has completed.
    void wait_pass_drained(int64_t pass);
    WeightSlotTable::PassSummary end_pass(int64_t pass);
    // After a forward died: synchronizes the device, then reclaims every
    // pending release and drops every open pass and slot
    // (WeightSlotTable::abort). The producer must be stopped.
    WeightSlotTable::AbortCounts abort_passes(bool lose_schedule);

    /* Producer side of the slot lifecycle */
    WeightSlotTable::Production next_production(const std::string& module_type);
    // kCopy: copy into buffer_idx. kSkip: buffer_idx (if any) is abandoned;
    // release the daemon lease, then retire_skip().
    WeightProduction claim_for_copy(const std::string& module_type,
                                    int64_t buffer_idx,
                                    const std::string& module_name);
    void retire_skip(const std::string& module_type,
                     const std::string& module_name);
    void abandon(const std::string& module_type, int64_t buffer_idx);
    // Returns false when a skip released the slot as the copy landed.
    bool weights_copy_complete(const std::string& module_type,
                               const std::string& module_name,
                               int64_t buffer_idx);
    // Abandons buffer_idx if still FILLING (buffer_idx < 0: no slot), then
    // records the failure so every waiter throws it.
    void record_producer_failure(const std::string& module_type,
                                 int64_t buffer_idx,
                                 const std::string& message);
    void clear_producer_failure();
    std::optional<std::string> producer_failure() const;
    // A producer state change resets the stall clock of every waiter.
    void note_producer_progress();

    SkipCounts skip_modules(const std::vector<std::string>& module_names);
    SlotSnapshot slot_snapshot() const;

    void reset_prefill_buffer();
    void reset_decoding_buffer();
    void reset_weight_stream_profile(bool enabled);
    pybind11::dict get_weight_stream_profile();
    // Refused while a pass is open or a slot is held.
    void set_weight_copy_task(
        const std::unordered_map<std::string, std::vector<std::string>>&
            task) {
        std::lock_guard<std::mutex> lock(this->mutex_);
        this->slots_.install(task);
    }

    static constexpr std::chrono::seconds kStallTimeout{300};

   private:
    struct PendingRelease {
        cudaEvent_t event;
        std::vector<std::pair<std::string, int64_t>> slots;
    };

    void reclaimCompletedReleasesLocked();
    void synchronizePendingReleasesLocked();
    // Waits for every pending release and returns its slots to FREE.
    void drainPendingReleasesLocked();
    void throwIfProducerFailedLocked() const;
    // Counts a landing, a claim, a retired skip or a producer state change.
    void bumpProgressLocked();
    // Waits on cv_ until ready() holds. Throws the producer failure, or a
    // stall once no progress was counted for kStallTimeout.
    void waitWithoutStallLocked(std::unique_lock<std::mutex>& lock,
                                const std::string& waiting_for,
                                const std::function<bool()>& ready);

    EngineConfig& engine_config_;
    ModelConfig& model_config_;
    std::shared_ptr<spdlog::logger> logger_;

    std::unordered_map<std::string, weight_buffers> buffers_;
    /* Thread safe buffers management */
    mutable std::mutex mutex_;
    std::condition_variable cv_;

    WeightSlotTable slots_;  // every slot's state and module, under mutex_
    std::optional<std::string> producer_failure_;  // under mutex_
    uint64_t progress_epoch_ = 0;                  // under mutex_
    std::deque<PendingRelease> pending_releases_;

    std::mutex weight_profile_mutex_;
    bool weight_profile_enabled_{false};
    std::unordered_map<std::string, uint64_t> weight_profile_requests_;
    std::unordered_map<std::string, double> weight_profile_wait_seconds_;
};
