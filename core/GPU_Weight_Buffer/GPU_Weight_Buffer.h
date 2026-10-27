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
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <future>
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
    // What the producer does with the module it reserved a slot for.
    enum class CopyClaim {
        kCopy,      // the slot is named; copy into it
        kSkipped,   // a routing skip was pending; the slot is abandoned
        kResident,  // already resident in another slot; the slot is abandoned
    };

    struct SkipCounts {
        int64_t freed_present = 0;
        int64_t release_on_landing = 0;
        int64_t pending = 0;
    };

    struct SlotSnapshot {
        // module type -> slot count per WeightSlotState
        std::unordered_map<std::string, std::vector<int64_t>> counts;
        std::vector<WeightSlotTable::HeldSlot> held;
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

    // std::shared_ptr<module_weight_tensor_map> get_weights(
    //     const std::string& module_name);
    // Both lease the slot they return; the caller owns its release.
    module_weight_tensor_map get_weights(
            const std::string& module_name,
            std::string& phase);
    // Waits without eviction.
    module_weight_tensor_map get_weights_pinned(
            const std::string& module_name);

    /* Producer side of the slot lifecycle */
    CopyClaim claim_for_copy(const std::string& module_type,
                             int64_t buffer_idx,
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

    SkipCounts skip_modules(const std::vector<std::string>& module_names);
    SlotSnapshot slot_snapshot() const;

    void clear_expert_buffer(int64_t layer_idx, int64_t expert_idx, std::string phase);
    void reset_prefill_buffer();
    void reset_decoding_buffer();
    void reset_weight_stream_profile(bool enabled);
    pybind11::dict get_weight_stream_profile();
    void set_weight_copy_task(std::unordered_map<std::string, std::vector<std::string>> task){
        std::lock_guard<std::mutex> lock(this->mutex_);
        this->slots_.install(task);
        this->weight_copy_tasks_ = task;
    }

   private:
    struct PendingRelease {
        cudaEvent_t event;
        std::vector<std::pair<std::string, int64_t>> slots;
    };

    void reclaimCompletedReleasesLocked();
    void synchronizePendingReleasesLocked();
    void throwIfProducerFailedLocked() const;

    EngineConfig& engine_config_;
    ModelConfig& model_config_;
    std::shared_ptr<spdlog::logger> logger_;

    std::unordered_map<std::string, weight_buffers> buffers_;
    /* Thread safe buffers management */
    mutable std::mutex mutex_;
    std::condition_variable cv_;

    WeightSlotTable slots_;  // every slot's state and module, under mutex_
    std::optional<std::string> producer_failure_;  // under mutex_
    std::deque<PendingRelease> pending_releases_;
    std::unordered_map<std::string, std::vector<std::string>>
        weight_copy_tasks_;

    std::mutex weight_profile_mutex_;
    bool weight_profile_enabled_{false};
    std::unordered_map<std::string, uint64_t> weight_profile_requests_;
    std::unordered_map<std::string, double> weight_profile_wait_seconds_;
};
