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

using module_weight_tensor_map = std::unordered_map<std::string, torch::Tensor>;
using weight_buffers = std::vector<module_weight_tensor_map>;
class GPU_Weight_Buffer {
   public:
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
    module_weight_tensor_map get_weights(
            const std::string& module_name,
            std::string& phase);
    // Return an existing lease without eviction; the caller owns its release.
    module_weight_tensor_map get_weights_pinned(
            const std::string& module_name);

    void weights_copy_complete(const std::string& module_type,
                               const std::string& module_name,
                               int64_t buffer_idx);

    void clear_expert_buffer(int64_t layer_idx, int64_t expert_idx, std::string phase);
    void reset_prefill_buffer();
    void reset_decoding_buffer();
    void reset_weight_stream_profile(bool enabled);
    pybind11::dict get_weight_stream_profile();
    void set_weight_copy_task(std::unordered_map<std::string, std::vector<std::string>> task){
        this->weight_copy_tasks_ = task;
    }

    // module_flat: back every ring slot of module_type with ONE uint8 block of
    // span_bytes and expose its tensors as views at the given byte extents.
    // Rebuilds the type's slots, so every slot must be idle. Returns false
    // (after logging why) when the layout disagrees with module_shapes or the
    // slots are busy; the type then keeps its per-tensor slots.
    bool set_flat_layout(
        const std::string& module_type, int64_t span_bytes,
        const std::unordered_map<std::string, module_tensor_extent>& layout);
    // The flat block behind a slot, or an undefined tensor when the type has
    // no flat layout.
    torch::Tensor get_flat_slot(const std::string& module_type,
                                int64_t buffer_idx);

   private:
    struct PendingRelease {
        cudaEvent_t event;
        std::vector<std::pair<std::string, int64_t>> slots;
    };

    struct FlatLayout {
        int64_t span_bytes = 0;
        std::unordered_map<std::string, module_tensor_extent> tensors;
    };

    void reclaimCompletedReleasesLocked();
    void synchronizePendingReleasesLocked();
    module_weight_tensor_map make_flat_slot_(const std::string& module_type,
                                             torch::Tensor& flat);
    void build_flat_slots_(const std::string& module_type);

    EngineConfig& engine_config_;
    ModelConfig& model_config_;
    std::shared_ptr<spdlog::logger> logger_;

    std::unordered_map<std::string, weight_buffers> buffers_;
    // module_flat only. A type has an entry in both or in neither; the slot
    // tensors in buffers_ are then views into flat_buffers_[type][idx].
    std::unordered_map<std::string, FlatLayout> flat_layouts_;
    std::unordered_map<std::string, std::vector<torch::Tensor>> flat_buffers_;
    /* Thread safe buffers management */
    mutable std::mutex mutex_;
    std::condition_variable cv_;

    std::unordered_map<std::string, std::pair<std::string, int64_t>>
        module_in_buffers_;  // module_name -> (module_type, buffer_idx)
    std::unordered_map<std::string, std::vector<int64_t>>
        buffer_status_;  // 0: empty, 1: in use, 2: pending compute completion
    std::deque<PendingRelease> pending_releases_;
    std::unordered_map<std::string, std::vector<std::string>>
        weight_copy_tasks_;

    std::mutex weight_profile_mutex_;
    bool weight_profile_enabled_{false};
    std::unordered_map<std::string, uint64_t> weight_profile_requests_;
    std::unordered_map<std::string, double> weight_profile_wait_seconds_;
};
