// clang-format off
/* ----------------------------------------------------------------------------  *
 *  BatchGen                                                                      *
 *  copyright (c) EfficientMoE team 2025                                             *
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
#include <cuda_runtime.h>
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
            std::string& phase,
            cudaStream_t consumer_stream);

    module_weight_tensor_map get_weights_pinned(
            const std::string& module_name,
            cudaStream_t consumer_stream);

    void weights_copy_enqueued(const std::string& module_type,
                               const std::string& module_name,
                               int64_t buffer_idx,
                               cudaStream_t copy_stream);

    void clear_expert_buffer(int64_t layer_idx, int64_t expert_idx, std::string phase);
    void reset_prefill_buffer();
    void reset_decoding_buffer();
    void set_weight_copy_task(std::unordered_map<std::string, std::vector<std::string>> task){
        this->weight_copy_tasks_ = task;
    }

   private:
    struct PendingRelease {
        cudaEvent_t event;
        std::vector<std::pair<std::string, int64_t>> slots;
    };

    void reclaimCompletedReleasesLocked();
    void clearPendingReleasesLocked();
    void resetReadyEventsLocked(const std::string& module_type,
                                int64_t num_buffers);
    cudaEvent_t readyEventLocked(const std::string& module_type,
                                 int64_t buffer_idx) const;

    EngineConfig& engine_config_;
    ModelConfig& model_config_;
    std::shared_ptr<spdlog::logger> logger_;

    std::unordered_map<std::string, weight_buffers> buffers_;
    /* Thread safe buffers management */
    mutable std::mutex mutex_;
    std::condition_variable cv_;

    std::unordered_map<std::string, std::pair<std::string, int64_t>>
        module_in_buffers_;  // module_name -> (module_type, buffer_idx)
    std::unordered_map<std::string, std::vector<int64_t>>
        buffer_status_;  // 0: empty, 1: in use, 2: pending compute completion
    std::unordered_map<std::string, std::vector<cudaEvent_t>> ready_events_;
    std::deque<PendingRelease> pending_releases_;
    std::unordered_map<std::string, std::vector<std::string>>
        weight_copy_tasks_;
};
