// clang-format off
/* ----------------------------------------------------------------------------  *
 *  MoE-Gen                                                                      *
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
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "spdlog/spdlog.h"
#include <future>
#include <memory>
#include <string>
#include <thread>
#include <torch/extension.h>
#include <torch/torch.h>
#include <unordered_map>
#include <vector>

#include "../GPU_KV_Buffer/GPU_KV_Buffer.h"
#include "../GPU_Weight_Buffer/GPU_Weight_Buffer.h"
#include "../KV_Storage/KV_Storage.h"
#include "../Weights_Storage/Weights_Storage.h"
#include "../data_structures.h"
#include "../threadsafe_queue.h"
#include "../utils.h"

namespace py = pybind11;
class HtoD_Engine {
   public:
    HtoD_Engine(const EngineConfig& engine_config,
                const ModelConfig& model_config,
                Weights_Storage& weights_storage, KV_Storage& kv_storage,
                GPU_Weight_Buffer& gpu_weight_buffer,
                GPU_KV_Buffer& gpu_kv_buffer);
    ~HtoD_Engine();
    // void Init(const py::dict& py_weight_copy_tasks);
    void Init(std::unordered_map<std::string, std::vector<std::string>>&
                  weights_copy_tasks);
    void Start();
    void Terminate();

    torch::Tensor tensor_on_demand_copy(torch::Tensor src_tensor);
    void submit_to_KV_queue(std::vector<int64_t>& micro_batch,
                            int64_t micro_batch_idx, int64_t layer_idx,
                            int64_t byte_size);
    void clear_kv_copy_queue();
    void clear_weight_copy_queue();
    void reset_weight_copy_queue();
    void set_global_routed_experts_data_ptr(const py::dict& experts_IPC_handles);
    void cuda_enable_peer_access(int rank, int world_size);
    

   private:
    /* Template */
    bool terminate_flag_;
    const EngineConfig& engine_config_;
    const ModelConfig& model_config_;
    cudaStream_t HtoD_stream;
    std::shared_ptr<spdlog::logger> logger_;
    std::mutex mutex_;
    std::condition_variable cv_;
    py::module p2p_tensor_copy_module;
    std::unordered_map<std::string, std::unordered_map<std::string, void*>>
                       global_device_experts_ptrs_;

    /* Need to query kv storage and weights storage. */
    GPU_Weight_Buffer& gpu_weight_buffer_;
    // GPU_KV_Buffer& gpu_kv_buffer_;
    Weights_Storage& weights_storage_;
    KV_Storage& kv_storage_;
    GPU_KV_Buffer& gpu_kv_buffer_;

    /* Task Queues. */
    threadsafe_queue<std::packaged_task<void()>> on_demand_task_queue_;
    threadsafe_queue<
        std::tuple<std::vector<int64_t>, int64_t, int64_t, int64_t>>
        kv_copy_task_queue_;  // micro_batch, micro_batch_idx, layer_idx,
                              // byte_size.

    std::unordered_map<std::string, std::vector<std::string>>
        weight_copy_tasks_;
    std::unordered_map<std::string, threadsafe_queue<std::string>>
        weights_copy_task_queue_;  // module type -> queue of module names.

    std::thread HtoD_worker_;
    void HtoD_Worker();
    void blocking_copy_(void* dst, void* src, int64_t size);
};
