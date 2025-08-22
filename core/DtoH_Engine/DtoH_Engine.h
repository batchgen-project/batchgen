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

class DtoH_Engine {
   public:
    DtoH_Engine(EngineConfig& engine_config);
    ~DtoH_Engine();
    void Init();
    void Terminate();

    torch::Tensor tensor_on_demand_copy(
        torch::Tensor&
            src_tensor);  // Would be submitted to the on_demand_task_queue_.
    void submit_to_queue_B(
        void* dst, void* src,
        int64_t size);  // Would be submitted to the kv_offloading_task_queue_.
    void wait_until_queue_B_empty();

    cudaStream_t DtoH_stream;

   private:
    const EngineConfig& engine_config_;
    bool terminate_flag_;
    std::shared_ptr<spdlog::logger> logger_;

    /* Task Queues. */
    threadsafe_queue<std::packaged_task<void()>> on_demand_task_queue_;
    threadsafe_queue<std::packaged_task<void()>> queue_B_;

    void blocking_copy_(void* dst, void* src, int64_t size);
    std::thread d2h_copy_worker_thread_;
    void d2h_copy_worker_thread_func_();
};
