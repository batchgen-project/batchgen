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

class GPU_KV_Buffer {
   public:
    GPU_KV_Buffer(EngineConfig& engine_config, ModelConfig& model_config);
    ~GPU_KV_Buffer();

    /* APIs */
    void Init();
    void init_kv_buffer();

    std::optional<std::tuple<void*, void*, int64_t>> acquireEmptyBuffer();
    void releaseBuffer(int64_t layer_idx, int64_t micro_batch_idx);

    void kv_copy_complete(int64_t layer_idx, int64_t micro_batch_idx,
                          int64_t buffer_idx);

    torch::Tensor get_k(int64_t layer_idx, int64_t micro_batch_idx,
                        std::vector<int64_t> tensor_shape);
    torch::Tensor get_v(int64_t layer_idx, int64_t micro_batch_idx,
                        std::vector<int64_t> tensor_shape);

    void clear_kv_buffer();

   private:
    EngineConfig& engine_config_;
    ModelConfig& model_config_;
    std::shared_ptr<spdlog::logger> logger_;

    std::vector<void*> k_buffers_;
    std::vector<void*> v_buffers_;

    std::mutex mutex_;
    std::condition_variable cv_;

    std::vector<int64_t> buffer_status_;
    std::unordered_map<std::string, int64_t> kv_buffer_map_;
};
