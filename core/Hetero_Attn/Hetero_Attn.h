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

/*
        Expose A function call to be called from Python.
        Need to interact with
        - the Python interpreter with pybind11
        - the internal component instances.
*/
#pragma once
#include <condition_variable>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <queue>
#include <spdlog/spdlog.h>
#include <string>
#include <thread>
#include <torch/extension.h>
#include <torch/torch.h>
#include <unordered_map>
#include <vector>

#include "../DtoH_Engine/DtoH_Engine.h"
#include "../GPU_Weight_Buffer/GPU_Weight_Buffer.h"
#include "../HtoD_Engine/HtoD_Engine.h"
#include "../KV_Storage/KV_Storage.h"
#include "../Weights_Storage/Weights_Storage.h"
#include "../data_structures.h"
#include "../utils.h"

class Hetero_Attn {
public:
    Hetero_Attn(const EngineConfig& engine_config,
                const ModelConfig& model_config, KV_Storage& kv_storage,
                GPU_KV_Buffer& gpu_kv_buffer, HtoD_Engine& h2d_engine,
                DtoH_Engine& d2h_engine);
    ~Hetero_Attn() = default;

    torch::Tensor attn(py::object& PyTorch_attn_module, int64_t layer_idx,
                       torch::Tensor& hidden_states,
                       torch::Tensor& attention_mask,
                       torch::Tensor& position_ids,
                       std::vector<std::vector<int64_t>> cur_batching_plan);

   private:
    const EngineConfig& engine_config_;
    const ModelConfig& model_config_;
    KV_Storage& kv_storage_;
    HtoD_Engine& h2d_engine_;
    DtoH_Engine& d2h_engine_;
    GPU_KV_Buffer& gpu_kv_buffer_;
    std::shared_ptr<spdlog::logger> logger_;

    int64_t attn_mode_;
    torch::Tensor _attn_mode_0(py::object& PyTorch_attn_module,
                               int64_t layer_idx, torch::Tensor& hidden_states,
                               torch::Tensor& attention_mask,
                               torch::Tensor& position_ids,
                               std::vector<int64_t> cur_batch);

    torch::Tensor _attn_mode_1(py::object& PyTorch_attn_module,
                               int64_t layer_idx, torch::Tensor& hidden_states,
                               torch::Tensor& attention_mask,
                               torch::Tensor& position_ids,
                               std::vector<std::vector<int64_t>> micro_batches);

    torch::Tensor _attn_mode_2(py::object& PyTorch_attn_module,
                               int64_t layer_idx, torch::Tensor& hidden_states,
                               torch::Tensor& attention_mask,
                               torch::Tensor& position_ids,
                               std::vector<std::vector<int64_t>> micro_batches);
        torch::Tensor _attn_mode_3(py::object& PyTorch_attn_module,
                               int64_t layer_idx, torch::Tensor& hidden_states,
                               torch::Tensor& attention_mask,
                               torch::Tensor& position_ids,
                               std::vector<std::vector<int64_t>> micro_batches);

    std::future<torch::Tensor> CPU_attn_mechanism(
        int64_t layer_idx, torch::Tensor query_states,
        torch::Tensor& key_states, torch::Tensor& value_states,
        torch::Tensor& attention_mask, std::vector<int64_t> cur_batch);
};
