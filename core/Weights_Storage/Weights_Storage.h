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

#include "spdlog/spdlog.h"
#include <memory>
#include <string>
#include <torch/extension.h>
#include <torch/torch.h>
#include <unordered_map>
#include <vector>

#include "../Parameter_Server/Parameter_Server.h"
#include "../data_structures.h"
#include "../utils.h"

struct tensor_buffer {
    void* data_ptr;
    std::vector<int64_t> tensor_shape;
    int64_t byte_size;

    // default constructor
    tensor_buffer() : data_ptr(nullptr), tensor_shape({}), byte_size(0) {};
    // constructor
    tensor_buffer(void* data_ptr, std::vector<int64_t> tensor_shape,
                  int64_t byte_size)
        : data_ptr(data_ptr),
          tensor_shape(tensor_shape),
          byte_size(byte_size) {};
};

class Weights_Storage {
   public:
    Weights_Storage(const EngineConfig& engine_config,
                    const ModelConfig& model_config);
    ~Weights_Storage();
    void Init(std::string& shm_name, int64_t byte_size,
              std::unordered_map<std::string,
                                 std::unordered_map<std::string, tensor_meta>>
                  module_weights_shm);
    std::unordered_map<std::string, tensor_buffer> get_module_weights_storage(
        std::string module_key);

    std::unordered_map<std::string, torch::Tensor> get_tensor(
        std::string module_key);

   private:
    /* Template */
    const EngineConfig& engine_config_;
    const ModelConfig& model_config_;
    std::shared_ptr<spdlog::logger> logger;

    std::string shm_name;
    void* weight_ptr_;
    int64_t byte_size_;
    /* "attn_0" -> "o_proj" -> ptr */
    std::unordered_map<std::string,
                       std::unordered_map<std::string, tensor_buffer>>
        module_weights_storage_;
};
