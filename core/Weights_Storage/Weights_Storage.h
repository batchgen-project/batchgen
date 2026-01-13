// clang-format off
/* ----------------------------------------------------------------------------  *
 * BatchGen                                                                      *
 * copyright (c) EfficientMoE team 2025                                             *
 * *
 * licensed under the apache license, version 2.0 (the "license");              *
 * you may not use this file except in compliance with the license.             *
 * *
 * you may obtain a copy of the license at                                      *
 * *
 * http://www.apache.org/licenses/license-2.0                   *
 * *
 * unless required by applicable law or agreed to in writing, software          *
 * distributed under the license is distributed on an "as is" basis,            *
 * without warranties or conditions of any kind, either express or implied.     *
 * see the license for the specific language governing permissions and          *
 * limitations under the license.                                               *
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

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

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
    // Simplified Constructor: takes device_id directly
    Weights_Storage(int device_id);
    
    ~Weights_Storage();
    
    void Init(std::string& shm_name, int64_t byte_size,
                std::string& tensor_meta_shm_name, bool enable_hugetlbfs);
                  
    std::unordered_map<std::string, tensor_buffer> get_module_weights_storage(
        std::string module_key);

    // Returns Python Dictionary for Pybind11
    py::dict get_tensor(std::string module_key);

    // Set weight dtype for get_tensor (fp8 for DeepSeek, uint8 for GPT-OSS MXFP4)
    // Takes int8_t enum value: torch.uint8 = 0, torch.float8_e4m3fn = 25
    void set_weight_dtype(int8_t dtype) { weight_dtype_ = static_cast<c10::ScalarType>(dtype); }

   private:
    int device_id_; // Stored device ID
    std::shared_ptr<spdlog::logger> logger;

    std::string shm_name;
    void* weight_ptr_;
    int64_t byte_size_;

    // Weight dtype for packed weights (default: fp8 for DeepSeek)
    c10::ScalarType weight_dtype_ = c10::kFloat8_e4m3fn;

    /* "attn_0" -> "o_proj" -> ptr */
    std::unordered_map<std::string,
                       std::unordered_map<std::string, tensor_buffer>>
        module_weights_storage_;
};