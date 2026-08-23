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
#include <mutex>
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
    std::string dtype;  // Per-tensor dtype: "bfloat16", "uint8", "float8_e4m3fn", etc.

    // default constructor
    tensor_buffer() : data_ptr(nullptr), tensor_shape({}), byte_size(0), dtype("") {};
    // constructor
    tensor_buffer(void* data_ptr, std::vector<int64_t> tensor_shape,
                  int64_t byte_size, std::string dtype = "")
        : data_ptr(data_ptr),
          tensor_shape(tensor_shape),
          byte_size(byte_size),
          dtype(dtype) {};
};

struct distributed_tensor_meta {
    std::vector<int64_t> tensor_shape;
    int64_t byte_size;
    std::string dtype;
    uint64_t compact_offset;
    uint64_t module_offset;
};

class Weights_Storage {
   public:
    // Simplified Constructor: takes device_id directly
    Weights_Storage(int device_id);
    
    ~Weights_Storage();
    
    void Init(std::string& shm_name, int64_t byte_size,
                std::string& tensor_meta_shm_name, bool enable_hugetlbfs,
                bool enable_memfd = false, int memfd_creator_pid = -1,
                int memfd_fd = -1);

    void InitDistributed(const std::string& config_path);
                  
    std::unordered_map<std::string, tensor_buffer> get_module_weights_storage(
        std::string module_key);

    // Returns Python Dictionary for Pybind11
    py::dict get_tensor(std::string module_key);

    void release_module(const std::string& module_key);

   private:
    int device_id_; // Stored device ID
    std::shared_ptr<spdlog::logger> logger;

    std::string shm_name;
    void* weight_ptr_;
    int64_t byte_size_;
    
    /* "attn_0" -> "o_proj" -> ptr */
    std::unordered_map<std::string,
                       std::unordered_map<std::string, tensor_buffer>>
        module_weights_storage_;

    struct active_lease {
        int slot = -1;
        uint64_t generation = 0;
    };

    bool distributed_ = false;
    bool hierarchical_gdr_ = false;
    int local_node_rank_ = -1;
    int compact_fd_ = -1;
    int staging_fd_ = -1;
    int daemon_socket_ = -1;
    void* compact_ptr_ = nullptr;
    int64_t compact_bytes_ = 0;
    void* staging_ptr_ = nullptr;
    int64_t staging_bytes_ = 0;
    int64_t distributed_module_bytes_ = 0;
    std::mutex daemon_mutex_;
    std::unordered_map<std::string,
                       std::unordered_map<std::string,
                                          distributed_tensor_meta>>
        remote_module_weights_;
    std::unordered_map<std::string, active_lease> active_leases_;

    active_lease acquire_remote_module(const std::string& module_key);
};
