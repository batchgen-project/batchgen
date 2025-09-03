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

#include "spdlog/spdlog.h"
#include <memory>
#include <string>
#include <torch/extension.h>
#include <torch/torch.h>
#include <unordered_map>
#include <vector>

#include <cuda_runtime_api.h>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "../data_structures.h"
#include "../utils.h"

/*
        Instantiate by main process.
        Allocate pinned shared memory region for model parameters.
        Worker process open the shared memory object and register it with CUDA.
*/

struct tensor_meta {
    int64_t offset;
    std::vector<int64_t> tensor_shape;
    int64_t byte_size;

    // default constructor
    tensor_meta() : offset(0), tensor_shape({}), byte_size(0) {};
    // constructor
    tensor_meta(int64_t offset, std::vector<int64_t> tensor_shape,
                int64_t byte_size)
        : offset(offset), tensor_shape(tensor_shape), byte_size(byte_size) {};
};

class Parameter_Server {
   public:
    Parameter_Server();
    ~Parameter_Server();
    void Init(std::string& shm_name, std::string& tensor_meta_shm_name,
              int64_t byte_size, std::string& model_weights_path,
              std::unordered_map<std::string,
                                 std::unordered_map<std::string, std::string>>&
                  state_dict_name_map);
    std::unordered_map<std::string, torch::Tensor> get_skeleton_state_dict();
    int64_t byte_size();
    std::unordered_map<std::string,
                       std::unordered_map<std::string, tensor_meta>>
    module_weights_shm();

   private:
    /* Template */
    std::shared_ptr<spdlog::logger> logger;

    /* "attn_0" -> "o_proj" -> ptr */
    std::string shm_name;
    std::string tensor_meta_shm_name;
    void* weight_ptr_;
    int64_t byte_size_;
    std::unordered_map<std::string,
                       std::unordered_map<std::string, tensor_meta>>
        module_weights_storage_;
    std::unordered_map<std::string, torch::Tensor> skeleton_state_dict_;
    bool enable_hugetlbfs;
    void _load_cus_format_file_to_host_mem(
        const std::string& model_weights_path,
        void* weight_ptr,
        std::unordered_map<std::string,std::unordered_map<std::string, std::string>>& state_dict_name_map);
};
