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

#include "spdlog/spdlog.h"
#include <memory>
#include <string>
#include <torch/extension.h>
#include <torch/torch.h>
#include <unordered_map>
#include <vector>

#include "../Parameter_Server/Parameter_Server.h"
#include "../Parameter_Server/posix_shm.h"
#include "../data_structures.h"
#include "../utils.h"
#include "Weights_Storage.h"


#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

namespace py = pybind11;

Weights_Storage::Weights_Storage(const EngineConfig& engine_config,
                                 const ModelConfig& model_config)
    : engine_config_(engine_config), model_config_(model_config) {
    // std::cerr << "Weights_Storage Instantiated." << std::endl;
    this->logger = init_logger(
        this->engine_config_.basic_config.log_level,
        "Weights_Storage_" +
            std::to_string(this->engine_config_.basic_config.device));
    this->logger->info("Weights_Storage Instantiated.");
};

Weights_Storage::~Weights_Storage() {
    // free_shared_pinned_memory(this->shm_name, this->weight_ptr_,
    //                           this->byte_size_, true);
    this->logger->info("Weights_Storage Destroyed.");
};

void Weights_Storage::Init(
    std::string& shm_name, int64_t byte_size,
    std::unordered_map<std::string,
                       std::unordered_map<std::string, tensor_meta>>
        module_weights_shm) {
    CUDA_CHECK(cudaSetDevice(
        this->engine_config_.basic_config.device));
    this->shm_name = shm_name;
    auto start_time = std::chrono::high_resolution_clock::now();
    this->byte_size_ = byte_size;
    this->logger->info(
        "Initializing Weights_Storage with shared memory name: {} and byte size: {}",
        shm_name, byte_size);
    void* weight_ptr =
        allocate_shared_pinned_memory(shm_name, byte_size, false);
    // Check if weight_ptr is null
    if (weight_ptr == nullptr) {
        this->logger->error("Failed to allocate shared pinned memory.");
        throw std::runtime_error("Failed to allocate shared pinned memory.");
    }
    this->weight_ptr_ = weight_ptr;
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time)
            .count();
    this->logger->info("Shared Pinned Memory Allocation Time: {} seconds.",
                       duration);
    for (auto& [module_key, tensor_map] : module_weights_shm) {
        for (auto& [tensor_key, meta] : tensor_map) {
            // char* tensor_ptr = static_cast<char*>(weight_ptr) +
            // static_cast<size_t>(meta.offset);
            this->module_weights_storage_[module_key][tensor_key] =
                tensor_buffer(weight_ptr + meta.offset, meta.tensor_shape,
                              meta.byte_size);
        }
    }
}

std::unordered_map<std::string, tensor_buffer>
Weights_Storage::get_module_weights_storage(std::string module_key) {
    /* To Facilitate Module Copy */
    if (this->module_weights_storage_.find(module_key) ==
        this->module_weights_storage_.end()) {
        this->logger->error("Module key not found in storage.");
        throw std::runtime_error("Module key not found in storage.");
    };

    return this->module_weights_storage_[module_key];
};

// std::unordered_map<std::string, torch::Tensor> Weights_Storage::get_tensor(std::string module_key){
//     /* Get the tensor from the weights storage. */
//     if (this->module_weights_storage_.find(module_key) ==
//         this->module_weights_storage_.end()) {
//         this->logger->error("Module key not found in storage.");
//         throw std::runtime_error("Module key not found in storage.");
//     };
//     auto module_weights = this->module_weights_storage_[module_key];
//     auto bf16_option = torch::TensorOptions()
//         .dtype(torch::kBFloat16)
//         .device(torch::kCPU)
//         .requires_grad(false)
//         .memory_format(torch::MemoryFormat::Contiguous);
//     auto fp8_option = torch::TensorOptions()
//         .dtype(torch::kFloat8_e4m3fn)
//         .device(torch::kCPU)
//         .requires_grad(false)
//         .memory_format(torch::MemoryFormat::Contiguous);
//     std::unordered_map<std::string, torch::Tensor> tensors;
//     for (auto& [tensor_key, tensor_buffer] : module_weights) {
//         // If "norm" in module_key, use bf16 dtype.
//         if (tensor_key.find("norm") != std::string::npos) {
//             auto tensor = torch::from_blob(
//                 tensor_buffer.data_ptr, tensor_buffer.tensor_shape,
//                 // [buffer_copy = tensor_buffer](void* ptr) {},
//                 bf16_option);
//             tensors[tensor_key] = tensor;
//         }
//         else{
//             auto tensor = torch::from_blob(
//                 tensor_buffer.data_ptr, tensor_buffer.tensor_shape,
//                 // [buffer_copy = tensor_buffer](void* ptr) {},
//                 fp8_option);
//             tensors[tensor_key] = tensor;
//         }
//     }
//     return tensors;
// }

py::dict Weights_Storage::get_tensor(std::string module_key) {
    /* Get the tensor from the weights storage and return as Python dict. */
    
    // Check if module key exists
    if (this->module_weights_storage_.find(module_key) ==
        this->module_weights_storage_.end()) {
        this->logger->error("Module key not found in storage.");
        throw std::runtime_error("Module key not found in storage.");
    }
    
    // Get module weights
    auto module_weights = this->module_weights_storage_[module_key];
    
    // Define tensor options
    auto bf16_option = torch::TensorOptions()
        .dtype(torch::kBFloat16)
        .device(torch::kCPU)
        .requires_grad(false)
        .memory_format(torch::MemoryFormat::Contiguous);
        
    auto fp8_option = torch::TensorOptions()
        .dtype(torch::kFloat8_e4m3fn)
        .device(torch::kCPU)
        .requires_grad(false)
        .memory_format(torch::MemoryFormat::Contiguous);
    
    // Create Python dict to store tensors
    py::dict tensors;
    
    // Iterate through module weights and create tensors
    for (auto& [tensor_key, tensor_buffer] : module_weights) {
        torch::Tensor tensor;
        
        // If "norm" in tensor_key, use bf16 dtype
        if (tensor_key.find("norm") != std::string::npos) {
            tensor = torch::from_blob(
                tensor_buffer.data_ptr, 
                tensor_buffer.tensor_shape,
                bf16_option
            );
        } else {
            tensor = torch::from_blob(
                tensor_buffer.data_ptr, 
                tensor_buffer.tensor_shape,
                fp8_option
            );
        }
        
        // Add tensor to Python dict
        // PyTorch tensors are automatically converted to Python tensors
        tensors[tensor_key.c_str()] = tensor;
    }
    
    return tensors;
}

