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

#include "Weights_Storage.h"
#include "../Parameter_Server/posix_shm.h"
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
#include <torch/extension.h>

// Weights_Storage.cpp

Weights_Storage::Weights_Storage(int device_id)
    : device_id_(device_id) {
    
    // Initialize logger with a default level (e.g., 2 for INFO) 
    // since we don't have engine_config anymore.
    this->logger = init_logger(
        "info", // Default Log Level: INFO
        "Weights_Storage_" + std::to_string(this->device_id_));
        
    this->logger->info("Weights_Storage Instantiated on device {}.", this->device_id_);
};

Weights_Storage::~Weights_Storage() {
    // free_shared_pinned_memory(this->shm_name, this->weight_ptr_,
    //                           this->byte_size_, true);
    if(this->logger) {
        this->logger->info("Weights_Storage Destroyed.");
    }
};

/*
auto weights_map = deserialize_from_shared_memory(tensor_meta_shm_name);

*/
void Weights_Storage::Init(
    std::string& shm_name, int64_t byte_size,
    std::string& tensor_meta_shm_name, bool enable_hugetlbfs) 
{
    this->logger->info(
        "Setting CUDA device to {} for Weights_Storage initialization.",
        this->device_id_);       
    // Use the stored device_id_ member
    CUDA_CHECK(cudaSetDevice(this->device_id_));
        
    this->shm_name = shm_name;
    auto start_time = std::chrono::high_resolution_clock::now();
    this->byte_size_ = byte_size;
    auto weights_map = deserialize_from_shared_memory(tensor_meta_shm_name);
    
    this->logger->info(
        "Initializing Weights_Storage with shared memory name: {} and byte size: {}",
        shm_name, byte_size);

    // Worker process: register with CUDA for DMA access (pin_for_cuda=true)
    void* weight_ptr =
        allocate_shared_pinned_memory(shm_name, byte_size, false, enable_hugetlbfs, true);
        
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
                       
    for (auto& [module_key, tensor_map] : weights_map) {
        for (auto& [tensor_key, meta] : tensor_map) {
            this->module_weights_storage_[module_key][tensor_key] =
                tensor_buffer(static_cast<char*>(weight_ptr) + meta.offset,
                              meta.tensor_shape,
                              meta.byte_size,
                              meta.dtype);  // Pass dtype from metadata
        }
    }
}
// void Weights_Storage::Init(
//     std::string& shm_name, int64_t byte_size,
//     std::unordered_map<std::string,
//                        std::unordered_map<std::string, tensor_meta>>
//         module_weights_shm, bool enable_hugetlbfs) 
// {
//     this->logger->info(
//         "Setting CUDA device to {} for Weights_Storage initialization.",
//         this->device_id_);       
//     // Use the stored device_id_ member
//     CUDA_CHECK(cudaSetDevice(this->device_id_));
        
//     this->shm_name = shm_name;
//     auto start_time = std::chrono::high_resolution_clock::now();
//     this->byte_size_ = byte_size;
    
//     this->logger->info(
//         "Initializing Weights_Storage with shared memory name: {} and byte size: {}",
//         shm_name, byte_size);
        
//     void* weight_ptr =
//         allocate_shared_pinned_memory(shm_name, byte_size, false, enable_hugetlbfs);
        
//     // Check if weight_ptr is null
//     if (weight_ptr == nullptr) {
//         this->logger->error("Failed to allocate shared pinned memory.");
//         throw std::runtime_error("Failed to allocate shared pinned memory.");
//     }
    
//     this->weight_ptr_ = weight_ptr;
//     auto end_time = std::chrono::high_resolution_clock::now();
//     auto duration =
//         std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time)
//             .count();
            
//     this->logger->info("Shared Pinned Memory Allocation Time: {} seconds.",
//                        duration);
                       
//     for (auto& [module_key, tensor_map] : module_weights_shm) {
//         for (auto& [tensor_key, meta] : tensor_map) {
//             this->module_weights_storage_[module_key][tensor_key] =
//                 tensor_buffer(static_cast<char*>(weight_ptr) + meta.offset, 
//                               meta.tensor_shape,
//                               meta.byte_size);
//         }
//     }
// }

std::unordered_map<std::string, tensor_buffer>
Weights_Storage::get_module_weights_storage(std::string module_key) {
    /* To Facilitate Module Copy */
    if (this->module_weights_storage_.find(module_key) ==
        this->module_weights_storage_.end()) {
        this->logger->error("Module key not found in storage: {}", module_key);
        throw std::runtime_error("Module key not found in storage.");
    };

    return this->module_weights_storage_[module_key];
};

py::dict Weights_Storage::get_tensor(std::string module_key) {
    /* Get the tensor from the weights storage and return as Python dict.
     *
     * Uses stored dtype from tensor metadata instead of guessing from tensor names.
     * Supports: bfloat16, uint8, float8_e4m3fn, float32, float16
     */

    // Check if module key exists
    if (this->module_weights_storage_.find(module_key) ==
        this->module_weights_storage_.end()) {
        this->logger->error("Module key not found in storage: {}", module_key);
        throw std::runtime_error("Module key not found in storage.");
    }

    // Get module weights
    auto module_weights = this->module_weights_storage_[module_key];

    // Create Python dict to store tensors
    py::dict tensors;

    // Iterate through module weights and create tensors
    for (auto& [tensor_key, tb] : module_weights) {
        torch::Tensor tensor;

        // Use stored dtype instead of guessing from tensor name
        torch::Dtype torch_dtype;
        std::string resolved_dtype_name;
        if (tb.dtype == "bfloat16") {
            torch_dtype = torch::kBFloat16;
            resolved_dtype_name = "bfloat16";
        } else if (tb.dtype == "uint8") {
            torch_dtype = torch::kUInt8;
            resolved_dtype_name = "uint8";
        } else if (tb.dtype == "float8_e4m3fn") {
            torch_dtype = torch::kFloat8_e4m3fn;
            resolved_dtype_name = "float8_e4m3fn";
        } else if (tb.dtype == "float32") {
            torch_dtype = torch::kFloat32;
            resolved_dtype_name = "float32";
        } else if (tb.dtype == "float16") {
            torch_dtype = torch::kFloat16;
            resolved_dtype_name = "float16";
        } else {
            // Fallback to fp8 for backward compatibility
            this->logger->warn("Unknown dtype '{}' for tensor '{}', defaulting to fp8",
                              tb.dtype, tensor_key);
            torch_dtype = torch::kFloat8_e4m3fn;
            resolved_dtype_name = "float8_e4m3fn (fallback)";
        }

        // Log raw dtype from metadata for debugging (info level for visibility)
        this->logger->info("[{}] tensor '{}': raw_dtype='{}' -> torch_dtype={}",
                           module_key, tensor_key, tb.dtype, resolved_dtype_name);

        auto options = torch::TensorOptions()
            .dtype(torch_dtype)
            .device(torch::kCPU)
            .requires_grad(false)
            .memory_format(torch::MemoryFormat::Contiguous);

        tensor = torch::from_blob(
            tb.data_ptr,
            tb.tensor_shape,
            options
        );

        // Add tensor to Python dict
        tensors[tensor_key.c_str()] = tensor;
    }

    return tensors;
}