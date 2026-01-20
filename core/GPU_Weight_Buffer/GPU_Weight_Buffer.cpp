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

#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <torch/extension.h>
#include <torch/torch.h>
#include <unordered_map>
#include <vector>

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

#include "../data_structures.h"
#include "../utils.h"
#include "GPU_Weight_Buffer.h"

using module_weight_tensor_map = std::unordered_map<std::string, torch::Tensor>;
using weight_buffers = std::vector<module_weight_tensor_map>;

GPU_Weight_Buffer::GPU_Weight_Buffer(EngineConfig& engine_config,
                                     ModelConfig& model_config)
    : engine_config_(engine_config), model_config_(model_config) {
    this->logger_ = init_logger(
        this->engine_config_.basic_config.log_level,
        "GPU_Weight_Buffer" +
            std::to_string(this->engine_config_.basic_config.device));
    this->logger_->info("GPU_Weight_Buffer Instantiated.");
};

void GPU_Weight_Buffer::Init() {
    auto& buffer_shapes = this->engine_config_.gpu_buffer_config.module_shapes;
    auto& num_buffers =
        this->engine_config_.gpu_buffer_config
            .num_prefill_module_buffer;  // TODO: {"attn": 1, "routed_expert":
                                         // 160, "shared_expert": 1}
    for (auto& [module_type, num_buffer] : num_buffers) {
        // Per-module dtype lookup with fallback to global weight_dtype
        torch::Dtype module_dtype;
        auto dtype_it = this->engine_config_.gpu_buffer_config.weight_dtypes.find(module_type);
        if (dtype_it != this->engine_config_.gpu_buffer_config.weight_dtypes.end()) {
            module_dtype = dtype_it->second;
        } else {
            module_dtype = this->engine_config_.basic_config.weight_dtype_torch;
        }
        auto options =
            torch::TensorOptions()
                .dtype(module_dtype)
                .device(torch::kCUDA, this->engine_config_.basic_config.device)
                .requires_grad(false)
                .memory_format(torch::MemoryFormat::Contiguous);
        this->buffers_[module_type].clear();
        this->buffers_[module_type].resize(num_buffer);
        this->buffer_status_[module_type].clear();
        this->buffer_status_[module_type].resize(num_buffer, 0);
        for (auto& [buffer_name, buffer_shape] : buffer_shapes[module_type]) {
            for (int64_t buffer_idx = 0; buffer_idx < num_buffer;
                 buffer_idx++) {
                // Norm and bias tensors are always BF16, even for MXFP4 quantized modules
                if (buffer_name.find("norm") == std::string::npos &&
                    buffer_name.find("bias") == std::string::npos) {
                    this->buffers_[module_type][buffer_idx][buffer_name] =
                        torch::zeros(buffer_shape, options);
                } else {
                    auto bf16_options =
                        torch::TensorOptions()
                            .dtype(torch::kBFloat16)
                            .device(torch::kCUDA,
                                    this->engine_config_.basic_config.device)
                            .requires_grad(false)
                            .memory_format(torch::MemoryFormat::Contiguous);
                    this->buffers_[module_type][buffer_idx][buffer_name] =
                        torch::ones(buffer_shape, bf16_options);
                }
            }
        }
    }
};

void GPU_Weight_Buffer::resize_buffer() {
    auto& buffer_shapes = this->engine_config_.gpu_buffer_config.module_shapes;
    auto& num_buffers =
        this->engine_config_.gpu_buffer_config
            .num_decoding_module_buffer;  // {"attn": 1, "routed_expert": 160,
                                          // "shared_expert": 1}
    // Per-module dtype lookup for routed_expert
    torch::Dtype module_dtype;
    auto dtype_it = this->engine_config_.gpu_buffer_config.weight_dtypes.find("routed_expert");
    if (dtype_it != this->engine_config_.gpu_buffer_config.weight_dtypes.end()) {
        module_dtype = dtype_it->second;
    } else {
        module_dtype = this->engine_config_.basic_config.weight_dtype_torch;
    }
    auto options =
        torch::TensorOptions()
            .dtype(module_dtype)
            .device(torch::kCUDA, this->engine_config_.basic_config.device)
            .requires_grad(false)
            .memory_format(torch::MemoryFormat::Contiguous);
    {
        std::lock_guard<std::mutex> lock(this->mutex_);
        int64_t to_add_buffer = num_buffers["routed_expert"] -
                                this->buffers_["routed_expert"].size();
        for (int64_t i = 0; i < to_add_buffer; i++) {
            module_weight_tensor_map new_buffer;
            for (auto& [buffer_name, buffer_shape] :
                 buffer_shapes["routed_expert"]) {
                // Norm and bias tensors are always BF16, even for MXFP4 quantized modules
                if (buffer_name.find("norm") == std::string::npos &&
                    buffer_name.find("bias") == std::string::npos) {
                    new_buffer[buffer_name] =
                        torch::zeros(buffer_shape, options);
                } else {
                    auto bf16_options =
                        torch::TensorOptions()
                            .dtype(torch::kBFloat16)
                            .device(torch::kCUDA,
                                    this->engine_config_.basic_config.device)
                            .requires_grad(false)
                            .memory_format(torch::MemoryFormat::Contiguous);
                    new_buffer[buffer_name] =
                        torch::ones(buffer_shape, bf16_options);
                }
            }
            this->buffers_["routed_expert"].push_back(new_buffer);
            this->buffer_status_["routed_expert"].push_back(0);
        }
    }
}

std::optional<
    std::pair<std::reference_wrapper<module_weight_tensor_map>, int64_t>>
GPU_Weight_Buffer::acquireEmptyBuffer(const std::string& module_type) {
    std::lock_guard<std::mutex> lock(this->mutex_);
    for (int64_t buffer_idx = 0;
         buffer_idx < this->buffer_status_[module_type].size(); buffer_idx++) {
        if (this->buffer_status_[module_type][buffer_idx] == 0) {
            this->buffer_status_[module_type][buffer_idx] = 1;
            // this->logger_->debug("module_type {} acquire buffer_idx: {}",
            // module_type, buffer_idx);
            return std::make_pair(
                std::ref(this->buffers_[module_type][buffer_idx]), buffer_idx);
        }
    }
    // this->logger_->debug("No available buffer for module_type: {}",
    // module_type); std::this_thread::sleep_for(std::chrono::milliseconds(10));
    return std::nullopt;
};

void GPU_Weight_Buffer::releaseBuffer(const std::string& module_name) {
    std::lock_guard<std::mutex> lock(this->mutex_);
    // this->buffer_status_[module_type][buffer_idx] = 0;
    auto [module_type, buffer_idx] = this->module_in_buffers_[module_name];
    this->module_in_buffers_.erase(module_name);
    this->buffer_status_[module_type][buffer_idx] = 0;
};

module_weight_tensor_map GPU_Weight_Buffer::get_weights(
    const std::string& module_name,
    std::string& phase) 
{
    this->logger_->debug("Get weights: {}", module_name);
    
    // Start timer for timeout tracking
    auto start_time = std::chrono::steady_clock::now();
    constexpr auto timeout_duration = std::chrono::seconds(2);
    
    try {
        while (true) {
            {
                std::unique_lock<std::mutex> lock(this->mutex_);
                if (this->cv_.wait_for(
                        lock, std::chrono::milliseconds(1),
                        [this, module_name] {
                            return this->module_in_buffers_.find(module_name) !=
                                   this->module_in_buffers_.end();
                        })) {
                    auto [module_type, buffer_idx] =
                        this->module_in_buffers_[module_name];
                    return this->buffers_[module_type][buffer_idx];
                }
                this->logger_->debug("Waiting for module: {}", module_name);
            }
            
            // Check for timeout
            auto current_time = std::chrono::steady_clock::now();
            auto elapsed = current_time - start_time;
            if (elapsed >= timeout_duration) {
                // Log timeout error with buffer contents
                std::ostringstream oss;
                oss << "Map keys: ";
                
                size_t count = 0;
                for (const auto& [key, value] : this->module_in_buffers_) {
                    oss << key;
                    if (++count < this->module_in_buffers_.size()) {
                        oss << ", ";
                    }
                }
                
                this->logger_->error("Timeout reached while waiting for module: {}. Buffer contents: {}", 
                                    module_name, oss.str());
                throw std::runtime_error("Timeout reached while waiting for module: " + module_name);
            }
            
            // Check if module_name starts with "routed_expert" and has enough
            // length
            if (module_name.substr(0, 13) == "routed_expert") {
                // Find the last two underscores
                size_t last_underscore = module_name.rfind('_');
                size_t second_last_underscore =
                    module_name.rfind('_', last_underscore - 1);

                if (last_underscore != std::string::npos &&
                    second_last_underscore != std::string::npos) {
                    // Extract indices using the underscore positions
                    std::string layer_str = module_name.substr(
                        second_last_underscore + 1,
                        last_underscore - second_last_underscore - 1);
                    std::string expert_str =
                        module_name.substr(last_underscore + 1);

                    int64_t layer_idx = std::stoi(layer_str);
                    int64_t expert_idx = std::stoi(expert_str);
                    this->logger_->debug(
                        "Clearing expert buffer: layer_idx: {}, expert_idx: {}",
                        layer_idx, expert_idx);
                    this->clear_expert_buffer(layer_idx, expert_idx, phase);
                } else {
                    this->logger_->error(
                        "Invalid format in module name: '{}', expected "
                        "format: routed_expert_X_Y",
                        module_name);
                }
            }
        }
    } catch (const c10::Error& e) {
        this->logger_->debug(
            "GPU_Weight_Buffer get_weights(): CUDA/PyTorch error: {}",
            e.what());
        throw std::runtime_error(e.what());
    }
    // Catch CUDA runtime errors
    catch (const cudaError_t& err) {
        this->logger_->debug(
            "GPU_Weight_Buffer get_weights(): CUDA runtime error: {}",
            cudaGetErrorString(err));
        throw std::runtime_error(cudaGetErrorString(err));
    }
    // Catch standard C++ exceptions
    catch (const std::exception& e) {
        this->logger_->debug("GPU_Weight_Buffer get_weights() Error: {}",
                             e.what());
        throw std::runtime_error("GPU_Weight_Buffer get_weights()");
    }
    // Catch any other unexpected errors
    catch (...) {
        this->logger_->debug("GPU_Weight_Buffer get_weights()");
        throw std::runtime_error("GPU_Weight_Buffer get_weights()");
    }
};

void GPU_Weight_Buffer::weights_copy_complete(const std::string& module_type,
                                              const std::string& module_name,
                                              int64_t buffer_idx) {
    {
        std::lock_guard<std::mutex> lock(this->mutex_);
        this->module_in_buffers_[module_name] =
            std::make_pair(module_type, buffer_idx);
        this->logger_->debug("Module: {} is in buffer: {}", module_name,
                             buffer_idx);
        this->cv_.notify_all();
    }
    
};


void GPU_Weight_Buffer::clear_expert_buffer(int64_t layer_idx, int64_t expert_idx, std::string phase) {
    // Log the keys of the module_in_buffers_ in the same log msg
    std::ostringstream oss;
    oss << "Map keys: ";
    
    size_t count = 0;
    for (const auto& [key, value] : this->module_in_buffers_) {
        oss << key;
        if (++count < this->module_in_buffers_.size()) {
            oss << ", ";
        }
    }
    this->logger_->debug("clearing expert buffer: layer_idx: {}, expert_idx: {}, existing keys: {}", layer_idx, expert_idx, oss.str());

    // Get the number of expert buffers based on phase
    int64_t num_expert_buffer;
    if (phase == "prefill") {
        num_expert_buffer = this->engine_config_.gpu_buffer_config.num_prefill_module_buffer["routed_expert"];
    } else if (phase == "decode") {
        num_expert_buffer = this->engine_config_.gpu_buffer_config.num_decoding_module_buffer["routed_expert"];
    } else {
        throw std::runtime_error("Invalid phase: " + std::string(phase));
    }
    
    // Create the current expert name once
    std::string current_expert_name = "routed_expert_" + std::to_string(layer_idx) + "_" + std::to_string(expert_idx);
    
    // Find index of current expert in weight_copy_tasks_ in one pass
    const auto& expert_tasks = this->weight_copy_tasks_["routed_expert"];
    auto task_it = std::find(expert_tasks.begin(), expert_tasks.end(), current_expert_name);
    if (task_it == expert_tasks.end()) {
        return; // Expert name not found, nothing to clear
    }
    
    // Build the set of allowed expert names efficiently
    std::unordered_set<std::string> allowed_expert_names;
    allowed_expert_names.reserve(num_expert_buffer); // Preallocate for performance
    allowed_expert_names.insert(current_expert_name);
    
    size_t task_size = expert_tasks.size();
    size_t idx = std::distance(expert_tasks.begin(), task_it);
    
    // Add the next (num_expert_buffer - 1) expert names
    for (int64_t i = 1; i < num_expert_buffer; i++) {
        idx = (idx + 1) % task_size; // Wrap around more efficiently
        allowed_expert_names.insert(expert_tasks[idx]);
    }
    
    // Remove buffers in one pass, avoiding temporary storage vector
    {
        std::lock_guard<std::mutex> lock(this->mutex_);
        for (auto it = this->module_in_buffers_.begin(); it != this->module_in_buffers_.end();) {
            const auto& key = it->first;
            // Check if this is a routed expert buffer not in our allowed list
            if (key.compare(0, 13, "routed_expert") == 0 && 
                allowed_expert_names.find(key) == allowed_expert_names.end()) {
                this->logger_->debug(
                    "Clearing expert buffer: layer_idx: {}, expert_idx: {}, {} cleared",
                    layer_idx, expert_idx, key);
                auto module_type = it->second.first;  // Access through iterator
                auto buffer_idx = it->second.second;  // Access through iterator
                it = this->module_in_buffers_.erase(it); // Erase and get next iterator
                this->buffer_status_[module_type][buffer_idx] = 0; // Release buffer
            } else {
                ++it; // Move to next element
            }
        }
    }
}


// void GPU_Weight_Buffer::reset_prefill_buffer() {
//     auto& buffer_shapes = this->engine_config_.gpu_buffer_config.module_shapes;
//     auto& num_buffers =
//         this->engine_config_.gpu_buffer_config
//             .num_prefill_module_buffer;  // {"attn": 1, "routed_expert": 160,
//                                          // "shared_expert": 1}
//     auto options =
//         torch::TensorOptions()
//             .dtype(this->engine_config_.basic_config.weight_dtype_torch)
//             .device(torch::kCUDA, this->engine_config_.basic_config.device)
//             .requires_grad(false)
//             .memory_format(torch::MemoryFormat::Contiguous);
//     {
//         std::lock_guard<std::mutex> lock(this->mutex_);
//         // reset the buffer size as the prefill buffer size.
//         this->buffers_["routed_expert"].clear();
//         this->buffers_["routed_expert"].resize(num_buffers["routed_expert"]);
//         for (auto& [buffer_name, buffer_shape] :
//              buffer_shapes["routed_expert"]) {
//             for (int64_t buffer_idx = 0;
//                  buffer_idx < num_buffers["routed_expert"]; buffer_idx++) {
//                 if (buffer_name.find("norm") == std::string::npos) {
//                     this->buffers_["routed_expert"][buffer_idx][buffer_name] =
//                         torch::zeros(buffer_shape, options);
//                 } else {
//                     auto bf16_options =
//                         torch::TensorOptions()
//                             .dtype(torch::kBFloat16)
//                             .device(torch::kCUDA,
//                                     this->engine_config_.basic_config.device)
//                             .requires_grad(false)
//                             .memory_format(torch::MemoryFormat::Contiguous);
//                     this->buffers_["routed_expert"][buffer_idx][buffer_name] =
//                         torch::ones(buffer_shape, bf16_options);
//                 }
//             }
//         }
//         this->module_in_buffers_.clear();

//         // Set all buffer status to 0.
//         for (auto& [module_type, num_buffer] : num_buffers) {
//             this->buffer_status_[module_type].clear();
//             this->buffer_status_[module_type].resize(num_buffer, 0);
//         }
//         // Log the buffer status and size of each buffer.
//         for (auto& [module_type, num_buffer] : num_buffers) {
//             this->logger_->debug("Module type: {}, Number of buffer: {}",
//                                  module_type, num_buffer);
//             for (int64_t buffer_idx = 0; buffer_idx < num_buffer;
//                  buffer_idx++) {
//                 this->logger_->debug(
//                     "Buffer_idx: {}, Buffer status: {}", buffer_idx,
//                     this->buffer_status_[module_type][buffer_idx]);
//             }
//         }
//     }
// }

// void GPU_Weight_Buffer::reset_prefill_buffer() {
//     auto& buffer_shapes = this->engine_config_.gpu_buffer_config.module_shapes;
//     auto& num_buffers = this->engine_config_.gpu_buffer_config.num_prefill_module_buffer;
    
//     auto options = torch::TensorOptions()
//         .dtype(this->engine_config_.basic_config.weight_dtype_torch)
//         .device(torch::kCUDA, this->engine_config_.basic_config.device)
//         .requires_grad(false)
//         .memory_format(torch::MemoryFormat::Contiguous);
    
//     auto bf16_options = torch::TensorOptions()
//         .dtype(torch::kBFloat16)
//         .device(torch::kCUDA, this->engine_config_.basic_config.device)
//         .requires_grad(false)
//         .memory_format(torch::MemoryFormat::Contiguous);
    
//     {
//         std::lock_guard<std::mutex> lock(this->mutex_);
        
//         // Step 1: Log initial memory state
//         size_t memory_before = 0;
//         if (torch::cuda::is_available()) {
//             size_t free_bytes, total_bytes;
//             cudaSetDevice(this->engine_config_.basic_config.device);
//             cudaMemGetInfo(&free_bytes, &total_bytes);
//             memory_before = total_bytes - free_bytes;
//             this->logger_->info("Prefill reset - Memory before: {} MB", 
//                               memory_before / (1024.0 * 1024.0));
//         }
        
//         // Step 2: Explicitly release ALL existing tensors
//         for (auto& [buffer_type, buffer_vector] : this->buffers_) {
//             for (auto& module_map : buffer_vector) {
//                 for (auto& [name, tensor] : module_map) {
//                     tensor = torch::Tensor();
//                 }
//                 module_map.clear();
//             }
//             buffer_vector.clear();
//             buffer_vector.shrink_to_fit();
//         }
        
//         // Step 3: Clear module mappings
//         this->module_in_buffers_.clear();
        
//         // Step 4: Force CUDA synchronization and cleanup
//         if (torch::cuda::is_available()) {
//             // Use CUDA runtime API
//             cudaDeviceSynchronize();
            
//             // Empty the cache
//             c10::cuda::CUDACachingAllocator::emptyCache();
            
//             // Log memory after clearing
//             size_t free_bytes, total_bytes;
//             cudaMemGetInfo(&free_bytes, &total_bytes);
//             size_t memory_after_clear = total_bytes - free_bytes;
//             this->logger_->info("Prefill reset - Memory after clearing: {} MB (freed: {} MB)", 
//                               memory_after_clear / (1024.0 * 1024.0),
//                               (memory_before - memory_after_clear) / (1024.0 * 1024.0));
//         }
        
//         // Step 5: Create new prefill buffers
//         this->buffers_["routed_expert"].reserve(num_buffers["routed_expert"]);
//         this->buffers_["routed_expert"].resize(num_buffers["routed_expert"]);
        
//         // Pre-calculate norm buffer names
//         std::unordered_set<std::string> norm_buffers;
//         for (const auto& [buffer_name, buffer_shape] : buffer_shapes["routed_expert"]) {
//             if (buffer_name.find("norm") != std::string::npos) {
//                 norm_buffers.insert(buffer_name);
//             }
//         }
        
//         // Create tensors
//         for (auto& [buffer_name, buffer_shape] : buffer_shapes["routed_expert"]) {
//             bool is_norm = norm_buffers.count(buffer_name) > 0;
            
//             for (int64_t buffer_idx = 0; buffer_idx < num_buffers["routed_expert"]; buffer_idx++) {
//                 if (!is_norm) {
//                     this->buffers_["routed_expert"][buffer_idx][buffer_name] =
//                         torch::zeros(buffer_shape, options);
//                 } else {
//                     // Use bf16 for norm layers in prefill
//                     this->buffers_["routed_expert"][buffer_idx][buffer_name] =
//                         torch::ones(buffer_shape, bf16_options);
//                 }
//             }
//         }
        
//         // Step 6: Reset buffer status
//         for (auto& [module_type, num_buffer] : num_buffers) {
//             this->buffer_status_[module_type].clear();
//             this->buffer_status_[module_type].shrink_to_fit();
//             this->buffer_status_[module_type].resize(num_buffer, 0);
//         }
        
//         // Step 7: Final memory check
//         if (torch::cuda::is_available()) {
//             size_t free_bytes, total_bytes;
//             cudaMemGetInfo(&free_bytes, &total_bytes);
//             size_t memory_final = total_bytes - free_bytes;
//             this->logger_->info("Prefill reset - Memory after recreation: {} MB (net change: {} MB)", 
//                               memory_final / (1024.0 * 1024.0),
//                               (static_cast<int64_t>(memory_final) - static_cast<int64_t>(memory_before)) / (1024.0 * 1024.0));
//         }
        
//         this->logger_->info("Prefill buffer reset complete");
//     }
// }

void GPU_Weight_Buffer::reset_prefill_buffer() {
    auto& buffer_shapes = this->engine_config_.gpu_buffer_config.module_shapes;
    auto& num_buffers = this->engine_config_.gpu_buffer_config.num_prefill_module_buffer;

    // Per-module dtype lookup for routed_expert
    torch::Dtype module_dtype;
    auto dtype_it = this->engine_config_.gpu_buffer_config.weight_dtypes.find("routed_expert");
    if (dtype_it != this->engine_config_.gpu_buffer_config.weight_dtypes.end()) {
        module_dtype = dtype_it->second;
    } else {
        module_dtype = this->engine_config_.basic_config.weight_dtype_torch;
    }
    auto options = torch::TensorOptions()
        .dtype(module_dtype)
        .device(torch::kCUDA, this->engine_config_.basic_config.device)
        .requires_grad(false)
        .memory_format(torch::MemoryFormat::Contiguous);

    {
        std::lock_guard<std::mutex> lock(this->mutex_);
        
        // IMPORTANT: Only clear routed_expert buffers, not all buffers!
        // First, properly release the tensors in routed_expert
        if (this->buffers_.find("routed_expert") != this->buffers_.end()) {
            for (auto& module_map : this->buffers_["routed_expert"]) {
                for (auto& [name, tensor] : module_map) {
                    // Explicitly release tensor memory
                    tensor = torch::Tensor();
                }
                module_map.clear();
            }
            this->buffers_["routed_expert"].clear();
            
            // Optional: Force memory cleanup for routed_expert tensors only
            if (torch::cuda::is_available()) {
                cudaDeviceSynchronize();
                c10::cuda::CUDACachingAllocator::emptyCache();
            }
        }
        
        // Reset the buffer size as the prefill buffer size
        this->buffers_["routed_expert"].resize(num_buffers["routed_expert"]);
        
        // Create new tensors
        for (auto& [buffer_name, buffer_shape] : buffer_shapes["routed_expert"]) {
            for (int64_t buffer_idx = 0; buffer_idx < num_buffers["routed_expert"]; buffer_idx++) {
                if (buffer_name.find("norm") == std::string::npos) {
                    this->buffers_["routed_expert"][buffer_idx][buffer_name] =
                        torch::zeros(buffer_shape, options);
                } else {
                    // Create bf16_options inline as in original
                    auto bf16_options = torch::TensorOptions()
                        .dtype(torch::kBFloat16)
                        .device(torch::kCUDA, this->engine_config_.basic_config.device)
                        .requires_grad(false)
                        .memory_format(torch::MemoryFormat::Contiguous);
                    this->buffers_["routed_expert"][buffer_idx][buffer_name] =
                        torch::ones(buffer_shape, bf16_options);
                }
            }
        }
        
        // Clear module_in_buffers
        this->module_in_buffers_.clear();
        
        // Set all buffer status to 0 (for ALL module types, not just routed_expert)
        for (auto& [module_type, num_buffer] : num_buffers) {
            this->buffer_status_[module_type].clear();
            this->buffer_status_[module_type].resize(num_buffer, 0);
        }
        
        // Log the buffer status and size of each buffer (keep original debug level)
        for (auto& [module_type, num_buffer] : num_buffers) {
            this->logger_->debug("Module type: {}, Number of buffer: {}",
                                 module_type, num_buffer);
            for (int64_t buffer_idx = 0; buffer_idx < num_buffer; buffer_idx++) {
                this->logger_->debug("Buffer_idx: {}, Buffer status: {}", 
                                   buffer_idx,
                                   this->buffer_status_[module_type][buffer_idx]);
            }
        }
    }
}


// void GPU_Weight_Buffer::reset_decoding_buffer() {
//     auto& buffer_shapes = this->engine_config_.gpu_buffer_config.module_shapes;
//     auto& num_buffers =
//         this->engine_config_.gpu_buffer_config
//             .num_decoding_module_buffer;  // {"attn": 1, "routed_expert": 160,
//                                           // "shared_expert": 1}
//     auto options =
//         torch::TensorOptions()
//             .dtype(this->engine_config_.basic_config.weight_dtype_torch)
//             .device(torch::kCUDA, this->engine_config_.basic_config.device)
//             .requires_grad(false)
//             .memory_format(torch::MemoryFormat::Contiguous);
//     {
//         std::lock_guard<std::mutex> lock(this->mutex_);
//         // reset the buffer size as the prefill buffer size.
//         this->buffers_["routed_expert"].clear();
//         this->buffers_["routed_expert"].resize(num_buffers["routed_expert"]);
//         for (auto& [buffer_name, buffer_shape] :
//              buffer_shapes["routed_expert"]) {
//             for (int64_t buffer_idx = 0;
//                  buffer_idx < num_buffers["routed_expert"]; buffer_idx++) {
//                 if (buffer_name.find("norm") == std::string::npos) {
//                     this->buffers_["routed_expert"][buffer_idx][buffer_name] =
//                         torch::zeros(buffer_shape, options);
//                 } else {
//                     auto options =
//                         torch::TensorOptions()
//                             .dtype(this->engine_config_.basic_config.weight_dtype_torch)
//                             .device(torch::kCUDA,
//                                     this->engine_config_.basic_config.device)
//                             .requires_grad(false)
//                             .memory_format(torch::MemoryFormat::Contiguous);
//                     this->buffers_["routed_expert"][buffer_idx][buffer_name] =
//                         torch::ones(buffer_shape, options);
//                 }
//             }
//         }
//         this->module_in_buffers_.clear();

//         // Set all buffer status to 0.
//         for (auto& [module_type, num_buffer] : num_buffers) {
//             this->buffer_status_[module_type].clear();
//             this->buffer_status_[module_type].resize(num_buffer, 0);
//         }
//         // Log the buffer status and size of each buffer.
//         for (auto& [module_type, num_buffer] : num_buffers) {
//             this->logger_->debug("Module type: {}, Number of buffer: {}",
//                                  module_type, num_buffer);
//             for (int64_t buffer_idx = 0; buffer_idx < num_buffer;
//                  buffer_idx++) {
//                 this->logger_->debug(
//                     "Buffer_idx: {}, Buffer status: {}", buffer_idx,
//                     this->buffer_status_[module_type][buffer_idx]);
//             }
//         }
//         this->logger_->debug("Decoding buffer reset complete.");
//     }
// }

// void GPU_Weight_Buffer::reset_decoding_buffer() {
//     auto& buffer_shapes = this->engine_config_.gpu_buffer_config.module_shapes;
//     auto& num_buffers = this->engine_config_.gpu_buffer_config.num_decoding_module_buffer;
    
//     auto options = torch::TensorOptions()
//         .dtype(this->engine_config_.basic_config.weight_dtype_torch)
//         .device(torch::kCUDA, this->engine_config_.basic_config.device)
//         .requires_grad(false)
//         .memory_format(torch::MemoryFormat::Contiguous);
    
//     {
//         std::lock_guard<std::mutex> lock(this->mutex_);
        
//         // Step 1: Log initial memory state
//         size_t memory_before = 0;
//         if (torch::cuda::is_available()) {
//             // Using CUDA runtime API for memory info
//             size_t free_bytes, total_bytes;
//             cudaSetDevice(this->engine_config_.basic_config.device);
//             cudaMemGetInfo(&free_bytes, &total_bytes);
//             memory_before = total_bytes - free_bytes;
//             this->logger_->info("Decoding reset - Memory before: {} MB", 
//                               memory_before / (1024.0 * 1024.0));
//         }
        
//         // Step 2: Explicitly release ALL existing tensors in ALL buffer types
//         for (auto& [buffer_type, buffer_vector] : this->buffers_) {
//             for (auto& module_map : buffer_vector) {
//                 for (auto& [name, tensor] : module_map) {
//                     // Explicitly release each tensor
//                     tensor = torch::Tensor();
//                 }
//                 module_map.clear();
//             }
//             buffer_vector.clear();
            
//             // Force deallocation of vector memory
//             buffer_vector.shrink_to_fit();
//         }
        
//         // Step 3: Clear module mappings
//         this->module_in_buffers_.clear();
        
//         // Step 4: Force CUDA synchronization and aggressive memory cleanup
//         if (torch::cuda::is_available()) {
//             // CORRECT SYNTAX: Use cudaDeviceSynchronize() from CUDA runtime
//             cudaDeviceSynchronize();
            
//             // OR use AT namespace correctly:
//             // at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
//             // stream.synchronize();
            
//             // Force return memory to system
//             c10::cuda::CUDACachingAllocator::emptyCache();
            
//             // Log memory after clearing
//             size_t free_bytes, total_bytes;
//             cudaMemGetInfo(&free_bytes, &total_bytes);
//             size_t memory_after_clear = total_bytes - free_bytes;
//             this->logger_->info("Decoding reset - Memory after clearing: {} MB (freed: {} MB)", 
//                               memory_after_clear / (1024.0 * 1024.0),
//                               (memory_before - memory_after_clear) / (1024.0 * 1024.0));
//         }
        
//         // Step 5: Create new decoding buffers for routed_expert
//         this->buffers_["routed_expert"].reserve(num_buffers["routed_expert"]);
//         this->buffers_["routed_expert"].resize(num_buffers["routed_expert"]);
        
//         // Pre-calculate buffer names that need norm (for efficiency)
//         std::unordered_set<std::string> norm_buffers;
//         for (const auto& [buffer_name, buffer_shape] : buffer_shapes["routed_expert"]) {
//             if (buffer_name.find("norm") != std::string::npos) {
//                 norm_buffers.insert(buffer_name);
//             }
//         }
        
//         // Create tensors
//         for (auto& [buffer_name, buffer_shape] : buffer_shapes["routed_expert"]) {
//             bool is_norm = norm_buffers.count(buffer_name) > 0;
            
//             for (int64_t buffer_idx = 0; buffer_idx < num_buffers["routed_expert"]; buffer_idx++) {
//                 if (!is_norm) {
//                     this->buffers_["routed_expert"][buffer_idx][buffer_name] =
//                         torch::zeros(buffer_shape, options);
//                 } else {
//                     this->buffers_["routed_expert"][buffer_idx][buffer_name] =
//                         torch::ones(buffer_shape, options);
//                 }
//             }
//         }
        
//         // Step 6: Reset buffer status for ALL module types
//         for (auto& [module_type, num_buffer] : num_buffers) {
//             this->buffer_status_[module_type].clear();
//             this->buffer_status_[module_type].shrink_to_fit();
//             this->buffer_status_[module_type].resize(num_buffer, 0);
//         }
        
//         // Step 7: Final memory check and logging
//         if (torch::cuda::is_available()) {
//             size_t free_bytes, total_bytes;
//             cudaMemGetInfo(&free_bytes, &total_bytes);
//             size_t memory_final = total_bytes - free_bytes;
//             this->logger_->info("Decoding reset - Memory after recreation: {} MB (net change: {} MB)", 
//                               memory_final / (1024.0 * 1024.0),
//                               (static_cast<int64_t>(memory_final) - static_cast<int64_t>(memory_before)) / (1024.0 * 1024.0));
//         }
        
//         // Log buffer status
//         for (auto& [module_type, num_buffer] : num_buffers) {
//             this->logger_->debug("Module type: {}, Number of buffer: {}",
//                                module_type, num_buffer);
//             for (int64_t buffer_idx = 0; buffer_idx < num_buffer; buffer_idx++) {
//                 this->logger_->debug("Buffer_idx: {}, Buffer status: {}", 
//                                    buffer_idx, this->buffer_status_[module_type][buffer_idx]);
//             }
//         }
        
//         this->logger_->debug("Decoding buffer reset complete.");
//     }
// }


void GPU_Weight_Buffer::reset_decoding_buffer() {
    auto& buffer_shapes = this->engine_config_.gpu_buffer_config.module_shapes;
    auto& num_buffers = this->engine_config_.gpu_buffer_config.num_decoding_module_buffer;

    // Per-module dtype lookup for routed_expert
    torch::Dtype module_dtype;
    auto dtype_it = this->engine_config_.gpu_buffer_config.weight_dtypes.find("routed_expert");
    if (dtype_it != this->engine_config_.gpu_buffer_config.weight_dtypes.end()) {
        module_dtype = dtype_it->second;
    } else {
        module_dtype = this->engine_config_.basic_config.weight_dtype_torch;
    }
    auto options = torch::TensorOptions()
        .dtype(module_dtype)
        .device(torch::kCUDA, this->engine_config_.basic_config.device)
        .requires_grad(false)
        .memory_format(torch::MemoryFormat::Contiguous);

    {
        std::lock_guard<std::mutex> lock(this->mutex_);
        
        // IMPORTANT: Only clear routed_expert buffers!
        if (this->buffers_.find("routed_expert") != this->buffers_.end()) {
            for (auto& module_map : this->buffers_["routed_expert"]) {
                for (auto& [name, tensor] : module_map) {
                    // Explicitly release tensor memory
                    tensor = torch::Tensor();
                }
                module_map.clear();
            }
            this->buffers_["routed_expert"].clear();
            
            // Optional: Force memory cleanup
            if (torch::cuda::is_available()) {
                cudaDeviceSynchronize();
                c10::cuda::CUDACachingAllocator::emptyCache();
            }
        }
        
        // Reset the buffer size
        this->buffers_["routed_expert"].resize(num_buffers["routed_expert"]);
        
        // Create new tensors
        for (auto& [buffer_name, buffer_shape] : buffer_shapes["routed_expert"]) {
            for (int64_t buffer_idx = 0; buffer_idx < num_buffers["routed_expert"]; buffer_idx++) {
                if (buffer_name.find("norm") == std::string::npos) {
                    this->buffers_["routed_expert"][buffer_idx][buffer_name] =
                        torch::zeros(buffer_shape, options);
                } else {
                    // Norm buffers use per-module dtype (same as main options)
                    auto norm_options = torch::TensorOptions()
                        .dtype(module_dtype)
                        .device(torch::kCUDA, this->engine_config_.basic_config.device)
                        .requires_grad(false)
                        .memory_format(torch::MemoryFormat::Contiguous);
                    this->buffers_["routed_expert"][buffer_idx][buffer_name] =
                        torch::ones(buffer_shape, norm_options);
                }
            }
        }
        
        // Clear module_in_buffers
        this->module_in_buffers_.clear();
        
        // Set all buffer status to 0
        for (auto& [module_type, num_buffer] : num_buffers) {
            this->buffer_status_[module_type].clear();
            this->buffer_status_[module_type].resize(num_buffer, 0);
        }
        
        // Log the buffer status and size of each buffer
        for (auto& [module_type, num_buffer] : num_buffers) {
            this->logger_->debug("Module type: {}, Number of buffer: {}",
                                 module_type, num_buffer);
            for (int64_t buffer_idx = 0; buffer_idx < num_buffer; buffer_idx++) {
                this->logger_->debug("Buffer_idx: {}, Buffer status: {}", 
                                   buffer_idx,
                                   this->buffer_status_[module_type][buffer_idx]);
            }
        }
        
        this->logger_->debug("Decoding buffer reset complete.");
    }
}