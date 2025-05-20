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
    auto options =
        torch::TensorOptions()
            .dtype(this->engine_config_.basic_config.weight_dtype_torch)
            .device(torch::kCUDA, this->engine_config_.basic_config.device)
            .requires_grad(false)
            .memory_format(torch::MemoryFormat::Contiguous);
    for (auto& [module_type, num_buffer] : num_buffers) {
        this->buffers_[module_type].clear();
        this->buffers_[module_type].resize(num_buffer);
        this->buffer_status_[module_type].clear();
        this->buffer_status_[module_type].resize(num_buffer, 0);
        for (auto& [buffer_name, buffer_shape] : buffer_shapes[module_type]) {
            for (int64_t buffer_idx = 0; buffer_idx < num_buffer;
                 buffer_idx++) {
                if (buffer_name.find("norm") == std::string::npos) {
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
    auto options =
        torch::TensorOptions()
            .dtype(this->engine_config_.basic_config.weight_dtype_torch)
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
                if (buffer_name.find("norm") == std::string::npos) {
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

// std::shared_ptr<module_weight_tensor_map> GPU_Weight_Buffer::get_weights(
//     const std::string& module_name) {
//     this->logger_->debug("Get weights: {}", module_name);
//     try {
//         while (true) {
//             {
//                 std::unique_lock<std::mutex> lock(this->mutex_);
//                 if (this->cv_.wait_for(
//                         lock, std::chrono::milliseconds(1),
//                         [this, module_name] {
//                             return this->module_in_buffers_.find(module_name) !=
//                                    this->module_in_buffers_.end();
//                         })) {
//                     auto [module_type, buffer_idx] =
//                         this->module_in_buffers_[module_name];
//                     return std::make_shared<module_weight_tensor_map>(
//                         this->buffers_[module_type][buffer_idx]);
//                 }
//                 this->logger_->debug("Waiting for module: {}", module_name);
//             }
//             // Check if module_name starts with "routed_expert" and has enough
//             // length
//             if (module_name.substr(0, 13) == "routed_expert") {
//                 // Find the last two underscores
//                 size_t last_underscore = module_name.rfind('_');
//                 size_t second_last_underscore =
//                     module_name.rfind('_', last_underscore - 1);

//                 if (last_underscore != std::string::npos &&
//                     second_last_underscore != std::string::npos) {
//                     // Extract indices using the underscore positions
//                     std::string layer_str = module_name.substr(
//                         second_last_underscore + 1,
//                         last_underscore - second_last_underscore - 1);
//                     std::string expert_str =
//                         module_name.substr(last_underscore + 1);

//                     int64_t layer_idx = std::stoi(layer_str);
//                     int64_t expert_idx = std::stoi(expert_str);
//                     this->logger_->debug(
//                         "Clearing expert buffer: layer_idx: {}, expert_idx: {}",
//                         layer_idx, expert_idx);
//                     this->clear_expert_buffer(layer_idx, expert_idx);
//                 } else {
//                     this->logger_->error(
//                         "Invalid format in module name: '{}', expected "
//                         "format: routed_expert_X_Y",
//                         module_name);
//                 }
//             }
//         }
//     } catch (const c10::Error& e) {
//         this->logger_->debug(
//             "GPU_Weight_Buffer get_weights(): CUDA/PyTorch error: {}",
//             e.what());
//         throw std::runtime_error(e.what());
//     }
//     // Catch CUDA runtime errors
//     catch (const cudaError_t& err) {
//         this->logger_->debug(
//             "GPU_Weight_Buffer get_weights(): CUDA runtime error: {}",
//             cudaGetErrorString(err));
//         throw std::runtime_error(cudaGetErrorString(err));
//     }
//     // Catch standard C++ exceptions
//     catch (const std::exception& e) {
//         this->logger_->debug("GPU_Weight_Buffer get_weights() Error: {}",
//                              e.what());
//         throw std::runtime_error("GPU_Weight_Buffer get_weights()");
//     }
//     // Catch any other unexpected errors
//     catch (...) {
//         this->logger_->debug("GPU_Weight_Buffer get_weights()");
//         throw std::runtime_error("GPU_Weight_Buffer get_weights()");
//     }
// };
// module_weight_tensor_map GPU_Weight_Buffer::get_weights(
//     const std::string& module_name,
//     std::string& phase) 
// {
//     this->logger_->debug("Get weights: {}", module_name);
//     try {
//         while (true) {
//             {
//                 std::unique_lock<std::mutex> lock(this->mutex_);
//                 if (this->cv_.wait_for(
//                         lock, std::chrono::milliseconds(1),
//                         [this, module_name] {
//                             return this->module_in_buffers_.find(module_name) !=
//                                    this->module_in_buffers_.end();
//                         })) {
//                     auto [module_type, buffer_idx] =
//                         this->module_in_buffers_[module_name];
//                     return this->buffers_[module_type][buffer_idx];
//                 }
//                 this->logger_->debug("Waiting for module: {}", module_name);
//             }
//             // Check if module_name starts with "routed_expert" and has enough
//             // length
//             if (module_name.substr(0, 13) == "routed_expert") {
//                 // Find the last two underscores
//                 size_t last_underscore = module_name.rfind('_');
//                 size_t second_last_underscore =
//                     module_name.rfind('_', last_underscore - 1);

//                 if (last_underscore != std::string::npos &&
//                     second_last_underscore != std::string::npos) {
//                     // Extract indices using the underscore positions
//                     std::string layer_str = module_name.substr(
//                         second_last_underscore + 1,
//                         last_underscore - second_last_underscore - 1);
//                     std::string expert_str =
//                         module_name.substr(last_underscore + 1);

//                     int64_t layer_idx = std::stoi(layer_str);
//                     int64_t expert_idx = std::stoi(expert_str);
//                     this->logger_->debug(
//                         "Clearing expert buffer: layer_idx: {}, expert_idx: {}",
//                         layer_idx, expert_idx);
//                     this->clear_expert_buffer(layer_idx, expert_idx, phase);
//                 } else {
//                     this->logger_->error(
//                         "Invalid format in module name: '{}', expected "
//                         "format: routed_expert_X_Y",
//                         module_name);
//                 }
//             }
//         }
//     } catch (const c10::Error& e) {
//         this->logger_->debug(
//             "GPU_Weight_Buffer get_weights(): CUDA/PyTorch error: {}",
//             e.what());
//         throw std::runtime_error(e.what());
//     }
//     // Catch CUDA runtime errors
//     catch (const cudaError_t& err) {
//         this->logger_->debug(
//             "GPU_Weight_Buffer get_weights(): CUDA runtime error: {}",
//             cudaGetErrorString(err));
//         throw std::runtime_error(cudaGetErrorString(err));
//     }
//     // Catch standard C++ exceptions
//     catch (const std::exception& e) {
//         this->logger_->debug("GPU_Weight_Buffer get_weights() Error: {}",
//                              e.what());
//         throw std::runtime_error("GPU_Weight_Buffer get_weights()");
//     }
//     // Catch any other unexpected errors
//     catch (...) {
//         this->logger_->debug("GPU_Weight_Buffer get_weights()");
//         throw std::runtime_error("GPU_Weight_Buffer get_weights()");
//     }
// };

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


// void GPU_Weight_Buffer::clear_expert_buffer(int64_t layer_idx,
//                                             int64_t expert_idx) {
//     // Step 1: clear current layer.
//     for (int64_t idx = 0; idx < expert_idx; idx++) {
//         std::string module_name = "routed_expert_" + std::to_string(layer_idx) +
//                                   "_" + std::to_string(idx);
//         if (this->module_in_buffers_.find(module_name) !=
//             this->module_in_buffers_.end()) {
//             this->releaseBuffer(module_name);
//         }
//     };
//     // Step 2: Clear expert not belongs the current layer and the next layer.
//     int64_t next_layer_idx;
//     if (this->model_config_.model_type == "deepseek_v2") {
//         if ((layer_idx == 0) || (layer_idx == (this->model_config_.num_hidden_layers - 1))) {
//             next_layer_idx = 1;
//         } else {
//             next_layer_idx = layer_idx + 1;
//         }
//     } else if (this->model_config_.model_type == "deepseek_v3") {
//         if (layer_idx == 0 || layer_idx == 1 || layer_idx == 2 || 
//             layer_idx == this->model_config_.num_hidden_layers - 1) {
//             next_layer_idx = 3;
//         } else {
//             next_layer_idx = layer_idx + 1;
//         }
//     }
//     std::vector<std::string> keys_to_remove;
//     for (const auto& [key, value] : this->module_in_buffers_) {
//         // Skip if key doesn't match the routed_expert pattern
//         if (key.find("routed_expert_") != 0) {
//             continue;
//         }
        
//         // Parse layer_id from the key
//         // Format: routed_expert_{layer_id}_{expert_id}
//         size_t first_underscore = key.find('_', 13); // "routed_expert_" is 13 chars
//         size_t second_underscore = key.find('_', first_underscore + 1);
        
//         if (first_underscore == std::string::npos || second_underscore == std::string::npos) {
//             // Malformed key, skip it
//             continue;
//         }
        
//         std::string layer_id_str = key.substr(first_underscore + 1, second_underscore - first_underscore - 1);
//         int64_t layer_id;
        
//         try {
//             layer_id = std::stoll(layer_id_str);
//         } catch (const std::exception& e) {
//             // Invalid layer_id, skip it
//             continue;
//         }
        
//         // Keep if it's current layer or next layer, otherwise mark for removal
//         if (layer_id != layer_idx && layer_id != next_layer_idx) {
//             // this->logger_->info(
//             //     "Current layer: {}, current expert: {}, clearing expert buffer: {}",
//             //     layer_idx, expert_idx, key);
//             keys_to_remove.push_back(key);
//         }
//     }
//     for (const auto& key : keys_to_remove) {
//         this->releaseBuffer(key);
//     }
// }

// void GPU_Weight_Buffer::clear_expert_buffer(
//     int64_t layer_idx,
//     int64_t expert_idx,
//     std::string phase) 
// {
//     // Step 1: Prepare a hashset contains the routed expert name that is allowed.
//     // Allowded expert name is from current expert to current expert + engine_config num expert buffer.
//     // Find the current expert name of the weight_copy_tasks_. The insert the names to the hashset.
//     // If we reach the end of the task list, append start from the beginning.
//     std::unordered_set<std::string> allowed_expert_names;
//     int64_t num_expert_buffer;
//     if(phase == "prefill"){
//         num_expert_buffer = this->engine_config_.gpu_buffer_config.num_prefill_module_buffer["routed_expert"];
//     } else if(phase == "decoding"){
//         num_expert_buffer = this->engine_config_.gpu_buffer_config.num_decoding_module_buffer["routed_expert"];
//     } else {
//         throw std::runtime_error("Invalid phase: " + phase);
//     }
//     // Find the index for the current expert in the weight_copy_tasks_.
//     int64_t idx = 0;
//     std::string current_expert_name = "routed_expert_" + std::to_string(layer_idx) + "_" + std::to_string(expert_idx);
//     for (const auto& module_name : this->weight_copy_tasks_["routed_expert"]) {
//         if (module_name == current_expert_name) {
//             break;
//         }
//         idx++;
//     }
//     // Insert the current expert name to the hashset.
//     allowed_expert_names.insert(current_expert_name);
//     // Insert the next num_expert_buffer - 1 expert names to the hashset.
//     // If we reach the end of the task list, append start from the beginning.
//     for (int64_t i = 1; i < num_expert_buffer; i++) {
//         idx++;
//         if (idx >= this->weight_copy_tasks_["routed_expert"].size()) {
//             idx = 0;
//         }
//         allowed_expert_names.insert(this->weight_copy_tasks_["routed_expert"][idx]);
//     }

//     // Step 2: Clear the expert buffer that is not in the hashset.
//     std::vector<std::string> keys_to_remove;
//     for (const auto& [key, value] : this->module_in_buffers_) {
//         // Skip if key doesn't match the routed_expert pattern
//         if (key.find("routed_expert_") != 0) {
//             continue;
//         }
//         else if (allowed_expert_names.find(key) == allowed_expert_names.end()) {
//             // Add to key_to_remove if the key is not in the hashset.
//             keys_to_remove.push_back(key);
//         }
//     }
//     for (const auto& key : keys_to_remove) {
//         this->(key);
//     }
// }

void GPU_Weight_Buffer::clear_expert_buffer(int64_t layer_idx, int64_t expert_idx, std::string phase) {
    // Log the keys of the module_in_buffers_ in the same log msg
    // std::ostringstream oss;
    // oss << "Map keys: ";
    
    // size_t count = 0;
    // for (const auto& [key, value] : this->module_in_buffers_) {
    //     oss << key;
    //     if (++count < this->module_in_buffers_.size()) {
    //         oss << ", ";
    //     }
    // }
    // this->logger_->debug("clearing expert buffer: layer_idx: {}, expert_idx: {}, existing keys: {}", layer_idx, expert_idx, oss.str());

    // Get the number of expert buffers based on phase
    int64_t num_expert_buffer;
    if (phase == "prefill") {
        num_expert_buffer = this->engine_config_.gpu_buffer_config.num_prefill_module_buffer["routed_expert"];
    } else if (phase == "decoding") {
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


void GPU_Weight_Buffer::reset_prefill_buffer() {
    auto& buffer_shapes = this->engine_config_.gpu_buffer_config.module_shapes;
    auto& num_buffers =
        this->engine_config_.gpu_buffer_config
            .num_prefill_module_buffer;  // {"attn": 1, "routed_expert": 160,
                                         // "shared_expert": 1}
    auto options =
        torch::TensorOptions()
            .dtype(this->engine_config_.basic_config.weight_dtype_torch)
            .device(torch::kCUDA, this->engine_config_.basic_config.device)
            .requires_grad(false)
            .memory_format(torch::MemoryFormat::Contiguous);
    {
        std::lock_guard<std::mutex> lock(this->mutex_);
        // reset the buffer size as the prefill buffer size.
        this->buffers_["routed_expert"].clear();
        this->buffers_["routed_expert"].resize(num_buffers["routed_expert"]);
        for (auto& [buffer_name, buffer_shape] :
             buffer_shapes["routed_expert"]) {
            for (int64_t buffer_idx = 0;
                 buffer_idx < num_buffers["routed_expert"]; buffer_idx++) {
                if (buffer_name.find("norm") == std::string::npos) {
                    this->buffers_["routed_expert"][buffer_idx][buffer_name] =
                        torch::zeros(buffer_shape, options);
                } else {
                    auto bf16_options =
                        torch::TensorOptions()
                            .dtype(torch::kBFloat16)
                            .device(torch::kCUDA,
                                    this->engine_config_.basic_config.device)
                            .requires_grad(false)
                            .memory_format(torch::MemoryFormat::Contiguous);
                    this->buffers_["routed_expert"][buffer_idx][buffer_name] =
                        torch::ones(buffer_shape, bf16_options);
                }
            }
        }
        this->module_in_buffers_.clear();

        // Set all buffer status to 0.
        for (auto& [module_type, num_buffer] : num_buffers) {
            this->buffer_status_[module_type].clear();
            this->buffer_status_[module_type].resize(num_buffer, 0);
        }
        // Log the buffer status and size of each buffer.
        for (auto& [module_type, num_buffer] : num_buffers) {
            this->logger_->debug("Module type: {}, Number of buffer: {}",
                                 module_type, num_buffer);
            for (int64_t buffer_idx = 0; buffer_idx < num_buffer;
                 buffer_idx++) {
                this->logger_->debug(
                    "Buffer_idx: {}, Buffer status: {}", buffer_idx,
                    this->buffer_status_[module_type][buffer_idx]);
            }
        }
    }
}


void GPU_Weight_Buffer::reset_decoding_buffer() {
    auto& buffer_shapes = this->engine_config_.gpu_buffer_config.module_shapes;
    auto& num_buffers =
        this->engine_config_.gpu_buffer_config
            .num_decoding_module_buffer;  // {"attn": 1, "routed_expert": 160,
                                          // "shared_expert": 1}
    auto options =
        torch::TensorOptions()
            .dtype(this->engine_config_.basic_config.weight_dtype_torch)
            .device(torch::kCUDA, this->engine_config_.basic_config.device)
            .requires_grad(false)
            .memory_format(torch::MemoryFormat::Contiguous);
    {
        std::lock_guard<std::mutex> lock(this->mutex_);
        // reset the buffer size as the prefill buffer size.
        this->buffers_["routed_expert"].clear();
        this->buffers_["routed_expert"].resize(num_buffers["routed_expert"]);
        for (auto& [buffer_name, buffer_shape] :
             buffer_shapes["routed_expert"]) {
            for (int64_t buffer_idx = 0;
                 buffer_idx < num_buffers["routed_expert"]; buffer_idx++) {
                if (buffer_name.find("norm") == std::string::npos) {
                    this->buffers_["routed_expert"][buffer_idx][buffer_name] =
                        torch::zeros(buffer_shape, options);
                } else {
                    auto options =
                        torch::TensorOptions()
                            .dtype(this->engine_config_.basic_config.weight_dtype_torch)
                            .device(torch::kCUDA,
                                    this->engine_config_.basic_config.device)
                            .requires_grad(false)
                            .memory_format(torch::MemoryFormat::Contiguous);
                    this->buffers_["routed_expert"][buffer_idx][buffer_name] =
                        torch::ones(buffer_shape, options);
                }
            }
        }
        this->module_in_buffers_.clear();

        // Set all buffer status to 0.
        for (auto& [module_type, num_buffer] : num_buffers) {
            this->buffer_status_[module_type].clear();
            this->buffer_status_[module_type].resize(num_buffer, 0);
        }
        // Log the buffer status and size of each buffer.
        for (auto& [module_type, num_buffer] : num_buffers) {
            this->logger_->debug("Module type: {}, Number of buffer: {}",
                                 module_type, num_buffer);
            for (int64_t buffer_idx = 0; buffer_idx < num_buffer;
                 buffer_idx++) {
                this->logger_->debug(
                    "Buffer_idx: {}, Buffer status: {}", buffer_idx,
                    this->buffer_status_[module_type][buffer_idx]);
            }
        }
        this->logger_->debug("Decoding buffer reset complete.");
    }
}