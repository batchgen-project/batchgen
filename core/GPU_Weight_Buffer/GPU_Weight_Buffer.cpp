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
            .dtype(this->engine_config_.basic_config.dtype_torch)
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
            .dtype(this->engine_config_.basic_config.dtype_torch)
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
module_weight_tensor_map GPU_Weight_Buffer::get_weights(
    const std::string& module_name) {
    this->logger_->debug("Get weights: {}", module_name);
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
                    this->clear_expert_buffer(layer_idx, expert_idx);
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

void GPU_Weight_Buffer::clear_expert_buffer(int64_t layer_idx,
                                            int64_t expert_idx) {
    // clear expert in the buffer that is in previous of current expert.
    // this->logger_->debug("Clearing expert buffer prev of: layer_idx: {},
    // expert_idx: {}", layer_idx, expert_idx); for(auto const& [key, val] :
    // this->module_in_buffers_){ 	this->logger_->debug("Before Clearing
    // Module_in_buffers_: Key: {}", key);
    // }
    // Step 1: clear current layer.
    for (int64_t idx = 0; idx < expert_idx; idx++) {
        std::string module_name = "routed_expert_" + std::to_string(layer_idx) +
                                  "_" + std::to_string(idx);
        if (this->module_in_buffers_.find(module_name) !=
            this->module_in_buffers_.end()) {
            this->releaseBuffer(module_name);
        }
    };
    // Step 2: clear previous layer.
    int64_t prev_layer_idx;
    if (this->model_config_.model_type == "deepseek_v2") {
        if ((layer_idx == 0) || (layer_idx == 1)) {
            prev_layer_idx = this->model_config_.num_hidden_layers - 1;
        } else {
            prev_layer_idx = layer_idx - 1;
        }
    } else if (this->model_config_.model_type == "deepseek_v3") {
        if ((layer_idx >= 0) && (layer_idx <= 3)) {
            prev_layer_idx = this->model_config_.num_hidden_layers - 1;
        } else {
            prev_layer_idx = layer_idx - 1;
        }
    } else {
        prev_layer_idx = layer_idx - 1;
    }

    for (int64_t idx = 0; idx < this->model_config_.num_local_experts; idx++) {
        std::string module_name = "routed_expert_" +
                                  std::to_string(prev_layer_idx) + "_" +
                                  std::to_string(idx);
        if (this->module_in_buffers_.find(module_name) !=
            this->module_in_buffers_.end()) {
            this->releaseBuffer(module_name);
        }
    };
    // for(auto const& [key, val] : this->module_in_buffers_){
    // 	this->logger_->debug("After Clearing Module_in_buffers_: Key: {}", key);
    // }
}

void GPU_Weight_Buffer::reset_prefill_buffer() {
    auto& buffer_shapes = this->engine_config_.gpu_buffer_config.module_shapes;
    auto& num_buffers =
        this->engine_config_.gpu_buffer_config
            .num_prefill_module_buffer;  // {"attn": 1, "routed_expert": 160,
                                         // "shared_expert": 1}
    auto options =
        torch::TensorOptions()
            .dtype(this->engine_config_.basic_config.dtype_torch)
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
            .dtype(this->engine_config_.basic_config.dtype_torch)
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
        this->logger_->debug("Decoding buffer reset complete.");
    }
}