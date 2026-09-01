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

#include <chrono>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <torch/extension.h>
#include <torch/torch.h>
#include <unordered_map>
#include <unordered_set>
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

namespace {
// Helper function to determine the dtype for a specific tensor.
// Priority: 1) tensor_dtypes[module_type][buffer_name]
//           2) weight_dtypes[module_type]
//           3) basic_config.weight_dtype_torch
torch::Dtype get_tensor_dtype(
    const GPU_Buffer_Config& gpu_config,
    const Basic_Config& basic_config,
    const std::string& module_type,
    const std::string& buffer_name) {

    // 1. Check per-tensor dtype override
    auto module_it = gpu_config.tensor_dtypes.find(module_type);
    if (module_it != gpu_config.tensor_dtypes.end()) {
        auto tensor_it = module_it->second.find(buffer_name);
        if (tensor_it != module_it->second.end()) {
            return tensor_it->second;
        }
    }

    // 2. Check per-module dtype
    auto dtype_it = gpu_config.weight_dtypes.find(module_type);
    if (dtype_it != gpu_config.weight_dtypes.end()) {
        return dtype_it->second;
    }

    // 3. Fall back to global weight dtype
    return basic_config.weight_dtype_torch;
}
}  // namespace

GPU_Weight_Buffer::GPU_Weight_Buffer(EngineConfig& engine_config,
                                     ModelConfig& model_config)
    : engine_config_(engine_config), model_config_(model_config) {
    this->logger_ = init_logger(
        this->engine_config_.basic_config.log_level,
        "GPU_Weight_Buffer" +
            std::to_string(this->engine_config_.basic_config.device));
    this->logger_->info("GPU_Weight_Buffer Instantiated.");
};

GPU_Weight_Buffer::~GPU_Weight_Buffer() {
    cudaSetDevice(this->engine_config_.basic_config.device);
    std::lock_guard<std::mutex> lock(this->mutex_);
    this->clearPendingReleasesLocked();
    for (auto& [module_type, events] : this->ready_events_) {
        for (auto event : events) {
            if (event != nullptr) {
                cudaEventDestroy(event);
            }
        }
    }
}

void GPU_Weight_Buffer::clearPendingReleasesLocked() {
    for (auto& pending : this->pending_releases_) {
        if (pending.event != nullptr) {
            cudaEventDestroy(pending.event);
        }
    }
    this->pending_releases_.clear();
}

void GPU_Weight_Buffer::resetReadyEventsLocked(
    const std::string& module_type, int64_t num_buffers) {
    CUDA_CHECK(cudaSetDevice(this->engine_config_.basic_config.device));
    auto& events = this->ready_events_[module_type];
    for (auto event : events) {
        if (event != nullptr) {
            CUDA_CHECK(cudaEventDestroy(event));
        }
    }
    events.assign(num_buffers, nullptr);
    for (auto& event : events) {
        CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    }
}

cudaEvent_t GPU_Weight_Buffer::readyEventLocked(
    const std::string& module_type, int64_t buffer_idx) const {
    auto events_it = this->ready_events_.find(module_type);
    if (events_it == this->ready_events_.end() || buffer_idx < 0 ||
        buffer_idx >= static_cast<int64_t>(events_it->second.size())) {
        throw std::runtime_error(
            "Missing ready event for weight buffer type=" + module_type +
            " idx=" + std::to_string(buffer_idx));
    }
    return events_it->second[buffer_idx];
}

void GPU_Weight_Buffer::reclaimCompletedReleasesLocked() {
    for (auto it = this->pending_releases_.begin();
         it != this->pending_releases_.end();) {
        const cudaError_t status = cudaEventQuery(it->event);
        if (status == cudaErrorNotReady) {
            ++it;
            continue;
        }
        CUDA_CHECK(status);
        for (const auto& [module_type, buffer_idx] : it->slots) {
            auto status_it = this->buffer_status_.find(module_type);
            if (status_it == this->buffer_status_.end() || buffer_idx < 0 ||
                buffer_idx >= static_cast<int64_t>(status_it->second.size()) ||
                status_it->second[buffer_idx] != 2) {
                throw std::runtime_error(
                    "Invalid pending weight-buffer release: type=" +
                    module_type + " idx=" + std::to_string(buffer_idx));
            }
            status_it->second[buffer_idx] = 0;
        }
        CUDA_CHECK(cudaEventDestroy(it->event));
        it = this->pending_releases_.erase(it);
    }
}

void GPU_Weight_Buffer::Init() {
    auto& buffer_shapes = this->engine_config_.gpu_buffer_config.module_shapes;
    auto& num_buffers =
        this->engine_config_.gpu_buffer_config
            .num_prefill_module_buffer;  // TODO: {"attn": 1, "routed_expert":
                                         // 160, "shared_expert": 1}
    for (auto& [module_type, num_buffer] : num_buffers) {
        this->buffers_[module_type].clear();
        this->buffers_[module_type].resize(num_buffer);
        this->buffer_status_[module_type].clear();
        this->buffer_status_[module_type].resize(num_buffer, 0);
        for (auto& [buffer_name, buffer_shape] : buffer_shapes[module_type]) {
            // Get per-tensor dtype using the helper function
            torch::Dtype tensor_dtype = get_tensor_dtype(
                this->engine_config_.gpu_buffer_config,
                this->engine_config_.basic_config,
                module_type,
                buffer_name);
            auto options =
                torch::TensorOptions()
                    .dtype(tensor_dtype)
                    .device(torch::kCUDA, this->engine_config_.basic_config.device)
                    .requires_grad(false)
                    .memory_format(torch::MemoryFormat::Contiguous);
            for (int64_t buffer_idx = 0; buffer_idx < num_buffer;
                 buffer_idx++) {
                this->buffers_[module_type][buffer_idx][buffer_name] =
                    torch::zeros(buffer_shape, options);
            }
        }
        this->resetReadyEventsLocked(module_type, num_buffer);
    }
};

void GPU_Weight_Buffer::resize_buffer() {
    auto& buffer_shapes = this->engine_config_.gpu_buffer_config.module_shapes;
    auto& num_buffers =
        this->engine_config_.gpu_buffer_config
            .num_decoding_module_buffer;  // {"attn": 1, "routed_expert": 160,
                                          // "shared_expert": 1}
    {
        std::lock_guard<std::mutex> lock(this->mutex_);
        CUDA_CHECK(cudaSetDevice(this->engine_config_.basic_config.device));
        int64_t to_add_buffer = num_buffers["routed_expert"] -
                                this->buffers_["routed_expert"].size();
        for (int64_t i = 0; i < to_add_buffer; i++) {
            module_weight_tensor_map new_buffer;
            for (auto& [buffer_name, buffer_shape] :
                 buffer_shapes["routed_expert"]) {
                // Get per-tensor dtype using the helper function
                torch::Dtype tensor_dtype = get_tensor_dtype(
                    this->engine_config_.gpu_buffer_config,
                    this->engine_config_.basic_config,
                    "routed_expert",
                    buffer_name);
                auto options =
                    torch::TensorOptions()
                        .dtype(tensor_dtype)
                        .device(torch::kCUDA,
                                this->engine_config_.basic_config.device)
                        .requires_grad(false)
                        .memory_format(torch::MemoryFormat::Contiguous);
                new_buffer[buffer_name] = torch::zeros(buffer_shape, options);
            }
            this->buffers_["routed_expert"].push_back(new_buffer);
            this->buffer_status_["routed_expert"].push_back(0);
            cudaEvent_t ready_event = nullptr;
            CUDA_CHECK(cudaEventCreateWithFlags(
                &ready_event, cudaEventDisableTiming));
            this->ready_events_["routed_expert"].push_back(ready_event);
        }
    }
}

std::optional<
    std::pair<std::reference_wrapper<module_weight_tensor_map>, int64_t>>
GPU_Weight_Buffer::acquireEmptyBuffer(const std::string& module_type) {
    std::lock_guard<std::mutex> lock(this->mutex_);
    this->reclaimCompletedReleasesLocked();
    for (int64_t buffer_idx = 0;
         buffer_idx < this->buffer_status_[module_type].size(); buffer_idx++) {
        if (this->buffer_status_[module_type][buffer_idx] == 0) {
            this->buffer_status_[module_type][buffer_idx] = 1;
            this->logger_->debug("Acquired empty buffer: type={}, idx={}",
                                module_type, buffer_idx);
            return std::make_pair(
                std::ref(this->buffers_[module_type][buffer_idx]), buffer_idx);
        }
    }
    return std::nullopt;
};

void GPU_Weight_Buffer::releaseBuffer(const std::string& module_name) {
    std::lock_guard<std::mutex> lock(this->mutex_);
    auto module_it = this->module_in_buffers_.find(module_name);
    if (module_it == this->module_in_buffers_.end()) {
        throw std::runtime_error(
            "Cannot release non-resident weight module: " + module_name);
    }
    const auto [module_type, buffer_idx] = module_it->second;
    auto status_it = this->buffer_status_.find(module_type);
    if (status_it == this->buffer_status_.end() || buffer_idx < 0 ||
        buffer_idx >= static_cast<int64_t>(status_it->second.size())) {
        throw std::runtime_error(
            "Invalid weight-buffer ownership for module: " + module_name);
    }
    this->module_in_buffers_.erase(module_it);
    status_it->second[buffer_idx] = 0;
    this->logger_->debug("Released buffer: module={}, type={}, idx={}",
                         module_name, module_type, buffer_idx);
};

void GPU_Weight_Buffer::releaseBuffersAsync(
    const std::vector<std::string>& module_names,
    cudaStream_t consumer_stream) {
    if (module_names.empty()) {
        return;
    }

    std::lock_guard<std::mutex> lock(this->mutex_);
    std::unordered_set<std::string> unique_names;
    std::vector<std::pair<std::string, int64_t>> slots;
    slots.reserve(module_names.size());
    for (const auto& module_name : module_names) {
        if (!unique_names.insert(module_name).second) {
            throw std::runtime_error(
                "Duplicate module in asynchronous weight release: " +
                module_name);
        }
        auto module_it = this->module_in_buffers_.find(module_name);
        if (module_it == this->module_in_buffers_.end()) {
            throw std::runtime_error(
                "Cannot asynchronously release non-resident weight module: " +
                module_name);
        }
        const auto [module_type, buffer_idx] = module_it->second;
        auto status_it = this->buffer_status_.find(module_type);
        if (status_it == this->buffer_status_.end() || buffer_idx < 0 ||
            buffer_idx >= static_cast<int64_t>(status_it->second.size()) ||
            status_it->second[buffer_idx] != 1) {
            throw std::runtime_error(
                "Invalid asynchronous weight-buffer ownership for module: " +
                module_name);
        }
        slots.emplace_back(module_type, buffer_idx);
    }

    cudaEvent_t completion_event = nullptr;
    CUDA_CHECK(cudaEventCreateWithFlags(
        &completion_event, cudaEventDisableTiming));
    const cudaError_t record_status =
        cudaEventRecord(completion_event, consumer_stream);
    if (record_status != cudaSuccess) {
        cudaEventDestroy(completion_event);
        CUDA_CHECK(record_status);
    }

    for (size_t i = 0; i < module_names.size(); ++i) {
        this->module_in_buffers_.erase(module_names[i]);
        const auto& [module_type, buffer_idx] = slots[i];
        this->buffer_status_[module_type][buffer_idx] = 2;
    }
    this->pending_releases_.push_back(
        PendingRelease{completion_event, std::move(slots)});
    this->logger_->debug(
        "Queued asynchronous release for {} weight buffers",
        module_names.size());
}

module_weight_tensor_map GPU_Weight_Buffer::get_weights_pinned(
    const std::string& module_name,
    cudaStream_t consumer_stream) {
    constexpr auto timeout = std::chrono::seconds(120);
    std::unique_lock<std::mutex> lock(this->mutex_);
    const bool ready = this->cv_.wait_for(lock, timeout, [this, &module_name] {
        return this->module_in_buffers_.find(module_name) !=
               this->module_in_buffers_.end();
    });
    if (!ready) {
        this->logger_->error(
            "Timeout waiting for pinned weight module: {}", module_name);
        throw std::runtime_error(
            "Timeout waiting for pinned weight module: " + module_name);
    }
    const auto [module_type, buffer_idx] =
        this->module_in_buffers_.at(module_name);
    auto tensors = this->buffers_.at(module_type).at(buffer_idx);
    const cudaEvent_t ready_event =
        this->readyEventLocked(module_type, buffer_idx);
    lock.unlock();
    CUDA_CHECK(cudaStreamWaitEvent(consumer_stream, ready_event, 0));
    return tensors;
}

module_weight_tensor_map GPU_Weight_Buffer::get_weights(
    const std::string& module_name,
    std::string& phase,
    cudaStream_t consumer_stream)
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
                    auto tensors = this->buffers_[module_type][buffer_idx];
                    const cudaEvent_t ready_event =
                        this->readyEventLocked(module_type, buffer_idx);
                    lock.unlock();
                    CUDA_CHECK(cudaStreamWaitEvent(
                        consumer_stream, ready_event, 0));
                    return tensors;
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

void GPU_Weight_Buffer::weights_copy_enqueued(
    const std::string& module_type,
    const std::string& module_name,
    int64_t buffer_idx,
    cudaStream_t copy_stream) {
    cudaEvent_t ready_event = nullptr;
    {
        std::lock_guard<std::mutex> lock(this->mutex_);
        ready_event = this->readyEventLocked(module_type, buffer_idx);
    }
    CUDA_CHECK(cudaEventRecord(ready_event, copy_stream));
    {
        std::lock_guard<std::mutex> lock(this->mutex_);
        this->module_in_buffers_[module_name] =
            std::make_pair(module_type, buffer_idx);
        this->logger_->debug(
            "Module: {} copy is enqueued in buffer: {}", module_name,
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
        this->clearPendingReleasesLocked();

        // Reset the buffer size as the prefill buffer size
        this->buffers_["routed_expert"].resize(num_buffers["routed_expert"]);

        // Create new tensors
        for (auto& [buffer_name, buffer_shape] : buffer_shapes["routed_expert"]) {
            // Get per-tensor dtype using the helper function
            torch::Dtype tensor_dtype = get_tensor_dtype(
                this->engine_config_.gpu_buffer_config,
                this->engine_config_.basic_config,
                "routed_expert",
                buffer_name);
            auto options = torch::TensorOptions()
                .dtype(tensor_dtype)
                .device(torch::kCUDA, this->engine_config_.basic_config.device)
                .requires_grad(false)
                .memory_format(torch::MemoryFormat::Contiguous);
            for (int64_t buffer_idx = 0; buffer_idx < num_buffers["routed_expert"]; buffer_idx++) {
                this->buffers_["routed_expert"][buffer_idx][buffer_name] =
                    torch::zeros(buffer_shape, options);
            }
        }

        // Clear module_in_buffers
        this->module_in_buffers_.clear();

        // Set all buffer status to 0 (for ALL module types, not just routed_expert)
        for (auto& [module_type, num_buffer] : num_buffers) {
            this->buffer_status_[module_type].clear();
            this->buffer_status_[module_type].resize(num_buffer, 0);
        }
        this->resetReadyEventsLocked(
            "routed_expert", num_buffers["routed_expert"]);

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
        this->clearPendingReleasesLocked();

        // Reset the buffer size
        this->buffers_["routed_expert"].resize(num_buffers["routed_expert"]);

        // Create new tensors
        for (auto& [buffer_name, buffer_shape] : buffer_shapes["routed_expert"]) {
            // Get per-tensor dtype using the helper function
            torch::Dtype tensor_dtype = get_tensor_dtype(
                this->engine_config_.gpu_buffer_config,
                this->engine_config_.basic_config,
                "routed_expert",
                buffer_name);
            auto options = torch::TensorOptions()
                .dtype(tensor_dtype)
                .device(torch::kCUDA, this->engine_config_.basic_config.device)
                .requires_grad(false)
                .memory_format(torch::MemoryFormat::Contiguous);
            for (int64_t buffer_idx = 0; buffer_idx < num_buffers["routed_expert"]; buffer_idx++) {
                this->buffers_["routed_expert"][buffer_idx][buffer_name] =
                    torch::zeros(buffer_shape, options);
            }
        }

        // Clear module_in_buffers
        this->module_in_buffers_.clear();

        // Set all buffer status to 0
        for (auto& [module_type, num_buffer] : num_buffers) {
            this->buffer_status_[module_type].clear();
            this->buffer_status_[module_type].resize(num_buffer, 0);
        }
        this->resetReadyEventsLocked(
            "routed_expert", num_buffers["routed_expert"]);

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
