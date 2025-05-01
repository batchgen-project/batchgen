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

#include <condition_variable>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <queue>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <string>
#include <thread>
#include <torch/extension.h>
#include <torch/torch.h>
#include <unordered_map>
#include <vector>

#include "../data_structures.h"
#include "../utils.h"
#include "GPU_KV_Buffer.h"

GPU_KV_Buffer::GPU_KV_Buffer(EngineConfig& engine_config,
                             ModelConfig& model_config)
    : engine_config_(engine_config), model_config_(model_config) {
    this->logger_ = init_logger(
        this->engine_config_.basic_config.log_level,
        "GPU_KV_Buffer" +
            std::to_string(this->engine_config_.basic_config.device));
    this->logger_->info("GPU_KV_Buffer Instantiated.");
};

void GPU_KV_Buffer::Init() {
    CUDA_CHECK(cudaSetDevice(this->engine_config_.basic_config.device));
    for (int64_t buffer_idx = 0;
         buffer_idx < this->engine_config_.gpu_buffer_config.num_k_buffer;
         buffer_idx++) {
        this->buffer_status_.push_back(0);
    }
};

void GPU_KV_Buffer::init_kv_buffer() {
    CUDA_CHECK(cudaSetDevice(this->engine_config_.basic_config.device));
    std::lock_guard<std::mutex> lock(this->mutex_);
    if (this->model_config_.model_type.find("deepseek") == std::string::npos) {
        this->logger_->debug(
            "Num K Buffers: {}",
            this->engine_config_.gpu_buffer_config.num_k_buffer);
        int64_t buffer_size =
            this->engine_config_.gpu_buffer_config.kv_buffer_num_tokens *
            this->model_config_.num_key_value_heads *
            this->model_config_.head_dim * 2;
        for (int64_t buffer_idx = 0;
             buffer_idx < this->engine_config_.gpu_buffer_config.num_k_buffer;
             buffer_idx++) {
            void* k_buffer = nullptr;
            void* v_buffer = nullptr;
            CUDA_CHECK(cudaMalloc(&k_buffer, buffer_size));
            CUDA_CHECK(cudaMalloc(&v_buffer, buffer_size));
            this->k_buffers_.push_back(k_buffer);
            this->v_buffers_.push_back(v_buffer);
        }
    } else {
        // FP8 Buffer
        // int64_t buffer_size =
        //     this->engine_config_.gpu_buffer_config.kv_buffer_num_tokens *
        //     this->model_config_.compressed_kv_dim * 2;
        int64_t buffer_size =
            this->engine_config_.gpu_buffer_config.kv_buffer_num_tokens *
            this->model_config_.compressed_kv_dim;
        for (int64_t buffer_idx = 0;
             buffer_idx < this->engine_config_.gpu_buffer_config.num_k_buffer;
             buffer_idx++) {
            void* k_buffer = nullptr;
            void* v_buffer = nullptr;
            CUDA_CHECK(cudaMalloc(&k_buffer, buffer_size));
            CUDA_CHECK(cudaMemset(k_buffer, 0, buffer_size));
            // CUDA_CHECK(cudaMalloc(&v_buffer, buffer_size));
            this->k_buffers_.push_back(k_buffer);
            this->v_buffers_.push_back(v_buffer);
        }
    }
}

std::optional<std::tuple<void*, void*, int64_t>>
GPU_KV_Buffer::acquireEmptyBuffer() {
    try {
        std::lock_guard<std::mutex> lock(this->mutex_);
        for (int64_t buffer_idx = 0; buffer_idx < this->buffer_status_.size();
             buffer_idx++) {
            if (this->buffer_status_[buffer_idx] == 0) {
                this->buffer_status_[buffer_idx] = 1;
                this->logger_->debug(
                    "acquireEmptyBuffer() Acquired KV buffer: {}", buffer_idx);
                return std::make_tuple(this->k_buffers_[buffer_idx],
                                       this->v_buffers_[buffer_idx],
                                       buffer_idx);
            }
        }
        return std::nullopt;
    } catch (...) {
        this->logger_->error("Failed to acquireEmptyBuffer()");
        throw std::runtime_error("Failed to acquireEmptyBuffer()");
    }
};

void GPU_KV_Buffer::releaseBuffer(int64_t layer_idx, int64_t micro_batch_idx) {
    std::lock_guard<std::mutex> lock(this->mutex_);
    std::string key =
        std::to_string(layer_idx) + "_" + std::to_string(micro_batch_idx);
    int64_t buffer_idx = this->kv_buffer_map_[key];
    this->buffer_status_[buffer_idx] = 0;
    this->kv_buffer_map_.erase(key);
    this->logger_->debug(
        "Released KV buffer for Layer: {}, Micro Batch Idx: {}, Buffer Idx: {}",
        layer_idx, micro_batch_idx, buffer_idx);
};

void GPU_KV_Buffer::kv_copy_complete(int64_t layer_idx, int64_t micro_batch_idx,
                                     int64_t buffer_idx) {
    std::lock_guard<std::mutex> lock(this->mutex_);
    std::string key =
        std::to_string(layer_idx) + "_" + std::to_string(micro_batch_idx);
    this->logger_->debug(
        "KV copy complete. Layer: {}, Micro Batch Idx: {}, "
        "Buffer Idx: {}, key: {}",
        layer_idx, micro_batch_idx, buffer_idx, key);
    this->kv_buffer_map_[key] = buffer_idx;
    this->cv_.notify_all();
};



torch::Tensor GPU_KV_Buffer::get_k(int64_t layer_idx, int64_t micro_batch_idx,
                                   std::vector<int64_t> tensor_shape) {
    CUDA_CHECK(cudaSetDevice(this->engine_config_.basic_config.device));
    /* Sync */
    std::string key =
        std::to_string(layer_idx) + "_" + std::to_string(micro_batch_idx);
    std::unique_lock<std::mutex> lock(this->mutex_);
    this->cv_.wait(lock, [this, key] {
        return this->kv_buffer_map_.find(key) != this->kv_buffer_map_.end();
    });
    int64_t buffer_idx = this->kv_buffer_map_[key];
    // Note: kv buffer is always in bfloat16.
    CUDA_CHECK(cudaSetDevice(this->engine_config_.basic_config.device));
    CUDA_CHECK(cudaStreamSynchronize(0));
    CUDA_CHECK(cudaDeviceSynchronize());
    auto option =
        torch::TensorOptions()
            // .dtype(torch::kBFloat16)
            .dtype(c10::kFloat8_e4m3fn)
            .device(torch::kCUDA, this->engine_config_.basic_config.device)
            .requires_grad(false);
            // .memory_format(torch::MemoryFormat::Contiguous); 
    // auto [bsz, kv_seq_len, num_kv_heads, head_dim] = tensor_shape;

    torch::Tensor k_tensor =
        torch::from_blob(this->k_buffers_[buffer_idx], tensor_shape,
                         // [](void*){},
                         option).clone();
    // Check if k_tensor contains nan
    if (torch::any(torch::isnan(k_tensor)).item<bool>()) {
        for (int i = 0; i < k_tensor.size(0); i++) {
            if (torch::any(torch::isnan(k_tensor[i])).item<bool>()) {
                this->logger_->error(
                    "k_tensor contains NaN values, rank: {}, i: {}, layer_idx: {}",
                    this->engine_config_.basic_config.device, i, layer_idx
                );
            }
        }
        // torch::save(k_tensor,
        //             "/workspace/rank_" +
        //                 std::to_string(this->engine_config_.basic_config.device) +
        //                 "_layer_" + std::to_string(layer_idx) +
        //                 "_micro_batch_" + std::to_string(micro_batch_idx) +
        //                 "_k.pt");
        this->logger_->error("k_tensor contains NaN values, rank: {}",
                             this->engine_config_.basic_config.device);
        throw std::runtime_error("k_tensor contains NaN values.");
    }
    if (this->model_config_.model_type.find("deepseek") == std::string::npos) {
        k_tensor = k_tensor.permute({0, 2, 1, 3});
    }
    return k_tensor;
};

torch::Tensor GPU_KV_Buffer::get_v(int64_t layer_idx, int64_t micro_batch_idx,
                                   std::vector<int64_t> tensor_shape) {
    /* Sync */
    CUDA_CHECK(cudaSetDevice(this->engine_config_.basic_config.device));
    std::string key =
        std::to_string(layer_idx) + "_" + std::to_string(micro_batch_idx);
    std::unique_lock<std::mutex> lock(this->mutex_);
    this->cv_.wait(lock, [this, key] {
        return this->kv_buffer_map_.find(key) != this->kv_buffer_map_.end();
    });
    int64_t buffer_idx = this->kv_buffer_map_[key];
    auto option =
        torch::TensorOptions()
            .dtype(this->engine_config_.basic_config.dtype_torch)
            .device(torch::kCUDA, this->engine_config_.basic_config.device)
            .requires_grad(false)
            .memory_format(torch::MemoryFormat::Contiguous);
    torch::Tensor v_tensor =
        torch::from_blob(this->v_buffers_[buffer_idx], tensor_shape,
                         // [](void*){},
                         option);
    return v_tensor.permute({0, 2, 1, 3});
};

GPU_KV_Buffer::~GPU_KV_Buffer() {
    // for (int64_t buffer_idx = 0;
    //      buffer_idx < this->engine_config_.gpu_buffer_config.num_kv_buffer;
    //      buffer_idx++) {
    //     cudaFree(this->k_buffers_[buffer_idx]);
    //     // v_buffer can be nullptr
    //     if (this->v_buffers_[buffer_idx] != nullptr) {
    //         cudaFree(this->v_buffers_[buffer_idx]);
    //     }
    // }
};

void GPU_KV_Buffer::clear_kv_buffer() {
    // Mark all buffers as empty.
    std::lock_guard<std::mutex> lock(this->mutex_);
    for (int64_t buffer_idx = 0;
         buffer_idx < this->engine_config_.gpu_buffer_config.num_k_buffer;
         buffer_idx++) {
        this->buffer_status_[buffer_idx] = 0;
    }
    this->kv_buffer_map_.clear();
}
