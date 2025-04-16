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

#include "spdlog/spdlog.h"
#include <c10/util/BFloat16.h>
#include <memory>
#include <string>
#include <torch/extension.h>
#include <torch/torch.h>
#include <unordered_map>
#include <vector>

#include "../data_structures.h"
#include "../utils.h"
#include "KV_Storage.h"
#include "tqdm.hpp"
#include <random>
#include <cstdint>
#include <cstring>

typedef uint16_t bf16;

bf16 float_to_bf16(float f) {
    uint32_t float_bits;
    std::memcpy(&float_bits, &f, sizeof(float));
    return static_cast<bf16>(float_bits >> 16);
}

void fill_with_random_bf16_uniform(void* k_ptr, size_t per_layer_storage_size, 
                                   float min_val = -1.0f, float max_val = 1.0f) {
    size_t num_elements = per_layer_storage_size / sizeof(bf16);
    bf16* bf16_ptr = static_cast<bf16*>(k_ptr);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(min_val, max_val);
    
    for (size_t i = 0; i < num_elements; ++i) {
        bf16_ptr[i] = float_to_bf16(dist(gen));
    }
}

// Normal distribution variant
void fill_with_random_bf16_normal(void* k_ptr, size_t per_layer_storage_size, 
                                 float mean = 0.0f, float stddev = 1.0f) {
    size_t num_elements = per_layer_storage_size / sizeof(bf16);
    bf16* bf16_ptr = static_cast<bf16*>(k_ptr);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(mean, stddev);
    
    for (size_t i = 0; i < num_elements; ++i) {
        bf16_ptr[i] = float_to_bf16(dist(gen));
    }
}

KV_Storage::KV_Storage(EngineConfig& engine_config, ModelConfig& model_config,
                       DtoH_Engine& d2h_engine)
    : engine_config_(engine_config),
      model_config_(model_config),
      d2h_engine_(d2h_engine),
      per_element_mutex_(model_config.num_hidden_layers *
                         engine_config.kv_storage_config.num_host_slots) {
    try {
        this->logger_ = init_logger(
            this->engine_config_.basic_config.log_level,
            "KV_Storage" +
                std::to_string(this->engine_config_.basic_config.device));
        CUDA_CHECK(cudaSetDevice(this->engine_config_.basic_config.device));
        this->logger_->info("Starting KV_Storage Initialization.");
        /* Reserve Pinned Memory for K and V. */
        const auto& storage_size =
            this->engine_config_.kv_storage_config.storage_byte_size;
        auto per_layer_storage_size =
            storage_size / this->model_config_.num_hidden_layers;

        auto bar = tq::trange(this->model_config_.num_hidden_layers);
        bar.set_prefix("Allocating Pinned Memory for KV cache");
        if (this->model_config_.model_type.find("deepseek") ==
            std::string::npos) {
            for (auto layer_idx : bar) {
                void* k_ptr = nullptr;
                void* v_ptr = nullptr;
                CUDA_CHECK(cudaHostAlloc(&k_ptr, per_layer_storage_size,
                                         cudaHostAllocDefault));
                CUDA_CHECK(cudaHostAlloc(&v_ptr, per_layer_storage_size,
                                         cudaHostAllocDefault));
                this->k_pinned_memory.push_back(k_ptr);
                this->v_pinned_memory.push_back(v_ptr);
            }
        } else {
            for (auto layer_idx : bar) {
                void* k_ptr = nullptr;
                CUDA_CHECK(cudaHostAlloc(&k_ptr, per_layer_storage_size,
                                         cudaHostAllocDefault));
                // memset random generated number to k_ptr
                // fill_with_random_bf16_uniform(k_ptr, per_layer_storage_size, -0.5f, 0.5f);
                // memset(k_ptr, 0, per_layer_storage_size);
                this->k_pinned_memory.push_back(k_ptr);
            }
        }
        this->logger_->info("KV Storage Pinned Memory Allocated.");
    } catch (...) {
        this->logger_->debug(
            "KV_Storage: Failed to update K and V to the storage.");
        throw std::runtime_error(
            "KV_Storage: Failed to update K and V to the storage.");
    }
};

void KV_Storage::Init() {
    try {
        if (this->model_config_.model_type.find("deepseek") ==
            std::string::npos) {
            /* Slicing reserved K and V to slots. */
            auto num_slots =
                this->engine_config_.kv_storage_config.num_host_slots;
            this->logger_->debug("num_slots: {}", num_slots);
            auto slot_byte_size =
                this->engine_config_.kv_storage_config.slot_byte_size;
            this->logger_->debug("slot_byte_size: {}", slot_byte_size);
            this->k_storage.resize(num_slots);
            this->v_storage.resize(num_slots);
            this->empty_slots.clear();
            this->query_idx_to_slot_idx_map.clear();
            for (int64_t slot_idx = 0; slot_idx < num_slots; slot_idx++) {
                this->k_storage[slot_idx].resize(
                    this->model_config_.num_hidden_layers);
                this->v_storage[slot_idx].resize(
                    this->model_config_.num_hidden_layers);
                this->empty_slots.insert(slot_idx);
                for (int64_t layer_idx = 0;
                     layer_idx < this->model_config_.num_hidden_layers;
                     layer_idx++) {
                    this->k_storage[slot_idx][layer_idx].start_ptr =
                        this->k_pinned_memory[layer_idx] +
                        slot_idx * slot_byte_size;
                    this->v_storage[slot_idx][layer_idx].start_ptr =
                        this->v_pinned_memory[layer_idx] +
                        slot_idx * slot_byte_size;
                    this->k_storage[slot_idx][layer_idx].used_byte_size = 0;
                    this->v_storage[slot_idx][layer_idx].used_byte_size = 0;
                    this->k_storage[slot_idx][layer_idx].num_tokens = 0;
                    this->v_storage[slot_idx][layer_idx].num_tokens = 0;
                }
            }
        } else {
            /* Slicing reserved K and V to slots. */
            auto num_slots =
                this->engine_config_.kv_storage_config.num_host_slots;
            this->logger_->debug("num_slots: {}", num_slots);
            auto slot_byte_size =
                this->engine_config_.kv_storage_config.slot_byte_size;
            this->logger_->debug("slot_byte_size: {}", slot_byte_size);
            this->k_storage.resize(num_slots);
            // this->v_storage.resize(num_slots);
            this->empty_slots.clear();
            this->query_idx_to_slot_idx_map.clear();
            for (int64_t slot_idx = 0; slot_idx < num_slots; slot_idx++) {
                this->k_storage[slot_idx].resize(
                    this->model_config_.num_hidden_layers);
                this->empty_slots.insert(slot_idx);
                for (int64_t layer_idx = 0;
                     layer_idx < this->model_config_.num_hidden_layers;
                     layer_idx++) {
                    this->k_storage[slot_idx][layer_idx].start_ptr =
                        this->k_pinned_memory[layer_idx] +
                        slot_idx * slot_byte_size;
                    this->k_storage[slot_idx][layer_idx].used_byte_size = 0;
                    this->k_storage[slot_idx][layer_idx].num_tokens = 0;
                }
            }
        }
        this->logger_->debug("KV_Storage Initialized.");
    } catch (const c10::Error& e) {
        this->logger_->debug("KV_Storage: CUDA/PyTorch error: {}", e.what());
        throw std::runtime_error(e.what());
    }
    // Catch CUDA runtime errors
    catch (const cudaError_t& err) {
        this->logger_->debug("KV_Storage: CUDA runtime error: {}",
                             cudaGetErrorString(err));
        throw std::runtime_error(cudaGetErrorString(err));
    }
    // Catch standard C++ exceptions
    catch (const std::exception& e) {
        this->logger_->debug(
            "KV_Storage: Failed to update K and V to the storage. Error: {}",
            e.what());
        throw std::runtime_error(
            "KV_Storage: Failed to update K and V to the storage.");
    }
    // Catch any other unexpected errors
    catch (...) {
        this->logger_->debug(
            "KV_Storage: Failed to update K and V to the storage.");
        throw std::runtime_error(
            "KV_Storage: Failed to update K and V to the storage.");
    }
};

void KV_Storage::offload(int64_t layer_idx,
                         std::vector<int64_t> query_global_idx, torch::Tensor k,
                         torch::Tensor v) {
    SAFE_CALL(
        [&]() {
            auto worker = std::thread(&KV_Storage::offload_helper_, this,
                                      layer_idx, query_global_idx, k, v);
            worker.detach();
        },
        this->logger_);
};

void KV_Storage::offload_helper_(int64_t layer_idx,
                                 std::vector<int64_t> query_global_idx,
                                 torch::Tensor k, torch::Tensor v) {
    this->logger_->debug("Offloading layer_idx: {}", layer_idx);
    try {
        if (this->model_config_.model_type.find("deepseek") ==
            std::string::npos) {
            /* Step 1: Permute k and v in the device. */
            int64_t bsz = k.size(0);
            int64_t seq_len = k.size(2);
            int64_t k_seq_byte_size =
                k.size(1) * k.size(2) * k.size(3) * k.element_size();
            int64_t v_seq_byte_size =
                v.size(1) * v.size(2) * v.size(3) * v.element_size();
            k = k.permute({0, 2, 1, 3}).contiguous();
            v = v.permute({0, 2, 1, 3}).contiguous();

            /* Launch DMA per sequence. */
            this->logger_->debug("query_global_idx.size(): {}",
                                 query_global_idx.size());
            for (int64_t i = 0; i < query_global_idx.size(); i++) {
                // this->logger_->debug("{}/{}", i, query_global_idx.size());
                auto query_idx = query_global_idx[i];
                auto device_k_ptr = k.data_ptr() + i * k_seq_byte_size;
                auto device_v_ptr = v.data_ptr() + i * v_seq_byte_size;

                /* Get an empty slot for new query. */
                int64_t slot_idx = -1;
                if (layer_idx == 0) {
                    std::lock_guard<std::mutex> lock(this->mutex_);
                    if (this->empty_slots.empty()) {
                        this->logger_->debug(
                            "KV_Storage: No empty slot available.");
                        throw std::runtime_error(
                            "KV_Storage: No empty slot available.");
                    }
                    slot_idx = *this->empty_slots.begin();
                    this->empty_slots.erase(slot_idx);
                    this->query_idx_to_slot_idx_map[query_idx] = slot_idx;
                } else {
                    std::lock_guard<std::mutex> lock(this->mutex_);
                    slot_idx = this->query_idx_to_slot_idx_map[query_idx];
                }

                auto host_k_ptr =
                    this->k_storage[slot_idx][layer_idx].start_ptr;
                auto host_v_ptr =
                    this->v_storage[slot_idx][layer_idx].start_ptr;

                if (host_k_ptr == nullptr || host_v_ptr == nullptr) {
                    if (host_k_ptr == nullptr) {
                        this->logger_->debug(
                            "KV_Storage: host_k_ptr is nullptr.");
                    }
                    if (host_v_ptr == nullptr) {
                        this->logger_->debug(
                            "KV_Storage: host_v_ptr is nullptr.");
                    }
                    throw std::runtime_error(
                        "KV_Storage: Host memory ptr is nullptr.");
                }
                {
                    std::lock_guard<std::mutex> lock(
                        this->per_element_mutex_
                            [slot_idx * this->model_config_.num_hidden_layers +
                             layer_idx]);
                    this->d2h_engine_.submit_to_queue_B(/* Blocking and sync
                                                           copy function call */
                                                        host_k_ptr,
                                                        device_k_ptr,
                                                        k_seq_byte_size);
                    // torch::cuda::synchronize(this->engine_config_.basic_config.device);
                    this->k_storage[slot_idx][layer_idx].used_byte_size =
                        k_seq_byte_size;
                    this->k_storage[slot_idx][layer_idx].num_tokens = seq_len;
                }
                {
                    std::lock_guard<std::mutex> lock(
                        this->per_element_mutex_
                            [slot_idx * this->model_config_.num_hidden_layers +
                             layer_idx]);
                    this->d2h_engine_.submit_to_queue_B(/* Blocking and sync
                                                           copy function call */
                                                        host_v_ptr,
                                                        device_v_ptr,
                                                        v_seq_byte_size);

                    this->v_storage[slot_idx][layer_idx].used_byte_size =
                        v_seq_byte_size;
                    this->v_storage[slot_idx][layer_idx].num_tokens = seq_len;
                }
                {
                    std::lock_guard<std::mutex> lock(this->mutex_);
                    this->query_idx_to_slot_idx_map[query_idx] = slot_idx;
                }
            }
            this->logger_->debug("Offloading layer_idx: {} completed.",
                                 layer_idx);
        } else {
            /* Step 1: Permute k and v in the device. */
            int64_t bsz = k.size(0);
            int64_t seq_len = k.size(1);
            int64_t k_seq_byte_size = k.size(1) * k.size(2) * k.element_size();
            k = k.contiguous();

            /* Launch DMA per sequence. */
            this->logger_->debug("query_global_idx.size(): {}",
                                 query_global_idx.size());
            for (int64_t i = 0; i < query_global_idx.size(); i++) {
                // this->logger_->debug("{}/{}", i, query_global_idx.size());
                auto query_idx = query_global_idx[i];
                auto device_k_ptr = k.data_ptr() + i * k_seq_byte_size;

                /* Get an empty slot for new query. */
                int64_t slot_idx = -1;
                if (layer_idx == 0) {
                    std::lock_guard<std::mutex> lock(this->mutex_);
                    if (this->empty_slots.empty()) {
                        this->logger_->debug(
                            "KV_Storage: No empty slot available.");
                        throw std::runtime_error(
                            "KV_Storage: No empty slot available.");
                    }
                    slot_idx = *this->empty_slots.begin();
                    this->empty_slots.erase(slot_idx);
                    this->query_idx_to_slot_idx_map[query_idx] = slot_idx;
                } else {
                    std::lock_guard<std::mutex> lock(this->mutex_);
                    slot_idx = this->query_idx_to_slot_idx_map[query_idx];
                }

                // this->logger_->debug("slot_idx: {}", slot_idx);
                auto host_k_ptr =
                    this->k_storage[slot_idx][layer_idx].start_ptr;

                {
                    std::lock_guard<std::mutex> lock(
                        this->per_element_mutex_
                            [slot_idx * this->model_config_.num_hidden_layers +
                             layer_idx]);
                    this->d2h_engine_.submit_to_queue_B(/* Blocking and sync
                                                           copy function call */
                                                        host_k_ptr,
                                                        device_k_ptr,
                                                        k_seq_byte_size);
                    // torch::cuda::synchronize(this->engine_config_.basic_config.device);
                    this->k_storage[slot_idx][layer_idx].used_byte_size =
                        k_seq_byte_size;
                    this->k_storage[slot_idx][layer_idx].num_tokens = seq_len;
                }
                {
                    std::lock_guard<std::mutex> lock(this->mutex_);
                    this->query_idx_to_slot_idx_map[query_idx] = slot_idx;
                }
            }
            this->logger_->debug("Offloading layer_idx: {} completed.",
                                 layer_idx);
        }
    } catch (const c10::Error& e) {
        this->logger_->debug(
            "KV_Storage - offload_helper_(): CUDA/PyTorch error: {}", e.what());
        throw std::runtime_error(e.what());
    }
    // Catch CUDA runtime errors
    catch (const cudaError_t& err) {
        this->logger_->debug(
            "KV_Storage - offload_helper_(): CUDA runtime error: {}",
            cudaGetErrorString(err));
        throw std::runtime_error(cudaGetErrorString(err));
    }
    // Catch standard C++ exceptions
    catch (const std::exception& e) {
        this->logger_->debug(
            "KV_Storage - offload_helper_(): Failed to offload K "
            "and V to the storage. Error: {}",
            e.what());
        throw std::runtime_error(
            "KV_Storage - offload_helper_(): Failed to "
            "offload K and V to the storage.");
    }
    // Catch any other unexpected errors
    catch (...) {
        this->logger_->debug(
            "KV_Storage - offload_helper_(): Failed to offload K "
            "and V to the storage.");
        throw std::runtime_error(
            "KV_Storage - offload_helper_(): Failed to "
            "offload K and V to the storage.");
    }
};

void KV_Storage::update(int64_t layer_idx,
                        std::vector<int64_t> query_global_indices,
                        torch::Tensor k, torch::Tensor v) {
    try {
        auto worker = std::thread(&KV_Storage::update_helper_, this, layer_idx,
                                  query_global_indices, k, v);
        worker.detach();
    } catch (const std::exception& e) {
        this->logger_->debug(
            "KV_Storage update(): Failed to update K and V to the "
            "storage. Error: {}",
            e.what());
        throw std::runtime_error(
            "KV_Storage update(): Failed to update K and V to the storage.");
    }
};

void KV_Storage::update_helper_(int64_t layer_idx,
                                std::vector<int64_t> query_global_indices,
                                torch::Tensor k, torch::Tensor v) {
    try {
        CUDA_CHECK(cudaSetDevice(this->engine_config_.basic_config.device));
        if (this->model_config_.model_type.find("deepseek") ==
            std::string::npos) {
            k = k.contiguous();
            v = v.contiguous();
            // int64_t token_byte_size = this->model_config_.head_dim *
            // this->model_config_.num_key_value_heads * k.element_size();
            int64_t k_token_byte_size =
                k.size(1) * k.size(2) * k.size(3) * k.element_size();
            int64_t v_token_byte_size =
                v.size(1) * v.size(2) * v.size(3) * v.element_size();
            this->logger_->debug("k_token_byte_size: {}", k_token_byte_size);
            this->logger_->debug("v_token_byte_size: {}", v_token_byte_size);
            for (int64_t i = 0;
                 i < static_cast<int64_t>(query_global_indices.size()); i++) {
                auto query_idx = query_global_indices[i];
                auto device_k_ptr = k.data_ptr() + i * k_token_byte_size;
                auto device_v_ptr = v.data_ptr() + i * v_token_byte_size;
                int64_t slot_idx;
                {
                    std::lock_guard<std::mutex> lock(this->mutex_);
                    slot_idx = this->query_idx_to_slot_idx_map[query_idx];
                }
                auto host_k_ptr =
                    this->k_storage[slot_idx][layer_idx].start_ptr +
                    this->k_storage[slot_idx][layer_idx].used_byte_size;
                auto host_v_ptr =
                    this->v_storage[slot_idx][layer_idx].start_ptr +
                    this->v_storage[slot_idx][layer_idx].used_byte_size;
                {
                    // std::lock_guard<std::mutex>
                    // lock(this->per_element_mutex_[slot_idx
                    // * this->model_config_.num_hidden_layers + layer_idx]);
                    // this->d2h_engine_.submit_to_queue_B(
                    // 	host_k_ptr, device_k_ptr, k_token_byte_size
                    // );
                    CUDA_CHECK(cudaMemcpyAsync(
                        host_k_ptr, device_k_ptr, k_token_byte_size,
                        cudaMemcpyDeviceToHost, this->d2h_engine_.DtoH_stream));
                    this->k_storage[slot_idx][layer_idx].used_byte_size +=
                        k_token_byte_size;
                    this->k_storage[slot_idx][layer_idx].num_tokens += 1;
                }
                {
                    // std::lock_guard<std::mutex>
                    // lock(this->per_element_mutex_[slot_idx
                    // * this->model_config_.num_hidden_layers + layer_idx]);
                    // this->d2h_engine_.submit_to_queue_B(
                    // 	host_v_ptr, device_v_ptr, v_token_byte_size
                    // );
                    CUDA_CHECK(cudaMemcpyAsync(
                        host_v_ptr, device_v_ptr, v_token_byte_size,
                        cudaMemcpyDeviceToHost, this->d2h_engine_.DtoH_stream));
                    this->v_storage[slot_idx][layer_idx].used_byte_size +=
                        v_token_byte_size;
                    this->v_storage[slot_idx][layer_idx].num_tokens += 1;
                }
            }
            CUDA_CHECK(cudaStreamSynchronize(this->d2h_engine_.DtoH_stream));
        } else {
            if ((k.dtype() != torch::kBFloat16)) {
                throw std::runtime_error(
                    "KV_Storage update_helper_(): k.dtype and "
                    "v.dtype should be torch::kBFloat16.");
            }
            k = k.contiguous();
            // int64_t token_byte_size = this->model_config_.head_dim *
            // this->model_config_.num_key_value_heads * k.element_size();
            int64_t k_token_byte_size =
                k.size(1) * k.size(2) * k.element_size();
            this->logger_->debug("k_token_byte_size: {}", k_token_byte_size);
            for (int64_t i = 0; i < query_global_indices.size(); i++) {
                auto query_idx = query_global_indices[i];
                auto device_k_ptr = k.data_ptr() + i * k_token_byte_size;
                int64_t slot_idx;
                {
                    std::lock_guard<std::mutex> lock(this->mutex_);
                    slot_idx = this->query_idx_to_slot_idx_map[query_idx];
                }
                auto host_k_ptr =
                    this->k_storage[slot_idx][layer_idx].start_ptr +
                    this->k_storage[slot_idx][layer_idx].used_byte_size;
                {
                    std::lock_guard<std::mutex> lock(
                        this->per_element_mutex_
                            [slot_idx * this->model_config_.num_hidden_layers +
                             layer_idx]);
                    this->d2h_engine_.submit_to_queue_B(
                        host_k_ptr, device_k_ptr, k_token_byte_size);
                    this->k_storage[slot_idx][layer_idx].used_byte_size +=
                        k_token_byte_size;
                    this->k_storage[slot_idx][layer_idx].num_tokens += 1;
                }
            }
        }
    }
    // Catch PyTorch/CUDA specific exceptions
    catch (const c10::Error& e) {
        this->logger_->debug(
            "KV_Storage update_helper_(): CUDA/PyTorch error: {}", e.what());
        throw;
    }
    // Catch CUDA runtime errors
    catch (const cudaError_t& err) {
        this->logger_->debug(
            "KV_Storage update_helper_(): CUDA runtime error: {}",
            cudaGetErrorString(err));
        throw std::runtime_error(cudaGetErrorString(err));
    }
    // Catch standard C++ exceptions
    catch (const std::exception& e) {
        this->logger_->debug(
            "KV_Storage update_helper_(): Failed to update K and "
            "V to the storage. Error: {}",
            e.what());
        throw std::runtime_error(
            "KV_Storage: Failed to update K and V to the storage.");
    }
    // Catch any other unexpected errors
    catch (...) {
        this->logger_->debug(
            "KV_Storage update_helper_(): Failed to update K and "
            "V to the storage.");
        throw std::runtime_error(
            "KV_Storage: Failed to update K and V to the storage.");
    }
};

std::vector<c10::BFloat16*> KV_Storage::get_k_ptrs(int64_t layer_idx,
                                                   std::vector<int64_t> batch) {
    try {
        std::vector<c10::BFloat16*> k_ptrs;
        for (int64_t i = 0; i < static_cast<int64_t>(batch.size()); i++) {
            auto query_idx = batch[i];
            int64_t slot_idx = -1;
            {
                std::lock_guard<std::mutex> lock(this->mutex_);
                slot_idx = this->query_idx_to_slot_idx_map[query_idx];
            }
            c10::BFloat16* k_ptr = nullptr;
            {
                std::lock_guard<std::mutex> lock(
                    this->per_element_mutex_[slot_idx * this->model_config_
                                                            .num_hidden_layers +
                                             layer_idx]);
                k_ptr = static_cast<c10::BFloat16*>(
                    this->k_storage[slot_idx][layer_idx].start_ptr);
            }
            k_ptrs.push_back(k_ptr);
        }
        return k_ptrs;
    }
    // Catch PyTorch/CUDA specific exceptions
    catch (const c10::Error& e) {
        this->logger_->debug("KV_Storage get_k_ptrs(): CUDA/PyTorch error: {}",
                             e.what());
        throw;
    }
    // Catch CUDA runtime errors
    catch (const cudaError_t& err) {
        this->logger_->debug("KV_Storage get_k_ptrs(): CUDA runtime error: {}",
                             cudaGetErrorString(err));
        throw std::runtime_error(cudaGetErrorString(err));
    }
    // Catch standard C++ exceptions
    catch (const std::exception& e) {
        this->logger_->debug(
            "KV_Storage get_k_ptrs(): Failed to update K and V to "
            "the storage. Error: {}",
            e.what());
        throw std::runtime_error(
            "KV_Storage: Failed to update K and V to the storage.");
    }
    // Catch any other unexpected errors
    catch (...) {
        this->logger_->debug(
            "KV_Storage get_k_ptrs(): Failed to update K and V to the "
            "storage.");
        throw std::runtime_error(
            "KV_Storage: Failed to update K and V to the storage.");
    }
};

std::vector<c10::BFloat16*> KV_Storage::get_v_ptrs(int64_t layer_idx,
                                                   std::vector<int64_t> batch) {
    try {
        std::vector<c10::BFloat16*> v_ptrs;
        for (int64_t i = 0; i < static_cast<int64_t>(batch.size()); i++) {
            auto query_idx = batch[i];
            int64_t slot_idx = -1;
            {
                std::lock_guard<std::mutex> lock(this->mutex_);
                slot_idx = this->query_idx_to_slot_idx_map[query_idx];
            }
            c10::BFloat16* v_ptr = nullptr;
            {
                std::lock_guard<std::mutex> lock(
                    this->per_element_mutex_[slot_idx * this->model_config_
                                                            .num_hidden_layers +
                                             layer_idx]);
                v_ptr = static_cast<c10::BFloat16*>(
                    this->v_storage[slot_idx][layer_idx].start_ptr);
            }
            v_ptrs.push_back(v_ptr);
        }
        return v_ptrs;
    }
    // Catch PyTorch/CUDA specific exceptions
    catch (const c10::Error& e) {
        this->logger_->debug("KV_Storage get_v_ptrs(): CUDA/PyTorch error: {}",
                             e.what());
        throw;
    }
    // Catch CUDA runtime errors
    catch (const cudaError_t& err) {
        this->logger_->debug("KV_Storage get_v_ptrs(): CUDA runtime error: {}",
                             cudaGetErrorString(err));
        throw std::runtime_error(cudaGetErrorString(err));
    }
    // Catch standard C++ exceptions
    catch (const std::exception& e) {
        this->logger_->debug(
            "KV_Storage get_v_ptrs(): Failed to update K and V to "
            "the storage. Error: {}",
            e.what());
        throw std::runtime_error(
            "KV_Storage: Failed to update K and V to the storage.");
    }
    // Catch any other unexpected errors
    catch (...) {
        this->logger_->debug(
            "KV_Storage get_v_ptrs(): Failed to update K and V to the "
            "storage.");
        throw std::runtime_error(
            "KV_Storage: Failed to update K and V to the storage.");
    }
};

void KV_Storage::clear_kv_storage() {
    // Clear K and V storage. Mark all slots as empty.
    try {
        std::lock_guard<std::mutex> lock(this->mutex_);
        this->empty_slots.clear();
        this->query_idx_to_slot_idx_map.clear();
        // distinguish between deepseek and other models
        if (this->model_config_.model_type.find("deepseek") ==
            std::string::npos) {
            for (int64_t slot_idx = 0;
                 slot_idx <
                 this->engine_config_.kv_storage_config.num_host_slots;
                 slot_idx++) {
                for (int64_t layer_idx = 0;
                     layer_idx < this->model_config_.num_hidden_layers;
                     layer_idx++) {
                    this->k_storage[slot_idx][layer_idx].used_byte_size = 0;
                    this->k_storage[slot_idx][layer_idx].num_tokens = 0;
                    this->v_storage[slot_idx][layer_idx].used_byte_size = 0;
                    this->v_storage[slot_idx][layer_idx].num_tokens = 0;
                    this->empty_slots.insert(slot_idx);
                }
            }
        } else {
            for (int64_t slot_idx = 0;
                 slot_idx <
                 this->engine_config_.kv_storage_config.num_host_slots;
                 slot_idx++) {
                for (int64_t layer_idx = 0;
                     layer_idx < this->model_config_.num_hidden_layers;
                     layer_idx++) {
                    this->k_storage[slot_idx][layer_idx].used_byte_size = 0;
                    this->k_storage[slot_idx][layer_idx].num_tokens = 0;
                    this->empty_slots.insert(slot_idx);
                }
            }
        }
        this->logger_->debug(
            "KV_Storage clear_kv_storage(): K and V storage cleared.");
    } catch (const std::exception& e) {
        this->logger_->debug(
            "KV_Storage clear_kv_storage(): Failed to clear K and "
            "V storage. Error: {}",
            e.what());
        throw std::runtime_error(
            "KV_Storage clear_kv_storage(): Failed to clear K and V storage.");
    }
}


// Function to create a fake kv storage for test.
// Fill the storage with random data.
// Fill the query_idx_to_slot_idx_map with one-to-one mapping.
// There is no empty slots.
// We only care deepseek models.
void KV_Storage::create_fake_kv_storage() {
    try {
        std::lock_guard<std::mutex> lock(this->mutex_);
        // Set query_idx_to_slot_idx_map
        for (int64_t slot_idx = 0;
             slot_idx < this->engine_config_.kv_storage_config.num_host_slots;
             slot_idx++) {
            for (int64_t layer_idx = 0;
                 layer_idx < this->model_config_.num_hidden_layers;
                 layer_idx++) {
                this->k_storage[slot_idx][layer_idx].used_byte_size = 14000 * 576 * 2;
                this->k_storage[slot_idx][layer_idx].num_tokens = 13000;
                this->query_idx_to_slot_idx_map[slot_idx] = slot_idx;
            }
        }
    } catch (const std::exception& e) {
        this->logger_->debug(
            "KV_Storage create_fake_kv_storage(): Failed to create "
            "fake kv storage. Error: {}",
            e.what());
        throw std::runtime_error(
            "KV_Storage create_fake_kv_storage(): Failed to create fake kv "
            "storage.");
    }
}