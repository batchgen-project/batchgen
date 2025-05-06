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

constexpr float FP8_MAX = 448.0f;
std::tuple<at::Tensor, at::Tensor> per_token_quant(torch::Tensor x) {
    /* 
     * Quantize a [bsz, seq, 576] BF16 tensor to FP8 per-token.
     * Args:
     *   x: Input tensor of shape [bsz, seq, 576] with dtype bfloat16
     * Returns:
     *   q: Quantized tensor [bsz, seq, 576] with dtype float8_e4m3fn
     *   s: Scale factors [bsz, seq] with dtype float32
     */
    // TORCH_CHECK(x.scalar_type() == at::ScalarType::BFloat16, 
    //             "Input tensor must be of dtype BFloat16");
    // TORCH_CHECK(x.size(-1) == 576, 
    //             "Last dimension of input tensor must be 576");
    // TORCH_CHECK(x.is_contiguous(), 
    //             "Input tensor must be contiguous");
    // TORCH_CHECK(x.dim() == 3, 
    //             "Input tensor must have 3 dimensions");
    
    const auto device = x.device();
    const auto bsz = x.size(0);
    const auto seq_len = x.size(1);
    const auto dim = x.size(2);
    const auto M = bsz * seq_len;
    
    // Cast to float32 and reshape for reduction
    auto x_flat = x.view({M, dim}).to(at::ScalarType::Float);
    
    // Compute max absolute value per token
    auto amax = at::amax(at::abs(x_flat), /*dim=*/1);
    amax = at::clamp(amax, /*min=*/1e-6f);
    
    // Compute scales
    auto scale = amax / FP8_MAX;
    
    // Scale and cast to fp8
    auto y = x_flat / scale.unsqueeze(1);
    auto q = y.to(at::ScalarType::Float8_e4m3fn);
    
    // Reshape output tensors
    q = q.view({bsz, seq_len, dim});
    scale = scale.view({bsz, seq_len});
    
    return std::make_tuple(q, scale);
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
                memset(k_ptr, 999, per_layer_storage_size); 
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

torch::Tensor KV_Storage::get_k_quantize_scale(
    int64_t layer_idx, std::vector<int64_t> cur_batch, int64_t padding_length) 
{
    CUDA_CHECK(cudaSetDevice(this->engine_config_.basic_config.device));
    torch::Device device(torch::kCUDA, this->engine_config_.basic_config.device);
    auto opt = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(device)  // Use the device object instead of a raw integer
        .requires_grad(false);
    auto k_quantize_scale = torch::ones({cur_batch.size(), padding_length}, opt);
                                            
    for (int64_t i = 0; i < static_cast<int64_t>(cur_batch.size()); i++) {
        auto query_idx = cur_batch[i];
        int64_t slot_idx = -1;
        {
            std::lock_guard<std::mutex> lock(this->mutex_);
            slot_idx = this->query_idx_to_slot_idx_map[query_idx];
        }
        auto scale = this->k_storage[slot_idx][layer_idx].quantize_scale;
        // log scale shape
        // this->logger_->info("scale shape: {}",
        //                      get_tensor_shape(scale));
        int64_t scale_size = scale.size(1);
        // copy scale to k_quantize_scale[i]'s first scale's size
        if(scale_size > padding_length) {
            this->logger_->info("scale_size: {}, padding_length: {}",
                                 scale_size, padding_length);
            throw std::runtime_error("scale_size > padding_length");
        }
        k_quantize_scale.index({i, torch::indexing::Slice(0, scale_size)}) = scale.index({0, torch::indexing::Slice(0, scale_size)}); 
        // Log padding_length, and the first 5 elements of k_quantize_scale[i]
        // this->logger_->info("padding_length: {}, k_quantize_scale[{}]: {}{}{}{}{}",
        //                      padding_length, i,
        //                      k_quantize_scale[i][-1].item<float>(),
        //                      k_quantize_scale[i][-2].item<float>(),
        //                      k_quantize_scale[i][-3].item<float>(),
        //                      k_quantize_scale[i][-4].item<float>(),
        //                      k_quantize_scale[i][-5].item<float>());
    }
    // Check k_quantize_scale has nan values
    if (torch::any(torch::isnan(k_quantize_scale)).item<bool>()){
        for (int64_t i = 0; i < k_quantize_scale.size(0); i++) {
            if(torch::any(torch::isnan(k_quantize_scale[i])).item<bool>()) {
                this->logger_->debug("k_quantize_scale[{}] has nan values, rank: {}, layer_idx: {}",
                                     i, this->engine_config_.basic_config.device,
                                     layer_idx);
            }
        }
        throw std::runtime_error("k_quantize_scale has nan values");
    }
    // CUDA_CHECK(cudaStreamSynchronize(0));
    // CUDA_CHECK(cudaDeviceSynchronize());
    return k_quantize_scale;
}
            
            
void KV_Storage::offload(
    int64_t layer_idx,
    std::vector<int64_t> query_global_idx, 
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor attention_mask)
{
    // SAFE_CALL(
    //     [&]() {
    //         auto worker = std::thread(&KV_Storage::offload_helper_, this,
    //                                   layer_idx, query_global_idx, k, v);
    //         // worker.detach();
    //         worker.join();
    //     },
    //     this->logger_);
    // Check if k has nan values
    CUDA_CHECK(cudaStreamSynchronize(0));
    // if (torch::any(torch::isnan(k)).item<bool>()){
    //     for (int64_t i = 0; i < k.size(0); i++) {
    //         if(torch::any(torch::isnan(k[i])).item<bool>()) {
    //             this->logger_->debug("k[{}] has nan values, rank: {}, layer_idx: {}",
    //                                  i, this->engine_config_.basic_config.device,
    //                                  layer_idx);
    //         }
    //     }
    //     throw std::runtime_error("k has nan values");
    // }
    this->offload_helper_(layer_idx, query_global_idx, k, v, attention_mask);
};

void KV_Storage::offload_helper_(
    int64_t layer_idx,
    std::vector<int64_t> query_global_idx,
    torch::Tensor bf16_k, 
    torch::Tensor v,
    torch::Tensor attention_mask) 
{
    this->logger_->debug("Offloading layer_idx: {}", layer_idx);
    try {
        if (this->model_config_.model_type.find("deepseek") ==
            std::string::npos) {
            /* Step 1: Permute k and v in the device. */
            auto k = bf16_k;
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
            CUDA_CHECK(cudaSetDevice(
                this->engine_config_.basic_config.device));
            auto [k, k_quantize_scale] = per_token_quant(bf16_k);
            // Check k or k_quantize_scale has nan values
            if (torch::any(torch::isnan(k)).item<bool>()){
                for (int64_t i = 0; i < k.size(0); i++) {
                    if(torch::any(torch::isnan(k[i])).item<bool>()) {
                        this->logger_->debug("k[{}] has nan values, rank: {}, layer_idx: {}",
                                             i, this->engine_config_.basic_config.device,
                                             layer_idx);
                    }
                }
                throw std::runtime_error("k has nan values");
            }
            if (torch::any(torch::isnan(k_quantize_scale)).item<bool>()){
                for (int64_t i = 0; i < k_quantize_scale.size(0); i++) {
                    if(torch::any(torch::isnan(k_quantize_scale[i])).item<bool>()) {
                        this->logger_->debug("k_quantize_scale[{}] has nan values, rank: {}, layer_idx: {}",
                                             i, this->engine_config_.basic_config.device,
                                             layer_idx);
                    }
                }
                throw std::runtime_error("k_quantize_scale has nan values");
            }


            // If k_quantize_scale[layer_idx] is torch.empty({0,0}), assign it to k_quantize_scale
            // other wise concatenate it.
            // if (this->k_quantize_scale[layer_idx].numel() == 0) {
            //     this->k_quantize_scale[layer_idx] = k_quantize_scale;
            // } else {
            //     this->k_quantize_scale[layer_idx] = torch::cat(
            //         {this->k_quantize_scale[layer_idx], k_quantize_scale}, 0);
            // }
            /* Step 1: Permute k and v in the device. */
            int64_t bsz = k.size(0);
            int64_t seq_len = k.size(1);
            int64_t k_seq_byte_size = k.size(1) * k.size(2) * k.element_size();
            int64_t token_byte_size = k.size(2) * k.element_size();
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
                    // Only note non-padding token.
                    // num_tokens = attention_mask.sum(1).item<int64_t>();
                    // used_byte_size = num_tokens * token_byte_size;
                    this->k_storage[slot_idx][layer_idx].num_tokens = attention_mask[i].sum().item<int64_t>();
                    this->k_storage[slot_idx][layer_idx].used_byte_size =
                        attention_mask[i].sum().item<int64_t>() * token_byte_size;
                    this->k_storage[slot_idx][layer_idx].quantize_scale = k_quantize_scale.index({i, torch::indexing::Slice(0, this->k_storage[slot_idx][layer_idx].num_tokens)}).clone().unsqueeze(0);                     
                        
                    // this->logger_->info("k_storage[{}][{}].num_tokens: {}, used_byte_size: {}",
                    //                     slot_idx, layer_idx,
                    //                     this->k_storage[slot_idx][layer_idx].num_tokens,
                    //                     this->k_storage[slot_idx][layer_idx].used_byte_size);
                    // this->k_storage[slot_idx][layer_idx].used_byte_size =
                    //     k_seq_byte_size;
                    // this->k_storage[slot_idx][layer_idx].num_tokens = seq_len;
                }
                {
                    std::lock_guard<std::mutex> lock(this->mutex_);
                    this->query_idx_to_slot_idx_map[query_idx] = slot_idx;
                }
            }
            // CUDA_CHECK(cudaStreamSynchronize(
            //     this->d2h_engine_.DtoH_stream));
            CUDA_CHECK(cudaStreamSynchronize(0));
            // CUDA_CHECK(cudaDeviceSynchronize());
            
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
                        torch::Tensor k, torch::Tensor v, torch::Tensor k_quantize_scale) {
    try {
        // auto worker = std::thread(&KV_Storage::update_helper_, this, layer_idx,
        //                           query_global_indices, k, v, k_quantize_scale);
        // worker.detach();
        // worker.join();
        this->update_helper_(layer_idx, query_global_indices, k, v,
                              k_quantize_scale);
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
                                torch::Tensor k, torch::Tensor v, torch::Tensor k_quantize_scale) {
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
            // this->k_quantize_scale[layer_idx] = torch::cat(
            //     {this->k_quantize_scale[layer_idx], k_quantize_scale}, 1);
            k = k.contiguous();
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
                    this->k_storage[slot_idx][layer_idx].quantize_scale = torch::cat(
                        {this->k_storage[slot_idx][layer_idx].quantize_scale,
                            k_quantize_scale.index({i}).unsqueeze(0)}, 1);
                    // this->k_quantize_scale[layer_idx]
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
        // std::vector<c10::Float8_e4m3fn*> k_ptrs;
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


std::vector<c10::Float8_e4m3fn*> KV_Storage::get_k_ptrs_fp8(
    int64_t layer_idx,
    std::vector<int64_t> batch) 
{
    try{
        std::vector<c10::Float8_e4m3fn*> k_ptrs;
        for (int64_t i = 0; i < static_cast<int64_t>(batch.size()); i++) {
            auto query_idx = batch[i];
            int64_t slot_idx = -1;
            {
                std::lock_guard<std::mutex> lock(this->mutex_);
                slot_idx = this->query_idx_to_slot_idx_map[query_idx];
            }
            c10::Float8_e4m3fn* k_ptr = nullptr;
            {
                std::lock_guard<std::mutex> lock(
                    this->per_element_mutex_[slot_idx *
                                                this->model_config_
                                                    .num_hidden_layers +
                                            layer_idx]);
                k_ptr = static_cast<c10::Float8_e4m3fn*>(
                    this->k_storage[slot_idx][layer_idx].start_ptr);
            }
            k_ptrs.push_back(k_ptr);
        }
        return k_ptrs;
    }
    // Catch PyTorch/CUDA specific exceptions
    catch (const c10::Error& e) {
        this->logger_->debug("KV_Storage get_k_ptrs_fp8(): CUDA/PyTorch error: {}",
                             e.what());
        throw;
    }
    // Catch CUDA runtime errors
    catch (const cudaError_t& err) {
        this->logger_->debug("KV_Storage get_k_ptrs_fp8(): CUDA runtime error: {}",
                             cudaGetErrorString(err));
        throw std::runtime_error(cudaGetErrorString(err));
    }
    // Catch standard C++ exceptions
    catch (const std::exception& e) {
        this->logger_->debug(
            "KV_Storage get_k_ptrs_fp8(): Failed to update K and V to "
            "the storage. Error: {}",
            e.what());
        throw std::runtime_error(
            "KV_Storage: Failed to update K and V to the storage.");
    }
    // Catch any other unexpected errors
    catch (...) {
        this->logger_->debug(
            "KV_Storage get_k_ptrs_fp8(): Failed to update K and V to the "
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
            // Init empty tensors for quantization scale
            // for(auto layer_idx = 0; layer_idx < this->model_config_.num_hidden_layers; layer_idx++) {
            //     auto options = torch::TensorOptions()
            //         .dtype(torch::kFloat32)
            //         .device(this->engine_config_.basic_config.device_torch)
            //         .requires_grad(false);
            //     torch::Tensor k_quantize_scale = torch::empty({0,0}, options);
            //     this->k_quantize_scale.push_back(k_quantize_scale);
            // }
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
                this->k_storage[slot_idx][layer_idx].used_byte_size = 14000 * 576;
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


void KV_Storage::save_compressed_kv(){
    // Save all the k in k_storage in a .pt file.
    this->logger_->info("KV_Storage save_compressed_kv(): Saving compressed kv.");
    std::vector<torch::Tensor> compressed_kv;
    for (int64_t slot_idx = 0;
         slot_idx < this->engine_config_.kv_storage_config.num_host_slots;
         slot_idx++) {
        for (int64_t layer_idx = 0;
             layer_idx < this->model_config_.num_hidden_layers;
             layer_idx++) {
            auto k_ptr = this->k_storage[slot_idx][layer_idx].start_ptr;
            auto k_size = this->k_storage[slot_idx][layer_idx].used_byte_size;
            auto k_tensor = torch::from_blob(
                k_ptr,                                     // pointer to data
                {1, k_size},                               // shape
                torch::TensorOptions()
                    .dtype(torch::kBFloat16)               // move dtype into TensorOptions
                    .device(torch::kCPU)                   // device
            ).clone();
            // Fill the tensor with random data
            // compressed_kv = torch::cat({compressed_kv, k_tensor}, 1);
            compressed_kv.push_back(k_tensor);
        }
    }
    // Concatenate all tensors in compressed_kv
    auto t = torch::cat(compressed_kv, 0);
    // Save the tensor with name compressed_kv_{device_id}.pt
    std::string device_id = std::to_string(this->engine_config_.basic_config.device);
    std::string file_name = "./compressed_kv_" + device_id + ".pt";
    this->logger_->info("Saving to file: {}", file_name);
    torch::save(t, file_name);
    this->logger_->info("KV_Storage save_compressed_kv(): Compressed kv saved.");  
}