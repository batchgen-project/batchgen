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
#include <c10/util/BFloat16.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <memory>
#include <mutex>
#include <string>
#include <torch/extension.h>
#include <torch/torch.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../DtoH_Engine/DtoH_Engine.h"
#include "../data_structures.h"
#include "../utils.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

/*
        Manage the Key-Value Cache Storage in the Host Memory.
*/
struct sequence_storage {
    void* start_ptr;
    int64_t used_byte_size;
    int64_t num_tokens;
    torch::Tensor quantize_scale;
    sequence_storage() : start_ptr(nullptr), used_byte_size(0), num_tokens(0) {}
};

class DtoH_Engine;

class KV_Storage {
   public:
    KV_Storage(EngineConfig& engine_config, ModelConfig& model_config,
               DtoH_Engine& d2h_engine);
    ~KV_Storage() = default;
    void Init();

    void offload(int64_t layer_idx, std::vector<int64_t> query_global_idx,
                 torch::Tensor k, torch::Tensor v, torch::Tensor attention_mask);
    void update(int64_t layer_idx, std::vector<int64_t> query_global_idx,
                torch::Tensor k, torch::Tensor v, torch::Tensor k_quantize_scale);

    // void fetch_tasks(std::vector<std::vector<int64_t>>& micro_batches);

    std::vector<c10::BFloat16*> get_k_ptrs(int64_t layer_idx,
                                           std::vector<int64_t> batch);
    std::vector<c10::BFloat16*> get_v_ptrs(int64_t layer_idx,
                                           std::vector<int64_t> batch);
    std::vector<c10::Float8_e4m3fn*> get_k_ptrs_fp8(
        int64_t layer_idx,
        std::vector<int64_t> batch
    );

    void clear_kv_storage();
    void create_fake_kv_storage();
    void save_compressed_kv();

    torch::Tensor get_k_quantize_scale(int64_t layer_idx, const std::vector<int64_t>& cur_batch, int64_t padding_lentgh);
    void copy_kv_to_worker(std::vector<int64_t> query_global_idx, int64_t context_length);
    torch::Tensor get_k(int64_t layer_idx, std::vector<int64_t> cur_batch, std::vector<int64_t> tensor_shape);
    void gpu_kv_update(
        int64_t layer_idx,
        std::vector<int64_t> query_global_indices,
        torch::Tensor k, torch::Tensor v, torch::Tensor k_quantize_scale);
    void gpu_kv_update_func(
        int64_t layer_idx,
        std::vector<int64_t> query_global_indices,
        torch::Tensor k, torch::Tensor v, torch::Tensor k_quantize_scale);
    void clear_kv_gpu_storage();
    std::vector<torch::Tensor> get_kv_scale(std::vector<int64_t> query_global_indices, int64_t seq_len);
    //  std::vector<torch::Tensor> get_past_key_states(std::vector<int64_t> query_global_indices, int64_t max_seq_len);
    py::list get_past_key_states(std::vector<int64_t> query_global_indices, int64_t max_seq_len);

   private:
    /* Template */
    std::shared_ptr<spdlog::logger> logger_;
    const EngineConfig& engine_config_;
    const ModelConfig& model_config_;
    DtoH_Engine& d2h_engine_;
    cudaStream_t stream_;

    std::vector<void*> k_pinned_memory;
    std::vector<void*> v_pinned_memory;
    std::vector<std::vector<sequence_storage>> k_storage;
    std::vector<std::vector<sequence_storage>> v_storage;
    std::vector<std::mutex> per_element_mutex_;
    
    std::vector<void*> k_gpu_memory;
    std::vector<std::vector<sequence_storage>> k_gpu_storage;


    std::mutex mutex_;  // protect query_idx_to_slot_idx_map and empty_slots.
    std::unordered_map<int64_t, int64_t> query_idx_to_slot_idx_map;
    std::unordered_map<int64_t, int64_t> gpu_query_idx_to_slot_idx_map;
    std::unordered_set<int64_t> empty_slots;

    void offload_helper_(int64_t layer_idx,
                         std::vector<int64_t> query_global_idx, 
                         torch::Tensor k,
                         torch::Tensor v,
                         torch::Tensor attention_mask);
    void update_helper_(int64_t layer_idx,
                        std::vector<int64_t> query_global_idx, 
                        torch::Tensor k,
                        torch::Tensor v,
                        torch::Tensor k_quantize_scale);
    
};
