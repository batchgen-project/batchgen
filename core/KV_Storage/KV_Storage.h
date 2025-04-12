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

/*
        Manage the Key-Value Cache Storage in the Host Memory.
*/
struct sequence_storage {
    void* start_ptr;
    int64_t used_byte_size;
    int64_t num_tokens;
};

class DtoH_Engine;

class KV_Storage {
   public:
    KV_Storage(EngineConfig& engine_config, ModelConfig& model_config,
               DtoH_Engine& d2h_engine);
    ~KV_Storage() = default;
    void Init();

    void offload(int64_t layer_idx, std::vector<int64_t> query_global_idx,
                 torch::Tensor k, torch::Tensor v);
    void update(int64_t layer_idx, std::vector<int64_t> query_global_idx,
                torch::Tensor k, torch::Tensor v);

    // void fetch_tasks(std::vector<std::vector<int64_t>>& micro_batches);

    std::vector<c10::BFloat16*> get_k_ptrs(int64_t layer_idx,
                                           std::vector<int64_t> batch);
    std::vector<c10::BFloat16*> get_v_ptrs(int64_t layer_idx,
                                           std::vector<int64_t> batch);

    void clear_kv_storage();
    void create_fake_kv_storage();

   private:
    /* Template */
    std::shared_ptr<spdlog::logger> logger_;
    const EngineConfig& engine_config_;
    const ModelConfig& model_config_;
    DtoH_Engine& d2h_engine_;

    std::vector<void*> k_pinned_memory;
    std::vector<void*> v_pinned_memory;
    std::vector<std::vector<sequence_storage>> k_storage;
    std::vector<std::vector<sequence_storage>> v_storage;
    std::vector<std::mutex> per_element_mutex_;

    std::mutex mutex_;  // protect query_idx_to_slot_idx_map and empty_slots.
    std::unordered_map<int64_t, int64_t> query_idx_to_slot_idx_map;
    std::unordered_set<int64_t> empty_slots;

    void offload_helper_(int64_t layer_idx,
                         std::vector<int64_t> query_global_idx, torch::Tensor k,
                         torch::Tensor v);
    void update_helper_(int64_t layer_idx,
                        std::vector<int64_t> query_global_idx, torch::Tensor k,
                        torch::Tensor v);
};
