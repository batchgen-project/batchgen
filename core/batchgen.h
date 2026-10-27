// clang-format off
/* ----------------------------------------------------------------------------  *
 *  BatchGen                                                                      *
 *  Copyright (c) 2025-2026 BatchGen Team                                            *
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
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <torch/extension.h>
#include <torch/torch.h>
#include <unordered_map>
#include <vector>

#include "DtoH_Engine/DtoH_Engine.h"
#include "GPU_KV_Buffer/GPU_KV_Buffer.h"
#include "GPU_Weight_Buffer/GPU_Weight_Buffer.h"
#include "Hetero_Attn/Hetero_Attn.h"
#include "HtoD_Engine/HtoD_Engine.h"
#include "KV_Storage/KV_Storage.h"
#include "Weights_Storage/Weights_Storage.h"
#include "data_structures.h"
#include "threadsafe_queue.h"
#include "utils.h"

namespace py = pybind11;

class BatchGen {
   public:
    BatchGen(py::object engine_config, py::object model_config, Weights_Storage& weights_storage);

    ~BatchGen();

    void Init();
    void init_weight_storage(std::string& shm_name, std::string& tensor_meta_shm_name,
                    int64_t byte_size, bool enable_hugetlbfs,
                    bool enable_thp = false, int memfd_creator_pid = -1,
                    int memfd_fd = -1, int tensor_meta_memfd_fd = -1);
    void Terminate();

    /*
            Exposing APIs to PyTorch Side.
    */
    void prefill_complete_sync();
    // std::unordered_map<std::string, torch::Tensor> get_skeleton_state_dict();
    /* Holding */
    void set_batching_plan(std::vector<std::vector<int64_t>>
                               batching_plan);  // Called every round.

    void submit_to_KV_queue(std::vector<int64_t> micro_batch,
                            int64_t micro_batch_idx, int64_t layer_idx,
                            int64_t byte_size);

    /* KV */
    void kv_offload(int64_t layer_idx, std::vector<int64_t> query_idx,
                    torch::Tensor key_states, torch::Tensor value_states,torch::Tensor attention_mask);

    /* Weights */
    // void add_weight_storage(
    // 	const std::string& module_key,
    // 	const std::string& tensor_key,
    // 	int64_t tensor_ptr,
    // 	const std::vector<int64_t>& tensor_shape,
    // 	int64_t byte_size
    // );

    // Leases module_name in the current weight pass; see begin_weight_pass.
    std::unordered_map<std::string, torch::Tensor> get_weights(
        const std::string& module_name);
    void free_weights_buffer(const std::string& module_name);
    void free_weights_buffer_async(const std::string& module_name);
    void free_weights_buffers_async(
        const std::vector<std::string>& module_names);

    /* Attn */
    torch::Tensor attn(py::object torch_module, int64_t layer_idx,
                       torch::Tensor& hidden_states,
                       torch::Tensor& attention_mask,
                       torch::Tensor& position_ids,
                       std::vector<std::vector<int64_t>> cur_batching_plan);

    void set_phase(std::string phase);

    void register_signal_handler();

    void clear_kv_storage();
    void clear_kv_buffer();
    void clear_kv_copy_queue();
    void reset_prefill_buffer();
    void create_fake_kv_storage();
    // std::unordered_map<std::string, torch::Tensor> get_tensor(
        // std::string module_key);
    py::dict get_tensor(std::string module_key);
    void start_h2d_worker();
    void set_global_routed_experts_data_ptr(
        const py::dict& experts_IPC_handles,
        const py::dict& expert_location_map
    );
    void cuda_enable_peer_access(int rank, int world_size);
    void save_compressed_kv();
    void set_weight_copy_queue(
        std::unordered_map<std::string, std::vector<std::string>>&
            weight_copy_tasks);
    void reset_weight_stream_profile(bool enabled);
    py::dict get_weight_stream_profile();
    // Routing-informed skip in the current pass: frees, releases on landing,
    // or skips the copy of each module. Returns {"freed_present",
    // "release_on_landing", "pending"} counts.
    py::dict skip_weight_modules(const std::vector<std::string>& module_names);
    py::dict weight_stream_status();

    /* Weight passes: one traversal of every installed list, i.e. one
     * forward. */
    // Arms the next pass and returns its id (0: nothing installed). Arming a
    // second pass while one is open is the explicit carry; a third throws.
    int64_t begin_weight_pass();
    // Waits until the producer is past the end of pass_id (the oldest open
    // pass) and its copies and releases completed, asserts that it holds no
    // slot, skip or daemon lease, and closes it. Returns {"pass", "copied",
    // "skipped", "released_on_landing"} (per type) and "carried" (slots
    // already produced for the next pass).
    py::dict end_weight_pass(int64_t pass_id);
    // Declares every module of an open pass that is not leased as skipped.
    // The caller still ends the pass. Returns {"pass", "freed_present",
    // "release_on_landing", "pending"}.
    py::dict cancel_weight_pass(int64_t pass_id);

    void reset_decoding_buffer();
    void stop_h2d_worker();

    // Binds a Python-owned HostPagedKVWorkerView instance for reuse.
    void set_host_paged_kv_worker_view(py::object worker_view);

    // Returns the bound HostPagedKVWorkerView. Raises if unset.
    py::object host_paged_kv_worker_view() const;

    // Attach a Python GPU paged KV manager instance shared with BatchGen.
    void set_gpu_paged_kv_manager(py::object manager);

    // Returns the attached GPU paged KV manager.
    py::object gpu_paged_kv_manager() const;

    void copy_kv_to_worker(std::vector<int64_t> query_global_idx, int64_t context_length) {
        this->kv_storage_.copy_kv_to_worker(query_global_idx, context_length);
    }
    void clear_gpu_kv_storage(){
        this->kv_storage_.clear_kv_gpu_storage();
    }

    // std::vector<torch::Tensor> get_past_key_states(std::vector<int64_t> query_global_indices, int64_t max_seq_len) {
    //     return this->kv_storage_.get_past_key_states(query_global_indices, max_seq_len);
    // }
    py::list get_past_key_states(std::vector<int64_t> query_global_indices, int64_t max_seq_len) {
        return this->kv_storage_.get_past_key_states(query_global_indices, max_seq_len);
    }
    std::vector<torch::Tensor> get_kv_scale(std::vector<int64_t> query_global_indices, int64_t seq_len){
        return this->kv_storage_.get_kv_scale(query_global_indices, seq_len);
    }
        

   private:
    EngineConfig engine_config_;
    ModelConfig model_config_;
    std::shared_ptr<spdlog::logger> logger;

    std::string phase_;
    std::string shm_name_;
    std::string tensor_meta_shm_name_;
    KV_Storage kv_storage_;
    // Weights_Storage* weights_storage_ptr_;
    Weights_Storage& weights_storage_;
    HtoD_Engine h2d_engine_;
    DtoH_Engine d2h_engine_;
    GPU_Weight_Buffer gpu_weight_buffer_;
    GPU_KV_Buffer gpu_kv_buffer_;

    py::object parameter_server_;

    Hetero_Attn hetero_attn_;

    py::object host_paged_kv_worker_view_{py::none()};
    py::object gpu_paged_kv_manager_{py::none()};
};
