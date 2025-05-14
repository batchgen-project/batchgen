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
#include <ATen/cuda/CachingHostAllocator.h>
#include <filesystem>
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <torch/extension.h>
#include <torch/torch.h>
#include <unordered_map>
#include <vector>
// #include <numa.h>

#include "DtoH_Engine/DtoH_Engine.h"
#include "GPU_KV_Buffer/GPU_KV_Buffer.h"
#include "GPU_Weight_Buffer/GPU_Weight_Buffer.h"
#include "HtoD_Engine/HtoD_Engine.h"
#include "KV_Storage/KV_Storage.h"
#include "MoE_Gen.h"
#include "Parameter_Server/Parameter_Server.h"
#include "Parameter_Server/posix_shm.h"
#include "Weights_Storage/Weights_Storage.h"
#include "data_structures.h"
#include "utils.h"

namespace py = pybind11;
// namespace fs = std::filesystem;

MoE_Gen::MoE_Gen(py::object engine_config, py::object model_config)
    : engine_config_(parse_engine_config(engine_config)),
      model_config_(parse_model_config(model_config)),
      weights_storage_(engine_config_, model_config_),
      gpu_weight_buffer_(engine_config_, model_config_),
      gpu_kv_buffer_(engine_config_, model_config_),
      d2h_engine_(engine_config_),
      kv_storage_(engine_config_, model_config_, d2h_engine_),
      h2d_engine_(engine_config_, model_config_, weights_storage_, kv_storage_,
                  gpu_weight_buffer_, gpu_kv_buffer_),
      hetero_attn_(engine_config_, model_config_, kv_storage_, gpu_kv_buffer_,
                   h2d_engine_, d2h_engine_) {
    this->logger = init_logger(
        this->engine_config_.basic_config.log_level,
        "MoE_Gen" + std::to_string(this->engine_config_.basic_config.device));
    if (!this->logger) {
        throw std::runtime_error("Logger initialization failed.");
    }
    // struct bitmask* nodemask = numa_allocate_nodemask();
    // numa_bitmask_clearall(nodemask);
    // numa_bitmask_setbit(nodemask, 1);

    // numa_set_preferred(1);
    // this->weights_storage_.Init();
}

MoE_Gen::~MoE_Gen() { this->Terminate(); }

void MoE_Gen::Init(std::string& shm_name, std::string& tensor_meta_shm_name,
                    int64_t byte_size,
                    std::unordered_map<std::string, std::vector<std::string>>&
                       weights_copy_tasks
    ) 
{
    this->logger->info("MoE-Gen Init.");
    this->logger->info("model type: {}", this->model_config_.model_type);


    this->shm_name_ = shm_name;
    this->tensor_meta_shm_name_ = tensor_meta_shm_name;
    auto weights_map = deserialize_from_shared_memory(tensor_meta_shm_name);

    this->logger->info("weights_map deserialized.");
    this->logger->info("shm_name: {}", shm_name);
    this->logger->info("byte_size: {}", byte_size);
    this->weights_storage_.Init(shm_name, byte_size, weights_map);
    this->logger->info("weights_storage initialized.");
    // this->weights_storage_.Init(this->parameter_server_.attr("byte_size").cast<int64_t>(),
    // this->parameter_server_.attr("module_weights_shm").cast<std::unordered_map<std::string,
    // std::unordered_map<std::string, tensor_meta>>());
    this->gpu_kv_buffer_.Init();
    this->gpu_weight_buffer_.Init();
    this->kv_storage_.Init();
    this->h2d_engine_.Init(weights_copy_tasks);
    this->d2h_engine_.Init();

    // this->hetero_attn_.Init();
    this->register_signal_handler();
    this->logger->info("MoE-Gen Initialized.");
};

void MoE_Gen::Terminate() {
    this->h2d_engine_.Terminate();
    this->d2h_engine_.Terminate();
    
    // Now managed by the parameter server.
    // shm_unlink(this->shm_name_.c_str());
    // shm_unlink(this->tensor_meta_shm_name_.c_str());
};

// std::unordered_map<std::string, torch::Tensor>
// MoE_Gen::get_skeleton_state_dict(){
// 	/* Get the skeleton state dict. */
// 	return this->weights_storage_.get_skeleton_state_dict();
// };

void MoE_Gen::set_batching_plan(
    std::vector<std::vector<int64_t>> batching_plan) {
    /* Set the batching */
};

void MoE_Gen::set_phase(std::string phase) {
    this->phase_ = phase;
    if (phase == "decoding") {
        this->gpu_kv_buffer_.init_kv_buffer();
        this->gpu_weight_buffer_.resize_buffer();
    }
}

void MoE_Gen::kv_offload(int64_t layer_idx, std::vector<int64_t> query_idx,
                         torch::Tensor key_states, torch::Tensor value_states,torch::Tensor attention_mask) {
    /* Offload the kv to the kv storage. */
    // check if key_states contains any NaN values
    if (key_states.isnan().any().item<bool>()) {
        // check which sequence contains NaN values
        for(int i = 0; i < key_states.size(0); i++){
            if(key_states[i].isnan().any().item<bool>()){
                this->logger->error("key_states contains NaN values at index {}", i);
            }
        }
        this->logger->error("key_states contains NaN values");
        // throw std::runtime_error("key_states contains NaN values");
    }
    this->kv_storage_.offload(layer_idx, query_idx, key_states, value_states, attention_mask);
};

// void MoE_Gen::add_weight_storage(
// 	const std::string& module_key,
// 	const std::string& tensor_key,
// 	int64_t tensor_ptr,
// 	const std::vector<int64_t>& tensor_shape,
// 	int64_t byte_size)
// {
// 	/* Add the weights to the weights storage. */
// 	void* ptr = reinterpret_cast<void*>(tensor_ptr);
// 	this->weights_storage_.add_module_to_storage(
// 		module_key,
// 		tensor_key,
// 		ptr,
// 		tensor_shape,
// 		byte_size
// 	);
// };

// std::unordered_map<std::string, torch::Tensor> MoE_Gen::get_weights(
//     std::string module_key) {
//     /* Get the weights from the weights storage. */
//     return *this->gpu_weight_buffer_.get_weights(module_key);  // blocking.
// };

std::unordered_map<std::string, torch::Tensor> MoE_Gen::get_weights(
    std::string module_key,
    std::string& phase) 
{
    /* Get the weights from the weights storage. */
    CUDA_CHECK(cudaSetDevice(this->engine_config_.basic_config.device));
    // CUDA_CHECK(cudaStreamSynchronize(0));
    return this->gpu_weight_buffer_.get_weights(module_key, phase);  // blocking.
};

void MoE_Gen::free_weights_buffer(const std::string& module_name) {
    /* Free the weights buffer. */
    this->gpu_weight_buffer_.releaseBuffer(module_name);
};

torch::Tensor MoE_Gen::attn(py::object torch_module, int64_t layer_idx,
                            torch::Tensor& hidden_states,
                            torch::Tensor& attention_mask,
                            torch::Tensor& position_ids,
                            std::vector<std::vector<int64_t>> cur_batch) {
    return this->hetero_attn_.attn(torch_module, layer_idx, hidden_states,
                                   attention_mask, position_ids, cur_batch);
}

void MoE_Gen::submit_to_KV_queue(std::vector<int64_t> micro_batch,
                                 int64_t micro_batch_idx, int64_t layer_idx,
                                 int64_t byte_size) {
    this->h2d_engine_.submit_to_KV_queue(micro_batch, micro_batch_idx,
                                         layer_idx, byte_size);
};

void MoE_Gen::clear_expert_buffer(int64_t layer_idx, int64_t expert_idx, std::string phase) {
    this->gpu_weight_buffer_.clear_expert_buffer(layer_idx, expert_idx, phase);
};

void MoE_Gen::prefill_complete_sync() {
    this->d2h_engine_.wait_until_queue_B_empty();
};

void MoE_Gen::clear_kv_storage() { this->kv_storage_.clear_kv_storage(); };

void MoE_Gen::clear_kv_copy_queue() {
    this->h2d_engine_.clear_kv_copy_queue();
};

void MoE_Gen::clear_weight_copy_queue() {
    this->h2d_engine_.clear_weight_copy_queue();
};

void MoE_Gen::reset_weight_copy_queue() {
    this->h2d_engine_.reset_weight_copy_queue();
};

void MoE_Gen::reset_prefill_buffer() {
    this->gpu_weight_buffer_.reset_prefill_buffer();
};

void MoE_Gen::clear_kv_buffer() { this->gpu_kv_buffer_.clear_kv_buffer(); };
void MoE_Gen::create_fake_kv_storage() {
    this->kv_storage_.create_fake_kv_storage();
};

std::unordered_map<std::string, torch::Tensor> MoE_Gen::get_tensor(
    std::string module_key)
{
    return this->weights_storage_.get_tensor(module_key);
}

void MoE_Gen::start_h2d_worker() {
    this->h2d_engine_.Start();
}


void MoE_Gen::set_global_routed_experts_data_ptr(
    const py::dict& experts_IPC_handles,
    const py::dict& expert_location_map)
{
    this->h2d_engine_.set_global_routed_experts_data_ptr(
        experts_IPC_handles,
        expert_location_map
    );
}

void MoE_Gen::cuda_enable_peer_access(int rank, int world_size){
    this->h2d_engine_.cuda_enable_peer_access(rank, world_size);
}

void MoE_Gen::save_compressed_kv(){
    this->kv_storage_.save_compressed_kv();
}


void MoE_Gen::set_weight_copy_queue(
    std::unordered_map<std::string, std::vector<std::string>>&
        weight_copy_tasks) {
    this->h2d_engine_.set_weight_copy_queue(weight_copy_tasks);
    this->gpu_weight_buffer_.set_weight_copy_task(weight_copy_tasks);
}

void MoE_Gen::reset_decoding_buffer() {
    this->gpu_weight_buffer_.reset_decoding_buffer();
}

void MoE_Gen::stop_h2d_worker() {
    this->h2d_engine_.stop_h2d_worker();
}

#include <signal.h>
static MoE_Gen* engine_instance = nullptr;
void signalHandler(int signum) {
    if (engine_instance) {
        engine_instance->Terminate();
    }
    exit(signum);
}
void MoE_Gen::register_signal_handler() {
    engine_instance = this;
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);
}
