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

#include "spdlog/spdlog.h"
#include <ATen/cuda/CachingHostAllocator.h>
#include <ATen/cuda/CUDAContext.h>
#include <filesystem>
#include <memory>
#include <stdexcept>
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
#include "batchgen.h"
#include "Parameter_Server/Parameter_Server.h"
#include "Parameter_Server/posix_shm.h"
#include "Weights_Storage/Weights_Storage.h"
#include "data_structures.h"
#include "utils.h"

namespace py = pybind11;
// namespace fs = std::filesystem;

// weight_storage = WeightStorage()

// core_engine = CoreEngine(weight_storage)


// CoreEngine(weight_storage){
//     this->weight_storage_ = weight_storage;
//     this->h2d_engine_ = HtoDEngine(weight_storage_);
// }
// BatchGen::BatchGen(py::object engine_config, py::object model_config)
//     : engine_config_(parse_engine_config(engine_config)),
//       model_config_(parse_model_config(model_config)),
//       weights_storage_(engine_config_, model_config_),
//       gpu_weight_buffer_(engine_config_, model_config_),
//       gpu_kv_buffer_(engine_config_, model_config_),
//       d2h_engine_(engine_config_),
//       kv_storage_(engine_config_, model_config_, d2h_engine_),
//       h2d_engine_(engine_config_, model_config_, weights_storage_, kv_storage_,
//                   gpu_weight_buffer_, gpu_kv_buffer_),
//       hetero_attn_(engine_config_, model_config_, kv_storage_, gpu_kv_buffer_,
//                    h2d_engine_, d2h_engine_) {
//     this->logger = init_logger(
//         this->engine_config_.basic_config.log_level,
//         "BatchGen" + std::to_string(this->engine_config_.basic_config.device));
//     if (!this->logger) {
//         throw std::runtime_error("Logger initialization failed.");
//     }
// }
// BatchGen.cpp

BatchGen::BatchGen(py::object engine_config, py::object model_config, Weights_Storage& weights_storage)
    : 
      // 1. Parse Configs first
      engine_config_(parse_engine_config(engine_config)),
      model_config_(parse_model_config(model_config)),
      
      // 2. Store the pointer to weights (so you can access it later if needed)
      weights_storage_(weights_storage),

      // 3. Initialize Direct Objects (Order matches Header file)
      gpu_weight_buffer_(engine_config_, model_config_),
      
      gpu_kv_buffer_(engine_config_, model_config_),
      
      d2h_engine_(engine_config_),
      
      // Pass the *initialized* d2h_engine_ to kv_storage
      kv_storage_(engine_config_, model_config_, d2h_engine_),

      // 4. The Complex One: H2D Engine
      // We pass the ARGUMENT 'weights_storage' directly here.
      // We pass the MEMBERS (kv_storage_, etc.) which are already initialized above.
      h2d_engine_(engine_config_, model_config_, 
                  weights_storage, // <--- Passing the argument reference
                  kv_storage_, 
                  gpu_weight_buffer_, 
                  gpu_kv_buffer_),

      // 5. Hetero Attn
      hetero_attn_(engine_config_, model_config_, 
                   kv_storage_, 
                   gpu_kv_buffer_, 
                   h2d_engine_, 
                   d2h_engine_) 
{
    // Body: Just logging setup now
    this->logger = init_logger(
        this->engine_config_.basic_config.log_level,
        "BatchGen_" + std::to_string(this->engine_config_.basic_config.device));
    this->logger->info("BatchGen Instantiated.");
}

BatchGen::~BatchGen() { this->Terminate(); }

// void BatchGen::Init(std::string& shm_name, std::string& tensor_meta_shm_name,
//                     int64_t byte_size,
//                     std::unordered_map<std::string, std::vector<std::string>>&
//                        weights_copy_tasks
//     ) 
// {
//     this->logger->info("BatchGen Init.");
//     this->logger->info("model type: {}", this->model_config_.model_type);


//     this->shm_name_ = shm_name;
//     this->tensor_meta_shm_name_ = tensor_meta_shm_name;
//     auto weights_map = deserialize_from_shared_memory(tensor_meta_shm_name);

//     this->logger->info("weights_map deserialized.");
//     this->logger->info("shm_name: {}", shm_name);
//     this->logger->info("byte_size: {}", byte_size);
//     this->weights_storage_.Init(shm_name, byte_size, weights_map);
//     this->logger->info("weights_storage initialized.");
//     // this->weights_storage_.Init(this->parameter_server_.attr("byte_size").cast<int64_t>(),
//     // this->parameter_server_.attr("module_weights_shm").cast<std::unordered_map<std::string,
//     // std::unordered_map<std::string, tensor_meta>>());
//     this->gpu_kv_buffer_.Init();
//     this->gpu_weight_buffer_.Init();
//     this->kv_storage_.Init();
//     this->h2d_engine_.Init(weights_copy_tasks);
//     this->d2h_engine_.Init();

//     // this->hetero_attn_.Init();
//     this->register_signal_handler();
//     this->logger->info("BatchGen Initialized.");
// };

void BatchGen::init_weight_storage(std::string& shm_name, std::string& tensor_meta_shm_name,
                    int64_t byte_size, bool enable_hugetlbfs)
{
    this->logger->info("BatchGen Init Weight Storage.");
    this->logger->info("model type: {}", this->model_config_.model_type);
    this->shm_name_ = shm_name;
    this->tensor_meta_shm_name_ = tensor_meta_shm_name;
    // auto weights_map = deserialize_from_shared_memory(tensor_meta_shm_name);
    this->logger->info("weights_map deserialized.");
    this->logger->info("shm_name: {}", shm_name);
    this->logger->info("byte_size: {}", byte_size);
    this->weights_storage_.Init(shm_name, byte_size, tensor_meta_shm_name, enable_hugetlbfs);
    this->logger->info("weights_storage initialized.");
}



void BatchGen::Init() 
{
    this->logger->info("BatchGen Init.");
    this->logger->info("model type: {}", this->model_config_.model_type);


    // this->shm_name_ = shm_name;
    // this->tensor_meta_shm_name_ = tensor_meta_shm_name;
    // auto weights_map = deserialize_from_shared_memory(tensor_meta_shm_name);

    // this->logger->info("weights_map deserialized.");
    // this->logger->info("shm_name: {}", shm_name);
    // this->logger->info("byte_size: {}", byte_size);
    // this->weights_storage_.Init(shm_name, byte_size, weights_map, enable_hugetlbfs);
    // this->logger->info("weights_storage initialized.");
    
    // this->gpu_kv_buffer_.Init();
    this->gpu_weight_buffer_.Init();
    this->kv_storage_.Init();
    this->h2d_engine_.Init();
    this->d2h_engine_.Init();

    // this->hetero_attn_.Init();
    this->register_signal_handler();
    this->logger->info("BatchGen Initialized.");
};

void BatchGen::Terminate() {
    this->h2d_engine_.Terminate();
    this->d2h_engine_.Terminate();
    
    // Now managed by the parameter server.
    // shm_unlink(this->shm_name_.c_str());
    // shm_unlink(this->tensor_meta_shm_name_.c_str());
};

// std::unordered_map<std::string, torch::Tensor>
// BatchGen::get_skeleton_state_dict(){
// 	/* Get the skeleton state dict. */
// 	return this->weights_storage_.get_skeleton_state_dict();
// };

void BatchGen::set_batching_plan(
    std::vector<std::vector<int64_t>> batching_plan) {
    /* Set the batching */
};

void BatchGen::set_phase(std::string phase) {
    this->phase_ = phase;
    if (phase == "decode") {
        // this->gpu_kv_buffer_.init_kv_buffer();
        this->gpu_weight_buffer_.resize_buffer();
    }
}

void BatchGen::kv_offload(int64_t layer_idx, std::vector<int64_t> query_idx,
                         torch::Tensor key_states, torch::Tensor value_states,torch::Tensor attention_mask) {
    /* Offload the kv to the kv storage. */
    // check if key_states contains any NaN values
    // if (key_states.isnan().any().item<bool>()) {
    //     // check which sequence contains NaN values
    //     for(int i = 0; i < key_states.size(0); i++){
    //         if(key_states[i].isnan().any().item<bool>()){
    //             this->logger->error("key_states contains NaN values at index {}", i);
    //         }
    //     }
    //     this->logger->error("key_states contains NaN values");
    //     // throw std::runtime_error("key_states contains NaN values");
    // }
    this->kv_storage_.offload(layer_idx, query_idx, key_states, value_states, attention_mask);
};

// void BatchGen::add_weight_storage(
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

// std::unordered_map<std::string, torch::Tensor> BatchGen::get_weights(
//     std::string module_key) {
//     /* Get the weights from the weights storage. */
//     return *this->gpu_weight_buffer_.get_weights(module_key);  // blocking.
// };

std::unordered_map<std::string, torch::Tensor> BatchGen::get_weights(
    std::string module_key,
    std::string& phase) 
{
    /* Get the weights from the weights storage. */
    CUDA_CHECK(cudaSetDevice(this->engine_config_.basic_config.device));
    const cudaStream_t consumer_stream = at::cuda::getCurrentCUDAStream(
        this->engine_config_.basic_config.device).stream();
    return this->gpu_weight_buffer_.get_weights(
        module_key, phase, consumer_stream);
};

std::unordered_map<std::string, torch::Tensor> BatchGen::get_weights_pinned(
    std::string module_key) {
    CUDA_CHECK(cudaSetDevice(this->engine_config_.basic_config.device));
    const cudaStream_t consumer_stream = at::cuda::getCurrentCUDAStream(
        this->engine_config_.basic_config.device).stream();
    return this->gpu_weight_buffer_.get_weights_pinned(
        module_key, consumer_stream);
};

void BatchGen::free_weights_buffer(const std::string& module_name) {
    /* Free the weights buffer. */
    this->gpu_weight_buffer_.releaseBuffer(module_name);
};

void BatchGen::free_weights_buffer_async(const std::string& module_name) {
    this->free_weights_buffers_async({module_name});
};

void BatchGen::free_weights_buffers_async(
    const std::vector<std::string>& module_names) {
    CUDA_CHECK(cudaSetDevice(this->engine_config_.basic_config.device));
    const cudaStream_t consumer_stream = at::cuda::getCurrentCUDAStream(
        this->engine_config_.basic_config.device).stream();
    this->gpu_weight_buffer_.releaseBuffersAsync(
        module_names, consumer_stream);
};

torch::Tensor BatchGen::attn(py::object torch_module, int64_t layer_idx,
                            torch::Tensor& hidden_states,
                            torch::Tensor& attention_mask,
                            torch::Tensor& position_ids,
                            std::vector<std::vector<int64_t>> cur_batch) {
    return this->hetero_attn_.attn(torch_module, layer_idx, hidden_states,
                                   attention_mask, position_ids, cur_batch);
}

void BatchGen::submit_to_KV_queue(std::vector<int64_t> micro_batch,
                                 int64_t micro_batch_idx, int64_t layer_idx,
                                 int64_t byte_size) {
    this->h2d_engine_.submit_to_KV_queue(micro_batch, micro_batch_idx,
                                         layer_idx, byte_size);
};

void BatchGen::clear_expert_buffer(int64_t layer_idx, int64_t expert_idx, std::string phase) {
    this->gpu_weight_buffer_.clear_expert_buffer(layer_idx, expert_idx, phase);
};

void BatchGen::prefill_complete_sync() {
    this->d2h_engine_.wait_until_queue_B_empty();
};

void BatchGen::clear_kv_storage() { this->kv_storage_.clear_kv_storage(); };

void BatchGen::clear_kv_copy_queue() {
    this->h2d_engine_.clear_kv_copy_queue();
};

void BatchGen::clear_weight_copy_queue() {
    this->h2d_engine_.clear_weight_copy_queue();
};

void BatchGen::reset_weight_copy_queue() {
    this->h2d_engine_.reset_weight_copy_queue();
};

void BatchGen::reset_prefill_buffer() {
    this->gpu_weight_buffer_.reset_prefill_buffer();
};

void BatchGen::clear_kv_buffer() { this->gpu_kv_buffer_.clear_kv_buffer(); };
void BatchGen::create_fake_kv_storage() {
    this->kv_storage_.create_fake_kv_storage();
};

// std::unordered_map<std::string, torch::Tensor> BatchGen::get_tensor(
//     std::string module_key)
// {
//     return this->weights_storage_.get_tensor(module_key);
// }
py::dict BatchGen::get_tensor(std::string module_key)
{
    return this->weights_storage_.get_tensor(module_key);
}

void BatchGen::start_h2d_worker() {
    this->h2d_engine_.Start();
}


void BatchGen::set_global_routed_experts_data_ptr(
    const py::dict& experts_IPC_handles,
    const py::dict& expert_location_map)
{
    this->h2d_engine_.set_global_routed_experts_data_ptr(
        experts_IPC_handles,
        expert_location_map
    );
}

void BatchGen::cuda_enable_peer_access(int rank, int world_size){
    this->h2d_engine_.cuda_enable_peer_access(rank, world_size);
}

void BatchGen::save_compressed_kv(){
    this->kv_storage_.save_compressed_kv();
}


void BatchGen::set_weight_copy_queue(
    std::unordered_map<std::string, std::vector<std::string>>&
        weight_copy_tasks) {
    this->h2d_engine_.set_weight_copy_queue(weight_copy_tasks);
    this->gpu_weight_buffer_.set_weight_copy_task(weight_copy_tasks);
}

void BatchGen::reset_decoding_buffer() {
    this->gpu_weight_buffer_.reset_decoding_buffer();
}

void BatchGen::stop_h2d_worker() {
    this->h2d_engine_.stop_h2d_worker();
}

void BatchGen::set_host_paged_kv_worker_view(py::object worker_view) {
    if (worker_view.is_none()) {
        throw std::invalid_argument(
            "host_paged_kv_worker_view must not be None");
    }
    host_paged_kv_worker_view_ = std::move(worker_view);
}

py::object BatchGen::host_paged_kv_worker_view() const {
    return host_paged_kv_worker_view_;
}

void BatchGen::set_gpu_paged_kv_manager(py::object manager) {
    if (manager.is_none()) {
        throw std::invalid_argument(
            "gpu_paged_kv_manager must not be None");
    }
    gpu_paged_kv_manager_ = std::move(manager);
}

py::object BatchGen::gpu_paged_kv_manager() const {
    return gpu_paged_kv_manager_;
}



#include <signal.h>
static BatchGen* engine_instance = nullptr;
void signalHandler(int signum) {
    if (engine_instance) {
        engine_instance->Terminate();
    }
    exit(signum);
}
void BatchGen::register_signal_handler() {
    engine_instance = this;
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);
}
