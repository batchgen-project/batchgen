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
#include <cuda.h>
#include <cuda_runtime.h>
#include <memory>
#include <queue>
#include <torch/extension.h>
#include <torch/torch.h>

struct MemoryChunk {
    void* ptr;
    int64_t size;
    void* data_ptr() { return ptr; }
    // Default constructor
    MemoryChunk() : ptr(nullptr), size(0) {}
    // Parameterized constructor
    MemoryChunk(void* p, int64_t s) : ptr(p), size(s) {}
};

// struct kv_slot{
// 	bool occupied;
// 	int64_t layer_idx;
// 	int64_t query_global_idx;
// 	void* memory_address;
// 	int64_t used_byte_size;
// 	int64_t num_tokens;
// 	void* data_ptr() {
// 		return memory_address;
// 	};
// };
struct kv_host_slot {
    bool occupied;
    int64_t query_global_idx;
    std::vector<void*> k_ptrs;
    std::vector<void*> v_ptrs;
    std::vector<int64_t> used_byte_size;  // For kv update;
    std::vector<int64_t> num_tokens;
};

struct cpu_compute_task {
    std::vector<int64_t> query_global_idx;
    torch::Tensor attention_weights;

    cpu_compute_task(std::vector<int64_t> query_global_idx,
                     torch::Tensor attention_weights)
        : query_global_idx(query_global_idx),
          attention_weights(attention_weights) {};
};

struct micro_batch {
    std::vector<int64_t> query_global_idx;
    std::vector<std::vector<int64_t>> micro_batch_boundaries;
};

struct KV_Storage_Config {
    int64_t num_host_slots;
    int64_t reserved_length;  // Num tokens stored in each slot.
    int64_t
        slot_byte_size;  // Byte size of each slot in a layer. [reserved_length
                         // * sizeof a token]
    int64_t storage_byte_size;  // storage size for key or value.
                                // int64_t num_device_buffers;
    // int64_t device_buffer_byte_size; // "Area" of buffer in device. Number of
    // tokens reserved for each device buffer.
};

// struct Weights_Storage_Config{
// 	std::vector<std::string> module_types; // {"attn", "expert", "dense"}
// 	std::vector<int64_t> num_device_buffers; // Number of device buffers for
// each module type.

// 	/* “attn” -> “o_proj” -> [4096, 14336] */
// 	std::unordered_map<std::string, std::unordered_map<std::string,
// std::vector<int64_t>>> module_weights_shape;
// std::unordered_map<std::string, std::unordered_map<std::string, int64_t>>
// module_weights_byte_size;
// };

struct GPU_Buffer_Config {
    std::unordered_map<std::string, int64_t> num_prefill_module_buffer;
    std::unordered_map<std::string, int64_t> num_decoding_module_buffer;
    std::unordered_map<std::string,
                       std::unordered_map<std::string, std::vector<int64_t>>>
        module_shapes;
    int64_t num_k_buffer;
    int64_t num_v_buffer;
    int64_t num_kv_buffer;
    int64_t kv_buffer_num_tokens;  // Size of the buffer in tokens.
};

struct Basic_Config {
    std::string log_level;
    int64_t device;
    // torch::Dtype dtype_torch;
    // std::string dtype_str;
    std::string weight_dtype;
    torch::Dtype weight_dtype_torch;
    std::string kv_dtype;
    torch::Dtype kv_dtype_torch;
    std::string activation_dtype;
    torch::Dtype activation_dtype_torch;
    torch::Device device_torch = torch::Device(torch::kCPU);
    int64_t attn_mode;  // 0: kv disaggreated. 1. kv cpu. 2. kv gpu.
    int64_t num_threads = 16;
    std::vector<std::string> module_types;  // {"attn", "expert", "dense"}
    int64_t rank;
    int64_t world_size;
};

struct Module_Batching_Config {
    int64_t prefill_micro_batch_size;
    // int64_t refill_batch_size; // Lower bound for refill.
    int64_t attn_decoding_micro_batch_size;  // Batch size for CPU-GPU pipeline
                                             // or GPU if non-pipeline strategy
                                             // is used.
    int64_t expert_prefill_batch_size_upper_bound;   // CUDA OOM if exceed.
    int64_t expert_decoding_batch_size_upper_bound;  // CUDA OOM if exceed.
};

struct EngineConfig {
    Basic_Config basic_config;
    KV_Storage_Config kv_storage_config;
    // Weights_Storage_Config weights_storage_config;
    GPU_Buffer_Config gpu_buffer_config;
    Module_Batching_Config module_batching_config;
};

struct ModelConfig {
    std::string model_type;
    int64_t num_hidden_layers;
    int64_t num_local_experts;
    int64_t num_attention_heads;
    int64_t num_key_value_heads;
    int64_t hidden_size;
    int64_t intermediate_size;
    int64_t head_dim;
    int64_t compressed_kv_dim;
};

// struct copy_task{
// 	void* dst;
// 	void* src;
// 	int64_t size;
// 	std::promise<void> completion_promise;
// };

struct batch_copy_task {
    std::vector<void*> dst;
    std::vector<void*> src;
    std::vector<int64_t> sizes;
};

struct kv_h2d_copy_task {
    int64_t layer_idx;
    int64_t micro_batch_idx;
    int64_t micro_batch_width;  // longest sequence length in the micro batch.
    std::vector<int64_t> query_global_idx;
};
