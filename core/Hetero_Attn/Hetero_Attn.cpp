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

#include <condition_variable>
#include <cuda.h>
#include <cuda_runtime.h>
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
#include <sstream>
#include <string>

#include "../DtoH_Engine/DtoH_Engine.h"
#include "../GPU_Weight_Buffer/GPU_Weight_Buffer.h"
#include "../HtoD_Engine/HtoD_Engine.h"
#include "../KV_Storage/KV_Storage.h"
#include "../Weights_Storage/Weights_Storage.h"
#include "../data_structures.h"
#include "../utils.h"
#include "CPU_Kernels/cpu_kernels.h"
#include "Hetero_Attn.h"

constexpr float FP8_MAX = 448.0f;
std::tuple<torch::Tensor, torch::Tensor> compressed_kv_bf16_to_fp8_per_token(torch::Tensor x) {
    /*
        * Quantize a [bsz, seq, 576] BF16 tensor to FP8 per 128-element block.
        * Args:
        *   x: Input tensor of shape [bsz, seq, 576] with dtype bfloat16
        * Returns:
        *   q: Quantized tensor [bsz, seq, 576] with dtype float8_e4m3fn
        *   s: Scale factors [bsz, seq, num_blocks] with dtype float32
        */
    TORCH_CHECK(x.scalar_type() == at::ScalarType::BFloat16,
                "Input tensor must be of dtype BFloat16");
    TORCH_CHECK(x.size(-1) == 576,
                "Last dimension of input tensor must be 576");
    TORCH_CHECK(x.is_contiguous(),
                "Input tensor must be contiguous");
    TORCH_CHECK(x.dim() == 3,
                "Input tensor must have 3 dimensions");

    const int64_t bsz = x.size(0);
    const int64_t seq_len = x.size(1);
    const int64_t dim = x.size(2);
    const int64_t M = bsz * seq_len;

    const int64_t block_size = 128;
    const int64_t num_full_blocks = dim / block_size; // 576 / 128 = 4
    const bool has_last_block = (dim % block_size != 0);
    const int64_t last_block_size = dim % block_size; // 64
    const int64_t num_blocks = num_full_blocks + (has_last_block ? 1 : 0); // 5

    // Flatten and cast to float32
    auto x_flat = x.view({M, dim}).to(at::ScalarType::Float);

    // Prepare output tensors
    auto scale_flat = at::empty({M, num_blocks}, x_flat.options().dtype(at::ScalarType::Float));
    auto q_flat     = at::empty({M, dim},   x_flat.options().dtype(at::ScalarType::Float8_e4m3fn));

    // Process each block independently
    for (int64_t b = 0; b < num_blocks; ++b) {
        const int64_t start  = b * block_size;
        const int64_t length = (b < num_full_blocks ? block_size : last_block_size);

        auto x_block = x_flat.narrow(1, start, length);
        auto amax    = at::amax(at::abs(x_block), /*dim=*/1);
        amax = at::clamp(amax, /*min=*/1e-6f);

        // Compute scale for this block
        auto scale = amax / FP8_MAX;
        scale_flat.select(1, b).copy_(scale);

        // Quantize block
        auto y       = x_block / scale.unsqueeze(1);
        auto q_block = y.to(at::ScalarType::Float8_e4m3fn);
        q_flat.narrow(1, start, length).copy_(q_block);
    }

    // Reshape back to [bsz, seq_len, dim] and [bsz, seq_len, num_blocks]
    auto q     = q_flat.view({bsz, seq_len, dim});
    auto scale = scale_flat.view({bsz, seq_len, num_blocks});

    return std::make_tuple(q, scale);
}
    


torch::Tensor compressed_kv_fp8_to_bf16_per_token(torch::Tensor q, torch::Tensor scale) {
    /*
     * Dequantize the output of bf16_to_fp8_per_token back to BF16.
     * Args:
     *   q: Quantized tensor [bsz, seq, 576] with dtype float8_e4m3fn
     *   scale: Scale factors [bsz, seq] with dtype float32
     * Returns:
     *   x_bf16: Dequantized tensor [bsz, seq, 576] with dtype bfloat16
     */
    // TORCH_CHECK(q.scalar_type() == at::ScalarType::Float8_e4m3fn, 
    //             "Quantized tensor must be of dtype Float8_e4m3fn");
    // TORCH_CHECK(scale.scalar_type() == at::ScalarType::Float, 
    //             "Scale tensor must be of dtype Float");
    
    const auto bsz = q.size(0);
    const auto seq_len = q.size(1);
    const auto dim = q.size(2);
    const auto M = bsz * seq_len;
    
    // Flatten tensors
    auto q_flat = q.view({M, dim}).to(at::ScalarType::Float);  // upcast FP8→FP32
    auto scale_flat = scale.view({M, 1});
    
    // Rescale
    auto x_rec = q_flat * scale_flat;
    
    // Cast back to BF16 and reshape
    return x_rec.to(at::ScalarType::BFloat16).view({bsz, seq_len, dim});
}

at::Tensor dequant_per_token(const at::Tensor& q, const at::Tensor& scale) {
    /*
     * Dequantize a [bsz, seq, 576] FP8 tensor back to BF16 per 128-element block.
     * Args:
     *   q: Quantized tensor [bsz, seq, 576] with dtype float8_e4m3fn
     *   scale: Scale factors [bsz, seq, num_blocks] with dtype float32
     * Returns:
     *   x_bf16: Dequantized tensor [bsz, seq, 576] with dtype bfloat16
     */
    TORCH_CHECK(q.scalar_type() == at::ScalarType::Float8_e4m3fn,
                "Quantized tensor must be of dtype Float8_e4m3fn");
    TORCH_CHECK(scale.scalar_type() == at::ScalarType::Float,
                "Scale tensor must be of dtype Float");
    TORCH_CHECK(q.dim() == 3 && scale.dim() == 3,
                "Input tensors must be 3D");
    TORCH_CHECK(q.size(0) == scale.size(0) && q.size(1) == scale.size(1),
                "Batch and sequence dimensions must match between q and scale");

    const int64_t bsz = q.size(0);
    const int64_t seq_len = q.size(1);
    const int64_t dim = q.size(2);
    const int64_t M = bsz * seq_len;

    const int64_t block_size = 128;
    const int64_t num_full_blocks = dim / block_size;
    const bool has_last_block = (dim % block_size != 0);
    const int64_t last_block_size = dim % block_size;
    const int64_t num_blocks = num_full_blocks + (has_last_block ? 1 : 0);

    TORCH_CHECK(scale.size(2) == num_blocks,
                "Scale tensor last dimension must match number of blocks");

    // Flatten
    auto q_flat     = q.view({M, dim});
    auto scale_flat = scale.view({M, num_blocks});

    // Prepare output buffer in float32
    auto x_flat = at::empty({M, dim}, q_flat.options().dtype(at::ScalarType::Float));

    // Process each block
    for (int64_t b = 0; b < num_blocks; ++b) {
        const int64_t start  = b * block_size;
        const int64_t length = (b < num_full_blocks ? block_size : last_block_size);

        // Extract block of q, upcast to float
        auto q_block = q_flat.narrow(1, start, length).to(at::ScalarType::Float);

        // Get scale for this block [M]
        auto s_block = scale_flat.select(1, b).unsqueeze(1);

        // Reconstruct
        auto x_block = q_block * s_block;

        // Store
        x_flat.narrow(1, start, length).copy_(x_block);
    }

    // Cast back to BF16 and reshape
    auto x_bf16 = x_flat.to(at::ScalarType::BFloat16).view({bsz, seq_len, dim});
    return x_bf16;
}


Hetero_Attn::Hetero_Attn(const EngineConfig& engine_config,
                         const ModelConfig& model_config,
                         KV_Storage& kv_storage, GPU_KV_Buffer& gpu_kv_buffer,
                         HtoD_Engine& h2d_engine, DtoH_Engine& d2h_engine)
    : engine_config_(engine_config),
      model_config_(model_config),
      kv_storage_(kv_storage),
      h2d_engine_(h2d_engine),
      d2h_engine_(d2h_engine),
      gpu_kv_buffer_(gpu_kv_buffer) {
    this->logger_ = init_logger(
        this->engine_config_.basic_config.log_level,
        "Hetero_Attn" +
            std::to_string(this->engine_config_.basic_config.device));
    this->attn_mode_ = this->engine_config_.basic_config.attn_mode;
};

torch::Tensor Hetero_Attn::attn(
    py::object& PyTorch_attn_module, int64_t layer_idx,
    torch::Tensor& hidden_states, torch::Tensor& attention_mask,
    torch::Tensor& position_ids,
    std::vector<std::vector<int64_t>> cur_batching_plan) {
    // Step 1: Checkings.
    if (!Py_IsInitialized()) {
        PyErr_Print();
        throw std::runtime_error("Python interpreter is not running!");
    };

    // Step 2: call
    switch (this->attn_mode_) {
        case 0:
            return this->_attn_mode_0(PyTorch_attn_module, layer_idx,
                                      hidden_states, attention_mask,
                                      position_ids, cur_batching_plan[0]);
        case 1:
            return this->_attn_mode_1(PyTorch_attn_module, layer_idx,
                                      hidden_states, attention_mask,
                                      position_ids, cur_batching_plan);
        case 2:
            return this->_attn_mode_2(PyTorch_attn_module, layer_idx,
                                      hidden_states, attention_mask,
                                      position_ids, cur_batching_plan);
        case 3:
            return this->_attn_mode_3(PyTorch_attn_module, layer_idx,
                                      hidden_states, attention_mask,
                                      position_ids, cur_batching_plan);
        default:
            throw std::runtime_error("Unsupported attn_mode_");
            
    };
};

// std::string get_tensor_shape(const torch::Tensor& t) {
//     std::stringstream ss;
//     ss << "[";
//     for (int i = 0; i < t.dim(); ++i) {
//         ss << t.size(i);
//         if (i < t.dim() - 1) ss << ", ";
//     }
//     ss << "]";
//     return ss.str();
// }

torch::Tensor Hetero_Attn::_attn_mode_0(py::object& PyTorch_attn_module,
                                        int64_t layer_idx,
                                        torch::Tensor& hidden_states,
                                        torch::Tensor& attention_mask,
                                        torch::Tensor& position_ids,
                                        std::vector<int64_t> batch) {
    /*
      Single thread.
      - Call python module's QKV_Proj
      - Call full attn_mechanism kernel. attn_QKT_softmax_CPU + attn_AV
      - Call python module's O_Proj
    */
    CUDA_CHECK(cudaSetDevice(this->engine_config_.basic_config.device));

    this->logger_->debug("MS 1");

    auto kv_seq_len = attention_mask.size(-1);
    auto qkv_result =
        PyTorch_attn_module
            .attr("QKV_Proj")(hidden_states, position_ids, kv_seq_len)
            .cast<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>>();
    CUDA_CHECK(cudaStreamSynchronize(0));

    this->logger_->debug("QKV_Proj computed.");
    auto [query_states, key_states, value_states] = qkv_result;
    this->logger_->debug("QKV decomposed.");
    attention_mask = attention_mask.to(torch::kCPU);
    this->logger_->debug("Attention mask shape: {}",
                         get_tensor_shape(attention_mask));
    // query_states = this->d2h_engine_.tensor_on_demand_copy(query_states);
    query_states = query_states.to(torch::kCPU);

    // TODO: revise for other models.
    // this->kv_storage_.update(layer_idx, batch, key_states, value_states,
    //                          this->kv_storage_.get_k_quantize_scale(layer_idx));
    this->logger_->debug("K and V updated to the storage.");

    auto k_ptrs = this->kv_storage_.get_k_ptrs(layer_idx, batch);
    auto v_ptrs = this->kv_storage_.get_v_ptrs(layer_idx, batch);

    this->logger_->debug("K and V pointers size: {}", k_ptrs.size());
    this->logger_->debug("num threads: {}",
                         this->engine_config_.basic_config.num_threads);
    this->logger_->debug("query_states shape: {}",
                         get_tensor_shape(query_states));
    torch::Tensor attn_output;
    try {
        auto kernel_start = std::chrono::high_resolution_clock::now();
        attn_output = grouped_query_attention_cpu_avx2(
            query_states, k_ptrs, v_ptrs, attention_mask,
            attention_mask.size(-1),
            this->model_config_.num_attention_heads /
                this->model_config_.num_key_value_heads,
            this->model_config_.num_attention_heads,
            this->model_config_.num_key_value_heads,
            this->model_config_.head_dim,
            this->engine_config_.basic_config.num_threads);
        auto kernel_end = std::chrono::high_resolution_clock::now();
        auto kernel_duration =
            std::chrono::duration_cast<std::chrono::microseconds>(kernel_end -
                                                                  kernel_start);
        this->logger_->debug("CPU Attn kernel duration: {} microseconds.",
                             kernel_duration.count());

        this->logger_->debug("CPU Attn output computed.");
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        throw;
    } catch (...) {
        std::cerr << "Unknown error occurred." << std::endl;
        throw;
    }
    attn_output = this->h2d_engine_.tensor_on_demand_copy(attn_output);

    auto final_output =
        PyTorch_attn_module.attr("O_Proj")(attn_output).cast<torch::Tensor>();
    CUDA_CHECK(cudaStreamSynchronize(0));
    // CUDA_CHECK(cudaDeviceSynchronize());

    return final_output;
};

torch::Tensor Hetero_Attn::_attn_mode_1(
    py::object& PyTorch_attn_module, int64_t layer_idx,
    torch::Tensor& hidden_states, torch::Tensor& attention_mask,
    torch::Tensor& position_ids,
    std::vector<std::vector<int64_t>> micro_batches)
{
    // this->logger_->info("Hetero_Attn::_attn_mode_1");
    /*
            FULL GPU MODE.
    */
    CUDA_CHECK(cudaSetDevice(this->engine_config_.basic_config.device));
    std::vector<torch::Tensor> full_new_k;
    std::vector<torch::Tensor> full_factor;
    torch::Tensor final_output = torch::zeros_like(hidden_states);
    if (this->model_config_.model_type.find("deepseek") == std::string::npos) {
        for (int64_t micro_batch_idx = 0;
             micro_batch_idx < micro_batches.size(); micro_batch_idx++) {
            auto cur_batch = micro_batches[micro_batch_idx];
            auto cur_batch_size = cur_batch.size();

            int64_t bsz = cur_batch_size;
            int64_t kv_seq_len = attention_mask.size(-1) - 1;
            std::vector<int64_t> tensor_shape = {
                bsz, kv_seq_len, this->model_config_.num_key_value_heads,
                this->model_config_.head_dim};

            auto cur_k = this->gpu_kv_buffer_.get_k(layer_idx, micro_batch_idx,
                                                    tensor_shape);
            auto cur_v = this->gpu_kv_buffer_.get_v(layer_idx, micro_batch_idx,
                                                    tensor_shape);

            int64_t cur_batch_start_idx = 0;
            for (int64_t i = 0; i < micro_batch_idx; i++) {
                cur_batch_start_idx += micro_batches[i].size();
            };
            auto cur_hidden_states = hidden_states.index(
                {torch::indexing::Slice(cur_batch_start_idx,
                                        cur_batch_start_idx + cur_batch_size)});
            auto module_output =
                PyTorch_attn_module
                    .attr("decoding_attn")(
                        cur_hidden_states, cur_k, cur_v,
                        attention_mask.index({torch::indexing::Slice(
                            cur_batch_start_idx,
                            cur_batch_start_idx + cur_batch_size)}),
                        position_ids.index({torch::indexing::Slice(
                            cur_batch_start_idx,
                            cur_batch_start_idx + cur_batch_size)}))
                    .cast<std::tuple<torch::Tensor, torch::Tensor,
                                     torch::Tensor>>();
            CUDA_CHECK(cudaStreamSynchronize(0));

            auto [attn_result, new_k, new_v] = module_output;
            this->gpu_kv_buffer_.releaseBuffer(layer_idx, micro_batch_idx);
            
            // this->kv_storage_.update(layer_idx, cur_batch, new_k,
            //                          new_v, );  // Todo.
            final_output.index_put_(
                {torch::indexing::Slice(cur_batch_start_idx,
                                        cur_batch_start_idx + cur_batch_size)},
                attn_result);
            // py::gil_scoped_release release;
        }
    } else {
        CUDA_CHECK(cudaSetDevice(this->engine_config_.basic_config.device));
        // this->logger_->info("Hetero_Attn::_attn_mode_1 deepseek");
        // auto dequantize_factor = this->kv_storage_.get_k_quantize_scale(layer_idx);
        // this->logger_->info("dequantize_factor got");
        // Check if the dequantize_factor contains nan
        // if (torch::any(torch::isnan(dequantize_factor)).item<bool>()) {
        //     this->logger_->error("Dequantize factor contains NaN values, rank: {}",
        //                          this->engine_config_.basic_config.device);
        //     throw std::runtime_error("Dequantize factor contains NaN values.");
        // }
        for (int64_t micro_batch_idx = 0;
             micro_batch_idx < micro_batches.size(); micro_batch_idx++) {
            auto cur_batch = micro_batches[micro_batch_idx];
            auto cur_batch_size = cur_batch.size();

            int64_t bsz = cur_batch_size;
            // int64_t kv_seq_len = attention_mask.size(-1) - 1;
            int64_t kv_seq_len = attention_mask.size(-1); // We copy one more token which is the place holder for new Q.
            std::vector<int64_t> tensor_shape = {
                bsz, kv_seq_len, this->model_config_.compressed_kv_dim};
            
            // CUDA_CHECK(cudaDeviceSynchronize());
            auto cur_k = this->gpu_kv_buffer_.get_k(
                layer_idx, micro_batch_idx, tensor_shape);
            // Check if the external_tensor contains nan
            // if (torch::any(torch::isnan(cur_k)).item<bool>()) {
            //     for(int i = 0; i < cur_k.size(0); i++) {
            //         if (torch::any(torch::isnan(cur_k[i])).item<bool>()) {
            //             this->logger_->error("cur_k contains NaN values, rank: {}, i",
            //                                     this->engine_config_.basic_config.device, i);
            //         }
                
            //     }
            //     this->logger_->error("cur_k contains NaN values, rank: {}",
            //                          this->engine_config_.basic_config.device);
            //     // throw std::runtime_error("cur_k contains NaN values.");
            // }


            // auto cur_k = torch::zeros_like(external_tensor);
            // cur_k.copy_(external_tensor);
            // auto cur_k = external_tensor.clone();
            // CUDA_CHECK(cudaStreamSynchronize(0));
            // CUDA_CHECK(cudaDeviceSynchronize());
            // Check if cur_k contains nan
            // if (torch::any(torch::isnan(cur_k)).item<bool>()) {
            //     this->logger_->error("cur_k contains NaN values, rank: {}",
            //                          this->engine_config_.basic_config.device);
            //     // throw std::runtime_error("cur_k contains NaN values.");
            // }

            // file name: rank_0_layer_0_micro_batch_0_k.pt
            // std::string file_name = "/workspace/rank_" +
            //     std::to_string(this->engine_config_.basic_config.device) +
            //     "_layer_" + std::to_string(layer_idx) + "_micro_batch_" +
            //     std::to_string(micro_batch_idx) + "_k.pt";
            // torch::save(cur_k, file_name);
            // exit(0);
            

            this->gpu_kv_buffer_.releaseBuffer(layer_idx, micro_batch_idx);

            int64_t cur_batch_start_idx = 0;
            for (int64_t i = 0; i < micro_batch_idx; i++) {
                cur_batch_start_idx += micro_batches[i].size();
            };
            auto cur_hidden_states = hidden_states.index(
                {torch::indexing::Slice(cur_batch_start_idx,
                                        cur_batch_start_idx + cur_batch_size)});
            // CUDA_CHECK(cudaStreamSynchronize(0));
            // CUDA_CHECK(cudaDeviceSynchronize());
            torch::Tensor cur_v = torch::empty(
                {0}, torch::TensorOptions()
                         .dtype(this->engine_config_.basic_config.kv_dtype_torch)
                         .device(torch::kCUDA,
                                 this->engine_config_.basic_config.device)
                         .requires_grad(false)
                         .memory_format(torch::MemoryFormat::Contiguous));
            std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
                module_output;
            
            // this->logger_->info("start dequantization");
            // auto cur_factor = dequantize_factor.index(
            //     {torch::indexing::Slice(cur_batch_start_idx,
            //                             cur_batch_start_idx + cur_batch_size)});
            int64_t padding_length = cur_k.size(1);
            auto cur_factor = this->kv_storage_.get_k_quantize_scale(layer_idx, cur_batch, padding_length);            
            
            // auto dequant_k = dequant_per_token(cur_k, cur_factor);

            // Check if the dequant_k contains nan

            // if (torch::any(torch::isnan(dequant_k)).item<bool>()) {
            //     this->logger_->error("dequant_k contains NaN values, rank: {}",
            //                          this->engine_config_.basic_config.device);
            //     // throw std::runtime_error("dequant_k contains NaN values.");
            // }

            {
                py::gil_scoped_acquire acquire;
                module_output =
                    PyTorch_attn_module
                        .attr("decoding_attn")(
                            cur_hidden_states, cur_k, cur_v,
                            attention_mask.index({torch::indexing::Slice(
                                cur_batch_start_idx,
                                cur_batch_start_idx + cur_batch_size)}),
                            position_ids.index({torch::indexing::Slice(
                                cur_batch_start_idx,
                                cur_batch_start_idx + cur_batch_size)}),
                            cur_factor)
                        .cast<std::tuple<torch::Tensor, torch::Tensor,
                                        torch::Tensor>>();
            }
            CUDA_CHECK(cudaStreamSynchronize(0));
            // CUDA_CHECK(cudaDeviceSynchronize());    
            auto [attn_result, new_k, new_v] = module_output;
            // Check if the attn_result and new_k contain nan
            // if (torch::any(torch::isnan(attn_result)).item<bool>()) {
            //     this->logger_->error("attn_result contains NaN values, rank: {}",
            //                          this->engine_config_.basic_config.device);
            //     // throw std::runtime_error("attn_result contains NaN values.");
            // }
            // if (torch::any(torch::isnan(new_k)).item<bool>()) {
            //     this->logger_->error("new_k contains NaN values, rank: {}",
            //                          this->engine_config_.basic_config.device);
            //     // throw std::runtime_error("new_k contains NaN values.");
            // }
            auto [quant_k, factor] = compressed_kv_bf16_to_fp8_per_token(new_k);
            // this->kv_storage_.update(layer_idx, cur_batch, quant_k, new_v, factor);
            full_new_k.push_back(quant_k);
            full_factor.push_back(factor);
            // CUDA_CHECK(cudaStreamSynchronize(0));
            final_output.index_put_(
                {torch::indexing::Slice(cur_batch_start_idx,
                                        cur_batch_start_idx + cur_batch_size)},
                attn_result);

            // CUDA_CHECK(cudaStreamSynchronize(0)); 
            // CUDA_CHECK(cudaDeviceSynchronize());   
        }
    }
    torch::Tensor quant_k = torch::cat(full_new_k, 0);
    torch::Tensor factor = torch::cat(full_factor, 0);
    // Merge all micro batches into one vector.
    std::vector<int64_t> idx = {};
    for (int64_t i = 0; i < micro_batches.size(); i++) {
        auto cur_batch = micro_batches[i];
        for (int64_t j = 0; j < cur_batch.size(); j++) {
            idx.push_back(cur_batch[j]);
        }
    }
    // new_v is a place holder
    torch::Tensor new_v = torch::empty(
        {0}, torch::TensorOptions()
                 .dtype(this->engine_config_.basic_config.kv_dtype_torch)
                 .device(torch::kCUDA,
                         this->engine_config_.basic_config.device)
                 .requires_grad(false)
                 .memory_format(torch::MemoryFormat::Contiguous));

    this->kv_storage_.update(layer_idx, idx, quant_k, new_v, factor);
    // CUDA_CHECK(cudaStreamSynchronize(0));
    // CUDA_CHECK(cudaDeviceSynchronize());
    return final_output;
};

torch::Tensor Hetero_Attn::_attn_mode_2(
    py::object& PyTorch_attn_module, int64_t layer_idx,
    torch::Tensor& hidden_states, torch::Tensor& attention_mask,
    torch::Tensor& position_ids,
    std::vector<std::vector<int64_t>> micro_batches) {
    /*
            - Call python module's QKV_Proj
            - CPU GPU attn parallel.
            - Call python module's O_Proj
    */
    CUDA_CHECK(cudaSetDevice(this->engine_config_.basic_config.device));
    auto CPU_Batch = micro_batches[0];
    std::vector<std::vector<int64_t>> GPU_Batch;
    for (int64_t i = 1; i < micro_batches.size(); i++) {
        GPU_Batch.push_back(micro_batches[i]);
    }
    auto kv_seq_len = attention_mask.size(-1);
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> qkv_result;
    qkv_result =
        PyTorch_attn_module
            .attr("QKV_Proj")(hidden_states, position_ids, kv_seq_len)
            .cast<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>>();
    auto [query_states, key_states, value_states] = qkv_result;

    int64_t CPU_batch_start_idx = 0;
    int64_t CPU_batch_end_idx = CPU_Batch.size();
    int64_t GPU_batch_start_idx = CPU_Batch.size();
    int64_t GPU_batch_end_idx = query_states.size(0);

    auto CPU_query_states = query_states
                                .index({torch::indexing::Slice(
                                    CPU_batch_start_idx, CPU_batch_end_idx)})
                                .to(torch::kCPU);
    auto CPU_key_states = key_states.index(
        {torch::indexing::Slice(CPU_batch_start_idx, CPU_batch_end_idx)});
    auto CPU_value_states = value_states.index(
        {torch::indexing::Slice(CPU_batch_start_idx, CPU_batch_end_idx)});
    auto CPU_attention_mask = attention_mask
                                  .index({torch::indexing::Slice(
                                      CPU_batch_start_idx, CPU_batch_end_idx)})
                                  .to(torch::kCPU);

    std::future<torch::Tensor> CPU_result_future;
    CPU_result_future = this->CPU_attn_mechanism(
        layer_idx, CPU_query_states, CPU_key_states, CPU_value_states,
        CPU_attention_mask, CPU_Batch);

    auto GPU_query_states = query_states.index(
        {torch::indexing::Slice(GPU_batch_start_idx, GPU_batch_end_idx)});
    auto GPU_key_states = key_states.index(
        {torch::indexing::Slice(GPU_batch_start_idx, GPU_batch_end_idx)});
    auto GPU_value_states = value_states.index(
        {torch::indexing::Slice(GPU_batch_start_idx, GPU_batch_end_idx)});
    auto GPU_attention_mask = attention_mask.index(
        {torch::indexing::Slice(GPU_batch_start_idx, GPU_batch_end_idx)});
    GPU_attention_mask =
        this->h2d_engine_.tensor_on_demand_copy(GPU_attention_mask);
    std::vector<torch::Tensor> GPU_result;
    this->logger_->debug("GPU Batch size: {}", GPU_Batch.size());
    this->logger_->debug("GPU Computation Start.");
    auto GPU_start = std::chrono::high_resolution_clock::now();
    for (int64_t GPU_micro_batch_idx = 0;
         GPU_micro_batch_idx < GPU_Batch.size(); GPU_micro_batch_idx++) {
        auto cur_batch = GPU_Batch[GPU_micro_batch_idx];
        auto cur_batch_size = cur_batch.size();
        int64_t bsz = cur_batch_size;
        int64_t kv_seq_len = attention_mask.size(-1) - 1;
        std::vector<int64_t> tensor_shape = {
            bsz, kv_seq_len, this->model_config_.num_key_value_heads,
            this->model_config_.head_dim};
        auto cur_k = this->gpu_kv_buffer_.get_k(layer_idx, GPU_micro_batch_idx,
                                                tensor_shape);
        auto cur_v = this->gpu_kv_buffer_.get_v(layer_idx, GPU_micro_batch_idx,
                                                tensor_shape);
        this->logger_->debug("cur batch size: {}", cur_batch_size);
        // CUDA_CHECK(cudaStreamSynchronize(0));
        int64_t cur_batch_start_idx = 0;
        for (int64_t i = 0; i < GPU_micro_batch_idx; i++) {
            cur_batch_start_idx += GPU_Batch[i].size();
        };
        auto cur_query_states = GPU_query_states.index({torch::indexing::Slice(
            cur_batch_start_idx, cur_batch_start_idx + cur_batch_size)});
        auto new_k = GPU_key_states.index({torch::indexing::Slice(
            cur_batch_start_idx, cur_batch_start_idx + cur_batch_size)});
        auto new_v = GPU_value_states.index({torch::indexing::Slice(
            cur_batch_start_idx, cur_batch_start_idx + cur_batch_size)});

        cur_k = torch::cat({cur_k, new_k}, 2);
        cur_v = torch::cat({cur_v, new_v}, 2);

        torch::Tensor attn_output;
        this->logger_->debug(
            "starting attn computation for GPU micro batch: {}",
            GPU_micro_batch_idx);
        auto attn_weights =
            PyTorch_attn_module
                .attr("attn_QKT_softmax")(
                    cur_query_states, cur_k,
                    GPU_attention_mask.index({torch::indexing::Slice(
                        cur_batch_start_idx,
                        cur_batch_start_idx + cur_batch_size)}))
                .cast<torch::Tensor>();
        this->logger_->debug("attn_weights shape: {}",
                             get_tensor_shape(attn_weights));
        attn_output = PyTorch_attn_module.attr("attn_AV")(attn_weights, cur_v)
                          .cast<torch::Tensor>();

        this->gpu_kv_buffer_.releaseBuffer(layer_idx, GPU_micro_batch_idx);
        // this->kv_storage_.update(layer_idx, cur_batch, new_k, new_v); //TODO
        GPU_result.push_back(attn_output);
    };
    auto GPU_end = std::chrono::high_resolution_clock::now();
    auto GPU_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        GPU_end - GPU_start);
    this->logger_->debug("GPU Computation Duration: {} ms",
                         GPU_duration.count());
    // CUDA_CHECK(cudaStreamSynchronize(0));
    auto cat_start = std::chrono::high_resolution_clock::now();
    auto GPU_result_tensor = torch::cat(GPU_result, 0);
    auto cat_end = std::chrono::high_resolution_clock::now();
    auto cat_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        cat_end - cat_start);
    this->logger_->debug("GPU Result cat duration: {} ms",
                         cat_duration.count());
    this->logger_->debug("GPU Result shape: {}, start waiting for CPU result.",
                         get_tensor_shape(GPU_result_tensor));
    auto start = std::chrono::high_resolution_clock::now();
    auto CPU_result = CPU_result_future.get();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    this->logger_->debug("Wait for CPU result: {} ms", duration.count());

    auto attn_output = torch::cat({CPU_result, GPU_result_tensor}, 0);
    auto final_output =
        PyTorch_attn_module.attr("O_Proj")(attn_output).cast<torch::Tensor>();
    return final_output;
};

std::future<torch::Tensor> Hetero_Attn::CPU_attn_mechanism(
    int64_t layer_idx, torch::Tensor query_states, torch::Tensor& key_states,
    torch::Tensor& value_states, torch::Tensor& attention_mask,
    std::vector<int64_t> batch) {
    auto promise = std::make_shared<std::promise<torch::Tensor>>();
    std::future<torch::Tensor> future = promise->get_future();
    std::thread([this, promise, query_states, key_states, value_states,
                 attention_mask, layer_idx, batch]() {
        try {
            // auto start = std::chrono::high_resolution_clock::now();
            CUDA_CHECK(cudaSetDevice(this->engine_config_.basic_config.device));
            // this->kv_storage_.update(layer_idx, batch, key_states,
            //                          value_states); // TODO
            this->logger_->debug("num threads: {}",
                                 this->engine_config_.basic_config.num_threads);
            auto k_ptrs = this->kv_storage_.get_k_ptrs(layer_idx, batch);
            auto v_ptrs = this->kv_storage_.get_v_ptrs(layer_idx, batch);
            auto start = std::chrono::high_resolution_clock::now();
            this->logger_->debug("Calling CPU kernel.");
            auto attn_output = grouped_query_attention_cpu_avx2(
                query_states, k_ptrs, v_ptrs, attention_mask,
                attention_mask.size(-1),
                this->model_config_.num_attention_heads /
                    this->model_config_.num_key_value_heads,
                this->model_config_.num_attention_heads,
                this->model_config_.num_key_value_heads,
                this->model_config_.head_dim,
                this->engine_config_.basic_config.num_threads);
            // attn_output =
            // this->h2d_engine_.tensor_on_demand_copy(attn_output);
            attn_output =
                attn_output.to(this->engine_config_.basic_config.device_torch);
            auto end = std::chrono::high_resolution_clock::now();
            auto duration =
                std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                      start);
            this->logger_->debug("CPU Attn kernel duration: {} ms",
                                 duration.count());
            auto set_start = std::chrono::high_resolution_clock::now();
            promise->set_value(attn_output);
            auto set_end = std::chrono::high_resolution_clock::now();
            auto set_duration =
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    set_end - set_start);
            this->logger_->debug("CPU Attn set value duration: {} ms",
                                 set_duration.count());
        } catch (...) {
            promise->set_exception(std::current_exception());
        }
    }).detach();

    return future;
};


torch::Tensor Hetero_Attn::_attn_mode_3(
    py::object& PyTorch_attn_module, int64_t layer_idx,
    torch::Tensor& hidden_states, torch::Tensor& attention_mask,
    torch::Tensor& position_ids,
    std::vector<std::vector<int64_t>> micro_batches)
{
    /*
        Decoding. KV Managed in GPU with DP pattern.
    */
    CUDA_CHECK(cudaSetDevice(this->engine_config_.basic_config.device));
    std::vector<torch::Tensor> full_new_k;
    std::vector<torch::Tensor> full_factor;
    torch::Tensor final_output = torch::zeros_like(hidden_states);
    if (this->model_config_.model_type.find("deepseek") == std::string::npos) {
        for (int64_t micro_batch_idx = 0;
             micro_batch_idx < micro_batches.size(); micro_batch_idx++) {
            auto cur_batch = micro_batches[micro_batch_idx];
            auto cur_batch_size = cur_batch.size();

            int64_t bsz = cur_batch_size;
            int64_t kv_seq_len = attention_mask.size(-1) - 1;
            std::vector<int64_t> tensor_shape = {
                bsz, kv_seq_len, this->model_config_.num_key_value_heads,
                this->model_config_.head_dim};

            auto cur_k = this->gpu_kv_buffer_.get_k(layer_idx, micro_batch_idx,
                                                    tensor_shape);
            auto cur_v = this->gpu_kv_buffer_.get_v(layer_idx, micro_batch_idx,
                                                    tensor_shape);

            int64_t cur_batch_start_idx = 0;
            for (int64_t i = 0; i < micro_batch_idx; i++) {
                cur_batch_start_idx += micro_batches[i].size();
            };
            auto cur_hidden_states = hidden_states.index(
                {torch::indexing::Slice(cur_batch_start_idx,
                                        cur_batch_start_idx + cur_batch_size)});
            auto module_output =
                PyTorch_attn_module
                    .attr("decoding_attn")(
                        cur_hidden_states, cur_k, cur_v,
                        attention_mask.index({torch::indexing::Slice(
                            cur_batch_start_idx,
                            cur_batch_start_idx + cur_batch_size)}),
                        position_ids.index({torch::indexing::Slice(
                            cur_batch_start_idx,
                            cur_batch_start_idx + cur_batch_size)}))
                    .cast<std::tuple<torch::Tensor, torch::Tensor,
                                     torch::Tensor>>();
            CUDA_CHECK(cudaStreamSynchronize(0));

            auto [attn_result, new_k, new_v] = module_output;
            this->gpu_kv_buffer_.releaseBuffer(layer_idx, micro_batch_idx);
            
            // this->kv_storage_.update(layer_idx, cur_batch, new_k,
            //                          new_v, );  // Todo.
            final_output.index_put_(
                {torch::indexing::Slice(cur_batch_start_idx,
                                        cur_batch_start_idx + cur_batch_size)},
                attn_result);
            // py::gil_scoped_release release;
        }
    } else {
        CUDA_CHECK(cudaSetDevice(this->engine_config_.basic_config.device));
        for (int64_t micro_batch_idx = 0;
             micro_batch_idx < micro_batches.size(); micro_batch_idx++) {
            auto cur_batch = micro_batches[micro_batch_idx];
            auto cur_batch_size = cur_batch.size();

            int64_t bsz = cur_batch_size;
            // int64_t kv_seq_len = attention_mask.size(-1) - 1;
            int64_t kv_seq_len = attention_mask.size(-1); // We copy one more token which is the place holder for new Q.
            std::vector<int64_t> tensor_shape = {
                bsz, kv_seq_len, this->model_config_.compressed_kv_dim};
            
            // auto cur_k = this->gpu_kv_buffer_.get_gpu_k(
            //     layer_idx, micro_batch_idx, tensor_shape);

            // this->gpu_kv_buffer_.releaseBuffer(layer_idx, micro_batch_idx);
            auto cur_k = this->kv_storage_.get_k(layer_idx, cur_batch, tensor_shape);

            int64_t cur_batch_start_idx = 0;
            for (int64_t i = 0; i < micro_batch_idx; i++) {
                cur_batch_start_idx += micro_batches[i].size();
            };
            auto cur_hidden_states = hidden_states.index(
                {torch::indexing::Slice(cur_batch_start_idx,
                                        cur_batch_start_idx + cur_batch_size)});
            // CUDA_CHECK(cudaStreamSynchronize(0));
            // CUDA_CHECK(cudaDeviceSynchronize());
            torch::Tensor cur_v = torch::empty(
                {0}, torch::TensorOptions()
                         .dtype(this->engine_config_.basic_config.kv_dtype_torch)
                         .device(torch::kCUDA,
                                 this->engine_config_.basic_config.device)
                         .requires_grad(false)
                         .memory_format(torch::MemoryFormat::Contiguous));
            std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
                module_output;
            

            int64_t padding_length = cur_k.size(1);
            auto cur_factor = this->kv_storage_.get_k_quantize_scale(layer_idx, cur_batch, padding_length);            

            {
                py::gil_scoped_acquire acquire;
                module_output =
                    PyTorch_attn_module
                        .attr("decoding_attn")(
                            cur_hidden_states, cur_k, cur_v,
                            attention_mask.index({torch::indexing::Slice(
                                cur_batch_start_idx,
                                cur_batch_start_idx + cur_batch_size)}),
                            position_ids.index({torch::indexing::Slice(
                                cur_batch_start_idx,
                                cur_batch_start_idx + cur_batch_size)}),
                            cur_factor)
                        .cast<std::tuple<torch::Tensor, torch::Tensor,
                                        torch::Tensor>>();
            }
            CUDA_CHECK(cudaStreamSynchronize(0));   
            auto [attn_result, new_k, new_v] = module_output;
            auto [quant_k, factor] = compressed_kv_bf16_to_fp8_per_token(new_k);
            // this->kv_storage_.update(layer_idx, cur_batch, quant_k, new_v, factor);
            full_new_k.push_back(quant_k);
            full_factor.push_back(factor);
            // CUDA_CHECK(cudaStreamSynchronize(0));
            final_output.index_put_(
                {torch::indexing::Slice(cur_batch_start_idx,
                                        cur_batch_start_idx + cur_batch_size)},
                attn_result);

            // CUDA_CHECK(cudaStreamSynchronize(0)); 
            // CUDA_CHECK(cudaDeviceSynchronize());   
        }
    }
    torch::Tensor quant_k = torch::cat(full_new_k, 0);
    torch::Tensor factor = torch::cat(full_factor, 0);
    // Merge all micro batches into one vector.
    std::vector<int64_t> idx = {};
    for (int64_t i = 0; i < micro_batches.size(); i++) {
        auto cur_batch = micro_batches[i];
        for (int64_t j = 0; j < cur_batch.size(); j++) {
            idx.push_back(cur_batch[j]);
        }
    }
    // new_v is a place holder
    torch::Tensor new_v = torch::empty(
        {0}, torch::TensorOptions()
                 .dtype(this->engine_config_.basic_config.kv_dtype_torch)
                 .device(torch::kCUDA,
                         this->engine_config_.basic_config.device)
                 .requires_grad(false)
                 .memory_format(torch::MemoryFormat::Contiguous));

    this->kv_storage_.gpu_kv_update_func(layer_idx, idx, quant_k, new_v, factor);
    return final_output;
};