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

#include <c10/util/BFloat16.h>
#include <chrono>
#include <immintrin.h>
#include <numa.h>
#include <numaif.h>
#include <sched.h>
#include <torch/extension.h>
#include <torch/torch.h>
#include <vector>

#include "cpu_kernels.h"

torch::Tensor repeat_kv(const torch::Tensor& hidden_states, int64_t n_rep) {
    auto sizes = hidden_states.sizes();
    int64_t batch = sizes[0];
    int64_t num_key_value_heads = sizes[1];
    int64_t slen = sizes[2];
    int64_t head_dim = sizes[3];

    if (n_rep == 1) {
        return hidden_states;
    }

    // Add new dimension: [:, :, None, :, :]
    auto expanded = hidden_states.unsqueeze(2);

    // Expand along the new dimension
    std::vector<int64_t> expand_sizes = {batch, num_key_value_heads, n_rep,
                                         slen, head_dim};
    auto expanded_tensor = expanded.expand(expand_sizes);

    // Reshape to final shape
    return expanded_tensor.reshape(
        {batch, num_key_value_heads * n_rep, slen, head_dim});
};

int main(int argc, char* argv[]) {
    // Initialize NUMA
    if (numa_available() < 0) {
        std::cerr << "NUMA is not available\n";
        return 1;
    }

    int64_t bsz = 16;
    int64_t num_attention_heads = 128;
    int64_t num_kv_head = 128;
    int64_t head_dim = 192;
    int64_t cache_length = 16384;
    int64_t num_threads = 16;
    if (argc > 1) {
        num_threads = atoi(argv[1]);
    }
    torch::manual_seed(42);
    auto option = torch::TensorOptions()
                      .dtype(torch::kFloat32)
                      .requires_grad(false)
                      .device(torch::kCPU);
    torch::Tensor query_states =
        torch::empty({bsz, num_attention_heads, 1, head_dim}, option)
            .to(torch::kBFloat16)
            .contiguous();
    std::vector<c10::BFloat16*> key_states;
    std::vector<c10::BFloat16*> value_states;
    auto key_tensor =
        torch::empty({bsz, cache_length, num_kv_head, head_dim}, option)
            .to(torch::kBFloat16);
    auto value_tensor =
        torch::empty({bsz, cache_length, num_kv_head, head_dim}, option)
            .to(torch::kBFloat16);
    for (int i = 0; i < bsz; ++i) {
        // auto key_tensor = torch::randn({cache_length * num_kv_head *
        // head_dim}, option); auto value_tensor = torch::randn({cache_length *
        // num_kv_head * head_dim}, option);
        c10::BFloat16* key =
            static_cast<c10::BFloat16*>(key_tensor[i].data_ptr());
        c10::BFloat16* value =
            static_cast<c10::BFloat16*>(value_tensor[i].data_ptr());
        key_states.push_back(key);
        value_states.push_back(value);
    }
    torch::Tensor attention_mask =
        torch::randint(0, 2, {bsz, cache_length}, option)
            .to(torch::kBFloat16)
            .contiguous();

    // Convert values:
    // 0 -> negative extreme of bfloat16 (around -3.389e+38)
    // 1 -> 0.0
    attention_mask =
        torch::where(
            attention_mask == 0,
            torch::full_like(attention_mask,
                             -std::numeric_limits<at::BFloat16>::max()),
            torch::zeros_like(attention_mask))
            .contiguous();

    // c10::BFloat16* attn_weights = static_cast<c10::BFloat16*>(
    // 	std::malloc(bsz * num_attention_heads * 1 * cache_length *
    // sizeof(c10::BFloat16)));
    std::cout << "Input tensors created" << std::endl;

    // Force all previous operations to complete
    // asm volatile("" ::: "memory");
    // sched_yield();  // Let other processes run

    // // Start measuring here
    // asm volatile("###START###" ::: "memory");  // Marker for perf

    // Set thread affinity
    // cpu_set_t cpuset;
    // CPU_ZERO(&cpuset);
    // for (int i = 0; i < num_threads; i++) {
    //     CPU_SET(i, &cpuset);
    // }
    // sched_setaffinity(0, sizeof(cpuset), &cpuset);

    torch::Tensor attn_output;
    for (int i = 0; i < 10; i++) {
        auto start_time = std::chrono::high_resolution_clock::now();
        attn_output = grouped_query_attention_cpu_avx2(
            query_states, key_states, value_states, attention_mask,
            cache_length, 1, num_attention_heads, num_kv_head, head_dim,
            num_threads);
        auto end_time = std::chrono::high_resolution_clock::now();
        std::cout << "Time taken: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(
                         end_time - start_time)
                         .count()
                  << " ms" << std::endl;
    };

    // End measuring here
    // asm volatile("###END###" ::: "memory");    // Marker for perf

    // Verify correctness
    // auto key_tensor_permuted = key_tensor.permute({0, 2, 3, 1});
    // auto value_tensor_permuted = value_tensor.permute({0, 2, 1, 3});
    // std::cout<< "key tensor permuted shape: " << key_tensor_permuted.sizes()
    // << std::endl; auto start = std::chrono::high_resolution_clock::now();
    // auto torch_result = query_states.matmul(key_tensor_permuted) /
    // std::sqrt(static_cast<float>(head_dim)); torch_result +=
    // attention_mask.unsqueeze(1).unsqueeze(2); torch_result =
    // torch::softmax(torch_result, 3); torch_result =
    // torch_result.matmul(value_tensor_permuted); auto end =
    // std::chrono::high_resolution_clock::now(); std::cout << "PyTorch time
    // taken: " << std::chrono::duration_cast<std::chrono::milliseconds>(end -
    // start).count() << " ms" << std::endl; bool is_identical =
    // torch::allclose(attn_output, torch_result, 7.8125e-3, 7.8125e-3);
    // // bool is_identical = torch::allclose(attn_output, torch_result, 1e-2,
    // 1e-2); std::cout << "Outputs are identical: " << is_identical <<
    // std::endl;

    // // Check difference
    // auto diff = attn_output - torch_result;
    // auto diff_norm = diff.norm().item<double>();
    // std::cout << "Difference norm: " << diff_norm << std::endl;
    // // Ratio of elements that difference larger than 1/128
    // auto diff_ratio = torch::count_nonzero(diff.abs()
    // > 7.8125e-3).item<int>() / float(diff.numel()); std::cout << "Proportion
    // of elements with difference > 1/128: " << diff_ratio << std::endl;
    // // Show the absolute number of elements that difference larger than 1/128
    // and print the first 5 auto diff_indices = (diff.abs()
    // > 7.8125e-3).nonzero(); std::cout << "Number of elements with difference
    // > 1/128: " << diff_indices.size(0) << std::endl; for (int i = 0; i < 5;
    // i++) { 	auto idx = diff_indices[i]; 	std::cout << "C++ output: " <<
    // attn_output[idx[0]][idx[1]][idx[2]][idx[3]] << ", Python output: " <<
    // torch_result[idx[0]][idx[1]][idx[2]][idx[3]] << std::endl;
    // }

    return 0;
};

// int main() {
//     // Initialize NUMA
//     if (numa_available() < 0) {
//         std::cerr << "NUMA is not available\n";
//         return 1;
//     }

//     // Configure dimensions (unchanged)
//     int64_t bsz = 400;
//     int64_t num_attention_heads = 32;
//     int64_t num_kv_head = 8;
//     int64_t head_dim = 128;
//     int64_t cache_length = 512;
//     int64_t num_threads = 16;

//     // Bind the current thread to NUMA node 0
//     int numa_node = 0;
//     numa_run_on_node(numa_node);
//     numa_set_preferred(numa_node);

//     // Set thread affinity
//     cpu_set_t cpuset;
//     CPU_ZERO(&cpuset);
//     for (int i = 0; i < num_threads; i++) {
//         CPU_SET(i, &cpuset);
//     }
//     sched_setaffinity(0, sizeof(cpuset), &cpuset);

//     // Create query_states with original shape and dtype
//     auto query_options = torch::TensorOptions()
//         .dtype(torch::kFloat16)
//         .device(torch::kCPU);
//     torch::Tensor query_states = torch::randn(
//         {bsz, num_attention_heads, 1, head_dim},
//         query_options
//     ).contiguous();

//     // Allocate key and value states
//     std::vector<c10::BFloat16*> key_states;
//     std::vector<c10::BFloat16*> value_states;
//     key_states.reserve(bsz);
//     value_states.reserve(bsz);

//     // Create key and value tensors with original shape
//     auto kv_options = torch::TensorOptions()
//         .dtype(torch::kBFloat16)
//         .device(torch::kCPU)
//         .requires_grad(false);

//     // Allocate on NUMA node 0
//     auto key_tensor = torch::randn(
//         {bsz, cache_length, num_kv_head, head_dim},
//         kv_options
//     ).contiguous();
//     auto value_tensor = torch::randn(
//         {bsz, cache_length, num_kv_head, head_dim},
//         kv_options
//     ).contiguous();

//     // Move tensors to NUMA node 0
//     int64_t key_size = key_tensor.numel() * sizeof(c10::BFloat16);
//     int64_t value_size = value_tensor.numel() * sizeof(c10::BFloat16);

//     void* key_numa_mem = numa_alloc_onnode(key_size, numa_node);
//     void* value_numa_mem = numa_alloc_onnode(value_size, numa_node);

//     // Copy data to NUMA memory
//     std::memcpy(key_numa_mem, key_tensor.data_ptr(), key_size);
//     std::memcpy(value_numa_mem, value_tensor.data_ptr(), value_size);

//     // Update tensors to use NUMA memory
//     key_tensor = torch::from_blob(
//         key_numa_mem,
//         key_tensor.sizes(),
//         kv_options
//     ).contiguous();

//     value_tensor = torch::from_blob(
//         value_numa_mem,
//         value_tensor.sizes(),
//         kv_options
//     ).contiguous();

//     // Setup key and value state pointers (unchanged)
//     for (int i = 0; i < bsz; ++i) {
//         key_states.push_back(static_cast<c10::BFloat16*>(key_tensor[i].data_ptr()));
//         value_states.push_back(static_cast<c10::BFloat16*>(value_tensor[i].data_ptr()));
//     }

//     // Create attention mask with original shape
//     torch::Tensor attention_mask = torch::randn(
//         {bsz, cache_length},
//         torch::TensorOptions().dtype(torch::kFloat32)
//     ).contiguous();

//     // Allocate attention weights on NUMA node 0
//     size_t attn_weights_size = bsz * num_attention_heads * 1 * cache_length *
//     sizeof(c10::BFloat16); c10::BFloat16* attn_weights =
//     static_cast<c10::BFloat16*>(
//         numa_alloc_onnode(attn_weights_size, numa_node)
//     );

//     // Verify tensor shapes before proceeding
//     std::cout << "Shape verification:" << std::endl;
//     std::cout << "query_states: " << query_states.sizes() << std::endl;
//     std::cout << "key_tensor: " << key_tensor.sizes() << std::endl;
//     std::cout << "value_tensor: " << value_tensor.sizes() << std::endl;
//     std::cout << "attention_mask: " << attention_mask.sizes() << std::endl;

//     std::cout << "Input tensors created on NUMA node " << numa_node <<
//     std::endl;

//     // Memory fence and yield
//     asm volatile("" ::: "memory");
//     sched_yield();

//     // Performance markers
//     asm volatile("###START###" ::: "memory");
//     auto start_time = std::chrono::high_resolution_clock::now();

//     torch::Tensor attn_output = grouped_query_attention_cpu_avx2(
//         query_states,
//         key_states,
//         value_states,
//         cache_length,
//         attention_mask,
//         attn_weights,
//         4,
//         num_threads,
//         num_attention_heads,
//         num_kv_head,
//         head_dim
//     );

//     auto end_time = std::chrono::high_resolution_clock::now();
//     asm volatile("###END###" ::: "memory");

//     std::cout << "Time taken: "
//               <<
//               std::chrono::duration_cast<std::chrono::milliseconds>(end_time
//               - start_time).count()
//               << " ms" << std::endl;

//     // Cleanup NUMA allocations
//     numa_free(key_numa_mem, key_size);
//     numa_free(value_numa_mem, value_size);
//     numa_free(attn_weights, attn_weights_size);

//     return 0;
// }
