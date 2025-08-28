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
#include <numa.h>
#include <numaif.h>
#include <cstdlib>       // posix_memalign      
#include <cuda_runtime.h>
#include <unistd.h>      // getpagesize
#include <linux/mman.h>  // For MAP_HUGE_2MB

// Fallback calculation if MAP_HUGE_2MB is not directly available
#ifndef MAP_HUGE_2MB
#define MAP_HUGE_2MB (21 << MAP_HUGE_SHIFT)
#endif


constexpr float FP8_MAX = 448.0f;
// std::tuple<at::Tensor, at::Tensor> per_token_quant(torch::Tensor x) {
//     /* 
//      * Quantize a [bsz, seq, 576] BF16 tensor to FP8 per-token.
//      * Args:
//      *   x: Input tensor of shape [bsz, seq, 576] with dtype bfloat16
//      * Returns:
//      *   q: Quantized tensor [bsz, seq, 576] with dtype float8_e4m3fn
//      *   s: Scale factors [bsz, seq] with dtype float32
//      */
//     // TORCH_CHECK(x.scalar_type() == at::ScalarType::BFloat16, 
//     //             "Input tensor must be of dtype BFloat16");
//     // TORCH_CHECK(x.size(-1) == 576, 
//     //             "Last dimension of input tensor must be 576");
//     // TORCH_CHECK(x.is_contiguous(), 
//     //             "Input tensor must be contiguous");
//     // TORCH_CHECK(x.dim() == 3, 
//     //             "Input tensor must have 3 dimensions");
    
//     const auto device = x.device();
//     const auto bsz = x.size(0);
//     const auto seq_len = x.size(1);
//     const auto dim = x.size(2);
//     const auto M = bsz * seq_len;
    
//     // Cast to float32 and reshape for reduction
//     auto x_flat = x.view({M, dim}).to(at::ScalarType::Float);
    
//     // Compute max absolute value per token
//     auto amax = at::amax(at::abs(x_flat), /*dim=*/1);
//     amax = at::clamp(amax, /*min=*/1e-6f);
    
//     // Compute scales
//     auto scale = amax / FP8_MAX;
    
//     // Scale and cast to fp8
//     auto y = x_flat / scale.unsqueeze(1);
//     auto q = y.to(at::ScalarType::Float8_e4m3fn);
    
//     // Reshape output tensors
//     q = q.view({bsz, seq_len, dim});
//     scale = scale.view({bsz, seq_len});
    
//     return std::make_tuple(q, scale);
// }

std::tuple<at::Tensor, at::Tensor> quant_per_token(const at::Tensor& x) {
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
        amax = at::clamp(amax, /*min=*/1e-4f);

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



// KV_Storage::KV_Storage(EngineConfig& engine_config, ModelConfig& model_config,
//                        DtoH_Engine& d2h_engine)
//     : engine_config_(engine_config),
//       model_config_(model_config),
//       d2h_engine_(d2h_engine),
//       per_element_mutex_(model_config.num_hidden_layers *
//                          engine_config.kv_storage_config.num_host_slots) {
//     try {
//         this->logger_ = init_logger(
//             this->engine_config_.basic_config.log_level,
//             "KV_Storage" +
//                 std::to_string(this->engine_config_.basic_config.device));
//         CUDA_CHECK(cudaSetDevice(this->engine_config_.basic_config.device));
//         this->logger_->info("Starting KV_Storage Initialization.");
//         /* Reserve Pinned Memory for K and V. */
//         const auto& storage_size =
//             this->engine_config_.kv_storage_config.storage_byte_size;
//         auto per_layer_storage_size =
//             storage_size / this->model_config_.num_hidden_layers;

//         auto bar = tq::trange(this->model_config_.num_hidden_layers);
//         bar.set_prefix("Allocating Pinned Memory for KV cache");
//         if (this->model_config_.model_type.find("deepseek") ==
//             std::string::npos) {
//             for (auto layer_idx : bar) {
//                 void* k_ptr = nullptr;
//                 void* v_ptr = nullptr;
//                 CUDA_CHECK(cudaHostAlloc(&k_ptr, per_layer_storage_size,
//                                          cudaHostAllocDefault));
//                 CUDA_CHECK(cudaHostAlloc(&v_ptr, per_layer_storage_size,
//                                          cudaHostAllocDefault));
//                 this->k_pinned_memory.push_back(k_ptr);
//                 this->v_pinned_memory.push_back(v_ptr);
//             }
//         } else {
//             for (auto layer_idx : bar) {
//                 void* k_ptr = nullptr;
//                 CUDA_CHECK(cudaHostAlloc(&k_ptr, per_layer_storage_size,
//                                          cudaHostAllocDefault));
//                 // memset(k_ptr, 999, per_layer_storage_size); 
//                 this->k_pinned_memory.push_back(k_ptr);
//             }
//         }
//         this->logger_->info("KV Storage Pinned Memory Allocated.");
//     } catch (...) {
//         this->logger_->debug(
//             "KV_Storage: Failed to update K and V to the storage.");
//         throw std::runtime_error(
//             "KV_Storage: Failed to update K and V to the storage.");
//     }
// };

// KV_Storage::KV_Storage(EngineConfig& engine_config, ModelConfig& model_config,
//                        DtoH_Engine& d2h_engine)
//     : engine_config_(engine_config),
//       model_config_(model_config),
//       d2h_engine_(d2h_engine),
//       per_element_mutex_(model_config.num_hidden_layers *
//                          engine_config.kv_storage_config.num_host_slots) {
//     try {
//         this->logger_ = init_logger(
//             this->engine_config_.basic_config.log_level,
//             "KV_Storage" + std::to_string(this->engine_config_.basic_config.device));
        
//         int device_id = this->engine_config_.basic_config.device;
//         CUDA_CHECK(cudaSetDevice(device_id));

//         int numa_node = -1;

//         if (numa_available() >= 0) {
//             numa_node = (device_id < 4) ? 0 : 1;
//             numa_set_preferred(numa_node);
//             numa_run_on_node(numa_node);  // Apply thread binding after check
//         } else {
//             this->logger_->warn("NUMA is not available. Falling back to default memory allocation.");
//         }

//         this->logger_->info("Starting KV_Storage Initialization on device {} using NUMA node {}.",
//                             device_id, numa_node);

//         const auto& storage_size = this->engine_config_.kv_storage_config.storage_byte_size;
//         auto per_layer_storage_size = storage_size / this->model_config_.num_hidden_layers;

//         auto bar = tq::trange(this->model_config_.num_hidden_layers);
//         bar.set_prefix("Allocating Pinned Memory for KV cache");

//         bool is_deepseek = (this->model_config_.model_type.find("deepseek") != std::string::npos);

//         for (auto layer_idx : bar) {
//             auto allocate = [&](void** ptr) {
//                 if (numa_node >= 0) {
//                     *ptr = numa_alloc_onnode(per_layer_storage_size, numa_node);
//                     if (*ptr != nullptr) {
//                         // Touch the pages to commit allocation
//                         volatile char* p = static_cast<volatile char*>(*ptr);
//                         for (size_t i = 0; i < per_layer_storage_size; i += 4096)
//                             p[i] = 0;

//                         if (cudaHostRegister(*ptr, per_layer_storage_size, cudaHostRegisterDefault) != cudaSuccess) {
//                             this->logger_->warn("cudaHostRegister failed on NUMA memory, falling back to cudaHostAlloc.");
//                             numa_free(*ptr, per_layer_storage_size);
//                             *ptr = nullptr;
//                         }
//                     }
//                 }

//                 if (*ptr == nullptr) {
//                     CUDA_CHECK(cudaHostAlloc(ptr, per_layer_storage_size, cudaHostAllocDefault));
//                 }
//             };

//             void* k_ptr = nullptr;
//             allocate(&k_ptr);
//             this->k_pinned_memory.push_back(k_ptr);

//             if (!is_deepseek) {
//                 void* v_ptr = nullptr;
//                 allocate(&v_ptr);
//                 this->v_pinned_memory.push_back(v_ptr);
//             }
//         }

//         this->logger_->info("KV Storage Pinned Memory Allocated on NUMA node {}.", numa_node);
//     } catch (const std::exception& e) {
//         this->logger_->error("KV_Storage: Exception during initialization: {}", e.what());
//         throw;
//     } catch (...) {
//         this->logger_->error("KV_Storage: Unknown exception during initialization");
//         throw std::runtime_error("KV_Storage: Failed to initialize storage.");
//     }

//     // NUMA verification (optional)
//     if (!this->k_pinned_memory.empty()) {
//         int node = -1;
//         get_mempolicy(&node, nullptr, 0, this->k_pinned_memory[0], MPOL_F_NODE | MPOL_F_ADDR);
//         this->logger_->info("Verify device {} memory allocation on NUMA node {}.",
//                             this->engine_config_.basic_config.device, node);
//     }
// }

KV_Storage::KV_Storage(EngineConfig& engine_config,
                       ModelConfig& model_config,
                       DtoH_Engine& d2h_engine)
    : engine_config_(engine_config),
      model_config_(model_config),
      d2h_engine_(d2h_engine),
      per_element_mutex_(model_config.num_hidden_layers *
                         engine_config.kv_storage_config.num_host_slots) {
    try {
        this->logger_ = init_logger(
            this->engine_config_.basic_config.log_level,
            "KV_Storage" + std::to_string(this->engine_config_.basic_config.device));

        // int device_id = this->engine_config_.basic_config.device;
        // CUDA_CHECK(cudaSetDevice(device_id));

        // // Determine target NUMA node based on device ID
        // int numa_node = -1;
        // bool numa_avail = (numa_available() >= 0);
        // if (numa_avail) {
        //     numa_node = (device_id < 4) ? 0 : 1;
        // }

        // this->logger_->info("Starting KV_Storage Initialization on device {} targeting NUMA node {}.",
        //                     device_id, numa_node);

        // const size_t storage_size       = this->engine_config_.kv_storage_config.storage_byte_size;
        // const size_t per_layer_size     = storage_size / this->model_config_.num_hidden_layers;
        // const size_t alignment          = 2 * 1024 * 1024;  // 2 MiB

        // bool is_deepseek =
        //     (this->model_config_.model_type.find("deepseek") != std::string::npos);

        // auto allocate_numa_wc = [&](void** out_ptr) -> bool {
        //     if (!numa_avail || numa_node < 0) {
        //         // Fallback to regular cudaHostAlloc if NUMA not available
        //         cudaError_t err = cudaHostAlloc(out_ptr, per_layer_size, cudaHostAllocWriteCombined);
        //         if (err == cudaSuccess) {
        //             this->logger_->info("Allocated {} bytes using cudaHostAlloc (no NUMA)", per_layer_size);
        //             return true;
        //         }
        //         // memset
        //         // CUDA_CHECK(cudaMemset(*out_ptr, 9999999.0, per_layer_size));
        //         return false;
        //     }

        //     // Set NUMA policy to bind to target node before allocation
        //     struct bitmask *old_mask = numa_get_membind();
        //     struct bitmask *target_mask = numa_allocate_nodemask();
        //     numa_bitmask_setbit(target_mask, numa_node);
            
        //     // Set binding policy - use MPOL_BIND for strict placement
        //     numa_set_bind_policy(1);
        //     numa_set_membind(target_mask);

        //     // Allocate write-combined memory
        //     cudaError_t err = cudaHostAlloc(out_ptr, per_layer_size, cudaHostAllocWriteCombined);
            
        //     if (err == cudaSuccess) {
        //         // Verify the allocation is on correct NUMA node
        //         int actual_node = -1;
        //         if (get_mempolicy(&actual_node, nullptr, 0, *out_ptr, MPOL_F_NODE | MPOL_F_ADDR) == 0) {
        //             if (actual_node == numa_node) {
        //                 this->logger_->info("Successfully allocated {} bytes on NUMA node {}", 
        //                                   per_layer_size, actual_node);
        //             } else {
        //                 this->logger_->warn("Allocated on NUMA node {} instead of target node {}", 
        //                                   actual_node, numa_node);
                        
        //                 // Attempt to migrate the memory to correct NUMA node
        //                 unsigned long nodemask = 1UL << numa_node;
        //                 if (mbind(*out_ptr, per_layer_size, MPOL_BIND, &nodemask, 
        //                         sizeof(nodemask) * 8, MPOL_MF_MOVE | MPOL_MF_STRICT) == 0) {
        //                     this->logger_->info("Successfully migrated memory to NUMA node {}", numa_node);
        //                 } else {
        //                     this->logger_->warn("Failed to migrate memory to NUMA node {}", numa_node);
        //                 }
        //             }
        //         }

        //         // Touch pages to ensure physical allocation and proper NUMA placement
        //         // Use first-touch with proper alignment
        //         volatile char* ptr = static_cast<volatile char*>(*out_ptr);
        //         size_t page_size = getpagesize();
                
        //         #pragma omp parallel for if(per_layer_size > 1024*1024)
        //         for (size_t offset = 0; offset < per_layer_size; offset += page_size) {
        //             ptr[offset] = 0;
        //         }
                
        //         // Check alignment
        //         if (reinterpret_cast<uintptr_t>(*out_ptr) % alignment == 0) {
        //             this->logger_->debug("Memory is properly aligned to {} bytes", alignment);
        //         } else {
        //             this->logger_->warn("Memory alignment is not optimal (requested {} bytes)", alignment);
        //         }
        //     }

        //     // Restore original NUMA policy
        //     numa_set_membind(old_mask);
        //     numa_set_bind_policy(0);
        //     numa_free_nodemask(target_mask);
        //     numa_free_nodemask(old_mask);

        //     return (err == cudaSuccess);
        // };

        // // -------------------------------------------------
        // // Allocate per-layer KV buffers
        // // -------------------------------------------------
        // auto start_time = std::chrono::high_resolution_clock::now();
        // auto bar = tq::trange(this->model_config_.num_hidden_layers);
        // bar.set_prefix("Allocating NUMA-aware Pinned Memory for KV cache");
        // for (auto layer_idx : bar) {
        //     void* k_ptr = nullptr;
        //     if (!allocate_numa_wc(&k_ptr)) {
        //         throw std::runtime_error("Failed to allocate K cache memory for layer " + std::to_string(layer_idx));
        //     }
        //     this->k_pinned_memory.push_back(k_ptr);

        //     if (!is_deepseek) {
        //         void* v_ptr = nullptr;
        //         if (!allocate_numa_wc(&v_ptr)) {
        //             throw std::runtime_error("Failed to allocate V cache memory for layer " + std::to_string(layer_idx));
        //         }
        //         this->v_pinned_memory.push_back(v_ptr);
        //     }
        // }
        // auto end_time = std::chrono::high_resolution_clock::now();
        // auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
        // this->logger_->info("KV Storage Pinned Memory Allocated in {} seconds on NUMA node {}.",
        //                     duration, numa_node);

        // // Final NUMA verification for first allocation
        // if (!this->k_pinned_memory.empty() && numa_available) {
        //     int actual_node = -1;
        //     if (get_mempolicy(&actual_node, nullptr, 0,
        //                     this->k_pinned_memory[0],
        //                     MPOL_F_NODE | MPOL_F_ADDR) == 0) {
        //         this->logger_->info("Final verification: Device {} memory allocated on NUMA node {} (target was {}).",
        //                           device_id, actual_node, numa_node);
        //     }
            
        //     // Optional: Verify multiple pages for large allocations
        //     this->verify_numa_placement(this->k_pinned_memory[0], per_layer_size, numa_node);
        // }

    } catch (const std::exception& e) {
        this->logger_->error("KV_Storage: Exception during initialization: {}", e.what());
        throw;
    } catch (...) {
        this->logger_->error("KV_Storage: Unknown exception during initialization");
        throw std::runtime_error("KV_Storage: Failed to initialize storage.");
    }
}

// Helper method to verify NUMA placement across pages
void KV_Storage::verify_numa_placement(void* ptr, size_t size, int expected_node) {
    if (!ptr || size == 0) return;
    
    size_t page_size = getpagesize();
    size_t num_pages = (size + page_size - 1) / page_size;
    
    // Only check first few and last few pages for large allocations
    size_t pages_to_check = std::min(num_pages, static_cast<size_t>(10));
    
    std::vector<void*> pages(pages_to_check);
    std::vector<int> status(pages_to_check);
    
    // Check first few pages
    for (size_t i = 0; i < pages_to_check / 2 && i < num_pages; i++) {
        pages[i] = static_cast<char*>(ptr) + i * page_size;
    }
    
    // Check last few pages
    size_t start_idx = pages_to_check / 2;
    for (size_t i = 0; i < pages_to_check - start_idx && (num_pages - pages_to_check + start_idx + i) < num_pages; i++) {
        pages[start_idx + i] = static_cast<char*>(ptr) + (num_pages - pages_to_check + start_idx + i) * page_size;
    }
    
    if (move_pages(0, pages_to_check, pages.data(), nullptr, status.data(), 0) == 0) {
        int correct_placement = 0;
        for (size_t i = 0; i < pages_to_check; i++) {
            if (status[i] == expected_node) {
                correct_placement++;
            } else if (status[i] >= 0) {
                this->logger_->debug("Page {} on NUMA node {} (expected {})", i, status[i], expected_node);
            }
        }
        
        double placement_ratio = static_cast<double>(correct_placement) / pages_to_check;
        if (placement_ratio >= 0.9) {
            this->logger_->info("NUMA placement verification: {:.1f}% pages on correct node", placement_ratio * 100);
        } else {
            this->logger_->warn("NUMA placement verification: Only {:.1f}% pages on correct node {}", 
                              placement_ratio * 100, expected_node);
        }
    }
}





// KV_Storage::KV_Storage(EngineConfig& engine_config, ModelConfig& model_config,
//                        DtoH_Engine& d2h_engine)
//     : engine_config_(engine_config),
//       model_config_(model_config),
//       d2h_engine_(d2h_engine),
//       per_element_mutex_(model_config.num_hidden_layers *
//                          engine_config.kv_storage_config.num_host_slots) {
//     try {
//         this->logger_ = init_logger(
//             this->engine_config_.basic_config.log_level,
//             "KV_Storage" +
//                 std::to_string(this->engine_config_.basic_config.device));
        
//         int device_id = this->engine_config_.basic_config.device;
//         CUDA_CHECK(cudaSetDevice(device_id));
        
//         // Determine which NUMA node to use based on device ID
//         int numa_node = (device_id < 4) ? 0 : 1;
//         numa_run_on_node(numa_node);
//         numa_set_preferred(numa_node);
        
//         this->logger_->info("Starting KV_Storage Initialization on device {} using NUMA node {}.", 
//                            device_id, numa_node);
        
//         // Check if NUMA is available
//         if (numa_available() < 0) {
//             this->logger_->warn("NUMA is not available. Falling back to default memory allocation.");
//             numa_node = -1; // Disable NUMA binding
//         }
        
//         /* Reserve Pinned Memory for K and V. */
//         const auto& storage_size =
//             this->engine_config_.kv_storage_config.storage_byte_size;
//         auto per_layer_storage_size =
//             storage_size / this->model_config_.num_hidden_layers;

//         auto bar = tq::trange(this->model_config_.num_hidden_layers);
//         bar.set_prefix("Allocating Pinned Memory for KV cache");
        
//         // Check if it's a deepseek model which only uses K
//         bool is_deepseek = (this->model_config_.model_type.find("deepseek") != std::string::npos);
        
//         for (auto layer_idx : bar) {
//             // Allocate K memory
//             void* k_ptr = nullptr;
            
//             if (numa_node >= 0) {
//                 // Use NUMA-aware allocation
//                 k_ptr = numa_alloc_onnode(per_layer_storage_size, numa_node);
//                 if (k_ptr == nullptr) {
//                     this->logger_->warn("NUMA allocation failed for K cache. Falling back to cudaHostAlloc.");
//                     CUDA_CHECK(cudaHostAlloc(&k_ptr, per_layer_storage_size, cudaHostAllocDefault));
//                 } else {
//                     // Register the NUMA memory with CUDA for pinned access
//                     CUDA_CHECK(cudaHostRegister(k_ptr, per_layer_storage_size, 
//                                                cudaHostRegisterDefault));
//                 }
//             } else {
//                 // Use regular pinned memory
//                 CUDA_CHECK(cudaHostAlloc(&k_ptr, per_layer_storage_size, cudaHostAllocDefault));
//             }
            
//             this->k_pinned_memory.push_back(k_ptr);
            
//             // Allocate V memory if not deepseek model
//             if (!is_deepseek) {
//                 void* v_ptr = nullptr;
                
//                 if (numa_node >= 0) {
//                     // Use NUMA-aware allocation
//                     v_ptr = numa_alloc_onnode(per_layer_storage_size, numa_node);
//                     if (v_ptr == nullptr) {
//                         this->logger_->warn("NUMA allocation failed for V cache. Falling back to cudaHostAlloc.");
//                         CUDA_CHECK(cudaHostAlloc(&v_ptr, per_layer_storage_size, cudaHostAllocDefault));
//                     } else {
//                         // Register the NUMA memory with CUDA for pinned access
//                         CUDA_CHECK(cudaHostRegister(v_ptr, per_layer_storage_size, 
//                                                    cudaHostRegisterDefault));
//                     }
//                 } else {
//                     // Use regular pinned memory
//                     CUDA_CHECK(cudaHostAlloc(&v_ptr, per_layer_storage_size, cudaHostAllocDefault));
//                 }
                
//                 this->v_pinned_memory.push_back(v_ptr);
//             }
//         }
        
//         this->logger_->info("KV Storage Pinned Memory Allocated on NUMA node {}.", numa_node);
//     } catch (const std::exception& e) {
//         this->logger_->error("KV_Storage: Exception during initialization: {}", e.what());
//         throw;
//     } catch (...) {
//         this->logger_->error("KV_Storage: Unknown exception during initialization");
//         throw std::runtime_error("KV_Storage: Failed to update K and V to the storage.");
//     }
    
//     // Add memory allocation verification (optional)
//     if (!this->k_pinned_memory.empty()) {
//         int node;
//         get_mempolicy(&node, NULL, 0, this->k_pinned_memory[0], MPOL_F_NODE | MPOL_F_ADDR);
//         this->logger_->info("Verify device {} memory allocation on NUMA node {}.", 
//                            this->engine_config_.basic_config.device, node);
//     }
// };


void KV_Storage::Init() {
    try {
        int device_id = this->engine_config_.basic_config.device;
        CUDA_CHECK(cudaSetDevice(device_id));

        // Determine target NUMA node based on device ID
        int numa_node = -1;
        bool numa_avail = (numa_available() >= 0);
        if (numa_avail) {
            numa_node = (device_id < 4) ? 0 : 1;
        }

        this->logger_->info("Starting KV_Storage Initialization on device {} targeting NUMA node {}.",
                            device_id, numa_node);

        const size_t storage_size       = this->engine_config_.kv_storage_config.storage_byte_size;
        const size_t per_layer_size     = storage_size / this->model_config_.num_hidden_layers;
        const size_t alignment          = 2 * 1024 * 1024;  // 2 MiB

        bool is_deepseek =
            (this->model_config_.model_type.find("deepseek") != std::string::npos);

        auto allocate_numa_wc = [&](void** out_ptr) -> bool {
            if (!numa_avail || numa_node < 0) {
                // Fallback to regular cudaHostAlloc if NUMA not available
                cudaError_t err = cudaHostAlloc(out_ptr, per_layer_size, cudaHostAllocWriteCombined);
                if (err == cudaSuccess) {
                    this->logger_->info("Allocated {} bytes using cudaHostAlloc (no NUMA)", per_layer_size);
                    return true;
                }
                // memset
                CUDA_CHECK(cudaMemset(*out_ptr, 9999999.0, per_layer_size));
                return false;
            }

            // Set NUMA policy to bind to target node before allocation
            struct bitmask *old_mask = numa_get_membind();
            struct bitmask *target_mask = numa_allocate_nodemask();
            numa_bitmask_setbit(target_mask, numa_node);
            
            // Set binding policy - use MPOL_BIND for strict placement
            numa_set_bind_policy(1);
            numa_set_membind(target_mask);

            // Allocate write-combined memory
            cudaError_t err = cudaHostAlloc(out_ptr, per_layer_size, cudaHostAllocWriteCombined);
            CUDA_CHECK(cudaMemset(*out_ptr, 9999999.0, per_layer_size));
            
            if (err == cudaSuccess) {
                // Verify the allocation is on correct NUMA node
                int actual_node = -1;
                if (get_mempolicy(&actual_node, nullptr, 0, *out_ptr, MPOL_F_NODE | MPOL_F_ADDR) == 0) {
                    if (actual_node == numa_node) {
                        this->logger_->debug("Successfully allocated {} bytes on NUMA node {}", 
                                          per_layer_size, actual_node);
                    } else {
                        this->logger_->warn("Allocated on NUMA node {} instead of target node {}", 
                                          actual_node, numa_node);
                        
                        // Attempt to migrate the memory to correct NUMA node
                        unsigned long nodemask = 1UL << numa_node;
                        if (mbind(*out_ptr, per_layer_size, MPOL_BIND, &nodemask, 
                                sizeof(nodemask) * 8, MPOL_MF_MOVE | MPOL_MF_STRICT) == 0) {
                            this->logger_->info("Successfully migrated memory to NUMA node {}", numa_node);
                        } else {
                            this->logger_->warn("Failed to migrate memory to NUMA node {}", numa_node);
                        }
                    }
                }

                // Touch pages to ensure physical allocation and proper NUMA placement
                // Use first-touch with proper alignment
                volatile char* ptr = static_cast<volatile char*>(*out_ptr);
                size_t page_size = getpagesize();
                
                #pragma omp parallel for if(per_layer_size > 1024*1024)
                for (size_t offset = 0; offset < per_layer_size; offset += page_size) {
                    ptr[offset] = 0;
                }
                
                // Check alignment
                if (reinterpret_cast<uintptr_t>(*out_ptr) % alignment == 0) {
                    this->logger_->debug("Memory is properly aligned to {} bytes", alignment);
                } else {
                    this->logger_->warn("Memory alignment is not optimal (requested {} bytes)", alignment);
                }
            }

            // Restore original NUMA policy
            numa_set_membind(old_mask);
            numa_set_bind_policy(0);
            numa_free_nodemask(target_mask);
            numa_free_nodemask(old_mask);

            return (err == cudaSuccess);
        };

        // -------------------------------------------------
        // Allocate per-layer KV buffers
        // -------------------------------------------------
        auto start_time = std::chrono::high_resolution_clock::now();
        auto bar = tq::trange(this->model_config_.num_hidden_layers);
        bar.set_prefix("Allocating NUMA-aware Pinned Memory for KV cache");
        for (auto layer_idx : bar) {
            void* k_ptr = nullptr;
            if (!allocate_numa_wc(&k_ptr)) {
                throw std::runtime_error("Failed to allocate K cache memory for layer " + std::to_string(layer_idx));
            }
            this->k_pinned_memory.push_back(k_ptr);

            if (!is_deepseek) {
                void* v_ptr = nullptr;
                if (!allocate_numa_wc(&v_ptr)) {
                    throw std::runtime_error("Failed to allocate V cache memory for layer " + std::to_string(layer_idx));
                }
                this->v_pinned_memory.push_back(v_ptr);
            }
        }
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
        this->logger_->info("KV Storage Pinned Memory Allocated in {} seconds on NUMA node {}.",
                            duration, numa_node);

        // Final NUMA verification for first allocation
        if (!this->k_pinned_memory.empty() && numa_available) {
            int actual_node = -1;
            if (get_mempolicy(&actual_node, nullptr, 0,
                            this->k_pinned_memory[0],
                            MPOL_F_NODE | MPOL_F_ADDR) == 0) {
                this->logger_->info("Final verification: Device {} memory allocated on NUMA node {} (target was {}).",
                                  device_id, actual_node, numa_node);
            }
            
            // Optional: Verify multiple pages for large allocations
            this->verify_numa_placement(this->k_pinned_memory[0], per_layer_size, numa_node);
        }

        // Bookkeeping init
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
        CUDA_CHECK(
            cudaStreamCreateWithFlags(&this->stream_, cudaStreamNonBlocking));
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

// torch::Tensor KV_Storage::get_k_quantize_scale(
//     int64_t layer_idx, std::vector<int64_t> cur_batch, int64_t padding_length) 
// {
//     CUDA_CHECK(cudaSetDevice(this->engine_config_.basic_config.device));
//     torch::Device device(torch::kCUDA, this->engine_config_.basic_config.device);
//     auto opt = torch::TensorOptions()
//         .dtype(torch::kFloat32)
//         .device(device)  // Use the device object instead of a raw integer
//         .requires_grad(false);
//     auto k_quantize_scale = torch::ones({cur_batch.size(), padding_length, 5}, opt); // TODO:
//     // std::vector<torch::Tensor> k_quantize_scale_vec;
                                            
//     for (int64_t i = 0; i < static_cast<int64_t>(cur_batch.size()); i++) {
//         auto query_idx = cur_batch[i];
//         int64_t slot_idx = -1;
//         {
//             std::lock_guard<std::mutex> lock(this->mutex_);
//             slot_idx = this->query_idx_to_slot_idx_map[query_idx];
//         }
//         auto scale = this->k_storage[slot_idx][layer_idx].quantize_scale;
//         // log scale shape
//         // this->logger_->info("scale shape: {}",
//         //                      get_tensor_shape(scale));
//         int64_t scale_size = scale.size(1);
//         // copy scale to k_quantize_scale[i]'s first scale's size
//         if(scale_size > padding_length) {
//             this->logger_->info("scale_size: {}, padding_length: {}",
//                                  scale_size, padding_length);
//             throw std::runtime_error("scale_size > padding_length");
//         }
//         k_quantize_scale.index({i, torch::indexing::Slice(0, scale_size)}) = scale.index({0, torch::indexing::Slice(0, scale_size)}); 
//         // k_quantize_scale_vec.push_back(scale.index({0, torch::indexing::Slice(0, scale_size)}).unsqueeze(0));
//     }
//     // Check k_quantize_scale has nan values
//     // if (torch::any(torch::isnan(k_quantize_scale)).item<bool>()){
//     //     for (int64_t i = 0; i < k_quantize_scale.size(0); i++) {
//     //         if(torch::any(torch::isnan(k_quantize_scale[i])).item<bool>()) {
//     //             this->logger_->debug("k_quantize_scale[{}] has nan values, rank: {}, layer_idx: {}",
//     //                                  i, this->engine_config_.basic_config.device,
//     //                                  layer_idx);
//     //         }
//     //     }
//     //     // throw std::runtime_error("k_quantize_scale has nan values");
//     // }
//     // CUDA_CHECK(cudaStreamSynchronize(0));
//     // CUDA_CHECK(cudaDeviceSynchronize());
//     // torch::Tensor k_quantize_scale = torch::cat(k_quantize_scale_vec, 0);
//     // Concat to padding_length
//     // if(k_quantize_scale.size(1) < padding_length) {
//     //     auto padding = torch::ones({k_quantize_scale.size(0), padding_length - k_quantize_scale.size(1),k_quantize_scale.size(2)}, opt);
//     //     k_quantize_scale = torch::cat({k_quantize_scale, padding}, 1);
//     // }
//     // this->logger_->info("k_quantize_scale shape: {}",
//     //                              get_tensor_shape(k_quantize_scale));
//     return k_quantize_scale;
// }

torch::Tensor KV_Storage::get_k_quantize_scale(
    int64_t layer_idx, 
    const std::vector<int64_t>& cur_batch, 
    int64_t padding_length) {
    
    CUDA_CHECK(cudaSetDevice(this->engine_config_.basic_config.device));
    torch::Device device(torch::kCUDA, this->engine_config_.basic_config.device);
    
    auto opt = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(device)
        .requires_grad(false);
    
    // Check if all scales have the same size for potential optimization
    std::vector<std::pair<torch::Tensor, int64_t>> tensor_info;
    tensor_info.reserve(cur_batch.size());
    
    {
        std::lock_guard<std::mutex> lock(this->mutex_);
        for (const auto& query_idx : cur_batch) {
            int64_t slot_idx = this->query_idx_to_slot_idx_map[query_idx];
            auto& scale = this->k_storage[slot_idx][layer_idx].quantize_scale;
            int64_t scale_size = scale.size(1);
            
            if (scale_size > padding_length) {
                this->logger_->info("scale_size: {}, padding_length: {}", 
                                   scale_size, padding_length);
                throw std::runtime_error("scale_size > padding_length");
            }
            
            tensor_info.emplace_back(scale, scale_size);
        }
    }
    
    // Check if we can use torch::stack for uniform sizes
    bool uniform_sizes = true;
    int64_t first_size = tensor_info.empty() ? 0 : tensor_info[0].second;
    for (const auto& info : tensor_info) {
        if (info.second != first_size) {
            uniform_sizes = false;
            break;
        }
    }
    
    if (uniform_sizes && first_size > 0 && first_size <= padding_length) {
        // All tensors have the same size - use efficient stack operation
        std::vector<torch::Tensor> uniform_tensors;
        uniform_tensors.reserve(tensor_info.size());
        
        for (const auto& info : tensor_info) {
            uniform_tensors.push_back(
                info.first.index({0, torch::indexing::Slice(0, first_size)})
            );
        }
        
        auto stacked = torch::stack(uniform_tensors, 0);
        
        // If we need padding, create result tensor and copy
        if (first_size < padding_length) {
            auto result = torch::ones({static_cast<int64_t>(cur_batch.size()), padding_length, 5}, opt);
            result.index({torch::indexing::Slice(), torch::indexing::Slice(0, first_size)}) = stacked;
            return result;
        } else {
            return stacked;
        }
    } else {
        // Fall back to individual copies (but still more efficient than original)
        auto k_quantize_scale = torch::ones({static_cast<int64_t>(cur_batch.size()), padding_length, 5}, opt);
        
        for (int64_t i = 0; i < static_cast<int64_t>(tensor_info.size()); i++) {
            const auto& [scale, scale_size] = tensor_info[i];
            if (scale_size > 0) {
                k_quantize_scale.index({i, torch::indexing::Slice(0, scale_size)}).copy_(
                    scale.index({0, torch::indexing::Slice(0, scale_size)})
                );
            }
        }
        
        return k_quantize_scale;
    }
}
            
            
// void KV_Storage::offload(
//     int64_t layer_idx,
//     std::vector<int64_t> query_global_idx, 
//     torch::Tensor k,
//     torch::Tensor v,
//     torch::Tensor attention_mask)
// {
//     // SAFE_CALL(
//     //     [&]() {
//     //         auto worker = std::thread(&KV_Storage::offload_helper_, this,
//     //                                   layer_idx, query_global_idx, k, v);
//     //         // worker.detach();
//     //         worker.join();
//     //     },
//     //     this->logger_);
//     // Check if k has nan values
//     CUDA_CHECK(cudaStreamSynchronize(0));
//     // CUDA_CHECK(cudaDeviceSynchronize());
//     // if (torch::any(torch::isnan(k)).item<bool>()){
//     //     for (int64_t i = 0; i < k.size(0); i++) {
//     //         if(torch::any(torch::isnan(k[i])).item<bool>()) {
//     //             this->logger_->debug("k[{}] has nan values, rank: {}, layer_idx: {}",
//     //                                  i, this->engine_config_.basic_config.device,
//     //                                  layer_idx);
//     //         }
//     //     }
//     //     // throw std::runtime_error("k has nan values");
//     // }
//     this->offload_helper_(layer_idx, query_global_idx, k, v, attention_mask);
// };


void KV_Storage::offload(
    int64_t layer_idx,
    std::vector<int64_t> query_global_idx, 
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor attention_mask)
{
    CUDA_CHECK(cudaStreamSynchronize(0));
    SAFE_CALL(
        [&]() {
            auto worker = std::thread(&KV_Storage::offload_helper_, this,
                                      layer_idx, query_global_idx, k, v, attention_mask);
            worker.detach();
            // worker.join();
        },
        this->logger_);

    // this->offload_helper_(layer_idx, query_global_idx, k, v, attention_mask);
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
            auto [k, k_quantize_scale] = quant_per_token(bf16_k);
            // log k_quantize_scale shape
            // this->logger_->info("k_quantize_scale shape: {}",
            //                      get_tensor_shape(k_quantize_scale));
            // Check k or k_quantize_scale has nan values
            CUDA_CHECK(cudaStreamSynchronize(0));
            // if (torch::any(torch::isnan(k)).item<bool>()){
            //     for (int64_t i = 0; i < k.size(0); i++) {
            //         if(torch::any(torch::isnan(k[i])).item<bool>()) {
            //             this->logger_->debug("k[{}] has nan values, rank: {}, layer_idx: {}",
            //                                  i, this->engine_config_.basic_config.device,
            //                                  layer_idx);
            //         }
            //     }
            //     // throw std::runtime_error("k has nan values");
            // }
            // if (torch::any(torch::isnan(k_quantize_scale)).item<bool>()){
            //     for (int64_t i = 0; i < k_quantize_scale.size(0); i++) {
            //         if(torch::any(torch::isnan(k_quantize_scale[i])).item<bool>()) {
            //             this->logger_->debug("k_quantize_scale[{}] has nan values, rank: {}, layer_idx: {}",
            //                                  i, this->engine_config_.basic_config.device,
            //                                  layer_idx);
            //         }
            //     }
            //     // throw std::runtime_error("k_quantize_scale has nan values");
            // }


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
                    // log quantize_scale shape
                    // this->logger_->info("quantize_scale shape: {}",
                    //                      get_tensor_shape(this->k_storage[slot_idx][layer_idx].quantize_scale));              
                        
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
            // CUDA_CHECK(cudaStreamSynchronize(0));
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
        auto worker = std::thread(&KV_Storage::update_helper_, this, layer_idx,
                                  query_global_indices, k, v, k_quantize_scale);
        worker.detach();
        // worker.join();
        // this->update_helper_(layer_idx, query_global_indices, k, v,
        //                       k_quantize_scale);
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

void KV_Storage::copy_kv_to_worker(std::vector<int64_t> query_global_idx, int64_t max_length) {
    /*
        Copy the prefilled KV to the worker. 
        Here we copy the k storage to k_gpu_storage
    */
    this->logger_->info("KV_Storage copy_kv_to_worker(): Copying KV to worker.");
    CUDA_CHECK(cudaSetDevice(this->engine_config_.basic_config.device));
    try{
        // Init the k_gpu_storage and v_gpu_storage
        int64_t num_slots = query_global_idx.size();
        // int64_t max_length = this->engine_config_.basic_config.padding_length + this->engine_config_.basic_config.max_decoding_length;
        // TODO:
        this->k_gpu_memory.resize(this->model_config_.num_hidden_layers);
        int64_t per_layer_size = num_slots * max_length * 576;
        for(int64_t layer_idx = 0; layer_idx < this->model_config_.num_hidden_layers; layer_idx++){
            void* k_ptr = nullptr;
            CUDA_CHECK(cudaMalloc(&k_ptr, per_layer_size));
            this->k_gpu_memory[layer_idx] = k_ptr;
        };
        this->logger_->info("KV_Storage copy_kv_to_worker(): num_slots: {}, per_layer_size: {}", num_slots, per_layer_size);

        this->k_gpu_storage.resize(num_slots);
        for(int64_t slot_idx = 0; slot_idx < num_slots; slot_idx++){
            this->k_gpu_storage[slot_idx].resize(this->model_config_.num_hidden_layers);
            for(int64_t layer_idx = 0; layer_idx < this->model_config_.num_hidden_layers; layer_idx++){
                void* slot_ptr = nullptr;
                slot_ptr = static_cast<void*>(
                    static_cast<char*>(this->k_gpu_memory[layer_idx]) +
                    slot_idx * max_length * 576);
                this->k_gpu_storage[slot_idx][layer_idx].start_ptr = slot_ptr;
                this->k_gpu_storage[slot_idx][layer_idx].used_byte_size = 0;
                this->k_gpu_storage[slot_idx][layer_idx].num_tokens = 0;

            }
        };

        // Update gpu_query_idx_to_slot_idx_map
        this->gpu_query_idx_to_slot_idx_map.clear();
        for(int64_t i = 0; i < query_global_idx.size(); i++){
            int64_t query_idx = query_global_idx[i];
            this->gpu_query_idx_to_slot_idx_map[query_idx] = i;
        };
            
        this->logger_->info("KV_Storage copy_kv_to_worker(): GPU storage initialized.");

                         
        for(int64_t idx = 0; idx < query_global_idx.size(); idx++){
            int64_t cpu_slot_idx = -1;
            {
                std::lock_guard<std::mutex> lock(this->mutex_);
                cpu_slot_idx = this->query_idx_to_slot_idx_map[query_global_idx[idx]];
            }
            for(int64_t layer_idx = 0; layer_idx < this->model_config_.num_hidden_layers; layer_idx++){
                auto host_k_ptr = this->k_storage[cpu_slot_idx][layer_idx].start_ptr;
                // auto host_v_ptr = this->v_storage[slot_idx][layer_idx].start_ptr;
                auto k_size = this->k_storage[cpu_slot_idx][layer_idx].used_byte_size;
                // auto v_size = this->v_storage[slot_idx][layer_idx].used_byte_size;
                // Copy the k and v to the worker
                CUDA_CHECK(cudaMemcpyAsync(
                    this->k_gpu_storage[idx][layer_idx].start_ptr,
                    host_k_ptr, k_size, cudaMemcpyHostToDevice, this->stream_));
                // CUDA_CHECK(cudaMemcpyAsync(
                //     this->v_gpu_storage[slot_idx][layer_idx].start_ptr,
                //     host_v_ptr, v_size, cudaMemcpyHostToDevice, this->stream_));
                // Update the used byte size and num tokens in the gpu storage
                this->k_gpu_storage[idx][layer_idx].used_byte_size = k_size;
                this->k_gpu_storage[idx][layer_idx].num_tokens = this->k_storage[cpu_slot_idx][layer_idx].num_tokens;
                this->k_gpu_storage[idx][layer_idx].quantize_scale = this->k_storage[cpu_slot_idx][layer_idx].quantize_scale.clone();
                // this->v_gpu_storage[slot_idx][layer_idx].used_byte_size = v_size;
                // this->v_gpu_storage[slot_idx][layer_idx].num_tokens = this->v_storage[slot_idx][layer_idx].num_tokens;
            }
        }
        // Synchronize the stream
        CUDA_CHECK(cudaStreamSynchronize(this->stream_));
        this->logger_->info("KV_Storage copy_kv_to_worker(): KV copied to worker.");
    }
    // Catch PyTorch/CUDA specific exceptions
    catch (const c10::Error& e) {
        this->logger_->debug("KV_Storage get_v_ptrs(): CUDA/PyTorch error: {}",
                             e.what());
        throw std::runtime_error(
            "KV_Storage: Failed to copy KV to worker.");
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



// get_k(layer_idx, cur_batch, tensor_shape)
torch::Tensor KV_Storage::get_k(int64_t layer_idx, std::vector<int64_t> cur_batch, std::vector<int64_t> tensor_shape) {
    try {
        CUDA_CHECK(cudaSetDevice(this->engine_config_.basic_config.device));
        // Check if the layer_idx is valid
        if (layer_idx < 0 || layer_idx >= this->model_config_.num_hidden_layers) {
            throw std::runtime_error("Invalid layer index: " + std::to_string(layer_idx));
        }
        // Check if the batch is empty
        if (cur_batch.empty()) {
            throw std::runtime_error("Batch is empty.");
        }
        
        // Get the tensor from k_gpu_storage
        auto k_tensor = torch::empty(tensor_shape, torch::TensorOptions()
            .dtype(this->engine_config_.basic_config.kv_dtype_torch)
            .device(this->engine_config_.basic_config.device_torch)
            .requires_grad(false));
        
        int64_t seq_byte_size = tensor_shape[1] * tensor_shape[2] * k_tensor.element_size();
        // Fill the tensor with data from k_gpu_storage
        for (int64_t i = 0; i < static_cast<int64_t>(cur_batch.size()); i++) {
            auto query_idx = cur_batch[i];
            int64_t slot_idx = -1;
            {
                std::lock_guard<std::mutex> lock(this->mutex_);
                slot_idx = this->gpu_query_idx_to_slot_idx_map[query_idx];
            }
            // this->engine_config_.basic_config.kv_dtype_torch* k_ptr = nullptr;
            void* src_ptr = nullptr;
            {
                std::lock_guard<std::mutex> lock(
                    this->per_element_mutex_[slot_idx * this->model_config_.num_hidden_layers + layer_idx]);
                    src_ptr = this->k_gpu_storage[slot_idx][layer_idx].start_ptr;
            }
            // Copy the data from k_ptr to k_tensor
            auto dst_ptr = k_tensor.data_ptr() + i * seq_byte_size;
            CUDA_CHECK(cudaMemcpyAsync(
                dst_ptr, src_ptr, seq_byte_size, cudaMemcpyDeviceToDevice, this->stream_));
        }
        // Synchronize the stream
        CUDA_CHECK(cudaStreamSynchronize(this->stream_));
        return k_tensor;
    }     // Catch PyTorch/CUDA specific exceptions
    catch (const c10::Error& e) {
        this->logger_->debug("KV_Storage get_v_ptrs(): CUDA/PyTorch error: {}",
                             e.what());
        throw std::runtime_error(
            "KV_Storage: Failed to copy KV to worker.");
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

void KV_Storage::gpu_kv_update(
    int64_t layer_idx,
    std::vector<int64_t> query_global_indices,
    torch::Tensor k, torch::Tensor v, torch::Tensor k_quantize_scale) 
{
    std::thread worker(&KV_Storage::gpu_kv_update_func, this, layer_idx,
                        query_global_indices, k, v, k_quantize_scale);
    worker.detach();
}

void KV_Storage::gpu_kv_update_func(
        int64_t layer_idx,
        std::vector<int64_t> query_global_indices,
        torch::Tensor k, torch::Tensor v, torch::Tensor k_quantize_scale) 
{
    try{
        CUDA_CHECK(cudaSetDevice(this->engine_config_.basic_config.device));
        k = k.contiguous();
        int64_t k_token_byte_size =
            k.size(1) * k.size(2) * k.element_size();
        this->logger_->debug("k_token_byte_size: {}", k_token_byte_size);
        for (int64_t i = 0; i < query_global_indices.size(); i++) {
            auto query_idx = query_global_indices[i];
            auto src_ptr = k.data_ptr() + i * k_token_byte_size;
            int64_t slot_idx;
            {
                std::lock_guard<std::mutex> lock(this->mutex_);
                slot_idx = this->gpu_query_idx_to_slot_idx_map[query_idx];
            }
            auto dst_k_ptr =
                this->k_gpu_storage[slot_idx][layer_idx].start_ptr +
                this->k_gpu_storage[slot_idx][layer_idx].used_byte_size;
            {
                std::lock_guard<std::mutex> lock(
                    this->per_element_mutex_
                        [slot_idx * this->model_config_.num_hidden_layers +
                        layer_idx]);
                CUDA_CHECK(cudaMemcpyAsync(
                    dst_k_ptr, src_ptr, k_token_byte_size,
                    cudaMemcpyDeviceToDevice, this->stream_));
                this->k_gpu_storage[slot_idx][layer_idx].used_byte_size +=
                    k_token_byte_size;
                this->k_gpu_storage[slot_idx][layer_idx].num_tokens += 1;
                this->k_gpu_storage[slot_idx][layer_idx].quantize_scale = torch::cat(
                    {this->k_gpu_storage[slot_idx][layer_idx].quantize_scale,
                        k_quantize_scale.index({i}).unsqueeze(0)}, 1);
            }
        }
        // Synchronize the stream
        CUDA_CHECK(cudaStreamSynchronize(this->stream_));
    }
    // Catch CUDA runtime errors
    catch (const cudaError_t& err) {
        this->logger_->debug(
            "KV_Storage gpu_kv_update(): CUDA runtime error: {}",
            cudaGetErrorString(err));
        throw std::runtime_error(cudaGetErrorString(err));
    }
    // Catch standard C++ exceptions
    catch (const std::exception& e) {
        this->logger_->debug(
            "KV_Storage gpu_kv_update(): Failed to update K and "
            "V to the storage. Error: {}",
            e.what());
        throw std::runtime_error(
            "KV_Storage gpu_kv_update(): Failed to update K and V to the storage.");
    }
    // Catch any other unexpected errors
    catch (...) {
        this->logger_->debug(
            "KV_Storage gpu_kv_update(): Failed to update K and "
            "V to the storage.");
        throw std::runtime_error(
            "KV_Storage gpu_kv_update(): Failed to update K and V to the storage.");
    }
};

void KV_Storage::clear_kv_gpu_storage(){
    // Deallocate the k_gpu_storage and v_gpu_storage
    this->logger_->debug("KV_Storage clear_kv_gpu_storage(): Clearing KV GPU storage.");
    CUDA_CHECK(cudaSetDevice(this->engine_config_.basic_config.device));
    for(int64_t layer_idx = 0; layer_idx < this->k_gpu_memory.size(); layer_idx++){
        CUDA_CHECK(cudaFree(this->k_gpu_memory[layer_idx]));
    };
    this->logger_->info("KV_Storage clear_kv_gpu_storage(): KV GPU storage cleared.");
}


// std::vector<torch::Tensor> KV_Storage::get_past_key_states(std::vector<int64_t> query_global_indices, int64_t max_seq_len) {
//     // Currently only support deepseek models. TODO:
//     // From blob in cpu to get the torch tensor representation of the kv cache in kv_storage.
//     // The slot for the same layer should be concatenated together.
//     // Copy the tensor to GPU device.
//     // Return a vector of torch::Tensor, each tensor is the past key states for a layer.
//     // The result tensor shape is [batch_size, max_seq_len, 576] 

//     auto start_time = std::chrono::high_resolution_clock::now();
//     this->logger_->debug("KV_Storage get_past_key_states(): Getting past key states.");
//     std::vector<torch::Tensor> past_key_states;
//     for (int64_t layer_idx = 0; layer_idx < this->model_config_.num_hidden_layers; layer_idx++) {
//         // std::vector<torch::Tensor> k_tensors;
//         torch::Tensor k_tensor = torch::zeros(
//             {static_cast<int64_t>(query_global_indices.size()), max_seq_len, 576},
//             torch::TensorOptions()
//                 .dtype(this->engine_config_.basic_config.kv_dtype_torch)
//                 .device(torch::kCPU)
//                 .requires_grad(false));
//         // Fill the tensor with data from k_storage
//         for (int64_t i = 0; i < static_cast<int64_t>(query_global_indices.size()); i++) {
//             auto query_idx = query_global_indices[i];
//             int64_t slot_idx = -1;
//             {
//                 std::lock_guard<std::mutex> lock(this->mutex_);
//                 slot_idx = this->query_idx_to_slot_idx_map[query_idx];
//             }
//             c10::Float8_e4m3fn* k_ptr = nullptr;
//             {
//                 std::lock_guard<std::mutex> lock(
//                     this->per_element_mutex_[slot_idx * this->model_config_.num_hidden_layers + layer_idx]);
//                 k_ptr = static_cast<c10::Float8_e4m3fn*>(
//                     this->k_storage[slot_idx][layer_idx].start_ptr);
//             }
//             // Copy the data from k_ptr to k_tensor
//             auto dst_ptr = k_tensor.data_ptr() + i * max_seq_len * 576 * sizeof(this->engine_config_.basic_config.kv_dtype_torch);
//             CUDA_CHECK(cudaMemcpyAsync(
//                 dst_ptr, k_ptr, max_seq_len * 576 * sizeof(this->engine_config_.basic_config.kv_dtype_torch),
//                 cudaMemcpyHostToHost, this->stream_));
//         }
//         // Synchronize the stream
//         CUDA_CHECK(cudaStreamSynchronize(this->stream_));
//         past_key_states.push_back(k_tensor.to(this->engine_config_.basic_config.device_torch));
//     }
//     this->logger_->debug("KV_Storage get_past_key_states(): Past key states retrieved.");
//     auto end_time = std::chrono::high_resolution_clock::now();
//     auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time -
//         start_time);
//     this->logger_->info("KV_Storage get_past_key_states(): Time taken to get past key states: {} seconds", duration.count());
//     return past_key_states;
// }

// std::vector<torch::Tensor> KV_Storage::get_past_key_states(std::vector<int64_t> query_global_indices, int64_t max_seq_len) {
//     // Currently only support deepseek models. TODO:
//     // From blob in cpu to get the torch tensor representation of the kv cache in kv_storage.
//     // The slot for the same layer should be concatenated together.
//     // Copy the tensor to GPU device.
//     // Return a vector of torch::Tensor, each tensor is the past key states for a layer.
//     // The result tensor shape is [batch_size, max_seq_len, 576] 

//     auto start_time = std::chrono::high_resolution_clock::now();
//     this->logger_->debug("KV_Storage get_past_key_states(): Getting past key states.");
//     std::vector<torch::Tensor> past_key_states;
//     for (int64_t layer_idx = 0; layer_idx < this->model_config_.num_hidden_layers; layer_idx++) {
//         // std::vector<torch::Tensor> k_tensors;
//         torch::Tensor k_tensor = torch::zeros(
//             {static_cast<int64_t>(query_global_indices.size()), max_seq_len, 576},
//             torch::TensorOptions()
//                 .dtype(this->engine_config_.basic_config.kv_dtype_torch)
//                 .device(this->engine_config_.basic_config.device_torch)
//                 .requires_grad(false));
//         // Fill the tensor with data from k_storage
//         for (int64_t i = 0; i < static_cast<int64_t>(query_global_indices.size()); i++) {
//             auto query_idx = query_global_indices[i];
//             int64_t slot_idx = -1;
//             {
//                 std::lock_guard<std::mutex> lock(this->mutex_);
//                 slot_idx = this->query_idx_to_slot_idx_map[query_idx];
//             }
//             c10::Float8_e4m3fn* k_ptr = nullptr;
//             {
//                 std::lock_guard<std::mutex> lock(
//                     this->per_element_mutex_[slot_idx * this->model_config_.num_hidden_layers + layer_idx]);
//                 k_ptr = static_cast<c10::Float8_e4m3fn*>(
//                     this->k_storage[slot_idx][layer_idx].start_ptr);
//             }
//             // Copy the data from k_ptr to k_tensor
//             auto dst_ptr = k_tensor.data_ptr() + i * max_seq_len * 576 * sizeof(this->engine_config_.basic_config.kv_dtype_torch);
//             CUDA_CHECK(cudaMemcpyAsync(
//                 dst_ptr, k_ptr, max_seq_len * 576 * sizeof(this->engine_config_.basic_config.kv_dtype_torch),
//                 cudaMemcpyHostToDevice, this->stream_));
//         }
//         // Synchronize the stream
//         CUDA_CHECK(cudaStreamSynchronize(this->stream_));
//         past_key_states.push_back(k_tensor);
//     }
//     this->logger_->debug("KV_Storage get_past_key_states(): Past key states retrieved.");
//     auto end_time = std::chrono::high_resolution_clock::now();
//     auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time -
//         start_time);
//     this->logger_->debug("KV_Storage get_past_key_states(): Time taken to get past key states: {} seconds", duration.count());
//     return past_key_states;
// }

std::vector<torch::Tensor> KV_Storage::get_past_key_states(
    std::vector<int64_t> query_global_indices, 
    int64_t max_seq_len) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    this->logger_->debug("KV_Storage get_past_key_states(): Getting past key states.");
    
    std::vector<torch::Tensor> past_key_states;
    past_key_states.reserve(this->model_config_.num_hidden_layers); // Pre-allocate
    
    for (int64_t layer_idx = 0; layer_idx < this->model_config_.num_hidden_layers; layer_idx++) {
        // Create tensor WITH pinned memory for better async copy
        auto options = torch::TensorOptions()
            .dtype(this->engine_config_.basic_config.kv_dtype_torch)
            .device(this->engine_config_.basic_config.device_torch) 
            .requires_grad(false);
            
        torch::Tensor k_tensor = torch::empty(  // Use empty instead of zeros for performance
            {static_cast<int64_t>(query_global_indices.size()), max_seq_len, 576},
            options);
        
        // Ensure tensor is contiguous
        k_tensor = k_tensor.contiguous();
        
        // Your copy logic with proper pointer arithmetic
        for (int64_t i = 0; i < static_cast<int64_t>(query_global_indices.size()); i++) {
            auto query_idx = query_global_indices[i];
            int64_t slot_idx = -1;
            {
                std::lock_guard<std::mutex> lock(this->mutex_);
                slot_idx = this->query_idx_to_slot_idx_map[query_idx];
            }
            
            c10::Float8_e4m3fn* k_ptr = nullptr;
            {
                std::lock_guard<std::mutex> lock(
                    this->per_element_mutex_[slot_idx * this->model_config_.num_hidden_layers + layer_idx]);
                k_ptr = static_cast<c10::Float8_e4m3fn*>(
                    this->k_storage[slot_idx][layer_idx].start_ptr);
            }
            
            // Fix: Proper pointer arithmetic for float8
            auto element_size = k_tensor.element_size();
            auto dst_ptr = static_cast<char*>(k_tensor.data_ptr()) + 
                           i * max_seq_len * 576 * element_size;
            
            CUDA_CHECK(cudaMemcpyAsync(
                dst_ptr, 
                k_ptr, 
                max_seq_len * 576 * element_size,
                cudaMemcpyHostToDevice, 
                this->stream_));
        }
        
        // Move tensor into vector
        past_key_states.emplace_back(std::move(k_tensor));
    }
    
    // Synchronize after all copies
    CUDA_CHECK(cudaStreamSynchronize(this->stream_));
    
    this->logger_->debug("KV_Storage get_past_key_states(): Past key states retrieved.");
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    this->logger_->debug("KV_Storage get_past_key_states(): Time taken: {} seconds", duration.count());
    
    return past_key_states;
}


std::vector<torch::Tensor> KV_Storage::get_kv_scale(std::vector<int64_t> query_global_indices, int64_t seq_len) {
    // Return the quantization scale for k
    this->logger_->debug("KV_Storage get_kv_scale(): Getting kv scale.");
    auto start_time = std::chrono::high_resolution_clock::now();
    std::vector<torch::Tensor> kv_scale;
    for (int64_t layer_idx = 0; layer_idx < this->model_config_.num_hidden_layers; layer_idx++) {
        // Get the quantization scale for k
        std::vector<torch::Tensor> k_quantize_scale;
        for (int64_t i = 0; i < static_cast<int64_t>(query_global_indices.size()); i++) {
            auto query_idx = query_global_indices[i];
            int64_t slot_idx = -1;
            {
                std::lock_guard<std::mutex> lock(this->mutex_);
                slot_idx = this->query_idx_to_slot_idx_map[query_idx];
            }
            // Get the scale tensor from k_storage
            // The scale should be of shape [bsz, seq_len, num_block].
            // If the scale's seq_len is less than the seq_len, pad it with zeros.
            torch::Tensor scale_tensor = this->k_storage[slot_idx][layer_idx].quantize_scale;
            if (scale_tensor.size(1) < seq_len) {
                // Pad the scale tensor with zeros
                auto options = torch::TensorOptions()
                    .dtype(torch::kFloat32)
                    .device(this->engine_config_.basic_config.device_torch)
                    .requires_grad(false);
                torch::Tensor padded_scale_tensor = torch::zeros(
                    {scale_tensor.size(0), seq_len - scale_tensor.size(1), scale_tensor.size(2)},
                    options);
                scale_tensor = torch::cat({scale_tensor, padded_scale_tensor}, 1);
            }
            k_quantize_scale.push_back(scale_tensor);
        }
        // Concatenate the scale tensors for each layer
        torch::Tensor k_scale_tensor = torch::cat(k_quantize_scale, 0);
        kv_scale.push_back(k_scale_tensor);
    }
    this->logger_->debug("KV_Storage get_kv_scale(): KV scale retrieved.");
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time -
        start_time);
    this->logger_->debug("KV_Storage get_kv_scale(): Time taken to get kv scale: {} seconds", duration.count());
    return kv_scale;
}


// std::vector<torch::Tensor> KV_Storage::get_kv_scale(std::vector<int64_t> query_global_indices, int64_t seq_len) {
//     // Return the quantization scale for k
//     this->logger_->debug("KV_Storage get_kv_scale(): Getting kv scale.");
    
//     int64_t bsz = static_cast<int64_t>(query_global_indices.size());
//     std::vector<torch::Tensor> kv_scale;
//     kv_scale.reserve(this->model_config_.num_hidden_layers);
    
//     auto options = torch::TensorOptions()
//         .dtype(torch::kFloat32)
//         .device(this->engine_config_.basic_config.device_torch)
//         .requires_grad(false);
    
//     for (int64_t layer_idx = 0; layer_idx < this->model_config_.num_hidden_layers; layer_idx++) {
//         // Get the first tensor to determine the shape for num_blocks dimension
//         int64_t query_idx = query_global_indices[0];
//         int64_t slot_idx = -1;
//         {
//             std::lock_guard<std::mutex> lock(this->mutex_);
//             slot_idx = this->query_idx_to_slot_idx_map[query_idx];
//         }
//         torch::Tensor first_scale = this->k_storage[slot_idx][layer_idx].quantize_scale;
//         int64_t num_blocks = first_scale.size(2);
        
//         // Pre-allocate tensor with shape [bsz, seq_len, num_blocks]
//         torch::Tensor k_scale_tensor = torch::zeros({bsz, seq_len, num_blocks}, options);
        
//         // Copy data for each query
//         for (int64_t i = 0; i < bsz; i++) {
//             query_idx = query_global_indices[i];
//             {
//                 std::lock_guard<std::mutex> lock(this->mutex_);
//                 slot_idx = this->query_idx_to_slot_idx_map[query_idx];
//             }
            
//             torch::Tensor scale_tensor = this->k_storage[slot_idx][layer_idx].quantize_scale;
//             int64_t actual_seq_len = scale_tensor.size(1);
            
//             // Copy the existing data into the pre-allocated tensor
//             // The remaining positions stay as zeros (padding)
//             if (actual_seq_len > 0) {
//                 k_scale_tensor[i].slice(0, 0, std::min(actual_seq_len, seq_len)) = 
//                     scale_tensor[0].slice(0, 0, std::min(actual_seq_len, seq_len));
//             }
//         }
        
//         kv_scale.push_back(k_scale_tensor);
//     }
    
//     this->logger_->debug("KV_Storage get_kv_scale(): KV scale retrieved.");
//     return kv_scale;
// }
