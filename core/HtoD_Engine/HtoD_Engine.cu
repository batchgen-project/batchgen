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
#include <memory>
#include <string>
#include <torch/extension.h>
#include <torch/torch.h>
#include <unordered_map>
#include <vector>

#include "../KV_Storage/KV_Storage.h"
#include "../Weights_Storage/Weights_Storage.h"
#include "../data_structures.h"
#include "../threadsafe_queue.h"
#include "../utils.h"
#include "HtoD_Engine.h"

HtoD_Engine::HtoD_Engine(const EngineConfig& engine_config,
                         const ModelConfig& model_config,
                         Weights_Storage& weights_storage,
                         KV_Storage& kv_storage,
                         GPU_Weight_Buffer& gpu_weight_buffer,
                         GPU_KV_Buffer& gpu_kv_buffer)
    : engine_config_(engine_config),
      model_config_(model_config),
      weights_storage_(weights_storage),
      kv_storage_(kv_storage),
      gpu_weight_buffer_(gpu_weight_buffer),
      gpu_kv_buffer_(gpu_kv_buffer) {
    cudaSetDevice(this->engine_config_.basic_config.device);
    this->logger_ = init_logger(
        this->engine_config_.basic_config.log_level,
        "HtoD_Engine" +
            std::to_string(this->engine_config_.basic_config.device));
    CUDA_CHECK(
        cudaStreamCreateWithFlags(&this->HtoD_stream, cudaStreamNonBlocking));
};

// Enable peer access for all devices
void HtoD_Engine::cuda_enable_peer_access(int rank, int world_size){
    for (int i = 0; i < world_size; ++i) {
        if (i != rank) {
            int can_access = 0;
            cudaDeviceCanAccessPeer(&can_access, rank, i);
            if (can_access) {
                cudaSetDevice(rank);
                cudaDeviceEnablePeerAccess(i, 0);
            }
        }
    }
}

// void HtoD_Engine::setup_torch_dist_nccl_p2p(){
//     // TODO:
//     auto options = torch::distributed::ProcessGroupNCCL::Options();
//     options.set_rank(this->engine_config_.basic_config.device);
//     options.set_world_size(8);
//     options.set_init_method("tcp://localhost:12355");

//     torch::distributed::init_process_group(options);

//     at::cuda::set_device(this->engine_config_.basic_config.device);

//     // Enable peer access for all devices
//     int world_size = torch::distributed::get_world_size();
//     for(int i = 0; i < world_size; ++i) {
//         if (i != this->engine_config_.basic_config.device) {
//             int can_access = 0;
//             cudaDeviceCanAccessPeer(&can_access, this->engine_config_.basic_config.device, i);
//             if (can_access) {
//                 cudaSetDevice(this->engine_config_.basic_config.device);
//                 cudaDeviceEnablePeerAccess(i, 0);
//             }
//         }
//     }
//     this->logger_->info("Peer access enabled for all devices.");
// }


void HtoD_Engine::set_global_routed_experts_data_ptr(
    const py::dict& experts_IPC_handles,
    const py::dict& expert_location_map)
{
    // Set std::unordered_set<std::string> for local_expert_names
    for (const auto& item : expert_location_map) {
        std::string module_name = item.first.cast<std::string>();
        int64_t location = item.second.cast<int64_t>();
        this->expert_location_map_[module_name] = location;
    }

    // reinterpret_cast ptrs to void* 
    for(auto& item : experts_IPC_handles) {
        std::string module_name = item.first.cast<std::string>();
        py::dict tensor_ptrs_dict = item.second.cast<py::dict>();
        for(const auto& tensor_item : tensor_ptrs_dict) {
            auto start_time = std::chrono::high_resolution_clock::now();
            std::string tensor_name = tensor_item.first.cast<std::string>();
            std::tuple<py::object, py::object> IPC_context = tensor_item.second.cast<std::tuple<py::object, py::object>>();
            int rank_id = std::get<0>(IPC_context).cast<int>();
            py::bytes handle_bytes = std::get<1>(IPC_context).cast<py::bytes>();
            if(rank_id == this->engine_config_.basic_config.device) {
                this->global_device_experts_ptrs_[module_name][tensor_name] = nullptr;
                continue;
            }
            // Get raw bytes from Python bytes object
            std::string handle_str = handle_bytes;
            // Verify the size matches cudaIpcMemHandle_t
            if (handle_str.size() != sizeof(cudaIpcMemHandle_t)) {
                throw std::runtime_error("Invalid IPC handle size: " + 
                    std::to_string(handle_str.size()) + " != " + 
                    std::to_string(sizeof(cudaIpcMemHandle_t)));
            }            

            // Cast the bytes to cudaIpcMemHandle_t
            cudaIpcMemHandle_t handle;
            std::memcpy(&handle, handle_str.data(), sizeof(cudaIpcMemHandle_t));
            auto end_time = std::chrono::high_resolution_clock::now();
            this->logger_->info("IPC handle cast time: {} ms", 
                std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count());

            start_time = std::chrono::high_resolution_clock::now();
            // Print used cuda memory
            size_t free_mem = 0;
            size_t total_mem = 0;
            CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
            this->logger_->info("Free memory: {} MB, Total memory: {} MB", 
                free_mem / (1024 * 1024), total_mem / (1024 * 1024));
            // Open the handle to get a device pointer
            CUDA_CHECK(cudaSetDevice(rank_id));
            void* dev_ptr = nullptr;
            cudaError_t err = cudaIpcOpenMemHandle(&dev_ptr, handle, cudaIpcMemLazyEnablePeerAccess);
            if (err != cudaSuccess) {
                throw std::runtime_error("Failed to open IPC handle for expert '" + 
                    module_name + "': " + cudaGetErrorString(err));
            }
            // Close IPC
            // CUDA_CHECK(cudaIpcCloseMemHandle(dev_ptr));
            end_time = std::chrono::high_resolution_clock::now();
            this->logger_->info("IPC handle open time: {} ms", 
                std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count());

            this->global_device_experts_ptrs_[module_name][tensor_name] = dev_ptr;
        }
    }                    
}

// void HtoD_Engine::Init(
//     std::unordered_map<std::string, std::vector<std::string>>&
//         weight_copy_tasks) {
//     std::lock_guard<std::mutex> lock(this->mutex_);
//     this->weight_copy_tasks_ = weight_copy_tasks;
//     for (auto& item : weight_copy_tasks) {
//         auto module_type = item.first;
//         auto module_names = item.second;
//         this->weights_copy_task_queue_.emplace(module_type,
//                                                threadsafe_queue<std::string>());
//         for (auto& module_name : module_names) {
//             this->weights_copy_task_queue_[module_type].push(module_name);
//         }
//     }
    
//     // this->p2p_tensor_copy_module = py::module::import("p2p_tensor_copy");
// };

void HtoD_Engine::Init() {    
    // this->p2p_tensor_copy_module = py::module::import("p2p_tensor_copy");
};

void HtoD_Engine::Start(){
    if (!this->HtoD_worker_.joinable()) {
        this->terminate_flag_ = false;
        this->HtoD_worker_ = std::thread(&HtoD_Engine::HtoD_Worker, this);
    }
}


void HtoD_Engine::Terminate() {
    this->terminate_flag_ = true;
    if (this->HtoD_worker_.joinable()) {
        this->HtoD_worker_.join();
    }
};

HtoD_Engine::~HtoD_Engine() { this->Terminate(); };

torch::Tensor HtoD_Engine::tensor_on_demand_copy(torch::Tensor src_tensor) {
    auto dst_tensor = torch::empty_like(src_tensor, torch::kCUDA).contiguous();
    // Create packaged task
    std::packaged_task<void()> task(
        [this, dst_ptr = dst_tensor.data_ptr(), src_ptr = src_tensor.data_ptr(),
         size = src_tensor.numel() * src_tensor.element_size()]() {
            this->blocking_copy_(dst_ptr, src_ptr, size);
        });

    // Get future before moving task
    std::future<void> completion_future = task.get_future();

    // Push task to queue
    this->on_demand_task_queue_.push(std::move(task));

    // Wait for completion
    completion_future.wait();

    return dst_tensor;
};

void HtoD_Engine::submit_to_KV_queue(std::vector<int64_t>& micro_batch,
                                     int64_t micro_batch_idx, int64_t layer_idx,
                                     int64_t byte_size) {
    this->kv_copy_task_queue_.push(
        std::make_tuple(micro_batch, micro_batch_idx, layer_idx, byte_size));
};

void HtoD_Engine::blocking_copy_(void* dst, void* src, int64_t size) {
    CUDA_CHECK(cudaSetDevice(this->engine_config_.basic_config.device));
    CUDA_CHECK(cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice,
                               this->HtoD_stream));
    // CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaStreamSynchronize(this->HtoD_stream));
};

void HtoD_Engine::clear_kv_copy_queue() {
    CUDA_CHECK(cudaSetDevice(this->engine_config_.basic_config.device));
    try {
        while (!this->kv_copy_task_queue_.empty()) {
            std::tuple<std::vector<int64_t>, int64_t, int64_t, int64_t> task;
            this->kv_copy_task_queue_.wait_and_pop(task);
        }
    } catch (...) {
        this->logger_->error("Failed to clear kv_copy_queue.");
    }
    CUDA_CHECK(cudaStreamSynchronize(this->HtoD_stream));
};

void HtoD_Engine::clear_weight_copy_queue() {
    CUDA_CHECK(cudaSetDevice(this->engine_config_.basic_config.device));
    std::lock_guard<std::mutex> lock(this->mutex_);
    for (auto& [module_type, queue] : this->weights_copy_task_queue_) {
        while (!queue.empty()) {
            queue.clear();
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(this->HtoD_stream));
}

void HtoD_Engine::reset_weight_copy_queue() {
    std::lock_guard<std::mutex> lock(this->mutex_);

    // re-init the weights_copy_task_queue_ according to the weight_copy_tasks
    auto new_weights_copy_task_queue =
        std::unordered_map<std::string, threadsafe_queue<std::string>>();
    for (auto& item : this->weight_copy_tasks_) {
        auto module_type = item.first;
        auto module_names = item.second;
        new_weights_copy_task_queue.emplace(module_type,
                                            threadsafe_queue<std::string>());
        for (auto& module_name : module_names) {
            new_weights_copy_task_queue[module_type].push(module_name);
        }
    }
    this->weights_copy_task_queue_ = new_weights_copy_task_queue;
};

void HtoD_Engine::HtoD_Worker() {
    CUDA_CHECK(cudaSetDevice(this->engine_config_.basic_config.device));
    while (!terminate_flag_) {
        std::packaged_task<void()> task;
        while (on_demand_task_queue_.try_pop(task)) {
            task();
        }
        if (!this->kv_copy_task_queue_.empty()) {
            auto optional_buffer = this->gpu_kv_buffer_.acquireEmptyBuffer();
            if (optional_buffer.has_value()) {
                if (this->model_config_.model_type.find("deepseek") ==
                    std::string::npos) {
                    // this->logger_->debug("DeepSeek model detected. Copying
                    // both K to GPU.");
                    auto [dst_k_ptr, dst_v_ptr, buffer_idx] =
                        optional_buffer.value();
                    std::tuple<std::vector<int64_t>, int64_t, int64_t, int64_t>
                        task;
                    this->kv_copy_task_queue_.wait_and_pop(task);
                    auto& [cur_batch, micro_batch_idx, layer_idx, byte_size] =
                        task;
                    this->logger_->debug(
                        "copying micro_batch_idx: {}, layer_idx: {}, "
                        "byte_size: {}",
                        micro_batch_idx, layer_idx, byte_size);
                    auto host_k_ptrs =
                        this->kv_storage_.get_k_ptrs(layer_idx, cur_batch);
                    auto host_v_ptrs =
                        this->kv_storage_.get_v_ptrs(layer_idx, cur_batch);
                    int64_t k_offset = 0;
                    int64_t v_offset = 0;
                    int64_t k_byte_size = byte_size;
                    int64_t v_byte_size = byte_size;
                    for (int64_t i = 0; i < cur_batch.size(); i++) {
                        CUDA_CHECK(cudaMemcpyAsync(
                            dst_k_ptr + k_offset, host_k_ptrs[i], k_byte_size,
                            cudaMemcpyHostToDevice, this->HtoD_stream));
                        CUDA_CHECK(cudaMemcpyAsync(
                            dst_v_ptr + v_offset, host_v_ptrs[i], v_byte_size,
                            cudaMemcpyHostToDevice, this->HtoD_stream));
                        k_offset += k_byte_size;
                        v_offset += v_byte_size;
                    }
                    CUDA_CHECK(cudaStreamSynchronize(this->HtoD_stream));
                    this->gpu_kv_buffer_.kv_copy_complete(
                        layer_idx, micro_batch_idx, buffer_idx);
                    this->logger_->debug("Copied KV to buffer: {}", buffer_idx);
                } else {
                    CUDA_CHECK(cudaSetDevice(this->engine_config_.basic_config.device));
                    auto [dst_k_ptr, dst_v_ptr, buffer_idx] =
                        optional_buffer.value();
                    std::tuple<std::vector<int64_t>, int64_t, int64_t, int64_t>
                        task;
                    this->kv_copy_task_queue_.wait_and_pop(task);
                    auto& [cur_batch, micro_batch_idx, layer_idx, byte_size] =
                        task;
                    auto host_k_ptrs =
                        this->kv_storage_.get_k_ptrs_fp8(layer_idx, cur_batch);
                    // auto host_v_ptrs =
                    // this->kv_storage_.get_v_ptrs(layer_idx, cur_batch);
                    int64_t k_offset = 0;
                    // int64_t v_offset = 0;
                    int64_t k_byte_size = byte_size; // TODO:
                    // int64_t v_byte_size = byte_size;

                    this->logger_->debug(
                        "copying micro_batch_idx: {}, layer_idx: {}, "
                        "byte_size: {}",
                        micro_batch_idx, layer_idx, k_byte_size);
                    // for (int64_t i = 0; i < cur_batch.size(); i++) {
                    //     CUDA_CHECK(cudaMemcpyAsync(
                    //         dst_k_ptr + k_offset, host_k_ptrs[i], k_byte_size,
                    //         cudaMemcpyHostToDevice, this->HtoD_stream));
                    //     k_offset += k_byte_size;
                    // }
                    for (int64_t i = 0; i < cur_batch.size(); i++) {
                        // Validation checks
                        if (dst_k_ptr == nullptr) {
                            this->logger_->error("dst_k_ptr is NULL!");
                            throw std::runtime_error("Invalid dst_k_ptr");
                        }
                        
                        if (host_k_ptrs[i] == nullptr) {
                            this->logger_->error("host_k_ptrs[{}] is NULL!", i);
                            throw std::runtime_error("Invalid host_k_ptrs at index " + std::to_string(i));
                        }
                        
                        if (k_byte_size <= 0) {
                            this->logger_->error("Invalid k_byte_size: {}", k_byte_size);
                            throw std::runtime_error("Invalid k_byte_size");
                        }
                        
                        // Check if destination pointer is valid CUDA memory
                        cudaPointerAttributes dst_attrs;
                        cudaError_t dst_err = cudaPointerGetAttributes(&dst_attrs, dst_k_ptr + k_offset);
                        if (dst_err != cudaSuccess) {
                            this->logger_->error("dst_k_ptr + {} offset is not valid CUDA memory: {}", 
                                            k_offset, cudaGetErrorString(dst_err));
                            cudaGetLastError(); // Clear the error
                        }
                        
                        // Check if source pointer is valid host memory
                        cudaPointerAttributes src_attrs;
                        cudaError_t src_err = cudaPointerGetAttributes(&src_attrs, host_k_ptrs[i]);
                        if (src_err != cudaSuccess) {
                            this->logger_->error("host_k_ptrs[{}] is not valid memory: {}", 
                                            i, cudaGetErrorString(src_err));
                            cudaGetLastError(); // Clear the error
                        }
                        

                        this->logger_->debug("Copying batch[{}]: dst_offset={}, size={}, "
                                            "dst_ptr={}, src_ptr={}", 
                                            i, k_offset, k_byte_size, 
                                            (void*)(dst_k_ptr + k_offset), (void*)host_k_ptrs[i]);
                        
                        CUDA_CHECK(cudaMemcpyAsync(
                            dst_k_ptr + k_offset, host_k_ptrs[i], k_byte_size,
                            cudaMemcpyHostToDevice, this->HtoD_stream));
                        k_offset += k_byte_size;
                    }
                    CUDA_CHECK(cudaStreamSynchronize(this->HtoD_stream));
                    // CUDA_CHECK(cudaDeviceSynchronize());
                    this->gpu_kv_buffer_.kv_copy_complete(
                        layer_idx, micro_batch_idx, buffer_idx);
                    this->logger_->debug("Copied KV to buffer: {}", buffer_idx);
                    // CUDA_CHECK(cudaStreamSynchronize(0));
                }
            }
        };

        for (auto& module_type :
             this->engine_config_.basic_config.module_types) {
            // Skip if queue is empty (no modules of this type need loading)
            // This prevents blocking when e.g. experts are persistent and not in queue
            if (this->weights_copy_task_queue_[module_type].empty()) {
                continue;
            }
            auto optional_buffer =
                this->gpu_weight_buffer_.acquireEmptyBuffer(module_type);
            if (optional_buffer.has_value()) {
                this->logger_->debug("Acquired buffer for module type: {}",
                                     module_type);
                CUDA_CHECK(
                    cudaSetDevice(this->engine_config_.basic_config.device));
                auto [buffer, buffer_idx] = optional_buffer.value();
                std::string module_name;
                while (this->weights_copy_task_queue_[module_type].try_pop(
                           module_name) == false) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
                auto dst = buffer.get();
                auto src = this->weights_storage_.get_module_weights_storage(
                    module_name);
                torch::Tensor tmp_src;
                void* src_ptr;
                int64_t src_byte_size;
                for (auto& [tensor_name, host_tensor_storage] : src) {
                    src_ptr = host_tensor_storage.data_ptr;
                    src_byte_size = host_tensor_storage.byte_size;
                    // find(), not operator[]: dst is an unordered_map and
                    // operator[] DEFAULT-INSERTS an undefined torch::Tensor on
                    // every miss, permanently growing a buffer map that is
                    // reused across ring slots.
                    auto slot = dst.find(tensor_name);
                    if (slot == dst.end() || !slot->second.defined() ||
                        !slot->second.has_storage()) {
                        // The host map carries a tensor module_shapes declares
                        // no slot for. Continuing drops it silently and the
                        // consumer reads whatever the slot last held.
                        this->logger_->error(
                            "Module {}: host tensor {} has no GPU slot -- "
                            "module_shapes declares no such key",
                            module_name, tensor_name);
                        throw std::runtime_error(
                            "HtoD: host tensor has no GPU slot: " +
                            tensor_name);
                    }
                    int64_t dst_byte_size = slot->second.nbytes();
                    if (src_byte_size != dst_byte_size) {
                        // blocking_copy_ writes src_byte_size bytes with no
                        // bound check: a short slot is overrun into its
                        // neighbour, a long one keeps a stale tail. Both are
                        // silent and both produce wrong weights.
                        this->logger_->error(
                            "Module {}: tensor {} size mismatch -- host {} B, "
                            "GPU slot {} B (module_shapes/dtype disagrees with "
                            "the checkpoint)",
                            module_name, tensor_name, src_byte_size,
                            dst_byte_size);
                        throw std::runtime_error(
                            "HtoD: host/GPU byte size mismatch for " +
                            tensor_name);
                    }
                    // The dedicated H2D worker is already asynchronous with
                    // respect to the Python compute thread.  Pace this stream
                    // one tensor at a time so a rank cannot queue multiple
                    // complete expert layers ahead of their consumers and
                    // monopolize shared host-memory/PCIe service.
                    this->blocking_copy_(slot->second.data_ptr(), src_ptr,
                                         src_byte_size);
                }
                this->logger_->debug(
                    "Enqueued module copy: {} to buffer: {}", module_name,
                    buffer_idx);
                this->gpu_weight_buffer_.weights_copy_enqueued(
                    module_type, module_name, buffer_idx, this->HtoD_stream);
                /* PUSH THE TASK BACK */
                {
                    std::lock_guard<std::mutex> lock(this->mutex_);
                    this->weights_copy_task_queue_[module_type].push(
                        module_name);
                }
            }
        }
    }
};


void HtoD_Engine::set_weight_copy_queue(
    std::unordered_map<std::string, std::vector<std::string>>&
        weight_copy_tasks) {
    std::lock_guard<std::mutex> lock(this->mutex_);
    this->weight_copy_tasks_ = weight_copy_tasks;
    for (auto& item : weight_copy_tasks) {
        auto module_type = item.first;
        auto module_names = item.second;
        this->weights_copy_task_queue_[module_type].clear();
        for (auto& module_name : module_names) {
            this->weights_copy_task_queue_[module_type].push(module_name);
        }
    }
}

void HtoD_Engine::stop_h2d_worker() {
    this->terminate_flag_ = true;
    if (this->HtoD_worker_.joinable()) {
        this->HtoD_worker_.join();
    }
}

// Forward declaration of CUDA kernel
__global__ void batched_page_copy_kernel(
    uint8_t** src_ptrs, 
    uint8_t** dst_ptrs, 
    size_t page_size, 
    int num_pages
);
void HtoD_Engine::batched_page_copy(const std::vector<void*>& gpu_ptrs,
                                    const std::vector<void*>& host_ptrs,
                                    int64_t page_byte_size) {
    CUDA_CHECK(cudaSetDevice(this->engine_config_.basic_config.device));
    int num_pages = host_ptrs.size();

    // Convert void* vectors to uint8_t* for the kernel
    std::vector<uint8_t*> src_ptrs(num_pages);
    std::vector<uint8_t*> dst_ptrs(num_pages);
    for (int i = 0; i < num_pages; ++i) {
        src_ptrs[i] = static_cast<uint8_t*>(host_ptrs[i]);
        dst_ptrs[i] = static_cast<uint8_t*>(gpu_ptrs[i]);
    }

    // Allocate device memory for pointer arrays
    uint8_t** d_src_ptrs;
    uint8_t** d_dst_ptrs;
    CUDA_CHECK(cudaMalloc(&d_src_ptrs, num_pages * sizeof(uint8_t*)));
    CUDA_CHECK(cudaMalloc(&d_dst_ptrs, num_pages * sizeof(uint8_t*)));

    // Copy pointer arrays to device (this is the only cudaMemcpy needed)
    CUDA_CHECK(cudaMemcpyAsync(d_src_ptrs, src_ptrs.data(), 
                               num_pages * sizeof(uint8_t*), 
                               cudaMemcpyHostToDevice, 
                               this->HtoD_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_dst_ptrs, dst_ptrs.data(), 
                               num_pages * sizeof(uint8_t*), 
                               cudaMemcpyHostToDevice, 
                               this->HtoD_stream));
    // Launch the batched copy kernel to do the actual data transfer
    // One block per page, 256 threads per block
    constexpr int THREADS_PER_BLOCK = 256;
    batched_page_copy_kernel<<<num_pages, THREADS_PER_BLOCK, 0, this->HtoD_stream>>>(
        d_src_ptrs, d_dst_ptrs, page_byte_size, num_pages
    );

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());

    // Synchronize to ensure copy is complete (blocking behavior like blocking_copy_)
    CUDA_CHECK(cudaStreamSynchronize(this->HtoD_stream));

    // Clean up device pointer arrays
    CUDA_CHECK(cudaFree(d_src_ptrs));
    CUDA_CHECK(cudaFree(d_dst_ptrs));
    
};
