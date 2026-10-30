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

#include "spdlog/spdlog.h"
#include <algorithm>
#include <chrono>
#include <memory>
#include <string>
#include <torch/extension.h>
#include <torch/torch.h>
#include <unordered_map>
#include <unordered_set>
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
        this->set_producer_state_(this->producer_failed_
                                      ? ProducerState::kFailed
                                      : ProducerState::kIdle);
        this->HtoD_worker_ = std::thread(&HtoD_Engine::HtoD_Worker, this);
    }
}


// Never throws: it runs from the destructor. stop_h2d_worker reports a
// producer failure instead.
void HtoD_Engine::Terminate() {
    this->terminate_flag_ = true;
    // Only a running producer needs the wake. From ~HtoD_Engine the weight
    // buffer it would notify may already be destroyed.
    if (this->HtoD_worker_.joinable()) {
        this->wake_producer();
        this->HtoD_worker_.join();
    }
    if (!this->producer_failed_) {
        this->producer_state_ = static_cast<int>(ProducerState::kParked);
    }
};

HtoD_Engine::~HtoD_Engine() {
    this->Terminate();
    // The joined producer landed or abandoned every copy.
    cudaSetDevice(this->engine_config_.basic_config.device);
    for (cudaEvent_t event : this->copy_events_) {
        cudaEventDestroy(event);
    }
};

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
    this->wake_producer();

    // Wait for completion
    completion_future.wait();

    return dst_tensor;
};

void HtoD_Engine::submit_to_KV_queue(std::vector<int64_t>& micro_batch,
                                     int64_t micro_batch_idx, int64_t layer_idx,
                                     int64_t byte_size) {
    this->kv_copy_task_queue_.push(
        std::make_tuple(micro_batch, micro_batch_idx, layer_idx, byte_size));
    this->wake_producer();
};

void HtoD_Engine::wake_producer() {
    {
        std::lock_guard<std::mutex> lock(this->producer_wake_mutex_);
        ++this->producer_wake_;
    }
    this->producer_wake_cv_.notify_all();
    // A producer waiting for a free weight slot wakes as well.
    this->gpu_weight_buffer_.wake_producer();
}

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

void HtoD_Engine::reset_weight_stream_profile(bool enabled) {
    std::lock_guard<std::mutex> lock(this->weight_profile_mutex_);
    this->weight_profile_enabled_ = enabled;
    this->weight_profile_modules_.clear();
    this->weight_profile_tensors_.clear();
    this->weight_profile_bytes_.clear();
    this->weight_profile_copy_seconds_.clear();
}

py::dict HtoD_Engine::get_weight_stream_profile() {
    std::lock_guard<std::mutex> lock(this->weight_profile_mutex_);
    py::dict result;
    result["enabled"] = this->weight_profile_enabled_;
    py::dict by_type;
    for (const auto& [module_type, modules] : this->weight_profile_modules_) {
        py::dict entry;
        entry["modules"] = modules;
        entry["tensors"] = this->weight_profile_tensors_[module_type];
        entry["bytes"] = this->weight_profile_bytes_[module_type];
        entry["copy_s"] = this->weight_profile_copy_seconds_[module_type];
        by_type[module_type.c_str()] = entry;
    }
    result["by_type"] = by_type;
    return result;
}

void HtoD_Engine::set_producer_state_(ProducerState state) {
    {
        std::lock_guard<std::mutex> lock(this->status_mutex_);
        this->producer_state_ = static_cast<int>(state);
    }
    // After status_mutex_: the buffer mutex never nests inside it.
    this->gpu_weight_buffer_.note_producer_progress();
}

void HtoD_Engine::set_copying_or_idle_() {
    // An outstanding acquire alone is no copy in flight: it shows as
    // acquiring, so a stall names the daemon rather than the copy engine.
    if (this->in_flight_.empty()) {
        this->set_producer_state_(ProducerState::kIdle);
    } else if (this->in_flight_.size() == 1 &&
               this->in_flight_.front().acquiring) {
        this->set_producer_state_(ProducerState::kAcquiring);
    } else {
        this->set_producer_state_(ProducerState::kCopying);
    }
}

namespace {
const char* const kProducerStates[] = {"idle", "acquiring", "copying",
                                       "parked", "failed"};
}  // namespace

std::string HtoD_Engine::describe_producer_locked_() const {
    std::string description =
        std::string("state=") + kProducerStates[this->producer_state_.load()];
    if (!this->in_flight_.empty()) {
        // The front entry: an acquired-ahead entry moves to the back as its
        // copies are queued, so this is the first in stream order, not
        // necessarily the first started.
        const InFlightCopy& front = this->in_flight_.front();
        description += " module=" + front.module_name +
                       " pass=" + std::to_string(front.pass);
        if (this->in_flight_.size() > 1) {
            description +=
                " in_flight=" + std::to_string(this->in_flight_.size());
        }
        for (const InFlightCopy& copy : this->in_flight_) {
            if (copy.acquiring) {
                description += " acquiring=" + copy.module_name;
            }
        }
    }
    return description;
}

std::string HtoD_Engine::describe_producer() {
    std::lock_guard<std::mutex> lock(this->status_mutex_);
    return this->describe_producer_locked_();
}

void HtoD_Engine::begin_in_flight_(const std::string& module_type,
                                   const std::string& module_name,
                                   int64_t buffer_idx,
                                   module_weight_tensor_map* buffer,
                                   int64_t pass) {
    const cudaEvent_t event = this->take_copy_event_();
    InFlightCopy copy;
    copy.module_type = module_type;
    copy.module_name = module_name;
    copy.buffer_idx = buffer_idx;
    copy.buffer = buffer;
    copy.pass = pass;
    copy.event = event;
    std::lock_guard<std::mutex> lock(this->status_mutex_);
    this->in_flight_.push_back(std::move(copy));
}

void HtoD_Engine::end_in_flight_(bool oldest) {
    this->end_in_flight_at_(oldest ? 0 : this->in_flight_.size() - 1);
}

// The entry of an outstanding acquire need not be the newest: modules of
// this node's store may start behind it.
void HtoD_Engine::end_in_flight_at_(size_t index) {
    cudaEvent_t event = nullptr;
    {
        std::lock_guard<std::mutex> lock(this->status_mutex_);
        event = this->in_flight_[index].event;
        this->in_flight_.erase(this->in_flight_.begin() +
                               static_cast<std::ptrdiff_t>(index));
    }
    // Reserved for kCopyEvents: never reallocates.
    this->free_copy_events_.push_back(event);
}

cudaEvent_t HtoD_Engine::take_copy_event_() {
    // On the producer's device, which it set as it started.
    this->copy_events_.reserve(kCopyEvents);
    this->free_copy_events_.reserve(kCopyEvents);
    while (this->copy_events_.size() < kCopyEvents) {
        cudaEvent_t event = nullptr;
        CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
        this->copy_events_.push_back(event);
        this->free_copy_events_.push_back(event);
    }
    if (this->free_copy_events_.empty()) {
        throw std::runtime_error(
            "HtoD: no free copy event with " +
            std::to_string(this->in_flight_.size()) + " modules in flight");
    }
    const cudaEvent_t event = this->free_copy_events_.back();
    this->free_copy_events_.pop_back();
    return event;
}

void HtoD_Engine::require_no_daemon_lease_for_pass(int64_t pass) {
    // While status_mutex_ is held the producer cannot change its modules in
    // flight, and it acquires a lease only after naming its module in flight
    // and releases it before dropping the name: any lease is of a module in
    // flight.
    std::lock_guard<std::mutex> lock(this->status_mutex_);
    for (const auto& module_name : this->weights_storage_.leased_modules()) {
        const bool carried = std::any_of(
            this->in_flight_.begin(), this->in_flight_.end(),
            [&module_name, pass](const InFlightCopy& copy) {
                return copy.module_name == module_name && copy.pass > pass;
            });
        if (carried) {
            continue;
        }
        throw std::runtime_error(
            "Cannot end weight pass " + std::to_string(pass) +
            ": the node daemon still leases " + module_name +
            " to this worker (producer " + this->describe_producer_locked_() +
            ")");
    }
}

void HtoD_Engine::count_module_(
    std::unordered_map<std::string, uint64_t>& counter,
    const std::string& module_type) {
    std::lock_guard<std::mutex> lock(this->status_mutex_);
    counter[module_type] += 1;
}

// Called from the producer's catch: must not throw. The copy stream is
// synchronized first, so no copy still writes into a slot about to be
// abandoned. An outstanding acquire is drained, and every daemon lease in
// flight is released and every slot in flight abandoned before waiters see
// the failure.
void HtoD_Engine::fail_producer_(const std::string& module_type,
                                 int64_t buffer_idx,
                                 const std::string& module_name,
                                 const std::string& message) {
    std::string failure = module_name.empty()
                              ? message
                              : module_name + ": " + message;
    const cudaError_t synchronized = cudaStreamSynchronize(this->HtoD_stream);
    if (synchronized != cudaSuccess) {
        failure += "; synchronizing the copy stream also failed: " +
                   std::string(cudaGetErrorString(synchronized));
    }
    // Before the walk: a lease in the reply is released with the others.
    this->drain_acquire_on_failure_(&failure);
    std::vector<std::pair<std::string, int64_t>> filling;
    if (buffer_idx >= 0) {
        filling.emplace_back(module_type, buffer_idx);
    }
    for (InFlightCopy& copy : this->in_flight_) {
        if (copy.buffer_idx >= 0) {
            filling.emplace_back(copy.module_type, copy.buffer_idx);
        }
        if (!copy.leased) {
            continue;
        }
        copy.leased = false;
        try {
            this->weights_storage_.release_module(copy.module_name);
        } catch (const std::exception& error) {
            failure += "; releasing the daemon lease of " + copy.module_name +
                       " also failed: " + std::string(error.what());
        }
    }
    this->logger_->error("HtoD producer failed: {}", failure);
    {
        std::lock_guard<std::mutex> lock(this->mutex_);
        this->producer_failed_ = true;
        try {
            this->gpu_weight_buffer_.record_producer_failure(filling, failure);
        } catch (const std::exception& error) {
            this->logger_->error(
                "HtoD producer could not record its failure: {}",
                error.what());
        }
        while (!this->in_flight_.empty()) {
            this->end_in_flight_(true);
        }
        this->set_producer_state_(ProducerState::kFailed);
    }
}

size_t HtoD_Engine::land_copies_(bool wait) {
    size_t landed = 0;
    size_t index = 0;
    while (index < this->in_flight_.size()) {
        InFlightCopy& copy = this->in_flight_[index];
        // No copy recorded yet: its acquire's reply is outstanding. Its
        // pooled event keeps the state of its last use, so it is never
        // queried or waited on: that would land an uncopied slot. Landing
        // steps over it instead of stopping. While it is outstanding no
        // module of its type and no remote module starts, so every entry
        // behind it is a module of another type read from this node's
        // store; claim_module_ moves an entry to the back as it records its
        // copies, so the recorded entries stay in stream order and each
        // type still lands in its cursor order.
        if (!copy.recorded) {
            if (copy.acquiring) {
                ++index;
                continue;
            }
            // Only an outstanding acquire's entry is unrecorded between
            // producer steps.
            if (wait && landed == 0) {
                throw std::runtime_error(
                    "HtoD: module in flight " + copy.module_name +
                    " has no copy recorded");
            }
            break;
        }
        if (wait && landed == 0) {
            // Bounded by the copies queued up to and including this module.
            const auto wait_start = std::chrono::steady_clock::now();
            CUDA_CHECK(cudaEventSynchronize(copy.event));
            const auto wait_end = std::chrono::steady_clock::now();
            std::lock_guard<std::mutex> lock(this->weight_profile_mutex_);
            if (this->weight_profile_enabled_) {
                this->weight_profile_copy_seconds_[copy.module_type] +=
                    std::chrono::duration<double>(wait_end - wait_start)
                        .count();
            }
        } else {
            // One stream: a copy not done yet means none after it is.
            const cudaError_t status = cudaEventQuery(copy.event);
            if (status == cudaErrorNotReady) {
                break;
            }
            CUDA_CHECK(status);
        }
        // Released before the landing: once no slot of a pass is FILLING,
        // the producer holds no lease of it either.
        if (copy.leased) {
            copy.leased = false;
            this->weights_storage_.release_module(copy.module_name);
        }
        {
            std::lock_guard<std::mutex> lock(this->weight_profile_mutex_);
            if (this->weight_profile_enabled_) {
                this->weight_profile_modules_[copy.module_type] += 1;
            }
        }
        this->logger_->debug("Copied module: {} to buffer: {}",
                             copy.module_name, copy.buffer_idx);
        // The only FILLING -> PRESENT writer; it wakes the consumer.
        const bool present = this->gpu_weight_buffer_.weights_copy_complete(
            copy.module_type, copy.module_name, copy.buffer_idx);
        this->count_module_(this->copied_modules_, copy.module_type);
        if (!present) {
            this->count_module_(this->released_on_landing_modules_,
                                copy.module_type);
        }
        // Dropped only after the landing: until then a failure abandons the
        // slot. The next entry shifts into index.
        this->end_in_flight_at_(index);
        ++landed;
    }
    if (landed > 0) {
        this->set_copying_or_idle_();
    }
    return landed;
}

py::dict HtoD_Engine::weight_stream_status() {
    py::dict result;
    result["producer_state"] = kProducerStates[this->producer_state_.load()];
    {
        std::lock_guard<std::mutex> lock(this->status_mutex_);
        // The front module in flight, then every one in stream order; an
        // entry whose acquire is outstanding is marked acquiring.
        result["module_in_flight"] =
            this->in_flight_.empty()
                ? py::object(py::none())
                : py::object(py::str(this->in_flight_.front().module_name));
        result["pass_in_flight"] =
            this->in_flight_.empty() ? 0 : this->in_flight_.front().pass;
        py::list in_flight;
        for (const InFlightCopy& copy : this->in_flight_) {
            py::dict entry;
            entry["module_type"] = copy.module_type;
            entry["module"] = copy.module_name;
            entry["pass"] = copy.pass;
            entry["index"] = copy.buffer_idx;
            entry["acquiring"] = copy.acquiring;
            entry["leased"] = copy.leased;
            in_flight.append(entry);
        }
        result["in_flight"] = in_flight;
        std::unordered_set<std::string> module_types;
        for (const auto* counter :
             {&this->copied_modules_, &this->skipped_modules_,
              &this->released_on_landing_modules_}) {
            for (const auto& [module_type, count] : *counter) {
                module_types.insert(module_type);
            }
        }
        py::dict by_type;
        for (const auto& module_type : module_types) {
            auto count = [&module_type](
                             const std::unordered_map<std::string, uint64_t>&
                                 counter) -> uint64_t {
                auto found = counter.find(module_type);
                return found == counter.end() ? 0 : found->second;
            };
            py::dict entry;
            entry["copied"] = count(this->copied_modules_);
            entry["skipped"] = count(this->skipped_modules_);
            entry["released_on_landing"] =
                count(this->released_on_landing_modules_);
            by_type[module_type.c_str()] = entry;
        }
        result["by_type"] = by_type;
    }
    const auto failure = this->gpu_weight_buffer_.producer_failure();
    result["producer_failure"] = failure.has_value()
                                     ? py::object(py::str(*failure))
                                     : py::object(py::none());

    const auto snapshot = this->gpu_weight_buffer_.slot_snapshot();
    py::dict slot_counts;
    for (const auto& [module_type, counts] : snapshot.counts) {
        py::dict entry;
        for (int state = 0; state < kWeightSlotStateCount; ++state) {
            entry[weight_slot_state_name(
                static_cast<WeightSlotState>(state))] = counts[state];
        }
        slot_counts[module_type.c_str()] = entry;
    }
    py::list held;
    for (const auto& slot : snapshot.held) {
        py::dict entry;
        entry["module_type"] = slot.module_type;
        entry["index"] = slot.index;
        entry["module"] = slot.module_name.empty()
                              ? py::object(py::none())
                              : py::object(py::str(slot.module_name));
        entry["state"] = weight_slot_state_name(slot.state);
        entry["pass"] = slot.pass;
        held.append(entry);
    }
    py::dict buffer;
    buffer["slot_counts"] = slot_counts;
    buffer["held"] = held;
    result["buffer"] = buffer;

    // The oldest open pass is the one the consumer executes.
    py::list open_passes;
    for (const int64_t pass : snapshot.open_passes) {
        open_passes.append(pass);
    }
    result["open_passes"] = open_passes;
    py::dict producer_cursor;
    for (const auto& [module_type, cursor] : snapshot.cursors) {
        py::dict entry;
        entry["pass"] = cursor.pass;
        entry["position"] = cursor.position;
        producer_cursor[module_type.c_str()] = entry;
    }
    result["producer_cursor"] = producer_cursor;
    py::dict passes;
    for (const auto& [pass, by_type] : snapshot.pass_counts) {
        py::dict pass_entry;
        for (const auto& [module_type, counts] : by_type) {
            py::dict entry;
            entry["copied"] = counts.copied;
            entry["skipped"] = counts.skipped;
            entry["released_on_landing"] = counts.released_on_landing;
            pass_entry[module_type.c_str()] = entry;
        }
        passes[py::int_(pass)] = pass_entry;
    }
    result["passes"] = passes;
    result["schedule_lost"] = snapshot.schedule_lost;
    return result;
}

size_t HtoD_Engine::acquiring_index_() const {
    for (size_t index = 0; index < this->in_flight_.size(); ++index) {
        if (this->in_flight_[index].acquiring) {
            return index;
        }
    }
    return this->in_flight_.size();
}

void HtoD_Engine::claim_module_(
    size_t index, const std::unordered_map<std::string, tensor_buffer>& src) {
    // Copied: a skip drops the entry.
    const std::string module_type = this->in_flight_[index].module_type;
    const std::string module_name = this->in_flight_[index].module_name;
    const int64_t buffer_idx = this->in_flight_[index].buffer_idx;
    auto& queue = this->weights_copy_task_queue_.at(module_type);
    const auto claim = this->gpu_weight_buffer_.claim_for_copy(
        module_type, buffer_idx, module_name);
    // Only this thread pops: the head is still module_name. It goes back to
    // the tail, so the next pass reads the list in the same order.
    std::string head;
    queue.try_pop(head);
    queue.push(module_name);
    if (claim == WeightProduction::kSkip) {
        // No copy, but the daemon lease is still released so the node FIFO
        // advances. The cursor moves past the module only after that, so its
        // pass cannot end while the lease is held.
        this->in_flight_[index].leased = false;
        this->weights_storage_.release_module(module_name);
        this->gpu_weight_buffer_.retire_skip(module_type, module_name);
        this->count_module_(this->skipped_modules_, module_type);
        this->end_in_flight_at_(index);
        this->set_copying_or_idle_();
        return;
    }
    if (this->in_flight_[index].buffer == nullptr) {
        throw std::runtime_error("HtoD: " + module_name +
                                 " was claimed for a copy without a slot");
    }
    // An acquired-ahead entry may have local modules' copies queued behind
    // it. Its copies go to the stream after theirs, so it moves to the back:
    // the recorded entries of in_flight_ are always in stream order, which
    // land_copies_ relies on.
    if (index + 1 != this->in_flight_.size()) {
        std::lock_guard<std::mutex> lock(this->status_mutex_);
        InFlightCopy moved = std::move(this->in_flight_[index]);
        this->in_flight_.erase(this->in_flight_.begin() +
                               static_cast<std::ptrdiff_t>(index));
        this->in_flight_.push_back(std::move(moved));
        index = this->in_flight_.size() - 1;
    }
    this->set_producer_state_(ProducerState::kCopying);
    auto dst = *this->in_flight_[index].buffer;
    void* src_ptr;
    int64_t src_byte_size;
    uint64_t submitted_tensors = 0;
    uint64_t submitted_bytes = 0;
    const auto submit_start = std::chrono::steady_clock::now();
    for (auto& [tensor_name, host_tensor_storage] : src) {
        src_ptr = host_tensor_storage.data_ptr;
        src_byte_size = host_tensor_storage.byte_size;
        // find(), not operator[]: dst is an unordered_map and operator[]
        // DEFAULT-INSERTS an undefined torch::Tensor on every miss,
        // permanently growing a buffer map that is reused across ring slots.
        auto slot = dst.find(tensor_name);
        if (slot == dst.end() || !slot->second.defined() ||
            !slot->second.has_storage()) {
            // The host map carries a tensor module_shapes declares no slot
            // for. Continuing drops it silently and the consumer reads
            // whatever the slot last held.
            this->logger_->error(
                "Module {}: host tensor {} has no GPU slot -- module_shapes "
                "declares no such key",
                module_name, tensor_name);
            throw std::runtime_error("HtoD: host tensor has no GPU slot: " +
                                     tensor_name);
        }
        int64_t dst_byte_size = slot->second.nbytes();
        if (src_byte_size != dst_byte_size) {
            // The copy writes src_byte_size bytes with no bound check: a
            // short slot is overrun into its neighbour, a long one keeps a
            // stale tail. Both are silent and both produce wrong weights.
            this->logger_->error(
                "Module {}: tensor {} size mismatch -- host {} B, GPU slot {} "
                "B (module_shapes/dtype disagrees with the checkpoint)",
                module_name, tensor_name, src_byte_size, dst_byte_size);
            throw std::runtime_error(
                "HtoD: host/GPU byte size mismatch for " + tensor_name);
        }
        CUDA_CHECK(cudaMemcpyAsync(slot->second.data_ptr(), src_ptr,
                                   src_byte_size, cudaMemcpyHostToDevice,
                                   this->HtoD_stream));
        submitted_tensors += 1;
        submitted_bytes += static_cast<uint64_t>(src_byte_size);
    }
    // One event after the module's last copy. The stream runs its copies in
    // order; land_copies_ observes the event, releases the lease and lands
    // the slot, oldest entry first.
    CUDA_CHECK(cudaEventRecord(this->in_flight_[index].event,
                               this->HtoD_stream));
    this->in_flight_[index].recorded = true;
    const auto submit_end = std::chrono::steady_clock::now();
    {
        std::lock_guard<std::mutex> lock(this->weight_profile_mutex_);
        if (this->weight_profile_enabled_) {
            this->weight_profile_tensors_[module_type] += submitted_tensors;
            this->weight_profile_bytes_[module_type] += submitted_bytes;
            this->weight_profile_copy_seconds_[module_type] +=
                std::chrono::duration<double>(submit_end - submit_start)
                    .count();
        }
    }
    this->logger_->debug("Submitted module: {} to buffer: {}", module_name,
                         buffer_idx);
}

bool HtoD_Engine::poll_acquire_(bool block) {
    const size_t index = this->acquiring_index_();
    if (index == this->in_flight_.size()) {
        return false;
    }
    // A retry was re-sent at once, so the acquire stays outstanding.
    acquire_poll polled = this->weights_storage_.poll_acquire(block);
    if (polled.status != acquire_poll::Status::kReady) {
        return false;
    }
    // release_module() is owed for the module from here.
    this->in_flight_[index].acquiring = false;
    this->in_flight_[index].leased = true;
    this->claim_module_(index, polled.tensors);
    return true;
}

// Stopping with an acquire outstanding. Its reply comes within one daemon
// slice. A lease is claimed like any other: released at its cursor, the
// daemon would free the slot and never ready the module again.
void HtoD_Engine::drain_acquire_on_stop_() {
    const size_t index = this->acquiring_index_();
    if (index == this->in_flight_.size()) {
        return;
    }
    this->in_flight_[index].acquiring = false;
    acquire_drain drained = this->weights_storage_.drain_acquire();
    if (!drained.error.empty()) {
        throw std::runtime_error("draining the daemon acquire of " +
                                 this->in_flight_[index].module_name +
                                 " failed: " + drained.error);
    }
    if (!drained.leased) {
        // A retry: nothing was acquired, and the module stays at its cursor
        // and the head of its queue, as for a cancelled acquire.
        const InFlightCopy& copy = this->in_flight_[index];
        if (copy.buffer_idx >= 0) {
            this->gpu_weight_buffer_.abandon(copy.module_type,
                                             copy.buffer_idx);
        }
        this->end_in_flight_at_(index);
        return;
    }
    this->in_flight_[index].leased = true;
    this->claim_module_(index, drained.tensors);
}

void HtoD_Engine::drain_acquire_on_failure_(std::string* failure) noexcept {
    const size_t index = this->acquiring_index_();
    if (index == this->in_flight_.size()) {
        return;
    }
    InFlightCopy& copy = this->in_flight_[index];
    copy.acquiring = false;
    const acquire_drain drained = this->weights_storage_.drain_acquire();
    if (!drained.error.empty()) {
        *failure += "; draining the daemon acquire of " + copy.module_name +
                    " also failed: " + drained.error;
        return;
    }
    if (!drained.leased) {
        // A retry acquired nothing; the module stays at its cursor.
        return;
    }
    copy.leased = true;
    // The claim moves the cursor and the queue head past the module. No
    // copy is queued: the slot is abandoned and the lease released with
    // every other entry by the caller, or now for a pending skip.
    try {
        auto& queue = this->weights_copy_task_queue_.at(copy.module_type);
        const auto claim = this->gpu_weight_buffer_.claim_for_copy(
            copy.module_type, copy.buffer_idx, copy.module_name);
        std::string head;
        queue.try_pop(head);
        queue.push(copy.module_name);
        if (claim == WeightProduction::kSkip) {
            // The claim abandoned the slot.
            copy.buffer_idx = -1;
            copy.leased = false;
            this->weights_storage_.release_module(copy.module_name);
            this->gpu_weight_buffer_.retire_skip(copy.module_type,
                                                 copy.module_name);
            this->count_module_(this->skipped_modules_, copy.module_type);
        }
    } catch (const std::exception& error) {
        *failure += "; claiming the drained acquire of " + copy.module_name +
                    " also failed: " + std::string(error.what());
    } catch (...) {
        *failure += "; claiming the drained acquire of " + copy.module_name +
                    " also failed";
    }
}

void HtoD_Engine::HtoD_Worker() {
    CUDA_CHECK(cudaSetDevice(this->engine_config_.basic_config.device));
    while (!terminate_flag_) {
        // A wake after this read keeps the iteration from parking below.
        uint64_t wake_seen = 0;
        {
            std::lock_guard<std::mutex> lock(this->producer_wake_mutex_);
            wake_seen = this->producer_wake_;
        }
        // Read as early as wake_seen: a slot freed or a wake after this read
        // ends the full-ring wait below at once.
        const uint64_t slot_epoch_seen = this->gpu_weight_buffer_.slot_epoch();
        // Set when this iteration lands, copies or runs anything; an idle
        // iteration waits below instead of immediately re-taking the
        // weight-buffer mutex.
        bool progressed = false;
        // Stays true while every weight type is uninstalled or parked: only
        // begin_weight_pass can give the producer weight work.
        bool weights_parked = true;
        // The first type whose ring had no FREE slot this iteration.
        std::string full_ring_type;
        // Land every copy that completed, oldest first, before anything
        // else: the consumer may be waiting for one of them. The reply of an
        // acquire outstanding is polled before and after: a lease queues its
        // module's copies at once, a retry is re-sent at once.
        try {
            progressed |= this->poll_acquire_(false);
            if (this->land_copies_(false) > 0) {
                progressed = true;
            }
            progressed |= this->poll_acquire_(false);
        } catch (const std::exception& error) {
            // Never rethrow: waiters see the failure instead.
            this->fail_producer_("", -1, "", error.what());
        } catch (...) {
            this->fail_producer_("", -1, "", "unknown exception");
        }
        std::packaged_task<void()> task;
        while (on_demand_task_queue_.try_pop(task)) {
            task();
            progressed = true;
        }
        if (!this->kv_copy_task_queue_.empty()) {
            auto optional_buffer = this->gpu_kv_buffer_.acquireEmptyBuffer();
            if (optional_buffer.has_value()) {
                progressed = true;
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
            // A failed producer copies nothing until set_weight_copy_queue;
            // a stop request ends the iteration between modules; no module
            // starts while kMaxCopiesInFlight are in flight.
            if (this->producer_failed_ || this->terminate_flag_ ||
                this->in_flight_.size() >= kMaxCopiesInFlight) {
                break;
            }
            // The type's cursor (pass, position) lives in the weight buffer,
            // next to the passes the consumer armed.
            const auto next =
                this->gpu_weight_buffer_.next_production(module_type);
            if (next.kind == WeightProduction::kNone ||
                next.kind == WeightProduction::kParked) {
                // Nothing installed (e.g. persistent experts), or the type
                // finished every armed pass: it parks until the next one.
                continue;
            }
            weights_parked = false;
            // The primary daemon connection carries one request at a time,
            // so at most one acquire is outstanding, issued in cursor order.
            const size_t acquiring = this->acquiring_index_();
            const bool acquire_outstanding =
                acquiring < this->in_flight_.size();
            if (acquire_outstanding &&
                this->in_flight_[acquiring].module_type == module_type) {
                // The type's cursor module waits for its reply: nothing past
                // it starts.
                continue;
            }
            // A module of this node's registered store holds no lease and is
            // read at once, so it pipelines freely, even behind an acquire.
            const bool remote =
                this->weights_storage_.is_remote_module(next.module_name);
            if (remote && acquire_outstanding) {
                // No other request, not even a try, goes out before the
                // outstanding acquire's reply.
                continue;
            }
            // With no module in flight, hence no lease held, a remote fetch
            // blocks in the daemon acquire. With copies in flight it is
            // acquired ahead instead: one waiting request whose reply is
            // polled while the copies land, so the copy engine never drains
            // to idle for it; meanwhile the landings release their leases on
            // the daemon's release-only connection.
            const bool may_wait = this->in_flight_.empty();
            // A pending skip needs no slot: the module is only acquired and
            // released, so the node daemon's FIFO stays aligned across ranks.
            module_weight_tensor_map* buffer = nullptr;
            int64_t buffer_idx = -1;
            if (next.kind == WeightProduction::kCopy) {
                auto optional_buffer =
                    this->gpu_weight_buffer_.acquireEmptyBuffer(module_type);
                if (!optional_buffer.has_value()) {
                    if (full_ring_type.empty()) {
                        full_ring_type = module_type;
                    }
                    continue;
                }
                buffer = &optional_buffer->first.get();
                buffer_idx = optional_buffer->second;
            }
            std::string module_name;
            try {
                CUDA_CHECK(
                    cudaSetDevice(this->engine_config_.basic_config.device));
                auto& queue = this->weights_copy_task_queue_.at(module_type);
                // Peek, do not pop: a cancelled acquire leaves the module at
                // the head, so the schedule order is unchanged.
                if (!queue.try_front(module_name) ||
                    module_name != next.module_name) {
                    throw std::runtime_error(
                        "HtoD queue head '" + module_name + "' of " +
                        module_type + " is not its cursor module " +
                        next.module_name);
                }
                // Named before the fetch and dropped only after its daemon
                // lease is released: any lease is of a module in flight.
                this->begin_in_flight_(module_type, module_name, buffer_idx,
                                       buffer, next.pass);
                if (remote && !may_wait) {
                    // Acquired ahead: the entry holds the reserved slot and
                    // lands nothing until its reply's copies are queued. An
                    // acquire issued is no progress, and changes no state.
                    this->in_flight_.back().acquiring = true;
                    this->weights_storage_.begin_acquire(module_name);
                    continue;
                }
                if (may_wait) {
                    // Only a fetch that may wait shows as acquiring.
                    this->set_producer_state_(ProducerState::kAcquiring);
                }
                // Waits only for a remote module, hence only with nothing in
                // flight; a local one is ready at once.
                auto fetched =
                    this->weights_storage_.get_module_weights_storage(
                        module_name, &this->terminate_flag_);
                if (fetched.status == module_fetch::Status::kCancelled) {
                    // Stop requested before the daemon readied the module:
                    // nothing is leased, and the module stays at the head of
                    // its queue and at its cursor.
                    if (buffer_idx >= 0) {
                        this->gpu_weight_buffer_.abandon(module_type,
                                                         buffer_idx);
                    }
                    this->end_in_flight_(false);
                    this->set_producer_state_(ProducerState::kParked);
                    break;
                }
                progressed = true;
                // release_module() is owed for module_name: a remote module
                // holds a daemon lease from its fetch until then.
                this->in_flight_.back().leased = true;
                this->claim_module_(this->in_flight_.size() - 1,
                                    fetched.tensors);
            } catch (const std::exception& error) {
                // Never rethrow: an exception leaving this thread is
                // std::terminate. Waiters see the failure instead.
                this->fail_producer_(module_type, buffer_idx, module_name,
                                     error.what());
                break;
            } catch (...) {
                this->fail_producer_(module_type, buffer_idx, module_name,
                                     "unknown exception");
                break;
            }
        }
        if (!progressed) {
            if (!this->in_flight_.empty()) {
                // Nothing else to do: wait for the oldest copy and land it.
                // Never park or wait for a slot with a copy un-landed or an
                // acquire outstanding: the consumer may be waiting for it
                // while it holds every other slot. Bounded by the copies
                // already queued. Only when the outstanding acquire is the
                // sole entry is its reply waited for instead, at most one
                // daemon slice, and a lease queues its copies; with any copy
                // recorded, the oldest copy is waited for, so a landing is
                // never held behind the daemon.
                try {
                    if (this->in_flight_.size() == 1 &&
                        this->in_flight_.front().acquiring) {
                        this->poll_acquire_(true);
                    } else {
                        this->land_copies_(true);
                    }
                } catch (const std::exception& error) {
                    this->fail_producer_("", -1, "", error.what());
                } catch (...) {
                    this->fail_producer_("", -1, "", "unknown exception");
                }
                continue;
            }
            if (weights_parked && this->kv_copy_task_queue_.empty()) {
                // No type has an armed position to produce and no KV copy is
                // queued: block until a pass is armed, a task is queued or
                // the producer is stopped, instead of sleeping in a loop.
                std::unique_lock<std::mutex> lock(this->producer_wake_mutex_);
                this->producer_wake_cv_.wait(lock, [this, wake_seen] {
                    return this->terminate_flag_ ||
                           this->producer_wake_ != wake_seen;
                });
                continue;
            }
            if (!this->kv_copy_task_queue_.empty()) {
                // The KV buffer is full, and GPU_KV_Buffer signals no freed
                // slot. Without a sleep the iteration re-locks the buffer
                // mutexes in a tight loop and starves the consumer. 50 us is
                // negligible against a slot's copy time.
                std::this_thread::sleep_for(std::chrono::microseconds(50));
                continue;
            }
            if (full_ring_type.empty() || this->producer_failed_) {
                // No ring was found full, or a later type failed before its
                // fetch returned: the iteration ended on a stop or a producer
                // failure, which the next iteration parks on.
                continue;
            }
            // Every unparked weight type found its ring full. The weight
            // buffer reclaims, waits on the oldest pending release or on its
            // slot notification, and returns on any slot change, wake or
            // stop since slot_epoch_seen. It counts no consumer progress.
            try {
                this->gpu_weight_buffer_.wait_for_free_slot(
                    full_ring_type, slot_epoch_seen, this->terminate_flag_);
            } catch (const std::exception& error) {
                // Never rethrow: waiters see the failure instead.
                this->fail_producer_(full_ring_type, -1, "", error.what());
            } catch (...) {
                this->fail_producer_(full_ring_type, -1, "",
                                     "unknown exception");
            }
        }
    }
    // Stopping: an acquire outstanding is drained first, bounded by one
    // daemon slice; a lease in its reply is copied or retired as a skip.
    // Then nothing more is submitted, and every copy in flight lands before
    // the thread exits, bounded by the copies already queued. A failure
    // abandons the rest.
    try {
        this->drain_acquire_on_stop_();
    } catch (const std::exception& error) {
        this->fail_producer_("", -1, "", error.what());
    } catch (...) {
        this->fail_producer_("", -1, "", "unknown exception");
    }
    while (!this->in_flight_.empty()) {
        try {
            this->land_copies_(true);
        } catch (const std::exception& error) {
            this->fail_producer_("", -1, "", error.what());
        } catch (...) {
            this->fail_producer_("", -1, "", "unknown exception");
        }
    }
};


void HtoD_Engine::set_weight_copy_queue(
    std::unordered_map<std::string, std::vector<std::string>>&
        weight_copy_tasks) {
    // The producer reads the queues and its cursors without the engine
    // mutex, so both are replaced only while it is stopped.
    if (this->HtoD_worker_.joinable()) {
        throw std::runtime_error(
            "Cannot install a weight copy queue while the HtoD producer is "
            "running; stop it first");
    }
    // Refuses an open pass or a held slot before any queue changes.
    this->gpu_weight_buffer_.set_weight_copy_task(weight_copy_tasks);
    std::lock_guard<std::mutex> lock(this->mutex_);
    this->weight_copy_tasks_ = weight_copy_tasks;
    // A type missing from the new tasks must not keep its old queue.
    this->weights_copy_task_queue_.clear();
    for (auto& item : weight_copy_tasks) {
        auto module_type = item.first;
        auto module_names = item.second;
        for (auto& module_name : module_names) {
            this->weights_copy_task_queue_[module_type].push(module_name);
        }
    }
    // A new queue clears a failed producer's failure.
    this->gpu_weight_buffer_.clear_producer_failure();
    this->producer_failed_ = false;
    {
        std::lock_guard<std::mutex> status_lock(this->status_mutex_);
        this->copied_modules_.clear();
        this->skipped_modules_.clear();
        this->released_on_landing_modules_.clear();
        if (this->producer_state_ == static_cast<int>(ProducerState::kFailed)) {
            this->producer_state_ = static_cast<int>(
                this->HtoD_worker_.joinable() ? ProducerState::kIdle
                                              : ProducerState::kParked);
        }
    }
}

void HtoD_Engine::stop_h2d_worker() {
    this->terminate_flag_ = true;
    this->wake_producer();
    if (this->HtoD_worker_.joinable()) {
        this->HtoD_worker_.join();
    }
    const auto failure = this->gpu_weight_buffer_.producer_failure();
    if (failure.has_value()) {
        throw std::runtime_error("HtoD producer failed: " + *failure);
    }
    this->producer_state_ = static_cast<int>(ProducerState::kParked);
}

HtoD_Engine::WeightPassAbort HtoD_Engine::abort_weight_passes() {
    // The producer reads the queues and the cursors without the engine
    // mutex, and a running one could acquire from the daemon meanwhile.
    if (this->HtoD_worker_.joinable()) {
        throw std::runtime_error(
            "Cannot abort the weight passes while the HtoD producer is "
            "running; stop it first");
    }
    WeightPassAbort result;
    // The producer releases a lease before its copy lands, so one is left
    // only where that release failed. Each is released; a failure is
    // reported after the passes are dropped.
    std::string release_failures;
    for (const auto& module_name : this->weights_storage_.leased_modules()) {
        try {
            this->weights_storage_.release_module(module_name);
            ++result.daemon_leases;
        } catch (const std::exception& error) {
            release_failures += (release_failures.empty() ? "" : "; ") +
                                module_name + ": " + error.what();
        }
    }
    // The daemon serves every rank of the node the same cyclic schedule.
    // This rank stopped somewhere inside a pass its siblings may have
    // finished or not started, so its position cannot be matched again.
    result.schedule_lost = this->weights_storage_.is_distributed();
    result.slots = this->gpu_weight_buffer_.abort_passes(result.schedule_lost);
    {
        // The table parked every cursor at the head of its list.
        std::lock_guard<std::mutex> lock(this->mutex_);
        this->weights_copy_task_queue_.clear();
        for (const auto& [module_type, module_names] :
             this->weight_copy_tasks_) {
            for (const auto& module_name : module_names) {
                this->weights_copy_task_queue_[module_type].push(module_name);
            }
        }
    }
    // The stopped producer left nothing in flight: its stop landed every
    // copy and a failure abandoned them.
    this->logger_->warn(
        "Aborted {} weight passes and {} pending skips; released {} daemon "
        "leases; schedule_lost={}",
        result.slots.passes, result.slots.pending_skips, result.daemon_leases,
        result.schedule_lost);
    if (!release_failures.empty()) {
        throw std::runtime_error(
            "Aborted the weight passes, but releasing a daemon lease failed: " +
            release_failures);
    }
    return result;
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
