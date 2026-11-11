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
                                      : ProducerState::kIdle,
                                  "");
        this->HtoD_worker_ = std::thread(&HtoD_Engine::HtoD_Worker, this);
    }
}


// Never throws: it runs from the destructor. stop_h2d_worker reports a
// producer failure instead.
void HtoD_Engine::Terminate() {
    this->terminate_flag_ = true;
    this->wake_producer();
    if (this->HtoD_worker_.joinable()) {
        this->HtoD_worker_.join();
    }
    if (!this->producer_failed_) {
        this->producer_state_ = static_cast<int>(ProducerState::kParked);
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

void HtoD_Engine::set_producer_state_(ProducerState state,
                                      const std::string& module_name,
                                      int64_t pass) {
    {
        std::lock_guard<std::mutex> lock(this->status_mutex_);
        this->module_in_flight_ = module_name;
        this->pass_in_flight_ = module_name.empty() ? 0 : pass;
        this->producer_state_ = static_cast<int>(state);
    }
    // After status_mutex_: the buffer mutex never nests inside it.
    this->gpu_weight_buffer_.note_producer_progress();
}

namespace {
const char* const kProducerStates[] = {"idle", "acquiring", "copying",
                                       "parked", "failed"};
}  // namespace

std::string HtoD_Engine::describe_producer_locked_() const {
    std::string description =
        std::string("state=") + kProducerStates[this->producer_state_.load()];
    if (!this->module_in_flight_.empty()) {
        description += " module=" + this->module_in_flight_ +
                       " pass=" + std::to_string(this->pass_in_flight_);
    }
    return description;
}

std::string HtoD_Engine::describe_producer() {
    std::lock_guard<std::mutex> lock(this->status_mutex_);
    return this->describe_producer_locked_();
}

void HtoD_Engine::require_no_daemon_lease_for_pass(int64_t pass) {
    // While status_mutex_ is held the producer cannot change its module in
    // flight, and it acquires a lease only after naming the module and
    // releases it before clearing the name: any lease is of that module.
    std::lock_guard<std::mutex> lock(this->status_mutex_);
    for (const auto& module_name : this->weights_storage_.leased_modules()) {
        if (module_name == this->module_in_flight_ &&
            this->pass_in_flight_ > pass) {
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

// Called from the producer's catch: must not throw. The slot is abandoned and
// the daemon lease released before waiters see the failure.
void HtoD_Engine::fail_producer_(const std::string& module_type,
                                 int64_t buffer_idx,
                                 const std::string& module_name, bool leased,
                                 const std::string& message) {
    std::string failure = module_name.empty()
                              ? message
                              : module_name + ": " + message;
    if (this->replica_plan_ != nullptr) {
        // Peers polling this member's barriers fail fast on the abort word
        // instead of waiting out the stall timeout.
        this->replica_plan_->block.abort(this->replica_plan_->member_index);
    }
    if (leased) {
        try {
            this->weights_storage_.release_module(module_name);
        } catch (const std::exception& error) {
            failure += "; releasing its daemon lease also failed: " +
                       std::string(error.what());
        }
    }
    this->logger_->error("HtoD producer failed: {}", failure);
    {
        std::lock_guard<std::mutex> lock(this->mutex_);
        this->producer_failed_ = true;
        try {
            this->gpu_weight_buffer_.record_producer_failure(
                module_type, buffer_idx, failure);
        } catch (const std::exception& error) {
            this->logger_->error(
                "HtoD producer could not record its failure: {}",
                error.what());
        }
        this->set_producer_state_(ProducerState::kFailed, "");
    }
}

py::dict HtoD_Engine::weight_stream_status() {
    py::dict result;
    result["producer_state"] = kProducerStates[this->producer_state_.load()];
    {
        std::lock_guard<std::mutex> lock(this->status_mutex_);
        result["module_in_flight"] =
            this->module_in_flight_.empty()
                ? py::object(py::none())
                : py::object(py::str(this->module_in_flight_));
        result["pass_in_flight"] = this->pass_in_flight_;
        std::unordered_set<std::string> module_types;
        for (const auto* counter :
             {&this->copied_modules_, &this->skipped_modules_,
              &this->released_on_landing_modules_,
              &this->replica_pulled_modules_}) {
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
            entry["replica_pulled"] = count(this->replica_pulled_modules_);
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

void HtoD_Engine::install_replica_plan(
    int64_t member_index, int64_t group_size, uint64_t layers_per_pass,
    uint32_t records_per_frame,
    const std::vector<ReplicaModuleInstall>& modules,
    torch::Tensor control_block,
    const std::unordered_map<std::string, std::vector<int64_t>>& peer_fds,
    const std::unordered_map<std::string, std::vector<int64_t>>& peer_bytes,
    const std::unordered_map<std::string, int64_t>& strides) {
    if (this->HtoD_worker_.joinable()) {
        throw std::runtime_error(
            "install_replica_plan requires a stopped producer");
    }
    if (group_size < 2) {
        throw std::runtime_error(
            "a single-member group installs no plan; the uninstalled engine "
            "is that group");
    }
    auto runtime = std::make_unique<ReplicaPlanRuntime>(
        ReplicaControlBlock::attach(control_block.data_ptr(),
                                    static_cast<size_t>(
                                        control_block.nbytes())));
    // Types in sorted order, identical on every member (each installs the
    // same task). Barrier words, owned and total counts are per type.
    std::set<std::string> type_names;
    for (const auto& module : modules) {
        type_names.insert(module.module_type);
    }
    size_t next_type = 0;
    for (const auto& name : type_names) {
        runtime->type_index[name] = next_type++;
    }
    if (runtime->block.members() != static_cast<size_t>(group_size) ||
        runtime->block.records_per_frame() !=
            static_cast<size_t>(records_per_frame) ||
        runtime->block.num_types() != runtime->type_index.size()) {
        throw std::runtime_error(
            "control block geometry does not match the installed plan");
    }
    runtime->member_index = member_index;
    runtime->group_size = group_size;
    runtime->layers_per_pass = layers_per_pass;
    runtime->records_per_frame = records_per_frame;
    runtime->block_backing = std::move(control_block);
    runtime->owned_per_type_pos.assign(
        runtime->type_index.size(),
        std::vector<uint32_t>(layers_per_pass, 0));
    runtime->total_per_type_pos.assign(
        runtime->type_index.size(),
        std::vector<uint32_t>(layers_per_pass, 0));
    for (const auto& module : modules) {
        if (module.layer_position >= layers_per_pass) {
            throw std::runtime_error(
                "module " + module.module + " is outside the pass layout");
        }
        ReplicaModuleInstall entry = module;
        entry.type_index = static_cast<uint32_t>(
            runtime->type_index.at(module.module_type));
        ++runtime->total_per_type_pos[entry.type_index]
                                     [module.layer_position];
        if (entry.kind !=
            static_cast<uint8_t>(WeightRecordKind::Replica)) {
            ++runtime->owned_per_type_pos[entry.type_index]
                                         [module.layer_position];
        }
        runtime->modules[module.module] = entry;
    }
    CUDA_CHECK(cudaSetDevice(this->engine_config_.basic_config.device));
    const int device = this->engine_config_.basic_config.device;
    for (const auto& [module_type, fds] : peer_fds) {
        if (fds.size() != static_cast<size_t>(group_size)) {
            throw std::runtime_error(
                "peer fds of " + module_type +
                " must hold one entry per member");
        }
        auto bytes_entry = peer_bytes.find(module_type);
        if (bytes_entry == peer_bytes.end() ||
            bytes_entry->second.size() != static_cast<size_t>(group_size)) {
            throw std::runtime_error(
                "peer byte sizes of " + module_type +
                " must hold one entry per member");
        }
        auto stride = strides.find(module_type);
        if (stride == strides.end() || stride->second <= 0) {
            throw std::runtime_error("missing ring stride for " + module_type);
        }
        const auto [own_base, own_bytes] =
            this->gpu_weight_buffer_.ring_segment_base(module_type);
        auto& bases = runtime->segment_base[module_type];
        for (int64_t member = 0; member < group_size; ++member) {
            if (member == member_index) {
                if (fds[member] >= 0) {
                    throw std::runtime_error(
                        "this member's own fd slot of " + module_type +
                        " must be -1");
                }
                bases.push_back(own_base);
                continue;
            }
            const int64_t mapped = bytes_entry->second[member];
            if (fds[member] < 0 || mapped <= 0 || mapped < own_bytes) {
                throw std::runtime_error(
                    "peer segment of " + module_type + " from member " +
                    std::to_string(member) + " has fd " +
                    std::to_string(fds[member]) + " and " +
                    std::to_string(mapped) + " bytes, need >= " +
                    std::to_string(own_bytes));
            }
            auto import = std::make_shared<batchgen_vmm::VmmPeerImport>(
                static_cast<int>(fds[member]), static_cast<size_t>(mapped),
                device);
            bases.push_back(import->data());
            runtime->keepalive.push_back(std::move(import));
        }
        runtime->stride[module_type] = stride->second;
    }
    runtime->pull_streams.resize(group_size, nullptr);
    for (int64_t member = 0; member < group_size; ++member) {
        if (member == member_index) {
            continue;
        }
        CUDA_CHECK(cudaStreamCreateWithFlags(&runtime->pull_streams[member],
                                             cudaStreamNonBlocking));
    }
    ReplicaControlBlock block = runtime->block;
    this->gpu_weight_buffer_.set_peer_release_gate(
        [block, types = runtime->type_index](
            const std::string& module_type, int64_t frame) mutable -> bool {
            auto found = types.find(module_type);
            if (found == types.end()) {
                // A slot of a type outside the plan is never peer-gated.
                return true;
            }
            return block.first_waiting(found->second,
                                       ReplicaControlBlock::Barrier::B,
                                       static_cast<uint64_t>(frame)) ==
                   block.members();
        });
    this->replica_plan_ = std::move(runtime);
    this->logger_->info(
        "Replica plan installed: member {} of {}, {} records/frame, {} "
        "layer positions",
        member_index, group_size, records_per_frame, layers_per_pass);
}

void HtoD_Engine::uninstall_replica_plan() {
    if (this->replica_plan_ == nullptr) {
        return;
    }
    if (this->HtoD_worker_.joinable()) {
        throw std::runtime_error(
            "uninstall_replica_plan requires a stopped producer");
    }
    CUDA_CHECK(cudaSetDevice(this->engine_config_.basic_config.device));
    for (cudaStream_t stream : this->replica_plan_->pull_streams) {
        if (stream != nullptr) {
            CUDA_CHECK(cudaStreamSynchronize(stream));
            CUDA_CHECK(cudaStreamDestroy(stream));
        }
    }
    this->gpu_weight_buffer_.set_peer_release_gate(nullptr);
    this->replica_plan_.reset();
    this->logger_->info("Replica plan uninstalled");
}

void HtoD_Engine::replica_mark_owned_landed_(size_t type,
                                             uint64_t flat_frame) {
    ReplicaPlanRuntime& plan = *this->replica_plan_;
    const auto key = std::make_pair(type, flat_frame);
    const uint32_t landed = ++plan.owned_landed[key];
    const uint32_t owned =
        plan.owned_per_type_pos[type][flat_frame % plan.layers_per_pass];
    if (landed == owned) {
        plan.block.advance(plan.member_index, type,
                           ReplicaControlBlock::Barrier::A, flat_frame);
        plan.owned_landed.erase(key);
    }
}

void HtoD_Engine::replica_mark_terminal_(size_t type, uint64_t flat_frame) {
    ReplicaPlanRuntime& plan = *this->replica_plan_;
    const auto key = std::make_pair(type, flat_frame);
    const uint32_t terminal = ++plan.terminal[key];
    if (terminal ==
        plan.total_per_type_pos[type][flat_frame % plan.layers_per_pass]) {
        plan.block.advance(plan.member_index, type,
                           ReplicaControlBlock::Barrier::B, flat_frame);
        plan.terminal.erase(key);
        plan.wait_since.erase(key);
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
        // Set when this iteration copies or runs anything; an idle iteration
        // backs off below instead of immediately re-taking the weight-buffer
        // mutex.
        bool progressed = false;
        // Stays true while every weight type is uninstalled or parked: only
        // begin_weight_pass can give the producer weight work.
        bool weights_parked = true;
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
            // a stop request ends the iteration between modules.
            if (this->producer_failed_ || this->terminate_flag_) {
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
            // Record kind under the installed replica plan. Without a plan
            // (single-member group, decode, daemon sources) every module is
            // HostPrivate and the flow below is exactly the historic one.
            const ReplicaModuleInstall* role = nullptr;
            uint64_t flat = 0;
            if (this->replica_plan_ != nullptr) {
                auto found = this->replica_plan_->modules.find(
                    next.module_name);
                if (found == this->replica_plan_->modules.end()) {
                    this->fail_producer_(module_type, -1, next.module_name,
                                         false,
                                         "module is absent from the installed "
                                         "replica plan");
                    break;
                }
                role = &found->second;
                flat = flat_frame(static_cast<uint64_t>(next.pass),
                                  role->layer_position,
                                  this->replica_plan_->layers_per_pass);
            }
            if (role != nullptr &&
                role->kind ==
                    static_cast<uint8_t>(WeightRecordKind::Replica) &&
                next.kind == WeightProduction::kCopy) {
                ReplicaPlanRuntime& plan = *this->replica_plan_;
                const size_t type = role->type_index;
                // A frame this member owns nothing of (for this type) still
                // posts A once.
                const uint32_t owned =
                    plan.owned_per_type_pos[type]
                                           [flat % plan.layers_per_pass];
                const uint64_t own_a = plan.block.passed(
                    plan.member_index, type, ReplicaControlBlock::Barrier::A);
                if (owned == 0 && (own_a == ReplicaControlBlock::kNone ||
                                   own_a < flat)) {
                    plan.block.advance(plan.member_index, type,
                                       ReplicaControlBlock::Barrier::A, flat);
                }
                const uint64_t owner_a = plan.block.passed(
                    static_cast<size_t>(role->owner_member), type,
                    ReplicaControlBlock::Barrier::A);
                const bool owner_ready =
                    owner_a != ReplicaControlBlock::kNone && owner_a >= flat;
                if (plan.block.aborted(
                        static_cast<size_t>(role->owner_member))) {
                    this->fail_producer_(module_type, -1, next.module_name,
                                         false,
                                         "peer member " +
                                             std::to_string(
                                                 role->owner_member) +
                                             " aborted its weight pass");
                    break;
                }
                if (!owner_ready) {
                    auto waiting =
                        plan.wait_since.emplace(
                                std::make_pair(type, flat),
                                std::chrono::steady_clock::now())
                            .first;
                    const auto waited =
                        std::chrono::steady_clock::now() - waiting->second;
                    if (waited >= GPU_Weight_Buffer::kStallTimeout) {
                        this->fail_producer_(
                            module_type, -1, next.module_name, false,
                            "peer member " +
                                std::to_string(role->owner_member) +
                                " has not passed barrier A of frame " +
                                std::to_string(flat) + " for " +
                                std::to_string(
                                    std::chrono::duration_cast<
                                        std::chrono::seconds>(waited)
                                        .count()) +
                                " s (module " + next.module_name + ")");
                        break;
                    }
                    // Not an error and not progress: the loop goes on to
                    // serve the on-demand and KV queues, then backs off.
                    progressed = false;
                    continue;
                }
                plan.wait_since.erase(std::make_pair(type, flat));
            }
            // A pending skip needs no slot: the module is only acquired and
            // released, so the node daemon's FIFO stays aligned across ranks.
            std::optional<std::reference_wrapper<module_weight_tensor_map>>
                buffer;
            int64_t buffer_idx = -1;
            if (next.kind == WeightProduction::kCopy) {
                auto optional_buffer =
                    this->gpu_weight_buffer_.acquireEmptyBuffer(module_type);
                if (!optional_buffer.has_value()) {
                    continue;
                }
                buffer = optional_buffer->first;
                buffer_idx = optional_buffer->second;
            }
            progressed = true;
            std::string module_name;
            // release_module() is owed for module_name: a remote module
            // holds a daemon lease from its fetch until then.
            bool lease_held = false;
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
                if (role != nullptr &&
                    role->kind ==
                        static_cast<uint8_t>(WeightRecordKind::Replica)) {
                    // Pull the owner's landed slot over NVLink. No host
                    // acquisition and no daemon lease on this path.
                    ReplicaPlanRuntime& plan = *this->replica_plan_;
                    const auto claim = this->gpu_weight_buffer_.claim_for_copy(
                        module_type, buffer_idx, module_name);
                    queue.try_pop(module_name);
                    queue.push(module_name);
                    if (claim == WeightProduction::kSkip) {
                        this->gpu_weight_buffer_.retire_skip(module_type,
                                                             module_name);
                        this->count_module_(this->skipped_modules_,
                                            module_type);
                        this->replica_mark_terminal_(role->type_index, flat);
                        this->set_producer_state_(ProducerState::kIdle, "");
                        continue;
                    }
                    this->set_producer_state_(ProducerState::kCopying,
                                              module_name, next.pass);
                    const uint64_t slot_index = plan.block.owned_slot(
                        static_cast<size_t>(role->owner_member), flat,
                        role->record_index);
                    if (slot_index == ReplicaControlBlock::kNone) {
                        throw std::runtime_error(
                            "owner passed barrier A of frame " +
                            std::to_string(flat) + " without publishing " +
                            module_name);
                    }
                    const int64_t stride = plan.stride.at(module_type);
                    char* src = static_cast<char*>(
                                    plan.segment_base.at(module_type)
                                        [role->owner_member]) +
                                static_cast<int64_t>(slot_index) * stride;
                    char* dst = static_cast<char*>(
                                    plan.segment_base.at(module_type)
                                        [plan.member_index]) +
                                buffer_idx * stride;
                    cudaStream_t stream =
                        plan.pull_streams[role->owner_member];
                    CUDA_CHECK(cudaMemcpyAsync(dst, src,
                                               static_cast<size_t>(stride),
                                               cudaMemcpyDeviceToDevice,
                                               stream));
                    CUDA_CHECK(cudaStreamSynchronize(stream));
                    const bool present =
                        this->gpu_weight_buffer_.weights_copy_complete(
                            module_type, module_name, buffer_idx);
                    this->count_module_(this->replica_pulled_modules_,
                                        module_type);
                    if (!present) {
                        this->count_module_(
                            this->released_on_landing_modules_, module_type);
                    }
                    this->replica_mark_terminal_(role->type_index, flat);
                    this->set_producer_state_(ProducerState::kIdle, "");
                    continue;
                }
                this->set_producer_state_(ProducerState::kAcquiring,
                                          module_name, next.pass);
                auto fetched =
                    this->weights_storage_.get_module_weights_storage(
                        module_name, &this->terminate_flag_);
                if (!fetched.has_value()) {
                    // Stop requested before the daemon readied the module:
                    // nothing is leased, and the module stays at the head of
                    // its queue and at its cursor.
                    if (buffer_idx >= 0) {
                        this->gpu_weight_buffer_.abandon(module_type,
                                                         buffer_idx);
                    }
                    this->set_producer_state_(ProducerState::kParked, "");
                    break;
                }
                lease_held = true;
                const auto src = std::move(*fetched);
                const auto claim = this->gpu_weight_buffer_.claim_for_copy(
                    module_type, buffer_idx, module_name);
                // Only this thread pops: the head is still module_name. It
                // goes back to the tail, so the next pass reads the list in
                // the same order.
                queue.try_pop(module_name);
                queue.push(module_name);
                if (claim == WeightProduction::kSkip) {
                    if (role != nullptr &&
                        role->kind == static_cast<uint8_t>(
                                          WeightRecordKind::HostShared)) {
                        // Produce-only (copy for the peers, release without
                        // a consumer) is not wired yet, and no enabled
                        // configuration can reach it: replicated schedules
                        // never skip. Fail loud rather than strand peers.
                        throw std::runtime_error(
                            "routing-informed skip of HostShared module " +
                            module_name +
                            " needs the produce-only path, which this "
                            "transport slice does not implement");
                    }
                    // No copy, but the daemon lease is still released so the
                    // node FIFO advances. The cursor moves past the module
                    // only after that, so its pass cannot end while the lease
                    // is held.
                    lease_held = false;
                    this->weights_storage_.release_module(module_name);
                    this->gpu_weight_buffer_.retire_skip(module_type,
                                                         module_name);
                    this->count_module_(this->skipped_modules_, module_type);
                    if (role != nullptr) {
                        this->replica_mark_terminal_(role->type_index, flat);
                    }
                    this->set_producer_state_(ProducerState::kIdle, "");
                    continue;
                }
                this->set_producer_state_(ProducerState::kCopying, module_name,
                                          next.pass);
                auto dst = buffer->get();
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
                    auto copy_start = std::chrono::steady_clock::now();
                    this->blocking_copy_(slot->second.data_ptr(), src_ptr,
                                         src_byte_size);
                    auto copy_end = std::chrono::steady_clock::now();
                    {
                        std::lock_guard<std::mutex> lock(
                            this->weight_profile_mutex_);
                        if (this->weight_profile_enabled_) {
                            this->weight_profile_tensors_[module_type] += 1;
                            this->weight_profile_bytes_[module_type] +=
                                static_cast<uint64_t>(src_byte_size);
                            this->weight_profile_copy_seconds_[module_type] +=
                                std::chrono::duration<double>(copy_end -
                                                              copy_start)
                                    .count();
                        }
                    }
                }
                // Released before the landing: once no slot of the pass is
                // FILLING, the producer holds no lease of it either.
                lease_held = false;
                this->weights_storage_.release_module(module_name);
                {
                    std::lock_guard<std::mutex> lock(
                        this->weight_profile_mutex_);
                    if (this->weight_profile_enabled_) {
                        this->weight_profile_modules_[module_type] += 1;
                    }
                }
                this->logger_->debug("Copied module: {} to buffer: {}",
                                     module_name, buffer_idx);
                const bool present =
                    this->gpu_weight_buffer_.weights_copy_complete(
                        module_type, module_name, buffer_idx);
                this->count_module_(this->copied_modules_, module_type);
                if (!present) {
                    this->count_module_(this->released_on_landing_modules_,
                                        module_type);
                }
                if (role != nullptr &&
                    role->kind == static_cast<uint8_t>(
                                      WeightRecordKind::HostShared)) {
                    if (!present) {
                        throw std::runtime_error(
                            "HostShared module " + module_name +
                            " was released on landing; the produce-only "
                            "path is not implemented in this slice");
                    }
                    ReplicaPlanRuntime& plan = *this->replica_plan_;
                    this->gpu_weight_buffer_.arm_peer_release_gate(
                        module_type, buffer_idx, static_cast<int64_t>(flat));
                    plan.block.publish_slot(plan.member_index, flat,
                                            role->record_index,
                                            static_cast<uint64_t>(
                                                buffer_idx));
                    this->replica_mark_owned_landed_(role->type_index, flat);
                }
                if (role != nullptr) {
                    this->replica_mark_terminal_(role->type_index, flat);
                }
                this->set_producer_state_(ProducerState::kIdle, "");
            } catch (const std::exception& error) {
                // Never rethrow: an exception leaving this thread is
                // std::terminate. Waiters see the failure instead.
                this->fail_producer_(module_type, buffer_idx, module_name,
                                     lease_held, error.what());
                break;
            } catch (...) {
                this->fail_producer_(module_type, buffer_idx, module_name,
                                     lease_held, "unknown exception");
                break;
            }
        }
        if (!progressed) {
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
            // A full ring frees only when the consumer's release events
            // complete, which nothing signals. Without a sleep the iteration
            // re-locks the weight-buffer mutex in a tight loop and starves
            // the consumer's get_weights. 50 us is negligible against a
            // slot's copy time.
            std::this_thread::sleep_for(std::chrono::microseconds(50));
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
        this->replica_pulled_modules_.clear();
        if (this->producer_state_ == static_cast<int>(ProducerState::kFailed)) {
            this->producer_state_ = static_cast<int>(
                this->HtoD_worker_.joinable() ? ProducerState::kIdle
                                              : ProducerState::kParked);
            this->module_in_flight_.clear();
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
    {
        std::lock_guard<std::mutex> status_lock(this->status_mutex_);
        this->module_in_flight_.clear();
        this->pass_in_flight_ = 0;
    }
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
