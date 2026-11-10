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
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "spdlog/spdlog.h"
#include <atomic>
#include <chrono>
#include <cstdint>
#include <future>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <torch/extension.h>
#include <torch/torch.h>
#include <unordered_map>
#include <vector>

#include "../GPU_KV_Buffer/GPU_KV_Buffer.h"
#include "../GPU_Weight_Buffer/GPU_Weight_Buffer.h"
#include "../GPU_Weight_Buffer/replica_control_block.h"
#include "../GPU_Weight_Buffer/weight_replica_plan.h"
#include "../KV_Storage/KV_Storage.h"
#include "../Weights_Storage/Weights_Storage.h"
#include "../data_structures.h"
#include "../threadsafe_queue.h"
#include "../utils.h"

namespace py = pybind11;
class HtoD_Engine {
   public:
    struct WeightPassAbort {
        WeightSlotTable::AbortCounts slots;
        int64_t daemon_leases = 0;  // released
        bool schedule_lost = false;
    };

    HtoD_Engine(const EngineConfig& engine_config,
                const ModelConfig& model_config,
                Weights_Storage& weights_storage, KV_Storage& kv_storage,
                GPU_Weight_Buffer& gpu_weight_buffer,
                GPU_KV_Buffer& gpu_kv_buffer);
    ~HtoD_Engine();
    // void Init(const py::dict& py_weight_copy_tasks);
    // void Init(std::unordered_map<std::string, std::vector<std::string>>&
    //               weights_copy_tasks);
    void Init();

    void Start();
    void Terminate();

    torch::Tensor tensor_on_demand_copy(torch::Tensor src_tensor);
    void submit_to_KV_queue(std::vector<int64_t>& micro_batch,
                            int64_t micro_batch_idx, int64_t layer_idx,
                            int64_t byte_size);
    void clear_kv_copy_queue();
    // Installs the tasks in the weight buffer and the producer queues and
    // restarts pass numbering. Throws while the producer runs, a pass is open
    // or a slot is held.
    void set_weight_copy_queue(
        std::unordered_map<std::string, std::vector<std::string>>&
            weight_copy_tasks);
    void reset_weight_stream_profile(bool enabled);
    py::dict get_weight_stream_profile();
    py::dict weight_stream_status();
    // Joins the producer, then rethrows its recorded failure, if any.
    void stop_h2d_worker();
    // After a forward died inside its weight passes; the producer must be
    // stopped. Releases every daemon lease still recorded, drops every open
    // pass and slot (GPU_Weight_Buffer::abort_passes) and puts every queue
    // back at the head of its list, where the cursors now are. With
    // distributed storage the daemon schedule is lost: every later
    // begin_weight_pass and set_weight_copy_queue throws. Otherwise the next
    // set_weight_copy_queue, or start + begin, proceeds as after a phase end.
    WeightPassAbort abort_weight_passes();
    // Un-parks the producer: a pass was armed or a task was queued.
    void wake_producer();

    /* Node replica transport (spec v4 slice 2). Installed per prefill phase
       by the worker when the replica group has more than one member; the
       uninstalled engine behaves as the single-member group (every record
       HostPrivate). Install and uninstall run only while the producer is
       stopped. */
    struct ReplicaModuleInstall {
        std::string module_type;
        std::string module;
        uint8_t kind = 1;          // WeightRecordKind: 0 shared 1 private 2 replica
        int32_t owner_member = 0;  // index into the ordered group
        uint32_t layer_position = 0;
        uint32_t record_index = 0;  // within one frame, identical on every member
    };
    void install_replica_plan(
        int64_t member_index, int64_t group_size, uint64_t layers_per_pass,
        uint32_t records_per_frame,
        const std::vector<uint32_t>& owned_per_position,
        const std::vector<ReplicaModuleInstall>& modules,
        torch::Tensor control_block,
        const std::unordered_map<std::string,
                                 std::vector<torch::Tensor>>& peer_segments,
        const std::unordered_map<std::string, int64_t>& strides);
    // Syncs and destroys the pull streams, drops the imported segments and
    // the control-block view, clears the reclaim gate. The caller has run
    // the group barrier first.
    void uninstall_replica_plan();
    bool replica_plan_installed() const { return this->replica_plan_ != nullptr; }
    // Throws when the node daemon still leases this worker a module of pass
    // or an earlier one. A lease of the module in flight for a later
    // (carried) pass is allowed.
    void require_no_daemon_lease_for_pass(int64_t pass);
    // Producer state, module and pass in flight, for error messages.
    std::string describe_producer();
    void batched_page_copy(const std::vector<void*>& gpu_ptrs,
                           const std::vector<void*>& host_ptrs,
                           int64_t page_byte_size);


   private:
    // idle: running, nothing in flight. acquiring: fetching a module's host
    // storage (a remote module waits for the node daemon). copying: copying
    // it into its slot. parked: the thread is not running. failed: a copy
    // failed; nothing is copied until set_weight_copy_queue.
    enum class ProducerState : int {
        kIdle = 0,
        kAcquiring = 1,
        kCopying = 2,
        kParked = 3,
        kFailed = 4,
    };
    void set_producer_state_(ProducerState state,
                             const std::string& module_name,
                             int64_t pass = 0);
    std::string describe_producer_locked_() const;  // under status_mutex_
    void count_module_(
        std::unordered_map<std::string, uint64_t>& counter,
        const std::string& module_type);
    void fail_producer_(const std::string& module_type, int64_t buffer_idx,
                        const std::string& module_name, bool leased,
                        const std::string& message);

    /* Template */
    std::atomic<bool> terminate_flag_{false};
    const EngineConfig& engine_config_;
    const ModelConfig& model_config_;
    cudaStream_t HtoD_stream;
    std::shared_ptr<spdlog::logger> logger_;
    std::mutex mutex_;
    std::condition_variable cv_;
    py::module p2p_tensor_copy_module;

    /* Need to query kv storage and weights storage. */
    GPU_Weight_Buffer& gpu_weight_buffer_;
    // GPU_KV_Buffer& gpu_kv_buffer_;
    Weights_Storage& weights_storage_;
    KV_Storage& kv_storage_;
    GPU_KV_Buffer& gpu_kv_buffer_;

    /* Task Queues. */
    threadsafe_queue<std::packaged_task<void()>> on_demand_task_queue_;
    threadsafe_queue<
        std::tuple<std::vector<int64_t>, int64_t, int64_t, int64_t>>
        kv_copy_task_queue_;  // micro_batch, micro_batch_idx, layer_idx,
                              // byte_size.

    std::unordered_map<std::string, std::vector<std::string>>
        weight_copy_tasks_;
    // module type -> queue of module names, in schedule order. Replaced only
    // while the producer is stopped; the producer pops its head and pushes it
    // back to the tail once per claim, so every pass reads the same order.
    std::unordered_map<std::string, threadsafe_queue<std::string>>
        weights_copy_task_queue_;
    // Set by a failed copy, cleared only by set_weight_copy_queue.
    std::atomic<bool> producer_failed_{false};

    // A producer with no type to produce blocks on producer_wake_cv_ until
    // producer_wake_ moves (wake_producer) or it is told to stop.
    std::mutex producer_wake_mutex_;
    std::condition_variable producer_wake_cv_;
    uint64_t producer_wake_ = 0;  // under producer_wake_mutex_

    std::atomic<int> producer_state_{static_cast<int>(ProducerState::kParked)};
    std::mutex status_mutex_;
    std::string module_in_flight_;  // under status_mutex_
    int64_t pass_in_flight_ = 0;    // under status_mutex_
    // Per module type since the last set_weight_copy_queue, under
    // status_mutex_.
    std::unordered_map<std::string, uint64_t> copied_modules_;
    std::unordered_map<std::string, uint64_t> skipped_modules_;
    std::unordered_map<std::string, uint64_t> released_on_landing_modules_;
    std::unordered_map<std::string, uint64_t> replica_pulled_modules_;

    std::mutex weight_profile_mutex_;
    bool weight_profile_enabled_{false};
    std::unordered_map<std::string, uint64_t> weight_profile_modules_;
    std::unordered_map<std::string, uint64_t> weight_profile_tensors_;
    std::unordered_map<std::string, uint64_t> weight_profile_bytes_;
    std::unordered_map<std::string, double> weight_profile_copy_seconds_;

    std::thread HtoD_worker_;
    void HtoD_Worker();
    void blocking_copy_(void* dst, void* src, int64_t size);

    /* Replica transport runtime. Written only while the producer is stopped;
       read by the producer thread. */
    struct ReplicaPlanRuntime {
        int64_t member_index = 0;
        int64_t group_size = 1;
        uint64_t layers_per_pass = 0;
        uint32_t records_per_frame = 0;
        std::vector<uint32_t> owned_per_position;
        // Installed records per layer position; a frame of a partly
        // covered pass (GLM dense-prefix layers) has fewer than
        // records_per_frame live records.
        std::vector<uint32_t> total_per_position;
        std::unordered_map<std::string, ReplicaModuleInstall> modules;
        ReplicaControlBlock block;
        torch::Tensor block_backing;  // keep-alive of the shared mapping view
        // per type: per member, base pointer of their single ring segment
        std::unordered_map<std::string, std::vector<void*>> segment_base;
        std::unordered_map<std::string, int64_t> stride;
        std::vector<torch::Tensor> keepalive;  // imported segment tensors
        std::vector<cudaStream_t> pull_streams;  // one per member
        // frame bookkeeping, producer thread only
        std::map<uint64_t, uint32_t> owned_landed;
        std::map<uint64_t, uint32_t> terminal;
        std::map<uint64_t, std::chrono::steady_clock::time_point> wait_since;
        ReplicaPlanRuntime(ReplicaControlBlock attached)
            : block(attached) {}
    };
    std::unique_ptr<ReplicaPlanRuntime> replica_plan_;
    // Producer-side frame hooks; no-ops without an installed plan.
    void replica_mark_terminal_(uint64_t flat_frame);
    void replica_mark_owned_landed_(uint64_t flat_frame);
};
