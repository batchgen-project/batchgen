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
#include <cstdint>
#include <deque>
#include <future>
#include <memory>
#include <string>
#include <thread>
#include <torch/extension.h>
#include <torch/torch.h>
#include <unordered_map>
#include <vector>

#include "../GPU_KV_Buffer/GPU_KV_Buffer.h"
#include "../GPU_Weight_Buffer/GPU_Weight_Buffer.h"
#include "../KV_Storage/KV_Storage.h"
#include "../Weights_Storage/Weights_Storage.h"
#include "../Weights_Storage/distributed_weights_protocol.h"
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
    void set_global_routed_experts_data_ptr(
        const py::dict& experts_IPC_handles,
        const py::dict& expert_location_map);
    void cuda_enable_peer_access(int rank, int world_size);
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
    // Un-parks the producer: a pass was armed, a task was queued or a stop
    // was requested. Also ends its wait for a free weight slot.
    void wake_producer();
    // Throws when the node daemon still leases this worker a module of pass
    // or an earlier one. A lease of a module in flight for a later (carried)
    // pass is allowed.
    void require_no_daemon_lease_for_pass(int64_t pass);
    // Producer state and its oldest module and pass in flight, for error
    // messages.
    std::string describe_producer();
    void batched_page_copy(const std::vector<void*>& gpu_ptrs,
                           const std::vector<void*>& host_ptrs,
                           int64_t page_byte_size);


   private:
    // idle: running, nothing in flight. acquiring: fetching a module's host
    // storage with nothing in flight (a remote module waits for the node
    // daemon). copying: copies are in flight; a remote module is acquired
    // ahead of them. parked: the thread is not running. failed: a copy
    // failed; nothing is copied until set_weight_copy_queue.
    enum class ProducerState : int {
        kIdle = 0,
        kAcquiring = 1,
        kCopying = 2,
        kParked = 3,
        kFailed = 4,
    };

    // A module from the moment the producer names it, before its fetch,
    // until its copy landed or it was retired as a skip, after its daemon
    // lease was released. Its copies are queued on HtoD_stream; event is
    // recorded after the last of them. The producer only queries or waits on
    // an event while copies are in flight, which needs an armed pass: a CUDA
    // graph is captured with none armed, while the producer is parked.
    struct InFlightCopy {
        std::string module_type;
        std::string module_name;
        int64_t buffer_idx = -1;  // -1: a skip, which holds no slot
        // The reserved slot's tensors; null for a skip.
        module_weight_tensor_map* buffer = nullptr;
        int64_t pass = 0;
        bool leased = false;  // release_module() is still owed
        // Its waiting daemon acquire is outstanding: nothing is leased or
        // copied yet. At most one entry.
        bool acquiring = false;
        // event was recorded after its last copy. A pooled event keeps the
        // state of its last use, so only a recorded entry can land.
        bool recorded = false;
        cudaEvent_t event = nullptr;
    };
    // Enough to keep the copy engine busy behind the producer's per-module
    // host work; more only holds more slots and leases.
    static constexpr size_t kMaxCopiesInFlight = 4;
    static constexpr size_t kCopyEvents = kMaxCopiesInFlight + 1;
    // While its one acquire waits, the producer leases at most every other
    // entry; the node daemon sizes its staging ring for that bound.
    static_assert(
        batchgen::distributed_weights::kMaxLeasesWhileAcquiring ==
            kMaxCopiesInFlight - 1,
        "the daemon's ring bound must match the producer's copy pipeline");

    void set_producer_state_(ProducerState state);
    // copying while a copy is in flight, else idle.
    void set_copying_or_idle_();
    std::string describe_producer_locked_() const;  // under status_mutex_
    void count_module_(
        std::unordered_map<std::string, uint64_t>& counter,
        const std::string& module_type);
    // Producer thread only. Pushes or pops in_flight_ under status_mutex_
    // and takes or returns the entry's event.
    void begin_in_flight_(const std::string& module_type,
                          const std::string& module_name, int64_t buffer_idx,
                          module_weight_tensor_map* buffer, int64_t pass);
    void end_in_flight_(bool oldest);
    void end_in_flight_at_(size_t index);
    cudaEvent_t take_copy_event_();
    // Lands the copies whose event completed, oldest first, and stops at
    // the first that has not. The entry of an outstanding acquire has no
    // copy recorded and is stepped over: every entry behind it is another
    // type's local module. wait: first waits for the oldest recorded copy,
    // so at least one lands. Returns how many landed.
    size_t land_copies_(bool wait);
    // (module_type, buffer_idx): a slot reserved for module_name that may
    // not be in flight yet (-1: none).
    void fail_producer_(const std::string& module_type, int64_t buffer_idx,
                        const std::string& module_name,
                        const std::string& message);
    // The index of the entry whose acquire is outstanding, else
    // in_flight_.size().
    size_t acquiring_index_() const;
    // The claim path of in_flight_[index], whose host storage is in hand:
    // claims it at its type's cursor and moves its queue head to the tail,
    // then queues its copies behind its event or, for a pending skip,
    // releases its lease and retires it.
    void claim_module_(
        size_t index,
        const std::unordered_map<std::string, tensor_buffer>& src);
    // Polls the outstanding acquire's reply, if any; block waits for one
    // whole reply, at most one daemon slice. A lease runs the claim path.
    // Returns whether it did.
    bool poll_acquire_(bool block);
    // On stop: drains the outstanding acquire's reply. A lease runs the
    // claim path; a retry drops the entry and leaves its module at its
    // cursor and the head of its queue.
    void drain_acquire_on_stop_();
    // From fail_producer_: drains the outstanding acquire's reply and
    // claims a lease in it, so its release below moves past the module.
    // Never throws; what fails is appended to failure.
    void drain_acquire_on_failure_(std::string* failure) noexcept;

    /* Template */
    std::atomic<bool> terminate_flag_{false};
    const EngineConfig& engine_config_;
    const ModelConfig& model_config_;
    cudaStream_t HtoD_stream;
    std::shared_ptr<spdlog::logger> logger_;
    std::mutex mutex_;
    std::condition_variable cv_;
    py::module p2p_tensor_copy_module;
    std::unordered_map<std::string, std::unordered_map<std::string, void*>>
                       global_device_experts_ptrs_;
    std::unordered_map<std::string, int64_t> expert_location_map_;

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
    // Oldest first. Written by the producer under status_mutex_, read by it
    // without; empty whenever the producer is not running (a stop lands
    // every copy, a failure abandons them).
    std::deque<InFlightCopy> in_flight_;
    // Created lazily on the engine device, destroyed in ~HtoD_Engine.
    // Producer thread only.
    std::vector<cudaEvent_t> copy_events_;
    std::vector<cudaEvent_t> free_copy_events_;
    // Per module type since the last set_weight_copy_queue, under
    // status_mutex_.
    std::unordered_map<std::string, uint64_t> copied_modules_;
    std::unordered_map<std::string, uint64_t> skipped_modules_;
    std::unordered_map<std::string, uint64_t> released_on_landing_modules_;

    std::mutex weight_profile_mutex_;
    bool weight_profile_enabled_{false};
    std::unordered_map<std::string, uint64_t> weight_profile_modules_;
    std::unordered_map<std::string, uint64_t> weight_profile_tensors_;
    std::unordered_map<std::string, uint64_t> weight_profile_bytes_;
    std::unordered_map<std::string, double> weight_profile_copy_seconds_;

    std::thread HtoD_worker_;
    void HtoD_Worker();
    void blocking_copy_(void* dst, void* src, int64_t size);
};
