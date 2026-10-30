// clang-format off
/* ----------------------------------------------------------------------------  *
 * BatchGen                                                                      *
 * Copyright (c) 2025-2026 BatchGen Team                                            *
 * *
 * licensed under the apache license, version 2.0 (the "license");              *
 * you may not use this file except in compliance with the license.             *
 * *
 * you may obtain a copy of the license at                                      *
 * *
 * http://www.apache.org/licenses/license-2.0                   *
 * *
 * unless required by applicable law or agreed to in writing, software          *
 * distributed under the license is distributed on an "as is" basis,            *
 * without warranties or conditions of any kind, either express or implied.     *
 * see the license for the specific language governing permissions and          *
 * limitations under the license.                                               *
 * ---------------------------------------------------------------------------- */
// clang-format on

#pragma once

#include "spdlog/spdlog.h"
#include <array>
#include <atomic>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <torch/extension.h>
#include <torch/torch.h>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../Parameter_Server/Parameter_Server.h"
#include "../data_structures.h"
#include "../utils.h"
#include "distributed_weights_protocol.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

struct tensor_buffer {
    void* data_ptr;
    std::vector<int64_t> tensor_shape;
    int64_t byte_size;
    std::string dtype;  // Per-tensor dtype: "bfloat16", "uint8", "float8_e4m3fn", etc.

    // default constructor
    tensor_buffer() : data_ptr(nullptr), tensor_shape({}), byte_size(0), dtype("") {};
    // constructor
    tensor_buffer(void* data_ptr, std::vector<int64_t> tensor_shape,
                  int64_t byte_size, std::string dtype = "")
        : data_ptr(data_ptr),
          tensor_shape(tensor_shape),
          byte_size(byte_size),
          dtype(dtype) {};
};

// The outcome of Weights_Storage::get_module_weights_storage. tensors is
// filled only when kReady; kNotReady and kCancelled lease nothing.
struct module_fetch {
    enum class Status {
        kReady,
        kNotReady,   // without wait: the node daemon has not readied it yet
        kCancelled,  // with wait: stop was requested before it was ready
    };
    Status status = Status::kCancelled;
    std::unordered_map<std::string, tensor_buffer> tensors;
};

// The outcome of Weights_Storage::poll_acquire. tensors is filled only when
// kReady, and then release_module() is owed for the module.
struct acquire_poll {
    enum class Status {
        kPending,    // no whole reply yet
        kRetrySent,  // its slice ended first; the same request was re-sent
        kReady,      // leased
    };
    Status status = Status::kPending;
    std::unordered_map<std::string, tensor_buffer> tensors;
};

// The outcome of Weights_Storage::drain_acquire. leased: tensors is filled
// and release_module() is owed. error: the reply could not be read; nothing
// is outstanding any more.
struct acquire_drain {
    bool leased = false;
    std::unordered_map<std::string, tensor_buffer> tensors;
    std::string error;
};

struct distributed_tensor_meta {
    std::vector<int64_t> tensor_shape;
    int64_t byte_size;
    std::string dtype;
    uint64_t compact_offset;
    uint64_t module_offset;
};

class Weights_Storage {
   public:
    // Simplified Constructor: takes device_id directly
    Weights_Storage(int device_id);
    
    ~Weights_Storage();
    
    // memfd_creator_pid owns both the weights and tensor-metadata memfds in
    // every mode, hugetlbfs included, so both fds are always used.
    void Init(std::string& shm_name, int64_t byte_size,
                std::string& tensor_meta_shm_name, bool enable_hugetlbfs,
                bool enable_thp = false, int memfd_creator_pid = -1,
                int memfd_fd = -1, int tensor_meta_memfd_fd = -1);

    // cohort_epoch is the worker cohort the server spawned this worker
    // into; the daemon refuses a hello that names any other cohort.
    void InitDistributed(const std::string& config_path,
                         uint64_t cohort_epoch);
                  
    // A remote module is acquired from the node daemon. With wait, the daemon
    // answers in bounded slices and stop_requested (may be null) is polled
    // between them: kCancelled once stop was requested first. Without wait,
    // one request that the daemon answers at once: kNotReady when the module
    // is not ready yet. A local module is always kReady.
    module_fetch get_module_weights_storage(
        std::string module_key,
        const std::atomic<bool>* stop_requested = nullptr, bool wait = true);

    // Whether module_key is acquired from the node daemon, not read from
    // this node's registered store.
    bool is_remote_module(const std::string& module_key) const;

    // Two-phase acquire of a remote module on the primary daemon connection,
    // which carries one request at a time: from begin_acquire until its
    // lease arrives, or drain_acquire, no other acquire is sent. begin_acquire
    // sends one waiting acquire request.
    void begin_acquire(const std::string& module_key);
    // Reads the outstanding acquire's reply, completing it across partial
    // reads. Never blocks unless block: then it waits for one whole reply,
    // at most one daemon slice. On kStatusRetry the same request is re-sent.
    acquire_poll poll_acquire(bool block = false);
    // Ends the outstanding acquire without re-sending: waits for its reply,
    // at most one daemon slice. A lease in it is returned, never released
    // here. A no-op when none is outstanding. Never throws.
    acquire_drain drain_acquire() noexcept;

    // Returns Python Dictionary for Pybind11
    py::dict get_tensor(std::string module_key);

    // On the release-only daemon connection, never behind an acquire.
    void release_module(const std::string& module_key);

    // Modules the node daemon currently leases to this worker. Never waits
    // on a daemon request in flight.
    std::vector<std::string> leased_modules() const;

    // Whether remote modules come from the node daemon, whose schedule every
    // rank of the node consumes in the same cyclic order.
    bool is_distributed() const { return this->distributed_; }

   private:
    int device_id_; // Stored device ID
    std::shared_ptr<spdlog::logger> logger;

    std::string shm_name;
    // Only the ordinary Init() path owns this mapping; the distributed path
    // leaves it null so the destructor cannot release a mapping it never made.
    void* weight_ptr_ = nullptr;
    int64_t byte_size_ = 0;
    // Bytes actually mapped by allocate_shared_pinned_memory, which rounds up
    // to a huge page. munmap on a huge page mapping rejects a non-aligned
    // length, so the release must use this, not byte_size_.
    int64_t mapped_size_ = 0;

    /* "attn_0" -> "o_proj" -> ptr */
    std::unordered_map<std::string,
                       std::unordered_map<std::string, tensor_buffer>>
        module_weights_storage_;

    struct active_lease {
        int slot = -1;
        uint64_t generation = 0;
    };

    bool distributed_ = false;
    bool hierarchical_gdr_ = false;
    int local_node_rank_ = -1;
    // hierarchical_gdr pins only the routed experts this worker streams:
    // TP slot device_id_ owns experts [pin_expert_begin_, pin_expert_end_),
    // and only the node that sources that slot registers them.
    bool pin_source_ = false;
    int pin_expert_begin_ = -1;
    int pin_expert_end_ = -1;
    // The resident-EP decode shard of this worker (EP rank
    // local_node_rank_ * 8 + device_id_) is always local to this node's
    // store; pin it too, so the one-time shard materialization after the
    // first streamed prefill is a pinned H2D copy on every rank (not just
    // the one whose slot range happens to cover it).
    int resident_expert_begin_ = -1;
    int resident_expert_end_ = -1;
    // Every cudaHostRegister that actually succeeded, so the destructor can
    // unregister exactly those ranges before munmap, including after a
    // partially completed InitDistributed.
    std::vector<std::pair<void*, size_t>> registered_ranges_;
    int compact_fd_ = -1;
    int staging_fd_ = -1;
    int daemon_socket_ = -1;   // primary: hello and acquires
    int release_socket_ = -1;  // release-only (kHelloReleaseChannel)
    void* compact_ptr_ = nullptr;
    int64_t compact_bytes_ = 0;
    void* staging_ptr_ = nullptr;
    int64_t staging_bytes_ = 0;
    int64_t distributed_module_bytes_ = 0;
    // Guards daemon_socket_ and the acquire outstanding on it.
    std::mutex daemon_mutex_;
    // Guards release_socket_.
    std::mutex release_mutex_;
    std::unordered_map<std::string,
                       std::unordered_map<std::string,
                                          distributed_tensor_meta>>
        remote_module_weights_;
    // Every access, read or write, holds lease_index_mutex_: acquires write
    // it under daemon_mutex_, releases under release_mutex_.
    std::unordered_map<std::string, active_lease> active_leases_;
    mutable std::mutex lease_index_mutex_;

    // The acquire outstanding on daemon_socket_, under daemon_mutex_. An
    // empty key: none. Response is fixed-size POD, so its reply is collected
    // in acquire_reply_ across partial reads.
    std::string acquire_key_;
    batchgen::distributed_weights::Request acquire_request_{};
    std::array<char, sizeof(batchgen::distributed_weights::Response)>
        acquire_reply_{};
    size_t acquire_reply_bytes_ = 0;

    // What one whole reply of the outstanding acquire was.
    enum class AcquireReply { kPending, kRetry, kLeased };

    // std::nullopt leases nothing: with wait, stop was requested before the
    // module was ready; without, the daemon's one answer was not ready.
    // Built on begin_acquire_locked_ and receive_acquire_reply_locked_.
    std::optional<active_lease> acquire_remote_module(
        const std::string& module_key,
        const std::atomic<bool>* stop_requested, bool wait);
    // Under daemon_mutex_. Throws while an acquire is outstanding.
    void begin_acquire_locked_(const std::string& module_key, bool wait);
    // Under daemon_mutex_. kRetry leaves the acquire outstanding until it is
    // re-sent or abandoned; kLeased records the lease and ends it. A throw
    // ends it too.
    AcquireReply receive_acquire_reply_locked_(bool block,
                                               active_lease* lease);
    // Under daemon_mutex_: re-sends the outstanding acquire after a retry.
    void resend_acquire_locked_();
    // The staging views of a leased remote module.
    std::unordered_map<std::string, tensor_buffer> staging_views_(
        const std::string& module_key, const active_lease& lease) const;
};
