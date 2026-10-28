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

#include <chrono>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <torch/extension.h>
#include <torch/torch.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <condition_variable>
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

#include "../data_structures.h"
#include "../utils.h"
#include "GPU_Weight_Buffer.h"

using module_weight_tensor_map = std::unordered_map<std::string, torch::Tensor>;
using weight_buffers = std::vector<module_weight_tensor_map>;

namespace {
// Helper function to determine the dtype for a specific tensor.
// Priority: 1) tensor_dtypes[module_type][buffer_name]
//           2) weight_dtypes[module_type]
//           3) basic_config.weight_dtype_torch
torch::Dtype get_tensor_dtype(
    const GPU_Buffer_Config& gpu_config,
    const Basic_Config& basic_config,
    const std::string& module_type,
    const std::string& buffer_name) {

    // 1. Check per-tensor dtype override
    auto module_it = gpu_config.tensor_dtypes.find(module_type);
    if (module_it != gpu_config.tensor_dtypes.end()) {
        auto tensor_it = module_it->second.find(buffer_name);
        if (tensor_it != module_it->second.end()) {
            return tensor_it->second;
        }
    }

    // 2. Check per-module dtype
    auto dtype_it = gpu_config.weight_dtypes.find(module_type);
    if (dtype_it != gpu_config.weight_dtypes.end()) {
        return dtype_it->second;
    }

    // 3. Fall back to global weight dtype
    return basic_config.weight_dtype_torch;
}
}  // namespace

GPU_Weight_Buffer::GPU_Weight_Buffer(EngineConfig& engine_config,
                                     ModelConfig& model_config)
    : engine_config_(engine_config), model_config_(model_config) {
    this->logger_ = init_logger(
        this->engine_config_.basic_config.log_level,
        "GPU_Weight_Buffer" +
            std::to_string(this->engine_config_.basic_config.device));
    this->logger_->info("GPU_Weight_Buffer Instantiated.");
};

GPU_Weight_Buffer::~GPU_Weight_Buffer() {
    cudaSetDevice(this->engine_config_.basic_config.device);
    std::lock_guard<std::mutex> lock(this->mutex_);
    for (auto& pending : this->pending_releases_) {
        cudaEventSynchronize(pending.event->get());
    }
    this->pending_releases_.clear();
}

void GPU_Weight_Buffer::synchronizePendingReleasesLocked() {
    for (auto& pending : this->pending_releases_) {
        CUDA_CHECK(cudaEventSynchronize(pending.event->get()));
    }
    this->pending_releases_.clear();
}

void GPU_Weight_Buffer::drainPendingReleasesLocked() {
    while (!this->pending_releases_.empty()) {
        PendingRelease& pending = this->pending_releases_.front();
        CUDA_CHECK(cudaEventSynchronize(pending.event->get()));
        for (const auto& [module_type, buffer_idx] : pending.slots) {
            this->slots_.reclaim(module_type, buffer_idx);
        }
        this->pending_releases_.pop_front();
        this->notifySlotsLocked();
    }
}

void GPU_Weight_Buffer::bumpProgressLocked() {
    ++this->progress_epoch_;
    this->cv_.notify_all();
}

void GPU_Weight_Buffer::notifySlotsLocked() {
    ++this->slot_epoch_;
    this->slot_cv_.notify_all();
}

void GPU_Weight_Buffer::waitWithoutStallLocked(
    std::unique_lock<std::mutex>& lock, const std::string& waiting_for,
    const std::function<bool()>& ready) {
    uint64_t seen = this->progress_epoch_;
    auto last_progress = std::chrono::steady_clock::now();
    while (true) {
        this->throwIfProducerFailedLocked();
        if (ready()) {
            return;
        }
        const auto deadline = last_progress + kStallTimeout;
        if (std::chrono::steady_clock::now() >= deadline) {
            const std::string message =
                "Weight stream stalled waiting for " + waiting_for +
                ": no slot landed, no skip was retired and the producer did "
                "not change state for " +
                std::to_string(kStallTimeout.count()) + " s (" +
                this->slots_.describe_passes() +
                "). Held: " + this->slots_.describe_held();
            this->logger_->error("{}", message);
            throw std::runtime_error(message);
        }
        this->cv_.wait_until(lock, deadline, [this, seen, &ready] {
            return this->progress_epoch_ != seen ||
                   this->producer_failure_.has_value() || ready();
        });
        if (this->progress_epoch_ != seen) {
            seen = this->progress_epoch_;
            last_progress = std::chrono::steady_clock::now();
        }
    }
}

void GPU_Weight_Buffer::reclaimCompletedReleasesLocked() {
    for (auto it = this->pending_releases_.begin();
         it != this->pending_releases_.end();) {
        const cudaError_t status = cudaEventQuery(it->event->get());
        if (status == cudaErrorNotReady) {
            ++it;
            continue;
        }
        CUDA_CHECK(status);
        for (const auto& [module_type, buffer_idx] : it->slots) {
            this->slots_.reclaim(module_type, buffer_idx);
        }
        it = this->pending_releases_.erase(it);
        this->notifySlotsLocked();
    }
}

void GPU_Weight_Buffer::reset_weight_stream_profile(bool enabled) {
    std::lock_guard<std::mutex> lock(this->weight_profile_mutex_);
    this->weight_profile_enabled_ = enabled;
    this->weight_profile_requests_.clear();
    this->weight_profile_wait_seconds_.clear();
}

pybind11::dict GPU_Weight_Buffer::get_weight_stream_profile() {
    std::lock_guard<std::mutex> lock(this->weight_profile_mutex_);
    pybind11::dict result;
    result["enabled"] = this->weight_profile_enabled_;
    pybind11::dict by_type;
    for (const auto& [module_type, requests] :
         this->weight_profile_requests_) {
        pybind11::dict entry;
        entry["requests"] = requests;
        entry["wait_s"] = this->weight_profile_wait_seconds_[module_type];
        by_type[module_type.c_str()] = entry;
    }
    result["by_type"] = by_type;
    return result;
}

void GPU_Weight_Buffer::Init() {
    auto& buffer_shapes = this->engine_config_.gpu_buffer_config.module_shapes;
    auto& num_buffers =
        this->engine_config_.gpu_buffer_config
            .num_prefill_module_buffer;  // TODO: {"attn": 1, "routed_expert":
                                         // 160, "shared_expert": 1}
    for (auto& [module_type, num_buffer] : num_buffers) {
        this->buffers_[module_type].clear();
        this->buffers_[module_type].resize(num_buffer);
        {
            std::lock_guard<std::mutex> lock(this->mutex_);
            this->slots_.reset_type(module_type, num_buffer);
            this->notifySlotsLocked();
        }
        for (auto& [buffer_name, buffer_shape] : buffer_shapes[module_type]) {
            // Get per-tensor dtype using the helper function
            torch::Dtype tensor_dtype = get_tensor_dtype(
                this->engine_config_.gpu_buffer_config,
                this->engine_config_.basic_config,
                module_type,
                buffer_name);
            auto options =
                torch::TensorOptions()
                    .dtype(tensor_dtype)
                    .device(torch::kCUDA, this->engine_config_.basic_config.device)
                    .requires_grad(false)
                    .memory_format(torch::MemoryFormat::Contiguous);
            for (int64_t buffer_idx = 0; buffer_idx < num_buffer;
                 buffer_idx++) {
                this->buffers_[module_type][buffer_idx][buffer_name] =
                    torch::zeros(buffer_shape, options);
            }
        }
    }
};

void GPU_Weight_Buffer::resize_buffer() {
    auto& buffer_shapes = this->engine_config_.gpu_buffer_config.module_shapes;
    auto& num_buffers =
        this->engine_config_.gpu_buffer_config
            .num_decoding_module_buffer;  // {"attn": 1, "routed_expert": 160,
                                          // "shared_expert": 1}
    {
        std::lock_guard<std::mutex> lock(this->mutex_);
        int64_t to_add_buffer = num_buffers["routed_expert"] -
                                this->buffers_["routed_expert"].size();
        for (int64_t i = 0; i < to_add_buffer; i++) {
            module_weight_tensor_map new_buffer;
            for (auto& [buffer_name, buffer_shape] :
                 buffer_shapes["routed_expert"]) {
                // Get per-tensor dtype using the helper function
                torch::Dtype tensor_dtype = get_tensor_dtype(
                    this->engine_config_.gpu_buffer_config,
                    this->engine_config_.basic_config,
                    "routed_expert",
                    buffer_name);
                auto options =
                    torch::TensorOptions()
                        .dtype(tensor_dtype)
                        .device(torch::kCUDA,
                                this->engine_config_.basic_config.device)
                        .requires_grad(false)
                        .memory_format(torch::MemoryFormat::Contiguous);
                new_buffer[buffer_name] = torch::zeros(buffer_shape, options);
            }
            this->buffers_["routed_expert"].push_back(new_buffer);
        }
        this->slots_.grow_type(
            "routed_expert",
            static_cast<int64_t>(this->buffers_["routed_expert"].size()));
        this->notifySlotsLocked();
    }
}

std::optional<
    std::pair<std::reference_wrapper<module_weight_tensor_map>, int64_t>>
GPU_Weight_Buffer::acquireEmptyBuffer(const std::string& module_type) {
    std::lock_guard<std::mutex> lock(this->mutex_);
    this->reclaimCompletedReleasesLocked();
    const std::optional<int64_t> buffer_idx = this->slots_.reserve(module_type);
    if (!buffer_idx.has_value()) {
        return std::nullopt;
    }
    this->logger_->debug("Acquired empty buffer: type={}, idx={}",
                         module_type, *buffer_idx);
    return std::make_pair(std::ref(this->buffers_[module_type][*buffer_idx]),
                          *buffer_idx);
};

void GPU_Weight_Buffer::releaseBuffer(const std::string& module_name) {
    std::lock_guard<std::mutex> lock(this->mutex_);
    const auto slot = this->slots_.release(module_name);
    this->notifySlotsLocked();
    this->logger_->debug("Released buffer: module={}, type={}, idx={}",
                         module_name, slot.module_type, slot.index);
};

void GPU_Weight_Buffer::releaseBuffersAsync(
    const std::vector<std::string>& module_names,
    cudaStream_t consumer_stream) {
    if (module_names.empty()) {
        return;
    }

    std::lock_guard<std::mutex> lock(this->mutex_);
    // The handle destroys the event on every path that does not queue it.
    auto completion_event = std::make_shared<CudaEventHandle>();
    CUDA_CHECK(cudaEventRecord(completion_event->get(), consumer_stream));

    // All or nothing: every name must be LEASED, none duplicated.
    const std::vector<WeightSlotTable::SlotRef> refs =
        this->slots_.release_async(module_names);
    std::vector<std::pair<std::string, int64_t>> slots;
    slots.reserve(refs.size());
    for (const auto& ref : refs) {
        slots.emplace_back(ref.module_type, ref.index);
    }
    this->pending_releases_.push_back(
        PendingRelease{std::move(completion_event), std::move(slots)});
    // A producer with a full ring waits on this release's event.
    this->notifySlotsLocked();
    this->logger_->debug(
        "Queued asynchronous release for {} weight buffers",
        module_names.size());
}

module_weight_tensor_map GPU_Weight_Buffer::get_weights(
    const std::string& module_name) {
    this->logger_->debug("Get weights: {}", module_name);
    const auto start_time = std::chrono::steady_clock::now();
    std::unique_lock<std::mutex> lock(this->mutex_);
    // check_lease throws at once for a module that can never be leased in
    // this pass, so the wait only covers FILLING and not yet produced.
    this->waitWithoutStallLocked(lock, module_name, [this, &module_name] {
        return this->slots_.check_lease(module_name) ==
               WeightLeaseCheck::kReady;
    });
    const auto slot = this->slots_.lease(module_name);
    const auto wait_end = std::chrono::steady_clock::now();
    {
        std::lock_guard<std::mutex> profile_lock(this->weight_profile_mutex_);
        if (this->weight_profile_enabled_) {
            this->weight_profile_requests_[slot.module_type] += 1;
            this->weight_profile_wait_seconds_[slot.module_type] +=
                std::chrono::duration<double>(wait_end - start_time).count();
        }
    }
    return this->buffers_.at(slot.module_type).at(slot.index);
}

int64_t GPU_Weight_Buffer::begin_pass() {
    std::lock_guard<std::mutex> lock(this->mutex_);
    return this->slots_.begin_pass();
}

GPU_Weight_Buffer::SkipCounts GPU_Weight_Buffer::cancel_pass(int64_t pass) {
    std::lock_guard<std::mutex> lock(this->mutex_);
    const SkipCounts counts = this->slots_.cancel_pass(pass);
    this->cv_.notify_all();
    this->notifySlotsLocked();
    return counts;
}

void GPU_Weight_Buffer::wait_pass_drained(int64_t pass) {
    if (pass == 0) {
        return;
    }
    std::unique_lock<std::mutex> lock(this->mutex_);
    const auto current = this->slots_.current_pass();
    if (!current.has_value() || *current != pass) {
        throw std::runtime_error(
            "Cannot end weight pass " + std::to_string(pass) +
            ": it is not the oldest open pass (" +
            this->slots_.describe_passes() + ")");
    }
    // The producer retires the pass's pending skips on its way past its end.
    this->waitWithoutStallLocked(
        lock, "the end of weight pass " + std::to_string(pass),
        [this, pass] {
            return this->slots_.producer_finished(pass) &&
                   this->slots_.count_in_pass(pass,
                                              WeightSlotState::kFilling) == 0;
        });
    // A forward boundary: waiting for the consumer's release events is fine.
    this->drainPendingReleasesLocked();
}

WeightSlotTable::PassSummary GPU_Weight_Buffer::end_pass(int64_t pass) {
    std::lock_guard<std::mutex> lock(this->mutex_);
    this->throwIfProducerFailedLocked();
    return this->slots_.end_pass(pass);
}

WeightSlotTable::AbortCounts GPU_Weight_Buffer::abort_passes(
    bool lose_schedule) {
    // The forward that leased a slot died with its kernels possibly still
    // queued: none of its slots is reusable before they finish.
    CUDA_CHECK(cudaSetDevice(this->engine_config_.basic_config.device));
    CUDA_CHECK(cudaDeviceSynchronize());
    std::lock_guard<std::mutex> lock(this->mutex_);
    // Every release event has completed; the table frees the slots.
    this->synchronizePendingReleasesLocked();
    const WeightSlotTable::AbortCounts counts =
        this->slots_.abort(lose_schedule);
    // A waiter left in get_weights finds no open pass and throws.
    this->bumpProgressLocked();
    this->notifySlotsLocked();
    return counts;
}

WeightSlotTable::Production GPU_Weight_Buffer::next_production(
    const std::string& module_type) {
    std::lock_guard<std::mutex> lock(this->mutex_);
    return this->slots_.next_production(module_type);
}

void GPU_Weight_Buffer::retire_skip(const std::string& module_type,
                                    const std::string& module_name) {
    std::lock_guard<std::mutex> lock(this->mutex_);
    this->slots_.retire_skip(module_type, module_name);
    this->bumpProgressLocked();
}

void GPU_Weight_Buffer::note_producer_progress() {
    std::lock_guard<std::mutex> lock(this->mutex_);
    this->bumpProgressLocked();
}

void GPU_Weight_Buffer::throwIfProducerFailedLocked() const {
    if (this->producer_failure_.has_value()) {
        throw std::runtime_error("HtoD producer failed: " +
                                 *this->producer_failure_);
    }
}

WeightProduction GPU_Weight_Buffer::claim_for_copy(
    const std::string& module_type, int64_t buffer_idx,
    const std::string& module_name) {
    // One critical section with skip_modules() and cancel_pass(): a skip
    // lands either before this check (pending) or after the slot is named
    // (release-on-landing).
    std::lock_guard<std::mutex> lock(this->mutex_);
    const WeightProduction claim =
        this->slots_.claim(module_type, buffer_idx, module_name);
    this->bumpProgressLocked();
    return claim;
}

void GPU_Weight_Buffer::abandon(const std::string& module_type,
                                int64_t buffer_idx) {
    std::lock_guard<std::mutex> lock(this->mutex_);
    this->slots_.abandon(module_type, buffer_idx);
    this->notifySlotsLocked();
}

uint64_t GPU_Weight_Buffer::slot_epoch() const {
    std::lock_guard<std::mutex> lock(this->mutex_);
    return this->slot_epoch_;
}

void GPU_Weight_Buffer::wait_for_free_slot(const std::string& module_type,
                                           uint64_t seen,
                                           const std::atomic<bool>& stop) {
    std::unique_lock<std::mutex> lock(this->mutex_);
    while (true) {
        // Moves the slot epoch for every release it returns to FREE.
        this->reclaimCompletedReleasesLocked();
        if (stop || this->slot_epoch_ != seen ||
            this->slots_.count(module_type, WeightSlotState::kFree) > 0) {
            return;
        }
        if (!this->pending_releases_.empty()) {
            // The oldest release was recorded first. Its event completes
            // with work its consumer already enqueued, the same bound as the
            // pass-end drain. The handle outlives a drain, reset or abort
            // that drops the release while the mutex is not held.
            const std::shared_ptr<CudaEventHandle> oldest =
                this->pending_releases_.front().event;
            lock.unlock();
            const cudaError_t status = cudaEventSynchronize(oldest->get());
            lock.lock();
            CUDA_CHECK(status);
            continue;
        }
        // Only a consumer release, skip, cancel, abort, reset or a producer
        // wake can free a slot now; each moves the epoch under mutex_.
        this->slot_cv_.wait(lock, [this, seen, &stop] {
            return stop || this->slot_epoch_ != seen;
        });
    }
}

void GPU_Weight_Buffer::wake_producer() {
    std::lock_guard<std::mutex> lock(this->mutex_);
    this->notifySlotsLocked();
}

bool GPU_Weight_Buffer::weights_copy_complete(const std::string& module_type,
                                              const std::string& module_name,
                                              int64_t buffer_idx) {
    std::lock_guard<std::mutex> lock(this->mutex_);
    if (this->slots_.name_at(module_type, buffer_idx) != module_name) {
        throw std::runtime_error(
            "Copy of " + module_name + " landed in " + module_type + "[" +
            std::to_string(buffer_idx) + "], which holds " +
            this->slots_.name_at(module_type, buffer_idx) +
            ". Held: " + this->slots_.describe_held());
    }
    const bool present = this->slots_.landed(module_type, buffer_idx);
    this->logger_->debug("Module: {} is in buffer: {} (present={})",
                         module_name, buffer_idx, present);
    this->bumpProgressLocked();
    if (!present) {
        // Released on landing: the slot is FREE again.
        this->notifySlotsLocked();
    }
    return present;
};

void GPU_Weight_Buffer::record_producer_failure(
    const std::vector<std::pair<std::string, int64_t>>& filling,
    const std::string& message) {
    std::lock_guard<std::mutex> lock(this->mutex_);
    this->producer_failure_ = message;
    bool abandoned = false;
    // A slot listed twice is FREE the second time and left alone.
    for (const auto& [module_type, buffer_idx] : filling) {
        if (buffer_idx >= 0 &&
            buffer_idx < this->slots_.slot_count(module_type) &&
            this->slots_.state_at(module_type, buffer_idx) ==
                WeightSlotState::kFilling) {
            this->slots_.abandon(module_type, buffer_idx);
            abandoned = true;
        }
    }
    if (abandoned) {
        this->notifySlotsLocked();
    }
    this->cv_.notify_all();
}

void GPU_Weight_Buffer::clear_producer_failure() {
    std::lock_guard<std::mutex> lock(this->mutex_);
    this->producer_failure_.reset();
}

std::optional<std::string> GPU_Weight_Buffer::producer_failure() const {
    std::lock_guard<std::mutex> lock(this->mutex_);
    return this->producer_failure_;
}

GPU_Weight_Buffer::SkipCounts GPU_Weight_Buffer::skip_modules(
    const std::vector<std::string>& module_names) {
    std::lock_guard<std::mutex> lock(this->mutex_);
    this->throwIfProducerFailedLocked();
    // Validate every name before changing any slot.
    std::unordered_set<std::string> unique_names;
    for (const auto& module_name : module_names) {
        if (!unique_names.insert(module_name).second) {
            throw std::runtime_error("Duplicate module in weight skip: " +
                                     module_name);
        }
        this->slots_.check_skip(module_name);
    }
    SkipCounts counts;
    for (const auto& module_name : module_names) {
        switch (this->slots_.skip(module_name)) {
            case WeightSkipOutcome::kFreedPresent:
                ++counts.freed_present;
                break;
            case WeightSkipOutcome::kReleaseOnLanding:
                ++counts.release_on_landing;
                break;
            case WeightSkipOutcome::kPending:
                ++counts.pending;
                break;
        }
    }
    // As cancel_pass: a waiter on a module just skipped re-checks and throws.
    this->cv_.notify_all();
    if (counts.freed_present > 0) {
        this->notifySlotsLocked();
    }
    return counts;
}

GPU_Weight_Buffer::SlotSnapshot GPU_Weight_Buffer::slot_snapshot() const {
    std::lock_guard<std::mutex> lock(this->mutex_);
    SlotSnapshot snapshot;
    for (const auto& module_type : this->slots_.module_types()) {
        auto& counts = snapshot.counts[module_type];
        counts.resize(kWeightSlotStateCount, 0);
        for (int state = 0; state < kWeightSlotStateCount; ++state) {
            counts[state] = this->slots_.count(
                module_type, static_cast<WeightSlotState>(state));
        }
    }
    snapshot.held = this->slots_.held();
    snapshot.open_passes = this->slots_.open_passes();
    snapshot.cursors = this->slots_.cursors();
    snapshot.pass_counts = this->slots_.pass_counts();
    snapshot.schedule_lost = this->slots_.schedule_lost();
    return snapshot;
}


// void GPU_Weight_Buffer::reset_prefill_buffer() {
//     auto& buffer_shapes = this->engine_config_.gpu_buffer_config.module_shapes;
//     auto& num_buffers =
//         this->engine_config_.gpu_buffer_config
//             .num_prefill_module_buffer;  // {"attn": 1, "routed_expert": 160,
//                                          // "shared_expert": 1}
//     auto options =
//         torch::TensorOptions()
//             .dtype(this->engine_config_.basic_config.weight_dtype_torch)
//             .device(torch::kCUDA, this->engine_config_.basic_config.device)
//             .requires_grad(false)
//             .memory_format(torch::MemoryFormat::Contiguous);
//     {
//         std::lock_guard<std::mutex> lock(this->mutex_);
//         // reset the buffer size as the prefill buffer size.
//         this->buffers_["routed_expert"].clear();
//         this->buffers_["routed_expert"].resize(num_buffers["routed_expert"]);
//         for (auto& [buffer_name, buffer_shape] :
//              buffer_shapes["routed_expert"]) {
//             for (int64_t buffer_idx = 0;
//                  buffer_idx < num_buffers["routed_expert"]; buffer_idx++) {
//                 if (buffer_name.find("norm") == std::string::npos) {
//                     this->buffers_["routed_expert"][buffer_idx][buffer_name] =
//                         torch::zeros(buffer_shape, options);
//                 } else {
//                     auto bf16_options =
//                         torch::TensorOptions()
//                             .dtype(torch::kBFloat16)
//                             .device(torch::kCUDA,
//                                     this->engine_config_.basic_config.device)
//                             .requires_grad(false)
//                             .memory_format(torch::MemoryFormat::Contiguous);
//                     this->buffers_["routed_expert"][buffer_idx][buffer_name] =
//                         torch::ones(buffer_shape, bf16_options);
//                 }
//             }
//         }
//         this->module_in_buffers_.clear();

//         // Set all buffer status to 0.
//         for (auto& [module_type, num_buffer] : num_buffers) {
//             this->buffer_status_[module_type].clear();
//             this->buffer_status_[module_type].resize(num_buffer, 0);
//         }
//         // Log the buffer status and size of each buffer.
//         for (auto& [module_type, num_buffer] : num_buffers) {
//             this->logger_->debug("Module type: {}, Number of buffer: {}",
//                                  module_type, num_buffer);
//             for (int64_t buffer_idx = 0; buffer_idx < num_buffer;
//                  buffer_idx++) {
//                 this->logger_->debug(
//                     "Buffer_idx: {}, Buffer status: {}", buffer_idx,
//                     this->buffer_status_[module_type][buffer_idx]);
//             }
//         }
//     }
// }

// void GPU_Weight_Buffer::reset_prefill_buffer() {
//     auto& buffer_shapes = this->engine_config_.gpu_buffer_config.module_shapes;
//     auto& num_buffers = this->engine_config_.gpu_buffer_config.num_prefill_module_buffer;
    
//     auto options = torch::TensorOptions()
//         .dtype(this->engine_config_.basic_config.weight_dtype_torch)
//         .device(torch::kCUDA, this->engine_config_.basic_config.device)
//         .requires_grad(false)
//         .memory_format(torch::MemoryFormat::Contiguous);
    
//     auto bf16_options = torch::TensorOptions()
//         .dtype(torch::kBFloat16)
//         .device(torch::kCUDA, this->engine_config_.basic_config.device)
//         .requires_grad(false)
//         .memory_format(torch::MemoryFormat::Contiguous);
    
//     {
//         std::lock_guard<std::mutex> lock(this->mutex_);
        
//         // Step 1: Log initial memory state
//         size_t memory_before = 0;
//         if (torch::cuda::is_available()) {
//             size_t free_bytes, total_bytes;
//             cudaSetDevice(this->engine_config_.basic_config.device);
//             cudaMemGetInfo(&free_bytes, &total_bytes);
//             memory_before = total_bytes - free_bytes;
//             this->logger_->info("Prefill reset - Memory before: {} MB", 
//                               memory_before / (1024.0 * 1024.0));
//         }
        
//         // Step 2: Explicitly release ALL existing tensors
//         for (auto& [buffer_type, buffer_vector] : this->buffers_) {
//             for (auto& module_map : buffer_vector) {
//                 for (auto& [name, tensor] : module_map) {
//                     tensor = torch::Tensor();
//                 }
//                 module_map.clear();
//             }
//             buffer_vector.clear();
//             buffer_vector.shrink_to_fit();
//         }
        
//         // Step 3: Clear module mappings
//         this->module_in_buffers_.clear();
        
//         // Step 4: Force CUDA synchronization and cleanup
//         if (torch::cuda::is_available()) {
//             // Use CUDA runtime API
//             cudaDeviceSynchronize();
            
//             // Empty the cache
//             c10::cuda::CUDACachingAllocator::emptyCache();
            
//             // Log memory after clearing
//             size_t free_bytes, total_bytes;
//             cudaMemGetInfo(&free_bytes, &total_bytes);
//             size_t memory_after_clear = total_bytes - free_bytes;
//             this->logger_->info("Prefill reset - Memory after clearing: {} MB (freed: {} MB)", 
//                               memory_after_clear / (1024.0 * 1024.0),
//                               (memory_before - memory_after_clear) / (1024.0 * 1024.0));
//         }
        
//         // Step 5: Create new prefill buffers
//         this->buffers_["routed_expert"].reserve(num_buffers["routed_expert"]);
//         this->buffers_["routed_expert"].resize(num_buffers["routed_expert"]);
        
//         // Pre-calculate norm buffer names
//         std::unordered_set<std::string> norm_buffers;
//         for (const auto& [buffer_name, buffer_shape] : buffer_shapes["routed_expert"]) {
//             if (buffer_name.find("norm") != std::string::npos) {
//                 norm_buffers.insert(buffer_name);
//             }
//         }
        
//         // Create tensors
//         for (auto& [buffer_name, buffer_shape] : buffer_shapes["routed_expert"]) {
//             bool is_norm = norm_buffers.count(buffer_name) > 0;
            
//             for (int64_t buffer_idx = 0; buffer_idx < num_buffers["routed_expert"]; buffer_idx++) {
//                 if (!is_norm) {
//                     this->buffers_["routed_expert"][buffer_idx][buffer_name] =
//                         torch::zeros(buffer_shape, options);
//                 } else {
//                     // Use bf16 for norm layers in prefill
//                     this->buffers_["routed_expert"][buffer_idx][buffer_name] =
//                         torch::ones(buffer_shape, bf16_options);
//                 }
//             }
//         }
        
//         // Step 6: Reset buffer status
//         for (auto& [module_type, num_buffer] : num_buffers) {
//             this->buffer_status_[module_type].clear();
//             this->buffer_status_[module_type].shrink_to_fit();
//             this->buffer_status_[module_type].resize(num_buffer, 0);
//         }
        
//         // Step 7: Final memory check
//         if (torch::cuda::is_available()) {
//             size_t free_bytes, total_bytes;
//             cudaMemGetInfo(&free_bytes, &total_bytes);
//             size_t memory_final = total_bytes - free_bytes;
//             this->logger_->info("Prefill reset - Memory after recreation: {} MB (net change: {} MB)", 
//                               memory_final / (1024.0 * 1024.0),
//                               (static_cast<int64_t>(memory_final) - static_cast<int64_t>(memory_before)) / (1024.0 * 1024.0));
//         }
        
//         this->logger_->info("Prefill buffer reset complete");
//     }
// }

void GPU_Weight_Buffer::reset_prefill_buffer() {
    auto& buffer_shapes = this->engine_config_.gpu_buffer_config.module_shapes;
    auto& num_buffers = this->engine_config_.gpu_buffer_config.num_prefill_module_buffer;

    {
        std::lock_guard<std::mutex> lock(this->mutex_);
        // Between forwards only: a closed pass holds no slot and no skip.
        this->slots_.require_no_open_pass("reset the prefill weight ring");
        this->synchronizePendingReleasesLocked();

        // IMPORTANT: Only clear routed_expert buffers, not all buffers!
        // First, properly release the tensors in routed_expert
        if (this->buffers_.find("routed_expert") != this->buffers_.end()) {
            for (auto& module_map : this->buffers_["routed_expert"]) {
                for (auto& [name, tensor] : module_map) {
                    // Explicitly release tensor memory
                    tensor = torch::Tensor();
                }
                module_map.clear();
            }
            this->buffers_["routed_expert"].clear();

            // Optional: Force memory cleanup for routed_expert tensors only
            if (torch::cuda::is_available()) {
                cudaDeviceSynchronize();
                c10::cuda::CUDACachingAllocator::emptyCache();
            }
        }

        // Reset the buffer size as the prefill buffer size
        this->buffers_["routed_expert"].resize(num_buffers["routed_expert"]);

        // Create new tensors
        for (auto& [buffer_name, buffer_shape] : buffer_shapes["routed_expert"]) {
            // Get per-tensor dtype using the helper function
            torch::Dtype tensor_dtype = get_tensor_dtype(
                this->engine_config_.gpu_buffer_config,
                this->engine_config_.basic_config,
                "routed_expert",
                buffer_name);
            auto options = torch::TensorOptions()
                .dtype(tensor_dtype)
                .device(torch::kCUDA, this->engine_config_.basic_config.device)
                .requires_grad(false)
                .memory_format(torch::MemoryFormat::Contiguous);
            for (int64_t buffer_idx = 0; buffer_idx < num_buffers["routed_expert"]; buffer_idx++) {
                this->buffers_["routed_expert"][buffer_idx][buffer_name] =
                    torch::zeros(buffer_shape, options);
            }
        }

        // Every slot FREE (for ALL module types, not just routed_expert)
        for (auto& [module_type, num_buffer] : num_buffers) {
            this->slots_.reset_type(module_type, num_buffer);
        }
        this->notifySlotsLocked();

        // Log the buffer size of each type (keep original debug level)
        for (auto& [module_type, num_buffer] : num_buffers) {
            this->logger_->debug("Module type: {}, Number of buffer: {}",
                                 module_type, num_buffer);
        }
    }
}


// void GPU_Weight_Buffer::reset_decoding_buffer() {
//     auto& buffer_shapes = this->engine_config_.gpu_buffer_config.module_shapes;
//     auto& num_buffers =
//         this->engine_config_.gpu_buffer_config
//             .num_decoding_module_buffer;  // {"attn": 1, "routed_expert": 160,
//                                           // "shared_expert": 1}
//     auto options =
//         torch::TensorOptions()
//             .dtype(this->engine_config_.basic_config.weight_dtype_torch)
//             .device(torch::kCUDA, this->engine_config_.basic_config.device)
//             .requires_grad(false)
//             .memory_format(torch::MemoryFormat::Contiguous);
//     {
//         std::lock_guard<std::mutex> lock(this->mutex_);
//         // reset the buffer size as the prefill buffer size.
//         this->buffers_["routed_expert"].clear();
//         this->buffers_["routed_expert"].resize(num_buffers["routed_expert"]);
//         for (auto& [buffer_name, buffer_shape] :
//              buffer_shapes["routed_expert"]) {
//             for (int64_t buffer_idx = 0;
//                  buffer_idx < num_buffers["routed_expert"]; buffer_idx++) {
//                 if (buffer_name.find("norm") == std::string::npos) {
//                     this->buffers_["routed_expert"][buffer_idx][buffer_name] =
//                         torch::zeros(buffer_shape, options);
//                 } else {
//                     auto options =
//                         torch::TensorOptions()
//                             .dtype(this->engine_config_.basic_config.weight_dtype_torch)
//                             .device(torch::kCUDA,
//                                     this->engine_config_.basic_config.device)
//                             .requires_grad(false)
//                             .memory_format(torch::MemoryFormat::Contiguous);
//                     this->buffers_["routed_expert"][buffer_idx][buffer_name] =
//                         torch::ones(buffer_shape, options);
//                 }
//             }
//         }
//         this->module_in_buffers_.clear();

//         // Set all buffer status to 0.
//         for (auto& [module_type, num_buffer] : num_buffers) {
//             this->buffer_status_[module_type].clear();
//             this->buffer_status_[module_type].resize(num_buffer, 0);
//         }
//         // Log the buffer status and size of each buffer.
//         for (auto& [module_type, num_buffer] : num_buffers) {
//             this->logger_->debug("Module type: {}, Number of buffer: {}",
//                                  module_type, num_buffer);
//             for (int64_t buffer_idx = 0; buffer_idx < num_buffer;
//                  buffer_idx++) {
//                 this->logger_->debug(
//                     "Buffer_idx: {}, Buffer status: {}", buffer_idx,
//                     this->buffer_status_[module_type][buffer_idx]);
//             }
//         }
//         this->logger_->debug("Decoding buffer reset complete.");
//     }
// }

// void GPU_Weight_Buffer::reset_decoding_buffer() {
//     auto& buffer_shapes = this->engine_config_.gpu_buffer_config.module_shapes;
//     auto& num_buffers = this->engine_config_.gpu_buffer_config.num_decoding_module_buffer;
    
//     auto options = torch::TensorOptions()
//         .dtype(this->engine_config_.basic_config.weight_dtype_torch)
//         .device(torch::kCUDA, this->engine_config_.basic_config.device)
//         .requires_grad(false)
//         .memory_format(torch::MemoryFormat::Contiguous);
    
//     {
//         std::lock_guard<std::mutex> lock(this->mutex_);
        
//         // Step 1: Log initial memory state
//         size_t memory_before = 0;
//         if (torch::cuda::is_available()) {
//             // Using CUDA runtime API for memory info
//             size_t free_bytes, total_bytes;
//             cudaSetDevice(this->engine_config_.basic_config.device);
//             cudaMemGetInfo(&free_bytes, &total_bytes);
//             memory_before = total_bytes - free_bytes;
//             this->logger_->info("Decoding reset - Memory before: {} MB", 
//                               memory_before / (1024.0 * 1024.0));
//         }
        
//         // Step 2: Explicitly release ALL existing tensors in ALL buffer types
//         for (auto& [buffer_type, buffer_vector] : this->buffers_) {
//             for (auto& module_map : buffer_vector) {
//                 for (auto& [name, tensor] : module_map) {
//                     // Explicitly release each tensor
//                     tensor = torch::Tensor();
//                 }
//                 module_map.clear();
//             }
//             buffer_vector.clear();
            
//             // Force deallocation of vector memory
//             buffer_vector.shrink_to_fit();
//         }
        
//         // Step 3: Clear module mappings
//         this->module_in_buffers_.clear();
        
//         // Step 4: Force CUDA synchronization and aggressive memory cleanup
//         if (torch::cuda::is_available()) {
//             // CORRECT SYNTAX: Use cudaDeviceSynchronize() from CUDA runtime
//             cudaDeviceSynchronize();
            
//             // OR use AT namespace correctly:
//             // at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
//             // stream.synchronize();
            
//             // Force return memory to system
//             c10::cuda::CUDACachingAllocator::emptyCache();
            
//             // Log memory after clearing
//             size_t free_bytes, total_bytes;
//             cudaMemGetInfo(&free_bytes, &total_bytes);
//             size_t memory_after_clear = total_bytes - free_bytes;
//             this->logger_->info("Decoding reset - Memory after clearing: {} MB (freed: {} MB)", 
//                               memory_after_clear / (1024.0 * 1024.0),
//                               (memory_before - memory_after_clear) / (1024.0 * 1024.0));
//         }
        
//         // Step 5: Create new decoding buffers for routed_expert
//         this->buffers_["routed_expert"].reserve(num_buffers["routed_expert"]);
//         this->buffers_["routed_expert"].resize(num_buffers["routed_expert"]);
        
//         // Pre-calculate buffer names that need norm (for efficiency)
//         std::unordered_set<std::string> norm_buffers;
//         for (const auto& [buffer_name, buffer_shape] : buffer_shapes["routed_expert"]) {
//             if (buffer_name.find("norm") != std::string::npos) {
//                 norm_buffers.insert(buffer_name);
//             }
//         }
        
//         // Create tensors
//         for (auto& [buffer_name, buffer_shape] : buffer_shapes["routed_expert"]) {
//             bool is_norm = norm_buffers.count(buffer_name) > 0;
            
//             for (int64_t buffer_idx = 0; buffer_idx < num_buffers["routed_expert"]; buffer_idx++) {
//                 if (!is_norm) {
//                     this->buffers_["routed_expert"][buffer_idx][buffer_name] =
//                         torch::zeros(buffer_shape, options);
//                 } else {
//                     this->buffers_["routed_expert"][buffer_idx][buffer_name] =
//                         torch::ones(buffer_shape, options);
//                 }
//             }
//         }
        
//         // Step 6: Reset buffer status for ALL module types
//         for (auto& [module_type, num_buffer] : num_buffers) {
//             this->buffer_status_[module_type].clear();
//             this->buffer_status_[module_type].shrink_to_fit();
//             this->buffer_status_[module_type].resize(num_buffer, 0);
//         }
        
//         // Step 7: Final memory check and logging
//         if (torch::cuda::is_available()) {
//             size_t free_bytes, total_bytes;
//             cudaMemGetInfo(&free_bytes, &total_bytes);
//             size_t memory_final = total_bytes - free_bytes;
//             this->logger_->info("Decoding reset - Memory after recreation: {} MB (net change: {} MB)", 
//                               memory_final / (1024.0 * 1024.0),
//                               (static_cast<int64_t>(memory_final) - static_cast<int64_t>(memory_before)) / (1024.0 * 1024.0));
//         }
        
//         // Log buffer status
//         for (auto& [module_type, num_buffer] : num_buffers) {
//             this->logger_->debug("Module type: {}, Number of buffer: {}",
//                                module_type, num_buffer);
//             for (int64_t buffer_idx = 0; buffer_idx < num_buffer; buffer_idx++) {
//                 this->logger_->debug("Buffer_idx: {}, Buffer status: {}", 
//                                    buffer_idx, this->buffer_status_[module_type][buffer_idx]);
//             }
//         }
        
//         this->logger_->debug("Decoding buffer reset complete.");
//     }
// }


void GPU_Weight_Buffer::reset_decoding_buffer() {
    auto& buffer_shapes = this->engine_config_.gpu_buffer_config.module_shapes;
    auto& num_buffers = this->engine_config_.gpu_buffer_config.num_decoding_module_buffer;

    {
        std::lock_guard<std::mutex> lock(this->mutex_);
        // Between forwards only: a closed pass holds no slot and no skip.
        this->slots_.require_no_open_pass("reset the decoding weight ring");
        this->synchronizePendingReleasesLocked();

        // IMPORTANT: Only clear routed_expert buffers!
        if (this->buffers_.find("routed_expert") != this->buffers_.end()) {
            for (auto& module_map : this->buffers_["routed_expert"]) {
                for (auto& [name, tensor] : module_map) {
                    // Explicitly release tensor memory
                    tensor = torch::Tensor();
                }
                module_map.clear();
            }
            this->buffers_["routed_expert"].clear();

            // Optional: Force memory cleanup
            if (torch::cuda::is_available()) {
                cudaDeviceSynchronize();
                c10::cuda::CUDACachingAllocator::emptyCache();
            }
        }

        // Reset the buffer size
        this->buffers_["routed_expert"].resize(num_buffers["routed_expert"]);

        // Create new tensors
        for (auto& [buffer_name, buffer_shape] : buffer_shapes["routed_expert"]) {
            // Get per-tensor dtype using the helper function
            torch::Dtype tensor_dtype = get_tensor_dtype(
                this->engine_config_.gpu_buffer_config,
                this->engine_config_.basic_config,
                "routed_expert",
                buffer_name);
            auto options = torch::TensorOptions()
                .dtype(tensor_dtype)
                .device(torch::kCUDA, this->engine_config_.basic_config.device)
                .requires_grad(false)
                .memory_format(torch::MemoryFormat::Contiguous);
            for (int64_t buffer_idx = 0; buffer_idx < num_buffers["routed_expert"]; buffer_idx++) {
                this->buffers_["routed_expert"][buffer_idx][buffer_name] =
                    torch::zeros(buffer_shape, options);
            }
        }

        // Every slot FREE
        for (auto& [module_type, num_buffer] : num_buffers) {
            this->slots_.reset_type(module_type, num_buffer);
        }
        this->notifySlotsLocked();

        // Log the buffer size of each type
        for (auto& [module_type, num_buffer] : num_buffers) {
            this->logger_->debug("Module type: {}, Number of buffer: {}",
                                 module_type, num_buffer);
        }

        this->logger_->debug("Decoding buffer reset complete.");
    }
}
