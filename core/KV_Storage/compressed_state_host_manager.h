#ifndef COMPRESSED_STATE_HOST_MANAGER_H_
#define COMPRESSED_STATE_HOST_MANAGER_H_

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "host_paged_kv_worker_view.h"
#include "transformed_host_paged_kv_utils.h"

namespace batchgen::kv {

struct CompressedStateHostConfig {
    std::size_t num_layers = 0;
    std::size_t num_state_items = 0;
    std::size_t ring_size = 0;
    std::size_t state_token_bytes = 0;
    std::size_t sequence_table_capacity = 0;
    std::size_t alignment_bytes = 64;
    LayerMapping logical_to_physical_layer;
    std::string logger_name;
};

inline std::string ToString(const CompressedStateHostConfig& config) {
    std::ostringstream oss;
    oss << "CompressedStateHostConfig(num_layers=" << config.num_layers
        << ", num_state_items=" << config.num_state_items
        << ", ring_size=" << config.ring_size
        << ", state_token_bytes=" << config.state_token_bytes
        << ", sequence_table_capacity=" << config.sequence_table_capacity
        << ", alignment_bytes=" << config.alignment_bytes
        << ", logical_to_physical_layer=";
    AppendLogicalLayerMapping(oss, config.logical_to_physical_layer);
    oss << ")";
    return oss.str();
}

struct CompressedStateHostStats {
    std::size_t num_total_state_items = 0;
    std::size_t num_free_state_items = 0;
    std::size_t num_used_state_items = 0;
    std::size_t num_active_sequences = 0;
    std::size_t sequence_table_capacity = 0;
    std::size_t total_bytes = 0;
};

inline std::string ToString(const CompressedStateHostStats& stats) {
    std::ostringstream oss;
    oss << "CompressedStateHostStats(num_total_state_items="
        << stats.num_total_state_items
        << ", num_free_state_items=" << stats.num_free_state_items
        << ", num_used_state_items=" << stats.num_used_state_items
        << ", num_active_sequences=" << stats.num_active_sequences
        << ", sequence_table_capacity=" << stats.sequence_table_capacity
        << ", total_bytes=" << stats.total_bytes << ")";
    return oss.str();
}

template <std::size_t Ratio, bool Overlap>
class CompressedStateHostManager {
   public:
    static_assert(Ratio > 0, "Ratio must be greater than zero");
    static constexpr std::size_t kRatio = Ratio;
    static constexpr bool kOverlap = Overlap;

    explicit CompressedStateHostManager(CompressedStateHostConfig config)
        : config_(SanitizeConfig(std::move(config))),
          state_item_bytes_(config_.ring_size * config_.state_token_bytes),
          state_item_stride_bytes_(
              detail::AlignUp(state_item_bytes_, config_.alignment_bytes)),
          layer_stride_bytes_(detail::AlignUp(
              config_.num_state_items * state_item_stride_bytes_,
              config_.alignment_bytes)),
          total_bytes_(config_.num_layers * layer_stride_bytes_),
          logger_(init_logger(
              "info", config_.logger_name.empty()
                          ? "CompressedStateHostManager"
                          : config_.logger_name)) {
        ResetFreeStateItems();
    }

    CompressedStateHostManager(const CompressedStateHostManager&) = delete;
    CompressedStateHostManager& operator=(
        const CompressedStateHostManager&) = delete;
    CompressedStateHostManager(CompressedStateHostManager&&) = delete;
    CompressedStateHostManager& operator=(CompressedStateHostManager&&) =
        delete;

    ~CompressedStateHostManager() {
        try {
            Shutdown();
        } catch (const std::exception& ex) {
            logger_->error("CompressedStateHostManager shutdown failed: {}",
                           ex.what());
        }
    }

    void Initialize(int device_index) {
        if (device_index < 0) {
            throw std::invalid_argument("device_index must be >= 0");
        }
        std::lock_guard<std::mutex> lock(mutex_);
        if (initialized_) {
            return;
        }
        device_index_ = device_index;
        storage_.assign(total_bytes_, std::byte{0});
        if (!storage_.empty()) {
            worker_detail::RegisterPinnedRange(
                storage_.data(), storage_.size(), device_index_, logger_);
            pinned_ = true;
        }
        initialized_ = true;
        logger_->info(
            "CompressedStateHostManager ready (device_index={}, ratio={}, "
            "overlap={}, layers={}, state_items={}, state_item_bytes={}, "
            "total_bytes={})",
            device_index_, Ratio, Overlap, config_.num_layers,
            config_.num_state_items, state_item_bytes_, total_bytes_);
    }

    void Shutdown() {
        std::lock_guard<std::mutex> lock(mutex_);
        pending_tasks_.Drain();
        if (pinned_ && !storage_.empty()) {
            worker_detail::UnregisterPinnedRange(storage_.data(), device_index_,
                                                 logger_);
        }
        pinned_ = false;
        initialized_ = false;
        device_index_ = -1;
        storage_.clear();
        sequence_states_.clear();
        ResetFreeStateItems();
    }

    std::vector<std::int32_t> AllocateStateItemsForSequences(
        const std::vector<std::int64_t>& sequence_ids) {
        EnsureInitialized();
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<std::int32_t> allocations;
        allocations.reserve(sequence_ids.size());
        for (std::int64_t sequence_id : sequence_ids) {
            allocations.push_back(EnsureStateItemLocked(sequence_id));
        }
        return allocations;
    }

    std::int32_t AllocateStateItem(std::int64_t sequence_id) {
        auto allocations = AllocateStateItemsForSequences({sequence_id});
        return allocations.empty() ? -1 : allocations.front();
    }

    void ReleaseSequenceStates(
        const std::vector<std::int64_t>& sequence_ids) {
        EnsureInitialized();
        std::lock_guard<std::mutex> lock(mutex_);
        pending_tasks_.Drain();
        for (std::int64_t sequence_id : sequence_ids) {
            auto it = sequence_states_.find(sequence_id);
            if (it == sequence_states_.end()) {
                continue;
            }
            free_state_items_.push_back(it->second.state_item_id);
            sequence_states_.erase(it);
        }
    }

    KVAsyncTask AsyncOffloadDecodeStateToHost(
        std::size_t layer_idx, std::vector<std::int64_t> sequence_ids,
        torch::Tensor state_tensor, SequenceLengths raw_positions) {
        constexpr std::string_view kOpName =
            "AsyncOffloadDecodeStateToHost";
        auto prepared =
            PrepareDecodeRows(layer_idx, sequence_ids, state_tensor,
                              raw_positions, CopyDirection::kDeviceToHost,
                              kOpName);
        auto task = transformed_detail::MakeAsyncTask(
            [prepared = std::move(prepared)]() mutable {
                CopyDecodeRows(prepared, CopyDirection::kDeviceToHost);
            });
        TrackTask(task);
        return task;
    }

    KVAsyncTask AsyncLoadDecodeStateToDevice(
        std::size_t layer_idx, std::vector<std::int64_t> sequence_ids,
        torch::Tensor state_tensor, SequenceLengths raw_positions) {
        constexpr std::string_view kOpName = "AsyncLoadDecodeStateToDevice";
        auto prepared =
            PrepareDecodeRows(layer_idx, sequence_ids, state_tensor,
                              raw_positions, CopyDirection::kHostToDevice,
                              kOpName);
        auto task = transformed_detail::MakeAsyncTask(
            [prepared = std::move(prepared)]() mutable {
                CopyDecodeRows(prepared, CopyDirection::kHostToDevice);
            });
        TrackTask(task);
        return task;
    }

    KVAsyncTask AsyncOffloadStateItemsToHost(
        std::vector<std::int64_t> sequence_ids,
        torch::Tensor state_device_ptrs) {
        auto prepared = PrepareStateItemRows(
            std::move(sequence_ids), std::move(state_device_ptrs),
            "AsyncOffloadStateItemsToHost");
        auto task = transformed_detail::MakeAsyncTask(
            [prepared = std::move(prepared)]() mutable {
                CopyStateItems(prepared, CopyDirection::kDeviceToHost);
            });
        TrackTask(task);
        return task;
    }

    KVAsyncTask AsyncLoadStateItemsToDevice(
        std::vector<std::int64_t> sequence_ids,
        torch::Tensor state_device_ptrs) {
        auto prepared = PrepareStateItemRows(
            std::move(sequence_ids), std::move(state_device_ptrs),
            "AsyncLoadStateItemsToDevice");
        auto task = transformed_detail::MakeAsyncTask(
            [prepared = std::move(prepared)]() mutable {
                CopyStateItems(prepared, CopyDirection::kHostToDevice);
            });
        TrackTask(task);
        return task;
    }

    [[nodiscard]] std::uintptr_t GetSequenceLayerStateItemPointer(
        std::int64_t sequence_id, std::size_t layer_idx) const {
        EnsureInitialized();
        const std::size_t physical_layer =
            ResolvePhysicalLayer(layer_idx,
                                 "GetSequenceLayerStateItemPointer");
        std::lock_guard<std::mutex> lock(mutex_);
        const std::int32_t state_item_id =
            SequenceStateItemLocked(sequence_id);
        return reinterpret_cast<std::uintptr_t>(
            StateItemPtrPhysical(physical_layer, state_item_id));
    }

    [[nodiscard]] std::uintptr_t StateItemPtr(std::size_t layer_idx,
                                              std::int32_t state_item_id) {
        EnsureInitialized();
        const std::size_t physical_layer =
            ResolvePhysicalLayer(layer_idx, "StateItemPtr");
        return reinterpret_cast<std::uintptr_t>(
            StateItemPtrPhysical(physical_layer, state_item_id));
    }

    [[nodiscard]] std::int64_t ResolveStateSlot(std::int64_t sequence_id,
                                                std::size_t raw_position) {
        EnsureInitialized();
        std::lock_guard<std::mutex> lock(mutex_);
        const SlotAddress slot =
            ResolveSlotLocked(sequence_id, raw_position, 0,
                              "ResolveStateSlot");
        return static_cast<std::int64_t>(
            slot.state_item_id *
                static_cast<std::int32_t>(config_.ring_size) +
            static_cast<std::int32_t>(slot.ring_offset));
    }

    [[nodiscard]] std::size_t ResolvePhysicalLayer(
        std::size_t logical_layer_idx, std::string_view context) const {
        if (config_.logical_to_physical_layer.empty()) {
            if (logical_layer_idx >= config_.num_layers) {
                std::ostringstream oss;
                oss << context << ": layer " << logical_layer_idx
                    << " is out of range for num_layers="
                    << config_.num_layers;
                throw std::out_of_range(oss.str());
            }
            return logical_layer_idx;
        }
        if (logical_layer_idx >= config_.logical_to_physical_layer.size()) {
            std::ostringstream oss;
            oss << context << ": logical layer " << logical_layer_idx
                << " has no physical layer";
            throw std::out_of_range(oss.str());
        }
        const std::int32_t physical =
            config_.logical_to_physical_layer[logical_layer_idx];
        if (physical < 0) {
            std::ostringstream oss;
            oss << context << ": logical layer " << logical_layer_idx
                << " has no physical layer";
            throw std::out_of_range(oss.str());
        }
        return static_cast<std::size_t>(physical);
    }

    [[nodiscard]] CompressedStateHostStats GetStats() const {
        EnsureInitialized();
        std::lock_guard<std::mutex> lock(mutex_);
        CompressedStateHostStats stats;
        stats.num_total_state_items = config_.num_state_items;
        stats.num_free_state_items = free_state_items_.size();
        stats.num_used_state_items =
            config_.num_state_items - free_state_items_.size();
        stats.num_active_sequences = sequence_states_.size();
        stats.sequence_table_capacity = config_.sequence_table_capacity;
        stats.total_bytes = total_bytes_;
        return stats;
    }

    [[nodiscard]] const CompressedStateHostConfig& config() const {
        return config_;
    }

    [[nodiscard]] std::uintptr_t DataBaseAddress() const {
        EnsureInitialized();
        return reinterpret_cast<std::uintptr_t>(storage_.data());
    }

    [[nodiscard]] int device_index() const { return device_index_; }
    [[nodiscard]] std::size_t ratio() const { return Ratio; }
    [[nodiscard]] bool overlap() const { return Overlap; }
    [[nodiscard]] std::size_t num_state_items() const {
        return config_.num_state_items;
    }
    [[nodiscard]] std::size_t ring_size() const { return config_.ring_size; }
    [[nodiscard]] std::size_t state_token_bytes() const {
        return config_.state_token_bytes;
    }
    [[nodiscard]] std::size_t state_item_bytes() const {
        return state_item_bytes_;
    }
    [[nodiscard]] std::size_t state_item_stride_bytes() const {
        return state_item_stride_bytes_;
    }
    [[nodiscard]] std::size_t layer_stride_bytes() const {
        return layer_stride_bytes_;
    }

    [[nodiscard]] std::string DebugString() const {
        std::ostringstream oss;
        oss << "CompressedStateHostManager(ratio=" << Ratio
            << ", overlap=" << Overlap << ", config=" << ToString(config_)
            << ")";
        return oss.str();
    }

   private:
    enum class CopyDirection { kHostToDevice, kDeviceToHost };

    struct SequenceState {
        std::int32_t state_item_id = -1;
    };

    struct SlotAddress {
        std::int32_t state_item_id = -1;
        std::size_t ring_offset = 0;
        std::byte* host_ptr = nullptr;
    };

    struct DecodeRowCopy {
        std::byte* host_ptr = nullptr;
        std::byte* tensor_ptr = nullptr;
    };

    struct PreparedDecodeRows {
        torch::Tensor tensor;
        std::vector<DecodeRowCopy> rows;
        std::size_t row_bytes = 0;
        int device_index = -1;
    };

    struct StateItemCopy {
        std::byte* host_ptr = nullptr;
        std::byte* device_ptr = nullptr;
    };

    struct PreparedStateItemRows {
        torch::Tensor pointer_tensor;
        std::vector<StateItemCopy> state_items;
        std::size_t state_item_bytes = 0;
        int device_index = -1;
    };

    static CompressedStateHostConfig SanitizeConfig(
        CompressedStateHostConfig config) {
        if (config.sequence_table_capacity == 0) {
            config.sequence_table_capacity = config.num_state_items;
        }
        if (config.alignment_bytes == 0) {
            config.alignment_bytes = 64;
        }
        std::vector<std::string> errors;
        if (config.num_layers == 0) {
            errors.emplace_back("num_layers must be > 0");
        }
        if (config.num_state_items == 0) {
            errors.emplace_back("num_state_items must be > 0");
        }
        if (config.ring_size == 0) {
            errors.emplace_back("ring_size must be > 0");
        }
        if (config.state_token_bytes == 0) {
            errors.emplace_back("state_token_bytes must be > 0");
        }
        if (config.ring_size != 0 && config.ring_size % Ratio != 0) {
            errors.emplace_back("ring_size must be divisible by ratio");
        }
        for (std::size_t logical_layer = 0;
             logical_layer < config.logical_to_physical_layer.size();
             ++logical_layer) {
            const std::int32_t physical =
                config.logical_to_physical_layer[logical_layer];
            if (physical < -1) {
                std::ostringstream oss;
                oss << "logical_to_physical_layer[" << logical_layer
                    << "] must be >= -1";
                errors.emplace_back(oss.str());
            } else if (config.num_layers > 0 && physical >= 0 &&
                       static_cast<std::size_t>(physical) >=
                           config.num_layers) {
                std::ostringstream oss;
                oss << "logical_to_physical_layer[" << logical_layer
                    << "] physical layer id " << physical
                    << " must be < num_layers (" << config.num_layers
                    << ")";
                errors.emplace_back(oss.str());
            }
        }
        if (!errors.empty()) {
            std::string message = "Invalid CompressedStateHostConfig: ";
            for (std::size_t i = 0; i < errors.size(); ++i) {
                message.append(errors[i]);
                if (i + 1 < errors.size()) {
                    message.append(", ");
                }
            }
            throw std::invalid_argument(message);
        }
        return config;
    }

    void EnsureInitialized() const {
        if (!initialized_) {
            throw std::runtime_error(
                "CompressedStateHostManager.Initialize must be called before "
                "use");
        }
    }

    void ResetFreeStateItems() {
        free_state_items_.clear();
        free_state_items_.reserve(config_.num_state_items);
        for (std::int32_t state_item =
                 static_cast<std::int32_t>(config_.num_state_items);
             state_item > 0; --state_item) {
            free_state_items_.push_back(state_item - 1);
        }
    }

    std::int32_t EnsureStateItemLocked(std::int64_t sequence_id) {
        auto it = sequence_states_.find(sequence_id);
        if (it != sequence_states_.end()) {
            return it->second.state_item_id;
        }
        if (sequence_states_.size() >= config_.sequence_table_capacity) {
            std::ostringstream oss;
            oss << "CompressedStateHostManager: sequence table capacity "
                << config_.sequence_table_capacity << " exceeded";
            throw std::runtime_error(oss.str());
        }
        if (free_state_items_.empty()) {
            throw std::runtime_error(
                "CompressedStateHostManager: insufficient free state items");
        }
        const std::int32_t state_item_id = free_state_items_.back();
        free_state_items_.pop_back();
        sequence_states_.emplace(sequence_id, SequenceState{state_item_id});
        return state_item_id;
    }

    [[nodiscard]] std::int32_t SequenceStateItemLocked(
        std::int64_t sequence_id) const {
        const auto it = sequence_states_.find(sequence_id);
        if (it == sequence_states_.end()) {
            std::ostringstream oss;
            oss << "CompressedStateHostManager: sequence " << sequence_id
                << " has no allocated state item";
            throw std::out_of_range(oss.str());
        }
        return it->second.state_item_id;
    }

    [[nodiscard]] SlotAddress ResolveSlotLocked(
        std::int64_t sequence_id, std::size_t raw_position,
        std::size_t physical_layer, std::string_view context) {
        (void)context;
        const std::int32_t state_item_id =
            SequenceStateItemLocked(sequence_id);
        const std::size_t ring_offset = raw_position % config_.ring_size;
        return {state_item_id, ring_offset,
                StateSlotPtrPhysical(physical_layer, state_item_id,
                                     ring_offset)};
    }

    [[nodiscard]] std::byte* StateItemPtrPhysical(
        std::size_t physical_layer, std::int32_t state_item_id) const {
        if (physical_layer >= config_.num_layers) {
            throw std::out_of_range("physical layer out of range");
        }
        if (state_item_id < 0 ||
            static_cast<std::size_t>(state_item_id) >=
                config_.num_state_items) {
            throw std::out_of_range("state item id out of range");
        }
        return const_cast<std::byte*>(
            storage_.data() + physical_layer * layer_stride_bytes_ +
            static_cast<std::size_t>(state_item_id) *
                state_item_stride_bytes_);
    }

    [[nodiscard]] std::byte* StateSlotPtrPhysical(
        std::size_t physical_layer, std::int32_t state_item_id,
        std::size_t ring_offset) const {
        if (ring_offset >= config_.ring_size) {
            throw std::out_of_range("ring offset out of range");
        }
        return StateItemPtrPhysical(physical_layer, state_item_id) +
               ring_offset * config_.state_token_bytes;
    }

    PreparedDecodeRows PrepareDecodeRows(
        std::size_t layer_idx, const std::vector<std::int64_t>& sequence_ids,
        torch::Tensor state_tensor, const SequenceLengths& raw_positions,
        CopyDirection direction, std::string_view op_name) {
        EnsureInitialized();
        const std::size_t physical_layer =
            ResolvePhysicalLayer(layer_idx, op_name);
        if (sequence_ids.empty()) {
            return {std::move(state_tensor), {}, config_.state_token_bytes};
        }
        if (state_tensor.size(0) !=
            static_cast<std::int64_t>(sequence_ids.size())) {
            std::ostringstream oss;
            oss << op_name
                << ": state_tensor batch dimension must equal sequence_ids "
                   "size";
            throw std::invalid_argument(oss.str());
        }
        if (direction == CopyDirection::kHostToDevice &&
            !state_tensor.is_contiguous()) {
            std::ostringstream oss;
            oss << op_name << ": destination state_tensor must be contiguous";
            throw std::invalid_argument(oss.str());
        }
        if (!state_tensor.is_contiguous()) {
            state_tensor = state_tensor.contiguous();
        }
        const std::size_t row_bytes = RowBytes(state_tensor, op_name);
        PreparedDecodeRows prepared;
        prepared.tensor = std::move(state_tensor);
        prepared.row_bytes = row_bytes;
        prepared.device_index = device_index_;
        prepared.rows.reserve(sequence_ids.size());
        std::lock_guard<std::mutex> lock(mutex_);
        for (std::size_t batch_idx = 0; batch_idx < sequence_ids.size();
             ++batch_idx) {
            const std::int64_t sequence_id = sequence_ids[batch_idx];
            const std::size_t raw_position =
                transformed_detail::ResolveLength(
                    raw_positions, batch_idx, sequence_id, op_name);
            if (raw_position ==
                std::numeric_limits<std::size_t>::max()) {
                throw std::out_of_range(
                    "raw position overflow in compressed state manager");
            }
            EnsureStateItemLocked(sequence_id);
            const SlotAddress slot =
                ResolveSlotLocked(sequence_id, raw_position, physical_layer,
                                  op_name);
            prepared.rows.push_back(
                {slot.host_ptr,
                 TensorRowPtr(prepared.tensor, batch_idx, row_bytes)});
        }
        return prepared;
    }

    PreparedStateItemRows PrepareStateItemRows(
        std::vector<std::int64_t> sequence_ids,
        torch::Tensor state_device_ptrs, std::string_view op_name) {
        EnsureInitialized();
        if (state_device_ptrs.scalar_type() != torch::kInt64) {
            std::ostringstream oss;
            oss << op_name << ": state_device_ptrs must be int64";
            throw std::invalid_argument(oss.str());
        }
        if (state_device_ptrs.dim() != 2 ||
            state_device_ptrs.size(0) !=
                static_cast<std::int64_t>(config_.num_layers) ||
            state_device_ptrs.size(1) !=
                static_cast<std::int64_t>(sequence_ids.size())) {
            std::ostringstream oss;
            oss << op_name
                << ": state_device_ptrs must have shape [num_layers, batch]";
            throw std::invalid_argument(oss.str());
        }
        if (state_device_ptrs.device().is_cuda()) {
            state_device_ptrs =
                state_device_ptrs.to(torch::Device(torch::kCPU));
        }
        state_device_ptrs = state_device_ptrs.contiguous();
        PreparedStateItemRows prepared;
        prepared.pointer_tensor = std::move(state_device_ptrs);
        prepared.state_item_bytes = state_item_bytes_;
        prepared.device_index = device_index_;
        std::lock_guard<std::mutex> lock(mutex_);
        const auto* ptrs =
            prepared.pointer_tensor.template data_ptr<std::int64_t>();
        const std::size_t batch_size = sequence_ids.size();
        for (std::size_t seq_idx = 0; seq_idx < batch_size; ++seq_idx) {
            const std::int32_t state_item_id =
                SequenceStateItemLocked(sequence_ids[seq_idx]);
            for (std::size_t layer = 0; layer < config_.num_layers; ++layer) {
                const std::size_t ptr_index = layer * batch_size + seq_idx;
                const std::int64_t device_ptr = ptrs[ptr_index];
                if (device_ptr == 0) {
                    std::ostringstream oss;
                    oss << op_name
                        << ": state_device_ptrs contains null pointer";
                    throw std::invalid_argument(oss.str());
                }
                prepared.state_items.push_back(
                    {StateItemPtrPhysical(layer, state_item_id),
                     reinterpret_cast<std::byte*>(
                         static_cast<std::uintptr_t>(device_ptr))});
            }
        }
        return prepared;
    }

    [[nodiscard]] std::size_t RowBytes(const torch::Tensor& tensor,
                                       std::string_view op_name) const {
        if (tensor.dim() < 1) {
            throw std::invalid_argument("state tensor must have batch dim");
        }
        const std::size_t batch = static_cast<std::size_t>(tensor.size(0));
        if (batch == 0) {
            return config_.state_token_bytes;
        }
        const std::size_t tensor_bytes =
            static_cast<std::size_t>(tensor.numel()) *
            static_cast<std::size_t>(tensor.element_size());
        if (tensor_bytes % batch != 0) {
            std::ostringstream oss;
            oss << op_name << ": state_tensor row bytes are not integral";
            throw std::invalid_argument(oss.str());
        }
        const std::size_t row_bytes = tensor_bytes / batch;
        if (row_bytes != config_.state_token_bytes) {
            std::ostringstream oss;
            oss << op_name << ": state tensor row has " << row_bytes
                << " bytes, expected " << config_.state_token_bytes;
            throw std::invalid_argument(oss.str());
        }
        return row_bytes;
    }

    [[nodiscard]] static std::byte* TensorRowPtr(torch::Tensor& tensor,
                                                 std::size_t batch_idx,
                                                 std::size_t row_bytes) {
        return static_cast<std::byte*>(tensor.data_ptr()) +
               batch_idx * row_bytes;
    }

    static void CopyDecodeRows(const PreparedDecodeRows& prepared,
                               CopyDirection direction) {
        if (prepared.device_index >= 0) {
            CUDA_CHECK(cudaSetDevice(prepared.device_index));
        }
        for (const DecodeRowCopy& row : prepared.rows) {
            if (direction == CopyDirection::kDeviceToHost) {
                CopyBytes(row.tensor_ptr, row.host_ptr, prepared.row_bytes);
            } else {
                CopyBytes(row.host_ptr, row.tensor_ptr, prepared.row_bytes);
            }
        }
    }

    static void CopyStateItems(const PreparedStateItemRows& prepared,
                               CopyDirection direction) {
        if (prepared.device_index >= 0) {
            CUDA_CHECK(cudaSetDevice(prepared.device_index));
        }
        for (const StateItemCopy& state_item : prepared.state_items) {
            if (direction == CopyDirection::kDeviceToHost) {
                CopyBytes(state_item.device_ptr, state_item.host_ptr,
                          prepared.state_item_bytes);
            } else {
                CopyBytes(state_item.host_ptr, state_item.device_ptr,
                          prepared.state_item_bytes);
            }
        }
    }

    static void CopyBytes(const std::byte* src, std::byte* dst,
                          std::size_t bytes) {
        if (bytes == 0) {
            return;
        }
        CUDA_CHECK(cudaMemcpy(static_cast<void*>(dst),
                              static_cast<const void*>(src), bytes,
                              cudaMemcpyDefault));
    }

    void TrackTask(const KVAsyncTask& task) {
        std::lock_guard<std::mutex> lock(mutex_);
        pending_tasks_.Track(task);
    }

    CompressedStateHostConfig config_;
    std::size_t state_item_bytes_ = 0;
    std::size_t state_item_stride_bytes_ = 0;
    std::size_t layer_stride_bytes_ = 0;
    std::size_t total_bytes_ = 0;
    std::shared_ptr<spdlog::logger> logger_;

    mutable std::mutex mutex_;
    bool initialized_ = false;
    bool pinned_ = false;
    int device_index_ = -1;
    std::vector<std::byte> storage_;
    std::vector<std::int32_t> free_state_items_;
    std::unordered_map<std::int64_t, SequenceState> sequence_states_;
    transformed_detail::PendingHostWriteTasks pending_tasks_;
};

using OverlapCompressedState4HostManager =
    CompressedStateHostManager<4, true>;
using NonOverlapCompressedState4HostManager =
    CompressedStateHostManager<4, false>;
using OverlapCompressedState128HostManager =
    CompressedStateHostManager<128, true>;
using NonOverlapCompressedState128HostManager =
    CompressedStateHostManager<128, false>;

}  // namespace batchgen::kv

#endif  // COMPRESSED_STATE_HOST_MANAGER_H_
