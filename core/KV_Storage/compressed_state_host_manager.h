#ifndef COMPRESSED_STATE_HOST_MANAGER_H_
#define COMPRESSED_STATE_HOST_MANAGER_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
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
    std::size_t num_pages = 0;
    std::size_t state_page_size_tokens = 0;
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
        << ", num_pages=" << config.num_pages
        << ", state_page_size_tokens=" << config.state_page_size_tokens
        << ", ring_size=" << config.ring_size
        << ", state_token_bytes=" << config.state_token_bytes
        << ", sequence_table_capacity=" << config.sequence_table_capacity
        << ", alignment_bytes=" << config.alignment_bytes
        << ", logical_to_physical_layer=";
    AppendLogicalLayerMapping(oss, config.logical_to_physical_layer);
    oss << ")";
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
          page_bytes_(config_.ring_size * config_.state_token_bytes),
          page_stride_bytes_(
              detail::AlignUp(page_bytes_, config_.alignment_bytes)),
          layer_stride_bytes_(detail::AlignUp(
              config_.num_pages * page_stride_bytes_,
              config_.alignment_bytes)),
          total_bytes_(config_.num_layers * layer_stride_bytes_),
          logger_(init_logger(
              "info", config_.logger_name.empty()
                          ? "CompressedStateHostManager"
                          : config_.logger_name)) {
        ResetFreePages();
    }

    CompressedStateHostManager(const CompressedStateHostManager&) = delete;
    CompressedStateHostManager& operator=(const CompressedStateHostManager&) =
        delete;
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
            "overlap={}, layers={}, pages={}, page_bytes={}, "
            "total_bytes={})",
            device_index_, Ratio, Overlap, config_.num_layers,
            config_.num_pages, page_bytes_, total_bytes_);
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
        ResetFreePages();
    }

    std::vector<std::vector<std::int32_t>> AllocatePagesForSequences(
        const std::vector<std::int64_t>& sequence_ids,
        const std::vector<std::size_t>& raw_num_tokens) {
        EnsureInitialized();
        if (sequence_ids.size() != raw_num_tokens.size()) {
            throw std::invalid_argument(
                "sequence_ids and raw_num_tokens must have the same length");
        }
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<std::vector<std::int32_t>> allocations;
        allocations.reserve(sequence_ids.size());
        for (std::size_t i = 0; i < sequence_ids.size(); ++i) {
            allocations.push_back(EnsureCapacityLocked(sequence_ids[i],
                                                       raw_num_tokens[i]));
        }
        return allocations;
    }

    std::vector<std::int32_t> AllocatePages(std::int64_t sequence_id,
                                            std::size_t raw_num_tokens) {
        auto allocations = AllocatePagesForSequences({sequence_id},
                                                     {raw_num_tokens});
        return allocations.empty() ? std::vector<std::int32_t>{}
                                   : std::move(allocations.front());
    }

    void ReleaseSequencePages(
        const std::vector<std::int64_t>& sequence_ids) {
        EnsureInitialized();
        std::lock_guard<std::mutex> lock(mutex_);
        pending_tasks_.Drain();
        for (std::int64_t sequence_id : sequence_ids) {
            auto it = sequence_states_.find(sequence_id);
            if (it == sequence_states_.end()) {
                continue;
            }
            for (auto page_it = it->second.pages.rbegin();
                 page_it != it->second.pages.rend(); ++page_it) {
                free_pages_.push_back(*page_it);
            }
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

    KVAsyncTask AsyncOffloadStatePagesToHost(
        std::vector<std::int64_t> sequence_ids,
        std::vector<std::size_t> active_page_counts,
        torch::Tensor state_device_ptrs) {
        auto prepared = PreparePageRows(std::move(sequence_ids),
                                        std::move(active_page_counts),
                                        std::move(state_device_ptrs),
                                        "AsyncOffloadStatePagesToHost");
        auto task = transformed_detail::MakeAsyncTask(
            [prepared = std::move(prepared)]() mutable {
                CopyPages(prepared, CopyDirection::kDeviceToHost);
            });
        TrackTask(task);
        return task;
    }

    KVAsyncTask AsyncLoadStatePagesToDevice(
        std::vector<std::int64_t> sequence_ids,
        std::vector<std::size_t> active_page_counts,
        torch::Tensor state_device_ptrs) {
        auto prepared = PreparePageRows(std::move(sequence_ids),
                                        std::move(active_page_counts),
                                        std::move(state_device_ptrs),
                                        "AsyncLoadStatePagesToDevice");
        auto task = transformed_detail::MakeAsyncTask(
            [prepared = std::move(prepared)]() mutable {
                CopyPages(prepared, CopyDirection::kHostToDevice);
            });
        TrackTask(task);
        return task;
    }

    [[nodiscard]] std::vector<std::vector<std::int32_t>> BuildPageTable(
        const std::vector<std::int64_t>& sequence_ids) const {
        EnsureInitialized();
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<std::vector<std::int32_t>> table;
        table.reserve(sequence_ids.size());
        for (std::int64_t sequence_id : sequence_ids) {
            table.push_back(SequencePagesLocked(sequence_id));
        }
        return table;
    }

    [[nodiscard]] std::vector<std::uintptr_t>
    GetSequenceLayerStatePagePointers(std::int64_t sequence_id,
                                      std::size_t layer_idx) const {
        EnsureInitialized();
        const std::size_t physical_layer =
            ResolvePhysicalLayer(layer_idx,
                                 "GetSequenceLayerStatePagePointers");
        std::lock_guard<std::mutex> lock(mutex_);
        const auto pages = SequencePagesLocked(sequence_id);
        std::vector<std::uintptr_t> ptrs;
        ptrs.reserve(pages.size());
        for (std::int32_t page : pages) {
            ptrs.push_back(reinterpret_cast<std::uintptr_t>(
                StatePagePtrPhysical(physical_layer, page)));
        }
        return ptrs;
    }

    [[nodiscard]] std::uintptr_t StatePagePtr(std::size_t layer_idx,
                                              std::int32_t page_idx) {
        EnsureInitialized();
        const std::size_t physical_layer =
            ResolvePhysicalLayer(layer_idx, "StatePagePtr");
        return reinterpret_cast<std::uintptr_t>(
            StatePagePtrPhysical(physical_layer, page_idx));
    }

    [[nodiscard]] std::int64_t ResolveStateSlot(std::int64_t sequence_id,
                                                std::size_t raw_position) {
        EnsureInitialized();
        std::lock_guard<std::mutex> lock(mutex_);
        const SlotAddress slot =
            ResolveSlotLocked(sequence_id, raw_position, 0,
                              "ResolveStateSlot");
        return static_cast<std::int64_t>(
            slot.page_id * static_cast<std::int32_t>(config_.ring_size) +
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

    [[nodiscard]] HostPagedKVStats GetStats() const {
        EnsureInitialized();
        std::lock_guard<std::mutex> lock(mutex_);
        HostPagedKVStats stats;
        stats.num_total_pages = config_.num_pages;
        stats.num_free_pages = free_pages_.size();
        stats.num_used_pages = config_.num_pages - free_pages_.size();
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
    [[nodiscard]] std::size_t state_page_size_tokens() const {
        return config_.state_page_size_tokens;
    }
    [[nodiscard]] std::size_t ring_size() const { return config_.ring_size; }
    [[nodiscard]] std::size_t state_token_bytes() const {
        return config_.state_token_bytes;
    }
    [[nodiscard]] std::size_t page_bytes() const { return page_bytes_; }
    [[nodiscard]] std::size_t page_stride_bytes() const {
        return page_stride_bytes_;
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
        std::vector<std::int32_t> pages;
    };

    struct SlotAddress {
        std::int32_t page_id = -1;
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

    struct PageCopy {
        std::byte* host_ptr = nullptr;
        std::byte* device_ptr = nullptr;
    };

    struct PreparedPageRows {
        torch::Tensor pointer_tensor;
        std::vector<PageCopy> pages;
        std::size_t page_bytes = 0;
        int device_index = -1;
    };

    static CompressedStateHostConfig SanitizeConfig(
        CompressedStateHostConfig config) {
        if (config.sequence_table_capacity == 0) {
            config.sequence_table_capacity = config.num_pages;
        }
        if (config.alignment_bytes == 0) {
            config.alignment_bytes = 64;
        }
        std::vector<std::string> errors;
        if (config.num_layers == 0) {
            errors.emplace_back("num_layers must be > 0");
        }
        if (config.num_pages == 0) {
            errors.emplace_back("num_pages must be > 0");
        }
        if (config.state_page_size_tokens == 0) {
            errors.emplace_back("state_page_size_tokens must be > 0");
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
        if (config.ring_size != 0 && config.state_page_size_tokens != 0 &&
            config.state_page_size_tokens % config.ring_size != 0) {
            errors.emplace_back(
                "state_page_size_tokens must be divisible by ring_size");
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

    void ResetFreePages() {
        free_pages_.clear();
        free_pages_.reserve(config_.num_pages);
        for (std::int32_t page =
                 static_cast<std::int32_t>(config_.num_pages);
             page > 0; --page) {
            free_pages_.push_back(page - 1);
        }
    }

    [[nodiscard]] std::size_t RequiredPages(std::size_t raw_tokens) const {
        raw_tokens = std::max<std::size_t>(1, raw_tokens);
        return (raw_tokens + config_.state_page_size_tokens - 1) /
               config_.state_page_size_tokens;
    }

    std::vector<std::int32_t> EnsureCapacityLocked(
        std::int64_t sequence_id, std::size_t raw_num_tokens) {
        const std::size_t required_pages = RequiredPages(raw_num_tokens);
        auto& state = sequence_states_[sequence_id];
        const std::size_t current_pages = state.pages.size();
        if (required_pages <= current_pages) {
            return {};
        }
        const std::size_t missing = required_pages - current_pages;
        if (missing > free_pages_.size()) {
            std::ostringstream oss;
            oss << "CompressedStateHostManager: insufficient free pages "
                << "(need=" << missing << ", free=" << free_pages_.size()
                << ")";
            throw std::runtime_error(oss.str());
        }
        std::vector<std::int32_t> new_pages;
        new_pages.reserve(missing);
        for (std::size_t i = 0; i < missing; ++i) {
            new_pages.push_back(free_pages_.back());
            free_pages_.pop_back();
        }
        state.pages.insert(state.pages.end(), new_pages.begin(),
                           new_pages.end());
        return new_pages;
    }

    [[nodiscard]] std::vector<std::int32_t> SequencePagesLocked(
        std::int64_t sequence_id) const {
        const auto it = sequence_states_.find(sequence_id);
        if (it == sequence_states_.end()) {
            std::ostringstream oss;
            oss << "CompressedStateHostManager: sequence " << sequence_id
                << " has no allocated state pages";
            throw std::out_of_range(oss.str());
        }
        return it->second.pages;
    }

    [[nodiscard]] SlotAddress ResolveSlotLocked(
        std::int64_t sequence_id, std::size_t raw_position,
        std::size_t physical_layer, std::string_view context) {
        const auto it = sequence_states_.find(sequence_id);
        if (it == sequence_states_.end()) {
            std::ostringstream oss;
            oss << context << ": sequence " << sequence_id
                << " has no allocated state pages";
            throw std::out_of_range(oss.str());
        }
        const std::size_t page_ordinal =
            raw_position / config_.state_page_size_tokens;
        if (page_ordinal >= it->second.pages.size()) {
            std::ostringstream oss;
            oss << context << ": raw position " << raw_position
                << " maps to page ordinal " << page_ordinal
                << " but sequence " << sequence_id << " has only "
                << it->second.pages.size() << " pages";
            throw std::out_of_range(oss.str());
        }
        const std::int32_t page_id = it->second.pages[page_ordinal];
        const std::size_t ring_offset = raw_position % config_.ring_size;
        return {page_id, ring_offset,
                StateSlotPtrPhysical(physical_layer, page_id, ring_offset)};
    }

    [[nodiscard]] std::byte* StatePagePtrPhysical(
        std::size_t physical_layer, std::int32_t page_idx) const {
        if (physical_layer >= config_.num_layers) {
            throw std::out_of_range("physical layer out of range");
        }
        if (page_idx < 0 ||
            static_cast<std::size_t>(page_idx) >= config_.num_pages) {
            throw std::out_of_range("state page id out of range");
        }
        return const_cast<std::byte*>(
            storage_.data() + physical_layer * layer_stride_bytes_ +
            static_cast<std::size_t>(page_idx) * page_stride_bytes_);
    }

    [[nodiscard]] std::byte* StateSlotPtrPhysical(
        std::size_t physical_layer, std::int32_t page_idx,
        std::size_t ring_offset) const {
        if (ring_offset >= config_.ring_size) {
            throw std::out_of_range("ring offset out of range");
        }
        return StatePagePtrPhysical(physical_layer, page_idx) +
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
            EnsureCapacityLocked(sequence_id, raw_position + 1);
            const SlotAddress slot =
                ResolveSlotLocked(sequence_id, raw_position, physical_layer,
                                  op_name);
            prepared.rows.push_back(
                {slot.host_ptr,
                 TensorRowPtr(prepared.tensor, batch_idx, row_bytes)});
        }
        return prepared;
    }

    PreparedPageRows PreparePageRows(
        std::vector<std::int64_t> sequence_ids,
        std::vector<std::size_t> active_page_counts,
        torch::Tensor state_device_ptrs, std::string_view op_name) {
        EnsureInitialized();
        if (sequence_ids.size() != active_page_counts.size()) {
            throw std::invalid_argument(
                "sequence_ids and active_page_counts must have the same "
                "length");
        }
        if (state_device_ptrs.scalar_type() != torch::kInt64) {
            std::ostringstream oss;
            oss << op_name << ": state_device_ptrs must be int64";
            throw std::invalid_argument(oss.str());
        }
        if (state_device_ptrs.dim() != 3 ||
            state_device_ptrs.size(0) !=
                static_cast<std::int64_t>(config_.num_layers) ||
            state_device_ptrs.size(1) !=
                static_cast<std::int64_t>(sequence_ids.size())) {
            std::ostringstream oss;
            oss << op_name
                << ": state_device_ptrs must have shape "
                   "[num_layers, batch, max_pages]";
            throw std::invalid_argument(oss.str());
        }
        if (state_device_ptrs.device().is_cuda()) {
            state_device_ptrs =
                state_device_ptrs.to(torch::Device(torch::kCPU));
        }
        state_device_ptrs = state_device_ptrs.contiguous();
        const auto max_pages =
            static_cast<std::size_t>(state_device_ptrs.size(2));
        PreparedPageRows prepared;
        prepared.pointer_tensor = std::move(state_device_ptrs);
        prepared.page_bytes = page_bytes_;
        prepared.device_index = device_index_;
        std::lock_guard<std::mutex> lock(mutex_);
        const auto* ptrs =
            prepared.pointer_tensor.template data_ptr<std::int64_t>();
        const std::size_t layer_stride = sequence_ids.size() * max_pages;
        for (std::size_t seq_idx = 0; seq_idx < sequence_ids.size();
             ++seq_idx) {
            const auto pages = SequencePagesLocked(sequence_ids[seq_idx]);
            const std::size_t count = active_page_counts[seq_idx];
            if (count > pages.size() || count > max_pages) {
                std::ostringstream oss;
                oss << op_name << ": active_page_count " << count
                    << " exceeds allocated/capacity pages for sequence "
                    << sequence_ids[seq_idx];
                throw std::out_of_range(oss.str());
            }
            for (std::size_t layer = 0; layer < config_.num_layers; ++layer) {
                for (std::size_t page_ordinal = 0; page_ordinal < count;
                     ++page_ordinal) {
                    const std::size_t ptr_index =
                        layer * layer_stride + seq_idx * max_pages +
                        page_ordinal;
                    const std::int64_t device_ptr = ptrs[ptr_index];
                    if (device_ptr == 0) {
                        std::ostringstream oss;
                        oss << op_name
                            << ": state_device_ptrs contains null pointer";
                        throw std::invalid_argument(oss.str());
                    }
                    prepared.pages.push_back(
                        {StatePagePtrPhysical(layer, pages[page_ordinal]),
                         reinterpret_cast<std::byte*>(
                             static_cast<std::uintptr_t>(device_ptr))});
                }
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

    static void CopyPages(const PreparedPageRows& prepared,
                          CopyDirection direction) {
        if (prepared.device_index >= 0) {
            CUDA_CHECK(cudaSetDevice(prepared.device_index));
        }
        for (const PageCopy& page : prepared.pages) {
            if (direction == CopyDirection::kDeviceToHost) {
                CopyBytes(page.device_ptr, page.host_ptr,
                          prepared.page_bytes);
            } else {
                CopyBytes(page.host_ptr, page.device_ptr,
                          prepared.page_bytes);
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
    std::size_t page_bytes_ = 0;
    std::size_t page_stride_bytes_ = 0;
    std::size_t layer_stride_bytes_ = 0;
    std::size_t total_bytes_ = 0;
    std::shared_ptr<spdlog::logger> logger_;

    mutable std::mutex mutex_;
    bool initialized_ = false;
    bool pinned_ = false;
    int device_index_ = -1;
    std::vector<std::byte> storage_;
    std::vector<std::int32_t> free_pages_;
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
