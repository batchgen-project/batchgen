#ifndef SWA_HOST_PAGED_KV_WORKER_VIEW_H_
#define SWA_HOST_PAGED_KV_WORKER_VIEW_H_

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <future>
#include <limits>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "host_paged_kv_worker_view.h"
#include "transformed_host_paged_kv_utils.h"

namespace batchgen::kv {

template <typename BaseView>
class SWAHostPagedKVWorkerView {
   public:
    using BatchedKVEntry = typename BaseView::BatchedKVEntry;
    static constexpr bool kHasVCache = BaseView::kHasVCache;
    static constexpr bool kUsesLogicalLayerMapping =
        BaseView::kUsesLogicalLayerMapping;

    SWAHostPagedKVWorkerView(const EngineConfig& engine_config,
                             const ModelConfig& model_config,
                             std::size_t window_size_tokens)
        : base_view_(engine_config, model_config),
          page_size_tokens_(base_view_.config().page_size_tokens),
          window_size_tokens_(window_size_tokens) {
        ValidateWindowConfig();
    }

    explicit SWAHostPagedKVWorkerView(const HostPagedKVConfig& config,
                                      std::size_t window_size_tokens)
        : base_view_(config),
          page_size_tokens_(base_view_.config().page_size_tokens),
          window_size_tokens_(window_size_tokens) {
        ValidateWindowConfig();
    }

    SWAHostPagedKVWorkerView(const SWAHostPagedKVWorkerView&) = delete;
    SWAHostPagedKVWorkerView& operator=(const SWAHostPagedKVWorkerView&) =
        delete;
    SWAHostPagedKVWorkerView(SWAHostPagedKVWorkerView&&) = delete;
    SWAHostPagedKVWorkerView& operator=(SWAHostPagedKVWorkerView&&) = delete;

    void Initialize(int device_index, bool create_region = false) {
        base_view_.Initialize(device_index, create_region);
    }

    void Shutdown() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            pending_host_writes_.Drain();
            sequence_states_.clear();
        }
        base_view_.Shutdown();
    }

    std::byte* DataBase() { return base_view_.DataBase(); }
    const std::byte* DataBase() const { return base_view_.DataBase(); }

    void* KPagePtr(std::size_t layer_idx, std::int32_t page_idx) {
        return base_view_.KPagePtr(layer_idx, page_idx);
    }

    const void* KPagePtr(std::size_t layer_idx,
                         std::int32_t page_idx) const {
        return base_view_.KPagePtr(layer_idx, page_idx);
    }

    template <bool Enabled = kHasVCache, typename = std::enable_if_t<Enabled>>
    void* VPagePtr(std::size_t layer_idx, std::int32_t page_idx) {
        return base_view_.VPagePtr(layer_idx, page_idx);
    }

    template <bool Enabled = kHasVCache, typename = std::enable_if_t<Enabled>>
    const void* VPagePtr(std::size_t layer_idx,
                         std::int32_t page_idx) const {
        return base_view_.VPagePtr(layer_idx, page_idx);
    }

    [[nodiscard]] std::size_t ResolvePhysicalLayer(
        std::size_t logical_layer_idx, std::string_view context) const {
        return base_view_.ResolvePhysicalLayer(logical_layer_idx, context);
    }

    const HostPagedKVConfig& config() const { return base_view_.config(); }
    const auto& layout() const { return base_view_.layout(); }
    HostPagedKVStats GetStats() const { return base_view_.GetStats(); }
    int device_index() const { return base_view_.device_index(); }
    std::size_t page_size_tokens() const { return page_size_tokens_; }
    std::size_t window_size_tokens() const { return window_size_tokens_; }
    std::size_t window_pages() const { return window_pages_; }

    std::string DebugString() const {
        std::ostringstream oss;
        oss << "SWAHostPagedKVWorkerView(window_size_tokens="
            << window_size_tokens_ << ", page_size_tokens="
            << page_size_tokens_ << ", window_pages=" << window_pages_
            << ", base=" << base_view_.DebugString() << ")";
        return oss.str();
    }

    std::vector<std::vector<std::int32_t>> BuildPageTable(
        const std::vector<std::int64_t>& sequence_ids) const {
        return base_view_.BuildPageTable(sequence_ids);
    }

    std::pair<std::vector<void*>, std::optional<std::vector<void*>>>
    GetSequenceLayerPagePointers(
        std::int64_t sequence_id, std::size_t layer_idx,
        std::optional<std::size_t> max_tokens = std::nullopt) const {
        return base_view_.GetSequenceLayerPagePointers(sequence_id, layer_idx,
                                                       max_tokens);
    }

    void RegisterSequences(const std::vector<std::int64_t>& sequence_ids) {
        base_view_.RegisterSequences(sequence_ids);
        std::lock_guard<std::mutex> lock(mutex_);
        for (std::int64_t sequence_id : sequence_ids) {
            sequence_states_.try_emplace(sequence_id);
        }
    }

    void UnregisterSequence(std::int64_t sequence_id) {
        base_view_.UnregisterSequence(sequence_id);
        std::lock_guard<std::mutex> lock(mutex_);
        sequence_states_.erase(sequence_id);
    }

    void UnregisterSequences(const std::vector<std::int64_t>& sequence_ids) {
        base_view_.UnregisterSequences(sequence_ids);
        std::lock_guard<std::mutex> lock(mutex_);
        for (std::int64_t sequence_id : sequence_ids) {
            sequence_states_.erase(sequence_id);
        }
    }

    std::vector<std::vector<std::int32_t>> AllocatePagesForSequences(
        const std::vector<std::int64_t>& sequence_ids,
        const std::vector<std::size_t>& raw_num_tokens) {
        if (sequence_ids.size() != raw_num_tokens.size()) {
            throw std::invalid_argument(
                "sequence_ids and raw_num_tokens must have the same length");
        }
        std::vector<std::size_t> active_tokens;
        active_tokens.reserve(raw_num_tokens.size());
        std::vector<WindowForRawEnd> windows;
        windows.reserve(raw_num_tokens.size());
        for (std::size_t raw_tokens : raw_num_tokens) {
            const auto window = ComputeWindowForRawEnd(raw_tokens);
            if (window.active_tokens == 0) {
                throw std::invalid_argument(
                    "AllocatePagesForSequences: raw_num_tokens entries must "
                    "be greater than zero");
            }
            active_tokens.push_back(window.active_tokens);
            windows.push_back(window);
        }

        auto allocations =
            base_view_.AllocatePagesForSequences(sequence_ids, active_tokens);
        std::lock_guard<std::mutex> lock(mutex_);
        for (std::size_t i = 0; i < sequence_ids.size(); ++i) {
            auto& state = sequence_states_[sequence_ids[i]];
            state.window_start_page = windows[i].window_start_page;
            state.active_pages = windows[i].required_pages;
            state.max_seen_raw_pos = raw_num_tokens[i] - 1;
            state.has_tokens = true;
        }
        return allocations;
    }

    void ReleaseSequencePages(const std::vector<std::int64_t>& sequence_ids) {
        std::lock_guard<std::mutex> lock(mutex_);
        pending_host_writes_.Drain();
        base_view_.ReleaseSequencePages(sequence_ids);
        for (std::int64_t sequence_id : sequence_ids) {
            sequence_states_.erase(sequence_id);
        }
    }

    KVAsyncTask AsyncLoadLayerKVToDevice(
        torch::Tensor sequence_ids, torch::Tensor k_device_ptrs,
        std::optional<torch::Tensor> v_device_ptrs = std::nullopt) {
        return base_view_.AsyncLoadLayerKVToDevice(
            std::move(sequence_ids), std::move(k_device_ptrs),
            std::move(v_device_ptrs));
    }

    KVAsyncTask AsyncLoadLayerPagedKVToDevice(
        torch::Tensor sequence_ids, torch::Tensor active_page_counts,
        torch::Tensor k_device_ptrs,
        std::optional<torch::Tensor> v_device_ptrs = std::nullopt) {
        return base_view_.AsyncLoadLayerPagedKVToDevice(
            std::move(sequence_ids), std::move(active_page_counts),
            std::move(k_device_ptrs), std::move(v_device_ptrs));
    }

    KVAsyncTask AsyncOffloadPagedKVToHost(
        torch::Tensor gpu_page_ptrs, torch::Tensor host_page_ptrs,
        std::size_t page_bytes) {
        return base_view_.AsyncOffloadPagedKVToHost(
            std::move(gpu_page_ptrs), std::move(host_page_ptrs), page_bytes);
    }

    KVAsyncTask AsyncOffloadLayerKVToHost(
        std::size_t layer_idx, std::vector<std::int64_t> sequence_ids,
        torch::Tensor k_tensor, std::optional<torch::Tensor> v_tensor,
        SequenceLengths raw_sequence_lengths) {
        if (sequence_ids.empty()) {
            return transformed_detail::MakeAsyncTask([] {});
        }
        std::vector<KVAsyncTask> tasks;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            const std::size_t batch = sequence_ids.size();
            for (std::size_t batch_idx = 0; batch_idx < batch; ++batch_idx) {
                const std::int64_t sequence_id = sequence_ids[batch_idx];
                const std::size_t raw_tokens =
                    transformed_detail::ResolveLength(
                        raw_sequence_lengths, batch_idx, sequence_id,
                        "SWAHostPagedKVWorkerView::"
                        "AsyncOffloadLayerKVToHost");
                const auto active_tokens =
                    UpdateWindowForRawEndLocked(sequence_id, raw_tokens);
                if (active_tokens == 0) {
                    continue;
                }
                const auto source_start =
                    static_cast<std::int64_t>(raw_tokens - active_tokens);
                auto k_slice = k_tensor
                                   .narrow(0, static_cast<std::int64_t>(
                                                  batch_idx),
                                           1)
                                   .narrow(1, source_start,
                                           static_cast<std::int64_t>(
                                               active_tokens))
                                   .contiguous();
                std::optional<torch::Tensor> v_slice;
                if (v_tensor.has_value()) {
                    v_slice = v_tensor->narrow(
                                          0, static_cast<std::int64_t>(
                                                 batch_idx),
                                          1)
                                  .narrow(1, source_start,
                                          static_cast<std::int64_t>(
                                              active_tokens))
                                  .contiguous();
                }
                auto task = base_view_.AsyncOffloadLayerKVToHost(
                    layer_idx, {sequence_id}, std::move(k_slice),
                    std::move(v_slice), SequenceLengthVector{active_tokens});
                pending_host_writes_.Track(task);
                tasks.emplace_back(std::move(task));
            }
        }
        return transformed_detail::MakeCombinedTask(std::move(tasks));
    }

    KVAsyncTask AsyncAppendDecodeKVToHost(
        std::size_t layer_idx, std::vector<std::int64_t> sequence_ids,
        torch::Tensor k_tensor, std::optional<torch::Tensor> v_tensor,
        SequenceLengths raw_positions) {
        if (sequence_ids.empty()) {
            return transformed_detail::MakeAsyncTask([] {});
        }
        KVAsyncTask task;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto storage_positions =
                PrepareStoragePositionsLocked(sequence_ids, raw_positions);
            task = base_view_.AsyncAppendDecodeKVToHost(
                layer_idx, std::move(sequence_ids), std::move(k_tensor),
                std::move(v_tensor), std::move(storage_positions));
            pending_host_writes_.Track(task);
        }
        return task;
    }

    KVAsyncTask AsyncAppendDecodeKVToHostBatchedKernel(
        std::vector<BatchedKVEntry> entries,
        std::vector<std::int64_t> sequence_ids, SequenceLengths raw_positions) {
        if (entries.empty() || sequence_ids.empty()) {
            return transformed_detail::MakeAsyncTask([] {});
        }
        KVAsyncTask task;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto storage_positions =
                PrepareStoragePositionsLocked(sequence_ids, raw_positions);
            task = base_view_.AsyncAppendDecodeKVToHostBatchedKernel(
                std::move(entries), std::move(sequence_ids),
                std::move(storage_positions));
            pending_host_writes_.Track(task);
        }
        return task;
    }

    std::pair<torch::Tensor, torch::Tensor> ReadSequenceKVToCPU(
        std::int64_t sequence_id) const {
        return base_view_.ReadSequenceKVToCPU(sequence_id);
    }

    void WriteSequenceKVFromCPU(
        std::int64_t sequence_id, const torch::Tensor& k_tensor,
        const std::optional<torch::Tensor>& v_tensor = std::nullopt) {
        base_view_.WriteSequenceKVFromCPU(sequence_id, k_tensor, v_tensor);
    }

   private:
    struct SWASequenceState {
        std::size_t window_start_page = 0;
        std::size_t active_pages = 0;
        std::size_t max_seen_raw_pos = 0;
        bool has_tokens = false;
    };

    struct WindowForRawEnd {
        std::size_t window_start_page = 0;
        std::size_t active_tokens = 0;
        std::size_t required_pages = 0;
    };

    void ValidateWindowConfig() {
        if (page_size_tokens_ == 0) {
            throw std::invalid_argument(
                "SWAHostPagedKVWorkerView requires page_size_tokens > 0");
        }
        if (window_size_tokens_ == 0) {
            throw std::invalid_argument(
                "SWAHostPagedKVWorkerView requires window_size_tokens > 0");
        }
        if (window_size_tokens_ % page_size_tokens_ != 0) {
            throw std::invalid_argument(
                "SWAHostPagedKVWorkerView requires window_size_tokens to be "
                "divisible by page_size_tokens");
        }
        window_pages_ = window_size_tokens_ / page_size_tokens_;
    }

    WindowForRawEnd ComputeWindowForRawEnd(std::size_t raw_end_tokens) const {
        if (raw_end_tokens == 0) {
            return {};
        }
        const std::size_t first_needed_token =
            raw_end_tokens > window_size_tokens_
                ? raw_end_tokens - window_size_tokens_
                : 0;
        const std::size_t window_start_page =
            first_needed_token / page_size_tokens_;
        const std::size_t window_start_token =
            window_start_page * page_size_tokens_;
        const std::size_t active_tokens =
            raw_end_tokens - window_start_token;
        const std::size_t required_pages =
            (active_tokens + page_size_tokens_ - 1) / page_size_tokens_;
        if (required_pages > window_pages_ + 1) {
            std::ostringstream oss;
            oss << "SWAHostPagedKVWorkerView: active pages "
                << required_pages << " exceed window_pages + 1 ("
                << (window_pages_ + 1) << ")";
            throw std::logic_error(oss.str());
        }
        return {window_start_page, active_tokens, required_pages};
    }

    std::size_t UpdateWindowForRawEndLocked(std::int64_t sequence_id,
                                            std::size_t raw_end_tokens) {
        const auto window = ComputeWindowForRawEnd(raw_end_tokens);
        auto& state = sequence_states_[sequence_id];
        if (state.has_tokens &&
            window.window_start_page < state.window_start_page) {
            throw std::out_of_range(
                "SWAHostPagedKVWorkerView does not support writing a raw "
                "token range that is older than the current SWA window");
        }
        if (state.has_tokens &&
            window.window_start_page > state.window_start_page) {
            const std::size_t pages_to_release =
                window.window_start_page - state.window_start_page;
            if (pages_to_release > state.active_pages) {
                std::ostringstream oss;
                oss << "SWAHostPagedKVWorkerView: sequence " << sequence_id
                    << " cannot release " << pages_to_release
                    << " pages with only " << state.active_pages
                    << " active pages";
                throw std::out_of_range(oss.str());
            }
            pending_host_writes_.Drain();
            base_view_.ReleaseSequencePrefixPages(sequence_id,
                                                  pages_to_release);
            state.active_pages -= pages_to_release;
        }
        state.window_start_page = window.window_start_page;
        EnsureCapacityForActivePagesLocked(sequence_id, state,
                                           window.required_pages);
        if (raw_end_tokens > 0) {
            state.max_seen_raw_pos =
                std::max(state.max_seen_raw_pos, raw_end_tokens - 1);
            state.has_tokens = true;
        }
        return window.active_tokens;
    }

    void EnsureCapacityForActivePagesLocked(std::int64_t sequence_id,
                                            SWASequenceState& state,
                                            std::size_t required_pages) {
        if (required_pages == 0) {
            return;
        }
        if (required_pages > window_pages_ + 1) {
            std::ostringstream oss;
            oss << "SWAHostPagedKVWorkerView: sequence " << sequence_id
                << " requires " << required_pages
                << " active pages, exceeding window_pages + 1 ("
                << (window_pages_ + 1) << ")";
            throw std::out_of_range(oss.str());
        }
        if (state.active_pages < required_pages) {
            const std::size_t missing_pages =
                required_pages - state.active_pages;
            base_view_.GrowSequencePages(sequence_id,
                                         missing_pages);
            state.active_pages += missing_pages;
        }
    }

    SequenceLengthVector PrepareStoragePositionsLocked(
        const std::vector<std::int64_t>& sequence_ids,
        const SequenceLengths& raw_positions) {
        SequenceLengthVector storage_positions;
        storage_positions.reserve(sequence_ids.size());
        for (std::size_t batch_idx = 0; batch_idx < sequence_ids.size();
             ++batch_idx) {
            const std::int64_t sequence_id = sequence_ids[batch_idx];
            const std::size_t raw_pos = transformed_detail::ResolveLength(
                raw_positions, batch_idx, sequence_id,
                "SWAHostPagedKVWorkerView::AsyncAppendDecodeKVToHost");
            if (raw_pos == std::numeric_limits<std::size_t>::max()) {
                throw std::out_of_range(
                    "SWAHostPagedKVWorkerView: raw position overflow");
            }
            const std::size_t active_tokens_after =
                UpdateWindowForRawEndLocked(sequence_id, raw_pos + 1);
            storage_positions.push_back(active_tokens_after - 1);
        }
        return storage_positions;
    }

    BaseView base_view_;
    std::size_t page_size_tokens_ = 0;
    std::size_t window_size_tokens_ = 0;
    std::size_t window_pages_ = 0;
    mutable std::mutex mutex_;
    std::unordered_map<std::int64_t, SWASequenceState> sequence_states_;
    transformed_detail::PendingHostWriteTasks pending_host_writes_;
};

using SWADefaultHostPagedKVWorkerView =
    SWAHostPagedKVWorkerView<DefaultHostPagedKVWorkerView>;
using SWAMLAHostPagedKVWorkerView =
    SWAHostPagedKVWorkerView<MLAHostPagedKVWorkerView>;
using SWAMappedDefaultHostPagedKVWorkerView =
    SWAHostPagedKVWorkerView<MappedDefaultHostPagedKVWorkerView>;
using SWAMappedMLAHostPagedKVWorkerView =
    SWAHostPagedKVWorkerView<MappedMLAHostPagedKVWorkerView>;

}  // namespace batchgen::kv

#endif  // SWA_HOST_PAGED_KV_WORKER_VIEW_H_
