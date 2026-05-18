#ifndef COMPRESSED_RATIO_HOST_PAGED_KV_WORKER_VIEW_H_
#define COMPRESSED_RATIO_HOST_PAGED_KV_WORKER_VIEW_H_

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

namespace batchgen::kv {

template <typename BaseView, std::size_t CompressionRatio>
class CompressedRatioHostPagedKVWorkerView : public BaseView {
   public:
    static_assert(CompressionRatio > 0,
                  "CompressionRatio must be greater than zero");

    using BatchedKVEntry = typename BaseView::BatchedKVEntry;
    static constexpr bool kHasVCache = BaseView::kHasVCache;
    static constexpr bool kUsesLogicalLayerMapping =
        BaseView::kUsesLogicalLayerMapping;
    static constexpr std::size_t kCompressionRatio = CompressionRatio;

    CompressedRatioHostPagedKVWorkerView(
        const EngineConfig& engine_config, const ModelConfig& model_config)
        : BaseView(engine_config, model_config) {
        page_size_tokens_ = BaseView::config().page_size_tokens;
        ValidateConfig();
    }

    explicit CompressedRatioHostPagedKVWorkerView(
        const HostPagedKVConfig& config)
        : BaseView(config) {
        page_size_tokens_ = BaseView::config().page_size_tokens;
        ValidateConfig();
    }

    CompressedRatioHostPagedKVWorkerView(
        const CompressedRatioHostPagedKVWorkerView&) = delete;
    CompressedRatioHostPagedKVWorkerView& operator=(
        const CompressedRatioHostPagedKVWorkerView&) = delete;
    CompressedRatioHostPagedKVWorkerView(
        CompressedRatioHostPagedKVWorkerView&&) = delete;
    CompressedRatioHostPagedKVWorkerView& operator=(
        CompressedRatioHostPagedKVWorkerView&&) = delete;

    void Shutdown() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            DrainPendingHostWritesLocked();
            sequence_states_.clear();
        }
        BaseView::Shutdown();
    }

    std::size_t compression_ratio() const { return CompressionRatio; }

    std::string DebugString() const {
        std::ostringstream oss;
        oss << "CompressedRatioHostPagedKVWorkerView(compression_ratio="
            << CompressionRatio
            << ", base=" << BaseView::DebugString() << ")";
        return oss.str();
    }

    void UnregisterSequence(std::int64_t sequence_id) {
        BaseView::UnregisterSequence(sequence_id);
        std::lock_guard<std::mutex> lock(mutex_);
        sequence_states_.erase(sequence_id);
    }

    void UnregisterSequences(const std::vector<std::int64_t>& sequence_ids) {
        BaseView::UnregisterSequences(sequence_ids);
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

        std::vector<std::size_t> storage_tokens;
        storage_tokens.reserve(raw_num_tokens.size());
        std::vector<std::size_t> compressed_tokens;
        compressed_tokens.reserve(raw_num_tokens.size());
        for (std::size_t raw_tokens : raw_num_tokens) {
            const std::size_t tokens = CompressedTokens(raw_tokens);
            compressed_tokens.push_back(tokens);
            storage_tokens.push_back(StorageCapacityTokens(tokens));
        }

        auto allocations =
            BaseView::AllocatePagesForSequences(sequence_ids, storage_tokens);
        std::lock_guard<std::mutex> lock(mutex_);
        for (std::size_t i = 0; i < sequence_ids.size(); ++i) {
            auto& state = sequence_states_[sequence_ids[i]];
            state.allocated_pages =
                std::max(state.allocated_pages,
                         RequiredPagesForTokens(storage_tokens[i]));
            state.max_exposed_tokens = compressed_tokens[i];
        }
        return allocations;
    }

    std::vector<std::int32_t> GrowSequencePages(
        std::int64_t sequence_id, std::size_t num_pages) {
        auto pages = BaseView::GrowSequencePages(sequence_id, num_pages);
        std::lock_guard<std::mutex> lock(mutex_);
        sequence_states_[sequence_id].allocated_pages += pages.size();
        return pages;
    }

    std::vector<std::vector<std::int32_t>> GrowPagesForSequences(
        const std::vector<std::int64_t>& sequence_ids,
        const std::vector<std::size_t>& num_pages) {
        auto allocations =
            BaseView::GrowPagesForSequences(sequence_ids, num_pages);
        std::lock_guard<std::mutex> lock(mutex_);
        for (std::size_t i = 0; i < sequence_ids.size(); ++i) {
            sequence_states_[sequence_ids[i]].allocated_pages +=
                allocations[i].size();
        }
        return allocations;
    }

    void ReleaseSequencePages(const std::vector<std::int64_t>& sequence_ids) {
        std::lock_guard<std::mutex> lock(mutex_);
        DrainPendingHostWritesLocked();
        BaseView::ReleaseSequencePages(sequence_ids);
        for (std::int64_t sequence_id : sequence_ids) {
            sequence_states_.erase(sequence_id);
        }
    }

    KVAsyncTask AsyncOffloadLayerKVToHost(
        std::size_t layer_idx, std::vector<std::int64_t> sequence_ids,
        torch::Tensor k_tensor, std::optional<torch::Tensor> v_tensor,
        SequenceLengths raw_sequence_lengths) {
        if (sequence_ids.empty()) {
            return MakeAsyncTask([] {});
        }

        std::vector<std::int64_t> valid_sequence_ids;
        std::vector<std::int64_t> valid_rows;
        SequenceLengthVector compressed_lengths;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            PrepareOffloadRowsLocked(sequence_ids, raw_sequence_lengths,
                                     valid_sequence_ids, valid_rows,
                                     compressed_lengths);
        }
        if (valid_sequence_ids.empty()) {
            return MakeAsyncTask([] {});
        }

        auto prepared_k = SelectRows(k_tensor, valid_rows);
        std::optional<torch::Tensor> prepared_v;
        if (v_tensor.has_value()) {
            prepared_v = SelectRows(*v_tensor, valid_rows);
        }
        auto task = BaseView::AsyncOffloadLayerKVToHost(
            layer_idx, std::move(valid_sequence_ids), std::move(prepared_k),
            std::move(prepared_v), std::move(compressed_lengths));
        {
            std::lock_guard<std::mutex> lock(mutex_);
            TrackHostWriteTaskLocked(task);
        }
        return task;
    }

    KVAsyncTask AsyncAppendDecodeKVToHost(
        std::size_t layer_idx, std::vector<std::int64_t> sequence_ids,
        torch::Tensor k_tensor, std::optional<torch::Tensor> v_tensor,
        SequenceLengths raw_positions) {
        if (sequence_ids.empty()) {
            return MakeAsyncTask([] {});
        }

        std::vector<std::int64_t> valid_sequence_ids;
        std::vector<std::int64_t> valid_rows;
        SequenceLengthVector storage_positions;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            PrepareAppendRowsLocked(sequence_ids, raw_positions,
                                    valid_sequence_ids, valid_rows,
                                    storage_positions);
        }
        if (valid_sequence_ids.empty()) {
            return MakeAsyncTask([] {});
        }

        auto prepared_k = SelectRows(k_tensor, valid_rows);
        std::optional<torch::Tensor> prepared_v;
        if (v_tensor.has_value()) {
            prepared_v = SelectRows(*v_tensor, valid_rows);
        }
        auto task = BaseView::AsyncAppendDecodeKVToHost(
            layer_idx, std::move(valid_sequence_ids), std::move(prepared_k),
            std::move(prepared_v), std::move(storage_positions));
        {
            std::lock_guard<std::mutex> lock(mutex_);
            TrackHostWriteTaskLocked(task);
        }
        return task;
    }

    KVAsyncTask AsyncAppendDecodeKVToHostBatchedKernel(
        std::vector<BatchedKVEntry> entries,
        std::vector<std::int64_t> sequence_ids, SequenceLengths raw_positions) {
        if (entries.empty() || sequence_ids.empty()) {
            return MakeAsyncTask([] {});
        }

        std::vector<std::int64_t> valid_sequence_ids;
        std::vector<std::int64_t> valid_rows;
        SequenceLengthVector storage_positions;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            PrepareAppendRowsLocked(sequence_ids, raw_positions,
                                    valid_sequence_ids, valid_rows,
                                    storage_positions);
        }
        if (valid_sequence_ids.empty()) {
            return MakeAsyncTask([] {});
        }

        for (auto& entry : entries) {
            entry.k_tensor = SelectRows(entry.k_tensor, valid_rows);
            if (entry.v_tensor.has_value()) {
                entry.v_tensor = SelectRows(*entry.v_tensor, valid_rows);
            }
        }
        auto task = BaseView::AsyncAppendDecodeKVToHostBatchedKernel(
            std::move(entries), std::move(valid_sequence_ids),
            std::move(storage_positions));
        {
            std::lock_guard<std::mutex> lock(mutex_);
            TrackHostWriteTaskLocked(task);
        }
        return task;
    }

   private:
    struct SequenceState {
        std::size_t allocated_pages = 0;
        std::size_t max_exposed_tokens = 0;
    };

    void ValidateConfig() {
        if (page_size_tokens_ == 0) {
            throw std::invalid_argument(
                "CompressedRatioHostPagedKVWorkerView requires "
                "page_size_tokens > 0");
        }
    }

    static std::size_t CompressedTokens(std::size_t raw_tokens) {
        return raw_tokens / CompressionRatio;
    }

    static bool ShouldAppend(std::size_t raw_position) {
        if (raw_position == std::numeric_limits<std::size_t>::max()) {
            throw std::out_of_range(
                "CompressedRatioHostPagedKVWorkerView: raw position overflow");
        }
        return (raw_position + 1) % CompressionRatio == 0;
    }

    static std::size_t StorageCapacityTokens(std::size_t exposed_tokens) {
        return std::max<std::size_t>(1, exposed_tokens);
    }

    static std::size_t CompressedPosition(std::size_t raw_position) {
        return raw_position / CompressionRatio;
    }

    std::size_t RequiredPagesForTokens(std::size_t tokens) const {
        return (StorageCapacityTokens(tokens) + page_size_tokens_ - 1) /
               page_size_tokens_;
    }

    static std::size_t ResolveLength(const SequenceLengths& sequence_lengths,
                                     std::size_t batch_idx,
                                     std::int64_t sequence_id,
                                     std::string_view op_name) {
        return std::visit(
            [&](const auto& container) -> std::size_t {
                using Container = std::decay_t<decltype(container)>;
                if constexpr (std::is_same_v<Container, SequenceLengthMap>) {
                    const auto it = container.find(sequence_id);
                    if (it == container.end()) {
                        std::ostringstream oss;
                        oss << op_name
                            << ": missing sequence length for sequence "
                            << sequence_id;
                        throw std::out_of_range(oss.str());
                    }
                    return it->second;
                } else {
                    if (batch_idx >= container.size()) {
                        std::ostringstream oss;
                        oss << op_name
                            << ": sequence_lengths vector is missing batch "
                               "index "
                            << batch_idx;
                        throw std::out_of_range(oss.str());
                    }
                    return container[batch_idx];
                }
            },
            sequence_lengths);
    }

    void EnsureCapacityLocked(std::int64_t sequence_id,
                              std::size_t exposed_tokens) {
        const std::size_t required_pages =
            RequiredPagesForTokens(exposed_tokens);
        auto& state = sequence_states_[sequence_id];
        if (state.allocated_pages == 0) {
            BaseView::AllocatePagesForSequences(
                {sequence_id}, {StorageCapacityTokens(exposed_tokens)});
            state.allocated_pages = required_pages;
        } else if (required_pages > state.allocated_pages) {
            BaseView::GrowSequencePages(
                sequence_id, required_pages - state.allocated_pages);
            state.allocated_pages = required_pages;
        }
        state.max_exposed_tokens =
            std::max(state.max_exposed_tokens, exposed_tokens);
    }

    void PrepareOffloadRowsLocked(
        const std::vector<std::int64_t>& sequence_ids,
        const SequenceLengths& raw_sequence_lengths,
        std::vector<std::int64_t>& valid_sequence_ids,
        std::vector<std::int64_t>& valid_rows,
        SequenceLengthVector& compressed_lengths) {
        const std::size_t batch = sequence_ids.size();
        valid_sequence_ids.reserve(batch);
        valid_rows.reserve(batch);
        compressed_lengths.reserve(batch);
        for (std::size_t batch_idx = 0; batch_idx < batch; ++batch_idx) {
            const std::int64_t sequence_id = sequence_ids[batch_idx];
            const std::size_t raw_tokens = ResolveLength(
                raw_sequence_lengths, batch_idx, sequence_id,
                "CompressedRatioHostPagedKVWorkerView::"
                "AsyncOffloadLayerKVToHost");
            const std::size_t tokens = CompressedTokens(raw_tokens);
            EnsureCapacityLocked(sequence_id, tokens);
            if (tokens == 0) {
                continue;
            }
            valid_sequence_ids.push_back(sequence_id);
            valid_rows.push_back(static_cast<std::int64_t>(batch_idx));
            compressed_lengths.push_back(tokens);
        }
    }

    void PrepareAppendRowsLocked(
        const std::vector<std::int64_t>& sequence_ids,
        const SequenceLengths& raw_positions,
        std::vector<std::int64_t>& valid_sequence_ids,
        std::vector<std::int64_t>& valid_rows,
        SequenceLengthVector& storage_positions) {
        const std::size_t batch = sequence_ids.size();
        valid_sequence_ids.reserve(batch);
        valid_rows.reserve(batch);
        storage_positions.reserve(batch);
        for (std::size_t batch_idx = 0; batch_idx < batch; ++batch_idx) {
            const std::int64_t sequence_id = sequence_ids[batch_idx];
            const std::size_t raw_position = ResolveLength(
                raw_positions, batch_idx, sequence_id,
                "CompressedRatioHostPagedKVWorkerView::"
                "AsyncAppendDecodeKVToHost");
            const bool should_append = ShouldAppend(raw_position);
            const std::size_t raw_end = raw_position + 1;
            if (!should_append) {
                EnsureCapacityLocked(sequence_id, CompressedTokens(raw_end));
                continue;
            }
            const std::size_t storage_position =
                CompressedPosition(raw_position);
            EnsureCapacityLocked(sequence_id, storage_position + 1);
            valid_sequence_ids.push_back(sequence_id);
            valid_rows.push_back(static_cast<std::int64_t>(batch_idx));
            storage_positions.push_back(storage_position);
        }
    }

    static torch::Tensor SelectRows(const torch::Tensor& tensor,
                                    const std::vector<std::int64_t>& rows) {
        if (rows.size() == static_cast<std::size_t>(tensor.size(0))) {
            return tensor;
        }
        auto indices =
            torch::tensor(rows, torch::TensorOptions()
                                    .dtype(torch::kLong)
                                    .device(tensor.device()));
        return tensor.index_select(0, indices).contiguous();
    }

    void DrainPendingHostWritesLocked() {
        for (const auto& task : pending_host_write_tasks_) {
            task.wait();
        }
        pending_host_write_tasks_.clear();
    }

    void PruneDoneHostWritesLocked() {
        pending_host_write_tasks_.erase(
            std::remove_if(pending_host_write_tasks_.begin(),
                           pending_host_write_tasks_.end(),
                           [](const KVAsyncTask& task) { return task.done(); }),
            pending_host_write_tasks_.end());
    }

    void TrackHostWriteTaskLocked(const KVAsyncTask& task) {
        PruneDoneHostWritesLocked();
        pending_host_write_tasks_.push_back(task);
    }

    template <typename Fn>
    static KVAsyncTask MakeAsyncTask(Fn&& fn) {
        auto future =
            std::async(std::launch::async, std::forward<Fn>(fn)).share();
        const std::uint64_t id =
            task_id_counter_.fetch_add(1, std::memory_order_relaxed) + 1;
        return KVAsyncTask{id, std::move(future)};
    }

    std::size_t page_size_tokens_ = 0;
    mutable std::mutex mutex_;
    std::unordered_map<std::int64_t, SequenceState> sequence_states_;
    std::vector<KVAsyncTask> pending_host_write_tasks_;

    inline static std::atomic<std::uint64_t> task_id_counter_{0};
};

using CompressedRatio4DefaultHostPagedKVWorkerView =
    CompressedRatioHostPagedKVWorkerView<DefaultHostPagedKVWorkerView, 4>;
using CompressedRatio4MLAHostPagedKVWorkerView =
    CompressedRatioHostPagedKVWorkerView<MLAHostPagedKVWorkerView, 4>;
using CompressedRatio4MappedDefaultHostPagedKVWorkerView =
    CompressedRatioHostPagedKVWorkerView<MappedDefaultHostPagedKVWorkerView, 4>;
using CompressedRatio4MappedMLAHostPagedKVWorkerView =
    CompressedRatioHostPagedKVWorkerView<MappedMLAHostPagedKVWorkerView, 4>;

using CompressedRatio128DefaultHostPagedKVWorkerView =
    CompressedRatioHostPagedKVWorkerView<DefaultHostPagedKVWorkerView, 128>;
using CompressedRatio128MLAHostPagedKVWorkerView =
    CompressedRatioHostPagedKVWorkerView<MLAHostPagedKVWorkerView, 128>;
using CompressedRatio128MappedDefaultHostPagedKVWorkerView =
    CompressedRatioHostPagedKVWorkerView<MappedDefaultHostPagedKVWorkerView,
                                         128>;
using CompressedRatio128MappedMLAHostPagedKVWorkerView =
    CompressedRatioHostPagedKVWorkerView<MappedMLAHostPagedKVWorkerView, 128>;

}  // namespace batchgen::kv

#endif  // COMPRESSED_RATIO_HOST_PAGED_KV_WORKER_VIEW_H_
