#ifndef HOST_PAGED_KV_WORKER_VIEW_H_
#define HOST_PAGED_KV_WORKER_VIEW_H_

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <future>
#include <iomanip>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "../utils.h"
#include "host_kv_page_table.h"
#include "host_paged_kv_backend.h"
#include "host_paged_kv_geometry.h"
#include "host_paged_kv_layout.h"
#include "spdlog/spdlog.h"
#include "util_measure_time.h"

namespace batchgen::kv {

namespace worker_detail {
void RegisterPinnedRange(void* base, std::size_t bytes, int device_index,
                         const std::shared_ptr<spdlog::logger>& logger);
void UnregisterPinnedRange(void* base, int device_index,
                           const std::shared_ptr<spdlog::logger>& logger);
}  // namespace worker_detail

struct KVAsyncTask {
    KVAsyncTask() = default;
    KVAsyncTask(std::uint64_t id, std::shared_future<void> future)
        : id_(id), future_(std::move(future)) {}

    std::uint64_t id() const { return id_; }
    bool done() const {
        if (!future_.valid()) {
            return true;
        }
        return future_.wait_for(std::chrono::seconds(0)) ==
               std::future_status::ready;
    }
    void wait() const {
        if (future_.valid()) {
            future_.wait();
        }
    }
    void result() const {
        if (future_.valid()) {
            future_.get();
        }
    }

   private:
    std::uint64_t id_ = 0;
    std::shared_future<void> future_;
};

using SequenceLengthMap = std::unordered_map<std::int64_t, std::size_t>;
using SequenceLengthVector = std::vector<std::size_t>;
using SequenceLengths = std::variant<SequenceLengthMap, SequenceLengthVector>;

template <HostKVMode Mode, typename Layout = HostPagedKVLayout<Mode>>
class HostPagedKVWorkerView {
   public:
    using ModeTraits = HostKVModeTraits<Mode>;
    static constexpr HostKVMode kMode = Mode;
    static constexpr bool kHasVCache = Layout::kHasVCache;

    explicit HostPagedKVWorkerView(const HostPagedKVConfig& config)
        : config_(detail::SanitizeConfig(ModeTraits::Adjust(config))),
          geometry_(config_),
          layout_(config_),
          backend_(config_, layout_.DataSectionBytes(), layout_.Fingerprint(),
                   Layout::kHasVCache),
          logger_(init_logger("info", "HostPagedKVWorkerView")) {}

    HostPagedKVWorkerView(const HostPagedKVWorkerView&) = delete;
    HostPagedKVWorkerView& operator=(const HostPagedKVWorkerView&) = delete;
    HostPagedKVWorkerView(HostPagedKVWorkerView&&) = delete;
    HostPagedKVWorkerView& operator=(HostPagedKVWorkerView&&) = delete;

    ~HostPagedKVWorkerView() {
        try {
            ResetCopyStreams();
            UnregisterPinnedMemory();
        } catch (const std::exception& ex) {
            logger_->error(
                "Failed to unregister HostPagedKVWorkerView pinned range: "
                "{}",
                ex.what());
        }
    }

    void Initialize(int device_index, bool create_region = false) {
        if (device_index < 0) {
            throw std::invalid_argument(
                "device_index must be greater than or equal to zero");
        }
        device_index_ = device_index;
        logger_->info(
            "Attaching HostPagedKVWorkerView (device_index={}, "
            "create_region={})",
            device_index_, create_region);
        backend_.Initialize(create_region);
        const HostPagedKVStats stats = backend_.CollectStats();
        const auto duration_us =
            util::MeasureTime([this] { RegisterPinnedMemory(); });
        InitializeCopyStreams();
        const double duration_sec =
            std::chrono::duration_cast<std::chrono::duration<double>>(
                duration_us)
                .count();

        logger_->info(
            "HostPagedKVWorkerView ready "
            "(device_index={}, create_region={}, total_pages={}, "
            "free_pages={}, "
            "data_bytes={:.3f}GB, total_bytes={:.3f}GB), "
            "pinned_mem_setup={:.3f}s)",
            device_index_, create_region, stats.num_total_pages,
            stats.num_free_pages, BytesToGigabytes(layout_.DataSectionBytes()),
            BytesToGigabytes(stats.total_bytes), duration_sec);
    }

    std::vector<std::vector<std::int32_t>> AllocatePagesForSequences(
        const std::vector<std::int64_t>& sequence_ids,
        const std::vector<std::size_t>& num_tokens) {
        if (sequence_ids.size() != num_tokens.size()) {
            throw std::invalid_argument(
                "sequence_ids and num_tokens must have the same length");
        }
        EnsureSequencesRegistered(sequence_ids);
        for (std::size_t i = 0; i < num_tokens.size(); ++i) {
            if (num_tokens[i] == 0) {
                throw std::invalid_argument(
                    "num_tokens must be greater than zero");
            }
        }
        auto allocated_pages =
            backend_.AcquirePagesForSequences(sequence_ids, num_tokens);
        for (std::size_t i = 0; i < sequence_ids.size(); ++i) {
            page_table_.RegisterOrUpdate(sequence_ids[i], allocated_pages[i]);
        }
        return allocated_pages;
    }

    void Shutdown() {
        logger_->info("Shutting down HostPagedKVWorkerView (device_index={})",
                      device_index_);
        ResetCopyStreams();
        UnregisterPinnedMemory();
        page_table_.Clear();
    }

    std::byte* DataBase() { return backend_.DataBase(); }
    const std::byte* DataBase() const { return backend_.DataBase(); }

    void* KPagePtr(std::size_t layer_idx, std::int32_t page_idx) {
        geometry_.EnsureLayerBounds(layer_idx, kClassTag);
        geometry_.EnsurePageBounds(page_idx, kClassTag);
        return static_cast<void*>(
            layout_.KPageAddress(backend_.DataBase(), layer_idx, page_idx));
    }

    const void* KPagePtr(std::size_t layer_idx, std::int32_t page_idx) const {
        geometry_.EnsureLayerBounds(layer_idx, kClassTag);
        geometry_.EnsurePageBounds(page_idx, kClassTag);
        return static_cast<const void*>(
            layout_.KPageAddress(backend_.DataBase(), layer_idx, page_idx));
    }

    template <bool Enabled = Layout::kHasVCache,
              typename = std::enable_if_t<Enabled>>
    void* VPagePtr(std::size_t layer_idx, std::int32_t page_idx) {
        geometry_.EnsureLayerBounds(layer_idx, kClassTag);
        geometry_.EnsurePageBounds(page_idx, kClassTag);
        return static_cast<void*>(layout_.template VPageAddress<>(
            backend_.DataBase(), layer_idx, page_idx));
    }

    template <bool Enabled = Layout::kHasVCache,
              typename = std::enable_if_t<Enabled>>
    const void* VPagePtr(std::size_t layer_idx, std::int32_t page_idx) const {
        geometry_.EnsureLayerBounds(layer_idx, kClassTag);
        geometry_.EnsurePageBounds(page_idx, kClassTag);
        return static_cast<const void*>(layout_.template VPageAddress<>(
            backend_.DataBase(), layer_idx, page_idx));
    }

    const HostPagedKVConfig& config() const { return config_; }
    const Layout& layout() const { return layout_; }
    HostPagedKVStats GetStats() const { return backend_.CollectStats(); }
    int device_index() const { return device_index_; }

    std::string DebugString() const {
        std::ostringstream oss;
        oss << "HostPagedKVWorkerView(mode=" << static_cast<int>(Mode)
            << ", has_v_cache=" << std::boolalpha << kHasVCache
            << ", device_index=" << device_index_
            << ", config=" << ToString(config_);
        if (pinned_registered_) {
            oss << ", pinned_base=" << pinned_base_
                << ", pinned_bytes=" << pinned_bytes_;
        } else {
            oss << ", pinned_base=null";
        }
        oss << ")";
        return oss.str();
    }

    std::vector<std::vector<std::int32_t>> BuildPageTable(
        const std::vector<std::int64_t>& sequence_ids) const {
        std::vector<std::vector<std::int32_t>> table;
        table.reserve(sequence_ids.size());
        for (std::int64_t seq_id : sequence_ids) {
            try {
                table.emplace_back(page_table_.Pages(seq_id));
            } catch (const std::out_of_range&) {
                table.emplace_back();
            }
        }
        return table;
    }

    void RegisterSequences(const std::vector<std::int64_t>& sequence_ids) {
        if (sequence_ids.empty()) {
            return;
        }
        for (std::int64_t sequence_id : sequence_ids) {
            std::vector<std::int32_t> pages;
            try {
                pages = backend_.SequencePages(sequence_id, std::nullopt);
            } catch (const std::out_of_range&) {
                pages.clear();
            }
            page_table_.RegisterOrUpdate(sequence_id, std::move(pages));
        }
    }

    void UnregisterSequence(std::int64_t sequence_id) {
        page_table_.Remove(sequence_id);
    }

    void UnregisterSequences(const std::vector<std::int64_t>& sequence_ids) {
        if (sequence_ids.empty()) {
            return;
        }
        std::for_each(sequence_ids.begin(), sequence_ids.end(),
                      [this](std::int64_t sequence_id) {
                          UnregisterSequence(sequence_id);
                      });
    }

    KVAsyncTask AsyncOffloadLayerKVToHost(
        std::size_t layer_idx, std::vector<std::int64_t> sequence_ids,
        torch::Tensor k_tensor, std::optional<torch::Tensor> v_tensor,
        // [B, S, H, D]
        SequenceLengths sequence_lengths) {
        geometry_.EnsureLayerBounds(layer_idx, "AsyncOffloadLayerKVToHost");
        EnsureDeviceReady();
        const std::size_t batch = sequence_ids.size();
        if (batch == 0) {
            return LaunchAsyncTask([] {});
        }
        const std::size_t tokens_per_sequence =
            ValidateKTensorShape(k_tensor, batch);
        ValidateSequenceLengthsInput(sequence_lengths, batch,
                                     "AsyncOffloadLayerKVToHost");
        torch::Tensor prepared_k = k_tensor;
        std::optional<torch::Tensor> prepared_v;
        if (v_tensor.has_value()) {
            if constexpr (kHasVCache) {
                ValidateVTensorShape(*v_tensor, batch, tokens_per_sequence);
                prepared_v = *v_tensor;
            } else {
                throw std::invalid_argument(
                    "V tensor provided but V cache is disabled");
            }
        }

        return LaunchAsyncTask([this, layer_idx,
                                sequence_ids = std::move(sequence_ids),
                                sequence_lengths = std::move(sequence_lengths),
                                prepared_k, prepared_v, tokens_per_sequence]() {
            c10::cuda::OptionalCUDAGuard device_guard(device_index_);
            const auto cuda_stream = CopyStream(CopyDirection::kDeviceToHost);

            const auto* k_base =
                static_cast<const std::byte*>(prepared_k.data_ptr());
            const std::size_t k_token_bytes = geometry_.KTokenBytes();
            const std::size_t k_seq_stride =
                tokens_per_sequence * k_token_bytes;

            const std::byte* v_base = nullptr;
            std::size_t v_token_bytes = 0;
            std::size_t v_seq_stride = 0;
            if (prepared_v.has_value()) {
                if constexpr (kHasVCache) {
                    v_base =
                        static_cast<const std::byte*>(prepared_v->data_ptr());
                    v_token_bytes =
                        geometry_.template VTokenBytes<kHasVCache>();
                    v_seq_stride = tokens_per_sequence * v_token_bytes;
                }
            }

            std::byte* host_base = backend_.DataBase();

            for (std::size_t batch_idx = 0; batch_idx < sequence_ids.size();
                 ++batch_idx) {
                const std::int64_t sequence_id = sequence_ids[batch_idx];
                const auto pages = page_table_.Pages(sequence_id);
                const std::size_t tokens_to_copy = ResolveSequenceLength(
                    sequence_lengths, batch_idx, sequence_id,
                    tokens_per_sequence, "AsyncOffloadLayerKVToHost");
                if (tokens_to_copy == 0) {
                    continue;
                }
                geometry_.ValidatePageCapacity(pages, tokens_to_copy,
                                               "AsyncOffloadLayerKVToHost");

                const auto* seq_k_src = k_base + batch_idx * k_seq_stride;

                ForEachPageChunk(
                    pages, 0, tokens_to_copy,
                    [&](std::int32_t page_idx, std::size_t page_offset_tokens,
                        std::size_t chunk_tokens,
                        std::size_t relative_token_offset) {
                        std::byte* dst = layout_.KPageAddress(
                                             host_base, layer_idx, page_idx) +
                                         page_offset_tokens * k_token_bytes;
                        const std::byte* src =
                            seq_k_src + relative_token_offset * k_token_bytes;
                        EnqueueCopy(src, dst, chunk_tokens * k_token_bytes,
                                    CopyDirection::kDeviceToHost, cuda_stream);
                    });
                if constexpr (kHasVCache) {
                    if (v_base != nullptr) {
                        const auto* seq_v_src =
                            v_base + batch_idx * v_seq_stride;
                        ForEachPageChunk(
                            pages, 0, tokens_to_copy,
                            [&](std::int32_t page_idx,
                                std::size_t page_offset_tokens,
                                std::size_t chunk_tokens,
                                std::size_t relative_token_offset) {
                                std::byte* dst =
                                    layout_.template VPageAddress<>(
                                        host_base, layer_idx, page_idx) +
                                    page_offset_tokens * v_token_bytes;
                                const std::byte* src =
                                    seq_v_src +
                                    relative_token_offset * v_token_bytes;
                                EnqueueCopy(
                                    src, dst, chunk_tokens * v_token_bytes,
                                    CopyDirection::kDeviceToHost, cuda_stream);
                            });
                    }
                }
            }

            this->SynchronizeWithEvent(cuda_stream);
            // LogFirstTokenPerPage(layer_idx, sequence_ids, sequence_lengths,
            //                      tokens_per_sequence, host_base);
        });
    }

    KVAsyncTask AsyncAppendDecodeKVToHost(
        std::size_t layer_idx, std::vector<std::int64_t> sequence_ids,
        torch::Tensor k_tensor, std::optional<torch::Tensor> v_tensor,
        SequenceLengths sequence_lengths) {
        geometry_.EnsureLayerBounds(layer_idx, "AsyncAppendDecodeKVToHost");
        EnsureDeviceReady();
        const std::size_t batch = sequence_ids.size();
        if (batch == 0) {
            return LaunchAsyncTask([] {});
        }
        const std::size_t tokens_per_sequence =
            ValidateKTensorShape(k_tensor, batch);
        ValidateSequenceLengthsInput(sequence_lengths, batch,
                                     "AsyncAppendDecodeKVToHost");
        torch::Tensor prepared_k = k_tensor;
        std::optional<torch::Tensor> prepared_v;
        if (v_tensor.has_value()) {
            if constexpr (kHasVCache) {
                ValidateVTensorShape(*v_tensor, batch, tokens_per_sequence);
                prepared_v = *v_tensor;
            } else {
                throw std::invalid_argument(
                    "V tensor provided but V cache is disabled");
            }
        }

        if (tokens_per_sequence != 1) {
            throw std::invalid_argument(
                "AsyncAppendDecodeKVToHost expects tensors with a single token "
                "per sequence");
        }

        return LaunchAsyncTask([this, layer_idx,
                                sequence_ids = std::move(sequence_ids),
                                sequence_lengths = std::move(sequence_lengths),
                                prepared_k, prepared_v]() mutable {
            c10::cuda::OptionalCUDAGuard device_guard(device_index_);
            const auto cuda_stream = CopyStream(CopyDirection::kDeviceToHost);

            const auto* k_base =
                static_cast<const std::byte*>(prepared_k.data_ptr());
            const std::size_t k_token_bytes = geometry_.KTokenBytes();
            const std::size_t k_seq_stride = k_token_bytes;

            const std::byte* v_base = nullptr;
            std::size_t v_token_bytes = 0;
            std::size_t v_seq_stride = 0;
            if (prepared_v.has_value()) {
                if constexpr (kHasVCache) {
                    v_base =
                        static_cast<const std::byte*>(prepared_v->data_ptr());
                    v_token_bytes =
                        geometry_.template VTokenBytes<kHasVCache>();
                    v_seq_stride = v_token_bytes;
                }
            }

            std::byte* host_base = backend_.DataBase();

            for (std::size_t batch_idx = 0; batch_idx < sequence_ids.size();
                 ++batch_idx) {
                const std::int64_t sequence_id = sequence_ids[batch_idx];
                const std::size_t start_token = ResolveSequenceLength(
                    sequence_lengths, batch_idx, sequence_id, std::nullopt,
                    "AsyncAppendDecodeKVToHost");
                const auto pages = page_table_.Pages(sequence_id);
                geometry_.ValidatePageCapacity(pages, start_token + 1,
                                               "AsyncAppendDecodeKVToHost");

                const auto location =
                    ResolvePageLocation(pages, sequence_id, start_token,
                                        "AsyncAppendDecodeKVToHost");

                const auto* seq_k_src = k_base + batch_idx * k_seq_stride;
                std::byte* k_dst = layout_.KPageAddress(host_base, layer_idx,
                                                        location.page_idx) +
                                   location.page_offset_tokens * k_token_bytes;
                EnqueueCopy(seq_k_src, k_dst, k_token_bytes,
                            CopyDirection::kDeviceToHost, cuda_stream);

                if constexpr (kHasVCache) {
                    if (v_base != nullptr) {
                        const auto* seq_v_src =
                            v_base + batch_idx * v_seq_stride;
                        std::byte* v_dst =
                            layout_.template VPageAddress<>(
                                host_base, layer_idx, location.page_idx) +
                            location.page_offset_tokens * v_token_bytes;
                        EnqueueCopy(seq_v_src, v_dst, v_token_bytes,
                                    CopyDirection::kDeviceToHost, cuda_stream);
                    }
                }
            }

            this->SynchronizeWithEvent(cuda_stream);
        });
    }

   private:
    struct PageLocation {
        std::int32_t page_idx = -1;
        std::size_t page_offset_tokens = 0;
    };

    static inline constexpr std::string_view kClassTag =
        "HostPagedKVWorkerView";

    void RegisterPinnedMemory() {
        if (pinned_registered_) {
            return;
        }
        if (device_index_ < 0) {
            throw std::runtime_error(
                "device_index must be set before registering pinned memory");
        }
        std::byte* base = backend_.DataBase();
        const std::size_t bytes = layout_.DataSectionBytes();
        if (base == nullptr || bytes == 0) {
            return;
        }
        worker_detail::RegisterPinnedRange(static_cast<void*>(base), bytes,
                                           device_index_, logger_);
        pinned_registered_ = true;
        pinned_base_ = static_cast<void*>(base);
        pinned_bytes_ = bytes;
    }

    void UnregisterPinnedMemory() {
        if (!pinned_registered_) {
            return;
        }
        worker_detail::UnregisterPinnedRange(pinned_base_, device_index_,
                                             logger_);
        pinned_registered_ = false;
        pinned_base_ = nullptr;
        pinned_bytes_ = 0;
    }

    enum class CopyDirection { kHostToDevice, kDeviceToHost };

    void InitializeCopyStreams() {
        if (copy_streams_ready_) {
            return;
        }
        if (device_index_ < 0) {
            throw std::runtime_error(
                "device_index must be set before initializing copy streams");
        }
        c10::cuda::OptionalCUDAGuard guard(device_index_);
        h2d_stream_.emplace(at::cuda::getStreamFromPool(
            /* isHighPriority */ false, device_index_));
        d2h_stream_.emplace(at::cuda::getStreamFromPool(
            /* isHighPriority */ false,
            device_index_));  // the flag is NonBlocking Stream
        copy_streams_ready_ = true;
        logger_->info("Initialized dedicated copy streams on device {}",
                      device_index_);
    }

    void ResetCopyStreams() noexcept {
        copy_streams_ready_ = false;
        h2d_stream_.reset();
        d2h_stream_.reset();
    }

    cudaStream_t CopyStream(CopyDirection direction) const {
        if (!copy_streams_ready_) {
            throw std::runtime_error(
                "Copy streams are not initialized; call Initialize first");
        }
        const at::cuda::CUDAStream& stream =
            direction == CopyDirection::kHostToDevice ? *h2d_stream_
                                                      : *d2h_stream_;
        return stream.stream();
    }

    static constexpr cudaMemcpyKind ToCudaMemcpyKind(CopyDirection direction) {
        return direction == CopyDirection::kHostToDevice
                   ? cudaMemcpyHostToDevice
                   : cudaMemcpyDeviceToHost;
    }

    void EnqueueCopy(const std::byte* src, std::byte* dst,
                     std::size_t num_bytes, CopyDirection direction,
                     cudaStream_t stream) const {
        CUDA_CHECK(cudaMemcpyAsync(static_cast<void*>(dst),
                                   static_cast<const void*>(src), num_bytes,
                                   ToCudaMemcpyKind(direction), stream));
    }

    void EnsureDeviceReady() const {
        if (device_index_ < 0) {
            throw std::runtime_error(
                "HostPagedKVWorkerView must be initialized before async "
                "operations");
        }
        if (!copy_streams_ready_) {
            throw std::runtime_error(
                "Copy streams are not initialized; call Initialize first");
        }
    }

    void ValidateCudaTensor(const torch::Tensor& tensor) const {
        if (!tensor.is_cuda()) {
            throw std::invalid_argument("Tensor must reside on CUDA device");
        }
        if (tensor.device().index() != device_index_) {
            throw std::invalid_argument(
                "Tensor device index does not match worker view device");
        }
    }

    void EnsureContiguousTensor(const torch::Tensor& tensor,
                                std::string_view tensor_name) const {
        if (!tensor.is_contiguous()) {
            throw std::invalid_argument(std::string(tensor_name) +
                                        " must be contiguous");
        }
    }

    void EnsureSequenceRegistered(std::int64_t sequence_id) const {
        if (page_table_.Contains(sequence_id)) {
            return;
        }
        std::ostringstream oss;
        oss << "Sequence " << sequence_id
            << " is not registered. Call RegisterSequences before allocating "
               "or growing pages.";
        throw std::logic_error(oss.str());
    }

    void EnsureSequencesRegistered(
        const std::vector<std::int64_t>& sequence_ids) const {
        for (std::int64_t sequence_id : sequence_ids) {
            EnsureSequenceRegistered(sequence_id);
        }
    }

    void ValidateSequenceLengthsInput(const SequenceLengths& sequence_lengths,
                                      std::size_t expected_batch,
                                      std::string_view op_name) const {
        std::visit(
            [&](const auto& container) {
                using Container = std::decay_t<decltype(container)>;
                if constexpr (std::is_same_v<Container, SequenceLengthVector>) {
                    if (container.size() != expected_batch) {
                        std::ostringstream oss;
                        oss << op_name << ": sequence_lengths vector size ("
                            << container.size()
                            << ") must match sequence_ids size ("
                            << expected_batch << ")";
                        throw std::invalid_argument(oss.str());
                    }
                }
            },
            sequence_lengths);
    }

    void LogFirstTokenPerPage(std::size_t layer_idx,
                              const std::vector<std::int64_t>& sequence_ids,
                              const SequenceLengths& sequence_lengths,
                              std::size_t tokens_per_sequence,
                              std::byte* host_base) const {
        constexpr std::string_view kOpName = "AsyncOffloadLayerKVToHost";
        const std::size_t tokens_per_page = geometry_.PageSizeTokens();
        if (host_base == nullptr) {
            logger_->warn(
                "LogFirstTokenPerPage: host base is null, skipping logging");
            return;
        }
        for (std::size_t batch_idx = 0; batch_idx < sequence_ids.size();
             ++batch_idx) {
            const std::int64_t sequence_id = sequence_ids[batch_idx];
            const std::size_t tokens_to_copy =
                ResolveSequenceLength(sequence_lengths, batch_idx, sequence_id,
                                      tokens_per_sequence, kOpName);
            if (tokens_to_copy == 0) {
                continue;
            }
            const auto pages = page_table_.Pages(sequence_id);
            for (std::size_t slot = 0; slot < pages.size(); ++slot) {
                if (!PageContainsTokens(slot, tokens_to_copy,
                                        tokens_per_page)) {
                    break;
                }
                const std::int32_t page_idx = pages[slot];
                const std::byte* token_ptr =
                    layout_.KPageAddress(host_base, layer_idx, page_idx);
                logger_->info("Layer {} Seq {} Page {} first_token_hex {}",
                              layer_idx, sequence_id, page_idx,
                              geometry_.DescribeBytes(token_ptr,
                                                      geometry_.KTokenBytes()));
            }
        }
    }

    bool PageContainsTokens(std::size_t page_slot, std::size_t tokens_available,
                            std::size_t tokens_per_page) const {
        const std::size_t page_start_token = page_slot * tokens_per_page;
        return tokens_available > page_start_token;
    }

    PageLocation ResolvePageLocation(const std::vector<std::int32_t>& pages,
                                     std::int64_t sequence_id,
                                     std::size_t token_index,
                                     std::string_view op_name) const {
        if (pages.empty()) {
            std::ostringstream oss;
            oss << op_name << ": sequence " << sequence_id
                << " has no allocated pages";
            throw std::out_of_range(oss.str());
        }
        const std::size_t tokens_per_page = geometry_.PageSizeTokens();
        const std::size_t page_slot = token_index / tokens_per_page;
        if (page_slot >= pages.size()) {
            std::ostringstream oss;
            oss << op_name << ": sequence " << sequence_id << " token index "
                << token_index
                << " exceeds allocated capacity (pages=" << pages.size() << ")";
            throw std::out_of_range(oss.str());
        }
        PageLocation location;
        location.page_idx = pages[page_slot];
        location.page_offset_tokens = token_index % tokens_per_page;
        return location;
    }

    std::size_t ResolveSequenceLength(const SequenceLengths& sequence_lengths,
                                      std::size_t batch_idx,
                                      std::int64_t sequence_id,
                                      std::optional<std::size_t> max_tokens,
                                      std::string_view op_name) const {
        return std::visit(
            [&](const auto& container) -> std::size_t {
                using Container = std::decay_t<decltype(container)>;
                if constexpr (std::is_same_v<Container, SequenceLengthMap>) {
                    const auto it = RequireSequenceLengthIterator(
                        container, sequence_id, op_name);
                    const std::size_t length = it->second;
                    EnsureLengthWithinTensorCapacity(sequence_id, length,
                                                     max_tokens, op_name);
                    return length;
                } else {
                    EnsureSequenceLengthsVectorIndex(batch_idx,
                                                     container.size(), op_name);
                    const std::size_t length = container[batch_idx];
                    EnsureLengthWithinTensorCapacity(sequence_id, length,
                                                     max_tokens, op_name);
                    return length;
                }
            },
            sequence_lengths);
    }

    void EnsureSequenceLengthsVectorIndex(std::size_t batch_idx,
                                          std::size_t size,
                                          std::string_view op_name) const {
        if (batch_idx >= size) {
            std::ostringstream oss;
            oss << op_name
                << ": sequence_lengths vector is missing entry for batch index "
                << batch_idx << " (size=" << size << ")";
            throw std::out_of_range(oss.str());
        }
    }

    void EnsureLengthWithinTensorCapacity(std::int64_t sequence_id,
                                          std::size_t length,
                                          std::optional<std::size_t> max_tokens,
                                          std::string_view op_name) const {
        if (!max_tokens.has_value()) {
            return;
        }
        if (length > max_tokens.value()) {
            std::ostringstream oss;
            oss << op_name
                << ": sequence length exceeds tensor capacity for sequence "
                << sequence_id << " (length=" << length
                << ", capacity=" << max_tokens.value() << ")";
            throw std::out_of_range(oss.str());
        }
    }

    std::size_t ValidateKTensorShape(const torch::Tensor& tensor,
                                     std::size_t expected_batch) const {
        ValidateCudaTensor(tensor);
        EnsureContiguousTensor(tensor, "K tensor");
        if (tensor.dim() != 4) {
            throw std::invalid_argument(
                "Expected K tensor with shape [B, T, H, D]");
        }
        if (static_cast<std::size_t>(tensor.size(0)) != expected_batch) {
            throw std::invalid_argument(
                "Batch size of K tensor does not match sequence_ids");
        }
        if (tensor.size(2) != static_cast<long>(config_.num_k_heads) ||
            tensor.size(3) != static_cast<long>(config_.k_head_dim)) {
            throw std::invalid_argument(
                "K tensor head dimensions do not match configuration");
        }
        if (tensor.element_size() !=
            static_cast<int64_t>(config_.k_element_size_bytes)) {
            throw std::invalid_argument(
                "K tensor element size does not match configuration");
        }
        return static_cast<std::size_t>(tensor.size(1));
    }

    template <bool Enabled = Layout::kHasVCache>
    std::enable_if_t<Enabled, void> ValidateVTensorShape(
        const torch::Tensor& tensor, std::size_t expected_batch,
        std::size_t expected_tokens) const {
        ValidateCudaTensor(tensor);
        EnsureContiguousTensor(tensor, "V tensor");
        if (tensor.dim() != 4) {
            throw std::invalid_argument(
                "Expected V tensor with shape [B, T, H, D]");
        }
        if (static_cast<std::size_t>(tensor.size(0)) != expected_batch) {
            throw std::invalid_argument(
                "Batch size of V tensor does not match sequence_ids");
        }
        if (static_cast<std::size_t>(tensor.size(1)) != expected_tokens) {
            throw std::invalid_argument(
                "Token dimension of V tensor does not match K tensor");
        }
        if (tensor.size(2) != static_cast<long>(config_.num_v_heads) ||
            tensor.size(3) != static_cast<long>(config_.v_head_dim)) {
            throw std::invalid_argument(
                "V tensor head dimensions do not match configuration");
        }
        if (tensor.element_size() !=
            static_cast<int64_t>(config_.v_element_size_bytes)) {
            throw std::invalid_argument(
                "V tensor element size does not match configuration");
        }
    }

    template <bool Enabled = Layout::kHasVCache>
    std::enable_if_t<!Enabled, void> ValidateVTensorShape(const torch::Tensor&,
                                                          std::size_t,
                                                          std::size_t) const {
        throw std::logic_error("V cache is disabled for this worker view");
    }

    template <typename Map>
    auto RequireSequenceLengthIterator(Map& sequence_lengths,
                                       std::int64_t sequence_id,
                                       std::string_view op_name) const
        -> decltype(sequence_lengths.find(sequence_id)) {
        const auto it = sequence_lengths.find(sequence_id);
        if (it == sequence_lengths.end()) {
            std::ostringstream oss;
            oss << op_name << ": missing sequence length for sequence "
                << sequence_id;
            throw std::out_of_range(oss.str());
        }
        return it;
    }

    template <typename Fn>
    void ForEachPageChunk(const std::vector<std::int32_t>& pages,
                          std::size_t start_token, std::size_t num_tokens,
                          Fn&& fn) const {
        if (num_tokens == 0) {
            return;
        }
        const std::size_t tokens_per_page = geometry_.PageSizeTokens();
        std::size_t tokens_remaining = num_tokens;
        std::size_t token_cursor = start_token;
        while (tokens_remaining > 0) {
            const std::size_t page_slot = token_cursor / tokens_per_page;
            const std::size_t page_offset = token_cursor % tokens_per_page;
            if (page_slot >= pages.size()) {
                throw std::out_of_range(
                    "Page table is too small for requested tokens");
            }
            const std::size_t chunk_tokens =
                std::min(tokens_remaining, tokens_per_page - page_offset);
            fn(pages[page_slot], page_offset, chunk_tokens,
               token_cursor - start_token);
            token_cursor += chunk_tokens;
            tokens_remaining -= chunk_tokens;
        }
    }

    template <typename Fn>
    KVAsyncTask LaunchAsyncTask(Fn&& fn) const {
        auto future =
            std::async(std::launch::async, std::forward<Fn>(fn)).share();
        const std::uint64_t id =
            task_id_counter_.fetch_add(1, std::memory_order_relaxed) + 1;
        return KVAsyncTask{id, std::move(future)};
    }

    void SynchronizeWithEvent(cudaStream_t stream) const {
        ScopedCudaEvent event(logger_);
        CUDA_CHECK(cudaEventRecord(event.get(), stream));
        CUDA_CHECK(cudaEventSynchronize(event.get()));
    }

    class ScopedCudaEvent final {
       public:
        explicit ScopedCudaEvent(std::shared_ptr<spdlog::logger> logger,
                                 unsigned int flags = cudaEventDisableTiming)
            : logger_(std::move(logger)) {
            CUDA_CHECK(cudaEventCreateWithFlags(&event_, flags));
        }

        ScopedCudaEvent(const ScopedCudaEvent&) = delete;
        ScopedCudaEvent& operator=(const ScopedCudaEvent&) = delete;
        ScopedCudaEvent(ScopedCudaEvent&&) = delete;
        ScopedCudaEvent& operator=(ScopedCudaEvent&&) = delete;

        ~ScopedCudaEvent() {
            if (event_ == nullptr) {
                return;
            }
            const auto status = cudaEventDestroy(event_);
            if (status != cudaSuccess && logger_ != nullptr) {
                logger_->error("Failed to destroy CUDA event: {}",
                               cudaGetErrorString(status));
            }
        }

        cudaEvent_t get() const { return event_; }

       private:
        cudaEvent_t event_ = nullptr;
        std::shared_ptr<spdlog::logger> logger_;
    };

    HostPagedKVConfig config_;
    HostPagedKVGeometry geometry_;
    Layout layout_;
    HostPagedKVBackend backend_;
    std::shared_ptr<spdlog::logger> logger_;
    bool pinned_registered_ = false;
    void* pinned_base_ = nullptr;
    std::size_t pinned_bytes_ = 0;
    int device_index_ = -1;
    bool copy_streams_ready_ = false;
    std::optional<at::cuda::CUDAStream> h2d_stream_;
    std::optional<at::cuda::CUDAStream> d2h_stream_;
    HostKVPageTable page_table_;

    inline static std::atomic<std::uint64_t> task_id_counter_{0};
};

using DefaultHostPagedKVWorkerView = HostPagedKVWorkerView<HostKVMode::kMHA>;
using MLAHostPagedKVWorkerView = HostPagedKVWorkerView<HostKVMode::kMLA>;

}  // namespace batchgen::kv

#endif  // HOST_PAGED_KV_WORKER_VIEW_H_
