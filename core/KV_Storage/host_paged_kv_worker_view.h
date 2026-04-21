#ifndef HOST_PAGED_KV_WORKER_VIEW_H_
#define HOST_PAGED_KV_WORKER_VIEW_H_

#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>
#include <torch/torch.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <future>
#include <iomanip>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include "../utils.h"
#include "host_kv_page_table.h"
#include "host_paged_kv_backend.h"
#include "host_paged_kv_config_utils.h"
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

template <typename T>
class DeviceBuffer {
   public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t count) { Allocate(count); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept { Swap(other); }
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            Swap(other);
        }
        return *this;
    }

    ~DeviceBuffer() { Reset(); }

    void Allocate(std::size_t count) {
        Reset();
        if (count == 0) {
            return;
        }
        CUDA_CHECK(
            cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)));
        size_ = count;
    }

    void Reset() noexcept {
        if (data_ == nullptr) {
            return;
        }
        cudaFree(data_);
        data_ = nullptr;
        size_ = 0;
    }

    [[nodiscard]] T* get() const noexcept { return data_; }
    [[nodiscard]] std::size_t size() const noexcept { return size_; }

   private:
    void Swap(DeviceBuffer& other) noexcept {
        using std::swap;
        swap(data_, other.data_);
        swap(size_, other.size_);
    }

    T* data_ = nullptr;
    std::size_t size_ = 0;
};

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

    [[nodiscard]] cudaEvent_t get() const { return event_; }

   private:
    cudaEvent_t event_ = nullptr;
    std::shared_ptr<spdlog::logger> logger_;
};

void LaunchUvaPageCopyKernel(uint8_t** src_ptrs, uint8_t** dst_ptrs,
                             std::size_t page_size_bytes, int num_pages,
                             cudaStream_t stream);
}  // namespace worker_detail

struct KVAsyncTask {
    KVAsyncTask() = default;
    KVAsyncTask(std::uint64_t id, std::shared_future<void> future)
        : id_(id), future_(std::move(future)) {}

    [[nodiscard]] std::uint64_t id() const { return id_; }

    [[nodiscard]] bool done() const {
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

using SequenceLengthMap = batchgen::kv::SequenceLengthMap;
using SequenceLengthVector = batchgen::kv::SequenceLengthVector;
using SequenceLengths = batchgen::kv::SequenceLengths;

template <HostKVMode Mode, typename Layout = HostPagedKVLayout<Mode>>
class HostPagedKVWorkerView {
   public:
    using ModeTraits = HostKVModeTraits<Mode>;
    static constexpr HostKVMode kMode = Mode;
    static constexpr bool kHasVCache = Layout::kHasVCache;

        explicit HostPagedKVWorkerView(const EngineConfig& engine_config,
                                                                     const ModelConfig& model_config)
                : HostPagedKVWorkerView(config::BuildHostPagedKVConfig(
                            engine_config, model_config)) {}

    explicit HostPagedKVWorkerView(const HostPagedKVConfig& config)
        : config_(detail::SanitizeConfig(ModeTraits::Adjust(config))),
          geometry_(config_),
          layout_(config_),
          backend_(config_, layout_.DataSectionBytes(), layout_.Fingerprint(),
                   Layout::kHasVCache),
          logger_(init_logger("info",
              config_.logger_name.empty() ? "HostPagedKVWorkerView" : config_.logger_name)) {}

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
            AppendAllocatedPages(sequence_ids[i], allocated_pages[i]);
        }
        return allocated_pages;
    }

    std::vector<std::int32_t> GrowSequencePages(
        std::int64_t sequence_id, std::size_t num_pages) {
        if (num_pages == 0) {
            throw std::invalid_argument(
                "GrowSequencePages: num_pages must be greater than zero");
        }
        EnsureSequenceRegistered(sequence_id);
        auto new_pages = backend_.AcquirePages(sequence_id, num_pages);
        AppendAllocatedPages(sequence_id, new_pages);
        return new_pages;
    }

    std::vector<std::vector<std::int32_t>> GrowPagesForSequences(
        const std::vector<std::int64_t>& sequence_ids,
        const std::vector<std::size_t>& num_pages) {
        if (sequence_ids.size() != num_pages.size()) {
            throw std::invalid_argument(
                "sequence_ids and num_pages must have the same length");
        }
        if (sequence_ids.empty()) {
            return {};
        }
        EnsureSequencesRegistered(sequence_ids);
        std::vector<std::vector<std::int32_t>> allocations;
        allocations.reserve(sequence_ids.size());
        for (std::size_t i = 0; i < sequence_ids.size(); ++i) {
            const std::size_t pages = num_pages[i];
            if (pages == 0) {
                throw std::invalid_argument(
                    "GrowPagesForSequences: num_pages entries must be greater "
                    "than zero");
            }
            allocations.emplace_back(
                GrowSequencePages(sequence_ids[i], pages));
        }
        return allocations;
    }

    void Shutdown() {
        logger_->info("Shutting down HostPagedKVWorkerView (device_index={})",
                      device_index_);
        ResetCopyStreams();
        UnregisterPinnedMemory();
        page_table_.Clear();
    }

    KVAsyncTask AsyncLoadLayerKVToDevice(
        torch::Tensor sequence_ids, torch::Tensor k_device_ptrs,
        std::optional<torch::Tensor> v_device_ptrs = std::nullopt) {
        EnsureDeviceReady();
        constexpr std::string_view kOpName = "AsyncLoadLayerKVToDevice";
        auto batch = PrepareDevicePointerBatch(
            std::move(sequence_ids), std::move(k_device_ptrs),
            std::move(v_device_ptrs), kOpName);
        const auto sequence_vector = TensorToInt64Vector(batch.sequence_ids);
        EnsureSequencesRegistered(sequence_vector);
        const auto page_table = BuildPageTable(sequence_vector);
        const auto start = std::chrono::high_resolution_clock::now();
        const std::size_t total_pages = std::accumulate(
            page_table.begin(), page_table.end(), static_cast<std::size_t>(0),
            [](std::size_t sum, const std::vector<std::int32_t>& pages) {
                return sum + pages.size();
            });
        const auto pointer_columns =
            static_cast<std::size_t>(batch.k_device_ptrs.size(1));
        if (pointer_columns != total_pages) {
            std::ostringstream oss;
            oss << kOpName << ": pointer tensor columns (" << pointer_columns
                << ") must equal total host pages (" << total_pages << ")";
            throw std::invalid_argument(oss.str());
        }
        std::vector<std::size_t> sequence_offsets(page_table.size(), 0);
        std::size_t running_offset = 0;
        for (std::size_t seq_idx = 0; seq_idx < page_table.size(); ++seq_idx) {
            sequence_offsets[seq_idx] = running_offset;
            running_offset += page_table[seq_idx].size();
        }
        if (running_offset != total_pages) {
            throw std::logic_error(
                "AsyncLoadLayerKVToDevice: mismatch computing page offsets");
        }
        const std::size_t num_layers = config_.num_layers;
        const std::size_t copy_entries = num_layers * total_pages;
        if (copy_entries == 0) {
            return LaunchAsyncTask([] {});
        }
        const auto kernel_limit =
            static_cast<std::size_t>(std::numeric_limits<int>::max());
        if (copy_entries > kernel_limit) {
            std::ostringstream oss;
            oss << kOpName << ": num_pages=" << copy_entries
                << " exceeds kernel limit=" << kernel_limit;
            throw std::invalid_argument(oss.str());
        }
        const auto end = std::chrono::high_resolution_clock::now();
        double prep_ms =
            std::chrono::duration_cast<
                std::chrono::duration<double, std::milli>>(end - start)
                .count();
        logger_->debug(
            "Prepared AsyncLoadLayerKVToDevice (num_layers={}, total_pages={}, "
            "prep_time_ms={:.3f})",
            num_layers, total_pages, prep_ms);
        return LaunchAsyncTask([this, batch = std::move(batch),
                                page_table = std::move(page_table),
                                sequence_offsets = std::move(sequence_offsets),
                                total_pages, num_layers, pointer_columns,
                                copy_entries]() mutable {
            const auto start = std::chrono::high_resolution_clock::now();
            c10::cuda::OptionalCUDAGuard device_guard(device_index_);
            const auto cuda_stream = CopyStream(CopyDirection::kHostToDevice);
            const std::size_t k_page_bytes = layout_.KPageBytes();
            if (k_page_bytes == 0) {
                return;
            }

            auto* k_dest_ptr =
                batch.k_device_ptrs.template data_ptr<std::int64_t>();
            const std::int64_t* v_dest_ptr =
                batch.v_device_ptrs.has_value()
                    ? batch.v_device_ptrs->template data_ptr<std::int64_t>()
                    : nullptr;
            const auto row_stride = pointer_columns;
            auto build_plan = [&](const std::int64_t* dest_ptrs,
                                  auto&& host_ptr_provider) {
                return this->BuildPageCopyPlan(
                    page_table, sequence_offsets, num_layers, row_stride,
                    copy_entries, dest_ptrs,
                    std::forward<decltype(host_ptr_provider)>(
                        host_ptr_provider),
                    kOpName);
            };

            const auto k_plan = build_plan(
                k_dest_ptr,
                [this](std::size_t layer_idx, std::int32_t page_idx) -> void* {
                    return this->KPagePtr(layer_idx, page_idx);
                });
            const auto end = std::chrono::high_resolution_clock::now();
            double plan_ms =
                std::chrono::duration_cast<
                    std::chrono::duration<double, std::milli>>(end - start)
                    .count();
            logger_->debug(
                "Built K page copy plan (num_layers={}, total_pages={}, "
                "plan_time_ms={:.3f})",
                num_layers, total_pages, plan_ms);

            std::optional<PageCopyPlan> v_plan;
            if constexpr (kHasVCache) {
                if (v_dest_ptr != nullptr) {
                    v_plan = build_plan(v_dest_ptr,
                                        [this](std::size_t layer_idx,
                                               std::int32_t page_idx) -> void* {
                                            return this->template VPagePtr<>(
                                                layer_idx, page_idx);
                                        });
                }
            }

            worker_detail::DeviceBuffer<uint8_t*> k_device_src_ptrs(
                copy_entries);
            worker_detail::DeviceBuffer<uint8_t*> k_device_dst_ptrs(
                copy_entries);
            worker_detail::DeviceBuffer<uint8_t*> v_device_src_ptrs(
                v_plan.has_value() ? copy_entries : 0);
            worker_detail::DeviceBuffer<uint8_t*> v_device_dst_ptrs(
                v_plan.has_value() ? copy_entries : 0);

            auto enqueue_plan =
                [&](const PageCopyPlan& plan,
                    worker_detail::DeviceBuffer<uint8_t*>& dev_src_ptrs,
                    worker_detail::DeviceBuffer<uint8_t*>& dev_dst_ptrs,
                    std::size_t page_bytes) {
                    if (plan.host_sources.empty() || page_bytes == 0) {
                        return;
                    }
                    const std::size_t ptr_bytes =
                        plan.host_sources.size() * sizeof(uint8_t*);
                    EnqueueCopy(
                        reinterpret_cast<const std::byte*>(
                            plan.host_sources.data()),
                        reinterpret_cast<std::byte*>(dev_src_ptrs.get()),
                        ptr_bytes, CopyDirection::kHostToDevice, cuda_stream);
                    EnqueueCopy(
                        reinterpret_cast<const std::byte*>(
                            plan.device_dests.data()),
                        reinterpret_cast<std::byte*>(dev_dst_ptrs.get()),
                        ptr_bytes, CopyDirection::kHostToDevice, cuda_stream);
                    worker_detail::LaunchUvaPageCopyKernel(
                        dev_src_ptrs.get(), dev_dst_ptrs.get(), page_bytes,
                        static_cast<int>(plan.host_sources.size()),
                        cuda_stream);
                };

            enqueue_plan(k_plan, k_device_src_ptrs, k_device_dst_ptrs,
                         k_page_bytes);

            if constexpr (kHasVCache) {
                if (v_plan.has_value()) {
                    const std::size_t v_page_bytes = layout_.VPageBytes();
                    enqueue_plan(*v_plan, v_device_src_ptrs, v_device_dst_ptrs,
                                 v_page_bytes);
                }
            }
            this->logger_->debug(
                "AsyncLoadLayerKVToDevice completed (num_layers={}, "
                "total_pages={}, "
                "k_page_bytes={})",
                num_layers, total_pages, k_page_bytes);
            this->SynchronizeWithEvent(cuda_stream);
        });
    }

    KVAsyncTask AsyncLoadLayerPagedKVToDevice(
        torch::Tensor sequence_ids, torch::Tensor active_page_counts,
        torch::Tensor k_device_ptrs,
        std::optional<torch::Tensor> v_device_ptrs = std::nullopt) {
        EnsureDeviceReady();
        constexpr std::string_view kOpName =
            "AsyncLoadLayerPagedKVToDevice";

        auto validated_ids = ValidateCpuTensor1D(
            std::move(sequence_ids), torch::kInt64, "sequence_ids",
            kOpName);
        const auto batch_size =
            static_cast<std::size_t>(validated_ids.size(0));

        auto validated_counts = ValidatePageCountTensor(
            std::move(active_page_counts), batch_size, kOpName);

        auto validated_k_ptrs = ValidatePointerTensor3D(
            std::move(k_device_ptrs), "k_device_ptrs", batch_size,
            kOpName);

        std::optional<torch::Tensor> validated_v_ptrs;
        if (v_device_ptrs.has_value()) {
            if constexpr (!kHasVCache) {
                throw std::invalid_argument(std::string(kOpName) +
                                            ": V cache is disabled");
            }
            auto tensor = ValidatePointerTensor3D(
                std::move(*v_device_ptrs), "v_device_ptrs", batch_size,
                kOpName);
            if (tensor.sizes() != validated_k_ptrs.sizes()) {
                std::ostringstream oss;
                oss << kOpName
                    << ": v_device_ptrs must match k_device_ptrs shape";
                throw std::invalid_argument(oss.str());
            }
            validated_v_ptrs = std::move(tensor);
        }

        if (batch_size == 0) {
            return LaunchAsyncTask([] {});
        }

        const auto prep_start = std::chrono::high_resolution_clock::now();
        auto sequence_vector = TensorToInt64Vector(validated_ids);
        EnsureSequencesRegistered(sequence_vector);
        auto page_counts = TensorToSizeVector(
            validated_counts, "active_page_counts", kOpName);
        auto page_table = BuildPageTable(sequence_vector);

        const auto max_sequence_pages =
            static_cast<std::size_t>(validated_k_ptrs.size(2));
        std::vector<std::size_t> sequence_offsets(batch_size, 0);
        std::size_t total_pages = 0;
        for (std::size_t seq_idx = 0; seq_idx < batch_size; ++seq_idx) {
            const std::size_t requested = page_counts[seq_idx];
            if (requested > max_sequence_pages) {
                std::ostringstream oss;
                oss << kOpName << ": requested pages " << requested
                    << " exceed provided pointer tensor capacity "
                    << max_sequence_pages << " for sequence index "
                    << seq_idx;
                throw std::out_of_range(oss.str());
            }
            const auto available = page_table[seq_idx].size();
            if (requested > available) {
                std::ostringstream oss;
                oss << kOpName << ": requested pages " << requested
                    << " exceed host allocation " << available
                    << " for sequence " << sequence_vector[seq_idx];
                throw std::out_of_range(oss.str());
            }
            sequence_offsets[seq_idx] = total_pages;
            page_table[seq_idx].resize(requested);
            total_pages += requested;
        }

        if (total_pages == 0) {
            return LaunchAsyncTask([] {});
        }

        auto flattened_k_ptrs = FlattenActivePointerTensor(
            validated_k_ptrs, sequence_offsets, page_counts, total_pages,
            "k_device_ptrs", kOpName);

        std::optional<torch::Tensor> flattened_v_ptrs;
        if (validated_v_ptrs.has_value()) {
            flattened_v_ptrs = FlattenActivePointerTensor(
                *validated_v_ptrs, sequence_offsets, page_counts,
                total_pages, "v_device_ptrs", kOpName);
        }

        const std::size_t num_layers = config_.num_layers;
        const std::size_t copy_entries = num_layers * total_pages;
        if (copy_entries == 0) {
            return LaunchAsyncTask([] {});
        }
        const auto kernel_limit =
            static_cast<std::size_t>(std::numeric_limits<int>::max());
        if (copy_entries > kernel_limit) {
            std::ostringstream oss;
            oss << kOpName << ": num_pages=" << copy_entries
                << " exceeds kernel limit=" << kernel_limit;
            throw std::invalid_argument(oss.str());
        }

        const auto prep_end = std::chrono::high_resolution_clock::now();
        const double prep_ms =
            std::chrono::duration_cast<
                std::chrono::duration<double, std::milli>>(prep_end -
                                                          prep_start)
                .count();
        logger_->debug(
            "Prepared AsyncLoadLayerPagedKVToDevice (num_layers={}, total_pages={}, max_sequence_pages={}, prep_time_ms={:.3f})",
            num_layers, total_pages, max_sequence_pages, prep_ms);

        return LaunchAsyncTask([
            this,
            page_table = std::move(page_table),
            sequence_offsets = std::move(sequence_offsets),
            k_tensor = std::move(flattened_k_ptrs),
            v_tensor = std::move(flattened_v_ptrs),
            total_pages,
            num_layers,
            copy_entries,
            kOpName
        ]() mutable {
            const auto start = std::chrono::high_resolution_clock::now();
            c10::cuda::OptionalCUDAGuard device_guard(device_index_);
            const auto cuda_stream = CopyStream(CopyDirection::kHostToDevice);
            const std::size_t k_page_bytes = layout_.KPageBytes();
            if (k_page_bytes == 0) {
                return;
            }

            auto* k_dest_ptr = k_tensor.template data_ptr<std::int64_t>();
            const std::int64_t* v_dest_ptr =
                v_tensor.has_value()
                    ? v_tensor->data_ptr<std::int64_t>()
                    : nullptr;
            const std::size_t row_stride = total_pages;
            auto build_plan = [&](const std::int64_t* dest_ptrs,
                                  auto&& host_ptr_provider) {
                if (dest_ptrs == nullptr) {
                    throw std::invalid_argument(std::string(kOpName) +
                                                ": null device pointers");
                }
                return this->BuildPageCopyPlan(
                    page_table, sequence_offsets, num_layers, row_stride,
                    copy_entries, dest_ptrs,
                    std::forward<decltype(host_ptr_provider)>(
                        host_ptr_provider),
                    kOpName);
            };

            const auto k_plan = build_plan(
                k_dest_ptr,
                [this](std::size_t layer_idx, std::int32_t page_idx) -> void* {
                    return this->KPagePtr(layer_idx, page_idx);
                });
            const auto plan_end = std::chrono::high_resolution_clock::now();
            const double plan_ms =
                std::chrono::duration_cast<
                    std::chrono::duration<double, std::milli>>(plan_end -
                                                              start)
                    .count();
            logger_->debug(
                "Built paged K copy plan (num_layers={}, total_pages={}, plan_time_ms={:.3f})",
                num_layers, total_pages, plan_ms);

            std::optional<PageCopyPlan> v_plan;
            if constexpr (kHasVCache) {
                if (v_dest_ptr != nullptr) {
                    v_plan = build_plan(
                        v_dest_ptr, [this](std::size_t layer_idx,
                                           std::int32_t page_idx) -> void* {
                            return this->template VPagePtr<>(layer_idx,
                                                             page_idx);
                        });
                }
            }

            worker_detail::DeviceBuffer<uint8_t*> k_device_src_ptrs(
                copy_entries);
            worker_detail::DeviceBuffer<uint8_t*> k_device_dst_ptrs(
                copy_entries);
            worker_detail::DeviceBuffer<uint8_t*> v_device_src_ptrs(
                v_plan.has_value() ? copy_entries : 0);
            worker_detail::DeviceBuffer<uint8_t*> v_device_dst_ptrs(
                v_plan.has_value() ? copy_entries : 0);

            auto enqueue_plan =
                [&](const PageCopyPlan& plan,
                    worker_detail::DeviceBuffer<uint8_t*>& dev_src_ptrs,
                    worker_detail::DeviceBuffer<uint8_t*>& dev_dst_ptrs,
                    std::size_t page_bytes) {
                    if (plan.host_sources.empty() || page_bytes == 0) {
                        return;
                    }
                    const std::size_t ptr_bytes =
                        plan.host_sources.size() * sizeof(uint8_t*);
                    EnqueueCopy(
                        reinterpret_cast<const std::byte*>(
                            plan.host_sources.data()),
                        reinterpret_cast<std::byte*>(dev_src_ptrs.get()),
                        ptr_bytes, CopyDirection::kHostToDevice, cuda_stream);
                    EnqueueCopy(
                        reinterpret_cast<const std::byte*>(
                            plan.device_dests.data()),
                        reinterpret_cast<std::byte*>(dev_dst_ptrs.get()),
                        ptr_bytes, CopyDirection::kHostToDevice, cuda_stream);
                    worker_detail::LaunchUvaPageCopyKernel(
                        dev_src_ptrs.get(), dev_dst_ptrs.get(), page_bytes,
                        static_cast<int>(plan.host_sources.size()),
                        cuda_stream);
                };

            enqueue_plan(k_plan, k_device_src_ptrs, k_device_dst_ptrs,
                         k_page_bytes);

            if constexpr (kHasVCache) {
                if (v_plan.has_value()) {
                    const std::size_t v_page_bytes = layout_.VPageBytes();
                    enqueue_plan(*v_plan, v_device_src_ptrs,
                                 v_device_dst_ptrs, v_page_bytes);
                }
            }

            logger_->debug(
                "AsyncLoadLayerPagedKVToDevice completed (num_layers={}, total_pages={}, k_page_bytes={})",
                num_layers, total_pages, k_page_bytes);
            this->SynchronizeWithEvent(cuda_stream);
        });
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
        table.resize(sequence_ids.size());
#pragma omp parallel for
        for (std::size_t i = 0; i < sequence_ids.size(); ++i) {
            std::int64_t seq_id = sequence_ids[i];
            try {
                table[i] = page_table_.Pages(seq_id);
            } catch (const std::out_of_range&) {
            }
        }
        return table;
    }

    std::pair<std::vector<void*>, std::optional<std::vector<void*>>>
    GetSequenceLayerPagePointers(
        std::int64_t sequence_id, std::size_t layer_idx,
        std::optional<std::size_t> max_tokens = std::nullopt) const {
        geometry_.EnsureLayerBounds(
            layer_idx, "HostPagedKVWorkerView::GetSequenceLayerPagePointers");
        std::optional<std::size_t> max_pages;
        if (max_tokens.has_value()) {
            max_pages = geometry_.RequiredPages(max_tokens.value());
        }
        auto page_indices = backend_.SequencePages(sequence_id, max_pages);
        std::vector<void*> k_ptrs;
        k_ptrs.reserve(page_indices.size());
        std::optional<std::vector<void*>> v_ptrs;
        if constexpr (Layout::kHasVCache) {
            v_ptrs.emplace();
            v_ptrs->reserve(page_indices.size());
        }
        std::byte* base = const_cast<std::byte*>(backend_.DataBase());
        for (std::int32_t page : page_indices) {
            void* k_ptr =
                static_cast<void*>(layout_.KPageAddress(base, layer_idx, page));
            k_ptrs.emplace_back(k_ptr);
            // LogPageBytes("K", layer_idx, sequence_id, page, k_ptr,
            //              geometry_.KTokenBytes());
            if constexpr (Layout::kHasVCache) {
                void* v_ptr = static_cast<void*>(
                    layout_.VPageAddress(base, layer_idx, page));
                v_ptrs->emplace_back(v_ptr);
                // LogPageBytes("V", layer_idx, sequence_id, page, v_ptr,
                //              geometry_.template VTokenBytes<true>());
            }
        }
        return {std::move(k_ptrs), std::move(v_ptrs)};
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

    void ReleaseSequencePages(const std::vector<std::int64_t>& sequence_ids) {
        if (sequence_ids.empty()) {
            return;
        }
        std::unordered_set<std::int64_t> seen_ids;
        seen_ids.reserve(sequence_ids.size());
        for (std::int64_t sequence_id : sequence_ids) {
            if (!seen_ids.insert(sequence_id).second) {
                std::ostringstream oss;
                oss << "ReleaseSequencePages: duplicate sequence_id="
                    << sequence_id;
                throw std::runtime_error(oss.str());
            }
        }
        EnsureSequencesRegistered(sequence_ids);
        backend_.ReleaseSequences(sequence_ids);
        UnregisterSequences(sequence_ids);
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
        c10::cuda::OptionalCUDAGuard producer_guard(device_index_);
        const auto producer_cuda_stream =
            at::cuda::getCurrentCUDAStream(device_index_).stream();

        return LaunchAsyncTask([this, layer_idx,
                                 sequence_ids = std::move(sequence_ids),
                                 sequence_lengths = std::move(sequence_lengths),
                                 prepared_k, prepared_v, tokens_per_sequence,
                                 producer_cuda_stream]() {
            c10::cuda::OptionalCUDAGuard device_guard(device_index_);
            const auto cuda_stream = CopyStream(CopyDirection::kDeviceToHost);
            this->WaitForProducerStream(cuda_stream, producer_cuda_stream);

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
        c10::cuda::OptionalCUDAGuard producer_guard(device_index_);
        const auto producer_cuda_stream =
            at::cuda::getCurrentCUDAStream(device_index_).stream();

        return LaunchAsyncTask([this, layer_idx,
                                 sequence_ids = std::move(sequence_ids),
                                 sequence_lengths = std::move(sequence_lengths),
                                 prepared_k, prepared_v,
                                 producer_cuda_stream]() mutable {
            c10::cuda::OptionalCUDAGuard device_guard(device_index_);
            const auto cuda_stream = CopyStream(CopyDirection::kDeviceToHost);
            this->WaitForProducerStream(cuda_stream, producer_cuda_stream);

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

    // ================================================================
    // Direct host→CPU read/write for migration (no GPU staging)
    // ================================================================

    /**
     * Read all KV pages for a sequence directly to CPU tensors.
     * Returns (k_tensor, v_tensor). For MLA mode, v_tensor is empty (0-dim).
     * Shape: [num_layers, num_pages, page_size_tokens, num_heads, head_dim]
     */
    std::pair<torch::Tensor, torch::Tensor> ReadSequenceKVToCPU(
        std::int64_t sequence_id) const {
        const auto pages = page_table_.Pages(sequence_id);
        const std::size_t num_pages = pages.size();
        const std::size_t num_layers = config_.num_layers;
        const std::size_t page_tokens = config_.page_size_tokens;
        const std::size_t k_heads = config_.num_k_heads;
        const std::size_t k_dim = config_.k_head_dim;
        const std::size_t k_elem = config_.k_element_size_bytes;
        const std::size_t k_page_bytes = layout_.KPageBytes();

        // Determine torch dtype from element size
        auto k_dtype = (k_elem == 2) ? torch::kBFloat16 : torch::kFloat32;

        auto k_out = torch::empty(
            {(int64_t)num_layers, (int64_t)num_pages, (int64_t)page_tokens,
             (int64_t)k_heads, (int64_t)k_dim},
            torch::TensorOptions().dtype(k_dtype).device(torch::kCPU));

        auto* k_ptr = static_cast<std::byte*>(k_out.data_ptr());
        const std::byte* host_base = backend_.DataBase();

        for (std::size_t layer = 0; layer < num_layers; ++layer) {
            for (std::size_t p = 0; p < num_pages; ++p) {
                const void* src =
                    layout_.KPageAddress(host_base, layer, pages[p]);
                void* dst = k_ptr + (layer * num_pages + p) * k_page_bytes;
                std::memcpy(dst, src, k_page_bytes);
            }
        }

        // V cache (MHA only)
        torch::Tensor v_out;
        if constexpr (kHasVCache) {
            const std::size_t v_heads = config_.num_v_heads;
            const std::size_t v_dim = config_.v_head_dim;
            const std::size_t v_elem = config_.v_element_size_bytes;
            const std::size_t v_page_bytes = layout_.VPageBytes();
            auto v_dtype = (v_elem == 2) ? torch::kBFloat16 : torch::kFloat32;

            v_out = torch::empty(
                {(int64_t)num_layers, (int64_t)num_pages,
                 (int64_t)page_tokens, (int64_t)v_heads, (int64_t)v_dim},
                torch::TensorOptions().dtype(v_dtype).device(torch::kCPU));

            auto* v_ptr = static_cast<std::byte*>(v_out.data_ptr());
            for (std::size_t layer = 0; layer < num_layers; ++layer) {
                for (std::size_t p = 0; p < num_pages; ++p) {
                    const void* src =
                        layout_.VPageAddress(host_base, layer, pages[p]);
                    void* dst =
                        v_ptr + (layer * num_pages + p) * v_page_bytes;
                    std::memcpy(dst, src, v_page_bytes);
                }
            }
        } else {
            v_out = torch::empty({0}, torch::kBFloat16);
        }

        logger_->info(
            "ReadSequenceKVToCPU: seq={}, pages={}, layers={}, k_page_bytes={}",
            sequence_id, num_pages, num_layers, k_page_bytes);

        return {std::move(k_out), std::move(v_out)};
    }

    /**
     * Write KV data from CPU tensors directly to host pages.
     * For MLA mode, v_tensor is ignored.
     * k_tensor shape: [num_layers, num_pages, page_size_tokens, num_k_heads, k_head_dim]
     */
    void WriteSequenceKVFromCPU(std::int64_t sequence_id,
                                const torch::Tensor& k_tensor,
                                const std::optional<torch::Tensor>& v_tensor = std::nullopt) {
        const auto pages = page_table_.Pages(sequence_id);
        const std::size_t num_pages = pages.size();
        const std::size_t num_layers = config_.num_layers;
        const std::size_t k_page_bytes = layout_.KPageBytes();

        if ((std::size_t)k_tensor.size(0) != num_layers ||
            (std::size_t)k_tensor.size(1) != num_pages) {
            throw std::invalid_argument(
                "WriteSequenceKVFromCPU: k_tensor shape mismatch "
                "(expected layers=" +
                std::to_string(num_layers) +
                " pages=" + std::to_string(num_pages) + ")");
        }

        const auto* k_ptr =
            static_cast<const std::byte*>(k_tensor.data_ptr());
        std::byte* host_base = backend_.DataBase();

        for (std::size_t layer = 0; layer < num_layers; ++layer) {
            for (std::size_t p = 0; p < num_pages; ++p) {
                void* dst =
                    layout_.KPageAddress(host_base, layer, pages[p]);
                const void* src =
                    k_ptr + (layer * num_pages + p) * k_page_bytes;
                std::memcpy(dst, src, k_page_bytes);
            }
        }

        if constexpr (kHasVCache) {
            if (v_tensor.has_value() && v_tensor->numel() > 0) {
                const std::size_t v_page_bytes = layout_.VPageBytes();
                const auto* v_ptr =
                    static_cast<const std::byte*>(v_tensor->data_ptr());
                for (std::size_t layer = 0; layer < num_layers; ++layer) {
                    for (std::size_t p = 0; p < num_pages; ++p) {
                        void* dst = layout_.VPageAddress(host_base, layer,
                                                        pages[p]);
                        const void* src =
                            v_ptr + (layer * num_pages + p) * v_page_bytes;
                        std::memcpy(dst, src, v_page_bytes);
                    }
                }
            }
        }

        logger_->info(
            "WriteSequenceKVFromCPU: seq={}, pages={}, layers={}",
            sequence_id, num_pages, num_layers);
    }

   private:
    struct PageLocation {
        std::int32_t page_idx = -1;
        std::size_t page_offset_tokens = 0;
    };

    struct DevicePointerBatch {
        torch::Tensor sequence_ids;
        torch::Tensor k_device_ptrs;
        std::optional<torch::Tensor> v_device_ptrs;
    };

    struct PageCopyPlan {
        std::vector<uint8_t*> host_sources;
        std::vector<uint8_t*> device_dests;
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

    void AppendAllocatedPages(std::int64_t sequence_id,
                              const std::vector<std::int32_t>& pages) {
        if (pages.empty()) {
            return;
        }
        if (page_table_.Contains(sequence_id)) {
            page_table_.AppendPages(sequence_id, pages);
        } else {
            page_table_.RegisterOrUpdate(sequence_id, pages);
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

    void LogDecodeNeighborhood(std::size_t layer_idx,
                               const std::vector<std::int64_t>& sequence_ids,
                               const SequenceLengths& sequence_lengths,
                               std::byte* host_base,
                               std::size_t neighborhood = 1) const {
        if (host_base == nullptr) {
            logger_->warn("LogDecodeNeighborhood: host_base is null, skipping");
            return;
        }

        const std::size_t k_token_bytes = geometry_.KTokenBytes();

        for (std::size_t batch_idx = 0; batch_idx < sequence_ids.size();
             ++batch_idx) {
            const std::int64_t sequence_id = sequence_ids[batch_idx];
            const std::size_t start_token =
                ResolveSequenceLength(sequence_lengths, batch_idx, sequence_id,
                                      std::nullopt, "LogDecodeNeighborhood");

            const auto pages = page_table_.Pages(sequence_id);

            const std::size_t begin =
                (start_token > neighborhood) ? start_token - neighborhood : 0;
            const std::size_t end = start_token + neighborhood;

            for (std::size_t token_idx = begin; token_idx <= end; ++token_idx) {
                try {
                    geometry_.ValidatePageCapacity(pages, token_idx + 1,
                                                   "LogDecodeNeighborhood");
                } catch (const std::exception& ex) {
                    logger_->warn(
                        "LogDecodeNeighborhood: layer={} seq={} token_idx={} "
                        "exceeds capacity: {}",
                        layer_idx, sequence_id, token_idx, ex.what());
                    continue;
                }

                const auto location = ResolvePageLocation(
                    pages, sequence_id, token_idx, "LogDecodeNeighborhood");

                const std::byte* token_ptr =
                    layout_.KPageAddress(host_base, layer_idx,
                                         location.page_idx) +
                    location.page_offset_tokens * k_token_bytes;

                logger_->info(
                    "DecodeDebug layer={} seq={} token_idx={} "
                    "(page_idx={}, page_offset={}) first_token_bytes={}",
                    layer_idx, sequence_id, token_idx, location.page_idx,
                    location.page_offset_tokens,
                    geometry_.DescribeBytes(token_ptr, k_token_bytes));
            }
        }
    }

    void LogPageBytes(const char* tag, std::size_t layer_idx,
                      std::int64_t sequence_id, std::int32_t page_idx,
                      const void* ptr, std::size_t token_bytes) const {
        if (logger_ == nullptr || ptr == nullptr || token_bytes == 0) {
            return;
        }
        const auto bytes = geometry_.DescribeBytes(
            static_cast<const std::byte*>(ptr), token_bytes);
        logger_->info(
            "GetSequenceLayerPagePointers {} layer={} seq={} page={} "
            "first_bytes={}",
            tag, layer_idx, sequence_id, page_idx, bytes);
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

    DevicePointerBatch PrepareDevicePointerBatch(
        torch::Tensor sequence_ids, torch::Tensor k_device_ptrs,
        std::optional<torch::Tensor> v_device_ptrs,
        std::string_view op_name) const {
        DevicePointerBatch batch;
        batch.sequence_ids = ValidateCpuTensor1D(
            std::move(sequence_ids), torch::kInt64, "sequence_ids", op_name);
        batch.k_device_ptrs = ValidatePointerMatrix(std::move(k_device_ptrs),
                                                    "k_device_ptrs", op_name);
        if (v_device_ptrs.has_value()) {
            if constexpr (!kHasVCache) {
                throw std::invalid_argument(
                    std::string(op_name) +
                    ": v_device_ptrs provided but V cache is disabled");
            }
            auto validated = ValidatePointerMatrix(std::move(*v_device_ptrs),
                                                   "v_device_ptrs", op_name);
            if (validated.sizes() != batch.k_device_ptrs.sizes()) {
                std::ostringstream oss;
                oss << op_name
                    << ": v_device_ptrs must match k_device_ptrs shape";
                throw std::invalid_argument(oss.str());
            }
            batch.v_device_ptrs = std::move(validated);
        }

        if (batch.k_device_ptrs.numel() > 0 &&
            batch.k_device_ptrs.le(0).any().template item<bool>()) {
            std::ostringstream oss;
            oss << op_name << ": k_device_ptrs must be non-zero";
            throw std::invalid_argument(oss.str());
        }
        if (batch.v_device_ptrs.has_value() &&
            batch.v_device_ptrs->numel() > 0 &&
            batch.v_device_ptrs->le(0).any().template item<bool>()) {
            std::ostringstream oss;
            oss << op_name << ": v_device_ptrs must be non-zero";
            throw std::invalid_argument(oss.str());
        }

        return batch;
    }

    template <typename HostPtrProvider>
    PageCopyPlan BuildPageCopyPlan(
        const std::vector<std::vector<std::int32_t>>& page_table,
        const std::vector<std::size_t>& sequence_offsets,
        std::size_t num_layers, std::size_t row_stride,
        std::size_t total_entries, const std::int64_t* device_ptr_matrix,
        HostPtrProvider&& host_ptr_provider, std::string_view op_name) const {
        if (device_ptr_matrix == nullptr) {
            throw std::invalid_argument(std::string(op_name) +
                                        ": device pointer tensor is null");
        }
        PageCopyPlan plan;
        plan.host_sources.resize(total_entries);
        plan.device_dests.resize(total_entries);
        auto&& provider = std::forward<HostPtrProvider>(host_ptr_provider);
        std::size_t cursor = 0;
        for (std::size_t layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
            const std::size_t layer_offset = layer_idx * row_stride;
            for (std::size_t seq_idx = 0; seq_idx < page_table.size();
                 ++seq_idx) {
                const auto& pages = page_table[seq_idx];
                const std::size_t seq_offset = sequence_offsets[seq_idx];
                for (std::size_t slot = 0; slot < pages.size(); ++slot) {
                    const std::int32_t page_idx = pages[slot];
                    if (page_idx < 0) {
                        std::ostringstream oss;
                        oss << op_name << ": negative page index=" << page_idx;
                        throw std::out_of_range(oss.str());
                    }
                    const std::size_t column = seq_offset + slot;
                    if (column >= row_stride) {
                        std::ostringstream oss;
                        oss << op_name
                            << ": pointer tensor columns insufficient for "
                               "sequence slot (column="
                            << column << ")";
                        throw std::out_of_range(oss.str());
                    }
                    const auto dest_raw =
                        device_ptr_matrix[layer_offset + column];
                    if (dest_raw == 0) {
                        std::ostringstream oss;
                        oss << op_name
                            << ": missing device pointer for sequence slot"
                            << " (column=" << column << ")";
                        throw std::runtime_error(oss.str());
                    }
                    auto* host_ptr =
                        static_cast<uint8_t*>(provider(layer_idx, page_idx));
                    auto* device_ptr = reinterpret_cast<uint8_t*>(
                        static_cast<std::uintptr_t>(dest_raw));
                    plan.host_sources[cursor] = host_ptr;
                    plan.device_dests[cursor] = device_ptr;
                    ++cursor;
                }
            }
        }
        if (cursor != total_entries) {
            std::ostringstream oss;
            oss << op_name << ": expected " << total_entries
                << " entries but prepared " << cursor;
            throw std::logic_error(oss.str());
        }
        return plan;
    }

    torch::Tensor ValidateCpuTensor1D(torch::Tensor tensor,
                                      torch::ScalarType dtype,
                                      std::string_view tensor_name,
                                      std::string_view op_name) const {
        if (tensor.device().type() != torch::kCPU) {
            std::ostringstream oss;
            oss << op_name << ": " << tensor_name
                << " must be on CPU (got device " << tensor.device().str()
                << ")";
            throw std::invalid_argument(oss.str());
        }
        if (tensor.dim() != 1) {
            std::ostringstream oss;
            oss << op_name << ": " << tensor_name << " must be 1-D";
            throw std::invalid_argument(oss.str());
        }
        if (tensor.scalar_type() != dtype) {
            std::ostringstream oss;
            oss << op_name << ": " << tensor_name << " must have dtype "
                << c10::toString(dtype) << " (got "
                << c10::toString(tensor.scalar_type()) << ")";
            throw std::invalid_argument(oss.str());
        }
        if (!tensor.is_contiguous()) {
            std::ostringstream oss;
            oss << op_name << ": " << tensor_name << " must be contiguous";
            throw std::invalid_argument(oss.str());
        }
        return tensor;
    }

    torch::Tensor ValidatePointerMatrix(torch::Tensor tensor,
                                        std::string_view tensor_name,
                                        std::string_view op_name) const {
        if (tensor.device().type() != torch::kCPU) {
            std::ostringstream oss;
            oss << op_name << ": " << tensor_name << " must reside on CPU (got "
                << tensor.device().str() << ')';
            throw std::invalid_argument(oss.str());
        }
        if (tensor.scalar_type() != torch::kInt64) {
            std::ostringstream oss;
            oss << op_name << ": " << tensor_name
                << " must have dtype int64 (got "
                << c10::toString(tensor.scalar_type()) << ')';
            throw std::invalid_argument(oss.str());
        }
        if (tensor.dim() != 2) {
            std::ostringstream oss;
            oss << op_name << ": " << tensor_name
                << " must be 2-D (got dim=" << tensor.dim() << ')';
            throw std::invalid_argument(oss.str());
        }
        if (!tensor.is_contiguous()) {
            std::ostringstream oss;
            oss << op_name << ": " << tensor_name << " must be contiguous";
            throw std::invalid_argument(oss.str());
        }
        if (tensor.size(0) != static_cast<long>(config_.num_layers)) {
            std::ostringstream oss;
            oss << op_name << ": first dimension of " << tensor_name
                << " must equal num_layers (" << tensor.size(0)
                << " != " << config_.num_layers << ')';
            throw std::out_of_range(oss.str());
        }
        return tensor;
    }

    torch::Tensor ValidatePointerTensor3D(
        torch::Tensor tensor, std::string_view tensor_name,
        std::size_t expected_sequences, std::string_view op_name) const {
        if (tensor.device().type() != torch::kCPU) {
            std::ostringstream oss;
            oss << op_name << ": " << tensor_name
                << " must reside on CPU (got " << tensor.device().str()
                << ')';
            throw std::invalid_argument(oss.str());
        }
        if (tensor.scalar_type() != torch::kInt64) {
            std::ostringstream oss;
            oss << op_name << ": " << tensor_name
                << " must have dtype int64 (got "
                << c10::toString(tensor.scalar_type()) << ')';
            throw std::invalid_argument(oss.str());
        }
        if (!tensor.is_contiguous()) {
            std::ostringstream oss;
            oss << op_name << ": " << tensor_name
                << " must be contiguous";
            throw std::invalid_argument(oss.str());
        }
        if (tensor.dim() != 3) {
            std::ostringstream oss;
            oss << op_name << ": " << tensor_name
                << " must be 3-D (got dim=" << tensor.dim() << ')';
            throw std::invalid_argument(oss.str());
        }
        if (tensor.size(0) != static_cast<long>(config_.num_layers)) {
            std::ostringstream oss;
            oss << op_name << ": first dimension of " << tensor_name
                << " must equal num_layers";
            throw std::out_of_range(oss.str());
        }
        if (tensor.size(1) != static_cast<long>(expected_sequences)) {
            std::ostringstream oss;
            oss << op_name << ": second dimension of " << tensor_name
                << " must equal sequence count (expected "
                << expected_sequences << ", got " << tensor.size(1)
                << ')';
            throw std::out_of_range(oss.str());
        }
        return tensor;
    }

    torch::Tensor ValidatePageCountTensor(
        torch::Tensor tensor, std::size_t expected_length,
        std::string_view op_name) const {
        if (tensor.device().type() != torch::kCPU) {
            std::ostringstream oss;
            oss << op_name << ": active_page_counts must reside on CPU";
            throw std::invalid_argument(oss.str());
        }
        if (!tensor.is_contiguous()) {
            std::ostringstream oss;
            oss << op_name
                << ": active_page_counts must be contiguous";
            throw std::invalid_argument(oss.str());
        }
        if (tensor.dim() != 1) {
            std::ostringstream oss;
            oss << op_name
                << ": active_page_counts must be 1-D";
            throw std::invalid_argument(oss.str());
        }
        if (tensor.size(0) != static_cast<long>(expected_length)) {
            std::ostringstream oss;
            oss << op_name << ": active_page_counts length="
                << tensor.size(0)
                << " does not match sequence_ids length="
                << expected_length;
            throw std::out_of_range(oss.str());
        }
        if (tensor.scalar_type() != torch::kInt64 &&
            tensor.scalar_type() != torch::kInt32) {
            std::ostringstream oss;
            oss << op_name
                << ": active_page_counts must have dtype int32 or int64";
            throw std::invalid_argument(oss.str());
        }
        return tensor;
    }

    std::vector<std::size_t> TensorToSizeVector(
        const torch::Tensor& tensor, std::string_view tensor_name,
        std::string_view op_name) const {
        const auto length = static_cast<std::size_t>(tensor.size(0));
        std::vector<std::size_t> values(length, 0);
        if (tensor.scalar_type() == torch::kInt64) {
            const auto* data = tensor.data_ptr<std::int64_t>();
            for (std::size_t idx = 0; idx < length; ++idx) {
                const auto value = data[idx];
                if (value < 0) {
                    std::ostringstream oss;
                    oss << op_name << ": " << tensor_name
                        << " must be non-negative (index=" << idx
                        << ", value=" << value << ')';
                    throw std::out_of_range(oss.str());
                }
                values[idx] = static_cast<std::size_t>(value);
            }
            return values;
        }
        const auto* data = tensor.data_ptr<std::int32_t>();
        for (std::size_t idx = 0; idx < length; ++idx) {
            const auto value = data[idx];
            if (value < 0) {
                std::ostringstream oss;
                oss << op_name << ": " << tensor_name
                    << " must be non-negative (index=" << idx
                    << ", value=" << value << ')';
                throw std::out_of_range(oss.str());
            }
            values[idx] = static_cast<std::size_t>(value);
        }
        return values;
    }

    torch::Tensor FlattenActivePointerTensor(
        const torch::Tensor& tensor,
        const std::vector<std::size_t>& sequence_offsets,
        const std::vector<std::size_t>& page_counts,
        std::size_t total_pages, std::string_view tensor_name,
        std::string_view op_name) const {
        if (total_pages == 0) {
            throw std::logic_error(std::string(op_name) +
                                   ": total_pages must be > 0");
        }
        const auto seq_count = static_cast<std::size_t>(tensor.size(1));
        if (seq_count != page_counts.size()) {
            throw std::logic_error(std::string(op_name) +
                                   ": page_counts size mismatch");
        }
        const auto max_pages = static_cast<std::size_t>(tensor.size(2));
        auto result = torch::empty(
            {tensor.size(0), static_cast<long>(total_pages)},
            tensor.options());
        const auto num_layers = static_cast<std::size_t>(tensor.size(0));
        const auto* src = tensor.data_ptr<std::int64_t>();
        auto* dst = result.data_ptr<std::int64_t>();
        const std::size_t layer_stride = seq_count * max_pages;
        const std::size_t seq_stride = max_pages;
        for (std::size_t layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
            const auto* layer_src = src + layer_idx * layer_stride;
            auto* layer_dst = dst + layer_idx * total_pages;
            for (std::size_t seq_idx = 0; seq_idx < seq_count; ++seq_idx) {
                const auto count = page_counts[seq_idx];
                if (count == 0) {
                    continue;
                }
                if (count > max_pages) {
                    std::ostringstream oss;
                    oss << op_name << ": " << tensor_name
                        << " lacks capacity for sequence index " << seq_idx;
                    throw std::out_of_range(oss.str());
                }
                const auto seq_offset = sequence_offsets[seq_idx];
                if (seq_offset + count > total_pages) {
                    throw std::logic_error(std::string(op_name) +
                                           ": sequence offsets overflow");
                }
                const auto* seq_src = layer_src + seq_idx * seq_stride;
                auto* seq_dst = layer_dst + seq_offset;
                for (std::size_t slot = 0; slot < count; ++slot) {
                    const auto value = seq_src[slot];
                    if (value == 0) {
                        std::ostringstream oss;
                        oss << op_name << ": " << tensor_name
                            << " contains null pointer for sequence index "
                            << seq_idx << " slot " << slot;
                        throw std::runtime_error(oss.str());
                    }
                    seq_dst[slot] = value;
                }
            }
        }
        return result;
    }

    std::vector<std::int64_t> TensorToInt64Vector(
        const torch::Tensor& tensor) const {
        const auto length = static_cast<std::size_t>(tensor.size(0));
        std::vector<std::int64_t> values(length);
        if (length == 0) {
            return values;
        }
        const auto* data = tensor.data_ptr<std::int64_t>();
        std::copy(data, data + length, values.begin());
        return values;
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
        worker_detail::ScopedCudaEvent event(logger_);
        CUDA_CHECK(cudaEventRecord(event.get(), stream));
        CUDA_CHECK(cudaEventSynchronize(event.get()));
    }

    void WaitForProducerStream(cudaStream_t consumer_stream,
                               cudaStream_t producer_stream) const {
        if (producer_stream == nullptr || consumer_stream == producer_stream) {
            return;
        }
        worker_detail::ScopedCudaEvent event(logger_);
        CUDA_CHECK(cudaEventRecord(event.get(), producer_stream));
        CUDA_CHECK(cudaStreamWaitEvent(consumer_stream, event.get(), 0));
    }

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
