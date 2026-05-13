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
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <exception>
#include <functional>
#include <future>
#include <iomanip>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
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
#include "host_prefix_cache.h"
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

inline std::size_t ReadEnvSize(const char* name, std::size_t fallback,
                               std::size_t min_value,
                               std::size_t max_value) {
    const char* raw = std::getenv(name);
    if (raw == nullptr || *raw == '\0') {
        return fallback;
    }
    char* end = nullptr;
    const unsigned long parsed = std::strtoul(raw, &end, 10);
    if (end == raw || parsed == 0) {
        return fallback;
    }
    return std::min<std::size_t>(
        max_value, std::max<std::size_t>(min_value, parsed));
}

class BoundedAsyncExecutor {
   public:
    static BoundedAsyncExecutor& Instance() {
        static BoundedAsyncExecutor executor;
        return executor;
    }

    template <typename Fn>
    std::shared_future<void> Submit(Fn&& fn) {
        auto task =
            std::make_shared<std::packaged_task<void()>>(std::forward<Fn>(fn));
        auto future = task->get_future().share();
        {
            std::unique_lock<std::mutex> lock(mutex_);
            queue_space_cv_.wait(lock, [this]() {
                return stopping_ || queue_.size() < max_queue_depth_;
            });
            if (stopping_) {
                throw std::runtime_error(
                    "Host KV async executor is shutting down");
            }
            queue_.emplace_back([task = std::move(task)]() { (*task)(); });
        }
        queue_cv_.notify_one();
        return future;
    }

   private:
    BoundedAsyncExecutor()
        : max_queue_depth_(ReadEnvSize("BATCHGEN_HOST_KV_ASYNC_QUEUE_DEPTH",
                                       2048, 1, 1 << 20)) {
        const unsigned int hw = std::max(1u, std::thread::hardware_concurrency());
        const std::size_t default_threads =
            std::min<std::size_t>(16, std::max<std::size_t>(4, hw / 8));
        const std::size_t num_threads =
            ReadEnvSize("BATCHGEN_HOST_KV_ASYNC_THREADS", default_threads, 1,
                        256);
        workers_.reserve(num_threads);
        for (std::size_t idx = 0; idx < num_threads; ++idx) {
            workers_.emplace_back([this]() { WorkerLoop(); });
        }
    }

    ~BoundedAsyncExecutor() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        queue_cv_.notify_all();
        queue_space_cv_.notify_all();
        for (auto& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    BoundedAsyncExecutor(const BoundedAsyncExecutor&) = delete;
    BoundedAsyncExecutor& operator=(const BoundedAsyncExecutor&) = delete;

    void WorkerLoop() {
        while (true) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                queue_cv_.wait(lock, [this]() {
                    return stopping_ || !queue_.empty();
                });
                if (stopping_ && queue_.empty()) {
                    return;
                }
                task = std::move(queue_.front());
                queue_.pop_front();
            }
            queue_space_cv_.notify_one();
            task();
        }
    }

    const std::size_t max_queue_depth_;
    std::mutex mutex_;
    std::condition_variable queue_cv_;
    std::condition_variable queue_space_cv_;
    std::deque<std::function<void()>> queue_;
    std::vector<std::thread> workers_;
    bool stopping_ = false;
};
}  // namespace worker_detail

class KVLayerCompletionState {
   public:
    explicit KVLayerCompletionState(std::size_t num_layers)
        : events_(num_layers, nullptr), ready_(num_layers, false) {}

    KVLayerCompletionState(const KVLayerCompletionState&) = delete;
    KVLayerCompletionState& operator=(const KVLayerCompletionState&) = delete;

    ~KVLayerCompletionState() {
        for (cudaEvent_t event : events_) {
            if (event != nullptr) {
                cudaEventDestroy(event);
            }
        }
    }

    [[nodiscard]] std::size_t num_layers() const { return events_.size(); }

    void RecordLayerEvent(std::size_t layer_idx, cudaEvent_t event) {
        std::lock_guard<std::mutex> lock(mutex_);
        CheckLayerBounds(layer_idx);
        if (ready_[layer_idx]) {
            throw std::runtime_error("Layer completion event recorded twice");
        }
        events_[layer_idx] = event;
        ready_[layer_idx] = true;
        cv_.notify_all();
    }

    void SetFailure(std::exception_ptr failure) {
        std::lock_guard<std::mutex> lock(mutex_);
        failure_ = std::move(failure);
        cv_.notify_all();
    }

    void WaitLayer(std::size_t layer_idx) const {
        cudaEvent_t event = nullptr;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            CheckLayerBounds(layer_idx);
            cv_.wait(lock, [this, layer_idx]() {
                return ready_[layer_idx] || failure_ != nullptr;
            });
            if (!ready_[layer_idx]) {
                std::rethrow_exception(failure_);
            }
            event = events_[layer_idx];
        }

        auto stream = c10::cuda::getCurrentCUDAStream();
        CUDA_CHECK(cudaStreamWaitEvent(stream.stream(), event, 0));
    }

   private:
    void CheckLayerBounds(std::size_t layer_idx) const {
        if (layer_idx >= events_.size()) {
            std::ostringstream oss;
            oss << "layer_idx=" << layer_idx
                << " is out of range for layered KV async task with "
                << events_.size() << " layers";
            throw std::out_of_range(oss.str());
        }
    }

    mutable std::mutex mutex_;
    mutable std::condition_variable cv_;
    std::vector<cudaEvent_t> events_;
    std::vector<bool> ready_;
    std::exception_ptr failure_;
};

struct KVAsyncTask {
    KVAsyncTask() = default;
    KVAsyncTask(std::uint64_t id, std::shared_future<void> future)
        : id_(id), future_(std::move(future)) {}
    KVAsyncTask(std::uint64_t id, std::shared_future<void> future,
                std::shared_ptr<KVLayerCompletionState> layer_state)
        : id_(id),
          future_(std::move(future)),
          layer_state_(std::move(layer_state)) {}

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
            future_.get();
        }
    }

    void wait_layer(std::size_t layer_idx) const {
        if (layer_state_ == nullptr) {
            wait();
            return;
        }
        layer_state_->WaitLayer(layer_idx);
    }

    [[nodiscard]] std::size_t layer_count() const {
        return layer_state_ == nullptr ? 0 : layer_state_->num_layers();
    }

    void result() const {
        if (future_.valid()) {
            future_.get();
        }
    }

   private:
    std::uint64_t id_ = 0;
    std::shared_future<void> future_;
    std::shared_ptr<KVLayerCompletionState> layer_state_;
};

using SequenceLengthMap = std::unordered_map<std::int64_t, std::size_t>;
using SequenceLengthVector = std::vector<std::size_t>;
using SequenceLengths = std::variant<SequenceLengthMap, SequenceLengthVector>;

using SequenceLengthMap = batchgen::kv::SequenceLengthMap;
using SequenceLengthVector = batchgen::kv::SequenceLengthVector;
using SequenceLengths = batchgen::kv::SequenceLengths;

struct PrefixAllocationRequest {
    std::int64_t sequence_id = 0;
    std::vector<std::int64_t> token_ids;
    std::size_t capacity_tokens = 0;
    std::uint64_t namespace_hash = 0;
};

struct PrefixAllocationResult {
    std::int64_t sequence_id = 0;
    std::vector<std::int32_t> shared_prefix_pages;
    std::vector<std::int32_t> private_pages;
    std::size_t shared_prefix_tokens = 0;
    std::size_t private_start_token = 0;
    std::size_t logical_page_count = 0;
    std::size_t physical_pages_allocated = 0;
    bool full_hit = false;
    std::string miss_reason;
};

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
            ClearPrefixCache();
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

    std::vector<PrefixAllocationResult> AllocatePagesForSequencesWithPrefix(
        const std::vector<PrefixAllocationRequest>& requests) {
        if (requests.empty()) {
            return {};
        }
        std::vector<std::int64_t> sequence_ids;
        sequence_ids.reserve(requests.size());
        for (const auto& request : requests) {
            sequence_ids.push_back(request.sequence_id);
        }
        EnsureSequencesRegistered(sequence_ids);

        struct AllocationPlan {
            const PrefixAllocationRequest* request = nullptr;
            PrefixLookupResult hit;
            std::size_t private_pages_required = 0;
            std::vector<std::int32_t> private_pages;
            bool attached_shared_pages = false;
        };

        std::vector<AllocationPlan> plans;
        plans.reserve(requests.size());
        std::unordered_set<std::int32_t> protected_pages;
        std::size_t total_private_pages_required = 0;

        for (const auto& request : requests) {
            if (request.token_ids.empty()) {
                throw std::invalid_argument(
                    "Prefix allocation requires at least one prompt token");
            }
            if (request.capacity_tokens < request.token_ids.size()) {
                throw std::invalid_argument(
                    "capacity_tokens must be >= token_ids.size()");
            }

            const PrefixLookupResult hit = prefix_cache_.Lookup(
                request.namespace_hash,
                static_cast<std::int32_t>(config_.page_size_tokens),
                request.token_ids);
            const std::size_t private_tokens =
                request.capacity_tokens - hit.matched_tokens;
            const std::size_t private_pages_required =
                private_tokens == 0 ? 0 : geometry_.RequiredPages(private_tokens);
            total_private_pages_required += private_pages_required;
            for (std::int32_t page : hit.host_pages) {
                protected_pages.insert(page);
            }
            AllocationPlan plan;
            plan.request = &request;
            plan.hit = std::move(hit);
            plan.private_pages_required = private_pages_required;
            plans.emplace_back(std::move(plan));
        }

        if (backend_.FreePageCount() < total_private_pages_required) {
            PrefixEvictionResult eviction = EvictPrefixCacheUntilFree(
                total_private_pages_required, protected_pages);
            if (!eviction.reached_target &&
                backend_.FreePageCount() < total_private_pages_required) {
                std::ostringstream oss;
                oss << "AllocatePagesForSequencesWithPrefix: insufficient "
                       "free pages after prefix cache eviction "
                    << "(required=" << total_private_pages_required
                    << ", available=" << backend_.FreePageCount()
                    << ", evicted_entries=" << eviction.entries_removed
                    << ", protected_skips="
                    << eviction.protected_entries_skipped << ")";
                throw std::runtime_error(oss.str());
            }
        }

        std::vector<std::size_t> allocated_plan_indices;
        std::vector<std::size_t> attached_plan_indices;
        try {
            for (std::size_t i = 0; i < plans.size(); ++i) {
                AllocationPlan& plan = plans[i];
                if (plan.private_pages_required == 0) {
                    continue;
                }
                plan.private_pages = backend_.AcquirePages(
                    plan.request->sequence_id, plan.private_pages_required);
                allocated_plan_indices.push_back(i);
            }
            for (std::size_t i = 0; i < plans.size(); ++i) {
                AllocationPlan& plan = plans[i];
                if (plan.hit.host_pages.empty()) {
                    continue;
                }
                backend_.AttachSequencePages(plan.hit.host_pages);
                plan.attached_shared_pages = true;
                attached_plan_indices.push_back(i);
            }
            for (const AllocationPlan& plan : plans) {
                page_table_.RegisterOrUpdate(
                    plan.request->sequence_id, plan.hit.host_pages,
                    plan.private_pages,
                    static_cast<std::int64_t>(plan.hit.matched_tokens),
                    static_cast<std::int64_t>(plan.hit.matched_tokens),
                    static_cast<std::int64_t>(plan.request->capacity_tokens));
                if (!plan.hit.host_pages.empty()) {
                    prefix_cache_.RecordAttachedPages(
                        plan.hit.host_pages.size());
                }
            }
        } catch (...) {
            for (auto it = attached_plan_indices.rbegin();
                 it != attached_plan_indices.rend(); ++it) {
                const AllocationPlan& plan = plans[*it];
                try {
                    backend_.DetachSequencePages(plan.hit.host_pages);
                } catch (const std::exception& ex) {
                    logger_->error(
                        "Failed to roll back attached prefix pages for "
                        "sequence {}: {}",
                        plan.request->sequence_id, ex.what());
                }
            }
            for (auto it = allocated_plan_indices.rbegin();
                 it != allocated_plan_indices.rend(); ++it) {
                const AllocationPlan& plan = plans[*it];
                try {
                    backend_.ReleaseSequence(plan.request->sequence_id);
                } catch (const std::exception& ex) {
                    logger_->error(
                        "Failed to roll back private pages for sequence {}: {}",
                        plan.request->sequence_id, ex.what());
                }
            }
            for (const auto& request : requests) {
                try {
                    page_table_.RegisterOrUpdate(
                        request.sequence_id, std::vector<std::int32_t>{});
                } catch (const std::exception& ex) {
                    logger_->error(
                        "Failed to reset page table for sequence {} after "
                        "allocation rollback: {}",
                        request.sequence_id, ex.what());
                }
            }
            throw;
        }

        std::vector<PrefixAllocationResult> results;
        results.reserve(plans.size());
        for (const AllocationPlan& plan : plans) {
            PrefixAllocationResult result;
            result.sequence_id = plan.request->sequence_id;
            result.shared_prefix_pages = plan.hit.host_pages;
            result.private_pages = plan.private_pages;
            result.shared_prefix_tokens = plan.hit.matched_tokens;
            result.private_start_token = plan.hit.matched_tokens;
            result.logical_page_count =
                plan.hit.host_pages.size() + plan.private_pages.size();
            result.physical_pages_allocated = plan.private_pages.size();
            result.full_hit = plan.hit.full_hit;
            result.miss_reason = plan.hit.miss_reason;
            results.emplace_back(std::move(result));
        }
        return results;
    }

    std::vector<PrefixAllocationResult> EstimatePagesForSequencesWithPrefix(
        const std::vector<PrefixAllocationRequest>& requests) {
        std::vector<PrefixAllocationResult> results;
        results.reserve(requests.size());
        for (const auto& request : requests) {
            if (request.token_ids.empty()) {
                throw std::invalid_argument(
                    "Prefix allocation estimate requires at least one prompt "
                    "token");
            }
            if (request.capacity_tokens < request.token_ids.size()) {
                throw std::invalid_argument(
                    "capacity_tokens must be >= token_ids.size()");
            }

            const PrefixLookupResult hit = prefix_cache_.Peek(
                request.namespace_hash,
                static_cast<std::int32_t>(config_.page_size_tokens),
                request.token_ids);
            const std::size_t private_tokens =
                request.capacity_tokens - hit.matched_tokens;
            const std::size_t private_pages_required =
                private_tokens == 0 ? 0 : geometry_.RequiredPages(private_tokens);

            PrefixAllocationResult result;
            result.sequence_id = request.sequence_id;
            result.shared_prefix_pages = hit.host_pages;
            result.shared_prefix_tokens = hit.matched_tokens;
            result.private_start_token = hit.matched_tokens;
            result.logical_page_count =
                hit.host_pages.size() + private_pages_required;
            result.physical_pages_allocated = private_pages_required;
            result.full_hit = hit.full_hit;
            result.miss_reason = hit.miss_reason;
            results.emplace_back(std::move(result));
        }
        return results;
    }

    std::size_t CommitSequencePrefixPages(
        std::int64_t sequence_id,
        const std::vector<std::int64_t>& token_ids,
        std::uint64_t namespace_hash = 0) {
        EnsureSequenceRegistered(sequence_id);
        const auto logical_pages = page_table_.Pages(sequence_id);
        return prefix_cache_.CommitPages(
            namespace_hash, static_cast<std::int32_t>(config_.page_size_tokens),
            token_ids, logical_pages,
            [this](std::int32_t page) { backend_.PinPrefixPage(page); });
    }

    PrefixCacheStats GetPrefixCacheStats() const {
        return prefix_cache_.Stats();
    }

    std::vector<PrefixDebugEntry> PrefixCacheDebugEntries(
        std::size_t limit = 0, bool cold_first = true) const {
        return prefix_cache_.DebugEntries(limit, cold_first);
    }

    void ClearPrefixCache() {
        prefix_cache_.Clear(
            [this](std::int32_t page) { backend_.UnpinPrefixPage(page); });
    }

    PrefixEvictionResult EvictPrefixCacheUntilFree(
        std::size_t target_free_pages,
        const std::unordered_set<std::int32_t>& protected_pages = {}) {
        PrefixEvictionOptions options;
        options.target_free_pages = target_free_pages;
        options.protected_pages = protected_pages;
        PrefixEvictionResult result = prefix_cache_.EvictLeafPages(
            options, [this](std::int32_t page) {
                backend_.UnpinPrefixPage(page);
            },
            [this]() { return backend_.FreePageCount(); });
        if (result.entries_removed > 0 || !result.reached_target) {
            logger_->info(
                "[PREFIX_EVICT] target_free={} entries_removed={} "
                "pins_released={} immediate_free={} active_ref_removed={} "
                "protected_skips={} reached_target={} free_pages={}",
                target_free_pages, result.entries_removed,
                result.prefix_pins_released, result.pages_immediately_freed,
                result.active_ref_entries_removed,
                result.protected_entries_skipped, result.reached_target,
                backend_.FreePageCount());
        }
        if (result.active_ref_entries_removed > 0) {
            logger_->debug(
                "[PREFIX_EVICT] {} evicted prefix entries remain held by "
                "active sequence refs",
                result.active_ref_entries_removed);
        }
        return result;
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
        ClearPrefixCache();
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

    KVAsyncTask AsyncLoadPrefixPagesToDevice(
        torch::Tensor host_page_ids, torch::Tensor k_device_ptrs,
        std::optional<torch::Tensor> v_device_ptrs = std::nullopt) {
        EnsureDeviceReady();
        constexpr std::string_view kOpName =
            "AsyncLoadPrefixPagesToDevice";

        auto validated_pages =
            ValidatePageIdTensor(std::move(host_page_ids), "host_page_ids",
                                 kOpName);
        const auto total_pages =
            static_cast<std::size_t>(validated_pages.size(0));

        auto validated_k_ptrs = ValidatePointerMatrix(
            std::move(k_device_ptrs), "k_device_ptrs", kOpName);
        if (static_cast<std::size_t>(validated_k_ptrs.size(1)) !=
            total_pages) {
            std::ostringstream oss;
            oss << kOpName << ": k_device_ptrs second dimension must match "
                << "host_page_ids length (" << validated_k_ptrs.size(1)
                << " != " << total_pages << ")";
            throw std::out_of_range(oss.str());
        }

        std::optional<torch::Tensor> validated_v_ptrs;
        if (v_device_ptrs.has_value()) {
            if constexpr (!kHasVCache) {
                throw std::invalid_argument(std::string(kOpName) +
                                            ": V cache is disabled");
            }
            auto tensor = ValidatePointerMatrix(
                std::move(*v_device_ptrs), "v_device_ptrs", kOpName);
            if (tensor.sizes() != validated_k_ptrs.sizes()) {
                std::ostringstream oss;
                oss << kOpName
                    << ": v_device_ptrs must match k_device_ptrs shape";
                throw std::invalid_argument(oss.str());
            }
            validated_v_ptrs = std::move(tensor);
        }

        if (total_pages == 0) {
            return LaunchAsyncTask([] {});
        }

        const std::size_t num_layers = config_.num_layers;
        const std::size_t copy_entries = num_layers * total_pages;
        const auto kernel_limit =
            static_cast<std::size_t>(std::numeric_limits<int>::max());
        if (copy_entries > kernel_limit) {
            std::ostringstream oss;
            oss << kOpName << ": copy_entries=" << copy_entries
                << " exceeds kernel limit=" << kernel_limit;
            throw std::invalid_argument(oss.str());
        }

        auto page_vector = TensorToInt32Vector(validated_pages, "host_page_ids",
                                               kOpName);
        std::vector<std::vector<std::int32_t>> page_table(1);
        page_table[0] = std::move(page_vector);
        std::vector<std::size_t> sequence_offsets{0};

        logger_->debug(
            "Prepared AsyncLoadPrefixPagesToDevice (num_layers={}, total_pages={})",
            num_layers, total_pages);

        return LaunchLayeredAsyncTask(num_layers, [
            this,
            page_table = std::move(page_table),
            sequence_offsets = std::move(sequence_offsets),
            k_tensor = std::move(validated_k_ptrs),
            v_tensor = std::move(validated_v_ptrs),
            total_pages,
            num_layers,
            copy_entries,
            kOpName
        ](KVLayerCompletionState& layer_state) mutable {
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
            auto build_layer_plan = [&](std::size_t layer_idx,
                                        const std::int64_t* dest_ptrs,
                                        auto&& host_ptr_provider) {
                if (dest_ptrs == nullptr) {
                    throw std::invalid_argument(std::string(kOpName) +
                                                ": null device pointers");
                }
                return this->BuildPageCopyPlan(
                    page_table, sequence_offsets, 1, row_stride, total_pages,
                    dest_ptrs + layer_idx * row_stride,
                    std::forward<decltype(host_ptr_provider)>(
                        host_ptr_provider),
                    kOpName);
            };

            worker_detail::DeviceBuffer<uint8_t*> k_device_src_ptrs(
                copy_entries);
            worker_detail::DeviceBuffer<uint8_t*> k_device_dst_ptrs(
                copy_entries);
            worker_detail::DeviceBuffer<uint8_t*> v_device_src_ptrs(
                v_dest_ptr != nullptr ? copy_entries : 0);
            worker_detail::DeviceBuffer<uint8_t*> v_device_dst_ptrs(
                v_dest_ptr != nullptr ? copy_entries : 0);

            auto enqueue_plan =
                [&](const PageCopyPlan& plan,
                    worker_detail::DeviceBuffer<uint8_t*>& dev_src_ptrs,
                    worker_detail::DeviceBuffer<uint8_t*>& dev_dst_ptrs,
                    std::size_t page_bytes, std::size_t layer_idx) {
                    if (plan.host_sources.empty() || page_bytes == 0) {
                        return;
                    }
                    const std::size_t device_offset = layer_idx * total_pages;
                    const std::size_t ptr_bytes =
                        plan.host_sources.size() * sizeof(uint8_t*);
                    EnqueueCopy(
                        reinterpret_cast<const std::byte*>(
                            plan.host_sources.data()),
                        reinterpret_cast<std::byte*>(
                            dev_src_ptrs.get() + device_offset),
                        ptr_bytes, CopyDirection::kHostToDevice, cuda_stream);
                    EnqueueCopy(
                        reinterpret_cast<const std::byte*>(
                            plan.device_dests.data()),
                        reinterpret_cast<std::byte*>(
                            dev_dst_ptrs.get() + device_offset),
                        ptr_bytes, CopyDirection::kHostToDevice, cuda_stream);
                    worker_detail::LaunchUvaPageCopyKernel(
                        dev_src_ptrs.get() + device_offset,
                        dev_dst_ptrs.get() + device_offset, page_bytes,
                        static_cast<int>(plan.host_sources.size()),
                        cuda_stream);
                };

            std::vector<PageCopyPlan> k_layer_plans;
            k_layer_plans.reserve(num_layers);
            std::vector<PageCopyPlan> v_layer_plans;
            if constexpr (kHasVCache) {
                if (v_dest_ptr != nullptr) {
                    v_layer_plans.reserve(num_layers);
                }
            }
            for (std::size_t layer_idx = 0; layer_idx < num_layers;
                 ++layer_idx) {
                k_layer_plans.emplace_back(build_layer_plan(
                    layer_idx, k_dest_ptr,
                    [this, layer_idx](
                        std::size_t /*unused*/, std::int32_t page_idx
                    ) -> void* {
                        return this->KPagePtr(layer_idx, page_idx);
                    }));
                enqueue_plan(k_layer_plans.back(), k_device_src_ptrs,
                             k_device_dst_ptrs, k_page_bytes, layer_idx);

                if constexpr (kHasVCache) {
                    if (v_dest_ptr != nullptr) {
                        v_layer_plans.emplace_back(build_layer_plan(
                            layer_idx, v_dest_ptr,
                            [this, layer_idx](
                                std::size_t /*unused*/,
                                std::int32_t page_idx
                            ) -> void* {
                                return this->template VPagePtr<>(layer_idx,
                                                                 page_idx);
                            }));
                        const std::size_t v_page_bytes = layout_.VPageBytes();
                        enqueue_plan(v_layer_plans.back(), v_device_src_ptrs,
                                     v_device_dst_ptrs, v_page_bytes,
                                     layer_idx);
                    }
                }

                cudaEvent_t layer_event = nullptr;
                CUDA_CHECK(cudaEventCreateWithFlags(
                    &layer_event, cudaEventDisableTiming));
                CUDA_CHECK(cudaEventRecord(layer_event, cuda_stream));
                layer_state.RecordLayerEvent(layer_idx, layer_event);
            }

            logger_->debug(
                "AsyncLoadPrefixPagesToDevice completed (num_layers={}, total_pages={}, k_page_bytes={})",
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
    std::size_t FreePageCount() const { return backend_.FreePageCount(); }
    HostPageRefState PageRefState(std::int32_t page) const {
        return backend_.PageRefState(page);
    }
    std::vector<HostPageRefState> PageRefStates(
        const std::vector<std::int32_t>& pages) const {
        return backend_.PageRefStates(pages);
    }
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

    std::vector<std::int32_t> SharedPrefixPages(
        std::int64_t sequence_id) const {
        return page_table_.SharedPrefixPages(sequence_id);
    }

    std::vector<std::int32_t> PrivatePages(std::int64_t sequence_id) const {
        return page_table_.PrivatePages(sequence_id);
    }

    std::int64_t SharedPrefixTokens(std::int64_t sequence_id) const {
        return page_table_.SharedPrefixTokens(sequence_id);
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
        const bool has_shared_prefix =
            page_table_.Contains(sequence_id) &&
            !page_table_.SharedPrefixPages(sequence_id).empty();
        auto page_indices =
            has_shared_prefix
                ? page_table_.Pages(sequence_id)
                : backend_.SequencePages(sequence_id, max_pages);
        if (max_pages.has_value()) {
            if (max_pages.value() > page_indices.size()) {
                throw std::out_of_range(
                    "HostPagedKVWorkerView::GetSequenceLayerPagePointers: "
                    "requested more pages than allocated");
            }
            page_indices.resize(max_pages.value());
        }
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
        std::vector<std::int64_t> registered_sequence_ids;
        registered_sequence_ids.reserve(sequence_ids.size());
        for (std::int64_t sequence_id : sequence_ids) {
            if (page_table_.Contains(sequence_id)) {
                registered_sequence_ids.push_back(sequence_id);
            }
        }
        if (registered_sequence_ids.empty()) {
            return;
        }
        const bool has_any_shared_prefix =
            std::any_of(registered_sequence_ids.begin(),
                        registered_sequence_ids.end(),
                        [this](std::int64_t sequence_id) {
                            return !page_table_.SharedPrefixPages(sequence_id)
                                        .empty();
                        });
        if (!has_any_shared_prefix) {
            backend_.ReleaseSequences(registered_sequence_ids);
        } else {
            for (std::int64_t sequence_id : registered_sequence_ids) {
                backend_.ReleaseSequenceLogical(sequence_id,
                                                page_table_.Pages(sequence_id));
            }
        }
        UnregisterSequences(registered_sequence_ids);
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

    KVAsyncTask AsyncOffloadLayerKVToHostWithOffsets(
        std::size_t layer_idx, std::vector<std::int64_t> sequence_ids,
        torch::Tensor k_tensor, std::optional<torch::Tensor> v_tensor,
        SequenceLengths sequence_lengths,
        std::vector<std::size_t> source_token_starts,
        std::vector<std::size_t> destination_token_starts) {
        constexpr std::string_view kOpName =
            "AsyncOffloadLayerKVToHostWithOffsets";
        geometry_.EnsureLayerBounds(layer_idx, kOpName);
        EnsureDeviceReady();
        const std::size_t batch = sequence_ids.size();
        if (source_token_starts.size() != batch ||
            destination_token_starts.size() != batch) {
            std::ostringstream oss;
            oss << kOpName
                << ": source_token_starts and destination_token_starts must "
                   "match sequence_ids size ("
                << batch << ")";
            throw std::invalid_argument(oss.str());
        }
        if (batch == 0) {
            return LaunchAsyncTask([] {});
        }

        const std::size_t tokens_per_sequence =
            ValidateKTensorShape(k_tensor, batch);
        ValidateSequenceLengthsInput(sequence_lengths, batch, kOpName);
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
        const std::string op_name(kOpName);

        return LaunchAsyncTask([this, layer_idx,
                                 sequence_ids = std::move(sequence_ids),
                                 sequence_lengths = std::move(sequence_lengths),
                                 source_token_starts =
                                     std::move(source_token_starts),
                                 destination_token_starts =
                                     std::move(destination_token_starts),
                                 op_name,
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
                    tokens_per_sequence, op_name);
                if (tokens_to_copy == 0) {
                    continue;
                }

                const std::size_t source_token_start =
                    source_token_starts[batch_idx];
                const std::size_t destination_token_start =
                    destination_token_starts[batch_idx];
                if (source_token_start + tokens_to_copy >
                    tokens_per_sequence) {
                    std::ostringstream oss;
                    oss << op_name << ": source range exceeds tensor capacity "
                        << "for sequence " << sequence_id
                        << " (source_start=" << source_token_start
                        << ", tokens=" << tokens_to_copy
                        << ", tensor_tokens=" << tokens_per_sequence << ")";
                    throw std::out_of_range(oss.str());
                }

                const std::size_t shared_prefix_tokens = static_cast<std::size_t>(
                    page_table_.SharedPrefixTokens(sequence_id));
                if (destination_token_start < shared_prefix_tokens) {
                    std::ostringstream oss;
                    oss << op_name << ": refusing to write into shared prefix "
                        << "for sequence " << sequence_id
                        << " (destination_start=" << destination_token_start
                        << ", shared_prefix_tokens="
                        << shared_prefix_tokens << ")";
                    throw std::invalid_argument(oss.str());
                }

                geometry_.ValidatePageCapacity(
                    pages, destination_token_start + tokens_to_copy, op_name);

                const auto* seq_k_src = k_base + batch_idx * k_seq_stride;
                ForEachPageChunk(
                    pages, destination_token_start, tokens_to_copy,
                    [&](std::int32_t page_idx, std::size_t page_offset_tokens,
                        std::size_t chunk_tokens,
                        std::size_t relative_token_offset) {
                        std::byte* dst = layout_.KPageAddress(
                                             host_base, layer_idx, page_idx) +
                                         page_offset_tokens * k_token_bytes;
                        const std::byte* src =
                            seq_k_src +
                            (source_token_start + relative_token_offset) *
                                k_token_bytes;
                        EnqueueCopy(src, dst, chunk_tokens * k_token_bytes,
                                    CopyDirection::kDeviceToHost, cuda_stream);
                    });

                if constexpr (kHasVCache) {
                    if (v_base != nullptr) {
                        const auto* seq_v_src =
                            v_base + batch_idx * v_seq_stride;
                        ForEachPageChunk(
                            pages, destination_token_start, tokens_to_copy,
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
                                    (source_token_start +
                                     relative_token_offset) *
                                        v_token_bytes;
                                EnqueueCopy(
                                    src, dst, chunk_tokens * v_token_bytes,
                                    CopyDirection::kDeviceToHost, cuda_stream);
                            });
                    }
                }
            }

            this->SynchronizeWithEvent(cuda_stream);
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
                const std::size_t shared_prefix_tokens = static_cast<std::size_t>(
                    page_table_.SharedPrefixTokens(sequence_id));
                if (start_token < shared_prefix_tokens) {
                    throw std::runtime_error(
                        "AsyncAppendDecodeKVToHost: refusing to write into "
                        "shared prefix pages");
                }
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
    // Batched decode KV offload — one UVA kernel launch for all
    // (layer × seq) token writes instead of 1248 cudaMemcpyAsync.
    // ================================================================

    struct BatchedKVEntry {
        std::size_t layer_idx;
        torch::Tensor k_tensor;
        std::optional<torch::Tensor> v_tensor;
    };

    /**
     * Collapse all per-(layer × seq) host-KV appends for this decode step
     * into a single kernel launch per cache. Reuses
     * `worker_detail::LaunchUvaPageCopyKernel` (direction-agnostic): src
     * ptrs are GPU-resident K/V token slots, dst ptrs are UVA-mapped host
     * pinned pages. GPU issues PCIe writes directly — no cudaMemcpyAsync
     * per copy pair. Replaces 78×batch_size cudaMemcpyAsync launches +
     * per-layer thread-pool task + per-task producer-stream wait with:
     *   - 2 small HtoD of ptr arrays (~KiB each)
     *   - 1 UvaPageCopyKernel launch
     *   - 1 event record
     */
    KVAsyncTask AsyncAppendDecodeKVToHostBatchedKernel(
        std::vector<BatchedKVEntry> entries,
        std::vector<std::int64_t> sequence_ids,
        SequenceLengths sequence_lengths) {
        if (entries.empty() || sequence_ids.empty()) {
            return LaunchAsyncTask([] {});
        }
        EnsureDeviceReady();
        const std::size_t batch = sequence_ids.size();

        // Validate each entry and detect V presence
        bool has_v = false;
        for (const auto& e : entries) {
            geometry_.EnsureLayerBounds(e.layer_idx,
                                        "AsyncAppendDecodeKVToHostBatchedKernel");
            const std::size_t tokens_per_seq =
                ValidateKTensorShape(e.k_tensor, batch);
            if (tokens_per_seq != 1) {
                throw std::invalid_argument(
                    "AsyncAppendDecodeKVToHostBatchedKernel expects 1 token "
                    "per sequence");
            }
            if (e.v_tensor.has_value()) {
                if constexpr (kHasVCache) {
                    ValidateVTensorShape(*e.v_tensor, batch, tokens_per_seq);
                    has_v = true;
                } else {
                    throw std::invalid_argument(
                        "V tensor provided but V cache is disabled");
                }
            }
        }
        ValidateSequenceLengthsInput(
            sequence_lengths, batch,
            "AsyncAppendDecodeKVToHostBatchedKernel");

        c10::cuda::OptionalCUDAGuard producer_guard(device_index_);
        const auto producer_cuda_stream =
            at::cuda::getCurrentCUDAStream(device_index_).stream();

        // Resolve per-seq page locations ONCE (shared across all layers).
        struct SeqLoc {
            std::size_t page_idx;
            std::size_t page_offset_tokens;
        };
        std::vector<SeqLoc> seq_locs;
        seq_locs.reserve(batch);
        for (std::size_t b = 0; b < batch; ++b) {
            const std::int64_t sid = sequence_ids[b];
            const std::size_t start_token = ResolveSequenceLength(
                sequence_lengths, b, sid, std::nullopt,
                "AsyncAppendDecodeKVToHostBatchedKernel");
            const std::size_t shared_prefix_tokens = static_cast<std::size_t>(
                page_table_.SharedPrefixTokens(sid));
            if (start_token < shared_prefix_tokens) {
                throw std::runtime_error(
                    "AsyncAppendDecodeKVToHostBatchedKernel: refusing to write "
                    "into shared prefix pages");
            }
            const auto pages = page_table_.Pages(sid);
            geometry_.ValidatePageCapacity(
                pages, start_token + 1,
                "AsyncAppendDecodeKVToHostBatchedKernel");
            const auto loc = ResolvePageLocation(
                pages, sid, start_token,
                "AsyncAppendDecodeKVToHostBatchedKernel");
            seq_locs.push_back({loc.page_idx, loc.page_offset_tokens});
        }

        const std::size_t k_token_bytes = geometry_.KTokenBytes();
        std::size_t v_token_bytes = 0;
        if constexpr (kHasVCache) {
            if (has_v) {
                v_token_bytes = geometry_.template VTokenBytes<kHasVCache>();
            }
        }

        const std::size_t k_total = entries.size() * batch;
        const std::size_t v_total = has_v ? k_total : 0;

        // Build src/dst ptr arrays on host (pinned unnecessary — staged
        // HtoD is one-shot per cache per step, bandwidth insignificant).
        std::vector<uint8_t*> k_src_host(k_total), k_dst_host(k_total);
        std::vector<uint8_t*> v_src_host(v_total), v_dst_host(v_total);

        std::byte* host_base = backend_.DataBase();
        for (std::size_t li = 0; li < entries.size(); ++li) {
            const auto& e = entries[li];
            auto* k_base = static_cast<uint8_t*>(e.k_tensor.data_ptr());
            uint8_t* v_base = nullptr;
            if constexpr (kHasVCache) {
                if (e.v_tensor.has_value()) {
                    v_base =
                        static_cast<uint8_t*>(e.v_tensor->data_ptr());
                }
            }
            for (std::size_t b = 0; b < batch; ++b) {
                const auto& loc = seq_locs[b];
                const std::size_t idx = li * batch + b;
                k_src_host[idx] = k_base + b * k_token_bytes;
                k_dst_host[idx] = reinterpret_cast<uint8_t*>(
                    layout_.KPageAddress(host_base, e.layer_idx, loc.page_idx)
                    + loc.page_offset_tokens * k_token_bytes);
                if constexpr (kHasVCache) {
                    if (v_base != nullptr) {
                        v_src_host[idx] = v_base + b * v_token_bytes;
                        v_dst_host[idx] = reinterpret_cast<uint8_t*>(
                            layout_.template VPageAddress<>(
                                host_base, e.layer_idx, loc.page_idx)
                            + loc.page_offset_tokens * v_token_bytes);
                    }
                }
            }
        }

        return LaunchAsyncTask([this, k_src_host = std::move(k_src_host),
                                 k_dst_host = std::move(k_dst_host),
                                 v_src_host = std::move(v_src_host),
                                 v_dst_host = std::move(v_dst_host),
                                 k_token_bytes, v_token_bytes, k_total,
                                 v_total, producer_cuda_stream]() mutable {
            c10::cuda::OptionalCUDAGuard device_guard(device_index_);
            const auto cuda_stream = CopyStream(CopyDirection::kDeviceToHost);
            this->WaitForProducerStream(cuda_stream, producer_cuda_stream);

            worker_detail::DeviceBuffer<uint8_t*> k_src_buf(k_total);
            worker_detail::DeviceBuffer<uint8_t*> k_dst_buf(k_total);

            const std::size_t ptr_bytes_k = k_total * sizeof(uint8_t*);
            EnqueueCopy(
                reinterpret_cast<const std::byte*>(k_src_host.data()),
                reinterpret_cast<std::byte*>(k_src_buf.get()),
                ptr_bytes_k, CopyDirection::kHostToDevice, cuda_stream);
            EnqueueCopy(
                reinterpret_cast<const std::byte*>(k_dst_host.data()),
                reinterpret_cast<std::byte*>(k_dst_buf.get()),
                ptr_bytes_k, CopyDirection::kHostToDevice, cuda_stream);

            worker_detail::LaunchUvaPageCopyKernel(
                k_src_buf.get(), k_dst_buf.get(),
                k_token_bytes, static_cast<int>(k_total), cuda_stream);

            if constexpr (kHasVCache) {
                if (v_total > 0) {
                    worker_detail::DeviceBuffer<uint8_t*> v_src_buf(v_total);
                    worker_detail::DeviceBuffer<uint8_t*> v_dst_buf(v_total);
                    const std::size_t ptr_bytes_v = v_total * sizeof(uint8_t*);
                    EnqueueCopy(
                        reinterpret_cast<const std::byte*>(v_src_host.data()),
                        reinterpret_cast<std::byte*>(
                            v_src_buf.get()),
                        ptr_bytes_v, CopyDirection::kHostToDevice, cuda_stream);
                    EnqueueCopy(
                        reinterpret_cast<const std::byte*>(v_dst_host.data()),
                        reinterpret_cast<std::byte*>(
                            v_dst_buf.get()),
                        ptr_bytes_v, CopyDirection::kHostToDevice, cuda_stream);
                    worker_detail::LaunchUvaPageCopyKernel(
                        v_src_buf.get(),
                        v_dst_buf.get(), v_token_bytes,
                        static_cast<int>(v_total), cuda_stream);
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

    struct CopyPointerPair {
        std::uintptr_t host = 0;
        std::uintptr_t device = 0;

        bool operator==(const CopyPointerPair& other) const {
            return host == other.host && device == other.device;
        }
    };

    struct CopyPointerPairHash {
        std::size_t operator()(const CopyPointerPair& pair) const {
            return static_cast<std::size_t>(HashCombine(pair.host, pair.device));
        }
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
        plan.host_sources.reserve(total_entries);
        plan.device_dests.reserve(total_entries);
        std::unordered_set<CopyPointerPair, CopyPointerPairHash> seen_copies;
        seen_copies.reserve(total_entries);
        auto&& provider = std::forward<HostPtrProvider>(host_ptr_provider);
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
                    const auto host_key =
                        reinterpret_cast<std::uintptr_t>(host_ptr);
                    const auto device_key =
                        reinterpret_cast<std::uintptr_t>(device_ptr);
                    const CopyPointerPair copy_key{host_key, device_key};
                    if (seen_copies.insert(copy_key).second) {
                        plan.host_sources.emplace_back(host_ptr);
                        plan.device_dests.emplace_back(device_ptr);
                    }
                }
            }
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

    torch::Tensor ValidatePageIdTensor(torch::Tensor tensor,
                                       std::string_view tensor_name,
                                       std::string_view op_name) const {
        if (tensor.device().type() != torch::kCPU) {
            std::ostringstream oss;
            oss << op_name << ": " << tensor_name
                << " must reside on CPU";
            throw std::invalid_argument(oss.str());
        }
        if (!tensor.is_contiguous()) {
            std::ostringstream oss;
            oss << op_name << ": " << tensor_name
                << " must be contiguous";
            throw std::invalid_argument(oss.str());
        }
        if (tensor.dim() != 1) {
            std::ostringstream oss;
            oss << op_name << ": " << tensor_name
                << " must be 1-D";
            throw std::invalid_argument(oss.str());
        }
        if (tensor.scalar_type() != torch::kInt64 &&
            tensor.scalar_type() != torch::kInt32) {
            std::ostringstream oss;
            oss << op_name << ": " << tensor_name
                << " must have dtype int32 or int64";
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

    std::vector<std::int32_t> TensorToInt32Vector(
        const torch::Tensor& tensor, std::string_view tensor_name,
        std::string_view op_name) const {
        const auto length = static_cast<std::size_t>(tensor.size(0));
        std::vector<std::int32_t> values(length);
        if (length == 0) {
            return values;
        }
        if (tensor.scalar_type() == torch::kInt64) {
            const auto* data = tensor.data_ptr<std::int64_t>();
            for (std::size_t idx = 0; idx < length; ++idx) {
                const auto value = data[idx];
                if (value < 0 ||
                    value > std::numeric_limits<std::int32_t>::max()) {
                    std::ostringstream oss;
                    oss << op_name << ": " << tensor_name
                        << " value out of int32 page range (index=" << idx
                        << ", value=" << value << ")";
                    throw std::out_of_range(oss.str());
                }
                values[idx] = static_cast<std::int32_t>(value);
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
                    << ", value=" << value << ")";
                throw std::out_of_range(oss.str());
            }
            values[idx] = value;
        }
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
        auto future = worker_detail::BoundedAsyncExecutor::Instance().Submit(
            std::forward<Fn>(fn));
        const std::uint64_t id =
            task_id_counter_.fetch_add(1, std::memory_order_relaxed) + 1;
        return KVAsyncTask{id, std::move(future)};
    }

    template <typename Fn>
    KVAsyncTask LaunchLayeredAsyncTask(std::size_t num_layers, Fn&& fn) const {
        auto layer_state = std::make_shared<KVLayerCompletionState>(num_layers);
        auto task = [
            layer_state,
            fn = std::forward<Fn>(fn)
        ]() mutable {
            try {
                fn(*layer_state);
            } catch (...) {
                layer_state->SetFailure(std::current_exception());
                throw;
            }
        };
        auto future = worker_detail::BoundedAsyncExecutor::Instance().Submit(
            std::move(task));
        const std::uint64_t id =
            task_id_counter_.fetch_add(1, std::memory_order_relaxed) + 1;
        return KVAsyncTask{id, std::move(future), std::move(layer_state)};
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
    HostPrefixCache prefix_cache_;

    // Scratch device buffers for AsyncAppendDecodeKVToHostBatchedKernel —
    // pointer arrays (src + dst) uploaded once per batched call. Sized
    // lazily (grow-only) on first use; typical steady-state usage is
    // num_layers × max_batch_size entries per cache (~few KiB).
    mutable worker_detail::DeviceBuffer<uint8_t*> uva_append_k_src_buf_;
    mutable worker_detail::DeviceBuffer<uint8_t*> uva_append_k_dst_buf_;
    mutable worker_detail::DeviceBuffer<uint8_t*> uva_append_v_src_buf_;
    mutable worker_detail::DeviceBuffer<uint8_t*> uva_append_v_dst_buf_;

    inline static std::atomic<std::uint64_t> task_id_counter_{0};
};

using DefaultHostPagedKVWorkerView = HostPagedKVWorkerView<HostKVMode::kMHA>;
using MLAHostPagedKVWorkerView = HostPagedKVWorkerView<HostKVMode::kMLA>;

}  // namespace batchgen::kv

#endif  // HOST_PAGED_KV_WORKER_VIEW_H_
