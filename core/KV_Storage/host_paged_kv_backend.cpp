#include "host_paged_kv_backend.h"

#include <fcntl.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <exception>
#include <future>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

namespace batchgen::kv {

namespace {

constexpr std::uint64_t kSharedMemoryMagic =
    0x484f53544b564d47ULL;  // "HOSTKVMG"
constexpr std::int32_t kInvalidPageIndex = -1;
constexpr std::int64_t kEmptySequenceId =
    std::numeric_limits<std::int64_t>::min();
constexpr std::int64_t kTombstoneSequenceId = kEmptySequenceId + 1;

enum class InitState : std::uint32_t {
    kUninitialized = 0,
    kInitializing = 1,
    kReady = 2,
};

std::size_t AlignUp(std::size_t value, std::size_t alignment) {
    if (alignment == 0) {
        return value;
    }
    const std::size_t remainder = value % alignment;
    if (remainder == 0) {
        return value;
    }
    return value + (alignment - remainder);
}

std::size_t GetSystemPageSize() {
    const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) {
        const int err = errno;
        throw std::system_error(err, std::generic_category(),
                                "sysconf(_SC_PAGESIZE) failed");
    }
    return static_cast<std::size_t>(page_size);
}

class ScopedMutexLock {
   public:
    explicit ScopedMutexLock(pthread_mutex_t* mu) : mu_(mu) {
        int rc = pthread_mutex_lock(mu_);
        if (rc == EOWNERDEAD) {
            const int consistent_rc = pthread_mutex_consistent(mu_);
            if (consistent_rc != 0) {
                throw std::system_error(consistent_rc, std::generic_category(),
                                        "pthread_mutex_consistent failed");
            }
        } else if (rc != 0) {
            throw std::system_error(rc, std::generic_category(),
                                    "pthread_mutex_lock failed");
        }
    }

    ScopedMutexLock(const ScopedMutexLock&) = delete;
    ScopedMutexLock& operator=(const ScopedMutexLock&) = delete;

    ~ScopedMutexLock() {
        const int rc = pthread_mutex_unlock(mu_);
        if (rc != 0) {
            std::terminate();  // Unlock failure is irrecoverable here.
        }
    }

   private:
    pthread_mutex_t* mu_;
};

struct SequenceEntry {
    std::int64_t sequence_id = kEmptySequenceId;
    std::uint32_t num_pages = 0;
    std::int32_t head_page = kInvalidPageIndex;
    std::int32_t tail_page = kInvalidPageIndex;
};

struct SharedHeader {
    std::atomic<std::uint32_t> init_state{
        static_cast<std::uint32_t>(InitState::kUninitialized)};
    std::uint64_t magic = kSharedMemoryMagic;
    std::uint64_t layout_fingerprint = 0;
    std::uint64_t config_hash = 0;
    std::uint64_t data_bytes = 0;
    std::uint64_t num_pages = 0;
    std::uint64_t sequence_capacity = 0;
    std::uint64_t alignment_bytes = 0;
    std::uint64_t num_layers = 0;
    std::uint64_t page_size_tokens = 0;
    std::uint32_t has_v_cache = 0;
    std::atomic<std::uint32_t> free_stack_top{0};
    std::atomic<std::uint32_t> active_sequences{0};
    pthread_mutex_t allocation_mutex{};
    pthread_mutex_t sequence_mutex{};
};

std::size_t SafeHardwareConcurrency() {
    const unsigned int hint = std::thread::hardware_concurrency() / 4;
    return hint == 0 ? 1 : static_cast<std::size_t>(hint);
}

}  // namespace

struct HostPagedKVBackend::SharedState {
    explicit SharedState(const HostPagedKVConfig& cfg, std::size_t data_bytes,
                         std::uint64_t fingerprint, bool has_v)
        : config(cfg),
          data_bytes(data_bytes),
          layout_fingerprint(fingerprint),
          has_v_cache(has_v) {
        sequence_capacity = config.sequence_table_capacity == 0
                                ? config.num_pages
                                : config.sequence_table_capacity;
        ComputeOffsets();
    }

    void Initialize(bool create_region);
    std::vector<std::int32_t> AcquirePages(std::int64_t sequence_id,
                                           std::size_t num_pages);
    void ReleaseSequence(std::int64_t sequence_id);
    std::vector<std::int32_t> SequencePages(
        std::int64_t sequence_id, std::optional<std::size_t> max_pages) const;
    HostPagedKVStats CollectStats() const;

    std::byte* DataBase() { return data_base; }
    const std::byte* DataBase() const { return data_base; }

    HostPagedKVConfig config;
    std::size_t data_bytes = 0;
    std::uint64_t layout_fingerprint = 0;
    bool has_v_cache = false;

    int shm_fd = -1;
    std::size_t total_bytes = 0;
    std::byte* mapping = nullptr;
    SharedHeader* header = nullptr;
    std::int32_t* free_stack = nullptr;
    std::int64_t* page_owners = nullptr;
    std::int32_t* page_links = nullptr;
    SequenceEntry* sequence_table = nullptr;
    std::byte* data_base = nullptr;

    bool created_region = false;

    std::size_t header_offset = 0;
    std::size_t free_stack_offset = 0;
    std::size_t page_owner_offset = 0;
    std::size_t page_link_offset = 0;
    std::size_t sequence_table_offset = 0;
    std::size_t data_offset = 0;
    std::size_t total_bytes_unaligned = 0;
    std::size_t sequence_capacity = 0;

   private:
    void ComputeOffsets();
    void MapPointers();
    void ConstructSharedState();
    void WaitForInitialization() const;
    void ValidateSharedState() const;
    SequenceEntry* FindOrInsertSequenceEntryLocked(std::int64_t sequence_id,
                                                   bool* is_new);
    SequenceEntry* FindSequenceEntryLocked(std::int64_t sequence_id) const;
    std::size_t HashSequenceId(std::int64_t sequence_id) const;
};

void HostPagedKVBackend::SharedState::ComputeOffsets() {
    std::size_t offset = 0;

    offset = AlignUp(offset, alignof(SharedHeader));
    header_offset = offset;
    offset += sizeof(SharedHeader);

    offset = AlignUp(offset, alignof(std::int32_t));
    free_stack_offset = offset;
    offset += sizeof(std::int32_t) * config.num_pages;

    offset = AlignUp(offset, alignof(std::int64_t));
    page_owner_offset = offset;
    offset += sizeof(std::int64_t) * config.num_pages;

    offset = AlignUp(offset, alignof(std::int32_t));
    page_link_offset = offset;
    offset += sizeof(std::int32_t) * config.num_pages;

    offset = AlignUp(offset, alignof(SequenceEntry));
    sequence_table_offset = offset;
    offset += sizeof(SequenceEntry) * sequence_capacity;

    // Align data_offset to system page size (typically 4096 bytes) because
    // cudaHostRegister requires page-aligned pointers. Using std::max_align_t
    // (16 bytes) is insufficient and causes "invalid argument" errors.
    offset = AlignUp(offset, GetSystemPageSize());
    data_offset = offset;
    offset += data_bytes;

    total_bytes_unaligned = offset;
}

void HostPagedKVBackend::SharedState::MapPointers() {
    header = reinterpret_cast<SharedHeader*>(mapping + header_offset);
    free_stack = reinterpret_cast<std::int32_t*>(mapping + free_stack_offset);
    page_owners = reinterpret_cast<std::int64_t*>(mapping + page_owner_offset);
    page_links = reinterpret_cast<std::int32_t*>(mapping + page_link_offset);
    sequence_table =
        reinterpret_cast<SequenceEntry*>(mapping + sequence_table_offset);
    data_base = mapping + data_offset;
}

void HostPagedKVBackend::SharedState::ConstructSharedState() {
    std::memset(mapping, 0, total_bytes);
    MapPointers();
    header->magic = kSharedMemoryMagic;
    header->layout_fingerprint = layout_fingerprint;
    header->config_hash = HashHostKVConfig(config);
    header->data_bytes = data_bytes;
    header->num_pages = config.num_pages;
    header->sequence_capacity = sequence_capacity;
    header->alignment_bytes = config.alignment_bytes;
    header->num_layers = config.num_layers;
    header->page_size_tokens = config.page_size_tokens;
    header->has_v_cache = has_v_cache ? 1 : 0;
    header->free_stack_top.store(static_cast<std::uint32_t>(config.num_pages),
                                 std::memory_order_relaxed);
    header->active_sequences.store(0, std::memory_order_relaxed);

    for (std::size_t i = 0; i < config.num_pages; ++i) {
        free_stack[i] = static_cast<std::int32_t>(config.num_pages - 1 - i);
        page_owners[i] = kEmptySequenceId;
        page_links[i] = kInvalidPageIndex;
    }
    for (std::size_t i = 0; i < sequence_capacity; ++i) {
        sequence_table[i] = SequenceEntry();
    }

    pthread_mutexattr_t attr;
    if (const int rc = pthread_mutexattr_init(&attr); rc != 0) {
        throw std::system_error(rc, std::generic_category(),
                                "pthread_mutexattr_init failed");
    }
    if (const int rc =
            pthread_mutexattr_setpshared(&attr, PTHREAD_PROCESS_SHARED);
        rc != 0) {
        pthread_mutexattr_destroy(&attr);
        throw std::system_error(rc, std::generic_category(),
                                "pthread_mutexattr_setpshared failed");
    }
    if (const int rc = pthread_mutexattr_setrobust(&attr, PTHREAD_MUTEX_ROBUST);
        rc != 0) {
        pthread_mutexattr_destroy(&attr);
        throw std::system_error(rc, std::generic_category(),
                                "pthread_mutexattr_setrobust failed");
    }

    if (const int rc = pthread_mutex_init(&header->allocation_mutex, &attr);
        rc != 0) {
        pthread_mutexattr_destroy(&attr);
        throw std::system_error(rc, std::generic_category(),
                                "pthread_mutex_init allocation_mutex failed");
    }
    if (const int rc = pthread_mutex_init(&header->sequence_mutex, &attr);
        rc != 0) {
        pthread_mutex_destroy(&header->allocation_mutex);
        pthread_mutexattr_destroy(&attr);
        throw std::system_error(rc, std::generic_category(),
                                "pthread_mutex_init sequence_mutex failed");
    }
    pthread_mutexattr_destroy(&attr);

    header->init_state.store(static_cast<std::uint32_t>(InitState::kReady),
                             std::memory_order_release);
}

void HostPagedKVBackend::SharedState::WaitForInitialization() const {
    while (true) {
        const auto state = static_cast<InitState>(
            header->init_state.load(std::memory_order_acquire));
        if (state == InitState::kReady) {
            return;
        }
        if (state == InitState::kUninitialized) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void HostPagedKVBackend::SharedState::ValidateSharedState() const {
    if (header->magic != kSharedMemoryMagic) {
        std::ostringstream oss;
        oss << "Shared memory magic mismatch (expected=" << kSharedMemoryMagic
            << ", found=" << header->magic << ")";
        throw std::runtime_error(oss.str());
    }
    if (header->layout_fingerprint != layout_fingerprint) {
        std::ostringstream oss;
        oss << "Shared memory layout fingerprint mismatch (expected="
            << layout_fingerprint << ", found=" << header->layout_fingerprint
            << ")";
        throw std::runtime_error(oss.str());
    }
    if (header->config_hash != HashHostKVConfig(config)) {
        std::ostringstream oss;
        oss << "Shared memory config hash mismatch (expected="
            << HashHostKVConfig(config) << ", found=" << header->config_hash
            << ")";
        throw std::runtime_error(oss.str());
    }
    if (header->data_bytes != data_bytes) {
        std::ostringstream oss;
        oss << "Shared memory data size mismatch (expected=" << data_bytes
            << ", found=" << header->data_bytes << ")";
        throw std::runtime_error(oss.str());
    }
    if (header->num_pages != config.num_pages) {
        std::ostringstream oss;
        oss << "Shared memory num_pages mismatch (expected=" << config.num_pages
            << ", found=" << header->num_pages << ")";
        throw std::runtime_error(oss.str());
    }
    if (header->sequence_capacity != sequence_capacity) {
        std::ostringstream oss;
        oss << "Shared memory sequence capacity mismatch (expected="
            << sequence_capacity << ", found=" << header->sequence_capacity
            << ")";
        throw std::runtime_error(oss.str());
    }
}

std::size_t HostPagedKVBackend::SharedState::HashSequenceId(
    std::int64_t sequence_id) const {
    std::uint64_t x = static_cast<std::uint64_t>(sequence_id);
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return static_cast<std::size_t>(x % sequence_capacity);
}

SequenceEntry* HostPagedKVBackend::SharedState::FindSequenceEntryLocked(
    std::int64_t sequence_id) const {
    std::size_t index = HashSequenceId(sequence_id);
    for (std::size_t probe = 0; probe < sequence_capacity; ++probe) {
        SequenceEntry* entry = &sequence_table[index];
        if (entry->sequence_id == sequence_id) {
            return entry;
        }
        if (entry->sequence_id == kEmptySequenceId) {
            return nullptr;
        }
        index = (index + 1) % sequence_capacity;
    }
    return nullptr;
}

SequenceEntry* HostPagedKVBackend::SharedState::FindOrInsertSequenceEntryLocked(
    std::int64_t sequence_id, bool* is_new) {
    std::size_t index = HashSequenceId(sequence_id);
    SequenceEntry* first_tombstone = nullptr;
    for (std::size_t probe = 0; probe < sequence_capacity; ++probe) {
        SequenceEntry* entry = &sequence_table[index];
        if (entry->sequence_id == sequence_id) {
            if (is_new != nullptr) {
                *is_new = false;
            }
            return entry;
        }
        if (entry->sequence_id == kEmptySequenceId) {
            SequenceEntry* target =
                first_tombstone != nullptr ? first_tombstone : entry;
            if (target->sequence_id != sequence_id) {
                *target = SequenceEntry();
                target->sequence_id = sequence_id;
            }
            if (is_new != nullptr) {
                *is_new = true;
            }
            return target;
        }
        if (entry->sequence_id == kTombstoneSequenceId &&
            first_tombstone == nullptr) {
            first_tombstone = entry;
        }
        index = (index + 1) % sequence_capacity;
    }
    throw std::runtime_error("Sequence table is full (capacity=" +
                             std::to_string(sequence_capacity) + ")");
}

void HostPagedKVBackend::SharedState::Initialize(bool create_region) {
    const std::size_t page_size = GetSystemPageSize();
    total_bytes = AlignUp(total_bytes_unaligned, page_size);

    auto resize_region = [&](std::size_t bytes) {
        if (ftruncate(shm_fd, static_cast<off_t>(bytes)) == -1) {
            const int err = errno;
            close(shm_fd);
            shm_fd = -1;
            throw std::system_error(err, std::generic_category(),
                                    "ftruncate failed");
        }
    };

    auto reset_region = [&]() {
        if (ftruncate(shm_fd, 0) == -1) {
            const int err = errno;
            close(shm_fd);
            shm_fd = -1;
            throw std::system_error(err, std::generic_category(),
                                    "ftruncate reset failed");
        }
        resize_region(total_bytes);
    };

    int flags = O_RDWR;
    if (create_region) {
        flags |= O_CREAT;
    }

    shm_fd = shm_open(config.shm_name.c_str(), flags, 0660);
    if (shm_fd == -1) {
        throw std::system_error(errno, std::generic_category(),
                                "shm_open failed");
    }

    if (create_region) {
        struct stat stat_buffer {};
        if (fstat(shm_fd, &stat_buffer) == -1) {
            const int err = errno;
            close(shm_fd);
            shm_fd = -1;
            throw std::system_error(err, std::generic_category(),
                                    "fstat failed");
        }

        const std::size_t current_size =
            static_cast<std::size_t>(stat_buffer.st_size);
        if (current_size == total_bytes && current_size != 0) {
            reset_region();
        } else {
            resize_region(total_bytes);
        }
        created_region = true;
    } else {
        struct stat stat_buffer {};
        if (fstat(shm_fd, &stat_buffer) == -1) {
            const int err = errno;
            close(shm_fd);
            shm_fd = -1;
            throw std::system_error(err, std::generic_category(),
                                    "fstat failed");
        }
        if (static_cast<std::size_t>(stat_buffer.st_size) < total_bytes) {
            close(shm_fd);
            shm_fd = -1;
            throw std::runtime_error(
                "Existing shared memory segment is too small");
        }
    }

    void* mapped = mmap(nullptr, total_bytes, PROT_READ | PROT_WRITE,
                        MAP_SHARED, shm_fd, 0);
    if (mapped == MAP_FAILED) {
        const int err = errno;
        close(shm_fd);
        shm_fd = -1;
        throw std::system_error(err, std::generic_category(), "mmap failed");
    }

    mapping = static_cast<std::byte*>(mapped);
    MapPointers();

    if (created_region) {
        header->init_state.store(
            static_cast<std::uint32_t>(InitState::kInitializing),
            std::memory_order_relaxed);
        ConstructSharedState();
    } else {
        WaitForInitialization();
        ValidateSharedState();
    }
}

std::vector<std::int32_t> HostPagedKVBackend::SharedState::AcquirePages(
    std::int64_t sequence_id, std::size_t num_pages) {
    if (num_pages == 0) {
        return {};
    }
    std::vector<std::int32_t> pages(num_pages);
    {
        ScopedMutexLock lock(&header->allocation_mutex);
        const std::uint32_t top =
            header->free_stack_top.load(std::memory_order_relaxed);
        if (top < num_pages) {
            throw std::runtime_error(
                "Insufficient free pages available for sequence " +
                std::to_string(sequence_id) +
                " (requested=" + std::to_string(num_pages) +
                ", available=" + std::to_string(top) + ")");
        }
        std::uint32_t new_top = top - static_cast<std::uint32_t>(num_pages);
        for (std::size_t i = 0; i < num_pages; ++i) {
            pages[i] = free_stack[new_top + i];
        }
        header->free_stack_top.store(new_top, std::memory_order_relaxed);
    }

    {
        ScopedMutexLock lock(&header->sequence_mutex);
        bool is_new = false;
        SequenceEntry* entry =
            FindOrInsertSequenceEntryLocked(sequence_id, &is_new);
        if (is_new) {
            header->active_sequences.fetch_add(1, std::memory_order_relaxed);
        }
        for (std::size_t i = 0; i < num_pages; ++i) {
            const std::int32_t page = pages[i];
            page_owners[page] = sequence_id;
            page_links[page] = kInvalidPageIndex;
            if (entry->head_page == kInvalidPageIndex) {
                entry->head_page = page;
                entry->tail_page = page;
            } else {
                page_links[entry->tail_page] = page;
                entry->tail_page = page;
            }
            ++entry->num_pages;
        }
    }
    return pages;
}

void HostPagedKVBackend::SharedState::ReleaseSequence(
    std::int64_t sequence_id) {
    std::vector<std::int32_t> pages;
    {
        ScopedMutexLock lock(&header->sequence_mutex);
        SequenceEntry* entry = FindSequenceEntryLocked(sequence_id);
        if (entry == nullptr) {
            throw std::out_of_range("Sequence ID " +
                                    std::to_string(sequence_id) +
                                    " not found during release");
        }
        pages.reserve(entry->num_pages);
        std::int32_t page = entry->head_page;
        while (page != kInvalidPageIndex) {
            pages.push_back(page);
            const std::int32_t next = page_links[page];
            page_links[page] = kInvalidPageIndex;
            page_owners[page] = kEmptySequenceId;
            page = next;
        }
        entry->sequence_id = kTombstoneSequenceId;
        entry->num_pages = 0;
        entry->head_page = kInvalidPageIndex;
        entry->tail_page = kInvalidPageIndex;
        header->active_sequences.fetch_sub(1, std::memory_order_relaxed);
    }

    if (!pages.empty()) {
        ScopedMutexLock lock(&header->allocation_mutex);
        std::uint32_t top =
            header->free_stack_top.load(std::memory_order_relaxed);
        for (std::int32_t page : pages) {
            free_stack[top++] = page;
        }
        header->free_stack_top.store(top, std::memory_order_relaxed);
    }
}

std::vector<std::int32_t> HostPagedKVBackend::SharedState::SequencePages(
    std::int64_t sequence_id, std::optional<std::size_t> max_pages) const {
    ScopedMutexLock lock(&header->sequence_mutex);
    SequenceEntry* entry = FindSequenceEntryLocked(sequence_id);
    if (entry == nullptr) {
        throw std::out_of_range("Sequence ID " + std::to_string(sequence_id) +
                                " not found when fetching pages");
    }
    const std::size_t available_pages = entry->num_pages;
    const std::size_t limit = max_pages.has_value()
                                  ? std::min(max_pages.value(), available_pages)
                                  : available_pages;
    if (max_pages.has_value() && max_pages.value() > available_pages) {
        throw std::out_of_range(
            "Requested " + std::to_string(max_pages.value()) +
            " pages but only " + std::to_string(available_pages) +
            " pages allocated for sequence " + std::to_string(sequence_id));
    }

    std::vector<std::int32_t> pages;
    pages.reserve(limit);
    std::int32_t page = entry->head_page;
    std::size_t count = 0;
    while (page != kInvalidPageIndex && count < limit) {
        pages.push_back(page);
        page = page_links[page];
        ++count;
    }
    return pages;
}

HostPagedKVStats HostPagedKVBackend::SharedState::CollectStats() const {
    HostPagedKVStats stats;
    stats.num_total_pages = config.num_pages;
    const std::uint32_t free_count =
        header->free_stack_top.load(std::memory_order_relaxed);
    stats.num_free_pages = free_count;
    stats.num_used_pages = config.num_pages - free_count;
    stats.num_active_sequences =
        header->active_sequences.load(std::memory_order_relaxed);
    stats.sequence_table_capacity = sequence_capacity;
    stats.total_bytes = total_bytes;
    return stats;
}

// HostPagedKVBackend public API

HostPagedKVBackend::HostPagedKVBackend(HostPagedKVConfig config,
                                       std::size_t data_bytes,
                                       std::uint64_t layout_fingerprint,
                                       bool has_v_cache)
    : config_(std::move(config)),
      data_bytes_(data_bytes),
      layout_fingerprint_(layout_fingerprint),
      has_v_cache_(has_v_cache),
      state_(std::make_unique<SharedState>(
          config_, data_bytes_, layout_fingerprint_, has_v_cache_)) {}

HostPagedKVBackend::~HostPagedKVBackend() {
    if (state_->mapping != nullptr && state_->total_bytes != 0) {
        munmap(state_->mapping, state_->total_bytes);
    }
    if (state_->shm_fd >= 0) {
        close(state_->shm_fd);
    }
}

void HostPagedKVBackend::Initialize(bool create_region) {
    state_->Initialize(create_region);
}

std::vector<std::int32_t> HostPagedKVBackend::AcquirePages(
    std::int64_t sequence_id, std::size_t num_pages) {
    return state_->AcquirePages(sequence_id, num_pages);
}

std::vector<std::vector<std::int32_t>>
HostPagedKVBackend::AcquirePagesForSequences(
    const std::vector<std::int64_t>& sequence_ids,
    const std::vector<std::size_t>& num_tokens) {
    if (sequence_ids.size() != num_tokens.size()) {
        throw std::invalid_argument(
            "sequence_ids and num_tokens must have the same length");
    }
    if (sequence_ids.empty()) {
        return {};
    }

    auto allocate_one = [this](std::int64_t sequence_id,
                               std::size_t num_tokens_value) {
        if (num_tokens_value == 0) {
            throw std::invalid_argument(
                "num_tokens must be greater than zero for sequence " +
                std::to_string(sequence_id));
        }
        const std::size_t required_pages =
            (num_tokens_value + config_.page_size_tokens - 1) /
            config_.page_size_tokens;
        return state_->AcquirePages(sequence_id, required_pages);
    };

    if (sequence_ids.size() == 1 || SafeHardwareConcurrency() == 1) {
        std::vector<std::vector<std::int32_t>> allocations;
        allocations.reserve(sequence_ids.size());
        for (std::size_t i = 0; i < sequence_ids.size(); ++i) {
            allocations.emplace_back(
                allocate_one(sequence_ids[i], num_tokens[i]));
        }
        return allocations;
    }

    std::vector<std::future<std::vector<std::int32_t>>> futures;
    futures.reserve(sequence_ids.size());
    for (std::size_t i = 0; i < sequence_ids.size(); ++i) {
        const std::int64_t sequence_id = sequence_ids[i];
        const std::size_t tokens = num_tokens[i];
        futures.emplace_back(std::async(std::launch::async, [=]() {
            return allocate_one(sequence_id, tokens);
        }));
    }

    std::vector<std::vector<std::int32_t>> allocations;
    allocations.reserve(futures.size());
    for (auto& future : futures) {
        allocations.emplace_back(future.get());
    }
    return allocations;
}

void HostPagedKVBackend::ReleaseSequence(std::int64_t sequence_id) {
    state_->ReleaseSequence(sequence_id);
}

void HostPagedKVBackend::ReleaseSequences(
    const std::vector<std::int64_t>& sequence_ids) {
    if (sequence_ids.empty()) {
        return;
    }
    if (sequence_ids.size() == 1 || SafeHardwareConcurrency() == 1) {
        for (std::int64_t sequence_id : sequence_ids) {
            state_->ReleaseSequence(sequence_id);
        }
        return;
    }
    std::vector<std::future<void>> futures;
    futures.reserve(sequence_ids.size());
    for (std::int64_t sequence_id : sequence_ids) {
        futures.emplace_back(std::async(std::launch::async,
                                        [state = state_.get(), sequence_id]() {
                                            state->ReleaseSequence(sequence_id);
                                        }));
    }
    for (auto& future : futures) {
        future.get();
    }
}

std::vector<std::int32_t> HostPagedKVBackend::SequencePages(
    std::int64_t sequence_id, std::optional<std::size_t> max_pages) const {
    return state_->SequencePages(sequence_id, max_pages);
}

HostPagedKVStats HostPagedKVBackend::CollectStats() const {
    return state_->CollectStats();
}

std::byte* HostPagedKVBackend::DataBase() { return state_->DataBase(); }

const std::byte* HostPagedKVBackend::DataBase() const {
    return state_->DataBase();
}

}  // namespace batchgen::kv
