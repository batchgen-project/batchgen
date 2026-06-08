#include "host_paged_kv_backend.h"

#include "shared_memory_utils.h"

#include <fcntl.h>
#include <linux/memfd.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
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
constexpr std::int64_t kPrefixResidentSequenceId = kEmptySequenceId + 2;

struct SequenceEntry {
    std::int64_t sequence_id = kEmptySequenceId;
    std::uint32_t num_pages = 0;
    std::int32_t head_page = kInvalidPageIndex;
    std::int32_t tail_page = kInvalidPageIndex;
};

struct SharedHeader {
    std::atomic<std::uint32_t> init_state{
        static_cast<std::uint32_t>(SharedMemoryInitState::kUninitialized)};
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

int memfd_create_wrapper(const char* name, unsigned int flags) {
    return static_cast<int>(syscall(SYS_memfd_create, name, flags));
}

void TouchPagesMultiThreaded(void* ptr, std::size_t size, std::size_t stride) {
    const int num_threads = std::min(16, static_cast<int>(std::thread::hardware_concurrency()));
    const std::size_t chunk_size = size / num_threads;
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([=]() {
            std::size_t start = i * chunk_size;
            std::size_t end = (i == num_threads - 1) ? size : start + chunk_size;
            volatile char* p = static_cast<volatile char*>(ptr);
            for (std::size_t off = start; off < end; off += stride) {
                p[off] = 0;
            }
        });
    }
    for (auto& t : threads) {
        t.join();
    }
}

// Helper function to perform aligned mmap
// This ensures the virtual address is aligned to the specified alignment (e.g., 2MB for huge pages)
// which is often required for cudaHostRegister to work correctly with huge pages.
void* mmap_aligned(size_t length, int prot, int flags, int fd, off_t offset, size_t alignment) {
    // Allocate extra space to ensure we can find an aligned segment
    size_t total_len = length + alignment;
    
    // Reserve address space using anonymous mapping
    void* addr = mmap(nullptr, total_len, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (addr == MAP_FAILED) {
        return MAP_FAILED;
    }

    uintptr_t raw_addr = reinterpret_cast<uintptr_t>(addr);
    uintptr_t aligned_addr = (raw_addr + alignment - 1) & ~(alignment - 1);
    void* final_addr = reinterpret_cast<void*>(aligned_addr);

    // Map the file into the aligned position using MAP_FIXED
    // This replaces the anonymous mapping at that location
    void* ret = mmap(final_addr, length, prot, flags | MAP_FIXED, fd, offset);
    
    // Unmap the unused parts of the reservation
    size_t prefix_len = aligned_addr - raw_addr;
    if (prefix_len > 0) {
        munmap(addr, prefix_len);
    }
    
    size_t suffix_len = total_len - length - prefix_len;
    if (suffix_len > 0) {
        munmap(reinterpret_cast<void*>(aligned_addr + length), suffix_len);
    }

    return ret;
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
    std::vector<std::int32_t> ReleasePrefixPages(std::int64_t sequence_id,
                                                 std::size_t num_pages);
    std::vector<std::int32_t> RetainPrefixPages(std::int64_t sequence_id,
                                                std::size_t num_pages);
    std::vector<std::int32_t> RetainPageRange(std::int64_t sequence_id,
                                              std::size_t start_page,
                                              std::size_t num_pages);
    void ReleaseResidentPages(const std::vector<std::int32_t>& page_ids);
    std::vector<std::int32_t> SequencePages(
        std::int64_t sequence_id, std::optional<std::size_t> max_pages) const;
    std::vector<std::int32_t> SequencePageRange(
        std::int64_t sequence_id, std::size_t start_page,
        std::size_t page_count) const;
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
    bool using_memfd = false;
    int memfd_fd_value = -1;

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

    // Align data_offset to 2MB (huge page size) because cudaHostRegister may require
    // huge-page aligned pointers when the underlying memory is backed by huge pages
    // (e.g., via Transparent Huge Pages or hugetlbfs).
    // Using simple page alignment (4KB) can cause "invalid argument" errors with cudaHostRegister
    // on some systems when THP is active.
    constexpr std::size_t kHugePageAlignment = 2 * 1024 * 1024;
    offset = AlignUp(offset, kHugePageAlignment);
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
    // Always zero the entire mapping — historically we skipped the data
    // region under --fast-init because memfd pages are kernel-zeroed on
    // first fault. That was fine for clean startup, but page 0 gets
    // legitimately written by some sequence during a run, and if any
    // gather path later reads page 0 via a -1->0 clamp fallback (see
    // sparse_gather.py comment on invalid_mask), the reader picks up
    // whatever that other sequence wrote. Explicit zero-init at
    // allocation is the belt to the gather-level suspenders.
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

    InitProcessSharedRobustMutex(
        &header->allocation_mutex,
        "pthread_mutex_init allocation_mutex failed");
    try {
        InitProcessSharedRobustMutex(
            &header->sequence_mutex,
            "pthread_mutex_init sequence_mutex failed");
    } catch (...) {
        pthread_mutex_destroy(&header->allocation_mutex);
        throw;
    }

    header->init_state.store(
        static_cast<std::uint32_t>(SharedMemoryInitState::kReady),
        std::memory_order_release);
}

void HostPagedKVBackend::SharedState::WaitForInitialization() const {
    while (true) {
        const auto state = static_cast<SharedMemoryInitState>(
            header->init_state.load(std::memory_order_acquire));
        if (state == SharedMemoryInitState::kReady) {
            return;
        }
        if (state == SharedMemoryInitState::kUninitialized) {
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
    const std::size_t page_size = SystemPageSize();
    total_bytes = AlignUp(total_bytes_unaligned, page_size);
    constexpr std::size_t kHugePageSize = 2 * 1024 * 1024;
    const std::size_t alignment = std::max(kHugePageSize, page_size);

    if (config.enable_memfd) {
        // ---- memfd_create fast path (--fast-init) ----
        if (create_region) {
            int fd = memfd_create_wrapper("batchgen_kv", 0);
            if (fd < 0) {
                throw std::runtime_error(
                    "--fast-init: memfd_create failed: " +
                    std::string(strerror(errno)));
            }
            if (ftruncate(fd, static_cast<off_t>(total_bytes)) == -1) {
                const int err = errno;
                close(fd);
                throw std::system_error(err, std::generic_category(),
                                        "--fast-init: ftruncate on memfd failed");
            }

            void* mapped = mmap_aligned(total_bytes, PROT_READ | PROT_WRITE,
                                        MAP_SHARED, fd, 0, alignment);
            if (mapped == MAP_FAILED) {
                const int err = errno;
                close(fd);
                throw std::system_error(err, std::generic_category(),
                                        "--fast-init: mmap on memfd failed");
            }

            madvise(mapped, total_bytes, MADV_HUGEPAGE);

            auto touch_start = std::chrono::high_resolution_clock::now();
            TouchPagesMultiThreaded(mapped, total_bytes, kHugePageSize);
            auto touch_dur = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - touch_start);

            shm_fd = fd;
            memfd_fd_value = fd;
            using_memfd = true;
            created_region = true;
            mapping = static_cast<std::byte*>(mapped);
            MapPointers();
            header->init_state.store(
                static_cast<std::uint32_t>(
                    SharedMemoryInitState::kInitializing),
                std::memory_order_relaxed);
            ConstructSharedState();
        } else {
            // Worker: open memfd via /proc/<pid>/fd/<N>
            if (config.memfd_creator_pid <= 0 || config.memfd_fd < 0) {
                throw std::runtime_error(
                    "--fast-init worker: invalid memfd_creator_pid or memfd_fd");
            }
            std::string proc_path = "/proc/" +
                                    std::to_string(config.memfd_creator_pid) +
                                    "/fd/" + std::to_string(config.memfd_fd);
            int fd = open(proc_path.c_str(), O_RDWR);
            if (fd < 0) {
                throw std::runtime_error(
                    "--fast-init worker: cannot open " + proc_path + ": " +
                    strerror(errno));
            }
            struct stat stat_buffer {};
            if (fstat(fd, &stat_buffer) == -1) {
                const int err = errno;
                close(fd);
                throw std::system_error(err, std::generic_category(),
                                        "--fast-init worker: fstat failed");
            }
            if (static_cast<std::size_t>(stat_buffer.st_size) < total_bytes) {
                close(fd);
                throw std::runtime_error(
                    "--fast-init worker: memfd too small");
            }

            void* mapped = mmap_aligned(total_bytes, PROT_READ | PROT_WRITE,
                                        MAP_SHARED, fd, 0, alignment);
            if (mapped == MAP_FAILED) {
                const int err = errno;
                close(fd);
                throw std::system_error(err, std::generic_category(),
                                        "--fast-init worker: mmap failed");
            }

            shm_fd = fd;
            using_memfd = true;
            mapping = static_cast<std::byte*>(mapped);
            MapPointers();
            WaitForInitialization();
            ValidateSharedState();
        }
        return;
    }

    // ---- Original shm_open path (unchanged) ----
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

    void* mapped = mmap_aligned(total_bytes, PROT_READ | PROT_WRITE,
                        MAP_SHARED, shm_fd, 0, alignment);

    if (mapped == MAP_FAILED) {
        const int err = errno;
        close(shm_fd);
        shm_fd = -1;
        throw std::system_error(err, std::generic_category(), "mmap failed");
    }

    madvise(mapped, total_bytes, MADV_HUGEPAGE);

    mapping = static_cast<std::byte*>(mapped);
    MapPointers();

    if (created_region) {
        header->init_state.store(
            static_cast<std::uint32_t>(SharedMemoryInitState::kInitializing),
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
        ScopedPthreadMutexLock lock(&header->allocation_mutex);
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
        ScopedPthreadMutexLock lock(&header->sequence_mutex);
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
        ScopedPthreadMutexLock lock(&header->sequence_mutex);
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
        ScopedPthreadMutexLock lock(&header->allocation_mutex);
        std::uint32_t top =
            header->free_stack_top.load(std::memory_order_relaxed);
        for (std::int32_t page : pages) {
            free_stack[top++] = page;
        }
        header->free_stack_top.store(top, std::memory_order_relaxed);
    }
}

std::vector<std::int32_t> HostPagedKVBackend::SharedState::ReleasePrefixPages(
    std::int64_t sequence_id, std::size_t num_pages) {
    if (num_pages == 0) {
        return {};
    }
    std::vector<std::int32_t> pages;
    {
        ScopedPthreadMutexLock lock(&header->sequence_mutex);
        SequenceEntry* entry = FindSequenceEntryLocked(sequence_id);
        if (entry == nullptr) {
            throw std::out_of_range("Sequence ID " +
                                    std::to_string(sequence_id) +
                                    " not found during prefix release");
        }
        if (num_pages > entry->num_pages) {
            throw std::out_of_range(
                "Requested prefix release of " + std::to_string(num_pages) +
                " pages but sequence " + std::to_string(sequence_id) +
                " only owns " + std::to_string(entry->num_pages) + " pages");
        }

        pages.reserve(num_pages);
        std::int32_t page = entry->head_page;
        for (std::size_t i = 0; i < num_pages; ++i) {
            if (page == kInvalidPageIndex) {
                throw std::logic_error(
                    "Corrupt page chain during prefix release for sequence " +
                    std::to_string(sequence_id));
            }
            pages.push_back(page);
            const std::int32_t next = page_links[page];
            page_links[page] = kInvalidPageIndex;
            page_owners[page] = kEmptySequenceId;
            page = next;
        }

        entry->head_page = page;
        entry->num_pages -= static_cast<std::uint32_t>(num_pages);
        if (entry->num_pages == 0) {
            entry->tail_page = kInvalidPageIndex;
        }
    }

    if (!pages.empty()) {
        ScopedPthreadMutexLock lock(&header->allocation_mutex);
        std::uint32_t top =
            header->free_stack_top.load(std::memory_order_relaxed);
        for (std::int32_t page : pages) {
            free_stack[top++] = page;
        }
        header->free_stack_top.store(top, std::memory_order_relaxed);
    }
    return pages;
}

std::vector<std::int32_t> HostPagedKVBackend::SharedState::RetainPrefixPages(
    std::int64_t sequence_id, std::size_t num_pages) {
    return RetainPageRange(sequence_id, 0, num_pages);
}

std::vector<std::int32_t> HostPagedKVBackend::SharedState::RetainPageRange(
    std::int64_t sequence_id, std::size_t start_page, std::size_t num_pages) {
    if (num_pages == 0) {
        return {};
    }
    std::vector<std::int32_t> pages;
    {
        ScopedPthreadMutexLock lock(&header->sequence_mutex);
        SequenceEntry* entry = FindSequenceEntryLocked(sequence_id);
        if (entry == nullptr) {
            throw std::out_of_range("Sequence ID " +
                                    std::to_string(sequence_id) +
                                    " not found during prefix retain");
        }
        if (start_page > entry->num_pages ||
            num_pages > entry->num_pages - start_page) {
            throw std::out_of_range(
                "Requested retain of " + std::to_string(num_pages) +
                " pages from offset " + std::to_string(start_page) +
                " but sequence " + std::to_string(sequence_id) +
                " only owns " + std::to_string(entry->num_pages) +
                " pages");
        }

        pages.reserve(num_pages);
        std::int32_t page = entry->head_page;
        std::int32_t previous_page = kInvalidPageIndex;
        for (std::size_t i = 0; i < start_page; ++i) {
            if (page == kInvalidPageIndex) {
                throw std::logic_error(
                    "Corrupt page chain before retained range for sequence " +
                    std::to_string(sequence_id));
            }
            previous_page = page;
            page = page_links[page];
        }

        for (std::size_t i = 0; i < num_pages; ++i) {
            if (page == kInvalidPageIndex) {
                throw std::logic_error(
                    "Corrupt page chain during range retain for sequence " +
                    std::to_string(sequence_id));
            }
            pages.push_back(page);
            const std::int32_t next = page_links[page];
            page_links[page] = kInvalidPageIndex;
            page_owners[page] = kPrefixResidentSequenceId;
            page = next;
        }

        if (previous_page == kInvalidPageIndex) {
            entry->head_page = page;
        } else {
            page_links[previous_page] = page;
        }
        entry->num_pages -= static_cast<std::uint32_t>(num_pages);
        if (entry->num_pages == 0) {
            entry->tail_page = kInvalidPageIndex;
        } else if (page == kInvalidPageIndex) {
            entry->tail_page = previous_page;
        }
    }
    return pages;
}

void HostPagedKVBackend::SharedState::ReleaseResidentPages(
    const std::vector<std::int32_t>& page_ids) {
    if (page_ids.empty()) {
        return;
    }
    {
        ScopedPthreadMutexLock lock(&header->sequence_mutex);
        for (const std::int32_t page : page_ids) {
            if (page < 0 ||
                static_cast<std::size_t>(page) >= config.num_pages) {
                throw std::out_of_range(
                    "Resident page id out of range: " +
                    std::to_string(page));
            }
            if (page_owners[page] != kPrefixResidentSequenceId) {
                throw std::runtime_error(
                    "Cannot release page " + std::to_string(page) +
                    " because it is not prefix-resident");
            }
            page_owners[page] = kEmptySequenceId;
            page_links[page] = kInvalidPageIndex;
        }
    }

    ScopedPthreadMutexLock lock(&header->allocation_mutex);
    std::uint32_t top =
        header->free_stack_top.load(std::memory_order_relaxed);
    for (const std::int32_t page : page_ids) {
        free_stack[top++] = page;
    }
    header->free_stack_top.store(top, std::memory_order_relaxed);
}

std::vector<std::int32_t> HostPagedKVBackend::SharedState::SequencePages(
    std::int64_t sequence_id, std::optional<std::size_t> max_pages) const {
    ScopedPthreadMutexLock lock(&header->sequence_mutex);
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

std::vector<std::int32_t> HostPagedKVBackend::SharedState::SequencePageRange(
    std::int64_t sequence_id, std::size_t start_page,
    std::size_t page_count) const {
    ScopedPthreadMutexLock lock(&header->sequence_mutex);
    SequenceEntry* entry = FindSequenceEntryLocked(sequence_id);
    if (entry == nullptr) {
        throw std::out_of_range("Sequence ID " + std::to_string(sequence_id) +
                                " not found when fetching page range");
    }
    const std::size_t available_pages = entry->num_pages;
    if (start_page > available_pages) {
        throw std::out_of_range(
            "Requested page range starting at " + std::to_string(start_page) +
            " but sequence " + std::to_string(sequence_id) + " only owns " +
            std::to_string(available_pages) + " pages");
    }
    if (page_count > available_pages - start_page) {
        throw std::out_of_range(
            "Requested " + std::to_string(page_count) +
            " pages from offset " + std::to_string(start_page) +
            " but sequence " + std::to_string(sequence_id) + " only has " +
            std::to_string(available_pages) + " pages");
    }

    std::vector<std::int32_t> pages;
    pages.reserve(page_count);
    std::int32_t page = entry->head_page;
    for (std::size_t skipped = 0; skipped < start_page; ++skipped) {
        if (page == kInvalidPageIndex) {
            throw std::logic_error(
                "Corrupt page chain while skipping to page range for sequence " +
                std::to_string(sequence_id));
        }
        page = page_links[page];
    }
    for (std::size_t count = 0; count < page_count; ++count) {
        if (page == kInvalidPageIndex) {
            throw std::logic_error(
                "Corrupt page chain while reading page range for sequence " +
                std::to_string(sequence_id));
        }
        pages.push_back(page);
        page = page_links[page];
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

std::vector<std::int32_t> HostPagedKVBackend::ReleaseSequencePrefixPages(
    std::int64_t sequence_id, std::size_t num_pages) {
    return state_->ReleasePrefixPages(sequence_id, num_pages);
}

std::vector<std::int32_t> HostPagedKVBackend::RetainSequencePrefixPages(
    std::int64_t sequence_id, std::size_t num_pages) {
    return state_->RetainPrefixPages(sequence_id, num_pages);
}

std::vector<std::int32_t> HostPagedKVBackend::RetainSequencePageRange(
    std::int64_t sequence_id, std::size_t start_page,
    std::size_t num_pages) {
    return state_->RetainPageRange(sequence_id, start_page, num_pages);
}

void HostPagedKVBackend::ReleaseResidentPages(
    const std::vector<std::int32_t>& page_ids) {
    state_->ReleaseResidentPages(page_ids);
}

std::vector<std::int32_t> HostPagedKVBackend::SequencePages(
    std::int64_t sequence_id, std::optional<std::size_t> max_pages) const {
    return state_->SequencePages(sequence_id, max_pages);
}

std::vector<std::int32_t> HostPagedKVBackend::SequencePageRange(
    std::int64_t sequence_id, std::size_t start_page,
    std::size_t page_count) const {
    return state_->SequencePageRange(sequence_id, start_page, page_count);
}

HostPagedKVStats HostPagedKVBackend::CollectStats() const {
    return state_->CollectStats();
}

std::byte* HostPagedKVBackend::DataBase() { return state_->DataBase(); }

const std::byte* HostPagedKVBackend::DataBase() const {
    return state_->DataBase();
}

int HostPagedKVBackend::memfd_fd() const {
    return state_->memfd_fd_value;
}

}  // namespace batchgen::kv
