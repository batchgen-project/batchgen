#include "host_paged_kv_backend.h"
#include "host_paged_kv_prefix_cache.h"

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
constexpr std::int32_t kInvalidIndex = -1;
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
            std::terminate();
        }
    }

   private:
    pthread_mutex_t* mu_;
};

// Helper function to perform aligned mmap.
void* mmap_aligned(size_t length, int prot, int flags, int fd, off_t offset,
                   size_t alignment) {
    const size_t total_len = length + alignment;
    void* addr =
        mmap(nullptr, total_len, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (addr == MAP_FAILED) {
        return MAP_FAILED;
    }

    const uintptr_t raw_addr = reinterpret_cast<uintptr_t>(addr);
    const uintptr_t aligned_addr =
        (raw_addr + alignment - 1) & ~(alignment - 1);
    void* final_addr = reinterpret_cast<void*>(aligned_addr);

    void* ret = mmap(final_addr, length, prot, flags | MAP_FIXED, fd, offset);

    const size_t prefix_len = aligned_addr - raw_addr;
    if (prefix_len > 0) {
        munmap(addr, prefix_len);
    }

    const size_t suffix_len = total_len - length - prefix_len;
    if (suffix_len > 0) {
        munmap(reinterpret_cast<void*>(aligned_addr + length), suffix_len);
    }

    return ret;
}

}  // namespace

struct SequenceEntry {
    std::int64_t sequence_id = kEmptySequenceId;
    std::uint32_t num_pages = 0;
    std::int32_t head_node = kInvalidIndex;
    std::int32_t tail_node = kInvalidIndex;
};

struct SeqPageNode {
    std::int32_t page_idx = kInvalidIndex;
    std::int32_t next_node = kInvalidIndex;
};
using RadixNode = HostKVRadixNode;
using RadixEdge = HostKVRadixEdge;
using PrefixEntry = HostKVPrefixEntry;
using PrefixPageRef = HostKVPrefixPageRef;

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
    std::uint32_t enable_prefix_reuse = 0;

    std::uint64_t sequence_page_node_capacity = 0;
    std::uint64_t radix_node_capacity = 0;
    std::uint64_t radix_edge_capacity = 0;
    std::uint64_t prefix_entry_capacity = 0;
    std::uint64_t prefix_page_ref_capacity = 0;
    std::uint64_t prefix_page_budget = 0;

    std::atomic<std::uint32_t> free_stack_top{0};
    std::atomic<std::uint32_t> seq_page_node_free_top{0};
    std::atomic<std::uint32_t> radix_node_free_top{0};
    std::atomic<std::uint32_t> radix_edge_free_top{0};
    std::atomic<std::uint32_t> prefix_entry_free_top{0};
    std::atomic<std::uint32_t> prefix_page_ref_free_top{0};

    std::atomic<std::uint32_t> active_sequences{0};
    std::atomic<std::uint32_t> prefix_entry_count{0};
    std::atomic<std::uint32_t> prefix_used_pages{0};

    std::atomic<std::uint64_t> prefix_access_epoch{0};
    std::atomic<std::uint64_t> prefix_hit_count{0};
    std::atomic<std::uint64_t> prefix_miss_count{0};
    std::atomic<std::uint64_t> prefix_evict_count{0};

    std::int32_t lru_head = kInvalidIndex;
    std::int32_t lru_tail = kInvalidIndex;

    pthread_mutex_t metadata_mutex{};
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
        sequence_page_node_capacity = config.sequence_page_node_capacity;
        radix_node_capacity = config.radix_node_capacity;
        radix_edge_capacity = config.radix_edge_capacity;
        prefix_entry_capacity = config.prefix_entry_capacity;
        prefix_page_ref_capacity = config.prefix_page_ref_capacity;
        ComputeOffsets();
    }

    void Initialize(bool create_region);
    std::vector<std::int32_t> AcquirePages(std::int64_t sequence_id,
                                           std::size_t num_pages);
    PrefixAllocationBatchResult AcquirePagesForSequencesWithPrefix(
        const std::vector<std::int64_t>& sequence_ids,
        const std::vector<std::size_t>& num_tokens,
        const std::vector<std::int32_t>& flat_prompt_tokens,
        const std::vector<std::size_t>& prompt_offsets);
    void ReleaseSequence(std::int64_t sequence_id);
    std::vector<std::int32_t> SequencePages(
        std::int64_t sequence_id, std::optional<std::size_t> max_pages) const;
    void CommitSequencePrefix(std::int64_t sequence_id,
                              const std::vector<std::int32_t>& prompt_tokens,
                              std::size_t prompt_token_count);
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
    std::uint32_t* page_refcount = nullptr;
    SequenceEntry* sequence_table = nullptr;
    SeqPageNode* seq_page_nodes = nullptr;
    std::int32_t* seq_page_node_free_stack = nullptr;

    RadixNode* radix_nodes = nullptr;
    RadixEdge* radix_edges = nullptr;
    std::int32_t* radix_node_free_stack = nullptr;
    std::int32_t* radix_edge_free_stack = nullptr;

    PrefixEntry* prefix_entries = nullptr;
    PrefixPageRef* prefix_page_refs = nullptr;
    std::int32_t* prefix_entry_free_stack = nullptr;
    std::int32_t* prefix_page_ref_free_stack = nullptr;

    std::byte* data_base = nullptr;

    bool created_region = false;
    bool using_memfd = false;
    int memfd_fd_value = -1;

    std::size_t header_offset = 0;
    std::size_t free_stack_offset = 0;
    std::size_t page_refcount_offset = 0;
    std::size_t sequence_table_offset = 0;
    std::size_t seq_page_nodes_offset = 0;
    std::size_t seq_page_node_free_stack_offset = 0;
    std::size_t radix_nodes_offset = 0;
    std::size_t radix_edges_offset = 0;
    std::size_t radix_node_free_stack_offset = 0;
    std::size_t radix_edge_free_stack_offset = 0;
    std::size_t prefix_entries_offset = 0;
    std::size_t prefix_page_refs_offset = 0;
    std::size_t prefix_entry_free_stack_offset = 0;
    std::size_t prefix_page_ref_free_stack_offset = 0;
    std::size_t data_offset = 0;
    std::size_t total_bytes_unaligned = 0;

    std::size_t sequence_capacity = 0;
    std::size_t sequence_page_node_capacity = 0;
    std::size_t radix_node_capacity = 0;
    std::size_t radix_edge_capacity = 0;
    std::size_t prefix_entry_capacity = 0;
    std::size_t prefix_page_ref_capacity = 0;
    HostKVPrefixCache prefix_cache_;

   private:
    void ComputeOffsets();
    void MapPointers();
    void BindPrefixCache();
    void ConstructSharedState();
    void WaitForInitialization() const;
    void ValidateSharedState() const;

    std::size_t HashSequenceId(std::int64_t sequence_id) const;
    SequenceEntry* FindSequenceEntryLocked(std::int64_t sequence_id) const;
    SequenceEntry* FindOrInsertSequenceEntryLocked(std::int64_t sequence_id,
                                                   bool* is_new,
                                                   std::int64_t* previous_marker);

    std::int32_t PopStackIndexLocked(std::int32_t* stack,
                                     std::atomic<std::uint32_t>* top,
                                     const char* what) const;
    void PushStackIndexLocked(std::int32_t* stack,
                              std::atomic<std::uint32_t>* top,
                              std::int32_t value) const;

    std::int32_t AllocateSeqPageNodeLocked(std::int32_t page_idx);
    void FreeSeqPageNodeLocked(std::int32_t node_idx);
    void AppendPageToSequenceLocked(SequenceEntry* entry, std::int32_t page_idx);
    std::vector<std::int32_t> CollectSequencePagesLocked(
        const SequenceEntry* entry, std::optional<std::size_t> max_pages) const;
    void ReleaseSequencePagesLocked(SequenceEntry* entry);

    std::int32_t PopFreePageLocked();
    void PushFreePageLocked(std::int32_t page_idx);
    void IncrementPageRefLocked(std::int32_t page_idx);
    void DecrementPageRefLocked(std::int32_t page_idx);
};

void HostPagedKVBackend::SharedState::ComputeOffsets() {
    std::size_t offset = 0;

    offset = AlignUp(offset, alignof(SharedHeader));
    header_offset = offset;
    offset += sizeof(SharedHeader);

    offset = AlignUp(offset, alignof(std::int32_t));
    free_stack_offset = offset;
    offset += sizeof(std::int32_t) * config.num_pages;

    offset = AlignUp(offset, alignof(std::uint32_t));
    page_refcount_offset = offset;
    offset += sizeof(std::uint32_t) * config.num_pages;

    offset = AlignUp(offset, alignof(SequenceEntry));
    sequence_table_offset = offset;
    offset += sizeof(SequenceEntry) * sequence_capacity;

    offset = AlignUp(offset, alignof(SeqPageNode));
    seq_page_nodes_offset = offset;
    offset += sizeof(SeqPageNode) * sequence_page_node_capacity;

    offset = AlignUp(offset, alignof(std::int32_t));
    seq_page_node_free_stack_offset = offset;
    offset += sizeof(std::int32_t) * sequence_page_node_capacity;

    offset = AlignUp(offset, alignof(RadixNode));
    radix_nodes_offset = offset;
    offset += sizeof(RadixNode) * radix_node_capacity;

    offset = AlignUp(offset, alignof(RadixEdge));
    radix_edges_offset = offset;
    offset += sizeof(RadixEdge) * radix_edge_capacity;

    offset = AlignUp(offset, alignof(std::int32_t));
    radix_node_free_stack_offset = offset;
    offset += sizeof(std::int32_t) * radix_node_capacity;

    offset = AlignUp(offset, alignof(std::int32_t));
    radix_edge_free_stack_offset = offset;
    offset += sizeof(std::int32_t) * radix_edge_capacity;

    offset = AlignUp(offset, alignof(PrefixEntry));
    prefix_entries_offset = offset;
    offset += sizeof(PrefixEntry) * prefix_entry_capacity;

    offset = AlignUp(offset, alignof(PrefixPageRef));
    prefix_page_refs_offset = offset;
    offset += sizeof(PrefixPageRef) * prefix_page_ref_capacity;

    offset = AlignUp(offset, alignof(std::int32_t));
    prefix_entry_free_stack_offset = offset;
    offset += sizeof(std::int32_t) * prefix_entry_capacity;

    offset = AlignUp(offset, alignof(std::int32_t));
    prefix_page_ref_free_stack_offset = offset;
    offset += sizeof(std::int32_t) * prefix_page_ref_capacity;

    constexpr std::size_t kHugePageAlignment = 2 * 1024 * 1024;
    offset = AlignUp(offset, kHugePageAlignment);
    data_offset = offset;
    offset += data_bytes;

    total_bytes_unaligned = offset;
}

void HostPagedKVBackend::SharedState::MapPointers() {
    header = reinterpret_cast<SharedHeader*>(mapping + header_offset);
    free_stack = reinterpret_cast<std::int32_t*>(mapping + free_stack_offset);
    page_refcount =
        reinterpret_cast<std::uint32_t*>(mapping + page_refcount_offset);
    sequence_table =
        reinterpret_cast<SequenceEntry*>(mapping + sequence_table_offset);
    seq_page_nodes =
        reinterpret_cast<SeqPageNode*>(mapping + seq_page_nodes_offset);
    seq_page_node_free_stack = reinterpret_cast<std::int32_t*>(
        mapping + seq_page_node_free_stack_offset);

    radix_nodes = reinterpret_cast<RadixNode*>(mapping + radix_nodes_offset);
    radix_edges = reinterpret_cast<RadixEdge*>(mapping + radix_edges_offset);
    radix_node_free_stack =
        reinterpret_cast<std::int32_t*>(mapping + radix_node_free_stack_offset);
    radix_edge_free_stack =
        reinterpret_cast<std::int32_t*>(mapping + radix_edge_free_stack_offset);

    prefix_entries =
        reinterpret_cast<PrefixEntry*>(mapping + prefix_entries_offset);
    prefix_page_refs =
        reinterpret_cast<PrefixPageRef*>(mapping + prefix_page_refs_offset);
    prefix_entry_free_stack = reinterpret_cast<std::int32_t*>(
        mapping + prefix_entry_free_stack_offset);
    prefix_page_ref_free_stack = reinterpret_cast<std::int32_t*>(
        mapping + prefix_page_ref_free_stack_offset);

    data_base = mapping + data_offset;
}

void HostPagedKVBackend::SharedState::BindPrefixCache() {
    HostKVPrefixCacheParams params;
    params.enable_prefix_reuse = config.enable_prefix_reuse;
    params.prefix_min_reuse_pages = config.prefix_min_reuse_pages;
    params.prefix_min_store_pages = config.prefix_min_store_pages;
    params.prefix_page_budget = config.prefix_page_budget;

    HostKVPrefixCache::SharedFields shared_fields;
    shared_fields.radix_node_free_top = &header->radix_node_free_top;
    shared_fields.radix_edge_free_top = &header->radix_edge_free_top;
    shared_fields.prefix_entry_free_top = &header->prefix_entry_free_top;
    shared_fields.prefix_page_ref_free_top = &header->prefix_page_ref_free_top;
    shared_fields.prefix_entry_count = &header->prefix_entry_count;
    shared_fields.prefix_used_pages = &header->prefix_used_pages;
    shared_fields.prefix_access_epoch = &header->prefix_access_epoch;
    shared_fields.prefix_hit_count = &header->prefix_hit_count;
    shared_fields.prefix_miss_count = &header->prefix_miss_count;
    shared_fields.prefix_evict_count = &header->prefix_evict_count;
    shared_fields.lru_head = &header->lru_head;
    shared_fields.lru_tail = &header->lru_tail;

    prefix_cache_.Bind(
        params, radix_nodes, radix_edges, prefix_entries, prefix_page_refs,
        radix_node_free_stack, radix_edge_free_stack, prefix_entry_free_stack,
        prefix_page_ref_free_stack, shared_fields,
        [this](std::int32_t page_idx) { IncrementPageRefLocked(page_idx); },
        [this](std::int32_t page_idx) { DecrementPageRefLocked(page_idx); });
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
    header->has_v_cache = has_v_cache ? 1U : 0U;
    header->enable_prefix_reuse = config.enable_prefix_reuse ? 1U : 0U;

    header->sequence_page_node_capacity = sequence_page_node_capacity;
    header->radix_node_capacity = radix_node_capacity;
    header->radix_edge_capacity = radix_edge_capacity;
    header->prefix_entry_capacity = prefix_entry_capacity;
    header->prefix_page_ref_capacity = prefix_page_ref_capacity;
    header->prefix_page_budget = config.prefix_page_budget;

    header->free_stack_top.store(static_cast<std::uint32_t>(config.num_pages),
                                 std::memory_order_relaxed);
    header->seq_page_node_free_top.store(
        static_cast<std::uint32_t>(sequence_page_node_capacity),
        std::memory_order_relaxed);
    header->radix_node_free_top.store(
        static_cast<std::uint32_t>(radix_node_capacity > 0
                                       ? radix_node_capacity - 1
                                       : 0),
        std::memory_order_relaxed);
    header->radix_edge_free_top.store(
        static_cast<std::uint32_t>(radix_edge_capacity),
        std::memory_order_relaxed);
    header->prefix_entry_free_top.store(
        static_cast<std::uint32_t>(prefix_entry_capacity),
        std::memory_order_relaxed);
    header->prefix_page_ref_free_top.store(
        static_cast<std::uint32_t>(prefix_page_ref_capacity),
        std::memory_order_relaxed);

    header->active_sequences.store(0, std::memory_order_relaxed);
    header->prefix_entry_count.store(0, std::memory_order_relaxed);
    header->prefix_used_pages.store(0, std::memory_order_relaxed);
    header->prefix_access_epoch.store(0, std::memory_order_relaxed);
    header->prefix_hit_count.store(0, std::memory_order_relaxed);
    header->prefix_miss_count.store(0, std::memory_order_relaxed);
    header->prefix_evict_count.store(0, std::memory_order_relaxed);
    header->lru_head = kInvalidIndex;
    header->lru_tail = kInvalidIndex;

    for (std::size_t i = 0; i < config.num_pages; ++i) {
        free_stack[i] = static_cast<std::int32_t>(config.num_pages - 1 - i);
        page_refcount[i] = 0;
    }

    for (std::size_t i = 0; i < sequence_capacity; ++i) {
        sequence_table[i] = SequenceEntry();
    }

    for (std::size_t i = 0; i < sequence_page_node_capacity; ++i) {
        seq_page_nodes[i] = SeqPageNode();
        seq_page_node_free_stack[i] =
            static_cast<std::int32_t>(sequence_page_node_capacity - 1 - i);
    }

    BindPrefixCache();
    prefix_cache_.InitializePools(radix_node_capacity, radix_edge_capacity,
                                  prefix_entry_capacity,
                                  prefix_page_ref_capacity);

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

    if (const int rc = pthread_mutex_init(&header->metadata_mutex, &attr);
        rc != 0) {
        pthread_mutexattr_destroy(&attr);
        throw std::system_error(rc, std::generic_category(),
                                "pthread_mutex_init metadata_mutex failed");
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
    if (header->sequence_page_node_capacity != sequence_page_node_capacity) {
        throw std::runtime_error("Shared memory sequence_page_node_capacity mismatch");
    }
    if (header->radix_node_capacity != radix_node_capacity ||
        header->radix_edge_capacity != radix_edge_capacity) {
        throw std::runtime_error("Shared memory radix pool capacity mismatch");
    }
    if (header->prefix_entry_capacity != prefix_entry_capacity ||
        header->prefix_page_ref_capacity != prefix_page_ref_capacity) {
        throw std::runtime_error("Shared memory prefix pool capacity mismatch");
    }
    if (header->prefix_page_budget != config.prefix_page_budget) {
        throw std::runtime_error("Shared memory prefix_page_budget mismatch");
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
    std::int64_t sequence_id, bool* is_new, std::int64_t* previous_marker) {
    std::size_t index = HashSequenceId(sequence_id);
    SequenceEntry* first_tombstone = nullptr;
    for (std::size_t probe = 0; probe < sequence_capacity; ++probe) {
        SequenceEntry* entry = &sequence_table[index];
        if (entry->sequence_id == sequence_id) {
            if (is_new != nullptr) {
                *is_new = false;
            }
            if (previous_marker != nullptr) {
                *previous_marker = sequence_id;
            }
            return entry;
        }
        if (entry->sequence_id == kEmptySequenceId) {
            SequenceEntry* target =
                first_tombstone != nullptr ? first_tombstone : entry;
            if (previous_marker != nullptr) {
                *previous_marker = target->sequence_id;
            }
            *target = SequenceEntry();
            target->sequence_id = sequence_id;
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

std::int32_t HostPagedKVBackend::SharedState::PopStackIndexLocked(
    std::int32_t* stack, std::atomic<std::uint32_t>* top,
    const char* what) const {
    const std::uint32_t current = top->load(std::memory_order_relaxed);
    if (current == 0) {
        throw std::runtime_error(std::string("Out of ") + what);
    }
    const std::uint32_t next = current - 1;
    const std::int32_t value = stack[next];
    top->store(next, std::memory_order_relaxed);
    return value;
}

void HostPagedKVBackend::SharedState::PushStackIndexLocked(
    std::int32_t* stack, std::atomic<std::uint32_t>* top,
    std::int32_t value) const {
    const std::uint32_t current = top->load(std::memory_order_relaxed);
    stack[current] = value;
    top->store(current + 1, std::memory_order_relaxed);
}

std::int32_t HostPagedKVBackend::SharedState::AllocateSeqPageNodeLocked(
    std::int32_t page_idx) {
    const std::int32_t node_idx = PopStackIndexLocked(
        seq_page_node_free_stack, &header->seq_page_node_free_top,
        "sequence page nodes");
    seq_page_nodes[node_idx].page_idx = page_idx;
    seq_page_nodes[node_idx].next_node = kInvalidIndex;
    return node_idx;
}

void HostPagedKVBackend::SharedState::FreeSeqPageNodeLocked(
    std::int32_t node_idx) {
    seq_page_nodes[node_idx] = SeqPageNode();
    PushStackIndexLocked(seq_page_node_free_stack,
                         &header->seq_page_node_free_top, node_idx);
}

void HostPagedKVBackend::SharedState::AppendPageToSequenceLocked(
    SequenceEntry* entry, std::int32_t page_idx) {
    const std::int32_t node_idx = AllocateSeqPageNodeLocked(page_idx);
    if (entry->head_node == kInvalidIndex) {
        entry->head_node = node_idx;
        entry->tail_node = node_idx;
    } else {
        seq_page_nodes[entry->tail_node].next_node = node_idx;
        entry->tail_node = node_idx;
    }
    ++entry->num_pages;
}

std::vector<std::int32_t> HostPagedKVBackend::SharedState::CollectSequencePagesLocked(
    const SequenceEntry* entry, std::optional<std::size_t> max_pages) const {
    const std::size_t available_pages = entry->num_pages;
    const std::size_t limit = max_pages.has_value()
                                  ? std::min(max_pages.value(), available_pages)
                                  : available_pages;
    if (max_pages.has_value() && max_pages.value() > available_pages) {
        throw std::out_of_range(
            "Requested " + std::to_string(max_pages.value()) +
            " pages but only " + std::to_string(available_pages) +
            " pages allocated for sequence " +
            std::to_string(entry->sequence_id));
    }

    std::vector<std::int32_t> pages;
    pages.reserve(limit);
    std::int32_t node_idx = entry->head_node;
    while (node_idx != kInvalidIndex && pages.size() < limit) {
        pages.push_back(seq_page_nodes[node_idx].page_idx);
        node_idx = seq_page_nodes[node_idx].next_node;
    }
    return pages;
}

void HostPagedKVBackend::SharedState::ReleaseSequencePagesLocked(
    SequenceEntry* entry) {
    std::int32_t node_idx = entry->head_node;
    while (node_idx != kInvalidIndex) {
        const std::int32_t next = seq_page_nodes[node_idx].next_node;
        const std::int32_t page_idx = seq_page_nodes[node_idx].page_idx;
        DecrementPageRefLocked(page_idx);
        FreeSeqPageNodeLocked(node_idx);
        node_idx = next;
    }
    entry->num_pages = 0;
    entry->head_node = kInvalidIndex;
    entry->tail_node = kInvalidIndex;
}

std::int32_t HostPagedKVBackend::SharedState::PopFreePageLocked() {
    return PopStackIndexLocked(free_stack, &header->free_stack_top,
                               "free pages");
}

void HostPagedKVBackend::SharedState::PushFreePageLocked(std::int32_t page_idx) {
    PushStackIndexLocked(free_stack, &header->free_stack_top, page_idx);
}

void HostPagedKVBackend::SharedState::IncrementPageRefLocked(
    std::int32_t page_idx) {
    ++page_refcount[page_idx];
}

void HostPagedKVBackend::SharedState::DecrementPageRefLocked(
    std::int32_t page_idx) {
    if (page_refcount[page_idx] == 0) {
        throw std::runtime_error("page_refcount underflow on page " +
                                 std::to_string(page_idx));
    }
    --page_refcount[page_idx];
    if (page_refcount[page_idx] == 0) {
        PushFreePageLocked(page_idx);
    }
}

void HostPagedKVBackend::SharedState::Initialize(bool create_region) {
    const std::size_t page_size = GetSystemPageSize();
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
                static_cast<std::uint32_t>(InitState::kInitializing),
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

    void* mapped = mmap_aligned(total_bytes, PROT_READ | PROT_WRITE, MAP_SHARED,
                                shm_fd, 0, alignment);
    if (mapped == MAP_FAILED) {
        const int err = errno;
        close(shm_fd);
        shm_fd = -1;
        throw std::system_error(err, std::generic_category(), "mmap failed");
    }

    madvise(mapped, total_bytes, MADV_HUGEPAGE);

    mapping = static_cast<std::byte*>(mapped);
    MapPointers();
    BindPrefixCache();

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

    std::vector<std::int32_t> pages;
    pages.reserve(num_pages);

    ScopedMutexLock lock(&header->metadata_mutex);

    if (header->free_stack_top.load(std::memory_order_relaxed) < num_pages) {
        throw std::runtime_error(
            "Insufficient free pages available for sequence " +
            std::to_string(sequence_id) + " (requested=" +
            std::to_string(num_pages) + ", available=" +
            std::to_string(header->free_stack_top.load(std::memory_order_relaxed)) +
            ")");
    }

    if (header->seq_page_node_free_top.load(std::memory_order_relaxed) < num_pages) {
        throw std::runtime_error("Insufficient sequence page nodes");
    }

    bool is_new = false;
    SequenceEntry* entry =
        FindOrInsertSequenceEntryLocked(sequence_id, &is_new, nullptr);
    if (is_new) {
        header->active_sequences.fetch_add(1, std::memory_order_relaxed);
    }

    for (std::size_t i = 0; i < num_pages; ++i) {
        const std::int32_t page_idx = PopFreePageLocked();
        IncrementPageRefLocked(page_idx);
        AppendPageToSequenceLocked(entry, page_idx);
        pages.push_back(page_idx);
    }

    return pages;
}

PrefixAllocationBatchResult
HostPagedKVBackend::SharedState::AcquirePagesForSequencesWithPrefix(
    const std::vector<std::int64_t>& sequence_ids,
    const std::vector<std::size_t>& num_tokens,
    const std::vector<std::int32_t>& flat_prompt_tokens,
    const std::vector<std::size_t>& prompt_offsets) {
    if (sequence_ids.size() != num_tokens.size()) {
        throw std::invalid_argument(
            "sequence_ids and num_tokens must have the same length");
    }
    if (prompt_offsets.size() != sequence_ids.size() + 1) {
        throw std::invalid_argument(
            "prompt_offsets must contain sequence_count + 1 entries");
    }
    if (prompt_offsets.empty() || prompt_offsets.front() != 0 ||
        prompt_offsets.back() != flat_prompt_tokens.size()) {
        throw std::invalid_argument("prompt_offsets boundaries are invalid");
    }

    PrefixAllocationBatchResult result;
    result.allocated_pages.resize(sequence_ids.size());
    result.reused_prefix_tokens.resize(sequence_ids.size(), 0);

    struct SequenceRollbackRecord {
        SequenceEntry* entry = nullptr;
        std::int64_t previous_marker = kEmptySequenceId;
        std::uint32_t previous_num_pages = 0;
        std::int32_t previous_head_node = kInvalidIndex;
        std::int32_t previous_tail_node = kInvalidIndex;
        bool was_new = false;
    };

    ScopedMutexLock lock(&header->metadata_mutex);
    std::vector<SequenceRollbackRecord> rollback_records;
    rollback_records.reserve(sequence_ids.size());

    try {
        for (std::size_t i = 0; i < sequence_ids.size(); ++i) {
            const std::size_t begin = prompt_offsets[i];
            const std::size_t end = prompt_offsets[i + 1];
            if (begin > end || end > flat_prompt_tokens.size()) {
                throw std::invalid_argument("prompt_offsets contain invalid slice");
            }
            if (num_tokens[i] == 0) {
                throw std::invalid_argument("num_tokens entries must be > 0");
            }

            const std::size_t required_pages =
                (num_tokens[i] + config.page_size_tokens - 1) /
                config.page_size_tokens;
            const std::size_t prompt_tokens = end - begin;
            const std::size_t prompt_full_pages =
                std::min(required_pages, prompt_tokens / config.page_size_tokens);

            std::size_t reused_pages = 0;
            std::vector<std::int32_t> reused_page_candidates;
            if (config.enable_prefix_reuse && prompt_full_pages > 0) {
                auto lookup = prefix_cache_.LookupPrefixPagesLocked(
                    flat_prompt_tokens.data() + begin,
                    prompt_full_pages * config.page_size_tokens,
                    prompt_full_pages);
                reused_pages =
                    std::min<std::size_t>(lookup.reused_pages, prompt_full_pages);
                reused_page_candidates = std::move(lookup.pages);
                if (reused_page_candidates.size() < reused_pages) {
                    throw std::runtime_error(
                        "Prefix lookup returned fewer pages than expected");
                }
            }

            const std::size_t new_pages = required_pages - reused_pages;
            if (header->free_stack_top.load(std::memory_order_relaxed) <
                new_pages) {
                throw std::runtime_error(
                    "Insufficient free pages for prefix-aware allocation of "
                    "sequence " +
                    std::to_string(sequence_ids[i]));
            }
            if (header->seq_page_node_free_top.load(std::memory_order_relaxed) <
                required_pages) {
                throw std::runtime_error(
                    "Insufficient sequence page nodes for prefix-aware allocation");
            }

            bool is_new = false;
            std::int64_t previous_marker = kEmptySequenceId;
            SequenceEntry* entry = FindOrInsertSequenceEntryLocked(
                sequence_ids[i], &is_new, &previous_marker);

            SequenceRollbackRecord rollback_record;
            rollback_record.entry = entry;
            rollback_record.previous_marker = previous_marker;
            rollback_record.previous_num_pages = entry->num_pages;
            rollback_record.previous_head_node = entry->head_node;
            rollback_record.previous_tail_node = entry->tail_node;
            rollback_record.was_new = is_new;
            rollback_records.push_back(rollback_record);

            if (is_new) {
                header->active_sequences.fetch_add(1, std::memory_order_relaxed);
            }

            std::vector<std::int32_t> pages;
            pages.reserve(required_pages);

            for (std::size_t j = 0; j < reused_pages; ++j) {
                const std::int32_t page_idx = reused_page_candidates[j];
                IncrementPageRefLocked(page_idx);
                AppendPageToSequenceLocked(entry, page_idx);
                pages.push_back(page_idx);
            }

            for (std::size_t j = 0; j < new_pages; ++j) {
                const std::int32_t page_idx = PopFreePageLocked();
                IncrementPageRefLocked(page_idx);
                AppendPageToSequenceLocked(entry, page_idx);
                pages.push_back(page_idx);
            }

            result.allocated_pages[i] = std::move(pages);
            result.reused_prefix_tokens[i] = reused_pages * config.page_size_tokens;
        }
    } catch (...) {
        for (auto it = rollback_records.rbegin(); it != rollback_records.rend();
             ++it) {
            SequenceRollbackRecord& record = *it;
            SequenceEntry* entry = record.entry;
            if (entry == nullptr) {
                continue;
            }

            std::int32_t first_added_node = kInvalidIndex;
            if (entry->num_pages > record.previous_num_pages) {
                if (record.previous_num_pages == 0) {
                    first_added_node = entry->head_node;
                } else {
                    first_added_node =
                        seq_page_nodes[record.previous_tail_node].next_node;
                }
            }

            std::int32_t node_idx = first_added_node;
            while (node_idx != kInvalidIndex) {
                const std::int32_t next_node = seq_page_nodes[node_idx].next_node;
                const std::int32_t page_idx = seq_page_nodes[node_idx].page_idx;
                DecrementPageRefLocked(page_idx);
                FreeSeqPageNodeLocked(node_idx);
                node_idx = next_node;
            }

            if (record.previous_tail_node != kInvalidIndex) {
                seq_page_nodes[record.previous_tail_node].next_node =
                    kInvalidIndex;
            }

            entry->num_pages = record.previous_num_pages;
            entry->head_node = record.previous_head_node;
            entry->tail_node = record.previous_tail_node;
            entry->sequence_id = record.previous_marker;

            if (record.was_new) {
                header->active_sequences.fetch_sub(1, std::memory_order_relaxed);
            }
        }
        throw;
    }

    return result;
}

void HostPagedKVBackend::SharedState::ReleaseSequence(
    std::int64_t sequence_id) {
    ScopedMutexLock lock(&header->metadata_mutex);
    SequenceEntry* entry = FindSequenceEntryLocked(sequence_id);
    if (entry == nullptr) {
        throw std::out_of_range("Sequence ID " + std::to_string(sequence_id) +
                                " not found during release");
    }

    ReleaseSequencePagesLocked(entry);
    entry->sequence_id = kTombstoneSequenceId;
    header->active_sequences.fetch_sub(1, std::memory_order_relaxed);
}

std::vector<std::int32_t> HostPagedKVBackend::SharedState::SequencePages(
    std::int64_t sequence_id, std::optional<std::size_t> max_pages) const {
    ScopedMutexLock lock(&header->metadata_mutex);
    SequenceEntry* entry = FindSequenceEntryLocked(sequence_id);
    if (entry == nullptr) {
        throw std::out_of_range("Sequence ID " + std::to_string(sequence_id) +
                                " not found when fetching pages");
    }
    return CollectSequencePagesLocked(entry, max_pages);
}

void HostPagedKVBackend::SharedState::CommitSequencePrefix(
    std::int64_t sequence_id, const std::vector<std::int32_t>& prompt_tokens,
    std::size_t prompt_token_count) {
    if (!config.enable_prefix_reuse || prompt_tokens.empty()) {
        return;
    }

    const std::size_t bounded_prompt_tokens =
        std::min(prompt_token_count, prompt_tokens.size());
    const std::size_t full_prompt_pages =
        bounded_prompt_tokens / config.page_size_tokens;
    if (full_prompt_pages < config.prefix_min_store_pages) {
        return;
    }

    try {
        ScopedMutexLock lock(&header->metadata_mutex);
        SequenceEntry* entry = FindSequenceEntryLocked(sequence_id);
        if (entry == nullptr || entry->num_pages == 0) {
            return;
        }

        const std::size_t storable_pages =
            std::min<std::size_t>(entry->num_pages, full_prompt_pages);
        if (storable_pages < config.prefix_min_store_pages) {
            return;
        }

        const std::size_t token_count = storable_pages * config.page_size_tokens;
        auto pages = CollectSequencePagesLocked(entry, storable_pages);
        if (pages.size() != storable_pages) {
            return;
        }
        prefix_cache_.CommitPrefixLocked(prompt_tokens.data(), token_count, pages);
    } catch (const std::exception&) {
        // Prefix commit is opportunistic. Ignore insertion failures and keep the
        // inference flow unaffected.
        return;
    }
}

HostPagedKVStats HostPagedKVBackend::SharedState::CollectStats() const {
    HostPagedKVStats stats;
    stats.num_total_pages = config.num_pages;
    stats.num_free_pages =
        header->free_stack_top.load(std::memory_order_relaxed);
    stats.num_used_pages = stats.num_total_pages - stats.num_free_pages;
    stats.num_active_sequences =
        header->active_sequences.load(std::memory_order_relaxed);
    stats.sequence_table_capacity = sequence_capacity;
    stats.total_bytes = total_bytes;

    stats.num_prefix_entries =
        header->prefix_entry_count.load(std::memory_order_relaxed);
    stats.num_prefix_hits =
        header->prefix_hit_count.load(std::memory_order_relaxed);
    stats.num_prefix_misses =
        header->prefix_miss_count.load(std::memory_order_relaxed);
    stats.num_prefix_evictions =
        header->prefix_evict_count.load(std::memory_order_relaxed);
    stats.num_cache_entry_pages =
        header->prefix_used_pages.load(std::memory_order_relaxed);

    std::size_t shared_pages = 0;
    {
        ScopedMutexLock lock(&header->metadata_mutex);
        for (std::size_t i = 0; i < config.num_pages; ++i) {
            if (page_refcount[i] > 1) {
                ++shared_pages;
            }
        }
    }
    stats.num_shared_pages = shared_pages;

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
    std::vector<std::vector<std::int32_t>> allocations;
    allocations.reserve(sequence_ids.size());

    for (std::size_t i = 0; i < sequence_ids.size(); ++i) {
        if (num_tokens[i] == 0) {
            throw std::invalid_argument(
                "num_tokens must be greater than zero for sequence " +
                std::to_string(sequence_ids[i]));
        }
        const std::size_t required_pages =
            (num_tokens[i] + config_.page_size_tokens - 1) /
            config_.page_size_tokens;
        allocations.emplace_back(state_->AcquirePages(sequence_ids[i], required_pages));
    }
    return allocations;
}

PrefixAllocationBatchResult HostPagedKVBackend::AcquirePagesForSequencesWithPrefix(
    const std::vector<std::int64_t>& sequence_ids,
    const std::vector<std::size_t>& num_tokens,
    const std::vector<std::int32_t>& flat_prompt_tokens,
    const std::vector<std::size_t>& prompt_offsets) {
    return state_->AcquirePagesForSequencesWithPrefix(
        sequence_ids, num_tokens, flat_prompt_tokens, prompt_offsets);
}

void HostPagedKVBackend::ReleaseSequence(std::int64_t sequence_id) {
    state_->ReleaseSequence(sequence_id);
}

void HostPagedKVBackend::ReleaseSequences(
    const std::vector<std::int64_t>& sequence_ids) {
    for (std::int64_t sequence_id : sequence_ids) {
        state_->ReleaseSequence(sequence_id);
    }
}

std::vector<std::int32_t> HostPagedKVBackend::SequencePages(
    std::int64_t sequence_id, std::optional<std::size_t> max_pages) const {
    return state_->SequencePages(sequence_id, max_pages);
}

void HostPagedKVBackend::CommitSequencePrefix(
    std::int64_t sequence_id, const std::vector<std::int32_t>& prompt_tokens,
    std::size_t prompt_token_count) {
    state_->CommitSequencePrefix(sequence_id, prompt_tokens, prompt_token_count);
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
