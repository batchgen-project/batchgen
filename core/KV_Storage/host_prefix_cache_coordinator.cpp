#include "host_prefix_cache_coordinator.h"

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
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <unordered_map>
#include <utility>

namespace batchgen::kv {

namespace {

constexpr std::uint64_t kPrefixCacheMagic = 0x484f535450434348ULL;
constexpr std::uint32_t kPrefixCacheAbiVersion = 1;

enum class InitState : std::uint32_t {
    kUninitialized = 0,
    kInitializing = 1,
    kReady = 2,
};

enum class EntryState : std::uint32_t {
    kEmpty = 0,
    kResident = 1,
    kTombstone = 2,
};

struct SharedHeader {
    std::atomic<std::uint32_t> init_state{
        static_cast<std::uint32_t>(InitState::kUninitialized)};
    std::uint64_t magic = kPrefixCacheMagic;
    std::uint32_t abi_version = kPrefixCacheAbiVersion;
    std::uint64_t create_time_ns = 0;

    std::uint32_t group_count = 0;
    std::uint32_t hash_block_tokens = 0;
    std::uint32_t commit_boundary_tokens = 0;
    std::uint32_t max_nodes = 0;
    std::uint32_t max_group_entries = 0;
    std::uint32_t max_page_handles = 0;
    std::uint32_t max_attachments = 0;

    std::atomic<std::uint32_t> next_group_entry{0};
    std::atomic<std::uint32_t> next_page_handle{0};
    std::atomic<std::uint64_t> next_attachment_handle{1};
    std::atomic<std::uint64_t> global_epoch{0};
    std::atomic<std::uint64_t> lookup_hits{0};
    std::atomic<std::uint64_t> lookup_misses{0};
    pthread_mutex_t mutex{};
};

struct SharedGroupSpec {
    std::uint32_t group_id = 0;
    std::uint32_t semantic = 0;
    std::uint32_t required_for_reuse = 0;
    std::uint32_t raw_page_tokens = 0;
    std::uint32_t compression_ratio = 1;
};

struct SharedPrefixNode {
    std::uint32_t state = static_cast<std::uint32_t>(EntryState::kEmpty);
    PrefixDigest digest{};
    std::uint32_t raw_end_token = 0;
    std::uint32_t first_group_entry = 0;
    std::uint32_t group_entry_count = 0;
    std::uint64_t last_access_epoch = 0;
};

struct SharedGroupEntry {
    std::uint32_t state = static_cast<std::uint32_t>(EntryState::kEmpty);
    std::uint32_t group_id = 0;
    std::uint32_t raw_end_token = 0;
    std::uint32_t first_page_handle = 0;
    std::uint32_t page_handle_count = 0;
    std::atomic<std::uint32_t> active_ref_count{0};
    std::atomic<std::uint32_t> pending_load_count{0};
};

struct SharedPageHandle {
    std::uint32_t host_region_id = 0;
    std::uint32_t page_id = 0;
};

struct SharedAttachment {
    std::uint32_t state = static_cast<std::uint32_t>(EntryState::kEmpty);
    std::uint64_t attachment_handle = 0;
    std::uint32_t node_index = 0;
    std::uint32_t pending_load_count = 0;
    std::uint32_t release_requested = 0;
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

std::size_t SystemPageSize() {
    const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) {
        throw std::system_error(errno, std::generic_category(),
                                "sysconf(_SC_PAGESIZE) failed");
    }
    return static_cast<std::size_t>(page_size);
}

std::uint64_t NowNs() {
    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
}

std::uint64_t SplitMix64(std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

PrefixDigest HashPrefixBlock(PrefixDigest namespace_digest,
                             PrefixDigest parent_digest,
                             const std::int64_t* tokens,
                             std::uint32_t token_count) {
    PrefixDigest state{
        0x243f6a8885a308d3ULL,
        0x13198a2e03707344ULL,
        0xa4093822299f31d0ULL,
        0x082efa98ec4e6c89ULL,
    };
    for (std::size_t lane = 0; lane < state.size(); ++lane) {
        state[lane] ^= SplitMix64(namespace_digest[lane]);
        state[lane] ^= SplitMix64(parent_digest[lane] + lane);
    }
    for (std::uint32_t idx = 0; idx < token_count; ++idx) {
        const auto token = static_cast<std::uint64_t>(tokens[idx]);
        const std::size_t lane = idx % state.size();
        state[lane] = SplitMix64(state[lane] ^ token ^
                                 (static_cast<std::uint64_t>(idx) << 32));
        state[(lane + 1) % state.size()] ^= state[lane];
    }
    for (std::size_t lane = 0; lane < state.size(); ++lane) {
        state[lane] = SplitMix64(state[lane] ^ token_count ^ lane);
    }
    return state;
}

bool DigestEquals(const PrefixDigest& lhs, const PrefixDigest& rhs) {
    return lhs == rhs;
}

void ResetGroupEntry(SharedGroupEntry& entry) {
    entry.state = static_cast<std::uint32_t>(EntryState::kEmpty);
    entry.group_id = 0;
    entry.raw_end_token = 0;
    entry.first_page_handle = 0;
    entry.page_handle_count = 0;
    entry.active_ref_count.store(0, std::memory_order_relaxed);
    entry.pending_load_count.store(0, std::memory_order_relaxed);
}

std::uint32_t Gcd(std::uint32_t lhs, std::uint32_t rhs) {
    return static_cast<std::uint32_t>(std::gcd(lhs, rhs));
}

std::uint32_t Lcm(std::uint32_t lhs, std::uint32_t rhs) {
    if (lhs == 0 || rhs == 0) {
        return 0;
    }
    return static_cast<std::uint32_t>(std::lcm(lhs, rhs));
}

void ValidateGroupSpec(const HostKVGroupSpec& spec) {
    if (spec.raw_page_tokens == 0) {
        throw std::invalid_argument("HostKVGroupSpec.raw_page_tokens must be > 0");
    }
    if (spec.compression_ratio == 0) {
        throw std::invalid_argument(
            "HostKVGroupSpec.compression_ratio must be > 0");
    }
}

std::uint32_t ComputeHashBlockTokens(
    const std::vector<HostKVGroupSpec>& group_specs) {
    std::uint32_t result = 0;
    for (const auto& spec : group_specs) {
        if (!spec.required_for_reuse) {
            continue;
        }
        result = result == 0 ? spec.raw_page_tokens
                             : Gcd(result, spec.raw_page_tokens);
    }
    return result;
}

std::uint32_t ComputeCommitBoundaryTokens(
    const std::vector<HostKVGroupSpec>& group_specs) {
    std::uint32_t result = 1;
    bool has_required_group = false;
    for (const auto& spec : group_specs) {
        if (!spec.required_for_reuse) {
            continue;
        }
        has_required_group = true;
        result = Lcm(result, spec.raw_page_tokens);
    }
    return has_required_group ? result : 0;
}

class ScopedMutexLock {
   public:
    explicit ScopedMutexLock(pthread_mutex_t* mutex) : mutex_(mutex) {
        const int rc = pthread_mutex_lock(mutex_);
        if (rc == EOWNERDEAD) {
            const int consistent_rc = pthread_mutex_consistent(mutex_);
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
        const int rc = pthread_mutex_unlock(mutex_);
        if (rc != 0) {
            std::terminate();
        }
    }

   private:
    pthread_mutex_t* mutex_;
};

}  // namespace

std::string ToString(const HostKVGroupSpec& spec) {
    std::ostringstream oss;
    oss << "HostKVGroupSpec(group_id=" << spec.group_id
        << ", semantic=" << static_cast<std::uint32_t>(spec.semantic)
        << ", required_for_reuse=" << spec.required_for_reuse
        << ", raw_page_tokens=" << spec.raw_page_tokens
        << ", compression_ratio=" << spec.compression_ratio << ")";
    return oss.str();
}

std::string ToString(const HostPrefixCacheStats& stats) {
    std::ostringstream oss;
    oss << "HostPrefixCacheStats(resident_nodes=" << stats.resident_nodes
        << ", active_attachments=" << stats.active_attachments
        << ", used_group_entries=" << stats.used_group_entries
        << ", used_page_handles=" << stats.used_page_handles
        << ", lookup_hits=" << stats.lookup_hits
        << ", lookup_misses=" << stats.lookup_misses << ")";
    return oss.str();
}

std::vector<std::pair<std::uint32_t, PrefixDigest>> BuildPrefixHashChain(
    PrefixDigest namespace_digest, const std::vector<std::int64_t>& token_ids,
    std::uint32_t block_tokens) {
    if (block_tokens == 0) {
        throw std::invalid_argument("block_tokens must be > 0");
    }
    std::vector<std::pair<std::uint32_t, PrefixDigest>> chain;
    const std::uint32_t full_tokens =
        static_cast<std::uint32_t>(token_ids.size()) -
        (static_cast<std::uint32_t>(token_ids.size()) % block_tokens);
    chain.reserve(full_tokens / block_tokens);
    PrefixDigest parent_digest{};
    for (std::uint32_t start = 0; start < full_tokens; start += block_tokens) {
        parent_digest = HashPrefixBlock(namespace_digest, parent_digest,
                                        token_ids.data() + start, block_tokens);
        chain.emplace_back(start + block_tokens, parent_digest);
    }
    return chain;
}

struct HostPrefixCacheCoordinator::SharedState {
    explicit SharedState(HostPrefixCacheConfig cfg,
                         std::uint32_t hash_block_tokens,
                         std::uint32_t commit_boundary_tokens)
        : config(std::move(cfg)),
          hash_block_tokens(hash_block_tokens),
          commit_boundary_tokens(commit_boundary_tokens) {
        ComputeOffsets();
    }

    void Initialize(bool create_region);
    PrefixCommitResult CommitPrefixPages(
        PrefixDigest namespace_digest,
        const std::vector<std::int64_t>& token_ids,
        std::uint32_t commit_tokens,
        const std::vector<GroupCommitPages>& group_pages);
    PrefixLookupResult LookupAndAttach(
        PrefixDigest namespace_digest,
        const std::vector<std::int64_t>& token_ids);
    PrefixLookupResult EstimateLookup(
        PrefixDigest namespace_digest,
        const std::vector<std::int64_t>& token_ids);
    void ReleaseAttachment(std::uint64_t attachment_handle);
    void BeginAttachmentLoad(std::uint64_t attachment_handle);
    void EndAttachmentLoad(std::uint64_t attachment_handle);
    PrefixEvictionResult EvictUntilFree(
        std::uint32_t min_free_nodes,
        std::uint32_t min_free_group_entries,
        std::uint32_t min_free_page_handles,
        std::uint32_t max_scan_nodes);
    PrefixEvictionResult ClearUnprotected();
    HostPrefixCacheStats GetStats() const;

    HostPrefixCacheConfig config;
    std::uint32_t hash_block_tokens = 0;
    std::uint32_t commit_boundary_tokens = 0;

    int shm_fd = -1;
    std::size_t total_bytes = 0;
    std::byte* mapping = nullptr;
    SharedHeader* header = nullptr;
    SharedGroupSpec* group_specs = nullptr;
    SharedPrefixNode* nodes = nullptr;
    SharedGroupEntry* group_entries = nullptr;
    SharedPageHandle* page_handles = nullptr;
    SharedAttachment* attachments = nullptr;

    std::size_t header_offset = 0;
    std::size_t group_spec_offset = 0;
    std::size_t node_offset = 0;
    std::size_t group_entry_offset = 0;
    std::size_t page_handle_offset = 0;
    std::size_t attachment_offset = 0;
    std::size_t total_bytes_unaligned = 0;

   private:
    void ComputeOffsets();
    void MapPointers();
    void ConstructSharedState();
    void WaitForInitialization() const;
    void ValidateSharedState() const;
    std::optional<std::uint32_t> FindNodeLocked(
        const PrefixDigest& digest) const;
    std::uint32_t AllocateNodeLocked();
    std::uint32_t AllocateAttachmentLocked();
    bool NodeHasRequiredGroupsLocked(const SharedPrefixNode& node) const;
    std::vector<GroupMaterializationSpan> BuildMaterializationSpansLocked(
        const SharedPrefixNode& node) const;
    std::uint64_t AttachNodeLocked(std::uint32_t node_index);
    std::uint32_t CountFreeNodeSlotsLocked() const;
    bool NodeIsProtectedLocked(const SharedPrefixNode& node) const;
    SharedAttachment* FindAttachmentLocked(std::uint64_t attachment_handle);
    void UpdateAttachmentLoadRefsLocked(SharedAttachment* attachment,
                                        int delta);
    void FinalizeAttachmentReleaseLocked(SharedAttachment* attachment);
    void AppendEvictedPagesLocked(const SharedPrefixNode& node,
                                  PrefixEvictionResult* result) const;
    bool ResidentNodeReferencesPageLocked(std::uint32_t group_id,
                                          const HostPageHandle& page) const;
    void FilterEvictedPagesStillReferencedLocked(
        PrefixEvictionResult* result) const;
    void CompactArenasLocked();
};

void HostPrefixCacheCoordinator::SharedState::ComputeOffsets() {
    std::size_t offset = 0;
    offset = AlignUp(offset, alignof(SharedHeader));
    header_offset = offset;
    offset += sizeof(SharedHeader);

    offset = AlignUp(offset, alignof(SharedGroupSpec));
    group_spec_offset = offset;
    offset += sizeof(SharedGroupSpec) * config.group_specs.size();

    offset = AlignUp(offset, alignof(SharedPrefixNode));
    node_offset = offset;
    offset += sizeof(SharedPrefixNode) * config.max_nodes;

    offset = AlignUp(offset, alignof(SharedGroupEntry));
    group_entry_offset = offset;
    offset += sizeof(SharedGroupEntry) * config.max_group_entries;

    offset = AlignUp(offset, alignof(SharedPageHandle));
    page_handle_offset = offset;
    offset += sizeof(SharedPageHandle) * config.max_page_handles;

    offset = AlignUp(offset, alignof(SharedAttachment));
    attachment_offset = offset;
    offset += sizeof(SharedAttachment) * config.max_attachments;

    total_bytes_unaligned = offset;
}

void HostPrefixCacheCoordinator::SharedState::MapPointers() {
    header = reinterpret_cast<SharedHeader*>(mapping + header_offset);
    group_specs =
        reinterpret_cast<SharedGroupSpec*>(mapping + group_spec_offset);
    nodes = reinterpret_cast<SharedPrefixNode*>(mapping + node_offset);
    group_entries =
        reinterpret_cast<SharedGroupEntry*>(mapping + group_entry_offset);
    page_handles =
        reinterpret_cast<SharedPageHandle*>(mapping + page_handle_offset);
    attachments =
        reinterpret_cast<SharedAttachment*>(mapping + attachment_offset);
}

void HostPrefixCacheCoordinator::SharedState::ConstructSharedState() {
    std::memset(mapping, 0, total_bytes);
    MapPointers();
    header->magic = kPrefixCacheMagic;
    header->abi_version = kPrefixCacheAbiVersion;
    header->create_time_ns = NowNs();
    header->group_count = static_cast<std::uint32_t>(config.group_specs.size());
    header->hash_block_tokens = hash_block_tokens;
    header->commit_boundary_tokens = commit_boundary_tokens;
    header->max_nodes = config.max_nodes;
    header->max_group_entries = config.max_group_entries;
    header->max_page_handles = config.max_page_handles;
    header->max_attachments = config.max_attachments;
    header->next_group_entry.store(0, std::memory_order_relaxed);
    header->next_page_handle.store(0, std::memory_order_relaxed);
    header->next_attachment_handle.store(1, std::memory_order_relaxed);
    header->global_epoch.store(0, std::memory_order_relaxed);
    header->lookup_hits.store(0, std::memory_order_relaxed);
    header->lookup_misses.store(0, std::memory_order_relaxed);

    for (std::size_t i = 0; i < config.group_specs.size(); ++i) {
        const HostKVGroupSpec& spec = config.group_specs[i];
        group_specs[i].group_id = spec.group_id;
        group_specs[i].semantic = static_cast<std::uint32_t>(spec.semantic);
        group_specs[i].required_for_reuse = spec.required_for_reuse ? 1 : 0;
        group_specs[i].raw_page_tokens = spec.raw_page_tokens;
        group_specs[i].compression_ratio = spec.compression_ratio;
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
    if (const int rc = pthread_mutex_init(&header->mutex, &attr); rc != 0) {
        pthread_mutexattr_destroy(&attr);
        throw std::system_error(rc, std::generic_category(),
                                "pthread_mutex_init failed");
    }
    pthread_mutexattr_destroy(&attr);
    header->init_state.store(static_cast<std::uint32_t>(InitState::kReady),
                             std::memory_order_release);
}

void HostPrefixCacheCoordinator::SharedState::WaitForInitialization() const {
    while (true) {
        const auto state = static_cast<InitState>(
            header->init_state.load(std::memory_order_acquire));
        if (state == InitState::kReady) {
            return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void HostPrefixCacheCoordinator::SharedState::ValidateSharedState() const {
    if (header->magic != kPrefixCacheMagic) {
        throw std::runtime_error("Host prefix cache shared memory magic mismatch");
    }
    if (header->abi_version != kPrefixCacheAbiVersion) {
        throw std::runtime_error("Host prefix cache ABI version mismatch");
    }
    if (header->group_count != config.group_specs.size()) {
        throw std::runtime_error("Host prefix cache group count mismatch");
    }
    if (header->hash_block_tokens != hash_block_tokens ||
        header->commit_boundary_tokens != commit_boundary_tokens ||
        header->max_nodes != config.max_nodes ||
        header->max_group_entries != config.max_group_entries ||
        header->max_page_handles != config.max_page_handles ||
        header->max_attachments != config.max_attachments) {
        throw std::runtime_error("Host prefix cache config mismatch");
    }
    for (std::size_t i = 0; i < config.group_specs.size(); ++i) {
        const HostKVGroupSpec& expected = config.group_specs[i];
        const SharedGroupSpec& actual = group_specs[i];
        if (actual.group_id != expected.group_id ||
            actual.semantic != static_cast<std::uint32_t>(expected.semantic) ||
            actual.required_for_reuse !=
                (expected.required_for_reuse ? 1U : 0U) ||
            actual.raw_page_tokens != expected.raw_page_tokens ||
            actual.compression_ratio != expected.compression_ratio) {
            throw std::runtime_error("Host prefix cache group spec mismatch");
        }
    }
}

void HostPrefixCacheCoordinator::SharedState::Initialize(bool create_region) {
    const std::size_t page_size = SystemPageSize();
    total_bytes = AlignUp(total_bytes_unaligned, page_size);
    int flags = O_RDWR;
    if (create_region) {
        flags |= O_CREAT;
    }
    shm_fd = shm_open(config.shm_name.c_str(), flags, 0660);
    if (shm_fd == -1) {
        throw std::system_error(errno, std::generic_category(),
                                "host prefix cache shm_open failed");
    }
    if (create_region) {
        if (ftruncate(shm_fd, static_cast<off_t>(total_bytes)) == -1) {
            const int err = errno;
            close(shm_fd);
            shm_fd = -1;
            throw std::system_error(err, std::generic_category(),
                                    "host prefix cache ftruncate failed");
        }
    } else {
        struct stat stat_buffer {};
        if (fstat(shm_fd, &stat_buffer) == -1) {
            const int err = errno;
            close(shm_fd);
            shm_fd = -1;
            throw std::system_error(err, std::generic_category(),
                                    "host prefix cache fstat failed");
        }
        if (static_cast<std::size_t>(stat_buffer.st_size) < total_bytes) {
            close(shm_fd);
            shm_fd = -1;
            throw std::runtime_error(
                "host prefix cache shared memory segment is too small");
        }
    }

    void* mapped = mmap(nullptr, total_bytes, PROT_READ | PROT_WRITE,
                        MAP_SHARED, shm_fd, 0);
    if (mapped == MAP_FAILED) {
        const int err = errno;
        close(shm_fd);
        shm_fd = -1;
        throw std::system_error(err, std::generic_category(),
                                "host prefix cache mmap failed");
    }
    mapping = static_cast<std::byte*>(mapped);
    MapPointers();

    if (create_region) {
        header->init_state.store(
            static_cast<std::uint32_t>(InitState::kInitializing),
            std::memory_order_relaxed);
        ConstructSharedState();
    } else {
        WaitForInitialization();
        ValidateSharedState();
    }
}

std::optional<std::uint32_t>
HostPrefixCacheCoordinator::SharedState::FindNodeLocked(
    const PrefixDigest& digest) const {
    for (std::uint32_t index = 0; index < config.max_nodes; ++index) {
        const SharedPrefixNode& node = nodes[index];
        if (node.state == static_cast<std::uint32_t>(EntryState::kResident) &&
            DigestEquals(node.digest, digest)) {
            return index;
        }
    }
    return std::nullopt;
}

std::uint32_t HostPrefixCacheCoordinator::SharedState::AllocateNodeLocked() {
    for (std::uint32_t index = 0; index < config.max_nodes; ++index) {
        SharedPrefixNode& node = nodes[index];
        if (node.state == static_cast<std::uint32_t>(EntryState::kEmpty) ||
            node.state == static_cast<std::uint32_t>(EntryState::kTombstone)) {
            node = SharedPrefixNode();
            return index;
        }
    }
    throw std::runtime_error("Host prefix cache node table is full");
}

std::uint32_t
HostPrefixCacheCoordinator::SharedState::AllocateAttachmentLocked() {
    for (std::uint32_t index = 0; index < config.max_attachments; ++index) {
        SharedAttachment& attachment = attachments[index];
        if (attachment.state ==
                static_cast<std::uint32_t>(EntryState::kEmpty) ||
            attachment.state ==
                static_cast<std::uint32_t>(EntryState::kTombstone)) {
            attachment = SharedAttachment();
            return index;
        }
    }
    throw std::runtime_error("Host prefix cache attachment table is full");
}

bool HostPrefixCacheCoordinator::SharedState::NodeHasRequiredGroupsLocked(
    const SharedPrefixNode& node) const {
    for (std::size_t spec_idx = 0; spec_idx < config.group_specs.size();
         ++spec_idx) {
        const SharedGroupSpec& spec = group_specs[spec_idx];
        if (spec.required_for_reuse == 0) {
            continue;
        }
        bool found = false;
        for (std::uint32_t offset = 0; offset < node.group_entry_count;
             ++offset) {
            const SharedGroupEntry& entry =
                group_entries[node.first_group_entry + offset];
            if (entry.state ==
                    static_cast<std::uint32_t>(EntryState::kResident) &&
                entry.group_id == spec.group_id &&
                entry.raw_end_token >= node.raw_end_token) {
                found = true;
                break;
            }
        }
        if (!found) {
            return false;
        }
    }
    return true;
}

std::vector<GroupMaterializationSpan>
HostPrefixCacheCoordinator::SharedState::BuildMaterializationSpansLocked(
    const SharedPrefixNode& node) const {
    std::vector<GroupMaterializationSpan> spans;
    spans.reserve(node.group_entry_count);
    for (std::uint32_t offset = 0; offset < node.group_entry_count; ++offset) {
        const SharedGroupEntry& entry =
            group_entries[node.first_group_entry + offset];
        if (entry.state !=
            static_cast<std::uint32_t>(EntryState::kResident)) {
            continue;
        }
        GroupMaterializationSpan span;
        span.group_id = entry.group_id;
        span.raw_end_token = entry.raw_end_token;
        span.pages.reserve(entry.page_handle_count);
        for (std::uint32_t page_idx = 0; page_idx < entry.page_handle_count;
             ++page_idx) {
            const SharedPageHandle& page =
                page_handles[entry.first_page_handle + page_idx];
            span.pages.push_back({page.host_region_id, page.page_id});
        }
        spans.emplace_back(std::move(span));
    }
    return spans;
}

std::uint64_t HostPrefixCacheCoordinator::SharedState::AttachNodeLocked(
    std::uint32_t node_index) {
    SharedPrefixNode& node = nodes[node_index];
    const std::uint32_t attachment_index = AllocateAttachmentLocked();
    const std::uint64_t handle =
        header->next_attachment_handle.fetch_add(1, std::memory_order_relaxed);
    for (std::uint32_t offset = 0; offset < node.group_entry_count; ++offset) {
        SharedGroupEntry& entry = group_entries[node.first_group_entry + offset];
        entry.active_ref_count.fetch_add(1, std::memory_order_relaxed);
    }
    const std::uint64_t epoch =
        header->global_epoch.fetch_add(1, std::memory_order_relaxed) + 1;
    node.last_access_epoch = epoch;

    SharedAttachment& attachment = attachments[attachment_index];
    attachment.state = static_cast<std::uint32_t>(EntryState::kResident);
    attachment.attachment_handle = handle;
    attachment.node_index = node_index;
    return handle;
}

std::uint32_t
HostPrefixCacheCoordinator::SharedState::CountFreeNodeSlotsLocked() const {
    std::uint32_t free_slots = 0;
    for (std::uint32_t index = 0; index < config.max_nodes; ++index) {
        const SharedPrefixNode& node = nodes[index];
        if (node.state == static_cast<std::uint32_t>(EntryState::kEmpty) ||
            node.state == static_cast<std::uint32_t>(EntryState::kTombstone)) {
            ++free_slots;
        }
    }
    return free_slots;
}

bool HostPrefixCacheCoordinator::SharedState::NodeIsProtectedLocked(
    const SharedPrefixNode& node) const {
    for (std::uint32_t offset = 0; offset < node.group_entry_count; ++offset) {
        const SharedGroupEntry& entry =
            group_entries[node.first_group_entry + offset];
        if (entry.state !=
            static_cast<std::uint32_t>(EntryState::kResident)) {
            continue;
        }
        if (entry.active_ref_count.load(std::memory_order_relaxed) != 0 ||
            entry.pending_load_count.load(std::memory_order_relaxed) != 0) {
            return true;
        }
    }
    return false;
}

SharedAttachment*
HostPrefixCacheCoordinator::SharedState::FindAttachmentLocked(
    std::uint64_t attachment_handle) {
    for (std::uint32_t index = 0; index < config.max_attachments; ++index) {
        SharedAttachment& candidate = attachments[index];
        if (candidate.state ==
                static_cast<std::uint32_t>(EntryState::kResident) &&
            candidate.attachment_handle == attachment_handle) {
            return &candidate;
        }
    }
    return nullptr;
}

void HostPrefixCacheCoordinator::SharedState::UpdateAttachmentLoadRefsLocked(
    SharedAttachment* attachment, int delta) {
    SharedPrefixNode& node = nodes[attachment->node_index];
    if (node.state != static_cast<std::uint32_t>(EntryState::kResident)) {
        throw std::runtime_error(
            "host prefix cache attachment refers to non-resident node");
    }
    for (std::uint32_t offset = 0; offset < node.group_entry_count; ++offset) {
        SharedGroupEntry& entry = group_entries[node.first_group_entry + offset];
        const std::uint32_t pending =
            entry.pending_load_count.load(std::memory_order_relaxed);
        if (delta > 0) {
            entry.pending_load_count.store(pending + 1,
                                           std::memory_order_relaxed);
        } else {
            if (pending == 0) {
                throw std::runtime_error(
                    "host prefix cache pending load ref underflow");
            }
            entry.pending_load_count.store(pending - 1,
                                           std::memory_order_relaxed);
        }
    }
}

void HostPrefixCacheCoordinator::SharedState::
    FinalizeAttachmentReleaseLocked(SharedAttachment* attachment) {
    if (attachment->pending_load_count != 0) {
        attachment->release_requested = 1;
        return;
    }
    attachment->state = static_cast<std::uint32_t>(EntryState::kTombstone);
}

void HostPrefixCacheCoordinator::SharedState::AppendEvictedPagesLocked(
    const SharedPrefixNode& node, PrefixEvictionResult* result) const {
    std::map<std::uint32_t, std::vector<HostPageHandle>> pages_by_group;
    for (const GroupCommitPages& group_pages : result->evicted_group_pages) {
        pages_by_group[group_pages.group_id] = group_pages.pages;
    }
    for (std::uint32_t offset = 0; offset < node.group_entry_count; ++offset) {
        const SharedGroupEntry& entry =
            group_entries[node.first_group_entry + offset];
        if (entry.state !=
            static_cast<std::uint32_t>(EntryState::kResident)) {
            continue;
        }
        std::vector<HostPageHandle>& pages = pages_by_group[entry.group_id];
        for (std::uint32_t page_idx = 0; page_idx < entry.page_handle_count;
             ++page_idx) {
            const SharedPageHandle& handle =
                page_handles[entry.first_page_handle + page_idx];
            pages.push_back({handle.host_region_id, handle.page_id});
        }
    }

    result->evicted_group_pages.clear();
    for (const HostKVGroupSpec& spec : config.group_specs) {
        auto iter = pages_by_group.find(spec.group_id);
        if (iter == pages_by_group.end() || iter->second.empty()) {
            continue;
        }
        result->evicted_group_pages.push_back(
            GroupCommitPages{iter->first, std::move(iter->second)});
    }
}

bool HostPrefixCacheCoordinator::SharedState::ResidentNodeReferencesPageLocked(
    std::uint32_t group_id, const HostPageHandle& page) const {
    for (std::uint32_t node_index = 0; node_index < config.max_nodes;
         ++node_index) {
        const SharedPrefixNode& node = nodes[node_index];
        if (node.state !=
            static_cast<std::uint32_t>(EntryState::kResident)) {
            continue;
        }
        for (std::uint32_t offset = 0; offset < node.group_entry_count;
             ++offset) {
            const SharedGroupEntry& entry =
                group_entries[node.first_group_entry + offset];
            if (entry.state !=
                    static_cast<std::uint32_t>(EntryState::kResident) ||
                entry.group_id != group_id) {
                continue;
            }
            for (std::uint32_t page_idx = 0;
                 page_idx < entry.page_handle_count; ++page_idx) {
                const SharedPageHandle& resident_page =
                    page_handles[entry.first_page_handle + page_idx];
                if (resident_page.host_region_id == page.host_region_id &&
                    resident_page.page_id == page.page_id) {
                    return true;
                }
            }
        }
    }
    return false;
}

void HostPrefixCacheCoordinator::SharedState::
    FilterEvictedPagesStillReferencedLocked(
        PrefixEvictionResult* result) const {
    for (GroupCommitPages& group_pages : result->evicted_group_pages) {
        std::vector<HostPageHandle> releasable_pages;
        for (const HostPageHandle& page : group_pages.pages) {
            if (ResidentNodeReferencesPageLocked(group_pages.group_id, page)) {
                continue;
            }
            const bool already_recorded = std::any_of(
                releasable_pages.begin(), releasable_pages.end(),
                [&page](const HostPageHandle& existing) {
                    return existing.host_region_id == page.host_region_id &&
                           existing.page_id == page.page_id;
                });
            if (!already_recorded) {
                releasable_pages.push_back(page);
            }
        }
        group_pages.pages = std::move(releasable_pages);
    }
    result->evicted_group_pages.erase(
        std::remove_if(result->evicted_group_pages.begin(),
                       result->evicted_group_pages.end(),
                       [](const GroupCommitPages& group_pages) {
                           return group_pages.pages.empty();
                       }),
        result->evicted_group_pages.end());
}

void HostPrefixCacheCoordinator::SharedState::CompactArenasLocked() {
    struct GroupEntrySnapshot {
        std::uint32_t group_id = 0;
        std::uint32_t raw_end_token = 0;
        std::uint32_t active_ref_count = 0;
        std::uint32_t pending_load_count = 0;
        std::vector<SharedPageHandle> pages;
    };
    struct NodeSnapshot {
        std::uint32_t node_index = 0;
        PrefixDigest digest{};
        std::uint32_t raw_end_token = 0;
        std::uint64_t last_access_epoch = 0;
        std::vector<GroupEntrySnapshot> groups;
    };

    std::vector<NodeSnapshot> snapshots;
    snapshots.reserve(config.max_nodes);
    for (std::uint32_t node_index = 0; node_index < config.max_nodes;
         ++node_index) {
        const SharedPrefixNode& node = nodes[node_index];
        if (node.state !=
            static_cast<std::uint32_t>(EntryState::kResident)) {
            continue;
        }
        NodeSnapshot snapshot;
        snapshot.node_index = node_index;
        snapshot.digest = node.digest;
        snapshot.raw_end_token = node.raw_end_token;
        snapshot.last_access_epoch = node.last_access_epoch;
        snapshot.groups.reserve(node.group_entry_count);
        for (std::uint32_t offset = 0; offset < node.group_entry_count;
             ++offset) {
            const SharedGroupEntry& entry =
                group_entries[node.first_group_entry + offset];
            if (entry.state !=
                static_cast<std::uint32_t>(EntryState::kResident)) {
                continue;
            }
            GroupEntrySnapshot group;
            group.group_id = entry.group_id;
            group.raw_end_token = entry.raw_end_token;
            group.active_ref_count =
                entry.active_ref_count.load(std::memory_order_relaxed);
            group.pending_load_count =
                entry.pending_load_count.load(std::memory_order_relaxed);
            group.pages.reserve(entry.page_handle_count);
            for (std::uint32_t page_idx = 0;
                 page_idx < entry.page_handle_count; ++page_idx) {
                group.pages.push_back(
                    page_handles[entry.first_page_handle + page_idx]);
            }
            snapshot.groups.emplace_back(std::move(group));
        }
        snapshots.emplace_back(std::move(snapshot));
    }

    for (std::uint32_t index = 0; index < config.max_group_entries; ++index) {
        ResetGroupEntry(group_entries[index]);
    }
    std::fill(page_handles, page_handles + config.max_page_handles,
              SharedPageHandle{});

    std::uint32_t next_group_entry = 0;
    std::uint32_t next_page_handle = 0;
    for (const NodeSnapshot& snapshot : snapshots) {
        SharedPrefixNode& node = nodes[snapshot.node_index];
        node.state = static_cast<std::uint32_t>(EntryState::kResident);
        node.digest = snapshot.digest;
        node.raw_end_token = snapshot.raw_end_token;
        node.first_group_entry = next_group_entry;
        node.group_entry_count =
            static_cast<std::uint32_t>(snapshot.groups.size());
        node.last_access_epoch = snapshot.last_access_epoch;

        for (const GroupEntrySnapshot& group : snapshot.groups) {
            SharedGroupEntry& entry = group_entries[next_group_entry++];
            ResetGroupEntry(entry);
            entry.state = static_cast<std::uint32_t>(EntryState::kResident);
            entry.group_id = group.group_id;
            entry.raw_end_token = group.raw_end_token;
            entry.first_page_handle = next_page_handle;
            entry.page_handle_count =
                static_cast<std::uint32_t>(group.pages.size());
            entry.active_ref_count.store(group.active_ref_count,
                                         std::memory_order_relaxed);
            entry.pending_load_count.store(group.pending_load_count,
                                           std::memory_order_relaxed);
            for (const SharedPageHandle& page : group.pages) {
                page_handles[next_page_handle++] = page;
            }
        }
    }
    header->next_group_entry.store(next_group_entry,
                                   std::memory_order_relaxed);
    header->next_page_handle.store(next_page_handle,
                                   std::memory_order_relaxed);
}

PrefixCommitResult HostPrefixCacheCoordinator::SharedState::CommitPrefixPages(
    PrefixDigest namespace_digest, const std::vector<std::int64_t>& token_ids,
    std::uint32_t commit_tokens,
    const std::vector<GroupCommitPages>& group_pages) {
    const std::uint32_t token_count =
        static_cast<std::uint32_t>(token_ids.size());
    commit_tokens = std::min(commit_tokens, token_count);
    commit_tokens -= commit_tokens % commit_boundary_tokens;
    if (commit_tokens == 0) {
        return {};
    }

    std::unordered_map<std::uint32_t, const std::vector<HostPageHandle>*>
        pages_by_group;
    pages_by_group.reserve(group_pages.size());
    for (const auto& pages : group_pages) {
        pages_by_group[pages.group_id] = &pages.pages;
    }
    for (const auto& spec : config.group_specs) {
        if (!spec.required_for_reuse) {
            continue;
        }
        const auto iter = pages_by_group.find(spec.group_id);
        if (iter == pages_by_group.end()) {
            throw std::invalid_argument("missing pages for required group " +
                                        std::to_string(spec.group_id));
        }
        const std::uint32_t required_pages =
            commit_tokens / spec.raw_page_tokens;
        if (iter->second->size() < required_pages) {
            throw std::invalid_argument(
                "insufficient pages for required group " +
                std::to_string(spec.group_id));
        }
    }

    const auto chain = BuildPrefixHashChain(namespace_digest, token_ids,
                                           hash_block_tokens);
    PrefixCommitResult result;
    result.committed_tokens = commit_tokens;

    ScopedMutexLock lock(&header->mutex);
    std::uint32_t new_nodes_needed = 0;
    std::uint32_t group_entries_needed = 0;
    std::uint32_t page_handles_needed = 0;
    for (const auto& [raw_end_token, digest] : chain) {
        if (raw_end_token > commit_tokens ||
            raw_end_token % commit_boundary_tokens != 0) {
            continue;
        }
        if (FindNodeLocked(digest).has_value()) {
            continue;
        }
        ++new_nodes_needed;
        for (const auto& spec : config.group_specs) {
            const auto iter = pages_by_group.find(spec.group_id);
            if (iter == pages_by_group.end()) {
                continue;
            }
            if (raw_end_token % spec.raw_page_tokens != 0) {
                continue;
            }
            const std::uint32_t pages_needed =
                raw_end_token / spec.raw_page_tokens;
            if (iter->second->size() < pages_needed) {
                continue;
            }
            ++group_entries_needed;
            page_handles_needed += pages_needed;
        }
    }

    std::uint32_t free_node_slots = 0;
    for (std::uint32_t index = 0; index < config.max_nodes; ++index) {
        const SharedPrefixNode& node = nodes[index];
        if (node.state == static_cast<std::uint32_t>(EntryState::kEmpty) ||
            node.state == static_cast<std::uint32_t>(EntryState::kTombstone)) {
            ++free_node_slots;
        }
    }
    if (free_node_slots < new_nodes_needed) {
        throw std::runtime_error("Host prefix cache node table is full");
    }
    const std::uint32_t first_group_entry =
        header->next_group_entry.load(std::memory_order_relaxed);
    const std::uint32_t first_page_handle =
        header->next_page_handle.load(std::memory_order_relaxed);
    if (first_group_entry + group_entries_needed > config.max_group_entries) {
        throw std::runtime_error(
            "Host prefix cache group entry table is full");
    }
    if (first_page_handle + page_handles_needed > config.max_page_handles) {
        throw std::runtime_error("Host prefix cache page handle arena is full");
    }

    for (const auto& [raw_end_token, digest] : chain) {
        if (raw_end_token > commit_tokens ||
            raw_end_token % commit_boundary_tokens != 0) {
            continue;
        }
        if (FindNodeLocked(digest).has_value()) {
            ++result.existing_nodes;
            continue;
        }

        std::uint32_t group_entry_count = 0;
        std::uint32_t page_handle_count = 0;
        for (const auto& spec : config.group_specs) {
            const auto iter = pages_by_group.find(spec.group_id);
            if (iter == pages_by_group.end()) {
                continue;
            }
            if (raw_end_token % spec.raw_page_tokens != 0) {
                if (spec.required_for_reuse) {
                    throw std::runtime_error(
                        "required group is not aligned to raw page tokens");
                }
                continue;
            }
            const std::uint32_t pages_needed =
                raw_end_token / spec.raw_page_tokens;
            if (iter->second->size() < pages_needed) {
                if (spec.required_for_reuse) {
                    throw std::runtime_error(
                        "required group page list became too short");
                }
                continue;
            }
            ++group_entry_count;
            page_handle_count += pages_needed;
        }

        const std::uint32_t first_group_entry =
            header->next_group_entry.load(std::memory_order_relaxed);
        const std::uint32_t first_page_handle =
            header->next_page_handle.load(std::memory_order_relaxed);
        if (first_group_entry + group_entry_count >
            config.max_group_entries) {
            throw std::runtime_error(
                "Host prefix cache group entry table is full");
        }
        if (first_page_handle + page_handle_count > config.max_page_handles) {
            throw std::runtime_error(
                "Host prefix cache page handle arena is full");
        }

        std::uint32_t next_group_entry = first_group_entry;
        std::uint32_t next_page_handle = first_page_handle;
        for (const auto& spec : config.group_specs) {
            const auto iter = pages_by_group.find(spec.group_id);
            if (iter == pages_by_group.end() ||
                raw_end_token % spec.raw_page_tokens != 0) {
                continue;
            }
            const std::uint32_t pages_needed =
                raw_end_token / spec.raw_page_tokens;
            if (iter->second->size() < pages_needed) {
                continue;
            }

            SharedGroupEntry& entry = group_entries[next_group_entry++];
            ResetGroupEntry(entry);
            entry.state = static_cast<std::uint32_t>(EntryState::kResident);
            entry.group_id = spec.group_id;
            entry.raw_end_token = raw_end_token;
            entry.first_page_handle = next_page_handle;
            entry.page_handle_count = pages_needed;
            for (std::uint32_t page_idx = 0; page_idx < pages_needed;
                 ++page_idx) {
                const HostPageHandle& handle = (*iter->second)[page_idx];
                page_handles[next_page_handle++] =
                    SharedPageHandle{handle.host_region_id, handle.page_id};
            }
        }

        const std::uint32_t node_index = AllocateNodeLocked();
        SharedPrefixNode& node = nodes[node_index];
        node.state = static_cast<std::uint32_t>(EntryState::kResident);
        node.digest = digest;
        node.raw_end_token = raw_end_token;
        node.first_group_entry = first_group_entry;
        node.group_entry_count = group_entry_count;
        node.last_access_epoch =
            header->global_epoch.fetch_add(1, std::memory_order_relaxed) + 1;
        header->next_group_entry.store(next_group_entry,
                                       std::memory_order_relaxed);
        header->next_page_handle.store(next_page_handle,
                                       std::memory_order_relaxed);
        ++result.inserted_nodes;
    }
    return result;
}

PrefixLookupResult HostPrefixCacheCoordinator::SharedState::LookupAndAttach(
    PrefixDigest namespace_digest,
    const std::vector<std::int64_t>& token_ids) {
    const auto chain = BuildPrefixHashChain(namespace_digest, token_ids,
                                           hash_block_tokens);
    PrefixLookupResult result;
    ScopedMutexLock lock(&header->mutex);
    for (auto iter = chain.rbegin(); iter != chain.rend(); ++iter) {
        const std::uint32_t raw_end_token = iter->first;
        if (raw_end_token % commit_boundary_tokens != 0) {
            continue;
        }
        const auto node_index = FindNodeLocked(iter->second);
        if (!node_index.has_value()) {
            continue;
        }
        SharedPrefixNode& node = nodes[node_index.value()];
        if (!NodeHasRequiredGroupsLocked(node)) {
            continue;
        }
        result.attachment_handle = AttachNodeLocked(node_index.value());
        result.common_cached_tokens = node.raw_end_token;
        result.materialization_spans = BuildMaterializationSpansLocked(node);
        header->lookup_hits.fetch_add(1, std::memory_order_relaxed);
        return result;
    }
    result.miss_reason_mask = 1;
    header->lookup_misses.fetch_add(1, std::memory_order_relaxed);
    return result;
}

PrefixLookupResult HostPrefixCacheCoordinator::SharedState::EstimateLookup(
    PrefixDigest namespace_digest,
    const std::vector<std::int64_t>& token_ids) {
    const auto chain = BuildPrefixHashChain(namespace_digest, token_ids,
                                           hash_block_tokens);
    PrefixLookupResult result;
    ScopedMutexLock lock(&header->mutex);
    for (auto iter = chain.rbegin(); iter != chain.rend(); ++iter) {
        const std::uint32_t raw_end_token = iter->first;
        if (raw_end_token % commit_boundary_tokens != 0) {
            continue;
        }
        const auto node_index = FindNodeLocked(iter->second);
        if (!node_index.has_value()) {
            continue;
        }
        const SharedPrefixNode& node = nodes[node_index.value()];
        if (!NodeHasRequiredGroupsLocked(node)) {
            continue;
        }
        result.common_cached_tokens = node.raw_end_token;
        result.materialization_spans = BuildMaterializationSpansLocked(node);
        return result;
    }
    result.miss_reason_mask = 1;
    return result;
}

void HostPrefixCacheCoordinator::SharedState::ReleaseAttachment(
    std::uint64_t attachment_handle) {
    if (attachment_handle == 0) {
        return;
    }
    ScopedMutexLock lock(&header->mutex);
    SharedAttachment* attachment = FindAttachmentLocked(attachment_handle);
    if (attachment == nullptr) {
        throw std::out_of_range("unknown host prefix cache attachment handle");
    }
    if (attachment->release_requested != 0) {
        throw std::runtime_error(
            "host prefix cache attachment release was already requested");
    }
    SharedPrefixNode& node = nodes[attachment->node_index];
    if (node.state == static_cast<std::uint32_t>(EntryState::kResident)) {
        for (std::uint32_t offset = 0; offset < node.group_entry_count;
             ++offset) {
            SharedGroupEntry& entry =
                group_entries[node.first_group_entry + offset];
            const std::uint32_t refs =
                entry.active_ref_count.load(std::memory_order_relaxed);
            if (refs > 0) {
                entry.active_ref_count.store(refs - 1,
                                             std::memory_order_relaxed);
            }
        }
    }
    FinalizeAttachmentReleaseLocked(attachment);
}

void HostPrefixCacheCoordinator::SharedState::BeginAttachmentLoad(
    std::uint64_t attachment_handle) {
    if (attachment_handle == 0) {
        throw std::invalid_argument(
            "host prefix cache load attachment handle must be non-zero");
    }
    ScopedMutexLock lock(&header->mutex);
    SharedAttachment* attachment = FindAttachmentLocked(attachment_handle);
    if (attachment == nullptr) {
        throw std::out_of_range("unknown host prefix cache attachment handle");
    }
    if (attachment->release_requested != 0) {
        throw std::runtime_error(
            "cannot begin load for a released host prefix cache attachment");
    }
    ++attachment->pending_load_count;
    UpdateAttachmentLoadRefsLocked(attachment, 1);
}

void HostPrefixCacheCoordinator::SharedState::EndAttachmentLoad(
    std::uint64_t attachment_handle) {
    if (attachment_handle == 0) {
        throw std::invalid_argument(
            "host prefix cache load attachment handle must be non-zero");
    }
    ScopedMutexLock lock(&header->mutex);
    SharedAttachment* attachment = FindAttachmentLocked(attachment_handle);
    if (attachment == nullptr) {
        throw std::out_of_range("unknown host prefix cache attachment handle");
    }
    if (attachment->pending_load_count == 0) {
        throw std::runtime_error(
            "host prefix cache attachment pending load underflow");
    }
    --attachment->pending_load_count;
    UpdateAttachmentLoadRefsLocked(attachment, -1);
    if (attachment->release_requested != 0 &&
        attachment->pending_load_count == 0) {
        attachment->state =
            static_cast<std::uint32_t>(EntryState::kTombstone);
    }
}

PrefixEvictionResult HostPrefixCacheCoordinator::SharedState::EvictUntilFree(
    std::uint32_t min_free_nodes,
    std::uint32_t min_free_group_entries,
    std::uint32_t min_free_page_handles,
    std::uint32_t max_scan_nodes) {
    PrefixEvictionResult result;
    ScopedMutexLock lock(&header->mutex);
    CompactArenasLocked();

    const auto has_enough_free_capacity = [&result, this, min_free_nodes,
                                           min_free_group_entries,
                                           min_free_page_handles]() {
        const std::uint32_t free_nodes = CountFreeNodeSlotsLocked();
        const std::uint32_t free_group_entries =
            config.max_group_entries -
            header->next_group_entry.load(std::memory_order_relaxed) +
            result.freed_group_entries;
        const std::uint32_t free_page_handles =
            config.max_page_handles -
            header->next_page_handle.load(std::memory_order_relaxed) +
            result.freed_page_handles;
        return free_nodes >= min_free_nodes &&
               free_group_entries >= min_free_group_entries &&
               free_page_handles >= min_free_page_handles;
    };

    if (has_enough_free_capacity()) {
        return result;
    }

    std::vector<std::uint32_t> candidates;
    candidates.reserve(config.max_nodes);
    for (std::uint32_t index = 0; index < config.max_nodes; ++index) {
        if (nodes[index].state ==
            static_cast<std::uint32_t>(EntryState::kResident)) {
            candidates.push_back(index);
        }
    }
    std::sort(candidates.begin(), candidates.end(),
              [this](std::uint32_t lhs, std::uint32_t rhs) {
                  return nodes[lhs].last_access_epoch <
                         nodes[rhs].last_access_epoch;
              });

    std::uint32_t scanned = 0;
    for (std::uint32_t node_index : candidates) {
        if (max_scan_nodes != 0 && scanned >= max_scan_nodes) {
            break;
        }
        ++scanned;
        SharedPrefixNode& node = nodes[node_index];
        if (node.state !=
            static_cast<std::uint32_t>(EntryState::kResident)) {
            continue;
        }
        if (NodeIsProtectedLocked(node)) {
            ++result.protected_nodes;
            continue;
        }

        AppendEvictedPagesLocked(node, &result);
        result.freed_group_entries += node.group_entry_count;
        for (std::uint32_t offset = 0; offset < node.group_entry_count;
             ++offset) {
            const SharedGroupEntry& entry =
                group_entries[node.first_group_entry + offset];
            if (entry.state ==
                static_cast<std::uint32_t>(EntryState::kResident)) {
                result.freed_page_handles += entry.page_handle_count;
            }
        }
        node = SharedPrefixNode();
        node.state = static_cast<std::uint32_t>(EntryState::kTombstone);
        ++result.evicted_nodes;

        if (has_enough_free_capacity()) {
            break;
        }
    }

    if (result.evicted_nodes != 0) {
        FilterEvictedPagesStillReferencedLocked(&result);
        CompactArenasLocked();
    }
    return result;
}

PrefixEvictionResult HostPrefixCacheCoordinator::SharedState::ClearUnprotected() {
    PrefixEvictionResult result;
    ScopedMutexLock lock(&header->mutex);
    CompactArenasLocked();

    for (std::uint32_t node_index = 0; node_index < config.max_nodes;
         ++node_index) {
        SharedPrefixNode& node = nodes[node_index];
        if (node.state !=
            static_cast<std::uint32_t>(EntryState::kResident)) {
            continue;
        }
        if (NodeIsProtectedLocked(node)) {
            ++result.protected_nodes;
            continue;
        }
        AppendEvictedPagesLocked(node, &result);
        result.freed_group_entries += node.group_entry_count;
        for (std::uint32_t offset = 0; offset < node.group_entry_count;
             ++offset) {
            const SharedGroupEntry& entry =
                group_entries[node.first_group_entry + offset];
            if (entry.state ==
                static_cast<std::uint32_t>(EntryState::kResident)) {
                result.freed_page_handles += entry.page_handle_count;
            }
        }
        node = SharedPrefixNode();
        node.state = static_cast<std::uint32_t>(EntryState::kTombstone);
        ++result.evicted_nodes;
    }

    if (result.evicted_nodes != 0) {
        FilterEvictedPagesStillReferencedLocked(&result);
        CompactArenasLocked();
    }
    return result;
}

HostPrefixCacheStats HostPrefixCacheCoordinator::SharedState::GetStats() const {
    ScopedMutexLock lock(&header->mutex);
    HostPrefixCacheStats stats;
    for (std::uint32_t index = 0; index < config.max_nodes; ++index) {
        if (nodes[index].state ==
            static_cast<std::uint32_t>(EntryState::kResident)) {
            ++stats.resident_nodes;
        }
    }
    for (std::uint32_t index = 0; index < config.max_attachments; ++index) {
        if (attachments[index].state ==
            static_cast<std::uint32_t>(EntryState::kResident)) {
            ++stats.active_attachments;
        }
    }
    stats.used_group_entries =
        header->next_group_entry.load(std::memory_order_relaxed);
    stats.used_page_handles =
        header->next_page_handle.load(std::memory_order_relaxed);
    stats.lookup_hits = header->lookup_hits.load(std::memory_order_relaxed);
    stats.lookup_misses =
        header->lookup_misses.load(std::memory_order_relaxed);
    return stats;
}

HostPrefixCacheCoordinator::HostPrefixCacheCoordinator(
    HostPrefixCacheConfig config)
    : config_(std::move(config)) {
    if (config_.shm_name.empty()) {
        throw std::invalid_argument("HostPrefixCacheConfig.shm_name is empty");
    }
    if (config_.group_specs.empty()) {
        throw std::invalid_argument(
            "HostPrefixCacheConfig.group_specs is empty");
    }
    if (config_.max_nodes == 0 || config_.max_group_entries == 0 ||
        config_.max_page_handles == 0 || config_.max_attachments == 0) {
        throw std::invalid_argument(
            "HostPrefixCacheConfig capacities must be positive");
    }
    bool has_required_group = false;
    for (const auto& spec : config_.group_specs) {
        ValidateGroupSpec(spec);
        has_required_group = has_required_group || spec.required_for_reuse;
    }
    if (!has_required_group) {
        throw std::invalid_argument(
            "HostPrefixCacheConfig needs at least one required group");
    }
    hash_block_tokens_ = config_.hash_block_tokens == 0
                             ? ComputeHashBlockTokens(config_.group_specs)
                             : config_.hash_block_tokens;
    commit_boundary_tokens_ =
        ComputeCommitBoundaryTokens(config_.group_specs);
    if (hash_block_tokens_ == 0 || commit_boundary_tokens_ == 0) {
        throw std::invalid_argument(
            "HostPrefixCacheConfig computed zero token boundary");
    }
    state_ = new SharedState(config_, hash_block_tokens_,
                             commit_boundary_tokens_);
}

HostPrefixCacheCoordinator::~HostPrefixCacheCoordinator() {
    if (state_ != nullptr) {
        if (state_->mapping != nullptr && state_->total_bytes != 0) {
            munmap(state_->mapping, state_->total_bytes);
        }
        if (state_->shm_fd >= 0) {
            close(state_->shm_fd);
        }
        delete state_;
    }
}

void HostPrefixCacheCoordinator::Initialize(bool create_region) {
    state_->Initialize(create_region);
}

PrefixCommitResult HostPrefixCacheCoordinator::CommitPrefixPages(
    PrefixDigest namespace_digest, const std::vector<std::int64_t>& token_ids,
    std::uint32_t commit_tokens,
    const std::vector<GroupCommitPages>& group_pages) {
    return state_->CommitPrefixPages(namespace_digest, token_ids, commit_tokens,
                                     group_pages);
}

PrefixLookupResult HostPrefixCacheCoordinator::LookupAndAttach(
    PrefixDigest namespace_digest,
    const std::vector<std::int64_t>& token_ids) {
    return state_->LookupAndAttach(namespace_digest, token_ids);
}

PrefixLookupResult HostPrefixCacheCoordinator::EstimateLookup(
    PrefixDigest namespace_digest,
    const std::vector<std::int64_t>& token_ids) {
    return state_->EstimateLookup(namespace_digest, token_ids);
}

void HostPrefixCacheCoordinator::ReleaseAttachment(
    std::uint64_t attachment_handle) {
    state_->ReleaseAttachment(attachment_handle);
}

void HostPrefixCacheCoordinator::BeginAttachmentLoad(
    std::uint64_t attachment_handle) {
    state_->BeginAttachmentLoad(attachment_handle);
}

void HostPrefixCacheCoordinator::EndAttachmentLoad(
    std::uint64_t attachment_handle) {
    state_->EndAttachmentLoad(attachment_handle);
}

PrefixEvictionResult HostPrefixCacheCoordinator::EvictUntilFree(
    std::uint32_t min_free_nodes,
    std::uint32_t min_free_group_entries,
    std::uint32_t min_free_page_handles,
    std::uint32_t max_scan_nodes) {
    return state_->EvictUntilFree(min_free_nodes, min_free_group_entries,
                                  min_free_page_handles, max_scan_nodes);
}

PrefixEvictionResult HostPrefixCacheCoordinator::ClearUnprotected() {
    return state_->ClearUnprotected();
}

HostPrefixCacheStats HostPrefixCacheCoordinator::GetStats() const {
    return state_->GetStats();
}

}  // namespace batchgen::kv
