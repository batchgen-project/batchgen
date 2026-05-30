#ifndef HOST_PREFIX_CACHE_COORDINATOR_H_
#define HOST_PREFIX_CACHE_COORDINATOR_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace batchgen::kv {

using PrefixDigest = std::array<std::uint64_t, 4>;

enum class HostKVGroupSemantic : std::uint32_t {
    kFullKV = 0,
    kMlaCompressedKV = 1,
    kSwaKV = 2,
    kCompressedRatioKV = 3,
};

struct HostKVGroupSpec {
    std::uint32_t group_id = 0;
    HostKVGroupSemantic semantic = HostKVGroupSemantic::kFullKV;
    bool required_for_reuse = true;
    std::uint32_t raw_page_tokens = 0;
    std::uint32_t compression_ratio = 1;
};

struct HostPageHandle {
    std::uint32_t page_id = 0;
};

struct GroupCommitPages {
    std::uint32_t group_id = 0;
    std::vector<HostPageHandle> pages;
};

struct GroupPageRequirement {
    std::uint32_t group_id = 0;
    std::uint32_t min_pages = 0;
};

struct GroupMaterializationSpan {
    std::uint32_t group_id = 0;
    std::uint32_t raw_end_token = 0;
    std::vector<HostPageHandle> pages;
};

struct PrefixLookupResult {
    std::uint64_t attachment_handle = 0;
    std::uint32_t common_cached_tokens = 0;
    std::vector<GroupMaterializationSpan> materialization_spans;
    std::uint64_t miss_reason_mask = 0;
};

struct PrefixCommitResult {
    std::uint32_t committed_tokens = 0;
    std::uint32_t inserted_nodes = 0;
    std::uint32_t existing_nodes = 0;
};

struct PrefixEvictionResult {
    std::uint32_t evicted_nodes = 0;
    std::uint32_t protected_nodes = 0;
    std::uint32_t freed_group_entries = 0;
    std::uint32_t freed_page_handles = 0;
    std::vector<GroupCommitPages> evicted_group_pages;
};

struct HostPrefixCacheStats {
    std::uint32_t resident_nodes = 0;
    std::uint32_t active_attachments = 0;
    std::uint32_t pending_load_entries = 0;
    std::uint32_t pending_load_refs = 0;
    std::uint32_t used_group_entries = 0;
    std::uint32_t used_page_handles = 0;
    std::uint64_t lookup_hits = 0;
    std::uint64_t lookup_misses = 0;
    std::uint64_t evicted_nodes = 0;
    std::uint64_t eviction_protected_skips = 0;
};

struct HostPrefixCacheConfig {
    std::string shm_name;
    std::vector<HostKVGroupSpec> group_specs;
    std::uint32_t hash_block_tokens = 0;
    std::uint32_t max_nodes = 0;
    std::uint32_t max_group_entries = 0;
    std::uint32_t max_page_handles = 0;
    std::uint32_t max_attachments = 0;
};

std::string ToString(const HostKVGroupSpec& spec);
std::string ToString(const HostPrefixCacheStats& stats);
std::vector<std::pair<std::uint32_t, PrefixDigest>> BuildPrefixHashChain(
    PrefixDigest namespace_digest, const std::vector<std::int64_t>& token_ids,
    std::uint32_t block_tokens);

class HostPrefixCacheCoordinator {
   public:
    explicit HostPrefixCacheCoordinator(HostPrefixCacheConfig config);
    HostPrefixCacheCoordinator(const HostPrefixCacheCoordinator&) = delete;
    HostPrefixCacheCoordinator& operator=(const HostPrefixCacheCoordinator&) =
        delete;
    HostPrefixCacheCoordinator(HostPrefixCacheCoordinator&&) = delete;
    HostPrefixCacheCoordinator& operator=(HostPrefixCacheCoordinator&&) =
        delete;
    ~HostPrefixCacheCoordinator();

    void Initialize(bool create_region);

    PrefixCommitResult CommitPrefixPages(
        PrefixDigest namespace_digest,
        const std::vector<std::int64_t>& token_ids, std::uint32_t commit_tokens,
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

    PrefixEvictionResult EvictUntilFree(std::uint32_t min_free_nodes,
                                        std::uint32_t min_free_group_entries,
                                        std::uint32_t min_free_page_handles,
                                        std::uint32_t max_scan_nodes);
    PrefixEvictionResult EvictUntilReleasablePages(
        const std::vector<GroupPageRequirement>& requirements,
        std::uint32_t max_scan_nodes);

    PrefixEvictionResult ClearUnprotected();
    PrefixEvictionResult ClearNamespace(PrefixDigest namespace_digest);

    HostPrefixCacheStats GetStats() const;

    std::uint32_t hash_block_tokens() const { return hash_block_tokens_; }
    std::uint32_t commit_boundary_tokens() const {
        return commit_boundary_tokens_;
    }
    const HostPrefixCacheConfig& config() const { return config_; }

   private:
    struct SharedState;

    HostPrefixCacheConfig config_;
    std::uint32_t hash_block_tokens_ = 0;
    std::uint32_t commit_boundary_tokens_ = 0;
    SharedState* state_ = nullptr;
};

}  // namespace batchgen::kv

#endif  // HOST_PREFIX_CACHE_COORDINATOR_H_
