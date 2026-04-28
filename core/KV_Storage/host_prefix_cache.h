#ifndef HOST_PREFIX_CACHE_H_
#define HOST_PREFIX_CACHE_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace batchgen::kv {

struct PrefixPageKey {
    std::uint64_t namespace_hash = 0;
    std::int32_t page_size = 0;
    std::int32_t page_index = 0;
    std::uint64_t parent_page_hash = 0;
    std::uint64_t page_token_hash = 0;

    bool operator==(const PrefixPageKey& other) const {
        return namespace_hash == other.namespace_hash &&
               page_size == other.page_size &&
               page_index == other.page_index &&
               parent_page_hash == other.parent_page_hash &&
               page_token_hash == other.page_token_hash;
    }
};

struct PrefixPageEntry {
    PrefixPageKey key;
    std::uint64_t page_chain_hash = 0;
    std::int32_t host_page_id = -1;
    std::int32_t page_size = 0;
    std::uint64_t token_validation_hash = 0;
    std::uint32_t pin_count = 0;
    std::uint64_t insert_epoch = 0;
    std::uint64_t last_access_epoch = 0;
    std::uint64_t hit_count = 0;
    std::uint32_t child_count = 0;
};

struct PrefixLookupResult {
    std::vector<std::int32_t> host_pages;
    std::size_t matched_pages = 0;
    std::size_t matched_tokens = 0;
    bool full_hit = false;
    std::string miss_reason;
};

struct PrefixCacheStats {
    std::size_t entries = 0;
    std::size_t lookup_hits = 0;
    std::size_t lookup_misses = 0;
    std::size_t shared_pages_attached = 0;
    std::size_t prefix_pin_increments = 0;
    std::size_t prefix_pin_decrements = 0;
    std::size_t host_pages_saved = 0;
    std::uint64_t eviction_epoch = 0;
    std::size_t eviction_runs = 0;
    std::size_t evicted_entries = 0;
    std::size_t evicted_prefix_pins = 0;
    std::size_t evicted_pages_immediately_freed = 0;
    std::size_t evicted_active_ref_entries = 0;
    std::size_t eviction_protected_skips = 0;
    std::size_t eviction_target_failures = 0;
};

struct PrefixEvictionOptions {
    std::size_t target_free_pages = 0;
    std::size_t max_entries_to_scan = 0;
    std::unordered_set<std::int32_t> protected_pages;
};

struct PrefixEvictionResult {
    std::size_t requested_free_pages = 0;
    std::size_t entries_removed = 0;
    std::size_t prefix_pins_released = 0;
    std::size_t pages_immediately_freed = 0;
    std::size_t protected_entries_skipped = 0;
    std::size_t active_ref_entries_removed = 0;
    bool reached_target = false;
    std::uint64_t eviction_epoch = 0;
};

struct PrefixDebugEntry {
    std::uint64_t namespace_hash = 0;
    std::int32_t page_index = 0;
    std::int32_t host_page_id = -1;
    std::uint64_t page_chain_hash = 0;
    std::uint64_t parent_page_hash = 0;
    std::uint64_t insert_epoch = 0;
    std::uint64_t last_access_epoch = 0;
    std::uint64_t hit_count = 0;
    std::uint32_t child_count = 0;
};

class HostPrefixCache {
   public:
    using PinCallback = std::function<void(std::int32_t)>;
    using UnpinCallback = std::function<void(std::int32_t)>;
    using FreePageCountCallback = std::function<std::size_t()>;

    HostPrefixCache() = default;
    HostPrefixCache(const HostPrefixCache&) = delete;
    HostPrefixCache& operator=(const HostPrefixCache&) = delete;

    PrefixLookupResult Lookup(std::uint64_t namespace_hash,
                              std::int32_t page_size,
                              const std::vector<std::int64_t>& token_ids);

    std::size_t CommitPages(std::uint64_t namespace_hash,
                            std::int32_t page_size,
                            const std::vector<std::int64_t>& token_ids,
                            const std::vector<std::int32_t>& logical_pages,
                            const PinCallback& on_pin);

    void RecordAttachedPages(std::size_t pages);

    PrefixEvictionResult EvictLeafPages(
        const PrefixEvictionOptions& options, const UnpinCallback& on_unpin,
        const FreePageCountCallback& free_pages);

    PrefixCacheStats Stats() const;
    std::vector<PrefixDebugEntry> DebugEntries(std::size_t limit = 0,
                                               bool cold_first = true) const;

    void Clear(const UnpinCallback& on_unpin);

    static std::uint64_t HashTokens(const std::int64_t* data,
                                    std::size_t count);
    static std::uint64_t HashPageKey(const PrefixPageKey& key);

   private:
    struct KeyHasher {
        std::size_t operator()(const PrefixPageKey& key) const {
            return static_cast<std::size_t>(HashPageKey(key));
        }
    };

    static std::uint64_t BuildPageChainHash(const PrefixPageKey& key);
    std::uint64_t NextAccessEpochLocked();
    void RefreshAccessLocked(PrefixPageEntry& entry);
    void IncrementParentChildCountLocked(std::uint64_t parent_hash);
    void DecrementParentChildCountLocked(std::uint64_t parent_hash);
    PrefixEvictionResult RemoveLeafEntriesLocked(
        const std::vector<PrefixPageKey>& victim_keys,
        const PrefixEvictionOptions& options, const UnpinCallback& on_unpin,
        const FreePageCountCallback& free_pages);

    mutable std::mutex mutex_;
    std::unordered_map<PrefixPageKey, PrefixPageEntry, KeyHasher> entries_;
    std::unordered_map<std::uint64_t, PrefixPageKey> chain_hash_to_key_;
    PrefixCacheStats stats_;
    std::uint64_t access_epoch_ = 0;
    std::uint64_t eviction_epoch_ = 0;
};

}  // namespace batchgen::kv

#endif  // HOST_PREFIX_CACHE_H_
