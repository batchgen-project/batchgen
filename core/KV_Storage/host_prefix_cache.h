#ifndef HOST_PREFIX_CACHE_H_
#define HOST_PREFIX_CACHE_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>
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
};

class HostPrefixCache {
   public:
    using PinCallback = std::function<void(std::int32_t)>;
    using UnpinCallback = std::function<void(std::int32_t)>;

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

    PrefixCacheStats Stats() const;

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

    mutable std::mutex mutex_;
    std::unordered_map<PrefixPageKey, PrefixPageEntry, KeyHasher> entries_;
    PrefixCacheStats stats_;
};

}  // namespace batchgen::kv

#endif  // HOST_PREFIX_CACHE_H_
