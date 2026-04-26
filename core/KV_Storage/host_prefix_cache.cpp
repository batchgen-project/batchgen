#include "host_prefix_cache.h"

#include <algorithm>
#include <stdexcept>

namespace batchgen::kv {

namespace {

constexpr std::uint64_t kFnvOffset = 1469598103934665603ULL;
constexpr std::uint64_t kFnvPrime = 1099511628211ULL;
constexpr std::uint64_t kRootPageHash = 0x524f4f545f504147ULL;  // "ROOT_PAG"

std::uint64_t HashCombine64(std::uint64_t seed, std::uint64_t value) {
    seed ^= value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
    return seed;
}

std::size_t FullPageCount(std::size_t token_count, std::int32_t page_size) {
    if (page_size <= 0) {
        throw std::invalid_argument("page_size must be positive");
    }
    return token_count / static_cast<std::size_t>(page_size);
}

}  // namespace

std::uint64_t HostPrefixCache::HashTokens(const std::int64_t* data,
                                          std::size_t count) {
    std::uint64_t hash = kFnvOffset;
    for (std::size_t i = 0; i < count; ++i) {
        std::uint64_t value = static_cast<std::uint64_t>(data[i]);
        for (int byte = 0; byte < 8; ++byte) {
            hash ^= value & 0xffULL;
            hash *= kFnvPrime;
            value >>= 8;
        }
    }
    return hash;
}

std::uint64_t HostPrefixCache::HashPageKey(const PrefixPageKey& key) {
    std::uint64_t seed = 0;
    seed = HashCombine64(seed, key.namespace_hash);
    seed = HashCombine64(seed, static_cast<std::uint64_t>(key.page_size));
    seed = HashCombine64(seed, static_cast<std::uint64_t>(key.page_index));
    seed = HashCombine64(seed, key.parent_page_hash);
    seed = HashCombine64(seed, key.page_token_hash);
    return seed;
}

std::uint64_t HostPrefixCache::BuildPageChainHash(const PrefixPageKey& key) {
    return HashPageKey(key);
}

PrefixLookupResult HostPrefixCache::Lookup(
    std::uint64_t namespace_hash, std::int32_t page_size,
    const std::vector<std::int64_t>& token_ids) {
    PrefixLookupResult result;
    const std::size_t full_pages = FullPageCount(token_ids.size(), page_size);
    if (full_pages == 0) {
        result.miss_reason = "no_full_prompt_pages";
        std::lock_guard<std::mutex> lock(mutex_);
        ++stats_.lookup_misses;
        return result;
    }

    std::uint64_t parent_hash = kRootPageHash;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (std::size_t page = 0; page < full_pages; ++page) {
            const auto page_size_count = static_cast<std::size_t>(page_size);
            const std::int64_t* page_tokens =
                token_ids.data() + page * page_size_count;
            const std::uint64_t token_hash =
                HashTokens(page_tokens, page_size_count);
            PrefixPageKey key{namespace_hash,
                              page_size,
                              static_cast<std::int32_t>(page),
                              parent_hash,
                              token_hash};
            const auto it = entries_.find(key);
            if (it == entries_.end()) {
                result.miss_reason =
                    page == 0 ? "first_page_miss" : "prefix_chain_miss";
                break;
            }
            const PrefixPageEntry& entry = it->second;
            if (entry.token_validation_hash != token_hash) {
                result.miss_reason = "token_validation_hash_mismatch";
                break;
            }
            result.host_pages.push_back(entry.host_page_id);
            parent_hash = entry.page_chain_hash;
        }
        result.matched_pages = result.host_pages.size();
        result.matched_tokens =
            result.matched_pages * static_cast<std::size_t>(page_size);
        result.full_hit = result.matched_tokens == token_ids.size();
        if (result.matched_pages == full_pages) {
            result.miss_reason.clear();
        }
        if (result.matched_pages == 0) {
            ++stats_.lookup_misses;
        } else {
            ++stats_.lookup_hits;
        }
    }
    return result;
}

std::size_t HostPrefixCache::CommitPages(
    std::uint64_t namespace_hash, std::int32_t page_size,
    const std::vector<std::int64_t>& token_ids,
    const std::vector<std::int32_t>& logical_pages,
    const PinCallback& on_pin) {
    const std::size_t full_pages = FullPageCount(token_ids.size(), page_size);
    if (full_pages == 0) {
        return 0;
    }
    if (logical_pages.size() < full_pages) {
        throw std::invalid_argument(
            "CommitPages: logical page table is smaller than full prompt pages");
    }

    std::size_t inserted = 0;
    std::uint64_t parent_hash = kRootPageHash;
    const auto page_size_count = static_cast<std::size_t>(page_size);
    std::lock_guard<std::mutex> lock(mutex_);
    for (std::size_t page = 0; page < full_pages; ++page) {
        const std::int64_t* page_tokens =
            token_ids.data() + page * page_size_count;
        const std::uint64_t token_hash =
            HashTokens(page_tokens, page_size_count);
        PrefixPageKey key{namespace_hash,
                          page_size,
                          static_cast<std::int32_t>(page),
                          parent_hash,
                          token_hash};
        const std::uint64_t chain_hash = BuildPageChainHash(key);
        const auto it = entries_.find(key);
        if (it == entries_.end()) {
            PrefixPageEntry entry;
            entry.key = key;
            entry.page_chain_hash = chain_hash;
            entry.host_page_id = logical_pages[page];
            entry.page_size = page_size;
            entry.token_validation_hash = token_hash;
            entry.pin_count = 1;
            entries_.emplace(key, entry);
            if (on_pin) {
                on_pin(entry.host_page_id);
            }
            ++inserted;
            ++stats_.prefix_pin_increments;
        }
        parent_hash = chain_hash;
    }
    stats_.entries = entries_.size();
    return inserted;
}

void HostPrefixCache::RecordAttachedPages(std::size_t pages) {
    if (pages == 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    stats_.shared_pages_attached += pages;
    stats_.host_pages_saved += pages;
}

PrefixCacheStats HostPrefixCache::Stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    PrefixCacheStats stats = stats_;
    stats.entries = entries_.size();
    return stats;
}

void HostPrefixCache::Clear(const UnpinCallback& on_unpin) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& item : entries_) {
        const PrefixPageEntry& entry = item.second;
        for (std::uint32_t i = 0; i < entry.pin_count; ++i) {
            if (on_unpin) {
                on_unpin(entry.host_page_id);
            }
            ++stats_.prefix_pin_decrements;
        }
    }
    entries_.clear();
    stats_.entries = 0;
}

}  // namespace batchgen::kv
