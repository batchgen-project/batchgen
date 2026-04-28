#include "host_prefix_cache.h"

#include <algorithm>
#include <limits>
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

bool IsRootParentHash(std::uint64_t parent_hash) {
    return parent_hash == kRootPageHash;
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

std::uint64_t HostPrefixCache::NextAccessEpochLocked() {
    if (access_epoch_ == std::numeric_limits<std::uint64_t>::max()) {
        access_epoch_ = 0;
    }
    return ++access_epoch_;
}

void HostPrefixCache::RefreshAccessLocked(PrefixPageEntry& entry) {
    entry.last_access_epoch = NextAccessEpochLocked();
    ++entry.hit_count;
}

void HostPrefixCache::IncrementParentChildCountLocked(
    std::uint64_t parent_hash) {
    if (IsRootParentHash(parent_hash)) {
        return;
    }
    const auto parent_key_it = chain_hash_to_key_.find(parent_hash);
    if (parent_key_it == chain_hash_to_key_.end()) {
        throw std::logic_error(
            "HostPrefixCache: missing parent chain hash during insert");
    }
    auto parent_it = entries_.find(parent_key_it->second);
    if (parent_it == entries_.end()) {
        throw std::logic_error(
            "HostPrefixCache: missing parent entry during insert");
    }
    ++parent_it->second.child_count;
}

void HostPrefixCache::DecrementParentChildCountLocked(
    std::uint64_t parent_hash) {
    if (IsRootParentHash(parent_hash)) {
        return;
    }
    const auto parent_key_it = chain_hash_to_key_.find(parent_hash);
    if (parent_key_it == chain_hash_to_key_.end()) {
        throw std::logic_error(
            "HostPrefixCache: missing parent chain hash during delete");
    }
    auto parent_it = entries_.find(parent_key_it->second);
    if (parent_it == entries_.end()) {
        throw std::logic_error(
            "HostPrefixCache: missing parent entry during delete");
    }
    if (parent_it->second.child_count == 0) {
        throw std::logic_error(
            "HostPrefixCache: parent child_count underflow");
    }
    --parent_it->second.child_count;
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
            PrefixPageEntry& entry = it->second;
            if (entry.token_validation_hash != token_hash) {
                result.miss_reason = "token_validation_hash_mismatch";
                break;
            }
            RefreshAccessLocked(entry);
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
            entry.insert_epoch = NextAccessEpochLocked();
            entry.last_access_epoch = entry.insert_epoch;
            entry.hit_count = 0;
            entry.child_count = 0;
            entries_.emplace(key, entry);
            chain_hash_to_key_[chain_hash] = key;
            IncrementParentChildCountLocked(key.parent_page_hash);
            if (on_pin) {
                on_pin(entry.host_page_id);
            }
            ++inserted;
            ++stats_.prefix_pin_increments;
        } else {
            RefreshAccessLocked(it->second);
        }
        parent_hash = chain_hash;
    }
    stats_.entries = entries_.size();
    return inserted;
}

PrefixEvictionResult HostPrefixCache::RemoveLeafEntriesLocked(
    const std::vector<PrefixPageKey>& victim_keys,
    const PrefixEvictionOptions& options, const UnpinCallback& on_unpin,
    const FreePageCountCallback& free_pages) {
    PrefixEvictionResult result;
    result.requested_free_pages = options.target_free_pages;
    for (const PrefixPageKey& key : victim_keys) {
        auto it = entries_.find(key);
        if (it == entries_.end()) {
            continue;
        }
        PrefixPageEntry entry = it->second;
        if (entry.child_count != 0) {
            continue;
        }
        if (options.protected_pages.find(entry.host_page_id) !=
            options.protected_pages.end()) {
            ++result.protected_entries_skipped;
            continue;
        }

        const std::size_t before_free = free_pages ? free_pages() : 0;
        DecrementParentChildCountLocked(entry.key.parent_page_hash);
        chain_hash_to_key_.erase(entry.page_chain_hash);
        entries_.erase(it);
        for (std::uint32_t pin = 0; pin < entry.pin_count; ++pin) {
            if (on_unpin) {
                on_unpin(entry.host_page_id);
            }
            ++result.prefix_pins_released;
            ++stats_.prefix_pin_decrements;
        }
        ++result.entries_removed;

        const std::size_t after_free = free_pages ? free_pages() : before_free;
        if (after_free > before_free) {
            result.pages_immediately_freed += after_free - before_free;
        } else {
            ++result.active_ref_entries_removed;
        }

        if (free_pages && after_free >= options.target_free_pages) {
            result.reached_target = true;
            break;
        }
    }
    stats_.entries = entries_.size();
    return result;
}

PrefixEvictionResult HostPrefixCache::EvictLeafPages(
    const PrefixEvictionOptions& options, const UnpinCallback& on_unpin,
    const FreePageCountCallback& free_pages) {
    PrefixEvictionResult total;
    total.requested_free_pages = options.target_free_pages;
    std::lock_guard<std::mutex> lock(mutex_);
    ++stats_.eviction_runs;
    if (free_pages && free_pages() >= options.target_free_pages) {
        total.reached_target = true;
        total.eviction_epoch = eviction_epoch_;
        return total;
    }

    while (true) {
        if (free_pages && free_pages() >= options.target_free_pages) {
            total.reached_target = true;
            break;
        }

        struct Candidate {
            PrefixPageKey key;
            std::uint64_t last_access_epoch = 0;
            std::uint64_t insert_epoch = 0;
            std::uint64_t page_chain_hash = 0;
            std::int32_t host_page_id = -1;
        };

        std::vector<Candidate> candidates;
        candidates.reserve(entries_.size());
        std::size_t scanned = 0;
        std::size_t protected_skips = 0;
        for (const auto& item : entries_) {
            if (options.max_entries_to_scan != 0 &&
                scanned >= options.max_entries_to_scan) {
                break;
            }
            ++scanned;
            const PrefixPageEntry& entry = item.second;
            if (entry.child_count != 0) {
                continue;
            }
            if (options.protected_pages.find(entry.host_page_id) !=
                options.protected_pages.end()) {
                ++protected_skips;
                continue;
            }
            candidates.push_back(Candidate{item.first,
                                           entry.last_access_epoch,
                                           entry.insert_epoch,
                                           entry.page_chain_hash,
                                           entry.host_page_id});
        }
        total.protected_entries_skipped += protected_skips;
        if (candidates.empty()) {
            break;
        }
        std::sort(candidates.begin(), candidates.end(),
                  [](const Candidate& lhs, const Candidate& rhs) {
                      if (lhs.last_access_epoch != rhs.last_access_epoch) {
                          return lhs.last_access_epoch < rhs.last_access_epoch;
                      }
                      if (lhs.insert_epoch != rhs.insert_epoch) {
                          return lhs.insert_epoch < rhs.insert_epoch;
                      }
                      if (lhs.page_chain_hash != rhs.page_chain_hash) {
                          return lhs.page_chain_hash < rhs.page_chain_hash;
                      }
                      return lhs.host_page_id < rhs.host_page_id;
                  });

        std::vector<PrefixPageKey> victim_keys;
        victim_keys.reserve(candidates.size());
        for (const Candidate& candidate : candidates) {
            victim_keys.push_back(candidate.key);
        }
        PrefixEvictionResult step = RemoveLeafEntriesLocked(
            victim_keys, options, on_unpin, free_pages);
        total.entries_removed += step.entries_removed;
        total.prefix_pins_released += step.prefix_pins_released;
        total.pages_immediately_freed += step.pages_immediately_freed;
        total.active_ref_entries_removed += step.active_ref_entries_removed;
        total.protected_entries_skipped += step.protected_entries_skipped;
        if (step.reached_target) {
            total.reached_target = true;
            break;
        }
        if (step.entries_removed == 0) {
            break;
        }
    }

    if (total.entries_removed > 0) {
        ++eviction_epoch_;
    }
    total.eviction_epoch = eviction_epoch_;
    stats_.entries = entries_.size();
    stats_.eviction_epoch = eviction_epoch_;
    stats_.evicted_entries += total.entries_removed;
    stats_.evicted_prefix_pins += total.prefix_pins_released;
    stats_.evicted_pages_immediately_freed += total.pages_immediately_freed;
    stats_.evicted_active_ref_entries += total.active_ref_entries_removed;
    stats_.eviction_protected_skips += total.protected_entries_skipped;
    if (!total.reached_target) {
        ++stats_.eviction_target_failures;
    }
    return total;
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
    stats.eviction_epoch = eviction_epoch_;
    return stats;
}

void HostPrefixCache::Clear(const UnpinCallback& on_unpin) {
    std::lock_guard<std::mutex> lock(mutex_);
    const bool had_entries = !entries_.empty();
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
    chain_hash_to_key_.clear();
    if (had_entries) {
        ++eviction_epoch_;
        stats_.eviction_epoch = eviction_epoch_;
    }
    stats_.entries = 0;
}

}  // namespace batchgen::kv
