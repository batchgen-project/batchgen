#ifndef HOST_PAGED_KV_PREFIX_CACHE_H_
#define HOST_PAGED_KV_PREFIX_CACHE_H_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace batchgen::kv {

constexpr std::int32_t kHostKVInvalidIndex = -1;
constexpr std::size_t kHostKVRadixEdgeLabelChunk = 64;

struct HostKVRadixNode {
    std::int32_t parent_node = kHostKVInvalidIndex;
    std::int32_t parent_edge = kHostKVInvalidIndex;
    std::int32_t first_edge = kHostKVInvalidIndex;
    std::int32_t terminal_entry = kHostKVInvalidIndex;
    std::uint32_t child_count = 0;
};

struct HostKVRadixEdge {
    std::int32_t child_node = kHostKVInvalidIndex;
    std::int32_t next_sibling_edge = kHostKVInvalidIndex;
    std::uint16_t label_len = 0;
    std::int32_t label_tokens[kHostKVRadixEdgeLabelChunk] = {0};
};

struct HostKVPrefixPageRef {
    std::int32_t page_idx = kHostKVInvalidIndex;
    std::int32_t next = kHostKVInvalidIndex;
};

struct HostKVPrefixEntry {
    std::int32_t terminal_node = kHostKVInvalidIndex;
    std::uint32_t num_pages = 0;
    std::int32_t page_ref_head = kHostKVInvalidIndex;
    std::int32_t lru_prev = kHostKVInvalidIndex;
    std::int32_t lru_next = kHostKVInvalidIndex;
    std::uint64_t last_access_epoch = 0;
    std::uint8_t in_use = 0;
};

struct HostKVPrefixCacheParams {
    bool enable_prefix_reuse = false;
    std::size_t prefix_min_reuse_pages = 1;
    std::size_t prefix_min_store_pages = 2;
    std::size_t prefix_page_budget = 0;
};

class HostKVPrefixCache {
   public:
    struct SharedFields {
        std::atomic<std::uint32_t>* radix_node_free_top = nullptr;
        std::atomic<std::uint32_t>* radix_edge_free_top = nullptr;
        std::atomic<std::uint32_t>* prefix_entry_free_top = nullptr;
        std::atomic<std::uint32_t>* prefix_page_ref_free_top = nullptr;

        std::atomic<std::uint32_t>* prefix_entry_count = nullptr;
        std::atomic<std::uint32_t>* prefix_used_pages = nullptr;

        std::atomic<std::uint64_t>* prefix_access_epoch = nullptr;
        std::atomic<std::uint64_t>* prefix_hit_count = nullptr;
        std::atomic<std::uint64_t>* prefix_miss_count = nullptr;
        std::atomic<std::uint64_t>* prefix_evict_count = nullptr;

        std::int32_t* lru_head = nullptr;
        std::int32_t* lru_tail = nullptr;
    };

    struct LookupResult {
        std::vector<std::int32_t> pages;
        std::size_t reused_pages = 0;
        std::int32_t entry_idx = kHostKVInvalidIndex;
    };

    using PageRefCallback = std::function<void(std::int32_t)>;

    HostKVPrefixCache() = default;

    void Bind(const HostKVPrefixCacheParams& params, HostKVRadixNode* radix_nodes,
              HostKVRadixEdge* radix_edges, HostKVPrefixEntry* prefix_entries,
              HostKVPrefixPageRef* prefix_page_refs,
              std::int32_t* radix_node_free_stack,
              std::int32_t* radix_edge_free_stack,
              std::int32_t* prefix_entry_free_stack,
              std::int32_t* prefix_page_ref_free_stack,
              const SharedFields& shared_fields,
              PageRefCallback increment_page_ref_cb,
              PageRefCallback decrement_page_ref_cb);

    void InitializePools(std::size_t radix_node_capacity,
                         std::size_t radix_edge_capacity,
                         std::size_t prefix_entry_capacity,
                         std::size_t prefix_page_ref_capacity);

    LookupResult LookupPrefixPagesLocked(const std::int32_t* tokens,
                                         std::size_t token_count,
                                         std::size_t max_pages);

    bool CommitPrefixLocked(const std::int32_t* tokens, std::size_t token_count,
                            const std::vector<std::int32_t>& pages);

   private:
    std::int32_t PopStackIndexLocked(std::int32_t* stack,
                                     std::atomic<std::uint32_t>* top,
                                     const char* what) const;
    void PushStackIndexLocked(std::int32_t* stack,
                              std::atomic<std::uint32_t>* top,
                              std::int32_t value) const;

    std::int32_t AllocateRadixNodeLocked(std::int32_t parent_node,
                                         std::int32_t parent_edge);
    void FreeRadixNodeLocked(std::int32_t node_idx);
    std::int32_t AllocateRadixEdgeLocked(std::int32_t child_node,
                                         std::int32_t next_sibling_edge,
                                         const std::int32_t* label_tokens,
                                         std::size_t label_len);
    void FreeRadixEdgeLocked(std::int32_t edge_idx);

    std::int32_t AllocatePrefixEntryLocked();
    void FreePrefixEntryLocked(std::int32_t entry_idx);
    std::int32_t AllocatePrefixPageRefLocked(std::int32_t page_idx,
                                             std::int32_t next);
    void FreePrefixPageRefLocked(std::int32_t page_ref_idx);

    std::int32_t FindEdgeByFirstTokenLocked(std::int32_t node_idx,
                                            std::int32_t first_token) const;
    std::int32_t AppendTokenPathLocked(std::int32_t start_node,
                                       const std::int32_t* tokens,
                                       std::size_t token_count);
    std::int32_t UpsertRadixPathLocked(const std::int32_t* tokens,
                                       std::size_t token_count);

    std::vector<std::int32_t> CollectPrefixEntryPagesLocked(
        std::int32_t entry_idx, std::size_t max_pages) const;

    void TouchPrefixEntryLocked(std::int32_t entry_idx);
    void LruDetachLocked(std::int32_t entry_idx);
    void LruAttachTailLocked(std::int32_t entry_idx);

    void PruneEmptyNodeChainLocked(std::int32_t node_idx);
    bool EvictOnePrefixEntryLocked();
    bool EnsurePrefixCapacityLocked(std::size_t required_nodes,
                                    std::size_t required_edges,
                                    std::size_t required_entries,
                                    std::size_t required_page_refs,
                                    std::size_t extra_budget_pages);

    std::size_t FreeRadixNodeCountLocked() const;
    std::size_t FreeRadixEdgeCountLocked() const;
    std::size_t FreePrefixEntryCountLocked() const;
    std::size_t FreePrefixPageRefCountLocked() const;

    HostKVPrefixCacheParams params_{};
    SharedFields shared_{};

    HostKVRadixNode* radix_nodes_ = nullptr;
    HostKVRadixEdge* radix_edges_ = nullptr;
    HostKVPrefixEntry* prefix_entries_ = nullptr;
    HostKVPrefixPageRef* prefix_page_refs_ = nullptr;

    std::int32_t* radix_node_free_stack_ = nullptr;
    std::int32_t* radix_edge_free_stack_ = nullptr;
    std::int32_t* prefix_entry_free_stack_ = nullptr;
    std::int32_t* prefix_page_ref_free_stack_ = nullptr;

    PageRefCallback increment_page_ref_cb_;
    PageRefCallback decrement_page_ref_cb_;
};

}  // namespace batchgen::kv

#endif  // HOST_PAGED_KV_PREFIX_CACHE_H_
