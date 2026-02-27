#include "host_paged_kv_prefix_cache.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

namespace batchgen::kv {

void HostKVPrefixCache::Bind(const HostKVPrefixCacheParams& params,
                             HostKVRadixNode* radix_nodes,
                             HostKVRadixEdge* radix_edges,
                             HostKVPrefixEntry* prefix_entries,
                             HostKVPrefixPageRef* prefix_page_refs,
                             std::int32_t* radix_node_free_stack,
                             std::int32_t* radix_edge_free_stack,
                             std::int32_t* prefix_entry_free_stack,
                             std::int32_t* prefix_page_ref_free_stack,
                             const SharedFields& shared_fields,
                             PageRefCallback increment_page_ref_cb,
                             PageRefCallback decrement_page_ref_cb) {
    params_ = params;
    shared_ = shared_fields;

    radix_nodes_ = radix_nodes;
    radix_edges_ = radix_edges;
    prefix_entries_ = prefix_entries;
    prefix_page_refs_ = prefix_page_refs;

    radix_node_free_stack_ = radix_node_free_stack;
    radix_edge_free_stack_ = radix_edge_free_stack;
    prefix_entry_free_stack_ = prefix_entry_free_stack;
    prefix_page_ref_free_stack_ = prefix_page_ref_free_stack;

    increment_page_ref_cb_ = std::move(increment_page_ref_cb);
    decrement_page_ref_cb_ = std::move(decrement_page_ref_cb);
}

void HostKVPrefixCache::InitializePools(std::size_t radix_node_capacity,
                                        std::size_t radix_edge_capacity,
                                        std::size_t prefix_entry_capacity,
                                        std::size_t prefix_page_ref_capacity) {
    for (std::size_t i = 0; i < radix_node_capacity; ++i) {
        radix_nodes_[i] = HostKVRadixNode();
    }
    if (radix_node_capacity > 0) {
        radix_nodes_[0].parent_node = kHostKVInvalidIndex;
        radix_nodes_[0].parent_edge = kHostKVInvalidIndex;
        radix_nodes_[0].first_edge = kHostKVInvalidIndex;
        radix_nodes_[0].terminal_entry = kHostKVInvalidIndex;
        radix_nodes_[0].child_count = 0;
        std::size_t cursor = 0;
        for (std::size_t i = radix_node_capacity; i > 1; --i) {
            radix_node_free_stack_[cursor++] = static_cast<std::int32_t>(i - 1);
        }
    }

    for (std::size_t i = 0; i < radix_edge_capacity; ++i) {
        radix_edges_[i] = HostKVRadixEdge();
        radix_edge_free_stack_[i] =
            static_cast<std::int32_t>(radix_edge_capacity - 1 - i);
    }

    for (std::size_t i = 0; i < prefix_entry_capacity; ++i) {
        prefix_entries_[i] = HostKVPrefixEntry();
        prefix_entry_free_stack_[i] =
            static_cast<std::int32_t>(prefix_entry_capacity - 1 - i);
    }

    for (std::size_t i = 0; i < prefix_page_ref_capacity; ++i) {
        prefix_page_refs_[i] = HostKVPrefixPageRef();
        prefix_page_ref_free_stack_[i] =
            static_cast<std::int32_t>(prefix_page_ref_capacity - 1 - i);
    }
}

std::int32_t HostKVPrefixCache::PopStackIndexLocked(
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

void HostKVPrefixCache::PushStackIndexLocked(std::int32_t* stack,
                                             std::atomic<std::uint32_t>* top,
                                             std::int32_t value) const {
    const std::uint32_t current = top->load(std::memory_order_relaxed);
    stack[current] = value;
    top->store(current + 1, std::memory_order_relaxed);
}

std::int32_t HostKVPrefixCache::AllocateRadixNodeLocked(std::int32_t parent_node,
                                                         std::int32_t parent_edge) {
    const std::int32_t node_idx = PopStackIndexLocked(
        radix_node_free_stack_, shared_.radix_node_free_top, "radix nodes");
    radix_nodes_[node_idx] = HostKVRadixNode();
    radix_nodes_[node_idx].parent_node = parent_node;
    radix_nodes_[node_idx].parent_edge = parent_edge;
    return node_idx;
}

void HostKVPrefixCache::FreeRadixNodeLocked(std::int32_t node_idx) {
    radix_nodes_[node_idx] = HostKVRadixNode();
    PushStackIndexLocked(radix_node_free_stack_, shared_.radix_node_free_top,
                         node_idx);
}

std::int32_t HostKVPrefixCache::AllocateRadixEdgeLocked(
    std::int32_t child_node, std::int32_t next_sibling_edge,
    const std::int32_t* label_tokens, std::size_t label_len) {
    if (label_len == 0 || label_len > kHostKVRadixEdgeLabelChunk) {
        throw std::invalid_argument("invalid radix edge label length");
    }
    const std::int32_t edge_idx = PopStackIndexLocked(
        radix_edge_free_stack_, shared_.radix_edge_free_top, "radix edges");
    radix_edges_[edge_idx] = HostKVRadixEdge();
    radix_edges_[edge_idx].child_node = child_node;
    radix_edges_[edge_idx].next_sibling_edge = next_sibling_edge;
    radix_edges_[edge_idx].label_len = static_cast<std::uint16_t>(label_len);
    std::memcpy(radix_edges_[edge_idx].label_tokens, label_tokens,
                sizeof(std::int32_t) * label_len);
    return edge_idx;
}

void HostKVPrefixCache::FreeRadixEdgeLocked(std::int32_t edge_idx) {
    radix_edges_[edge_idx] = HostKVRadixEdge();
    PushStackIndexLocked(radix_edge_free_stack_, shared_.radix_edge_free_top,
                         edge_idx);
}

std::int32_t HostKVPrefixCache::AllocatePrefixEntryLocked() {
    const std::int32_t entry_idx = PopStackIndexLocked(
        prefix_entry_free_stack_, shared_.prefix_entry_free_top,
        "prefix entries");
    prefix_entries_[entry_idx] = HostKVPrefixEntry();
    prefix_entries_[entry_idx].in_use = 1;
    return entry_idx;
}

void HostKVPrefixCache::FreePrefixEntryLocked(std::int32_t entry_idx) {
    prefix_entries_[entry_idx] = HostKVPrefixEntry();
    PushStackIndexLocked(prefix_entry_free_stack_, shared_.prefix_entry_free_top,
                         entry_idx);
}

std::int32_t HostKVPrefixCache::AllocatePrefixPageRefLocked(std::int32_t page_idx,
                                                             std::int32_t next) {
    const std::int32_t ref_idx = PopStackIndexLocked(
        prefix_page_ref_free_stack_, shared_.prefix_page_ref_free_top,
        "prefix page refs");
    prefix_page_refs_[ref_idx].page_idx = page_idx;
    prefix_page_refs_[ref_idx].next = next;
    return ref_idx;
}

void HostKVPrefixCache::FreePrefixPageRefLocked(std::int32_t page_ref_idx) {
    prefix_page_refs_[page_ref_idx] = HostKVPrefixPageRef();
    PushStackIndexLocked(prefix_page_ref_free_stack_,
                         shared_.prefix_page_ref_free_top, page_ref_idx);
}

std::int32_t HostKVPrefixCache::FindEdgeByFirstTokenLocked(
    std::int32_t node_idx, std::int32_t first_token) const {
    std::int32_t edge_idx = radix_nodes_[node_idx].first_edge;
    while (edge_idx != kHostKVInvalidIndex) {
        const HostKVRadixEdge& edge = radix_edges_[edge_idx];
        if (edge.label_len > 0 && edge.label_tokens[0] == first_token) {
            return edge_idx;
        }
        edge_idx = edge.next_sibling_edge;
    }
    return kHostKVInvalidIndex;
}

std::int32_t HostKVPrefixCache::AppendTokenPathLocked(std::int32_t start_node,
                                                       const std::int32_t* tokens,
                                                       std::size_t token_count) {
    std::int32_t node_idx = start_node;
    std::size_t pos = 0;
    while (pos < token_count) {
        const std::size_t chunk =
            std::min<std::size_t>(kHostKVRadixEdgeLabelChunk, token_count - pos);
        const std::int32_t child_idx =
            AllocateRadixNodeLocked(node_idx, kHostKVInvalidIndex);
        const std::int32_t edge_idx = AllocateRadixEdgeLocked(
            child_idx, radix_nodes_[node_idx].first_edge, tokens + pos, chunk);
        radix_nodes_[node_idx].first_edge = edge_idx;
        ++radix_nodes_[node_idx].child_count;
        radix_nodes_[child_idx].parent_edge = edge_idx;
        node_idx = child_idx;
        pos += chunk;
    }
    return node_idx;
}

std::int32_t HostKVPrefixCache::UpsertRadixPathLocked(const std::int32_t* tokens,
                                                       std::size_t token_count) {
    std::int32_t node_idx = 0;
    std::size_t pos = 0;

    while (pos < token_count) {
        std::int32_t edge_idx = FindEdgeByFirstTokenLocked(node_idx, tokens[pos]);
        if (edge_idx == kHostKVInvalidIndex) {
            return AppendTokenPathLocked(node_idx, tokens + pos, token_count - pos);
        }

        HostKVRadixEdge& edge = radix_edges_[edge_idx];
        const std::size_t remaining = token_count - pos;
        const std::size_t compare_len =
            std::min<std::size_t>(edge.label_len, remaining);
        std::size_t common = 0;
        while (common < compare_len &&
               edge.label_tokens[common] == tokens[pos + common]) {
            ++common;
        }

        if (common == edge.label_len) {
            pos += common;
            node_idx = edge.child_node;
            continue;
        }

        if (common == 0) {
            return AppendTokenPathLocked(node_idx, tokens + pos, token_count - pos);
        }

        const std::int32_t old_child = edge.child_node;
        const std::size_t old_suffix_len = edge.label_len - common;
        std::int32_t old_suffix_tokens[kHostKVRadixEdgeLabelChunk] = {0};
        std::memcpy(old_suffix_tokens, edge.label_tokens + common,
                    sizeof(std::int32_t) * old_suffix_len);

        const std::int32_t split_node = AllocateRadixNodeLocked(node_idx, edge_idx);
        const std::int32_t old_suffix_edge = AllocateRadixEdgeLocked(
            old_child, kHostKVInvalidIndex, old_suffix_tokens, old_suffix_len);

        radix_nodes_[split_node].first_edge = old_suffix_edge;
        radix_nodes_[split_node].child_count = 1;

        radix_nodes_[old_child].parent_node = split_node;
        radix_nodes_[old_child].parent_edge = old_suffix_edge;

        edge.child_node = split_node;
        edge.label_len = static_cast<std::uint16_t>(common);

        pos += common;
        if (pos == token_count) {
            return split_node;
        }
        return AppendTokenPathLocked(split_node, tokens + pos, token_count - pos);
    }

    return node_idx;
}

std::vector<std::int32_t> HostKVPrefixCache::CollectPrefixEntryPagesLocked(
    std::int32_t entry_idx, std::size_t max_pages) const {
    std::vector<std::int32_t> pages;
    if (entry_idx == kHostKVInvalidIndex || max_pages == 0) {
        return pages;
    }

    const HostKVPrefixEntry& entry = prefix_entries_[entry_idx];
    const std::size_t limit = std::min<std::size_t>(entry.num_pages, max_pages);
    pages.reserve(limit);

    std::int32_t ref_idx = entry.page_ref_head;
    while (ref_idx != kHostKVInvalidIndex && pages.size() < limit) {
        pages.push_back(prefix_page_refs_[ref_idx].page_idx);
        ref_idx = prefix_page_refs_[ref_idx].next;
    }
    return pages;
}

HostKVPrefixCache::LookupResult HostKVPrefixCache::LookupPrefixPagesLocked(
    const std::int32_t* tokens, std::size_t token_count, std::size_t max_pages) {
    LookupResult result;
    if (!params_.enable_prefix_reuse || tokens == nullptr || token_count == 0 ||
        max_pages == 0) {
        return result;
    }

    auto consider_entry = [&](std::int32_t entry_idx) {
        if (entry_idx == kHostKVInvalidIndex) {
            return;
        }
        const HostKVPrefixEntry& entry = prefix_entries_[entry_idx];
        if (entry.in_use == 0) {
            return;
        }
        const std::size_t candidate_pages =
            std::min<std::size_t>(entry.num_pages, max_pages);
        if (candidate_pages < params_.prefix_min_reuse_pages ||
            candidate_pages <= result.reused_pages) {
            return;
        }

        auto pages = CollectPrefixEntryPagesLocked(entry_idx, candidate_pages);
        if (pages.size() < candidate_pages) {
            return;
        }

        result.pages = std::move(pages);
        result.reused_pages = candidate_pages;
        result.entry_idx = entry_idx;
    };

    std::int32_t node_idx = 0;
    std::size_t pos = 0;
    consider_entry(radix_nodes_[node_idx].terminal_entry);

    while (pos < token_count) {
        const std::int32_t edge_idx =
            FindEdgeByFirstTokenLocked(node_idx, tokens[pos]);
        if (edge_idx == kHostKVInvalidIndex) {
            break;
        }
        const HostKVRadixEdge& edge = radix_edges_[edge_idx];
        const std::size_t remaining = token_count - pos;
        const std::size_t compare_len =
            std::min<std::size_t>(edge.label_len, remaining);

        std::size_t common = 0;
        while (common < compare_len &&
               edge.label_tokens[common] == tokens[pos + common]) {
            ++common;
        }
        if (common != edge.label_len) {
            break;
        }

        pos += common;
        node_idx = edge.child_node;
        consider_entry(radix_nodes_[node_idx].terminal_entry);
    }

    if (result.reused_pages >= params_.prefix_min_reuse_pages) {
        shared_.prefix_hit_count->fetch_add(1, std::memory_order_relaxed);
        TouchPrefixEntryLocked(result.entry_idx);
    } else {
        shared_.prefix_miss_count->fetch_add(1, std::memory_order_relaxed);
        result = LookupResult();
    }

    return result;
}

void HostKVPrefixCache::LruDetachLocked(std::int32_t entry_idx) {
    HostKVPrefixEntry& entry = prefix_entries_[entry_idx];
    if (entry.lru_prev != kHostKVInvalidIndex) {
        prefix_entries_[entry.lru_prev].lru_next = entry.lru_next;
    } else if (*shared_.lru_head == entry_idx) {
        *shared_.lru_head = entry.lru_next;
    }

    if (entry.lru_next != kHostKVInvalidIndex) {
        prefix_entries_[entry.lru_next].lru_prev = entry.lru_prev;
    } else if (*shared_.lru_tail == entry_idx) {
        *shared_.lru_tail = entry.lru_prev;
    }

    entry.lru_prev = kHostKVInvalidIndex;
    entry.lru_next = kHostKVInvalidIndex;
}

void HostKVPrefixCache::LruAttachTailLocked(std::int32_t entry_idx) {
    HostKVPrefixEntry& entry = prefix_entries_[entry_idx];
    entry.lru_prev = *shared_.lru_tail;
    entry.lru_next = kHostKVInvalidIndex;
    if (*shared_.lru_tail != kHostKVInvalidIndex) {
        prefix_entries_[*shared_.lru_tail].lru_next = entry_idx;
    } else {
        *shared_.lru_head = entry_idx;
    }
    *shared_.lru_tail = entry_idx;
}

void HostKVPrefixCache::TouchPrefixEntryLocked(std::int32_t entry_idx) {
    if (entry_idx == kHostKVInvalidIndex || prefix_entries_[entry_idx].in_use == 0) {
        return;
    }
    HostKVPrefixEntry& entry = prefix_entries_[entry_idx];
    entry.last_access_epoch =
        shared_.prefix_access_epoch->fetch_add(1, std::memory_order_relaxed) + 1;
    if (*shared_.lru_tail == entry_idx) {
        return;
    }
    LruDetachLocked(entry_idx);
    LruAttachTailLocked(entry_idx);
}

void HostKVPrefixCache::PruneEmptyNodeChainLocked(std::int32_t node_idx) {
    while (node_idx != 0 && node_idx != kHostKVInvalidIndex) {
        HostKVRadixNode& node = radix_nodes_[node_idx];
        if (node.terminal_entry != kHostKVInvalidIndex || node.child_count != 0) {
            break;
        }

        const std::int32_t parent_idx = node.parent_node;
        const std::int32_t parent_edge = node.parent_edge;
        if (parent_idx == kHostKVInvalidIndex ||
            parent_edge == kHostKVInvalidIndex) {
            break;
        }

        std::int32_t prev_edge = kHostKVInvalidIndex;
        std::int32_t cur_edge = radix_nodes_[parent_idx].first_edge;
        while (cur_edge != kHostKVInvalidIndex && cur_edge != parent_edge) {
            prev_edge = cur_edge;
            cur_edge = radix_edges_[cur_edge].next_sibling_edge;
        }
        if (cur_edge == kHostKVInvalidIndex) {
            break;
        }

        const std::int32_t next_edge = radix_edges_[cur_edge].next_sibling_edge;
        if (prev_edge == kHostKVInvalidIndex) {
            radix_nodes_[parent_idx].first_edge = next_edge;
        } else {
            radix_edges_[prev_edge].next_sibling_edge = next_edge;
        }
        if (radix_nodes_[parent_idx].child_count > 0) {
            --radix_nodes_[parent_idx].child_count;
        }

        FreeRadixEdgeLocked(cur_edge);
        const std::int32_t to_free = node_idx;
        node_idx = parent_idx;
        FreeRadixNodeLocked(to_free);
    }
}

bool HostKVPrefixCache::EvictOnePrefixEntryLocked() {
    const std::int32_t entry_idx = *shared_.lru_head;
    if (entry_idx == kHostKVInvalidIndex) {
        return false;
    }

    HostKVPrefixEntry& entry = prefix_entries_[entry_idx];
    if (entry.in_use == 0) {
        LruDetachLocked(entry_idx);
        FreePrefixEntryLocked(entry_idx);
        return true;
    }

    LruDetachLocked(entry_idx);

    if (entry.terminal_node != kHostKVInvalidIndex &&
        radix_nodes_[entry.terminal_node].terminal_entry == entry_idx) {
        radix_nodes_[entry.terminal_node].terminal_entry = kHostKVInvalidIndex;
    }

    std::int32_t ref_idx = entry.page_ref_head;
    while (ref_idx != kHostKVInvalidIndex) {
        const std::int32_t next = prefix_page_refs_[ref_idx].next;
        const std::int32_t page_idx = prefix_page_refs_[ref_idx].page_idx;
        decrement_page_ref_cb_(page_idx);
        FreePrefixPageRefLocked(ref_idx);
        ref_idx = next;
    }

    if (shared_.prefix_entry_count->load(std::memory_order_relaxed) > 0) {
        shared_.prefix_entry_count->fetch_sub(1, std::memory_order_relaxed);
    }
    const std::uint32_t used_pages =
        shared_.prefix_used_pages->load(std::memory_order_relaxed);
    shared_.prefix_used_pages->store(
        used_pages > entry.num_pages ? used_pages - entry.num_pages : 0,
        std::memory_order_relaxed);
    shared_.prefix_evict_count->fetch_add(1, std::memory_order_relaxed);

    const std::int32_t terminal_node = entry.terminal_node;
    FreePrefixEntryLocked(entry_idx);
    if (terminal_node != kHostKVInvalidIndex) {
        PruneEmptyNodeChainLocked(terminal_node);
    }

    return true;
}

bool HostKVPrefixCache::EnsurePrefixCapacityLocked(
    std::size_t required_nodes, std::size_t required_edges,
    std::size_t required_entries, std::size_t required_page_refs,
    std::size_t extra_budget_pages) {
    while (FreeRadixNodeCountLocked() < required_nodes ||
           FreeRadixEdgeCountLocked() < required_edges ||
           FreePrefixEntryCountLocked() < required_entries ||
           FreePrefixPageRefCountLocked() < required_page_refs ||
           shared_.prefix_used_pages->load(std::memory_order_relaxed) +
                   extra_budget_pages >
               params_.prefix_page_budget) {
        if (!EvictOnePrefixEntryLocked()) {
            return false;
        }
    }
    return true;
}

std::size_t HostKVPrefixCache::FreeRadixNodeCountLocked() const {
    return shared_.radix_node_free_top->load(std::memory_order_relaxed);
}

std::size_t HostKVPrefixCache::FreeRadixEdgeCountLocked() const {
    return shared_.radix_edge_free_top->load(std::memory_order_relaxed);
}

std::size_t HostKVPrefixCache::FreePrefixEntryCountLocked() const {
    return shared_.prefix_entry_free_top->load(std::memory_order_relaxed);
}

std::size_t HostKVPrefixCache::FreePrefixPageRefCountLocked() const {
    return shared_.prefix_page_ref_free_top->load(std::memory_order_relaxed);
}

bool HostKVPrefixCache::CommitPrefixLocked(const std::int32_t* tokens,
                                           std::size_t token_count,
                                           const std::vector<std::int32_t>& pages) {
    if (!params_.enable_prefix_reuse || tokens == nullptr || token_count == 0 ||
        pages.size() < params_.prefix_min_store_pages) {
        return false;
    }

    const std::size_t max_needed_nodes =
        (token_count + kHostKVRadixEdgeLabelChunk - 1) /
            kHostKVRadixEdgeLabelChunk +
        1;
    const std::size_t max_needed_edges = max_needed_nodes + 1;

    if (!EnsurePrefixCapacityLocked(max_needed_nodes, max_needed_edges, 1,
                                    pages.size(), pages.size())) {
        return false;
    }

    const std::int32_t terminal_node = UpsertRadixPathLocked(tokens, token_count);
    if (terminal_node == kHostKVInvalidIndex) {
        return false;
    }

    if (radix_nodes_[terminal_node].terminal_entry != kHostKVInvalidIndex) {
        TouchPrefixEntryLocked(radix_nodes_[terminal_node].terminal_entry);
        return true;
    }

    if (!EnsurePrefixCapacityLocked(0, 0, 1, pages.size(), pages.size())) {
        return false;
    }

    const std::int32_t entry_idx = AllocatePrefixEntryLocked();
    HostKVPrefixEntry& prefix_entry = prefix_entries_[entry_idx];
    prefix_entry.terminal_node = terminal_node;
    prefix_entry.num_pages = static_cast<std::uint32_t>(pages.size());
    prefix_entry.page_ref_head = kHostKVInvalidIndex;
    prefix_entry.last_access_epoch =
        shared_.prefix_access_epoch->fetch_add(1, std::memory_order_relaxed) + 1;

    std::int32_t tail_ref = kHostKVInvalidIndex;
    for (std::size_t i = 0; i < pages.size(); ++i) {
        const std::int32_t page_ref_idx =
            AllocatePrefixPageRefLocked(pages[i], kHostKVInvalidIndex);
        if (prefix_entry.page_ref_head == kHostKVInvalidIndex) {
            prefix_entry.page_ref_head = page_ref_idx;
            tail_ref = page_ref_idx;
        } else {
            prefix_page_refs_[tail_ref].next = page_ref_idx;
            tail_ref = page_ref_idx;
        }
        increment_page_ref_cb_(pages[i]);
    }

    radix_nodes_[terminal_node].terminal_entry = entry_idx;
    LruAttachTailLocked(entry_idx);
    shared_.prefix_entry_count->fetch_add(1, std::memory_order_relaxed);
    shared_.prefix_used_pages->fetch_add(static_cast<std::uint32_t>(pages.size()),
                                         std::memory_order_relaxed);

    while (shared_.prefix_used_pages->load(std::memory_order_relaxed) >
           params_.prefix_page_budget) {
        if (!EvictOnePrefixEntryLocked()) {
            break;
        }
    }

    return true;
}

}  // namespace batchgen::kv
