#ifndef HOST_KV_PAGE_TABLE_H_
#define HOST_KV_PAGE_TABLE_H_

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace batchgen::kv {

class HostKVPageTable {
   public:
    struct SequenceRecord {
        std::vector<std::int32_t> shared_prefix_pages;
        std::vector<std::int32_t> private_pages;
        std::int64_t shared_prefix_tokens = 0;
        std::int64_t private_start_token = 0;
        std::int64_t logical_context_tokens = 0;
    };

    HostKVPageTable() = default;
    HostKVPageTable(const HostKVPageTable&) = delete;
    HostKVPageTable& operator=(const HostKVPageTable&) = delete;
    HostKVPageTable(HostKVPageTable&&) = delete;
    HostKVPageTable& operator=(HostKVPageTable&&) = delete;

    void RegisterOrUpdate(std::int64_t sequence_id,
                          std::vector<std::int32_t> pages);

    void RegisterOrUpdate(std::int64_t sequence_id,
                          std::vector<std::int32_t> shared_prefix_pages,
                          std::vector<std::int32_t> private_pages,
                          std::int64_t shared_prefix_tokens,
                          std::int64_t private_start_token,
                          std::int64_t logical_context_tokens);

    void AppendPages(std::int64_t sequence_id,
                     const std::vector<std::int32_t>& additional_pages);

    [[nodiscard]] std::vector<std::int32_t> Pages(
        std::int64_t sequence_id) const;

    [[nodiscard]] std::vector<std::int32_t> SharedPrefixPages(
        std::int64_t sequence_id) const;

    [[nodiscard]] std::vector<std::int32_t> PrivatePages(
        std::int64_t sequence_id) const;

    [[nodiscard]] std::int64_t SharedPrefixTokens(
        std::int64_t sequence_id) const;

    [[nodiscard]] std::int64_t PrivateStartToken(
        std::int64_t sequence_id) const;

    [[nodiscard]] std::int64_t LogicalContextTokens(
        std::int64_t sequence_id) const;

    [[nodiscard]] bool Contains(std::int64_t sequence_id) const;

    void Remove(std::int64_t sequence_id);

    void Clear();

    [[nodiscard]] std::size_t Size() const;

   private:
    [[nodiscard]] std::string BuildMissingSequenceMessage(
        std::int64_t sequence_id) const;

    SequenceRecord& RequireRecordLocked(
        std::int64_t sequence_id, std::unique_lock<std::shared_mutex>& lock);
    const SequenceRecord& RequireRecordLocked(
        std::int64_t sequence_id,
        std::shared_lock<std::shared_mutex>& lock) const;

    mutable std::shared_mutex mutex_;
    std::unordered_map<std::int64_t, SequenceRecord> records_;
};

}  // namespace batchgen::kv

#endif  // HOST_KV_PAGE_TABLE_H_
