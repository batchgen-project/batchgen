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
        std::vector<std::int32_t> pages;
    };

    HostKVPageTable() = default;
    HostKVPageTable(const HostKVPageTable&) = delete;
    HostKVPageTable& operator=(const HostKVPageTable&) = delete;
    HostKVPageTable(HostKVPageTable&&) = delete;
    HostKVPageTable& operator=(HostKVPageTable&&) = delete;

    void RegisterOrUpdate(std::int64_t sequence_id,
                          std::vector<std::int32_t> pages);

    void AppendPages(std::int64_t sequence_id,
                     const std::vector<std::int32_t>& additional_pages);

    [[nodiscard]] std::vector<std::int32_t> Pages(
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
