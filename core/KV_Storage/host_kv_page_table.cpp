#include "host_kv_page_table.h"

#include <sstream>
#include <string_view>

namespace batchgen::kv {

namespace {

std::string BuildErrorMessage(std::string_view prefix,
                              std::int64_t sequence_id) {
    std::ostringstream oss;
    oss << prefix << sequence_id;
    return oss.str();
}

}  // namespace

void HostKVPageTable::RegisterOrUpdate(std::int64_t sequence_id,
                                       std::vector<std::int32_t> pages) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    SequenceRecord& record = records_[sequence_id];
    record.pages = std::move(pages);
}

void HostKVPageTable::AppendPages(
    std::int64_t sequence_id,
    const std::vector<std::int32_t>& additional_pages) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    SequenceRecord& record = RequireRecordLocked(sequence_id, lock);
    record.pages.insert(record.pages.end(), additional_pages.begin(),
                        additional_pages.end());
}

std::vector<std::int32_t> HostKVPageTable::Pages(
    std::int64_t sequence_id) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    const SequenceRecord& record = RequireRecordLocked(sequence_id, lock);
    return record.pages;
}

bool HostKVPageTable::Contains(std::int64_t sequence_id) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return records_.find(sequence_id) != records_.end();
}

void HostKVPageTable::Remove(std::int64_t sequence_id) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    records_.erase(sequence_id);
}

void HostKVPageTable::Clear() {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    records_.clear();
}

std::size_t HostKVPageTable::Size() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return records_.size();
}

std::string HostKVPageTable::BuildMissingSequenceMessage(
    std::int64_t sequence_id) const {
    return BuildErrorMessage("Sequence not registered in HostKVPageTable: ",
                             sequence_id);
}

HostKVPageTable::SequenceRecord& HostKVPageTable::RequireRecordLocked(
    std::int64_t sequence_id, std::unique_lock<std::shared_mutex>& lock) {
    auto it = records_.find(sequence_id);
    if (it == records_.end()) {
        throw std::out_of_range(BuildMissingSequenceMessage(sequence_id));
    }
    return it->second;
}

const HostKVPageTable::SequenceRecord& HostKVPageTable::RequireRecordLocked(
    std::int64_t sequence_id, std::shared_lock<std::shared_mutex>& lock) const {
    auto it = records_.find(sequence_id);
    if (it == records_.end()) {
        throw std::out_of_range(BuildMissingSequenceMessage(sequence_id));
    }
    return it->second;
}

}  // namespace batchgen::kv
