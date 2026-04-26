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
    record.shared_prefix_pages.clear();
    record.private_pages = std::move(pages);
    record.shared_prefix_tokens = 0;
    record.private_start_token = 0;
    record.logical_context_tokens = 0;
}

void HostKVPageTable::RegisterOrUpdate(
    std::int64_t sequence_id, std::vector<std::int32_t> shared_prefix_pages,
    std::vector<std::int32_t> private_pages,
    std::int64_t shared_prefix_tokens, std::int64_t private_start_token,
    std::int64_t logical_context_tokens) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    SequenceRecord& record = records_[sequence_id];
    record.shared_prefix_pages = std::move(shared_prefix_pages);
    record.private_pages = std::move(private_pages);
    record.shared_prefix_tokens = shared_prefix_tokens;
    record.private_start_token = private_start_token;
    record.logical_context_tokens = logical_context_tokens;
}

void HostKVPageTable::AppendPages(
    std::int64_t sequence_id,
    const std::vector<std::int32_t>& additional_pages) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    SequenceRecord& record = RequireRecordLocked(sequence_id, lock);
    record.private_pages.insert(record.private_pages.end(),
                                additional_pages.begin(),
                                additional_pages.end());
}

std::vector<std::int32_t> HostKVPageTable::Pages(
    std::int64_t sequence_id) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    const SequenceRecord& record = RequireRecordLocked(sequence_id, lock);
    std::vector<std::int32_t> pages;
    pages.reserve(record.shared_prefix_pages.size() +
                  record.private_pages.size());
    pages.insert(pages.end(), record.shared_prefix_pages.begin(),
                 record.shared_prefix_pages.end());
    pages.insert(pages.end(), record.private_pages.begin(),
                 record.private_pages.end());
    return pages;
}

std::vector<std::int32_t> HostKVPageTable::SharedPrefixPages(
    std::int64_t sequence_id) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    const SequenceRecord& record = RequireRecordLocked(sequence_id, lock);
    return record.shared_prefix_pages;
}

std::vector<std::int32_t> HostKVPageTable::PrivatePages(
    std::int64_t sequence_id) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    const SequenceRecord& record = RequireRecordLocked(sequence_id, lock);
    return record.private_pages;
}

std::int64_t HostKVPageTable::SharedPrefixTokens(
    std::int64_t sequence_id) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    const SequenceRecord& record = RequireRecordLocked(sequence_id, lock);
    return record.shared_prefix_tokens;
}

std::int64_t HostKVPageTable::PrivateStartToken(
    std::int64_t sequence_id) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    const SequenceRecord& record = RequireRecordLocked(sequence_id, lock);
    return record.private_start_token;
}

std::int64_t HostKVPageTable::LogicalContextTokens(
    std::int64_t sequence_id) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    const SequenceRecord& record = RequireRecordLocked(sequence_id, lock);
    return record.logical_context_tokens;
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
