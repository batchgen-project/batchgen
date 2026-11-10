#include "replica_control_block.h"

#include <new>
#include <stdexcept>

static_assert(std::atomic<uint64_t>::is_always_lock_free,
              "Replica control words must be lock-free across processes");

namespace {
constexpr uint64_t kReady = 0x4c455245504c4943ULL;

void check_mapping(void* mapping, size_t bytes, size_t required,
                   size_t alignment) {
    if (mapping == nullptr ||
        reinterpret_cast<uintptr_t>(mapping) % alignment != 0 ||
        bytes < required) {
        throw std::invalid_argument("Invalid replica control block mapping");
    }
}
}  // namespace

size_t ReplicaControlBlock::bytes_required(size_t members,
                                          size_t records_per_frame) {
    if (members == 0) {
        throw std::invalid_argument("Replica control block needs a member");
    }
    constexpr size_t max = std::numeric_limits<size_t>::max();
    if (records_per_frame > (max - sizeof(Member)) /
                                (2 * sizeof(std::atomic<uint64_t>))) {
        throw std::overflow_error("Replica owned-slot table size overflow");
    }
    const size_t per_member = sizeof(Member) +
        2 * records_per_frame * sizeof(std::atomic<uint64_t>);
    if (members > (max - sizeof(Header)) / per_member) {
        throw std::overflow_error("Replica control block size overflow");
    }
    return sizeof(Header) + members * per_member;
}

ReplicaControlBlock ReplicaControlBlock::initialize(
    void* mapping, size_t bytes, size_t members, size_t records_per_frame) {
    check_mapping(mapping, bytes, bytes_required(members, records_per_frame),
                  alignof(Header));
    auto* header = new (mapping) Header;
    header->members = members;
    header->records_per_frame = records_per_frame;
    auto* member_base = reinterpret_cast<Member*>(header + 1);
    for (size_t index = 0; index < members; ++index) {
        new (member_base + index) Member;
    }
    auto* slots = reinterpret_cast<std::atomic<uint64_t>*>(member_base + members);
    for (size_t index = 0; index < members * 2 * records_per_frame; ++index) {
        new (slots + index) std::atomic<uint64_t>(kNone);
    }
    header->ready.store(kReady, std::memory_order_release);
    return ReplicaControlBlock(header);
}

ReplicaControlBlock ReplicaControlBlock::attach(void* mapping, size_t bytes) {
    check_mapping(mapping, bytes, sizeof(Header), alignof(Header));
    auto* header = static_cast<Header*>(mapping);
    if (header->ready.load(std::memory_order_acquire) != kReady) {
        throw std::invalid_argument("Replica control block is not initialized");
    }
    check_mapping(mapping, bytes,
                  bytes_required(header->members, header->records_per_frame),
                  alignof(Header));
    return ReplicaControlBlock(header);
}

size_t ReplicaControlBlock::members() const { return header_->members; }

size_t ReplicaControlBlock::records_per_frame() const {
    return header_->records_per_frame;
}

ReplicaControlBlock::Member& ReplicaControlBlock::member_(size_t member) const {
    if (member >= members()) {
        throw std::out_of_range("Replica control block member is out of range");
    }
    return reinterpret_cast<Member*>(header_ + 1)[member];
}

std::atomic<uint64_t>& ReplicaControlBlock::slot_(
    size_t member, uint64_t frame, size_t record) const {
    member_(member);
    if (frame == kNone || record >= records_per_frame()) {
        throw std::out_of_range("Replica frame or record is out of range");
    }
    auto* slots = reinterpret_cast<std::atomic<uint64_t>*>(
        reinterpret_cast<Member*>(header_ + 1) + members());
    return slots[(member * 2 + frame % 2) * records_per_frame() + record];
}

void ReplicaControlBlock::publish_slot(size_t member, uint64_t frame,
                                       size_t record, uint64_t slot) {
    slot_(member, frame, record).store(slot, std::memory_order_release);
}

uint64_t ReplicaControlBlock::owned_slot(size_t member, uint64_t frame,
                                        size_t record) const {
    return slot_(member, frame, record).load(std::memory_order_acquire);
}

void ReplicaControlBlock::advance(size_t member, Barrier barrier, uint64_t frame) {
    if (frame == kNone) {
        throw std::invalid_argument("Replica barrier frame uses the none sentinel");
    }
    Member& words = member_(member);
    auto& word = barrier == Barrier::A ? words.a : words.b;
    const uint64_t previous = word.load(std::memory_order_acquire);
    if (previous != kNone && frame < previous) {
        throw std::logic_error("Replica barrier cannot move backwards");
    }
    // Only this member writes its words.
    word.store(frame, std::memory_order_release);
}

uint64_t ReplicaControlBlock::passed(size_t member, Barrier barrier) const {
    Member& words = member_(member);
    return (barrier == Barrier::A ? words.a : words.b)
        .load(std::memory_order_acquire);
}

size_t ReplicaControlBlock::first_waiting(Barrier barrier, uint64_t frame) const {
    if (frame == kNone) {
        throw std::invalid_argument("Replica barrier frame uses the none sentinel");
    }
    for (size_t member = 0; member < members(); ++member) {
        const uint64_t value = passed(member, barrier);
        if (value == kNone || value < frame) {
            return member;
        }
    }
    return members();
}

void ReplicaControlBlock::abort(size_t member) {
    uint64_t expected = 0;
    member_(member).abort.compare_exchange_strong(
        expected, 1, std::memory_order_release, std::memory_order_relaxed);
}

bool ReplicaControlBlock::aborted(size_t member) const {
    return member_(member).abort.load(std::memory_order_acquire) != 0;
}
