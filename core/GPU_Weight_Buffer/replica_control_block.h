#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>

// Non-owning view of a cohort's shared host mapping. The caller owns the
// mapping and orders initialize/attach at install with a group barrier.
// Members are indexed in the installed group's rank order. Record indices
// identify entries within one frame, across all module types.
class ReplicaControlBlock {
   public:
    static constexpr uint64_t kNone = std::numeric_limits<uint64_t>::max();
    enum class Barrier { A, B };

    static size_t bytes_required(size_t members, size_t records_per_frame);
    static ReplicaControlBlock initialize(void* mapping, size_t bytes,
                                          size_t members,
                                          size_t records_per_frame);
    static ReplicaControlBlock attach(void* mapping, size_t bytes);

    size_t members() const;
    size_t records_per_frame() const;
    void publish_slot(size_t member, uint64_t frame, size_t record,
                      uint64_t slot);
    uint64_t owned_slot(size_t member, uint64_t frame, size_t record) const;
    void advance(size_t member, Barrier barrier, uint64_t frame);
    uint64_t passed(size_t member, Barrier barrier) const;
    // Returns the first member that has not passed frame, or members(). One
    // acquire load per examined member; polling/back-off belongs to producer.
    size_t first_waiting(Barrier barrier, uint64_t frame) const;
    void abort(size_t member);
    bool aborted(size_t member) const;

   private:
    struct alignas(64) Header {
        std::atomic<uint64_t> ready{0};
        size_t members;
        size_t records_per_frame;
    };
    struct alignas(64) Member {
        std::atomic<uint64_t> a{kNone};
        std::atomic<uint64_t> b{kNone};
        std::atomic<uint64_t> abort{0};
    };
    explicit ReplicaControlBlock(Header* header) : header_(header) {}
    Member& member_(size_t member) const;
    std::atomic<uint64_t>& slot_(size_t member, uint64_t frame,
                                 size_t record) const;
    Header* header_;
};
