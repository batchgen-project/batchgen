#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>

// Non-owning view of a cohort's shared host mapping. The caller owns the
// mapping and orders initialize/attach at install with a group barrier.
// Members are indexed in the installed group's rank order. Record indices
// identify entries within one frame, across all module types.
//
// Barrier words are per (member, module type): the producer interleaves
// module types whose rings run at different depths (a 2-slot attention ring
// trails the forward while the routed ring prefetches whole frames), so
// frame completion is monotone only WITHIN a type. A single per-member word
// regressed on GLM dense-prefix frames: the owner landed routed frame 3
// before attention frame 2, advanced past it, and every consumer of attn_2
// read "owner passed barrier A without publishing". Types are indexed by
// the install's sorted type order, identical on every member.
class ReplicaControlBlock {
   public:
    static constexpr uint64_t kNone = std::numeric_limits<uint64_t>::max();
    enum class Barrier { A, B };

    static size_t bytes_required(size_t members, size_t records_per_frame,
                                 size_t num_types);
    static ReplicaControlBlock initialize(void* mapping, size_t bytes,
                                          size_t members,
                                          size_t records_per_frame,
                                          size_t num_types);
    static ReplicaControlBlock attach(void* mapping, size_t bytes);

    size_t members() const;
    size_t records_per_frame() const;
    size_t num_types() const;
    void publish_slot(size_t member, uint64_t frame, size_t record,
                      uint64_t slot);
    uint64_t owned_slot(size_t member, uint64_t frame, size_t record) const;
    void advance(size_t member, size_t type, Barrier barrier, uint64_t frame);
    uint64_t passed(size_t member, size_t type, Barrier barrier) const;
    // Returns the first member that has not passed frame for the type, or
    // members(). One acquire load per examined member; polling/back-off
    // belongs to the producer.
    size_t first_waiting(size_t type, Barrier barrier, uint64_t frame) const;
    void abort(size_t member);
    bool aborted(size_t member) const;

   private:
    struct alignas(64) Header {
        std::atomic<uint64_t> ready{0};
        size_t members;
        size_t records_per_frame;
        size_t num_types;
    };
    struct alignas(64) Member {
        std::atomic<uint64_t> abort{0};
    };
    explicit ReplicaControlBlock(Header* header) : header_(header) {}
    Member& member_(size_t member) const;
    std::atomic<uint64_t>& word_(size_t member, size_t type,
                                 Barrier barrier) const;
    std::atomic<uint64_t>& slot_(size_t member, uint64_t frame,
                                 size_t record) const;
    std::atomic<uint64_t>* words_base_() const;
    Header* header_;
};
