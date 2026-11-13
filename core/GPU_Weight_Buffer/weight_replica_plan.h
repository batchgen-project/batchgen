// -----------------------------------------------------------------------------
// Replica plan design note (slice 2, specification v4)
//
// The plan is consumed later at these exact seams:
//   * core/HtoD_Engine/HtoD_Engine.cu (HtoD_Worker)
//   * core/GPU_Weight_Buffer/GPU_Weight_Buffer.cpp
//     (reclaimCompletedReleasesLocked)
//   * batchgen/batchgen_worker.py
//     (_install_prefill_weight_copy_pipeline, _prepare_phase_exit)
// This header deliberately contains only CPU-testable ownership, frame,
// record-kind, capacity and barrier-accounting logic. The shared control
// block, engine adapters and torch CUDA IPC export/import live separately.
// -----------------------------------------------------------------------------

#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

enum class WeightModuleKind : uint8_t {
    kRoutedExpert = 0,
    kSingleModule = 1,
};

enum class WeightRecordKind : uint8_t {
    HostShared = 0,
    HostPrivate = 1,
    Replica = 2,
};

enum class WeightFrameRecordState : uint8_t {
    kPending = 0,
    kLanded = 1,
    kDeclined = 2,
    kSkipped = 3,
};

inline void validate_replica_group(const std::vector<int64_t>& group) {
    if (group.empty()) {
        throw std::invalid_argument("Replica group must not be empty");
    }
    std::set<int64_t> unique(group.begin(), group.end());
    if (unique.size() != group.size()) {
        throw std::invalid_argument("Replica group contains duplicate ranks");
    }
}

// Return the rank that owns one module in the ordered replica group. Routed
// experts use contiguous blocks. All other module types use layer round-robin.
inline int64_t source_owner(const std::vector<int64_t>& ordered_group,
                            WeightModuleKind module_kind, uint64_t layer,
                            uint64_t expert, uint64_t experts) {
    validate_replica_group(ordered_group);
    const uint64_t group_size = ordered_group.size();
    if (module_kind == WeightModuleKind::kRoutedExpert) {
        if (experts == 0 || expert >= experts) {
            throw std::invalid_argument(
                "Routed expert index is outside the expert count");
        }
        if (experts % group_size != 0) {
            throw std::invalid_argument(
                "Expert count " + std::to_string(experts) +
                " is not divisible by replica group size " +
                std::to_string(group_size));
        }
        const uint64_t block = experts / group_size;
        return ordered_group[expert / block];
    }
    return ordered_group[layer % group_size];
}

inline bool is_routed_expert_type(std::string_view module_type) {
    return module_type == "routed_expert" ||
           module_type == "routed_experts" || module_type == "routed";
}

inline int64_t source_owner(const std::vector<int64_t>& ordered_group,
                            std::string_view module_type, uint64_t layer,
                            uint64_t expert, uint64_t experts) {
    return source_owner(ordered_group,
                        is_routed_expert_type(module_type)
                            ? WeightModuleKind::kRoutedExpert
                            : WeightModuleKind::kSingleModule,
                        layer, expert, experts);
}

struct WeightRecord {
    std::string module;
    int64_t owner = 0;
    std::vector<int64_t> readers;
    WeightRecordKind kind = WeightRecordKind::HostPrivate;
};

// Build the install-time record map from the static module reader map. The
// caller supplies one map of owners for the same cohort and the rank whose
// local records are being installed.
inline std::vector<WeightRecord> install_record_kinds(
    const std::vector<int64_t>& ordered_group, int64_t rank,
    const std::map<std::string, int64_t>& owner_by_module,
    const std::map<std::string, std::vector<int64_t>>& static_reader_map) {
    validate_replica_group(ordered_group);
    if (std::find(ordered_group.begin(), ordered_group.end(), rank) ==
        ordered_group.end()) {
        throw std::invalid_argument("Rank " + std::to_string(rank) +
                                    " is not in the replica group");
    }
    std::vector<WeightRecord> records;
    records.reserve(owner_by_module.size());
    for (const auto& [module, owner] : owner_by_module) {
        if (std::find(ordered_group.begin(), ordered_group.end(), owner) ==
            ordered_group.end()) {
            throw std::invalid_argument("Owner " + std::to_string(owner) +
                                        " of " + module +
                                        " is outside the replica group");
        }
        const auto readers_it = static_reader_map.find(module);
        if (readers_it == static_reader_map.end()) {
            throw std::invalid_argument("Static reader map lacks module " +
                                        module);
        }
        const auto& readers = readers_it->second;
        if (std::find(readers.begin(), readers.end(), owner) ==
            readers.end()) {
            throw std::invalid_argument("Owner " + std::to_string(owner) +
                                        " of " + module +
                                        " is absent from its reader map");
        }
        const bool rank_reads =
            std::find(readers.begin(), readers.end(), rank) != readers.end();
        if (!rank_reads) {
            continue;
        }
        const bool shared = std::any_of(
            readers.begin(), readers.end(),
            [rank](int64_t reader) { return reader != rank; });
        records.push_back(WeightRecord{
            module, owner, readers,
            owner == rank
                ? (shared ? WeightRecordKind::HostShared
                          : WeightRecordKind::HostPrivate)
                : WeightRecordKind::Replica});
    }
    return records;
}

struct WeightFrame {
    uint64_t pass = 0;
    uint64_t layer_position = 0;
    uint64_t flat = 0;
};

inline uint64_t flat_frame(uint64_t pass, uint64_t layer_position,
                           uint64_t layers_per_pass) {
    if (pass == 0 || layers_per_pass == 0 ||
        layer_position >= layers_per_pass) {
        throw std::invalid_argument(
            "Invalid frame: pass, layer position or layers per pass");
    }
    constexpr uint64_t max = std::numeric_limits<uint64_t>::max();
    if (pass - 1 > (max - layer_position) / layers_per_pass) {
        throw std::overflow_error("flat frame index exceeds uint64_t");
    }
    return (pass - 1) * layers_per_pass + layer_position;
}

class WeightFrameMap {
   public:
    explicit WeightFrameMap(uint64_t layers_per_pass)
        : layers_per_pass_(layers_per_pass) {
        if (layers_per_pass_ == 0) {
            throw std::invalid_argument("layers_per_pass must be positive");
        }
    }

    WeightFrame at(uint64_t pass, uint64_t layer_position) const {
        return WeightFrame{pass, layer_position,
                           flat_frame(pass, layer_position,
                                      layers_per_pass_)};
    }

   private:
    uint64_t layers_per_pass_ = 0;
};

// Return an empty string when the ring can hold the two-frame rendezvous.
inline std::string ring_capacity_error(std::string_view module_type,
                                       uint64_t ring_slots,
                                       uint64_t frame_size) {
    const bool too_large =
        frame_size > std::numeric_limits<uint64_t>::max() / 2;
    if (!too_large && ring_slots >= 2 * frame_size) {
        return {};
    }
    return "Ring capacity for type " + std::string(module_type) +
           ": ring_slots=" + std::to_string(ring_slots) +
           ", frame_size=" + std::to_string(frame_size) +
           ", needs at least " + std::to_string(frame_size) + " * 2 = " +
           (too_large ? "overflow" : std::to_string(2 * frame_size));
}

inline void require_ring_capacity(std::string_view module_type,
                                  uint64_t ring_slots, uint64_t frame_size) {
    const std::string error =
        ring_capacity_error(module_type, ring_slots, frame_size);
    if (!error.empty()) {
        throw std::invalid_argument(error);
    }
}

struct WeightFrameAccounting {
    explicit WeightFrameAccounting(const std::vector<WeightRecord>& records)
        : kinds(), states(records.size(), WeightFrameRecordState::kPending) {
        kinds.reserve(records.size());
        for (const WeightRecord& record : records) {
            kinds.push_back(record.kind);
        }
    }

    void mark_landed(size_t index) { mark(index, WeightFrameRecordState::kLanded); }
    void mark_declined(size_t index) {
        mark(index, WeightFrameRecordState::kDeclined);
    }
    void mark_skipped(size_t index) {
        mark(index, WeightFrameRecordState::kSkipped);
    }

    // Barrier A is local to this rank and needs every owned record landed.
    bool may_post_barrier_a() const {
        for (size_t index = 0; index < states.size(); ++index) {
            if (kinds[index] != WeightRecordKind::Replica &&
                states[index] != WeightFrameRecordState::kLanded) {
                return false;
            }
        }
        return true;
    }

    // Barrier B needs every local record to be landed, declined or skipped.
    bool may_post_barrier_b() const {
        return std::all_of(
            states.begin(), states.end(), [](WeightFrameRecordState state) {
                return state != WeightFrameRecordState::kPending;
            });
    }

   private:
    std::vector<WeightRecordKind> kinds;
    std::vector<WeightFrameRecordState> states;

    void check_index(size_t index) const {
        if (index >= states.size()) {
            throw std::out_of_range("Frame record index is outside the frame");
        }
    }

    void mark(size_t index, WeightFrameRecordState next) {
        check_index(index);
        if (states[index] != WeightFrameRecordState::kPending) {
            throw std::logic_error("Frame record is already terminal");
        }
        states[index] = next;
    }
};

