// clang-format off
/* ----------------------------------------------------------------------------  *
 *  BatchGen                                                                      *
 *  Copyright (c) 2025-2026 BatchGen Team                                            *
 *                                                                               *
 *  licensed under the apache license, version 2.0 (the "license");              *
 *  you may not use this file except in compliance with the license.             *
 *                                                                               *
 *  you may obtain a copy of the license at                                      *
 *                                                                               *
 *                  http://www.apache.org/licenses/license-2.0                   *
 *                                                                               *
 *  unless required by applicable law or agreed to in writing, software          *
 *  distributed under the license is distributed on an "as is" basis,            *
 *  without warranties or conditions of any kind, either express or implied.     *
 *  see the license for the specific language governing permissions and          *
 *  limitations under the license.                                               *
 * ---------------------------------------------------------------------------- */
// clang-format on

#pragma once

// Slot bookkeeping of the GPU weight ring, kept free of torch, CUDA and
// pybind so it can be compiled and tested on its own. Not thread safe: the
// owner (GPU_Weight_Buffer) serialises every call under its buffer mutex.
//
// Each slot is in exactly one state:
//   FREE             empty, may be reserved by the producer
//   FILLING          reserved by the producer; named once claim() runs
//   PRESENT          copied, not yet handed to a consumer
//   LEASED           handed to a consumer by lease()
//   PENDING_RELEASE  released by the consumer, reusable once its compute
//                    stream passes the recorded event (reclaim())
// Every named slot carries the weight pass it was produced for.
//
// A weight pass is one traversal of every module type's installed list, i.e.
// one forward. Passes are numbered 1, 2, ... from each install(). The
// consumer arms a pass with begin_pass() and closes it with end_pass(). At
// most two passes are open: arming the second while the first is open is the
// explicit carry, which lets the producer continue into it once the first is
// produced. lease() and skip() address the oldest open pass, the one the
// consumer executes. The producer keeps one cursor (pass, position) per type
// and produces a position only while its pass is open.
//
// The name index covers FILLING (named), PRESENT and LEASED slots, keyed by
// (module, pass): a module whose slot is PENDING_RELEASE may already be
// copied again, and a carried copy is never confused with the current one.

#include <algorithm>
#include <cstdint>
#include <deque>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

enum class WeightSlotState : uint8_t {
    kFree = 0,
    kFilling = 1,
    kPresent = 2,
    kLeased = 3,
    kPendingRelease = 4,
};

constexpr int kWeightSlotStateCount = 5;

inline const char* weight_slot_state_name(WeightSlotState state) {
    switch (state) {
        case WeightSlotState::kFree:
            return "FREE";
        case WeightSlotState::kFilling:
            return "FILLING";
        case WeightSlotState::kPresent:
            return "PRESENT";
        case WeightSlotState::kLeased:
            return "LEASED";
        case WeightSlotState::kPendingRelease:
            return "PENDING_RELEASE";
    }
    return "UNKNOWN";
}

// What a routing-informed skip did to a module.
enum class WeightSkipOutcome {
    kFreedPresent,      // its PRESENT slot was freed at once
    kReleaseOnLanding,  // its copy is in flight; the slot frees when it lands
    kPending,           // not produced yet; the producer retires the skip
};

struct WeightSkipCounts {
    int64_t freed_present = 0;
    int64_t release_on_landing = 0;
    int64_t pending = 0;
};

// What the producer does next for one module type.
enum class WeightProduction {
    kNone,    // the type has no installed module
    kParked,  // the type finished every open pass
    kCopy,    // copy the module at the cursor
    kSkip,    // a skip is pending for it: acquire and release, no copy
};

enum class WeightLeaseCheck {
    kReady,  // PRESENT in the current pass
    kWait,   // FILLING, or not produced yet
};

class WeightSlotTable {
   public:
    static constexpr size_t kMaxOpenPasses = 2;

    struct SlotRef {
        std::string module_type;
        int64_t index = -1;
    };

    struct HeldSlot {
        std::string module_type;
        int64_t index = -1;
        std::string module_name;
        WeightSlotState state = WeightSlotState::kFree;
        int64_t pass = 0;
    };

    // The next position the producer handles for a type.
    struct Cursor {
        int64_t pass = 1;
        int64_t position = 0;
    };

    struct Production {
        WeightProduction kind = WeightProduction::kNone;
        int64_t pass = 0;
        int64_t position = 0;
        std::string module_name;
    };

    // copied counts every landed copy, including those released on landing.
    struct PassCounts {
        int64_t copied = 0;
        int64_t skipped = 0;
        int64_t released_on_landing = 0;
    };

    struct PassSummary {
        int64_t pass = 0;
        std::map<std::string, PassCounts> by_type;
        // Slots already produced for the next pass.
        int64_t carried = 0;
    };

    /* Layout */

    // Declare `count` FREE slots for `module_type`, dropping all of its
    // state. No pass may be open.
    void reset_type(const std::string& module_type, int64_t count) {
        this->require_no_open_pass("reset the " + module_type + " slots");
        auto existing = this->slots_.find(module_type);
        if (existing != this->slots_.end()) {
            for (size_t index = 0; index < existing->second.size(); ++index) {
                this->unindex(module_type, static_cast<int64_t>(index));
            }
        }
        this->slots_[module_type].assign(static_cast<size_t>(count), Slot{});
    }

    // Append FREE slots until `module_type` has at least `count`.
    void grow_type(const std::string& module_type, int64_t count) {
        auto& slots = this->slots_[module_type];
        if (static_cast<int64_t>(slots.size()) < count) {
            slots.resize(static_cast<size_t>(count));
        }
    }

    int64_t slot_count(const std::string& module_type) const {
        auto found = this->slots_.find(module_type);
        return found == this->slots_.end()
                   ? 0
                   : static_cast<int64_t>(found->second.size());
    }

    // Record the installed copy tasks (module type -> module names in
    // schedule order) and restart pass numbering at 1 with every cursor at
    // the head of its list. Refused while a pass is open or a slot is held:
    // either would carry state of the previous queue into pass 1.
    void install(const std::unordered_map<std::string,
                                          std::vector<std::string>>& tasks) {
        this->require_no_open_pass("install weight copy tasks");
        std::unordered_map<std::string, Installed> installed;
        for (const auto& [module_type, module_names] : tasks) {
            for (size_t position = 0; position < module_names.size();
                 ++position) {
                const auto& module_name = module_names[position];
                auto inserted = installed.emplace(
                    module_name,
                    Installed{module_type, static_cast<int64_t>(position)});
                if (inserted.second) {
                    continue;
                }
                if (inserted.first->second.module_type != module_type) {
                    throw std::runtime_error(
                        "Weight copy task " + module_name +
                        " is installed for both " +
                        inserted.first->second.module_type + " and " +
                        module_type);
                }
                throw std::runtime_error("Weight copy task " + module_name +
                                         " is installed twice for " +
                                         module_type);
            }
        }
        if (!this->held().empty()) {
            throw std::runtime_error(
                "Cannot install weight copy tasks while slots are held: " +
                this->describe_held());
        }
        this->installed_ = std::move(installed);
        this->lists_ = tasks;
        this->cursors_.clear();
        for (const auto& [module_type, module_names] : tasks) {
            this->cursors_[module_type] = Cursor{};
        }
        this->pending_skips_.clear();
        this->pass_counts_.clear();
        this->next_pass_ = 1;
    }

    /* Passes */

    // Arm the next pass and return its id, or 0 when no module is installed.
    int64_t begin_pass() {
        if (this->installed_.empty()) {
            return 0;
        }
        if (this->open_passes_.size() >= kMaxOpenPasses) {
            throw std::runtime_error(
                "Cannot begin weight pass " +
                std::to_string(this->next_pass_) + ": passes " +
                this->describe_open() + " are already open (at most " +
                std::to_string(kMaxOpenPasses) + ")");
        }
        const int64_t pass = this->next_pass_++;
        this->open_passes_.push_back(pass);
        return pass;
    }

    // Declare every module of an open pass that is not leased as skipped:
    // PRESENT frees, FILLING releases on landing, not yet produced becomes a
    // pending skip. A LEASED slot of the pass throws and changes nothing.
    // Pass 0 is a no-op.
    WeightSkipCounts cancel_pass(int64_t pass) {
        WeightSkipCounts counts;
        if (pass == 0) {
            return counts;
        }
        this->require_open(pass, "cancel");
        for (const auto& slot : this->held()) {
            if (slot.pass == pass && slot.state == WeightSlotState::kLeased) {
                throw std::runtime_error(
                    "Cannot cancel weight pass " + std::to_string(pass) +
                    ": " + slot.module_name +
                    " is still LEASED; its consumer must release it first. "
                    "Held: " +
                    this->describe_held());
            }
        }
        for (auto& [module_type, slots] : this->slots_) {
            for (Slot& slot : slots) {
                if (slot.name.empty() || slot.pass != pass) {
                    continue;
                }
                if (slot.state == WeightSlotState::kPresent) {
                    this->index_.erase(Key{slot.name, pass});
                    slot = Slot{};
                    ++counts.freed_present;
                } else if (slot.state == WeightSlotState::kFilling &&
                           !slot.release_on_landing) {
                    slot.release_on_landing = true;
                    ++counts.release_on_landing;
                }
            }
        }
        for (const auto& [module_type, module_names] : this->lists_) {
            for (size_t position = 0; position < module_names.size();
                 ++position) {
                if (this->produced(module_type, pass,
                                   static_cast<int64_t>(position))) {
                    continue;
                }
                if (this->pending_skips_
                        .insert({pass, module_names[position]})
                        .second) {
                    ++counts.pending;
                }
            }
        }
        return counts;
    }

    // True once the producer moved past the end of pass for every type.
    bool producer_finished(int64_t pass) const {
        for (const auto& [module_type, module_names] : this->lists_) {
            if (!module_names.empty() &&
                this->cursors_.at(module_type).pass <= pass) {
                return false;
            }
        }
        return true;
    }

    // Close the oldest open pass. Throws, changing nothing, unless the
    // producer is past its end, none of its slots is held and no skip of it
    // is pending. Pass 0 is a no-op.
    PassSummary end_pass(int64_t pass) {
        PassSummary summary;
        if (pass == 0) {
            return summary;
        }
        if (this->open_passes_.empty() || this->open_passes_.front() != pass) {
            throw std::runtime_error(
                "Cannot end weight pass " + std::to_string(pass) +
                ": it is not the oldest open pass (open " +
                this->describe_open() + ")");
        }
        std::ostringstream violations;
        if (!this->producer_finished(pass)) {
            violations << " the producer has not finished it ("
                       << this->describe_cursors() << ");";
        }
        for (const auto& slot : this->held()) {
            if (slot.pass == pass && !slot.module_name.empty()) {
                violations << " " << slot.module_name << " is "
                           << weight_slot_state_name(slot.state) << ";";
            }
        }
        const int64_t pending = this->pending_skip_count(pass);
        if (pending != 0) {
            violations << " " << pending << " skips are still pending;";
        }
        const std::string violation = violations.str();
        if (!violation.empty()) {
            throw std::runtime_error(
                "Cannot end weight pass " + std::to_string(pass) + ":" +
                violation + " Held: " + this->describe_held());
        }
        this->open_passes_.pop_front();
        summary.pass = pass;
        for (const auto& [module_type, module_names] : this->lists_) {
            if (!module_names.empty()) {
                summary.by_type[module_type];
            }
        }
        auto counts = this->pass_counts_.find(pass);
        if (counts != this->pass_counts_.end()) {
            for (const auto& [module_type, type_counts] : counts->second) {
                summary.by_type[module_type] = type_counts;
            }
            this->pass_counts_.erase(counts);
        }
        for (const auto& slot : this->held()) {
            if (slot.pass == pass + 1) {
                ++summary.carried;
            }
        }
        return summary;
    }

    // The pass the consumer executes, if any.
    std::optional<int64_t> current_pass() const {
        if (this->open_passes_.empty()) {
            return std::nullopt;
        }
        return this->open_passes_.front();
    }

    bool is_open(int64_t pass) const {
        return std::find(this->open_passes_.begin(), this->open_passes_.end(),
                         pass) != this->open_passes_.end();
    }

    std::vector<int64_t> open_passes() const {
        return std::vector<int64_t>(this->open_passes_.begin(),
                                    this->open_passes_.end());
    }

    void require_no_open_pass(const std::string& action) const {
        if (!this->open_passes_.empty()) {
            throw std::runtime_error("Cannot " + action +
                                     " while weight passes " +
                                     this->describe_open() + " are open");
        }
    }

    /* Producer */

    // FREE -> FILLING. The slot carries no name until claim().
    std::optional<int64_t> reserve(const std::string& module_type) {
        auto found = this->slots_.find(module_type);
        if (found == this->slots_.end()) {
            return std::nullopt;
        }
        for (size_t index = 0; index < found->second.size(); ++index) {
            Slot& slot = found->second[index];
            if (slot.state == WeightSlotState::kFree) {
                slot = Slot{};
                slot.state = WeightSlotState::kFilling;
                return static_cast<int64_t>(index);
            }
        }
        return std::nullopt;
    }

    Production next_production(const std::string& module_type) const {
        Production next;
        auto list = this->lists_.find(module_type);
        if (list == this->lists_.end() || list->second.empty()) {
            return next;
        }
        const Cursor& cursor = this->cursors_.at(module_type);
        next.pass = cursor.pass;
        next.position = cursor.position;
        next.module_name = list->second[static_cast<size_t>(cursor.position)];
        if (!this->is_open(cursor.pass)) {
            next.kind = WeightProduction::kParked;
        } else if (this->pending_skips_.count({cursor.pass, next.module_name}) !=
                   0) {
            next.kind = WeightProduction::kSkip;
        } else {
            next.kind = WeightProduction::kCopy;
        }
        return next;
    }

    // Claim the module at module_type's cursor. kCopy: the reserved slot is
    // named for the cursor's pass and the cursor advances. kSkip: a skip is
    // pending for the module, the slot (index >= 0) is abandoned and the
    // cursor stays until retire_skip(), so the pass cannot end while the
    // producer still holds the module's daemon lease.
    WeightProduction claim(const std::string& module_type, int64_t index,
                           const std::string& module_name) {
        const Production next = this->next_production(module_type);
        if ((next.kind != WeightProduction::kCopy &&
             next.kind != WeightProduction::kSkip) ||
            next.module_name != module_name) {
            throw std::runtime_error(
                "Producer claimed " + module_name + " for " + module_type +
                ", but its cursor is at " + this->describe_cursors() +
                " (open " + this->describe_open() + ")");
        }
        if (next.kind == WeightProduction::kSkip) {
            if (index >= 0) {
                this->abandon(module_type, index);
            }
            return WeightProduction::kSkip;
        }
        Slot& slot = this->slot_at(module_type, index);
        if (slot.state != WeightSlotState::kFilling || !slot.name.empty()) {
            throw this->transition_error("assign", module_name, slot.state,
                                         module_type, index);
        }
        const Key key{module_name, next.pass};
        auto resident = this->index_.find(key);
        if (resident != this->index_.end()) {
            throw this->transition_error(
                "assign", module_name,
                this->slot_at(resident->second.module_type,
                              resident->second.index)
                    .state,
                resident->second.module_type, resident->second.index);
        }
        slot.name = module_name;
        slot.pass = next.pass;
        this->index_[key] = SlotRef{module_type, index};
        this->advance(module_type);
        return WeightProduction::kCopy;
    }

    // Retire the pending skip claim() reported and advance the cursor.
    void retire_skip(const std::string& module_type,
                     const std::string& module_name) {
        const Production next = this->next_production(module_type);
        if (next.kind != WeightProduction::kSkip ||
            next.module_name != module_name) {
            throw std::runtime_error(
                "Producer retired a skip of " + module_name + " for " +
                module_type + ", but none is pending at its cursor " +
                this->describe_cursors());
        }
        this->pending_skips_.erase({next.pass, module_name});
        this->pass_counts_[next.pass][module_type].skipped += 1;
        this->advance(module_type);
    }

    // FILLING -> PRESENT, or FILLING -> FREE when a skip marked the slot
    // release-on-landing. Returns whether the module is now PRESENT.
    bool landed(const std::string& module_type, int64_t index) {
        Slot& slot = this->slot_at(module_type, index);
        if (slot.state != WeightSlotState::kFilling || slot.name.empty()) {
            throw this->transition_error("land", slot.name, slot.state,
                                         module_type, index);
        }
        PassCounts& counts = this->pass_counts_[slot.pass][module_type];
        counts.copied += 1;
        if (slot.release_on_landing) {
            counts.released_on_landing += 1;
            this->index_.erase(Key{slot.name, slot.pass});
            slot = Slot{};
            return false;
        }
        slot.state = WeightSlotState::kPresent;
        return true;
    }

    // FILLING -> FREE: the copy failed, was cancelled or was skipped.
    void abandon(const std::string& module_type, int64_t index) {
        Slot& slot = this->slot_at(module_type, index);
        if (slot.state != WeightSlotState::kFilling) {
            throw this->transition_error("abandon", slot.name, slot.state,
                                         module_type, index);
        }
        this->unindex(module_type, index);
        slot = Slot{};
    }

    /* Consumer */

    // Throws when module_name can never be leased in the current pass: no
    // pass is open, it is not installed, it is already LEASED, or it was
    // already consumed or skipped in this pass.
    WeightLeaseCheck check_lease(const std::string& module_name) const {
        const int64_t pass = this->require_current_pass("lease " + module_name);
        const Installed& installed =
            this->require_installed(module_name, "lease");
        auto resident = this->index_.find(Key{module_name, pass});
        if (resident != this->index_.end()) {
            const SlotRef& ref = resident->second;
            const Slot& slot = this->slot_at(ref.module_type, ref.index);
            if (slot.state == WeightSlotState::kPresent) {
                return WeightLeaseCheck::kReady;
            }
            if (slot.state == WeightSlotState::kFilling) {
                if (slot.release_on_landing) {
                    throw this->skipped_error("lease", module_name, pass);
                }
                return WeightLeaseCheck::kWait;
            }
            throw this->transition_error("lease", module_name, slot.state,
                                         ref.module_type, ref.index);
        }
        if (this->pending_skips_.count({pass, module_name}) != 0) {
            throw this->skipped_error("lease", module_name, pass);
        }
        if (this->produced(installed.module_type, pass, installed.position)) {
            throw this->consumed_error("lease", module_name, pass);
        }
        return WeightLeaseCheck::kWait;
    }

    // PRESENT -> LEASED, in the current pass.
    SlotRef lease(const std::string& module_name) {
        if (this->check_lease(module_name) != WeightLeaseCheck::kReady) {
            const int64_t pass = *this->current_pass();
            auto resident = this->index_.find(Key{module_name, pass});
            if (resident == this->index_.end()) {
                throw this->transition_error("lease", module_name,
                                             std::nullopt);
            }
            throw this->transition_error(
                "lease", module_name,
                this->slot_at(resident->second.module_type,
                              resident->second.index)
                    .state,
                resident->second.module_type, resident->second.index);
        }
        const SlotRef ref =
            this->index_.at(Key{module_name, *this->current_pass()});
        this->slot_at(ref.module_type, ref.index).state =
            WeightSlotState::kLeased;
        return ref;
    }

    // LEASED -> FREE, in the current pass.
    SlotRef release(const std::string& module_name) {
        const std::optional<int64_t> pass = this->current_pass();
        auto resident = this->index_.end();
        std::optional<WeightSlotState> state;
        if (pass.has_value()) {
            resident = this->index_.find(Key{module_name, *pass});
            if (resident != this->index_.end()) {
                state = this->slot_at(resident->second.module_type,
                                      resident->second.index)
                            .state;
            }
        }
        if (!state.has_value() || *state != WeightSlotState::kLeased) {
            throw std::runtime_error(
                "Cannot release missing weight-buffer lease for " +
                module_name + " (state " + this->state_text(state) +
                ", open passes " + this->describe_open() +
                "). Held: " + this->describe_held());
        }
        const SlotRef ref = resident->second;
        this->index_.erase(resident);
        this->slot_at(ref.module_type, ref.index) = Slot{};
        return ref;
    }

    // LEASED -> PENDING_RELEASE for every name, or for none when any name is
    // duplicated or not LEASED in the current pass.
    std::vector<SlotRef> release_async(
        const std::vector<std::string>& module_names) {
        const std::optional<int64_t> pass = this->current_pass();
        std::unordered_set<std::string> unique_names;
        std::vector<SlotRef> refs;
        refs.reserve(module_names.size());
        for (const auto& module_name : module_names) {
            if (!unique_names.insert(module_name).second) {
                throw std::runtime_error(
                    "Duplicate module in asynchronous weight release: " +
                    module_name);
            }
            std::optional<WeightSlotState> state;
            auto resident = this->index_.end();
            if (pass.has_value()) {
                resident = this->index_.find(Key{module_name, *pass});
                if (resident != this->index_.end()) {
                    state = this->slot_at(resident->second.module_type,
                                          resident->second.index)
                                .state;
                }
            }
            if (!state.has_value() || *state != WeightSlotState::kLeased) {
                throw std::runtime_error(
                    "Cannot asynchronously release missing weight-buffer "
                    "lease: " +
                    module_name + " (state " + this->state_text(state) +
                    ", open passes " + this->describe_open() +
                    "). Held: " + this->describe_held());
            }
            refs.push_back(resident->second);
        }
        for (size_t i = 0; i < module_names.size(); ++i) {
            this->index_.erase(Key{module_names[i], *pass});
            this->slot_at(refs[i].module_type, refs[i].index).state =
                WeightSlotState::kPendingRelease;
        }
        return refs;
    }

    // PENDING_RELEASE -> FREE, once the consumer's event has completed.
    void reclaim(const std::string& module_type, int64_t index) {
        Slot& slot = this->slot_at(module_type, index);
        if (slot.state != WeightSlotState::kPendingRelease) {
            throw this->transition_error("reclaim", slot.name, slot.state,
                                         module_type, index);
        }
        slot = Slot{};
    }

    /* Routing-informed skip, in the current pass */

    // Throws exactly when skip(module_name) would, without changing state.
    void check_skip(const std::string& module_name) const {
        const int64_t pass = this->require_current_pass("skip " + module_name);
        auto installed = this->installed_.find(module_name);
        if (installed == this->installed_.end()) {
            throw std::runtime_error(
                "Cannot skip " + module_name +
                ": it is not an installed weight copy task");
        }
        auto resident = this->index_.find(Key{module_name, pass});
        if (resident != this->index_.end()) {
            const Slot& slot = this->slot_at(resident->second.module_type,
                                             resident->second.index);
            if (slot.state == WeightSlotState::kLeased ||
                (slot.state == WeightSlotState::kFilling &&
                 slot.release_on_landing)) {
                throw this->transition_error(
                    "skip", module_name, slot.state,
                    resident->second.module_type, resident->second.index);
            }
            return;
        }
        if (this->pending_skips_.count({pass, module_name}) != 0) {
            throw std::runtime_error(
                "Duplicate skip of " + module_name + ": its copy in pass " +
                std::to_string(pass) +
                " is already skipped. Held: " + this->describe_held());
        }
        if (this->produced(installed->second.module_type, pass,
                           installed->second.position)) {
            throw this->consumed_error("skip", module_name, pass);
        }
    }

    WeightSkipOutcome skip(const std::string& module_name) {
        this->check_skip(module_name);
        const int64_t pass = *this->current_pass();
        auto resident = this->index_.find(Key{module_name, pass});
        if (resident != this->index_.end()) {
            const SlotRef ref = resident->second;
            Slot& slot = this->slot_at(ref.module_type, ref.index);
            if (slot.state == WeightSlotState::kFilling) {
                slot.release_on_landing = true;
                return WeightSkipOutcome::kReleaseOnLanding;
            }
            // PRESENT: no consumer holds it.
            this->index_.erase(resident);
            slot = Slot{};
            return WeightSkipOutcome::kFreedPresent;
        }
        this->pending_skips_.insert({pass, module_name});
        return WeightSkipOutcome::kPending;
    }

    /* Queries */

    // State of the slot indexed for (module_name, pass) (FILLING, PRESENT or
    // LEASED), or nullopt when it is not resident.
    std::optional<WeightSlotState> state_of(const std::string& module_name,
                                            int64_t pass) const {
        auto resident = this->index_.find(Key{module_name, pass});
        if (resident == this->index_.end()) {
            return std::nullopt;
        }
        return this->slot_at(resident->second.module_type,
                             resident->second.index)
            .state;
    }

    bool is_present(const std::string& module_name, int64_t pass) const {
        const auto state = this->state_of(module_name, pass);
        return state.has_value() && *state == WeightSlotState::kPresent;
    }

    bool skip_pending(const std::string& module_name, int64_t pass) const {
        return this->pending_skips_.count({pass, module_name}) != 0;
    }

    int64_t pending_skip_count(int64_t pass) const {
        int64_t count = 0;
        for (auto it = this->pending_skips_.lower_bound({pass, ""});
             it != this->pending_skips_.end() && it->first == pass; ++it) {
            ++count;
        }
        return count;
    }

    WeightSlotState state_at(const std::string& module_type,
                             int64_t index) const {
        return this->slot_at(module_type, index).state;
    }

    std::string name_at(const std::string& module_type, int64_t index) const {
        return this->slot_at(module_type, index).name;
    }

    int64_t pass_at(const std::string& module_type, int64_t index) const {
        return this->slot_at(module_type, index).pass;
    }

    int64_t count(const std::string& module_type,
                  WeightSlotState state) const {
        auto found = this->slots_.find(module_type);
        if (found == this->slots_.end()) {
            return 0;
        }
        return std::count_if(
            found->second.begin(), found->second.end(),
            [state](const Slot& slot) { return slot.state == state; });
    }

    // Named slots of pass in state, over every type.
    int64_t count_in_pass(int64_t pass, WeightSlotState state) const {
        int64_t total = 0;
        for (const auto& [module_type, slots] : this->slots_) {
            total += std::count_if(
                slots.begin(), slots.end(), [pass, state](const Slot& slot) {
                    return !slot.name.empty() && slot.pass == pass &&
                           slot.state == state;
                });
        }
        return total;
    }

    // Cursor of every type with an installed list, ordered by type.
    std::map<std::string, Cursor> cursors() const {
        std::map<std::string, Cursor> result;
        for (const auto& [module_type, module_names] : this->lists_) {
            if (!module_names.empty()) {
                result[module_type] = this->cursors_.at(module_type);
            }
        }
        return result;
    }

    // Counts of every pass with activity that has not ended yet.
    const std::map<int64_t, std::map<std::string, PassCounts>>& pass_counts()
        const {
        return this->pass_counts_;
    }

    std::vector<std::string> module_types() const {
        std::vector<std::string> types;
        types.reserve(this->slots_.size());
        for (const auto& [module_type, slots] : this->slots_) {
            types.push_back(module_type);
        }
        std::sort(types.begin(), types.end());
        return types;
    }

    // Every non-FREE slot, ordered by type and index.
    std::vector<HeldSlot> held() const {
        std::vector<HeldSlot> result;
        for (const auto& module_type : this->module_types()) {
            const auto& slots = this->slots_.at(module_type);
            for (size_t index = 0; index < slots.size(); ++index) {
                if (slots[index].state != WeightSlotState::kFree) {
                    result.push_back(HeldSlot{module_type,
                                              static_cast<int64_t>(index),
                                              slots[index].name,
                                              slots[index].state,
                                              slots[index].pass});
                }
            }
        }
        return result;
    }

    std::string describe_held() const {
        std::ostringstream oss;
        oss << "[";
        bool first = true;
        for (const auto& slot : this->held()) {
            oss << (first ? "" : ", ") << slot.module_type << "[" << slot.index
                << "]=";
            if (slot.module_name.empty()) {
                oss << "<unnamed>:" << weight_slot_state_name(slot.state);
            } else {
                oss << slot.module_name << ":"
                    << weight_slot_state_name(slot.state) << "@" << slot.pass;
            }
            first = false;
        }
        oss << "]";
        return oss.str();
    }

    std::string describe_open() const {
        std::ostringstream oss;
        oss << "[";
        for (size_t i = 0; i < this->open_passes_.size(); ++i) {
            oss << (i == 0 ? "" : ", ") << this->open_passes_[i];
        }
        oss << "]";
        return oss.str();
    }

    // "type=(pass, position/length)" per installed type.
    std::string describe_cursors() const {
        std::ostringstream oss;
        bool first = true;
        for (const auto& [module_type, cursor] : this->cursors()) {
            oss << (first ? "" : ", ") << module_type << "=(" << cursor.pass
                << ", " << cursor.position << "/"
                << this->lists_.at(module_type).size() << ")";
            first = false;
        }
        return oss.str();
    }

    std::string describe_passes() const {
        return "open passes " + this->describe_open() + ", cursors " +
               this->describe_cursors();
    }

   private:
    struct Slot {
        WeightSlotState state = WeightSlotState::kFree;
        std::string name;
        int64_t pass = 0;
        bool release_on_landing = false;
    };

    struct Installed {
        std::string module_type;
        int64_t position = 0;
    };

    // (module name, pass)
    using Key = std::pair<std::string, int64_t>;

    Slot& slot_at(const std::string& module_type, int64_t index) {
        return const_cast<Slot&>(
            static_cast<const WeightSlotTable*>(this)->slot_at(module_type,
                                                               index));
    }

    const Slot& slot_at(const std::string& module_type, int64_t index) const {
        auto found = this->slots_.find(module_type);
        if (found == this->slots_.end() || index < 0 ||
            index >= static_cast<int64_t>(found->second.size())) {
            throw std::runtime_error(
                "Invalid weight-buffer slot: type=" + module_type +
                " idx=" + std::to_string(index));
        }
        return found->second[static_cast<size_t>(index)];
    }

    // Drop the slot's (name, pass) from the index if it points at the slot.
    void unindex(const std::string& module_type, int64_t index) {
        const Slot& slot = this->slot_at(module_type, index);
        if (slot.name.empty()) {
            return;
        }
        auto resident = this->index_.find(Key{slot.name, slot.pass});
        if (resident != this->index_.end() &&
            resident->second.module_type == module_type &&
            resident->second.index == index) {
            this->index_.erase(resident);
        }
    }

    void advance(const std::string& module_type) {
        Cursor& cursor = this->cursors_.at(module_type);
        cursor.position += 1;
        if (cursor.position >=
            static_cast<int64_t>(this->lists_.at(module_type).size())) {
            cursor.pass += 1;
            cursor.position = 0;
        }
    }

    // Whether the producer already moved past position of pass.
    bool produced(const std::string& module_type, int64_t pass,
                  int64_t position) const {
        const Cursor& cursor = this->cursors_.at(module_type);
        return cursor.pass > pass ||
               (cursor.pass == pass && cursor.position > position);
    }

    int64_t require_current_pass(const std::string& action) const {
        if (this->open_passes_.empty()) {
            throw std::runtime_error("Cannot " + action +
                                     ": no weight pass is open");
        }
        return this->open_passes_.front();
    }

    void require_open(int64_t pass, const std::string& action) const {
        if (!this->is_open(pass)) {
            throw std::runtime_error(
                "Cannot " + action + " weight pass " + std::to_string(pass) +
                ": it is not open (open " + this->describe_open() + ")");
        }
    }

    const Installed& require_installed(const std::string& module_name,
                                       const std::string& action) const {
        auto installed = this->installed_.find(module_name);
        if (installed == this->installed_.end()) {
            throw std::runtime_error(
                "Cannot " + action + " " + module_name +
                ": it is not an installed weight copy task");
        }
        return installed->second;
    }

    static std::string state_text(std::optional<WeightSlotState> state) {
        return state.has_value() ? weight_slot_state_name(*state)
                                 : "not resident";
    }

    std::runtime_error consumed_error(const std::string& action,
                                      const std::string& module_name,
                                      int64_t pass) const {
        return std::runtime_error(
            "Cannot " + action + " " + module_name +
            ": it was already consumed in pass " + std::to_string(pass) +
            " (leased and released, or skipped). Held: " +
            this->describe_held());
    }

    std::runtime_error skipped_error(const std::string& action,
                                     const std::string& module_name,
                                     int64_t pass) const {
        return std::runtime_error(
            "Cannot " + action + " " + module_name +
            ": it is skipped in pass " + std::to_string(pass) +
            ". Held: " + this->describe_held());
    }

    std::runtime_error transition_error(
        const std::string& action, const std::string& module_name,
        std::optional<WeightSlotState> state,
        const std::string& module_type = "", int64_t index = -1) const {
        std::ostringstream oss;
        oss << "Illegal weight-buffer transition: " << action << " "
            << (module_name.empty() ? "<unnamed>" : module_name) << " in state "
            << state_text(state);
        if (!module_type.empty()) {
            oss << " (" << module_type << "[" << index << "])";
        }
        oss << ". Held: " << this->describe_held();
        return std::runtime_error(oss.str());
    }

    std::unordered_map<std::string, std::vector<Slot>> slots_;
    std::map<Key, SlotRef> index_;
    // (pass, module name) whose copy in that pass the producer skips.
    std::set<std::pair<int64_t, std::string>> pending_skips_;
    // installed module name -> its type and position in the type's list.
    std::unordered_map<std::string, Installed> installed_;
    std::unordered_map<std::string, std::vector<std::string>> lists_;
    std::unordered_map<std::string, Cursor> cursors_;
    std::deque<int64_t> open_passes_;
    int64_t next_pass_ = 1;
    std::map<int64_t, std::map<std::string, PassCounts>> pass_counts_;
};
