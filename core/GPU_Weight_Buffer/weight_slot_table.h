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
//   FILLING          reserved by the producer; named once assign() runs
//   PRESENT          copied, not yet handed to a consumer
//   LEASED           handed to a consumer by lease()
//   PENDING_RELEASE  released by the consumer, reusable once its compute
//                    stream passes the recorded event (reclaim())
// The name index covers FILLING (named), PRESENT and LEASED slots only, so a
// module whose slot is PENDING_RELEASE may already be copied again.

#include <algorithm>
#include <cstdint>
#include <optional>
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
    kPending,           // not resident; the producer skips its next copy
};

class WeightSlotTable {
   public:
    struct SlotRef {
        std::string module_type;
        int64_t index = -1;
    };

    struct HeldSlot {
        std::string module_type;
        int64_t index = -1;
        std::string module_name;
        WeightSlotState state = WeightSlotState::kFree;
    };

    /* Layout */

    // Declare `count` FREE slots for `module_type`, dropping all of its state
    // and its pending skips.
    void reset_type(const std::string& module_type, int64_t count) {
        auto existing = this->slots_.find(module_type);
        if (existing != this->slots_.end()) {
            for (const Slot& slot : existing->second) {
                this->unindex(slot.name, module_type);
            }
        }
        this->slots_[module_type].assign(static_cast<size_t>(count), Slot{});
        this->pending_skips_[module_type].clear();
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

    // Record the installed copy tasks (module type -> module names). Only an
    // installed module can be skipped. Every pending skip is dropped: it
    // described the routing of the previous queue.
    void install(const std::unordered_map<std::string,
                                          std::vector<std::string>>& tasks) {
        std::unordered_map<std::string, std::string> module_types;
        for (const auto& [module_type, module_names] : tasks) {
            for (const auto& module_name : module_names) {
                auto inserted = module_types.emplace(module_name, module_type);
                if (!inserted.second && inserted.first->second != module_type) {
                    throw std::runtime_error(
                        "Weight copy task " + module_name +
                        " is installed for both " + inserted.first->second +
                        " and " + module_type);
                }
            }
        }
        this->installed_types_ = std::move(module_types);
        this->clear_pending_skips();
    }

    void clear_pending_skips() {
        for (auto& [module_type, names] : this->pending_skips_) {
            names.clear();
        }
    }

    /* Producer */

    // FREE -> FILLING. The slot carries no name until assign().
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

    // Name a reserved slot. The module must not be resident anywhere else.
    void assign(const std::string& module_type, int64_t index,
                const std::string& module_name) {
        Slot& slot = this->slot_at(module_type, index);
        if (slot.state != WeightSlotState::kFilling || !slot.name.empty()) {
            throw this->transition_error("assign", module_name, slot.state,
                                         module_type, index);
        }
        auto resident = this->index_.find(module_name);
        if (resident != this->index_.end()) {
            throw this->transition_error(
                "assign", module_name,
                this->slot_at(resident->second.module_type,
                              resident->second.index)
                    .state,
                resident->second.module_type, resident->second.index);
        }
        slot.name = module_name;
        this->index_[module_name] = SlotRef{module_type, index};
    }

    // FILLING -> PRESENT, or FILLING -> FREE when a skip marked the slot
    // release-on-landing. Returns whether the module is now PRESENT.
    bool landed(const std::string& module_type, int64_t index) {
        Slot& slot = this->slot_at(module_type, index);
        if (slot.state != WeightSlotState::kFilling || slot.name.empty()) {
            throw this->transition_error("land", slot.name, slot.state,
                                         module_type, index);
        }
        if (slot.release_on_landing) {
            this->index_.erase(slot.name);
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
        this->unindex(slot.name, module_type);
        slot = Slot{};
    }

    /* Consumer */

    // PRESENT -> LEASED.
    SlotRef lease(const std::string& module_name) {
        auto resident = this->index_.find(module_name);
        if (resident == this->index_.end()) {
            throw this->transition_error("lease", module_name, std::nullopt);
        }
        const SlotRef ref = resident->second;
        Slot& slot = this->slot_at(ref.module_type, ref.index);
        if (slot.state != WeightSlotState::kPresent) {
            throw this->transition_error("lease", module_name, slot.state,
                                         ref.module_type, ref.index);
        }
        slot.state = WeightSlotState::kLeased;
        return ref;
    }

    // LEASED -> FREE.
    SlotRef release(const std::string& module_name) {
        auto resident = this->index_.find(module_name);
        std::optional<WeightSlotState> state;
        if (resident != this->index_.end()) {
            state = this->slot_at(resident->second.module_type,
                                  resident->second.index)
                        .state;
        }
        if (!state.has_value() || *state != WeightSlotState::kLeased) {
            throw std::runtime_error(
                "Cannot release missing weight-buffer lease for " +
                module_name + " (state " + this->state_text(state) +
                "). Held: " + this->describe_held());
        }
        const SlotRef ref = resident->second;
        this->index_.erase(resident);
        this->slot_at(ref.module_type, ref.index) = Slot{};
        return ref;
    }

    // LEASED -> PENDING_RELEASE for every name, or for none when any name is
    // duplicated or not LEASED.
    std::vector<SlotRef> release_async(
        const std::vector<std::string>& module_names) {
        std::unordered_set<std::string> unique_names;
        std::vector<SlotRef> refs;
        refs.reserve(module_names.size());
        for (const auto& module_name : module_names) {
            if (!unique_names.insert(module_name).second) {
                throw std::runtime_error(
                    "Duplicate module in asynchronous weight release: " +
                    module_name);
            }
            auto resident = this->index_.find(module_name);
            std::optional<WeightSlotState> state;
            if (resident != this->index_.end()) {
                state = this->slot_at(resident->second.module_type,
                                      resident->second.index)
                            .state;
            }
            if (!state.has_value() || *state != WeightSlotState::kLeased) {
                throw std::runtime_error(
                    "Cannot asynchronously release missing weight-buffer "
                    "lease: " +
                    module_name + " (state " + this->state_text(state) +
                    "). Held: " + this->describe_held());
            }
            refs.push_back(resident->second);
        }
        for (size_t i = 0; i < module_names.size(); ++i) {
            this->index_.erase(module_names[i]);
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

    /* Routing-informed skip */

    // Throws exactly when skip(module_name) would, without changing state.
    void check_skip(const std::string& module_name) const {
        auto installed = this->installed_types_.find(module_name);
        if (installed == this->installed_types_.end()) {
            throw std::runtime_error(
                "Cannot skip " + module_name +
                ": it is not an installed weight copy task");
        }
        auto resident = this->index_.find(module_name);
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
        auto pending = this->pending_skips_.find(installed->second);
        if (pending != this->pending_skips_.end() &&
            pending->second.count(module_name) != 0) {
            throw std::runtime_error(
                "Duplicate skip of " + module_name +
                ": its next copy is already skipped. Held: " +
                this->describe_held());
        }
    }

    WeightSkipOutcome skip(const std::string& module_name) {
        this->check_skip(module_name);
        auto resident = this->index_.find(module_name);
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
        this->pending_skips_[this->installed_types_.at(module_name)].insert(
            module_name);
        return WeightSkipOutcome::kPending;
    }

    // Consume the pending skip of module_name; true when there was one.
    bool take_pending_skip(const std::string& module_name) {
        auto installed = this->installed_types_.find(module_name);
        if (installed == this->installed_types_.end()) {
            return false;
        }
        auto pending = this->pending_skips_.find(installed->second);
        return pending != this->pending_skips_.end() &&
               pending->second.erase(module_name) != 0;
    }

    /* Eviction */

    // Free every PRESENT slot of module_type whose module is not in
    // allowed_names. LEASED, FILLING and PENDING_RELEASE slots are never
    // touched. Returns the evicted names.
    std::vector<std::string> evict_present_outside(
        const std::string& module_type,
        const std::unordered_set<std::string>& allowed_names) {
        std::vector<std::string> evicted;
        auto found = this->slots_.find(module_type);
        if (found == this->slots_.end()) {
            return evicted;
        }
        for (Slot& slot : found->second) {
            if (slot.state == WeightSlotState::kPresent &&
                allowed_names.count(slot.name) == 0) {
                evicted.push_back(slot.name);
                this->index_.erase(slot.name);
                slot = Slot{};
            }
        }
        return evicted;
    }

    /* Queries */

    // State of the slot indexed for module_name (FILLING, PRESENT or LEASED),
    // or nullopt when the module is not resident.
    std::optional<WeightSlotState> state_of(
        const std::string& module_name) const {
        auto resident = this->index_.find(module_name);
        if (resident == this->index_.end()) {
            return std::nullopt;
        }
        return this->slot_at(resident->second.module_type,
                             resident->second.index)
            .state;
    }

    bool is_present(const std::string& module_name) const {
        const auto state = this->state_of(module_name);
        return state.has_value() && *state == WeightSlotState::kPresent;
    }

    WeightSlotState state_at(const std::string& module_type,
                             int64_t index) const {
        return this->slot_at(module_type, index).state;
    }

    std::string name_at(const std::string& module_type, int64_t index) const {
        return this->slot_at(module_type, index).name;
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
                                              slots[index].state});
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
                << "]=" << (slot.module_name.empty() ? "<unnamed>"
                                                     : slot.module_name)
                << ":" << weight_slot_state_name(slot.state);
            first = false;
        }
        oss << "]";
        return oss.str();
    }

   private:
    struct Slot {
        WeightSlotState state = WeightSlotState::kFree;
        std::string name;
        bool release_on_landing = false;
    };

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

    // Drop name from the index if it points into module_type.
    void unindex(const std::string& module_name,
                 const std::string& module_type) {
        if (module_name.empty()) {
            return;
        }
        auto resident = this->index_.find(module_name);
        if (resident != this->index_.end() &&
            resident->second.module_type == module_type) {
            this->index_.erase(resident);
        }
    }

    static std::string state_text(std::optional<WeightSlotState> state) {
        return state.has_value() ? weight_slot_state_name(*state)
                                 : "not resident";
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
    std::unordered_map<std::string, SlotRef> index_;
    // module type -> names whose next copy the producer skips.
    std::unordered_map<std::string, std::unordered_set<std::string>>
        pending_skips_;
    // installed module name -> module type.
    std::unordered_map<std::string, std::string> installed_types_;
};
