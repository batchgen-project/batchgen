"""CPU-only: the GPU weight ring's slot state machine, compiled and run.

``core/GPU_Weight_Buffer/weight_slot_table.h`` is free of torch, CUDA and
pybind, so a small driver is built against it with the system C++ compiler.
The driver exercises every legal transition, every illegal one, routing skips
on each state, and the weight passes: numbering, the producer cursor, lease
and skip addressing the oldest open pass, the explicit carry, cancel, and the
conditions end_pass asserts.
"""

import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
HEADER_DIR = ROOT / "core" / "GPU_Weight_Buffer"

DRIVER = r"""
#include "weight_slot_table.h"

#include <cstdio>
#include <string>
#include <vector>

using S = WeightSlotState;
using P = WeightProduction;

static int failures = 0;

#define CHECK(cond)                                                       \
    do {                                                                  \
        if (!(cond)) {                                                    \
            std::fprintf(stderr, "line %d: CHECK failed: %s\n", __LINE__, \
                         #cond);                                          \
            ++failures;                                                   \
        }                                                                 \
    } while (0)

// The call must throw std::runtime_error whose message contains every needle.
template <typename F>
void expect_throw(int line, F&& call, std::vector<std::string> needles = {}) {
    try {
        call();
    } catch (const std::runtime_error& error) {
        const std::string message = error.what();
        for (const auto& needle : needles) {
            if (message.find(needle) == std::string::npos) {
                std::fprintf(stderr, "line %d: message lacks '%s': %s\n", line,
                             needle.c_str(), message.c_str());
                ++failures;
            }
        }
        return;
    }
    std::fprintf(stderr, "line %d: did not throw\n", line);
    ++failures;
}
#define EXPECT_THROW(call, ...) expect_throw(__LINE__, [&] { call; }, ##__VA_ARGS__)

static const std::string E = "routed_expert";
static const std::vector<std::string> EXPERTS = {"e0", "e1", "e2",
                                                 "e3", "e4", "e5"};

static WeightSlotTable make(int64_t slots) {
    WeightSlotTable table;
    table.reset_type(E, slots);
    table.install({{E, EXPERTS}});
    return table;
}

// Producer: reserve + claim + landed for the module at the type's cursor.
// Returns the slot, now PRESENT.
static int64_t produce(WeightSlotTable& table, const std::string& type = E) {
    const auto next = table.next_production(type);
    if (next.kind != P::kCopy) {
        std::fprintf(stderr, "produce(%s): cursor is not a copy\n",
                     type.c_str());
        ++failures;
        return -1;
    }
    const auto index = table.reserve(type);
    if (!index.has_value()) {
        std::fprintf(stderr, "produce(%s): no FREE slot\n",
                     next.module_name.c_str());
        ++failures;
        return -1;
    }
    CHECK(table.claim(type, *index, next.module_name) == P::kCopy);
    CHECK(table.pass_at(type, *index) == next.pass);
    if (!table.landed(type, *index)) {
        std::fprintf(stderr, "produce(%s): released on landing\n",
                     next.module_name.c_str());
        ++failures;
    }
    return *index;
}

// Producer: retire the pending skip at the cursor without a slot.
static void retire(WeightSlotTable& table, const std::string& type = E) {
    const auto next = table.next_production(type);
    CHECK(next.kind == P::kSkip);
    CHECK(table.claim(type, -1, next.module_name) == P::kSkip);
    table.retire_skip(type, next.module_name);
}

// Consumer: lease and release.
static void consume(WeightSlotTable& table, const std::string& name) {
    table.lease(name);
    table.release(name);
}

// One whole pass with produce and consume interleaved.
static void run_pass(WeightSlotTable& table) {
    for (const auto& name : EXPERTS) {
        CHECK(table.next_production(E).module_name == name);
        produce(table);
        consume(table, name);
    }
}

static void pass_numbering() {
    // Nothing installed: begin returns 0, and 0 ends and cancels as a no-op.
    WeightSlotTable fresh;
    CHECK(fresh.begin_pass() == 0);
    WeightSlotTable empty;
    empty.reset_type(E, 2);
    empty.install({{E, {}}});
    CHECK(empty.begin_pass() == 0);
    CHECK(!empty.current_pass().has_value());
    CHECK(empty.end_pass(0).pass == 0);
    CHECK(empty.cancel_pass(0).pending == 0);
    CHECK(empty.next_production(E).kind == P::kNone);

    WeightSlotTable table = make(2);
    CHECK(table.next_production("unknown_type").kind == P::kNone);
    CHECK(table.begin_pass() == 1);
    CHECK(table.begin_pass() == 2);
    EXPECT_THROW(table.begin_pass(),
                 {"Cannot begin weight pass 3", "[1, 2]", "at most 2"});
    CHECK(table.current_pass() == 1);
    CHECK(table.is_open(2));
    CHECK(table.open_passes() == std::vector<int64_t>({1, 2}));
    EXPECT_THROW(table.end_pass(2),
                 {"Cannot end weight pass 2", "not the oldest open pass",
                  "[1, 2]"});
    EXPECT_THROW(table.cancel_pass(3), {"weight pass 3", "not open"});
    // The layout and the tasks stay fixed while a pass is open.
    EXPECT_THROW(table.reset_type(E, 2),
                 {"Cannot reset the routed_expert slots",
                  "while weight passes [1, 2] are open"});
    EXPECT_THROW(table.require_no_open_pass("reset the ring"),
                 {"Cannot reset the ring"});
    EXPECT_THROW(table.install({{E, {"e0"}}}),
                 {"Cannot install weight copy tasks", "are open"});
}

static void producer_cursor_and_parking() {
    WeightSlotTable table = make(2);
    // No open pass: the type parks at the head of pass 1.
    auto next = table.next_production(E);
    CHECK(next.kind == P::kParked && next.pass == 1 && next.position == 0 &&
          next.module_name == "e0");
    CHECK(!table.producer_finished(1));
    CHECK(table.begin_pass() == 1);
    next = table.next_production(E);
    CHECK(next.kind == P::kCopy && next.pass == 1 && next.module_name == "e0");

    // A ring of 2 over a list of 6: two slots, then no FREE slot until the
    // consumer releases one.
    const int64_t first = produce(table);
    produce(table);
    CHECK(!table.reserve(E).has_value());
    CHECK(table.state_of("e0", 1) == S::kPresent);
    CHECK(table.pass_at(E, first) == 1);
    for (size_t i = 2; i < EXPERTS.size(); ++i) {
        consume(table, EXPERTS[i - 2]);
        produce(table);
    }
    consume(table, "e4");
    consume(table, "e5");

    // The end of the list moves the cursor to the head of pass 2, which is
    // not armed: the type parks instead of reading ahead.
    next = table.next_production(E);
    CHECK(next.kind == P::kParked && next.pass == 2 && next.position == 0 &&
          next.module_name == "e0");
    CHECK(table.producer_finished(1));
    const auto summary = table.end_pass(1);
    CHECK(summary.pass == 1);
    CHECK(summary.by_type.at(E).copied == 6);
    CHECK(summary.by_type.at(E).skipped == 0);
    CHECK(summary.by_type.at(E).released_on_landing == 0);
    CHECK(summary.carried == 0);
    CHECK(!table.current_pass().has_value());
    CHECK(table.held().empty());
    CHECK(table.pass_counts().empty());

    // The next pass reads the list in the same order.
    CHECK(table.begin_pass() == 2);
    run_pass(table);
    CHECK(table.end_pass(2).by_type.at(E).copied == 6);
}

static void claim_and_land() {
    WeightSlotTable table = make(3);
    table.begin_pass();
    const auto index = table.reserve(E);
    CHECK(index.has_value() && *index == 0);
    CHECK(table.state_at(E, 0) == S::kFilling);
    CHECK(table.name_at(E, 0).empty());
    // The claim must name the module at the cursor.
    EXPECT_THROW(table.claim(E, *index, "e1"),
                 {"Producer claimed e1", "routed_expert=(1, 0/6)"});
    EXPECT_THROW(table.claim(E, -1, "e0"), {"Invalid weight-buffer slot"});
    CHECK(table.claim(E, *index, "e0") == P::kCopy);
    CHECK(table.state_of("e0", 1) == S::kFilling);
    CHECK(table.name_at(E, 0) == "e0");
    // A named slot cannot be claimed again, and the cursor stays.
    EXPECT_THROW(table.claim(E, *index, "e1"), {"assign e1", "FILLING"});
    CHECK(table.next_production(E).module_name == "e1");
    EXPECT_THROW(table.retire_skip(E, "e1"), {"none is pending"});

    // landed: only a named FILLING slot.
    EXPECT_THROW(table.landed(E, 2), {"FREE"});
    const auto unnamed = table.reserve(E);
    EXPECT_THROW(table.landed(E, *unnamed), {"FILLING"});
    table.abandon(E, *unnamed);
    CHECK(table.state_at(E, *unnamed) == S::kFree);
    CHECK(table.landed(E, *index));
    EXPECT_THROW(table.landed(E, *index), {"e0", "PRESENT"});
    EXPECT_THROW(table.abandon(E, *index), {"e0", "PRESENT"});

    // An abandoned named fill leaves the module produced but not resident.
    const auto dropped = table.reserve(E);
    table.claim(E, *dropped, "e1");
    table.abandon(E, *dropped);
    CHECK(!table.state_of("e1", 1).has_value());
    EXPECT_THROW(table.check_lease("e1"), {"already consumed in pass 1"});

    // Out-of-range slots and unknown types.
    EXPECT_THROW(table.abandon(E, 99), {"Invalid weight-buffer slot"});
    EXPECT_THROW(table.abandon(E, -1), {"Invalid weight-buffer slot"});
    EXPECT_THROW(table.landed("unknown_type", 0),
                 {"Invalid weight-buffer slot"});
}

static void lease_and_release() {
    WeightSlotTable table = make(3);
    EXPECT_THROW(table.lease("e0"), {"Cannot lease e0", "no weight pass is open"});
    EXPECT_THROW(table.check_lease("e0"), {"no weight pass is open"});
    EXPECT_THROW(table.release("e0"),
                 {"Cannot release missing weight-buffer lease for e0"});
    table.begin_pass();
    EXPECT_THROW(table.check_lease("nope"),
                 {"nope", "not an installed weight copy task"});
    // Not produced yet, then FILLING: the consumer waits.
    CHECK(table.check_lease("e0") == WeightLeaseCheck::kWait);
    const auto index = table.reserve(E);
    table.claim(E, *index, "e0");
    CHECK(table.check_lease("e0") == WeightLeaseCheck::kWait);
    EXPECT_THROW(table.lease("e0"), {"e0", "FILLING"});
    table.landed(E, *index);
    CHECK(table.check_lease("e0") == WeightLeaseCheck::kReady);

    // PRESENT -> LEASED -> FREE.
    const auto lease = table.lease("e0");
    CHECK(lease.module_type == E && lease.index == *index);
    EXPECT_THROW(table.lease("e0"), {"e0", "LEASED", "Held:", "e0:LEASED@1"});
    EXPECT_THROW(table.abandon(E, *index), {"e0", "LEASED"});
    EXPECT_THROW(table.reclaim(E, *index), {"e0", "LEASED"});
    const auto released = table.release("e0");
    CHECK(released.module_type == E && released.index == *index);
    CHECK(table.state_at(E, *index) == S::kFree);
    // Consumed in this pass: an immediate throw, never a wait.
    EXPECT_THROW(table.check_lease("e0"),
                 {"Cannot lease e0", "already consumed in pass 1"});
    EXPECT_THROW(table.release("e0"),
                 {"Cannot release missing weight-buffer lease for e0",
                  "not resident"});

    // release refuses anything that is not LEASED.
    const int64_t present = produce(table);  // e1
    EXPECT_THROW(table.release("e1"),
                 {"Cannot release missing weight-buffer lease for e1",
                  "PRESENT"});
    CHECK(table.state_at(E, present) == S::kPresent);

    // Asynchronous: LEASED -> PENDING_RELEASE (unindexed) -> FREE, all or
    // nothing.
    table.lease("e1");
    const int64_t other = produce(table);  // e2
    EXPECT_THROW(table.release_async({"e1", "e2"}),
                 {"Cannot asynchronously release missing weight-buffer lease",
                  "e2", "PRESENT"});
    CHECK(table.state_of("e1", 1) == S::kLeased);
    EXPECT_THROW(table.release_async({"e1", "e1"}),
                 {"Duplicate module in asynchronous weight release: e1"});
    CHECK(table.state_of("e1", 1) == S::kLeased);
    const auto refs = table.release_async({"e1"});
    CHECK(refs.size() == 1 && refs[0].module_type == E &&
          refs[0].index == present);
    CHECK(table.state_at(E, present) == S::kPendingRelease);
    CHECK(table.pass_at(E, present) == 1);
    CHECK(table.name_at(E, present) == "e1");
    CHECK(!table.state_of("e1", 1).has_value());
    CHECK(table.count_in_pass(1, S::kPendingRelease) == 1);
    EXPECT_THROW(table.check_lease("e1"), {"already consumed in pass 1"});
    EXPECT_THROW(table.reclaim(E, other), {"PRESENT"});
    table.reclaim(E, present);
    CHECK(table.state_at(E, present) == S::kFree);
    EXPECT_THROW(table.reclaim(E, present), {"FREE"});
}

static void skips() {
    WeightSlotTable table = make(3);
    EXPECT_THROW(table.skip("e0"), {"Cannot skip e0", "no weight pass is open"});
    table.begin_pass();

    // PRESENT -> FREE at once; afterwards it is consumed for this pass.
    const int64_t present = produce(table);  // e0
    CHECK(table.skip("e0") == WeightSkipOutcome::kFreedPresent);
    CHECK(table.state_at(E, present) == S::kFree);
    CHECK(!table.state_of("e0", 1).has_value());
    EXPECT_THROW(table.skip("e0"), {"Cannot skip e0", "already consumed in pass 1"});
    EXPECT_THROW(table.check_lease("e0"), {"already consumed in pass 1"});

    // FILLING -> release-on-landing; a second skip and a lease throw.
    const auto filling = table.reserve(E);
    table.claim(E, *filling, "e1");
    CHECK(table.skip("e1") == WeightSkipOutcome::kReleaseOnLanding);
    CHECK(table.state_at(E, *filling) == S::kFilling);
    EXPECT_THROW(table.skip("e1"), {"e1", "FILLING"});
    EXPECT_THROW(table.check_lease("e1"), {"Cannot lease e1", "skipped in pass 1"});
    CHECK(!table.landed(E, *filling));
    CHECK(table.state_at(E, *filling) == S::kFree);
    CHECK(!table.state_of("e1", 1).has_value());

    // LEASED: the consumer holds it, so the skip throws and changes nothing.
    const int64_t leased = produce(table);  // e2
    table.lease("e2");
    EXPECT_THROW(table.skip("e2"), {"e2", "LEASED", "Held:"});
    CHECK(table.state_at(E, leased) == S::kLeased);
    table.release("e2");

    // Not produced -> pending; a duplicate and a lease of it throw.
    CHECK(table.skip("e3") == WeightSkipOutcome::kPending);
    CHECK(table.skip_pending("e3", 1));
    CHECK(!table.skip_pending("e3", 2));
    EXPECT_THROW(table.skip("e3"), {"Duplicate skip of e3", "pass 1"});
    EXPECT_THROW(table.check_lease("e3"), {"skipped in pass 1"});
    // check_skip throws exactly like skip, without changing state.
    EXPECT_THROW(table.check_skip("e3"), {"Duplicate skip of e3"});
    table.check_skip("e4");
    CHECK(!table.skip_pending("e4", 1));

    // The producer sees the skip at its cursor. The claim abandons the slot
    // it reserved and keeps the cursor until the skip is retired, i.e. until
    // the daemon lease of the module is released.
    const auto next = table.next_production(E);
    CHECK(next.kind == P::kSkip && next.module_name == "e3" && next.pass == 1);
    const auto slot = table.reserve(E);
    CHECK(table.claim(E, *slot, "e3") == P::kSkip);
    CHECK(table.state_at(E, *slot) == S::kFree);
    CHECK(table.next_production(E).module_name == "e3");
    CHECK(table.pending_skip_count(1) == 1);
    table.retire_skip(E, "e3");
    CHECK(!table.skip_pending("e3", 1));
    CHECK(table.pending_skip_count(1) == 0);
    CHECK(table.next_production(E).module_name == "e4");
    EXPECT_THROW(table.check_lease("e3"), {"already consumed in pass 1"});

    // Only installed modules can be skipped.
    EXPECT_THROW(table.skip("not_installed"),
                 {"not_installed", "not an installed weight copy task"});

    produce(table);
    consume(table, "e4");
    produce(table);
    consume(table, "e5");
    const auto summary = table.end_pass(1);
    // e0, e1 (released on landing), e2, e4 and e5 landed; e3 was retired.
    CHECK(summary.by_type.at(E).copied == 5);
    CHECK(summary.by_type.at(E).skipped == 1);
    CHECK(summary.by_type.at(E).released_on_landing == 1);
}

static void carry_is_keyed_by_pass() {
    // A ring larger than the list: the carried copy of a module coexists
    // with its copy in the current pass.
    WeightSlotTable table = make(8);
    CHECK(table.begin_pass() == 1);
    for (size_t i = 0; i < EXPERTS.size(); ++i) {
        produce(table);
    }
    CHECK(table.next_production(E).kind == P::kParked);
    CHECK(table.begin_pass() == 2);
    const int64_t carried = produce(table);  // e0 of pass 2
    CHECK(table.pass_at(E, carried) == 2);
    CHECK(table.state_of("e0", 1) == S::kPresent);
    CHECK(table.state_of("e0", 2) == S::kPresent);

    // Lease and skip address pass 1; pass 2's copy is untouched.
    const auto ref = table.lease("e0");
    CHECK(ref.index != carried);
    CHECK(table.state_of("e0", 2) == S::kPresent);
    table.release("e0");
    CHECK(table.skip("e1") == WeightSkipOutcome::kFreedPresent);
    CHECK(!table.state_of("e1", 2).has_value());

    // end_pass refuses while a slot of the pass is held, changing nothing.
    EXPECT_THROW(table.end_pass(1),
                 {"Cannot end weight pass 1", "e2 is PRESENT", "Held:"});
    CHECK(table.current_pass() == 1);
    for (const char* name : {"e2", "e3", "e4", "e5"}) {
        consume(table, name);
    }
    const auto summary = table.end_pass(1);
    CHECK(summary.carried == 1);
    CHECK(table.current_pass() == 2);
    // The carried copy now belongs to the pass the consumer executes.
    CHECK(table.check_lease("e0") == WeightLeaseCheck::kReady);
    consume(table, "e0");
    for (size_t i = 1; i < EXPERTS.size(); ++i) {
        produce(table);
        consume(table, EXPERTS[i]);
    }
    CHECK(table.end_pass(2).carried == 0);
}

static void cancel_the_current_pass() {
    WeightSlotTable table = make(3);
    table.begin_pass();
    const int64_t present = produce(table);  // e0
    produce(table);                          // e1
    table.lease("e1");
    const auto filling = table.reserve(E);
    table.claim(E, *filling, "e2");

    // A LEASED slot refuses the cancel, and nothing changes.
    EXPECT_THROW(table.cancel_pass(1),
                 {"Cannot cancel weight pass 1", "e1 is still LEASED"});
    CHECK(table.state_at(E, present) == S::kPresent);
    CHECK(table.pending_skip_count(1) == 0);
    table.release("e1");

    const auto counts = table.cancel_pass(1);
    CHECK(counts.freed_present == 1);       // e0
    CHECK(counts.release_on_landing == 1);  // e2
    CHECK(counts.pending == 3);             // e3, e4, e5
    CHECK(table.state_at(E, present) == S::kFree);
    // A second cancel finds nothing left to declare.
    const auto again = table.cancel_pass(1);
    CHECK(again.freed_present == 0 && again.release_on_landing == 0 &&
          again.pending == 0);
    EXPECT_THROW(table.check_lease("e3"), {"skipped in pass 1"});
    EXPECT_THROW(table.check_lease("e2"), {"skipped in pass 1"});

    // The pass ends only once the copy landed and every skip was retired.
    EXPECT_THROW(table.end_pass(1),
                 {"the producer has not finished it", "e2 is FILLING",
                  "3 skips are still pending"});
    CHECK(!table.landed(E, *filling));
    for (int i = 0; i < 3; ++i) {
        retire(table);
    }
    CHECK(table.producer_finished(1));
    const auto summary = table.end_pass(1);
    CHECK(summary.by_type.at(E).copied == 3);
    CHECK(summary.by_type.at(E).skipped == 3);
    CHECK(summary.by_type.at(E).released_on_landing == 1);
    CHECK(table.held().empty());
}

static void cancel_a_carried_pass() {
    WeightSlotTable table = make(2);
    CHECK(table.begin_pass() == 1);
    CHECK(table.begin_pass() == 2);
    // Pass 2 has not started: its whole list becomes pending skips, and
    // pass 1 keeps none.
    const auto counts = table.cancel_pass(2);
    CHECK(counts.pending == 6 && counts.freed_present == 0 &&
          counts.release_on_landing == 0);
    CHECK(table.pending_skip_count(2) == 6);
    CHECK(table.pending_skip_count(1) == 0);
    run_pass(table);
    // The producer continues into pass 2 but only retires skips.
    const auto next = table.next_production(E);
    CHECK(next.kind == P::kSkip && next.pass == 2);
    CHECK(table.end_pass(1).carried == 0);
    for (int i = 0; i < 6; ++i) {
        retire(table);
    }
    const auto summary = table.end_pass(2);
    CHECK(summary.by_type.at(E).skipped == 6);
    CHECK(summary.by_type.at(E).copied == 0);
    const auto parked = table.next_production(E);
    CHECK(parked.kind == P::kParked && parked.pass == 3);
    CHECK(table.begin_pass() == 3);
}

static void every_type_must_finish() {
    WeightSlotTable table;
    table.reset_type(E, 2);
    table.reset_type("attn", 1);
    table.install({{E, {"e0"}}, {"attn", {"a0", "a1"}}, {"shared", {}}});
    table.begin_pass();
    produce(table, E);
    consume(table, "e0");
    EXPECT_THROW(table.end_pass(1),
                 {"the producer has not finished it", "attn=(1, 0/2)"});
    produce(table, "attn");
    consume(table, "a0");
    produce(table, "attn");
    consume(table, "a1");
    const auto summary = table.end_pass(1);
    // A type without a list is not reported.
    CHECK(summary.by_type.size() == 2);
    CHECK(summary.by_type.at("attn").copied == 2);
    CHECK(summary.by_type.at(E).copied == 1);
    const auto cursors = table.cursors();
    CHECK(cursors.size() == 2);
    CHECK(cursors.at("attn").pass == 2 && cursors.at("attn").position == 0);
}

static void install_refusals_and_renumbering() {
    WeightSlotTable table = make(2);
    EXPECT_THROW(table.install({{E, {"x"}}, {"attn", {"x"}}}),
                 {"x", "installed for both"});
    EXPECT_THROW(table.install({{E, {"x", "x"}}}), {"x", "installed twice"});
    // A held slot would carry the previous queue into pass 1.
    const auto unnamed = table.reserve(E);
    EXPECT_THROW(table.install({{E, EXPERTS}}),
                 {"Cannot install weight copy tasks while slots are held"});
    table.abandon(E, *unnamed);

    CHECK(table.begin_pass() == 1);
    run_pass(table);
    table.end_pass(1);
    CHECK(table.begin_pass() == 2);
    table.cancel_pass(2);
    for (int i = 0; i < 6; ++i) {
        retire(table);
    }
    table.end_pass(2);
    // A reinstall restarts numbering at 1 with the cursor at the head.
    table.install({{E, EXPERTS}});
    const auto next = table.next_production(E);
    CHECK(next.kind == P::kParked && next.pass == 1 && next.position == 0);
    CHECK(table.begin_pass() == 1);
}

static void queries_and_layout() {
    WeightSlotTable table = make(4);
    table.reset_type("attn", 1);
    table.begin_pass();
    produce(table);  // e0
    produce(table);  // e1
    table.lease("e1");
    const auto filling = table.reserve(E);
    table.claim(E, *filling, "e2");

    CHECK(table.slot_count(E) == 4);
    CHECK(table.count(E, S::kFree) == 1);
    CHECK(table.count(E, S::kPresent) == 1);
    CHECK(table.count(E, S::kLeased) == 1);
    CHECK(table.count(E, S::kFilling) == 1);
    CHECK(table.count(E, S::kPendingRelease) == 0);
    CHECK(table.count_in_pass(1, S::kPresent) == 1);
    CHECK(table.count_in_pass(1, S::kFilling) == 1);
    CHECK(table.count_in_pass(2, S::kPresent) == 0);
    CHECK(table.count("attn", S::kFree) == 1);
    CHECK(table.module_types() == std::vector<std::string>({"attn", E}));
    const auto held = table.held();
    CHECK(held.size() == 3);
    CHECK(held[0].pass == 1 && held[0].module_name == "e0");
    std::string described = table.describe_held();
    CHECK(described.find("routed_expert[0]=e0:PRESENT@1") != std::string::npos);
    CHECK(described.find("e1:LEASED@1") != std::string::npos);
    CHECK(described.find("e2:FILLING@1") != std::string::npos);
    const auto unnamed = table.reserve(E);
    CHECK(table.describe_held().find("<unnamed>:FILLING") != std::string::npos);
    table.abandon(E, *unnamed);
    const std::string passes = table.describe_passes();
    CHECK(passes.find("open passes [1]") != std::string::npos);
    CHECK(passes.find("routed_expert=(1, 3/6)") != std::string::npos);
    CHECK(table.pass_counts().at(1).at(E).copied == 2);

    // grow keeps every existing state; a reset waits for every pass to end.
    table.grow_type(E, 6);
    CHECK(table.slot_count(E) == 6);
    CHECK(table.count(E, S::kFree) == 3);
    CHECK(table.state_of("e1", 1) == S::kLeased);
    table.grow_type(E, 2);
    CHECK(table.slot_count(E) == 6);
    EXPECT_THROW(table.reset_type(E, 3), {"Cannot reset the routed_expert slots", "[1]"});
    CHECK(table.state_of("e1", 1) == S::kLeased);

    // Between passes nothing is produced, so a reset is allowed.
    WeightSlotTable idle = make(3);
    CHECK(idle.next_production(E).kind == P::kParked);
    idle.reset_type(E, 2);
    CHECK(idle.slot_count(E) == 2);
    CHECK(idle.held().empty());

    CHECK(std::string(weight_slot_state_name(S::kFree)) == "FREE");
    CHECK(std::string(weight_slot_state_name(S::kPendingRelease)) ==
          "PENDING_RELEASE");
}

int main() {
    pass_numbering();
    producer_cursor_and_parking();
    claim_and_land();
    lease_and_release();
    skips();
    carry_is_keyed_by_pass();
    cancel_the_current_pass();
    cancel_a_carried_pass();
    every_type_must_finish();
    install_refusals_and_renumbering();
    queries_and_layout();
    if (failures != 0) {
        std::fprintf(stderr, "%d failures\n", failures);
        return 1;
    }
    std::printf("OK\n");
    return 0;
}
"""


def test_weight_slot_table_state_machine(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("no system C++ compiler")
    source = tmp_path / "weight_slot_table_driver.cpp"
    source.write_text(DRIVER)
    binary = tmp_path / "weight_slot_table_driver"
    build = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-Wall",
            "-I",
            str(HEADER_DIR),
            str(source),
            "-o",
            str(binary),
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert build.returncode == 0, build.stderr
    run = subprocess.run(
        [str(binary)], capture_output=True, text=True, timeout=60
    )
    assert run.returncode == 0, run.stdout + run.stderr
    assert run.stdout.strip() == "OK"


def test_weight_slot_table_header_is_free_of_torch_cuda_and_pybind():
    header = (HEADER_DIR / "weight_slot_table.h").read_text()
    includes = [
        line for line in header.splitlines() if line.startswith("#include")
    ]
    for line in includes:
        assert "torch" not in line
        assert "cuda" not in line
        assert "pybind" not in line
