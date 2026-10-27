"""CPU-only: the GPU weight ring's slot state machine, compiled and run.

``core/GPU_Weight_Buffer/weight_slot_table.h`` is free of torch, CUDA and
pybind, so a small driver is built against it with the system C++ compiler.
The driver exercises every legal transition, every illegal one, routing skips
on each state, and that eviction never touches a slot someone still owns.
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
#include <unordered_set>
#include <vector>

using S = WeightSlotState;

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

static WeightSlotTable make(int64_t slots) {
    WeightSlotTable table;
    table.reset_type(E, slots);
    table.reset_type("attn", 1);
    table.install({{E, {"e0", "e1", "e2", "e3", "e4", "e5"}},
                   {"attn", {"a0", "a1"}}});
    return table;
}

// reserve + assign + landed: the module ends PRESENT.
static int64_t fill(WeightSlotTable& table, const std::string& name) {
    const auto index = table.reserve(E);
    if (!index.has_value()) {
        std::fprintf(stderr, "fill(%s): no FREE slot\n", name.c_str());
        ++failures;
        return -1;
    }
    table.assign(E, *index, name);
    if (!table.landed(E, *index)) {
        std::fprintf(stderr, "fill(%s): released on landing\n", name.c_str());
        ++failures;
    }
    return *index;
}

static void legal_lifecycle() {
    WeightSlotTable table = make(2);

    // Producer: FREE -> FILLING (unnamed, not indexed) -> named -> PRESENT.
    const auto index = table.reserve(E);
    CHECK(index.has_value() && *index == 0);
    CHECK(table.state_at(E, 0) == S::kFilling);
    CHECK(!table.state_of("e0").has_value());
    table.assign(E, 0, "e0");
    CHECK(table.state_of("e0") == S::kFilling);
    CHECK(table.name_at(E, 0) == "e0");
    CHECK(table.landed(E, 0));
    CHECK(table.is_present("e0"));

    // Consumer: PRESENT -> LEASED -> FREE.
    const auto lease = table.lease("e0");
    CHECK(lease.module_type == E && lease.index == 0);
    CHECK(table.state_of("e0") == S::kLeased);
    CHECK(!table.is_present("e0"));
    const auto released = table.release("e0");
    CHECK(released.module_type == E && released.index == 0);
    CHECK(table.state_at(E, 0) == S::kFree);
    CHECK(!table.state_of("e0").has_value());

    // Asynchronous: LEASED -> PENDING_RELEASE (unindexed) -> FREE.
    const int64_t slot = fill(table, "e1");
    table.lease("e1");
    const auto refs = table.release_async({"e1"});
    CHECK(refs.size() == 1 && refs[0].module_type == E &&
          refs[0].index == slot);
    CHECK(table.state_at(E, slot) == S::kPendingRelease);
    CHECK(!table.state_of("e1").has_value());
    CHECK(table.name_at(E, slot) == "e1");
    // While its old slot drains, the module can be copied again.
    const int64_t refill = fill(table, "e1");
    CHECK(refill != slot);
    table.reclaim(E, slot);
    CHECK(table.state_at(E, slot) == S::kFree);
    CHECK(table.is_present("e1"));

    // Abandon: unnamed and named FILLING both return to FREE.
    WeightSlotTable other = make(2);
    const auto unnamed = other.reserve(E);
    other.abandon(E, *unnamed);
    CHECK(other.state_at(E, *unnamed) == S::kFree);
    const auto named = other.reserve(E);
    other.assign(E, *named, "e2");
    other.abandon(E, *named);
    CHECK(other.state_at(E, *named) == S::kFree);
    CHECK(!other.state_of("e2").has_value());

    // reserve() reports a full ring.
    WeightSlotTable full = make(1);
    fill(full, "e0");
    CHECK(!full.reserve(E).has_value());
    CHECK(!full.reserve("unknown_type").has_value());
}

static void illegal_transitions() {
    WeightSlotTable table = make(3);

    // assign: only an unnamed FILLING slot, and never a resident module.
    EXPECT_THROW(table.assign(E, 0, "e0"), {"e0", "FREE", "Held:"});
    const auto index = table.reserve(E);
    table.assign(E, *index, "e0");
    EXPECT_THROW(table.assign(E, *index, "e1"), {"FILLING"});
    const auto second = table.reserve(E);
    EXPECT_THROW(table.assign(E, *second, "e0"), {"e0", "FILLING"});
    table.abandon(E, *second);

    // landed: only a named FILLING slot.
    EXPECT_THROW(table.landed(E, 2), {"FREE"});
    const auto unnamed = table.reserve(E);
    EXPECT_THROW(table.landed(E, *unnamed), {"FILLING"});
    table.abandon(E, *unnamed);

    // lease: FILLING and unknown names throw; a second lease throws.
    EXPECT_THROW(table.lease("e0"), {"e0", "FILLING"});
    EXPECT_THROW(table.lease("nope"), {"nope", "not resident"});
    CHECK(table.landed(E, *index));
    EXPECT_THROW(table.landed(E, *index), {"e0", "PRESENT"});
    EXPECT_THROW(table.abandon(E, *index), {"e0", "PRESENT"});
    table.lease("e0");
    EXPECT_THROW(table.lease("e0"), {"e0", "LEASED", "Held:", "e0:LEASED"});
    EXPECT_THROW(table.abandon(E, *index), {"e0", "LEASED"});
    EXPECT_THROW(table.reclaim(E, *index), {"e0", "LEASED"});

    // release: PRESENT, FILLING and unknown names keep the historical prefix.
    const int64_t present = fill(table, "e1");
    EXPECT_THROW(table.release("e1"),
                 {"Cannot release missing weight-buffer lease for e1",
                  "PRESENT"});
    CHECK(table.state_at(E, present) == S::kPresent);
    EXPECT_THROW(table.release("nope"),
                 {"Cannot release missing weight-buffer lease for nope",
                  "not resident"});

    // release_async is all or nothing.
    EXPECT_THROW(table.release_async({"e0", "e1"}),
                 {"Cannot asynchronously release missing weight-buffer lease",
                  "e1", "PRESENT"});
    CHECK(table.state_of("e0") == S::kLeased);
    EXPECT_THROW(table.release_async({"e0", "e0"}),
                 {"Duplicate module in asynchronous weight release: e0"});
    CHECK(table.state_of("e0") == S::kLeased);

    // reclaim: only PENDING_RELEASE, and only once.
    EXPECT_THROW(table.reclaim(E, present), {"PRESENT"});
    const auto refs = table.release_async({"e0"});
    table.reclaim(E, refs[0].index);
    EXPECT_THROW(table.reclaim(E, refs[0].index), {"FREE"});

    // Out-of-range slots and unknown types.
    EXPECT_THROW(table.abandon(E, 99), {"Invalid weight-buffer slot"});
    EXPECT_THROW(table.abandon(E, -1), {"Invalid weight-buffer slot"});
    EXPECT_THROW(table.landed("unknown_type", 0),
                 {"Invalid weight-buffer slot"});

    // install refuses a module listed under two types.
    EXPECT_THROW(table.install({{E, {"x"}}, {"attn", {"x"}}}),
                 {"x", "installed for both"});
}

static void skips() {
    // PRESENT -> FREE at once.
    WeightSlotTable table = make(3);
    const int64_t present = fill(table, "e0");
    CHECK(table.skip("e0") == WeightSkipOutcome::kFreedPresent);
    CHECK(table.state_at(E, present) == S::kFree);
    CHECK(!table.state_of("e0").has_value());
    CHECK(!table.take_pending_skip("e0"));

    // FILLING -> release-on-landing; a second skip throws.
    const auto filling = table.reserve(E);
    table.assign(E, *filling, "e1");
    CHECK(table.skip("e1") == WeightSkipOutcome::kReleaseOnLanding);
    CHECK(table.state_at(E, *filling) == S::kFilling);
    EXPECT_THROW(table.skip("e1"), {"e1", "FILLING"});
    CHECK(!table.landed(E, *filling));
    CHECK(table.state_at(E, *filling) == S::kFree);
    CHECK(!table.state_of("e1").has_value());
    // The mark does not outlive the slot's fill.
    const int64_t again = fill(table, "e1");
    CHECK(table.is_present("e1"));
    table.lease("e1");
    table.release("e1");
    CHECK(table.state_at(E, again) == S::kFree);

    // An abandoned fill drops its release-on-landing mark too.
    const auto abandoned = table.reserve(E);
    table.assign(E, *abandoned, "e2");
    CHECK(table.skip("e2") == WeightSkipOutcome::kReleaseOnLanding);
    table.abandon(E, *abandoned);
    CHECK(table.skip("e2") == WeightSkipOutcome::kPending);
    CHECK(table.take_pending_skip("e2"));

    // LEASED: the consumer holds it, so the skip throws and changes nothing.
    const int64_t leased = fill(table, "e3");
    table.lease("e3");
    EXPECT_THROW(table.skip("e3"), {"e3", "LEASED", "Held:"});
    CHECK(table.state_at(E, leased) == S::kLeased);
    CHECK(!table.take_pending_skip("e3"));

    // Not resident -> pending; a duplicate pending skip throws; the producer
    // consumes it exactly once.
    CHECK(table.skip("e4") == WeightSkipOutcome::kPending);
    EXPECT_THROW(table.skip("e4"), {"Duplicate skip of e4"});
    CHECK(table.take_pending_skip("e4"));
    CHECK(!table.take_pending_skip("e4"));
    CHECK(table.skip("e4") == WeightSkipOutcome::kPending);

    // A module whose old slot is PENDING_RELEASE is not resident.
    const auto refs = table.release_async({"e3"});
    CHECK(table.skip("e3") == WeightSkipOutcome::kPending);
    table.reclaim(E, refs[0].index);

    // check_skip throws exactly like skip, without changing state.
    EXPECT_THROW(table.check_skip("e4"), {"Duplicate skip of e4"});
    table.check_skip("e5");
    CHECK(table.take_pending_skip("e4"));
    CHECK(!table.take_pending_skip("e5"));

    // Only installed modules can be skipped.
    EXPECT_THROW(table.skip("not_installed"),
                 {"not_installed", "not an installed weight copy task"});
    CHECK(!table.take_pending_skip("not_installed"));

    // A new install, or a reset, drops every pending skip.
    CHECK(table.skip("e5") == WeightSkipOutcome::kPending);
    CHECK(table.skip("a1") == WeightSkipOutcome::kPending);
    table.install({{E, {"e0", "e1", "e2", "e3", "e4", "e5"}},
                   {"attn", {"a0", "a1"}}});
    CHECK(!table.take_pending_skip("e5"));
    CHECK(!table.take_pending_skip("a1"));
    CHECK(table.skip("e5") == WeightSkipOutcome::kPending);
    table.reset_type(E, 3);
    CHECK(!table.take_pending_skip("e5"));
    CHECK(table.skip("a1") == WeightSkipOutcome::kPending);
    table.clear_pending_skips();
    CHECK(!table.take_pending_skip("a1"));
}

static void eviction() {
    WeightSlotTable table = make(6);
    const int64_t present = fill(table, "e0");
    const int64_t leased = fill(table, "e1");
    table.lease("e1");
    const auto filling = table.reserve(E);
    table.assign(E, *filling, "e2");
    const int64_t pending = fill(table, "e3");
    table.lease("e3");
    table.release_async({"e3"});
    const int64_t allowed = fill(table, "e4");
    const auto unnamed = table.reserve(E);

    const auto evicted =
        table.evict_present_outside(E, std::unordered_set<std::string>{"e4"});
    CHECK(evicted == std::vector<std::string>{"e0"});
    CHECK(table.state_at(E, present) == S::kFree);
    CHECK(!table.state_of("e0").has_value());
    CHECK(table.state_at(E, leased) == S::kLeased);
    CHECK(table.state_at(E, *filling) == S::kFilling);
    CHECK(table.state_at(E, pending) == S::kPendingRelease);
    CHECK(table.state_at(E, allowed) == S::kPresent);
    CHECK(table.state_at(E, *unnamed) == S::kFilling);

    // An empty allow-list still spares every owned slot.
    const auto rest = table.evict_present_outside(E, {});
    CHECK(rest == std::vector<std::string>{"e4"});
    CHECK(table.state_at(E, leased) == S::kLeased);
    CHECK(table.state_at(E, *filling) == S::kFilling);
    CHECK(table.state_at(E, pending) == S::kPendingRelease);
    CHECK(table.evict_present_outside("unknown_type", {}).empty());
}

static void queries_and_layout() {
    WeightSlotTable table = make(4);
    fill(table, "e0");
    fill(table, "e1");
    table.lease("e1");
    const auto filling = table.reserve(E);
    table.assign(E, *filling, "e2");

    CHECK(table.slot_count(E) == 4);
    CHECK(table.count(E, S::kFree) == 1);
    CHECK(table.count(E, S::kPresent) == 1);
    CHECK(table.count(E, S::kLeased) == 1);
    CHECK(table.count(E, S::kFilling) == 1);
    CHECK(table.count(E, S::kPendingRelease) == 0);
    CHECK(table.count("attn", S::kFree) == 1);
    CHECK(table.module_types() == std::vector<std::string>({"attn", E}));
    const auto held = table.held();
    CHECK(held.size() == 3);
    const std::string described = table.describe_held();
    CHECK(described.find("e0:PRESENT") != std::string::npos);
    CHECK(described.find("e1:LEASED") != std::string::npos);
    CHECK(described.find("e2:FILLING") != std::string::npos);

    // grow keeps every existing state; reset frees everything.
    table.grow_type(E, 6);
    CHECK(table.slot_count(E) == 6);
    CHECK(table.count(E, S::kFree) == 3);
    CHECK(table.state_of("e1") == S::kLeased);
    table.grow_type(E, 2);
    CHECK(table.slot_count(E) == 6);
    table.reset_type(E, 3);
    CHECK(table.slot_count(E) == 3);
    CHECK(table.count(E, S::kFree) == 3);
    CHECK(!table.state_of("e0").has_value());
    CHECK(!table.state_of("e1").has_value());
    CHECK(!table.state_of("e2").has_value());
    CHECK(table.held().empty());

    CHECK(std::string(weight_slot_state_name(S::kFree)) == "FREE");
    CHECK(std::string(weight_slot_state_name(S::kPendingRelease)) ==
          "PENDING_RELEASE");
}

int main() {
    legal_lifecycle();
    illegal_transitions();
    skips();
    eviction();
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
