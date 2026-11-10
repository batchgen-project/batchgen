"""CPU-only checks for the torch-free replica plan header."""

import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HEADER_DIR = ROOT / "core" / "GPU_Weight_Buffer"

DRIVER = r"""
#include "weight_replica_plan.h"

#include <cstdio>
#include <map>
#include <string>
#include <vector>

static int failures = 0;

#define CHECK(condition)                                                   \
    do {                                                                   \
        if (!(condition)) {                                                \
            std::fprintf(stderr, "line %d: CHECK failed: %s\n", __LINE__, \
                         #condition);                                      \
            ++failures;                                                    \
        }                                                                    \
    } while (0)

template <typename F>
void expect_throw(int line, F&& call, const char* needle) {
    try {
        call();
    } catch (const std::exception& error) {
        if (std::string(error.what()).find(needle) == std::string::npos) {
            std::fprintf(stderr, "line %d: message lacks '%s': %s\n", line,
                         needle, error.what());
            ++failures;
        }
        return;
    }
    std::fprintf(stderr, "line %d: did not throw\n", line);
    ++failures;
}

int main() {
    const std::vector<int64_t> group = {3, 5, 7, 9};
    CHECK(source_owner(group, WeightModuleKind::kRoutedExpert, 0, 0, 8) == 3);
    CHECK(source_owner(group, WeightModuleKind::kRoutedExpert, 0, 3, 8) == 5);
    CHECK(source_owner(group, WeightModuleKind::kRoutedExpert, 0, 7, 8) == 9);
    CHECK(source_owner(group, "attn", 0, 0, 8) == 3);
    CHECK(source_owner(group, "shared_expert", 1, 0, 8) == 5);
    CHECK(source_owner(group, "attn", 4, 0, 8) == 3);
    expect_throw(__LINE__, [&] {
        source_owner(group, WeightModuleKind::kRoutedExpert, 0, 0, 10);
    }, "not divisible");

    const std::map<std::string, int64_t> owners = {
        {"attn0", 0}, {"attn1", 1}, {"private", 0},
    };
    const std::map<std::string, std::vector<int64_t>> readers = {
        {"attn0", {0, 1}}, {"attn1", {0, 1}}, {"private", {0}},
    };
    const auto rank0 = install_record_kinds({0, 1}, 0, owners, readers);
    CHECK(rank0.size() == 3);
    CHECK(rank0[0].kind == WeightRecordKind::HostShared);
    CHECK(rank0[1].kind == WeightRecordKind::Replica);
    CHECK(rank0[2].kind == WeightRecordKind::HostPrivate);
    const auto rank1 = install_record_kinds({0, 1}, 1, owners, readers);
    CHECK(rank1[0].kind == WeightRecordKind::Replica);
    CHECK(rank1[1].kind == WeightRecordKind::HostShared);

    CHECK(flat_frame(2, 3, 4) == 7);
    const WeightFrameMap frames(4);
    CHECK(frames.at(2, 3).flat == 7);
    CHECK(ring_capacity_error("attn", 1, 1).find("attn") != std::string::npos);
    CHECK(ring_capacity_error("attn", 1, 1).find("1") != std::string::npos);
    CHECK(ring_capacity_error("attn", 2, 1).empty());
    expect_throw(__LINE__, [&] { require_ring_capacity("routed_expert", 8, 256); },
                 "routed_expert");

    WeightFrameAccounting accounting(rank0);
    CHECK(!accounting.may_post_barrier_a());
    CHECK(!accounting.may_post_barrier_b());
    accounting.mark_declined(1);
    CHECK(!accounting.may_post_barrier_a());
    accounting.mark_landed(0);
    accounting.mark_landed(2);
    CHECK(accounting.may_post_barrier_a());
    CHECK(accounting.may_post_barrier_b());

    if (failures != 0) {
        std::fprintf(stderr, "%d failures\n", failures);
        return 1;
    }
    std::printf("OK\n");
    return 0;
}
"""


def test_weight_replica_plan_header(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        import pytest

        pytest.skip("no system C++ compiler")
    source = tmp_path / "weight_replica_plan_driver.cpp"
    source.write_text(DRIVER)
    binary = tmp_path / "weight_replica_plan_driver"
    build = subprocess.run(
        [compiler, "-std=c++17", "-Wall", "-I", str(HEADER_DIR), str(source), "-o", str(binary)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert build.returncode == 0, build.stderr
    run = subprocess.run([str(binary)], capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stdout + run.stderr
    assert run.stdout.strip() == "OK"


def test_weight_replica_plan_header_is_torch_free():
    header = (HEADER_DIR / "weight_replica_plan.h").read_text()
    assert all(
        token not in header
        for token in ("#include <torch", "#include <cuda", "pybind")
    )

