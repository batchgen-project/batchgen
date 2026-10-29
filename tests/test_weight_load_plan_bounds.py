"""CPU-only: the parameter server's weight-load plan, compiled and run.

``core/Parameter_Server/weight_load_plan.h`` is free of torch, CUDA, pybind
and JSON, so a small driver is built against it with the system C++ compiler.
The driver replays the measured 2026-10-28 GPT-OSS overrun (23 files loaded,
the 24th, 10,488,712,704 bytes, past the 75,161,927,680-byte buffer) and checks
that the plan refuses it before any read, that a fitting plan passes with 4 KiB
aligned offsets, that a tensor outside its file is refused, and the single-read
bound at its edge.
"""

import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
HEADER_DIR = ROOT / "core" / "Parameter_Server"
SOURCE = ROOT / "core" / "Parameter_Server" / "Parameter_Server.cpp"

DRIVER = r"""
#include "weight_load_plan.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using batchgen::WeightFileExtent;
using batchgen::WeightLoadPlanError;

static int failures = 0;
#define CHECK(cond)                                                    \
    do {                                                               \
        if (!(cond)) {                                                 \
            std::fprintf(stderr, "line %d: %s\n", __LINE__, #cond);    \
            ++failures;                                                \
        }                                                              \
    } while (0)

static WeightFileExtent file(const std::string& name, int64_t size) {
    return WeightFileExtent{name + ".json", name + ".bin", size, {{name + ".w", 0, size}}};
}

static std::string plan_error(const std::vector<WeightFileExtent>& files, int64_t buffer) {
    try {
        batchgen::plan_weight_load(files, buffer);
    } catch (const WeightLoadPlanError& e) {
        return e.what();
    }
    return "";
}

static void replay_measured_overrun() {
    const int64_t layer = 1748118784, non_layer = 2316539520;
    const int64_t shard1 = 10544030848, shard = 10488712704;
    const int64_t buffer = 70LL * 1024 * 1024 * 1024;  // 75161927680
    // Loader order and sizes from the S0 run 1 log.
    std::vector<WeightFileExtent> files = {
        file("layer_22", layer), file("layer_8", layer), file("layer_26", layer),
        file("layer_31", layer), file("layer_9", layer), file("layer_13", layer),
        file("layer_16", layer), file("non_layer", non_layer), file("layer_24", layer),
        file("layer_23", layer), file("model--00001-of-00007", shard1), file("layer_1", layer),
        file("layer_0", layer), file("layer_11", layer), file("layer_5", layer),
        file("layer_34", layer), file("model--00002-of-00007", shard),
        file("model--00003-of-00007", shard), file("model--00007-of-00007", non_layer),
        file("layer_4", layer), file("layer_20", layer), file("layer_18", layer),
        file("layer_32", layer), file("model--00004-of-00007", shard)};
    const std::string msg = plan_error(files, buffer);
    CHECK(msg.rfind("WeightLoadPlanError: ", 0) == 0);
    CHECK(msg.find("75161927680-byte weight buffer") != std::string::npos);
    CHECK(msg.find("first file past the bound: model--00004-of-00007.json") != std::string::npos);
    CHECK(msg.find("10488712704 bytes at buffer offset") != std::string::npos);
    // The first 23 files alone fit.
    files.pop_back();
    CHECK(plan_error(files, buffer).empty());
}

static void fitting_plan_is_aligned() {
    const std::vector<WeightFileExtent> files = {file("a", 5000), file("b", 1), file("c", 4096)};
    const auto starts = batchgen::plan_weight_load(files, 3 * 4096 + 4096);
    CHECK(starts.size() == 3);
    CHECK(starts[0] == 0);
    CHECK(starts[1] == 8192);
    CHECK(starts[2] == 12288);
    CHECK(batchgen::planned_weight_load_bytes(files) == 16384);
    // One byte short of the exact end fails, naming the last file.
    const std::string msg = plan_error(files, 16383);
    CHECK(msg.find("first file past the bound: c.json") != std::string::npos);
    CHECK(msg.find("planned weight load of 16384 bytes over 3 files") != std::string::npos);
    CHECK(plan_error({}, 4096).empty());
    CHECK(!plan_error({}, 0).empty());
}

static void tensor_outside_file_is_refused() {
    WeightFileExtent f{"t.json", "t.bin", 100, {{"w", 64, 64}}};
    const std::string msg = plan_error({f}, 1 << 20);
    CHECK(msg.find("tensor w in t.json spans [64, 128)") != std::string::npos);
    WeightFileExtent neg{"n.json", "n.bin", -1, {}};
    CHECK(plan_error({neg}, 1 << 20).find("negative total_byte_size") != std::string::npos);
}

static void single_read_bound() {
    bool threw = false;
    batchgen::check_weight_read_bound(4096, 4096, 8192, "x.bin");
    try {
        batchgen::check_weight_read_bound(4096, 4097, 8192, "x.bin");
    } catch (const WeightLoadPlanError&) {
        threw = true;
    }
    CHECK(threw);
}

int main() {
    replay_measured_overrun();
    fitting_plan_is_aligned();
    tensor_outside_file_is_refused();
    single_read_bound();
    if (failures != 0) {
        std::fprintf(stderr, "%d failures\n", failures);
        return 1;
    }
    std::printf("OK\n");
    return 0;
}
"""


def test_weight_load_plan_bounds(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("no system C++ compiler")
    source = tmp_path / "weight_load_plan_driver.cpp"
    source.write_text(DRIVER)
    binary = tmp_path / "weight_load_plan_driver"
    build = subprocess.run(
        [compiler, "-std=c++17", "-Wall", "-Werror", "-I", str(HEADER_DIR),
         str(source), "-o", str(binary)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert build.returncode == 0, build.stderr
    run = subprocess.run([str(binary)], capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stdout + run.stderr
    assert run.stdout.strip() == "OK"


def test_weight_load_plan_header_is_free_of_torch_cuda_pybind_and_json():
    header = (HEADER_DIR / "weight_load_plan.h").read_text()
    includes = [line for line in header.splitlines() if line.startswith("#include")]
    assert includes
    for line in includes:
        assert "<" in line, line  # standard headers only
        for banned in ("torch", "cuda", "pybind", "json"):
            assert banned not in line, line


def test_loader_plans_before_reading_and_bounds_each_read():
    source = SOURCE.read_text()
    body = source.split("void Parameter_Server::_load_cus_format_file_to_host_mem(", 1)[1]
    body = body.split("void Parameter_Server::Init(", 1)[0]
    plan_at = body.index("batchgen::plan_weight_load(extents, buffer_bytes)")
    assert plan_at < body.index("pread(")
    assert plan_at < body.index("tail_file.read(")
    assert body.count("batchgen::check_weight_read_bound(") == 2
    assert "_load_cus_format_file_to_host_mem(model_weights_path, weight_ptr, byte_size," in source
