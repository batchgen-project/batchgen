"""CPU-only: the node weight daemon's latency histogram, compiled and run.

``core/Weights_Storage/latency_histogram.h`` is free of torch, CUDA, pybind
and JSON, so a small driver is built against it with the system C++ compiler.
The driver checks the fixed size, the empty histogram, the stated percentile
bound on single samples, every bucket edge, known and random populations and
a long recording run, weighted records, and the out-of-range clamps.
"""

import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
HEADER_DIR = ROOT / "core" / "Weights_Storage"

DRIVER = r"""
#include "latency_histogram.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <type_traits>
#include <vector>

using H = LatencyHistogram;

static_assert(std::is_trivially_copyable<H>::value, "snapshot copy");
// The bucket array and three scalars, however much is recorded.
static_assert(sizeof(H) == sizeof(std::uint64_t) * (H::kBuckets + 3),
              "fixed size");
static_assert(H::kBuckets ==
                  (H::kMaxOctave - H::kMinOctave) * H::kSubBuckets + 2,
              "underflow, range, overflow");

static int failures = 0;

#define CHECK(cond)                                                       \
    do {                                                                  \
        if (!(cond)) {                                                    \
            std::fprintf(stderr, "line %d: CHECK failed: %s\n", __LINE__, \
                         #cond);                                          \
            ++failures;                                                   \
        }                                                                 \
    } while (0)

// The stated guarantee for an in-range exact quantile e:
// e <= result < e * ratio.
static void check_bound(int line, double e, double result) {
    if (!(e <= result && result < e * H::ratio())) {
        std::fprintf(stderr, "line %d: %.17g outside [%.17g, %.17g)\n", line,
                     result, e, e * H::ratio());
        ++failures;
    }
}
#define CHECK_BOUND(e, result) check_bound(__LINE__, e, result)

// Exact nearest-rank percentile of an ascending population.
static double exact(const std::vector<double>& sorted, int percent) {
    size_t rank = (static_cast<size_t>(percent) * sorted.size() + 99) / 100;
    if (rank < 1) {
        rank = 1;
    }
    return sorted[rank - 1];
}

static void check_population(int line, std::vector<double> values) {
    H histogram;
    for (double value : values) {
        histogram.record(value);
    }
    std::sort(values.begin(), values.end());
    const double p50 = histogram.percentile(50);
    const double p95 = histogram.percentile(95);
    const double p99 = histogram.percentile(99);
    check_bound(line, exact(values, 50), p50);
    check_bound(line, exact(values, 95), p95);
    check_bound(line, exact(values, 99), p99);
    if (!(p50 <= p95 && p95 <= p99)) {
        std::fprintf(stderr, "line %d: percentiles not monotone\n", line);
        ++failures;
    }
    if (histogram.count() != values.size() ||
        histogram.max() != values.back()) {
        std::fprintf(stderr, "line %d: count or max wrong\n", line);
        ++failures;
    }
}

static void empty() {
    H histogram;
    CHECK(histogram.count() == 0);
    CHECK(histogram.max() == 0.0);
    CHECK(histogram.percentile(0) == 0.0);
    CHECK(histogram.percentile(50) == 0.0);
    CHECK(histogram.percentile(100) == 0.0);
}

static void constants() {
    CHECK(H::low_edge() < 0.001);
    // Past the daemon's 300 s acquire timeout, in milliseconds.
    CHECK(H::high_edge() > 300000.0);
    CHECK(H::ratio() == std::exp2(1.0 / H::kSubBuckets));
}

static void single_samples() {
    const double mantissas[] = {1.0,    1.0001, 1.03,   1.0905, 1.17,
                                1.3333, 1.4142, 1.5,    1.618,  1.77,
                                1.9,    1.95,   1.9999};
    for (int octave = H::kMinOctave; octave < H::kMaxOctave; ++octave) {
        for (double mantissa : mantissas) {
            const double value = std::ldexp(mantissa, octave);
            if (value <= H::low_edge()) {
                continue;
            }
            H histogram;
            histogram.record(value);
            CHECK_BOUND(value, histogram.percentile(50));
            CHECK(histogram.percentile(50) == histogram.percentile(99));
            CHECK(histogram.percentile(0) == histogram.percentile(100));
            CHECK(histogram.count() == 1);
            CHECK(histogram.max() == value);
        }
    }
    // The top of the range is in range.
    H top;
    top.record(H::high_edge());
    CHECK(top.percentile(50) == H::high_edge());
}

static void every_edge_is_its_own_bucket_top() {
    // A value exactly on an edge belongs to the bucket that edge closes.
    for (int k = H::kMinOctave * H::kSubBuckets + 1;
         k <= H::kMaxOctave * H::kSubBuckets; ++k) {
        const double edge =
            std::exp2(static_cast<double>(k) / H::kSubBuckets);
        H histogram;
        histogram.record(edge);
        if (histogram.percentile(50) != edge) {
            std::fprintf(stderr, "edge %d: %.17g reported %.17g\n", k, edge,
                         histogram.percentile(50));
            ++failures;
        }
        // Just above the edge falls in the next bucket, or in overflow past
        // the top of the range.
        H above;
        above.record(std::nextafter(edge, 2.0 * edge));
        if (k < H::kMaxOctave * H::kSubBuckets) {
            CHECK(above.percentile(50) > edge);
        } else {
            CHECK(above.percentile(50) == H::high_edge());
        }
    }
}

static void weighted_equals_repeated() {
    H weighted;
    H repeated;
    weighted.record(3.25, 5);
    weighted.record(0.75, 3);
    for (int i = 0; i < 5; ++i) {
        repeated.record(3.25);
    }
    for (int i = 0; i < 3; ++i) {
        repeated.record(0.75);
    }
    CHECK(std::memcmp(&weighted, &repeated, sizeof(H)) == 0);
    CHECK(weighted.count() == 8);
    CHECK(weighted.sum() == 18.5);
    CHECK(weighted.max() == 3.25);
    // A zero weight records nothing.
    H unchanged = weighted;
    unchanged.record(9.0, 0);
    CHECK(std::memcmp(&weighted, &unchanged, sizeof(H)) == 0);
}

static void known_populations() {
    std::mt19937_64 random(12345);
    std::vector<double> linear;
    for (int i = 1; i <= 1000; ++i) {
        linear.push_back(0.37 * i);
    }
    std::shuffle(linear.begin(), linear.end(), random);
    check_population(__LINE__, linear);

    std::vector<double> integers;
    for (int i = 1; i <= 100; ++i) {
        integers.push_back(i);
    }
    std::shuffle(integers.begin(), integers.end(), random);
    check_population(__LINE__, integers);

    // Two clusters far apart: p50 in the low one, p95 and p99 in the high.
    std::vector<double> bimodal;
    for (int i = 0; i < 900; ++i) {
        bimodal.push_back(2.0 + 0.001 * i);
    }
    for (int i = 0; i < 100; ++i) {
        bimodal.push_back(4000.0 + 3.0 * i);
    }
    std::shuffle(bimodal.begin(), bimodal.end(), random);
    check_population(__LINE__, bimodal);
}

static void out_of_range() {
    H histogram;
    histogram.record(0.0);
    histogram.record(H::low_edge() / 2.0);
    histogram.record(H::low_edge());
    histogram.record(H::high_edge() * 1.5);
    histogram.record(1e9);
    CHECK(histogram.count() == 5);
    CHECK(histogram.max() == 1e9);
    // Ranks 1-3 are underflow, 4-5 overflow.
    CHECK(histogram.percentile(0) == H::low_edge());
    CHECK(histogram.percentile(50) == H::low_edge());
    CHECK(histogram.percentile(60) == H::low_edge());
    CHECK(histogram.percentile(61) == H::high_edge());
    CHECK(histogram.percentile(99) == H::high_edge());
    CHECK(histogram.percentile(100) == H::high_edge());
}

static void long_run() {
    std::mt19937_64 random(2026);
    std::uniform_real_distribution<double> exponent(
        H::kMinOctave + 1.0, H::kMaxOctave - 1.0);
    std::vector<double> values;
    values.reserve(1000000);
    H histogram;
    for (int i = 0; i < 1000000; ++i) {
        const double value = std::exp2(exponent(random));
        values.push_back(value);
        histogram.record(value);
    }
    const H copy = histogram;
    CHECK(std::memcmp(&copy, &histogram, sizeof(H)) == 0);
    std::sort(values.begin(), values.end());
    const double p50 = copy.percentile(50);
    const double p95 = copy.percentile(95);
    const double p99 = copy.percentile(99);
    CHECK_BOUND(exact(values, 50), p50);
    CHECK_BOUND(exact(values, 95), p95);
    CHECK_BOUND(exact(values, 99), p99);
    CHECK(p50 <= p95 && p95 <= p99);
    CHECK(copy.count() == values.size());
    CHECK(copy.max() == values.back());
}

int main() {
    empty();
    constants();
    single_samples();
    every_edge_is_its_own_bucket_top();
    weighted_equals_repeated();
    known_populations();
    out_of_range();
    long_run();
    if (failures != 0) {
        std::fprintf(stderr, "%d failures\n", failures);
        return 1;
    }
    std::printf("OK\n");
    return 0;
}
"""


def test_latency_histogram_bounds(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("no system C++ compiler")
    source = tmp_path / "latency_histogram_driver.cpp"
    source.write_text(DRIVER)
    binary = tmp_path / "latency_histogram_driver"
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


def test_latency_histogram_header_is_free_of_torch_cuda_pybind_and_json():
    header = (HEADER_DIR / "latency_histogram.h").read_text()
    includes = [
        line for line in header.splitlines() if line.startswith("#include")
    ]
    assert includes
    for line in includes:
        lowered = line.lower()
        assert "torch" not in lowered
        assert "cuda" not in lowered
        assert "pybind" not in lowered
        assert "nlohmann" not in lowered
