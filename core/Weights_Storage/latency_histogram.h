#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <type_traits>

// Fixed-size latency histogram of the node weight daemon, kept free of torch,
// CUDA, pybind and JSON so it can be compiled and tested on its own. Not
// thread safe: the daemon records into it and copies it under stats_mutex.
//
// Buckets are log-spaced, kSubBuckets per octave, over the milliseconds range
// (2^kMinOctave, 2^kMaxOctave], from about a microsecond to past the daemon's
// 300 s acquire timeout. In-range bucket k covers
// (2^((k - 1) / kSubBuckets), 2^(k / kSubBuckets)]. One underflow bucket takes
// every value at or below 2^kMinOctave and one overflow bucket every value
// above 2^kMaxOctave. record() is O(1) and the size never grows, so a copy
// costs the same however long the daemon has run.
//
// counts() exposes the buckets by index and upper_edge(index) the edge that
// closes each one, so a consumer can export the histogram as it is; the
// overflow bucket has no finite upper edge, and upper_edge() reports
// 2^kMaxOctave for it, the clamp edge below.
//
// percentile() is nearest-rank over the buckets and returns the upper edge of
// the bucket holding the rank. For the exact nearest-rank quantile q of the
// recorded values, with ratio = 2^(1 / kSubBuckets):
//     result / ratio < q <= result
// An out-of-range q reports the clamp edge of its side: 2^kMinOctave for the
// underflow bucket, 2^kMaxOctave for the overflow bucket. max() is exact.
class LatencyHistogram {
public:
    static constexpr int kSubBuckets = 16;
    static constexpr int kMinOctave = -10;  // about 1 us
    static constexpr int kMaxOctave = 19;   // about 524 s
    static constexpr int kRangeBuckets =
        (kMaxOctave - kMinOctave) * kSubBuckets;
    // Underflow, the in-range buckets, overflow.
    static constexpr int kBuckets = kRangeBuckets + 2;

    static double low_edge() { return std::ldexp(1.0, kMinOctave); }
    static double high_edge() { return std::ldexp(1.0, kMaxOctave); }
    static double ratio() { return std::exp2(1.0 / kSubBuckets); }

    // Records weight samples of value ms.
    void record(double ms, std::uint64_t weight = 1) {
        if (weight == 0) {
            return;
        }
        counts_[bucket_of(ms)] += weight;
        count_ += weight;
        sum_ += ms * static_cast<double>(weight);
        max_ = std::max(max_, ms);
    }

    // Nearest-rank percentile, percent in [0, 100]. Empty: 0.0.
    double percentile(double percent) const {
        if (count_ == 0) {
            return 0.0;
        }
        const double wanted =
            std::ceil(percent * static_cast<double>(count_) / 100.0);
        std::uint64_t rank = count_;
        if (!(wanted >= 1.0)) {
            rank = 1;
        } else if (wanted < static_cast<double>(count_)) {
            rank = static_cast<std::uint64_t>(wanted);
        }
        std::uint64_t seen = 0;
        for (int index = 0; index < kBuckets; ++index) {
            seen += counts_[index];
            if (seen >= rank) {
                return upper_edge(index);
            }
        }
        return upper_edge(kBuckets - 1);
    }

    std::uint64_t count() const { return count_; }
    double sum() const { return sum_; }
    double max() const { return max_; }
    const std::array<std::uint64_t, kBuckets>& counts() const {
        return counts_;
    }

    // The edge that closes bucket index, index in [0, kBuckets).
    static double upper_edge(int index) {
        if (index == 0) {
            return low_edge();
        }
        if (index == kBuckets - 1) {
            return high_edge();
        }
        return edge(index + kMinOctave * kSubBuckets);
    }

private:
    // 2^(k / kSubBuckets); exact when k is a multiple of kSubBuckets.
    static double edge(int k) {
        return std::exp2(static_cast<double>(k) / kSubBuckets);
    }

    static int bucket_of(double ms) {
        if (!(ms > low_edge())) {
            return 0;
        }
        if (ms > high_edge()) {
            return kBuckets - 1;
        }
        int k = static_cast<int>(std::ceil(std::log2(ms) * kSubBuckets));
        // log2 may round across an edge; settle against edge() itself so
        // edge(k - 1) < ms <= edge(k) holds for the edges percentile reports.
        if (edge(k) < ms) {
            ++k;
        } else if (edge(k - 1) >= ms) {
            --k;
        }
        return k - kMinOctave * kSubBuckets;
    }

    std::array<std::uint64_t, kBuckets> counts_{};
    std::uint64_t count_ = 0;
    double sum_ = 0.0;
    double max_ = 0.0;
};

static_assert(std::is_trivially_copyable<LatencyHistogram>::value,
              "the daemon copies the histogram as a snapshot");
