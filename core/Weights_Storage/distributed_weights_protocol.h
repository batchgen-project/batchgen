#pragma once

#include <cstddef>
#include <cstdint>

namespace batchgen::distributed_weights {

constexpr std::uint64_t kProtocolMagic = 0x4b33523457454947ULL;
// Version 2: a kHello carries the worker cohort epoch in Request::generation;
// the daemon admits only the epoch it currently serves.
// Version 3: a kAcquire waits one bounded slice and answers kStatusRetry when
// its module is not ready yet; the worker re-sends it.
// Version 4: Request::flags. A kAcquire with kAcquireNoWait waits no slice:
// it answers kStatusRetry at once when its module is not ready.
constexpr std::uint32_t kProtocolVersion = 4;
constexpr std::size_t kModuleKeyBytes = 64;
constexpr std::size_t kErrorBytes = 192;

// Response::status: 0 success, negative an error that ends the connection,
// kStatusRetry an acquire whose module was not ready within one slice, or at
// once for kAcquireNoWait (nothing was acquired).
constexpr std::int32_t kStatusRetry = 1;

// Request::flags bits.
constexpr std::uint32_t kAcquireNoWait = 1;

enum class Operation : std::uint32_t {
    kHello = 1,
    kAcquire = 2,
    kRelease = 3,
};

struct Request {
    std::uint64_t magic = kProtocolMagic;
    std::uint32_t version = kProtocolVersion;
    std::uint32_t operation = 0;
    std::uint32_t worker_id = 0;
    std::uint32_t flags = 0;
    std::uint64_t generation = 0;
    char module_key[kModuleKeyBytes] = {};
};

struct Response {
    std::uint64_t magic = kProtocolMagic;
    std::uint32_t version = kProtocolVersion;
    std::int32_t status = 0;
    std::int32_t slot = -1;
    std::uint32_t reserved = 0;
    std::uint64_t generation = 0;
    std::uint64_t staging_bytes = 0;
    std::uint64_t module_bytes = 0;
    char error[kErrorBytes] = {};
};

}  // namespace batchgen::distributed_weights
