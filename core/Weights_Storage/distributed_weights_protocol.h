#pragma once

#include <cstddef>
#include <cstdint>

namespace batchgen::distributed_weights {

constexpr std::uint64_t kProtocolMagic = 0x4b33523457454947ULL;
constexpr std::uint32_t kProtocolVersion = 1;
constexpr std::size_t kModuleKeyBytes = 64;
constexpr std::size_t kErrorBytes = 192;

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
    std::uint32_t reserved = 0;
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
