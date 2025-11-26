#ifndef HOST_PAGED_KV_CONFIG_UTILS_H_
#define HOST_PAGED_KV_CONFIG_UTILS_H_

#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

#include "../data_structures.h"
#include "host_paged_kv_backend.h"

namespace batchgen::kv::config {
namespace detail {

template <typename T>
inline std::size_t RequirePositive(T value, std::string_view field_name) {
    if (value <= T{0}) {
        std::ostringstream oss;
        oss << field_name << " must be greater than zero";
        throw std::invalid_argument(oss.str());
    }
    return static_cast<std::size_t>(value);
}

inline std::size_t ResolveElementSizeBytes(std::string_view kv_dtype) {
    const std::string dtype =
        kv_dtype.empty() ? "bfloat16" : std::string(kv_dtype);
    if (dtype == "bfloat16" || dtype == "float16") {
        return 2;
    }
    if (dtype == "float32") {
        return 4;
    }
    if (dtype == "float8_e4m3fn" || dtype == "float8_e5m2") {
        return 1;
    }
    std::ostringstream oss;
    oss << "Unsupported kv_dtype='" << dtype << "'";
    throw std::invalid_argument(oss.str());
}

inline std::size_t DetermineSequenceTableCapacity(
    const KV_Storage_Config& storage_config, std::size_t fallback) {
    if (storage_config.num_host_slots > 0) {
        return static_cast<std::size_t>(storage_config.num_host_slots);
    }
    return fallback;
}

}  // namespace detail

inline HostPagedKVConfig BuildHostPagedKVConfig(
    const EngineConfig& engine_config, const ModelConfig& model_config) {
    static_cast<void>(model_config);

    HostPagedKVConfig config;
    const ::HostPagedKVConfig& external = engine_config.host_paged_kv_config;
    config.shm_name = external.shm_name;
    config.num_layers = detail::RequirePositive(
        external.num_layers, "Host_Paged_KV_Config.num_layers");
    const std::size_t pages_per_layer = detail::RequirePositive(
        external.num_pages_per_layer,
        "Host_Paged_KV_Config.num_pages_per_layer");
    config.num_pages = pages_per_layer;
    config.page_size_tokens = detail::RequirePositive(
        external.page_size, "Host_Paged_KV_Config.page_size");

    config.num_k_heads = detail::RequirePositive(
        external.num_k_heads, "Host_Paged_KV_Config.num_k_heads");
    config.k_head_dim = detail::RequirePositive(
        external.k_head_dim, "Host_Paged_KV_Config.k_head_dim");

    config.num_v_heads = external.num_v_heads;
    if (config.num_v_heads == 0) {
        config.v_head_dim = 0;
    } else {
        config.v_head_dim = detail::RequirePositive(
            external.v_head_dim, "Host_Paged_KV_Config.v_head_dim");
    }

    const std::size_t element_size_bytes =
        detail::ResolveElementSizeBytes(external.kv_dtype);
    config.k_element_size_bytes = element_size_bytes;
    config.v_element_size_bytes =
        config.num_v_heads == 0 ? 0 : element_size_bytes;
    config.sequence_table_capacity = detail::DetermineSequenceTableCapacity(
        engine_config.kv_storage_config, config.num_pages);
    config.alignment_bytes = 64;
    return config;
}

}  // namespace batchgen::kv::config

#endif  // HOST_PAGED_KV_CONFIG_UTILS_H_
