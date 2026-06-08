#ifndef HOST_PAGED_KV_BACKEND_H_
#define HOST_PAGED_KV_BACKEND_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace batchgen::kv {

using LayerMapping = std::vector<std::int32_t>;

struct HostPagedKVStats {
    std::size_t num_total_pages = 0;
    std::size_t num_free_pages = 0;
    std::size_t num_used_pages = 0;
    std::size_t num_active_sequences = 0;
    std::size_t sequence_table_capacity = 0;
    std::size_t total_bytes = 0;
};

struct HostPagedKVConfig {
    std::string shm_name;
    std::size_t num_layers = 0;
    std::size_t num_pages = 0;
    std::size_t page_size_tokens = 0;
    std::size_t num_k_heads = 0;
    std::size_t k_head_dim = 0;
    std::size_t num_v_heads = 0;
    std::size_t v_head_dim = 0;
    std::size_t k_element_size_bytes = 0;
    std::size_t v_element_size_bytes = 0;
    std::size_t sequence_table_capacity = 0;
    std::size_t alignment_bytes = 64;
    bool enable_memfd = false;
    int memfd_creator_pid = -1;
    int memfd_fd = -1;
    LayerMapping logical_to_physical_layer;
    std::string logger_name;  // Custom logger name (empty = use default)
};

inline std::uint64_t HashCombine(std::uint64_t seed, std::uint64_t value) {
    seed ^= value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
    return seed;
}

namespace detail {

inline HostPagedKVConfig SanitizeConfig(HostPagedKVConfig config) {
    if (config.sequence_table_capacity == 0) {
        config.sequence_table_capacity = config.num_pages;
    }
    if (config.alignment_bytes == 0) {
        config.alignment_bytes = 64;
    }
    std::vector<std::string> errors;
    if (config.num_layers == 0) {
        errors.emplace_back("num_layers must be > 0");
    }
    if (config.num_pages == 0) {
        errors.emplace_back("num_pages must be > 0");
    }
    if (config.page_size_tokens == 0) {
        errors.emplace_back("page_size_tokens must be > 0");
    }
    if (config.num_k_heads == 0) {
        errors.emplace_back("num_k_heads must be > 0");
    }
    if (config.k_head_dim == 0) {
        errors.emplace_back("k_head_dim must be > 0");
    }
    if (config.k_element_size_bytes == 0) {
        errors.emplace_back("k_element_size_bytes must be > 0");
    }
    for (std::size_t logical_layer = 0;
         logical_layer < config.logical_to_physical_layer.size();
         ++logical_layer) {
        const std::int32_t physical_layer =
            config.logical_to_physical_layer[logical_layer];
        if (physical_layer < -1) {
            std::ostringstream oss;
            oss << "logical_to_physical_layer[" << logical_layer
                << "] must be >= -1";
            errors.emplace_back(oss.str());
            continue;
        }
        if (config.num_layers > 0 && physical_layer >= 0 &&
            static_cast<std::size_t>(physical_layer) >= config.num_layers) {
            std::ostringstream oss;
            oss << "logical_to_physical_layer[" << logical_layer
                << "] physical layer id " << physical_layer
                << " must be < num_layers (" << config.num_layers << ")";
            errors.emplace_back(oss.str());
        }
    }
    if (!errors.empty()) {
        std::string message = "Invalid HostPagedKVConfig: ";
        for (std::size_t i = 0; i < errors.size(); ++i) {
            message.append(errors[i]);
            if (i + 1 < errors.size()) {
                message.append(", ");
            }
        }
        message.append(" (shm_name='" + config.shm_name + "')");
        throw std::invalid_argument(message);
    }
    if (config.num_v_heads == 0) {
        config.v_head_dim = 0;
        config.v_element_size_bytes = 0;
    } else {
        if (config.v_head_dim == 0) {
            config.v_head_dim = config.k_head_dim;
        }
        if (config.v_element_size_bytes == 0) {
            config.v_element_size_bytes = config.k_element_size_bytes;
        }
    }
    return config;
}

inline std::size_t AlignUp(std::size_t value, std::size_t alignment) {
    if (alignment == 0) {
        return value;
    }
    const std::size_t remainder = value % alignment;
    if (remainder == 0) {
        return value;
    }
    return value + (alignment - remainder);
}

}  // namespace detail

inline void AppendLogicalLayerMapping(std::ostringstream& oss,
                                      const LayerMapping& mapping) {
    oss << '[';
    constexpr std::size_t kMaxEntriesToPrint = 32;
    const std::size_t entries =
        mapping.size() < kMaxEntriesToPrint ? mapping.size()
                                            : kMaxEntriesToPrint;
    for (std::size_t i = 0; i < entries; ++i) {
        if (i != 0) {
            oss << ',';
        }
        oss << mapping[i];
    }
    if (mapping.size() > entries) {
        if (entries != 0) {
            oss << ',';
        }
        oss << "...(" << mapping.size() << " total)";
    }
    oss << ']';
}

inline std::string ToString(const HostPagedKVConfig& config) {
    std::ostringstream oss;
    oss << "HostPagedKVConfig(shm_name='" << config.shm_name
        << "', num_layers=" << config.num_layers
        << ", num_pages=" << config.num_pages
        << ", page_size_tokens=" << config.page_size_tokens
        << ", num_k_heads=" << config.num_k_heads
        << ", k_head_dim=" << config.k_head_dim
        << ", num_v_heads=" << config.num_v_heads
        << ", v_head_dim=" << config.v_head_dim
        << ", k_element_size_bytes=" << config.k_element_size_bytes
        << ", v_element_size_bytes=" << config.v_element_size_bytes
        << ", sequence_table_capacity=" << config.sequence_table_capacity
        << ", alignment_bytes=" << config.alignment_bytes
        << ", logical_to_physical_layer=";
    AppendLogicalLayerMapping(oss, config.logical_to_physical_layer);
    oss << ")";
    return oss.str();
}

inline std::string ToString(const HostPagedKVStats& stats) {
    std::ostringstream oss;
    oss << "HostPagedKVStats(total_pages=" << stats.num_total_pages
        << ", free_pages=" << stats.num_free_pages
        << ", used_pages=" << stats.num_used_pages
        << ", active_sequences=" << stats.num_active_sequences
        << ", sequence_table_capacity=" << stats.sequence_table_capacity
        << ", total_bytes=" << stats.total_bytes << ")";
    return oss.str();
}

inline std::uint64_t HashHostKVConfig(const HostPagedKVConfig& config) {
    const HostPagedKVConfig sanitized = detail::SanitizeConfig(config);
    std::uint64_t seed = 0;
    seed = HashCombine(seed, sanitized.num_layers);
    seed = HashCombine(seed, sanitized.num_pages);
    seed = HashCombine(seed, sanitized.page_size_tokens);
    seed = HashCombine(seed, sanitized.num_k_heads);
    seed = HashCombine(seed, sanitized.k_head_dim);
    seed = HashCombine(seed, sanitized.num_v_heads);
    seed = HashCombine(seed, sanitized.v_head_dim);
    seed = HashCombine(seed, sanitized.k_element_size_bytes);
    seed = HashCombine(seed, sanitized.v_element_size_bytes);
    seed = HashCombine(seed, sanitized.sequence_table_capacity);
    seed = HashCombine(seed, sanitized.alignment_bytes);
    seed = HashCombine(seed, static_cast<std::uint64_t>(sanitized.enable_memfd));
    // logical_to_physical_layer is view-level routing. It does not change the
    // physical shared-memory layout, so it is intentionally excluded here.
    return seed;
}

class HostPagedKVBackend {
   public:
    HostPagedKVBackend(HostPagedKVConfig config, std::size_t data_bytes,
                       std::uint64_t layout_fingerprint, bool has_v_cache);
    HostPagedKVBackend(const HostPagedKVBackend&) = delete;
    HostPagedKVBackend& operator=(const HostPagedKVBackend&) = delete;
    HostPagedKVBackend(HostPagedKVBackend&&) = delete;
    HostPagedKVBackend& operator=(HostPagedKVBackend&&) = delete;
    ~HostPagedKVBackend();

    void Initialize(bool create_region);

    std::vector<std::int32_t> AcquirePages(std::int64_t sequence_id,
                                           std::size_t num_pages);

    std::vector<std::vector<std::int32_t>> AcquirePagesForSequences(
        const std::vector<std::int64_t>& sequence_ids,
        const std::vector<std::size_t>& num_tokens);

    void ReleaseSequence(std::int64_t sequence_id);

    void ReleaseSequences(const std::vector<std::int64_t>& sequence_ids);

    std::vector<std::int32_t> ReleaseSequencePrefixPages(
        std::int64_t sequence_id, std::size_t num_pages);

    std::vector<std::int32_t> RetainSequencePrefixPages(
        std::int64_t sequence_id, std::size_t num_pages);

    std::vector<std::int32_t> RetainSequencePageRange(
        std::int64_t sequence_id, std::size_t start_page,
        std::size_t num_pages);

    std::vector<std::int32_t> RetainSequencePages(
        std::int64_t sequence_id,
        const std::vector<std::int32_t>& page_ids);

    void ReleaseResidentPages(const std::vector<std::int32_t>& page_ids);

    std::vector<std::int32_t> SequencePages(
        std::int64_t sequence_id, std::optional<std::size_t> max_pages) const;

    std::vector<std::int32_t> SequencePageRange(
        std::int64_t sequence_id, std::size_t start_page,
        std::size_t page_count) const;

    HostPagedKVStats CollectStats() const;

    std::byte* DataBase();
    const std::byte* DataBase() const;

    const HostPagedKVConfig& config() const { return config_; }
    bool has_v_cache() const { return has_v_cache_; }
    int memfd_fd() const;

   private:
    struct SharedState;

    HostPagedKVConfig config_;
    std::size_t data_bytes_ = 0;
    std::uint64_t layout_fingerprint_ = 0;
    bool has_v_cache_ = false;
    std::unique_ptr<SharedState> state_;
};

}  // namespace batchgen::kv

#endif  // HOST_PAGED_KV_BACKEND_H_
