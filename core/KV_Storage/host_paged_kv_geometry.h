#ifndef HOST_PAGED_KV_GEOMETRY_H_
#define HOST_PAGED_KV_GEOMETRY_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "host_paged_kv_backend.h"

namespace batchgen::kv {

class HostPagedKVGeometry {
   public:
    explicit HostPagedKVGeometry(const HostPagedKVConfig& config)
        : config_(&config) {
        if (config_ == nullptr) {
            throw std::invalid_argument(
                "HostPagedKVGeometry requires a valid config reference");
        }
    }

    HostPagedKVGeometry(const HostPagedKVGeometry&) = default;
    HostPagedKVGeometry& operator=(const HostPagedKVGeometry&) = default;
    HostPagedKVGeometry(HostPagedKVGeometry&&) = default;
    HostPagedKVGeometry& operator=(HostPagedKVGeometry&&) = default;

    void Reset(const HostPagedKVConfig& config) { config_ = &config; }

    void EnsureLayerBounds(std::size_t layer_idx,
                           std::string_view context) const {
        if (layer_idx >= config_->num_layers) {
            throw std::out_of_range(ComposeBoundsMessage(
                "layer_idx", layer_idx, config_->num_layers, context));
        }
    }

    void EnsurePageBounds(std::int32_t page_idx,
                          std::string_view context) const {
        if (page_idx < 0 ||
            static_cast<std::size_t>(page_idx) >= config_->num_pages) {
            throw std::out_of_range(ComposeBoundsMessage(
                "page_idx", static_cast<std::size_t>(page_idx),
                config_->num_pages, context));
        }
    }

    [[nodiscard]] std::size_t KTokenBytes() const noexcept {
        return config_->num_k_heads * config_->k_head_dim *
               config_->k_element_size_bytes;
    }

    template <bool HasVCache>
    [[nodiscard]] std::size_t VTokenBytes() const {
        if constexpr (HasVCache) {
            return config_->num_v_heads * config_->v_head_dim *
                   config_->v_element_size_bytes;
        } else {
            throw std::logic_error("V cache is disabled for this layout");
        }
    }

    [[nodiscard]] std::size_t PageSizeTokens() const noexcept {
        return config_->page_size_tokens;
    }

    [[nodiscard]] std::size_t RequiredPages(std::size_t num_tokens) const {
        if (num_tokens == 0) {
            throw std::invalid_argument("num_tokens must be greater than zero");
        }
        const std::size_t tokens_per_page = PageSizeTokens();
        return (num_tokens + tokens_per_page - 1) / tokens_per_page;
    }

    void ValidatePageCapacity(const std::vector<std::int32_t>& pages,
                              std::size_t required_tokens,
                              std::string_view context) const {
        const std::size_t capacity = pages.size() * PageSizeTokens();
        if (required_tokens > capacity) {
            std::ostringstream oss;
            oss << context << ": page table capacity (" << capacity
                << ") insufficient for required tokens (" << required_tokens
                << ")";
            throw std::out_of_range(oss.str());
        }
    }

    [[nodiscard]] std::string DescribeBytes(const std::byte* ptr,
                                            std::size_t token_bytes,
                                            std::size_t max_bytes = 64) const {
        if (ptr == nullptr) {
            return "<null>";
        }
        if (token_bytes == 0) {
            return "<empty>";
        }
        const std::size_t bytes_to_log = std::min(token_bytes, max_bytes);
        std::ostringstream oss;
        oss << "0x" << std::uppercase << std::hex;
        for (std::size_t i = 0; i < bytes_to_log; ++i) {
            const auto value = static_cast<unsigned int>(
                std::to_integer<std::uint8_t>(ptr[i]));
            oss << std::setw(2) << std::setfill('0') << value;
            if (i + 1 != bytes_to_log) {
                oss << ' ';
            }
        }
        if (token_bytes > max_bytes) {
            oss << " ...";
        }
        return oss.str();
    }

   private:
    [[nodiscard]] std::string ComposeBoundsMessage(
        std::string_view field, std::size_t index, std::size_t limit,
        std::string_view context) const {
        std::ostringstream oss;
        if (!context.empty()) {
            oss << context << ": ";
        }
        oss << field << " " << index << " exceeds configured limit " << limit;
        return oss.str();
    }

    const HostPagedKVConfig* config_ = nullptr;
};

}  // namespace batchgen::kv

#endif  // HOST_PAGED_KV_GEOMETRY_H_
