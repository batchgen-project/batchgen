#ifndef HOST_PAGED_KV_LAYOUT_H_
#define HOST_PAGED_KV_LAYOUT_H_

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "host_paged_kv_backend.h"

namespace batchgen::kv {

enum class HostKVMode { kMHA, kMLA };

template <HostKVMode Mode>
struct HostKVModeTraits;

template <>
struct HostKVModeTraits<HostKVMode::kMHA> {
    static constexpr bool kHasVCache = true;

    static HostPagedKVConfig Adjust(HostPagedKVConfig config) {
        if (config.num_v_heads == 0) {
            config.num_v_heads = config.num_k_heads;
        }
        if (config.v_head_dim == 0) {
            config.v_head_dim = config.k_head_dim;
        }
        if (config.v_element_size_bytes == 0) {
            config.v_element_size_bytes = config.k_element_size_bytes;
        }
        return config;
    }
};

template <>
struct HostKVModeTraits<HostKVMode::kMLA> {
    static constexpr bool kHasVCache = false;

    static HostPagedKVConfig Adjust(HostPagedKVConfig config) {
        config.num_v_heads = 0;
        config.v_head_dim = 0;
        config.v_element_size_bytes = 0;
        return config;
    }
};

template <HostKVMode Mode>
class HostPagedKVLayout {
   public:
    using Traits = HostKVModeTraits<Mode>;
    static constexpr bool kHasVCache = Traits::kHasVCache;

    explicit HostPagedKVLayout(const HostPagedKVConfig& config)
        : config_(detail::SanitizeConfig(Traits::Adjust(config))) {
        k_page_bytes_ = config_.page_size_tokens * config_.num_k_heads *
                        config_.k_head_dim * config_.k_element_size_bytes;

        if constexpr (kHasVCache) {
            v_page_bytes_ = config_.page_size_tokens * config_.num_v_heads *
                            config_.v_head_dim * config_.v_element_size_bytes;
            v_layer_offset_ = Align(config_.num_pages * k_page_bytes_);
            layer_stride_bytes_ =
                v_layer_offset_ + config_.num_pages * v_page_bytes_;
        } else {
            v_page_bytes_ = 0;
            v_layer_offset_ = 0;
            layer_stride_bytes_ = Align(config_.num_pages * k_page_bytes_);
        }

        data_section_bytes_ = layer_stride_bytes_ * config_.num_layers;
        fingerprint_ = HashCombine(HashHostKVConfig(config_),
                                   static_cast<std::uint64_t>(Mode));
    }

    constexpr bool HasVCache() const { return kHasVCache; }
    std::size_t KPageBytes() const { return k_page_bytes_; }

    template <bool Enabled = kHasVCache, typename = std::enable_if_t<Enabled>>
    std::size_t VPageBytes() const {
        return v_page_bytes_;
    }

    std::size_t LayerStrideBytes() const { return layer_stride_bytes_; }
    std::size_t DataSectionBytes() const { return data_section_bytes_; }
    std::uint64_t Fingerprint() const { return fingerprint_; }

    std::byte* KPageAddress(std::byte* base, std::size_t layer_idx,
                            std::int32_t page_idx) const {
        const std::size_t layer_offset = layer_idx * layer_stride_bytes_;
        const std::size_t page_offset =
            static_cast<std::size_t>(page_idx) * k_page_bytes_;
        return base + layer_offset + page_offset;
    }

    const std::byte* KPageAddress(const std::byte* base, std::size_t layer_idx,
                                  std::int32_t page_idx) const {
        const std::size_t layer_offset = layer_idx * layer_stride_bytes_;
        const std::size_t page_offset =
            static_cast<std::size_t>(page_idx) * k_page_bytes_;
        return base + layer_offset + page_offset;
    }

    template <bool Enabled = kHasVCache, typename = std::enable_if_t<Enabled>>
    std::byte* VPageAddress(std::byte* base, std::size_t layer_idx,
                            std::int32_t page_idx) const {
        const std::size_t layer_offset = layer_idx * layer_stride_bytes_;
        const std::size_t page_offset =
            static_cast<std::size_t>(page_idx) * v_page_bytes_;
        return base + layer_offset + v_layer_offset_ + page_offset;
    }

    template <bool Enabled = kHasVCache, typename = std::enable_if_t<Enabled>>
    const std::byte* VPageAddress(const std::byte* base, std::size_t layer_idx,
                                  std::int32_t page_idx) const {
        const std::size_t layer_offset = layer_idx * layer_stride_bytes_;
        const std::size_t page_offset =
            static_cast<std::size_t>(page_idx) * v_page_bytes_;
        return base + layer_offset + v_layer_offset_ + page_offset;
    }

   private:
    std::size_t Align(std::size_t value) const {
        return detail::AlignUp(value, config_.alignment_bytes);
    }

    HostPagedKVConfig config_;
    std::size_t k_page_bytes_ = 0;
    std::size_t v_page_bytes_ = 0;
    std::size_t layer_stride_bytes_ = 0;
    std::size_t data_section_bytes_ = 0;
    std::size_t v_layer_offset_ = 0;
    std::uint64_t fingerprint_ = 0;
};

}  // namespace batchgen::kv

#endif  // HOST_PAGED_KV_LAYOUT_H_
