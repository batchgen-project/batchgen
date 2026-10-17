#ifndef HOST_PAGED_KV_LAYOUT_H_
#define HOST_PAGED_KV_LAYOUT_H_

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
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

        // Intra-page geometry. k_head_dim is the per-token byte count in the
        // page-split format, so the two regions always sum back to the same
        // k_page_bytes_ the AoS format uses; only the OFFSET of a token's
        // bytes changes.
        if (config_.page_format == HostKVPageFormat::kPageSplitFp8Scale) {
            k_data_token_bytes_ =
                config_.k_head_dim - kHostKVScaleBytesPerToken;
            scale_token_bytes_ = kHostKVScaleBytesPerToken;
            scale_region_offset_ =
                config_.page_size_tokens * k_data_token_bytes_;
        } else {
            k_data_token_bytes_ = config_.num_k_heads * config_.k_head_dim *
                                  config_.k_element_size_bytes;
            scale_token_bytes_ = 0;
            scale_region_offset_ = 0;
        }

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

    HostKVPageFormat PageFormat() const { return config_.page_format; }

    bool IsPageSplit() const {
        return config_.page_format == HostKVPageFormat::kPageSplitFp8Scale;
    }

    // Bytes of ONE token's K data inside a page. Equal to the whole token
    // record in the AoS format; the e4m3 data bytes only (no scale) in the
    // page-split format.
    std::size_t KDataTokenBytes() const { return k_data_token_bytes_; }

    // Bytes of ONE token's scale inside a page; 0 unless page-split.
    std::size_t ScaleTokenBytes() const { return scale_token_bytes_; }

    // Byte offset of the scale region from the start of a K page.
    std::size_t ScaleRegionOffset() const { return scale_region_offset_; }

    // Address of token `page_offset_tokens` of this page's K data region.
    std::byte* KDataTokenAddress(std::byte* base, std::size_t layer_idx,
                                 std::int32_t page_idx,
                                 std::size_t page_offset_tokens) const {
        return KPageAddress(base, layer_idx, page_idx) +
               page_offset_tokens * k_data_token_bytes_;
    }

    const std::byte* KDataTokenAddress(const std::byte* base,
                                       std::size_t layer_idx,
                                       std::int32_t page_idx,
                                       std::size_t page_offset_tokens) const {
        return KPageAddress(base, layer_idx, page_idx) +
               page_offset_tokens * k_data_token_bytes_;
    }

    // Address of token `page_offset_tokens` of this page's scale region.
    // Only the page-split format has one.
    std::byte* ScaleTokenAddress(std::byte* base, std::size_t layer_idx,
                                 std::int32_t page_idx,
                                 std::size_t page_offset_tokens) const {
        RequirePageSplitScales();
        return KPageAddress(base, layer_idx, page_idx) + scale_region_offset_ +
               page_offset_tokens * scale_token_bytes_;
    }

    const std::byte* ScaleTokenAddress(const std::byte* base,
                                       std::size_t layer_idx,
                                       std::int32_t page_idx,
                                       std::size_t page_offset_tokens) const {
        RequirePageSplitScales();
        return KPageAddress(base, layer_idx, page_idx) + scale_region_offset_ +
               page_offset_tokens * scale_token_bytes_;
    }

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

    void RequirePageSplitScales() const {
        if (!IsPageSplit()) {
            throw std::logic_error(
                "HostPagedKVLayout: this page format has no scale region "
                "(page_format=" +
                ToString(config_.page_format) + ")");
        }
    }

    HostPagedKVConfig config_;
    std::size_t k_page_bytes_ = 0;
    std::size_t k_data_token_bytes_ = 0;
    std::size_t scale_token_bytes_ = 0;
    std::size_t scale_region_offset_ = 0;
    std::size_t v_page_bytes_ = 0;
    std::size_t layer_stride_bytes_ = 0;
    std::size_t data_section_bytes_ = 0;
    std::size_t v_layer_offset_ = 0;
    std::uint64_t fingerprint_ = 0;
};

}  // namespace batchgen::kv

#endif  // HOST_PAGED_KV_LAYOUT_H_
