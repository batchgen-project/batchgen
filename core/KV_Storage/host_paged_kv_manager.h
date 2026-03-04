#ifndef HOST_PAGED_KV_MANAGER_H_
#define HOST_PAGED_KV_MANAGER_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "../data_structures.h"
#include "../utils.h"
#include "host_paged_kv_backend.h"
#include "host_paged_kv_config_utils.h"
#include "host_paged_kv_geometry.h"
#include "host_paged_kv_layout.h"
#include "spdlog/spdlog.h"

namespace batchgen::kv {

struct IdentityLayoutPolicy {
    static HostPagedKVConfig Normalize(HostPagedKVConfig config) {
        return config;
    }
};

struct MHALayoutPolicy {
    static HostPagedKVConfig Normalize(HostPagedKVConfig config) {
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

template <HostKVMode Mode, typename LayoutPolicy = IdentityLayoutPolicy,
          typename Layout = HostPagedKVLayout<Mode>>
class HostPagedKVManager {
   public:
    using ModeTraits = HostKVModeTraits<Mode>;
    static constexpr HostKVMode kMode = Mode;
    static constexpr bool kHasVCache = Layout::kHasVCache;

    // Construct the manager from engine and model configuration objects.
    // The EngineConfig and ModelConfig are stored as references members so
    // callers must ensure they outlive this manager.
    explicit HostPagedKVManager(const EngineConfig& engine_config,
                                const ModelConfig& model_config)
        : engine_config_(std::ref(const_cast<EngineConfig&>(engine_config))),
          model_config_(std::ref(const_cast<ModelConfig&>(model_config))),
          config_(detail::SanitizeConfig(ModeTraits::Adjust(
              LayoutPolicy::Normalize(config::BuildHostPagedKVConfig(
                  engine_config, model_config))))),
          geometry_(config_),
          layout_(config_),
          backend_(config_, layout_.DataSectionBytes(), layout_.Fingerprint(),
                   Layout::kHasVCache) {
        this->logger_ = init_logger(
            this->engine_config_.value().get().basic_config.log_level,
            "HostPagedKVManager");
        this->logger_->info(
            "Constructed HostPagedKVManager (shm_name={}, layers={}, pages={}, "
            "page_tokens={}, "
            "k_heads={}, head_dim={}, has_v_cache={}, seq_capacity={})",
            config_.shm_name, config_.num_layers, config_.num_pages,
            config_.page_size_tokens, config_.num_k_heads, config_.k_head_dim,
            Layout::kHasVCache, config_.sequence_table_capacity);
    }

    explicit HostPagedKVManager(const HostPagedKVConfig& config)
        : config_(detail::SanitizeConfig(
              ModeTraits::Adjust(LayoutPolicy::Normalize(config)))),
          geometry_(config_),
          layout_(config_),
          backend_(config_, layout_.DataSectionBytes(), layout_.Fingerprint(),
                   Layout::kHasVCache) {
        this->logger_ = init_logger("info", "HostPagedKVManager");
        this->logger_->info(
            "Constructed HostPagedKVManager (shm_name={}, layers={}, pages={}, "
            "page_tokens={}, "
            "k_heads={}, head_dim={}, has_v_cache={}, seq_capacity={})",
            config_.shm_name, config_.num_layers, config_.num_pages,
            config_.page_size_tokens, config_.num_k_heads, config_.k_head_dim,
            Layout::kHasVCache, config_.sequence_table_capacity);
    }

    HostPagedKVManager(const HostPagedKVManager&) = delete;
    HostPagedKVManager& operator=(const HostPagedKVManager&) = delete;
    HostPagedKVManager(HostPagedKVManager&&) = delete;
    HostPagedKVManager& operator=(HostPagedKVManager&&) = delete;

    void Initialize(bool create_region) {
        this->logger_->info(
            "Initializing HostPagedKVBackend (create_region={})",
            create_region);
        backend_.Initialize(create_region);
        const HostPagedKVStats stats = backend_.CollectStats();
        this->logger_->info(
            "HostPagedKVBackend ready (total_pages={}, free_pages={}, "
            "has_v_cache={}, total_bytes={})",
            stats.num_total_pages, stats.num_free_pages, Layout::kHasVCache,
            stats.total_bytes);
        const std::size_t data_bytes = layout_.DataSectionBytes();
        const std::size_t metadata_bytes =
            stats.total_bytes > data_bytes ? stats.total_bytes - data_bytes : 0;
        std::size_t v_page_bytes = 0;
        std::string v_page_suffix;
        if constexpr (Layout::kHasVCache) {
            v_page_bytes = layout_.template VPageBytes<>();
            v_page_suffix = ", V_page_bytes=" + std::to_string(v_page_bytes) +
                            " bytes per page";
        }
        this->logger_->info(
            "HostPagedKV memory layout: total={} GB (data={} GB, "
            "metadata={} KB); layer_stride={} MB; K_page_bytes={} "
            "bytes{}",
            BytesToGigabytes(stats.total_bytes), BytesToGigabytes(data_bytes),
            BytesToKilobytes(metadata_bytes),
            BytesToMegabytes(layout_.LayerStrideBytes()), layout_.KPageBytes(),
            v_page_suffix);
    }

    std::vector<std::int32_t> AllocatePages(std::int64_t sequence_id,
                                            std::size_t num_tokens) {
        const std::size_t required_pages = geometry_.RequiredPages(num_tokens);
        return backend_.AcquirePages(sequence_id, required_pages);
    }

    std::string DebugString() const {
        std::ostringstream oss;
        oss << "HostPagedKVManager(mode=" << static_cast<int>(Mode)
            << ", config=" << ToString(config_);
        try {
            const HostPagedKVStats stats = backend_.CollectStats();
            oss << ", stats=" << ToString(stats);
        } catch (const std::exception& ex) {
            oss << ", stats_error='" << ex.what() << "'";
        }
        oss << ")";
        return oss.str();
    }

    void FreeSequence(std::int64_t sequence_id) {
        backend_.ReleaseSequence(sequence_id);
    }

    void FreeSequences(const std::vector<std::int64_t>& sequence_ids) {
        backend_.ReleaseSequences(sequence_ids);
    }

    std::pair<std::vector<void*>, std::optional<std::vector<void*>>>
    GetSequenceLayerPagePointers(
        std::int64_t sequence_id, std::size_t layer_idx,
        std::optional<std::size_t> max_tokens = std::nullopt) const {
        geometry_.EnsureLayerBounds(
            layer_idx, "HostPagedKVManager::GetSequenceLayerPagePointers");
        std::optional<std::size_t> max_pages;
        if (max_tokens.has_value()) {
            max_pages = geometry_.RequiredPages(max_tokens.value());
        }
        auto page_indices = backend_.SequencePages(sequence_id, max_pages);
        std::vector<void*> k_ptrs;
        k_ptrs.reserve(page_indices.size());
        std::optional<std::vector<void*>> v_ptrs;
        if constexpr (Layout::kHasVCache) {
            v_ptrs.emplace();
            v_ptrs->reserve(page_indices.size());
        }
        std::byte* base = const_cast<std::byte*>(backend_.DataBase());
        for (std::int32_t page : page_indices) {
            void* k_ptr =
                static_cast<void*>(layout_.KPageAddress(base, layer_idx, page));
            k_ptrs.emplace_back(k_ptr);
            // LogPageBytes("K", layer_idx, sequence_id, page, k_ptr,
            //              geometry_.KTokenBytes());
            if constexpr (Layout::kHasVCache) {
                void* v_ptr = static_cast<void*>(
                    layout_.VPageAddress(base, layer_idx, page));
                v_ptrs->emplace_back(v_ptr);
                // LogPageBytes("V", layer_idx, sequence_id, page, v_ptr,
                //              geometry_.template VTokenBytes<true>());
            }
        }
        return {std::move(k_ptrs), std::move(v_ptrs)};
    }

    std::vector<std::vector<std::int32_t>> BuildPageTable(
        const std::vector<std::int64_t>& sequence_ids) const {
        std::vector<std::vector<std::int32_t>> table;
        table.reserve(sequence_ids.size());
        for (std::int64_t seq_id : sequence_ids) {
            try {
                table.emplace_back(
                    backend_.SequencePages(seq_id, std::nullopt));
            } catch (const std::out_of_range&) {
                table.emplace_back();
            }
        }
        return table;
    }

    HostPagedKVStats GetStats() const { return backend_.CollectStats(); }

    void* MutableKPage(std::size_t layer_idx, std::int32_t page_idx) {
        geometry_.EnsureLayerBounds(layer_idx,
                                    "HostPagedKVManager::MutableKPage");
        geometry_.EnsurePageBounds(page_idx,
                                   "HostPagedKVManager::MutableKPage");
        return static_cast<void*>(
            layout_.KPageAddress(backend_.DataBase(), layer_idx, page_idx));
    }

    template <bool Enabled = Layout::kHasVCache,
              typename = std::enable_if_t<Enabled>>
    void* MutableVPage(std::size_t layer_idx, std::int32_t page_idx) {
        geometry_.EnsureLayerBounds(layer_idx,
                                    "HostPagedKVManager::MutableVPage");
        geometry_.EnsurePageBounds(page_idx,
                                   "HostPagedKVManager::MutableVPage");
        return static_cast<void*>(
            layout_.VPageAddress(backend_.DataBase(), layer_idx, page_idx));
    }

    const HostPagedKVConfig& config() const { return config_; }
    const Layout& layout() const { return layout_; }
    int memfd_fd() const { return backend_.memfd_fd(); }

   private:
    void LogPageBytes(const char* tag, std::size_t layer_idx,
                      std::int64_t sequence_id, std::int32_t page_idx,
                      const void* ptr, std::size_t token_bytes) const {
        if (logger_ == nullptr || ptr == nullptr || token_bytes == 0) {
            return;
        }
        const auto bytes = geometry_.DescribeBytes(
            static_cast<const std::byte*>(ptr), token_bytes);
        logger_->info(
            "GetSequenceLayerPagePointers {} layer={} seq={} page={} "
            "first_bytes={}",
            tag, layer_idx, sequence_id, page_idx, bytes);
    }

   protected:
    std::shared_ptr<spdlog::logger> logger_;

    // References to the global engine and model configuration. These are
    // non-owning references; the caller is responsible for their lifetime.
    // const EngineConfig& engine_config_;
    const std::optional<std::reference_wrapper<EngineConfig>> engine_config_;
    const std::optional<std::reference_wrapper<ModelConfig>> model_config_;

    // Derived host-paged KV configuration used by this manager. This is
    // constructed from `engine_config_` / `model_config_` during
    // initialization.
    HostPagedKVConfig config_;
    HostPagedKVGeometry geometry_;
    Layout layout_;
    HostPagedKVBackend backend_;
};

using DefaultHostPagedKVManager = HostPagedKVManager<HostKVMode::kMHA>;
using MHAHostPagedKVManager =
    HostPagedKVManager<HostKVMode::kMHA, MHALayoutPolicy>;
using MLAHostPagedKVManager = HostPagedKVManager<HostKVMode::kMLA>;

}  // namespace batchgen::kv

#endif  // HOST_PAGED_KV_MANAGER_H_