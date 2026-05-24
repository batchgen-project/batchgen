#ifndef SWA_HOST_PAGED_KV_WORKER_VIEW_H_
#define SWA_HOST_PAGED_KV_WORKER_VIEW_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#include "host_paged_kv_worker_view.h"

namespace batchgen::kv {

struct SWAHostPageRange {
    std::int64_t sequence_id = 0;
    std::size_t raw_context_len = 0;
    std::size_t window_start_token = 0;
    std::size_t first_page = 0;
    std::size_t page_count = 0;
    std::size_t local_kv_len = 0;
    std::size_t mask_start = 0;
};

template <typename BaseView>
class SWAHostPagedKVWorkerView : public BaseView {
   public:
    using BatchedKVEntry = typename BaseView::BatchedKVEntry;
    static constexpr bool kHasVCache = BaseView::kHasVCache;
    static constexpr bool kUsesLogicalLayerMapping =
        BaseView::kUsesLogicalLayerMapping;

    SWAHostPagedKVWorkerView(const EngineConfig& engine_config,
                             const ModelConfig& model_config,
                             std::size_t window_size_tokens)
        : BaseView(engine_config, model_config),
          window_size_tokens_(window_size_tokens) {
        ValidateWindowConfig();
    }

    explicit SWAHostPagedKVWorkerView(const HostPagedKVConfig& config,
                                      std::size_t window_size_tokens)
        : BaseView(config), window_size_tokens_(window_size_tokens) {
        ValidateWindowConfig();
    }

    SWAHostPagedKVWorkerView(const SWAHostPagedKVWorkerView&) = delete;
    SWAHostPagedKVWorkerView& operator=(const SWAHostPagedKVWorkerView&) =
        delete;
    SWAHostPagedKVWorkerView(SWAHostPagedKVWorkerView&&) = delete;
    SWAHostPagedKVWorkerView& operator=(SWAHostPagedKVWorkerView&&) = delete;

    std::size_t page_size_tokens() const {
        return this->config().page_size_tokens;
    }

    std::size_t window_size_tokens() const { return window_size_tokens_; }

    std::size_t window_pages() const {
        return CeilDiv(window_size_tokens_, page_size_tokens());
    }

    std::string DebugString() const {
        std::ostringstream oss;
        oss << "SWAHostPagedKVWorkerView(full_history=true, "
            << "window_size_tokens=" << window_size_tokens_
            << ", page_size_tokens=" << page_size_tokens()
            << ", window_pages=" << window_pages()
            << ", base=" << BaseView::DebugString() << ")";
        return oss.str();
    }

    SWAHostPageRange ComputeSWAHostPageRange(
        std::int64_t sequence_id, std::size_t raw_context_len) const {
        const std::size_t page_size = page_size_tokens();
        const std::size_t window_start_token =
            raw_context_len > window_size_tokens_
                ? raw_context_len - window_size_tokens_
                : 0;
        const std::size_t first_page = window_start_token / page_size;
        const std::size_t last_page_exclusive =
            CeilDiv(raw_context_len, page_size);
        const std::size_t first_page_token = first_page * page_size;
        const std::size_t page_count =
            last_page_exclusive > first_page
                ? last_page_exclusive - first_page
                : 0;
        return SWAHostPageRange{
            sequence_id,
            raw_context_len,
            window_start_token,
            first_page,
            page_count,
            raw_context_len - first_page_token,
            window_start_token - first_page_token,
        };
    }

    std::vector<SWAHostPageRange> ComputeSWAHostPageRanges(
        const std::vector<std::int64_t>& sequence_ids,
        const std::vector<std::size_t>& raw_context_lens) const {
        if (sequence_ids.size() != raw_context_lens.size()) {
            throw std::invalid_argument(
                "ComputeSWAHostPageRanges: sequence_ids and "
                "raw_context_lens must have the same length");
        }
        std::vector<SWAHostPageRange> ranges;
        ranges.reserve(sequence_ids.size());
        for (std::size_t i = 0; i < sequence_ids.size(); ++i) {
            ranges.push_back(
                ComputeSWAHostPageRange(sequence_ids[i], raw_context_lens[i]));
        }
        return ranges;
    }

    std::pair<std::vector<void*>, std::optional<std::vector<void*>>>
    GetSequenceLayerSWAWindowPagePointers(
        std::int64_t sequence_id, std::size_t layer_idx,
        std::size_t raw_context_len) const {
        const SWAHostPageRange range =
            ComputeSWAHostPageRange(sequence_id, raw_context_len);
        return this->GetSequenceLayerPageRangePointers(
            sequence_id, layer_idx, range.first_page, range.page_count);
    }

   private:
    static std::size_t CeilDiv(std::size_t value, std::size_t divisor) {
        if (divisor == 0) {
            throw std::invalid_argument("CeilDiv divisor must be non-zero");
        }
        return (value + divisor - 1) / divisor;
    }

    void ValidateWindowConfig() const {
        if (page_size_tokens() == 0) {
            throw std::invalid_argument(
                "SWAHostPagedKVWorkerView requires page_size_tokens > 0");
        }
        if (window_size_tokens_ == 0) {
            throw std::invalid_argument(
                "SWAHostPagedKVWorkerView requires window_size_tokens > 0");
        }
    }

    std::size_t window_size_tokens_ = 0;
};

using SWADefaultHostPagedKVWorkerView =
    SWAHostPagedKVWorkerView<DefaultHostPagedKVWorkerView>;
using SWAMLAHostPagedKVWorkerView =
    SWAHostPagedKVWorkerView<MLAHostPagedKVWorkerView>;
using SWAMappedDefaultHostPagedKVWorkerView =
    SWAHostPagedKVWorkerView<MappedDefaultHostPagedKVWorkerView>;
using SWAMappedMLAHostPagedKVWorkerView =
    SWAHostPagedKVWorkerView<MappedMLAHostPagedKVWorkerView>;

}  // namespace batchgen::kv

#endif  // SWA_HOST_PAGED_KV_WORKER_VIEW_H_
