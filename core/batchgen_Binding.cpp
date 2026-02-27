// clang-format off
/* ----------------------------------------------------------------------------  *
 *  BatchGen                                                                      *
 *  copyright (c) EfficientMoE team 2025                                             *
 *                                                                               *
 *  licensed under the apache license, version 2.0 (the "license");              *
 *  you may not use this file except in compliance with the license.             *
 *                                                                               *
 *  you may obtain a copy of the license at                                      *
 *                                                                               *
 *                  http://www.apache.org/licenses/license-2.0                   *
 *                                                                               *
 *  unless required by applicable law or agreed to in writing, software          *
 *  distributed under the license is distributed on an "as is" basis,            *
 *  without warranties or conditions of any kind, either express or implied.     *
 *  see the license for the specific language governing permissions and          *
 *  limitations under the license.                                               *
 * ---------------------------------------------------------------------------- */
// clang-format on

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <ATen/cuda/CachingHostAllocator.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include "KV_Storage/host_paged_kv_manager.h"
#include "KV_Storage/host_paged_kv_prefix_cache.h"
#include "KV_Storage/host_paged_kv_worker_view.h"
#include "Weights_Storage/Weights_Storage.h"
#include "allocator.h"
#include "batchgen.h"
#include "data_structures.h"

namespace py = pybind11;
namespace kv = batchgen::kv;

namespace {

struct HostKVPrefixCacheHarnessStats {
    std::uint32_t prefix_entry_count = 0;
    std::uint32_t prefix_used_pages = 0;
    std::uint64_t prefix_access_epoch = 0;
    std::uint64_t prefix_hit_count = 0;
    std::uint64_t prefix_miss_count = 0;
    std::uint64_t prefix_evict_count = 0;
    std::int32_t lru_head = kv::kHostKVInvalidIndex;
    std::int32_t lru_tail = kv::kHostKVInvalidIndex;
};

class HostKVPrefixCacheHarness {
   public:
    HostKVPrefixCacheHarness(std::size_t num_pages,
                             std::size_t radix_node_capacity,
                             std::size_t radix_edge_capacity,
                             std::size_t prefix_entry_capacity,
                             std::size_t prefix_page_ref_capacity,
                             bool enable_prefix_reuse = true,
                             std::size_t prefix_min_reuse_pages = 1,
                             std::size_t prefix_min_store_pages = 1,
                             std::size_t prefix_page_budget = 0)
        : num_pages_(num_pages),
          radix_node_capacity_(radix_node_capacity),
          radix_edge_capacity_(radix_edge_capacity),
          prefix_entry_capacity_(prefix_entry_capacity),
          prefix_page_ref_capacity_(prefix_page_ref_capacity),
          radix_nodes_(radix_node_capacity),
          radix_edges_(radix_edge_capacity),
          prefix_entries_(prefix_entry_capacity),
          prefix_page_refs_(prefix_page_ref_capacity),
          radix_node_free_stack_(radix_node_capacity),
          radix_edge_free_stack_(radix_edge_capacity),
          prefix_entry_free_stack_(prefix_entry_capacity),
          prefix_page_ref_free_stack_(prefix_page_ref_capacity),
          page_refcounts_(num_pages, 0) {
        if (num_pages_ == 0) {
            throw std::invalid_argument("num_pages must be > 0");
        }
        if (radix_node_capacity_ == 0) {
            throw std::invalid_argument("radix_node_capacity must be > 0");
        }
        if (radix_edge_capacity_ == 0) {
            throw std::invalid_argument("radix_edge_capacity must be > 0");
        }
        if (prefix_entry_capacity_ == 0) {
            throw std::invalid_argument("prefix_entry_capacity must be > 0");
        }
        if (prefix_page_ref_capacity_ == 0) {
            throw std::invalid_argument("prefix_page_ref_capacity must be > 0");
        }

        params_.enable_prefix_reuse = enable_prefix_reuse;
        params_.prefix_min_reuse_pages =
            std::max<std::size_t>(1, prefix_min_reuse_pages);
        params_.prefix_min_store_pages =
            std::max<std::size_t>(1, prefix_min_store_pages);
        params_.prefix_page_budget =
            prefix_page_budget == 0 ? num_pages_ : prefix_page_budget;

        kv::HostKVPrefixCache::SharedFields shared_fields;
        shared_fields.radix_node_free_top = &radix_node_free_top_;
        shared_fields.radix_edge_free_top = &radix_edge_free_top_;
        shared_fields.prefix_entry_free_top = &prefix_entry_free_top_;
        shared_fields.prefix_page_ref_free_top = &prefix_page_ref_free_top_;
        shared_fields.prefix_entry_count = &prefix_entry_count_;
        shared_fields.prefix_used_pages = &prefix_used_pages_;
        shared_fields.prefix_access_epoch = &prefix_access_epoch_;
        shared_fields.prefix_hit_count = &prefix_hit_count_;
        shared_fields.prefix_miss_count = &prefix_miss_count_;
        shared_fields.prefix_evict_count = &prefix_evict_count_;
        shared_fields.lru_head = &lru_head_;
        shared_fields.lru_tail = &lru_tail_;

        cache_.Bind(
            params_, radix_nodes_.data(), radix_edges_.data(),
            prefix_entries_.data(), prefix_page_refs_.data(),
            radix_node_free_stack_.data(), radix_edge_free_stack_.data(),
            prefix_entry_free_stack_.data(), prefix_page_ref_free_stack_.data(),
            shared_fields,
            [this](std::int32_t page_idx) {
                ValidatePageIdx(page_idx);
                ++page_refcounts_[page_idx];
            },
            [this](std::int32_t page_idx) {
                ValidatePageIdx(page_idx);
                if (page_refcounts_[page_idx] == 0) {
                    throw std::runtime_error(
                        "page_refcount underflow on page " +
                        std::to_string(page_idx));
                }
                --page_refcounts_[page_idx];
            });

        Reset();
    }

    std::pair<std::vector<std::int32_t>, std::size_t> Lookup(
        const std::vector<std::int32_t>& tokens, std::size_t max_pages) {
        const auto result =
            cache_.LookupPrefixPagesLocked(tokens.data(), tokens.size(), max_pages);
        return {result.pages, result.reused_pages};
    }

    bool Commit(const std::vector<std::int32_t>& tokens,
                const std::vector<std::int32_t>& pages) {
        ValidatePageIndices(pages);
        return cache_.CommitPrefixLocked(tokens.data(), tokens.size(), pages);
    }

    HostKVPrefixCacheHarnessStats GetStats() const {
        HostKVPrefixCacheHarnessStats stats;
        stats.prefix_entry_count =
            prefix_entry_count_.load(std::memory_order_relaxed);
        stats.prefix_used_pages =
            prefix_used_pages_.load(std::memory_order_relaxed);
        stats.prefix_access_epoch =
            prefix_access_epoch_.load(std::memory_order_relaxed);
        stats.prefix_hit_count = prefix_hit_count_.load(std::memory_order_relaxed);
        stats.prefix_miss_count =
            prefix_miss_count_.load(std::memory_order_relaxed);
        stats.prefix_evict_count =
            prefix_evict_count_.load(std::memory_order_relaxed);
        stats.lru_head = lru_head_;
        stats.lru_tail = lru_tail_;
        return stats;
    }

    std::int32_t PageRefcount(std::int32_t page_idx) const {
        ValidatePageIdx(page_idx);
        return static_cast<std::int32_t>(page_refcounts_[page_idx]);
    }

    std::vector<std::int32_t> PageRefcounts() const {
        std::vector<std::int32_t> result;
        result.reserve(page_refcounts_.size());
        for (std::uint32_t count : page_refcounts_) {
            result.push_back(static_cast<std::int32_t>(count));
        }
        return result;
    }

    void Reset() {
        std::fill(page_refcounts_.begin(), page_refcounts_.end(), 0);

        radix_node_free_top_.store(
            static_cast<std::uint32_t>(radix_node_capacity_ - 1),
            std::memory_order_relaxed);
        radix_edge_free_top_.store(
            static_cast<std::uint32_t>(radix_edge_capacity_),
            std::memory_order_relaxed);
        prefix_entry_free_top_.store(
            static_cast<std::uint32_t>(prefix_entry_capacity_),
            std::memory_order_relaxed);
        prefix_page_ref_free_top_.store(
            static_cast<std::uint32_t>(prefix_page_ref_capacity_),
            std::memory_order_relaxed);

        prefix_entry_count_.store(0, std::memory_order_relaxed);
        prefix_used_pages_.store(0, std::memory_order_relaxed);
        prefix_access_epoch_.store(0, std::memory_order_relaxed);
        prefix_hit_count_.store(0, std::memory_order_relaxed);
        prefix_miss_count_.store(0, std::memory_order_relaxed);
        prefix_evict_count_.store(0, std::memory_order_relaxed);
        lru_head_ = kv::kHostKVInvalidIndex;
        lru_tail_ = kv::kHostKVInvalidIndex;

        cache_.InitializePools(radix_node_capacity_, radix_edge_capacity_,
                               prefix_entry_capacity_,
                               prefix_page_ref_capacity_);
    }

   private:
    void ValidatePageIdx(std::int32_t page_idx) const {
        if (page_idx < 0 || static_cast<std::size_t>(page_idx) >= num_pages_) {
            throw std::out_of_range("page index out of range: " +
                                    std::to_string(page_idx));
        }
    }

    void ValidatePageIndices(const std::vector<std::int32_t>& pages) const {
        for (std::int32_t page_idx : pages) {
            ValidatePageIdx(page_idx);
        }
    }

    std::size_t num_pages_ = 0;
    std::size_t radix_node_capacity_ = 0;
    std::size_t radix_edge_capacity_ = 0;
    std::size_t prefix_entry_capacity_ = 0;
    std::size_t prefix_page_ref_capacity_ = 0;

    kv::HostKVPrefixCache cache_;
    kv::HostKVPrefixCacheParams params_{};

    std::vector<kv::HostKVRadixNode> radix_nodes_;
    std::vector<kv::HostKVRadixEdge> radix_edges_;
    std::vector<kv::HostKVPrefixEntry> prefix_entries_;
    std::vector<kv::HostKVPrefixPageRef> prefix_page_refs_;

    std::vector<std::int32_t> radix_node_free_stack_;
    std::vector<std::int32_t> radix_edge_free_stack_;
    std::vector<std::int32_t> prefix_entry_free_stack_;
    std::vector<std::int32_t> prefix_page_ref_free_stack_;

    std::vector<std::uint32_t> page_refcounts_;

    std::atomic<std::uint32_t> radix_node_free_top_{0};
    std::atomic<std::uint32_t> radix_edge_free_top_{0};
    std::atomic<std::uint32_t> prefix_entry_free_top_{0};
    std::atomic<std::uint32_t> prefix_page_ref_free_top_{0};

    std::atomic<std::uint32_t> prefix_entry_count_{0};
    std::atomic<std::uint32_t> prefix_used_pages_{0};

    std::atomic<std::uint64_t> prefix_access_epoch_{0};
    std::atomic<std::uint64_t> prefix_hit_count_{0};
    std::atomic<std::uint64_t> prefix_miss_count_{0};
    std::atomic<std::uint64_t> prefix_evict_count_{0};

    std::int32_t lru_head_ = kv::kHostKVInvalidIndex;
    std::int32_t lru_tail_ = kv::kHostKVInvalidIndex;
};

template <typename Manager>
void BindHostPagedManager(py::module& m, const char* name) {
    py::class_<Manager>(m, name)
        .def(py::init<EngineConfig, ModelConfig>())
        .def(py::init<kv::HostPagedKVConfig>())
        .def("initialize", &Manager::Initialize, py::arg("create_region"))
        .def("allocate_pages",
             [](Manager& self, std::int64_t sequence_id,
                std::size_t num_tokens) {
                 return self.AllocatePages(sequence_id, num_tokens);
             },
             py::arg("sequence_id"), py::arg("num_tokens"))
       .def("free_sequence", &Manager::FreeSequence,
           py::arg("sequence_id"))
       .def("free_sequences", &Manager::FreeSequences,
           py::arg("sequence_ids"))
        .def("build_page_table", &Manager::BuildPageTable,
             py::arg("sequence_ids"))
        .def("get_stats", &Manager::GetStats)
       .def("__repr__",
           [](const Manager& self) { return self.DebugString(); })
        .def("get_sequence_layer_page_pointers",
             [](Manager& self, std::int64_t sequence_id,
                std::size_t layer_idx,
                std::optional<std::size_t> max_tokens) {
                 auto result = self.GetSequenceLayerPagePointers(
                     sequence_id, layer_idx, max_tokens);
                 py::list k_ptrs;
                 for (void* ptr : result.first) {
                     k_ptrs.append(py::int_(
                         reinterpret_cast<std::uintptr_t>(ptr)));
                 }
                 py::object v_ptrs = py::none();
                 if (result.second.has_value()) {
                     py::list v_list;
                     for (void* ptr : result.second.value()) {
                         v_list.append(py::int_(
                             reinterpret_cast<std::uintptr_t>(ptr)));
                     }
                     v_ptrs = std::move(v_list);
                 }
                 return py::make_tuple(std::move(k_ptrs), v_ptrs);
             },
             py::arg("sequence_id"), py::arg("layer_idx"),
             py::arg("max_tokens") = py::none());
}

template <typename WorkerView>
void BindHostPagedWorkerView(py::module& m, const char* name) {
    py::class_<WorkerView>(m, name)
        .def(py::init<EngineConfig, ModelConfig>())
        .def(py::init<kv::HostPagedKVConfig>(), py::arg("config"))
        .def("initialize", &WorkerView::Initialize,
             py::arg("device_index"),
             py::arg("create_region") = false)
        .def("shutdown", &WorkerView::Shutdown)
        .def("data_base_address",
             [](WorkerView& self) -> std::uintptr_t {
                 return reinterpret_cast<std::uintptr_t>(self.DataBase());
             })
        .def("k_page_ptr",
             [](WorkerView& self, std::size_t layer_idx,
                std::int32_t page_idx) -> std::uintptr_t {
                 return reinterpret_cast<std::uintptr_t>(
                     self.KPagePtr(layer_idx, page_idx));
             },
             py::arg("layer_idx"), py::arg("page_idx"))
        .def(
            "v_page_ptr",
            [](WorkerView& self, std::size_t layer_idx,
               std::int32_t page_idx) -> std::uintptr_t {
                if constexpr (WorkerView::kHasVCache) {
                    return reinterpret_cast<std::uintptr_t>(
                        self.VPagePtr(layer_idx, page_idx));
                }
                throw std::runtime_error(
                    "V cache is disabled for this worker view");
            },
            py::arg("layer_idx"), py::arg("page_idx"))
        .def("get_stats", &WorkerView::GetStats)
       .def("build_page_table",
           [](WorkerView& self,
             const std::vector<std::int64_t>& sequence_ids) {
              return self.BuildPageTable(sequence_ids);
           },
           py::arg("sequence_ids"))
       .def("register_sequences", &WorkerView::RegisterSequences,
           py::arg("sequence_ids"))
       .def("unregister_sequence", &WorkerView::UnregisterSequence,
           py::arg("sequence_id"))
       .def("unregister_sequences", &WorkerView::UnregisterSequences,
           py::arg("sequence_ids"))
       .def("release_sequence_pages", &WorkerView::ReleaseSequencePages,
            py::arg("sequence_ids"))
       .def("async_offload_layer_kv_to_host",
           &WorkerView::AsyncOffloadLayerKVToHost,
           py::arg("layer_idx"), py::arg("sequence_ids"),
           py::arg("k_tensor"), py::arg("v_tensor") = py::none(),
           py::arg("sequence_lengths"))
       .def("async_append_decode_kv_to_host",
           &WorkerView::AsyncAppendDecodeKVToHost,
           py::arg("layer_idx"), py::arg("sequence_ids"),
           py::arg("k_tensor"), py::arg("v_tensor") = py::none(),
           py::arg("sequence_lengths"))
        .def(
            "async_load_layer_kv_to_device",
            [](WorkerView& self, torch::Tensor sequence_ids,
               torch::Tensor k_device_ptrs,
               std::optional<torch::Tensor> v_device_ptrs) {
                return self.AsyncLoadLayerKVToDevice(
                    std::move(sequence_ids), std::move(k_device_ptrs),
                    std::move(v_device_ptrs));
            },
            py::arg("sequence_ids"), py::arg("k_device_ptrs"),
            py::arg("v_device_ptrs") = py::none(),
            "Schedule host-paged KV pages to be loaded onto device memory "
            "using pre-allocated GPU destinations.")
        .def(
            "async_load_layer_paged_kv_to_device",
            [](WorkerView& self, torch::Tensor sequence_ids,
               torch::Tensor active_page_counts,
               torch::Tensor k_device_ptrs,
               std::optional<torch::Tensor> v_device_ptrs) {
                return self.AsyncLoadLayerPagedKVToDevice(
                    std::move(sequence_ids), std::move(active_page_counts),
                    std::move(k_device_ptrs), std::move(v_device_ptrs));
            },
            py::arg("sequence_ids"), py::arg("active_page_counts"),
            py::arg("k_device_ptrs"),
            py::arg("v_device_ptrs") = py::none(),
            "Load only the active per-sequence KV pages using padded page tables.")
        .def("__repr__",
             [](const WorkerView& self) { return self.DebugString(); })
        .def(
            "allocate_pages_for_sequences",
            [](WorkerView& self,
               const std::vector<std::pair<std::int64_t, std::size_t>>&
                   requests) {
                std::vector<std::int64_t> sequence_ids;
                std::vector<std::size_t> num_tokens;
                sequence_ids.reserve(requests.size());
                num_tokens.reserve(requests.size());
                for (const auto& request : requests) {
                    sequence_ids.push_back(request.first);
                    num_tokens.push_back(request.second);
                }
                return self.AllocatePagesForSequences(sequence_ids,
                                                      num_tokens);
            })
        .def(
            "allocate_pages_for_sequences_with_prefix",
            [](WorkerView& self,
               const std::vector<std::pair<std::int64_t, std::size_t>>&
                   requests,
               const std::vector<std::int32_t>& flat_prompt_tokens,
               const std::vector<std::size_t>& prompt_offsets) {
                std::vector<std::int64_t> sequence_ids;
                std::vector<std::size_t> num_tokens;
                sequence_ids.reserve(requests.size());
                num_tokens.reserve(requests.size());
                for (const auto& request : requests) {
                    sequence_ids.push_back(request.first);
                    num_tokens.push_back(request.second);
                }
                auto result = self.AllocatePagesForSequencesWithPrefix(
                    sequence_ids, num_tokens, flat_prompt_tokens,
                    prompt_offsets);
                return py::make_tuple(std::move(result.first),
                                      std::move(result.second));
            },
            py::arg("requests"), py::arg("flat_prompt_tokens"),
            py::arg("prompt_offsets"))
        .def("grow_sequence_pages",
             [](WorkerView& self, std::int64_t sequence_id,
                std::size_t num_pages) {
                 return self.GrowSequencePages(sequence_id, num_pages);
             },
             py::arg("sequence_id"), py::arg("num_pages"))
        .def(
            "grow_pages_for_sequences",
            [](WorkerView& self,
               const std::vector<std::pair<std::int64_t, std::size_t>>&
                   requests) {
                std::vector<std::int64_t> sequence_ids;
                std::vector<std::size_t> page_counts;
                sequence_ids.reserve(requests.size());
                page_counts.reserve(requests.size());
                for (const auto& request : requests) {
                    sequence_ids.push_back(request.first);
                    page_counts.push_back(request.second);
                }
                return self.GrowPagesForSequences(sequence_ids,
                                                   page_counts);
            })
        .def_property_readonly("device_index", &WorkerView::device_index)
        .def_property_readonly_static(
            "has_v_cache",
            [](py::object /* cls */) { return WorkerView::kHasVCache; })
        .def("get_sequence_layer_page_pointers",
             [](WorkerView& self, std::int64_t sequence_id,
                std::size_t layer_idx,
                std::optional<std::size_t> max_tokens) {
                 auto result = self.GetSequenceLayerPagePointers(
                     sequence_id, layer_idx, max_tokens);
                 py::list k_ptrs;
                 for (void* ptr : result.first) {
                     k_ptrs.append(py::int_(
                         reinterpret_cast<std::uintptr_t>(ptr)));
                 }
                 py::object v_ptrs = py::none();
                 if (result.second.has_value()) {
                     py::list v_list;
                     for (void* ptr : result.second.value()) {
                         v_list.append(py::int_(
                             reinterpret_cast<std::uintptr_t>(ptr)));
                     }
                     v_ptrs = std::move(v_list);
                 }
                 return py::make_tuple(std::move(k_ptrs), v_ptrs);
             },
             py::arg("sequence_id"), py::arg("layer_idx"),
             py::arg("max_tokens") = py::none());
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<BatchGen>(m, "batchgen")
        .def(py::init<py::object, py::object, Weights_Storage&>())
        .def("Init", &BatchGen::Init)
        .def("terminate", &BatchGen::Terminate)
        // .def("set_batching_plan", &BatchGen::set_batching_plan)
        .def("kv_offload", &BatchGen::kv_offload)
        // .def("add_weight_storage", &BatchGen::add_weight_storage)
        .def("get_weights", &BatchGen::get_weights)
        .def("free_weights_buffer", &BatchGen::free_weights_buffer)
        .def("attn", &BatchGen::attn)
        .def("submit_to_KV_queue", &BatchGen::submit_to_KV_queue)
        .def("clear_expert_buffer", &BatchGen::clear_expert_buffer)
        // .def("get_skeleton_state_dict", &BatchGen::get_skeleton_state_dict)
        .def("prefill_complete_sync", &BatchGen::prefill_complete_sync)
        .def("set_phase", &BatchGen::set_phase)
        .def("clear_kv_storage", &BatchGen::clear_kv_storage)
        .def("clear_kv_copy_queue", &BatchGen::clear_kv_copy_queue)
        .def("reset_weight_copy_queue", &BatchGen::reset_weight_copy_queue)
        .def("clear_kv_buffer", &BatchGen::clear_kv_buffer)
        .def("clear_weight_copy_queue", &BatchGen::clear_weight_copy_queue)
        .def("reset_prefill_buffer", &BatchGen::reset_prefill_buffer)
        .def("create_fake_kv_storage", &BatchGen::create_fake_kv_storage)
        // .def("get_tensor", &BatchGen::get_tensor,py::return_value_policy::take_ownership)
        .def("get_tensor", &BatchGen::get_tensor)
        .def("start_h2d_worker", &BatchGen::start_h2d_worker)
        .def("set_global_routed_experts_data_ptr",
             &BatchGen::set_global_routed_experts_data_ptr)
        .def("cuda_enable_peer_access", &BatchGen::cuda_enable_peer_access)
        .def("save_compressed_kv", &BatchGen::save_compressed_kv)
        .def("set_weight_copy_queue", &BatchGen::set_weight_copy_queue)
        .def("reset_decoding_buffer", &BatchGen::reset_decoding_buffer)
        .def("stop_h2d_worker", &BatchGen::stop_h2d_worker)
        .def("copy_kv_to_worker",
             &BatchGen::copy_kv_to_worker,
             "Copy KV to worker for the given query global index and context length.")
        .def("clear_kv_gpu_storage",
             &BatchGen::clear_gpu_kv_storage,
             "Clear the GPU KV storage.")
        .def("get_kv_scale",
             &BatchGen::get_kv_scale,
             "Get the quantization scale for KV storage.")
        .def("get_past_key_states",
             &BatchGen::get_past_key_states,
             "Get the past key states for the given query global indices and "
             "max sequence length.")
        .def("init_weight_storage", &BatchGen::init_weight_storage)
        .def_property(
            "host_paged_kv_worker_view", &BatchGen::host_paged_kv_worker_view,
            &BatchGen::set_host_paged_kv_worker_view,
            "Reference to the bound HostPagedKVWorkerView instance.")
        .def_property("gpu_paged_kv_manager", &BatchGen::gpu_paged_kv_manager,
                      &BatchGen::set_gpu_paged_kv_manager,
                      "Python GPU paged KV manager bound to this engine.");

    py::class_<Weights_Storage>(m, "Weights_Storage")
        // Updated Constructor Binding
        .def(py::init<int>(), py::arg("device_id"))
        .def("Init", &Weights_Storage::Init,
             py::arg("shm_name"), py::arg("byte_size"),
             py::arg("module_weights_shm"),
             py::arg("enable_hugetlbfs") = false)
        .def("get_tensor", &Weights_Storage::get_tensor,
             py::arg("module_key"));

    py::class_<kv::HostPagedKVConfig>(m, "HostPagedKVConfig")
        .def(py::init<>())
        .def_readwrite("shm_name", &kv::HostPagedKVConfig::shm_name)
        .def_readwrite("num_layers", &kv::HostPagedKVConfig::num_layers)
        .def_readwrite("num_pages", &kv::HostPagedKVConfig::num_pages)
        .def_readwrite("page_size_tokens",
                       &kv::HostPagedKVConfig::page_size_tokens)
        .def_readwrite("num_k_heads", &kv::HostPagedKVConfig::num_k_heads)
        .def_readwrite("k_head_dim", &kv::HostPagedKVConfig::k_head_dim)
        .def_readwrite("num_v_heads", &kv::HostPagedKVConfig::num_v_heads)
        .def_readwrite("v_head_dim", &kv::HostPagedKVConfig::v_head_dim)
        .def_readwrite("k_element_size_bytes",
                       &kv::HostPagedKVConfig::k_element_size_bytes)
        .def_readwrite("v_element_size_bytes",
                       &kv::HostPagedKVConfig::v_element_size_bytes)
        .def_readwrite("sequence_table_capacity",
                       &kv::HostPagedKVConfig::sequence_table_capacity)
        .def_readwrite("alignment_bytes",
                       &kv::HostPagedKVConfig::alignment_bytes)
        .def_readwrite("enable_prefix_reuse",
                       &kv::HostPagedKVConfig::enable_prefix_reuse)
        .def_readwrite("prefix_min_reuse_pages",
                       &kv::HostPagedKVConfig::prefix_min_reuse_pages)
        .def_readwrite("prefix_min_store_pages",
                       &kv::HostPagedKVConfig::prefix_min_store_pages)
        .def_readwrite("sequence_page_node_capacity",
                       &kv::HostPagedKVConfig::sequence_page_node_capacity)
        .def_readwrite("radix_node_capacity",
                       &kv::HostPagedKVConfig::radix_node_capacity)
        .def_readwrite("radix_edge_capacity",
                       &kv::HostPagedKVConfig::radix_edge_capacity)
        .def_readwrite("prefix_entry_capacity",
                       &kv::HostPagedKVConfig::prefix_entry_capacity)
        .def_readwrite("prefix_page_ref_capacity",
                       &kv::HostPagedKVConfig::prefix_page_ref_capacity)
        .def_readwrite("prefix_page_budget",
                       &kv::HostPagedKVConfig::prefix_page_budget)
        .def("__repr__",
             [](const kv::HostPagedKVConfig& self) {
                 return kv::ToString(self);
             });

    py::class_<kv::HostPagedKVStats>(m, "HostPagedKVStats")
        .def(py::init<>())
        .def_readwrite("num_total_pages", &kv::HostPagedKVStats::num_total_pages)
        .def_readwrite("num_free_pages", &kv::HostPagedKVStats::num_free_pages)
        .def_readwrite("num_used_pages", &kv::HostPagedKVStats::num_used_pages)
        .def_readwrite("num_active_sequences",
                       &kv::HostPagedKVStats::num_active_sequences)
        .def_readwrite("sequence_table_capacity",
                       &kv::HostPagedKVStats::sequence_table_capacity)
        .def_readwrite("total_bytes", &kv::HostPagedKVStats::total_bytes)
        .def_readwrite("num_prefix_entries",
                       &kv::HostPagedKVStats::num_prefix_entries)
        .def_readwrite("num_prefix_hits",
                       &kv::HostPagedKVStats::num_prefix_hits)
        .def_readwrite("num_prefix_misses",
                       &kv::HostPagedKVStats::num_prefix_misses)
        .def_readwrite("num_prefix_evictions",
                       &kv::HostPagedKVStats::num_prefix_evictions)
        .def_readwrite("num_prefix_pinned_pages",
                       &kv::HostPagedKVStats::num_prefix_pinned_pages)
        .def_readwrite("num_shared_pages",
                       &kv::HostPagedKVStats::num_shared_pages)
        .def("__repr__",
             [](const kv::HostPagedKVStats& self) {
                 return kv::ToString(self);
             });

    py::class_<HostKVPrefixCacheHarnessStats>(m, "HostKVPrefixCacheHarnessStats")
        .def(py::init<>())
        .def_readwrite("prefix_entry_count",
                       &HostKVPrefixCacheHarnessStats::prefix_entry_count)
        .def_readwrite("prefix_used_pages",
                       &HostKVPrefixCacheHarnessStats::prefix_used_pages)
        .def_readwrite("prefix_access_epoch",
                       &HostKVPrefixCacheHarnessStats::prefix_access_epoch)
        .def_readwrite("prefix_hit_count",
                       &HostKVPrefixCacheHarnessStats::prefix_hit_count)
        .def_readwrite("prefix_miss_count",
                       &HostKVPrefixCacheHarnessStats::prefix_miss_count)
        .def_readwrite("prefix_evict_count",
                       &HostKVPrefixCacheHarnessStats::prefix_evict_count)
        .def_readwrite("lru_head", &HostKVPrefixCacheHarnessStats::lru_head)
        .def_readwrite("lru_tail", &HostKVPrefixCacheHarnessStats::lru_tail);

    py::class_<HostKVPrefixCacheHarness>(m, "HostKVPrefixCacheHarness")
        .def(py::init<std::size_t, std::size_t, std::size_t, std::size_t,
                      std::size_t, bool, std::size_t, std::size_t,
                      std::size_t>(),
             py::arg("num_pages"), py::arg("radix_node_capacity"),
             py::arg("radix_edge_capacity"), py::arg("prefix_entry_capacity"),
             py::arg("prefix_page_ref_capacity"),
             py::arg("enable_prefix_reuse") = true,
             py::arg("prefix_min_reuse_pages") = 1,
             py::arg("prefix_min_store_pages") = 1,
             py::arg("prefix_page_budget") = 0)
        .def("lookup",
             [](HostKVPrefixCacheHarness& self,
                const std::vector<std::int32_t>& tokens,
                std::size_t max_pages) {
                 auto result = self.Lookup(tokens, max_pages);
                 return py::make_tuple(std::move(result.first), result.second);
             },
             py::arg("tokens"), py::arg("max_pages"))
        .def("commit", &HostKVPrefixCacheHarness::Commit, py::arg("tokens"),
             py::arg("pages"))
        .def("get_stats", &HostKVPrefixCacheHarness::GetStats)
        .def("page_refcount", &HostKVPrefixCacheHarness::PageRefcount,
             py::arg("page_idx"))
        .def("page_refcounts", &HostKVPrefixCacheHarness::PageRefcounts)
        .def("reset", &HostKVPrefixCacheHarness::Reset);

    py::class_<kv::KVAsyncTask>(m, "KVAsyncTask")
        .def_property_readonly("id", &kv::KVAsyncTask::id)
        .def("wait", &kv::KVAsyncTask::wait)
        .def("done", &kv::KVAsyncTask::done)
        .def("result", &kv::KVAsyncTask::result);

    BindHostPagedManager<kv::DefaultHostPagedKVManager>(
        m, "DefaultHostPagedKVManager");
    BindHostPagedManager<kv::MHAHostPagedKVManager>(
        m, "MHAHostPagedKVManager");
    BindHostPagedManager<kv::MLAHostPagedKVManager>(
        m, "MLAHostPagedKVManager");
    BindHostPagedWorkerView<kv::DefaultHostPagedKVWorkerView>(
        m, "DefaultHostPagedKVWorkerView");
    BindHostPagedWorkerView<kv::MLAHostPagedKVWorkerView>(
        m, "MLAHostPagedKVWorkerView");

    py::class_<Parameter_Server>(m, "Parameter_Server")
        .def(py::init<bool>())
        .def("Init", &Parameter_Server::Init)
        .def("get_skeleton_state_dict",
             &Parameter_Server::get_skeleton_state_dict)
        .def("byte_size", &Parameter_Server::byte_size)
        .def("module_weights_shm", &Parameter_Server::module_weights_shm);

    m.def(
        "set_data",
        [](torch::Tensor& dst, torch::Tensor& src) {
            dst.set_data(src);
            return dst;
        },
        "Set the data for the KV storage.");

    m.def("host_empty_cache", &at::cuda::CachingHostAllocator_emptyCache,
          "Empty the cache of caching host allocator");
    // Add version info
    m.attr("__version__") = "0.1.0";
}
