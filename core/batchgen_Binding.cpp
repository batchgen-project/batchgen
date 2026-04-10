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

#include "KV_Storage/host_paged_kv_manager.h"
#include "KV_Storage/host_paged_kv_worker_view.h"
#include "batchgen.h"
#include "Weights_Storage/Weights_Storage.h" 
#include "allocator.h"
#include "data_structures.h"
#include <ATen/cuda/CachingHostAllocator.h>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

namespace py = pybind11;
namespace kv = batchgen::kv;

namespace {

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
        .def("memfd_fd", &Manager::memfd_fd)
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
       .def("read_sequence_kv_to_cpu", &WorkerView::ReadSequenceKVToCPU,
            py::arg("sequence_id"),
            "Read all KV pages for a sequence directly to CPU tensors (no GPU). "
            "Returns (k_tensor, v_tensor). For MLA, v_tensor is empty.")
       .def("write_sequence_kv_from_cpu", &WorkerView::WriteSequenceKVFromCPU,
            py::arg("sequence_id"), py::arg("k_tensor"),
            py::arg("v_tensor") = std::nullopt,
            "Write KV data from CPU tensors directly to host pages (no GPU).")
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
             py::arg("max_tokens") = py::none());;
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
                "Get the past key states for the given query global indices and max sequence length.")
           .def("init_weight_storage", &BatchGen::init_weight_storage)
           .def_property(
              "host_paged_kv_worker_view",
              &BatchGen::host_paged_kv_worker_view,
              &BatchGen::set_host_paged_kv_worker_view,
              "Reference to the bound HostPagedKVWorkerView instance.")
          .def_property(
              "gpu_paged_kv_manager",
              &BatchGen::gpu_paged_kv_manager,
              &BatchGen::set_gpu_paged_kv_manager,
              "Python GPU paged KV manager bound to this engine.");
    
    py::class_<Weights_Storage>(m, "Weights_Storage")
        // Updated Constructor Binding
        .def(py::init<int>(), py::arg("device_id")) 
        
        .def("Init", &Weights_Storage::Init,
            py::arg("shm_name"),
            py::arg("byte_size"),
            py::arg("module_weights_shm"),
            py::arg("enable_hugetlbfs") = false,
            py::arg("enable_memfd") = false,
            py::arg("memfd_creator_pid") = -1,
            py::arg("memfd_fd") = -1)
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
        .def_readwrite("enable_memfd", &kv::HostPagedKVConfig::enable_memfd)
        .def_readwrite("memfd_creator_pid",
                       &kv::HostPagedKVConfig::memfd_creator_pid)
        .def_readwrite("memfd_fd", &kv::HostPagedKVConfig::memfd_fd)
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
        .def("__repr__",
             [](const kv::HostPagedKVStats& self) {
                 return kv::ToString(self);
             });

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
        .def(py::init<bool, bool>(), py::arg("enable_hugetlbfs"),
             py::arg("enable_memfd") = false)
        .def("Init", &Parameter_Server::Init)
        .def("get_skeleton_state_dict",
             &Parameter_Server::get_skeleton_state_dict)
        .def("byte_size", &Parameter_Server::byte_size)
        .def("module_weights_shm", &Parameter_Server::module_weights_shm)
        .def("weights_memfd_fd", &Parameter_Server::weights_memfd_fd);

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
