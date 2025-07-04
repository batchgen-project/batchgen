// clang-format off
/* ----------------------------------------------------------------------------  *
 *  MoE-Gen                                                                      *
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

#include "MoE_Gen.h"
#include "allocator.h"
#include <ATen/cuda/CachingHostAllocator.h>
#include <cstdlib>
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

namespace py = pybind11;
torch::Tensor pass_tensor(py::handle py_tensor) {
    // Borrow the tensor without increasing the reference count
    py::object borrowed_tensor = py::reinterpret_borrow<py::object>(py_tensor);

    // Cast to torch::Tensor
    auto tensor = borrowed_tensor.cast<torch::Tensor>();

    // Log tensor shape (for demonstration purposes)
    // spdlog::info("Tensor shape: {}", tensor.sizes());

    return tensor;  // Optionally return the tensor if needed
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<MoE_Gen>(m, "MoE_Gen")
        .def(py::init<py::object, py::object>())
        .def("Init", &MoE_Gen::Init)
        .def("terminate", &MoE_Gen::Terminate)
        // .def("set_batching_plan", &MoE_Gen::set_batching_plan)
        .def("kv_offload", &MoE_Gen::kv_offload)
        // .def("add_weight_storage", &MoE_Gen::add_weight_storage)
        .def("get_weights", &MoE_Gen::get_weights)
        .def("free_weights_buffer", &MoE_Gen::free_weights_buffer)
        .def("attn", &MoE_Gen::attn)
        .def("submit_to_KV_queue", &MoE_Gen::submit_to_KV_queue)
        .def("clear_expert_buffer", &MoE_Gen::clear_expert_buffer)
        // .def("get_skeleton_state_dict", &MoE_Gen::get_skeleton_state_dict)
        .def("prefill_complete_sync", &MoE_Gen::prefill_complete_sync)
        .def("set_phase", &MoE_Gen::set_phase)
        .def("clear_kv_storage", &MoE_Gen::clear_kv_storage)
        .def("clear_kv_copy_queue", &MoE_Gen::clear_kv_copy_queue)
        .def("reset_weight_copy_queue", &MoE_Gen::reset_weight_copy_queue)
        .def("clear_kv_buffer", &MoE_Gen::clear_kv_buffer)
        .def("clear_weight_copy_queue", &MoE_Gen::clear_weight_copy_queue)
        .def("reset_prefill_buffer", &MoE_Gen::reset_prefill_buffer)
        .def("create_fake_kv_storage", &MoE_Gen::create_fake_kv_storage)
        .def("get_tensor", &MoE_Gen::get_tensor)
        .def("start_h2d_worker", &MoE_Gen::start_h2d_worker)
        .def("set_global_routed_experts_data_ptr",
             &MoE_Gen::set_global_routed_experts_data_ptr)
        .def("cuda_enable_peer_access", &MoE_Gen::cuda_enable_peer_access)
        .def("save_compressed_kv", &MoE_Gen::save_compressed_kv)
        .def("set_weight_copy_queue", &MoE_Gen::set_weight_copy_queue)
        .def("reset_decoding_buffer", &MoE_Gen::reset_decoding_buffer)
        .def("stop_h2d_worker", &MoE_Gen::stop_h2d_worker)
        .def("copy_kv_to_worker",
             &MoE_Gen::copy_kv_to_worker,
             "Copy KV to worker for the given query global index and context length.")
        .def("clear_kv_gpu_storage",
             &MoE_Gen::clear_gpu_kv_storage,
             "Clear the GPU KV storage.")
        .def("get_kv_scale",
             &MoE_Gen::get_kv_scale,
             "Get the quantization scale for KV storage.")
        .def("get_past_key_states",
                &MoE_Gen::get_past_key_states,
                "Get the past key states for the given query global indices and max sequence length.");

             
    py::class_<Parameter_Server>(m, "Parameter_Server")
        .def(py::init<>())
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
