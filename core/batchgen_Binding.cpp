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

#include "batchgen.h"
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
    py::class_<BatchGen>(m, "batchgen")
        .def(py::init<py::object, py::object>())
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
