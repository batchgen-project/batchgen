/*
 * GPT-OSS-120B MoE Routing CUDA Extension.
 *
 * PyTorch C++ extension binding for CUDA routing kernels.
 * Built via torch.utils.cpp_extension.load() JIT compilation.
 */

#include <torch/extension.h>
#include "routing_ops.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gate_topk_softmax", &gate_topk_softmax_cuda,
          "Gate: fused top-k selection + softmax (CUDA)",
          py::arg("router_logits"),
          py::arg("k"),
          py::arg("topk_indices"),
          py::arg("topk_weights"),
          py::arg("num_valid_per_rank") = c10::nullopt,
          py::arg("bucket_size") = 0);

    m.def("dispatch_count_gather", &dispatch_count_gather_cuda,
          "Dispatch: count + prefix_sum + gather (CUDA)",
          py::arg("x"),
          py::arg("topk_indices"),
          py::arg("expert_start"),
          py::arg("num_local_experts"),
          py::arg("expert_counts"),
          py::arg("expert_offsets"),
          py::arg("expert_counters"),
          py::arg("dispatched_x"),
          py::arg("topk_pos"));

    m.def("reduce_weighted_scatter", &reduce_weighted_scatter_cuda,
          "Reduce: weighted scatter-add (CUDA)",
          py::arg("expert_output"),
          py::arg("topk_pos"),
          py::arg("topk_weights"),
          py::arg("N"),
          py::arg("H"),
          py::arg("K"),
          py::arg("output"));
}
