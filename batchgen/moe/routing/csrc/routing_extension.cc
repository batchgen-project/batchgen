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
          py::arg("num_valid_tokens") = -1);

    m.def("dispatch_count_gather", &dispatch_count_gather_cuda,
          "Dispatch: count+prefix_sum + gather (CUDA, 2 fused kernels)",
          py::arg("x"),
          py::arg("topk_indices"),
          py::arg("expert_start"),
          py::arg("num_local_experts"),
          py::arg("expert_counts"),
          py::arg("expert_offsets"),
          py::arg("expert_counters"),
          py::arg("dispatched_x"),
          py::arg("topk_pos"),
          py::arg("num_valid_tokens") = -1);

    m.def("router_bias_cast", &router_bias_cast_cuda,
          "Router epilogue: fused BF16 bias add + BF16->FP32 cast (CUDA)",
          py::arg("logits"),
          py::arg("bias"),
          py::arg("output"));

    m.def("create_fused_gate_context", &create_fused_gate_context,
          "Create cached fused gate context (SM90a WGMMA)",
          py::arg("router_weight"),
          py::arg("router_bias"),
          py::arg("topk"));

    m.def("destroy_fused_gate_context", &destroy_fused_gate_context,
          "Destroy cached fused gate context",
          py::arg("ctx_ptr"));

    m.def("fused_gate_forward", &fused_gate_forward,
          "Fused gate forward: WGMMA GEMM + bias + TopK + Softmax (SM90a)",
          py::arg("ctx_ptr"),
          py::arg("hidden_states"),
          py::arg("logits"),
          py::arg("topk_indices"),
          py::arg("topk_weights"),
          py::arg("num_valid_tokens") = -1);

    m.def("reduce_weighted_scatter", &reduce_weighted_scatter_cuda,
          "Reduce: weighted scatter-add (CUDA)",
          py::arg("expert_output"),
          py::arg("topk_pos"),
          py::arg("topk_weights"),
          py::arg("N"),
          py::arg("H"),
          py::arg("K"),
          py::arg("output"),
          py::arg("num_valid_tokens") = -1);
}
