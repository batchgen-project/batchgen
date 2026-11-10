/*
 * WGMMA fused routing gate extension (compute capability 9.x only).
 *
 * Bound separately from the generic routing extension so the generic module
 * (_C_routing) links on every supported architecture while the WGMMA fused
 * gate stays a Hopper-only companion (_C_routing_wgmma).
 */

#include <torch/python.h>
#include "routing_ops.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("create_fused_gate_context", &create_fused_gate_context,
          "Create cached fused gate context (WGMMA)",
          py::arg("router_weight"),
          py::arg("router_bias"),
          py::arg("topk"));

    m.def("destroy_fused_gate_context", &destroy_fused_gate_context,
          "Destroy cached fused gate context",
          py::arg("ctx_ptr"));

    m.def("fused_gate_warmup", &fused_gate_warmup,
          "Pre-create TMA descriptor for input buffer (call before graph capture)",
          py::arg("ctx_ptr"),
          py::arg("hidden_states"));

    m.def("fused_router_forward", &fused_router_forward,
          "WGMMA router GEMM: BF16 hidden x BF16 weight^T -> FP32 logits",
          py::arg("ctx_ptr"),
          py::arg("hidden_states"),
          py::arg("logits"),
          py::arg("num_valid_tokens") = -1);

    m.def("fused_gate_forward", &fused_gate_forward,
          "Fused gate forward: WGMMA GEMM + bias + TopK + Softmax",
          py::arg("ctx_ptr"),
          py::arg("hidden_states"),
          py::arg("logits"),
          py::arg("topk_indices"),
          py::arg("topk_weights"),
          py::arg("num_valid_tokens") = -1);
}
