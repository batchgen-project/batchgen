/*
 * MGN (MoE General Native) CUDA Extension.
 *
 * PyTorch pybind11 extension binding for MGN CUDA kernels:
 *   - moe_fused_gate: fused hierarchical top-k expert selection
 *   - expert_bincount: count tokens per expert + compact active experts
 *   - fused_moe_token_dispatch: dispatch tokens to experts
 *   - compact_expert_data: compact expert metadata (replaces torch.nonzero)
 *   - fused_rmsnorm: RMSNorm elementwise kernel
 *
 * Built via torch.utils.cpp_extension / CUDAExtension at pip install time.
 */

#include <torch/python.h>
#include "include/mgn_kernel_ops.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // --- MoE routing / gating ---
    m.def("moe_fused_gate", &moe_fused_gate,
          "Fused MoE gate: hierarchical group selection + top-k expert selection (CUDA)",
          py::arg("scores"),
          py::arg("e_score_correction_bias"),
          py::arg("n_group"),
          py::arg("topk_group"),
          py::arg("n_routed_experts"),
          py::arg("top_k"),
          py::arg("routed_scaling_factor"));

    m.def("expert_bincount", &expert_bincount_cuda,
          "Count tokens per expert + compact active expert metadata (CUDA)",
          py::arg("eids"),
          py::arg("routed_expert_start_idx"),
          py::arg("experts_per_rank"),
          py::arg("device"));

    m.def("expert_bincount_sliced", &expert_bincount_cuda_sliced,
          "Count tokens per expert + compact + slice to active (CUDA)",
          py::arg("eids"),
          py::arg("routed_expert_start_idx"),
          py::arg("experts_per_rank"),
          py::arg("device"));

    m.def("fused_moe_token_dispatch", &fused_moe_token_dispatch_cuda,
          "Dispatch tokens to experts based on top-k indices (CUDA)",
          py::arg("global_x"),
          py::arg("topk_idx"),
          py::arg("token_idx"),
          py::arg("topk_pos"),
          py::arg("routed_expert_start_idx"),
          py::arg("routed_expert_end_idx"));

    m.def("compact_expert_data", &compact_expert_data_cuda,
          "Compact expert counts into dense active-expert metadata (CUDA)",
          py::arg("expert_counts"));

    // --- Elementwise ---
    m.def("fused_rmsnorm", &simple_rmsnorm_forward,
          "Fused RMSNorm (CUDA)",
          py::arg("input"),
          py::arg("weight"),
          py::arg("eps"));
}
