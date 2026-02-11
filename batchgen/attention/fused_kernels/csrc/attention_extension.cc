#include <torch/extension.h>
#include "attention_ops.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rmsnorm_forward", &rmsnorm_forward,
          "RMSNorm forward (CUDA)",
          py::arg("input"),
          py::arg("weight"),
          py::arg("eps"));

    m.def("add_rmsnorm_forward", &add_rmsnorm_forward,
          "Fused Add + RMSNorm forward (CUDA)",
          py::arg("residual"),
          py::arg("hidden"),
          py::arg("weight"),
          py::arg("eps"));

    m.def("rope_forward", &rope_forward,
          "Fused RoPE for Q+K forward (CUDA)",
          py::arg("query"),
          py::arg("key"),
          py::arg("cos"),
          py::arg("sin"),
          py::arg("half_dim"));

    m.def("qkv_split_forward", &qkv_split_forward,
          "QKV Split: packed QKV to separate Q/K/V (CUDA, allocating)",
          py::arg("qkv"),
          py::arg("q_size"),
          py::arg("kv_size"));

    m.def("qkv_split_inplace", &qkv_split_inplace,
          "QKV Split: write to pre-allocated Q/K/V (CUDA, zero-alloc)",
          py::arg("qkv"),
          py::arg("q_out"),
          py::arg("k_out"),
          py::arg("v_out"),
          py::arg("q_size"),
          py::arg("kv_size"));
}
