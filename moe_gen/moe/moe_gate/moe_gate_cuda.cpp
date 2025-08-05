#include "moe_gate_launcher.h" // Include our header

// The Pybind11 module definition.
// This maps the Python function name "forward" to our C++ launcher function.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &launch_moe_gate_forward_kernel, "Fused MoE Gate Forward (CUDA)");
}