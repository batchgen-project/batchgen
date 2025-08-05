// #pragma once

// #include <torch/extension.h>
// #include <vector>

// // This struct definition is shared via the header.
// struct KeyValuePair {
//     float value;
//     int index;
// };

// // Declaration of the launcher function.
// std::vector<torch::Tensor> launch_moe_gate_forward_kernel(
//     torch::Tensor hidden_states,
//     torch::Tensor weight,
//     torch::Tensor e_score_correction_bias,
//     int64_t n_group,
//     int64_t topk_group,
//     int64_t n_routed_experts,
//     int64_t top_k,
//     double routed_scaling_factor);


#pragma once

#include <torch/extension.h>
#include <vector>
#include <cstdint>

// Shared between host/device. Keep SINGLE definition here.
struct alignas(8) KeyValuePair {
    float value;  // 4 bytes
    int   index;  // 4 bytes
};
static_assert(sizeof(KeyValuePair) == 8, "KeyValuePair must be 8 bytes");
static_assert(alignof(KeyValuePair) == 8, "KeyValuePair must be 8-aligned");

// Launcher declaration
std::vector<torch::Tensor> launch_moe_gate_forward_kernel(
    torch::Tensor hidden_states,
    torch::Tensor weight,
    torch::Tensor e_score_correction_bias,
    int64_t n_group,
    int64_t topk_group,
    int64_t n_routed_experts,
    int64_t top_k,
    double routed_scaling_factor);
