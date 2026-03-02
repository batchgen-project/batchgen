/* Copyright 2025 BatchGen Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <torch/extension.h>
#include <vector>

/*
 * From moe_fused_gate.cu
 */
std::vector<torch::Tensor> moe_fused_gate(
    torch::Tensor scores,                    // [n, n_routed_experts] - already sigmoid'd
    torch::Tensor e_score_correction_bias,   // [n_routed_experts]
    int64_t n_group,
    int64_t topk_group,
    int64_t n_routed_experts,
    int64_t top_k,
    double routed_scaling_factor);

/*
 * From expert_bin_count.cu
 */
std::vector<torch::Tensor> expert_bincount_cuda(
    torch::Tensor eids,
    int64_t routed_expert_start_idx,
    int64_t experts_per_rank,
    torch::Device device);

std::vector<torch::Tensor> expert_bincount_cuda_sliced(
    torch::Tensor eids,
    int64_t routed_expert_start_idx,
    int64_t experts_per_rank,
    torch::Device device);

std::vector<torch::Tensor> compact_expert_data_cuda(
    torch::Tensor expert_counts);

/*
 * From fused_moe_token_dispatch.cu
 */
std::vector<torch::Tensor> fused_moe_token_dispatch_cuda(
    torch::Tensor global_x,
    torch::Tensor topk_idx,
    torch::Tensor token_idx,
    torch::Tensor topk_pos,
    int64_t routed_expert_start_idx,
    int64_t routed_expert_end_idx);

/*
 * From rmsnorm.cu
 */
torch::Tensor simple_rmsnorm_forward(
    torch::Tensor input,
    torch::Tensor weight,
    double eps);
