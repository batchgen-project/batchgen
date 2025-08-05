/* Copyright 2025 SGLang Team. All Rights Reserved.

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
#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/all.h>
#include <torch/library.h>

#include "mgn_kernel_ops.h"
TORCH_LIBRARY_FRAGMENT(mgn_kernel, m) {
    /*
     * From csrc/moe
     */
    m.def(
        "moe_fused_gate(Tensor input, Tensor bias, int num_expert_group, int "
        "topk_group, int topk, int "
        "num_fused_shared_experts, float routed_scaling_factor) -> "
        "(Tensor[])");
    m.impl("moe_fused_gate", torch::kCUDA, &moe_fused_gate);
    m.def(
        "expert_bincount(Tensor eids, int routed_expert_start_idx, int "
        "experts_per_rank, Device device) -> "
        "(Tensor[])");
    m.impl("expert_bincount", torch::kCUDA, &expert_bincount_cuda);
    m.def(
        "fused_moe_token_dispatch(Tensor global_x, Tensor topk_idx, Tensor "
        "token_idx, Tensor topk_pos, int "
        "routed_expert_start_idx, int routed_expert_end_idx) -> "
        "(Tensor[])");
    m.impl("fused_moe_token_dispatch", torch::kCUDA,
           &fused_moe_token_dispatch_cuda);

    /*
     * From csrc/elementwise
     */
    m.def(
        "fused_rmsnorm(Tensor input, Tensor weight, float eps) -> "
        "(Tensor)");
    m.impl("fused_rmsnorm", torch::kCUDA, &fused_rmsnorm_forward);
}

REGISTER_EXTENSION(common_ops)
